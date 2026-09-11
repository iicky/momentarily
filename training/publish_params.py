"""Params-publishing family: build, write, and gate the R2 artifacts.

Extracted from training.train_em so the trainer module stays focused on the
EM fit and main() orchestration.  Every function here is a pure move — no
signature or behaviour changes — plus a CorpusStats TypedDict/dataclass that
was already the implicit contract.
"""

from __future__ import annotations

import io
import json
import math
import sys
import zipfile
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from botocore.exceptions import ClientError

from momentarily.hmm import EmissionParams, HMMParams
from training.dwell import DwellQuantiles
from training.gtfs_archive import digest_of
from training.gtfs_static import (
    RoutePattern,
    SegmentKey,
    dominant_successor,
    fetch_gtfs_zip,
    load_topology,
    patterns_to_json,
    read_version,
)
from training.headway import (
    SCHEDULED_HEADWAY_NOTE,
    load_gtfs_zip,
    scheduled_headway_baseline,
    scheduled_headway_to_json,
    select_reference_stops,
)
from training.load import TICK_SECONDS
from training.load_r2 import (
    MIN_THROUGHPUT_TICKS,
    StopFilter,
    build_segment_baseline,
    build_segment_throughput,
    fetch_vehicle_metrics,
    throughput_to_json,
)
from training.prov import (
    AgentFacts,
    ArtifactFacts,
    FeedFacts,
    ManifestFacts,
    build_trainer_run,
)
from training.provenance import code_provenance
from training.r2_client import (
    R2Config,
    get_object_bytes,
)
from training.reliability import MIN_SHARE
from training.segment_dwell import SegmentDwellStats, build_segment_dwell
from training.segments import canonical_adjacency

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


@dataclass(frozen=True)
class CorpusStats:
    """Audit metadata about the archive window a run actually trained on."""

    start_tick: int
    end_tick: int
    n_observations: int  # real (alert-bearing) tick-observations, pre-quiet-fill
    n_input_versions: int = 0  # archived alert-version objects that fed the fit
    n_vehicle_keys: int = 0  # archived vehicle-movement objects that fed the fit
    # BLAKE3 over the alert + vehicle object keys — the lineage fingerprint. See
    # INPUT_MANIFEST_VERSION for which key set a given hash covers.
    input_blake3: str = ""

    @property
    def span_seconds(self) -> int:
        return self.end_tick - self.start_tick


def aligned_window(start: date, end: date) -> tuple[int, int]:
    """Tick-aligned UTC window covering [start, end+1day)."""
    start_dt = datetime(start.year, start.month, start.day, tzinfo=UTC)
    end_dt = datetime(end.year, end.month, end.day, tzinfo=UTC) + timedelta(days=1)
    start_epoch = (int(start_dt.timestamp()) // TICK_SECONDS) * TICK_SECONDS
    end_epoch = (int(end_dt.timestamp()) // TICK_SECONDS) * TICK_SECONDS
    return start_epoch, end_epoch


PARAMS_KEY = "state/params.json"
# Immutable per-run snapshots live under this prefix as v<trained_at>.json.
VERSIONED_PARAMS_PREFIX = "state/params/"
SCHEMA_VERSION = "1"

# The composition of training_corpus.input_blake3. v1 fingerprinted the alert-
# version keys only; v2 folds the vehicle-archive keys in too, because the
# vehicle archive now feeds both the serialized movement_baseline and the EM
# normal-state advance prior, so a manifest over alerts alone under-describes the
# inputs. Bumped here (rather than silently redefining the hash) so a consumer can
# tell which key set a published input_blake3 covers.
INPUT_MANIFEST_VERSION = 2

# Emission channels dropped from the published params. The Worker reads
# service_mu/service_sigma as OPTIONAL (worker/src/hmm.ts: the service term only
# scores when `em.service_mu !== undefined`), so omitting them here turns the
# service channel off live via the exact back-compat gate pre-service params used
# — no Worker deploy required. This is the 2026-08-31 fit-or-drop verdict: fitting
# the service Gaussian showed the per-state means barely separate (median spread
# 0.15 on a ~1.0 scale, sigma ~0.25) and the fit is severity-INVERTED on 15/28
# routes, because assigned_n supply is statistically independent of the
# alert-defined disruption axis the states are anchored on (journal 2026-08-31
# assigned_n-independence result). A sub-nat channel that points the wrong way
# half the time cannot help a posterior already one-hot at log-odds in the
# hundreds, and the live suspended arm already reads assigned_n on its own axis.
# The channel is still FITTED (see --diagnose-service) so the decision stays
# reproducible; it is simply not shipped for scoring.
_DROPPED_EMISSION_KEYS = ("service_mu", "service_sigma")


def params_to_json(params: HMMParams) -> dict[str, Any]:
    """Serialize HMMParams to the loose schema the Worker reads.

    Drops the service Gaussian (see _DROPPED_EMISSION_KEYS): the trained value is
    not shipped, so the Worker's optional-param gate leaves the channel unscored.
    """

    def emit(em: EmissionParams) -> dict[str, Any]:
        d = asdict(em)
        for k in _DROPPED_EMISSION_KEYS:
            d.pop(k, None)
        return d

    body: dict[str, Any] = {
        "transition": [list(row) for row in params.transition],
        "initial": list(params.initial),
        "emissions": emit(params.emissions),
    }
    if params.emissions_by_bin is not None:
        body["emissions_by_bin"] = [emit(e) for e in params.emissions_by_bin]
    return body


@dataclass(frozen=True)
class DeferredPointer:
    """A live-pointer write held back until the whole run is durable.

    The publish is transactional (see main): every immutable v<trained_at>
    snapshot and the PROV doc are written first, then the live pointers are
    flipped last. Each held-back pointer carries its own bytes and cache header
    so the flush is a plain replay."""

    key: str
    body: bytes
    cache_control: str


def _publish(
    client: S3Client,
    bucket: str,
    live_key: str,
    versioned_key: str,
    body: bytes,
    cache_control: str,
    pending: list[DeferredPointer] | None,
) -> None:
    """Write the immutable versioned snapshot now. Write the live pointer now
    too (`pending` is None — the standalone/backfill path), or defer it into
    `pending` so a transactional caller can flip it only once every artifact and
    the PROV doc are durable. The versioned key is immutable and never read by
    the Worker (which reads only the live pointer), so writing it eagerly is
    always safe; deferring the pointer is what keeps the flip atomic per run."""
    client.put_object(
        Bucket=bucket,
        Key=versioned_key,
        Body=body,
        ContentType="application/json",
        CacheControl=cache_control,
    )
    if pending is None:
        client.put_object(
            Bucket=bucket,
            Key=live_key,
            Body=body,
            ContentType="application/json",
            CacheControl=cache_control,
        )
    else:
        pending.append(DeferredPointer(live_key, body, cache_control))


def build_params_doc(
    per_route: dict[str, HMMParams],
    *,
    corpus: CorpusStats,
    n_routes_trained: int,
    dwell_quantiles: dict[str, dict[str, DwellQuantiles]] | None = None,
    dwell_quantiles_by_alert: (
        dict[str, dict[str, dict[str, DwellQuantiles]]] | None
    ) = None,
    dwell_quantiles_by_cause: (
        dict[str, dict[str, dict[str, DwellQuantiles]]] | None
    ) = None,
    dwell_movement: dict[str, dict[str, DwellQuantiles]] | None = None,
    hyperparams: dict[str, Any] | None = None,
    input_profile: dict[str, Any] | None = None,
    movement_baseline: dict[str, Any] | None = None,
    movement_through_stops: dict[str, dict[str, list[str]]] | None = None,
    service_baseline: dict[str, Any] | None = None,
    schedule_rate: dict[str, Any] | None = None,
    trained_at: int | None = None,
    feed: FeedFacts | None = None,
    prov_ref: str | None = None,
) -> dict[str, Any]:
    """Assemble the params.json document the Worker reads.

    Pure: no I/O, so main can build it once to run the plausibility gate against
    the currently-live blob before anything is written, and write_params rebuilds
    the identical doc to publish. `trained_at` is resolved here so the caller can
    key the versioned snapshot off the same value the doc carries."""
    trained_at = trained_at or int(datetime.now(UTC).timestamp())
    routes_doc = {r: params_to_json(p) for r, p in per_route.items()}
    if dwell_quantiles:
        # Merge per-route empirical dwell into the same per-route subdoc — the
        # Worker reads it as an optional sibling of `emissions`/`transition`.
        for r, by_state in dwell_quantiles.items():
            if r in routes_doc:
                routes_doc[r]["dwell_quantiles"] = by_state
    if dwell_quantiles_by_alert:
        # Cause-segmented dwell, layered on top of the (route, state) aggregate.
        # The Worker prefers (route, state, alert_type) and falls back to the
        # aggregate above when a cause cell is absent.
        for r, by_state_alert in dwell_quantiles_by_alert.items():
            if r in routes_doc:
                routes_doc[r]["dwell_quantiles_by_alert"] = by_state_alert
    if dwell_quantiles_by_cause:
        # Cause-CATEGORY dwell for the episode-recovery grader (Episode.cause is a
        # coarse category, not a raw alert_type). The Worker ignores this key
        # (zod strips it); scorecard.dwell_lookup_from_params reads it so the
        # grade stops silently falling back to the (route, state) aggregate.
        for r, by_state_cause in dwell_quantiles_by_cause.items():
            if r in routes_doc:
                routes_doc[r]["dwell_quantiles_by_cause"] = by_state_cause
    doc = {
        "schema_version": SCHEMA_VERSION,
        "trained_at": trained_at,
        "provenance": code_provenance(),
        "hyperparams": hyperparams or {},
        "input_profile": input_profile or {},
        "training_corpus": {
            "start_tick": corpus.start_tick,
            "end_tick": corpus.end_tick,
            "n_routes_trained": n_routes_trained,
            "n_observations": corpus.n_observations,
            # input_blake3 is a BLAKE3 fingerprint over the alert AND vehicle
            # archive keys the fit read; input_manifest_version names which key
            # set it covers (2 = alerts + vehicles, see INPUT_MANIFEST_VERSION).
            # n_input_versions counts the alert keys, n_vehicle_keys the vehicle
            # keys, so the hash's composition is auditable, not just its value.
            "n_input_versions": corpus.n_input_versions,
            "n_vehicle_keys": corpus.n_vehicle_keys,
            "input_manifest_version": INPUT_MANIFEST_VERSION,
            "input_blake3": corpus.input_blake3,
        },
        "routes": routes_doc,
    }
    # Per-(route, direction, tod_bin) advance-rate baseline the Worker needs live to
    # gate and score the movement channel. Top-level (not per-route) so the assigned_n
    # service baseline can sit beside it under the same delivery.
    if movement_baseline:
        doc["movement_baseline"] = movement_baseline
    # The stops that baseline was fitted on: from_stops with a scheduled
    # predecessor and successor. The Worker counts a cross-tick advance or stall
    # only at these, so a terminal layover is not evidence of a stall. Travels in
    # the same object as the baseline deliberately — scoring against a stop set
    # the baseline was not fitted with judges layovers against a through-stop
    # normal. Absent means unfiltered on both sides.
    if movement_through_stops:
        doc["movement_through_stops"] = movement_through_stops
    # Per-(route, tod_bin) assigned_n baseline the Worker divides live assigned_n
    # by to form the service ratio the emission scores. Top-level beside
    # movement_baseline.
    if service_baseline:
        doc["service_baseline"] = service_baseline
    # Per-(route, schedule_bin) scheduled-presence rate the Worker uses to split a
    # no-service reading into suspended vs not_scheduled. Top-level beside the
    # baselines.
    if schedule_rate:
        doc["schedule_rate"] = schedule_rate
    # Movement-primary dwell (C2). Route scope only -- segment scope is
    # training.segment_dwell's own object at state/segment_dwell.json. Top-
    # level like the baselines above, not nested per-route: the Worker's
    # movementDwellFor and the scorecard's movement_dwell_lookup_from_params
    # both read it that way.
    if dwell_movement:
        doc["dwell_movement"] = dwell_movement
    # Which GTFS static timetable this run was measured against: the feed's
    # self-declared version AND a sha256 content digest computed over the exact
    # fetched bytes. The version string alone cannot pin the snapshot (MTA
    # republishes under the same name); the digest names it. Absent when the feed
    # fetch failed — a missing input is never a fabricated one.
    if feed is not None:
        doc["gtfs_feed"] = {
            "version": feed.version,
            "sha256": feed.sha256,
            "start": feed.start,
            "end": feed.end,
        }
    # Pointer to this run's W3C PROV-JSON sidecar (state/prov/v<trained_at>.json),
    # which states the full lineage in a standard vocabulary. The ad-hoc blocks
    # above stay authoritative for existing consumers; this only adds a reference.
    if prov_ref is not None:
        doc["prov_ref"] = prov_ref
    return doc


def write_params(
    client: S3Client,
    bucket: str,
    per_route: dict[str, HMMParams],
    *,
    corpus: CorpusStats,
    n_routes_trained: int,
    dwell_quantiles: dict[str, dict[str, DwellQuantiles]] | None = None,
    dwell_quantiles_by_alert: (
        dict[str, dict[str, dict[str, DwellQuantiles]]] | None
    ) = None,
    dwell_quantiles_by_cause: (
        dict[str, dict[str, dict[str, DwellQuantiles]]] | None
    ) = None,
    dwell_movement: dict[str, dict[str, DwellQuantiles]] | None = None,
    hyperparams: dict[str, Any] | None = None,
    input_profile: dict[str, Any] | None = None,
    movement_baseline: dict[str, Any] | None = None,
    movement_through_stops: dict[str, dict[str, list[str]]] | None = None,
    service_baseline: dict[str, Any] | None = None,
    schedule_rate: dict[str, Any] | None = None,
    trained_at: int | None = None,
    feed: FeedFacts | None = None,
    prov_ref: str | None = None,
    pending: list[DeferredPointer] | None = None,
) -> str:
    """Write the live params pointer plus an immutable versioned snapshot.

    The Worker reads state/params.json; the state/params/v<epoch>.json copies
    give us a per-run rollback trail. Returns the versioned key. `pending`, when
    given, defers the state/params.json pointer flip for a transactional caller
    (see main) so params.json never lands before its sidecars."""
    doc = build_params_doc(
        per_route,
        corpus=corpus,
        n_routes_trained=n_routes_trained,
        dwell_quantiles=dwell_quantiles,
        dwell_quantiles_by_alert=dwell_quantiles_by_alert,
        dwell_quantiles_by_cause=dwell_quantiles_by_cause,
        dwell_movement=dwell_movement,
        hyperparams=hyperparams,
        input_profile=input_profile,
        movement_baseline=movement_baseline,
        movement_through_stops=movement_through_stops,
        service_baseline=service_baseline,
        schedule_rate=schedule_rate,
        trained_at=trained_at,
        feed=feed,
        prov_ref=prov_ref,
    )
    body = json.dumps(doc).encode()
    versioned_key = f"{VERSIONED_PARAMS_PREFIX}v{doc['trained_at']}.json"
    _publish(
        client,
        bucket,
        PARAMS_KEY,
        versioned_key,
        body,
        "public, max-age=300, s-maxage=900",
        pending,
    )
    return versioned_key


SEGMENT_PARAMS_KEY = "state/segment_params.json"
VERSIONED_SEGMENT_PREFIX = "state/segment_params/"
# RETENTION: the versioned state/segment_params/ snapshots are kept forever.
# training.prune has no rule for this prefix -- its DATED_PREFIXES cover only
# the dated archive/v1 streams, and PARAMS_PREFIX covers only state/params/ --
# so nothing ever sweeps it. Each snapshot is one small per-run JSON doc
# (segment cells + adjacency + throughput, no raw per-tick data), negligible
# next to any dated archive prefix, and a past snapshot is only ever fetched
# by exact key during a manual rollback (docs/params-rollback.md) alongside
# the matching state/params/v<trained_at>.json -- never listed or iterated --
# so an unbounded prefix costs nothing at read time either. If this needs to
# become bounded, add a rule beside PARAMS_PREFIX in training/prune.py.


def static_topology() -> tuple[
    dict[SegmentKey, list[tuple[str, int]]] | None,
    dict[tuple[str, str], list[RoutePattern]] | None,
    str,
]:
    """The static successor skeleton and stopping patterns for this run, or
    (None, None, "observed") with the reason printed.

    Fetched once and shared: the advance baseline, the through-stop set it is
    fitted against and the published segment topology all have to describe the
    same timetable. A Worker scoring against a stop set the baseline was not
    fitted with would judge layovers against a through-stop normal.
    """
    try:
        successors, patterns = load_topology()
        return successors, patterns, "gtfs_static"
    except Exception as exc:
        print(
            f"gtfs static topology unavailable, using observed adjacency ({exc})",
            file=sys.stderr,
        )
        return None, None, "observed"


def _stop_filter(
    through: frozenset[tuple[str, str, str]] | None,
) -> StopFilter | None:
    """Admit only from_stops the timetable puts mid-chain. None passes
    everything, which is what an unavailable static feed leaves us with.

    One definition for every fit in this run: a rate fitted over a different
    stop set than the one published in movement_through_stops would have the
    Worker judging layovers against a through-stop normal.
    """
    if through is None:
        return None
    return lambda route, direction, frm: (route, direction, frm) in through


def write_segment_params(
    cfg: R2Config,
    client: S3Client,
    bucket: str,
    start_date: date,
    end_date: date,
    trained_at: int,
    static_successors: dict[SegmentKey, list[tuple[str, int]]] | None,
    static_patterns: dict[tuple[str, str], list[RoutePattern]] | None,
    topology_source: str,
    through: frozenset[tuple[str, str, str]] | None,
    prov_ref: str | None = None,
    pending: list[DeferredPointer] | None = None,
) -> int:
    """Write the segment baseline + adjacency as their OWN R2 object (not
    folded into params.json, which the Worker parses on the hot per-tick
    path). The Worker reads this at step 8b, off the publish path, to score
    per-segment movement and roll it up to station service flow.

    Topology (adjacency) comes from the static GTFS timetable the caller fetched
    for the whole run (`static_topology`), when that fetch succeeded: a segment
    exists because the schedule says so, keyed
    'route|dir|from'. A from_stop with more than one static successor
    (branch/express) keeps its full successor list, not just the modal
    winner. canonical_adjacency (observed cross-tick transitions) is now only
    the fallback for when the GTFS fetch itself fails, plus a `share`/`n`
    reliability annotation riding along on whichever entries the vehicle
    archive also observed that window — annotation only, it no longer decides
    whether an entry is published.

    cells (the pooled advance-rate baseline) still needs actual cross-tick
    vehicle data, so it's scoped to baseline.items() regardless of topology
    source; the whole object is skipped when that's empty (an archive hiccup
    leaves nothing to pair the topology with). `through` restricts those leaves
    to mid-chain from_stops, the same set params.json publishes — a leaf fitted
    over layovers would hand the Worker a normal it never scores against.

    Each cell also carries `lam`: its expected matched traversals per tick by
    time bin (load_r2.build_segment_throughput), the denominator that lets the
    Worker read an empty window as evidence instead of abstaining. Fitted over
    the same bodies and the same `through` filter as p0, so the rate and the
    normal it complements describe one stop set. Bins where the cell runs
    nothing are dropped from `lam` and read as zero against the bin set in
    `throughput.ticks` — see load_r2.throughput_to_json.

    Fail-soft: a vehicle-archive hiccup skips the object, leaving the last good
    one; the station-flow surface just goes stale, never blocks the params run.
    """
    try:
        bodies = fetch_vehicle_metrics(
            cfg, start_date=start_date, end_date=end_date, client=client
        )
        stop_filter = _stop_filter(through)
        baseline = build_segment_baseline(bodies, counts_from_stop=stop_filter)
        observed_adjacency = canonical_adjacency(bodies)
        rates, exposure = build_segment_throughput(bodies, counts_from_stop=stop_filter)
        lam = throughput_to_json(rates)

        cells: dict[str, dict[str, Any]] = {}
        for key, cell in baseline.items():
            entry: dict[str, Any] = {"p0": round(cell.p0, 6), "n": cell.n}
            cell_lam = lam.get("|".join(key))
            if cell_lam is not None:
                entry["lam"] = cell_lam
            cells["|".join(key)] = entry
        if not cells:
            print("segment params skipped (no through-segments)", file=sys.stderr)
            return 0

        adj_doc: dict[str, dict[str, Any]] = {}
        if static_successors is not None:
            for key, succs in static_successors.items():
                if not succs:
                    continue
                to_stop, _n_trips = dominant_successor(succs)
                adj_entry: dict[str, Any] = {
                    "to": to_stop,
                    "source": "gtfs_static",
                    "successors": [{"to": t, "n_trips": n} for t, n in succs],
                }
                obs = observed_adjacency.get(key)
                if obs is not None:
                    adj_entry["share"] = round(obs.share, 4)
                    adj_entry["n"] = obs.n
                adj_doc["|".join(key)] = adj_entry
        else:
            for key, adj in observed_adjacency.items():
                adj_doc["|".join(key)] = {
                    "to": adj.to_stop,
                    "source": "observed",
                    "share": round(adj.share, 4),
                    "n": adj.n,
                }

        doc = {
            "schema_version": SCHEMA_VERSION,
            "trained_at": trained_at,
            # Which code produced this doc, matching params.json/eval.json — the
            # off-Worker consumers (viz) read the topology and its ordering, so
            # they can name the tree that built it. See training/provenance.py.
            "provenance": code_provenance(),
            "min_share": MIN_SHARE,
            "topology_source": topology_source,
            "cells": cells,
            "adjacency": adj_doc,
            # How the per-cell `lam` rates were fitted: the bin function, the
            # exposure floor, and the observed ticks per published bin. The bin
            # set is exactly these keys — a bin missing here was never fitted
            # (the Worker abstains), a bin here but missing from a cell's `lam`
            # was fitted at zero (nothing scheduled, silence is normal).
            "throughput": {
                "bin": "schedule_bin",
                "min_ticks": MIN_THROUGHPUT_TICKS,
                "ticks": dict(sorted(exposure.items())),
            },
            # Canonical per-(route, direction) stop order: the actual scheduled
            # trip patterns, most-run first. A consumer reads line order off
            # these instead of relinearizing the single-successor adjacency
            # graph, which mangles express/local and branch splits. Empty on the
            # observed-adjacency fallback (no static feed to read patterns from).
            "route_stops": (
                patterns_to_json(static_patterns) if static_patterns is not None else {}
            ),
        }
        if prov_ref is not None:
            doc["prov_ref"] = prov_ref
        body = json.dumps(doc).encode()
        versioned = f"{VERSIONED_SEGMENT_PREFIX}v{trained_at}.json"
        _publish(
            client, bucket, SEGMENT_PARAMS_KEY, versioned, body, "no-store", pending
        )
        return len(cells)
    except Exception as exc:
        print(f"segment params skipped ({exc})", file=sys.stderr)
        return 0


SERVICE_BASELINE_KEY = "state/service_baseline.json"
VERSIONED_SERVICE_PREFIX = "state/service_baseline/"
# RETENTION: the versioned state/service_baseline/ snapshots are kept forever.
# Same absence of a training.prune rule as state/segment_params/ above (only
# DATED_PREFIXES and PARAMS_PREFIX are policed), and the same negligible size
# (one per-run assigned_n baseline doc). Doubly worth keeping here: this
# sidecar is versioned by its OWN `generated_at`, not `trained_at` (see
# write_service_baseline below), because both a full retrain AND a standalone
# training.backfill_service_baseline refresh write into this prefix on their
# own schedules -- pruning by age would delete the one sidecar a rollback of
# an OLDER params.json needs to pair with (docs/params-rollback.md), since a
# newer standalone refresh's timestamp doesn't track params.json's own age at
# all.

# Trailing window for the published service-baseline sidecar, deliberately wider
# than a typical HMM retrain window. Sized so a weekend-hourly (route,
# schedule_bin) cell clears the sidecar's SERVICE_MIN_NIGHTS gate with margin: 35
# days is 5 weekends = 10 Sat/Sun nights per cell against a floor of 8, so a
# normally-thin weekend late-night cell publishes a trusted median instead of
# abstaining. Still inside archive-retention headroom. Shared with
# backfill_service_baseline so a retrain publish and a standalone backfill write
# the same sidecar.
SERVICE_SIDECAR_WINDOW_DAYS = 35

# Trailing window for fitting the empirical dwell quantiles (dwell_quantiles,
# dwell_quantiles_by_alert, dwell_quantiles_by_cause and the pooled `normal`
# cells), deliberately wider than the 14d transition/emission window. The HMM
# transition/emission fit stays on --days because its self-loops describe the
# CURRENT regime persistence; the dwell curves are graded on recover-by-H and
# need the rare long severe incidents to be represented, which a 14d window is
# too short to hold — thin (n=5-15), short-tailed disrupted cells with
# curve_max well under 120min pin recover_by_120 at 1.0 near-permanently,
# regardless of shape. Mirrors SERVICE_SIDECAR_WINDOW_DAYS's 35d = 5 weekends
# so severe weekend incidents land in the fit, and stays inside archive
# retention. This only widens the regime_transitions/predictions read, which is
# NOT part of training_corpus.input_blake3 (INPUT_MANIFEST_VERSION covers the
# alert + vehicle archive keys only), so no manifest bump is needed; the
# resolved dwell window is recorded in hyperparams instead.
DWELL_WINDOW_DAYS = SERVICE_SIDECAR_WINDOW_DAYS


def write_service_baseline(
    client: S3Client,
    bucket: str,
    hourly: dict[str, Any],
    generated_at: int,
    params_trained_at: int | None = None,
    quantiles: dict[str, Any] | None = None,
    prov_ref: str | None = None,
    pending: list[DeferredPointer] | None = None,
) -> int:
    """Write the per-(route, schedule_bin) assigned_n baseline -- the supply
    axis's denominator -- as its OWN versioned R2 object, decoupled from
    params.json. Refreshing it never moves the HMM artifact's trained_at, so it
    cannot reseed the Worker's filter or split the grader's params-version
    window. Versioned by its OWN `generated_at` (not the model's trained_at) so a
    later refresh can't overwrite a prior immutable snapshot; `params_trained_at`
    records which frozen model it was computed to accompany. `quantiles` is the
    sibling per-(route, schedule_bin) p10/p90 spread (training.load_r2.
    compute_service_quantiles / service_quantiles_to_json), same keying as
    `hourly`; omitted from the doc when absent (None or empty), so a caller with
    no quantile data round-trips a sidecar exactly like today's. Mirrors
    write_segment_params: live pointer + immutable versioned snapshot, skipped
    when empty. Returns the route count."""
    if not hourly:
        return 0
    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "baseline": hourly,
    }
    if params_trained_at is not None:
        doc["params_trained_at"] = params_trained_at
    if quantiles:
        doc["quantiles"] = quantiles
    if prov_ref is not None:
        doc["prov_ref"] = prov_ref
    body = json.dumps(doc).encode()
    versioned = f"{VERSIONED_SERVICE_PREFIX}v{generated_at}.json"
    _publish(client, bucket, SERVICE_BASELINE_KEY, versioned, body, "no-store", pending)
    return len(hourly)


SCHEDULED_HEADWAY_KEY = "state/scheduled_headway.json"
VERSIONED_SCHEDULED_HEADWAY_PREFIX = "state/scheduled_headway/"
# RETENTION: the versioned state/scheduled_headway/ snapshots are kept
# forever. Same absence of a training.prune rule as the sidecars above (only
# DATED_PREFIXES and PARAMS_PREFIX are policed) and the same reasoning: one
# small per-run median-headway doc, versioned by `trained_at` like
# state/params/, so a rollback of a given params.json (docs/params-rollback.md)
# always has this sidecar's matching version still available to pair with it.


def write_scheduled_headway(
    client: S3Client,
    bucket: str,
    trained_at: int,
    feed_zip_bytes: bytes | None = None,
    prov_ref: str | None = None,
    pending: list[DeferredPointer] | None = None,
) -> int:
    """Write the scheduled-headway baseline as its OWN R2 object: median
    timetable time-between-trains at each route/direction's canonical reference
    stop, per hour-of-week 0..167 (see training.headway.scheduled_headway_baseline).

    Published so a consumer can render an observed headway as a ratio/deviation
    ("~2x the usual gap for this hour") with no second network fetch — it lands
    in state/ beside the other weekly-fit artifacts the Worker already reads. A
    readability normaliser for the viz, NOT the excess-wait severity baseline,
    which stays own-cell (a schedule baseline false-alarms ~45% of normal ticks).

    Self-contained and fail-soft, mirroring write_segment_dwell: one static-feed
    fetch, parsed with the same trip-by-trip streaming the topology read uses; a
    fetch or parse hiccup just skips the object and leaves the last good one,
    never blocking the params publish. Live pointer + immutable versioned
    snapshot. Returns the cell count."""
    try:
        # Reuse the run's already-fetched, already-digested feed bytes when the
        # caller passes them (so the published cells derive from the exact bytes
        # the PROV feed entity is named by); otherwise self-fetch, staying
        # standalone and fail-soft for the backfill entrypoint.
        zf = (
            zipfile.ZipFile(io.BytesIO(feed_zip_bytes))
            if feed_zip_bytes is not None
            else load_gtfs_zip()
        )
        try:
            reference_stops = select_reference_stops(zf)
            cells = scheduled_headway_baseline(zf, reference_stops)
            version = read_version(zf)
        finally:
            zf.close()
        if not cells:
            print("scheduled headway skipped (no cells)", file=sys.stderr)
            return 0
        doc = {
            "schema_version": SCHEMA_VERSION,
            "trained_at": trained_at,
            "provenance": code_provenance(),
            # Which timetable these headways were read from: a scheduled number
            # is meaningless without naming the feed version that produced it.
            "feed_version": version.version,
            "note": SCHEDULED_HEADWAY_NOTE,
            # 'route|direction' -> the static canonical reference stop the cell
            # is keyed on, so a consumer can see WHERE the baseline was measured
            # and detect a runtime reroute-fallback mismatch.
            "reference_stops": {
                f"{rs.route}|{rs.direction}": rs.stop_id
                for rs in reference_stops.values()
            },
            # 'route|direction|hour_of_week' -> {median_headway_s, n_trips}. An
            # absent cell is no scheduled service — never a fabricated 0.
            "cells": scheduled_headway_to_json(cells),
        }
        if prov_ref is not None:
            doc["prov_ref"] = prov_ref
        body = json.dumps(doc).encode()
        versioned = f"{VERSIONED_SCHEDULED_HEADWAY_PREFIX}v{trained_at}.json"
        _publish(
            client, bucket, SCHEDULED_HEADWAY_KEY, versioned, body, "no-store", pending
        )
        return len(cells)
    except Exception as exc:
        print(f"scheduled headway skipped ({exc})", file=sys.stderr)
        return 0


PROV_KEY = "state/prov/latest.json"
VERSIONED_PROV_PREFIX = "state/prov/"
# Public mirror of the PROV sidecars. Standard-vocabulary provenance is FOR
# outside consumers, and the docs are tiny and immutable, so they are served
# under the public v1/ prefix (the only prefix index.ts exposes) alongside the
# private state/ copies. A snapshot's provenance.prov_ref points at the
# versioned mirror's public URL, so a consumer can walk to the lineage without
# knowing bucket internals.
PUBLIC_PROV_KEY = "v1/prov/latest.json"
PUBLIC_PROV_PREFIX = "v1/prov/"
# RETENTION: the PROV sidecars are kept forever. training.prune never sweeps the
# state/prov/ or v1/prov/ prefixes (its rules cover only the dated archive
# prefixes and state/params/v* at PARAMS_RETENTION_DAYS), and prune-nightly.yml
# carries the matching keep-forever note. One tiny (~KB) immutable doc per weekly
# run is negligible next to any dated archive prefix; the public v1/prov mirror is
# advertised immutable (max-age 1y) and is the target of every published
# snapshot's prov_ref, so pruning it would strand lineage references; and
# provenance is an audit record whose value is permanence. A prov doc names
# artifacts by immutable key as a recorded fact, so it deliberately outlives the
# state/params/v* snapshot it describes (pruned at 180d) — a historical record,
# not a live pointer. If this policy changes, update prune-nightly.yml's note too.


def fetch_gtfs_feed() -> tuple[bytes, FeedFacts] | None:
    """Fetch the GTFS static feed once for the run's provenance.

    Returns the fetched zip bytes plus the feed's identity: its self-declared
    version AND a sha256 content digest computed over those exact bytes at fetch
    time. The digest is what pins the snapshot — MTA republishes under the same
    version name, so the version string alone cannot name which bytes were read.
    Returns None when the feed is unavailable, so the run publishes no GTFS
    provenance rather than an ungrounded claim; the bytes are handed to
    write_scheduled_headway so its cells derive from the same snapshot the PROV
    feed entity names."""
    try:
        data = fetch_gtfs_zip()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            version = read_version(zf)
    except Exception as exc:
        print(f"gtfs feed provenance unavailable ({exc})", file=sys.stderr)
        return None
    facts = FeedFacts(
        version=version.version,
        sha256=digest_of(data),
        start=version.start.isoformat() if version.start else None,
        end=version.end.isoformat() if version.end else None,
    )
    return data, facts


def write_prov(
    client: S3Client,
    bucket: str,
    *,
    trained_at: int,
    started_at: int,
    corpus: CorpusStats,
    artifacts: list[ArtifactFacts],
    feed: FeedFacts | None = None,
    pending: list[DeferredPointer] | None = None,
) -> str:
    """Publish the run's W3C PROV-JSON sidecar in two places: the private
    state/prov/ copies (latest.json pointer + v<trained_at>.json snapshot) that
    the other sidecars live beside, and a public mirror under v1/prov/ so
    outside consumers can read the standard-vocabulary lineage. The document is
    byte-stable for fixed inputs (training.prov.ProvDocument.to_json), and every
    relation it carries is grounded in a recorded fact — the grounding rule
    lives in training.prov, not here. Returns the private versioned key, which
    is the prov_ref the published artifacts point back at; the snapshot's own
    prov_ref is the public mirror's URL, derived Worker-side from trained_at."""
    prov = code_provenance()
    agent = AgentFacts(
        code_sha=prov["code_sha"], dirty=prov["dirty"], producer=prov["producer"]
    )
    doc = build_trainer_run(
        trained_at=trained_at,
        started_at=started_at,
        agent=agent,
        manifest=ManifestFacts(
            blake3=corpus.input_blake3,
            n_alert_keys=corpus.n_input_versions,
            n_vehicle_keys=corpus.n_vehicle_keys,
            manifest_version=INPUT_MANIFEST_VERSION,
        ),
        artifacts=artifacts,
        feed=feed,
    )
    body = doc.to_json().encode()
    versioned = f"{VERSIONED_PROV_PREFIX}v{trained_at}.json"
    public_versioned = f"{PUBLIC_PROV_PREFIX}v{trained_at}.json"
    # Immutable versioned snapshots FIRST: the prov_ref every published artifact
    # carries points at `versioned`, so the lineage doc must be durable before any
    # live pointer names this run. The public versioned mirror is immutable and
    # may cache for a year; the private copy is never served publicly (index.ts
    # gates reads to v1/), so its cache header is moot and stays no-store.
    client.put_object(
        Bucket=bucket,
        Key=versioned,
        Body=body,
        ContentType="application/json",
        CacheControl="no-store",
    )
    client.put_object(
        Bucket=bucket,
        Key=public_versioned,
        Body=body,
        ContentType="application/json",
        CacheControl="public, max-age=31536000, immutable",
    )
    # The latest.json pointers move every run and stay no-store. They flip with
    # the other live pointers in the transactional caller's phase-2 flush (or now,
    # standalone), so a failed versioned write above aborts before any pointer
    # advertises this run.
    if pending is None:
        client.put_object(
            Bucket=bucket,
            Key=PROV_KEY,
            Body=body,
            ContentType="application/json",
            CacheControl="no-store",
        )
        client.put_object(
            Bucket=bucket,
            Key=PUBLIC_PROV_KEY,
            Body=body,
            ContentType="application/json",
            CacheControl="no-store",
        )
    else:
        pending.append(DeferredPointer(PROV_KEY, body, "no-store"))
        pending.append(DeferredPointer(PUBLIC_PROV_KEY, body, "no-store"))
    return versioned


SEGMENT_DWELL_KEY = "state/segment_dwell.json"
VERSIONED_SEGMENT_DWELL_PREFIX = "state/segment_dwell/"
# RETENTION: the versioned state/segment_dwell/ snapshots are kept forever.
# Same absence of a training.prune rule as the other publish_params sidecars
# above (only DATED_PREFIXES and PARAMS_PREFIX are policed) and the same
# reasoning: one small per-run pooled-dwell-curve doc, versioned by
# `trained_at` like state/params/, so a rollback of a given params.json
# (docs/params-rollback.md) always has this sidecar's matching version still
# available to pair with it.


def write_segment_dwell(
    client: S3Client,
    bucket: str,
    start_date: date,
    end_date: date,
    trained_at: int,
    through: frozenset[tuple[str, str, str]] | None,
    pending: list[DeferredPointer] | None = None,
) -> tuple[int, SegmentDwellStats]:
    """Write the per-segment dwell curves as their OWN R2 object (not folded
    into segment_params.json), hierarchically pooled leaf -> route -> system
    (training.segment_dwell) from the segment-scope movement regimes over this
    run's training window.

    Episodes are reconstructed from archive/vehicles through the identical
    regime clock (training.regime) the Worker runs online, counting only
    `through` from_stops so the curves and the published baseline describe the
    same segments. The Worker's own committed v1/movement_transitions stream is
    deliberately not read here — see _movement_dwell.

    Fail-soft like write_segment_params: an archive hiccup or an empty
    stream just skips the object, leaving the last good one, and never
    blocks the params publish. Returns (n_cells, stats) — stats is all-zero
    on skip.
    """
    empty_stats = SegmentDwellStats(
        n_cells_own=0, n_cells_route=0, n_cells_system=0, n_cells_skipped=0
    )
    try:
        from training.movement_backfill import reconstruct_movement_transitions

        transitions = reconstruct_movement_transitions(
            client=client,
            bucket=bucket,
            start_date=start_date,
            end_date=end_date,
            scope="segment",
            counts_from_stop=_stop_filter(through),
        )
        # Same censoring boundary as the route-level dwell fit: "now", clamped
        # to the requested window.
        _, end_epoch = aligned_window(start_date, end_date)
        window_end = min(int(datetime.now(UTC).timestamp()), end_epoch)
        cells, stats = build_segment_dwell(transitions, window_end=window_end)
        if not cells:
            print(
                "segment dwell skipped (no segment-scope transitions)",
                file=sys.stderr,
            )
            return 0, stats
        doc = {
            "schema_version": SCHEMA_VERSION,
            "trained_at": trained_at,
            "cells": cells,
        }
        body = json.dumps(doc).encode()
        versioned = f"{VERSIONED_SEGMENT_DWELL_PREFIX}v{trained_at}.json"
        _publish(
            client, bucket, SEGMENT_DWELL_KEY, versioned, body, "no-store", pending
        )
        return len(cells), stats
    except Exception as exc:
        print(f"segment dwell skipped ({exc})", file=sys.stderr)
        return 0, empty_stats


# --- publish plausibility gate -------------------------------------------
#
# The structural gates elsewhere in main (severity floor, empty movement
# baseline, MIN_DATA_DAYS span) all admit a degenerate-but-non-empty fit: a
# collapsed transition matrix, dwell quantiles pinned at 0, a non-finite
# service_mu. This gate compares the run's params doc against the currently-live
# one and refuses the pointer flip when a value bound is violated or an aggregate
# jumps out of band, so a bad fit leaves the last good params.json in place
# instead of shipping. Skipped with --skip-plausibility for a first publish /
# bootstrap, where there is no live blob to gate against.

# A self-loop this high makes a state absorbing: the chain never leaves it, so
# the transition structure has collapsed to a single regime. Real fits are capped
# well below this by _cap_self_loops (MAX_SELF_LOOP peaks at 0.975), so only a
# degenerate blob trips it.
COLLAPSE_SELF_LOOP = 0.999
# Per-route total-variation shift of the stationary regime mix vs the live blob.
# Half the mass moving to different states is a different model, not a refit.
MAX_STATIONARY_TV = 0.5
# Tolerated jump in the fraction of routes that inherited the global prior
# (n_routes - n_routes_trained). A surge means the window went thin and most
# routes degenerated to the prior — refuse rather than ship a mostly-prior blob.
MAX_FALLBACK_INCREASE = 0.25


def _emission_dicts(route_doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Every emission subdoc a route carries: the unconditioned set plus any
    per-tod-bin sets, so a value bound covers whichever the Worker will read."""
    out: list[dict[str, Any]] = []
    em = route_doc.get("emissions")
    if isinstance(em, dict):
        out.append(cast("dict[str, Any]", em))
    by_bin = route_doc.get("emissions_by_bin")
    if isinstance(by_bin, list):
        for e in cast("list[Any]", by_bin):
            if isinstance(e, dict):
                out.append(cast("dict[str, Any]", e))
    return out


def _stationary(transition: list[list[float]]) -> list[float] | None:
    """Stationary distribution of a row-stochastic matrix by power iteration, or
    None when the rows do not form a usable distribution."""
    n = len(transition)
    if n == 0 or any(len(row) != n for row in transition):
        return None
    pi = [1.0 / n] * n
    for _ in range(256):
        nxt = [0.0] * n
        for i, row in enumerate(transition):
            for j, p in enumerate(row):
                nxt[j] += pi[i] * p
        total = sum(nxt)
        if total <= 0 or not math.isfinite(total):
            return None
        nxt = [x / total for x in nxt]
        if max(abs(nxt[k] - pi[k]) for k in range(n)) < 1e-9:
            return nxt
        pi = nxt
    return pi


def _tv(a: list[float], b: list[float]) -> float:
    """Total-variation distance between two same-length distributions."""
    return 0.5 * sum(abs(x - y) for x, y in zip(a, b, strict=True))


def _dwell_pinned_zero(dwell: dict[str, Any]) -> bool:
    """True when every state's dwell quantiles are pinned at 0 (q25/median/q75 all
    zero) — a degenerate empirical fit with no spread, not a recovery curve."""
    entries: list[dict[str, Any]] = [
        cast("dict[str, Any]", v) for v in dwell.values() if isinstance(v, dict)
    ]
    if not entries:
        return False
    return all(
        (
            int(e.get("q25_sec", 0)),
            int(e.get("median_sec", 0)),
            int(e.get("q75_sec", 0)),
        )
        == (0, 0, 0)
        for e in entries
    )


def _fallback_fraction(doc: dict[str, Any]) -> float | None:
    """Fraction of routes that inherited the global prior instead of their own
    fit, or None when the doc does not record enough to say."""
    routes = cast("dict[str, Any]", doc.get("routes") or {})
    n = len(routes)
    if n == 0:
        return None
    corpus = cast("dict[str, Any]", doc.get("training_corpus") or {})
    trained = corpus.get("n_routes_trained")
    if trained is None:
        return None
    return max(0, n - int(trained)) / n


def implausible_params(new: dict[str, Any], live: dict[str, Any]) -> str | None:
    """A named reason to refuse `new` in favour of the currently-live `live`, or
    None when `new` is plausible.

    Two families of check. Value bounds catch a degenerate fit on its own terms —
    a non-finite emission, an absorbing transition matrix, dwell quantiles pinned
    at 0 — regardless of what was serving. Diff-vs-live checks catch a fit that is
    individually well-formed but lurches away from the running model: a stationary
    regime mix that half-moves, or a surge of routes collapsing to the global
    prior. The reason names the check and the offending route so an operator sees
    WHY the publish was refused, not just that it was."""
    new_routes = cast("dict[str, dict[str, Any]]", new.get("routes") or {})
    live_routes = cast("dict[str, dict[str, Any]]", live.get("routes") or {})

    for route, rp in new_routes.items():
        for em in _emission_dicts(rp):
            for field in (
                "poisson_lambda",
                "gamma_alpha",
                "gamma_beta",
                "advance_rate",
            ):
                values = cast("list[float]", em.get(field) or [])
                if any(not math.isfinite(float(v)) for v in values):
                    return f"non_finite:{route}:{field}"
        transition = cast("list[list[float]]", rp.get("transition") or [])
        if any(not math.isfinite(float(v)) for row in transition for v in row):
            return f"non_finite:{route}:transition"
        diag = [
            transition[i][i] for i in range(len(transition)) if i < len(transition[i])
        ]
        if diag and max(diag) >= COLLAPSE_SELF_LOOP:
            return f"collapsed_transition:{route}"
        dwell = rp.get("dwell_quantiles")
        if (
            isinstance(dwell, dict)
            and dwell
            and _dwell_pinned_zero(cast("dict[str, Any]", dwell))
        ):
            return f"degenerate_dwell:{route}"

    for route, rp in new_routes.items():
        live_rp = live_routes.get(route)
        if live_rp is None:
            continue
        new_pi = _stationary(cast("list[list[float]]", rp.get("transition") or []))
        live_pi = _stationary(
            cast("list[list[float]]", live_rp.get("transition") or [])
        )
        if (
            new_pi is not None
            and live_pi is not None
            and len(new_pi) == len(live_pi)
            and _tv(new_pi, live_pi) > MAX_STATIONARY_TV
        ):
            return f"stationary_shift:{route}"

    new_fb = _fallback_fraction(new)
    live_fb = _fallback_fraction(live)
    if (
        new_fb is not None
        and live_fb is not None
        and new_fb - live_fb > MAX_FALLBACK_INCREASE
    ):
        return "prior_fallback_surge"
    return None


def load_live_params(client: S3Client, bucket: str) -> dict[str, Any] | None:
    """The currently-live params.json for the plausibility gate to compare
    against, or None when there is none yet (first publish / bootstrap). A blob
    that is present but unreadable raises — a corrupt live pointer is not a silent
    skip."""
    try:
        body = get_object_bytes(client, bucket, PARAMS_KEY)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in {"NoSuchKey", "404"}:
            return None
        raise
    return cast("dict[str, Any]", json.loads(body))
