"""Per-route EM trainer with empirical-Bayes prior anchoring.

Each run:
  1. Pull the alerts archive over a window (default 14 days) from R2.
  2. Pool all routes into one corpus and fit a global HMM — the prior.
  3. For each route with enough data, fit again with `prior_params=global`
     and Dirichlet/Gamma/Beta pseudo-counts (`prior_strength`).
  4. Routes with thin data inherit the global prior as-is.
  5. Write state/params.json (live pointer) + state/params/v<epoch>.json
     (immutable per-run snapshot) — the Worker picks up params.json on its
     next cron tick; the versioned copies are the rollback trail.

Run with:
    murk exec -- python -m training.train_em [--days 14] [--start/--end DATE]
        [--routes A,C,E] [--min-ticks N] [--prior-strength 100] [--dry-run]

params.json records what it took to produce it — provenance.code_sha, the
hyperparams block (resolved window + prior_strength + min_ticks + routes), and
training_corpus.input_blake3. Against the immutable archive, re-running this
tool at that code_sha with that hyperparams block reproduces the version.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from momentarily.hmm import EmissionParams, HMMParams, Observation, fit_em
from training.drift import build_input_profile
from training.dwell import (
    DwellQuantiles,
    compute_dwell_quantiles,
    compute_dwell_quantiles_by_alert,
    compute_dwell_quantiles_by_cause,
)
from training.load import TICK_SECONDS, TickObservation, fill_quiet_ticks
from training.load_r2 import (
    advance_baseline_to_json,
    build_movement_series_by_direction,
    build_service_series,
    build_tick_observations,
    compute_advance_baseline,
    compute_advance_baseline_by_route,
    compute_baseline,
    compute_schedule_rate,
    fetch_objects,
    fetch_trip_update_metrics,
    fetch_vehicle_metrics,
    input_manifest_hash,
    list_alert_keys,
    presence_mask_from_predictions,
    schedule_rate_to_json,
    service_baseline_to_json,
)
from training.provenance import code_provenance
from training.r2_client import R2Config, load_config, make_client
from training.run_filter import BOOTSTRAP_PARAMS
from training.survival import loglogistic_tail

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


PARAMS_KEY = "state/params.json"
# Immutable per-run snapshots live under this prefix as v<trained_at>.json.
VERSIONED_PARAMS_PREFIX = "state/params/"
SCHEMA_VERSION = "1"

# A route needs at least this many ticks of data to fit per-route — under that,
# we fall back to the global prior.
MIN_TICKS_PER_ROUTE = 288  # one day at the 5-min grid


@dataclass(frozen=True)
class CorpusStats:
    """Audit metadata about the archive window a run actually trained on."""

    start_tick: int
    end_tick: int
    n_observations: int  # real (alert-bearing) tick-observations, pre-quiet-fill
    n_input_versions: int = 0  # archived alert-version objects that fed the fit
    input_blake3: str = ""  # BLAKE3 over those object keys — lineage fingerprint

    @property
    def span_seconds(self) -> int:
        return self.end_tick - self.start_tick


# EM on a thin or mostly-quiet corpus drives transition self-loops toward 1.0,
# which pins the forward filter so a route can never leave a regime. Cap the
# diagonal, and refuse to publish at all under a week of archive. The original
# bound was two weeks; once the EM variance/Bernoulli floors landed (momentarily-p8y)
# the dominant risk of thin data — degenerate emissions — was no longer in
# play, so we relaxed the gate. _cap_self_loops still bounds the transition
# self-loops independently. See momentarily-625.
#
# Per-state ceilings, set from the actual median regime dwell in the
# v1/regime_transitions stream (14d): normal ~135min, disrupted ~45min,
# suspended ~50min. A single 0.97 cap modeled every regime as ~114min, making
# the filter 2.5x too pessimistic about recovery from disruption (it predicted
# 17% recovered-in-30min against 35% actual). self_loop = exp(ln(0.5) / (median
# dwell minutes / 5)) reproduces each regime's real persistence. Indexed
# (normal, disrupted, suspended). See momentarily-2jt.
MAX_SELF_LOOP: tuple[float, float, float] = (0.975, 0.93, 0.93)
MIN_DATA_DAYS = 5


def _cap_self_loops(
    params: HMMParams, max_self: tuple[float, float, float] = MAX_SELF_LOOP
) -> HMMParams:
    """Clamp each transition row's diagonal to its per-state ceiling `max_self[s]`,
    redistributing the freed mass across that row's off-diagonal entries
    (proportionally, or evenly when they're all zero)."""
    rows: list[tuple[float, float, float]] = []
    for s in range(3):
        row = list(params.transition[s])
        cap = max_self[s]
        if row[s] <= cap:
            rows.append((row[0], row[1], row[2]))
            continue
        freed = row[s] - cap
        row[s] = cap
        off = [j for j in range(3) if j != s]
        off_sum = sum(row[j] for j in off)
        for j in off:
            share = row[j] / off_sum if off_sum > 0 else 1.0 / len(off)
            row[j] += freed * share
        rows.append((row[0], row[1], row[2]))
    return HMMParams(
        transition=tuple(rows),
        initial=params.initial,
        emissions=params.emissions,
        emissions_by_bin=params.emissions_by_bin,
    )


def _aligned_window(start: date, end: date) -> tuple[int, int]:
    """Tick-aligned UTC window covering [start, end+1day)."""
    start_dt = datetime(start.year, start.month, start.day, tzinfo=UTC)
    end_dt = datetime(end.year, end.month, end.day, tzinfo=UTC) + timedelta(days=1)
    start_epoch = (int(start_dt.timestamp()) // TICK_SECONDS) * TICK_SECONDS
    end_epoch = (int(end_dt.timestamp()) // TICK_SECONDS) * TICK_SECONDS
    return start_epoch, end_epoch


def load_series_by_route(
    cfg: R2Config,
    start: date,
    end: date,
) -> tuple[dict[str, list[Observation]], CorpusStats, dict[str, Any]]:
    """Single R2 pass: fetch alerts, build per-route quiet-filled series.

    Returns (by_route, corpus, input_profile). `corpus` describes the *actual*
    observed ticks before quiet-filling — fill_quiet_ticks pads every series to
    the requested window, so series length can't tell us how much real archive
    we have. The publish gate and the params.json audit block both need the
    unpadded view. `input_profile` is the emission-channel reference profile (over
    the real ticks) that the eval job's drift check compares against.
    """
    client = make_client(cfg)
    # Hash the exact key set we fetch — the manifest fingerprint and the training
    # input are then guaranteed to describe the same objects.
    keys = list_alert_keys(client, cfg.bucket, start, end)
    bodies = fetch_objects(client, cfg.bucket, keys)
    input_blake3 = input_manifest_hash(keys)
    # Mask the reconstruction against what the live Worker actually saw active,
    # so an alert that left the feed without a superseding version doesn't train
    # as still-active to its active_period end. See momentarily-1a7. Degrades to
    # the raw reconstruction if predictions are unavailable (e.g. pre-stream).
    mask = None
    try:
        from training.eval import load_predictions

        predictions = load_predictions(client, cfg.bucket, start, end)
        mask = presence_mask_from_predictions(predictions)
    except Exception as exc:
        print(f"presence-mask: prediction load failed ({exc}); raw reconstruction")
    all_ticks = build_tick_observations(bodies, active_mask=mask)
    if not all_ticks:
        return (
            {},
            CorpusStats(
                start_tick=0,
                end_tick=0,
                n_observations=0,
                n_input_versions=len(keys),
                input_blake3=input_blake3,
            ),
            {},
        )

    input_profile = build_input_profile(all_ticks)
    ticks = [t.tick for t in all_ticks]
    corpus = CorpusStats(
        start_tick=min(ticks),
        end_tick=max(ticks),
        n_observations=len(all_ticks),
        n_input_versions=len(keys),
        input_blake3=input_blake3,
    )

    start_tick, end_tick_excl = _aligned_window(start, end)
    last_tick = end_tick_excl - TICK_SECONDS

    by_route: dict[str, list[Observation]] = {}
    seen_routes = {t.route_id for t in all_ticks}
    for route in sorted(seen_routes):
        filled: list[TickObservation] = fill_quiet_ticks(
            all_ticks, route, start_tick=start_tick, end_tick=last_tick
        )
        by_route[route] = [t.observation for t in filled]
    return by_route, corpus, input_profile


def train(
    series_by_route: dict[str, list[Observation]],
    *,
    prior_strength: float = 100.0,
    min_ticks: int = MIN_TICKS_PER_ROUTE,
    advance_priors: dict[str, float] | None = None,
) -> tuple[HMMParams, dict[str, HMMParams]]:
    """Returns (global_prior, per_route_params). Doesn't touch R2.

    `advance_priors` maps a route to its measured normal-state advance rate; when
    present, that route's prior carries the measured rate instead of the
    hardcoded default, so the movement emission's normal state anchors on the
    line's real cross-tick advance fraction. Routes without a measured baseline
    keep the global prior's default.
    """
    if not series_by_route:
        raise ValueError("no observations to train on")
    advance_priors = advance_priors or {}

    pooled: list[Observation] = []
    for series in series_by_route.values():
        pooled.extend(series)
    global_prior, _ = fit_em(pooled, BOOTSTRAP_PARAMS, max_iterations=50)
    # fit_em returns canonical state order (normal/disrupted/suspended), so the
    # per-state self-loop caps land on the regimes they were tuned for. Capping
    # before canonicalization applied them to arbitrary EM indices. See
    # momentarily-vk0.7.
    global_prior = _cap_self_loops(global_prior)

    out: dict[str, HMMParams] = {}
    for route, series in series_by_route.items():
        rate = advance_priors.get(route)
        prior = (
            _apply_advance_prior(global_prior, rate)
            if rate is not None
            else global_prior
        )
        if len(series) < min_ticks:
            out[route] = prior
            continue
        fitted, _ = fit_em(
            series,
            prior,
            max_iterations=30,
            prior_params=prior,
            prior_strength=prior_strength,
        )
        out[route] = _cap_self_loops(fitted)
    return global_prior, out


def _apply_advance_prior(params: HMMParams, normal_rate: float) -> HMMParams:
    """Return `params` with the normal state's advance-rate prior set to the
    measured route baseline `normal_rate`, leaving disrupted/suspended alone.
    The flat emissions carry it; per-bin emissions, if ever present, get the
    same override so the prior survives a TOD-conditioned model."""

    def override(em: EmissionParams) -> EmissionParams:
        a = em.advance_rate
        return replace(em, advance_rate=(normal_rate, a[1], a[2]))

    by_bin = (
        tuple(override(e) for e in params.emissions_by_bin)
        if params.emissions_by_bin is not None
        else None
    )
    return replace(
        params, emissions=override(params.emissions), emissions_by_bin=by_bin
    )


def _params_to_json(params: HMMParams) -> dict[str, Any]:
    """Serialize HMMParams to the loose schema the Worker reads."""
    emissions = asdict(params.emissions)
    body: dict[str, Any] = {
        "transition": [list(row) for row in params.transition],
        "initial": list(params.initial),
        "emissions": emissions,
    }
    if params.emissions_by_bin is not None:
        body["emissions_by_bin"] = [asdict(e) for e in params.emissions_by_bin]
    return body


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
    hyperparams: dict[str, Any] | None = None,
    input_profile: dict[str, Any] | None = None,
    movement_baseline: dict[str, Any] | None = None,
    service_baseline: dict[str, Any] | None = None,
    schedule_rate: dict[str, Any] | None = None,
    trained_at: int | None = None,
) -> str:
    """Write the live params pointer plus an immutable versioned snapshot.

    The Worker reads state/params.json; the state/params/v<epoch>.json copies
    give us a per-run rollback trail. Returns the versioned key.
    """
    trained_at = trained_at or int(datetime.now(UTC).timestamp())
    routes_doc = {r: _params_to_json(p) for r, p in per_route.items()}
    if dwell_quantiles:
        # Merge per-route empirical dwell into the same per-route subdoc — the
        # Worker reads it as an optional sibling of `emissions`/`transition`.
        for r, by_state in dwell_quantiles.items():
            if r in routes_doc:
                routes_doc[r]["dwell_quantiles"] = by_state
    if dwell_quantiles_by_alert:
        # Cause-segmented dwell, layered on top of the (route, state) aggregate.
        # The Worker prefers (route, state, alert_type) and falls back to the
        # aggregate above when a cause cell is absent. See momentarily-alu.
        for r, by_state_alert in dwell_quantiles_by_alert.items():
            if r in routes_doc:
                routes_doc[r]["dwell_quantiles_by_alert"] = by_state_alert
    if dwell_quantiles_by_cause:
        # Cause-CATEGORY dwell for the episode-recovery grader (Episode.cause is a
        # coarse category, not a raw alert_type). The Worker ignores this key
        # (zod strips it); scorecard.dwell_lookup_from_params reads it so the
        # grade stops silently falling back to the (route, state) aggregate. 1a6.
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
            "n_input_versions": corpus.n_input_versions,
            "input_blake3": corpus.input_blake3,
        },
        "routes": routes_doc,
    }
    # Per-(route, direction, tod_bin) advance-rate baseline the Worker needs live
    # to gate and score the movement channel. Top-level (not per-route) so 8zp's
    # assigned_n service baseline can sit beside it under the same delivery.
    if movement_baseline:
        doc["movement_baseline"] = movement_baseline
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
    body = json.dumps(doc).encode()
    versioned_key = f"{VERSIONED_PARAMS_PREFIX}v{trained_at}.json"
    for key in (PARAMS_KEY, versioned_key):
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            CacheControl="public, max-age=300, s-maxage=900",
        )
    return versioned_key


def _movement_baseline(
    cfg: R2Config,
    client: S3Client,
    start_date: date,
    end_date: date,
) -> tuple[dict[str, Any], int, dict[str, float]]:
    """Advance-rate baseline over the training window, in the two shapes the
    pipeline needs from one vehicle-archive fetch:

    - the per-(route, direction, tod) baseline, serialized for params.json
      delivery to the Worker's movement posterior;
    - the per-route normal advance rate that seeds each route's EM prior.

    Uses the explicit training window — fetch_vehicle_metrics defaults to
    yesterday..today, too narrow for a stable prior. Fail-soft: any
    vehicle-archive error returns empty baselines so a movement hiccup never
    blocks the params publish (the channel is optional and back-compat). Returns
    (serialized, n_cells, route_advance_rates)."""
    try:
        bodies = fetch_vehicle_metrics(
            cfg, start_date=start_date, end_date=end_date, client=client
        )
        series = build_movement_series_by_direction(bodies)
        baseline = compute_advance_baseline(series)
        route_rates = compute_advance_baseline_by_route(series)
        return advance_baseline_to_json(baseline), len(baseline), route_rates
    except Exception as exc:
        print(f"movement baseline skipped ({exc})", file=sys.stderr)
        return {}, 0, {}


def _service_baseline(
    cfg: R2Config,
    client: S3Client,
    start_date: date,
    end_date: date,
) -> tuple[dict[str, Any], int, dict[str, Any], int]:
    """Per-(route, tod) assigned_n baseline AND per-(route, schedule_bin)
    scheduled-presence rate over the training window, from one trip-updates fetch,
    serialized for params.json. The Worker forms the service ratio from the first
    and splits suspended vs not_scheduled with the second. Fail-soft: a trip-updates
    archive error returns empty sidecars (both optional and back-compat). Returns
    (baseline, n_baseline_cells, schedule, n_schedule_cells)."""
    try:
        bodies = fetch_trip_update_metrics(
            cfg, start_date=start_date, end_date=end_date, client=client
        )
        baseline = compute_baseline(build_service_series(bodies))
        rate = compute_schedule_rate(bodies)
        return (
            service_baseline_to_json(baseline),
            len(baseline),
            schedule_rate_to_json(rate),
            len(rate),
        )
    except Exception as exc:
        print(f"service baseline skipped ({exc})", file=sys.stderr)
        return {}, 0, {}, 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-route EM trainer")
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="trailing-window size when --start is unset",
    )
    parser.add_argument(
        "--start", help="window start date YYYY-MM-DD (overrides --days)"
    )
    parser.add_argument("--end", help="window end date YYYY-MM-DD (default: today UTC)")
    parser.add_argument(
        "--routes", help="comma-separated route whitelist (default: all observed)"
    )
    parser.add_argument(
        "--min-ticks",
        type=int,
        default=MIN_TICKS_PER_ROUTE,
        help="routes with fewer observations inherit the global prior",
    )
    parser.add_argument(
        "--prior-strength",
        type=float,
        default=100.0,
        help="pseudo-counts strength for per-route prior anchor (in tick units)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print learned params instead of writing to R2",
    )
    parser.add_argument(
        "--allow-empty-baseline",
        action="store_true",
        help="publish even if the movement advance-baseline is empty (0 cells); "
        "the movement-primary condition stays off. Default: refuse to publish.",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    end_date = date.fromisoformat(args.end) if args.end else datetime.now(UTC).date()
    start_date = (
        date.fromisoformat(args.start)
        if args.start
        else end_date - timedelta(days=args.days - 1)
    )
    series, corpus, input_profile = load_series_by_route(cfg, start_date, end_date)
    if not series:
        print("no observations in archive — skipping training", file=sys.stderr)
        return 1

    if args.routes:
        whitelist = {r.strip() for r in args.routes.split(",") if r.strip()}
        series = {r: s for r, s in series.items() if r in whitelist}
        if not series:
            print(
                f"none of --routes {sorted(whitelist)} present in archive",
                file=sys.stderr,
            )
            return 1

    client = make_client(cfg)
    # Movement advance-rate baseline over the training window: the per-cell form
    # ships to the Worker in params.json; the per-route rates seed each route's
    # normal-state advance prior below (fail-soft — see helper).
    movement_baseline, n_baseline_cells, route_advance_rates = _movement_baseline(
        cfg, client, start_date, end_date
    )
    if n_baseline_cells == 0 and not (args.dry_run or args.allow_empty_baseline):
        print(
            "ERROR: movement advance-baseline is EMPTY (0 cells) -- refusing to "
            "publish params that would silently disable the movement-primary "
            "condition (every route reads 'unknown'). Check vehicle-archive "
            "by_direction coverage over the training window, or pass "
            "--allow-empty-baseline to publish an alerts-only params set.",
            file=sys.stderr,
        )
        return 1
    if n_baseline_cells == 0:
        print(
            "WARNING: movement advance-baseline is EMPTY (0 cells) -- the "
            "movement-primary condition will publish 'unknown' for every route.",
            file=sys.stderr,
        )
    # Assigned_n service baseline (per route, tod) for the Worker's service
    # emission channel — it divides live assigned_n by this to form the ratio.
    service_baseline, n_service_cells, schedule_rate, n_schedule_cells = (
        _service_baseline(cfg, client, start_date, end_date)
    )

    global_prior, per_route = train(
        series,
        prior_strength=args.prior_strength,
        min_ticks=args.min_ticks,
        advance_priors=route_advance_rates,
    )

    # Empirical dwell quantiles from the regime_transitions stream over the
    # same window. Cells below MIN_SAMPLES_FOR_EMPIRICAL fall back to the
    # geometric dwell in the Worker — no-op if the stream is empty.
    from training.eval import load_transitions

    transitions = load_transitions(client, cfg.bucket, start_date, end_date)
    # Censoring boundary for still-open regimes: "now", clamped to the
    # requested window so a backdated --end doesn't fabricate giant censored
    # durations from regimes that actually ended after the window.
    _, end_epoch = _aligned_window(start_date, end_date)
    window_end = min(int(datetime.now(UTC).timestamp()), end_epoch)
    dwell_q = compute_dwell_quantiles(
        transitions, window_end=window_end, tail_fn=loglogistic_tail
    )
    dwell_q_by_alert = compute_dwell_quantiles_by_alert(
        transitions, tail_fn=loglogistic_tail
    )
    dwell_q_by_cause = compute_dwell_quantiles_by_cause(
        transitions, tail_fn=loglogistic_tail
    )
    n_dwell_cells = sum(len(by_state) for by_state in dwell_q.values())
    n_dwell_alert_cells = sum(
        len(by_alert)
        for by_state in dwell_q_by_alert.values()
        for by_alert in by_state.values()
    )
    n_dwell_cause_cells = sum(
        len(by_cause)
        for by_state in dwell_q_by_cause.values()
        for by_cause in by_state.values()
    )

    if args.dry_run:
        dry_routes = {r: _params_to_json(p) for r, p in per_route.items()}
        for r, by_state in dwell_q.items():
            if r in dry_routes:
                dry_routes[r]["dwell_quantiles"] = by_state
        for r, by_state_alert in dwell_q_by_alert.items():
            if r in dry_routes:
                dry_routes[r]["dwell_quantiles_by_alert"] = by_state_alert
        for r, by_state_cause in dwell_q_by_cause.items():
            if r in dry_routes:
                dry_routes[r]["dwell_quantiles_by_cause"] = by_state_cause
        print(
            json.dumps(
                {
                    "global_prior": _params_to_json(global_prior),
                    "routes": dry_routes,
                    "dwell_cells": n_dwell_cells,
                    "dwell_alert_cells": n_dwell_alert_cells,
                    "dwell_cause_cells": n_dwell_cause_cells,
                    "baseline_cells": n_baseline_cells,
                    "service_cells": n_service_cells,
                },
                indent=2,
            )
        )
        return 0

    if corpus.span_seconds < MIN_DATA_DAYS * 86_400:
        print(
            f"archive spans {corpus.span_seconds / 86_400:.1f}d "
            f"(< {MIN_DATA_DAYS}d minimum) — refusing to publish; thin data "
            "overfits transition self-loops",
            file=sys.stderr,
        )
        return 1

    n_routes_trained = sum(1 for s in series.values() if len(s) >= args.min_ticks)
    # The knobs that determine the fit — with code_sha + the immutable archive
    # these make a params_version re-derivable. Window is recorded as resolved
    # dates so it reproduces regardless of when --days was relative to.
    hyperparams = {
        "window_start": start_date.isoformat(),
        "window_end": end_date.isoformat(),
        "prior_strength": args.prior_strength,
        "min_ticks": args.min_ticks,
        "routes": sorted(args.routes.split(",")) if args.routes else None,
    }
    versioned_key = write_params(
        client,
        cfg.bucket,
        per_route,
        corpus=corpus,
        n_routes_trained=n_routes_trained,
        dwell_quantiles=dwell_q,
        dwell_quantiles_by_alert=dwell_q_by_alert,
        dwell_quantiles_by_cause=dwell_q_by_cause,
        hyperparams=hyperparams,
        input_profile=input_profile,
        movement_baseline=movement_baseline,
        service_baseline=service_baseline,
        schedule_rate=schedule_rate,
    )
    print(
        f"published {PARAMS_KEY} + {versioned_key}: "
        f"{n_routes_trained}/{len(per_route)} routes fitted "
        f"(prior_strength={args.prior_strength}, dwell_cells={n_dwell_cells}, "
        f"dwell_alert_cells={n_dwell_alert_cells}, "
        f"dwell_cause_cells={n_dwell_cause_cells}, baseline_cells={n_baseline_cells}, "
        f"service_cells={n_service_cells}, schedule_cells={n_schedule_cells})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
