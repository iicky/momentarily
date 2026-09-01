"""Load alert observations from the R2 archive.

The Worker stores one R2 object per (alert_id, updated_at) version. To train
the HMM we need per-tick observations — alert_count, severity_sum, etc., on
the 5-minute cron grid. Each alert version has an `active_period` (start/end
epochs); we walk the grid and count which versions were live at each tick.

Public surface mirrors training/load.py:
    load_route_series(route_id, start_date=None, end_date=None) -> list[TickObservation]

The Observation has tod_bin populated. Uses the same TickObservation type as
the local loader so run_filter.py can swap with a flag.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Callable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol, cast
from zoneinfo import ZoneInfo

from blake3 import blake3

from momentarily.hmm import Observation, schedule_bin, tod_bin
from momentarily.mapping import is_planned_work_id
from training.hierarchical import PooledCell, partially_pool
from training.load import TICK_SECONDS, TickObservation
from training.r2_client import R2Config, get_object_bytes, load_config, make_client

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


class PredictionLike(Protocol):
    """Structural type for the prediction fields the presence mask reads (see
    training.eval.PredictionRecord). A Protocol so load_r2 doesn't import eval —
    eval imports load_r2, and a mutual TYPE_CHECKING import confuses the checker.
    Read-only properties so the frozen PredictionRecord dataclass satisfies it."""

    @property
    def ts(self) -> int: ...
    @property
    def route(self) -> str: ...
    @property
    def primary_alert_type(self) -> str | None: ...


_SORT_ORDER_RE = re.compile(r":(\d+)$")


def _snap_tick(epoch: int) -> int:
    return (epoch // TICK_SECONDS) * TICK_SECONDS


def date_range(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def list_keys(client: S3Client, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents") or []:
            key = obj.get("Key")
            if key is not None:
                keys.append(key)
        if not resp.get("IsTruncated"):
            return keys
        token = resp.get("NextContinuationToken")


def _fetch_object(client: S3Client, bucket: str, key: str) -> dict[str, Any]:
    body = get_object_bytes(client, bucket, key)
    return cast(dict[str, Any], json.loads(body))


def list_alert_keys(client: S3Client, bucket: str, start: date, end: date) -> list[str]:
    """Every alert-version object key in the [start, end] window, in list order."""
    keys: list[str] = []
    for d in date_range(start, end):
        keys.extend(list_keys(client, bucket, f"archive/alerts/{d.isoformat()}/"))
    return keys


def fetch_objects(
    client: S3Client, bucket: str, keys: list[str]
) -> list[dict[str, Any]]:
    """Parallel GET of the given keys — R2 happily handles tens of concurrent GETs."""

    def _fetch(k: str) -> dict[str, Any]:
        return _fetch_object(client, bucket, k)

    bodies: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for body in pool.map(_fetch, keys):
            bodies.append(body)
    return bodies


def input_manifest_hash(keys: list[str]) -> str:
    """BLAKE3 over the sorted object keys that fed a fit.

    The archive is immutable and keys are timestamped, so the sorted key set is a
    deterministic fingerprint of exactly which feed snapshots trained the model —
    re-listing the same window reproduces the same digest. Empty key set hashes
    the empty string."""
    h = blake3()
    for k in sorted(keys):
        h.update(k.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def fetch_alert_versions(
    config: R2Config | None = None,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    client: S3Client | None = None,
) -> list[dict[str, Any]]:
    """Pull every alert version object in the [start_date, end_date] window.

    Defaults to "yesterday through today" if no dates provided.
    """
    cfg = config or load_config()
    client = client or make_client(cfg)

    today = datetime.now(UTC).date()
    start = start_date or (today - timedelta(days=1))
    end = end_date or today

    keys = list_alert_keys(client, cfg.bucket, start, end)
    return fetch_objects(client, cfg.bucket, keys)


def _sort_order(entity: dict[str, Any]) -> int:
    selector = cast(
        dict[str, Any], entity.get("transit_realtime.mercury_entity_selector") or {}
    )
    raw = selector.get("sort_order")
    if not isinstance(raw, str):
        return 0
    match = _SORT_ORDER_RE.search(raw)
    return int(match.group(1)) if match else 0


def _alert_type(alert_payload: dict[str, Any]) -> str:
    mercury = cast(
        dict[str, Any], alert_payload.get("transit_realtime.mercury_alert") or {}
    )
    return str(mercury.get("alert_type") or "")


@dataclass(frozen=True)
class PresenceMask:
    """Per-(route, tick) live alert presence taken from the v1/predictions stream.

    The archive dedupes by (alert_id, updated_at) and writes no marker when an
    alert leaves the feed, so build_tick_observations fills an alert's whole
    active_period — over-extending past when the live Worker actually saw it
    (open-ended/early-cleared alerts run to corpus_end). The predictions stream
    records, per (route, tick), whether the live Worker counted any active alert
    (primary_alert_type non-null). Intersecting against it drops the
    hallucinated tail.

    `active` is the set of (route, tick) the Worker saw active; `covered` is
    every tick the stream spans. A cell is only dropped when its tick is covered
    but not active — ticks the stream never saw (pre-stream history, write-gap
    skips) fall back to the raw reconstruction untouched, so masking can only
    remove over-extension, never under-count a real disruption.
    """

    active: frozenset[tuple[str, int]]
    covered: frozenset[int]

    def covers(self, tick: int) -> bool:
        return tick in self.covered

    def is_active(self, route_id: str, tick: int) -> bool:
        return (route_id, tick) in self.active


def presence_mask_from_predictions(
    predictions: Sequence[PredictionLike],
) -> PresenceMask:
    """Build a PresenceMask from loaded prediction rows. primary_alert_type is
    non-null iff the live Worker counted an active alert on that route at that
    tick (worker/src/index.ts). Snapped with the reconstruction's floor grid so
    the keys line up with build_tick_observations' ticks."""
    active: set[tuple[str, int]] = set()
    covered: set[int] = set()
    for p in predictions:
        tick = _snap_tick(p.ts)
        covered.add(tick)
        if p.primary_alert_type is not None:
            active.add((p.route, tick))
    return PresenceMask(active=frozenset(active), covered=frozenset(covered))


def build_tick_observations(
    bodies: list[dict[str, Any]],
    *,
    corpus_end: int | None = None,
    active_mask: PresenceMask | None = None,
) -> list[TickObservation]:
    """Reconstruct per-(route, tick) observations from alert-version events.

    For each alert version, walk its active_period start..end on the tick grid
    and add it to that (route, tick) bucket. Sort_order and alert_type come
    from the entity row matching the route within the version's payload.

    `corpus_end` caps any open-ended active_period so we don't extend alerts
    forever. Defaults to the max observed_at across all bodies + one day —
    a reasonable upper bound for "still active at corpus end."

    `active_mask`, when given, drops (route, tick) cells the live Worker never
    saw active — correcting the archive's over-extension past feed presence.
    """
    if not bodies:
        return []

    if corpus_end is None:
        max_observed = max(int(b.get("observed_at") or 0) for b in bodies)
        corpus_end = max_observed + 86_400  # one day past the latest observation

    # bucket[tick][route_id] = {alert_id: (sort_order, alert_type)}
    bucket: dict[int, dict[str, dict[str, tuple[int, str]]]] = {}
    masked_out = 0  # cells the presence mask dropped (diagnostic)
    kept_active = 0  # cells written through

    # All observed_at values per alert_id, sorted — so we can clamp version
    # windows at the start of the *next* version, not just the latest.
    versions_by_alert: dict[str, list[int]] = {}
    for body in bodies:
        alert_envelope = cast(dict[str, Any], body.get("alert") or {})
        alert_id = alert_envelope.get("id")
        if not isinstance(alert_id, str):
            continue
        observed_at = int(body.get("observed_at") or 0)
        versions_by_alert.setdefault(alert_id, []).append(observed_at)
    for arr in versions_by_alert.values():
        arr.sort()

    for body in bodies:
        alert_envelope = cast(dict[str, Any], body.get("alert") or {})
        alert_id = alert_envelope.get("id")
        if not isinstance(alert_id, str):
            continue
        observed_at = int(body.get("observed_at") or 0)
        inner = cast(dict[str, Any], alert_envelope.get("alert") or {})
        alert_type = _alert_type(inner)

        # Active window for this version
        periods = cast(list[Any], inner.get("active_period") or [])
        if periods and isinstance(periods[0], dict):
            period0 = cast(dict[str, Any], periods[0])
            start = int(period0.get("start") or observed_at)
            end_raw = period0.get("end")
            # Open-ended period → clamp to corpus_end so we don't generate
            # billions of ticks for a "still active" alert.
            end = int(end_raw) if end_raw else corpus_end
        else:
            start = observed_at
            end = corpus_end

        # Clamp at the next version's observed_at for this alert — that
        # version supersedes us and will populate the bucket from there on.
        versions = versions_by_alert[alert_id]
        next_idx = next((i for i, t in enumerate(versions) if t > observed_at), None)
        if next_idx is not None:
            end = min(end, versions[next_idx])

        # Hard cap: never go past corpus_end (defensive belt + suspenders)
        end = min(end, corpus_end)

        first_tick = _snap_tick(start)
        last_tick = _snap_tick(end)
        if last_tick < first_tick:
            continue

        # Per-route this alert mentions
        informed = cast(list[Any], inner.get("informed_entity") or [])
        route_entities: list[dict[str, Any]] = [
            entity
            for entity in (
                cast(dict[str, Any], e) for e in informed if isinstance(e, dict)
            )
            if entity.get("route_id")
        ]
        if not route_entities:
            continue

        for entity in route_entities:
            route_id = entity["route_id"]
            if not isinstance(route_id, str):
                continue
            sort_order = _sort_order(entity)
            tick = first_tick
            while tick <= last_tick:
                if (
                    active_mask is not None
                    and active_mask.covers(tick)
                    and not active_mask.is_active(route_id, tick)
                ):
                    # Live Worker saw no alert on this route here — the archived
                    # active_period over-extended past feed presence; drop it.
                    masked_out += 1
                    tick += TICK_SECONDS
                    continue
                tick_bucket = bucket.setdefault(tick, {})
                route_bucket = tick_bucket.setdefault(route_id, {})
                route_bucket.setdefault(alert_id, (sort_order, alert_type))
                kept_active += 1
                tick += TICK_SECONDS

    if active_mask is not None and (masked_out or kept_active):
        total = masked_out + kept_active
        pct = 100.0 * masked_out / total
        print(
            f"presence-mask: dropped {masked_out}/{total} ({pct:.1f}%) "
            f"over-extended alert-active cells"
        )

    out: list[TickObservation] = []
    for tick in sorted(bucket):
        for route_id, alerts in bucket[tick].items():
            # Planned/scheduled work (lmm:planned_work:*) drops out of the HMM
            # disruption observation so the filter reads quiet; real-time alerts
            # and any other id are counted. Mirrors load.py + derive.ts.
            counted = [
                (so, at)
                for aid, (so, at) in alerts.items()
                if not is_planned_work_id(aid)
            ]
            types = [at for _so, at in counted]
            obs = Observation(
                alert_count=len(counted),
                severity_sum=sum(so for so, _at in counted),
                has_suspended_alert=_match(
                    types,
                    ("Suspend", "No Trains"),
                    exclude_prefix="Planned -",
                ),
                has_delays=_match(
                    types, ("Delays", "Severe Delays"), exclude_prefix="Planned -"
                ),
                has_service_change=_match(
                    types,
                    (
                        "Service Change",
                        "Trains Rerouted",
                        "Reroute",
                        "Stops Skipped",
                        "Express to Local",
                        "Local to Express",
                    ),
                    exclude_prefix="Planned -",
                ),
                has_planned=any(at.startswith("Planned -") for at in types),
                tod_bin=tod_bin(tick),
            )
            out.append(
                TickObservation(
                    route_id=route_id,
                    tick=tick,
                    observation=obs,
                    disruptive_types=tuple(types),
                )
            )
    return out


def _match(
    types: list[str],
    needles: tuple[str, ...],
    *,
    exclude_prefix: str | None = None,
) -> bool:
    for at in types:
        if exclude_prefix and at.startswith(exclude_prefix):
            continue
        if any(needle in at for needle in needles):
            return True
    return False


def load_route_series_r2(
    route_id: str,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    config: R2Config | None = None,
) -> list[TickObservation]:
    """End-to-end: pull R2 archive, build observations, return one route's series."""
    bodies = fetch_alert_versions(config, start_date=start_date, end_date=end_date)
    obs = build_tick_observations(bodies)
    series = [o for o in obs if o.route_id == route_id]
    if not series:
        return []

    # Fill quiet ticks the same way load.py does, so the HMM filter sees a
    # contiguous grid (gaps in coverage mean "no alerts active," not "missing").
    first_tick = series[0].tick
    last_tick = series[-1].tick
    by_tick = {o.tick: o for o in series}
    out: list[TickObservation] = []
    tick = first_tick
    while tick <= last_tick:
        if tick in by_tick:
            out.append(by_tick[tick])
        else:
            out.append(
                TickObservation(
                    route_id=route_id,
                    tick=tick,
                    observation=Observation(
                        alert_count=0,
                        severity_sum=0,
                        has_suspended_alert=False,
                        tod_bin=tod_bin(tick),
                    ),
                )
            )
        tick += TICK_SECONDS
    return out


# --- Trip-updates service metric: independent recovery truth ---
#
# The Worker archives a compact per-route service metric each tick at
# archive/trip_updates/<date>/<observed_at>.json:
#   {observed_at, fresh_feeds, rows: {route: {assigned_n, trips_n, ...}}}
# assigned_n counts NYCT-assigned (dispatched, running) trains on a route — a
# signal orthogonal to both the alerts feed and the HMM argmax, so it gives an
# INDEPENDENT recovery truth (vs eval.recovery_metrics, which grades against the
# model's own transitions). It is service LEVEL, not service quality — a strong
# proxy, not ground truth (true recovery would need GTFS trip-update arrivals).


def fetch_trip_update_metrics(
    config: R2Config | None = None,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    client: S3Client | None = None,
) -> list[dict[str, Any]]:
    """Pull every archived trip-updates service-metric snapshot in the window
    (one object per tick). Defaults to yesterday-through-today."""
    cfg = config or load_config()
    client = client or make_client(cfg)

    today = datetime.now(UTC).date()
    start = start_date or (today - timedelta(days=1))
    end = end_date or today

    keys: list[str] = []
    for d in date_range(start, end):
        keys.extend(
            list_keys(client, cfg.bucket, f"archive/trip_updates/{d.isoformat()}/")
        )
    return fetch_objects(client, cfg.bucket, keys)


def build_service_series(bodies: list[dict[str, Any]]) -> dict[tuple[str, int], int]:
    """(route, tick) -> assigned_n, from the archived per-tick snapshots."""
    series: dict[tuple[str, int], int] = {}
    for body in bodies:
        tick = _snap_tick(int(body.get("observed_at") or 0))
        rows = cast(dict[str, Any], body.get("rows") or {})
        for route, row in rows.items():
            if isinstance(row, dict):
                assigned = cast(dict[str, Any], row).get("assigned_n") or 0
                series[(route, tick)] = int(assigned)
    return series


_NYC_TZ = ZoneInfo("America/New_York")

# A published supply cell must span at least this many DISTINCT nights before its
# median is trusted, on top of the raw min_samples tick floor. Rationale (the
# weekend-hourly thin-cell caveat): assigned_n is ~constant within an hour, so a
# dozen 5-min ticks from ONE night are a single autocorrelated draw, not twelve.
# On a short window a weekend-hourly (route, schedule_bin) cell clears min_samples
# on ~2 nights, and its median then sits between the bimodal trackwork/normal
# service modes — a genuinely normal weekend night reads 1.7-2.1x its own baseline.
# Requiring ~a month of distinct nights (8 weekend days = 4 weekends) makes such a
# cell ABSTAIN (omitted -> no baseline -> serviceRatioFor null -> service_condition
# 'unknown') until enough independent nights exist, rather than publish a median we
# do not believe. Applied to the published schedule_bin axis only; the tod_bin
# emission denominator keeps the default 1 (no gate) so the frozen operating point
# is untouched. Not a clamp: no ratio is fabricated, the reading is withheld.
SERVICE_MIN_NIGHTS = 8


def _et_date(epoch_seconds: int) -> date:
    """The America/New_York calendar date a tick falls on — the unit
    independent-night gating counts, so autocorrelated same-night ticks collapse
    to one draw. Same zone schedule_bin/tod_bin already bucket by."""
    return datetime.fromtimestamp(epoch_seconds, tz=_NYC_TZ).date()


# Baselines are keyed on (route, time bucket), and which bucket is a real
# choice. `tod_bin` (5 ET blocks) is what the HMM emission channel scores
# against and must keep; `schedule_bin` (ET weekday/weekend x hour) is what a
# degrade/recover call wants, because a 4-6 hour block's median is set by its
# busiest hour and a route running normal service at the quiet edge of the
# block reads as collapsed. The bucket type rides along so a baseline can only
# be paired with the bin_fn that built it.
def compute_baseline[BinKey: (int, str)](
    series: dict[tuple[str, int], int],
    *,
    bin_fn: Callable[[int], BinKey] = tod_bin,
    min_samples: int = 20,
    min_nights: int = 1,
) -> dict[tuple[str, BinKey], float]:
    """Per (route, time bucket) median of assigned_n — the expected running-train
    count at that time of day. The median resists the disrupted minority. Cells
    with fewer than `min_samples` observations, OR spanning fewer than
    `min_nights` distinct ET calendar dates, are omitted (insufficient/
    autocorrelated data), so callers treat a missing baseline as "can't judge",
    not "zero service". `min_nights` defaults to 1 (tick floor only); the
    published supply axis passes SERVICE_MIN_NIGHTS — see its note."""
    buckets: dict[tuple[str, BinKey], list[int]] = {}
    nights: dict[tuple[str, BinKey], set[date]] = {}
    for (route, tick), assigned in series.items():
        key = (route, bin_fn(tick))
        buckets.setdefault(key, []).append(assigned)
        nights.setdefault(key, set()).add(_et_date(tick))
    return {
        key: statistics.median(vals)
        for key, vals in buckets.items()
        if len(vals) >= min_samples and len(nights[key]) >= min_nights
    }


def service_baseline_to_json[BinKey: (int, str)](
    baseline: dict[tuple[str, BinKey], float],
) -> dict[str, dict[str, float]]:
    """Serialize an assigned_n baseline for params.json delivery to the Worker,
    nested route -> bin (stringified) -> median. Serves both the tod_bin baseline
    (the service emission's live-ratio denominator) and the finer schedule_bin
    baseline (the published service-degradation axis) — the bin key is whatever
    the baseline was built with. The Worker divides live assigned_n by the median
    to form the ratio."""
    out: dict[str, dict[str, float]] = {}
    for (route, bin_key), median in baseline.items():
        out.setdefault(route, {})[str(bin_key)] = median
    return out


@dataclass(frozen=True)
class ServiceQuantiles:
    """A (route, time bucket) cell's own spread of assigned_n, read against by
    the Worker instead of one global supply-notable multiple — a cell's high
    mark is its own p90, not baseline * 1.25."""

    p10: float
    p90: float


def compute_service_quantiles[BinKey: (int, str)](
    series: dict[tuple[str, int], int],
    *,
    bin_fn: Callable[[int], BinKey] = tod_bin,
    min_samples: int = 20,
    min_nights: int = 1,
) -> dict[tuple[str, BinKey], ServiceQuantiles]:
    """Per (route, time bucket) p10/p90 of assigned_n — the cell's own spread,
    the denominator-relative bounds the Worker draws as ticks on the existing
    supply meter. Same bucketing, `min_samples`, and `min_nights` gates as
    compute_baseline: a cell present in one is present in the other (pass the
    SAME min_nights), so a caller can always pair a quantile with its median.

    Nearest-rank on the cell's own sorted assigned_n samples: p10 is
    sorted[n // 10], p90 is sorted[int(n * 0.9)] (0-indexed) — both are always
    an OBSERVED assigned_n value, never interpolated between two samples."""
    buckets: dict[tuple[str, BinKey], list[int]] = {}
    nights: dict[tuple[str, BinKey], set[date]] = {}
    for (route, tick), assigned in series.items():
        key = (route, bin_fn(tick))
        buckets.setdefault(key, []).append(assigned)
        nights.setdefault(key, set()).add(_et_date(tick))
    out: dict[tuple[str, BinKey], ServiceQuantiles] = {}
    for key, vals in buckets.items():
        if len(vals) < min_samples or len(nights[key]) < min_nights:
            continue
        ordered = sorted(vals)
        n = len(ordered)
        out[key] = ServiceQuantiles(
            p10=float(ordered[n // 10]), p90=float(ordered[int(n * 0.9)])
        )
    return out


def service_quantiles_to_json[BinKey: (int, str)](
    quantiles: dict[tuple[str, BinKey], ServiceQuantiles],
) -> dict[str, dict[str, dict[str, float]]]:
    """Serialize per-cell assigned_n quantiles for the sidecar's `quantiles`
    key, nested route -> bin (stringified) -> {p10, p90} — sibling shape to
    service_baseline_to_json's route -> bin -> median. The Worker divides both
    by that cell's own median baseline to get service_low_ratio/
    service_high_ratio, on the same scale as service_ratio."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    for (route, bin_key), q in quantiles.items():
        out.setdefault(route, {})[str(bin_key)] = {"p10": q.p10, "p90": q.p90}
    return out


# Min usable ticks in a schedule bin before its in-service rate is trusted.
MIN_SCHEDULE_TICKS = 20


def compute_schedule_rate(
    bodies: list[dict[str, Any]],
    *,
    min_ticks: int = MIN_SCHEDULE_TICKS,
) -> dict[tuple[str, str], float]:
    """Per (route, schedule_bin) in-service rate: the share of usable ticks in that
    (weekend, hour) bin where the route was actually running — at least one
    dispatched train (assigned_n >= 1). Not running where it normally runs is a
    suspension; not running where it rarely runs (or never) is a planned gap.

    Uses dispatch (assigned_n), not mere timetable presence (trips_n >= 1): NYCT
    lists a rush-only route's scheduled trips at the fringe hours before/after it
    actually runs, so presence stays high there while dispatch is ~0 — presence
    can't separate the fringe from mid-service. The cost is a coupling to the
    outcome: a route down for most of the training window would learn a low rate
    and read not_scheduled, but the multi-week window dilutes transient outages,
    callers default a missing/unconfident rate to suspended, and the next retrain
    corrects. The denominator is usable ticks only — a globally empty tick (feed
    outage) is skipped so an outage doesn't depress every route's rate.

    A cell is emitted for the full grid of known routes x bins with at least
    `min_ticks` usable ticks — rate 0 where the route never ran at that bin. The
    explicit zeros let a caller tell a route that's off by timetable (real 0, e.g.
    a rush-only line at midday it never appears in) apart from a bin with too
    little data (omitted, treated as unknown)."""
    denom: dict[str, int] = {}
    routes: set[str] = set()
    running: dict[tuple[str, str], int] = {}
    for body in bodies:
        rows = cast(dict[str, Any], body.get("rows") or {})
        if not rows:  # feed-outage tick — don't let it depress the rate
            continue
        sb = schedule_bin(int(body.get("observed_at") or 0))
        denom[sb] = denom.get(sb, 0) + 1
        for route, row in rows.items():
            routes.add(route)
            if (
                isinstance(row, dict)
                and int(cast(dict[str, Any], row).get("assigned_n") or 0) >= 1
            ):
                running[(route, sb)] = running.get((route, sb), 0) + 1
    bins = sorted(sb for sb, total in denom.items() if total >= min_ticks)
    out: dict[tuple[str, str], float] = {}
    for route in sorted(routes):
        for sb in bins:
            out[(route, sb)] = running.get((route, sb), 0) / denom[sb]
    return out


def schedule_rate_to_json(
    rate: dict[tuple[str, str], float],
) -> dict[str, dict[str, float]]:
    """Serialize the scheduled-presence rate for params.json delivery to the
    Worker, nested route -> schedule_bin -> rate."""
    out: dict[str, dict[str, float]] = {}
    for (route, sb), r in rate.items():
        out.setdefault(route, {})[sb] = r
    return out


@dataclass(frozen=True)
class Disruption:
    """An independent disruption interval derived from the service metric."""

    route: str
    start_tick: int  # first degraded tick
    recovered_tick: int  # first recovered tick


def derive_actual_recovery[BinKey: (int, str)](
    series: dict[tuple[str, int], int],
    baseline: dict[tuple[str, BinKey], float],
    *,
    bin_fn: Callable[[int], BinKey] = tod_bin,
    degrade_ratio: float = 0.5,
    recover_ratio: float = 0.8,
    debounce: int = 2,
) -> list[Disruption]:
    """Independent disruptions from the service metric: a route is degraded when
    assigned_n falls below `degrade_ratio` x its (route, time bucket) baseline for
    `debounce` consecutive ticks, and recovered at the first tick back above
    `recover_ratio` for `debounce` consecutive ticks. Hysteresis (recover >
    degrade) avoids flapping. Ticks with no baseline reset the run counters but
    don't end an open disruption. Disruptions still open at the window end are
    censored (dropped).

    `bin_fn` must be the one the baseline was built with — see compute_baseline."""
    by_route: dict[str, list[tuple[int, int]]] = {}
    for (route, tick), assigned in series.items():
        by_route.setdefault(route, []).append((tick, assigned))

    out: list[Disruption] = []
    for route, points in by_route.items():
        points.sort()
        in_disruption = False
        start: int | None = None
        cand_start: int | None = None
        cand_recover: int | None = None
        low_run = 0
        high_run = 0
        for tick, assigned in points:
            base = baseline.get((route, bin_fn(tick)))
            if base is None or base <= 0:
                low_run = 0
                high_run = 0
                continue
            ratio = assigned / base
            if not in_disruption:
                if ratio < degrade_ratio:
                    if low_run == 0:
                        cand_start = tick
                    low_run += 1
                    if low_run >= debounce:
                        in_disruption = True
                        start = cand_start
                        high_run = 0
                else:
                    low_run = 0
            else:
                if ratio >= recover_ratio:
                    if high_run == 0:
                        cand_recover = tick
                    high_run += 1
                    if high_run >= debounce and start is not None:
                        out.append(Disruption(route, start, cand_recover or tick))
                        in_disruption = False
                        start = None
                        low_run = 0
                        high_run = 0
                else:
                    high_run = 0
    out.sort(key=lambda d: (d.route, d.start_tick))
    return out


# --- Vehicle-movement metric: independent current-state truth ---
#
# The Worker archives a compact per-route movement metric each tick at
# archive/vehicles/<date>/<observed_at>.json:
#   {observed_at, fresh_feeds, rows: {route: {vehicles_n, stopped_n, moving_n,
#                                             advanced_n, stalled_n}}}
# This is independent IN DERIVATION from the alerts feed and from assigned_n:
# it's where trains physically are (decoded VehiclePosition stop_ids), not how
# many trips are dispatched. Once assigned_n becomes a live HMM input it can no
# longer be held out as truth; vehicle movement still can. Same upstream feed,
# though — independent-in-derivation, not in-source.
#
# The headline signal is the CROSS-TICK advance fraction, advanced_n /
# (advanced_n + stalled_n): of the trips seen both this tick and last, the share
# that moved to a new stop. A route with trains dispatched but none advancing is
# physically frozen — the disruption mode assigned_n structurally cannot see.


def fetch_vehicle_metrics(
    config: R2Config | None = None,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    client: S3Client | None = None,
) -> list[dict[str, Any]]:
    """Pull every archived vehicle-movement snapshot in the window (one object
    per tick). Defaults to yesterday-through-today. Mirrors
    fetch_trip_update_metrics."""
    cfg = config or load_config()
    client = client or make_client(cfg)

    today = datetime.now(UTC).date()
    start = start_date or (today - timedelta(days=1))
    end = end_date or today

    keys: list[str] = []
    for d in date_range(start, end):
        keys.extend(list_keys(client, cfg.bucket, f"archive/vehicles/{d.isoformat()}/"))
    return fetch_objects(client, cfg.bucket, keys)


# (route, direction, from_stop) -> count this trip's cross-tick outcome? Passed
# where the caller wants the advance signal restricted to stops it is willing to
# judge — see training.gtfs_static.terminals / through_stops, and the terminal
# layover mass they exist to keep out of the advance rate.
StopFilter = Callable[[str, str, str], bool]


def _cross_tick_counts(
    drow: Mapping[str, Any],
    route: str,
    direction: str,
    counts_from_stop: StopFilter | None,
) -> tuple[int, int]:
    """One archived direction row's (advanced_n, stalled_n).

    Unfiltered, straight off the archived counters. Filtered, recomputed from
    that row's raw `transitions` map — the counters are already summed and can't
    be narrowed after the fact, while the transitions map still carries each
    trip's from_stop. Same population either way: measured over 2026-08-11's 288
    ticks, the route counters, the per-direction counters and the transitions
    map all give 87,912 advances and 59,657 stalls.
    """
    if counts_from_stop is None:
        return (
            int(drow.get("advanced_n") or 0),
            int(drow.get("stalled_n") or 0),
        )
    advanced = stalled = 0
    trans = cast(dict[str, Any], drow.get("transitions") or {})
    for pair, count in trans.items():
        frm, sep, to = pair.partition(">")
        if not sep or not frm or not to:
            continue
        n = int(count or 0)
        if n <= 0 or not counts_from_stop(route, direction, frm):
            continue
        if frm == to:
            stalled += n
        else:
            advanced += n
    return advanced, stalled


def build_movement_series(
    bodies: list[dict[str, Any]],
    *,
    counts_from_stop: StopFilter | None = None,
) -> dict[tuple[str, int], dict[str, int]]:
    """(route, tick) -> the full movement row, from the archived per-tick
    snapshots. Keeps every counter (not just one) because the current-state call
    needs both presence (vehicles_n) and the cross-tick advance fraction.

    With `counts_from_stop`, advanced_n/stalled_n are summed over the two
    directions' filtered transitions instead of read off the route counters; the
    presence counters are unaffected."""
    series: dict[tuple[str, int], dict[str, int]] = {}
    for body in bodies:
        tick = _snap_tick(int(body.get("observed_at") or 0))
        rows = cast(dict[str, Any], body.get("rows") or {})
        for route, row in rows.items():
            if not isinstance(row, dict):
                continue
            row = cast(dict[str, Any], row)
            out = {
                k: int(row.get(k) or 0) for k in ("vehicles_n", "stopped_n", "moving_n")
            }
            if counts_from_stop is None:
                out["advanced_n"] = int(row.get("advanced_n") or 0)
                out["stalled_n"] = int(row.get("stalled_n") or 0)
            else:
                by_dir = cast(dict[str, Any], row.get("by_direction") or {})
                advanced = stalled = 0
                for direction in _DIRECTIONS:
                    drow = by_dir.get(direction)
                    if not isinstance(drow, dict):
                        continue
                    a, s = _cross_tick_counts(
                        cast(dict[str, Any], drow), route, direction, counts_from_stop
                    )
                    advanced += a
                    stalled += s
                out["advanced_n"] = advanced
                out["stalled_n"] = stalled
            series[(route, tick)] = out
    return series


_DIRECTIONS: tuple[str, ...] = ("north", "south")


def build_movement_series_by_direction(
    bodies: list[dict[str, Any]],
    *,
    counts_from_stop: StopFilter | None = None,
) -> dict[tuple[str, str, int], dict[str, int]]:
    """(route, direction, tick) -> the per-direction movement counters, from the
    by_direction split the Worker archives (north/south). The cross-tick advance
    fraction is direction-specific because the two directions fail independently
    and the Bayesian model scores each line-direction against its own baseline.

    `counts_from_stop` restricts advanced_n/stalled_n to trips whose from_stop it
    admits — see _cross_tick_counts."""
    series: dict[tuple[str, str, int], dict[str, int]] = {}
    for body in bodies:
        tick = _snap_tick(int(body.get("observed_at") or 0))
        rows = cast(dict[str, Any], body.get("rows") or {})
        for route, row in rows.items():
            if not isinstance(row, dict):
                continue
            by_dir = cast(
                dict[str, Any], cast(dict[str, Any], row).get("by_direction") or {}
            )
            for direction in _DIRECTIONS:
                drow = by_dir.get(direction)
                if not isinstance(drow, dict):
                    continue
                advanced, stalled = _cross_tick_counts(
                    cast(dict[str, Any], drow), route, direction, counts_from_stop
                )
                series[(route, direction, tick)] = {
                    "vehicles_n": int(
                        cast(dict[str, Any], drow).get("vehicles_n") or 0
                    ),
                    "advanced_n": advanced,
                    "stalled_n": stalled,
                }
    return series


def build_segment_series(
    bodies: list[dict[str, Any]],
) -> dict[tuple[str, str, str, str, int], int]:
    """(route, direction, from_stop, to_stop, tick) -> cross-tick transition count,
    from the by_direction.transitions the Worker archives. The raw segment leaf for
    the (route, direction, segment) hierarchical baseline; from==to is a stall in
    place. Tick is kept so later phases can condition on tod/baseline windows;
    canonical segment mapping (needs static GTFS stop ordering) is layered on top."""
    series: dict[tuple[str, str, str, str, int], int] = {}
    for body in bodies:
        tick = _snap_tick(int(body.get("observed_at") or 0))
        rows = cast(dict[str, Any], body.get("rows") or {})
        for route, row in rows.items():
            if not isinstance(row, dict):
                continue
            by_dir = cast(
                dict[str, Any], cast(dict[str, Any], row).get("by_direction") or {}
            )
            for direction in _DIRECTIONS:
                drow = by_dir.get(direction)
                if not isinstance(drow, dict):
                    continue
                trans = cast(
                    dict[str, Any], cast(dict[str, Any], drow).get("transitions") or {}
                )
                for pair, count in trans.items():
                    if ">" not in pair:
                        continue
                    frm, to = pair.split(">", 1)
                    n = int(count or 0)
                    if not frm or not to or n <= 0:
                        continue
                    key = (route, direction, frm, to, tick)
                    series[key] = series.get(key, 0) + n
    return series


def build_segment_baseline(
    bodies: list[dict[str, Any]],
    *,
    counts_from_stop: StopFilter | None = None,
) -> dict[tuple[str, str, str], PooledCell]:
    """Hierarchical partial-pooling advance-rate baseline per (route, direction,
    from_stop) segment leaf, from the archived cross-tick transitions.

    Aggregates build_segment_series over the window into per-leaf advanced/stalled
    counts — a from>to pair with from!=to is an advance out of from_stop, from==to
    is a stall there — then shrinks each leaf toward its line-direction / route /
    system normal via the empirical-Bayes estimator (training.hierarchical). Sparse
    leaves borrow strength; data-rich leaves keep their own low rate. The finer
    leaf for the segment-aware classifier and station roll-ups; canonical
    from_stop->to_stop labelling layers on top.

    `counts_from_stop` drops leaves the caller won't judge. Restricted to through
    stops, every held-out leaf has training data (measured 2026-08-12: 177 leaves
    with none, down to 0) and pooling stops losing to each leaf's own raw rate,
    because a layover and a mid-line stop are no longer pooled as if exchangeable."""
    leaves: dict[tuple[str, str, str], list[int]] = {}
    for (route, direction, frm, to, _tick), n in build_segment_series(bodies).items():
        if counts_from_stop is not None and not counts_from_stop(route, direction, frm):
            continue
        cell = leaves.setdefault((route, direction, frm), [0, 0])
        if frm == to:
            cell[1] += n  # stall in place
        else:
            cell[0] += n  # advanced out of frm
    return partially_pool({k: (adv, stall) for k, (adv, stall) in leaves.items()})


# --- Expected throughput per segment cell ---------------------------------
#
# The advance-rate baseline above answers "of the trains that were here, what
# share moved on". It answers nothing when no train was here at all — and that
# is the common case: the Worker's MIN_EFF_MATCHED=5 over a ~25-minute decayed
# window needs sub-5-minute headways, so almost every baselined cell abstains
# almost always. The missing quantity is the denominator the timetable implies:
# how many traversals a cell should see per tick. With it, an empty window is
# evidence instead of an abstention.

# Bin the rate by ET (weekday|weekend, hour) — momentarily.hmm.schedule_bin —
# for the reason training.degradation_label pins the same choice for the
# assigned_n label: a tod_bin spans 4-6 hours and its centre is set by the
# busiest core, so the quiet edge of a block reads as a collapse against it.
# Throughput varies at exactly that scale (a 20-minute overnight headway inside
# the same tod_bin as a 4-minute rush headway), and the weekday/weekend split
# rides along for free.
THROUGHPUT_BIN_FN = schedule_bin

# Observed ticks a bin needs before its rate is published, matching the
# min_samples floor compute_service_quantiles uses for a per-cell spread. Below
# it no rate ships and the Worker keeps abstaining — the behaviour that
# preceded this fit — rather than judging silence against a handful of ticks.
MIN_THROUGHPUT_TICKS = 20


def throughput_exposure[BinKey: (int, str)](
    bodies: list[dict[str, Any]],
    *,
    bin_fn: Callable[[int], BinKey] = THROUGHPUT_BIN_FN,
) -> dict[BinKey, int]:
    """Observed ticks per time bin — the exposure a per-tick traversal rate
    divides by.

    Counted once per SNAPPED tick (a duplicated archive object must not inflate
    the denominator) and only for ticks the vehicle feed actually reported on: a
    body with no rows is a feed outage, and counting it would drag every cell's
    rate toward zero exactly where the archive is blind. Same guard
    compute_schedule_rate applies to its own denominator.
    """
    seen: dict[int, BinKey] = {}
    for body in bodies:
        if not cast(dict[str, Any], body.get("rows") or {}):
            continue
        tick = _snap_tick(int(body.get("observed_at") or 0))
        seen[tick] = bin_fn(tick)
    out: dict[BinKey, int] = {}
    for bin_key in seen.values():
        out[bin_key] = out.get(bin_key, 0) + 1
    return out


def build_segment_throughput[BinKey: (int, str)](
    bodies: list[dict[str, Any]],
    *,
    counts_from_stop: StopFilter | None = None,
    bin_fn: Callable[[int], BinKey] = THROUGHPUT_BIN_FN,
    min_ticks: int = MIN_THROUGHPUT_TICKS,
) -> tuple[dict[tuple[str, str, str], dict[BinKey, float]], dict[BinKey, int]]:
    """Expected matched traversals per tick for each (route, direction,
    from_stop) cell at each time bin, plus the exposure it was fitted over.

    Numerator: every archived cross-tick transition out of the from_stop,
    advances and stalls alike, because `matched` is what the Worker's decayed
    accumulator counts (segment_flow.ts tickCounts) and the rate has to be in
    the same units as the thing it is compared against. Denominator: the ticks
    the archive covered in that bin, NOT the ticks this cell was seen in — a
    cell missing from the transitions saw no train, which is the entire signal.

    Only bins clearing `min_ticks` are fitted; a cell's rate for such a bin is
    always present, zero included, because zero is the informative value.
    Returned exposure is the published bin set, so a consumer can tell "fitted
    at zero" from "never fitted".

    Cadence-defined: the rate is per TICK, so a change to the cron cadence the
    Worker accumulates on invalidates it and it has to be refitted in lockstep.
    """
    exposure = {
        bin_key: ticks
        for bin_key, ticks in throughput_exposure(bodies, bin_fn=bin_fn).items()
        if ticks >= min_ticks
    }
    matched: dict[tuple[str, str, str], dict[BinKey, int]] = {}
    for (route, direction, frm, _to, tick), n in build_segment_series(bodies).items():
        if counts_from_stop is not None and not counts_from_stop(route, direction, frm):
            continue
        bin_key = bin_fn(tick)
        if bin_key not in exposure:
            continue
        counts = matched.setdefault((route, direction, frm), {})
        counts[bin_key] = counts.get(bin_key, 0) + n
    return (
        {
            leaf: {b: counts.get(b, 0) / exposure[b] for b in exposure}
            for leaf, counts in matched.items()
        },
        exposure,
    )


def throughput_to_json(
    rates: Mapping[tuple[str, str, str], Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    """Serialize the per-cell rates for segment_params.json, keyed the way the
    Worker keys its cells ('route|direction|from_stop').

    Zero-rate bins are DROPPED: with the published bin set riding along in the
    doc's own exposure map, an absent bin inside a published one reads as zero
    unambiguously, and dropping them takes a fitted-everywhere 48-bin table
    down to the bins where trains actually run.
    """
    return {
        "|".join(leaf): {b: round(lam, 4) for b, lam in sorted(bins.items()) if lam > 0}
        for leaf, bins in rates.items()
    }


# Movement→state thresholds. MIN_MATCHED_TRIPS gates whether a direction has
# enough cross-tick matches to judge at all; under it the direction abstains.
MIN_MATCHED_TRIPS = 3  # advanced_n + stalled_n floor to make a cross-tick call

# Classification-time prior strength in pseudo-trials, distinct from
# ADVANCE_PRIOR_STRENGTH (which anchors the HMM emission accumulated over the
# whole training window). This one regularizes a single live tick's advance
# fraction toward the cell baseline so a thin sample can't swing the call; kept
# light enough that a decisive tick still speaks.
CLASSIFY_PRIOR_STRENGTH = 8.0

# A direction reads disrupted when its posterior advance rate sits at/under this
# fraction of the cell's own baseline p0 — advancing at under half its normal
# rate. Baseline-relative, so a shuttle and a trunk line are each judged against
# their own normal instead of one global cutoff.
DISRUPTED_RATIO = 0.5

# A large posterior drop only reads disrupted when the low advance count is also
# statistically significant against the cell baseline (binomial lower tail at or
# under this). Guards degenerate-low baselines — a short shuttle advances ~0 even
# when healthy, so a normal zero-advance tick would otherwise flip disrupted.
CLASSIFY_ALPHA = 0.05


# Pseudo-trials behind a baseline Beta prior — how much a cell's history outvotes
# the live tick when forming the posterior advance rate. ~50 trips is a few ticks
# of a busy line; enough to anchor a thin live sample without burying a real shift.
ADVANCE_PRIOR_STRENGTH = 50.0

# Keep p0 off the degenerate endpoints so the Beta shapes stay strictly positive
# (a healthy line where every matched trip advanced medians to 1.0 → beta=0
# otherwise). Mirrors hmm.py's BERNOULLI_FLOOR on the emission's advance_rate.
P0_FLOOR = 1e-3

# One-sided lower trim (fraction of a cell's worst ticks, by advance fraction,
# dropped before pooling) for the advance baseline. Defaults OFF. The trim was
# meant to stop outage ticks dragging the normal rate down, but measured on
# 2026-07-31..08-13 the drag is small (median p0 0.937 at trim 0) while any
# trim>0 re-creates the exact saturation this fix removes — most ticks are
# legitimately stall-free, so trimming the low tail walks p0 back toward 1.0
# (>=0.999 cells: 0% at trim 0, 19% at 0.1, 37% at 0.2). Kept as a knob to
# recalibrate against a movement-truth eval; 0 is correct on today's data.
ADVANCE_TRIM = 0.0


@dataclass(frozen=True)
class AdvanceBaseline:
    """The normal advance-rate prior for one (route, direction, tod_bin) cell.

    `p0` is the cell's baseline (normal) cross-tick advance fraction — the share
    of matched trips that advance a stop in a healthy tick. It anchors the HMM's
    movement emission: the normal state sits near p0, disrupted below it. Carried
    as the Beta(alpha, beta) prior the emission's responsibility-weighted update
    consumes, with alpha + beta = the prior strength in pseudo-trials.
    """

    p0: float  # baseline advance rate (trimmed pooled rate over the cell's ticks)
    n: int  # ticks contributing to the cell
    alpha: float  # Beta prior successes: prior_strength * p0
    beta: float  # Beta prior failures: prior_strength * (1 - p0)


def _trimmed_pooled_advance_rate(ticks: list[tuple[int, int]], trim: float) -> float:
    """Pooled Σadvanced / Σmatched over a cell's ticks, after dropping the lowest
    `trim` fraction of ticks by advance fraction.

    A one-sided (lower) trim: the dropped ticks are the cell's worst — outages and
    frozen stretches — so they can't drag the *normal* rate down, the job the old
    median did by construction. Unlike the median it does not saturate to ~1.0
    when most ticks are stall-free, because pooling keeps the ordinary
    few-percent stall rate in the denominator. `trim=0` is the raw pooled rate;
    each tick is (advanced_n, matched_n) with matched_n > 0. `trim` must be in
    [0, 1) — a trim that dropped every tick would leave nothing to pool and
    silently floor the baseline.
    """
    if not 0.0 <= trim < 1.0:
        raise ValueError(f"trim must be in [0, 1), got {trim!r}")
    kept = ticks
    if trim > 0.0 and len(ticks) > 1:
        ordered = sorted(ticks, key=lambda t: t[0] / t[1])
        kept = ordered[int(len(ordered) * trim) :]
    matched = sum(m for _a, m in kept)
    advanced = sum(a for a, _m in kept)
    return advanced / matched if matched > 0 else 0.0


def compute_advance_baseline(
    series: dict[tuple[str, str, int], dict[str, int]],
    *,
    prior_strength: float = ADVANCE_PRIOR_STRENGTH,
    min_matched: int = MIN_MATCHED_TRIPS,
    min_samples: int = 20,
    trim: float = ADVANCE_TRIM,
) -> dict[tuple[str, str, int], AdvanceBaseline]:
    """Per (route, direction, tod_bin) baseline advance rate, as a Beta prior.

    For each tick with at least `min_matched` cross-tick matches we keep its
    (advanced_n, matched_n). The cell's p0 is the trimmed *pooled* rate
    Σadvanced / Σmatched over those ticks (_trimmed_pooled_advance_rate) — the
    rate the cell actually ran at in normal operation. This replaces a median of
    per-tick fractions, which saturated to ~1.0 wherever most ticks were
    stall-free and published a p0 no cell truly ran at, making the downstream
    binomial significance test fire on a single stall (journal 2026-08-13). Cells
    below `min_samples` ticks are omitted (callers treat a missing baseline as
    "no prior", and the emission channel drops out — see hmm.py has_movement).
    """
    buckets: dict[tuple[str, str, int], list[tuple[int, int]]] = {}
    for (route, direction, tick), row in series.items():
        advanced = row.get("advanced_n", 0)
        matched = advanced + row.get("stalled_n", 0)
        if matched < min_matched:
            continue
        buckets.setdefault((route, direction, tod_bin(tick)), []).append(
            (advanced, matched)
        )

    out: dict[tuple[str, str, int], AdvanceBaseline] = {}
    for key, ticks in buckets.items():
        if len(ticks) < min_samples:
            continue
        rate = _trimmed_pooled_advance_rate(ticks, trim)
        p0 = min(max(rate, P0_FLOOR), 1.0 - P0_FLOOR)
        out[key] = AdvanceBaseline(
            p0=p0,
            n=len(ticks),
            alpha=prior_strength * p0,
            beta=prior_strength * (1.0 - p0),
        )
    return out


# A carried movement snapshot older than this (seconds) is a feed gap, not "now"
# — mirrors worker/src/movement_state.ts MAX_MOVEMENT_METRIC_LAG_SECONDS. The
# live filter folds the PREVIOUS tick's counts into an observation (option B
# lag) and drops anything staler than this; training admits the same window so
# the emission is fitted on exactly the samples inference will score it against.
MAX_MOVEMENT_METRIC_LAG_SECONDS = 600


def movement_observation_fields(
    movement_by_tick: dict[tuple[str, int], dict[str, int]],
    baseline_cells: set[tuple[str, str, int]],
    route: str,
    tick: int,
    *,
    min_matched: int = MIN_MATCHED_TRIPS,
) -> dict[str, Any] | None:
    """Movement fields for one (route, tick) HMM observation, reconstructing what
    the live filter folds in at that tick: the previous tick's cross-tick counts,
    both directions summed off the raw route counters. Returns None — channel
    stays off — exactly where the live filter abstains: no carried snapshot within
    the lag window, fewer than `min_matched` cross-tick matches, or no published
    baseline cell for the current tick's tod_bin. Straight port of
    worker/src/movement_state.ts movementObservationFields, so training and
    inference admit and score the same samples (no train/serve skew).
    """
    row: dict[str, int] | None = None
    lag = TICK_SECONDS
    while lag <= MAX_MOVEMENT_METRIC_LAG_SECONDS:
        row = movement_by_tick.get((route, tick - lag))
        if row is not None:
            break
        lag += TICK_SECONDS
    if row is None:
        return None
    advanced_n = int(row.get("advanced_n") or 0)
    matched_n = advanced_n + int(row.get("stalled_n") or 0)
    if matched_n < min_matched:
        return None
    tb = tod_bin(tick)
    if (route, "north", tb) not in baseline_cells and (
        route,
        "south",
        tb,
    ) not in baseline_cells:
        return None
    return {"advanced_n": advanced_n, "matched_n": matched_n, "has_movement": True}


def compute_advance_baseline_by_route(
    series: dict[tuple[str, str, int], dict[str, int]],
    *,
    min_matched: int = MIN_MATCHED_TRIPS,
    min_samples: int = 20,
    trim: float = ADVANCE_TRIM,
) -> dict[str, float]:
    """Per-route baseline (normal) advance rate — the trimmed pooled cross-tick
    advance rate over both directions and all times of day.

    Coarser than compute_advance_baseline: the trained emissions aren't
    TOD-conditioned, so the EM prior anchors one normal-state advance_rate per
    route, not a per-(direction, tod) grid. Same trimmed-pooled estimator and P0
    floor. Routes below min_samples ticks are omitted so a route with thin
    movement data gets no fabricated prior (the fit keeps the default).
    """
    buckets: dict[str, list[tuple[int, int]]] = {}
    for (route, _direction, _tick), row in series.items():
        advanced = row.get("advanced_n", 0)
        matched = advanced + row.get("stalled_n", 0)
        if matched < min_matched:
            continue
        buckets.setdefault(route, []).append((advanced, matched))

    out: dict[str, float] = {}
    for route, ticks in buckets.items():
        if len(ticks) < min_samples:
            continue
        rate = _trimmed_pooled_advance_rate(ticks, trim)
        out[route] = min(max(rate, P0_FLOOR), 1.0 - P0_FLOOR)
    return out


def advance_baseline_to_json(
    baseline: dict[tuple[str, str, int], AdvanceBaseline],
) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    """Serialize the advance baseline for params.json delivery to the Worker,
    nested route -> direction -> tod_bin -> cell. tod_bin keys are stringified
    (JSON object keys must be strings; the Worker parses them back to int)."""
    out: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for (route, direction, tod), cell in baseline.items():
        out.setdefault(route, {}).setdefault(direction, {})[str(tod)] = {
            "p0": cell.p0,
            "alpha": cell.alpha,
            "beta": cell.beta,
            "n": cell.n,
        }
    return out


def _binom_lower_tail(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p), via an iterative pmf sum. Exact for the
    tick-level counts here (n well under ~50) and free of special functions, so it
    mirrors the worker/viz binomLowerTail 1:1. p is the cell baseline p0, floored
    off 0 upstream."""
    if k >= n:
        return 1.0
    if k < 0:
        return 0.0
    q = 1.0 - p
    pmf = q**n  # P(X = 0)
    cdf = pmf
    for i in range(k):
        pmf *= (n - i) / (i + 1) * (p / q)
        cdf += pmf
    return cdf


def classify_direction(
    advanced_n: int,
    stalled_n: int,
    baseline: AdvanceBaseline | None,
    *,
    prior_strength: float = CLASSIFY_PRIOR_STRENGTH,
    disrupted_ratio: float = DISRUPTED_RATIO,
    min_matched: int = MIN_MATCHED_TRIPS,
    alpha: float = CLASSIFY_ALPHA,
) -> str | None:
    """Beta-Binomial call for one (route, direction) at one tick, three ways:

      normal    — posterior advance rate above `disrupted_ratio * p0`.
      disrupted — posterior at/under `disrupted_ratio * p0` AND the low advance
                  count is significant against p0 (binomial lower tail <= alpha).
      None      — can't judge: fewer than `min_matched` matches, no baseline, OR a
                  point-estimate drop not distinguishable from a low-p0 normal
                  fluctuation (a degenerate-baseline shuttle's zero-advance tick,
                  not a stall).

    Baseline-relative, so low-baseline lines aren't pinned disrupted; the
    significance gate additionally keeps a degenerate baseline from firing on its
    own normal zero-advance noise."""
    matched = advanced_n + stalled_n
    if matched < min_matched:
        return None
    if baseline is None:
        return None
    post = (prior_strength * baseline.p0 + advanced_n) / (prior_strength + matched)
    if post > disrupted_ratio * baseline.p0:
        return "normal"
    if _binom_lower_tail(advanced_n, matched, baseline.p0) <= alpha:
        return "disrupted"
    return None


def derive_movement_state(
    route_row: dict[str, int],
    dir_rows: Mapping[str, dict[str, int] | None],
    baselines: Mapping[str, AdvanceBaseline | None],
    *,
    prior_strength: float = CLASSIFY_PRIOR_STRENGTH,
    disrupted_ratio: float = DISRUPTED_RATIO,
    min_matched: int = MIN_MATCHED_TRIPS,
) -> str | None:
    """Independent current-state label for one route at one tick, or None when the
    movement channel can't support a call.

      suspended — no trains physically on the route (vehicles_n == 0).
      disrupted — at least one direction reads frozen against its own baseline.
      normal    — trains present, at least one direction judgeable, none frozen.

    Vehicle-only: a suspended route has no vehicles, so this is the sole no-service
    reading here (and the vehicle archive omits routes with no trains, so it rarely
    fires). The worker's deriveMovementState, which also sees the trip-updates feed
    and the schedule rate, is what splits suspended vs not_scheduled. Each direction
    is scored against its own (route, direction, tod_bin) baseline via
    classify_direction; the route takes the worse of the two."""
    if route_row.get("vehicles_n", 0) <= 0:
        return "suspended"
    calls: list[str] = []
    for direction in _DIRECTIONS:
        drow = dir_rows.get(direction)
        if drow is None:
            continue
        call = classify_direction(
            drow.get("advanced_n", 0),
            drow.get("stalled_n", 0),
            baselines.get(direction),
            prior_strength=prior_strength,
            disrupted_ratio=disrupted_ratio,
            min_matched=min_matched,
        )
        if call is not None:
            calls.append(call)
    if not calls:
        return None
    return "disrupted" if "disrupted" in calls else "normal"


def build_movement_truth(
    bodies: list[dict[str, Any]],
    *,
    movement_baseline: Mapping[tuple[str, str, int], AdvanceBaseline],
    counts_from_stop: StopFilter | None = None,
    prior_strength: float = CLASSIFY_PRIOR_STRENGTH,
    disrupted_ratio: float = DISRUPTED_RATIO,
    min_matched: int = MIN_MATCHED_TRIPS,
) -> dict[tuple[str, int], str]:
    """(route, tick) -> independent movement-derived state, judgeable ticks only.
    A drop-in alternate truth for confusion(): pass it where build_mta_truth's
    output goes to score the HMM condition against where trains physically are.

    `movement_baseline` is the per-(route, direction, tod_bin) advance prior
    applied to each tick — supply it explicitly (compute_advance_baseline over a
    clean/earlier window) rather than deriving it from the labeled bodies, so the
    truth stays causal and a sustained outage can't lower its own baseline.

    `counts_from_stop` MUST match the filter the baseline was fitted with. A
    through-stop baseline runs higher than an unfiltered one, so scoring
    unfiltered live counts against it reads spuriously disrupted."""
    route_series = build_movement_series(bodies, counts_from_stop=counts_from_stop)
    dir_series = build_movement_series_by_direction(
        bodies, counts_from_stop=counts_from_stop
    )
    truth: dict[tuple[str, int], str] = {}
    for (route, tick), route_row in route_series.items():
        tb = tod_bin(tick)
        dir_rows: dict[str, dict[str, int] | None] = {
            d: dir_series.get((route, d, tick)) for d in _DIRECTIONS
        }
        baselines: dict[str, AdvanceBaseline | None] = {
            d: movement_baseline.get((route, d, tb)) for d in _DIRECTIONS
        }
        state = derive_movement_state(
            route_row,
            dir_rows,
            baselines,
            prior_strength=prior_strength,
            disrupted_ratio=disrupted_ratio,
            min_matched=min_matched,
        )
        if state is not None:
            truth[(route, tick)] = state
    return truth
