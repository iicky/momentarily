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
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol, cast

from blake3 import blake3

from momentarily.hmm import Observation, schedule_bin, tod_bin
from momentarily.mapping import is_planned_work_id
from training.load import TICK_SECONDS, TickObservation
from training.r2_client import R2Config, load_config, make_client

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


def _date_range(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _list_keys(client: S3Client, bucket: str, prefix: str) -> list[str]:
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
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    return cast(dict[str, Any], json.loads(body))


def list_alert_keys(client: S3Client, bucket: str, start: date, end: date) -> list[str]:
    """Every alert-version object key in the [start, end] window, in list order."""
    keys: list[str] = []
    for d in _date_range(start, end):
        keys.extend(_list_keys(client, bucket, f"archive/alerts/{d.isoformat()}/"))
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
    hallucinated tail. See momentarily-1a7.

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
    See momentarily-1a7.
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


# --- Trip-updates service metric: independent recovery truth (momentarily-xum) ---
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
    for d in _date_range(start, end):
        keys.extend(
            _list_keys(client, cfg.bucket, f"archive/trip_updates/{d.isoformat()}/")
        )

    bucket = cfg.bucket
    fetched_client = client

    def _fetch(k: str) -> dict[str, Any]:
        return _fetch_object(fetched_client, bucket, k)

    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        out.extend(pool.map(_fetch, keys))
    return out


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


def compute_baseline(
    series: dict[tuple[str, int], int], *, min_samples: int = 20
) -> dict[tuple[str, int], float]:
    """Per (route, tod_bin) median of assigned_n — the expected running-train
    count at that time of day. The median resists the disrupted minority. Cells
    with fewer than `min_samples` observations are omitted (insufficient data),
    so callers treat a missing baseline as "can't judge", not "zero service"."""
    buckets: dict[tuple[str, int], list[int]] = {}
    for (route, tick), assigned in series.items():
        buckets.setdefault((route, tod_bin(tick)), []).append(assigned)
    return {
        key: statistics.median(vals)
        for key, vals in buckets.items()
        if len(vals) >= min_samples
    }


def service_baseline_to_json(
    baseline: dict[tuple[str, int], float],
) -> dict[str, dict[str, float]]:
    """Serialize the assigned_n baseline for params.json delivery to the Worker,
    nested route -> tod_bin (stringified int) -> median. The Worker divides the
    live assigned_n by this to form the service ratio the emission scores."""
    out: dict[str, dict[str, float]] = {}
    for (route, tod), median in baseline.items():
        out.setdefault(route, {})[str(tod)] = median
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


def derive_actual_recovery(
    series: dict[tuple[str, int], int],
    baseline: dict[tuple[str, int], float],
    *,
    degrade_ratio: float = 0.5,
    recover_ratio: float = 0.8,
    debounce: int = 2,
) -> list[Disruption]:
    """Independent disruptions from the service metric: a route is degraded when
    assigned_n falls below `degrade_ratio` x its (route, tod_bin) baseline for
    `debounce` consecutive ticks, and recovered at the first tick back above
    `recover_ratio` for `debounce` consecutive ticks. Hysteresis (recover >
    degrade) avoids flapping. Ticks with no baseline reset the run counters but
    don't end an open disruption. Disruptions still open at the window end are
    censored (dropped)."""
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
            base = baseline.get((route, tod_bin(tick)))
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


# --- Vehicle-movement metric: independent current-state truth (momentarily-vy0) ---
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
    for d in _date_range(start, end):
        keys.extend(
            _list_keys(client, cfg.bucket, f"archive/vehicles/{d.isoformat()}/")
        )

    bucket = cfg.bucket
    fetched_client = client

    def _fetch(k: str) -> dict[str, Any]:
        return _fetch_object(fetched_client, bucket, k)

    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        out.extend(pool.map(_fetch, keys))
    return out


def build_movement_series(
    bodies: list[dict[str, Any]],
) -> dict[tuple[str, int], dict[str, int]]:
    """(route, tick) -> the full movement row, from the archived per-tick
    snapshots. Keeps every counter (not just one) because the current-state call
    needs both presence (vehicles_n) and the cross-tick advance fraction."""
    series: dict[tuple[str, int], dict[str, int]] = {}
    for body in bodies:
        tick = _snap_tick(int(body.get("observed_at") or 0))
        rows = cast(dict[str, Any], body.get("rows") or {})
        for route, row in rows.items():
            if isinstance(row, dict):
                series[(route, tick)] = {
                    k: int(cast(dict[str, Any], row).get(k) or 0)
                    for k in (
                        "vehicles_n",
                        "stopped_n",
                        "moving_n",
                        "advanced_n",
                        "stalled_n",
                    )
                }
    return series


_DIRECTIONS: tuple[str, ...] = ("north", "south")


def build_movement_series_by_direction(
    bodies: list[dict[str, Any]],
) -> dict[tuple[str, str, int], dict[str, int]]:
    """(route, direction, tick) -> the per-direction movement counters, from the
    by_direction split the Worker archives (north/south). The cross-tick advance
    fraction is direction-specific because the two directions fail independently
    and the Bayesian model scores each line-direction against its own baseline."""
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
                series[(route, direction, tick)] = {
                    k: int(cast(dict[str, Any], drow).get(k) or 0)
                    for k in ("vehicles_n", "advanced_n", "stalled_n")
                }
    return series


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


# Pseudo-trials behind a baseline Beta prior — how much a cell's history outvotes
# the live tick when forming the posterior advance rate. ~50 trips is a few ticks
# of a busy line; enough to anchor a thin live sample without burying a real shift.
ADVANCE_PRIOR_STRENGTH = 50.0

# Keep p0 off the degenerate endpoints so the Beta shapes stay strictly positive
# (a healthy line where every matched trip advanced medians to 1.0 → beta=0
# otherwise). Mirrors hmm.py's BERNOULLI_FLOOR on the emission's advance_rate.
P0_FLOOR = 1e-3


@dataclass(frozen=True)
class AdvanceBaseline:
    """The normal advance-rate prior for one (route, direction, tod_bin) cell.

    `p0` is the cell's baseline (normal) cross-tick advance fraction — the share
    of matched trips that advance a stop in a healthy tick. It anchors the HMM's
    movement emission: the normal state sits near p0, disrupted below it. Carried
    as the Beta(alpha, beta) prior the emission's responsibility-weighted update
    consumes, with alpha + beta = the prior strength in pseudo-trials.
    """

    p0: float  # baseline advance rate (median over the cell's ticks)
    n: int  # ticks contributing to the cell
    alpha: float  # Beta prior successes: prior_strength * p0
    beta: float  # Beta prior failures: prior_strength * (1 - p0)


def compute_advance_baseline(
    series: dict[tuple[str, str, int], dict[str, int]],
    *,
    prior_strength: float = ADVANCE_PRIOR_STRENGTH,
    min_matched: int = MIN_MATCHED_TRIPS,
    min_samples: int = 20,
) -> dict[tuple[str, str, int], AdvanceBaseline]:
    """Per (route, direction, tod_bin) baseline advance rate, as a Beta prior.

    For each tick with at least `min_matched` cross-tick matches, the advance
    fraction is advanced_n / (advanced_n + stalled_n). The cell's p0 is the
    *median* of those fractions — like compute_baseline for assigned_n, the
    median resists the disrupted minority, so a line that mostly runs well keeps
    a high baseline even with occasional frozen stretches. Cells below
    `min_samples` ticks are omitted (callers treat a missing baseline as "no
    prior", and the emission channel drops out — see hmm.py has_movement).
    """
    buckets: dict[tuple[str, str, int], list[float]] = {}
    for (route, direction, tick), row in series.items():
        matched = row.get("advanced_n", 0) + row.get("stalled_n", 0)
        if matched < min_matched:
            continue
        frac = row.get("advanced_n", 0) / matched
        buckets.setdefault((route, direction, tod_bin(tick)), []).append(frac)

    out: dict[tuple[str, str, int], AdvanceBaseline] = {}
    for key, fracs in buckets.items():
        if len(fracs) < min_samples:
            continue
        p0 = min(max(statistics.median(fracs), P0_FLOOR), 1.0 - P0_FLOOR)
        out[key] = AdvanceBaseline(
            p0=p0,
            n=len(fracs),
            alpha=prior_strength * p0,
            beta=prior_strength * (1.0 - p0),
        )
    return out


def compute_advance_baseline_by_route(
    series: dict[tuple[str, str, int], dict[str, int]],
    *,
    min_matched: int = MIN_MATCHED_TRIPS,
    min_samples: int = 20,
) -> dict[str, float]:
    """Per-route baseline (normal) advance rate — the median cross-tick advance
    fraction pooled over both directions and all times of day.

    Coarser than compute_advance_baseline: the trained emissions aren't
    TOD-conditioned, so the EM prior anchors one normal-state advance_rate per
    route, not a per-(direction, tod) grid. Same median-of-fractions estimator
    and P0 floor. Routes below min_samples ticks are omitted so a route with
    thin movement data gets no fabricated prior (the fit keeps the default).
    """
    buckets: dict[str, list[float]] = {}
    for (route, _direction, _tick), row in series.items():
        matched = row.get("advanced_n", 0) + row.get("stalled_n", 0)
        if matched < min_matched:
            continue
        buckets.setdefault(route, []).append(row.get("advanced_n", 0) / matched)

    out: dict[str, float] = {}
    for route, fracs in buckets.items():
        if len(fracs) < min_samples:
            continue
        out[route] = min(max(statistics.median(fracs), P0_FLOOR), 1.0 - P0_FLOOR)
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


def classify_direction(
    advanced_n: int,
    stalled_n: int,
    baseline: AdvanceBaseline | None,
    *,
    prior_strength: float = CLASSIFY_PRIOR_STRENGTH,
    disrupted_ratio: float = DISRUPTED_RATIO,
    min_matched: int = MIN_MATCHED_TRIPS,
) -> str | None:
    """Beta-Binomial call for one (route, direction) at one tick, or None when it
    can't be judged (fewer than `min_matched` cross-tick matches, or no baseline
    prior for the cell). The posterior mean of the advance rate under a Beta prior
    centered on the cell baseline p0 reads disrupted when it sits at/under
    `disrupted_ratio * p0` — a drop relative to the direction's OWN normal, not a
    global cutoff, so low-baseline lines aren't pinned disrupted."""
    matched = advanced_n + stalled_n
    if matched < min_matched:
        return None
    if baseline is None:
        return None
    post = (prior_strength * baseline.p0 + advanced_n) / (prior_strength + matched)
    return "disrupted" if post <= disrupted_ratio * baseline.p0 else "normal"


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
    truth stays causal and a sustained outage can't lower its own baseline."""
    route_series = build_movement_series(bodies)
    dir_series = build_movement_series_by_direction(bodies)
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
