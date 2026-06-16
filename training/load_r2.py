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
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol, cast

from blake3 import blake3

from momentarily.hmm import Observation, tod_bin
from momentarily.mapping import is_hmm_excluded
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
            # Extra service (good news) and scheduled non-service stay on the
            # display surfaces but drop out of the HMM observation so the filter
            # reads quiet. Mirrors training/load.py and worker/src/derive.ts.
            counted = [
                (so, at) for so, at in alerts.values() if not is_hmm_excluded(at)
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
            out.append(TickObservation(route_id=route_id, tick=tick, observation=obs))
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
