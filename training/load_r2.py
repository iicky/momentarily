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
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from momentarily.hmm import Observation, tod_bin
from training.load import TICK_SECONDS, TickObservation
from training.r2_client import R2Config, load_config, make_client

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

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

    keys: list[str] = []
    for d in _date_range(start, end):
        prefix = f"archive/alerts/{d.isoformat()}/"
        keys.extend(_list_keys(client, cfg.bucket, prefix))

    # Parallel fetch — R2 happily handles tens of concurrent GETs.
    fetched_client = client

    def _fetch(k: str) -> dict[str, Any]:
        return _fetch_object(fetched_client, cfg.bucket, k)

    bodies: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for body in pool.map(_fetch, keys):
            bodies.append(body)
    return bodies


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


def build_tick_observations(
    bodies: list[dict[str, Any]],
    *,
    corpus_end: int | None = None,
) -> list[TickObservation]:
    """Reconstruct per-(route, tick) observations from alert-version events.

    For each alert version, walk its active_period start..end on the tick grid
    and add it to that (route, tick) bucket. Sort_order and alert_type come
    from the entity row matching the route within the version's payload.

    `corpus_end` caps any open-ended active_period so we don't extend alerts
    forever. Defaults to the max observed_at across all bodies + one day —
    a reasonable upper bound for "still active at corpus end."
    """
    if not bodies:
        return []

    if corpus_end is None:
        max_observed = max(int(b.get("observed_at") or 0) for b in bodies)
        corpus_end = max_observed + 86_400  # one day past the latest observation

    # bucket[tick][route_id] = {alert_id: (sort_order, alert_type)}
    bucket: dict[int, dict[str, dict[str, tuple[int, str]]]] = {}

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
                tick_bucket = bucket.setdefault(tick, {})
                route_bucket = tick_bucket.setdefault(route_id, {})
                route_bucket.setdefault(alert_id, (sort_order, alert_type))
                tick += TICK_SECONDS

    out: list[TickObservation] = []
    for tick in sorted(bucket):
        for route_id, alerts in bucket[tick].items():
            types = [at for _so, at in alerts.values()]
            obs = Observation(
                alert_count=len(alerts),
                severity_sum=sum(so for so, _at in alerts.values()),
                # "No Scheduled Service" is deliberately NOT a suspension: it's
                # scheduled absence (overnight/weekend non-service on B, W, 7X,
                # ...), which is normal operations. Counting it made ~41% of
                # truth ticks "suspended" in the 2026-06-09 shadow review. It
                # still contributes to alert_count/severity_sum. Mirrors
                # worker/src/derive.ts. See momentarily-vk0.3.
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
