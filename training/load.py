"""Load collector JSONL → per-route per-tick HMM observations.

The collector dumps the full set of currently-active alerts every 5 min. For
HMM training we want one Observation per (route_id, tick) — alert_count is the
distinct alert IDs active at that tick mentioning the route, severity_sum is
the sum of their sort_order values, has_suspended_alert is whether any of
their alert_types names a suspension.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from momentarily.hmm import Observation, tod_bin
from momentarily.mapping import is_hmm_excluded

# Cron cadence — collector polls every 5 min, ticks align on this boundary.
TICK_SECONDS = 300

# sort_order in mercury_entity_selector is "MTASBWY:6:29" — trailing integer.
_SORT_ORDER_RE = re.compile(r":(\d+)$")


@dataclass(frozen=True)
class TickObservation:
    """Observation tagged with route + tick so callers can group/sort."""

    route_id: str
    tick: int  # epoch seconds, snapped to TICK_SECONDS
    observation: Observation
    # Counted (HMM-included) alert_types active on this route-tick. Only the R2
    # truth builder populates it — used by the review to grade ground truth by
    # severity; the HMM training path leaves it empty.
    disruptive_types: tuple[str, ...] = ()


def _snap_tick(epoch: int) -> int:
    return (epoch // TICK_SECONDS) * TICK_SECONDS


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


def iter_records(paths: Iterable[Path]) -> Iterator[dict[str, Any]]:
    """Yield parsed JSONL records from a list of files, in file order."""
    for path in paths:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield cast(dict[str, Any], json.loads(line))


def build_observations(
    records: Iterable[dict[str, Any]],
) -> list[TickObservation]:
    """Aggregate raw alert poll records into per-(route, tick) observations.

    Multiple polls within a 5-min window are merged: an alert seen in any poll
    of that window counts once. Sort_order is summed across distinct alerts in
    the route×tick bucket.
    """
    # bucket[tick][route_id] = {alert_id: (sort_order, alert_type)}
    bucket: dict[int, dict[str, dict[str, tuple[int, str]]]] = {}

    for record in records:
        observed_at = int(record["observed_at"])
        tick = _snap_tick(observed_at)

        alert_envelope = cast(dict[str, Any], record.get("alert") or {})
        alert_id = alert_envelope.get("id")
        if not isinstance(alert_id, str):
            continue
        alert_payload = cast(dict[str, Any], alert_envelope.get("alert") or {})
        alert_type = _alert_type(alert_payload)

        # Each informed_entity contributes its sort_order to the route it names.
        # An entity with route_id is "this alert applies to that route."
        entities = cast(
            list[dict[str, Any]], alert_payload.get("informed_entity") or []
        )
        for entity in entities:
            route_id = entity.get("route_id")
            if not isinstance(route_id, str):
                continue
            sort_order = _sort_order(entity)
            tick_bucket = bucket.setdefault(tick, {})
            route_bucket = tick_bucket.setdefault(route_id, {})
            # Keep the first (alert_id, sort_order, alert_type) we see for this
            # alert in this tick — subsequent occurrences are duplicates from
            # other polls in the same window.
            route_bucket.setdefault(alert_id, (sort_order, alert_type))

    out: list[TickObservation] = []
    for tick in sorted(bucket):
        for route_id, alerts in bucket[tick].items():
            # Extra service (good news) and scheduled non-service drop out of the
            # HMM observation so the filter reads quiet. Mirrors training/load_r2.py
            # and worker/src/derive.ts.
            counted = [
                (so, at) for so, at in alerts.values() if not is_hmm_excluded(at)
            ]
            alert_count = len(counted)
            severity_sum = sum(so for so, _at in counted)
            types = [at for _so, at in counted]
            out.append(
                TickObservation(
                    route_id=route_id,
                    tick=tick,
                    observation=Observation(
                        alert_count=alert_count,
                        severity_sum=severity_sum,
                        has_suspended_alert=_match(
                            types,
                            ("Suspend", "No Trains"),
                            exclude_prefix="Planned -",
                        ),
                        has_delays=_match(
                            types,
                            ("Delays", "Severe Delays"),
                            exclude_prefix="Planned -",
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
                    ),
                )
            )
    return out


def _match(
    types: list[str],
    needles: tuple[str, ...],
    *,
    exclude_prefix: str | None = None,
) -> bool:
    """Whether any alert_type in `types` contains one of the needles.

    If exclude_prefix is set, alert_types starting with that prefix are skipped
    so that "Planned - Stops Skipped" doesn't double-count as a service change
    when has_planned already captures it.
    """
    for at in types:
        if exclude_prefix and at.startswith(exclude_prefix):
            continue
        if any(needle in at for needle in needles):
            return True
    return False


def fill_quiet_ticks(
    observations: list[TickObservation],
    route_id: str,
    start_tick: int | None = None,
    end_tick: int | None = None,
) -> list[TickObservation]:
    """Return a contiguous sequence of ticks for one route, inserting quiet
    observations (no alerts) for ticks where the route had no entry.

    The HMM needs evenly-spaced observations to compute dwell times correctly;
    without filling, a quiet route that vanishes from the data would look like
    its dwell time stretched across the gap.
    """
    route_obs = [o for o in observations if o.route_id == route_id]
    if not route_obs:
        return []

    ticks_present = {o.tick: o for o in route_obs}
    first = start_tick if start_tick is not None else min(ticks_present)
    last = end_tick if end_tick is not None else max(ticks_present)

    out: list[TickObservation] = []
    tick = first
    while tick <= last:
        if tick in ticks_present:
            out.append(ticks_present[tick])
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


def load_route_series(data_dir: Path, route_id: str) -> list[TickObservation]:
    """End-to-end convenience: read all alerts/*.jsonl under data_dir, build
    observations, return one route's contiguous tick series."""
    alerts_dir = data_dir / "alerts"
    paths = sorted(alerts_dir.glob("*.jsonl"))
    records = iter_records(paths)
    observations = build_observations(records)
    return fill_quiet_ticks(observations, route_id)
