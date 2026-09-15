"""Reconstruct 1-minute vehicle-movement bodies from the per-minute trace, so
the segment classifier can be graded on the 1-minute clock the archive already
carries.

The segment fit/replay stack (load_r2.build_segment_baseline /
build_segment_throughput, segment_replay.tick_inputs) all read one shape: a
vehicle-movement body
    {observed_at, rows: {route: {vehicles_n,
                                 by_direction: {north|south: {transitions:
                                     {"from>to": n}, ...}}}}}
archived every 5 minutes under archive/vehicles/. The Worker builds it with
deriveRouteMovementMetric, which diffs this tick's stop_id against the previous
tick's carry (state/vehicle_stops.json) — a 5-MINUTE cross-tick diff. At that
cadence 48.9% of observed transitions span 2+ stations (A>C skipping B), and the
accumulator credits the jump only to the from_stop cell, so the stations the
train provably crossed get no evidence (the multi-station-jump measurement in
journal.md, 2026-08-24: 48.9% of transitions, ~2.30x dropped credit).

The per-minute trace (archive/trace/, vehicles.deriveTrace) is a FULL snapshot
of every in-service trip's raw stop_id every minute. Diffing consecutive
1-minute snapshots per trip reproduces exactly the same cross-tick transition
the Worker computes — advances and stalls, raw stop_id, no arrival/departure
reconstruction — but at 1-minute resolution, so the intermediate hops are
OBSERVED rather than inferred. `trace_to_movement_bodies` does that diff and
emits one 1-minute body per snapped minute, in the identical shape the 5-minute
stack already consumes, so the only cadence knob the fit/replay need is
`tick_seconds` (=60 here, defaulted to load.TICK_SECONDS=300 everywhere else).

This is a PARALLEL, OFFLINE reconstruction: it reads only archived trace
objects and never touches archive/vehicles/ or state/vehicle_stops.json, so it
cannot perturb the 5-minute signal every trained param still assumes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

# Match the Worker's directionOf output (vehicles.ts) and the by_direction split
# the 5-minute archive carries.
_DIRECTIONS = ("north", "south")


def _snap(poll: int, tick_seconds: int) -> int:
    return (poll // tick_seconds) * tick_seconds


def _stops_by_tick(
    trace_bodies: Iterable[Mapping[str, Any]],
    tick_seconds: int,
) -> dict[int, dict[str, tuple[str, str | None, str]]]:
    """tick -> trip_id -> (route, direction, stop_id), from the raw trace.

    Snapped on `scheduled_at` (the cron minute the poll was meant to fire), the
    same clock the Worker gates on, so a poll that started a few seconds late
    still lands on its intended minute. Duplicate archive objects for one minute
    (idempotent retries write byte-identical rows) collapse onto the same tick,
    last write winning — a no-op when the bytes match. Rows with an empty
    trip_id or stop_id are dropped (can't be diffed / can't key a segment),
    mirroring deriveTrace's own skips.
    """
    by_tick: dict[int, dict[str, tuple[str, str | None, str]]] = {}
    for body in trace_bodies:
        rows = body.get("rows")
        if not isinstance(rows, list):
            continue
        poll = int(body.get("scheduled_at") or body.get("observed_at") or 0)
        if poll <= 0:
            continue
        tick = _snap(poll, tick_seconds)
        slot = by_tick.setdefault(tick, {})
        for raw in cast(list[Any], rows):
            if not isinstance(raw, dict):
                continue
            row = cast(dict[str, Any], raw)
            trip = str(row.get("trip_id") or "")
            stop = str(row.get("stop_id") or "")
            route = str(row.get("route_id") or "")
            if not trip or not stop or not route:
                continue
            direction = row.get("direction")
            direction = direction if direction in _DIRECTIONS else None
            slot[trip] = (route, direction, stop)
    return by_tick


def _empty_dir() -> dict[str, Any]:
    return {"vehicles_n": 0, "advanced_n": 0, "stalled_n": 0, "transitions": {}}


def _empty_row() -> dict[str, Any]:
    return {
        "vehicles_n": 0,
        "advanced_n": 0,
        "stalled_n": 0,
        "by_direction": {d: _empty_dir() for d in _DIRECTIONS},
    }


def trace_to_movement_bodies(
    trace_bodies: Iterable[Mapping[str, Any]],
    *,
    tick_seconds: int = 60,
) -> list[dict[str, Any]]:
    """Per-minute vehicle-movement bodies reconstructed from the trace.

    One body per snapped minute that carried any trip. Its transitions are the
    cross-tick diff against the IMMEDIATELY preceding minute only (tick -
    tick_seconds), exactly mirroring the Worker's carry, which overwrites
    state/vehicle_stops.json wholesale each tick: a trip absent last minute has
    no `prev` and so contributes no transition this minute. That strict
    adjacency is what keeps every credited transition a real single-minute hop
    rather than a jump re-assembled across a feed gap.

    A transition is keyed under the CURRENT tick's (route, direction) using the
    trip's previous stop_id — the same rule deriveRouteMovementMetric applies
    (prevStops carries only stop_id, keyed by trip). `vehicles_n` counts every
    trip present on the route this minute (the outage-guard liveness input),
    direction-known or not.
    """
    by_tick = _stops_by_tick(trace_bodies, tick_seconds)
    bodies: list[dict[str, Any]] = []
    for tick in sorted(by_tick):
        prev = by_tick.get(tick - tick_seconds, {})
        rows_out: dict[str, dict[str, Any]] = {}
        for trip, (route, direction, stop) in by_tick[tick].items():
            row = rows_out.get(route)
            if row is None:
                row = _empty_row()
                rows_out[route] = row
            row["vehicles_n"] += 1
            if direction is None:
                continue
            drow: dict[str, Any] = row["by_direction"][direction]
            drow["vehicles_n"] += 1
            p = prev.get(trip)
            if p is None:
                continue
            pstop = p[2]
            if not pstop:
                continue
            key = f"{pstop}>{stop}"
            drow["transitions"][key] = drow["transitions"].get(key, 0) + 1
            if pstop == stop:
                drow["stalled_n"] += 1
                row["stalled_n"] += 1
            else:
                drow["advanced_n"] += 1
                row["advanced_n"] += 1
        if rows_out:
            bodies.append({"observed_at": tick, "rows": rows_out})
    return bodies


def matched_credit_stats(
    bodies: Iterable[Mapping[str, Any]],
    *,
    counts_from_stop: Any | None = None,
) -> dict[str, int]:
    """Cross-tick transition credit over `bodies` (any shape build_segment_series
    reads), split into the pieces that mean different things.

    Two multipliers fall out of comparing the 5-minute vehicle bodies against
    the 1-minute reconstruction over the same window, and they must not be
    conflated:

      matched  — every transition out of a from_stop, advances AND stalls. This
                 is what the accumulator counts, so its 1min/5min ratio is the
                 total per-tick evidence gain — but that ratio is dominated by
                 the 5x sampling frequency (a train reports a transition every
                 minute instead of every fifth), NOT by intermediate hops.
      advanced — transitions where the train actually changed station
                 (frm != to). Its 1min/5min ratio is the intermediate-hop credit
                 multiplier proper: stations crossed per move, the ~2.30x the
                 earlier trip-pattern inference measured, with the
                 stall-every-minute inflation removed.
      stalled  — frm == to, a train standing still. Reported so the split is
                 auditable; it is the bulk of the raw 5x that `matched` carries.

    `distinct_cells` counts from_stop cells credited any transition; over a
    multi-day window it saturates near the full baselined set on both clocks, so
    the multiplier, not the cell count, is where the dropped evidence shows.
    """
    matched = advanced = stalled = 0
    cells: set[tuple[str, str, str]] = set()
    for body in bodies:
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
                    frm, sep, to = str(pair).partition(">")
                    if not sep or not frm or not to:
                        continue
                    n = int(count or 0)
                    if n <= 0:
                        continue
                    if counts_from_stop is not None and not counts_from_stop(
                        route, direction, frm
                    ):
                        continue
                    matched += n
                    if frm == to:
                        stalled += n
                    else:
                        advanced += n
                    cells.add((route, direction, frm))
    return {
        "matched": matched,
        "advanced": advanced,
        "stalled": stalled,
        "distinct_cells": len(cells),
    }
