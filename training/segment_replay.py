"""Faithful offline replay of the Worker's segment classifier.

WHY A THIRD REPLICA EXISTS, AND WHAT MAKES IT DIFFERENT

Two other modules reconstruct segment calls from the vehicle archive:

  - `movement_backfill.segment_ticks_from_vehicle_bodies` approximates the
    Worker's EWMA as a plain 6-tick trailing sum, deliberately: it is a one-shot
    backfill of the transitions stream and says so in its own comment.
  - `incidents.segment_ticks_with_baseline` copies that accumulation so it can
    substitute a published baseline for a self-trained one, for the incident-
    clustering measurement.

Neither can grade a Worker change, because neither runs the Worker's actual
accumulator. This module does: the same EWMA over advanced/matched, the same
decayed expectation over the timetable's per-bin traversal rates, the same
per-direction vehicle counter, the same two branches in the same order, over the
same baselined cell set. `tests/fixtures/parity_segment_flow.json` pins it to
worker/src/segment_flow.ts tick for tick, so a grade measured here is a
prediction about production rather than about a nearby model.

It reads a `state/segment_params.json`-shaped mapping directly, so the grade is
scored against the document the Worker will actually classify against.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from momentarily.hmm import schedule_bin
from training.hierarchical import PooledCell
from training.load import TICK_SECONDS
from training.load_r2 import ADVANCE_PRIOR_STRENGTH, StopFilter
from training.regime import DEBOUNCE_TICKS, MAX_IDLE_SEC, RegimeEntry, advance_regimes
from training.segments import classify_segment, classify_throughput

# Mirrors of worker/src/segment_flow.ts, including its retuned accumulator
# window. Kept as module constants rather than imported from anywhere: there is
# no shared source across the language boundary, and the parity fixture is what
# proves they agree.
SEGMENT_DECAY = 0.94
MIN_EFF_MATCHED = 3
PRUNE_MATCHED = 0.3
EFF_COUNT_SCALE = 1.0 + SEGMENT_DECAY


@dataclass(frozen=True)
class Policy:
    """The accumulator's two tunable levers, so a caller can replay a policy the
    Worker does not currently run and compare it against the one it does.

    Defaults are the SHIPPED values; `SHIPPED` below is the instance the parity
    fixture pins, so a change to these constants breaks that test rather than
    silently redefining what "the Worker" means.

    `decay` sets the effective window at ~1/(1-decay) ticks and is therefore not
    free: a verdict can be stale by up to that window, so buying coverage here
    is spending onset latency. `eff_count_scale` follows from it — the
    weighted-Poisson correction is a property of the weights.
    """

    decay: float = SEGMENT_DECAY
    min_eff_matched: int = MIN_EFF_MATCHED
    prune_matched: float = PRUNE_MATCHED

    @property
    def eff_count_scale(self) -> float:
        return 1.0 + self.decay

    @property
    def window_ticks(self) -> float:
        return 1.0 / (1.0 - self.decay)


SHIPPED = Policy()

_DIRECTIONS = ("north", "south")


def js_round(x: float) -> int:
    """JavaScript's Math.round: half UP, including on the negative side. Python's
    built-in round() is banker's rounding (round(2.5) == 2), so using it here
    would silently desync the replay from the Worker on every exact .5 — which
    the decayed sums hit often, being sums of integers times powers of
    SEGMENT_DECAY."""
    return math.floor(x + 0.5)


@dataclass(frozen=True)
class TickInput:
    """One archived tick, reduced to what the classifier reads.

    `counts` is {`route|direction|from_stop`: (advanced, matched)} — matched
    counts every cross-tick transition out of the stop, advances and stalls
    alike, the way the Worker's tickCounts does. `vehicles` is
    {route: tracked vehicles} for the routes the feed reported, the
    feed-liveness input to the outage guard.
    """

    observed_at: int
    counts: Mapping[str, tuple[int, int]]
    vehicles: Mapping[str, int]


@dataclass(frozen=True)
class Cell:
    """Decayed advanced / matched / expected for one segment cell."""

    a: float
    m: float
    e: float


@dataclass(frozen=True)
class FlowState:
    """The carried accumulator — segment_flow.json's cells, plus this tick's
    per-route vehicle counts (carried on the doc, not accumulated)."""

    observed_at: int = 0
    cells: dict[str, Cell] = field(default_factory=lambda: cast("dict[str, Cell]", {}))
    vehicles: dict[str, int] = field(default_factory=lambda: cast("dict[str, int]", {}))


def tick_inputs(
    bodies: Iterable[dict[str, Any]],
    *,
    counts_from_stop: StopFilter | None = None,
) -> list[TickInput]:
    """The archived vehicle bodies reduced to one TickInput per snapped tick,
    in time order.

    Ticks the feed reported nothing for (no rows) are DROPPED rather than
    replayed as an all-quiet tick: the Worker never ran a classification for
    them either, and folding them in would decay every accumulator against
    evidence that was never collected. Same guard load_r2.throughput_exposure
    applies to the rate's denominator, so the fit and the replay agree about
    which ticks exist.
    """
    by_tick: dict[int, tuple[dict[str, list[int]], dict[str, int]]] = {}
    for body in bodies:
        rows = cast(dict[str, Any], body.get("rows") or {})
        if not rows:
            continue
        tick = (int(body.get("observed_at") or 0) // TICK_SECONDS) * TICK_SECONDS
        counts, vehicles = by_tick.setdefault(tick, ({}, {}))
        for route, row in rows.items():
            if not isinstance(row, dict):
                continue
            row = cast(dict[str, Any], row)
            n_vehicles = int(row.get("vehicles_n") or 0)
            if n_vehicles > 0:
                vehicles[route] = n_vehicles
            by_dir = cast(dict[str, Any], row.get("by_direction") or {})
            for direction in _DIRECTIONS:
                drow = by_dir.get(direction)
                if not isinstance(drow, dict):
                    continue
                drow = cast(dict[str, Any], drow)
                for pair, count in cast(
                    dict[str, Any], drow.get("transitions") or {}
                ).items():
                    if ">" not in pair:
                        continue
                    frm, to = pair.split(">", 1)
                    n = int(count or 0)
                    if not frm or not to or n <= 0:
                        continue
                    if counts_from_stop is not None and not counts_from_stop(
                        route, direction, frm
                    ):
                        continue
                    acc = counts.setdefault(f"{route}|{direction}|{frm}", [0, 0])
                    if frm != to:
                        acc[0] += n
                    acc[1] += n
    return [
        TickInput(
            observed_at=tick,
            counts={k: (adv, matched) for k, (adv, matched) in counts.items()},
            vehicles=vehicles,
        )
        for tick, (counts, vehicles) in sorted(by_tick.items())
    ]


def fitted_bin(params: Mapping[str, Any], observed_at: int) -> str | None:
    """The published bin whose rates apply at `observed_at`, or None when the
    throughput fit does not cover it — a params doc predating the fit, or a bin
    that never cleared its exposure floor."""
    ticks = cast(dict[str, Any], params.get("throughput") or {}).get("ticks")
    if not isinstance(ticks, dict):
        return None
    bin_key = schedule_bin(observed_at)
    return bin_key if bin_key in ticks else None


def update_flow(
    prev: FlowState,
    tick: TickInput,
    params: Mapping[str, Any],
    policy: Policy = SHIPPED,
) -> FlowState:
    """Advance the accumulator one tick — the Python half of
    segment_flow.ts updateSegmentFlow."""
    cells_doc = cast(dict[str, Any], params.get("cells") or {})
    bin_key = fitted_bin(params, tick.observed_at)

    cells: dict[str, Cell] = {}
    for key, cell in cells_doc.items():
        p = prev.cells.get(key)
        pa, pm, pe = (p.a, p.m, p.e) if p is not None else (0.0, 0.0, 0.0)
        adv, matched = tick.counts.get(key, (0, 0))
        lam = 0.0
        if bin_key is not None:
            lam = float(cast(dict[str, Any], cell.get("lam") or {}).get(bin_key, 0.0))
        a = adv + policy.decay * pa
        m = matched + policy.decay * pm
        e = lam + policy.decay * pe
        if m < policy.prune_matched and e < policy.prune_matched:
            continue
        cells[key] = Cell(a=a, m=m, e=e)

    return FlowState(
        observed_at=tick.observed_at, cells=cells, vehicles=dict(tick.vehicles)
    )


def classify(
    state: FlowState,
    params: Mapping[str, Any],
    policy: Policy = SHIPPED,
) -> dict[str, str]:
    """{cell key: normal|quiet|disrupted} for every baselined cell the two
    branches can judge — the Python half of segment_flow.ts classifySegments.

    Iteration is over the params doc's cells, not the accumulator: a cell that
    saw no train has no accumulator entry, and that silence is exactly what the
    throughput branch reads.
    """
    cells_doc = cast(dict[str, Any], params.get("cells") or {})
    adjacency = cast(dict[str, Any], params.get("adjacency") or {})
    bin_fitted = fitted_bin(params, state.observed_at) is not None

    out: dict[str, str] = {}
    for key, cell_doc in cells_doc.items():
        if key not in adjacency:
            continue
        cell = state.cells.get(key)
        a, m, e = (cell.a, cell.m, cell.e) if cell is not None else (0.0, 0.0, 0.0)
        matched = js_round(m)
        advanced = min(js_round(a), matched)
        p0 = float(cell_doc["p0"])
        call: str | None = None
        if matched >= policy.min_eff_matched:
            call = classify_segment(
                advanced,
                matched - advanced,
                PooledCell(
                    p0=p0,
                    raw=p0,
                    n=int(cell_doc.get("n", 0)),
                    alpha=p0 * ADVANCE_PRIOR_STRENGTH,
                    beta=(1.0 - p0) * ADVANCE_PRIOR_STRENGTH,
                    source="published",
                ),
            )
        if call is None and bin_fitted:
            call = classify_throughput(
                m,
                e,
                key.split("|", 1)[0] in state.vehicles,
                eff_count_scale=policy.eff_count_scale,
            )
        if call is None:
            continue
        out[key] = call
    return out


def replay(
    ticks: Iterable[TickInput],
    params: Mapping[str, Any],
    policy: Policy = SHIPPED,
) -> list[tuple[int, dict[str, str]]]:
    """Per-tick {cell key: call} over the whole run, carrying the accumulator
    forward exactly as the Worker carries segment_flow.json."""
    state = FlowState()
    out: list[tuple[int, dict[str, str]]] = []
    for tick in ticks:
        state = update_flow(state, tick, params, policy)
        out.append((tick.observed_at, classify(state, params, policy)))
    return out


def published_states(
    calls: Sequence[tuple[int, Mapping[str, str]]],
    *,
    debounce_ticks: int = DEBOUNCE_TICKS,
    max_idle_sec: int = MAX_IDLE_SEC,
) -> list[tuple[int, dict[str, str]]]:
    """`replay`'s raw per-tick calls run through the regime clock, giving what the
    snapshot would actually SHOW at each tick rather than what the classifier
    decided at it.

    The two differ in a way that matters for anything time-sensitive, and the
    difference grows with the accumulator's window. A cell the classifier abstains
    on keeps its previous published state for up to `max_idle_sec`, so the surface
    can read disrupted long after the evidence stopped, and can read normal on a
    cell that has said nothing for fifty minutes. Grading raw calls credits a
    policy for opinions no rider ever saw and hides the staleness a wider window
    buys; grading these does neither.

    Goes through training.regime.advance_regimes, the same function the offline
    backfill uses and a hand-port pin of the Worker's own clock, rather than
    reimplementing the debounce.
    """
    entries: dict[str, RegimeEntry] = {}
    out: list[tuple[int, dict[str, str]]] = []
    for observed_at, tick_calls in sorted(calls, key=lambda t: t[0]):
        entries, _changes = advance_regimes(
            entries,
            tick_calls,
            observed_at,
            debounce_ticks=debounce_ticks,
            max_idle_sec=max_idle_sec,
        )
        out.append((observed_at, {k: v.state for k, v in entries.items()}))
    return out


def without_throughput(params: Mapping[str, Any]) -> dict[str, Any]:
    """`params` with the throughput fit stripped — the classifier as it behaved
    before this branch existed, for a before/after comparison that differs in
    exactly one thing. Dropping the exposure map is enough: `fitted_bin` then
    returns None for every tick and the branch never fires."""
    return {k: v for k, v in params.items() if k != "throughput"}
