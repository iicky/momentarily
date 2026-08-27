"""Generate the Python<->TypeScript segment-classifier parity fixture.

worker/src/segment_flow.ts and training/segment_replay.py run the same
accumulator and the same two branches over the same baselined cells. They will
drift unless something pins them together, and drift here is expensive in a
specific way: segment_replay is what grades a change to the classifier BEFORE it
ships. A replay that is a near-miss of the Worker produces a grade about a model
that never runs, which is worse than no grade at all.

The fixture is a segment_params-shaped document plus a canonical tick sequence,
and the accumulator state and calls Python produces for each tick. Both
languages replay it — tests/test_segment_replay.py and
worker/test/segment_parity.test.ts.

Coverage the tick sequence is built for: cold start; the advance branch taking
over once matched clears MIN_EFF_MATCHED; a cell going silent against a real
expectation (the whole point of the branch); a cell whose bin expects nothing
reading quiet; the expectation carrying across a schedule_bin edge; a
vehicle-feed dropout abstaining instead of blaming the railway; a cell pruning
out of the accumulator and re-entering.

Run:  uv run python -m scripts.gen_segment_parity_fixture
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from momentarily.hmm import schedule_bin
from training.segment_replay import FlowState, TickInput, classify, update_flow

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "parity_segment_flow.json"
)

TICK_SECONDS = 300

# Wed 2026-08-19 09:15 ET. Ticks 0-8 sit in schedule_bin wd09 and ticks 9-18 in
# wd10, so the run crosses exactly one bin edge, at tick 9, and the decayed
# expectation has to carry wd09's rate across it. A fixed epoch, not a computed
# "now": the fixture has to be stable, and both its weekday and its hour are
# load-bearing for the bin labels below.
START = 1787145300

BUSY = "F|south|A09S"  # scheduled, and the cell every branch is exercised on
SPARSE = "F|south|A10S"  # scheduled at the first bin only
IDLE = "F|south|A11S"  # never scheduled: always quiet
ORPHAN = "F|south|Z99S"  # baselined but missing from adjacency: never judged

# The two bins the run straddles. Spelled out so the fixture reads as data;
# _check_bins below proves the labels match the epoch.
BIN_A = "wd09"
BIN_B = "wd10"

PARAMS: dict[str, Any] = {
    "schema_version": "1",
    "trained_at": START,
    "min_share": 0.5,
    "topology_source": "gtfs_static",
    "cells": {
        BUSY: {"p0": 0.9, "n": 4000, "lam": {BIN_A: 2.0, BIN_B: 2.0}},
        SPARSE: {"p0": 0.9, "n": 900, "lam": {BIN_A: 1.9}},
        IDLE: {"p0": 0.05, "n": 40},
        ORPHAN: {"p0": 0.9, "n": 100, "lam": {BIN_A: 2.0, BIN_B: 2.0}},
    },
    "adjacency": {
        BUSY: {"to": "A10S", "source": "gtfs_static"},
        SPARSE: {"to": "A11S", "source": "gtfs_static"},
        IDLE: {"to": "A12S", "source": "gtfs_static"},
    },
    "throughput": {
        "bin": "schedule_bin",
        "min_ticks": 20,
        "ticks": {BIN_A: 600, BIN_B: 600},
    },
}


def _t(i: int) -> int:
    return START + i * TICK_SECONDS


# (advanced, matched) per cell, and tracked vehicles per ROUTE — the outage
# guard's input is "did the feed report this route at all this tick".
TICKS: list[TickInput] = [
    # 0-1: healthy traffic on BUSY. Too thin for the advance branch yet, but the
    # throughput branch already has an opinion.
    TickInput(_t(0), {BUSY: (2, 2), SPARSE: (2, 2)}, {"F": 20}),
    TickInput(_t(1), {BUSY: (2, 2), SPARSE: (2, 2)}, {"F": 20}),
    # 2: matched has decayed-summed past MIN_EFF_MATCHED, so the advance branch
    # takes over on BUSY and the throughput fit stops mattering for it.
    TickInput(_t(2), {BUSY: (2, 2), SPARSE: (2, 2)}, {"F": 20}),
    # 3: a stall-heavy tick on BUSY. Still well above DISRUPTED_RATIO * p0 once
    # the window's earlier advances are counted, so it reads normal — the point
    # is that the ADVANCE branch is the one answering, not the throughput fit.
    TickInput(_t(3), {BUSY: (0, 4)}, {"F": 20}),
    # 4-8: total silence on every cell while the feed still sees trains. BUSY
    # falls back through the advance branch to the throughput branch as its
    # matched decays away, and the expectation it accrues turns that silence
    # into a disrupted call. This is the behaviour the epic exists for.
    *[TickInput(_t(i), {}, {"F": 20}) for i in range(4, 9)],
    # 9: crosses into BIN_B, where SPARSE is scheduled for nothing. Its decayed
    # expectation still holds BIN_A's rate, so it cannot flip to quiet at once.
    TickInput(_t(9), {}, {"F": 20}),
    # 10-13: deeper into BIN_B; SPARSE's expectation decays until it drops under
    # the power floor and the call becomes quiet.
    *[TickInput(_t(i), {}, {"F": 20}) for i in range(10, 14)],
    # 14-17: the vehicle feed stops reporting route F entirely. Cells that still
    # expect traffic must abstain, not read disrupted; IDLE stays quiet
    # regardless, because quiet is a claim about the timetable.
    TickInput(_t(14), {}, {}),
    *[TickInput(_t(i), {}, {}) for i in range(15, 18)],
    # 18: traffic and the feed return together.
    TickInput(_t(18), {BUSY: (3, 3), SPARSE: (3, 3)}, {"F": 20}),
]


def _check_bins() -> None:
    """BIN_A/BIN_B are hand-written data, so prove they are the right data: the
    run must sit in BIN_A through tick 8 and in BIN_B from tick 9 on, where every
    comment above says it does."""
    bins = [schedule_bin(t.observed_at) for t in TICKS]
    expected = [BIN_A] * 9 + [BIN_B] * (len(TICKS) - 9)
    if bins != expected:
        raise AssertionError(f"fixture bin layout drifted: {bins} != {expected}")


def build() -> dict[str, Any]:
    _check_bins()
    steps: list[dict[str, Any]] = []
    state = FlowState()
    for tick in TICKS:
        state = update_flow(state, tick, PARAMS)
        steps.append(
            {
                "observed_at": tick.observed_at,
                "counts": {k: list(v) for k, v in sorted(tick.counts.items())},
                "vehicles": dict(sorted(tick.vehicles.items())),
                "cells": {k: asdict(v) for k, v in sorted(state.cells.items())},
                "state_vehicles": dict(sorted(state.vehicles.items())),
                "calls": dict(sorted(classify(state, PARAMS).items())),
            }
        )
    return {
        "tick_seconds": TICK_SECONDS,
        "params": PARAMS,
        "steps": steps,
    }


def main() -> int:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(build(), indent=2) + "\n")
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
