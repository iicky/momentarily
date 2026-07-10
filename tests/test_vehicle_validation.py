"""Independent current-state validation from the vehicle-movement metric.

Synthetic series — no R2. Covers the load_r2 derivation (build_movement_series,
derive_movement_state, build_movement_truth) that holds vehicle positions out as
a contemporaneous truth for the regime confusion matrix. See momentarily-vy0.
"""

from __future__ import annotations

from typing import Any

from momentarily.hmm import tod_bin
from training.load_r2 import (
    AdvanceBaseline,
    build_movement_series,
    build_movement_truth,
    classify_direction,
    derive_movement_state,
)

TICK = 300
T0 = 1_700_000_100  # tick-aligned


def _row(
    vehicles_n: int = 0,
    advanced_n: int = 0,
    stalled_n: int = 0,
    stopped_n: int = 0,
    moving_n: int = 0,
    by_direction: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "vehicles_n": vehicles_n,
        "stopped_n": stopped_n,
        "moving_n": moving_n,
        "advanced_n": advanced_n,
        "stalled_n": stalled_n,
    }
    if by_direction is not None:
        row["by_direction"] = by_direction
    return row


def test_build_movement_series_keeps_all_counters():
    bodies = [
        {
            "observed_at": T0,
            "fresh_feeds": ["ace"],
            "rows": {
                "A": _row(vehicles_n=12, advanced_n=9, stalled_n=1),
                "C": _row(vehicles_n=8),
            },
        },
        {"observed_at": T0 + TICK, "rows": {"A": _row(vehicles_n=0)}},
    ]
    series = build_movement_series(bodies)
    assert series[("A", T0)]["vehicles_n"] == 12
    assert series[("A", T0)]["advanced_n"] == 9
    assert series[("C", T0)]["vehicles_n"] == 8
    assert series[("A", T0 + TICK)]["vehicles_n"] == 0


def test_no_vehicles_reads_suspended():
    assert derive_movement_state(_row(vehicles_n=0), {}, {}) == "suspended"


def test_classify_direction_normal_trunk():
    # p0=0.9, advanced=8, stalled=1 (matched=9): post = (8*0.9+8)/(8+9) = 15.2/17 ~ 0.894 > 0.45.
    baseline = AdvanceBaseline(p0=0.9, n=50, alpha=45, beta=5)
    assert classify_direction(8, 1, baseline) == "normal"


def test_classify_direction_disrupted_trunk():
    # p0=0.9, advanced=0, stalled=12 (matched=12): post = 7.2/20 = 0.36 <= 0.45.
    baseline = AdvanceBaseline(p0=0.9, n=50, alpha=45, beta=5)
    assert classify_direction(0, 12, baseline) == "disrupted"


def test_classify_direction_shuttle_debiasing():
    # The whole point of vhh.11: a shuttle running at its own ~10% normal
    # advance rate must read normal — the old fixed-0.25 rule called this
    # disrupted. post = (8*0.1+1)/(8+10) = 1.8/18 = 0.10 > 0.05 (ratio*p0).
    baseline = AdvanceBaseline(p0=0.1, n=50, alpha=5, beta=45)
    assert classify_direction(1, 9, baseline) == "normal"


def test_classify_direction_too_few_matched_is_none():
    baseline = AdvanceBaseline(p0=0.9, n=50, alpha=45, beta=5)
    assert (
        classify_direction(1, 1, baseline) is None
    )  # matched=2 < 3, even with a baseline


def test_classify_direction_no_baseline_is_none():
    assert classify_direction(8, 1, None) is None


def test_derive_movement_state_worst_of_one_direction_disrupted():
    route_row = _row(vehicles_n=10)
    dir_rows = {
        "north": _row(advanced_n=0, stalled_n=12),  # disrupted
        "south": _row(advanced_n=8, stalled_n=1),  # normal
    }
    baselines = {
        "north": AdvanceBaseline(p0=0.9, n=50, alpha=45, beta=5),
        "south": AdvanceBaseline(p0=0.9, n=50, alpha=45, beta=5),
    }
    assert derive_movement_state(route_row, dir_rows, baselines) == "disrupted"


def test_derive_movement_state_both_directions_normal():
    route_row = _row(vehicles_n=10)
    dir_rows = {
        "north": _row(advanced_n=8, stalled_n=1),
        "south": _row(advanced_n=8, stalled_n=1),
    }
    baselines = {
        "north": AdvanceBaseline(p0=0.9, n=50, alpha=45, beta=5),
        "south": AdvanceBaseline(p0=0.9, n=50, alpha=45, beta=5),
    }
    assert derive_movement_state(route_row, dir_rows, baselines) == "normal"


def test_derive_movement_state_both_directions_unjudgeable_is_none():
    route_row = _row(vehicles_n=10)
    dir_rows = {
        "north": _row(advanced_n=1, stalled_n=1),  # too few matched
        "south": None,  # no data at all
    }
    baselines = {
        "north": AdvanceBaseline(p0=0.9, n=50, alpha=45, beta=5),
        "south": None,
    }
    assert derive_movement_state(route_row, dir_rows, baselines) is None


def test_build_movement_truth_drops_unjudgeable_ticks():
    tb = tod_bin(T0)
    bodies = [
        {
            "observed_at": T0,
            "rows": {
                "A": _row(vehicles_n=0),  # suspended
                "B": _row(
                    vehicles_n=10,
                    by_direction={
                        "north": {"vehicles_n": 5, "advanced_n": 8, "stalled_n": 1},
                        "south": {"vehicles_n": 5, "advanced_n": 4, "stalled_n": 1},
                    },
                ),  # north normal against its baseline -> route normal
                "C": _row(
                    vehicles_n=10,
                    by_direction={
                        "north": {"vehicles_n": 5, "advanced_n": 1, "stalled_n": 1},
                        "south": {"vehicles_n": 5, "advanced_n": 1, "stalled_n": 1},
                    },
                ),  # too few matched in both directions -> dropped
                "D": _row(
                    vehicles_n=10,
                    by_direction={
                        "north": {"vehicles_n": 5, "advanced_n": 8, "stalled_n": 1},
                        "south": {"vehicles_n": 5, "advanced_n": 4, "stalled_n": 1},
                    },
                ),  # enough matches but no baseline anywhere -> dropped
            },
        }
    ]
    movement_baseline = {
        ("B", "north", tb): AdvanceBaseline(p0=0.9, n=50, alpha=45, beta=5),
        ("C", "north", tb): AdvanceBaseline(p0=0.9, n=50, alpha=45, beta=5),
    }
    truth = build_movement_truth(bodies, movement_baseline=movement_baseline)
    assert truth[("A", T0)] == "suspended"
    assert truth[("B", T0)] == "normal"
    assert ("C", T0) not in truth
    assert ("D", T0) not in truth
