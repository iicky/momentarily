"""Independent current-state validation from the vehicle-movement metric.

Synthetic series — no R2. Covers the load_r2 derivation (build_movement_series,
derive_movement_state, build_movement_truth) that holds vehicle positions out as
a contemporaneous truth for the regime confusion matrix. See momentarily-vy0.
"""

from __future__ import annotations

from training.load_r2 import (
    build_movement_series,
    build_movement_truth,
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
) -> dict[str, int]:
    return {
        "vehicles_n": vehicles_n,
        "stopped_n": stopped_n,
        "moving_n": moving_n,
        "advanced_n": advanced_n,
        "stalled_n": stalled_n,
    }


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
    assert derive_movement_state(_row(vehicles_n=0)) == "suspended"


def test_present_but_frozen_reads_disrupted():
    # 10 trains, only 1 of 10 matched trips advanced → frozen.
    assert (
        derive_movement_state(_row(vehicles_n=10, advanced_n=1, stalled_n=9))
        == "disrupted"
    )


def test_present_and_advancing_reads_normal():
    assert (
        derive_movement_state(_row(vehicles_n=10, advanced_n=9, stalled_n=1))
        == "normal"
    )


def test_too_few_matched_trips_is_unjudgeable():
    # Trains present but only 2 matched across ticks (< min) → can't judge.
    assert derive_movement_state(_row(vehicles_n=4, advanced_n=2, stalled_n=0)) is None


def test_thresholds_are_tunable():
    row = _row(vehicles_n=10, advanced_n=3, stalled_n=7)  # advance_frac 0.30
    # Default frozen threshold 0.25 → above it → normal.
    assert derive_movement_state(row) == "normal"
    # Raise the threshold and the same row reads frozen.
    assert derive_movement_state(row, frozen_advance_frac=0.5) == "disrupted"


def test_build_movement_truth_drops_unjudgeable_ticks():
    bodies = [
        {
            "observed_at": T0,
            "rows": {
                "A": _row(vehicles_n=0),  # suspended
                "B": _row(vehicles_n=10, advanced_n=9, stalled_n=1),  # normal
                "C": _row(vehicles_n=3, advanced_n=1, stalled_n=0),  # too few → dropped
            },
        }
    ]
    truth = build_movement_truth(bodies)
    assert truth[("A", T0)] == "suspended"
    assert truth[("B", T0)] == "normal"
    assert ("C", T0) not in truth
