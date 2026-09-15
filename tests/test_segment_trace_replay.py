"""The 1-minute-clock reconstruction that the segment grade replays
(training/segment_trace_replay.py). Hermetic — no R2.

What these pin is the reconstruction's whole reason to exist: at 1-minute
resolution the intermediate hops of a multi-station move are OBSERVED and
credited to their own from_stop cells, where the 5-minute sampling credits the
whole jump to one cell and drops the rest. If that stops being true, the grade
that gates the migration is measuring nothing.
"""

from __future__ import annotations

from typing import Any

from training.segment_trace_replay import (
    matched_credit_stats,
    trace_to_movement_bodies,
)

T0 = 1_700_000_040  # a multiple of 60


def _trace(
    trip: str, route: str, direction: str | None, seq: list[tuple[int, str]]
) -> list[dict[str, Any]]:
    """One trace body per (minute_offset, stop_id) sighting of `trip`."""
    return [
        {
            "scheduled_at": T0 + m * 60,
            "observed_at": T0 + m * 60,
            "rows": [
                {
                    "trip_id": trip,
                    "route_id": route,
                    "direction": direction,
                    "stop_id": stop,
                    "stop_seq": None,
                    "stopped": False,
                    "vehicle_ts": T0 + m * 60,
                }
            ],
        }
        for m, stop in seq
    ]


def _transitions(
    bodies: list[dict[str, Any]], route: str, direction: str
) -> dict[int, dict[str, int]]:
    return {
        b["observed_at"]: b["rows"][route]["by_direction"][direction]["transitions"]
        for b in bodies
        if route in b["rows"]
    }


def test_each_minute_credits_one_single_hop_transition() -> None:
    """A train hopping A>B>C>D one stop per minute credits each intermediate
    from_stop its own transition — the evidence 5-minute sampling never sees."""
    trace = _trace(
        "t1", "A", "north", [(0, "A01N"), (1, "A02N"), (2, "A03N"), (3, "A04N")]
    )
    bodies = trace_to_movement_bodies(trace, tick_seconds=60)
    trans = _transitions(bodies, "A", "north")
    # First minute has no predecessor, so no transition; each later minute one.
    assert trans[T0] == {}
    assert trans[T0 + 60] == {"A01N>A02N": 1}
    assert trans[T0 + 120] == {"A02N>A03N": 1}
    assert trans[T0 + 180] == {"A03N>A04N": 1}


def test_no_transition_is_assembled_across_a_gap() -> None:
    """A trip absent for a minute has no `prev` when it returns, exactly like the
    Worker's carry, which is overwritten wholesale each tick. Without this a
    multi-minute gap would fabricate one long jump — the very artifact the
    1-minute clock exists to avoid."""
    # Present at minute 0 and 1, absent at 2, back at 3.
    trace = _trace("t1", "A", "north", [(0, "A01N"), (1, "A02N"), (3, "A04N")])
    bodies = trace_to_movement_bodies(trace, tick_seconds=60)
    trans = _transitions(bodies, "A", "north")
    assert trans[T0 + 60] == {"A01N>A02N": 1}  # adjacent, credited
    # Minute 3's body exists (the trip is present) but carries no transition:
    # minute 2 had no sighting, so there is no immediately-preceding stop.
    assert trans[T0 + 180] == {}


def test_a_stall_in_place_is_a_matched_transition() -> None:
    """Standing still (same stop_id two minutes running) is a matched-but-stalled
    transition, not an absence — the throughput branch counts it as a train that
    showed up."""
    trace = _trace("t1", "A", "north", [(0, "A01N"), (1, "A01N")])
    bodies = trace_to_movement_bodies(trace, tick_seconds=60)
    row = bodies[-1]["rows"]["A"]["by_direction"]["north"]
    assert row["transitions"] == {"A01N>A01N": 1}
    assert row["stalled_n"] == 1
    assert row["advanced_n"] == 0


def test_direction_unknown_counts_as_presence_but_credits_no_segment() -> None:
    """A sighting with no resolvable direction still marks the route present (the
    outage-guard input) but cannot key a directional segment cell."""
    trace = _trace("t1", "A", None, [(0, "A01N"), (1, "A02N")])
    bodies = trace_to_movement_bodies(trace, tick_seconds=60)
    row = bodies[-1]["rows"]["A"]
    assert row["vehicles_n"] == 1
    assert row["by_direction"]["north"]["transitions"] == {}
    assert row["by_direction"]["south"]["transitions"] == {}


def test_credit_multiplier_over_5min_sampling() -> None:
    """The measurement the migration rests on, with the sampling rate and the
    intermediate-hop expansion kept apart. A train advancing one station per
    minute for 6 minutes is credited every hop at 1 minute (5 advances across 5
    cells); the 5-minute sample of it sees one A01>A06 jump credited to one
    cell. Since every minute here is an advance (no stalls), the matched and the
    advanced multiplier coincide at 5x — the split matters only once stalls
    enter, which the mixed test below covers."""
    seq = [(m, f"A0{m + 1}N") for m in range(6)]  # A01..A06 over 6 minutes
    trace = _trace("t1", "A", "north", seq)
    one = matched_credit_stats(trace_to_movement_bodies(trace, tick_seconds=60))
    five_trace = [b for b in trace if (b["scheduled_at"] - T0) % 300 == 0]
    five = matched_credit_stats(trace_to_movement_bodies(five_trace, tick_seconds=300))
    assert one == {"matched": 5, "advanced": 5, "stalled": 0, "distinct_cells": 5}
    assert five == {"matched": 1, "advanced": 1, "stalled": 0, "distinct_cells": 1}


def test_stalls_inflate_matched_but_not_the_intermediate_hop_multiplier() -> None:
    """The correction the raw 5.25x needed: a train that stalls most minutes and
    advances occasionally credits a matched transition every minute at 1-minute
    cadence (the sampling-rate inflation) but only one ADVANCE per station it
    actually crossed. The advances-only count is the honest intermediate-hop
    signal; matched is not."""
    # 5 minutes at A01 (stalls), then one hop to A02.
    seq = [(0, "A01N"), (1, "A01N"), (2, "A01N"), (3, "A01N"), (4, "A02N")]
    one = matched_credit_stats(
        trace_to_movement_bodies(_trace("t1", "A", "north", seq), tick_seconds=60)
    )
    assert one["matched"] == 4  # four consecutive-minute transitions
    assert one["stalled"] == 3  # A01>A01 three times
    assert one["advanced"] == 1  # the single real station change


def test_duplicate_archive_object_does_not_double_count() -> None:
    """Idempotent retries write the same minute's rows twice; snapping collapses
    them so a duplicated object cannot inflate a cell's matched count."""
    trace = _trace("t1", "A", "north", [(0, "A01N"), (1, "A02N")])
    dup = [*trace, dict(trace[-1])]  # minute 1 archived twice
    bodies = trace_to_movement_bodies(dup, tick_seconds=60)
    assert _transitions(bodies, "A", "north")[T0 + 60] == {"A01N>A02N": 1}
