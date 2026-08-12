"""Arrival + traversal reconstruction from the per-minute trace (training/trace.py).

Synthetic snapshots only — no R2. Each case pins one rule the diff has to honour
from worker/src/vehicles.ts deriveTrace's contract.
"""

from __future__ import annotations

from typing import Any

from training.trace import (
    EXACT,
    INTERVAL,
    RIGHT,
    arrivals_from_trace,
    to_dwell_samples,
    traversals_from_trace,
)

T0 = 1_786_552_200


def _row(
    stop_id: str,
    *,
    stopped: bool,
    seq: int | None = None,
    trip: str = "t1",
    vehicle_ts: int | None = None,
) -> dict[str, Any]:
    return {
        "trip_id": trip,
        "route_id": "A",
        "direction": "south",
        "stop_id": stop_id,
        "stop_seq": seq if stopped else None,
        "stopped": stopped,
        "vehicle_ts": vehicle_ts,
    }


def _body(minute: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    at = T0 + minute * 60
    return {
        "observed_at": at + 3,
        "scheduled_at": at,
        "fresh_feeds": ["ace"],
        "rows": rows,
    }


def test_arrival_is_the_first_stopped_snapshot_not_every_one():
    """A train standing at a stop for three polls arrived once."""
    bodies = [
        _body(0, [_row("A1S", stopped=False)]),
        _body(1, [_row("A1S", stopped=True, seq=1)]),
        _body(2, [_row("A1S", stopped=True, seq=1)]),
        _body(3, [_row("A1S", stopped=True, seq=1)]),
    ]
    arrivals = arrivals_from_trace(bodies)
    assert [(a.stop_id, a.at) for a in arrivals] == [("A1S", T0 + 60)]


def test_heading_to_a_stop_is_not_an_arrival_at_it():
    """stop_id names the stop a train is heading TO while in transit; only the
    stopped=true row is the arrival."""
    bodies = [
        _body(0, [_row("A2S", stopped=False)]),
        _body(1, [_row("A2S", stopped=False)]),
    ]
    assert arrivals_from_trace(bodies) == []


def test_arrival_time_prefers_the_feeds_own_vehicle_ts():
    """vehicle_ts is finer than the 1-minute poll — that is what makes arrival
    timing better than the cadence."""
    bodies = [_body(0, [_row("A1S", stopped=True, seq=1, vehicle_ts=T0 - 17)])]
    (arrival,) = arrivals_from_trace(bodies)
    assert arrival.at == T0 - 17


def test_arrival_falls_back_to_the_scheduled_second_without_vehicle_ts():
    bodies = [_body(0, [_row("A1S", stopped=True, seq=1)])]
    (arrival,) = arrivals_from_trace(bodies)
    assert arrival.at == T0


def test_consecutive_stop_seq_is_an_exact_traversal():
    """Arrival to arrival is the upper bound; departure to arrival is the lower
    one, and the latter is what the timetable measures."""
    bodies = [
        _body(0, [_row("A1S", stopped=True, seq=1, vehicle_ts=T0)]),
        _body(1, [_row("A2S", stopped=False, vehicle_ts=T0 + 40)]),
        _body(2, [_row("A2S", stopped=True, seq=2, vehicle_ts=T0 + 95)]),
    ]
    traversals, stats = traversals_from_trace(bodies)
    (hop,) = traversals
    assert (hop.from_stop, hop.to_stop, hop.seconds) == ("A1S", "A2S", 95)
    assert hop.moving_seconds == 55  # 95 total less the 40s dwell at A1S
    assert (hop.n_hops, hop.censoring) == (1, EXACT)
    assert (stats.n_exact, stats.n_interval, stats.n_right) == (1, 0, 0)
    assert stats.n_with_moving_time == 1


def test_no_in_transit_sighting_leaves_moving_seconds_unknown():
    """Stopped at one stop, then stopped at the next: the poll never caught the
    train in between, so there is no lower bound on travel time — and 0 is not
    one."""
    bodies = [
        _body(0, [_row("A1S", stopped=True, seq=1, vehicle_ts=T0)]),
        _body(1, [_row("A2S", stopped=True, seq=2, vehicle_ts=T0 + 60)]),
    ]
    traversals, stats = traversals_from_trace(bodies)
    (hop,) = traversals
    assert hop.seconds == 60
    assert hop.moving_seconds is None
    assert to_dwell_samples(traversals) == []
    assert stats.n_with_moving_time == 0


def test_a_stop_seq_jump_is_interval_censored_not_a_single_hop():
    """31 -> 33 means an arrival was missed between polls: the two hop times are
    known only to sum to the span."""
    bodies = [
        _body(0, [_row("A1S", stopped=True, seq=31, vehicle_ts=T0)]),
        _body(2, [_row("A3S", stopped=True, seq=33, vehicle_ts=T0 + 200)]),
    ]
    traversals, stats = traversals_from_trace(bodies)
    (hop,) = traversals
    assert (hop.n_hops, hop.seconds, hop.censoring) == (2, 200, INTERVAL)
    assert stats.n_interval == 1
    assert stats.n_exact == 0


def test_disappearing_mid_hop_is_right_censored():
    """Last seen heading somewhere else: a next hop existed and was under way,
    so the traversal out of the last arrival is censored, not lost."""
    bodies = [
        _body(0, [_row("A1S", stopped=True, seq=1, vehicle_ts=T0)]),
        _body(1, [_row("A2S", stopped=False, vehicle_ts=T0 + 55)]),
    ]
    traversals, stats = traversals_from_trace(bodies)
    (hop,) = traversals
    assert (hop.from_stop, hop.to_stop, hop.censoring) == ("A1S", None, RIGHT)
    assert hop.seconds == 55
    assert (hop.n_hops, stats.n_right) == (None, 1)


def test_a_trip_that_ends_standing_at_its_last_stop_emits_no_censored_hop():
    """It may simply have finished its run. Inventing an 'at least this long'
    observation for a hop that never existed would bias every fit upward."""
    bodies = [
        _body(0, [_row("A9S", stopped=True, seq=9, vehicle_ts=T0)]),
        _body(1, [_row("A9S", stopped=True, seq=9, vehicle_ts=T0 + 60)]),
    ]
    traversals, stats = traversals_from_trace(bodies)
    assert traversals == []
    assert stats.n_right == 0


def test_a_trip_last_seen_standing_further_along_is_not_censored():
    """It already arrived — the feed just omitted stop_seq, so the hop can't be
    timed. Recording 'still running at T' there would be false, not merely
    unverified, and would stretch every fitted traversal."""
    bodies = [
        _body(0, [_row("A1S", stopped=True, seq=1, vehicle_ts=T0)]),
        _body(1, [_row("A2S", stopped=True, seq=None, vehicle_ts=T0 + 60)]),
    ]
    traversals, stats = traversals_from_trace(bodies)
    assert traversals == []
    assert (stats.n_right, stats.n_unknown_seq) == (0, 1)


def test_a_feed_gap_after_the_last_in_transit_sighting_keeps_the_bound_honest():
    """The censored duration ends at the last moment the train was OBSERVED in
    transit, so a feed outage that follows cannot inflate it."""
    bodies = [
        _body(0, [_row("A1S", stopped=True, seq=1, vehicle_ts=T0)]),
        _body(1, [_row("A2S", stopped=False, vehicle_ts=T0 + 50)]),
        # ace stopped decoding: the trip is absent from the next ten polls.
        *[
            {"scheduled_at": T0 + m * 60, "fresh_feeds": ["bdfm"], "rows": []}
            for m in range(2, 12)
        ],
    ]
    traversals, _stats = traversals_from_trace(bodies)
    (hop,) = traversals
    assert hop.censoring == RIGHT
    assert hop.seconds == 50  # not the 660s the gap would have implied


def test_backwards_stop_seq_is_counted_and_dropped():
    """Relays and trip_id reuse run the sequence backwards; that is not a
    traversal and must not be silently folded in as one."""
    bodies = [
        _body(0, [_row("A5S", stopped=True, seq=5, vehicle_ts=T0)]),
        _body(1, [_row("A2S", stopped=True, seq=2, vehicle_ts=T0 + 60)]),
    ]
    traversals, stats = traversals_from_trace(bodies)
    assert traversals == []
    assert stats.n_backwards == 1


def test_trips_are_reconstructed_independently():
    bodies = [
        _body(
            0,
            [
                _row("A1S", stopped=True, seq=1, trip="t1", vehicle_ts=T0),
                _row("B1S", stopped=True, seq=7, trip="t2", vehicle_ts=T0),
            ],
        ),
        _body(
            1,
            [
                _row("A2S", stopped=True, seq=2, trip="t1", vehicle_ts=T0 + 60),
                _row("B2S", stopped=True, seq=8, trip="t2", vehicle_ts=T0 + 90),
            ],
        ),
    ]
    traversals, stats = traversals_from_trace(bodies)
    assert {(t.trip_id, t.seconds) for t in traversals} == {("t1", 60), ("t2", 90)}
    assert stats.n_trips == 2


def test_out_of_order_bodies_are_sorted_before_diffing():
    """Objects come back from R2 in listing order, not time order."""
    late = _body(2, [_row("A2S", stopped=True, seq=2, vehicle_ts=T0 + 120)])
    early = _body(0, [_row("A1S", stopped=True, seq=1, vehicle_ts=T0)])
    traversals, _stats = traversals_from_trace([late, early])
    (hop,) = traversals
    assert (hop.from_stop, hop.to_stop, hop.seconds) == ("A1S", "A2S", 120)


def test_dwell_samples_keep_exact_and_censored_and_drop_interval():
    """The survival fitters carry a right-censored likelihood only; a multi-hop
    span is an upper bound they cannot express, so it is dropped rather than
    split into fabricated observations. Durations are departure to arrival, so
    the origin dwell never lands in a traversal fit."""
    bodies = [
        _body(0, [_row("A1S", stopped=True, seq=1, vehicle_ts=T0)]),
        _body(1, [_row("A2S", stopped=False, vehicle_ts=T0 + 30)]),
        _body(2, [_row("A2S", stopped=True, seq=2, vehicle_ts=T0 + 80)]),
        _body(3, [_row("A3S", stopped=False, vehicle_ts=T0 + 120)]),
        _body(4, [_row("A4S", stopped=True, seq=4, vehicle_ts=T0 + 300)]),
        _body(5, [_row("A5S", stopped=False, vehicle_ts=T0 + 340)]),
        _body(6, [_row("A5S", stopped=False, vehicle_ts=T0 + 400)]),
    ]
    traversals, stats = traversals_from_trace(bodies)
    assert (stats.n_exact, stats.n_interval, stats.n_right) == (1, 1, 1)
    # 50s for the observed hop (80 - 30, not the 80s including dwell), and the
    # in-progress hop censored at 60s (400 - 340), not at 100s from its arrival.
    assert sorted(to_dwell_samples(traversals)) == [(50, True), (60, False)]


def test_a_reused_trip_id_after_a_long_absence_starts_a_fresh_run():
    """NYCT reuses trip_ids. A later run that begins stopped where the previous
    one ended must not have its first arrival swallowed as 'still standing
    here', and must not inherit the old run's position."""
    bodies = [
        _body(0, [_row("A9S", stopped=True, seq=9, vehicle_ts=T0)]),
        # ... 30 minutes with no sighting of this trip_id at all ...
        _body(30, [_row("A9S", stopped=True, seq=9, vehicle_ts=T0 + 1800)]),
        _body(31, [_row("A8S", stopped=False, vehicle_ts=T0 + 1860)]),
        _body(32, [_row("A8S", stopped=True, seq=10, vehicle_ts=T0 + 1920)]),
    ]
    arrivals = arrivals_from_trace(bodies)
    assert [a.at for a in arrivals] == [T0, T0 + 1800, T0 + 1920]
    # The hop belongs to the second run; nothing spans the 30-minute gap.
    traversals, _stats = traversals_from_trace(bodies)
    assert [(t.from_stop, t.to_stop, t.seconds) for t in traversals] == [
        ("A9S", "A8S", 120)
    ]


def test_a_short_absence_does_not_split_a_run():
    """A couple of missed polls is a feed hiccup, not a different train: the hop
    across them is still one trip's traversal, interval-censored if an arrival
    was lost."""
    bodies = [
        _body(0, [_row("A1S", stopped=True, seq=1, vehicle_ts=T0)]),
        _body(4, [_row("A2S", stopped=True, seq=2, vehicle_ts=T0 + 240)]),
    ]
    traversals, stats = traversals_from_trace(bodies)
    (hop,) = traversals
    assert (hop.from_stop, hop.to_stop, hop.seconds) == ("A1S", "A2S", 240)
    assert stats.n_exact == 1
