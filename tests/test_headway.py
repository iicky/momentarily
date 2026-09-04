"""Reference-stop selection, headway reconstruction, and typical-actual baselines
(training/headway.py).

Synthetic data only — no R2, no network. Each case pins one rule: the wait
formula, the trunk-over-branch reference-stop choice, the midnight/night
bucketing, a feed stall vs a service gap, and the independent-night gate.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from training.headway import (
    DUP_ARRIVAL_SECONDS,
    FEED_GAP_SECONDS,
    HeadwayEvent,
    awt,
    headway_cv,
    headway_events,
    reference_arrivals,
    scheduled_swt,
    select_reference_stops,
    tick_aligned_waits,
    typical_actual_baseline,
)
from training.trace import Passing, arrivals_from_trace, passings_from_trace

from .conftest import make_gtfs_zip

_ET = ZoneInfo("America/New_York")


def _et_epoch(y: int, mo: int, d: int, h: int, mi: int = 0) -> int:
    return int(datetime(y, mo, d, h, mi, tzinfo=_ET).timestamp())


def _passing(
    stop: str, at: int, *, trip: str, route: str = "A", direction: str = "north"
) -> Passing:
    return Passing(
        trip_id=trip,
        route_id=route,
        direction=direction,
        stop_id=stop,
        stop_seq=1,
        at=at,
    )


# --- wait formulas ---


def test_awt_of_even_service_is_half_the_headway():
    # every gap 600s -> a rider waits on average 300s
    assert awt([600, 600, 600]) == 300.0


def test_awt_weights_the_long_gap():
    # one 1200s gap among 600s gaps pulls the average wait above 300
    even = awt([600, 600, 600, 600])
    bunched = awt([200, 200, 200, 1200])
    assert even is not None
    assert bunched is not None
    assert bunched > even


def test_awt_none_without_positive_headway():
    assert awt([]) is None
    assert awt([0]) is None


def test_headway_cv_zero_for_even_service():
    assert headway_cv([600, 600, 600]) == 0.0


def test_headway_cv_none_below_two():
    assert headway_cv([600]) is None


# --- reference-stop selection: trunk over branch ---


def _branching_zip():
    # route A: a local pattern A1..A5 and an express A1,A3,A5. A3 is the trunk
    # stop every train passes; A2/A4 are local-only branch stops.
    local = ["A01N", "A02N", "A03N", "A04N", "A05N"]
    express = ["A01N", "A03N", "A05N"]
    trips: list[str] = []
    stop_times: list[str] = []
    tid = 0

    def add(stops: list[str], base: int) -> None:
        nonlocal tid
        tid += 1
        trip = f"t{tid}..N"
        trips.append(f"A,{trip},Weekday,Uptown,0,sh")
        for i, s in enumerate(stops):
            sec = base + i * 120
            hhmmss = f"{sec // 3600:02d}:{(sec % 3600) // 60:02d}:{sec % 60:02d}"
            stop_times.append(f"{trip},{s},{hhmmss},{hhmmss},{i + 1}")

    # 4 local trips, 2 express trips, staggered so headways are regular
    for k in range(4):
        add(local, 8 * 3600 + k * 600)
    for k in range(2):
        add(express, 8 * 3600 + 300 + k * 1200)
    return make_gtfs_zip(trips, stop_times)


def test_reference_stop_is_the_trunk_not_a_branch_stop():
    stops = select_reference_stops(_branching_zip())
    rs = stops[("A", "north")]
    assert rs.stop_id == "A03N"  # the stop every pattern shares
    assert rs.coverage == 1.0
    assert rs.n_patterns == 2


def test_reference_stop_excludes_terminals():
    stops = select_reference_stops(_branching_zip())
    rs = stops[("A", "north")]
    # A01N (origin) and A05N (sink) are terminals, never the reference
    assert rs.stop_id not in ("A01N", "A05N")


# --- reconstruction: dedupe, feed stall, direction filter ---


def _covered(start: int, end: int, step: int = 60) -> list[int]:
    return list(range(start, end + 1, step))


def test_duplicate_trip_arrival_is_not_a_zero_headway():
    t0 = _et_epoch(2026, 8, 22, 12)
    arrs = {
        ("A", "north"): [
            _passing("A03N", t0, trip="x"),
            _passing("A03N", t0 + DUP_ARRIVAL_SECONDS - 10, trip="x"),  # re-report
            _passing("A03N", t0 + 600, trip="y"),
        ]
    }
    events = headway_events(arrs, _covered(t0 - 60, t0 + 700))
    hws = [e.headway_sec for e in events[("A", "north")]]
    assert hws == [600]  # the re-report was collapsed, one clean gap remains


def test_feed_stall_flags_the_headway_across_it():
    t0 = _et_epoch(2026, 8, 22, 3)
    # coverage present, then a long gap (stall), then present again
    covered = _covered(t0 - 60, t0 + 120) + _covered(
        t0 + 120 + FEED_GAP_SECONDS + 120, t0 + 3000
    )
    arrs = {
        ("A", "north"): [
            _passing("A03N", t0, trip="x"),
            _passing("A03N", t0 + 2400, trip="y"),  # spans the stall
            _passing("A03N", t0 + 3000, trip="z"),
        ]
    }
    events = headway_events(arrs, covered)
    evs = events[("A", "north")]
    assert evs[0].feed_gap is True  # gap inside (t0, t0+2400)
    assert evs[1].feed_gap is False


def test_feed_gap_headways_excluded_from_tick_waits():
    t0 = _et_epoch(2026, 8, 22, 12)
    good: list[HeadwayEvent] = [
        HeadwayEvent(
            "A", "north", "A03N", f"t{i}", f"t{i - 1}", t0 + i * 300, 300, False
        )
        for i in range(1, 20)
    ]
    stalled = HeadwayEvent("A", "north", "A03N", "z", "y", t0 + 6000, 5400, True)
    waits = tick_aligned_waits([*good, stalled])
    # the stalled 5400s "headway" never enters an AWT reading
    assert waits
    assert max(w.awt_sec for w in waits) < 1000


# --- night/ET bucketing and the independent-night gate ---


def test_typical_actual_gates_on_independent_nights():
    # one busy night of ticks in a single cell must NOT publish under the gate
    t_noon = _et_epoch(2026, 8, 22, 12)
    one_night = [
        HeadwayEvent(
            "A", "north", "A03N", f"t{i}", f"t{i - 1}", t_noon + i * 60, 360, False
        )
        for i in range(1, 60)
    ]
    waits = {("A", "north"): tick_aligned_waits(one_night)}
    cells = typical_actual_baseline(waits, min_nights=8)
    assert cells == {}  # 1 distinct ET date < 8


def test_typical_actual_publishes_with_enough_nights():
    events: list[HeadwayEvent] = []
    # eight distinct ET dates, same weekday-noon cell, enough ticks each
    for day in range(17, 29):  # 2026-08-17 .. 08-28 (>= 8 weekday dates)
        base = _et_epoch(2026, 8, day, 12)
        events += [
            HeadwayEvent(
                "A",
                "north",
                "A03N",
                f"{day}_{i}",
                f"{day}_{i - 1}",
                base + i * 60,
                360,
                False,
            )
            for i in range(1, 30)
        ]
    waits = {("A", "north"): tick_aligned_waits(events)}
    cells = typical_actual_baseline(waits, min_samples=20, min_nights=8)
    assert cells  # gate cleared
    cell = next(iter(cells.values()))
    assert cell.n_nights >= 8
    assert cell.p50 > 0


# --- scheduled SWT is the same wait formula on the timetable ---


def test_scheduled_swt_uses_the_wait_formula_on_the_timetable():
    stops = select_reference_stops(_branching_zip())
    swt = scheduled_swt(_branching_zip(), stops)
    # A03N is served by both local (~10-min) and express trips in the 8am hour;
    # the combined scheduled gaps are bunched, so SWT sits below a clean H/2 but
    # is a real positive wait computed the same way as observed AWT.
    key = ("A", "north", "wd08")
    assert key in swt
    assert 150 <= swt[key] <= 360


# --- direction filter ---


def test_reference_arrivals_keep_only_the_reference_stop():
    stops = select_reference_stops(_branching_zip())
    t0 = _et_epoch(2026, 8, 22, 12)
    arrs = [
        _passing("A03N", t0, trip="a"),  # reference stop
        _passing("A02N", t0 + 60, trip="a"),  # a branch stop, dropped
        _passing("A03N", t0 + 600, trip="b"),
    ]
    grouped = reference_arrivals(arrs, stops)
    kept = grouped[("A", "north")]
    assert all(a.stop_id == "A03N" for a in kept)
    assert len(kept) == 2


# --- the double-headway collapse (the bug this reconstruction fixes) -------------


def _hrow(
    stop: str, *, trip: str, stopped: bool, seq: int | None = None
) -> dict[str, Any]:
    return {
        "trip_id": trip,
        "route_id": "A",
        "direction": "north",
        "stop_id": stop,
        "stop_seq": seq if stopped else None,
        "stopped": stopped,
        "vehicle_ts": None,
    }


def _hbody(t0: int, minute: int, rows: list[dict[str, Any]]) -> dict[str, Any]:
    at = t0 + minute * 60
    return {
        "observed_at": at + 3,
        "scheduled_at": at,
        "fresh_feeds": ["ace"],
        "rows": rows,
    }


def test_a_sub_poll_dwell_train_collapses_a_double_headway():
    """End to end at reference stop A03N. Trains x and y are caught STOPPED_AT;
    the middle train m dwells shorter than the poll and is only ever seen
    approaching, never standing. The STOPPED_AT reconstruction misses m and so
    reports ONE double-length gap x->y; the transition reconstruction catches m
    and splits it into TWO real gaps — which is exactly the ~7% of headways the
    old rule merged."""
    t0 = _et_epoch(2026, 8, 22, 12)
    bodies = [
        _hbody(t0, 0, [_hrow("A03N", trip="x", stopped=True, seq=3)]),
        _hbody(t0, 1, [_hrow("A04N", trip="x", stopped=False)]),
        _hbody(t0, 2, [_hrow("A03N", trip="m", stopped=False)]),  # sub-poll dwell
        _hbody(t0, 3, [_hrow("A04N", trip="m", stopped=False)]),
        _hbody(t0, 7, [_hrow("A03N", trip="y", stopped=True, seq=3)]),
        _hbody(t0, 8, [_hrow("A04N", trip="y", stopped=False)]),
    ]
    covered = _covered(t0 - 60, t0 + 700)

    arrivals = [a for a in arrivals_from_trace(bodies) if a.stop_id == "A03N"]
    passings = [p for p in passings_from_trace(bodies) if p.stop_id == "A03N"]
    assert {a.trip_id for a in arrivals} == {"x", "y"}  # m never caught standing
    assert {p.trip_id for p in passings} == {"x", "m", "y"}

    # Same headway_events on each reconstruction: the merged double vs the split.
    arr_series: dict[tuple[str, str], list[Passing]] = {("A", "north"): arrivals}  # type: ignore[dict-item]
    arr_hw = headway_events(arr_series, covered)[("A", "north")]
    pass_hw = headway_events({("A", "north"): passings}, covered)[("A", "north")]
    assert len(arr_hw) == 1  # one double-length gap x->y
    assert len(pass_hw) == 2  # split into x->m and m->y
    assert arr_hw[0].headway_sec == pass_hw[0].headway_sec + pass_hw[1].headway_sec
