"""Grading the traversal measure against announced planned work
(training/planned_work.py).

Synthetic alert payloads and Traversal lists — no R2. Each case pins one rule
about which windows can be graded, with which measure, and against what.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

from training.planned_work import (
    GRADED,
    HEADWAY_ONLY,
    NO_CONTROL_PERIOD,
    NO_PAIRED_SERVICE,
    SERVICE_CHANGING,
    Window,
    control_reach,
    control_supply,
    coverage_state,
    measure,
    pattern_shift,
    split_by_local_day,
    unknown_types,
    windows_from_alerts,
)
from training.trace import EXACT, INTERVAL, Traversal

T0 = 1_786_552_200
HOUR = 3600


def _alert(
    *,
    alert_type: str = "Planned - Part Suspended",
    routes: list[str] | None = None,
    stops: list[str] | None = None,
    periods: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    entities: list[dict[str, Any]] = [
        {"route_id": r} for r in (routes if routes is not None else ["J"])
    ]
    entities += [
        {"route_id": (routes or ["J"])[0], "stop_id": s}
        for s in (stops if stops is not None else ["J20", "J21"])
    ]
    return {
        "alert": {
            "alert": {
                "transit_realtime.mercury_alert": {"alert_type": alert_type},
                "informed_entity": entities,
                "active_period": [
                    {"start": s, "end": e}
                    for s, e in (periods or [(T0, T0 + 4 * HOUR)])
                ],
            }
        }
    }


def _hop(
    at: int,
    seconds: int,
    *,
    frm: str = "J20N",
    to: str = "J21N",
    route: str = "J",
    trip: str = "072000_J..N40R",
) -> Traversal:
    return Traversal(
        trip_id=trip,
        route_id=route,
        direction="north",
        from_stop=frm,
        to_stop=to,
        at=at,
        seconds=seconds,
        moving_seconds=None,
        n_hops=1,
        censoring=EXACT,
    )


def test_windows_carry_the_stations_and_hours_the_alert_named():
    (window,) = windows_from_alerts([_alert()])
    assert window.alert_type == "Planned - Part Suspended"
    assert window.routes == frozenset({"J"})
    assert window.stops == frozenset({"J20", "J21"})
    assert window.contains(T0 + HOUR)
    assert not window.contains(T0 - 1)
    assert not window.contains(T0 + 5 * HOUR)


def test_a_republished_alert_does_not_become_two_windows():
    """Every tick rewrites the alert version, so the archive holds the same
    window hundreds of times over."""
    assert len(windows_from_alerts([_alert(), _alert(), _alert()])) == 1


def test_an_alert_naming_no_station_is_skipped_entirely():
    """A route-wide window has no control arm, which is the whole reason the
    unplanned feed cannot grade this measure. Widening it to the route would
    manufacture a comparison that does not exist."""
    assert windows_from_alerts([_alert(stops=[])]) == []


def test_a_window_matches_the_directional_platforms_of_the_stations_it_names():
    """Alerts name parent stations ('J20'); the trace reports platforms ('J20N').
    Joined literally this finds zero of 42 affected platforms and looks exactly
    like a measure that detected nothing."""
    (window,) = windows_from_alerts([_alert(stops=["J20"])])
    assert window.touches(("J", "north", "J20N", "J21N"))
    assert window.touches(("J", "north", "J19N", "J20N"))
    assert not window.touches(("J", "north", "J21N", "J22N"))


def test_only_types_a_traversal_can_see_are_graded():
    """Headway and signage changes move neither the pairs a train serves nor how
    long it takes over them, so grading them would count guaranteed misses."""
    for alert_type in SERVICE_CHANGING:
        (window,) = windows_from_alerts([_alert(alert_type=alert_type)])
        assert window.gradeable, alert_type
    for alert_type in HEADWAY_ONLY:
        (window,) = windows_from_alerts([_alert(alert_type=alert_type)])
        assert not window.gradeable, alert_type


def test_an_alert_type_this_module_has_never_seen_is_surfaced():
    """MTA adds types without notice, and the silent failure is that a new
    stop-changing type is excluded from every grade and read as thin supply."""
    windows = windows_from_alerts([_alert(alert_type="Planned - Something New")])
    assert unknown_types(windows) == {"Planned - Something New"}
    assert unknown_types(windows_from_alerts([_alert()])) == set()


def test_a_window_on_a_route_covers_its_express_variant():
    """Alerts name the base route. vehicles.ts folds 7X to 7, but a 7X traversal
    still belongs to a window announced for the 7."""
    (window,) = windows_from_alerts([_alert(routes=["7"], stops=["710"])])
    assert window.covers_route("7")
    assert window.covers_route("7X")
    assert not window.covers_route("6")


def test_only_hops_with_one_endpoint_in_scope_are_boundary_spillover():
    """A hop between two named stations sits inside the closure and stops running
    altogether. Counting it as an affected segment selects for absence, and the
    duration test would report no effect on every real closure."""
    (window,) = windows_from_alerts([_alert(stops=["J20", "J21"])])
    assert window.at_boundary(("J", "north", "J19N", "J20N"))
    assert not window.at_boundary(("J", "north", "J20N", "J21N"))  # both named
    assert window.touches(("J", "north", "J20N", "J21N"))
    assert not window.at_boundary(("J", "north", "J30N", "J31N"))


def _spillover_traversals() -> list[Traversal]:
    """A boundary segment (one endpoint named) that doubles inside the window, and
    a distant one on the same route that also rises by half — so only the excess
    is a real effect."""
    boundary = [_hop(T0 - HOUR + i, 100, frm="J19N", to="J20N") for i in range(8)]
    boundary += [_hop(T0 + HOUR + i, 200, frm="J19N", to="J20N") for i in range(8)]
    distant = [_hop(T0 - HOUR + i, 100, frm="J30N", to="J31N") for i in range(8)]
    distant += [_hop(T0 + HOUR + i, 150, frm="J30N", to="J31N") for i in range(8)]
    return boundary + distant


def test_the_effect_is_what_the_boundary_did_beyond_its_own_route():
    """The boundary hop doubled and the rest of the route rose by half, so the
    intervention is worth 2.0 / 1.5, not 2.0. Without the control arm this measure
    cannot tell planned work from the evening rush."""
    (window,) = windows_from_alerts([_alert(stops=["J20", "J21"])])
    (effect,) = measure(window, _spillover_traversals())
    assert effect.affected_lift == 2.0
    assert effect.control_lift == 1.5
    assert round(effect.effect, 3) == 1.333
    assert effect.n_affected_keys == 1
    assert effect.n_control_keys == 1


def test_a_second_closure_is_kept_out_of_the_control_period():
    """Overlapping planned work is the norm. Compared against as though it were
    normal service, it drags the baseline toward the disrupted state and hides
    the effect."""
    (window,) = windows_from_alerts([_alert(stops=["J20", "J21"])])
    (other,) = windows_from_alerts(
        [_alert(stops=["J20"], periods=[(T0 - 2 * HOUR, T0 - 1)])]
    )
    # Every "outside" observation now falls inside the other closure, so no
    # segment has a clean before arm and the honest answer is silence.
    assert measure(window, _spillover_traversals(), other_windows=[other]) == []


def test_a_thin_window_says_nothing_rather_than_dividing_two_tiny_medians():
    (window,) = windows_from_alerts([_alert(stops=["J20", "J21"])])
    thin = [_hop(T0 - HOUR, 100, frm="J19N", to="J20N"), _hop(T0 + HOUR, 400)]
    assert measure(window, thin) == []


def test_neither_measure_grades_a_headway_only_window():
    """Empty here does not mean "detected nothing" — it means the trace cannot
    see this kind of change at all."""
    (window,) = windows_from_alerts([_alert(alert_type="Reduced Service")])
    assert measure(window, _spillover_traversals()) == []
    assert pattern_shift(window, _spillover_traversals()) == []


def test_interval_and_multi_hop_spans_never_reach_either_measure():
    """Only a hop the realtime feed calls one hop measures that pair."""
    (window,) = windows_from_alerts([_alert(stops=["J20", "J21"])])
    spans = [
        Traversal(
            trip_id="072000_J..N40R",
            route_id="J",
            direction="north",
            from_stop="J19N",
            to_stop="J21N",
            at=T0 + HOUR,
            seconds=500,
            moving_seconds=None,
            n_hops=2,
            censoring=INTERVAL,
        )
    ] * 20
    assert measure(window, spans) == []


# A window at the same clock hours one week earlier: same weekday, so the same
# timetable, which is what the coverage control arm requires.
WEEK = 7 * 24 * HOUR


def test_pattern_shift_reports_the_express_apart_from_the_local():
    """Folded together, an express that stopped making its own long hops is a
    handful of keys against hundreds of local ones running normally, and averages
    to nothing. The local row is a control over the same hours."""
    (window,) = windows_from_alerts(
        [_alert(alert_type="Planned - Express to Local", routes=["7"], stops=["710"])]
    )
    express = "072000_7X..N"
    local = "072000_7..N01R"
    traversals = [
        # A week earlier, same weekday and same hours: the express ran its own
        # long hop.
        _hop(T0 - WEEK + HOUR, 300, frm="701N", to="710N", route="7", trip=express),
        _hop(T0 - WEEK + HOUR + 1, 300, frm="701N", to="710N", route="7", trip=express),
        # Inside the window it makes local stops instead.
        _hop(T0 + HOUR, 100, frm="701N", to="705N", route="7", trip=express),
        # The local serves the same pair in both periods.
        _hop(T0 - WEEK + HOUR, 100, frm="701N", to="705N", route="7", trip=local),
        _hop(T0 + HOUR, 100, frm="701N", to="705N", route="7", trip=local),
    ]
    shifts = {s.service: s for s in pattern_shift(window, traversals)}
    assert set(shifts) == {"7", "7X"}
    assert shifts["7X"].vanished == 1.0  # its long hop stopped appearing
    assert shifts["7"].vanished == 0.0  # the local carried on


def test_the_coverage_control_arm_rejects_the_rest_of_the_same_day():
    """Express service only runs at rush hour, which is when this work is
    scheduled. Measured 2026-08-13, a 7X had 13 traversals outside a 15:00-22:00
    window against 790 inside — 'outside' meant 'when no express runs', and every
    pair would have read as appearing from nowhere."""
    (window,) = windows_from_alerts(
        [
            _alert(
                alert_type="Planned - Express to Local",
                routes=["7"],
                stops=["710"],
                periods=[(T0, T0 + 4 * HOUR)],
            )
        ]
    )
    express = "072000_7X..N"
    same_day_but_other_hours = [
        _hop(T0 + 10 * HOUR, 300, frm="701N", to="710N", route="7", trip=express)
    ]
    inside = [_hop(T0 + HOUR, 100, frm="701N", to="705N", route="7", trip=express)]
    assert pattern_shift(window, same_day_but_other_hours + inside) == []


def test_the_coverage_control_arm_rejects_a_different_service_day():
    """The 7X does not run at all on a weekend, so a Saturday control against a
    weekday window would manufacture the entire signal."""
    (window,) = windows_from_alerts(
        [_alert(alert_type="Planned - Express to Local", routes=["7"], stops=["710"])]
    )
    express = "072000_7X..N"
    weekday = datetime.fromtimestamp(T0, ZoneInfo("America/New_York")).weekday()
    assert weekday < 5  # the fixture instant is a weekday
    saturday_same_hours = [
        _hop(
            T0 + (5 - weekday) * 24 * HOUR,
            300,
            frm="701N",
            to="710N",
            route="7",
            trip=express,
        )
    ]
    inside = [_hop(T0 + HOUR, 100, frm="701N", to="705N", route="7", trip=express)]
    assert pattern_shift(window, saturday_same_hours + inside) == []


def test_an_absent_control_period_is_distinguishable_from_an_absent_effect():
    """`pattern_shift` returns nothing both when the pattern did not move and
    when the archive holds no comparable period to move against, and those are
    opposite conclusions. An archive shorter than two comparable days can only
    produce the second, so reading the empty result as a miss would grade the
    measure on supply it never had."""
    (window,) = windows_from_alerts(
        [
            _alert(
                alert_type="Planned - Express to Local",
                routes=["7"],
                stops=["710"],
                periods=[(T0, T0 + 4 * HOUR)],
            )
        ]
    )
    express = "072000_7X..N"
    same_day_but_other_hours = [
        _hop(T0 + 10 * HOUR, 300, frm="701N", to="710N", route="7", trip=express)
    ]
    inside = [_hop(T0 + HOUR, 100, frm="701N", to="705N", route="7", trip=express)]
    traversals = same_day_but_other_hours + inside
    assert pattern_shift(window, traversals) == []
    assert control_supply(window, traversals) == {}


def test_the_same_clock_band_on_the_next_weekday_supplies_a_control():
    """The counterpart: once the archive reaches a comparable day the measure
    runs, and the supply count says so rather than staying silently zero."""
    (window,) = windows_from_alerts(
        [
            _alert(
                alert_type="Planned - Express to Local",
                routes=["7"],
                stops=["710"],
                periods=[(T0, T0 + 4 * HOUR)],
            )
        ]
    )
    express = "072000_7X..N"
    weekday = datetime.fromtimestamp(T0, ZoneInfo("America/New_York")).weekday()
    assert weekday < 4  # the fixture instant leaves a weekday available tomorrow
    next_weekday = [
        _hop(T0 + 24 * HOUR + HOUR, 300, frm="701N", to="710N", route="7", trip=express)
    ]
    inside = [_hop(T0 + HOUR, 100, frm="701N", to="705N", route="7", trip=express)]
    traversals = next_weekday + inside
    assert control_supply(window, traversals) == {"7X": 1}
    (shift,) = pattern_shift(window, traversals)
    assert shift.service == "7X"
    assert shift.n_outside == 1


def test_control_supply_is_counted_per_service_not_per_route():
    """A route-level total certifies a comparison that never happened. Here the
    7 local has a control day and the 7X the work actually named does not, and
    neither service has both arms, so `pattern_shift` is empty. Summed over the
    route the diagnostic would report control and the empty result would read as
    'compared, found nothing' — the opposite of the truth."""
    (window,) = windows_from_alerts(
        [
            _alert(
                alert_type="Planned - Express to Local",
                routes=["7"],
                stops=["710"],
                periods=[(T0, T0 + 4 * HOUR)],
            )
        ]
    )
    express, local = "072000_7X..N", "072000_7..N40R"
    # The express ran inside the window only; the local ran on the control day
    # only. Neither pairs, and the route-level sum is nonetheless 1.
    traversals = [
        _hop(T0 + HOUR, 100, frm="701N", to="705N", route="7", trip=express),
        _hop(T0 + 24 * HOUR + HOUR, 300, frm="701N", to="702N", route="7", trip=local),
    ]
    assert pattern_shift(window, traversals) == []
    supply = control_supply(window, traversals)
    assert supply == {"7": 1}
    assert "7X" not in supply  # the named service never had a comparison


def test_the_three_coverage_states_are_told_apart():
    """An empty `pattern_shift` has two innocent causes and no interesting one,
    and treating it as a result publishes a gap in the archive as a finding about
    the subway."""
    (window,) = windows_from_alerts(
        [
            _alert(
                alert_type="Planned - Express to Local",
                routes=["7"],
                stops=["710"],
                periods=[(T0, T0 + 4 * HOUR)],
            )
        ]
    )
    express, local = "072000_7X..N", "072000_7..N40R"
    tomorrow = T0 + 24 * HOUR + HOUR
    inside = _hop(T0 + HOUR, 100, frm="701N", to="705N", route="7", trip=express)

    # Nothing to compare against: the archive never reached a comparable day.
    assert coverage_state(window, [], [inside]) == NO_CONTROL_PERIOD

    # The 7 local has a control day, the 7X the alert named does not, and neither
    # service holds both arms. A route-level count would call this a comparison.
    mixed = [
        inside,
        _hop(tomorrow, 300, frm="701N", to="702N", route="7", trip=local),
    ]
    assert pattern_shift(window, mixed) == []
    assert coverage_state(window, [], mixed) == NO_PAIRED_SERVICE

    # Both arms on one service, and the pattern moved: the measure ran.
    moved = [
        inside,
        _hop(tomorrow, 300, frm="701N", to="710N", route="7", trip=express),
    ]
    shifts = pattern_shift(window, moved)
    assert [s.service for s in shifts] == ["7X"]
    assert coverage_state(window, shifts, moved) == GRADED

    # The case that separates "no pairing" from "paired, nothing moved": a
    # service running the SAME pairs on both sides still emits a row, of zeros.
    # Unchanged is a measurement and must never be filed as absent supply.
    unchanged = [
        inside,
        _hop(tomorrow, 100, frm="701N", to="705N", route="7", trip=express),
    ]
    (steady,) = pattern_shift(window, unchanged)
    assert (steady.vanished, steady.appeared) == (0.0, 0.0)
    assert coverage_state(window, [steady], unchanged) == GRADED


def test_a_multi_day_closure_is_graded_one_local_day_at_a_time():
    """The J part suspension announced for 2026-08-14 runs Fri 23:45 straight
    through to Mon 05:00 as ONE alert period: a Friday night, a whole Saturday, a
    whole Sunday and a Monday morning, on three timetables. Graded whole, the
    inside arm is every weekend train while the matched control arm is whatever
    ran in the 23:45-05:00 band — two different populations, not a comparison."""
    friday_2345 = int(
        datetime(2026, 8, 14, 23, 45, tzinfo=ZoneInfo("America/New_York")).timestamp()
    )
    monday_0500 = int(
        datetime(2026, 8, 17, 5, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()
    )
    (window,) = windows_from_alerts(
        [_alert(stops=["J20", "J21"], periods=[(friday_2345, monday_0500)])]
    )
    pieces = split_by_local_day(window)
    assert len(pieces) == 4
    # Each piece sits inside one local day, so each has one service class and one
    # clock band.
    assert [_class_of(p.start) for p in pieces] == [0, 1, 2, 0]
    assert pieces[0].start == friday_2345
    assert pieces[-1].end == monday_0500
    assert all(p.stops == window.stops for p in pieces)


def _class_of(at: int) -> int:
    weekday = datetime.fromtimestamp(at, ZoneInfo("America/New_York")).weekday()
    return 0 if weekday < 5 else weekday - 4


def test_a_single_day_window_is_not_split():
    (window,) = windows_from_alerts([_alert(stops=["J20"])])
    assert split_by_local_day(window) == [window]


def _friday_to_monday() -> Window:
    friday_2345 = int(
        datetime(2026, 8, 14, 23, 45, tzinfo=ZoneInfo("America/New_York")).timestamp()
    )
    monday_0500 = int(
        datetime(2026, 8, 17, 5, 0, tzinfo=ZoneInfo("America/New_York")).timestamp()
    )
    (window,) = windows_from_alerts(
        [_alert(stops=["J20", "J21"], periods=[(friday_2345, monday_0500)])]
    )
    return window


def test_measure_reports_one_effect_per_day_of_a_multi_day_closure():
    """Saturday's spillover is not Sunday's, and pooling the weekend hides both."""
    window = _friday_to_monday()
    clean = window.start - 7 * 24 * HOUR  # the same hours a week earlier
    rows: list[Traversal] = []
    for day, lift in ((1, 2.0), (2, 3.0)):  # Saturday doubles, Sunday triples
        at = window.start + day * 24 * HOUR
        rows += [_hop(at + i, int(100 * lift), frm="J19N", to="J20N") for i in range(8)]
        rows += [_hop(at + i, 100, frm="J30N", to="J31N") for i in range(8)]
    rows += [_hop(clean + i, 100, frm="J19N", to="J20N") for i in range(8)]
    rows += [_hop(clean + i, 100, frm="J30N", to="J31N") for i in range(8)]
    by_day = {
        datetime.fromtimestamp(e.start, ZoneInfo("America/New_York")).weekday(): e
        for e in measure(window, rows)
    }
    assert by_day[5].effect == 2.0  # Saturday
    assert by_day[6].effect == 3.0  # Sunday


def test_one_day_of_a_closure_is_not_the_control_for_another():
    """Without the unsplit window in the blackout, Saturday's disrupted traversals
    become Friday's idea of normal and the effect collapses toward 1.0."""
    window = _friday_to_monday()
    saturday = window.start + 24 * HOUR
    rows = [_hop(saturday + i, 200, frm="J19N", to="J20N") for i in range(8)]
    rows += [_hop(saturday + i, 100, frm="J30N", to="J31N") for i in range(8)]
    # Friday's only candidate "outside" observations are Saturday's, inside the
    # same closure. With them excluded there is no control period, so Friday says
    # nothing rather than grading itself against a disrupted baseline.
    fridays = [
        e
        for e in measure(window, rows)
        if datetime.fromtimestamp(e.start, ZoneInfo("America/New_York")).weekday() == 4
    ]
    assert fridays == []


def test_the_duration_control_period_need_not_match_the_clock():
    """Why `measure` needs no day-matched control while `pattern_shift` does.

    Both arms are measured against the SAME outside period, so a control period
    that is systematically faster or slower than the window inflates
    `affected_lift` and `control_lift` by the same factor and cancels in the
    ratio. Here the comparison period runs 3x quicker than the closure period at
    every segment; the effect is unchanged, because the effect is a ratio of
    ratios. Constraining this arm by clock and weekday would only cost samples.
    """
    window = _friday_to_monday()
    saturday = window.start + 24 * HOUR
    control_period = window.start - 7 * 24 * HOUR
    effects: list[float] = []
    for control_level in (100, 300):  # a fast comparison week, then a slow one
        rows = [
            _hop(saturday + i, control_level * 2, frm="J19N", to="J20N")
            for i in range(8)
        ]
        rows += [
            _hop(saturday + i, control_level, frm="J30N", to="J31N") for i in range(8)
        ]
        rows += [
            _hop(control_period + i, control_level, frm="J19N", to="J20N")
            for i in range(8)
        ]
        rows += [
            _hop(control_period + i, control_level, frm="J30N", to="J31N")
            for i in range(8)
        ]
        (saturday_effect,) = [
            e
            for e in measure(window, rows)
            if datetime.fromtimestamp(e.start, ZoneInfo("America/New_York")).weekday()
            == 5
        ]
        effects.append(saturday_effect.effect)
    assert effects == [2.0, 2.0]


def test_a_closure_ending_at_local_midnight_keeps_its_final_instant():
    """Window intervals are closed at both ends, so the last piece of a closure
    that ends exactly at midnight is one second wide -- and real."""
    ny = ZoneInfo("America/New_York")
    start = int(datetime(2026, 8, 14, 23, 45, tzinfo=ny).timestamp())
    end = int(datetime(2026, 8, 16, 0, 0, tzinfo=ny).timestamp())
    (window,) = windows_from_alerts([_alert(stops=["J20"], periods=[(start, end)])])
    pieces = split_by_local_day(window)
    assert [(p.start, p.end) for p in pieces][-1] == (end, end)
    assert pieces[-1].contains(end)
    assert sum(1 for p in pieces if p.contains(end)) == 1


_NY = ZoneInfo("America/New_York")
WED = date(2026, 8, 12)  # T0's local day
THU, FRI = date(2026, 8, 13), date(2026, 8, 14)
MON, TUE = date(2026, 8, 10), date(2026, 8, 11)


def _window(start: int, end: int, *, routes: tuple[str, ...] = ("J",)) -> Window:
    return Window(
        alert_type="Planned - Part Suspended",
        routes=frozenset(routes),
        stops=frozenset({"J20"}),
        start=start,
        end=end,
    )


def _daily(days: list[int]) -> list[Window]:
    """The same clock band as T0's window, on each offset day given."""
    return [_window(T0 + d * 24 * HOUR, T0 + d * 24 * HOUR + 4 * HOUR) for d in days]


def test_recurring_work_covering_every_certified_day_reports_no_reach():
    """The state all fifteen ungraded windows were in on 2026-08-17. Every
    comparable day the answer key can speak for carries the same work, so
    `certified == covered` and there is nothing to count down to -- the remedy is
    coverage, not a different comparison."""
    window = _window(T0, T0 + 4 * HOUR)
    reach = control_reach(window, [window, *_daily([1, 2])], [WED, THU, FRI])
    assert reach.day is None
    assert reach.lag_days is None
    assert reach.certified == reach.covered == 3


def test_a_free_comparable_day_inside_coverage_is_named_with_its_lag():
    """The counterpart, and the reason the diagnostic is worth reporting: a
    window one free day away from a grade must not read like one that can never
    have a control."""
    window = _window(T0, T0 + 4 * HOUR)
    reach = control_reach(window, [window, *_daily([1])], [WED, THU, FRI])
    assert reach.day == FRI
    assert reach.lag_days == 2
    assert (reach.certified, reach.covered) == (3, 2)


def test_a_day_the_answer_key_never_covered_is_never_named():
    """Absence of an announcement is evidence of a free day only where we hold
    alert snapshots. Monday and Tuesday carry no announced work here purely
    because the record does not reach them, and naming either would promise a
    control drawn from a period we cannot certify was free."""
    window = _window(T0, T0 + 4 * HOUR)
    announced = [window, *_daily([1, 2])]
    assert control_reach(window, announced, [WED, THU, FRI]).day is None
    # The same search, once those days are covered and known free.
    widened = control_reach(window, announced, [MON, TUE, WED, THU, FRI])
    assert widened.day == TUE  # the nearer of the two
    assert widened.lag_days == -1


def test_a_hole_in_coverage_is_not_read_as_a_free_day():
    """Coverage is a set, not a span. Thursday is uncovered here while the days
    either side are covered, so collapsing coverage to first..last would certify
    it -- and it is exactly the day the work runs."""
    window = _window(T0, T0 + 4 * HOUR)
    announced = [window, *_daily([1, 2])]
    reach = control_reach(window, announced, [WED, FRI])
    assert reach.day is None
    assert reach.certified == 2  # Thursday was never a candidate


def test_a_multi_day_closure_blocks_its_own_later_dates():
    """One announced period covers all of its own days. Without the unsplit window
    in the blackout every band would be free, so `covered` counts only what the
    closure itself blocks: its Thursday piece is enclosed on all three candidate
    days, and the Friday and Wednesday pieces are blocked wherever their bands sit
    inside it. The Wednesday whole-day band is NOT blocked -- the closure opens at
    23:45, so that day's earlier hours are admissible and the grade would take
    them."""
    start = int(datetime(2026, 8, 12, 23, 45, tzinfo=_NY).timestamp())
    end = int(datetime(2026, 8, 14, 5, 0, tzinfo=_NY).timestamp())
    window = _window(start, end)
    reach = control_reach(window, [window], [WED, THU, FRI])
    assert (reach.certified, reach.covered) == (9, 5)


def test_reach_is_judged_per_local_day_piece_not_the_unsplit_window():
    """A Friday-night-to-Monday-morning closure is graded as four pieces on three
    timetables, so its Saturday piece needs a SATURDAY control. Read off the
    unsplit window there is one clock band and one service class -- Friday's --
    and the free Saturday below is never even a candidate, so the diagnostic
    reports no reach while the grade could in fact run."""
    closure = _window(
        int(datetime(2026, 8, 14, 23, 45, tzinfo=_NY).timestamp()),
        int(datetime(2026, 8, 17, 5, 0, tzinfo=_NY).timestamp()),
    )
    # Blocks the Monday piece's 00:00-05:00 band on its only weekday candidate,
    # leaving the Saturday as the single free day in the record.
    friday_small_hours = _window(
        int(datetime(2026, 8, 14, 0, 0, tzinfo=_NY).timestamp()),
        int(datetime(2026, 8, 14, 5, 0, tzinfo=_NY).timestamp()),
    )
    free_saturday = date(2026, 8, 8)
    reach = control_reach(
        closure, [closure, friday_small_hours], [free_saturday, date(2026, 8, 14)]
    )
    assert reach.day == free_saturday
    assert reach.lag_days == -7  # against the Saturday piece, not the window start
    assert (reach.certified, reach.covered) == (3, 2)


def test_a_day_is_only_covered_when_every_named_route_is_blacked_out():
    """Work on the 5 does not deprive the 2 of a control arm. `coverage_state`
    calls a window graded when ANY service pairs, so a day blocked on one of two
    named routes still reaches -- reporting it as covered would publish a false
    negative for a grade that can run."""
    window = _window(T0, T0 + 4 * HOUR, routes=("2", "5"))
    on_the_five = _window(T0 + 24 * HOUR, T0 + 24 * HOUR + 4 * HOUR, routes=("5",))
    reach = control_reach(window, [window, on_the_five], [WED, THU])
    assert reach.day == THU
    assert reach.lag_days == 1
    assert (reach.certified, reach.covered) == (2, 1)
    # Once the 2 is blacked out over the same band the day is genuinely gone.
    on_the_two = _window(T0 + 24 * HOUR, T0 + 24 * HOUR + 4 * HOUR, routes=("2",))
    blocked = control_reach(window, [window, on_the_five, on_the_two], [WED, THU])
    assert blocked.day is None
    assert (blocked.certified, blocked.covered) == (2, 2)


def test_work_covering_part_of_a_band_leaves_the_rest_reachable():
    """The diagnostic must agree with the measure it describes. `_is_control`
    tests a traversal's own instant, so a closure over the first hour of a band
    leaves the remaining three admissible -- and `control_supply` proves the grade
    really does take them. Rejecting the whole band on any overlap reports no
    reach for a comparison that runs."""
    window = _window(
        int(datetime(2026, 8, 14, 12, 0, tzinfo=_NY).timestamp()),
        int(datetime(2026, 8, 14, 16, 0, tzinfo=_NY).timestamp()),
    )
    first_hour_thursday = _window(
        int(datetime(2026, 8, 13, 12, 0, tzinfo=_NY).timestamp()),
        int(datetime(2026, 8, 13, 13, 0, tzinfo=_NY).timestamp()),
    )
    reach = control_reach(window, [window, first_hour_thursday], [THU, FRI])
    assert reach.day == THU
    assert reach.lag_days == -1
    assert (reach.certified, reach.covered) == (2, 1)

    # The grade agrees, on the same blackout: 14:00 is outside the closed hour.
    free = _hop(int(datetime(2026, 8, 13, 14, 0, tzinfo=_NY).timestamp()), 100)
    assert control_supply(window, [free], other_windows=[first_hour_thursday]) == {
        "J": 1
    }
    # And a traversal inside that hour is still refused, so the arm is real.
    inside = _hop(int(datetime(2026, 8, 13, 12, 30, tzinfo=_NY).timestamp()), 100)
    assert control_supply(window, [inside], other_windows=[first_hour_thursday]) == {}


def test_blackouts_tiling_a_whole_band_leave_no_instant():
    """The other side of the sweep: two closures meeting inside the band cover it
    between them, and overlapping spans must not read as a gap."""
    window = _window(
        int(datetime(2026, 8, 14, 12, 0, tzinfo=_NY).timestamp()),
        int(datetime(2026, 8, 14, 16, 0, tzinfo=_NY).timestamp()),
    )
    early = _window(
        int(datetime(2026, 8, 13, 11, 0, tzinfo=_NY).timestamp()),
        int(datetime(2026, 8, 13, 14, 0, tzinfo=_NY).timestamp()),
    )
    late = _window(
        int(datetime(2026, 8, 13, 13, 30, tzinfo=_NY).timestamp()),
        int(datetime(2026, 8, 13, 17, 0, tzinfo=_NY).timestamp()),
    )
    reach = control_reach(window, [window, early, late], [THU, FRI])
    assert reach.day is None
    assert (reach.certified, reach.covered) == (2, 2)


def test_the_band_keeps_its_wall_clock_across_a_daylight_saving_shift():
    """2026-03-08 springs forward, so local 05:00 is 18,000 seconds after
    midnight only on days that have 24 hours. Projected arithmetically the band
    lands at 06:00, collides with the 06:00 work below, and the day reads
    covered; on the wall clock it stays at 05:00 and is free."""
    sunday, shift_day = date(2026, 3, 15), date(2026, 3, 8)
    window = _window(
        int(datetime(2026, 3, 15, 5, 0, tzinfo=_NY).timestamp()),
        int(datetime(2026, 3, 15, 5, 30, tzinfo=_NY).timestamp()),
    )
    elsewhere = _window(
        int(datetime(2026, 3, 8, 6, 0, tzinfo=_NY).timestamp()),
        int(datetime(2026, 3, 8, 7, 0, tzinfo=_NY).timestamp()),
    )
    reach = control_reach(window, [window, elsewhere], [shift_day, sunday])
    assert reach.day == shift_day
    assert reach.lag_days == -7
