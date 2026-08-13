"""Static-GTFS segment topology parsing (training/gtfs_static.py).

Synthetic in-memory GTFS zips only — no network access.
"""

from __future__ import annotations

import zipfile
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from training.gtfs_static import (
    Chain,
    Span,
    base_route,
    chains,
    direction_of,
    dominant_successor,
    load_successors,
    origin_seconds,
    service_dates,
    stops_to_json,
    successors,
    terminals,
    through_stops,
    timetable,
)

from .conftest import make_gtfs_bytes
from .conftest import make_gtfs_zip as _zip


def test_base_route_strips_trailing_express_x():
    assert base_route("6X") == "6"
    assert base_route("7X") == "7"
    assert base_route("FX") == "F"
    assert base_route("F") == "F"
    assert base_route("SI") == "SI"  # no trailing X, untouched


def test_direction_of_from_stop_id_suffix():
    assert direction_of("A09N", "anything") == "north"
    assert direction_of("A09S", "anything") == "south"


def test_direction_of_falls_back_to_trip_id_pattern():
    assert direction_of("A00", "SI..N01R") == "north"
    assert direction_of("A00", "SI..S01R") == "south"


def test_direction_of_none_when_neither_present():
    assert direction_of("A00", "SI01R") is None


# A branching fixture: route F south splits at B0S (2 trips continue to C0S,
# 1 to D0S), and an express variant (route FX, folds to F) skips B0S entirely
# (A0S -> C0S direct). A separate route Q north trip is unrelated.
_TRIPS = [
    "F,t1,Weekday,X,0,s1",
    "F,t2,Weekday,X,0,s1",
    "F,t3,Weekday,X,0,s1",
    "FX,t4,Weekday,X,0,s2",
    "Q,t5,Weekday,X,0,s3",
]
_STOP_TIMES = [
    "t1,A0S,00:00:00,00:00:00,1",
    "t1,B0S,00:01:00,00:01:00,2",
    "t1,C0S,00:02:00,00:02:00,3",
    "t2,A0S,00:00:00,00:00:00,1",
    "t2,B0S,00:01:00,00:01:00,2",
    "t2,C0S,00:02:00,00:02:00,3",
    "t3,A0S,00:00:00,00:00:00,1",
    "t3,B0S,00:01:00,00:01:00,2",
    "t3,D0S,00:02:00,00:02:00,3",
    "t4,A0S,00:00:00,00:00:00,1",
    "t4,C0S,00:01:00,00:01:00,2",
    "t5,X0N,00:00:00,00:00:00,1",
    "t5,Y0N,00:01:00,00:01:00,2",
]


def _branching_zip() -> zipfile.ZipFile:
    return _zip(_TRIPS, _STOP_TIMES)


def test_successors_counts_trips_per_consecutive_pair():
    succ = successors(_branching_zip())
    assert succ[("F", "south", "B0S")] == [("C0S", 2), ("D0S", 1)]
    assert succ[("Q", "north", "X0N")] == [("Y0N", 1)]


def test_successors_folds_express_route_to_base_and_keeps_full_branch_list():
    succ = successors(_branching_zip())
    # A0S has two static successors: the local (B0S, dominant) and the express
    # skip-stop (C0S, from the folded FX trip) — neither silently dropped.
    assert succ[("F", "south", "A0S")] == [("B0S", 3), ("C0S", 1)]
    assert ("FX", "south", "A0S") not in succ


def test_successors_ignores_trips_absent_from_trips_txt():
    zf = _zip(
        _TRIPS,
        [
            *_STOP_TIMES,
            "ghost,Z0S,00:00:00,00:00:00,1",
            "ghost,Z1S,00:01:00,00:01:00,2",
        ],
    )
    succ = successors(zf)
    assert all(frm != "Z0S" for (_r, _d, frm) in succ)


def test_successors_skips_stall_pairs():
    stop_times = [
        "t1,A0S,00:00:00,00:00:00,1",
        "t1,A0S,00:01:00,00:01:00,2",  # duplicate consecutive stop -> stall
        "t1,B0S,00:02:00,00:02:00,3",
    ]
    zf = _zip(["F,t1,Weekday,X,0,s1"], stop_times)
    succ = successors(zf)
    assert succ == {("F", "south", "A0S"): [("B0S", 1)]}


def test_dominant_successor_picks_highest_n_trips():
    assert dominant_successor([("C0S", 2), ("D0S", 1)]) == ("C0S", 2)


def test_dominant_successor_tie_breaks_on_smaller_stop_id():
    assert dominant_successor([("Y", 3), ("X", 3)]) == ("X", 3)


def test_chains_walks_a_linear_route_into_one_component():
    succ = successors(_zip(["F,t1,Weekday,X,0,s1"], _STOP_TIMES[:3]))
    result = chains(succ)
    assert result[("F", "south")] == [Chain(stops=("A0S", "B0S", "C0S"))]


def test_chains_drops_minor_branch_but_keeps_dominant_path():
    succ = successors(_branching_zip())
    (component,) = chains(succ)[("F", "south")]
    # D0S is B0S's minor (non-dominant) successor, so it never joins the walk.
    assert component.stops == ("A0S", "B0S", "C0S")
    assert "D0S" not in component.stops


def test_chains_reports_disjoint_pieces_as_separate_components():
    stop_times = [
        "t1,A0S,00:00:00,00:00:00,1",
        "t1,B0S,00:01:00,00:01:00,2",
        "t2,X0S,00:00:00,00:00:00,1",
        "t2,Y0S,00:01:00,00:01:00,2",
    ]
    zf = _zip(["Q,t1,Weekday,X,0,s1", "Q,t2,Weekday,X,0,s1"], stop_times)
    result = chains(successors(zf))[("Q", "south")]
    assert len(result) == 2
    assert {c.stops for c in result} == {("A0S", "B0S"), ("X0S", "Y0S")}


def test_terminals_names_both_ends_of_a_linear_route():
    succ = successors(_zip(["F,t1,Weekday,X,0,s1"], _STOP_TIMES[:3]))
    assert terminals(succ) == {("F", "south", "A0S"), ("F", "south", "C0S")}


def test_through_stops_excludes_both_ends():
    succ = successors(_zip(["F,t1,Weekday,X,0,s1"], _STOP_TIMES[:3]))
    assert through_stops(succ) == {("F", "south", "B0S")}


def test_terminals_and_through_stops_partition_the_skeleton():
    succ = successors(_zip(["F,t1,Weekday,X,0,s1"], _STOP_TIMES[:3]))
    term, through = terminals(succ), through_stops(succ)
    assert not term & through
    assert term | through == {("F", "south", stop) for stop in ("A0S", "B0S", "C0S")}


def test_terminals_names_every_origin_of_a_two_entry_component():
    """Two branches merging into one trunk: both branch heads are origins, and
    only the source/sink test finds both — a Chain's stops concatenates one walk
    per entry, so its first element names just one of them."""
    stop_times = [
        "t1,A0S,00:00:00,00:00:00,1",
        "t1,C0S,00:01:00,00:01:00,2",
        "t2,B0S,00:00:00,00:00:00,1",
        "t2,C0S,00:01:00,00:01:00,2",
    ]
    succ = successors(_zip(["Q,t1,Weekday,X,0,s1", "Q,t2,Weekday,X,0,s1"], stop_times))
    assert terminals(succ) == {
        ("Q", "south", "A0S"),
        ("Q", "south", "B0S"),
        ("Q", "south", "C0S"),
    }
    assert through_stops(succ) == set()


def test_stops_to_json_nests_route_direction_sorted():
    stops = frozenset(
        {("F", "south", "C0S"), ("F", "south", "A0S"), ("F", "north", "A0N")}
    )
    assert stops_to_json(stops) == {"F": {"north": ["A0N"], "south": ["A0S", "C0S"]}}


def test_load_successors_composes_fetch_and_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    zip_bytes = make_gtfs_bytes(["F,t1,Weekday,X,0,s1"], _STOP_TIMES[:3])

    def fake_fetch(url: str = "") -> bytes:
        return zip_bytes

    import training.gtfs_static as gtfs_static

    monkeypatch.setattr(gtfs_static, "fetch_gtfs_zip", fake_fetch)
    result = load_successors()
    assert result == {
        ("F", "south", "A0S"): [("B0S", 1)],
        ("F", "south", "B0S"): [("C0S", 1)],
    }


# --- the timetable: what the schedule allowed a specific trip -------------------

WEEKDAY = date(2026, 8, 12)  # a Wednesday
SATURDAY = date(2026, 8, 15)


def _at(day: date, hour: int, minute: int = 0) -> int:
    """Epoch seconds at a local wall-clock time on a service day."""
    return int(
        datetime.combine(
            day, time(hour, minute), tzinfo=ZoneInfo("America/New_York")
        ).timestamp()
    )


def test_hops_measure_arrival_to_arrival_including_scheduled_dwell():
    """The observed quantity is arrival to arrival, so the reference has to be
    too: s1N arrives at 00:00:00 and s2N at 00:02:30, which is 150s even though
    only 90s of it is running and the train stood at s1N for the other minute."""
    zf = _zip(
        ["A,000000_A..N01R,Weekday,X,0,s1"],
        [
            "000000_A..N01R,s1N,00:00:00,00:01:00,1",
            "000000_A..N01R,s2N,00:02:30,00:02:40,2",
        ],
    )
    assert timetable(zf).day(WEEKDAY).hops == {("A", "north", "s1N", "s2N"): 150}


def test_hops_take_the_median_across_trips_serving_them():
    """Three trips run the same hop in 60s, 90s and 300s. The median (90) is what
    ships — a mean would let one padded late-night trip stretch the hop."""
    zf = _zip(
        [
            "A,000000_A..N01R,Weekday,X,0,s1",
            "A,006000_A..N01R,Weekday,X,0,s1",
            "A,012000_A..N01R,Weekday,X,0,s1",
        ],
        [
            "000000_A..N01R,s1N,00:00:00,00:00:00,1",
            "000000_A..N01R,s2N,00:01:00,00:01:00,2",
            "006000_A..N01R,s1N,01:00:00,01:00:00,1",
            "006000_A..N01R,s2N,01:01:30,01:01:30,2",
            "012000_A..N01R,s1N,02:00:00,02:00:00,1",
            "012000_A..N01R,s2N,02:05:00,02:05:00,2",
        ],
    )
    assert timetable(zf).day(WEEKDAY).hops[("A", "north", "s1N", "s2N")] == 90


def test_hops_handle_times_past_midnight():
    """NYCT expresses a post-midnight trip as hours >= 24 (25:14:00 is real), so
    the parse cannot use a wall-clock time type."""
    zf = _zip(
        ["A,149900_A..N01R,Weekday,X,0,s1"],
        [
            "149900_A..N01R,s1N,24:59:00,24:59:00,1",
            "149900_A..N01R,s2N,25:01:00,25:01:00,2",
        ],
    )
    assert timetable(zf).day(WEEKDAY).hops == {("A", "north", "s1N", "s2N"): 120}


def test_hops_drop_nonincreasing_and_unparseable_times():
    """An arrival at or before the preceding one is not a usable run time, and
    neither is a blank — both drop out rather than becoming a zero or a negative
    hop that would read as infinite speed downstream."""
    zf = _zip(
        ["A,000000_A..N01R,Weekday,X,0,s1", "A,006000_A..N01R,Weekday,X,0,s1"],
        [
            "000000_A..N01R,s1N,00:05:00,00:05:00,1",
            "000000_A..N01R,s2N,00:04:00,00:04:00,2",
            "006000_A..N01R,s1N,,,1",
            "006000_A..N01R,s2N,00:02:00,00:02:00,2",
        ],
    )
    assert timetable(zf).day(WEEKDAY).hops == {}


def test_hops_fold_express_route_to_base():
    zf = _zip(
        ["6X,000000_6X..N01R,Weekday,X,0,s1"],
        [
            "000000_6X..N01R,s1N,00:00:00,00:00:00,1",
            "000000_6X..N01R,s2N,00:02:00,00:02:00,2",
        ],
    )
    assert timetable(zf).day(WEEKDAY).hops == {("6", "north", "s1N", "s2N"): 120}


SATURDAY_TRIP = "143500_A..N02R"  # origin 23:55, a run that crosses midnight


def _two_calendars() -> zipfile.ZipFile:
    """The same hop, three minutes on a weekday and five on a Saturday. The
    Saturday run leaves at 23:55, so it is still going after midnight."""
    return _zip(
        ["A,000000_A..N01R,Weekday,X,0,s1", f"A,{SATURDAY_TRIP},Saturday,X,0,s1"],
        [
            "000000_A..N01R,s1N,00:00:00,00:00:00,1",
            "000000_A..N01R,s2N,00:03:00,00:03:00,2",
            f"{SATURDAY_TRIP},s1N,23:55:00,23:55:00,1",
            f"{SATURDAY_TRIP},s2N,24:00:00,24:00:00,2",
        ],
    )


def test_a_day_only_sees_the_services_its_calendar_runs():
    """Pooling the weekend timetable into the weekday one gets a quarter of
    observed hops; the calendar is what keeps them apart."""
    tt = timetable(_two_calendars())
    key = ("A", "north", "s1N", "s2N")
    assert tt.day(WEEKDAY).hops[key] == 180
    assert tt.day(SATURDAY).hops[key] == 300


def test_a_calendar_dates_exception_moves_a_service_onto_another_day():
    """A holiday runs Saturday service on a weekday, and calendar_dates.txt is
    the only place that says so."""
    zf = _zip(
        ["A,000000_A..N01R,Weekday,X,0,s1", "A,000000_A..N02R,Saturday,X,0,s1"],
        [
            "000000_A..N01R,s1N,00:00:00,00:00:00,1",
            "000000_A..N01R,s2N,00:03:00,00:03:00,2",
            "000000_A..N02R,s1N,00:00:00,00:00:00,1",
            "000000_A..N02R,s2N,00:05:00,00:05:00,2",
        ],
        calendar_dates_rows=["Weekday,20260812,2", "Saturday,20260812,1"],
    )
    assert timetable(zf).day(WEEKDAY).hops[("A", "north", "s1N", "s2N")] == 300


def _bypass_feed() -> zipfile.ZipFile:
    """A local calling at three stops and an express skipping the middle one."""
    return _zip(
        ["A,000000_A..N01R,Weekday,X,0,s1", "A,006000_A..N88R,Weekday,X,0,s1"],
        [
            "000000_A..N01R,s1N,00:00:00,00:00:00,1",
            "000000_A..N01R,s2N,00:02:00,00:02:00,2",
            "000000_A..N01R,s3N,00:04:00,00:04:00,3",
            "006000_A..N88R,s1N,01:00:00,01:00:00,1",
            "006000_A..N88R,s3N,01:03:00,01:03:00,3",
        ],
    )


def test_span_measures_a_bypass_along_the_trips_own_stops():
    """The local is scheduled 240s for s1N->s3N because it stops at s2N; the
    express is scheduled 180s for the same pair because it does not. Reading
    either off the other is the skip-stop inversion — a train that bypassed a
    station gets credited with the time it saved and reads as running fast."""
    day = timetable(_bypass_feed()).day(WEEKDAY)
    local = day.span("000000_A..N01R", "s1N", "s3N")
    assert local == Span(seconds=240, n_hops=2)
    express = day.span("006000_A..N88R", "s1N", "s3N")
    assert express == Span(seconds=180, n_hops=1)


def test_span_resolves_a_truncated_realtime_path_code():
    """The realtime feed publishes some ids without the path suffix ('W..N' for
    'W..N30R'), so an exact match on the code is not enough."""
    day = timetable(_bypass_feed()).day(WEEKDAY)
    assert day.span("006000_A..N88", "s1N", "s3N") == Span(seconds=180, n_hops=1)


def test_span_is_none_when_candidate_patterns_disagree():
    """A truncated code that matches both the local and the express is not a
    reading of the timetable — the caller falls back to the day's median rather
    than picking whichever pattern sorts first."""
    day = timetable(_bypass_feed()).day(WEEKDAY)
    assert day.span("006000_A..N", "s1N", "s3N") is None


def test_origin_seconds_reads_hundredths_of_a_minute():
    """'060250' is 602.50 minutes past midnight, 10:02:30 — not 06:02:50. Read as
    HH:MM:SS it matches 273 of the feed's 20,621 trips against 20,311."""
    assert origin_seconds("060250_A..N01R") == 10 * 3600 + 2 * 60 + 30
    assert origin_seconds("t1") is None


def test_service_dates_put_a_post_midnight_run_on_the_day_it_started():
    """A run scheduled to leave at 25:30 and seen at 01:40 belongs to yesterday's
    timetable, which is where its origin lives."""
    trip = "153000_A..N01R"  # 25:30
    assert service_dates(_at(date(2026, 8, 13), 1, 40), trip)[0] == date(2026, 8, 12)


def test_service_dates_keep_a_train_put_in_service_early_on_today():
    """Seen at 07:55 against an 08:00 origin, a train is five minutes early, not
    a day late."""
    trip = "048000_A..N01R"  # 08:00
    assert service_dates(_at(WEEKDAY, 7, 55), trip)[0] == WEEKDAY


def test_day_for_passes_over_a_day_whose_calendar_never_ran_the_trip():
    """Ranking by scheduled origin picks a day and the calendar gets a veto.

    The Saturday run leaves at 23:55, so ten past midnight on Sunday its origin
    sits nearest to Saturday's midnight and Saturday describes it. Seen five past
    midnight on SATURDAY, the same id looks like a Friday 23:55 departure — but
    Friday runs weekday service, which has no such pattern, so the veto keeps it
    on Saturday instead of judging a weekend train by the weekday timetable.
    """
    tt = timetable(_two_calendars())
    key = ("A", "north", "s1N", "s2N")
    assert tt.day_for(_at(date(2026, 8, 16), 0, 10), SATURDAY_TRIP).hops[key] == 300
    assert service_dates(_at(SATURDAY, 0, 5), SATURDAY_TRIP)[0] == date(2026, 8, 14)
    assert tt.day_for(_at(SATURDAY, 0, 5), SATURDAY_TRIP).hops[key] == 300
