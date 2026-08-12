"""Static-GTFS segment topology parsing (training/gtfs_static.py).

Synthetic in-memory GTFS zips only — no network access.
"""

from __future__ import annotations

import io
import zipfile

import pytest

from training.gtfs_static import (
    Chain,
    base_route,
    chains,
    direction_of,
    dominant_successor,
    hop_seconds,
    load_successors,
    successors,
)


def _csv(header: str, rows: list[str]) -> str:
    return "\n".join([header, *rows]) + "\n"


def _zip(trips_rows: list[str], stop_times_rows: list[str]) -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "trips.txt",
            _csv(
                "route_id,trip_id,service_id,trip_headsign,direction_id,shape_id",
                trips_rows,
            ),
        )
        zf.writestr(
            "stop_times.txt",
            _csv(
                "trip_id,stop_id,arrival_time,departure_time,stop_sequence",
                stop_times_rows,
            ),
        )
    buf.seek(0)
    return zipfile.ZipFile(buf)


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


def test_load_successors_composes_fetch_and_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(
            "trips.txt",
            _csv(
                "route_id,trip_id,service_id,trip_headsign,direction_id,shape_id",
                ["F,t1,Weekday,X,0,s1"],
            ),
        )

        zf.writestr(
            "stop_times.txt",
            _csv(
                "trip_id,stop_id,arrival_time,departure_time,stop_sequence",
                _STOP_TIMES[:3],
            ),
        )
    zip_bytes = buf.getvalue()

    def fake_fetch(url: str = "") -> bytes:
        return zip_bytes

    import training.gtfs_static as gtfs_static

    monkeypatch.setattr(gtfs_static, "fetch_gtfs_zip", fake_fetch)
    result = load_successors()
    assert result == {
        ("F", "south", "A0S"): [("B0S", 1)],
        ("F", "south", "B0S"): [("C0S", 1)],
    }


# --- hop_seconds: the scheduled run time between adjacent stops -----------------


def test_hop_seconds_measures_departure_to_arrival_not_including_dwell():
    """The quantity is travel time, so a scheduled dwell at from_stop must not be
    counted: s1N departs at 00:01:00 and s2N arrives at 00:02:30, so the hop is
    90s even though the train was standing at s1N for a minute beforehand."""
    zf = _zip(
        ["A,t1,Weekday,X,0,s1"],
        [
            "t1,s1N,00:00:00,00:01:00,1",
            "t1,s2N,00:02:30,00:02:40,2",
        ],
    )
    assert hop_seconds(zf) == {("A", "north", "s1N", "s2N"): 90}


def test_hop_seconds_takes_the_median_across_trips_serving_the_hop():
    """Three trips run the same hop in 60s, 90s and 300s. The median (90) is what
    ships — a mean would let one padded late-night trip stretch the hop."""
    zf = _zip(
        ["A,t1,Weekday,X,0,s1", "A,t2,Weekday,X,0,s1", "A,t3,Weekday,X,0,s1"],
        [
            "t1,s1N,00:00:00,00:00:00,1",
            "t1,s2N,00:01:00,00:01:00,2",
            "t2,s1N,01:00:00,01:00:00,1",
            "t2,s2N,01:01:30,01:01:30,2",
            "t3,s1N,02:00:00,02:00:00,1",
            "t3,s2N,02:05:00,02:05:00,2",
        ],
    )
    assert hop_seconds(zf)[("A", "north", "s1N", "s2N")] == 90


def test_hop_seconds_handles_times_past_midnight():
    """NYCT expresses a post-midnight trip as hours >= 24 (25:14:00 is real), so
    the parse cannot use a wall-clock time type."""
    zf = _zip(
        ["A,t1,Weekday,X,0,s1"],
        [
            "t1,s1N,24:59:00,24:59:00,1",
            "t1,s2N,25:01:00,25:01:00,2",
        ],
    )
    assert hop_seconds(zf) == {("A", "north", "s1N", "s2N"): 120}


def test_hop_seconds_drops_nonincreasing_and_unparseable_times():
    """An arrival at or before the preceding departure is not a usable run time,
    and neither is a blank — both drop out rather than becoming a zero or a
    negative hop that would read as infinite speed downstream."""
    zf = _zip(
        ["A,t1,Weekday,X,0,s1", "A,t2,Weekday,X,0,s1"],
        [
            "t1,s1N,00:00:00,00:05:00,1",
            "t1,s2N,00:04:00,00:04:00,2",
            "t2,s1N,,,1",
            "t2,s2N,00:02:00,00:02:00,2",
        ],
    )
    assert hop_seconds(zf) == {}


def test_hop_seconds_folds_express_route_to_base():
    zf = _zip(
        ["6X,t1,Weekday,X,0,s1"],
        [
            "t1,s1N,00:00:00,00:00:00,1",
            "t1,s2N,00:02:00,00:02:00,2",
        ],
    )
    assert hop_seconds(zf) == {("6", "north", "s1N", "s2N"): 120}
