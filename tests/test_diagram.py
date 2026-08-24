"""Diagram geometry: the drawable graph the map overlay paints status onto.

The property that matters is the join between the two sides: every edge the
diagram draws has to carry the exact `segment_flow` key the Worker publishes
for that (route, direction, from_stop) cell, or the map silently paints nothing
where a reading exists.
"""

from __future__ import annotations

import csv
import io
import itertools
import math
import zipfile
from datetime import date

import pytest

from training.diagram import (
    OFFSET_SPACING,
    PAD,
    VIEW_WIDTH,
    axis_deviation,
    build,
    components,
    edge_directions,
    edge_seconds,
    octilinear,
    project,
    representative_days,
    route_sort_key,
    to_json,
)
from training.gtfs_static import FeedVersion


def _approx(expected: float, tol: float | None = None) -> object:
    """Typed wrapper around ``pytest.approx``, mirroring tests/test_dwell.py:
    ``approx`` leaks ``Unknown`` through ``ApproxBase`` under strict pyright."""
    if tol is None:
        return pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]
    return pytest.approx(expected, abs=tol)  # pyright: ignore[reportUnknownMemberType]


# A toy trunk: two routes sharing the 100 <-> 101 hop, then diverging, plus a
# one-way branch that only the 1 runs northbound.
STOPS = [
    ("100", "South End", 40.700, -74.010),
    ("101", "Trunk Middle", 40.710, -74.000),
    ("102", "North End", 40.720, -73.990),
    ("103", "Branch End", 40.715, -73.970),
]

TRIPS = [
    # trip_id, route_id, service_id, [(stop_id, arrival_time), ...].
    # arrival_time is HH:MM:SS, "" when the feed leaves that stop's arrival
    # blank — the untimed slice of hops a real feed carries (Timetable's own
    # docstring measures this at 1.2% of hops feed-wide).
    (
        "t1",
        "1",
        "WKD",
        [("100N", "08:00:00"), ("101N", "08:03:00"), ("102N", "08:07:00")],
    ),
    # Two more weekday runs of route 1's 100N->101N leg, one a padded
    # late-night outlier: [180, 190, 1000] pools to a median of 190s, not the
    # ~457s a mean would give the hop. This is what exercises the
    # trip-weighted-median rule.
    ("t1b", "1", "WKD", [("100N", "09:00:00"), ("101N", "09:03:10")]),
    ("t1c", "1", "WKD", [("100N", "22:00:00"), ("101N", "22:16:40")]),
    (
        "t2",
        "1",
        "WKD",
        [("102S", "08:10:00"), ("101S", "08:14:00"), ("100S", "08:17:00")],
    ),
    (
        "t3",
        "2",
        "WKD",
        [("100N", "08:05:00"), ("101N", "08:08:00"), ("103N", "08:15:00")],
    ),
    (
        "t4",
        "2",
        "WKD",
        [("103S", "08:20:00"), ("101S", "08:27:00"), ("100S", "08:31:00")],
    ),
    # Northbound-only branch hop: 102 -> 103 is never scheduled southbound,
    # and its arrival at 103N is blank, so the hop carries no scheduled time
    # either — topology without timing.
    ("t5", "1", "WKD", [("102N", "08:20:00"), ("103N", "")]),
    # Saturday runs the SAME 100->101 northbound hop as t1/t1b/t1c, on a
    # slower schedule. This is the case that proves service classes are kept
    # apart rather than pooled: weekday resolves to 190s, Saturday to 300s.
    # Sunday runs nothing in this fixture at all, which is the class-level
    # absence case (as opposed to the direction-level one t5 covers).
    ("t1sat", "1", "SAT", [("100N", "10:00:00"), ("101N", "10:05:00")]),
]
# A second system sharing no track and no complex with the trunk, sitting far
# southwest of it — the Staten Island Railway's relationship to the subway.
DETACHED_STOPS = [
    ("900", "Island South", 40.510, -74.250),
    ("901", "Island North", 40.560, -74.180),
]

DETACHED_TRIPS = [
    ("t9", "SI", "WKD", [("900N", ""), ("901N", "")]),
    ("t10", "SI", "WKD", [("901S", ""), ("900S", "")]),
]


def _feed(detached: bool = False) -> zipfile.ZipFile:
    """A minimal but structurally real GTFS zip: parent stations plus the two
    platform rows the real feed carries, so the parent-station filter is
    actually exercised. `detached` adds a second, unconnected system."""
    stops_rows = STOPS + (DETACHED_STOPS if detached else [])
    trips_rows = TRIPS + (DETACHED_TRIPS if detached else [])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        stops = io.StringIO()
        writer = csv.writer(stops)
        writer.writerow(
            [
                "stop_id",
                "stop_name",
                "stop_lat",
                "stop_lon",
                "location_type",
                "parent_station",
            ]
        )
        for sid, name, lat, lon in stops_rows:
            writer.writerow([sid, name, lat, lon, "1", ""])
            for suffix in ("N", "S"):
                writer.writerow([f"{sid}{suffix}", name, lat, lon, "0", sid])
        zf.writestr("stops.txt", stops.getvalue())

        zf.writestr(
            "routes.txt",
            "route_id,route_long_name,route_color\n"
            "1,One Line,D82233\n2,Two Line,009952\nSI,Island Railway,08179C\n",
        )
        zf.writestr(
            "trips.txt",
            "route_id,service_id,trip_id\n"
            + "".join(
                f"{route},{service},{trip}\n" for trip, route, service, _ in trips_rows
            ),
        )
        rows = ["trip_id,stop_id,stop_sequence,arrival_time"]
        for trip, _, _, stops_on_trip in trips_rows:
            for i, (stop, arrival) in enumerate(stops_on_trip, start=1):
                rows.append(f"{trip},{stop},{i},{arrival}")
        zf.writestr("stop_times.txt", "\n".join(rows) + "\n")
        zf.writestr(
            "feed_info.txt",
            "feed_version,feed_start_date,feed_end_date\ntest-feed,20260801,20261031\n",
        )
        # Two real service classes, so the diagram has an actual Mon-Fri vs
        # Saturday split to resolve rather than the single WKD class the
        # fixture used to carry — a fixture that never crosses a service
        # class exercises the splitting code without exercising the reason
        # splitting exists. Sunday is deliberately absent: nothing in TRIPS
        # runs on it, which is the class-level "no timetable at all" case.
        zf.writestr(
            "calendar.txt",
            "service_id,monday,tuesday,wednesday,thursday,friday,saturday,"
            "sunday,start_date,end_date\n"
            "WKD,1,1,1,1,1,0,0,20260801,20261031\n"
            "SAT,0,0,0,0,0,1,0,20260801,20261031\n",
        )
    return zipfile.ZipFile(buf)


def test_an_edge_carries_the_segment_flow_key_for_each_direction() -> None:
    """The key is the join to the published surface. It must be built from the
    timetable's own directional stop id, in travel order: northbound across
    100<->101 is measured at 100N, southbound at 101S."""
    pairs = edge_directions({("1", "north", "100N"): [("101N", 9)]})
    assert pairs == {("1", "100", "101"): {"north": "1|north|100N"}}


def test_a_pair_scheduled_both_ways_is_one_edge_with_two_keys() -> None:
    pairs = edge_directions(
        {
            ("1", "north", "100N"): [("101N", 9)],
            ("1", "south", "101S"): [("100S", 9)],
        }
    )
    assert pairs == {
        ("1", "100", "101"): {
            "north": "1|north|100N",
            "south": "1|south|101S",
        }
    }


def test_a_one_way_pair_gets_one_key_not_a_fabricated_twin() -> None:
    """Half the value of keying off the timetable: a branch hop the schedule
    only runs one way must not claim a southbound cell that can never be
    published, or the map would paint it permanently unreadable."""
    diagram = build(_feed())
    branch = next(e for e in diagram.edges if e.route == "1" and e.b == "103")
    assert set(branch.keys) == {"north"}
    assert branch.keys["north"] == "1|north|102N"


def test_the_busiest_pattern_wins_a_contested_successor() -> None:
    """Same rule as gtfs_static.dominant_successor: a from_stop reaching one
    neighbour under two patterns resolves to the busier, not to whichever the
    dict happened to iterate last."""
    pairs = edge_directions({("1", "north", "100N"): [("101N", 2), ("101N", 40)]})
    assert pairs[("1", "100", "101")]["north"] == "1|north|100N"


def test_an_edge_carries_seconds_for_each_service_class_and_direction() -> None:
    """seconds is class -> direction -> scheduled seconds, one entry per
    (class, direction) the timetable actually ran a trip for, read off the
    same (from_stop, to_stop) hop edge_directions keyed."""
    succ = {
        ("1", "north", "100N"): [("101N", 9)],
        ("1", "south", "101S"): [("100S", 9)],
    }
    hops_by_class = {
        "weekday": {
            ("1", "north", "100N", "101N"): 180,
            ("1", "south", "101S", "100S"): 190,
        },
        "saturday": {
            ("1", "north", "100N", "101N"): 240,
        },
    }
    assert edge_seconds(succ, hops_by_class) == {
        ("1", "100", "101"): {
            "weekday": {"north": 180, "south": 190},
            "saturday": {"north": 240},
        }
    }


def test_a_class_or_direction_the_timetable_never_timed_is_absent_not_zero() -> None:
    """A hop the topology schedules but a class's hop dict has no entry for —
    a service class that simply doesn't run the hop, the way Sunday runs
    nothing in the fixture below — must be missing entirely, never a
    fabricated 0 seconds."""
    succ = {("1", "north", "100N"): [("101N", 9)]}
    assert edge_seconds(succ, {"weekday": {}, "saturday": {}}) == {}


def test_edge_seconds_and_edge_directions_resolve_the_same_contested_hop() -> None:
    """Two candidate origins reaching one station pair must resolve to the
    SAME winner in both functions — the segment_flow key and its scheduled
    time can never end up describing two different physical hops. (The
    southbound-suffixed second origin is synthetic: it isolates the
    dominance rule from direction_of, which this function never calls.)"""
    succ = {
        ("1", "north", "100N"): [("101N", 3)],
        ("1", "north", "100S"): [("101S", 40)],
    }
    hops_by_class = {
        "weekday": {
            ("1", "north", "100N", "101N"): 999,  # the losing hop's time
            ("1", "north", "100S", "101S"): 150,  # the winning hop's time
        }
    }
    assert edge_directions(succ)[("1", "100", "101")]["north"] == "1|north|100S"
    seconds = edge_seconds(succ, hops_by_class)
    assert seconds[("1", "100", "101")]["weekday"]["north"] == 150


def test_representative_days_find_one_of_each_class_from_the_window_start() -> None:
    """Derived from the feed's own validity window, not a hardcoded date, so
    the picks don't quietly go stale the next time MTA republishes with a
    shifted window. 2026-08-01 is a Saturday."""
    version = FeedVersion(version="v", start=date(2026, 8, 1), end=None)
    assert representative_days(version) == {
        "saturday": date(2026, 8, 1),
        "sunday": date(2026, 8, 2),
        "weekday": date(2026, 8, 3),
    }


def test_an_edges_seconds_are_the_arrival_to_arrival_scheduled_time() -> None:
    """seconds is read off the same timetable that measures the network,
    arrival-to-arrival, not re-derived from the drawn geometry."""
    diagram = build(_feed())
    by_pair = {(e.route, e.a, e.b): e for e in diagram.edges}
    # 101<->102 is a plain single-trip weekday hop in both directions: 4
    # minutes each way, straight off the fixture's arrival times.
    assert by_pair[("1", "101", "102")].seconds["weekday"] == {
        "north": 240,
        "south": 240,
    }
    assert by_pair[("1", "100", "101")].seconds["weekday"]["south"] == 180
    assert by_pair[("2", "100", "101")].seconds["weekday"] == {
        "north": 180,
        "south": 240,
    }
    assert by_pair[("2", "101", "103")].seconds["weekday"] == {
        "north": 420,
        "south": 420,
    }


def test_multiple_trips_on_one_hop_resolve_to_the_median_not_the_mean() -> None:
    """Route 1's 100->101 northbound weekday hop is run by three trips in the
    fixture — 180s, 190s and a padded 1000s late-night run. The published
    seconds must be the pooled median (190), not the mean (~457) the one
    outlier would otherwise drag the number toward."""
    diagram = build(_feed())
    edge = next(
        e for e in diagram.edges if e.route == "1" and (e.a, e.b) == ("100", "101")
    )
    assert edge.seconds["weekday"]["north"] == 190


def test_saturday_and_weekday_resolve_to_different_scheduled_times() -> None:
    """The reason service classes are kept apart rather than pooled: the SAME
    physical hop (route 1, 100->101 northbound) runs on a genuinely different
    schedule Saturday (300s) than it does on a weekday (190s). Sunday runs
    nothing at all in this fixture, so it must be entirely absent from the
    edge rather than falling back to either of the other two."""
    diagram = build(_feed())
    edge = next(
        e for e in diagram.edges if e.route == "1" and (e.a, e.b) == ("100", "101")
    )
    assert edge.seconds["weekday"]["north"] == 190
    assert edge.seconds["saturday"]["north"] == 300
    assert "sunday" not in edge.seconds


def test_a_branch_hop_with_no_scheduled_arrival_carries_no_seconds() -> None:
    """102->103 exists in the topology (t5 runs it) but its arrival at 103N is
    blank in the fixture, mirroring the slice of real hops the static feed's
    arrival_time never covers. The edge must publish no seconds at all, in
    any class or direction, rather than a fabricated 0."""
    diagram = build(_feed())
    branch = next(e for e in diagram.edges if e.route == "1" and e.b == "103")
    assert branch.seconds == {}


def test_stations_are_parent_rows_only_and_carry_their_routes() -> None:
    diagram = build(_feed())
    assert set(diagram.stations) == {"100", "101", "102", "103"}
    assert diagram.stations["101"].routes == ("1", "2")
    assert diagram.stations["103"].routes == ("1", "2")
    assert diagram.stations["100"].name == "South End"


def test_routes_sharing_a_pair_are_fanned_onto_distinct_strokes() -> None:
    """The reason the diagram is drawn per (route, pair) rather than per shape:
    two routes on one trunk are two separately-measured cells, so they must not
    land on the same pixels.

    The gap is the full configured spacing at 100, where the trunk is all
    either route runs, and tapers at 101, where each route also owns a solo hop
    pulling its vertex back toward the centre line. That taper is the price of
    strokes that actually join, and it must still leave the two apart.
    """
    diagram = build(_feed())
    trunk = {e.route: e.path for e in diagram.edges if (e.a, e.b) == ("100", "101")}
    assert set(trunk) == {"1", "2"}
    # Endpoints, not path indices: an off-axis hop is routed as a dogleg, so
    # path[1] may be an interior bend rather than the far station.
    gaps = [math.dist(trunk["1"][i], trunk["2"][i]) for i in (0, -1)]
    assert gaps[0] == _approx(OFFSET_SPACING, 0.05)
    assert 0 < gaps[1] < gaps[0]


def test_consecutive_edges_of_one_route_meet_exactly() -> None:
    """A fan offset applied per hop leaves a gap at every station where the
    bearing turns. Averaging the offset at the shared vertex is what closes it,
    and a diagram of broken strokes is the visible symptom if this regresses."""
    diagram = build(_feed())
    first = next(
        e for e in diagram.edges if e.route == "1" and (e.a, e.b) == ("100", "101")
    )
    second = next(
        e for e in diagram.edges if e.route == "1" and (e.a, e.b) == ("101", "102")
    )
    assert first.path[-1] == second.path[0]


def test_a_near_axis_hop_is_left_straight() -> None:
    """Every NYC hop sits within 22° of a 45° axis, so doglegging all of them
    turns a long gentle run into a staircase of shallow bends. Inside the
    tolerance the hop already reads as octilinear and is left alone."""
    assert octilinear((0.0, 0.0), (100.0, 0.0)) == ((0.0, 0.0), (100.0, 0.0))
    assert octilinear((0.0, 0.0), (100.0, 100.0)) == ((0.0, 0.0), (100.0, 100.0))
    # 5° off horizontal: inside tolerance, still one straight segment.
    assert len(octilinear((0.0, 0.0), (100.0, 8.7))) == 2


def test_an_off_axis_hop_is_routed_on_the_eight_bearings() -> None:
    path = octilinear((0.0, 0.0), (100.0, 30.0))
    assert path[0] == (0.0, 0.0)
    assert path[-1] == (100.0, 30.0)
    for p, q in itertools.pairwise(path):
        assert axis_deviation(p, q) == _approx(0.0, 1e-9)


def test_a_dogleg_leaves_both_ends_along_an_axis() -> None:
    """What makes consecutive edges join without kinking: each edge departs its
    station axis-aligned, so two edges meeting at a through-stop are collinear
    there. Splitting the straight run evenly across both ends is how."""
    path = octilinear((0.0, 0.0), (100.0, 30.0))
    assert len(path) == 4
    assert path[1][1] == path[0][1], "leaves the first station horizontally"
    assert path[-2][1] == path[-1][1], "arrives at the second horizontally"
    assert math.dist(path[0], path[1]) == _approx(math.dist(path[-2], path[-1]))


def test_a_dogleg_stays_inside_its_own_hop() -> None:
    """The view box and the inset search are both computed from station
    positions, so a route that bulged outside the hop's bounding box would
    escape the canvas and the occupancy scan alike."""
    for target in ((100.0, 30.0), (-100.0, 30.0), (30.0, 100.0), (-30.0, -100.0)):
        for x, y in octilinear((0.0, 0.0), target):
            assert min(0.0, target[0]) - 1e-9 <= x <= max(0.0, target[0]) + 1e-9
            assert min(0.0, target[1]) - 1e-9 <= y <= max(0.0, target[1]) + 1e-9


def test_the_view_box_preserves_the_projection_aspect() -> None:
    """A stretched subway map lies about direction, so height follows from the
    projection rather than being fitted to a target box."""
    diagram = build(_feed())
    _, _, width, height = diagram.view_box
    assert width == _approx(VIEW_WIDTH + 2 * PAD)
    corners = [project(lat, lon) for _, _, lat, lon in STOPS]
    span_x = max(x for x, _ in corners) - min(x for x, _ in corners)
    span_y = max(y for _, y in corners) - min(y for _, y in corners)
    assert (height - 2 * PAD) / (width - 2 * PAD) == _approx(span_y / span_x)


def test_components_split_a_track_graph_that_shares_no_stops() -> None:
    groups = components([("1", "100", "101"), ("SI", "S30", "S31")])
    assert groups == [frozenset({"100", "101"}), frozenset({"S30", "S31"})]


def test_transfers_merge_divisions_that_use_different_ids_at_one_complex() -> None:
    """The trap this guards: the IRT, IND/BMT, 7, L and shuttles each have their
    own stop id at a shared complex, so a pure track graph reads the subway as
    six detached systems and would inset five of them. transfers.txt is how the
    feed says two ids are one place."""
    pairs = [("1", "126", "127"), ("A", "A27", "A28"), ("SI", "S30", "S31")]
    assert len(components(pairs)) == 3
    groups = components(pairs, [("127", "A27")])
    assert groups == [
        frozenset({"126", "127", "A27", "A28"}),
        frozenset({"S30", "S31"}),
    ]


def test_a_detached_component_is_inset_and_declared() -> None:
    """Placed at true position, a component across water stretches the box over
    empty canvas. It's scaled into a hole in the main body instead — and marked,
    because an unmarked relocation is a map that lies about where it is."""
    diagram = build(_feed(detached=True))
    assert len(diagram.insets) == 1
    inset = diagram.insets[0]
    assert inset.routes == ("SI",)
    assert inset.scale < 1
    # The box holds the component it names, and nothing else moved into it.
    x, y, width, height = inset.box
    for sid in ("900", "901"):
        station = diagram.stations[sid]
        assert x - 0.01 <= station.x <= x + width + 0.01
        assert y - 0.01 <= station.y <= y + height + 0.01


def test_the_inset_does_not_land_on_top_of_the_main_body() -> None:
    """The whole point of searching for an empty block: an inset over Coney
    Island would be worse than the wasted canvas it saves."""
    diagram = build(_feed(detached=True))
    x, y, width, height = diagram.insets[0].box
    inset_stations = {"900", "901"}
    for edge in diagram.edges:
        if edge.a in inset_stations:
            continue
        for px, py in edge.path:
            assert not (x <= px <= x + width and y <= py <= y + height), (
                f"{edge.route} {edge.a}->{edge.b} crosses the inset box"
            )


def test_the_view_box_ignores_the_detached_component() -> None:
    """The main body sets the scale. Letting a ferry-only line stretch the box
    is the waste the inset exists to remove, so the box must not grow when one
    appears."""
    plain = build(_feed()).view_box
    with_inset = build(_feed(detached=True)).view_box
    assert plain == with_inset


def test_route_metadata_comes_from_the_feed_and_covers_every_drawn_route() -> None:
    diagram = build(_feed())
    assert set(diagram.routes) == {"1", "2"}
    assert diagram.routes["1"].color == "#D82233"
    assert diagram.routes["2"].name == "Two Line"


def test_feed_version_rides_along_as_provenance() -> None:
    payload = to_json(build(_feed()))
    assert payload["feed_version"] == {
        "version": "test-feed",
        "start": "2026-08-01",
        "end": "2026-10-31",
    }


def test_the_asset_is_stable_across_builds_of_the_same_feed() -> None:
    """No timestamp, sorted output: regenerating against an unchanged timetable
    has to produce an identical file or the committed asset churns on every
    run and a real topology change stops standing out in the diff."""
    assert to_json(build(_feed())) == to_json(build(_feed()))


def test_route_order_puts_unlisted_routes_last() -> None:
    assert route_sort_key("1") < route_sort_key("A")
    assert route_sort_key("SI") < route_sort_key("ZZ")
    assert route_sort_key("ZZ") < route_sort_key("ZZZ")


def test_a_feed_with_no_drawable_pairs_is_an_error_not_an_empty_diagram() -> None:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("stops.txt", "stop_id,stop_name,stop_lat,stop_lon,parent_station\n")
        zf.writestr("routes.txt", "route_id,route_long_name,route_color\n")
        zf.writestr("trips.txt", "route_id,service_id,trip_id\n")
        zf.writestr("stop_times.txt", "trip_id,stop_id,stop_sequence\n")
    with pytest.raises(ValueError, match="no drawable station pairs"):
        build(zipfile.ZipFile(buf))
