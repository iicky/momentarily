"""Unit tests for the station-facts generator (scripts/gen_station_facts.py).

Covers the pure join logic with no network: geohash decode, the layered
matcher's tier order and thresholds, the collision-aware authoritative index
(the Transitland Onestop -> GTFS stop_id map), the Transitland response parse,
the twelve-month ridership window floor, and the opening-year fold.
"""

from __future__ import annotations

from datetime import date

from scripts.gen_station_facts import (
    COORD_MAX_M,
    ONESTOP_MAX_M,
    Station,
    build_authoritative_index,
    geohash_decode,
    haversine_m,
    match_station,
    onestop_latlon,
    opened_year,
    parse_transitland_stops,
    window_start,
)

# A real Times Sq Onestop ID and the coordinate its geohash encodes.
TIMES_SQ_ONESTOP = "s-dr5ru77by2-timessq~42st"
TIMES_SQ_LAT, TIMES_SQ_LON = 40.7553, -73.9875


def _station(sid: str, lat: float, lon: float) -> Station:
    return Station(sid, sid, lat, lon, complex_id="0", routes=frozenset())


def test_geohash_decode_lands_on_the_encoded_point() -> None:
    lat, lon = geohash_decode("dr5ru77by2")
    assert haversine_m((lat, lon), (TIMES_SQ_LAT, TIMES_SQ_LON)) < 60


def test_onestop_latlon_reads_the_embedded_geohash() -> None:
    ll = onestop_latlon(TIMES_SQ_ONESTOP)
    assert ll is not None
    assert haversine_m(ll, (TIMES_SQ_LAT, TIMES_SQ_LON)) < 60


def test_onestop_latlon_rejects_a_non_stop_id() -> None:
    assert onestop_latlon("o-dr5r-mta") is None


def test_haversine_symmetric_and_zero() -> None:
    a, b = (40.0, -73.0), (40.1, -73.1)
    assert haversine_m(a, a) == 0.0
    assert abs(haversine_m(a, b) - haversine_m(b, a)) < 1e-6


def test_parse_transitland_strips_direction_suffix() -> None:
    payload = {
        "stops": [
            {"onestop_id": "s-a-x", "stop_id": "127N"},
            {"onestop_id": "s-b-y", "stop_id": "127S"},
            {"onestop_id": "s-c-z", "stop_id": "R16"},
            {"onestop_id": "s-d-w"},  # no stop_id -> skipped
        ]
    }
    assert parse_transitland_stops(payload) == {
        "s-a-x": "127",
        "s-b-y": "127",
        "s-c-z": "R16",
    }


def test_authoritative_index_accepts_one_item_per_stop() -> None:
    onestop_pts = [
        (40.0, -73.0, "Q1", "s-a-x"),
        (40.0, -73.0, "Q1", "s-b-y"),  # same item, second platform
    ]
    transitland = {"s-a-x": "127", "s-b-y": "127"}
    index, ambiguous = build_authoritative_index(onestop_pts, transitland)
    assert index["127"] == ("Q1", "s-a-x") or index["127"] == ("Q1", "s-b-y")
    assert ambiguous == {}


def test_authoritative_index_flags_conflicting_items() -> None:
    onestop_pts = [
        (40.0, -73.0, "Q1", "s-a-x"),
        (40.0, -73.0, "Q2", "s-b-y"),  # a DIFFERENT item on the same stop
    ]
    transitland = {"s-a-x": "127", "s-b-y": "127"}
    index, ambiguous = build_authoritative_index(onestop_pts, transitland)
    assert "127" not in index
    assert ambiguous == {"127": {"Q1", "Q2"}}


def test_match_prefers_authoritative_over_nearer_geohash() -> None:
    # A wrong item's platform sits 1 m away; the authoritative map names the
    # right item for this stop. The direct lookup must win regardless.
    st = _station("127", 40.0, -73.0)
    onestop_pts = [
        (40.0, -73.0, "Q_WRONG", "s-wrong"),  # 0 m, but not our stop's item
        (40.01, -73.0, "Q_RIGHT", "s-right"),  # ~1.1 km away
    ]
    authoritative = {"127": ("Q_RIGHT", "s-right")}
    m = match_station(st, onestop_pts, [], authoritative, set())
    assert m is not None
    assert m.method == "transitland_onestop"
    assert m.authoritative is True
    assert m.qid == "Q_RIGHT"


def test_match_ambiguous_authoritative_is_unmatched() -> None:
    st = _station("127", 40.0, -73.0)
    onestop_pts = [(40.0, -73.0, "Q1", "s-a")]
    m = match_station(st, onestop_pts, [], {}, {"127"})
    assert m is None


def test_match_falls_back_to_geohash_then_coordinate() -> None:
    st = _station("127", 40.0, -73.0)
    # Onestop point ~14 m away (inside ONESTOP_MAX_M) -> geohash tier.
    near = (40.0001, -73.0, "Q_OS", "s-os")
    m = match_station(st, [near], [(40.0, -73.0, "Q_IT")], {}, set())
    assert m is not None
    assert m.method == "onestop_geohash"
    assert not m.authoritative
    assert m.distance_m is not None
    assert m.distance_m <= ONESTOP_MAX_M

    # No Onestop point in range -> item-coordinate tier.
    far_os = (40.01, -73.0, "Q_OS", "s-os")  # ~1.1 km
    m2 = match_station(st, [far_os], [(40.0005, -73.0, "Q_IT")], {}, set())
    assert m2 is not None
    assert m2.method == "coordinate"
    assert m2.distance_m is not None
    assert m2.distance_m <= COORD_MAX_M


def test_match_returns_none_beyond_all_thresholds() -> None:
    st = _station("127", 40.0, -73.0)
    far = (41.0, -74.0, "Q", "s")  # ~130 km
    assert match_station(st, [far], [(41.0, -74.0, "Q")], {}, set()) is None


def test_geohash_tie_between_distinct_items_abstains() -> None:
    # Both items' platforms sit on the same spot (an identical-geohash collision):
    # the spatial fallback cannot tell them apart, so it must not guess.
    st = _station("127", 40.0, -73.0)
    onestop_pts = [
        (40.0, -73.0, "Q1", "s-a-one"),
        (40.0, -73.0, "Q2", "s-b-two"),
    ]
    assert match_station(st, onestop_pts, [], {}, set()) is None


def test_coordinate_tie_between_distinct_items_abstains() -> None:
    st = _station("127", 40.0, -73.0)
    item_pts = [(40.0005, -73.0, "Q1"), (40.0005, -73.0, "Q2")]  # ~55 m, both
    assert match_station(st, [], item_pts, {}, set()) is None


def test_window_start_is_twelve_complete_months() -> None:
    assert window_start(date(2026, 6, 1)) == date(2025, 7, 1)
    assert window_start(date(2026, 1, 1)) == date(2025, 2, 1)
    assert window_start(date(2026, 12, 1)) == date(2026, 1, 1)


def test_opened_year_takes_the_earliest() -> None:
    year, iso = opened_year({"1932-01-01T00:00:00Z", "1904-10-27T00:00:00Z"})
    assert year == 1904
    assert iso == "1904-10-27"
    assert opened_year(set()) == (None, None)
