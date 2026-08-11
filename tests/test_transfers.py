"""Transfer viability from station metadata (training/transfers.py).

Synthetic stations dict only — no network access.
"""

from __future__ import annotations

from training.transfers import (
    StationRecord,
    build_complexes,
    complexes_joining,
    routes_at,
    station_id,
)


def _station(
    gtfs_stop_id: str,
    routes_served: list[str],
    *,
    complex_id: str | None,
    ada: int = 0,
) -> StationRecord:
    return StationRecord(
        gtfs_stop_id=gtfs_stop_id,
        station_complex_id=complex_id,
        name=gtfs_stop_id,
        borough=None,
        routes_served=routes_served,
        ada=ada,  # type: ignore[typeddict-item]
        ada_northbound=False,
        ada_southbound=False,
    )


def test_station_id_strips_direction_suffix():
    assert station_id("A09N") == "A09"
    assert station_id("A09S") == "A09"
    assert station_id("A09") == "A09"
    assert station_id("") == ""


def test_ns_strip_maps_both_directional_stops_to_one_station():
    # stations.json is keyed on the bare (non-directional) gtfs_stop_id, as
    # 39hk-dx4f publishes it. Directional segment-world ids collapse to it.
    stations = {"A09": _station("A09", ["A", "C"], complex_id=None)}
    assert routes_at(stations, "A09N") == frozenset({"A", "C"})
    assert routes_at(stations, "A09S") == frozenset({"A", "C"})
    assert routes_at(stations, "A09N") == routes_at(stations, "A09S")


def test_routes_at_unknown_stop_is_empty():
    assert routes_at({}, "Z99N") == frozenset()


def test_two_routes_at_same_complex_are_transferable():
    stations = {
        "127": _station("127", ["1"], complex_id="611"),
        "902": _station("902", ["7"], complex_id="611"),
    }
    complexes = build_complexes(stations)
    result = complexes_joining(complexes, "1", "7")
    assert result.viable
    assert result.complexes == ("611",)


def test_same_routes_at_different_complexes_are_not_transferable():
    stations = {
        "127": _station("127", ["1"], complex_id="611"),
        "725": _station("725", ["7"], complex_id="723"),  # different complex
    }
    complexes = build_complexes(stations)
    result = complexes_joining(complexes, "1", "7")
    assert not result.viable
    assert result.complexes == ()


def test_null_complex_is_a_singleton_not_merged_with_other_nulls():
    # Two physically distinct stops, both with no documented complex id, each
    # serving only one route: routes_served alone must not fabricate a
    # transfer that the topology never asserted.
    stations = {
        "P01": _station("P01", ["A"], complex_id=None),
        "P02": _station("P02", ["B"], complex_id=None),
    }
    complexes = build_complexes(stations)
    assert set(complexes) == {"stop:P01", "stop:P02"}
    result = complexes_joining(complexes, "A", "B")
    assert not result.viable


def test_null_complex_singleton_still_joins_routes_at_the_same_stop():
    # A single stop naturally serves multiple routes without any documented
    # complex id — that's still a real, viable transfer at that one platform.
    stations = {"P03": _station("P03", ["A", "B"], complex_id=None)}
    complexes = build_complexes(stations)
    assert set(complexes) == {"stop:P03"}
    result = complexes_joining(complexes, "A", "B")
    assert result.viable
    assert result.complexes == ("stop:P03",)


def test_ada_viability_is_separate_axis_and_never_flips_plain_viability():
    # Both stops in the complex are transfer-viable for A<->B, but neither is
    # ADA accessible: plain viable stays True, ada_viable is a distinct False.
    stations = {
        "127": _station("127", ["1"], complex_id="611", ada=0),
        "902": _station("902", ["7"], complex_id="611", ada=0),
    }
    complexes = build_complexes(stations)
    result = complexes_joining(complexes, "1", "7")
    assert result.viable
    assert not result.ada_viable
    assert result.ada_complexes == ()


def test_ada_viability_true_when_both_sides_of_transfer_are_accessible():
    stations = {
        "127": _station("127", ["1"], complex_id="611", ada=1),
        "902": _station("902", ["7"], complex_id="611", ada=2),  # partial counts
    }
    complexes = build_complexes(stations)
    result = complexes_joining(complexes, "1", "7")
    assert result.viable
    assert result.ada_viable
    assert result.ada_complexes == ("611",)


def test_ada_viability_false_when_only_one_side_is_accessible():
    stations = {
        "127": _station("127", ["1"], complex_id="611", ada=1),
        "902": _station("902", ["7"], complex_id="611", ada=0),  # not accessible
    }
    complexes = build_complexes(stations)
    result = complexes_joining(complexes, "1", "7")
    assert result.viable  # plain viability unaffected by the accessible side
    assert not result.ada_viable


def test_complex_rolls_up_routes_across_all_its_stops():
    stations = {
        "127": _station("127", ["1", "2", "3"], complex_id="611"),
        "902": _station("902", ["7"], complex_id="611"),
        "901": _station("901", ["S"], complex_id="611"),
    }
    complexes = build_complexes(stations)
    assert complexes["611"].routes == frozenset({"1", "2", "3", "7", "S"})
    assert complexes["611"].stops == ("127", "901", "902")
