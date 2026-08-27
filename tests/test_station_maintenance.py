"""Unit tests for the per-station maintenance sidecar reduction
(training/station_maintenance.py).

The network-touching parts (listing/fetching R2, reading the window archive)
are exercised by running the tool; these tests pin the PURE reductions the tool
composes — episode reconstruction from the hourly feed, the per-stop outage
tally, the planned-closure count, and the published document shape — against
synthetic inputs, so a change in what a number means fails here.
"""

from __future__ import annotations

from momentarily.schema import Equipment, EquipmentOutage
from training.planned_work import Window
from training.station_maintenance import (
    Episode,
    build_doc,
    equipment_stop_map,
    reconstruct_episodes,
    reduce_closures,
    reduce_outages,
)


def _unit(
    equipment_id: str,
    kind: str,
    *,
    since: int | None,
    reason: str | None = None,
    est_return: int | None = None,
) -> Equipment:
    return Equipment(
        equipment_id=equipment_id,
        type=kind,  # type: ignore[arg-type]
        outage=EquipmentOutage(reason=reason, est_return=est_return, since=since),
    )


def test_reconstruct_keys_episode_by_equipment_and_since() -> None:
    # Same unit, same MTA-reported start across two hours -> ONE outage spanning
    # both; a later record with a DIFFERENT start is a second, re-broken outage.
    snaps = [
        (1000, [_unit("EL1", "elevator", since=900)]),
        (4600, [_unit("EL1", "elevator", since=900)]),
        (90000, [_unit("EL1", "elevator", since=89000)]),
    ]
    eps = sorted(reconstruct_episodes(snaps), key=lambda e: e.first_seen)
    assert len(eps) == 2
    assert (eps[0].since, eps[0].first_seen, eps[0].last_seen) == (900, 1000, 4600)
    assert (eps[1].since, eps[1].first_seen, eps[1].last_seen) == (89000, 90000, 90000)


def test_reconstruct_bridges_a_collection_gap() -> None:
    # A missed middle hour must not split one outage: the shared `since` keys it,
    # so first/last span the whole observed range regardless of the hole.
    snaps = [
        (1000, [_unit("ES9", "escalator", since=500)]),
        # (missed 4600)
        (8200, [_unit("ES9", "escalator", since=500)]),
    ]
    (ep,) = reconstruct_episodes(snaps)
    assert ep.kind == "escalator"
    assert (ep.first_seen, ep.last_seen) == (1000, 8200)


def test_reduce_outages_counts_by_type_and_medians_resolved() -> None:
    # last snapshot at 100_000, gap 3600. E1 last seen well before the tail ->
    # resolved, TTR = (last_seen - since). E2 seen at the tail -> still open, no
    # repair time. Escalator counted on its own axis.
    episodes = [
        Episode(
            "EL1",
            "elevator",
            since=10_000,
            first_seen=10_000,
            last_seen=17_200,
            reason=None,
            est_return=None,
        ),
        Episode(
            "EL2",
            "elevator",
            since=50_000,
            first_seen=50_000,
            last_seen=100_000,
            reason=None,
            est_return=None,
        ),
        Episode(
            "ES1",
            "escalator",
            since=0,
            first_seen=0,
            last_seen=3_600,
            reason=None,
            est_return=None,
        ),
    ]
    eq_stops = {"EL1": ["A"], "EL2": ["A"], "ES1": ["A"]}
    per = reduce_outages(episodes, eq_stops, last_observed=100_000, gap=3600)
    a = per["A"]
    assert a.elevator == 2
    assert a.escalator == 1
    # Only the two resolved outages (EL1: 2.0h, ES1: 1.0h) back the repair list;
    # EL2 is still open at the tail.
    assert sorted(a.repair_hours) == [1.0, 2.0]


def test_reduce_outages_attributes_to_every_stop_of_a_complex() -> None:
    # A compound equipment id spans a multi-stop complex; the outage counts once
    # against EACH of its stops, so any stop page of the complex sees it.
    episodes = [
        Episode(
            "EL1",
            "elevator",
            since=1,
            first_seen=1,
            last_seen=1,
            reason=None,
            est_return=None,
        ),
    ]
    per = reduce_outages(
        episodes, {"EL1": ["112", "A09"]}, last_observed=100_000, gap=3600
    )
    assert per["112"].elevator == 1
    assert per["A09"].elevator == 1


def test_reduce_outages_drops_equipment_absent_from_catalog() -> None:
    episodes = [
        Episode(
            "GONE",
            "elevator",
            since=1,
            first_seen=1,
            last_seen=1,
            reason=None,
            est_return=None,
        ),
    ]
    assert reduce_outages(episodes, {}, last_observed=100_000) == {}


def test_equipment_stop_map_splits_compound_ids() -> None:
    catalog = [
        {"equipmentno": "EL1", "elevatorsgtfsstopid": "L06"},
        {"equipmentno": "EL2", "elevatorsgtfsstopid": "112/A09"},
        {"equipmentno": "EL3", "elevatorsgtfsstopid": None},  # non-NYCT: dropped
        {"equipmentno": "EL4"},  # no gtfs id: dropped
    ]
    assert equipment_stop_map(catalog) == {"EL1": ["L06"], "EL2": ["112", "A09"]}


def _window(alert_type: str, stops: set[str], start: int, end: int) -> Window:
    return Window(
        alert_type=alert_type,
        routes=frozenset({"A"}),
        stops=frozenset(stops),
        start=start,
        end=end,
    )


def test_reduce_closures_counts_only_service_removing_planned_types() -> None:
    windows = [
        _window("Planned - Suspended", {"A01"}, 1000, 2000),
        _window("Planned - Stops Skipped", {"A01", "A02"}, 1000, 2000),
        # A reroute keeps the station served — announced work, not a closure.
        _window("Planned - Reroute", {"A01"}, 1000, 2000),
        # The real-time (unplanned) counterpart is not planned work.
        _window("Suspended", {"A01"}, 1000, 2000),
    ]
    per = reduce_closures(windows, year_start=0, now=100_000)
    assert per == {"A01": 2, "A02": 1}


def test_reduce_closures_filters_to_the_year_window() -> None:
    windows = [
        # Ended before the year began: excluded.
        _window("Planned - Suspended", {"A01"}, 100, 200),
        # Starts in the future: excluded.
        _window("Planned - Suspended", {"A01"}, 9000, 9500),
        # Open-ended and already started: overlaps, counted.
        _window("Planned - Suspended", {"A01"}, 1500, 0),
    ]
    per = reduce_closures(windows, year_start=1000, now=5000)
    assert per == {"A01": 1}


def test_build_doc_shape_is_pure_and_stamped() -> None:
    outages = reduce_outages(
        [
            Episode(
                "EL1",
                "elevator",
                since=10,
                first_seen=10,
                last_seen=20,
                reason=None,
                est_return=None,
            )
        ],
        {"EL1": ["A"]},
        last_observed=100_000,
        gap=3600,
    )
    doc = build_doc(
        outages,
        {"A": 3, "B": 1},
        generated_at=1_700_000_000,
        month_start=1_699_000_000,
        year_start=1_690_000_000,
        window_end=1_700_000_000,
        ene_coverage=None,
        windows_coverage=None,
    )
    assert doc["schema_version"] == "1"
    assert doc["generated_at"] == 1_700_000_000
    assert doc["n_stations"] == 2
    a = doc["stations"]["A"]
    assert a["elevator_outages"] == 1
    assert a["escalator_outages"] == 0
    assert a["resolved_outages"] == 1
    assert a["median_repair_hours"] is not None
    assert a["planned_closures"] == 3
    # A stop with only a closure still appears, with zeroed outage fields.
    assert doc["stations"]["B"]["planned_closures"] == 1
    assert doc["stations"]["B"]["median_repair_hours"] is None
