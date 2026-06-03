"""Tests for the E&E feed parser + station-status derive."""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from momentarily.derive import derive_station_status
from momentarily.ene import (
    is_active_outage,
    is_long_running,
    parse_feed_payload,
    parse_outage_record,
)
from momentarily.schema import (
    Alert,
    Equipment,
    EquipmentOutage,
    InformedEntity,
    Station,
)

NY = ZoneInfo("America/New_York")


def _et_epoch(year: int, month: int, day: int, hour: int = 12) -> int:
    return int(datetime(year, month, day, hour, tzinfo=NY).astimezone(UTC).timestamp())


def test_parse_outage_record_basic() -> None:
    record = {
        "station": "61 St-Woodside",
        "equipment": "ES448",
        "equipmenttype": "ES",
        "serving": "61 St & Roosevelt Ave (SE corner) to mezzanine",
        "ADA": "N",
        "outagedate": "09/30/2024 12:05:00 PM",
        "estimatedreturntoservice": "05/31/2026 11:59:00 PM",
        "reason": "Capital Replacement",
        "isupcomingoutage": "N",
        "ismaintenanceoutage": "N",
    }
    eq = parse_outage_record(record)
    assert eq is not None
    assert eq.equipment_id == "ES448"
    assert eq.type == "escalator"
    assert eq.station_complex_id == "61 St-Woodside"
    assert eq.ada_pathway is False
    assert eq.outage is not None
    assert eq.outage.reason == "Capital Replacement"
    # outagedate parses to roughly 2024-09-30 in ET → UTC
    assert eq.outage.since == _et_epoch(2024, 9, 30, 12) + 5 * 60


def test_parse_handles_missing_estimated_return() -> None:
    record = {
        "equipment": "EL999",
        "equipmenttype": "EL",
        "outagedate": "01/15/2026 03:00:00 AM",
        "estimatedreturntoservice": "",
        "reason": "Mechanical",
        "ADA": "Y",
    }
    eq = parse_outage_record(record)
    assert eq is not None
    assert eq.type == "elevator"
    assert eq.ada_pathway is True
    assert eq.outage is not None
    assert eq.outage.est_return is None
    assert eq.outage.since is not None


def test_parse_rejects_unknown_equipment_type() -> None:
    record = {"equipment": "X1", "equipmenttype": "GLOBAL_WARMING"}
    assert parse_outage_record(record) is None


def test_parse_feed_payload_skips_garbage() -> None:
    payload = [
        {
            "equipment": "EL1",
            "equipmenttype": "EL",
            "outagedate": "01/01/2026 12:00:00 PM",
        },
        "not a dict",
        None,
        {
            "equipment": "ES2",
            "equipmenttype": "ES",
            "outagedate": "01/02/2026 12:00:00 PM",
        },
        {"equipmenttype": "EL"},  # missing equipment id
    ]
    out = parse_feed_payload(payload)
    assert len(out) == 2
    assert [e.equipment_id for e in out] == ["EL1", "ES2"]


def test_active_outage_respects_past_estimated_return() -> None:
    past = EquipmentOutage(
        since=_et_epoch(2026, 1, 1), est_return=_et_epoch(2026, 1, 2)
    )
    current = EquipmentOutage(
        since=_et_epoch(2026, 1, 1), est_return=_et_epoch(2026, 12, 1)
    )
    now = _et_epoch(2026, 5, 1)
    assert not is_active_outage(past, now=now)
    assert is_active_outage(current, now=now)


def test_long_running_flag() -> None:
    fresh = EquipmentOutage(since=_et_epoch(2026, 4, 30))
    stale = EquipmentOutage(since=_et_epoch(2024, 1, 1))
    now = _et_epoch(2026, 5, 1)
    assert not is_long_running(fresh, now=now)
    assert is_long_running(stale, now=now)


# ---------------------------------------------------------------------------
# derive_station_status
# ---------------------------------------------------------------------------


def _station() -> Station:
    return Station(
        gtfs_stop_id="R03",
        station_complex_id="R03",
        name="Times Sq-42 St",
        borough="M",
        routes_served=["1", "2", "3"],
        ada=2,
        ada_northbound=True,
        ada_southbound=True,
    )


def _eq(
    eid: str,
    typ: str = "elevator",
    *,
    out: bool = True,
    ada_pathway: bool = True,
    since: int | None = None,
    est_return: int | None = None,
) -> Equipment:
    outage = (
        EquipmentOutage(since=since, est_return=est_return, reason="x") if out else None
    )
    return Equipment(
        equipment_id=eid,
        type=typ,  # type: ignore[arg-type]
        station_complex_id="R03",
        ada_pathway=ada_pathway,
        outage=outage,
    )


def test_station_status_all_operational() -> None:
    now = _et_epoch(2026, 5, 1)
    eq_in_service = _eq("EL1", out=False, ada_pathway=True)
    status = derive_station_status(_station(), [eq_in_service], [], now)
    assert status.elevators_total == 1
    assert status.elevators_out == 0
    assert status.ada_status == "operational"
    assert status.earliest_elevator_return is None
    assert status.oldest_outage_since is None


def test_station_status_ada_degraded_picks_earliest_return() -> None:
    now = _et_epoch(2026, 5, 1)
    e1 = _eq(
        "EL1",
        out=True,
        ada_pathway=True,
        since=_et_epoch(2026, 4, 1),
        est_return=_et_epoch(2026, 5, 10),
    )
    e2 = _eq(
        "EL2",
        out=True,
        ada_pathway=True,
        since=_et_epoch(2026, 3, 15),
        est_return=_et_epoch(2026, 6, 1),
    )
    status = derive_station_status(_station(), [e1, e2], [], now)
    assert status.elevators_out == 2
    assert status.ada_status == "ada_degraded"
    # min of est_returns
    assert status.earliest_elevator_return == _et_epoch(2026, 5, 10)
    # min of sinces
    assert status.oldest_outage_since == _et_epoch(2026, 3, 15)


def test_station_status_non_ada_when_no_ada_path_equipment() -> None:
    now = _et_epoch(2026, 5, 1)
    only_escalators = [
        _eq("ES1", typ="escalator", out=False, ada_pathway=False),
        _eq("ES2", typ="escalator", out=False, ada_pathway=False),
    ]
    status = derive_station_status(_station(), only_escalators, [], now)
    assert status.ada_status == "non_ada"


def test_station_status_filters_alerts_by_stop() -> None:
    now = _et_epoch(2026, 5, 1)
    relevant = Alert(
        id="a",
        alert_type="Delays",
        sort_order=22,
        active_period=[],
        informed_entities=[InformedEntity(stop_id="R03")],
        source="subway",
    )
    other = Alert(
        id="b",
        alert_type="Delays",
        sort_order=22,
        active_period=[],
        informed_entities=[InformedEntity(stop_id="OTHER")],
        source="subway",
    )
    status = derive_station_status(_station(), [], [relevant, other], now)
    assert status.alerts == ["a"]


def test_station_status_no_est_return_means_none() -> None:
    now = _et_epoch(2026, 5, 1)
    e1 = _eq(
        "EL1",
        out=True,
        ada_pathway=True,
        since=_et_epoch(2026, 4, 1),
        est_return=None,
    )
    status = derive_station_status(_station(), [e1], [], now)
    assert status.elevators_out == 1
    assert status.earliest_elevator_return is None  # nothing to surface
    assert status.oldest_outage_since == _et_epoch(2026, 4, 1)
