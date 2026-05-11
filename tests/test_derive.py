"""Tests for the derivation logic (alerts → RouteStatus, SystemStatus, etc.)."""

from __future__ import annotations

from momentarily.derive import (
    alerts_for_route,
    derive_compat_route,
    derive_route_status,
    derive_system_status,
)
from momentarily.schema import Equipment, EquipmentOutage, Route

from .conftest import make_alert


def test_no_alerts_yields_good_service(line_1: Route, now: int) -> None:
    status = derive_route_status(line_1, [], now)
    assert status.label == "Good Service"
    assert status.primary_alert_type is None
    assert status.alerts == []


def test_active_alert_drives_status(line_1: Route, now: int) -> None:
    alert = make_alert(alert_type="Delays", route_id="1")
    status = derive_route_status(line_1, [alert], now)
    assert status.label == "Delays"
    assert status.primary_alert_type == "Delays"
    assert status.alerts == ["alert-1"]


def test_alert_outside_active_window_is_ignored(line_1: Route, now: int) -> None:
    alert = make_alert(start=1, end=2, route_id="1")
    status = derive_route_status(line_1, [alert], now)
    assert status.label == "Good Service"


def test_alert_for_other_route_is_ignored(line_1: Route, now: int) -> None:
    alert = make_alert(route_id="A")
    status = derive_route_status(line_1, [alert], now)
    assert status.label == "Good Service"


def test_highest_sort_order_wins(line_1: Route, now: int) -> None:
    minor = make_alert(id="m", alert_type="Slow Speeds", sort_order=16, route_id="1")
    major = make_alert(id="M", alert_type="Suspended", sort_order=39, route_id="1")
    status = derive_route_status(line_1, [minor, major], now)
    assert status.primary_alert_type == "Suspended"
    assert status.label == "Suspended"


def test_direction_filtering(line_1: Route, now: int) -> None:
    north = make_alert(id="n", route_id="1", direction_id=0, alert_type="Delays")
    south = make_alert(
        id="s", route_id="1", direction_id=1, alert_type="Service Change"
    )
    status = derive_route_status(line_1, [north, south], now)
    assert status.by_direction["northbound"].primary_alert_type == "Delays"
    assert status.by_direction["southbound"].primary_alert_type == "Service Change"


def test_alerts_for_route_filters_by_route_and_time(now: int) -> None:
    active = make_alert(id="a", route_id="1")
    other_route = make_alert(id="b", route_id="A")
    expired = make_alert(id="c", route_id="1", start=1, end=2)
    matching = alerts_for_route([active, other_route, expired], "1", now)
    assert [a.id for a in matching] == ["a"]


def test_compat_route_shape(line_1: Route, now: int) -> None:
    alert = make_alert(alert_type="Delays", route_id="1")
    status = derive_route_status(line_1, [alert], now)
    compat = derive_compat_route(line_1, status, [alert])
    assert compat.id == "1"
    assert compat.name == "1"
    assert compat.color == "#ee352e"
    assert compat.status == "Delays"
    assert compat.scheduled is True


def test_compat_route_falls_back_to_good_service(line_1: Route, now: int) -> None:
    status = derive_route_status(line_1, [], now)
    compat = derive_compat_route(line_1, status, [])
    assert compat.status == "Good Service"


def test_system_status_all_clear() -> None:
    system = derive_system_status([], [])
    assert system.overall_label == "All systems normal"
    assert system.accessibility.elevators_out == 0


def test_system_status_with_alerts_and_outage(line_1: Route, now: int) -> None:
    rs = derive_route_status(line_1, [make_alert(route_id="1")], now)
    elevator = Equipment(
        equipment_id="EL-1",
        type="elevator",
        ada_pathway=True,
        outage=EquipmentOutage(reason="Repair", since=now - 3600),
    )
    system = derive_system_status([rs], [elevator])
    assert "1" in system.by_mode["subway"].routes_with_alerts
    assert system.accessibility.elevators_out == 1
    assert system.accessibility.ada_pathways_degraded == 1
    assert "ADA pathway(s) degraded" in system.overall_label
