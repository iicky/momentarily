"""Tests for the derivation logic (alerts → RouteStatus, SystemStatus, etc.)."""

from __future__ import annotations

from momentarily.derive import (
    alerts_for_route,
    derive_compat_route,
    derive_route_status,
    derive_system_status,
)
from momentarily.schema import (
    Alert,
    Equipment,
    EquipmentOutage,
    InformedEntity,
    Route,
    TimeRange,
    TranslatedString,
    TranslatedText,
)

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
    compat = derive_compat_route(line_1, status, [alert], now)
    assert compat.id == "1"
    assert compat.name == "1"
    assert compat.color == "#ee352e"
    assert compat.status == "Delays"
    assert compat.scheduled is True


def test_compat_route_falls_back_to_good_service(line_1: Route, now: int) -> None:
    status = derive_route_status(line_1, [], now)
    compat = derive_compat_route(line_1, status, [], now)
    assert compat.status == "Good Service"


def test_system_status_all_clear() -> None:
    system = derive_system_status([], [], [])
    assert system.overall_label == "All systems normal"
    assert system.accessibility.elevators_out == 0


def test_system_status_with_alerts_and_outage(line_1: Route, now: int) -> None:
    alert = make_alert(route_id="1")
    rs = derive_route_status(line_1, [alert], now)
    elevator = Equipment(
        equipment_id="EL-1",
        type="elevator",
        ada_pathway=True,
        outage=EquipmentOutage(reason="Repair", since=now - 3600),
    )
    system = derive_system_status([rs], [alert], [elevator])
    assert "1" in system.by_mode["subway"].routes_with_alerts
    assert system.accessibility.elevators_out == 1
    assert system.accessibility.ada_pathways_degraded == 1
    assert "ADA pathway(s) degraded" in system.overall_label


def test_direction_split_does_not_leak_across_routes(
    line_1: Route, line_a: Route, now: int
) -> None:
    """An alert mentioning route 1 northbound AND route A southbound should
    tag route 1 only in northbound — not in southbound just because the
    alert has direction_id=1 on a different route's entity."""
    alert = Alert(
        id="shared",
        alert_type="Delays",
        sort_order=22,
        active_period=[TimeRange(start=1_699_000_000, end=1_799_000_000)],
        informed_entities=[
            InformedEntity(route_id="1", direction_id=0),
            InformedEntity(route_id="A", direction_id=1),
        ],
        header_text=TranslatedString(
            translation=[TranslatedText(text="shared", language="en")]
        ),
        source="subway",
    )

    status_1 = derive_route_status(line_1, [alert], now)
    assert status_1.by_direction["northbound"].primary_alert_type == "Delays"
    assert status_1.by_direction["southbound"].primary_alert_type is None

    status_a = derive_route_status(line_a, [alert], now)
    assert status_a.by_direction["southbound"].primary_alert_type == "Delays"
    assert status_a.by_direction["northbound"].primary_alert_type is None


def test_compat_summaries_filtered_per_direction(line_1: Route, now: int) -> None:
    """Compat delay/irreg summaries must come from route- AND direction-filtered
    alerts, not from list position in the raw alert list."""
    north_delay = make_alert(
        id="n",
        route_id="1",
        direction_id=0,
        alert_type="Delays",
        header_en="Northbound 1 trains are delayed.",
    )
    south_irreg = make_alert(
        id="s",
        route_id="1",
        direction_id=1,
        alert_type="Trains Rerouted",
        header_en="Southbound 1 trains rerouted.",
    )
    status = derive_route_status(line_1, [north_delay, south_irreg], now)
    compat = derive_compat_route(line_1, status, [north_delay, south_irreg], now)

    assert compat.delay_summaries is not None
    assert compat.delay_summaries.north == "Northbound 1 trains are delayed."
    assert compat.delay_summaries.south is None

    assert compat.service_irregularity_summaries is not None
    assert compat.service_irregularity_summaries.north is None
    assert (
        compat.service_irregularity_summaries.south == "Southbound 1 trains rerouted."
    )


def test_compat_ignores_other_routes_alerts(line_1: Route, now: int) -> None:
    """A route-A delay must not appear in route 1's compat summaries."""
    a_delay = make_alert(
        id="a",
        route_id="A",
        alert_type="Delays",
        header_en="A trains are delayed.",
    )
    status = derive_route_status(line_1, [a_delay], now)
    compat = derive_compat_route(line_1, status, [a_delay], now)
    assert compat.status == "Good Service"
    assert compat.delay_summaries is not None
    assert compat.delay_summaries.north is None
    assert compat.delay_summaries.south is None


def test_system_severity_max_populated(line_1: Route, now: int) -> None:
    """severity_max should reflect the max sort_order across subway alerts."""
    minor = make_alert(id="m", alert_type="Delays", sort_order=16, route_id="1")
    major = make_alert(id="M", alert_type="Suspended", sort_order=39, route_id="1")
    rs = derive_route_status(line_1, [minor, major], now)
    system = derive_system_status([rs], [minor, major], [])
    assert system.by_mode["subway"].severity_max == 39


def test_system_severity_max_zero_when_no_alerts() -> None:
    system = derive_system_status([], [], [])
    assert system.by_mode["subway"].severity_max == 0
