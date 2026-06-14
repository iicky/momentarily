"""Tests for the alert_type → coarse status mapping table."""

from __future__ import annotations

import pytest

from momentarily.mapping import (
    NO_ALERTS_FALLBACK,
    coarse_status,
    is_known_alert_type,
)


def test_none_maps_to_fallback() -> None:
    assert coarse_status(None) == NO_ALERTS_FALLBACK == "Good Service"


@pytest.mark.parametrize(
    ("alert_type", "expected"),
    [
        ("Delays", "Delays"),
        ("Severe Delays", "Delays"),
        ("Slow Speeds", "Delays"),
        ("Service Change", "Service Change"),
        ("Trains Rerouted", "Service Change"),
        ("Reroute", "Service Change"),
        ("Stations Skipped", "Service Change"),
        ("Stops Skipped", "Service Change"),
        ("Reduced Service", "Service Change"),
        ("Boarding Change", "Service Change"),
        ("Suspended", "Suspended"),
        ("No Trains", "Suspended"),
        ("Information", "Information"),
        ("Other", "Information"),
        ("Station Notice", "Information"),
        ("Special Schedule", "Information"),
    ],
)
def test_known_alert_types(alert_type: str, expected: str) -> None:
    assert coarse_status(alert_type) == expected


@pytest.mark.parametrize(
    "alert_type",
    [
        "Planned Work",
        "Planned – Service Change",  # MTA uses an en-dash in these alert_type strings
        "Planned – Multiple Changes",
        "Planned – Express to Local",
    ],
)
def test_planned_prefix_maps_to_planned_work(alert_type: str) -> None:
    assert coarse_status(alert_type) == "Planned Work"


def test_no_direction_service_maps_to_suspended() -> None:
    assert coarse_status("No Uptown Service") == "Suspended"
    assert coarse_status("No Manhattan-Bound Service") == "Suspended"


def test_unknown_alert_type_passes_through() -> None:
    """New MTA alert_type values should not be lost."""
    assert coarse_status("Brand New Mystery Type") == "Brand New Mystery Type"
    assert is_known_alert_type("Brand New Mystery Type") is False


def test_known_alert_type_helper() -> None:
    assert is_known_alert_type("Delays") is True
    assert is_known_alert_type("Planned Work") is True
    assert is_known_alert_type("No Uptown Service") is True
