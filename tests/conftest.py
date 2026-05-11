"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from momentarily.schema import (
    Alert,
    InformedEntity,
    Route,
    TimeRange,
    TranslatedString,
    TranslatedText,
)


@pytest.fixture
def now() -> int:
    """A deterministic 'now' for time-window math."""
    return 1_700_000_000


@pytest.fixture
def line_1() -> Route:
    return Route(
        id="1",
        mode="subway",
        short_name="1",
        long_name="Broadway-7 Avenue Local",
        color="#ee352e",
    )


@pytest.fixture
def line_a() -> Route:
    return Route(
        id="A",
        mode="subway",
        short_name="A",
        long_name="Eighth Avenue Express",
        color="#0039a6",
    )


def make_alert(
    *,
    id: str = "alert-1",
    alert_type: str = "Delays",
    route_id: str = "1",
    direction_id: int | None = None,
    sort_order: int = 22,
    start: int = 1_699_000_000,
    end: int = 1_799_000_000,
    header_en: str = "1 trains are delayed.",
) -> Alert:
    """Helper to build an Alert for tests."""
    return Alert(
        id=id,
        alert_type=alert_type,
        sort_order=sort_order,
        active_period=[TimeRange(start=start, end=end)],
        informed_entities=[
            InformedEntity(route_id=route_id, direction_id=direction_id)
        ],
        header_text=TranslatedString(
            translation=[TranslatedText(text=header_en, language="en")]
        ),
        source="subway",
    )
