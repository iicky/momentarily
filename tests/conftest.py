"""Shared pytest fixtures."""

from __future__ import annotations

import io
import zipfile

import pytest

from momentarily.schema import (
    Alert,
    InformedEntity,
    Route,
    TimeRange,
    TranslatedString,
    TranslatedText,
)

FEED_VERSION = "TEST-20260807"


def make_gtfs_bytes(
    trips_rows: list[str],
    stop_times_rows: list[str],
    *,
    calendar_rows: list[str] | None = None,
    calendar_dates_rows: list[str] | None = None,
    feed_info_row: str | None = None,
) -> bytes:
    """A synthetic static-GTFS zip, so the parsers can be exercised without the
    5 MB network fetch. Rows are raw CSV lines in the column order the headers
    below declare."""
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
        zf.writestr(
            "calendar.txt",
            _csv(
                "service_id,monday,tuesday,wednesday,thursday,friday,saturday,"
                "sunday,start_date,end_date",
                calendar_rows
                or [
                    "Weekday,1,1,1,1,1,0,0,19700101,20991231",
                    "Saturday,0,0,0,0,0,1,0,19700101,20991231",
                ],
            ),
        )
        zf.writestr(
            "calendar_dates.txt",
            _csv("service_id,date,exception_type", calendar_dates_rows or []),
        )
        zf.writestr(
            "feed_info.txt",
            _csv(
                "feed_publisher_name,feed_lang,feed_start_date,feed_end_date,"
                "feed_version",
                [feed_info_row or f"Test,EN,20260101,20261231,{FEED_VERSION}"],
            ),
        )
    return buf.getvalue()


def make_gtfs_zip(
    trips_rows: list[str],
    stop_times_rows: list[str],
    *,
    calendar_rows: list[str] | None = None,
    calendar_dates_rows: list[str] | None = None,
    feed_info_row: str | None = None,
) -> zipfile.ZipFile:
    """make_gtfs_bytes, opened for the parsers that take a zip directly."""
    return zipfile.ZipFile(
        io.BytesIO(
            make_gtfs_bytes(
                trips_rows,
                stop_times_rows,
                calendar_rows=calendar_rows,
                calendar_dates_rows=calendar_dates_rows,
                feed_info_row=feed_info_row,
            )
        )
    )


def _csv(header: str, rows: list[str]) -> str:
    return "\n".join([header, *rows]) + "\n"


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
