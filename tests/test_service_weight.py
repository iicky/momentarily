"""Unit tests for the scheduled-service baseline ingest
(training/service_weight.py).

Covers the pure pieces -- the after-midnight hour wrap, the stop_times
reduction to per-directional-platform hourly counts, representative-day
selection through the calendar (including a calendar_dates exception, so the
raw calendar.txt union is provably NOT what drives it), and the document
wrapper -- plus the R2 write via a fake client. No real network or R2.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from training.gtfs_static import Calendar, FeedVersion, Weekly
from training.service_weight import (
    BASELINE_KEY,
    SCHEMA_VERSION,
    VERSIONED_BASELINE_PREFIX,
    ReferenceDays,
    build_doc,
    hour_of,
    reduce_service_weights,
    select_reference_days,
    write_baseline,
)

_WD = frozenset({0, 1, 2, 3, 4})
_SAT = frozenset({5})
_SUN = frozenset({6})


def _st(trip_id: str, stop_id: str, departure_time: str) -> dict[str, str]:
    return {"trip_id": trip_id, "stop_id": stop_id, "departure_time": departure_time}


def test_hour_of_excludes_after_midnight_trips() -> None:
    # GTFS encodes an after-midnight departure on the prior service day as
    # 24:xx/25:xx. Those run on the NEXT wall-clock day, whose wd/we class can
    # differ (Friday 25:10 is Saturday; Sunday 25:10 is Monday), so a two-bin
    # artifact must drop them rather than wrap them into the wrong class.
    assert hour_of("08:14:00") == 8
    assert hour_of("25:10:00") is None
    assert hour_of("24:00:00") is None
    assert hour_of("23:59:59") == 23
    assert hour_of("") is None
    assert hour_of("abc") is None


def test_reduce_excludes_class_crossing_after_midnight_rows() -> None:
    # The two class-crossing cases the wrap bug corrupted: a Friday-night
    # weekday trip (runs Saturday 01:10, would have landed wd[1]) and a
    # Sunday-night weekend trip (runs Monday 01:10, would have landed we[1]).
    # Both are excluded outright; the stops keep their in-day counts.
    trip_service = {"wd1": "Weekday", "sun1": "Sunday"}
    stops = reduce_service_weights(
        [
            _st("wd1", "631N", "23:50:00"),
            _st("wd1", "631N", "25:10:00"),  # Friday-night spillover -> dropped
            _st("sun1", "631N", "25:10:00"),  # Sunday-night spillover -> dropped
            _st("sun1", "631N", "10:00:00"),
        ],
        trip_service,
        weekday_services=frozenset({"Weekday"}),
        weekend_services=frozenset({"Sunday"}),
    )
    assert stops["631N"]["wd"][23] == 1
    assert stops["631N"]["wd"][1] == 0
    assert stops["631N"]["we"][1] == 0
    assert stops["631N"]["we"][10] == 1
    assert sum(stops["631N"]["wd"]) == 1
    assert sum(stops["631N"]["we"]) == 1


def test_reduce_counts_departures_per_stop_hour_by_class() -> None:
    trip_service = {"wd1": "Weekday", "sat1": "Saturday", "other": "Holiday"}
    stop_times = [
        _st("wd1", "631N", "08:03:00"),
        _st("wd1", "631N", "08:59:00"),  # same stop, same hour -> counts twice
        _st("wd1", "901S", "08:10:00"),
        _st("sat1", "631N", "08:20:00"),  # weekend class only
        _st("other", "631N", "08:30:00"),  # neither class -> skipped
    ]
    stops = reduce_service_weights(
        stop_times,
        trip_service,
        weekday_services=frozenset({"Weekday"}),
        weekend_services=frozenset({"Saturday"}),
    )
    assert stops["631N"]["wd"][8] == 2
    assert stops["631N"]["we"][8] == 1
    assert stops["901S"]["wd"][8] == 1
    # A stop only ever seen at hour 8 still carries a full 24-slot pair, zeros
    # everywhere else -- a real 0, not a missing cell.
    assert len(stops["631N"]["wd"]) == 24
    assert stops["631N"]["wd"][9] == 0
    assert stops["901S"]["we"] == [0] * 24
    # The unclassified-service stop never appears.
    assert "other" not in stops


def test_reduce_skips_unparseable_times_without_dropping_the_stop() -> None:
    stops = reduce_service_weights(
        [
            _st("wd1", "A01N", "07:00:00"),
            _st("wd1", "A01N", ":::"),  # unparseable -> skipped, stop stays
        ],
        {"wd1": "Weekday"},
        weekday_services=frozenset({"Weekday"}),
        weekend_services=frozenset(),
    )
    assert stops["A01N"]["wd"][7] == 1
    assert sum(stops["A01N"]["wd"]) == 1


def _calendar(*, added: dict[str, frozenset[str]] | None = None) -> Calendar:
    return Calendar(
        weekly=(
            Weekly("Weekday", _WD, "20260101", "20261231"),
            Weekly("Saturday", _SAT, "20260101", "20261231"),
            Weekly("Sunday", _SUN, "20260101", "20261231"),
        ),
        added=added or {},
        removed={},
    )


def test_select_reference_days_picks_a_plain_day_of_each_type() -> None:
    feed = FeedVersion(version="v", start=date(2026, 6, 1), end=date(2026, 12, 31))
    ref = select_reference_days(feed, _calendar())
    assert ref == ReferenceDays(
        weekday=date(2026, 6, 1),  # a Monday
        saturday=date(2026, 6, 6),
        sunday=date(2026, 6, 7),
    )


def test_select_reference_days_skips_a_calendar_dates_exception_date() -> None:
    # A holiday adds Sunday service on Monday 2026-06-01 (calendar_dates), so
    # that date carries an exception. The weekday reference must move OFF it to
    # a plain weekday -- proof the choice runs through the calendar's exception
    # model, not a raw calendar.txt weekday-flag union.
    feed = FeedVersion(version="v", start=date(2026, 6, 1), end=date(2026, 12, 31))
    cal = _calendar(added={"20260601": frozenset({"Sunday"})})
    ref = select_reference_days(feed, cal)
    assert ref.weekday == date(2026, 6, 2)  # Tuesday, no exception
    # The resolved weekday services are the plain weekday set, never the
    # holiday's Sunday service.
    assert cal.active(ref.weekday) == frozenset({"Weekday"})


def test_build_doc_shape_is_pure_and_stamped() -> None:
    stops = {"631N": {"wd": [0] * 24, "we": [0] * 24}}
    doc = build_doc(
        stops,
        generated_at=1_787_000_000,
        url="https://example/gtfs.zip",
        feed_version="20260807",
        reference_weekday=date(2026, 6, 2),
        reference_saturday=date(2026, 6, 6),
        reference_sunday=date(2026, 6, 7),
        weekday_services=["Weekday"],
        weekend_services=["Sunday", "Saturday"],
    )
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["generated_at"] == 1_787_000_000
    assert doc["n_stops"] == 1
    assert doc["source"]["dataset"] == "gtfs_subway"
    assert doc["source"]["feed_version"] == "20260807"
    assert doc["source"]["reference_saturday"] == "2026-06-06"
    # Service lists are sorted for a stable diff.
    assert doc["source"]["weekend_services"] == ["Saturday", "Sunday"]
    assert doc["stops"] == stops


class _FakeS3:
    def __init__(self) -> None:
        self.puts: list[dict[str, Any]] = []

    def put_object(self, **kwargs: Any) -> None:
        self.puts.append(kwargs)


def test_write_baseline_writes_live_pointer_and_versioned_snapshot() -> None:
    client = _FakeS3()
    doc: dict[str, Any] = {"generated_at": 1_787_000_000, "stops": {}}
    versioned = write_baseline(client, "momentarily", doc)  # type: ignore[arg-type]
    assert versioned == f"{VERSIONED_BASELINE_PREFIX}v1787000000.json"
    keys = {p["Key"] for p in client.puts}
    assert keys == {BASELINE_KEY, versioned}
    # Both objects carry identical bytes, JSON of the same doc.
    assert all(json.dumps(doc).encode() == p["Body"] for p in client.puts)
    for p in client.puts:
        assert p["ContentType"] == "application/json"
