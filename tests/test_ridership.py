"""Unit tests for the ridership-baseline ingest (training/ridership.py).

Covers the pure reduction (rows -> per-complex baseline), the calendar-based
day-count denominator, the window-floor logic, and the truncation guard on
the aggregate fetch -- via a fake S3 client and hand-built Socrata rows, no
real network or R2.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, cast

import pytest

from training.ridership import (
    BASELINE_KEY,
    QUERY_LIMIT,
    SCHEMA_VERSION,
    VERSIONED_BASELINE_PREFIX,
    build_doc,
    fetch_hourly_rows,
    reduce_baseline,
    resolve_window,
    weekday_weekend_day_counts,
    write_baseline,
)


def _row(
    complex_id: str,
    hour: object,
    ridership: object,
    transfers: object,
    *,
    name: str = "Times Sq-42 St",
    borough: str = "Manhattan",
) -> dict[str, Any]:
    """One Socrata aggregate row, numeric fields as whatever type a caller
    wants to probe (str, like the real feed, or plain numbers)."""
    return {
        "station_complex_id": complex_id,
        "station_complex": name,
        "borough": borough,
        "hh": hour,
        "ridership": ridership,
        "transfers": transfers,
    }


def test_weekday_weekend_day_counts_mid_week_span() -> None:
    """A span starting Wed 2026-07-01 and ending (exclusive) Wed 2026-07-15
    covers 07-01..07-14: 10 weekdays, 4 weekend days -- neither boundary date
    is itself a weekend, so this isn't just exercising a whole-week multiple."""
    wd, we = weekday_weekend_day_counts(datetime(2026, 7, 1), datetime(2026, 7, 15))
    assert (wd, we) == (10, 4)


def test_weekday_weekend_day_counts_rejects_empty_span() -> None:
    with pytest.raises(ValueError, match="window_end must be after window_start"):
        weekday_weekend_day_counts(datetime(2026, 7, 1), datetime(2026, 7, 1))


def test_reduce_baseline_missing_hour_cells_get_zero_and_truthful_n_cells() -> None:
    """A complex present at only 2 of the 48 possible cells reads 0.0 at every
    other hour -- not a fabricated rate -- and n_cells reports 2, not 48."""
    weekday_rows = [
        _row("611", 8, 6000.0, 100.0),
        _row("611", 17, 9000.0, 150.0),
    ]
    complexes = reduce_baseline(weekday_rows, [], weekday_days=60, weekend_days=26)
    entry = complexes["611"]
    assert entry["n_cells"] == 2
    assert entry["entries_per_min"]["wd"][8] == round(6000.0 / 60 / 60, 3)
    assert entry["entries_per_min"]["wd"][17] == round(9000.0 / 60 / 60, 3)
    untouched_hours = [h for h in range(24) if h not in (8, 17)]
    assert all(entry["entries_per_min"]["wd"][h] == 0.0 for h in untouched_hours)
    assert entry["entries_per_min"]["we"] == [0.0] * 24


def test_reduce_baseline_rank_orders_by_entries_total_desc() -> None:
    """rank 1 goes to the complex with the larger summed entries_total, across
    both classes, not to whichever complex the rows happened to list first."""
    weekday_rows = [
        _row("100", 8, 1000.0, 0.0, name="Quiet St"),
        _row("611", 8, 50000.0, 0.0, name="Times Sq-42 St"),
    ]
    weekend_rows = [_row("100", 8, 500.0, 0.0, name="Quiet St")]
    complexes = reduce_baseline(
        weekday_rows, weekend_rows, weekday_days=60, weekend_days=26
    )
    assert complexes["611"]["rank"] == 1
    assert complexes["100"]["rank"] == 2
    assert complexes["100"]["entries_total"] == 1500.0


def test_reduce_baseline_coerces_stringified_socrata_numbers() -> None:
    """Socrata's aggregate endpoint stringifies every numeric column,
    including the extracted hour -- '8' and '1234.0' must parse the same as
    the equivalent real numbers, not raise or silently mis-key the array."""
    string_rows = [_row("611", "8", "1234.0", "56.0")]
    numeric_rows = [_row("611", 8, 1234.0, 56.0)]
    from_strings = reduce_baseline(string_rows, [], weekday_days=60, weekend_days=26)
    from_numbers = reduce_baseline(numeric_rows, [], weekday_days=60, weekend_days=26)
    assert from_strings == from_numbers
    assert from_strings["611"]["entries_per_min"]["wd"][8] == round(1234.0 / 60 / 60, 3)


def test_reduce_baseline_rejects_nonpositive_day_counts() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        reduce_baseline([], [], weekday_days=0, weekend_days=26)


def test_fetch_hourly_rows_raises_when_response_hits_query_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A response that reaches QUERY_LIMIT means Socrata's own cap silently
    truncated an aggregate that should have had every group -- returning it
    unflagged would quietly bias every cell drawn from it."""
    saturated = [_row("611", 8, 1.0, 0.0)] * QUERY_LIMIT

    def _fake_get_json(url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        return saturated

    monkeypatch.setattr("training.ridership._get_json", _fake_get_json)
    with pytest.raises(RuntimeError, match="would silently truncate"):
        fetch_hourly_rows(
            "https://example.test",
            window_start=datetime(2026, 1, 1),
            window_end=datetime(2026, 2, 1),
            weekend=False,
        )


def test_fetch_hourly_rows_passes_through_under_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A response one row short of the cap is real data, not truncation."""
    rows = [_row("611", 8, 1.0, 0.0)] * (QUERY_LIMIT - 1)

    def _fake_get_json(url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        return rows

    monkeypatch.setattr("training.ridership._get_json", _fake_get_json)
    result = fetch_hourly_rows(
        "https://example.test",
        window_start=datetime(2026, 1, 1),
        window_end=datetime(2026, 2, 1),
        weekend=True,
    )
    assert len(result) == QUERY_LIMIT - 1


def test_resolve_window_includes_the_full_final_day_when_latest_hour_is_23() -> None:
    """latest_hour at 23:00 means that whole day is complete -- window_end
    must land on the FOLLOWING midnight so the day's 24 hours are all
    included, not excluded by a naive same-day floor."""
    start, end = resolve_window(
        days=90, end=None, latest_hour="2026-07-28T23:00:00.000"
    )
    assert end == datetime(2026, 7, 29)
    assert start == datetime(2026, 7, 29) - timedelta(days=90)


def test_resolve_window_drops_a_partial_final_day() -> None:
    """latest_hour short of 23:00 means that day's later hours are missing --
    window_end must floor back to the START of that day so the partial day is
    excluded entirely rather than averaged in as if it were whole."""
    _, end = resolve_window(days=90, end=None, latest_hour="2026-07-28T14:00:00.000")
    assert end == datetime(2026, 7, 28)


def test_resolve_window_end_override_wins_over_latest_hour() -> None:
    start, end = resolve_window(
        days=14, end="2026-06-15", latest_hour="2026-07-28T23:00:00.000"
    )
    assert end == datetime(2026, 6, 15)
    assert start == datetime(2026, 6, 1)


def test_build_doc_records_source_and_complex_count() -> None:
    complexes = reduce_baseline(
        [_row("611", 8, 1234.0, 56.0)], [], weekday_days=60, weekend_days=26
    )
    doc = build_doc(
        complexes,
        generated_at=42,
        url="https://data.ny.gov/resource/5wq4-mkjj.json",
        window_start=datetime(2026, 5, 1),
        window_end=datetime(2026, 7, 29),
        latest_hour="2026-07-28T23:00:00.000",
        weekday_days=63,
        weekend_days=26,
    )
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["n_complexes"] == 1
    assert doc["source"]["window_start"] == "2026-05-01T00:00:00"
    assert doc["source"]["window_end"] == "2026-07-29T00:00:00"
    assert doc["source"]["weekday_days"] == 63
    assert doc["source"]["weekend_days"] == 26
    assert "provenance" in doc


class _FakeS3:
    """Minimal stand-in for the boto3 S3 client -- captures put_object calls."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_: object) -> None:
        self.objects[Key] = Body


def test_write_baseline_writes_live_and_versioned_keys() -> None:
    fake = _FakeS3()
    doc: dict[str, Any] = {"generated_at": 1_701_300_000, "complexes": {}}
    versioned_key = write_baseline(cast(Any, fake), "test-bucket", doc)
    assert versioned_key == f"{VERSIONED_BASELINE_PREFIX}v1701300000.json"
    assert set(fake.objects) == {BASELINE_KEY, versioned_key}
    assert fake.objects[BASELINE_KEY] == fake.objects[versioned_key]
