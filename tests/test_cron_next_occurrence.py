"""Tests for scripts/cron_next_occurrence.py — next_cron_occurrence / parse_crons."""

from __future__ import annotations

import datetime

import pytest

from scripts.cron_next_occurrence import next_cron_occurrence, parse_crons


def _utc(iso: str) -> datetime.datetime:
    return datetime.datetime.fromisoformat(iso).replace(tzinfo=datetime.UTC)


def test_resume_to_first_sunday() -> None:
    # Production case: resumed 2026-09-08T13:15:48Z -> first slot 2026-09-13T05:00Z.
    assert next_cron_occurrence("0 5 * * SUN", _utc("2026-09-08T13:15:48")) == _utc(
        "2026-09-13T05:00:00"
    )


def test_unpause_exactly_on_slot_rolls_to_next_week() -> None:
    # Slot is not strictly after itself; must advance one week.
    assert next_cron_occurrence("0 5 * * SUN", _utc("2026-09-13T05:00:00")) == _utc(
        "2026-09-20T05:00:00"
    )


def test_dow_sun_zero_seven_equivalent() -> None:
    # SUN, 0, and 7 must all resolve to the same slot.
    after = _utc("2026-09-08T13:15:48")
    expected = _utc("2026-09-13T05:00:00")
    assert next_cron_occurrence("0 5 * * 0", after) == expected
    assert next_cron_occurrence("0 5 * * 7", after) == expected


def test_multiple_crons_returns_earliest() -> None:
    # Saturday 03:00 < Sunday 05:00; earliest must win.
    after = _utc("2026-09-08T13:15:48")
    slots = [next_cron_occurrence(c, after) for c in ["0 5 * * SUN", "0 3 * * SAT"]]
    assert min(slots) == _utc("2026-09-12T03:00:00")


def test_unsupported_field_raises() -> None:
    # Non-wildcard dom is unsupported and must raise, not silently misbehave.
    with pytest.raises(ValueError, match="day-of-month"):
        next_cron_occurrence("0 5 15 * SUN", _utc("2026-09-08T00:00:00"))


def test_parse_crons_extracts_expressions() -> None:
    toml = '[triggers]\ncrons = ["0 5 * * SUN", "0 3 * * SAT"]\n'
    assert parse_crons(toml) == ["0 5 * * SUN", "0 3 * * SAT"]


def test_parse_crons_missing_raises() -> None:
    with pytest.raises(RuntimeError):
        parse_crons("[triggers]\n")
