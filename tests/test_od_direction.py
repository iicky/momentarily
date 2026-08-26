"""Unit tests for the OD direction-demand diagnostic (training/od_direction.py).

Covers the pure reduction only -- aggregate rows to per-cell north share and
per-cell demand, None for empty cells, and the busiest-weekday-hour selection
that keeps the headline off thin overnight cells. No network.
"""

from __future__ import annotations

from typing import Any

import pytest

from training.od_direction import reduce_origin


def _approx(expected: float, abs_: float | None = None) -> object:
    """Typed wrapper around ``pytest.approx`` (pins the Unknown return type),
    mirroring the convention in the rest of the suite."""
    return pytest.approx(expected, abs=abs_)  # pyright: ignore[reportUnknownMemberType]


def _row(cls: str, hh: str, direction: str, r: float) -> dict[str, Any]:
    return {"cls": cls, "hh": hh, "dir": direction, "r": r}


def test_north_share_is_demand_weighted_per_cell() -> None:
    rows = [
        _row("wd", "8", "N", 30.0),
        _row("wd", "8", "S", 70.0),
        _row("we", "8", "N", 50.0),
        _row("we", "8", "S", 50.0),
    ]
    origin = reduce_origin("610", rows)
    assert origin.north_share["wd"][8] == _approx(0.30)
    assert origin.north_share["we"][8] == _approx(0.50)
    assert origin.demand["wd"][8] == _approx(100.0)
    assert origin.total_ridership == _approx(200.0)


def test_empty_cell_is_none_not_a_fabricated_half() -> None:
    # An hour with no demand either way must read None, never 0.5 -- the split
    # would otherwise treat "no data" as "perfectly balanced".
    origin = reduce_origin("610", [_row("wd", "8", "N", 10.0)])
    assert origin.north_share["wd"][8] == _approx(1.0)  # all northbound
    assert origin.north_share["wd"][9] is None  # untouched hour
    assert origin.north_share["we"][8] is None
    assert origin.demand["wd"][9] == 0.0


def test_rush_reads_the_busiest_hour_not_a_thin_overnight_cell() -> None:
    # A tiny 3am cell is wildly one-directional; the busy 8am cell is the real
    # peak. The rush metrics must follow demand, not the extreme thin cell.
    rows = [
        _row("wd", "3", "S", 2.0),  # 0% north, but only 2 riders
        _row("wd", "8", "N", 800.0),  # the real peak
        _row("wd", "8", "S", 200.0),
    ]
    origin = reduce_origin("1", rows)
    assert origin.busiest_weekday_hour() == 8
    assert origin.rush_north_share() == _approx(0.80)
    assert origin.rush_bias() == _approx(0.30)


def test_rush_metrics_are_defined_away_when_there_is_no_weekday_demand() -> None:
    origin = reduce_origin("x", [_row("we", "10", "N", 5.0)])
    assert origin.busiest_weekday_hour() is None
    assert origin.rush_north_share() is None
    assert origin.rush_bias() == 0.0


def test_unexpected_class_hour_and_direction_are_rejected() -> None:
    with pytest.raises(ValueError, match="unexpected class"):
        reduce_origin("x", [_row("holiday", "8", "N", 1.0)])
    with pytest.raises(ValueError, match="out of range"):
        reduce_origin("x", [_row("wd", "27", "N", 1.0)])
    with pytest.raises(ValueError, match="unexpected direction"):
        reduce_origin("x", [_row("wd", "8", "n", 1.0)])
