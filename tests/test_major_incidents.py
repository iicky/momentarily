"""Ingest + join logic for the MTA Major Incidents source: line mapping (the
JZ fan-out and the three named shuttles), row normalization and null handling,
the 2023->2024 methodology-era split, and the month/coverage classification
that decides which signals can speak to which incident-month.

Everything here is hermetic — no network, no R2. The archive-prevalence read
lives behind main() and is not exercised.
"""

from __future__ import annotations

from datetime import date

import pytest

from training.divisions import ALL_KNOWN_ROUTES
from training.major_incidents import (
    AMBIGUOUS_LINES,
    METHODOLOGY_BREAK,
    SIX_CATEGORIES,
    SignalBounds,
    classify_coverage,
    era_for,
    join,
    map_line,
    month_span,
    parse_rows,
)

# --- line mapping ------------------------------------------------------------


def test_jz_fans_out_to_both_routes():
    """The MTA counts the Nassau St skip-stop pair as one line; it maps to BOTH
    J and Z, and is flagged ambiguous so the report can say a per-route join
    double-counts it."""
    assert map_line("JZ") == ("J", "Z")
    assert "JZ" in AMBIGUOUS_LINES


def test_named_shuttles_map_to_their_route_ids():
    assert map_line("S 42nd") == ("GS",)
    assert map_line("S Rock") == ("H",)
    assert map_line("S Fkln") == ("FS",)


def test_plain_lines_are_identity_when_known():
    for line in ("1", "7", "A", "L", "Q"):
        assert map_line(line) == (line,)


def test_unmappable_and_null_lines_yield_empty():
    assert map_line(None) == ()
    assert map_line("") == ()
    assert map_line("   ") == ()
    assert map_line("99") == ()  # a line we do not model -> dropped, not crashed


def test_si_maps_but_dataset_never_sends_it():
    """SI (Staten Island Railway) is a known route id, so map_line resolves it;
    the Major Incidents dataset simply never carries an SI row."""
    assert map_line("SI") == ("SI",)


def test_every_mapped_route_is_a_known_route_id():
    """The whole point of raising in map_line: a source line can never smuggle in
    a route id the trainer does not model."""
    for line in (
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "JZ",
        "L",
        "M",
        "N",
        "Q",
        "R",
        "W",
        "S 42nd",
        "S Rock",
        "S Fkln",
    ):
        for route in map_line(line):
            assert route in ALL_KNOWN_ROUTES


def test_mapped_route_absent_from_known_space_raises():
    from training import major_incidents as mi

    saved = dict(mi.LINE_TO_ROUTES)
    mi.LINE_TO_ROUTES["ZZ"] = ("NOPE",)
    try:
        with pytest.raises(ValueError, match="unknown route ids"):
            map_line("ZZ")
    finally:
        mi.LINE_TO_ROUTES.clear()
        mi.LINE_TO_ROUTES.update(saved)


# --- row parsing -------------------------------------------------------------


def _row(**kw: object) -> dict[str, object]:
    base: dict[str, object] = {
        "month": "2026-07-01T00:00:00.000",
        "division": "A DIVISION",
        "line": "2",
        "day_type": "1",
        "category": "Signals",
        "count": "2",
    }
    base.update(kw)
    return base


def test_parse_normalizes_a_row():
    (inc,) = parse_rows([_row()])
    assert inc.month == date(2026, 7, 1)
    assert inc.routes == ("2",)
    assert inc.day_type == "weekday"
    assert inc.category == "Signals"
    assert inc.count == 2
    assert inc.era == "post_2024"
    assert inc.ambiguous_line is False


def test_parse_maps_day_type_2_to_weekend():
    (inc,) = parse_rows([_row(day_type="2")])
    assert inc.day_type == "weekend"


def test_parse_skips_null_rows():
    """The dataset carries an occasional all-null row; it must drop out silently
    rather than crash the ingest."""
    null_row: dict[str, object] = {
        "month": None,
        "line": None,
        "count": None,
        "category": None,
    }
    assert len(parse_rows([_row(), null_row])) == 1


def test_parse_skips_unmodeled_line_but_keeps_count_int():
    rows = [_row(line="99"), _row(line="1", count="5")]
    parsed = parse_rows(rows)
    assert len(parsed) == 1
    assert parsed[0].routes == ("1",)
    assert parsed[0].count == 5


def test_categories_constant_matches_the_documented_six():
    assert len(SIX_CATEGORIES) == 6
    assert "Persons on Trackbed/Police/Medical" in SIX_CATEGORIES


# --- methodology era ---------------------------------------------------------


def test_era_split_at_the_2024_boundary():
    assert era_for(date(2023, 12, 1)) == "pre_2024"
    assert era_for(METHODOLOGY_BREAK) == "post_2024"
    assert era_for(date(2026, 7, 1)) == "post_2024"


# --- coverage classification -------------------------------------------------


def test_month_span_handles_month_lengths():
    assert month_span(date(2026, 2, 1)) == (date(2026, 2, 1), date(2026, 2, 28))
    assert month_span(date(2026, 7, 1)) == (date(2026, 7, 1), date(2026, 7, 31))


def test_full_partial_none_coverage():
    july = date(2026, 7, 1)
    # spans all of July -> full
    full = SignalBounds("alerts", date(2026, 6, 3), date(2026, 9, 2))
    assert classify_coverage(july, full) == "full"
    # starts mid-June, so June is partial but July is still full
    supply = SignalBounds("supply", date(2026, 6, 15), date(2026, 9, 2))
    assert classify_coverage(july, supply) == "full"
    assert classify_coverage(date(2026, 6, 1), supply) == "partial"
    # trace starts after July ends -> no overlap with any <=July month
    trace = SignalBounds("movement", date(2026, 8, 12), date(2026, 9, 2))
    assert classify_coverage(july, trace) == "none"
    assert classify_coverage(date(2026, 6, 1), trace) == "none"


def test_boundary_day_counts_as_overlap_not_none():
    """A signal whose last archived day is the month's first day still overlaps
    that month (partial), never 'none'."""
    b = SignalBounds("x", date(2026, 5, 1), date(2026, 6, 1))
    assert classify_coverage(date(2026, 6, 1), b) == "partial"
    b2 = SignalBounds("x", date(2026, 5, 1), date(2026, 5, 31))
    assert classify_coverage(date(2026, 6, 1), b2) == "none"


# --- join --------------------------------------------------------------------


def test_join_aggregates_by_route_and_category_and_classifies_coverage():
    rows = parse_rows(
        [
            _row(line="2", category="Signals", count="2"),
            _row(line="2", category="Track", count="1"),
            _row(line="JZ", category="Signals", count="1"),
        ]
    )
    bounds = [
        SignalBounds("alerts", date(2026, 6, 3), date(2026, 9, 2)),
        SignalBounds("movement", date(2026, 8, 12), date(2026, 9, 2)),
    ]
    (jm,) = join(rows, bounds)
    assert jm.month == date(2026, 7, 1)
    # JZ fans to both J and Z -> each gets the 1
    assert jm.mta_by_route["J"] == 1
    assert jm.mta_by_route["Z"] == 1
    assert jm.mta_by_route["2"] == 3  # 2 signals + 1 track
    assert jm.mta_by_category == {"Signals": 3, "Track": 1}
    assert jm.coverage == {"alerts": "full", "movement": "none"}


def test_join_movement_has_zero_overlap_with_july():
    """The load-bearing negative result, asserted directly: no incident-month
    the movement/headway trace covers."""
    rows = parse_rows([_row(month="2026-06-01T00:00:00.000"), _row()])
    trace = [SignalBounds("movement", date(2026, 8, 12), date(2026, 9, 2))]
    joined = join(rows, trace)
    assert all(jm.coverage["movement"] == "none" for jm in joined)
