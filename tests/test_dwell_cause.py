"""Tests for cause-conditioned dwell-quantile aggregation (training/dwell.py)."""

from __future__ import annotations

import pytest

from momentarily.mapping import category_for_label, coarse_status
from training.dwell import (
    MIN_SAMPLES_FOR_EMPIRICAL,
    cause_of,
    compute_dwell_quantiles_by_cause,
)
from training.eval import TransitionRecord

# Derived from the real mapping tables rather than hardcoded, so a future
# rename of the category vocabulary doesn't silently invalidate these tests.
DELAYS_CAUSE = category_for_label(coarse_status("Severe Delays"))
SUSPENDED_CAUSE = category_for_label(coarse_status("Suspended"))


def _tr(
    route: str,
    prev: str,
    dwell_sec: int,
    ts: int = 0,
    alert_type: str | None = None,
    new_state: str = "normal",
) -> TransitionRecord:
    return TransitionRecord(
        ts=ts,
        route=route,
        prev_state=prev,
        new_state=new_state,
        regime_entered_at=ts - dwell_sec,
        exited_at=ts,
        dwell_sec=dwell_sec,
        alert_type_at_entry=alert_type,
    )


def test_cause_of_none_alert_type_is_none() -> None:
    assert cause_of(None) is None


@pytest.mark.parametrize(
    "alert_type",
    [
        "Severe Delays",
        "Some Delays",
        "Suspended",
        "No Uptown Service",
        "Planned - Stops Skipped",
        "Brand New Mystery Type",
    ],
)
def test_cause_of_matches_category_for_label_of_coarse_status(alert_type: str) -> None:
    # cause_of is the episode grader's cause key, so it must track
    # category_for_label(coarse_status(...)) exactly across every branch of
    # coarse_status: an exact-match table entry, the "No X Service" prefix,
    # the "Planned" prefix, and the unknown-passthrough case.
    assert cause_of(alert_type) == category_for_label(coarse_status(alert_type))


def test_cause_groups_different_alert_types_under_one_category() -> None:
    # "Delays" and "Severe Delays" are distinct alert_type strings that both
    # fall in the delays-family cause category; individually each is below
    # the 5-sample floor, but grouped by cause they clear it. This is exactly
    # what distinguishes cause grouping from compute_dwell_quantiles_by_alert's
    # raw-alert_type grouping.
    transitions = [
        _tr("A", "disrupted", 300, alert_type="Delays") for _ in range(3)
    ] + [_tr("A", "disrupted", 600, alert_type="Severe Delays") for _ in range(3)]
    out = compute_dwell_quantiles_by_cause(transitions)
    assert set(out["A"]["disrupted"].keys()) == {DELAYS_CAUSE}
    assert out["A"]["disrupted"][DELAYS_CAUSE]["n"] == 6


def test_cause_below_floor_is_dropped_while_other_cause_is_kept() -> None:
    # 6 Delays (at the floor, emitted) + 3 Suspended (below floor, omitted).
    transitions = [
        _tr("A", "disrupted", 300, alert_type="Delays") for _ in range(6)
    ] + [_tr("A", "disrupted", 900, alert_type="Suspended") for _ in range(3)]
    out = compute_dwell_quantiles_by_cause(transitions)
    assert set(out["A"]["disrupted"].keys()) == {DELAYS_CAUSE}


def test_transitions_with_no_alert_type_are_skipped() -> None:
    transitions = [
        _tr("A", "disrupted", 300, alert_type="Delays")
        for _ in range(MIN_SAMPLES_FOR_EMPIRICAL)
    ] + [_tr("A", "disrupted", 300, alert_type=None) for _ in range(4)]
    out = compute_dwell_quantiles_by_cause(transitions)
    # If the null-alert transitions leaked into the cause cell, n would read
    # MIN_SAMPLES_FOR_EMPIRICAL + 4 instead.
    assert out["A"]["disrupted"][DELAYS_CAUSE]["n"] == MIN_SAMPLES_FOR_EMPIRICAL


def test_cause_cells_are_isolated_per_route_and_state() -> None:
    transitions = [
        _tr("A", "disrupted", 300, alert_type="Delays")
        for _ in range(MIN_SAMPLES_FOR_EMPIRICAL)
    ] + [
        _tr("B", "suspended", 900, alert_type="Suspended")
        for _ in range(MIN_SAMPLES_FOR_EMPIRICAL)
    ]
    out = compute_dwell_quantiles_by_cause(transitions)
    assert set(out.keys()) == {"A", "B"}
    assert set(out["A"]["disrupted"].keys()) == {DELAYS_CAUSE}
    assert set(out["B"]["suspended"].keys()) == {SUSPENDED_CAUSE}


def test_min_samples_override_lowers_the_floor() -> None:
    transitions = [_tr("A", "disrupted", 300, alert_type="Suspended") for _ in range(2)]
    assert compute_dwell_quantiles_by_cause(transitions) == {}
    out = compute_dwell_quantiles_by_cause(transitions, min_samples=2)
    assert out["A"]["disrupted"][SUSPENDED_CAUSE]["n"] == 2
