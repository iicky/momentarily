"""Changepoint alignment against grid-filled MTA truth (training/review.py).

The truth dict only holds alert-bearing ticks; the matcher must walk the full
tick grid so alert-clear (recovery) changepoints exist. See momentarily-vk0.2.
"""

from __future__ import annotations

import pytest

from momentarily.hmm import Observation
from momentarily.mapping import CANONICAL_SEVERITY_FLOOR
from training.eval import TransitionRecord
from training.load import TickObservation
from training.review import (
    changepoint_alignment,
    derive_graded_mta_state,
    mta_truth,
)

TICK = 300
T0 = 1_700_000_100  # tick-aligned


def _transition(
    exited_at: int,
    route: str = "1",
    prev: str = "disrupted",
    new: str = "normal",
) -> TransitionRecord:
    return TransitionRecord(
        ts=exited_at,
        route=route,
        prev_state=prev,
        new_state=new,
        regime_entered_at=T0,
        exited_at=exited_at,
        dwell_sec=exited_at - T0,
    )


def test_recovery_changepoint_is_visible():
    # Alerts active T0..T0+25min, absent after. MTA changepoints: T0
    # (normal→disrupted) and T0+30min (first alert-free tick → normal). An HMM
    # exit 2 minutes after the recovery must match it — before the grid fill,
    # recoveries never appeared in the truth series and this returned None.
    truth = {("1", T0 + i * TICK): "disrupted" for i in range(6)}
    deltas = changepoint_alignment(
        [_transition(T0 + 6 * TICK + 120)],
        truth,
        window_start=T0 - 2 * TICK,
        window_end=T0 + 24 * TICK,
    )
    assert deltas == [-2.0]


def test_unmatched_when_no_nearby_change():
    truth = {("1", T0): "disrupted"}
    deltas = changepoint_alignment(
        [_transition(T0 + 7200)],
        truth,
        window_start=T0,
        window_end=T0 + 7200,
    )
    assert deltas == [None]


def test_route_without_truth_entries_is_unmatched():
    truth = {("1", T0): "disrupted"}
    deltas = changepoint_alignment(
        [_transition(T0 + TICK, route="Q")],
        truth,
        window_start=T0,
        window_end=T0 + 7200,
    )
    assert deltas == [None]


# ---------------------------------------------------------------------------
# Severity-graded MTA truth (momentarily-zl6)
# ---------------------------------------------------------------------------


def test_graded_state_demotes_minor_alerts_at_severe_floor():
    """Ordinary delays / routine reroutes (tier 1) read normal at floor 2, so the
    HMM filtering them is scored as correct rather than a miss."""
    assert derive_graded_mta_state(("Delays",), floor=2) == "normal"
    assert derive_graded_mta_state(("Trains Rerouted",), floor=2) == "normal"
    # At floor 1 the same tick counts as disrupted — the breadth truth.
    assert derive_graded_mta_state(("Delays",), floor=1) == "disrupted"


def test_graded_state_keeps_severe_and_suspension():
    assert derive_graded_mta_state(("Severe Delays",), floor=2) == "disrupted"
    assert derive_graded_mta_state(("Suspended",), floor=2) == "suspended"
    # A severe alert alongside minor ones still grades disrupted.
    assert derive_graded_mta_state(("Delays", "Severe Delays"), floor=2) == "disrupted"
    # Suspension outranks everything regardless of floor.
    assert derive_graded_mta_state(("Suspended", "Delays"), floor=5) == "suspended"


def test_graded_state_no_alerts_is_normal():
    assert derive_graded_mta_state((), floor=2) == "normal"
    assert derive_graded_mta_state(("Planned - Service Change",), floor=2) == "normal"


def _tick_obs(route: str, tick: int, types: tuple[str, ...]) -> TickObservation:
    return TickObservation(
        route_id=route,
        tick=tick,
        observation=Observation(
            alert_count=len(types),
            severity_sum=0,
            has_suspended_alert=any("Suspend" in t or "No Trains" in t for t in types),
            has_delays=any("Delays" in t for t in types),
            has_service_change=any(
                "Rerouted" in t or "Service Change" in t for t in types
            ),
        ),
        disruptive_types=types,
    )


def test_mta_truth_floor_tightens_truth():
    """The graded floor reclassifies a minor-delays route-tick from disrupted to
    normal while leaving a severe one disrupted."""
    obs = [
        _tick_obs("1", T0, ("Delays",)),
        _tick_obs("A", T0, ("Severe Delays",)),
    ]
    broad = mta_truth(obs, severity_floor=1)
    graded = mta_truth(obs, severity_floor=2)
    assert broad[("1", T0)] == "disrupted"
    assert graded[("1", T0)] == "normal"
    assert broad[("A", T0)] == graded[("A", T0)] == "disrupted"


# ---------------------------------------------------------------------------
# Canonical default severity floor: mta_truth(obs) with no severity_floor arg
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("types", "expected"),
    [
        (("Delays",), "normal"),
        (("Trains Rerouted",), "normal"),
        (("Severe Delays",), "disrupted"),
        (("Suspended",), "suspended"),
        (("Planned - Service Change",), "normal"),
    ],
)
def test_mta_truth_default_uses_canonical_severe_only_floor(
    types: tuple[str, ...], expected: str
) -> None:
    """With no severity_floor argument, mta_truth grades by the severe-only
    canonical floor: a minor delay or routine reroute reads normal, and only
    Severe Delays / a suspension registers as disrupted / suspended. Planned
    work stays normal too, so it never masquerades as a stochastic episode."""
    obs = [_tick_obs("1", T0, types)]
    assert mta_truth(obs)[("1", T0)] == expected


def test_mta_truth_default_matches_explicit_canonical_floor():
    """Omitting severity_floor must be identical to passing
    severity_floor=CANONICAL_SEVERITY_FLOOR explicitly — the default is an
    alias for the canonical floor, not a separate code path that could drift
    from it."""
    obs = [
        _tick_obs("1", T0, ("Delays",)),
        _tick_obs("A", T0, ("Severe Delays",)),
        _tick_obs("Q", T0, ("Suspended",)),
        _tick_obs("N", T0, ("Planned - Service Change",)),
    ]
    assert mta_truth(obs) == mta_truth(obs, severity_floor=CANONICAL_SEVERITY_FLOOR)


def test_mta_truth_explicit_floor_1_still_recovers_breadth_truth():
    """Even though breadth is no longer the default, explicitly passing
    severity_floor=1 must still reproduce the legacy breadth truth (any
    delay counts as disrupted) — the sensitivity path stays reachable."""
    obs = [_tick_obs("1", T0, ("Delays",))]
    assert mta_truth(obs, severity_floor=1)[("1", T0)] == "disrupted"
