"""Changepoint alignment against grid-filled MTA truth (training/review.py).

The truth dict only holds alert-bearing ticks; the matcher must walk the full
tick grid so alert-clear (recovery) changepoints exist. See momentarily-vk0.2.
"""

from __future__ import annotations

from training.eval import TransitionRecord
from training.review import changepoint_alignment

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
