"""Tests for the empirical dwell-quantile sidecar."""

from __future__ import annotations

from training.dwell import (
    MIN_SAMPLES_FOR_EMPIRICAL,
    compute_dwell_quantiles,
)
from training.eval import TransitionRecord


def _tr(route: str, prev: str, dwell_sec: int, ts: int = 0) -> TransitionRecord:
    return TransitionRecord(
        ts=ts,
        route=route,
        prev_state=prev,
        new_state="normal",
        regime_entered_at=ts - dwell_sec,
        exited_at=ts,
        dwell_sec=dwell_sec,
    )


def test_cell_below_floor_is_omitted() -> None:
    transitions = [
        _tr("A", "disrupted", 600) for _ in range(MIN_SAMPLES_FOR_EMPIRICAL - 1)
    ]
    out = compute_dwell_quantiles(transitions)
    assert out == {}, f"thin cell should be omitted, got {out}"


def test_cell_at_floor_is_emitted() -> None:
    transitions = [_tr("A", "disrupted", 600) for _ in range(MIN_SAMPLES_FOR_EMPIRICAL)]
    out = compute_dwell_quantiles(transitions)
    assert "A" in out
    cell = out["A"]["disrupted"]
    assert cell["n"] == MIN_SAMPLES_FOR_EMPIRICAL
    assert cell["median_sec"] == 600


def test_quantiles_are_per_route_per_state() -> None:
    transitions = (
        [_tr("A", "disrupted", 60 * (i + 1)) for i in range(10)]  # 1..10 min
        + [_tr("A", "suspended", 3600) for _ in range(8)]
        + [_tr("B", "disrupted", 60 * i) for i in range(1, 6)]  # 1..5 min
    )
    out = compute_dwell_quantiles(transitions)
    assert set(out.keys()) == {"A", "B"}
    assert set(out["A"].keys()) == {"disrupted", "suspended"}
    assert set(out["B"].keys()) == {"disrupted"}
    a_disrupted = out["A"]["disrupted"]
    assert a_disrupted["q25_sec"] < a_disrupted["median_sec"] < a_disrupted["q75_sec"]
    # B has exactly MIN_SAMPLES so it should appear; A=disrupted has 10 samples
    assert out["B"]["disrupted"]["n"] == 5


def test_min_samples_override() -> None:
    transitions = [_tr("A", "disrupted", 600) for _ in range(3)]
    assert compute_dwell_quantiles(transitions) == {}
    out = compute_dwell_quantiles(transitions, min_samples=2)
    assert out["A"]["disrupted"]["n"] == 3
