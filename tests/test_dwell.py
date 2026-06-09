"""Tests for the empirical dwell-quantile sidecar."""

from __future__ import annotations

from training.dwell import (
    MIN_SAMPLES_FOR_EMPIRICAL,
    compute_dwell_quantiles,
    compute_dwell_quantiles_by_alert,
)
from training.eval import TransitionRecord


def _tr(
    route: str,
    prev: str,
    dwell_sec: int,
    ts: int = 0,
    alert_type: str | None = None,
) -> TransitionRecord:
    return TransitionRecord(
        ts=ts,
        route=route,
        prev_state=prev,
        new_state="normal",
        regime_entered_at=ts - dwell_sec,
        exited_at=ts,
        dwell_sec=dwell_sec,
        alert_type_at_entry=alert_type,
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


def test_recovery_fractions() -> None:
    # 3 regimes recover in 5 min, 3 in 90 min.
    transitions = [_tr("A", "disrupted", 300) for _ in range(3)] + [
        _tr("A", "disrupted", 5400) for _ in range(3)
    ]
    cell = compute_dwell_quantiles(transitions)["A"]["disrupted"]
    assert cell["recover_by_30"] == 0.5  # only the 5-min regimes
    assert cell["recover_by_60"] == 0.5  # 90 min still out at 60
    assert cell["recover_by_120"] == 1.0  # all recovered by 120


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


def test_by_alert_segments_on_alert_type() -> None:
    transitions = [
        _tr("A", "disrupted", 300, alert_type="Delays") for _ in range(6)
    ] + [
        _tr("A", "disrupted", 3600, alert_type="Planned - Stops Skipped")
        for _ in range(6)
    ]
    out = compute_dwell_quantiles_by_alert(transitions)
    assert set(out["A"]["disrupted"].keys()) == {"Delays", "Planned - Stops Skipped"}
    # Cause conditioning separates the short delay regime from the long planned one.
    assert out["A"]["disrupted"]["Delays"]["median_sec"] == 300
    assert out["A"]["disrupted"]["Planned - Stops Skipped"]["median_sec"] == 3600


def test_by_alert_skips_null_alert_type() -> None:
    # Null-alert transitions belong to the (route, state) aggregate only.
    transitions = [_tr("A", "disrupted", 600, alert_type=None) for _ in range(8)]
    assert compute_dwell_quantiles_by_alert(transitions) == {}
    # ...but they still feed the aggregate.
    assert compute_dwell_quantiles(transitions)["A"]["disrupted"]["n"] == 8


def test_by_alert_thin_cause_cell_is_omitted() -> None:
    # 6 Delays (emitted) + 3 Planned (below floor, omitted).
    transitions = [
        _tr("A", "disrupted", 300, alert_type="Delays") for _ in range(6)
    ] + [
        _tr("A", "disrupted", 3600, alert_type="Planned - Stops Skipped")
        for _ in range(3)
    ]
    out = compute_dwell_quantiles_by_alert(transitions)
    assert set(out["A"]["disrupted"].keys()) == {"Delays"}
