"""Independent recovery validation from the trip-updates service metric.

Synthetic series — no R2. Covers the load_r2 derivation (build_service_series,
compute_baseline, derive_actual_recovery) and the eval grading
(independent_recovery_metrics). See momentarily-xum / up0.
"""

from __future__ import annotations

from momentarily.hmm import tod_bin
from training.eval import PredictionRecord, independent_recovery_metrics
from training.load_r2 import (
    Disruption,
    build_service_series,
    compute_baseline,
    derive_actual_recovery,
)

TICK = 300
T0 = 1_700_000_100  # tick-aligned


def test_build_service_series_maps_assigned_n():
    bodies = [
        {
            "observed_at": T0,
            "fresh_feeds": ["ace"],
            "rows": {"A": {"assigned_n": 12, "trips_n": 15}, "C": {"assigned_n": 8}},
        },
        {"observed_at": T0 + TICK, "rows": {"A": {"assigned_n": 0}}},
    ]
    series = build_service_series(bodies)
    assert series[("A", T0)] == 12
    assert series[("C", T0)] == 8
    assert series[("A", T0 + TICK)] == 0


def test_compute_baseline_medians_and_min_samples():
    series: dict[tuple[str, int], int] = {}
    for i in range(8):  # route A: many samples of 10
        series[("A", T0 + i * TICK)] = 10
    series[("B", T0)] = 5  # route B: a single sample
    base = compute_baseline(series, min_samples=3)
    a_cells = [v for (r, _b), v in base.items() if r == "A"]
    assert a_cells
    assert all(v == 10 for v in a_cells)
    # Below the sample floor → omitted (caller treats as "can't judge").
    assert not any(r == "B" for (r, _b) in base)


def _series(route: str, vals: list[int]) -> dict[tuple[str, int], int]:
    return {(route, T0 + i * TICK): v for i, v in enumerate(vals)}


def _flat_baseline(
    series: dict[tuple[str, int], int], level: float
) -> dict[tuple[str, int], float]:
    return {(r, tod_bin(t)): level for (r, t) in series}


def test_derive_actual_recovery_detects_one_disruption():
    # normal(10)x3, degraded(2)x4, recovered(10)x5
    series = _series("A", [10, 10, 10, 2, 2, 2, 2, 10, 10, 10, 10, 10])
    baseline = _flat_baseline(series, 10.0)
    out = derive_actual_recovery(
        series, baseline, degrade_ratio=0.5, recover_ratio=0.8, debounce=2
    )
    assert len(out) == 1
    d = out[0]
    assert d.route == "A"
    assert d.start_tick == T0 + 3 * TICK  # first degraded tick
    assert d.recovered_tick == T0 + 7 * TICK  # first recovered tick


def test_derive_censors_an_ongoing_disruption():
    series = _series("A", [10, 10, 2, 2, 2, 2])  # dips and never recovers
    baseline = _flat_baseline(series, 10.0)
    assert derive_actual_recovery(series, baseline) == []


def test_derive_ignores_a_single_tick_dip_via_debounce():
    # One isolated low tick must not start a disruption (debounce=2).
    series = _series("A", [10, 10, 2, 10, 10, 10])
    baseline = _flat_baseline(series, 10.0)
    assert derive_actual_recovery(series, baseline, debounce=2) == []


def _pred(
    *,
    ts: int,
    route: str = "A",
    condition: str = "suspended",
    recovery_minutes: int = 30,
    low: int = 15,
    high: int = 60,
    source: str | None = None,
    indeterminate: bool = False,
) -> PredictionRecord:
    return PredictionRecord(
        ts=ts,
        route=route,
        condition=condition,
        regime_entered_at=ts,
        p_normal=0.1,
        p_disrupted=0.8,
        p_suspended=0.1,
        p_normal_in_30min=0.0,
        p_normal_in_60min=0.0,
        p_normal_in_120min=0.0,
        recovery_minutes=recovery_minutes,
        recovery_minutes_low=low,
        recovery_minutes_high=high,
        recovery_indeterminate=indeterminate,
        recovery_source=source,
    )


def test_independent_recovery_grades_against_the_covering_disruption():
    # Disruption [T0, T0+1800): a prediction at T0 saying "30 min" is exactly
    # right (1800s = 30 min remaining) and inside its [15,60] band.
    d = Disruption(route="A", start_tick=T0, recovered_tick=T0 + 1800)
    res = independent_recovery_metrics([_pred(ts=T0, recovery_minutes=30)], [d])
    assert res.overall.n == 1
    assert res.overall.mae_min == 0
    assert res.overall.iqr_coverage == 1.0


def test_independent_recovery_applies_the_standard_exclusions():
    d = Disruption(route="A", start_tick=T0, recovered_tick=T0 + 1800)
    preds = [
        _pred(ts=T0, condition="normal"),  # skipped: not recovering
        _pred(ts=T0, source="schedule"),  # excluded: deterministic resume
        _pred(ts=T0, indeterminate=True),  # skipped: clamped, not predicted
        _pred(ts=T0 + 300, recovery_minutes=25),  # graded: (1800-300)/60 = 25
    ]
    res = independent_recovery_metrics(preds, [d])
    assert res.overall.n == 1
    assert res.overall.mae_min == 0
    assert res.excluded_schedule == 1


def test_independent_recovery_skips_predictions_outside_any_disruption():
    d = Disruption(route="A", start_tick=T0, recovered_tick=T0 + 1800)
    # Prediction after recovery — no covering disruption.
    res = independent_recovery_metrics([_pred(ts=T0 + 99_999)], [d])
    assert res.overall.n == 0
