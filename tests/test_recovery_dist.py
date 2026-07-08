"""Tests for the CRPS/PIT recovery-distribution scorecard (training/recovery_dist.py).

Ports the 9-assertion parity oracle in viz/tests/recovery_dist.test.ts to
Python — each test below mirrors one of those JS `test(...)` blocks — plus a
few additional contracts (curve shape, histogram/JSON invariants) called out
separately.
"""

from __future__ import annotations

import json
import math
from itertools import pairwise

import pytest

from training.recovery_dist import (
    RECOVERY_TMAX_MIN,
    VERDICT_MIN_INCIDENTS,
    RecoveryDistReport,
    RecoveryDistSample,
    RecoveryWeighting,
    predicted_recovery_curve,
    recovery_dist_report,
    recovery_verdict,
    report_as_dict,
)

UNIFORM: list[int] = [10] * 10


def _step_curve(at: int) -> list[float]:
    """Step CDF at integer minutes 0..4 that jumps to 1 at minute ``at`` —
    mirrors the TS oracle's stepCurve."""
    return [1.0 if t >= at else 0.0 for t in range(5)]


def _sample(regime_key: str, actual_min: float, jump_at: int) -> RecoveryDistSample:
    return RecoveryDistSample(
        pred_curve=_step_curve(jump_at), actual_min=actual_min, regime_key=regime_key
    )


def _report(
    pit: list[int], mean_pit: float, regimes: int, skill: float
) -> RecoveryDistReport:
    """Minimal report carrying only the fields recovery_verdict reads —
    mirrors the TS oracle's report() builder."""
    per_tick = RecoveryWeighting(
        n=0, mean_crps=0.0, baseline_crps=0.0, skill=0.0, mean_pit=0.0
    )
    per_regime = RecoveryWeighting(
        n=regimes, mean_crps=0.0, baseline_crps=0.0, skill=skill, mean_pit=mean_pit
    )
    return RecoveryDistReport(
        n=0,
        mean_crps=0.0,
        baseline_crps=0.0,
        skill=skill,
        mean_pit=mean_pit,
        per_tick=per_tick,
        per_regime=per_regime,
        pit=pit,
        grid=[],
        predicted_curve=[],
        empirical_curve=[],
        horizons=[],
    )


# --- recovery_dist_report: per-tick vs per-regime weighting ---


def test_report_separates_per_tick_from_per_regime_weighting() -> None:
    # One long, well-forecast incident (8 ticks, curve nails the recovery) and
    # one short, badly-forecast incident (2 ticks, curve says "already back").
    # Per-tick is dominated by the 8 good ticks; per-regime weights the two
    # incidents equally.
    samples: list[RecoveryDistSample] = [_sample("good:0", 2, 2) for _ in range(8)] + [
        _sample("bad:0", 3, 0) for _ in range(2)
    ]
    r = recovery_dist_report(samples)

    assert r.per_tick.n == 10
    assert r.per_regime.n == 2
    # Top-level headline stays per-tick for the curve view's back-compat.
    assert r.n == 10
    assert r.mean_crps == r.per_tick.mean_crps
    # The bad incident is one tick-heavy regime's worth of error spread across
    # only two ticks, so equal-per-incident weighting must score worse than
    # per-tick.
    assert r.per_regime.mean_crps > r.per_tick.mean_crps
    assert math.isfinite(r.per_tick.skill)
    assert math.isfinite(r.per_regime.skill)


def test_report_collapses_ticks_from_one_regime_into_one_incident() -> None:
    samples: list[RecoveryDistSample] = [_sample("solo:100", 2, 2) for _ in range(12)]
    r = recovery_dist_report(samples)
    assert r.per_tick.n == 12
    assert r.per_regime.n == 1


def test_report_handles_the_empty_window() -> None:
    r = recovery_dist_report([])
    assert r.n == 0
    assert r.per_tick.n == 0
    assert r.per_regime.n == 0
    assert math.isnan(r.per_regime.mean_crps)


# --- recovery_verdict: reading the PIT shape ---


def test_verdict_too_few_incidents_reads_inconclusive() -> None:
    v = recovery_verdict(_report(UNIFORM, 0.5, VERDICT_MIN_INCIDENTS - 1, 0.2))
    assert v.verdict == "Inconclusive"
    assert v.tone == "muted"


def test_verdict_empty_histogram_reads_no_data() -> None:
    v = recovery_verdict(_report([0] * 10, float("nan"), 0, float("nan")))
    assert v.verdict == "Not enough data yet"


def test_verdict_uniform_pit_with_positive_skill_is_well_calibrated() -> None:
    v = recovery_verdict(_report(UNIFORM, 0.5, 20, 0.3))
    assert v.verdict == "Well calibrated"
    assert v.tone == "good"
    assert v.warning is None


def test_verdict_calibrated_shape_but_negative_skill_warns_of_the_conflict() -> None:
    v = recovery_verdict(_report(UNIFORM, 0.5, 20, -0.3))
    assert v.verdict == "Well calibrated"
    assert v.warning is not None
    assert "baseline" in v.warning


def test_verdict_left_piled_pit_leans_cautious() -> None:
    pit = [30, 25, 20, 15, 5, 2, 1, 1, 1, 0]
    v = recovery_verdict(_report(pit, 0.3, 20, 0.1))
    assert v.verdict == "Leans cautious"


def test_verdict_u_shaped_pit_reads_overconfident() -> None:
    pit = [40, 5, 3, 2, 1, 1, 2, 3, 5, 38]
    v = recovery_verdict(_report(pit, 0.5, 20, -0.2))
    assert v.verdict == "Overconfident"
    assert v.tone == "warn"


# --- predicted_recovery_curve: shape contract ---


@pytest.mark.parametrize(
    ("elapsed_sec", "tail_ll"),
    [
        (0.0, None),
        (1200.0, None),
        (1200.0, [2.0, 1200.0]),
    ],
)
def test_predicted_recovery_curve_is_length_241_and_monotone(
    elapsed_sec: float, tail_ll: list[float] | None
) -> None:
    # Synthetic 21-point curve_sec (CURVE_POINTS): dwell uniform on 0..1200s.
    # elapsed_sec=1200 sits past the curve's max, forcing the tail branch (and
    # so exercising both the exponential patch and the log-logistic tail_ll).
    curve_sec = [i * 60 for i in range(21)]
    curve = predicted_recovery_curve(elapsed_sec, curve_sec, tail_ll)

    assert len(curve) == RECOVERY_TMAX_MIN + 1
    assert all(b >= a - 1e-9 for a, b in pairwise(curve))
    assert all(-1e-9 <= v <= 1.0 + 1e-9 for v in curve)


# --- histogram + serialization invariants ---


def test_pit_histogram_sums_to_n() -> None:
    # actual_min at and past both ends of the tick range exercises the
    # idx = min(t_max, max(0, round(y))) clamp in recovery_dist_report — every
    # sample must still land in exactly one PIT bin.
    linear_curve = [t / RECOVERY_TMAX_MIN for t in range(RECOVERY_TMAX_MIN + 1)]
    samples = [
        RecoveryDistSample(pred_curve=linear_curve, actual_min=-10.0, regime_key="r1"),
        RecoveryDistSample(pred_curve=linear_curve, actual_min=0.0, regime_key="r2"),
        RecoveryDistSample(pred_curve=linear_curve, actual_min=120.0, regime_key="r3"),
        RecoveryDistSample(pred_curve=linear_curve, actual_min=500.0, regime_key="r4"),
    ]
    r = recovery_dist_report(samples)
    assert len(r.pit) == 10
    assert sum(r.pit) == r.n


def test_report_as_dict_is_json_serializable() -> None:
    samples: list[RecoveryDistSample] = [_sample("good:0", 2, 2) for _ in range(8)] + [
        _sample("bad:0", 3, 0) for _ in range(2)
    ]
    r = recovery_dist_report(samples)

    round_tripped = json.loads(json.dumps(report_as_dict(r)))
    assert round_tripped["n"] == 10
    assert round_tripped["per_tick"]["n"] == 10
    assert round_tripped["per_regime"]["n"] == 2
    assert sum(round_tripped["pit"]) == 10
