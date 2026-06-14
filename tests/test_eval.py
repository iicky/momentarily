"""Unit tests for the self-grading math (training/eval.py).

Uses synthetic prediction/transition records — no R2 access.
"""

from __future__ import annotations

from training.eval import (
    BIN_COUNT,
    PredictionRecord,
    TransitionRecord,
    build_calibration,
    build_eval,
    calibrate,
    recovery_metrics,
    snap_tick,
)


def _pred(
    *,
    ts: int,
    route: str = "1",
    condition: str = "normal",
    regime_entered_at: int | None = None,
    p_normal_in_30min: float = 0.9,
    p_normal_in_60min: float = 0.8,
    p_normal_in_120min: float = 0.7,
    recovery_minutes: int = 30,
    recovery_minutes_low: int = 15,
    recovery_minutes_high: int = 60,
    recovery_indeterminate: bool = False,
    params_version: int = 0,
) -> PredictionRecord:
    return PredictionRecord(
        ts=ts,
        route=route,
        condition=condition,
        params_version=params_version,
        regime_entered_at=regime_entered_at if regime_entered_at is not None else ts,
        p_normal=0.95,
        p_disrupted=0.04,
        p_suspended=0.01,
        p_normal_in_30min=p_normal_in_30min,
        p_normal_in_60min=p_normal_in_60min,
        p_normal_in_120min=p_normal_in_120min,
        recovery_minutes=recovery_minutes,
        recovery_minutes_low=recovery_minutes_low,
        recovery_minutes_high=recovery_minutes_high,
        recovery_indeterminate=recovery_indeterminate,
    )


def test_snap_tick_rounds_to_nearest_5min():
    assert snap_tick(1_700_000_000) == 1_700_000_100  # nearest 5-min slot
    assert snap_tick(1_700_000_100) == 1_700_000_100
    assert snap_tick(1_700_000_299) == 1_700_000_400


def test_calibrate_perfect_predictions():
    # Two routes; predictions made every 5 min for 90 min.
    # All predictions say p_normal_in_30min=1.0, and the actual condition is
    # always 'normal' → Brier should be 0, all weight in the top bin.
    ts0 = 1_700_000_000
    preds: list[PredictionRecord] = []
    for i in range(18):  # 90 minutes of ticks
        preds.append(
            _pred(
                ts=ts0 + i * 300,
                route="1",
                condition="normal",
                p_normal_in_30min=1.0,
            )
        )
    result = calibrate(preds, horizon_min=30)
    assert result.n == 12  # 18 ticks − 6 (last 30 min have no T+30 lookup)
    assert result.brier == 0.0
    top = result.bins[BIN_COUNT - 1]
    assert top.n == 12
    assert top.mean_pred == 1.0
    assert top.mean_outcome == 1.0


def test_calibrate_wrong_predictions():
    # All predictions say p_normal_in_30min=1.0, but actual condition is
    # 'disrupted' → Brier should be 1.0, top bin holds them with outcome=0.
    ts0 = 1_700_000_000
    preds: list[PredictionRecord] = []
    for i in range(18):
        preds.append(
            _pred(
                ts=ts0 + i * 300,
                route="1",
                condition="disrupted",
                p_normal_in_30min=1.0,
            )
        )
    result = calibrate(preds, horizon_min=30)
    assert result.n == 12
    assert result.brier == 1.0
    assert result.bins[BIN_COUNT - 1].mean_outcome == 0.0


def test_calibrate_baselines_persistence_and_climatology():
    # Condition alternates normal/disrupted each tick; the +30min future lands
    # on the same parity, so persistence is a perfect baseline (Brier 0 → BSS
    # undefined) while the model's constant 0.5 exactly matches the route's
    # climatology (base rate 0.5 → BSS vs climatology = 0).
    ts0 = 1_700_000_000
    preds = [
        _pred(
            ts=ts0 + i * 300,
            condition="normal" if i % 2 == 0 else "disrupted",
            p_normal_in_30min=0.5,
        )
        for i in range(18)
    ]
    result = calibrate(preds, horizon_min=30)
    assert result.n == 12
    assert result.brier == 0.25
    assert result.brier_persistence == 0.0
    assert result.bss_persistence is None  # reference already perfect
    assert result.brier_climatology == 0.25
    assert result.bss_climatology == 0.0


def test_calibrate_model_beats_baselines():
    # Disrupted for 60 min, then normal. A model that calls the recovery
    # perfectly has Brier 0; persistence (condition holds) misses every tick in
    # the last half-hour of the disruption, climatology hedges at the base rate.
    ts0 = 1_700_000_000
    preds = [
        _pred(
            ts=ts0 + i * 300,
            condition="disrupted" if i < 12 else "normal",
            p_normal_in_30min=0.0 if i < 6 else 1.0,
        )
        for i in range(18)
    ]
    result = calibrate(preds, horizon_min=30)
    assert result.n == 12
    assert result.brier == 0.0
    assert result.brier_persistence == 0.5
    assert result.bss_persistence == 1.0
    assert result.brier_climatology == 0.25
    assert result.bss_climatology == 1.0


def test_calibrate_no_lookup_when_gap():
    # Single prediction with no T+30 future → not graded.
    preds = [_pred(ts=1_700_000_000)]
    result = calibrate(preds, horizon_min=30)
    assert result.n == 0
    assert result.brier is None  # distinguishes "no data" from "perfect"


def test_recovery_perfect_when_dwell_matches_prediction():
    # Regime starts at t0, exits at t0 + 30min. Prediction at t0 says
    # recovery_minutes=30 → error 0.
    t0 = 1_700_000_000
    preds = [
        _pred(
            ts=t0,
            condition="disrupted",
            regime_entered_at=t0,
            recovery_minutes=30,
            recovery_minutes_low=15,
            recovery_minutes_high=45,
        )
    ]
    transitions = [
        TransitionRecord(
            ts=t0 + 1800,
            route="1",
            prev_state="disrupted",
            new_state="normal",
            regime_entered_at=t0,
            exited_at=t0 + 1800,
            dwell_sec=1800,
        )
    ]
    r = recovery_metrics(preds, transitions)
    assert r.overall.n == 1
    assert r.overall.mae_min == 0.0
    assert r.overall.iqr_coverage == 1.0


def test_recovery_mae_when_off_by_15_min():
    t0 = 1_700_000_000
    # Predicted 30 min, actual 45 min → MAE 15.
    preds = [
        _pred(ts=t0, condition="disrupted", regime_entered_at=t0, recovery_minutes=30)
    ]
    transitions = [
        TransitionRecord(
            ts=t0 + 2700,
            route="1",
            prev_state="disrupted",
            new_state="normal",
            regime_entered_at=t0,
            exited_at=t0 + 2700,
            dwell_sec=2700,
        )
    ]
    r = recovery_metrics(preds, transitions)
    assert r.overall.n == 1
    assert r.overall.mae_min is not None
    assert abs(r.overall.mae_min - 15.0) < 1e-9


def test_recovery_skips_ongoing_regimes():
    t0 = 1_700_000_000
    # Prediction exists but no matching transition → not graded.
    preds = [_pred(ts=t0, regime_entered_at=t0)]
    r = recovery_metrics(preds, [])
    assert r.overall.n == 0


def test_recovery_skips_indeterminate_predictions():
    """Indeterminate rows are clamps, not predictions — including them would
    bias MAE toward the recovery_minutes ceiling. See momentarily-x25."""
    t0 = 1_700_000_000
    preds = [
        _pred(
            ts=t0,
            condition="disrupted",
            regime_entered_at=t0,
            recovery_minutes=1440,  # clamped at ceiling
            recovery_minutes_low=1440,
            recovery_minutes_high=1440,
            recovery_indeterminate=True,
        ),
        _pred(
            ts=t0 + 100,
            route="2",
            condition="disrupted",
            regime_entered_at=t0 + 100,
            recovery_minutes=30,
            recovery_minutes_low=15,
            recovery_minutes_high=60,
        ),
    ]
    transitions = [
        TransitionRecord(
            ts=t0 + 2700,
            route="1",
            prev_state="disrupted",
            new_state="normal",
            regime_entered_at=t0,
            exited_at=t0 + 2700,  # actual 45 min — would be a huge error if scored
            dwell_sec=2700,
        ),
        TransitionRecord(
            ts=t0 + 1900,
            route="2",
            prev_state="disrupted",
            new_state="normal",
            regime_entered_at=t0 + 100,
            exited_at=t0 + 1900,
            dwell_sec=1800,
        ),
    ]
    r = recovery_metrics(preds, transitions)
    # Only route 2's prediction is graded.
    assert r.overall.n == 1
    assert "1" not in r.by_route
    assert "2" in r.by_route


def test_prediction_record_from_json_defaults_indeterminate() -> None:
    """Predictions written before momentarily-x25 didn't carry the field;
    from_json must default it to False so old archives still parse."""
    raw = {
        "ts": 1_700_000_000,
        "route": "1",
        "condition": "normal",
        "regime_entered_at": 1_700_000_000,
        "p_normal": 0.95,
        "p_disrupted": 0.04,
        "p_suspended": 0.01,
        "p_normal_in_30min": 0.9,
        "p_normal_in_60min": 0.8,
        "p_normal_in_120min": 0.7,
        "recovery_minutes": 30,
        "recovery_minutes_low": 15,
        "recovery_minutes_high": 60,
    }
    p = PredictionRecord.from_json(raw)
    assert p.recovery_indeterminate is False


def test_recovery_iqr_coverage():
    t0 = 1_700_000_000
    # Actual = 50 min, low=15, high=60 → covered. Actual = 80, low=15, high=60 → not.
    preds = [
        _pred(
            ts=t0,
            condition="disrupted",
            regime_entered_at=t0,
            recovery_minutes=30,
            recovery_minutes_low=15,
            recovery_minutes_high=60,
        ),
        _pred(
            ts=t0 + 100,
            route="2",
            condition="disrupted",
            regime_entered_at=t0 + 100,
            recovery_minutes=30,
            recovery_minutes_low=15,
            recovery_minutes_high=60,
        ),
    ]
    transitions = [
        TransitionRecord(
            ts=t0 + 3000,
            route="1",
            prev_state="disrupted",
            new_state="normal",
            regime_entered_at=t0,
            exited_at=t0 + 3000,  # actual 50 min from t0
            dwell_sec=3000,
        ),
        TransitionRecord(
            ts=t0 + 4900,
            route="2",
            prev_state="disrupted",
            new_state="normal",
            regime_entered_at=t0 + 100,
            exited_at=t0 + 4900,  # actual 80 min
            dwell_sec=4800,
        ),
    ]
    r = recovery_metrics(preds, transitions)
    assert r.overall.n == 2
    assert r.overall.iqr_coverage == 0.5


def test_recovery_skips_normal_predictions():
    """A route already in `normal` isn't recovering — its recovery_minutes=0
    prediction must not be graded against time-until-the-next-disruption, which
    would swamp the metric. See momentarily-qsl."""
    t0 = 1_700_000_000
    preds = [
        _pred(
            ts=t0,
            condition="normal",
            regime_entered_at=t0,
            recovery_minutes=0,
            recovery_minutes_low=0,
            recovery_minutes_high=0,
        )
    ]
    transitions = [
        TransitionRecord(
            ts=t0 + 3600,
            route="1",
            prev_state="normal",
            new_state="disrupted",
            regime_entered_at=t0,
            exited_at=t0 + 3600,
            dwell_sec=3600,
        )
    ]
    r = recovery_metrics(preds, transitions)
    assert r.overall.n == 0


def test_recovery_per_regime_macro_average():
    """A long regime contributes many ticks; per-tick MAE is dominated by it,
    per-regime weights both regimes equally. See momentarily-vk0.9."""
    t0 = 1_700_000_000
    # Regime A: 10 ticks, each off by 10 min. Regime B: 1 tick, off by 100 min.
    preds = [
        _pred(
            ts=t0 + i * 300,
            route="1",
            condition="disrupted",
            regime_entered_at=t0,
            recovery_minutes=round((t0 + 3600 - (t0 + i * 300)) / 60) + 10,
            recovery_minutes_low=0,
            recovery_minutes_high=10_000,
        )
        for i in range(10)
    ] + [
        _pred(
            ts=t0,
            route="2",
            condition="disrupted",
            regime_entered_at=t0,
            recovery_minutes=60 + 100,
            recovery_minutes_low=0,
            recovery_minutes_high=10_000,
        )
    ]
    transitions = [
        TransitionRecord(
            ts=t0 + 3600,
            route="1",
            prev_state="disrupted",
            new_state="normal",
            regime_entered_at=t0,
            exited_at=t0 + 3600,
            dwell_sec=3600,
        ),
        TransitionRecord(
            ts=t0 + 3600,
            route="2",
            prev_state="disrupted",
            new_state="normal",
            regime_entered_at=t0,
            exited_at=t0 + 3600,
            dwell_sec=3600,
        ),
    ]
    r = recovery_metrics(preds, transitions)
    # Per-tick: (10*10 + 100) / 11 ≈ 18.2 — dominated by the long regime.
    assert r.overall.n == 11
    assert r.overall.mae_min is not None
    assert abs(r.overall.mae_min - 200 / 11) < 1e-9
    # Per-regime: regime means are 10 and 100 → macro MAE 55, n = 2 regimes.
    assert r.per_regime.n == 2
    assert r.per_regime.mae_min is not None
    assert abs(r.per_regime.mae_min - 55.0) < 1e-9
    assert r.per_regime.iqr_coverage == 1.0


def test_build_eval_structure():
    t0 = 1_700_000_000
    preds = [_pred(ts=t0)]
    doc = build_eval(preds, [], window_start=t0, window_end=t0 + 86400)
    assert doc["predictions_seen"] == 1
    assert doc["transitions_seen"] == 0
    assert {c["horizon_min"] for c in doc["calibration"]} == {30, 60, 120}
    assert "overall" in doc["recovery"]
    assert doc["recovery"]["overall"]["n"] == 0
    # No version tags → no current-params segment.
    assert doc["current_params"] is None


def test_build_eval_segments_by_latest_params_version():
    # 12 ticks under params v100, then 12 under v200. The current-params
    # segment must grade only the v200 predictions — full-window metrics mix
    # model versions and dilute the latest retrain. See momentarily-vk0.5.
    t0 = 1_700_000_000
    preds = [
        _pred(ts=t0 + i * 300, params_version=100 if i < 12 else 200) for i in range(24)
    ]
    doc = build_eval(preds, [], window_start=t0, window_end=t0 + 86400)
    cp = doc["current_params"]
    assert cp is not None
    assert cp["trained_at"] == 200
    assert cp["n_predictions"] == 12
    # 12 v200 ticks, 30-min horizon → 12 − 6 gradeable... minus none: futures
    # exist within the v200 block for the first 6. n = 6.
    cal30 = next(c for c in cp["calibration"] if c["horizon_min"] == 30)
    assert cal30["n"] == 6


def test_build_calibration_is_compact_subset():
    # build_calibration keeps the window-aggregate reliability + recovery but
    # drops the per-route/per-alert/per-version breakdowns that bloat eval.json.
    t0 = 1_700_000_000
    preds = [_pred(ts=t0 + i * 300, params_version=200) for i in range(24)]
    eval_doc = build_eval(preds, [], window_start=t0, window_end=t0 + 86400)
    matrices = {
        "trained_at": 200,
        "states": ["normal", "disrupted", "suspended"],
        "routes": {"1": [[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.0, 0.3, 0.7]]},
    }
    calib = build_calibration(eval_doc, matrices)

    assert {c["horizon_min"] for c in calib["calibration"]} == {30, 60, 120}
    assert calib["predictions_seen"] == eval_doc["predictions_seen"]
    assert calib["transition_matrices"] == matrices
    # Aggregate recovery only — no route/alert/version explosion.
    assert set(calib["recovery"]) == {"overall", "per_regime"}
    assert "by_route" not in calib["recovery"]
    assert "current_params" not in calib


def test_prediction_record_from_json_defaults_params_version():
    raw = {
        "ts": 1,
        "route": "1",
        "condition": "normal",
        "regime_entered_at": 1,
        "p_normal": 0.9,
        "p_disrupted": 0.05,
        "p_suspended": 0.05,
        "p_normal_in_30min": 0.9,
        "p_normal_in_60min": 0.9,
        "p_normal_in_120min": 0.9,
        "recovery_minutes": 0,
        "recovery_minutes_low": 0,
        "recovery_minutes_high": 0,
    }
    assert PredictionRecord.from_json(raw).params_version == 0
    assert (
        PredictionRecord.from_json({**raw, "params_version": 17}).params_version == 17
    )
