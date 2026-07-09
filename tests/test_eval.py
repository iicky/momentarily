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
    prequential_calibration,
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
    recovery_source: str | None = None,
) -> PredictionRecord:
    return PredictionRecord(
        ts=ts,
        route=route,
        condition=condition,
        params_version=params_version,
        recovery_source=recovery_source,
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


def test_calibrate_by_current_splits_normal_vs_recovery():
    """The persistence loss decomposes by current condition: a route that is
    normal-now and stays normal lands in 'normal_now'; a disrupted-now route in
    'not_normal_now'. Counts and the realized normal rate must line up."""
    ts0 = 1_700_000_000
    preds: list[PredictionRecord] = []
    for i in range(18):
        preds.append(_pred(ts=ts0 + i * 300, route="N", condition="normal"))
    for i in range(18):
        preds.append(_pred(ts=ts0 + i * 300, route="D", condition="disrupted"))
    result = calibrate(preds, horizon_min=30)
    normal = result.by_current["normal_now"]
    recovery = result.by_current["not_normal_now"]
    assert normal.n == 12  # 18 − 6 unmatched tail
    assert recovery.n == 12
    # Route N is always normal → outcome rate 1.0; route D always disrupted → 0.0
    assert normal.mean_outcome == 1.0
    assert recovery.mean_outcome == 0.0
    # The two strata partition the matched set.
    assert normal.n + recovery.n == result.n


def test_calibrate_by_current_flags_underconfident_normal():
    """A normal-now route that stays normal but is forecast under-confidently
    (p_normal_in_30 = 0.8) loses to persistence on that slice — the eeh signature:
    the forecast trails the realized rate, so its Brier is worse than the hard
    persistence call."""
    ts0 = 1_700_000_000
    preds = [
        _pred(
            ts=ts0 + i * 300,
            route="1",
            condition="normal",
            p_normal_in_30min=0.8,
        )
        for i in range(18)
    ]
    normal = calibrate(preds, horizon_min=30).by_current["normal_now"]
    assert normal.mean_outcome == 1.0  # it really did stay normal
    assert normal.mean_pred is not None
    assert normal.mean_pred < 1.0  # but the forecast hedged below the realized rate
    assert normal.brier is not None
    assert normal.brier_persistence is not None
    # Persistence (predict 1.0) beats the hedged 0.8 forecast on this slice.
    assert normal.brier > normal.brier_persistence


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


def test_schedule_rows_excluded_from_calibration_and_recovery():
    # A schedule-recovery row is a deterministic resume lookup, not an HMM
    # forecast — it must not be graded for calibration or recovery (it would
    # otherwise flatter the model). Mirrors the viz exclusion.
    t0 = 1_700_000_000
    # One HMM disruption that recovers, one schedule row in the same window.
    preds = [
        _pred(ts=t0, condition="suspended", regime_entered_at=t0, recovery_minutes=30),
        _pred(
            ts=t0,
            route="2",
            condition="suspended",
            regime_entered_at=t0,
            recovery_source="schedule",
        ),
    ]
    trans = [
        TransitionRecord(
            ts=t0 + 1800,
            route="1",
            prev_state="suspended",
            new_state="normal",
            regime_entered_at=t0,
            exited_at=t0 + 1800,
            dwell_sec=1800,
            alert_type_at_entry=None,
        ),
    ]
    rec = recovery_metrics(preds, trans)
    assert rec.excluded_schedule == 1
    # Only the HMM row (route 1) is graded; route 2's schedule row is skipped.
    assert rec.overall.n == 1

    # Schedule row is also skipped as a calibration predictor.
    cal = calibrate(preds, 30)
    assert cal.excluded_schedule == 1


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


def test_calibrate_truth_overrides_outcome_and_persistence():
    """With truth_by_key, outcome and persistence come from the held-out truth
    map, not the model's own condition — the load-bearing switch once the
    train-movement channel feeds the HMM and can no longer self-grade."""
    t0 = 1_700_000_100
    h_sec = 30 * 60
    preds = [
        # Model's own condition says the opposite of truth at both ticks.
        _pred(ts=t0, condition="normal", p_normal_in_30min=1.0),
        _pred(ts=t0 + h_sec, condition="disrupted"),
    ]
    truth = {("1", t0): "disrupted", ("1", t0 + h_sec): "normal"}
    result = calibrate(preds, horizon_min=30, truth_by_key=truth)
    assert result.n == 1
    assert result.brier == 0.0  # truth outcome "normal", not future.condition
    assert (
        result.brier_persistence == 1.0
    )  # truth persistence "disrupted", not p.condition


def test_calibrate_sparse_absent_truth_defaults_to_normal():
    """A route-tick missing from the sparse truth map is normal, not unknown:
    the sample is still graded (not skipped), and the missing side defaults to
    outcome/persistence 1.0."""
    t0 = 1_700_000_100
    h_sec = 30 * 60
    # Route A: T present ("disrupted"), T+H absent → outcome must default to
    # normal (1.0), not drop the sample from grading.
    # Route B: T+H present ("disrupted"), T absent → persistence must default
    # to normal (1.0).
    truth = {("A", t0): "disrupted", ("B", t0 + h_sec): "disrupted"}
    preds = [
        _pred(ts=t0, route="A", p_normal_in_30min=1.0),
        _pred(ts=t0 + h_sec, route="A"),
        _pred(ts=t0, route="B", p_normal_in_30min=0.0),
        _pred(ts=t0 + h_sec, route="B"),
    ]
    result = calibrate(preds, horizon_min=30, truth_by_key=truth)
    assert result.n == 2  # neither sample skipped despite the sparse gaps
    assert result.brier == 0.0  # route A: outcome defaulted to normal (1.0)
    assert (
        result.brier_persistence == 1.0
    )  # route B: persistence defaulted to normal (1.0)


def test_calibrate_coverage_gated_on_prediction_not_truth():
    # Truth present at both T and T+H, but no prediction at T+H → truth
    # presence alone doesn't create coverage; the window guard still requires
    # a future prediction to exist.
    t0 = 1_700_000_100
    h_sec = 30 * 60
    preds = [_pred(ts=t0)]
    truth = {("1", t0): "disrupted", ("1", t0 + h_sec): "normal"}
    result = calibrate(preds, horizon_min=30, truth_by_key=truth)
    assert result.n == 0
    assert result.brier is None


def test_calibrate_default_path_unchanged_without_truth_by_key():
    # Same predictions as the truth-override test above, but no truth_by_key
    # → must still grade against future.condition/p.condition (regression
    # guard for the pre-existing self-consistency path).
    t0 = 1_700_000_100
    h_sec = 30 * 60
    preds = [
        _pred(ts=t0, condition="normal", p_normal_in_30min=1.0),
        _pred(ts=t0 + h_sec, condition="disrupted"),
    ]
    result = calibrate(preds, horizon_min=30)
    assert result.n == 1
    assert result.brier == 1.0  # future.condition="disrupted" → outcome=0.0
    assert result.brier_persistence == 1.0  # p.condition="normal" → persistence=1.0


def test_calibrate_by_current_keys_off_truth_persistence():
    """by_current splits on truth-derived persistence, not the model's own
    condition. Each route's model condition is set to the opposite of its
    truth persistence, so a regression to p.condition would flip which
    stratum the sample lands in (and its mean_outcome)."""
    t0 = 1_700_000_100
    h_sec = 30 * 60
    truth = {
        ("N", t0): "normal",  # truth: normal now
        ("N", t0 + h_sec): "disrupted",
        ("D", t0): "disrupted",  # truth: not normal now
        ("D", t0 + h_sec): "normal",
    }
    preds = [
        _pred(ts=t0, route="N", condition="disrupted"),
        _pred(ts=t0 + h_sec, route="N", condition="normal"),
        _pred(ts=t0, route="D", condition="normal"),
        _pred(ts=t0 + h_sec, route="D", condition="disrupted"),
    ]
    result = calibrate(preds, horizon_min=30, truth_by_key=truth)
    normal_now = result.by_current["normal_now"]
    not_normal_now = result.by_current["not_normal_now"]
    assert normal_now.n == 1
    assert not_normal_now.n == 1
    assert normal_now.mean_outcome == 0.0  # route N, per truth (not condition)
    assert not_normal_now.mean_outcome == 1.0  # route D, per truth (not condition)


def test_prequential_calibration_segments_and_flags_low_sample():
    """prequential_calibration segments by params_version, flags low_sample
    below min_samples, and echoes the truth-source/floor metadata."""
    t0 = 1_700_000_100
    truth: dict[tuple[str, int], str] = {}  # empty sparse truth: all "normal"

    # v100: 8 predictions, 5min apart → 2 have a T+30min future (i=0,1).
    v100 = [_pred(ts=t0 + i * 300, route="V1", params_version=100) for i in range(8)]
    # v200: 10 predictions → 4 have a T+30min future (i=0..3).
    v200 = [_pred(ts=t0 + i * 300, route="V2", params_version=200) for i in range(10)]

    result = prequential_calibration(
        v100 + v200, truth, severity_floor=2, min_samples=3
    )

    assert result["truth_source"] == "alert_feed_clearance"
    assert result["severity_floor"] == 2
    assert len(result["overall"]) == 3
    assert {c["horizon_min"] for c in result["overall"]} == {30, 60, 120}

    by_version = result["by_params_version"]
    assert set(by_version) == {"100", "200"}

    v100_cal30 = next(
        c for c in by_version["100"]["calibration"] if c["horizon_min"] == 30
    )
    v200_cal30 = next(
        c for c in by_version["200"]["calibration"] if c["horizon_min"] == 30
    )
    assert v100_cal30["n"] == 2
    assert v200_cal30["n"] == 4
    assert by_version["100"]["n_predictions"] == 8
    assert by_version["200"]["n_predictions"] == 10
    assert by_version["100"]["n_matched_primary"] == 2
    assert by_version["200"]["n_matched_primary"] == 4
    assert by_version["100"]["low_sample"] is True  # 2 < min_samples=3
    assert by_version["200"]["low_sample"] is False  # 4 >= min_samples=3


def test_calibrate_coverage_predictions_spans_retrain_boundary():
    """coverage_predictions supplies the T+horizon lookup index independently
    of `predictions`: a v100 forecast whose only T+H row already retrained to
    v200 isn't dropped for lack of a same-segment future — the lookup comes
    from the full stream, the outcome from held-out truth."""
    t0 = 1_700_000_100
    h_sec = 30 * 60
    v100_forecast = _pred(ts=t0, params_version=100)
    v200_future = _pred(ts=t0 + h_sec, params_version=200)
    both = [v100_forecast, v200_future]
    truth = {("1", t0): "disrupted", ("1", t0 + h_sec): "normal"}

    # Segment-only coverage: no same-set future → not graded.
    solo = calibrate([v100_forecast], horizon_min=30, truth_by_key=truth)
    assert solo.n == 0

    # Full-stream coverage: the v200 row supplies the T+H lookup → graded.
    spanned = calibrate(
        [v100_forecast],
        horizon_min=30,
        truth_by_key=truth,
        coverage_predictions=both,
    )
    assert spanned.n == 1

    # prequential_calibration always passes the full stream as coverage, so
    # the v100 segment picks up the boundary forecast too.
    result = prequential_calibration(both, truth, severity_floor=2, min_samples=1)
    assert result["by_params_version"]["100"]["calibration"][0]["n"] == 1
