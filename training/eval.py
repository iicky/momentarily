"""Self-grading job: reliability + dwell accuracy from R2 streams.

Reads:
    v1/predictions/<date>/<ts>.jsonl
    v1/regime_transitions/<date>/<ts>.jsonl

Computes:
    Calibration  — Brier score + 10-bin reliability table for
                   p_normal_in_30/60/120min, grading against the published
                   `condition` k minutes later (snapped to the 5-min tick grid).
    Recovery     — MAE, RMSE, IQR coverage of recovery_minutes against actual
                   remaining-time-in-regime for every prediction made during a
                   regime that subsequently ended.

Emits:
    v1/eval.json — public, R2 custom-domain readable, max-age=300.

Run with:
    murk exec -- python -m training.eval [--days 7]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from training.r2_client import R2Config, load_config, make_client

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

    # load_r2 no longer imports eval (it uses a PredictionLike Protocol), so this
    # is no longer a cycle.
    from training.load_r2 import Disruption

TICK_SECONDS = 300  # publisher cron grid
HORIZONS_MIN = (30, 60, 120)
BIN_COUNT = 10  # 0.0-0.1, ..., 0.9-1.0
STATES = ("normal", "disrupted", "suspended")  # transition-matrix row/col order
EVAL_KEY = "v1/eval.json"
CALIBRATION_KEY = "v1/calibration.json"
PARAMS_KEY = "state/params.json"


# --- Records mirror the JSONL written by worker/src/grading.ts ---


@dataclass(frozen=True)
class PredictionRecord:
    ts: int
    route: str
    condition: str
    regime_entered_at: int
    p_normal: float
    p_disrupted: float
    p_suspended: float
    p_normal_in_30min: float
    p_normal_in_60min: float
    p_normal_in_120min: float
    recovery_minutes: int
    recovery_minutes_low: int
    recovery_minutes_high: int
    # True when the dwell estimate saturated the clamp; recovery_minutes is not
    # a real prediction for these rows. Defaults False so JSONL written before
    # momentarily-x25 still parses.
    recovery_indeterminate: bool = False
    # primary_alert_type at this tick. Defaults None for JSONL written before
    # momentarily-22k. Lets the grader segment calibration by cause.
    primary_alert_type: str | None = None
    # trained_at of the params.json active when this prediction was made.
    # 0 for bootstrap params or JSONL written before momentarily-vk0.5.
    # Predictions are prequential (params are always trained on data strictly
    # before the prediction), so this is a version tag for segmentation, not a
    # leakage guard.
    params_version: int = 0
    # "schedule" rows are deterministic planned-resume lookups, not HMM dwell
    # estimates — excluded from calibration/recovery grading (graded for
    # schedule adherence elsewhere). None for JSONL written before schedule
    # recovery shipped; treated as "hmm".
    recovery_source: str | None = None

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> PredictionRecord:
        return cls(
            ts=int(raw["ts"]),
            route=str(raw["route"]),
            condition=str(raw["condition"]),
            regime_entered_at=int(raw["regime_entered_at"]),
            p_normal=float(raw["p_normal"]),
            p_disrupted=float(raw["p_disrupted"]),
            p_suspended=float(raw["p_suspended"]),
            p_normal_in_30min=float(raw["p_normal_in_30min"]),
            p_normal_in_60min=float(raw["p_normal_in_60min"]),
            p_normal_in_120min=float(raw["p_normal_in_120min"]),
            recovery_minutes=int(raw["recovery_minutes"]),
            recovery_minutes_low=int(raw["recovery_minutes_low"]),
            recovery_minutes_high=int(raw["recovery_minutes_high"]),
            recovery_indeterminate=bool(raw.get("recovery_indeterminate", False)),
            primary_alert_type=raw.get("primary_alert_type"),
            params_version=int(raw.get("params_version") or 0),
            recovery_source=raw.get("recovery_source"),
        )


@dataclass(frozen=True)
class TransitionRecord:
    ts: int
    route: str
    prev_state: str
    new_state: str
    regime_entered_at: int
    exited_at: int
    dwell_sec: int
    # primary_alert_type when prev_state began. None for records written before
    # momentarily-22k or when no alert was active at regime start. Phase 2
    # (momentarily-alu) segments dwell quantiles on this.
    alert_type_at_entry: str | None = None

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> TransitionRecord:
        return cls(
            ts=int(raw["ts"]),
            route=str(raw["route"]),
            prev_state=str(raw["prev_state"]),
            new_state=str(raw["new_state"]),
            regime_entered_at=int(raw["regime_entered_at"]),
            exited_at=int(raw["exited_at"]),
            dwell_sec=int(raw["dwell_sec"]),
            alert_type_at_entry=raw.get("alert_type_at_entry"),
        )


# --- R2 I/O ---


def _date_range(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _list_keys(client: S3Client, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token: str | None = None
    while True:
        if token:
            resp = client.list_objects_v2(
                Bucket=bucket, Prefix=prefix, MaxKeys=1000, ContinuationToken=token
            )
        else:
            resp = client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1000)
        for obj in resp.get("Contents") or []:
            key = obj.get("Key")
            if key:
                keys.append(key)
        if not resp.get("IsTruncated"):
            return keys
        token = resp.get("NextContinuationToken")
        if not token:
            return keys


def _read_jsonl(client: S3Client, bucket: str, key: str) -> list[dict[str, Any]]:
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read().decode()
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def load_predictions(
    client: S3Client,
    bucket: str,
    start_date: date,
    end_date: date,
) -> list[PredictionRecord]:
    keys: list[str] = []
    for d in _date_range(start_date, end_date):
        keys.extend(_list_keys(client, bucket, f"v1/predictions/{d.isoformat()}/"))
    out: list[PredictionRecord] = []

    def fetch(k: str) -> list[dict[str, Any]]:
        return _read_jsonl(client, bucket, k)

    with ThreadPoolExecutor(max_workers=16) as pool:
        for rows in pool.map(fetch, keys):
            out.extend(PredictionRecord.from_json(r) for r in rows)
    return out


def load_transitions(
    client: S3Client,
    bucket: str,
    start_date: date,
    end_date: date,
) -> list[TransitionRecord]:
    keys: list[str] = []
    for d in _date_range(start_date, end_date):
        keys.extend(
            _list_keys(client, bucket, f"v1/regime_transitions/{d.isoformat()}/")
        )
    out: list[TransitionRecord] = []

    def fetch(k: str) -> list[dict[str, Any]]:
        return _read_jsonl(client, bucket, k)

    with ThreadPoolExecutor(max_workers=16) as pool:
        for rows in pool.map(fetch, keys):
            out.extend(TransitionRecord.from_json(r) for r in rows)
    return out


# --- Calibration math ---


def snap_tick(ts: int) -> int:
    return ((ts + TICK_SECONDS // 2) // TICK_SECONDS) * TICK_SECONDS


@dataclass
class ReliabilityBin:
    bin_lo: float
    bin_hi: float
    n: int = 0
    sum_pred: float = 0.0
    sum_outcome: float = 0.0

    @property
    def mean_pred(self) -> float | None:
        return self.sum_pred / self.n if self.n else None

    @property
    def mean_outcome(self) -> float | None:
        return self.sum_outcome / self.n if self.n else None


@dataclass
class CalibrationResult:
    horizon_min: int
    n: int
    brier: float | None  # None when n=0 — distinguishes "no data" from "perfect"
    # Reference forecasts on the same matched samples. Persistence predicts the
    # current condition holds at T+horizon (the baseline to beat for a sticky
    # process on short horizons); climatology predicts the per-route base rate
    # of normal over the eval window (in-sample, the standard reference).
    # A raw Brier score is uninterpretable without these. See momentarily-vk0.4.
    brier_persistence: float | None
    brier_climatology: float | None
    # Brier skill scores: 1 − brier/brier_ref. Positive = beats the baseline.
    # None when the reference is 0 (baseline already perfect) or n=0.
    bss_persistence: float | None
    bss_climatology: float | None
    bins: list[ReliabilityBin]
    # Schedule-recovery rows skipped as predictors — they're deterministic resume
    # lookups, not HMM forecasts, so grading them would flatter the model.
    excluded_schedule: int = 0


def _skill(brier: float | None, reference: float | None) -> float | None:
    if brier is None or reference is None or reference == 0.0:
        return None
    return 1.0 - brier / reference


def calibrate(
    predictions: list[PredictionRecord], horizon_min: int
) -> CalibrationResult:
    """Pair each prediction at T with the actual condition at T + horizon_min."""
    # Index predictions by (route, snapped_ts) so T+horizon lookup is O(1).
    by_key: dict[tuple[str, int], PredictionRecord] = {}
    for p in predictions:
        by_key[(p.route, snap_tick(p.ts))] = p

    bins = [
        ReliabilityBin(bin_lo=i / BIN_COUNT, bin_hi=(i + 1) / BIN_COUNT)
        for i in range(BIN_COUNT)
    ]

    horizon_sec = horizon_min * 60
    pred_field = f"p_normal_in_{horizon_min}min"

    # (model_pred, persistence_pred, outcome, route) per matched sample. Two
    # passes: climatology needs the per-route outcome base rate first.
    matched: list[tuple[float, float, float, str]] = []
    route_outcome_sum: dict[str, float] = {}
    route_outcome_n: dict[str, int] = {}
    excluded_schedule = 0

    for p in predictions:
        # Deterministic planned-resume lookup, not an HMM forecast — skip as a
        # predictor (it can still be a future outcome; condition is HMM-derived).
        if p.recovery_source == "schedule":
            excluded_schedule += 1
            continue
        future_key = (p.route, snap_tick(p.ts) + horizon_sec)
        future = by_key.get(future_key)
        if future is None:
            continue
        outcome = 1.0 if future.condition == "normal" else 0.0
        pred: float = getattr(p, pred_field)
        persistence = 1.0 if p.condition == "normal" else 0.0
        matched.append((pred, persistence, outcome, p.route))
        route_outcome_sum[p.route] = route_outcome_sum.get(p.route, 0.0) + outcome
        route_outcome_n[p.route] = route_outcome_n.get(p.route, 0) + 1

    base_rate = {
        route: route_outcome_sum[route] / route_outcome_n[route]
        for route in route_outcome_n
    }

    n = len(matched)
    brier_sum = 0.0
    persistence_sum = 0.0
    climatology_sum = 0.0
    for pred, persistence, outcome, route in matched:
        brier_sum += (pred - outcome) ** 2
        persistence_sum += (persistence - outcome) ** 2
        climatology_sum += (base_rate[route] - outcome) ** 2
        idx = min(int(pred * BIN_COUNT), BIN_COUNT - 1)
        b = bins[idx]
        b.n += 1
        b.sum_pred += pred
        b.sum_outcome += outcome

    brier = brier_sum / n if n else None
    brier_persistence = persistence_sum / n if n else None
    brier_climatology = climatology_sum / n if n else None
    return CalibrationResult(
        horizon_min=horizon_min,
        n=n,
        brier=brier,
        brier_persistence=brier_persistence,
        brier_climatology=brier_climatology,
        bss_persistence=_skill(brier, brier_persistence),
        bss_climatology=_skill(brier, brier_climatology),
        bins=bins,
        excluded_schedule=excluded_schedule,
    )


# --- Recovery / dwell math ---


@dataclass
class RecoveryStats:
    n: int
    mae_min: float | None
    rmse_min: float | None
    iqr_coverage: (
        float | None
    )  # fraction of predictions whose [low,high] contained actual


@dataclass
class RecoveryResult:
    overall: RecoveryStats
    # Macro-average with one sample per regime (each regime's per-tick errors
    # are averaged first, then regimes weighted equally). The per-tick view
    # weights a 6-hour regime ~72x a 30-minute one, so a couple of marathon
    # planned-work regimes dominate MAE. n = number of regimes. See
    # momentarily-vk0.9.
    per_regime: RecoveryStats = field(
        default_factory=lambda: RecoveryStats(
            n=0, mae_min=None, rmse_min=None, iqr_coverage=None
        )
    )
    by_route: dict[str, RecoveryStats] = field(
        default_factory=lambda: {}  # noqa: PIE807
    )
    # Recovery accuracy segmented by the prediction-tick's primary_alert_type.
    # Surfaces whether cause-conditioned dwell quantiles (momentarily-alu) are
    # actually tightening the interval per cause. Predictions with no alert type
    # are omitted from this breakdown.
    by_alert_type: dict[str, RecoveryStats] = field(
        default_factory=lambda: {}  # noqa: PIE807
    )
    # Schedule-recovery rows skipped — deterministic resume lookups graded for
    # adherence elsewhere, not against HMM dwell.
    excluded_schedule: int = 0


# Resolve a prediction to (actual_recovery_tick | None, regime_key). This is the
# ONLY thing that differs between the HMM-argmax truth (recovery_metrics) and the
# independent trip-updates truth (independent_recovery_metrics).
ExitResolver = Callable[["PredictionRecord"], tuple[int | None, tuple[str, int]]]


def _grade_recovery(
    predictions: list[PredictionRecord],
    exit_for: ExitResolver,
) -> RecoveryResult:
    """Shared recovery grading: for every prediction made during a disruption that
    subsequently ended, compare recovery_minutes against actual remaining time and
    check IQR coverage. Each prediction-tick is one grading sample. `exit_for`
    supplies the actual recovery time and the regime key to group by."""
    abs_errors: list[float] = []
    sq_errors: list[float] = []
    covered = 0
    excluded_schedule = 0
    by_route_abs: dict[str, list[float]] = {}
    by_route_sq: dict[str, list[float]] = {}
    by_route_cov: dict[str, list[int]] = {}
    by_alert_abs: dict[str, list[float]] = {}
    by_alert_sq: dict[str, list[float]] = {}
    by_alert_cov: dict[str, list[int]] = {}
    by_regime: dict[tuple[str, int], list[tuple[float, float, int]]] = {}

    for p in predictions:
        # Recovery time is only meaningful during a disruption. A route that is
        # already normal isn't "recovering" — it predicts recovery_minutes=0, and
        # grading that against time-until-the-next-disruption (the end of the
        # current normal regime) swamps MAE and pins IQR coverage near zero. Skip
        # them so the metric reflects actual recoveries. See momentarily-qsl.
        if p.condition == "normal":
            continue
        # Indeterminate rows are clamped, not predicted — including them would
        # bias MAE toward the clamp ceiling. See momentarily-x25.
        if p.recovery_indeterminate:
            continue
        # Schedule recoveries are deterministic resume lookups, graded for
        # adherence elsewhere — not against HMM dwell.
        if p.recovery_source == "schedule":
            excluded_schedule += 1
            continue
        exited_at, regime_key = exit_for(p)
        if exited_at is None or exited_at <= p.ts:
            continue
        actual_remaining_min = (exited_at - p.ts) / 60.0
        err = abs(p.recovery_minutes - actual_remaining_min)
        sq_err = (p.recovery_minutes - actual_remaining_min) ** 2
        within = (
            p.recovery_minutes_low <= actual_remaining_min <= p.recovery_minutes_high
        )
        abs_errors.append(err)
        sq_errors.append(sq_err)
        covered += 1 if within else 0
        by_regime.setdefault(regime_key, []).append(
            (err, sq_err, 1 if within else 0)
        )
        by_route_abs.setdefault(p.route, []).append(err)
        by_route_sq.setdefault(p.route, []).append(sq_err)
        by_route_cov.setdefault(p.route, []).append(1 if within else 0)
        if p.primary_alert_type is not None:
            by_alert_abs.setdefault(p.primary_alert_type, []).append(err)
            by_alert_sq.setdefault(p.primary_alert_type, []).append(sq_err)
            by_alert_cov.setdefault(p.primary_alert_type, []).append(1 if within else 0)

    overall = _stats_from(abs_errors, sq_errors, covered)
    by_route = {
        route: _stats_from(
            by_route_abs[route], by_route_sq[route], sum(by_route_cov[route])
        )
        for route in by_route_abs
    }
    by_alert_type = {
        at: _stats_from(by_alert_abs[at], by_alert_sq[at], sum(by_alert_cov[at]))
        for at in by_alert_abs
    }

    # Macro-average: collapse each regime to its mean error/coverage first,
    # then average regimes equally.
    n_regimes = len(by_regime)
    if n_regimes:
        regime_maes: list[float] = []
        regime_mses: list[float] = []
        regime_covs: list[float] = []
        for ticks in by_regime.values():
            k = len(ticks)
            regime_maes.append(sum(e for e, _sq, _w in ticks) / k)
            regime_mses.append(sum(sq for _e, sq, _w in ticks) / k)
            regime_covs.append(sum(w for _e, _sq, w in ticks) / k)
        per_regime = RecoveryStats(
            n=n_regimes,
            mae_min=sum(regime_maes) / n_regimes,
            rmse_min=(sum(regime_mses) / n_regimes) ** 0.5,
            iqr_coverage=sum(regime_covs) / n_regimes,
        )
    else:
        per_regime = RecoveryStats(n=0, mae_min=None, rmse_min=None, iqr_coverage=None)

    return RecoveryResult(
        overall=overall,
        per_regime=per_regime,
        by_route=by_route,
        by_alert_type=by_alert_type,
        excluded_schedule=excluded_schedule,
    )


def recovery_metrics(
    predictions: list[PredictionRecord],
    transitions: list[TransitionRecord],
) -> RecoveryResult:
    """Grade recovery_minutes against the HMM's OWN regime transitions (the
    filter's argmax flips). Self-consistent — a sanity check, not an independent
    validation. See momentarily-9bm and independent_recovery_metrics."""
    exits: dict[tuple[str, int], int] = {}
    for t in transitions:
        exits[(t.route, t.regime_entered_at)] = t.exited_at

    def exit_for(p: PredictionRecord) -> tuple[int | None, tuple[str, int]]:
        key = (p.route, p.regime_entered_at)
        return exits.get(key), key

    return _grade_recovery(predictions, exit_for)


def independent_recovery_metrics(
    predictions: list[PredictionRecord],
    disruptions: Sequence[Disruption],
) -> RecoveryResult:
    """Grade recovery_minutes against trip-updates-derived actual recovery — an
    INDEPENDENT truth (real trains running), unlike recovery_metrics which grades
    against the model's own argmax. A prediction is matched to the disruption
    interval [start_tick, recovered_tick) covering its tick. Same exclusions
    (normal / indeterminate / schedule). Truth is service LEVEL, a strong proxy,
    not service quality. See momentarily-xum / up0."""
    by_route: dict[str, list[Disruption]] = {}
    for d in disruptions:
        by_route.setdefault(d.route, []).append(d)
    for lst in by_route.values():
        lst.sort(key=lambda d: d.start_tick)

    def exit_for(p: PredictionRecord) -> tuple[int | None, tuple[str, int]]:
        for d in by_route.get(p.route, []):
            if d.start_tick <= p.ts < d.recovered_tick:
                return d.recovered_tick, (p.route, d.start_tick)
        return None, ("", 0)

    return _grade_recovery(predictions, exit_for)


def _stats_from(
    abs_errors: list[float], sq_errors: list[float], covered: int
) -> RecoveryStats:
    n = len(abs_errors)
    if n == 0:
        return RecoveryStats(n=0, mae_min=None, rmse_min=None, iqr_coverage=None)
    mae = sum(abs_errors) / n
    rmse = (sum(sq_errors) / n) ** 0.5
    return RecoveryStats(n=n, mae_min=mae, rmse_min=rmse, iqr_coverage=covered / n)


# --- Eval assembly + publish ---


def _calibration_as_dicts(
    calibrations: list[CalibrationResult],
) -> list[dict[str, Any]]:
    return [
        {
            "horizon_min": c.horizon_min,
            "n": c.n,
            "brier": c.brier,
            "brier_persistence": c.brier_persistence,
            "brier_climatology": c.brier_climatology,
            "bss_persistence": c.bss_persistence,
            "bss_climatology": c.bss_climatology,
            "excluded_schedule": c.excluded_schedule,
            "bins": [
                {
                    "bin_lo": b.bin_lo,
                    "bin_hi": b.bin_hi,
                    "n": b.n,
                    "mean_pred": b.mean_pred,
                    "mean_outcome": b.mean_outcome,
                }
                for b in c.bins
            ],
        }
        for c in calibrations
    ]


def _recovery_as_dict(recovery: RecoveryResult) -> dict[str, Any]:
    return {
        "overall": _stats_as_dict(recovery.overall),
        "per_regime": _stats_as_dict(recovery.per_regime),
        "by_route": {r: _stats_as_dict(s) for r, s in recovery.by_route.items()},
        "by_alert_type": {
            at: _stats_as_dict(s) for at, s in recovery.by_alert_type.items()
        },
        "excluded_schedule": recovery.excluded_schedule,
    }


def build_eval(
    predictions: list[PredictionRecord],
    transitions: list[TransitionRecord],
    *,
    window_start: int,
    window_end: int,
) -> dict[str, Any]:
    calibrations = [calibrate(predictions, h) for h in HORIZONS_MIN]
    recovery = recovery_metrics(predictions, transitions)

    # Per-params-version segment: the full-window metrics mix every params
    # version active during the window, which dilutes (or masks) the effect of
    # the latest retrain. The pipeline is prequential — params are always
    # trained on data strictly before the prediction — so this is isolation of
    # the current model's performance, not a leakage guard. Empty/None when no
    # prediction carries a version tag (pre-vk0.5 JSONL). See momentarily-vk0.5.
    latest_version = max((p.params_version for p in predictions), default=0)
    current_params: dict[str, Any] | None = None
    if latest_version > 0:
        current = [p for p in predictions if p.params_version == latest_version]
        current_recovery = recovery_metrics(current, transitions)
        current_params = {
            "trained_at": latest_version,
            "n_predictions": len(current),
            "calibration": _calibration_as_dicts(
                [calibrate(current, h) for h in HORIZONS_MIN]
            ),
            "recovery": _recovery_as_dict(current_recovery),
        }

    return {
        "generated_at": int(datetime.now(UTC).timestamp()),
        "window": {"start": window_start, "end": window_end},
        "predictions_seen": len(predictions),
        "transitions_seen": len(transitions),
        "current_params": current_params,
        "calibration": _calibration_as_dicts(calibrations),
        "recovery": _recovery_as_dict(recovery),
    }


def _stats_as_dict(s: RecoveryStats) -> dict[str, Any]:
    return {
        "n": s.n,
        "mae_min": s.mae_min,
        "rmse_min": s.rmse_min,
        "iqr_coverage": s.iqr_coverage,
    }


def publish_eval(client: S3Client, bucket: str, eval_doc: dict[str, Any]) -> None:
    client.put_object(
        Bucket=bucket,
        Key=EVAL_KEY,
        Body=json.dumps(eval_doc).encode(),
        ContentType="application/json",
        CacheControl="public, max-age=300, s-maxage=900",
    )


def load_transition_matrices(client: S3Client, bucket: str) -> dict[str, Any]:
    """Pull the per-route 3x3 transition matrices out of state/params.json.

    Bundled into calibration.json so a browser-only viz can draw the transition
    heatmap without LIST/credentialed access to state/. trained_at is the params
    version the matrices came from; routes maps route -> matrix in STATES order.
    Empty (trained_at=None) before the first weekly train.
    """
    try:
        body = client.get_object(Bucket=bucket, Key=PARAMS_KEY)["Body"].read()
        params: dict[str, Any] = json.loads(body)
    except Exception:
        return {"trained_at": None, "states": list(STATES), "routes": {}}
    raw_routes: dict[str, Any] = params.get("routes") or {}
    routes: dict[str, Any] = {
        route: p["transition"]
        for route, p in raw_routes.items()
        if isinstance(p.get("transition"), list) and len(p["transition"]) == len(STATES)
    }
    return {
        "trained_at": params.get("trained_at"),
        "states": list(STATES),
        "routes": routes,
    }


def build_calibration(
    eval_doc: dict[str, Any], transition_matrices: dict[str, Any]
) -> dict[str, Any]:
    """Compact public subset of eval.json for the hosted viz Models tab.

    Keeps the window-aggregate reliability bins, Brier/skill per horizon, and
    overall + per-regime recovery, plus the transition matrices. Drops the heavy
    breakdowns (current_params, recovery.by_route, recovery.by_alert_type) that
    multiply by route/alert-type/params-version — those stay in eval.json.
    """
    return {
        "generated_at": eval_doc["generated_at"],
        "window": eval_doc["window"],
        "predictions_seen": eval_doc["predictions_seen"],
        "transitions_seen": eval_doc["transitions_seen"],
        "calibration": eval_doc["calibration"],
        "recovery": {
            "overall": eval_doc["recovery"]["overall"],
            "per_regime": eval_doc["recovery"]["per_regime"],
        },
        "transition_matrices": transition_matrices,
    }


def publish_calibration(
    client: S3Client, bucket: str, calibration_doc: dict[str, Any]
) -> None:
    client.put_object(
        Bucket=bucket,
        Key=CALIBRATION_KEY,
        Body=json.dumps(calibration_doc).encode(),
        ContentType="application/json",
        CacheControl="public, max-age=300, s-maxage=900",
    )


# --- CLI ---


def build_independent_recovery(
    client: S3Client,
    predictions: list[PredictionRecord],
    start_date: date,
    end_date: date,
) -> dict[str, Any] | None:
    """Load the trip-updates service metric, derive independent disruptions, and
    grade recovery_minutes against them — a recovery truth independent of the
    HMM's own argmax. Returns None until the archive accumulates (the metric
    ships archive-first; ~2 weeks before the baseline is trustworthy). A load
    failure is non-fatal. See momentarily-xum / up0."""
    from training.load_r2 import (
        build_service_series,
        compute_baseline,
        derive_actual_recovery,
        fetch_trip_update_metrics,
    )

    try:
        bodies = fetch_trip_update_metrics(
            start_date=start_date, end_date=end_date, client=client
        )
    except Exception as exc:
        print(f"recovery_independent: trip-updates load failed ({exc})")
        return None
    if not bodies:
        return None
    series = build_service_series(bodies)
    baseline = compute_baseline(series)
    disruptions = derive_actual_recovery(series, baseline)
    result = independent_recovery_metrics(predictions, disruptions)
    return {
        **_recovery_as_dict(result),
        "truth_source": "trip_updates_service_level",
        "n_disruptions": len(disruptions),
        "n_baseline_cells": len(baseline),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Self-grading job")
    parser.add_argument("--days", type=int, default=7, help="window length in days")
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="print eval doc instead of writing to R2",
    )
    args = parser.parse_args(argv)

    cfg: R2Config = load_config()
    client = make_client(cfg)

    today = datetime.now(UTC).date()
    start_date = today - timedelta(days=args.days - 1)
    window_end = int(datetime.now(UTC).timestamp())
    window_start = int(
        datetime(
            start_date.year, start_date.month, start_date.day, tzinfo=UTC
        ).timestamp()
    )

    predictions = load_predictions(client, cfg.bucket, start_date, today)
    transitions = load_transitions(client, cfg.bucket, start_date, today)
    eval_doc = build_eval(
        predictions, transitions, window_start=window_start, window_end=window_end
    )
    eval_doc["recovery_independent"] = build_independent_recovery(
        client, predictions, start_date, today
    )
    transition_matrices = load_transition_matrices(client, cfg.bucket)
    calibration_doc = build_calibration(eval_doc, transition_matrices)

    if args.no_publish:
        print(json.dumps(eval_doc, indent=2))
        print(json.dumps(calibration_doc, indent=2))
    else:
        publish_eval(client, cfg.bucket, eval_doc)
        publish_calibration(client, cfg.bucket, calibration_doc)
        print(
            f"published {EVAL_KEY} + {CALIBRATION_KEY}: "
            f"{len(predictions)} predictions, {len(transitions)} transitions"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
