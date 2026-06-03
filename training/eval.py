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
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from training.r2_client import R2Config, load_config, make_client

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

TICK_SECONDS = 300  # publisher cron grid
HORIZONS_MIN = (30, 60, 120)
BIN_COUNT = 10  # 0.0-0.1, ..., 0.9-1.0
EVAL_KEY = "v1/eval.json"


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
    bins: list[ReliabilityBin]


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
    n = 0
    brier_sum = 0.0
    pred_field = f"p_normal_in_{horizon_min}min"

    for p in predictions:
        future_key = (p.route, snap_tick(p.ts) + horizon_sec)
        future = by_key.get(future_key)
        if future is None:
            continue
        outcome = 1.0 if future.condition == "normal" else 0.0
        pred: float = getattr(p, pred_field)
        n += 1
        brier_sum += (pred - outcome) ** 2
        idx = min(int(pred * BIN_COUNT), BIN_COUNT - 1)
        b = bins[idx]
        b.n += 1
        b.sum_pred += pred
        b.sum_outcome += outcome

    brier = brier_sum / n if n else None
    return CalibrationResult(horizon_min=horizon_min, n=n, brier=brier, bins=bins)


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


def recovery_metrics(
    predictions: list[PredictionRecord],
    transitions: list[TransitionRecord],
) -> RecoveryResult:
    """For every prediction made during a regime that subsequently ended, compare
    recovery_minutes (median) against actual remaining time, and check IQR
    coverage. Each prediction-tick is one grading sample."""
    # Map (route, regime_entered_at) -> exited_at via transition records.
    exits: dict[tuple[str, int], int] = {}
    for t in transitions:
        exits[(t.route, t.regime_entered_at)] = t.exited_at

    abs_errors: list[float] = []
    sq_errors: list[float] = []
    covered = 0
    by_route_abs: dict[str, list[float]] = {}
    by_route_sq: dict[str, list[float]] = {}
    by_route_cov: dict[str, list[int]] = {}
    by_alert_abs: dict[str, list[float]] = {}
    by_alert_sq: dict[str, list[float]] = {}
    by_alert_cov: dict[str, list[int]] = {}

    for p in predictions:
        # Indeterminate rows are clamped, not predicted — including them would
        # bias MAE toward the clamp ceiling. See momentarily-x25.
        if p.recovery_indeterminate:
            continue
        exited_at = exits.get((p.route, p.regime_entered_at))
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
    return RecoveryResult(
        overall=overall, by_route=by_route, by_alert_type=by_alert_type
    )


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


def build_eval(
    predictions: list[PredictionRecord],
    transitions: list[TransitionRecord],
    *,
    window_start: int,
    window_end: int,
) -> dict[str, Any]:
    calibrations = [calibrate(predictions, h) for h in HORIZONS_MIN]
    recovery = recovery_metrics(predictions, transitions)
    return {
        "generated_at": int(datetime.now(UTC).timestamp()),
        "window": {"start": window_start, "end": window_end},
        "predictions_seen": len(predictions),
        "transitions_seen": len(transitions),
        "calibration": [
            {
                "horizon_min": c.horizon_min,
                "n": c.n,
                "brier": c.brier,
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
        ],
        "recovery": {
            "overall": _stats_as_dict(recovery.overall),
            "by_route": {r: _stats_as_dict(s) for r, s in recovery.by_route.items()},
            "by_alert_type": {
                at: _stats_as_dict(s) for at, s in recovery.by_alert_type.items()
            },
        },
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


# --- CLI ---


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

    if args.no_publish:
        print(json.dumps(eval_doc, indent=2))
    else:
        publish_eval(client, cfg.bucket, eval_doc)
        print(
            f"published {EVAL_KEY}: "
            f"{len(predictions)} predictions, {len(transitions)} transitions"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
