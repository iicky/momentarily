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
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from momentarily.mapping import TRUTH_VERSION
from training.drift import unmapped_alert_type_drift
from training.provenance import code_provenance
from training.r2_client import R2Config, get_object_bytes, load_config, make_client

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
    # a real prediction for these rows. Defaults False so older JSONL still parses.
    recovery_indeterminate: bool = False
    # primary_alert_type at this tick. Defaults None for older JSONL. Lets the grader
    # segment calibration by cause.
    primary_alert_type: str | None = None
    # trained_at of the params.json active when this prediction was made.
    # 0 for bootstrap params or older JSONL.
    # Predictions are prequential (params are always trained on data strictly
    # before the prediction), so this is a version tag for segmentation, not a
    # leakage guard.
    params_version: int = 0
    # "schedule" rows are deterministic planned-resume lookups, not HMM dwell
    # estimates — excluded from calibration/recovery grading (graded for
    # schedule adherence elsewhere). None for JSONL written before schedule
    # recovery shipped; treated as "hmm".
    recovery_source: str | None = None
    # The published movement-primary current-state condition + its source at this
    # tick (the alert-shadow is `condition` above). None for JSONL written before
    # escalation-arm grading shipped; the review scores movement escalations —
    # disrupted where the alert feed read normal — against later alerts.
    published_condition: str | None = None
    condition_source: str | None = None
    # When the published movement regime was entered. 0 for JSONL written before
    # the movement regime clock shipped. Elapsed time in the regime is what the
    # movement dwell curve is conditioned on, so a grader cannot reconstruct the
    # published forecast without it.
    movement_regime_entered_at: int = 0

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
            published_condition=raw.get("published_condition"),
            condition_source=raw.get("condition_source"),
            movement_regime_entered_at=int(raw.get("movement_regime_entered_at") or 0),
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
    # primary_alert_type when prev_state began. None for older records or when no alert
    # was active at regime start. Phase 2 segments dwell quantiles on this.
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


@dataclass(frozen=True)
class MovementTransitionRecord:
    """A committed movement-regime change, at route or segment scope.

    Mirrors the Worker's MovementTransitionRecord. The clock fields carry the
    same names as TransitionRecord so both streams feed one dwell-fitting path;
    `scope` and `key` say which cell moved.
    """

    ts: int
    scope: str
    key: str
    route: str
    prev_state: str
    new_state: str
    regime_entered_at: int
    exited_at: int
    dwell_sec: int

    @classmethod
    def from_json(cls, raw: dict[str, Any]) -> MovementTransitionRecord:
        return cls(
            ts=int(raw["ts"]),
            scope=str(raw["scope"]),
            key=str(raw["key"]),
            route=str(raw["route"]),
            prev_state=str(raw["prev_state"]),
            new_state=str(raw["new_state"]),
            regime_entered_at=int(raw["regime_entered_at"]),
            exited_at=int(raw["exited_at"]),
            dwell_sec=int(raw["dwell_sec"]),
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
    body = get_object_bytes(client, bucket, key).decode()
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def _read_listed_jsonl(
    client: S3Client,
    bucket: str,
    prefixes: Iterable[str],
) -> list[dict[str, Any]]:
    """Read every key listed under the prefixes, in parallel."""
    keys: list[str] = []
    for prefix in prefixes:
        keys.extend(_list_keys(client, bucket, prefix))

    def fetch(k: str) -> list[dict[str, Any]]:
        return _read_jsonl(client, bucket, k)

    out: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        for rows in pool.map(fetch, keys):
            out.extend(rows)
    return out


def load_predictions(
    client: S3Client,
    bucket: str,
    start_date: date,
    end_date: date,
) -> list[PredictionRecord]:
    rows = _read_listed_jsonl(
        client,
        bucket,
        (f"v1/predictions/{d.isoformat()}/" for d in _date_range(start_date, end_date)),
    )
    return [PredictionRecord.from_json(r) for r in rows]


def load_transitions(
    client: S3Client,
    bucket: str,
    start_date: date,
    end_date: date,
) -> list[TransitionRecord]:
    rows = _read_listed_jsonl(
        client,
        bucket,
        (
            f"v1/regime_transitions/{d.isoformat()}/"
            for d in _date_range(start_date, end_date)
        ),
    )
    return [TransitionRecord.from_json(r) for r in rows]


def load_movement_transitions(
    client: S3Client,
    bucket: str,
    start_date: date,
    end_date: date,
    *,
    scope: str | None = None,
) -> list[MovementTransitionRecord]:
    """Committed movement-regime changes over the window. `scope` filters to
    'route' or 'segment'; None returns both."""
    rows = _read_listed_jsonl(
        client,
        bucket,
        (
            f"v1/movement_transitions/{d.isoformat()}/"
            for d in _date_range(start_date, end_date)
        ),
    )
    out = [MovementTransitionRecord.from_json(r) for r in rows]
    return [r for r in out if scope is None or r.scope == scope]


def open_regimes_from_predictions(
    predictions: list[PredictionRecord],
    *,
    window_end: int | None = None,
) -> dict[str, tuple[str, int]]:
    """Each route's regime still open at the end of the window, read off its
    newest prediction row: {route: (state, regime_entered_at)}.

    Keyed on the filter's own condition and regime clock, because these censored
    observations feed the dwell curves that same filter projects forward. The
    prediction stream carries every live route every tick, so unlike the
    transition stream it also covers routes that never changed state.

    `window_end` is the censoring boundary. Rows after it are ignored, because
    predictions load by whole-day prefix: a backdated run would otherwise read
    the regime open *now* rather than the one open at the boundary, and a route
    that flipped in between would be dropped instead of censored.

    Routes off the timetable are dropped: `not_scheduled` is not a regime the
    filter dwells in, and banking scheduled downtime as a long healthy run would
    inflate exactly the curve this is meant to correct. Rows with no regime clock
    (bootstrap ticks before the filter had state) are dropped for the same
    reason — a zero would censor from the epoch.
    """
    newest: dict[str, PredictionRecord] = {}
    for p in predictions:
        if window_end is not None and p.ts > window_end:
            continue
        prev = newest.get(p.route)
        if prev is None or p.ts > prev.ts:
            newest[p.route] = p
    return {
        route: (p.condition, p.regime_entered_at)
        for route, p in newest.items()
        if p.condition != "not_scheduled" and p.regime_entered_at > 0
    }


# Conditions that count as a live disruption. Everything else — `normal`,
# `not_scheduled`, and the movement arm's `unknown` — is not a disruption to
# grade: `unknown` in particular is an absence of reading, not evidence of calm.
NOT_NORMAL = ("disrupted", "suspended")


def published_arm(p: PredictionRecord) -> str:
    """The movement-primary condition consumers actually read.

    Falls back to the alert-shadow `condition` for rows archived before the
    published arm shipped. The two arms disagree materially, so which one a
    metric grades is a load-bearing choice, not a detail.
    """
    return p.published_condition or p.condition


# Conditions the movement arm can actually grade. `unknown` is excluded on
# purpose: movement had no reading, which is an absence of evidence rather than
# evidence of calm.
GRADEABLE_ARM = ("normal", "disrupted", "suspended")


def movement_truth_by_key(
    predictions: list[PredictionRecord],
) -> dict[tuple[str, int], str]:
    """(route, tick) -> published movement condition, over the ticks movement
    could actually call.

    Dense over gradeable ticks rather than sparse-over-disruptions, so pair it
    with `truth_default=None`: an `unknown` tick then drops out of the sample
    instead of scoring as a recovery the forecast never earned.
    """
    out: dict[tuple[str, int], str] = {}
    for p in predictions:
        arm = published_arm(p)
        if arm in GRADEABLE_ARM:
            out[(p.route, snap_tick(p.ts))] = arm
    return out


def published_condition_coverage(
    predictions: list[PredictionRecord],
) -> dict[str, Any]:
    """How much of the window the published arm could actually call.

    A tick with no movement reading publishes `unknown`; it is neither a
    detection opportunity nor evidence of calm, so any rate computed over a
    window that is largely unknown is not what it appears to be.

    `gradeable_share` counts only GRADEABLE_ARM, so an off-timetable route is
    excluded alongside `unknown`: a line that is not running is not a line the
    arm judged healthy, and folding it into the denominator's complement would
    overstate how much of the window was actually under test.
    """
    n = len(predictions)
    counts = Counter(published_arm(p) for p in predictions)
    gradeable = sum(counts.get(c, 0) for c in GRADEABLE_ARM)
    return {
        "n_ticks": n,
        "by_condition": dict(counts),
        "unknown_share": counts.get("unknown", 0) / n if n else None,
        "gradeable_share": gradeable / n if n else None,
    }


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
class StratumStats:
    """Calibration sliced to one subset of matched samples — used to localize
    where the persistence baseline beats the model. mean_pred vs mean_outcome is
    the sharpness/bias view: a forecast that under-shoots a near-certain outcome
    (low mean_pred, high mean_outcome) is exactly what loses to a hard
    persistence call on a sticky regime."""

    n: int
    brier: float | None
    brier_persistence: float | None
    bss_persistence: float | None
    mean_pred: float | None  # average forecast (sharpness)
    mean_outcome: float | None  # realized P(normal at T+horizon) in this subset
    # Rank discrimination within the stratum. The normal_now slice is the one
    # that matters: persistence is pinned at 1.0 there and cannot discriminate at
    # all, so this is the only number that says whether the model's p_normal
    # actually falls ahead of a route leaving normal.
    auc: float | None = None


def _empty_strata() -> dict[str, StratumStats]:
    return {}


@dataclass
class CalibrationResult:
    horizon_min: int
    n: int
    brier: float | None  # None when n=0 — distinguishes "no data" from "perfect"
    # Reference forecasts on the same matched samples. Persistence predicts the
    # current condition holds at T+horizon (the baseline to beat for a sticky
    # process on short horizons); climatology predicts the per-route base rate
    # of normal over the eval window (in-sample, the standard reference).
    # A raw Brier score is uninterpretable without these.
    brier_persistence: float | None
    brier_climatology: float | None
    # Brier skill scores: 1 − brier/brier_ref. Positive = beats the baseline.
    # None when the reference is 0 (baseline already perfect) or n=0.
    bss_persistence: float | None
    bss_climatology: float | None
    bins: list[ReliabilityBin]
    # Rank discrimination over all matched samples. Brier answers "how close are
    # the numbers"; this answers "are they pointed the right way at all". A
    # forecast can post a competitive Brier against a rare outcome while ranking
    # backwards, so neither number is interpretable without the other.
    # 0.5 = none, < 0.5 = anti-predictive.
    auc: float | None = None
    # Persistence loss decomposed by the current condition at T: "normal_now"
    # (persistence predicts 1.0 — the sticky-regime case that dominates the
    # corpus) vs "not_normal_now" (persistence predicts 0.0 — the recovery
    # forecast). Isolates which slice drags the overall BSS negative.
    by_current: dict[str, StratumStats] = field(default_factory=_empty_strata)
    # Schedule-recovery rows skipped as predictors — they're deterministic resume
    # lookups, not HMM forecasts, so grading them would flatter the model.
    excluded_schedule: int = 0


def _skill(brier: float | None, reference: float | None) -> float | None:
    if brier is None or reference is None or reference == 0.0:
        return None
    return 1.0 - brier / reference


def _auc(samples: list[tuple[float, float]]) -> float | None:
    """Rank-based AUC over (pred, outcome): P(pred | stayed > pred | left), ties
    counted as half. None when one class is absent — discrimination is undefined
    without both.

    This is the metric Brier cannot supply. When the outcome is rare (normal
    routes stay normal ~99.8% of the time over these horizons) Brier is dominated
    by the calibration-to-1 term, so a forecast that is *anti*-predictive — one
    that reads p_normal higher right before a route leaves normal — can score
    close to a well-behaved one and better than a hedged one. AUC separates them:
    0.5 is no discrimination, below 0.5 is backwards.

    Mann-Whitney U with midranks, so tied forecasts (a constant predictor, say)
    land exactly at 0.5 rather than being scored by input order.
    """
    pos = [pred for pred, outcome in samples if outcome == 1.0]
    neg = [pred for pred, outcome in samples if outcome != 1.0]
    if not pos or not neg:
        return None
    ordered = sorted(pos + neg)
    midrank: dict[float, float] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[i]:
            j += 1
        # 1-based ranks; ties share the average of the positions they span.
        midrank[ordered[i]] = (i + j) / 2.0 + 1.0
        i = j + 1
    rank_sum = sum(midrank[pred] for pred in pos)
    n_pos, n_neg = len(pos), len(neg)
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _stratum(samples: list[tuple[float, float, float]]) -> StratumStats:
    """Brier/persistence/sharpness/discrimination over (pred, persistence, outcome)."""
    n = len(samples)
    if n == 0:
        return StratumStats(0, None, None, None, None, None)
    brier = sum((pred - out) ** 2 for pred, _per, out in samples) / n
    persistence = sum((per - out) ** 2 for _pred, per, out in samples) / n
    return StratumStats(
        n=n,
        brier=brier,
        brier_persistence=persistence,
        bss_persistence=_skill(brier, persistence),
        mean_pred=sum(pred for pred, _per, _out in samples) / n,
        mean_outcome=sum(out for _pred, _per, out in samples) / n,
        auc=_auc([(pred, out) for pred, _per, out in samples]),
    )


def calibrate(
    predictions: list[PredictionRecord],
    horizon_min: int,
    *,
    truth_by_key: dict[tuple[str, int], str] | None = None,
    coverage_predictions: list[PredictionRecord] | None = None,
    truth_default: str | None = "normal",
) -> CalibrationResult:
    """Pair each prediction at T with the realized state at T + horizon_min.

    By default the outcome and persistence come from the model's own published
    condition — self-consistency. With `truth_by_key` (a (route, tick) -> state
    map) they come from that held-out truth instead, turning this into a temporal
    forecast-skill test.

    `truth_default` decides what a route-tick missing from that map means:

      - `"normal"` — absent is calm. Right for alert-clearance truth, which is
        sparse by construction: no alert on file *is* the evidence of calm.
      - `None` — absent is not gradeable, and the sample is dropped. Right for
        the movement arm, where a missing reading is `unknown`. Scoring those as
        calm would credit the forecast for every tick movement could not judge
        and make the arm look far calmer than the evidence supports.

    `coverage_predictions` supplies the T+horizon lookup index; it defaults to
    `predictions`. Pass the full stream when segmenting `predictions` (e.g. by
    params_version) so a forecast whose T+horizon lands in another segment isn't
    dropped for lack of a same-segment future row — the outcome is external."""
    # Index by (route, snapped_ts) so T+horizon lookup is O(1).
    by_key: dict[tuple[str, int], PredictionRecord] = {}
    for p in coverage_predictions if coverage_predictions is not None else predictions:
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

    use_truth = truth_by_key is not None
    for p in predictions:
        # Deterministic planned-resume lookup, not an HMM forecast — skip as a
        # predictor (it can still be a future outcome; condition is HMM-derived).
        if p.recovery_source == "schedule":
            excluded_schedule += 1
            continue
        cur = snap_tick(p.ts)
        future_key = (p.route, cur + horizon_sec)
        future = by_key.get(future_key)
        if future is None:
            continue
        if use_truth:
            assert truth_by_key is not None
            truth_future = truth_by_key.get(future_key, truth_default)
            truth_now = truth_by_key.get((p.route, cur), truth_default)
            if truth_future is None or truth_now is None:
                continue
            outcome = 1.0 if truth_future == "normal" else 0.0
            persistence = 1.0 if truth_now == "normal" else 0.0
        else:
            outcome = 1.0 if future.condition == "normal" else 0.0
            persistence = 1.0 if p.condition == "normal" else 0.0
        pred: float = getattr(p, pred_field)
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

    # Split by current condition: persistence == 1.0 iff the route is normal now.
    samples = [(pred, per, out) for pred, per, out, _route in matched]
    by_current = {
        "normal_now": _stratum([s for s in samples if s[1] == 1.0]),
        "not_normal_now": _stratum([s for s in samples if s[1] != 1.0]),
    }

    return CalibrationResult(
        horizon_min=horizon_min,
        n=n,
        brier=brier,
        brier_persistence=brier_persistence,
        brier_climatology=brier_climatology,
        bss_persistence=_skill(brier, brier_persistence),
        bss_climatology=_skill(brier, brier_climatology),
        auc=_auc([(pred, out) for pred, _per, out, _route in matched]),
        bins=bins,
        by_current=by_current,
        excluded_schedule=excluded_schedule,
    )


def prequential_calibration(
    predictions: list[PredictionRecord],
    truth_by_key: dict[tuple[str, int], str],
    *,
    severity_floor: int,
    min_samples: int = 200,
) -> dict[str, Any]:
    """Temporal forecast skill against held-out alert-clearance truth: p_normal_in_H
    made at T graded vs the alert state at T+H, across every horizon. The truth map
    is whatever `truth_by_key` carries; `severity_floor` only labels the output.

    This is the primary yardstick once movement (and, later, assigned_n) feed the
    HMM — instantaneous agreement is no longer independent, but a forecast made H
    ticks early is still out-of-sample in time. The truth is alert clearance, which
    is itself a model input, so the independence here is temporal only, not signal.

    Segmented by params_version so a pre-movement fit is never pooled with the
    post-movement one; a version with fewer than `min_samples` matched pairs at the
    primary (shortest) horizon is reported but flagged low_sample, not diluted.
    """

    def _blocks(preds: list[PredictionRecord]) -> list[CalibrationResult]:
        # Coverage against the full stream so a forecast whose T+H lands in another
        # params-version segment is still graded (its outcome is external truth).
        return [
            calibrate(
                preds, h, truth_by_key=truth_by_key, coverage_predictions=predictions
            )
            for h in HORIZONS_MIN
        ]

    by_version: dict[str, Any] = {}
    for version in sorted({p.params_version for p in predictions}):
        seg = [p for p in predictions if p.params_version == version]
        cals = _blocks(seg)
        primary_n = cals[0].n if cals else 0
        by_version[str(version)] = {
            "n_predictions": len(seg),
            "n_matched_primary": primary_n,
            "low_sample": primary_n < min_samples,
            "calibration": _calibration_as_dicts(cals),
        }

    return {
        "truth_source": "alert_feed_clearance",
        "severity_floor": severity_floor,
        "independence": (
            "temporal only — alert truth is a model input, so a forecast H ticks "
            "ahead is out-of-sample but instantaneous agreement is not"
        ),
        "horizons_min": list(HORIZONS_MIN),
        "min_samples": min_samples,
        "overall": _calibration_as_dicts(_blocks(predictions)),
        "by_params_version": by_version,
    }


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
    # planned-work regimes dominate MAE. n = number of regimes.
    per_regime: RecoveryStats = field(
        default_factory=lambda: RecoveryStats(
            n=0, mae_min=None, rmse_min=None, iqr_coverage=None
        )
    )
    by_route: dict[str, RecoveryStats] = field(
        default_factory=lambda: {}  # noqa: PIE807
    )
    # Recovery accuracy segmented by the prediction-tick's primary_alert_type. Surfaces
    # whether cause-conditioned dwell quantiles are actually tightening the interval per
    # cause. Predictions with no alert type are omitted from this breakdown.
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

# Which condition stream decides a route is disrupted at a given tick. The arm
# has to match the truth's frame: HMM-argmax truth is keyed on the filter's own
# regimes, so it must gate on the filter's condition or the regime key it looks
# up will describe a different regime than the one being graded. A truth derived
# outside the model carries no such coupling and gates on the published movement
# arm, which is what consumers actually read.
ConditionArm = Callable[["PredictionRecord"], str]


def _grade_recovery(
    predictions: list[PredictionRecord],
    exit_for: ExitResolver,
    *,
    arm: ConditionArm,
) -> RecoveryResult:
    """Shared recovery grading: for every prediction made during a disruption that
    subsequently ended, compare recovery_minutes against actual remaining time and
    check IQR coverage. Each prediction-tick is one grading sample. `exit_for`
    supplies the actual recovery time and the regime key to group by; `arm`
    supplies the condition stream that decides which ticks are disruptions."""
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
        # them so the metric reflects actual recoveries. `unknown` and
        # `not_scheduled` are skipped for a different reason: they are the absence
        # of a reading, so there is no disruption to time.
        if arm(p) not in NOT_NORMAL:
            continue
        # Indeterminate rows are clamped, not predicted — including them would
        # bias MAE toward the clamp ceiling.
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
        by_regime.setdefault(regime_key, []).append((err, sq_err, 1 if within else 0))
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
    validation. See independent_recovery_metrics.

    Gates on the alert-shadow `condition`, not the published movement arm,
    because the exit lookup is keyed on the filter's own regime clock: gating on
    a different arm would select ticks whose regime key describes an unrelated
    regime. This grades the shadow, and only the shadow."""
    exits: dict[tuple[str, int], int] = {}
    for t in transitions:
        exits[(t.route, t.regime_entered_at)] = t.exited_at

    def exit_for(p: PredictionRecord) -> tuple[int | None, tuple[str, int]]:
        key = (p.route, p.regime_entered_at)
        return exits.get(key), key

    return _grade_recovery(predictions, exit_for, arm=lambda p: p.condition)


def independent_recovery_metrics(
    predictions: list[PredictionRecord],
    disruptions: Sequence[Disruption],
) -> RecoveryResult:
    """Grade recovery_minutes against trip-updates-derived actual recovery — an
    INDEPENDENT truth (real trains running), unlike recovery_metrics which grades
    against the model's own argmax. A prediction is matched to the disruption
    interval [start_tick, recovered_tick) covering its tick. Truth is service
    LEVEL, a strong proxy, not service quality.

    Gates on the published movement arm: the truth is derived outside the model,
    so nothing forces the shadow's frame here, and the movement arm is the one
    consumers read.

    Note the two sides measure different things. This truth counts trains in
    service; the movement arm times how fast they advance between stations. A
    line running a full fleet at crawl speed moves the arm and not the truth, and
    a thinned but free-flowing line does the reverse."""
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

    return _grade_recovery(predictions, exit_for, arm=published_arm)


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
            "auc": c.auc,
            "excluded_schedule": c.excluded_schedule,
            "by_current": {
                stratum: {
                    "n": s.n,
                    "brier": s.brier,
                    "brier_persistence": s.brier_persistence,
                    "bss_persistence": s.bss_persistence,
                    "mean_pred": s.mean_pred,
                    "mean_outcome": s.mean_outcome,
                    "auc": s.auc,
                }
                for stratum, s in c.by_current.items()
            },
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


def recovery_as_dict(recovery: RecoveryResult, *, graded_arm: str) -> dict[str, Any]:
    """Serialize a recovery result, tagged with the condition arm it graded.

    `graded_arm` is required rather than defaulted: the alert-shadow and the
    movement arm produce materially different numbers, and an untagged block
    reads as the product grade whichever one it happens to be.
    """
    return {
        "graded_arm": graded_arm,
        "overall": _stats_as_dict(recovery.overall),
        "per_regime": _stats_as_dict(recovery.per_regime),
        "by_route": {r: _stats_as_dict(s) for r, s in recovery.by_route.items()},
        "by_alert_type": {
            at: _stats_as_dict(s) for at, s in recovery.by_alert_type.items()
        },
        "excluded_schedule": recovery.excluded_schedule,
    }


# Arm labels for the published eval doc.
SHADOW_ARM_LABEL = "condition (alert-shadow)"
MOVEMENT_ARM_LABEL = "published_condition (movement-primary)"


def build_eval(
    predictions: list[PredictionRecord],
    transitions: list[TransitionRecord],
    *,
    window_start: int,
    window_end: int,
) -> dict[str, Any]:
    calibrations = [calibrate(predictions, h) for h in HORIZONS_MIN]
    # The same forecast graded against the arm consumers actually read. The
    # shadow block above answers "does the filter predict its own alert-driven
    # label"; this one answers "does it predict what movement will say", which is
    # the question the product surface poses.
    movement_truth = movement_truth_by_key(predictions)
    movement_calibrations = [
        calibrate(predictions, h, truth_by_key=movement_truth, truth_default=None)
        for h in HORIZONS_MIN
    ]
    recovery = recovery_metrics(predictions, transitions)

    # Per-params-version segment: the full-window metrics mix every params
    # version active during the window, which dilutes (or masks) the effect of
    # the latest retrain. The pipeline is prequential — params are always
    # trained on data strictly before the prediction — so this is isolation of
    # the current model's performance, not a leakage guard. Empty/None when no
    # prediction carries a version tag (older JSONL).
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
            "recovery": recovery_as_dict(current_recovery, graded_arm=SHADOW_ARM_LABEL),
        }

    return {
        "generated_at": int(datetime.now(UTC).timestamp()),
        "provenance": code_provenance(),
        # Truth-definition version in effect. `calibration` and `recovery` grade
        # the alert-shadow `condition` against itself — self-consistency, not the
        # severity truth. `calibration_movement` and `recovery_independent` grade
        # the movement arm consumers actually read; every block carries the arm
        # it graded, because the two disagree materially.
        "truth_version": TRUTH_VERSION,
        "window": {"start": window_start, "end": window_end},
        "predictions_seen": len(predictions),
        "transitions_seen": len(transitions),
        "current_params": current_params,
        "calibration": _calibration_as_dicts(calibrations),
        "calibration_arm": SHADOW_ARM_LABEL,
        "calibration_movement": {
            "graded_arm": MOVEMENT_ARM_LABEL,
            "coverage": published_condition_coverage(predictions),
            "horizons": _calibration_as_dicts(movement_calibrations),
        },
        "recovery": recovery_as_dict(recovery, graded_arm=SHADOW_ARM_LABEL),
        "drift": {"unmapped_alert_type": unmapped_alert_type_drift(predictions)},
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
        "truth_version": eval_doc["truth_version"],
        "provenance": eval_doc["provenance"],
        "window": eval_doc["window"],
        "predictions_seen": eval_doc["predictions_seen"],
        "transitions_seen": eval_doc["transitions_seen"],
        "calibration": eval_doc["calibration"],
        "recovery": {
            "overall": eval_doc["recovery"]["overall"],
            "per_regime": eval_doc["recovery"]["per_regime"],
        },
        "drift": eval_doc["drift"],
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
    failure is non-fatal."""
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
        **recovery_as_dict(result, graded_arm=MOVEMENT_ARM_LABEL),
        "truth_source": "trip_updates_service_level",
        "n_disruptions": len(disruptions),
        "n_baseline_cells": len(baseline),
        # A zero graded n against a non-zero n_disruptions is the diagnostic
        # case: the arm and the truth never overlapped on the same route-tick.
        # Coverage separates "movement had no reading" from genuine disagreement.
        "coverage": published_condition_coverage(list(predictions)),
    }


def build_emission_drift(
    client: S3Client,
    bucket: str,
    predictions: Sequence[PredictionRecord],
    start: date,
    end: date,
) -> dict[str, Any]:
    """Drift of the emission channels vs the training reference in params.json.

    Builds the current window's per-(route, tod_bin) profile the same way the
    trainer built the stored reference (same masking), then scores PSI + flag
    deltas. Returns {available: False} when params predate the stored profile."""
    from training.drift import build_input_profile, emission_channel_drift
    from training.load_r2 import (
        build_tick_observations,
        fetch_objects,
        list_alert_keys,
        presence_mask_from_predictions,
    )

    try:
        body = client.get_object(Bucket=bucket, Key=PARAMS_KEY)["Body"].read()
        params: dict[str, Any] = json.loads(body)
        reference: dict[str, Any] = params.get("input_profile") or {}
    except Exception as exc:
        print(f"emission-drift: params load failed ({exc})")
        return {"available": False}
    if not reference:
        return {"available": False}

    keys = list_alert_keys(client, bucket, start, end)
    bodies = fetch_objects(client, bucket, keys)
    mask = presence_mask_from_predictions(predictions)
    ticks = build_tick_observations(bodies, active_mask=mask)
    current = build_input_profile(ticks)
    return {"available": True, **emission_channel_drift(reference, current)}


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
    eval_doc["drift"]["emission_channels"] = build_emission_drift(
        client, cfg.bucket, predictions, start_date, today
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
