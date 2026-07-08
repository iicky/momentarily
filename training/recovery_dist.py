"""Recovery time as a distribution — grade the model's full predicted recovery
curve as one object instead of three horizon Brier scores.

Faithful numeric port of viz/lib/recovery_dist.ts (plus the predicted-curve
builder in viz/lib/dwell.ts) so the offline event scorecard scores the same
CRPS/PIT distribution the dashboard shows. Reuses training.dwell's dwell_cdf /
p_leave_by as the underlying conditional-survival primitives — keep all three
(this module, viz/lib/recovery_dist.ts, viz/lib/dwell.ts) in sync.

Each sample carries the model's recovery CDF (reconstructed from the
params.json dwell curve, sampled at every integer minute) and the realized
time-to-normal. We score with:
  - CRPS: integral of (F_pred(t) - 1{t >= actual})^2 dt over the curve, in
    minutes — one proper score over the whole curve. A climatology baseline
    (the empirical realized CDF used as everyone's forecast) gives a skill
    score.
  - PIT: F_pred(actual). Calibrated => uniform on [0,1]; the average
    (mean_pit) is a single readable "lean": <0.5 the model is too pessimistic
    (recoveries beat its forecast), >0.5 too optimistic.

Graded only on cases that did recover, so the predicted object is the timing
of recovery *given it recovers* — see predicted_recovery_curve.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Literal

from training.dwell import p_leave_by

# Recovery-time CDF horizon, in integer minutes — matches viz/lib/dwell.ts's
# RECOVERY_TMAX_MIN. predicted_recovery_curve samples every minute 0..this.
RECOVERY_TMAX_MIN = 240

# Curve display sampling, in minutes (viz/lib/recovery_dist.ts's GRID_STEP).
GRID_STEP = 5

# Headline recovery horizons reported alongside the full curve.
_HORIZON_MINUTES: tuple[int, ...] = (30, 60, 120)


def predicted_recovery_curve(
    elapsed_sec: float,
    curve_sec: list[int],
    tail_ll: list[float] | None = None,
) -> list[float]:
    """The model's recovery-time CDF for one prediction, sampled at every
    integer minute 0..RECOVERY_TMAX_MIN. This is P(resolved within t | already
    survived elapsed) — the timing of recovery *given the regime resolves*,
    NOT multiplied by the to-normal share. Mirrors viz/lib/dwell.ts's
    predictedRecoveryCurve."""
    return [
        p_leave_by(curve_sec, elapsed_sec, t * 60, tail_ll)
        for t in range(RECOVERY_TMAX_MIN + 1)
    ]


@dataclass(frozen=True)
class RecoveryDistSample:
    """One (predicted recovery CDF, realized duration) pair to grade."""

    pred_curve: list[float]  # F_pred at integer minutes 0..TMAX
    actual_min: float  # realized minutes until the route next returned to normal
    # Ties every tick from one disruption episode together (route + regime
    # onset) so scoring can weight per incident, not per forecast tick.
    regime_key: str


@dataclass(frozen=True)
class RecoveryWeighting:
    """CRPS/PIT under one weighting. Per-tick weights every prediction tick
    equally (operational forecast load — long incidents dominate); per-regime
    averages each episode's ticks, then weights episodes equally (incident-
    level quality)."""

    n: int  # ticks (per-tick) or distinct regimes (per-regime)
    mean_crps: float  # minutes, lower better
    baseline_crps: float  # climatology (empirical CDF) CRPS, minutes
    skill: float  # 1 - mean_crps/baseline_crps; >0 beats climatology
    mean_pit: float  # <0.5 pessimistic, >0.5 optimistic, 0.5 calibrated


@dataclass(frozen=True)
class RecoveryDistReport:
    """Full CRPS/PIT grading of a batch of recovery predictions, faithfully
    mirroring viz/lib/recovery_dist.ts's RecoveryDistReport. Build with
    recovery_dist_report."""

    # Per-tick headline kept at the top level for the curve view's back-compat.
    n: int
    mean_crps: float
    baseline_crps: float
    skill: float
    mean_pit: float
    per_tick: RecoveryWeighting  # mirrors the top-level fields, named explicitly
    per_regime: RecoveryWeighting  # each disruption episode weighted equally
    pit: list[int]  # 10-bin per-tick PIT histogram counts
    grid: list[int]  # minutes (display sampling)
    predicted_curve: list[float]  # mean F_pred at each grid minute
    empirical_curve: list[float]  # realized recovery CDF at each grid minute
    horizons: list[dict[str, float]]  # [{h, predicted, observed}, ...] at 30/60/120min


def _ecdf(sorted_asc: list[float], t: float) -> float:
    """Empirical CDF (fraction <= t) over a sorted array, via binary search."""
    if not sorted_asc:
        return 0.0
    return bisect_right(sorted_asc, t) / len(sorted_asc)


def _js_round(x: float) -> int:
    """Match JS Math.round: ties break toward +Infinity (half-up), unlike
    Python's round() (round-half-to-even)."""
    return math.floor(x + 0.5)


def _empty_weighting() -> RecoveryWeighting:
    nan = float("nan")
    return RecoveryWeighting(
        n=0, mean_crps=nan, baseline_crps=nan, skill=nan, mean_pit=nan
    )


def _empty_report(t_max: int) -> RecoveryDistReport:
    nan = float("nan")
    grid = list(range(0, t_max + 1, GRID_STEP))
    return RecoveryDistReport(
        n=0,
        mean_crps=nan,
        baseline_crps=nan,
        skill=nan,
        mean_pit=nan,
        per_tick=_empty_weighting(),
        per_regime=_empty_weighting(),
        pit=[0] * 10,
        grid=grid,
        predicted_curve=[0.0] * len(grid),
        empirical_curve=[0.0] * len(grid),
        horizons=[
            {"h": float(h), "predicted": nan, "observed": nan} for h in _HORIZON_MINUTES
        ],
    )


def recovery_dist_report(samples: list[RecoveryDistSample]) -> RecoveryDistReport:
    """Faithful port of recoveryDistReport (viz/lib/recovery_dist.ts): CRPS/PIT
    of each sample's predicted recovery CDF against its realized duration,
    reported per-tick and per-regime (equal weight per distinct regime_key)."""
    n = len(samples)
    if not n:
        return _empty_report(RECOVERY_TMAX_MIN)

    t_max = len(samples[0].pred_curve) - 1
    grid = list(range(0, t_max + 1, GRID_STEP))

    actuals_asc = sorted(s.actual_min for s in samples)

    def emp_at(t: float) -> float:
        return _ecdf(actuals_asc, t)

    pit = [0] * 10
    crps_sum = 0.0
    base_sum = 0.0
    pit_sum = 0.0
    pred_accum = [0.0] * len(grid)

    # Per-regime accumulators: each episode's per-tick scores are averaged
    # first, then episodes are weighted equally so one long incident can't
    # dominate.
    regime_crps: dict[str, float] = defaultdict(float)
    regime_base: dict[str, float] = defaultdict(float)
    regime_pit: dict[str, float] = defaultdict(float)
    regime_count: dict[str, int] = defaultdict(int)

    for s in samples:
        f = s.pred_curve
        y = s.actual_min
        crps = 0.0
        base = 0.0
        for t in range(t_max):
            ind = 1.0 if t >= y else 0.0
            dp = f[t] - ind
            crps += dp * dp
            db = emp_at(t) - ind
            base += db * db
        crps_sum += crps
        base_sum += base
        idx = min(t_max, max(0, _js_round(y)))
        u = f[idx]
        pit_sum += u
        pit[min(9, max(0, math.floor(u * 10)))] += 1
        for i, t in enumerate(grid):
            pred_accum[i] += f[t]

        regime_crps[s.regime_key] += crps
        regime_base[s.regime_key] += base
        regime_pit[s.regime_key] += u
        regime_count[s.regime_key] += 1

    mean_crps = crps_sum / n
    baseline_crps = base_sum / n
    per_tick = RecoveryWeighting(
        n=n,
        mean_crps=mean_crps,
        baseline_crps=baseline_crps,
        skill=1 - mean_crps / baseline_crps if baseline_crps > 0 else float("nan"),
        mean_pit=pit_sum / n,
    )

    # Average within each regime, then across regimes (equal weight per episode).
    regimes = len(regime_count)
    r_crps = sum(regime_crps[k] / regime_count[k] for k in regime_count)
    r_base = sum(regime_base[k] / regime_count[k] for k in regime_count)
    r_pit = sum(regime_pit[k] / regime_count[k] for k in regime_count)
    regime_baseline = r_base / regimes
    per_regime = RecoveryWeighting(
        n=regimes,
        mean_crps=r_crps / regimes,
        baseline_crps=regime_baseline,
        skill=1 - r_crps / r_base if regime_baseline > 0 else float("nan"),
        mean_pit=r_pit / regimes,
    )

    horizons: list[dict[str, float]] = []
    for h in _HORIZON_MINUTES:
        idx = grid.index(h) if h in grid else None
        predicted = pred_accum[idx] / n if idx is not None else float("nan")
        horizons.append({"h": float(h), "predicted": predicted, "observed": emp_at(h)})

    return RecoveryDistReport(
        n=n,
        mean_crps=mean_crps,
        baseline_crps=baseline_crps,
        skill=per_tick.skill,
        mean_pit=per_tick.mean_pit,
        per_tick=per_tick,
        per_regime=per_regime,
        pit=pit,
        grid=grid,
        predicted_curve=[v / n for v in pred_accum],
        empirical_curve=[emp_at(t) for t in grid],
        horizons=horizons,
    )


def report_as_dict(report: RecoveryDistReport) -> dict[str, Any]:
    return asdict(report)


# --- Verdict: read the calibration story off the PIT shape ---

# Minimum distinct incidents before the PIT shape is worth reading. Below this
# the histogram is noise, so the card says so rather than inventing a verdict.
VERDICT_MIN_INCIDENTS = 8


@dataclass(frozen=True)
class RecoveryVerdict:
    verdict: str
    explain: str
    tone: Literal["good", "warn", "muted"]
    # Surfaced when calibration shape and baseline skill tell different stories.
    warning: str | None = None


def recovery_verdict(result: RecoveryDistReport) -> RecoveryVerdict:
    """Derive the verdict from the actual PIT histogram shape (not a fixed
    sentence): left/right lean, U-shape (overconfident) vs hump
    (underconfident), with a small-n guard and a skill-vs-shape conflict
    check. Faithful port of recoveryVerdict (viz/lib/recovery_dist.ts)."""
    pit = result.pit
    total = sum(pit)
    if not total or math.isnan(result.mean_pit):
        return RecoveryVerdict(
            verdict="Not enough data yet",
            explain="No recovery forecasts scored in this window yet.",
            tone="muted",
        )

    incidents = result.per_regime.n
    if incidents < VERDICT_MIN_INCIDENTS:
        plural = "" if incidents == 1 else "s"
        return RecoveryVerdict(
            verdict="Inconclusive",
            explain=(
                f"Only {incidents} distinct incident{plural} recovered in this "
                "window — too few to read the calibration shape. Widen the "
                "window."
            ),
            tone="muted",
        )

    expected = total / len(pit)
    ends = pit[0] + pit[-1]
    mid = pit[3] + pit[4] + pit[5] + pit[6]
    lean = result.mean_pit
    off = abs(lean - 0.5)
    u_shape = ends > expected * 2 * 1.6  # extremes overweight → too narrow
    humped = mid > expected * 4 * 1.3  # middle overweight → too wide
    skill = result.per_regime.skill

    tone: Literal["good", "warn"]
    if u_shape and not humped:
        verdict = "Overconfident"
        explain = (
            "Outcomes pile up at the edges of the model's predicted range — "
            "its recovery intervals are too narrow, so reality lands outside "
            "them more often than it should."
        )
        tone = "warn"
    elif humped and not u_shape:
        verdict = "Underconfident"
        explain = (
            "Outcomes cluster in the middle of the predicted range — the "
            "intervals are wider than they need to be."
        )
        tone = "warn"
    elif off < 0.05:
        verdict = "Well calibrated"
        explain = (
            "Recovery outcomes fall about evenly across the model's "
            "predicted range — the timing odds are honest."
        )
        tone = "good"
    elif lean < 0.5:
        verdict = "Leans cautious"
        explain = "Lines tend to recover a little sooner than the model expects."
        tone = "warn"
    else:
        verdict = "Leans optimistic"
        explain = (
            "Lines tend to take a little longer to recover than the model expects."
        )
        tone = "warn"

    warning: str | None = None
    if tone == "good" and skill < 0:
        warning = (
            f"But it scores {abs(skill * 100):.0f}% worse than the "
            "dead-simple baseline — calibrated, yet no sharper than "
            "guessing the average. Calibration isn't skill."
        )
    elif tone == "warn" and skill >= 0.1:
        warning = (
            f"Even so, it beats the simple baseline by {skill * 100:.0f}% "
            "— miscalibrated but still more informative than guessing the "
            "average."
        )

    return RecoveryVerdict(verdict=verdict, explain=explain, tone=tone, warning=warning)


def verdict_as_dict(verdict: RecoveryVerdict) -> dict[str, Any]:
    return asdict(verdict)
