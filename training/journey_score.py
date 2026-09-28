"""Scoring for journey-time forecasts on the fixed QUANTILES grid.

Measurement contract: a forecast is a Quantiles mapping over
journey_types.QUANTILES; the realized outcome is SpanSample.seconds. CRPS is
approximated as the mean over the grid of twice the pinball loss — for a point
forecast (every quantile equal to p) this reduces exactly to |y - p|, because
the grid's mean quantile level is 0.5, so the schedule baseline's CRPS IS its
MAE and the two baselines are comparable on one number. PIT is the
piecewise-linear CDF implied by the grid evaluated at y, clamped to [0, 1].
Abstentions (None forecasts) are counted, never scored; every forecaster in a
run is handed the SAME samples, so n_scored + n_abstained is constant across a
gate run.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from training.journey_types import QUANTILES, Quantiles, SpanSample

# PIT histogram resolution: bin i counts pit values in [i/10, (i+1)/10), with
# pit == 1.0 folded into the last bin.
PIT_BINS = 10


def crps_from_quantiles(q: Quantiles, y: float) -> float:
    """Pinball-loss approximation of CRPS on the fixed grid.

    mean over tau in QUANTILES of 2 * pinball_tau(y, q[tau]). A point forecast
    (all quantiles equal to p) reduces to |y - p| since mean(QUANTILES) == 0.5:
    the schedule baseline's CRPS is its MAE.
    """
    total = 0.0
    for tau in QUANTILES:
        v = q[tau]
        diff = y - v
        total += 2.0 * (tau * diff if diff >= 0 else (tau - 1.0) * diff)
    return total / len(QUANTILES)


def pit(q: Quantiles, y: float) -> float:
    """Probability integral transform of y under the grid's implied CDF.

    Piecewise-linear between the (value, tau) knots; y below q05 -> 0.0, above
    q95 -> 1.0. A flat run of equal values maps y onto the midpoint of its
    tau range, so a degenerate point forecast puts its own value at 0.5.
    """
    values = [q[tau] for tau in QUANTILES]
    if y < values[0]:
        return 0.0
    if y > values[-1]:
        return 1.0
    # Indices whose value equals or straddles y.
    lo = 0
    while lo < len(values) and values[lo] < y:
        lo += 1
    if lo < len(values) and values[lo] == y:
        hi = lo
        while hi + 1 < len(values) and values[hi + 1] == y:
            hi += 1
        return (QUANTILES[lo] + QUANTILES[hi]) / 2.0
    # values[lo - 1] < y < values[lo]: interpolate.
    v0, v1 = values[lo - 1], values[lo]
    t0, t1 = QUANTILES[lo - 1], QUANTILES[lo]
    return t0 + (t1 - t0) * (y - v0) / (v1 - v0)


@dataclass(frozen=True)
class Report:
    """One forecaster's scorecard over one shared sample set."""

    name: str
    n_scored: int
    n_abstained: int
    crps_mean: float
    mae_median: float  # mean |y - q50|
    coverage_10_90: float  # share of scored y inside [q10, q90]
    pit_bins: tuple[int, ...]  # PIT_BINS counts, uniform iff calibrated

    def skill_vs(self, other: Report) -> float:
        """1 - self.crps_mean / other.crps_mean; positive means better."""
        return 1.0 - self.crps_mean / other.crps_mean


def score(name: str, pairs: Sequence[tuple[SpanSample, Quantiles | None]]) -> Report:
    """Aggregate one forecaster's (sample, forecast-or-abstain) pairs."""
    n_abstained = 0
    crps_sum = 0.0
    mae_sum = 0.0
    covered = 0
    bins = [0] * PIT_BINS
    n = 0
    for sample, q in pairs:
        if q is None:
            n_abstained += 1
            continue
        y = float(sample.seconds)
        n += 1
        crps_sum += crps_from_quantiles(q, y)
        mae_sum += abs(y - q[0.50])
        if q[0.10] <= y <= q[0.90]:
            covered += 1
        p = pit(q, y)
        bins[min(int(p * PIT_BINS), PIT_BINS - 1)] += 1
    return Report(
        name=name,
        n_scored=n,
        n_abstained=n_abstained,
        crps_mean=crps_sum / n if n else float("nan"),
        mae_median=mae_sum / n if n else float("nan"),
        coverage_10_90=covered / n if n else float("nan"),
        pit_bins=tuple(bins),
    )
