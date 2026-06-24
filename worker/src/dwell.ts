/**
 * Conditional survival math over the empirical dwell curve.
 *
 * The trainer ships each (route, state[, alert_type]) dwell cell with
 * `curve_sec`: the dwell distribution as quantiles at evenly spaced
 * probabilities. Every recovery output must be conditioned on how long the
 * regime has already lasted — the unconditional quantiles are only correct at
 * elapsed=0, and for heavy-tailed dwells P(recover in 30min | disrupted 3h
 * already) is far below P(dwell <= 30min). See momentarily-vk0.1.
 *
 * Mirrors the reference implementation in training/dwell.py; keep in sync.
 */

export interface ConditionalRecovery {
  median_sec: number;
  q25_sec: number;
  q75_sec: number;
  recover_by_30: number;
  recover_by_60: number;
  recover_by_120: number;
}

/** Empirical P(dwell <= x) from the quantile curve, interpolated. */
export function dwellCdf(curveSec: number[], x: number): number {
  const k = curveSec.length;
  // Upper bound first so a degenerate flat curve (all samples equal) reads
  // as "outlived" at x == that value, not as P=0.
  if (x >= curveSec[k - 1]!) return 1.0;
  if (x <= curveSec[0]!) return 0.0;
  for (let i = 0; i < k - 1; i++) {
    const lo = curveSec[i]!;
    const hi = curveSec[i + 1]!;
    if (lo <= x && x <= hi) {
      const frac = hi === lo ? 0.0 : (x - lo) / (hi - lo);
      return (i + frac) / (k - 1);
    }
  }
  return 1.0; // unreachable for a monotone curve
}

/** Inverse of dwellCdf: dwell duration at cumulative probability p. */
function dwellQuantile(curveSec: number[], p: number): number {
  const k = curveSec.length;
  const pos = Math.min(Math.max(p, 0.0), 1.0) * (k - 1);
  const i = Math.min(Math.floor(pos), k - 2);
  const frac = pos - i;
  return curveSec[i]! + frac * (curveSec[i + 1]! - curveSec[i]!);
}

/**
 * All conditional recovery outputs for a regime that has survived elapsedSec.
 *
 * Returns null when the regime has outlived every observed dwell — the
 * empirical distribution says nothing about it and the caller should mark the
 * prediction indeterminate rather than fabricate a number.
 */
export function conditionalRecovery(
  curveSec: number[],
  elapsedSec: number,
): ConditionalRecovery | null {
  if (curveSec.length < 2) return null;
  const pElapsed = dwellCdf(curveSec, elapsedSec);
  if (pElapsed >= 1.0) return null;

  const remaining = (q: number): number => {
    const total = dwellQuantile(curveSec, pElapsed + q * (1.0 - pElapsed));
    return Math.max(0.0, total - elapsedSec);
  };
  const recoverBy = (horizonSec: number): number =>
    (dwellCdf(curveSec, elapsedSec + horizonSec) - pElapsed) / (1.0 - pElapsed);

  return {
    median_sec: remaining(0.5),
    q25_sec: remaining(0.25),
    q75_sec: remaining(0.75),
    recover_by_30: recoverBy(1800),
    recover_by_60: recoverBy(3600),
    recover_by_120: recoverBy(7200),
  };
}

/**
 * P(dwell <= elapsed+horizon | dwell > elapsed), with a tail extrapolation once
 * the regime has outlived every observed dwell instead of saturating at the
 * curve max. Unlike conditionalRecovery (which returns null past the curve and is
 * used for a recovery *time* that we won't fabricate), this keeps the conditional
 * exit *probability* meaningful in the long-lived tail.
 *
 * Past the curve the tail is the fitted log-logistic conditional survival when
 * `tailLl` ([shape, scale]) is supplied, else a constant-hazard exponential patch
 * read off the top segment. The log-logistic's decreasing hazard fits the heavy
 * dwell tail better, so a long-calm regime stays confident (gtq.5 Brier
 * backtest). The body stays empirical either way. Mirrors training/dwell.py.
 */
export function pLeaveBy(
  curveSec: number[],
  elapsedSec: number,
  horizonSec: number,
  tailLl?: number[],
): number {
  const k = curveSec.length;
  if (k < 2) return 0;
  const pElapsed = dwellCdf(curveSec, elapsedSec);
  if (pElapsed < 1.0) {
    return (dwellCdf(curveSec, elapsedSec + horizonSec) - pElapsed) / (1.0 - pElapsed);
  }
  if (tailLl !== undefined) {
    const [shape, scale] = tailLl as [number, number];
    const sNow = loglogisticSurvival(elapsedSec, shape, scale);
    if (sNow <= 0.0) return 1.0;
    const sFut = loglogisticSurvival(elapsedSec + horizonSec, shape, scale);
    return Math.max(0.0, Math.min(1.0, 1.0 - sFut / sNow));
  }
  // Outlived the curve: constant tail hazard from the top segment (the top
  // 1/(k-1) of mass is lost over its width), projected across the horizon.
  const seg = curveSec[k - 1]! - curveSec[k - 2]!;
  const lam = seg > 0 ? 1.0 / (k - 1) / seg : 1.0 / Math.max(1, curveSec[k - 1]!);
  return 1.0 - Math.exp(-Math.max(lam, 1e-12) * horizonSec);
}

/** S(t) = 1 / (1 + (t/scale)^shape) for the log-logistic dwell tail. */
function loglogisticSurvival(t: number, shape: number, scale: number): number {
  if (t <= 0.0 || scale <= 0.0) return 1.0;
  return 1.0 / (1.0 + (t / scale) ** shape);
}
