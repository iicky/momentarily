// Conditional survival over the empirical dwell curve — a faithful port of
// worker/src/dwell.ts (which mirrors training/dwell.py). Lets the dashboard
// reconstruct the model's full recovery-time curve from the params.json dwell
// cells, instead of the three published checkpoints. Keep in sync with the
// worker.

/** Empirical P(dwell <= x) from the quantile curve, interpolated. */
export function dwellCdf(curveSec: number[], x: number): number {
  const k = curveSec.length;
  if (x >= curveSec[k - 1]) return 1.0;
  if (x <= curveSec[0]) return 0.0;
  for (let i = 0; i < k - 1; i++) {
    const lo = curveSec[i];
    const hi = curveSec[i + 1];
    if (lo <= x && x <= hi) {
      const frac = hi === lo ? 0.0 : (x - lo) / (hi - lo);
      return (i + frac) / (k - 1);
    }
  }
  return 1.0;
}

/** S(t) = 1 / (1 + (t/scale)^shape) for the log-logistic dwell tail. */
function loglogisticSurvival(t: number, shape: number, scale: number): number {
  if (t <= 0.0 || scale <= 0.0) return 1.0;
  return 1.0 / (1.0 + (t / scale) ** shape);
}

/**
 * P(dwell <= elapsed + horizon | dwell > elapsed). Past the last observed
 * quantile it extrapolates with the fitted log-logistic tail when `tailLl`
 * ([shape, scale]) is present, else a constant-hazard exponential patch — so the
 * curve keeps climbing instead of flatlining at the curve max.
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
  const seg = curveSec[k - 1] - curveSec[k - 2];
  const lam = seg > 0 ? 1.0 / (k - 1) / seg : 1.0 / Math.max(1, curveSec[k - 1]);
  return 1.0 - Math.exp(-Math.max(lam, 1e-12) * horizonSec);
}

export const RECOVERY_TMAX_MIN = 240;

/**
 * The model's recovery-time CDF for one prediction, sampled at every integer
 * minute 0..RECOVERY_TMAX_MIN. This is P(resolved within t | already survived
 * elapsed) — the timing of recovery *given the regime resolves*, NOT multiplied
 * by the to-normal share. We grade against cases that did recover, so the
 * apples-to-apples object is the conditional timing; whether a regime escalates
 * instead is a separate (competing-risks) question.
 */
export function predictedRecoveryCurve(
  elapsedSec: number,
  curveSec: number[],
  tailLl?: number[],
): number[] {
  const out = new Array<number>(RECOVERY_TMAX_MIN + 1);
  for (let t = 0; t <= RECOVERY_TMAX_MIN; t++) {
    out[t] = pLeaveBy(curveSec, elapsedSec, t * 60, tailLl);
  }
  return out;
}
