// Conditional survival over the empirical dwell curve — a faithful port of
// worker/src/dwell.ts (which mirrors training/dwell.py). Lets the dashboard
// reconstruct the model's full recovery-time curve from the params.json dwell
// cells, instead of the three published checkpoints. Keep in sync with the
// worker.

/**
 * Empirical P(dwell <= x) from the quantile curve, interpolated.
 *
 * The lower guard is strict (`x < curveSec[0]`, not `<=`), and a repeated
 * knot reads at the TOP of its flat run. `curveSec` is a quantile function,
 * so a run of equal knots is a point mass, and P(dwell <= x) at that value
 * has to cover the whole mass. The inclusive-`<=` version used to zero out
 * the CDF at a disrupted cell's own first knot (one publisher tick, its
 * majority outcome) — every one-tick episode graded with PIT=0.
 *
 * Numerically identical to the old formula on a strictly increasing curve;
 * only the flat-run case changes.
 */
export function dwellCdf(curveSec: number[], x: number): number {
  const k = curveSec.length;
  // Upper bound first so a degenerate flat curve (all samples equal) reads
  // as "outlived" at x == that value, not as P=0.
  if (x >= curveSec[k - 1]) return 1.0;
  if (x < curveSec[0]) return 0.0;
  // Largest index at or below x; scanning past equal knots lands on the top
  // of a flat run.
  let j = 0;
  for (let i = 0; i < k; i++) {
    if (curveSec[i] <= x) j = i;
    else break;
  }
  if (j >= k - 1) return 1.0;
  if (curveSec[j] === x) return j / (k - 1);
  const span = curveSec[j + 1] - curveSec[j];
  const frac = span === 0 ? 0.0 : (x - curveSec[j]) / span;
  return (j + frac) / (k - 1);
}

/** S(t) = 1 / (1 + (t/scale)^shape) for the log-logistic dwell tail. */
function loglogisticSurvival(t: number, shape: number, scale: number): number {
  if (t <= 0.0 || scale <= 0.0) return 1.0;
  return 1.0 / (1.0 + (t / scale) ** shape);
}

// --- Atom + truncated log-logistic mixture ---
//
// A dwell cell carrying (atomP, atomSec) publishes a point mass at atomSec
// mixed with a log-logistic left-truncated there:
//
//   S(t) = 1                                       t <  atomSec
//   S(t) = (1 - atomP) * S_ll(t) / S_ll(atomSec)    t >= atomSec
//
// so F(atomSec) == atomP exactly. That exactness is the whole fix: the
// quantile-curve representation can only reach the mass as F(atomSec + eps),
// never at the grid point itself, which is what graded the one-tick majority
// at PIT=0. The closed form has no such boundary.
//
// One useful property, worth keeping in mind when reading the call sites:
// for elapsed >= atomSec the atom cancels out of the conditional and the
// answer reduces to the plain log-logistic 1 - S_ll(e+h)/S_ll(e). The
// mixture only moves anything inside the first tick — which is exactly where
// a forecast made at regime onset lives, and where the old fit was worst.

/** S(t) for the atom + left-truncated log-logistic mixture. */
export function mixtureSurvival(
  t: number,
  shape: number,
  scale: number,
  atomP: number,
  atomSec: number,
): number {
  if (t < atomSec) return 1.0;
  const sTau = loglogisticSurvival(atomSec, shape, scale);
  if (sTau <= 0.0) return 0.0;
  return Math.max(
    0.0,
    Math.min(1.0, ((1.0 - atomP) * loglogisticSurvival(t, shape, scale)) / sTau),
  );
}

/**
 * Inverse CDF of the mixture: smallest t with F(t) >= u.
 *
 * Flat at atomSec for every u up to atomP — the atom is an interval of the
 * quantile function, not a point, which is why curveSec renders it as a run
 * of equal knots.
 */
export function mixtureQuantile(
  u: number,
  shape: number,
  scale: number,
  atomP: number,
  atomSec: number,
): number {
  const uu = Math.min(Math.max(u, 0.0), 1.0 - 1e-12);
  if (uu <= atomP) return atomSec;
  if (shape <= 0.0 || scale <= 0.0) return atomSec;
  const sTau = loglogisticSurvival(atomSec, shape, scale);
  const sTarget = ((1.0 - uu) * sTau) / (1.0 - atomP);
  if (sTarget <= 0.0) return Infinity;
  if (sTarget >= 1.0) return atomSec;
  return scale * ((1.0 - sTarget) / sTarget) ** (1.0 / shape);
}

/**
 * (shape, scale, atomP, atomSec) when a cell carries a usable mixture, else
 * null. The mixture needs BOTH a tail and an atom — `tailLl` alone is the
 * legacy unconditional fit, and an atom without a tail has nothing to spend
 * its remaining mass on. Anything partial falls back to the curve path
 * instead of inventing a component. Mirrors `_atom_params` in
 * training/dwell.py.
 */
function atomParams(
  tailLl: number[] | undefined,
  atom: { p: number; sec: number } | undefined,
): { shape: number; scale: number; atomP: number; atomSec: number } | null {
  if (atom === undefined || tailLl === undefined || tailLl.length < 2) return null;
  const { p: atomP, sec: atomSec } = atom;
  if (!(atomP > 0.0 && atomP < 1.0) || atomSec <= 0.0) return null;
  const [shape, scale] = tailLl;
  if (shape <= 0.0 || scale <= 0.0) return null;
  return { shape, scale, atomP, atomSec };
}

/**
 * P(dwell <= elapsed + horizon | dwell > elapsed). Past the last observed
 * quantile it extrapolates with the fitted log-logistic tail when `tailLl`
 * ([shape, scale]) is present, else a constant-hazard exponential patch — so
 * the curve keeps climbing instead of flatlining at the curve max.
 *
 * When `atom` ([p, sec] of a point mass at `sec`, mixed with `tailLl` as a
 * log-logistic left-truncated at `sec`) is present and paired with a usable
 * `tailLl`, this is fully analytic: it does not consult `curveSec` at all,
 * and needs no past-the-curve splice — the closed form covers every elapsed
 * time, including inside the curve.
 */
export function pLeaveBy(
  curveSec: number[],
  elapsedSec: number,
  horizonSec: number,
  tailLl?: number[],
  atom?: { p: number; sec: number },
): number {
  const mix = atomParams(tailLl, atom);
  if (mix !== null) {
    const sNow = mixtureSurvival(elapsedSec, mix.shape, mix.scale, mix.atomP, mix.atomSec);
    if (sNow <= 0.0) return 1.0;
    const sFut = mixtureSurvival(
      elapsedSec + horizonSec,
      mix.shape,
      mix.scale,
      mix.atomP,
      mix.atomSec,
    );
    return Math.max(0.0, Math.min(1.0, 1.0 - sFut / sNow));
  }
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
  atom?: { p: number; sec: number },
): number[] {
  const out = new Array<number>(RECOVERY_TMAX_MIN + 1);
  for (let t = 0; t <= RECOVERY_TMAX_MIN; t++) {
    out[t] = pLeaveBy(curveSec, elapsedSec, t * 60, tailLl, atom);
  }
  return out;
}
