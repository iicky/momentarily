/**
 * Conditional survival math over the empirical dwell curve.
 *
 * The trainer ships each (route, state[, alert_type]) dwell cell with
 * `curve_sec`: the dwell distribution as quantiles at evenly spaced
 * probabilities. Every recovery output must be conditioned on how long the
 * regime has already lasted — the unconditional quantiles are only correct at
 * elapsed=0, and for heavy-tailed dwells P(recover in 30min | disrupted 3h
 * already) is far below P(dwell <= 30min).
 *
 * A cell may also carry `atom_p`/`atom_sec`: a point mass at one publisher
 * tick, mixed with `tail_ll` refit as a log-logistic LEFT-TRUNCATED at that
 * tick (mixtureSurvival/mixtureQuantile below). pLeaveBy and
 * conditionalRecovery both take an optional trailing `atom` argument — when
 * it is supplied alongside `tailLl`, they switch to that closed form and
 * ignore `curve_sec`/the past-the-curve splice entirely. Without atom fields
 * both behave exactly as they did before this mixture was added.
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
  // Strict, not <=: curve[0] is one publisher tick for a disrupted cell, and
  // an inclusive guard here used to zero the CDF at exactly that tick — the
  // PIT=0 defect. x == curve[0] now falls through to the knot scan below.
  if (x < curveSec[0]!) return 0.0;
  // Largest index j with curve[j] <= x. curveSec is non-decreasing but not
  // necessarily strictly increasing (a mixture cell repeats the atom's tick
  // across every knot below its cumulative share), so this is a flat-run-
  // aware replacement for the old single-interval scan, not just a rename.
  let j = 0;
  for (let i = 1; i < k; i++) {
    if (curveSec[i]! <= x) j = i;
    else break;
  }
  if (j >= k - 1) return 1.0;
  if (curveSec[j] === x) return j / (k - 1); // inclusive at a knot / flat run
  const lo = curveSec[j]!;
  const hi = curveSec[j + 1]!;
  return (j + (x - lo) / (hi - lo)) / (k - 1);
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
 *
 * `tailLl`/`atom` are only consulted together: when both are present, the
 * mixture closed form (mixtureSurvival/mixtureQuantile) replaces the curve
 * entirely, including recover_by_X (routed through pLeaveBy, which applies
 * the same rule). Either missing falls back to the empirical curve, unchanged
 * from before this mixture was added.
 */
export function conditionalRecovery(
  curveSec: number[],
  elapsedSec: number,
  tailLl?: number[],
  atom?: { p: number; sec: number },
): ConditionalRecovery | null {
  const mix = atomParams(tailLl, atom);
  if (mix !== null) {
    const { shape, scale, atomP, atomSec } = mix;
    const fElapsed = 1.0 - mixtureSurvival(elapsedSec, shape, scale, atomP, atomSec);
    if (fElapsed >= 1.0) return null;
    const remaining = (q: number): number => {
      const total = mixtureQuantile(fElapsed + q * (1.0 - fElapsed), shape, scale, atomP, atomSec);
      return Math.max(0.0, total - elapsedSec);
    };
    return {
      median_sec: remaining(0.5),
      q25_sec: remaining(0.25),
      q75_sec: remaining(0.75),
      recover_by_30: pLeaveBy(curveSec, elapsedSec, 1800, tailLl, atom),
      recover_by_60: pLeaveBy(curveSec, elapsedSec, 3600, tailLl, atom),
      recover_by_120: pLeaveBy(curveSec, elapsedSec, 7200, tailLl, atom),
    };
  }

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
 * (shape, scale, atomP, atomSec) when a cell carries a USABLE mixture, else null.
 *
 * Presence is not enough. The mixture needs both halves and needs them valid:
 * tailLl alone is the legacy unconditional fit, an atom alone has nothing to
 * spend its remaining mass on, and a degenerate parameter silently produces a
 * different distribution rather than an error — atomSec <= 0 would apply the
 * point mass from t=0, atomP >= 1 divides by zero in mixtureQuantile, and a
 * non-positive shape/scale makes loglogisticSurvival return 1 everywhere so the
 * "mixture" flattens to a constant. Anything partial or out of range falls back
 * to the curve path instead of inventing a component.
 *
 * Mirrors training/dwell.py _atom_params exactly; keep in sync. The parity
 * fixture carries invalid-atom cases specifically to pin this fallback.
 */
function atomParams(
  tailLl?: number[],
  atom?: { p: number; sec: number },
): { shape: number; scale: number; atomP: number; atomSec: number } | null {
  if (atom === undefined || tailLl === undefined || tailLl.length < 2) return null;
  const atomP = atom.p;
  const atomSec = atom.sec;
  if (!(atomP > 0.0 && atomP < 1.0) || atomSec <= 0.0) return null;
  const shape = tailLl[0]!;
  const scale = tailLl[1]!;
  if (shape <= 0.0 || scale <= 0.0) return null;
  return { shape, scale, atomP, atomSec };
}

/**
 * P(dwell <= elapsed+horizon | dwell > elapsed), with a tail extrapolation once
 * the regime has outlived every observed dwell instead of saturating at the
 * curve max. Unlike conditionalRecovery (which returns null past the curve and is
 * used for a recovery *time* that we won't fabricate), this keeps the conditional
 * exit *probability* meaningful in the long-lived tail.
 *
 * Past the curve the tail is the fitted log-logistic conditional survival when
 * `tailLl` ([shape, scale]) is supplied, else a constant-hazard exponential
 * patch read off the top segment. The log-logistic's decreasing hazard fits the
 * heavy dwell tail better, so a long-calm regime stays confident (per the Brier
 * backtest). The body stays empirical either way.
 *
 * `atom` ({ p, sec }) activates the mixture closed form when paired with
 * `tailLl`: point mass `p` at tick `sec`, log-logistic left-truncated at `sec`
 * for the rest (mixtureSurvival). That fully replaces the curve for this call —
 * no past-the-curve splice, `curveSec` is not read on this path at all. Mirrors
 * training/dwell.py p_leave_by.
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
    const { shape, scale, atomP, atomSec } = mix;
    const sNow = mixtureSurvival(elapsedSec, shape, scale, atomP, atomSec);
    if (sNow <= 0.0) return 1.0;
    const sFut = mixtureSurvival(elapsedSec + horizonSec, shape, scale, atomP, atomSec);
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
  // Outlived the curve: constant tail hazard from the top segment (the top
  // 1/(k-1) of mass is lost over its width), projected across the horizon.
  const seg = curveSec[k - 1]! - curveSec[k - 2]!;
  const lam = seg > 0 ? 1.0 / (k - 1) / seg : 1.0 / Math.max(1, curveSec[k - 1]!);
  return 1.0 - Math.exp(-Math.max(lam, 1e-12) * horizonSec);
}

/** S(t) = 1 / (1 + (t/scale)^shape) for the log-logistic dwell tail. Also the
 * building block for mixtureSurvival's truncated tail below. */
function loglogisticSurvival(t: number, shape: number, scale: number): number {
  if (t <= 0.0 || scale <= 0.0 || shape <= 0.0) return 1.0;
  return 1.0 / (1.0 + (t / scale) ** shape);
}

/**
 * Mixture survival: P(dwell > t) under the atom + left-truncated log-logistic
 * mixture — a point mass `atomP` at `atomSec`, with the log-logistic tail
 * scaled so its own survival at `atomSec` lands exactly on the mixture's
 * (1 - atomP) there. S(atomSec) == 1 - atomP, so F(atomSec) == atomP exactly:
 * the atom is inclusive at its own location. Mirrors training/dwell.py
 * mixture_survival; keep in sync.
 */
export function mixtureSurvival(
  t: number,
  shape: number,
  scale: number,
  atomP: number,
  atomSec: number,
): number {
  if (t < atomSec) return 1.0;
  const sTau = loglogisticSurvival(atomSec, shape, scale);
  const sT = loglogisticSurvival(t, shape, scale);
  return Math.max(0.0, Math.min(1.0, (1.0 - atomP) * (sT / sTau)));
}

/**
 * Inverse of mixtureSurvival: the mixture quantile function for u in [0, 1).
 * u <= atomP lands on the atom itself; above it, the log-logistic tail is
 * inverted against the truncated target survival. Mirrors training/dwell.py
 * mixture_quantile; keep in sync.
 */
export function mixtureQuantile(
  u: number,
  shape: number,
  scale: number,
  atomP: number,
  atomSec: number,
): number {
  if (u <= atomP) return atomSec;
  const sTau = loglogisticSurvival(atomSec, shape, scale);
  const sTarget = ((1.0 - u) * sTau) / (1.0 - atomP);
  return scale * ((1.0 - sTarget) / sTarget) ** (1.0 / shape);
}
