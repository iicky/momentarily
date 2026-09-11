/**
 * Conditional recovery outputs built on the shared dwell survival math.
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
 * tick (mixtureSurvival/mixtureQuantile in shared/dwell). pLeaveBy and
 * conditionalRecovery both take an optional trailing `atom` argument — when
 * it is supplied alongside `tailLl`, they switch to that closed form and
 * ignore `curve_sec`/the past-the-curve splice entirely. Without atom fields
 * both behave exactly as they did before this mixture was added.
 *
 * The survival primitives (dwellCdf, mixtureSurvival, mixtureQuantile,
 * atomParams, pLeaveBy) live in shared/dwell.ts, the single source shared with
 * viz/lib/dwell.ts and mirroring training/dwell.py. This module keeps only the
 * snapshot-facing rollup (conditionalRecovery) and re-exports the primitives
 * its callers already import from here.
 */

import {
  atomParams,
  dwellCdf,
  mixtureQuantile,
  mixtureSurvival,
  pLeaveBy,
} from '../../shared/dwell';

export { dwellCdf, mixtureQuantile, mixtureSurvival, pLeaveBy };

export interface ConditionalRecovery {
  median_sec: number;
  q25_sec: number;
  q75_sec: number;
  recover_by_30: number;
  recover_by_60: number;
  recover_by_120: number;
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
