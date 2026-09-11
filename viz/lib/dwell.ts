// Conditional survival over the empirical dwell curve. The math primitives
// (dwellCdf, loglogisticSurvival, mixtureSurvival, mixtureQuantile, atomParams,
// pLeaveBy) live in shared/dwell.ts — the single source shared with
// worker/src/dwell.ts and mirroring training/dwell.py. This module reconstructs
// the model's full recovery-time curve from the published dwell cells (instead
// of the three published checkpoints) and re-exports the primitives its callers
// and tests already import from here.

import { pLeaveBy } from "../../shared/dwell.ts";

export { dwellCdf, mixtureSurvival, mixtureQuantile, pLeaveBy } from "../../shared/dwell.ts";

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
