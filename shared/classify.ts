/**
 * The movement classifier's decision rule and its constants — the SINGLE source
 * of truth shared by worker/src/movement_state.ts (the live per-direction call)
 * and viz/lib/movement.ts (the offline confusion matrix over the archive). Both
 * import these; neither redeclares the thresholds. Mirrors classify_direction /
 * _binom_lower_tail in training/load_r2.py.
 */

// Classification-time prior strength in pseudo-trials — regularizes a single
// tick's advance fraction toward the cell baseline so a thin sample can't swing
// the call. Distinct from the trainer's advance-baseline prior strength, which
// anchors the HMM emission accumulated over the whole training window.
export const CLASSIFY_PRIOR_STRENGTH = 8;
// A direction reads disrupted when its posterior advance rate sits at/under this
// fraction of the cell's own baseline p0 — advancing at under half its normal
// rate. Baseline-relative, so shuttles and trunk lines are each judged against
// their own normal instead of one global cutoff.
export const DISRUPTED_RATIO = 0.5;
// A large posterior drop only reads disrupted when the low advance count is also
// statistically significant against the cell baseline (binomial lower tail at or
// under this). Guards degenerate-low baselines — a short shuttle advances ~0 even
// when healthy, so a normal zero-advance tick would otherwise flip disrupted.
export const CLASSIFY_ALPHA = 0.05;
// A cell whose baseline advance rate p0 sits under this floor is degenerate: it
// advances ~0 even in healthy operation (short shuttles, terminal/relay stops
// that dwell and reverse), so its 'normal' is near-zero advance and a zero-advance
// tick carries no disruption signal. Below the floor the disrupted determination
// is abstained (null / 'unknown') rather than run — a tail that would read
// 'significant' against that untrustworthy p0 is deliberately not credited. The
// cell can still read 'normal' when it is visibly advancing. Chosen from the live
// advance-baseline distribution (state/params.json movement_baseline, 211
// (route, direction, tod_bin) cells): median p0 0.94, p10 0.73 — trunk lines sit
// high. Only the H (Rockaway) shuttle's south cells fall low: p0 0.113 (tod0) and
// 0.145 (tod1), the lowest two in the fit; the next real cell up is M south
// overnight at 0.224. 0.15 and 0.2 abstain the same 2 cells and nothing else
// (0.9%); 0.25 wrongly pulls in M's overnight cell. 0.2 over 0.15 for headroom —
// 0.15 leaves only 0.005 above H's 0.145 cell (fragile to retrain drift), whereas
// 0.2 sits in the upper 0.145->0.224 gap: 0.055 above H, still 0.024 below M.
// Keyed on the cell's own p0, so it is one number shared by the direction call
// (movement_state.ts) and the segment call (segment_flow.ts).
export const CLASSIFY_P0_FLOOR = 0.2;
// advanced_n + stalled_n floor to make a cross-tick call.
export const MIN_MATCHED_TRIPS = 3;

// P(X <= k) for X ~ Binomial(n, p) via an iterative pmf sum. Exact for the
// tick-level counts here (n well under ~50) and free of special functions, so it
// mirrors 1:1 in Python/viz. p is the cell baseline p0, floored off 0 upstream.
export function binomLowerTail(k: number, n: number, p: number): number {
  if (k >= n) return 1;
  if (k < 0) return 0;
  const q = 1 - p;
  let pmf = q ** n; // P(X = 0)
  let cdf = pmf;
  for (let i = 0; i < k; i++) {
    pmf *= ((n - i) / (i + 1)) * (p / q);
    cdf += pmf;
  }
  return cdf;
}

// Beta-Binomial call against a baseline advance rate p0, three ways:
//   normal    — posterior advance rate above DISRUPTED_RATIO * p0.
//   disrupted — posterior at/under DISRUPTED_RATIO * p0 AND the low advance count
//               is significant against p0 (binomial lower tail <= CLASSIFY_ALPHA).
//   null      — too few matches; a degenerate baseline (p0 < CLASSIFY_P0_FLOOR)
//               where the disrupted test has no power; or a point-estimate drop
//               not distinguishable from a low-p0 normal fluctuation (a
//               degenerate-baseline zero-advance tick, not a stall).
// The one decision rule shared by the direction classifier and the segment
// classifier (segment_flow.ts), so the two never disagree. NOTE: with smoothed
// (decayed) counts the binomial tail is a tuned score, not a calibrated p-value —
// CLASSIFY_ALPHA is an empirical threshold.
export function classifyAdvance(
  advancedN: number,
  stalledN: number,
  p0: number,
): 'normal' | 'disrupted' | null {
  const matched = advancedN + stalledN;
  if (matched < MIN_MATCHED_TRIPS) return null;
  const post =
    (CLASSIFY_PRIOR_STRENGTH * p0 + advancedN) / (CLASSIFY_PRIOR_STRENGTH + matched);
  if (post > DISRUPTED_RATIO * p0) return 'normal';
  // Degenerate baseline: a cell that advances ~0 even when healthy (shuttles,
  // terminals) has no trustworthy 'normal' to fall from, so a zero-advance tick
  // carries no disruption signal. Abstain outright rather than credit a tail
  // computed against that degenerate p0.
  if (p0 < CLASSIFY_P0_FLOOR) return null;
  return binomLowerTail(advancedN, matched, p0) <= CLASSIFY_ALPHA ? 'disrupted' : null;
}
