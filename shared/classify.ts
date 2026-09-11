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
//   null      — too few matches, or a point-estimate drop not distinguishable from
//               a low-p0 normal fluctuation (a degenerate-baseline zero-advance
//               tick, not a stall).
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
  return binomLowerTail(advancedN, matched, p0) <= CLASSIFY_ALPHA ? 'disrupted' : null;
}
