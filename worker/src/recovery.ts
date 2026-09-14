/**
 * The recovery-duration climatology served to the public snapshot.
 *
 * Recovery is the product's differentiator and every fitted dwell curve has
 * lost to the empirical duration climatology (causal_skill still negative), so
 * the Worker serves the climatology straight rather than a curve: given a route
 * whose PUBLISHED condition is disrupted/suspended with a known onset and a
 * primary alert type, it reads the realized incident durations for that
 * (route, alert_type) cell (pooled up when thin, see params.resolveRecoveryCell)
 * and publishes the REMAINING duration distribution conditioned on how long the
 * live disruption has already run — D - t | D > t. This is NOT a fitted curve,
 * so it is NOT subject to PUBLISH_FITTED_RECOVERY; it is the yardstick fitted
 * curves must beat.
 *
 * The conditional math mirrors training/recovery_baseline.py exactly
 * (empirical_quantile / conditional_remaining_quantiles), pinned by the parity
 * fixture tests/fixtures/parity_recovery.json — if this drifts from the Python
 * reference it fails in worker/test/recovery_parity.test.ts.
 */

import type { RecoveryBaselineDoc } from './params';
import { resolveRecoveryCell } from './params';

/**
 * The q-th quantile of an ascending sample list by linear interpolation between
 * order statistics (numpy's default 'linear'/type-7). Byte-for-byte reproducible
 * against empirical_quantile in training/recovery_baseline.py.
 */
export function empiricalQuantile(sorted: number[], q: number): number {
  const n = sorted.length;
  if (n === 0) throw new Error('empiricalQuantile of an empty sample');
  if (n === 1) return sorted[0]!;
  const pos = q * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.min(lo + 1, n - 1);
  const frac = pos - lo;
  return sorted[lo]! + frac * (sorted[hi]! - sorted[lo]!);
}

export interface RemainingQuantiles {
  p25: number;
  p50: number;
  p75: number;
  /** Samples still running past the elapsed time. */
  n: number;
}

/**
 * Median and IQR of how much LONGER a disruption runs given it has already
 * lasted `elapsedMin`, from a cell's realized durations (minutes). Keeps only
 * durations exceeding the elapsed time, subtracts it, and reads the empirical
 * p25/p50/p75 of what remains. Null when fewer than `minSamples` durations run
 * past the elapsed time — the disruption has outlived its population and the
 * honest answer is indeterminate, never an extrapolated tail. Mirrors
 * conditional_remaining_quantiles in training/recovery_baseline.py.
 */
export function conditionalRemainingQuantiles(
  samplesMin: number[],
  elapsedMin: number,
  minSamples: number,
): RemainingQuantiles | null {
  const remaining = samplesMin
    .filter((s) => s > elapsedMin)
    .map((s) => s - elapsedMin)
    .sort((a, b) => a - b);
  if (remaining.length < minSamples) return null;
  return {
    p25: empiricalQuantile(remaining, 0.25),
    p50: empiricalQuantile(remaining, 0.5),
    p75: empiricalQuantile(remaining, 0.75),
    n: remaining.length,
  };
}

export interface ClimatologyRecovery {
  /** Median remaining minutes, or null when the disruption outlived its population. */
  recovery_minutes: number | null;
  recovery_minutes_low: number | null;
  recovery_minutes_high: number | null;
  /** Support (n) and pooling level of the cell served, so a consumer sees how thin it is. */
  recovery_baseline_n: number;
  recovery_baseline_level: string;
  /** True exactly when the disruption outlived its population (minutes null). */
  recovery_indeterminate: boolean;
}

/**
 * Serve recovery for a disrupted route from the published climatology.
 *
 * Null when no cell describes (route, alertType) at all — an absent or empty
 * climatology, in which case the caller leaves recovery null (the fitted arms
 * stay withheld). Otherwise the median/IQR of the remaining duration, or an
 * indeterminate record (minutes null, recovery_indeterminate true) when the
 * disruption has outlived its population. `elapsedSec` is now - the badge's own
 * onset (condition_entered_at), clamped at 0.
 */
export function climatologyRecovery(
  doc: RecoveryBaselineDoc | null,
  routeId: string,
  alertType: string | null,
  elapsedSec: number,
): ClimatologyRecovery | null {
  if (!doc) return null;
  const cell = resolveRecoveryCell(doc, routeId, alertType);
  if (!cell) return null;
  const elapsedMin = Math.max(0, elapsedSec) / 60;
  const rq = conditionalRemainingQuantiles(cell.samples_min, elapsedMin, doc.min_samples);
  if (rq === null) {
    return {
      recovery_minutes: null,
      recovery_minutes_low: null,
      recovery_minutes_high: null,
      recovery_baseline_n: cell.n,
      recovery_baseline_level: cell.level,
      recovery_indeterminate: true,
    };
  }
  return {
    // Math.round is half-up on the non-negative remaining durations here,
    // matching the Python reference's floor(x + 0.5) so the published integer
    // minutes are identical cross-language.
    recovery_minutes: Math.round(rq.p50),
    recovery_minutes_low: Math.round(rq.p25),
    recovery_minutes_high: Math.round(rq.p75),
    recovery_baseline_n: cell.n,
    recovery_baseline_level: cell.level,
    recovery_indeterminate: false,
  };
}
