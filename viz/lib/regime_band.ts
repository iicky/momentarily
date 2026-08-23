// Regime probabilities as a time series.
//
// The status drawer shows one instantaneous stack (p_normal / p_disrupted /
// p_suspended right now) and the swimlane next to this shows the *published
// condition* — a hard label per instant. Neither shows the belief underneath
// the label, which is where hedging, near-flips and slow escalations live: a
// line can sit at 60/40 for an hour before the argmax ever moves, and no hard
// label can express that.
//
// This buckets the prediction stream onto a fixed time grid and averages the
// three probabilities within each bucket, so the result is a stacked band that
// always sums to 1 and can be plotted against the same axis as the swimlane.

import type { PredictionRecord } from "./types";

export interface RegimeBandBucket {
  /** Bucket start, epoch seconds; width is the series' bucketSec. */
  t: number;
  /** Prediction ticks averaged into this bucket. Always >= 1 — empty buckets
   *  are omitted entirely, so a break in `buckets` is a real hole in the
   *  archive and the renderer must break the fill rather than interpolate. */
  n: number;
  pNormal: number;
  pDisrupted: number;
  pSuspended: number;
}

export interface RegimeBandSeries {
  route: string;
  /** Ascending by `t`, empty buckets dropped. */
  buckets: RegimeBandBucket[];
  /** Ticks behind the whole series. */
  n: number;
  /** Expected minutes spent away from normal: sum over buckets of
   *  (1 - pNormal) * bucket width. Bucket-based rather than tick-based so a
   *  densely-sampled line doesn't outrank a genuinely worse one. Used to pick
   *  which lines are worth a row. */
  nonNormalMin: number;
}

export interface RegimeBands {
  /** Grid bounds, epoch seconds. t0 is bucket-aligned; t1 is the exclusive end. */
  t0: number;
  t1: number;
  bucketSec: number;
  series: RegimeBandSeries[];
  /** Series that ranked below `limit` and were dropped. */
  truncated: number;
}

// Bucket widths we're willing to snap to — all whole minutes, so tick labels
// land on round times. The grid is aligned to epoch multiples of the chosen
// width, which keeps buckets stable across reloads and across window changes.
const BUCKET_STEPS = [60, 120, 300, 600, 900, 1800, 3600, 7200, 10800, 21600];
const TARGET_BUCKETS = 240;
const DEFAULT_LIMIT = 14;

/** Narrowest step that keeps the grid under `target` buckets. */
export function chooseBucketSec(spanSec: number, target = TARGET_BUCKETS): number {
  for (const step of BUCKET_STEPS) if (spanSec / step <= target) return step;
  return BUCKET_STEPS[BUCKET_STEPS.length - 1];
}

interface Acc {
  n: number;
  normal: number;
  disrupted: number;
  suspended: number;
}

const EMPTY: RegimeBands = {
  t0: 0,
  t1: 0,
  bucketSec: BUCKET_STEPS[0],
  series: [],
  truncated: 0,
};

export function regimeBands(
  predictions: PredictionRecord[],
  opts: { limit?: number } = {},
): RegimeBands {
  const limit = opts.limit ?? DEFAULT_LIMIT;
  if (!predictions.length) return EMPTY;

  let lo = Infinity;
  let hi = -Infinity;
  for (const p of predictions) {
    if (!Number.isFinite(p.ts)) continue;
    if (p.ts < lo) lo = p.ts;
    if (p.ts > hi) hi = p.ts;
  }
  if (!Number.isFinite(lo)) return EMPTY;

  const bucketSec = chooseBucketSec(Math.max(1, hi - lo));
  const t0 = Math.floor(lo / bucketSec) * bucketSec;
  const t1 = Math.floor(hi / bucketSec) * bucketSec + bucketSec;

  const byRoute = new Map<string, Map<number, Acc>>();
  for (const p of predictions) {
    if (!Number.isFinite(p.ts)) continue;
    const a = p.p_normal;
    const b = p.p_disrupted;
    const c = p.p_suspended;
    if (!Number.isFinite(a) || !Number.isFinite(b) || !Number.isFinite(c)) continue;
    const sum = a + b + c;
    // Normalize per tick, not per bucket: every tick then contributes exactly
    // one unit of mass, so a record published at 0.99 total can't quietly
    // under-weight itself against one published at 1.0.
    if (!(sum > 0)) continue;

    let buckets = byRoute.get(p.route);
    if (buckets === undefined) {
      buckets = new Map();
      byRoute.set(p.route, buckets);
    }
    const k = Math.floor((p.ts - t0) / bucketSec);
    let acc = buckets.get(k);
    if (acc === undefined) {
      acc = { n: 0, normal: 0, disrupted: 0, suspended: 0 };
      buckets.set(k, acc);
    }
    acc.n += 1;
    acc.normal += a / sum;
    acc.disrupted += b / sum;
    acc.suspended += c / sum;
  }

  const bucketMin = bucketSec / 60;
  const ranked: RegimeBandSeries[] = [];
  for (const [route, accs] of byRoute) {
    const keys = [...accs.keys()].sort((x, y) => x - y);
    const buckets: RegimeBandBucket[] = [];
    let n = 0;
    let nonNormalMin = 0;
    for (const k of keys) {
      const acc = accs.get(k)!;
      const pNormal = acc.normal / acc.n;
      buckets.push({
        t: t0 + k * bucketSec,
        n: acc.n,
        pNormal,
        pDisrupted: acc.disrupted / acc.n,
        pSuspended: acc.suspended / acc.n,
      });
      n += acc.n;
      nonNormalMin += (1 - pNormal) * bucketMin;
    }
    if (buckets.length) ranked.push({ route, buckets, n, nonNormalMin });
  }

  ranked.sort((a, b) => b.nonNormalMin - a.nonNormalMin || a.route.localeCompare(b.route));
  // Prefer lines that actually left normal; a wall of flat green rows says
  // nothing. But when nothing was disrupted — a quiet window, or a filter
  // pinned to one calm line — a flat band is the honest answer, so fall back
  // to the full ranking rather than rendering an empty panel.
  const active = ranked.filter((s) => s.nonNormalMin > 0);
  const pick = active.length ? active : ranked;

  return {
    t0,
    t1,
    bucketSec,
    series: pick.slice(0, limit),
    truncated: Math.max(0, pick.length - limit),
  };
}

/**
 * Split a series' buckets into runs of grid-adjacent buckets.
 *
 * Renderers must draw each run as its own shape. A stacked area over the whole
 * bucket list would slide a straight edge across an archive gap, turning "we
 * have no idea what happened here" into a confident-looking interpolation —
 * and on this chart a flat green stretch is exactly the claim we can't make.
 */
export function bucketRuns(
  buckets: RegimeBandBucket[],
  bucketSec: number,
): RegimeBandBucket[][] {
  const runs: RegimeBandBucket[][] = [];
  let run: RegimeBandBucket[] = [];
  for (const b of buckets) {
    const prev = run[run.length - 1];
    if (prev !== undefined && b.t !== prev.t + bucketSec) {
      runs.push(run);
      run = [];
    }
    run.push(b);
  }
  if (run.length) runs.push(run);
  return runs;
}
