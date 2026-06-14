// Calibration metrics for the recovery forecasts.
//
// Ground truth comes from the transition stream (when a line ACTUALLY returned
// to normal), not from the model's own labels — so these plots are a real,
// non-circular test of the forecasts. The two headline questions:
//
//   1. Reliability: when the model said "P(normal in H min) = x", did the line
//      reach normal within H minutes in x fraction of those cases?
//   2. Recovery coverage: did the actual time-to-normal fall inside the model's
//      predicted [25th, 75th] recovery band ~50% of the time?
//
// Predictions whose outcome isn't yet observable (the window ends before ts+H,
// or before the line next returned to normal) are CENSORED and excluded — a
// missing recovery isn't evidence of a wrong forecast.

import type { PredictionRecord, TransitionRecord } from "./types";

export interface Segment {
  state: string;
  start: number; // epoch sec
  end: number; // epoch sec (observedUntil for the open/current segment)
}

export interface RouteTimeline {
  route: string;
  segments: Segment[];
  /** Sorted epoch-sec starts of every observed "normal" regime. */
  normalStarts: number[];
  /** Latest time we know this route's state; beyond it, outcomes are censored. */
  observedUntil: number;
}

/**
 * Reconstruct each route's regime timeline from its transitions.
 *
 * Each transition says: prev_state ran [regime_entered_at, exited_at), then
 * new_state began at exited_at. Chaining them yields contiguous segments; the
 * final new_state is the open/current regime, capped at observedUntil.
 */
export function buildTimelines(
  transitions: TransitionRecord[],
  nowSec: number,
): Map<string, RouteTimeline> {
  const byRoute = new Map<string, TransitionRecord[]>();
  for (const t of transitions) {
    const arr = byRoute.get(t.route) ?? [];
    arr.push(t);
    byRoute.set(t.route, arr);
  }

  const out = new Map<string, RouteTimeline>();
  for (const [route, recs] of byRoute) {
    recs.sort((a, b) => a.exited_at - b.exited_at);
    const segments: Segment[] = [];
    const normalStarts = new Set<number>();

    for (const r of recs) {
      segments.push({
        state: r.prev_state,
        start: r.regime_entered_at,
        end: r.exited_at,
      });
      if (r.prev_state === "normal") normalStarts.add(r.regime_entered_at);
      if (r.new_state === "normal") normalStarts.add(r.exited_at);
    }

    const last = recs[recs.length - 1];
    const observedUntil = Math.max(last.exited_at, nowSec);
    // Open current regime.
    segments.push({
      state: last.new_state,
      start: last.exited_at,
      end: observedUntil,
    });

    out.set(route, {
      route,
      segments,
      normalStarts: [...normalStarts].sort((a, b) => a - b),
      observedUntil,
    });
  }
  return out;
}

/** First time strictly after `ts` that the route entered a normal regime. */
function nextNormalStart(tl: RouteTimeline, ts: number): number | null {
  for (const s of tl.normalStarts) if (s > ts) return s;
  return null;
}

export interface ReliabilityBin {
  // bucket midpoint in [0,1]
  p: number;
  predictedMean: number;
  observedFreq: number;
  n: number;
}

export interface ReliabilityResult {
  horizonMin: number;
  bins: ReliabilityBin[];
  brier: number;
  n: number;
  // schedule-recovery predictions skipped — they're deterministic resume
  // lookups, perfect by construction, so grading them would flatter the HMM.
  excludedSchedule: number;
}

const HORIZON_FIELD: Record<number, keyof PredictionRecord> = {
  30: "p_normal_in_30min",
  60: "p_normal_in_60min",
  120: "p_normal_in_120min",
};

/**
 * Reliability of the "P(normal in H min)" forecast against observed recoveries.
 * Evaluates predictions made while the line was NOT normal, whose H-minute
 * outcome is observable within the data window.
 */
export function reliability(
  predictions: PredictionRecord[],
  timelines: Map<string, RouteTimeline>,
  horizonMin: number,
  nBins = 10,
): ReliabilityResult {
  const field = HORIZON_FIELD[horizonMin];
  const horizonSec = horizonMin * 60;
  const bins: { sumP: number; sumY: number; n: number }[] = Array.from(
    { length: nBins },
    () => ({ sumP: 0, sumY: 0, n: 0 }),
  );
  let brierSum = 0;
  let n = 0;
  let excludedSchedule = 0;

  for (const pr of predictions) {
    if (pr.condition === "normal") continue;
    if (pr.recovery_source === "schedule") {
      excludedSchedule += 1;
      continue;
    }
    const tl = timelines.get(pr.route);
    if (!tl) continue;
    if (pr.ts + horizonSec > tl.observedUntil) continue; // censored

    const nn = nextNormalStart(tl, pr.ts);
    const y = nn != null && nn - pr.ts <= horizonSec ? 1 : 0;
    const p = pr[field] as number;
    if (typeof p !== "number" || Number.isNaN(p)) continue;

    const idx = Math.min(nBins - 1, Math.max(0, Math.floor(p * nBins)));
    bins[idx].sumP += p;
    bins[idx].sumY += y;
    bins[idx].n += 1;
    brierSum += (p - y) * (p - y);
    n += 1;
  }

  return {
    horizonMin,
    n,
    excludedSchedule,
    brier: n ? brierSum / n : NaN,
    bins: bins.map((b, i) => ({
      p: (i + 0.5) / nBins,
      predictedMean: b.n ? b.sumP / b.n : NaN,
      observedFreq: b.n ? b.sumY / b.n : NaN,
      n: b.n,
    })),
  };
}

export interface RecoveryPoint {
  route: string;
  ts: number;
  predictedMin: number;
  lowMin: number;
  highMin: number;
  actualMin: number;
  inIqr: boolean;
}

export interface RecoveryResult {
  points: RecoveryPoint[];
  coverage: number; // fraction inside [low, high]
  n: number;
  medianAbsErrorMin: number;
  // schedule-recovery predictions skipped (graded for adherence instead).
  excludedSchedule: number;
}

/**
 * Compare predicted recovery (median + IQR) to the actual observed time until
 * the line next returned to normal. Indeterminate forecasts are excluded.
 */
export function recoveryError(
  predictions: PredictionRecord[],
  timelines: Map<string, RouteTimeline>,
): RecoveryResult {
  const points: RecoveryPoint[] = [];
  let excludedSchedule = 0;
  for (const pr of predictions) {
    if (pr.condition === "normal" || pr.recovery_indeterminate) continue;
    if (pr.recovery_source === "schedule") {
      excludedSchedule += 1;
      continue;
    }
    const tl = timelines.get(pr.route);
    if (!tl) continue;
    const nn = nextNormalStart(tl, pr.ts);
    if (nn == null || nn > tl.observedUntil) continue; // never recovered in window
    const actualMin = (nn - pr.ts) / 60;
    const inIqr =
      actualMin >= pr.recovery_minutes_low &&
      actualMin <= pr.recovery_minutes_high;
    points.push({
      route: pr.route,
      ts: pr.ts,
      predictedMin: pr.recovery_minutes,
      lowMin: pr.recovery_minutes_low,
      highMin: pr.recovery_minutes_high,
      actualMin,
      inIqr,
    });
  }
  const n = points.length;
  const errs = points.map((p) => Math.abs(p.predictedMin - p.actualMin)).sort(
    (a, b) => a - b,
  );
  return {
    points,
    n,
    excludedSchedule,
    coverage: n ? points.filter((p) => p.inIqr).length / n : NaN,
    medianAbsErrorMin: n ? errs[Math.floor(n / 2)] : NaN,
  };
}

// --- Detection latency ---

export interface DetectionLatencyPoint {
  route: string;
  alertType: string;
  onsetTs: number;
  detectedTs: number;
  latencyMin: number;
}

export interface DetectionLatencyResult {
  points: DetectionLatencyPoint[];
  n: number;
  // Alert onsets that cleared before the HMM ever flipped to disrupted/suspended
  // — not a latency, a non-detection (e.g. not_scheduled, or sub-threshold).
  missed: number;
  medianLatencyMin: number;
  byAlertType: {
    alertType: string;
    n: number;
    medianLatencyMin: number;
    missed: number;
  }[];
}

function median(sorted: number[]): number {
  return sorted.length ? sorted[Math.floor(sorted.length / 2)] : NaN;
}

/**
 * Detection latency: minutes between a real alert first appearing on a route
 * and the HMM published condition flipping to disrupted/suspended.
 *
 * Walks each route's per-tick predictions: an alert onset is the first tick a
 * primary_alert_type appears after none; detection is the first disrupted/
 * suspended tick before the alert clears. An onset whose alert clears with no
 * flip is a miss (counted, not a latency). Resolution is the prediction tick
 * (~5 min). Onsets already in-flight at the window's first tick are imperfectly
 * attributed — a known window-edge artifact.
 */
export function detectionLatency(
  predictions: PredictionRecord[],
): DetectionLatencyResult {
  const byRoute = new Map<string, PredictionRecord[]>();
  for (const p of predictions) {
    const arr = byRoute.get(p.route) ?? [];
    arr.push(p);
    byRoute.set(p.route, arr);
  }

  const points: DetectionLatencyPoint[] = [];
  const missedByType = new Map<string, number>();
  let missed = 0;

  for (const [route, recs] of byRoute) {
    recs.sort((a, b) => a.ts - b.ts);
    let onsetTs: number | null = null;
    let onsetType: string | null = null;
    let prevHadAlert = false;

    for (const pr of recs) {
      const hasAlert = pr.primary_alert_type != null;
      const disrupted =
        pr.condition === "disrupted" || pr.condition === "suspended";

      if (hasAlert && !prevHadAlert && onsetTs == null) {
        onsetTs = pr.ts;
        onsetType = pr.primary_alert_type;
      }
      if (onsetTs != null && disrupted) {
        points.push({
          route,
          alertType: onsetType ?? "(unknown)",
          onsetTs,
          detectedTs: pr.ts,
          latencyMin: (pr.ts - onsetTs) / 60,
        });
        onsetTs = null;
        onsetType = null;
      } else if (onsetTs != null && !hasAlert) {
        missed += 1;
        const k = onsetType ?? "(unknown)";
        missedByType.set(k, (missedByType.get(k) ?? 0) + 1);
        onsetTs = null;
        onsetType = null;
      }
      prevHadAlert = hasAlert;
    }
  }

  const byTypeMap = new Map<string, number[]>();
  for (const p of points) {
    const arr = byTypeMap.get(p.alertType) ?? [];
    arr.push(p.latencyMin);
    byTypeMap.set(p.alertType, arr);
  }
  const types = new Set<string>([...byTypeMap.keys(), ...missedByType.keys()]);
  const byAlertType = [...types]
    .map((alertType) => {
      const lats = (byTypeMap.get(alertType) ?? []).sort((a, b) => a - b);
      return {
        alertType,
        n: lats.length,
        medianLatencyMin: median(lats),
        missed: missedByType.get(alertType) ?? 0,
      };
    })
    .sort((a, b) => b.n - a.n || b.missed - a.missed);

  const allLats = points.map((p) => p.latencyMin).sort((a, b) => a - b);
  return {
    points,
    n: points.length,
    missed,
    medianLatencyMin: median(allLats),
    byAlertType,
  };
}

/** Distinct route ids present in either stream, sorted naturally. */
export function routeUniverse(
  predictions: PredictionRecord[],
  transitions: TransitionRecord[],
): string[] {
  const set = new Set<string>();
  for (const p of predictions) set.add(p.route);
  for (const t of transitions) set.add(t.route);
  return [...set].sort((a, b) =>
    a.localeCompare(b, undefined, { numeric: true }),
  );
}
