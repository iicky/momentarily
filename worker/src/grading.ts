/**
 * Self-grading streams.
 *
 * Each cron tick appends two append-only JSONL files to R2:
 *
 *   v1/predictions/YYYY-MM-DD/<ts>.jsonl
 *     One line per route — the inference fields the snapshot publishes,
 *     timestamped so a grader can align prediction-at-T to outcome-at-T+k.
 *
 *   v1/regime_transitions/YYYY-MM-DD/<ts>.jsonl
 *     One line per route whose filter argmax flipped this tick. Empty ticks
 *     write no file. Provides ground-truth dwell times for recovery_minutes
 *     calibration.
 *
 * Both prefixes are listable by date for the Python grader (momentarily-3lb).
 */

import type { RouteRoll } from './alpha';
import { STATES } from './hmm';
import type { State } from './hmm';

export interface PredictionRecord {
  ts: number;
  route: string;
  condition: string;
  regime_entered_at: number;
  p_normal: number;
  p_disrupted: number;
  p_suspended: number;
  p_normal_in_30min: number;
  p_normal_in_60min: number;
  p_normal_in_120min: number;
  recovery_minutes: number;
  recovery_minutes_low: number;
  recovery_minutes_high: number;
  // True when the dwell estimate saturated MAX_RECOVERY_MINUTES — the geometric
  // self-loop projection is uninformative and the recovery_minutes value is a
  // clamp, not a real prediction. The grader must skip these rows so they don't
  // drag MAE around. See momentarily-x25.
  recovery_indeterminate: boolean;
  // "schedule" recoveries are deterministic lookups of the planned resume time,
  // not dwell estimates — the grader excludes them from HMM calibration and
  // instead grades them against the announced resumes_at (schedule adherence).
  recovery_source: 'hmm' | 'schedule';
  resumes_at: number | null;
  // primary_alert_type at this tick (the cause label currently associated with
  // the route). null when no alert is active. Lets the grader segment
  // calibration by cause. See momentarily-22k.
  primary_alert_type: string | null;
  // trained_at of the params.json that produced this prediction (0 = bootstrap).
  // The grader segments by this so a fresh retrain's predictions are judged
  // separately from old-params rows in the same window. See momentarily-vk0.5.
  params_version: number;
}

export interface TransitionRecord {
  ts: number;
  route: string;
  prev_state: State;
  new_state: State;
  regime_entered_at: number;
  exited_at: number;
  dwell_sec: number;
  // primary_alert_type when the prev_state regime *began*. Together with
  // (route, prev_state) this is the cell the trainer keys empirical dwell
  // quantiles on once enough data accumulates. See momentarily-alu.
  alert_type_at_entry: string | null;
}

function utcDate(epoch: number): string {
  return new Date(epoch * 1000).toISOString().slice(0, 10);
}

export async function writePredictions(
  bucket: R2Bucket,
  observedAt: number,
  records: PredictionRecord[],
): Promise<void> {
  if (records.length === 0) return;
  const key = `v1/predictions/${utcDate(observedAt)}/${observedAt}.jsonl`;
  const body = records.map((r) => JSON.stringify(r)).join('\n');
  await bucket.put(key, body, {
    httpMetadata: { contentType: 'application/x-ndjson' },
  });
}

export async function writeTransitions(
  bucket: R2Bucket,
  observedAt: number,
  records: TransitionRecord[],
): Promise<void> {
  if (records.length === 0) return;
  const key = `v1/regime_transitions/${utcDate(observedAt)}/${observedAt}.jsonl`;
  const body = records.map((r) => JSON.stringify(r)).join('\n');
  await bucket.put(key, body, {
    httpMetadata: { contentType: 'application/x-ndjson' },
  });
}

/**
 * Detect filter-argmax flips between two alpha-state snapshots. A transition
 * is emitted only when the regime_entered_at advanced, meaning forwardUpdate
 * decided the argmax changed this tick.
 */
export function detectTransitions(
  prev: Record<string, RouteRoll>,
  next: Record<string, RouteRoll>,
  observedAt: number,
): TransitionRecord[] {
  const out: TransitionRecord[] = [];
  for (const [route, newRoll] of Object.entries(next)) {
    const prevRoll = prev[route];
    if (!prevRoll) continue;
    if (newRoll.filter.regime_entered_at <= prevRoll.filter.regime_entered_at) continue;
    out.push({
      ts: observedAt,
      route,
      prev_state: STATES[argmax3(prevRoll.filter.probabilities)]!,
      new_state: STATES[argmax3(newRoll.filter.probabilities)]!,
      regime_entered_at: prevRoll.filter.regime_entered_at,
      exited_at: newRoll.filter.regime_entered_at,
      dwell_sec: newRoll.filter.regime_entered_at - prevRoll.filter.regime_entered_at,
      alert_type_at_entry: prevRoll.alert_type_at_entry ?? null,
    });
  }
  return out;
}

function argmax3(v: readonly [number, number, number]): 0 | 1 | 2 {
  if (v[0] >= v[1] && v[0] >= v[2]) return 0;
  if (v[1] >= v[2]) return 1;
  return 2;
}
