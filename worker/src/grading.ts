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
 *   v1/movement_transitions/YYYY-MM-DD/<ts>-<scope>.jsonl
 *     The same, for the movement arm's debounced regimes, at both route and
 *     segment scope. This is the stream the movement dwell curves are fitted
 *     from — the filter's argmax flips describe a different signal.
 *
 *     The `-<scope>` suffix is load-bearing: index.ts commits the route clock
 *     and the segment clock in two separate puts on the same tick, so a key of
 *     `<ts>.jsonl` alone made the second put overwrite the first. It did —
 *     route scope has been empty since 2026-08-27 and stays empty until this
 *     file is deployed, see journal.md 2026-09-03. Readers list by date prefix
 *     and filter on the record's own `scope` field, so the unscoped keys the
 *     archive already holds still load beside the new ones.
 *
 * All three prefixes are listable by date for the Python grader.
 */

import type { RouteRoll } from './alpha';
import { STATES } from './hmm';
import type { State } from './hmm';
import type { RegimeChange } from './regime';

export interface PredictionRecord {
  ts: number;
  route: string;
  condition: string;
  regime_entered_at: number;
  p_normal: number;
  p_disrupted: number;
  p_suspended: number;
  // Null when withheld. All three horizons can be, for two different reasons:
  // 30 when the forecast came from a different arm than the published
  // condition, 60/120 because they measured worse than naive persistence and
  // are only populated for deterministic schedule rows. The grader must skip
  // nulls rather than coerce them — a null read as 0 would score as a
  // confident "will not recover", which is not what was published.
  p_normal_in_30min: number | null;
  p_normal_in_60min: number | null;
  p_normal_in_120min: number | null;
  recovery_minutes: number;
  recovery_minutes_low: number;
  recovery_minutes_high: number;
  // True when the dwell estimate saturated MAX_RECOVERY_MINUTES — the geometric
  // self-loop projection is uninformative and the recovery_minutes value is a
  // clamp, not a real prediction. The grader must skip these rows so they don't
  // drag MAE around.
  recovery_indeterminate: boolean;
  // "schedule" recoveries are deterministic lookups of the planned resume time,
  // not dwell estimates; "movement" is the movement-clock dwell curve. The
  // grader excludes "schedule" rows from HMM calibration and instead grades
  // them against the announced resumes_at (schedule adherence).
  recovery_source: 'hmm' | 'schedule' | 'movement';
  resumes_at: number | null;
  // primary_alert_type at this tick (the cause label currently associated with
  // the route). null when no alert is active. Lets the grader segment
  // calibration by cause.
  primary_alert_type: string | null;
  // trained_at of the params.json that produced this prediction (0 = bootstrap).
  // The grader segments by this so a fresh retrain's predictions are judged
  // separately from old-params rows in the same window.
  params_version: number;
  // The published movement-primary current-state condition and its source at
  // this tick (the alert-shadow lives in `condition` above). Lets the grader
  // score the escalation arm — movement disrupted where the alert feed read
  // normal — against later alerts as delayed truth.
  published_condition: string;
  condition_source: string;
  // When the published movement regime was entered (the movement arm's own
  // clock, distinct from regime_entered_at above, which is the filter's). 0
  // when movement has no regime for the route this tick. Elapsed time in the
  // regime is what the movement dwell curve is conditioned on, so a grader
  // cannot reconstruct the forecast without it.
  movement_regime_entered_at: number;
  // The movement channel's inputs, as this tick's posterior was actually
  // computed from them. Null when the channel was gated off (logEmission
  // contributes 0 for it) or the tick had no observation at all.
  //
  // Present because the posterior saturates and the grader cannot tell from the
  // posterior alone which channel did it: of the seven channels in
  // logEmission, six evaluate one scalar or flag per tick, while this one's
  // log-likelihood ratio grows as matched_n * KL(rate_normal || rate_disrupted)
  // — linear in the tick's trip count, which has a floor of MIN_MATCHED_TRIPS
  // and no cap. With these two and the params, that term is computable in nats
  // per tick instead of inferred. See journal.md 2026-08-23.
  matched_n: number | null;
  advanced_n: number | null;
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
  // quantiles on once enough data accumulates.
  alert_type_at_entry: string | null;
}

/**
 * A committed movement-regime change, at route or segment scope. Carries the
 * same clock fields as TransitionRecord so both streams grade through one code
 * path; `scope` and `key` say which cell moved.
 */
export interface MovementTransitionRecord {
  ts: number;
  scope: 'route' | 'segment';
  // routeId at route scope, `route|direction|from_stop` at segment scope.
  key: string;
  // The route either scope belongs to, so segment dwells can pool up to their
  // route without re-parsing the key.
  route: string;
  prev_state: string;
  new_state: string;
  regime_entered_at: number;
  exited_at: number;
  dwell_sec: number;
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
 * Commit this tick's movement-regime changes, one R2 object per scope present.
 *
 * Partitioning by scope is the whole point: index.ts calls this twice on the
 * same tick with the same observedAt, once per clock, and an R2 put replaces
 * rather than appends. Keying on observedAt alone therefore made the segment
 * write delete the route write, which it did for eight days — see the module
 * comment. Grouping here rather than trusting the caller to pass a single
 * scope means no key can be mislabelled and no two puts from one tick can
 * target the same key.
 */
export async function writeMovementTransitions(
  bucket: R2Bucket,
  observedAt: number,
  records: MovementTransitionRecord[],
): Promise<void> {
  if (records.length === 0) return;
  const byScope: Record<string, MovementTransitionRecord[]> = {};
  for (const r of records) {
    (byScope[r.scope] ??= []).push(r);
  }
  const date = utcDate(observedAt);
  await Promise.all(
    Object.entries(byScope).map(([scope, group]) =>
      bucket.put(
        `v1/movement_transitions/${date}/${observedAt}-${scope}.jsonl`,
        group.map((r) => JSON.stringify(r)).join('\n'),
        { httpMetadata: { contentType: 'application/x-ndjson' } },
      ),
    ),
  );
}

/**
 * Turn this tick's committed regime changes into transition records. Segment
 * keys are `route|direction|from_stop`, so the route is the first field.
 */
export function movementTransitions(
  changes: RegimeChange[],
  scope: 'route' | 'segment',
  observedAt: number,
): MovementTransitionRecord[] {
  return changes.map((c) => ({
    ts: observedAt,
    scope,
    key: c.key,
    route: scope === 'route' ? c.key : c.key.slice(0, c.key.indexOf('|')),
    prev_state: c.prev_state,
    new_state: c.new_state,
    regime_entered_at: c.entered_at,
    exited_at: c.exited_at,
    dwell_sec: c.dwell_sec,
  }));
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
