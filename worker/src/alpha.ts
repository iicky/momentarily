/**
 * Rolling HMM posterior per route, persisted in R2.
 *
 * Lives at r2://momentarily/state/alpha.json. Read at the start of each tick,
 * advanced through the forward filter for every observed route, written back
 * at the end.
 */

import type { FilterState, PublishedState } from './hmm';
import { conditionalPut } from './r2';
import type { VersionedRead } from './r2';

const ALPHA_KEY = 'state/alpha.json';

export interface RouteRoll {
  filter: FilterState;
  published: PublishedState;
  // primary_alert_type observed at the moment filter.regime_entered_at was last
  // advanced. null when no alert was active then. Threaded into TransitionRecord
  // so the trainer can segment dwell distributions by cause.
  alert_type_at_entry: string | null;
  // The published route condition this tick and the epoch second it began — the
  // severity-graded alert read (snapshot.resolveAlertCondition) and its own
  // regime clock. published_condition is carried so the next tick can detect a
  // change and back-date condition_entered_at to the onset; condition_entered_at
  // is what route_status.condition_entered_at publishes on the alerts arm. Both
  // optional for back-compat with alpha.json written before they shipped and for
  // hand-built test rolls; the Worker populates them every tick.
  published_condition?: string;
  condition_entered_at?: number | null;
  // The primary_alert_type captured when the condition_entered_at clock last
  // restarted — the ONSET primary alert type of the current graded condition.
  // The recovery climatology is keyed (route, alert_type) at onset on BOTH
  // sides (the trainer keys each episode by its onset primary; the Worker must
  // serve on the same key, not the current tick's primary which drifts as alert
  // types change mid-incident). Restarts with the clock, carried while the
  // condition holds, null on schedule/unknown. Optional for the same back-compat
  // reason as published_condition/condition_entered_at above.
  condition_alert_type_at_entry?: string | null;
}

export interface AlphaState {
  /** Which params.json trained_at produced these posteriors (for audit). */
  params_version: number;
  /** Last tick when alpha was advanced. */
  updated_at: number;
  /** filter + published state per route_id */
  routes: Record<string, RouteRoll>;
}

export function emptyAlphaState(): AlphaState {
  return { params_version: 0, updated_at: 0, routes: {} };
}

/**
 * The alert regime's onset clock for route_status.condition_entered_at:
 * back-date to when the current graded state began. Only the alerts arm carries
 * an honest clock — 'not_scheduled' (schedule) and 'unknown' have no onset, so
 * null there. On the alerts arm the clock holds while the graded state is
 * unchanged from the previous tick, and restarts at `observedAt` on a state
 * change, on a route's first tick (no prev roll), or on return from a
 * null-clock tick. Named because the back-dating rule is not obvious from the
 * inlined expression, and it is a test seam for the alpha-loop behavior.
 */
export function alertConditionOnset(
  prev: RouteRoll | undefined,
  condition: string,
  observedAt: number,
): number | null {
  if (condition === 'not_scheduled' || condition === 'unknown') return null;
  return prev?.published_condition === condition && prev?.condition_entered_at != null
    ? prev.condition_entered_at
    : observedAt;
}

/**
 * The primary alert type at the moment the alert-condition clock
 * (condition_entered_at) last restarted — the ONSET primary alert type. Keyed
 * to the SAME restart predicate as alertConditionOnset so the type and the clock
 * move together: captured when a new graded condition begins (or there is no
 * prior clock), carried while the condition holds, null on schedule/unknown.
 * This is what the recovery climatology serves on, matching the (route,
 * alert_type) key the trainer fit each episode's duration under at its onset.
 *
 * A LEGACY in-flight roll — condition_entered_at present (from an earlier
 * deploy) but no condition_alert_type_at_entry yet — hits the carry branch and
 * returns null: the true onset type was never recorded, so the climatology
 * withholds (correct-or-nothing, its ethos everywhere) rather than guess the
 * drifting current primary. It self-heals on the next clock restart, which
 * captures the current primary as the new onset type.
 */
export function alertConditionTypeAtOnset(
  prev: RouteRoll | undefined,
  condition: string,
  primaryAlertType: string | null,
): string | null {
  if (condition === 'not_scheduled' || condition === 'unknown') return null;
  return prev?.published_condition === condition && prev?.condition_entered_at != null
    ? (prev.condition_alert_type_at_entry ?? null)
    : primaryAlertType;
}

// Posterior weight placed on the old argmax when reseeding across a params
// swap. High enough that the predict step doesn't flip the regime on its own,
// low enough that one tick of contrary evidence can.
const RESEED_PROB = 0.7;

/**
 * Reseed a roll for freshly published params: the posterior numbers are stale
 * (filtered under the old emissions) and get replaced with a soft one-hot on
 * the old argmax, but the regime clock and its cause are observation-derived
 * facts and carry over. Recovery predictions condition on regime age, so
 * zeroing the clock on every retrain would reset long-running regimes to
 * fresh-regime optimism.
 */
export function reseedForNewParams(roll: RouteRoll): RouteRoll {
  const probs = roll.filter.probabilities;
  let argmax = 0;
  for (let i = 1; i < probs.length; i += 1) {
    if (probs[i]! > probs[argmax]!) argmax = i;
  }
  const rest = (1 - RESEED_PROB) / (probs.length - 1);
  const reseeded = probs.map((_p, i) => (i === argmax ? RESEED_PROB : rest)) as [
    number,
    number,
    number,
  ];
  return {
    filter: {
      probabilities: reseeded,
      regime_entered_at: roll.filter.regime_entered_at,
      last_updated_at: roll.filter.last_updated_at,
    },
    published: roll.published,
    alert_type_at_entry: roll.alert_type_at_entry,
    // The alert-derived condition clock is observation-derived, like the regime
    // clock above — carry it across the params swap rather than resetting onset.
    // Spread conditionally so an older roll without these fields stays absent
    // (exactOptionalPropertyTypes forbids an explicit undefined here).
    ...(roll.published_condition !== undefined
      ? { published_condition: roll.published_condition }
      : {}),
    ...(roll.condition_entered_at !== undefined
      ? { condition_entered_at: roll.condition_entered_at }
      : {}),
    ...(roll.condition_alert_type_at_entry !== undefined
      ? { condition_alert_type_at_entry: roll.condition_alert_type_at_entry }
      : {}),
  };
}

export async function readAlphaState(
  bucket: R2Bucket,
): Promise<VersionedRead<AlphaState>> {
  const obj = await bucket.get(ALPHA_KEY);
  if (!obj) return { state: emptyAlphaState(), etag: null };
  try {
    return { state: (await obj.json()) as AlphaState, etag: obj.etag };
  } catch (err) {
    console.error('alpha.json parse failed; resetting:', err);
    return { state: emptyAlphaState(), etag: obj.etag };
  }
}

/**
 * Write alpha.json with compare-and-swap on `etag` (from readAlphaState).
 * Returns false when a concurrent tick already advanced the object.
 */
export async function writeAlphaState(
  bucket: R2Bucket,
  state: AlphaState,
  etag: string | null,
): Promise<boolean> {
  return conditionalPut(bucket, ALPHA_KEY, JSON.stringify(state), etag, {
    contentType: 'application/json',
    cacheControl: 'no-store',
  });
}
