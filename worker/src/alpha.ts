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
  // so the trainer can segment dwell distributions by cause. See momentarily-22k.
  alert_type_at_entry: string | null;
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
