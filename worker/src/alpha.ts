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
