/**
 * Rolling HMM posterior per route, persisted in R2.
 *
 * Lives at r2://momentarily/state/alpha.json. Read at the start of each tick,
 * advanced through the forward filter for every observed route, written back
 * at the end.
 */

import type { FilterState, PublishedState } from './hmm';

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

export async function readAlphaState(bucket: R2Bucket): Promise<AlphaState> {
  const obj = await bucket.get(ALPHA_KEY);
  if (!obj) return emptyAlphaState();
  try {
    return (await obj.json()) as AlphaState;
  } catch (err) {
    console.error('alpha.json parse failed; resetting:', err);
    return emptyAlphaState();
  }
}

export async function writeAlphaState(
  bucket: R2Bucket,
  state: AlphaState,
): Promise<void> {
  await bucket.put(ALPHA_KEY, JSON.stringify(state), {
    httpMetadata: {
      contentType: 'application/json',
      cacheControl: 'no-store',
    },
  });
}
