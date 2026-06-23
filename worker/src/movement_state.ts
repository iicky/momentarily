/**
 * Determine a route's current condition directly from where its trains physically
 * are, rather than inferring it from the alerts feed. This is the published
 * current-state signal: the alert-derived HMM is good at forecasting (recovery,
 * p_normal_in_H) but weak at "is this route disrupted right now" — and right now
 * is directly observable from movement.
 *
 * Inputs are the two per-route metrics already derived each tick:
 *   - MovementRow (vehicle positions): the cross-tick advance fraction
 *     advanced_n / (advanced_n + stalled_n) — of trips seen both this tick and
 *     last, the share that moved to a new stop. The disrupted/normal axis.
 *   - ServiceRow (trip-updates): assigned_n, dispatched trains. assigned_n == 0
 *     with trips still scheduled is the suspension signal — more reliable than
 *     vehicles_n == 0, since the vehicle feed tends to carry a few trains even on
 *     a suspended route.
 *
 * Returns null when movement can't support a call (cold start, feed gap, too few
 * cross-tick matches). The caller treats null as "fall back to the alert/HMM
 * condition", never as a silent "normal".
 *
 * Thresholds mirror training/load_r2.py (derive_movement_state) so the live
 * signal and the offline series agree on what "frozen" means.
 */

import type { MovementRow } from './vehicles';
import type { ServiceRow } from './trip_updates';

// OFF: this fixed-threshold derivation has severe per-route bias (shuttles read
// ~100% disrupted, trunk lines ~0%) and is NOT published. The Bayesian
// per-direction model (momentarily-vhh) replaces the derivation; flip this on
// once that lands. While off, movement state is never written and the published
// condition stays HMM-derived. The direction-split archive accrues regardless,
// so the baseline data clock runs.
export const MOVEMENT_STATE_PUBLISH = false;

export const FROZEN_ADVANCE_FRAC = 0.25; // advance fraction at/under which a route reads frozen
export const MIN_MATCHED_TRIPS = 3; // advanced_n + stalled_n floor to make a cross-tick call

export type MovementCondition = 'normal' | 'disrupted' | 'suspended';

export function deriveMovementState(
  move: MovementRow | undefined,
  svc: ServiceRow | undefined,
): MovementCondition | null {
  // Suspended: trips scheduled but none dispatched (assigned_n == 0). Primary
  // signal from the trip-updates feed.
  if (svc && svc.trips_n > 0 && svc.assigned_n === 0) return 'suspended';
  // Secondary: no trains reporting position, and trip-updates doesn't contradict
  // it (no assigned trains). Guards against a feed inconsistency where trains are
  // assigned but absent from vehicle positions.
  if (move && move.vehicles_n === 0 && (!svc || svc.assigned_n === 0)) {
    return 'suspended';
  }
  if (!move) return null;
  const matched = move.advanced_n + move.stalled_n;
  if (matched < MIN_MATCHED_TRIPS) return null; // trains present but too few cross-tick matches
  return move.advanced_n / matched <= FROZEN_ADVANCE_FRAC ? 'disrupted' : 'normal';
}

/**
 * Per-route movement-derived condition across all routes either feed saw this
 * tick. Routes whose movement can't be judged are omitted (the caller falls back
 * to the alert/HMM condition for those).
 */
export function deriveMovementStates(
  moveRows: Map<string, MovementRow>,
  svcRows: Map<string, ServiceRow>,
): Record<string, MovementCondition> {
  const out: Record<string, MovementCondition> = {};
  const routes = new Set<string>([...moveRows.keys(), ...svcRows.keys()]);
  for (const route of routes) {
    const state = deriveMovementState(moveRows.get(route), svcRows.get(route));
    if (state !== null) out[route] = state;
  }
  return out;
}
