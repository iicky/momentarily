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

import { tod_bin } from './hmm';
import type { Observation } from './hmm';
import { advanceBaselineFor, serviceBaselineFor } from './params';
import type { TrainedParams } from './params';
import type { MovementMetricDoc, ServiceMetricDoc } from './state';
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

// A carried movement metric older than this (seconds) is a feed gap, not "now" —
// don't fold a stale cross-tick sample into the filter. One tick of slack past
// the intended ~5-min lag.
export const MAX_MOVEMENT_METRIC_LAG_SECONDS = 600;

/**
 * Movement fields for a route's Observation at derive time, from the PREVIOUS
 * tick's carried counts (option B, ~5-min lag). Returns null — leave the
 * observation's movement channel off — when there's no usable signal: no carried
 * metric, a stale one, no counts for the route, too few cross-tick matches, or
 * no trainer baseline for the cell that produced the counts.
 *
 * The route-level filter takes one Observation per route, so both directions are
 * aggregated. The baseline gate keys off the CURRENT tick's tod_bin — the same
 * bin emissionsFor() scores the sample with — so a sample is never admitted
 * under one bin's baseline and scored under another's advance_rate.
 */
export function movementObservationFields(
  metric: MovementMetricDoc | null,
  trained: TrainedParams | null,
  routeId: string,
  observedAt: number,
): Pick<Observation, 'advanced_n' | 'matched_n' | 'has_movement'> | null {
  if (!metric) return null;
  if (observedAt - metric.observed_at > MAX_MOVEMENT_METRIC_LAG_SECONDS) return null;
  const row = metric.rows[routeId];
  if (!row) return null;
  const advanced_n = row.north.advanced_n + row.south.advanced_n;
  const matched_n = advanced_n + row.north.stalled_n + row.south.stalled_n;
  if (matched_n < MIN_MATCHED_TRIPS) return null;
  const todBin = tod_bin(observedAt);
  const hasBaseline =
    advanceBaselineFor(trained, routeId, 'north', todBin) !== null
    || advanceBaselineFor(trained, routeId, 'south', todBin) !== null;
  if (!hasBaseline) return null;
  return { advanced_n, matched_n, has_movement: true };
}

// A carried service metric older than this (seconds) is a feed gap, not "now".
export const MAX_SERVICE_METRIC_LAG_SECONDS = 600;

/**
 * Service fields for a route's Observation at derive time, from the PREVIOUS
 * tick's carried assigned_n (option B, ~5-min lag). Returns null — leave the
 * service channel off — when there's no usable signal: no carried metric, a
 * stale one, no assigned_n for the route, or no trainer baseline for the cell.
 * The ratio is assigned_n / baseline(route, current-tick tod_bin); the gate keys
 * off the current tick's bin so admit and score share the same bin.
 */
export function serviceObservationFields(
  metric: ServiceMetricDoc | null,
  trained: TrainedParams | null,
  routeId: string,
  observedAt: number,
): Pick<Observation, 'service_ratio' | 'has_service'> | null {
  if (!metric) return null;
  if (observedAt - metric.observed_at > MAX_SERVICE_METRIC_LAG_SECONDS) return null;
  const assigned = metric.rows[routeId];
  if (assigned === undefined) return null;
  const baseline = serviceBaselineFor(trained, routeId, tod_bin(observedAt));
  if (baseline === null || baseline <= 0) return null;
  return { service_ratio: assigned / baseline, has_service: true };
}
