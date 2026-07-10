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
import type { AdvanceBaselineCell, TrainedParams } from './params';
import type { MovementMetricDoc, ServiceMetricDoc } from './state';
import type { MovementRow } from './vehicles';
import type { ServiceRow } from './trip_updates';

// OFF: the movement-derived condition is not published yet — the public flip is a
// separate step. While off, movement state is never written and the published
// condition stays HMM-derived. The direction-split archive accrues regardless, so
// the baseline data clock runs.
export const MOVEMENT_STATE_PUBLISH = false;

// Classification-time prior strength in pseudo-trials — regularizes a single
// tick's advance fraction toward the cell baseline so a thin sample can't swing
// the call. Distinct from the trainer's advance-baseline prior strength, which
// anchors the HMM emission accumulated over the whole training window.
const CLASSIFY_PRIOR_STRENGTH = 8;
// A direction reads disrupted when its posterior advance rate sits at/under this
// fraction of the cell's own baseline p0 — advancing at under half its normal
// rate. Baseline-relative, so shuttles and trunk lines are each judged against
// their own normal instead of one global cutoff.
const DISRUPTED_RATIO = 0.5;
export const MIN_MATCHED_TRIPS = 3; // advanced_n + stalled_n floor to make a cross-tick call

export type MovementCondition = 'normal' | 'disrupted' | 'suspended';

// Beta-Binomial call for one (route, direction) at one tick, or null when it
// can't be judged (too few cross-tick matches, or no baseline prior for the
// cell). Posterior mean of the advance rate under a Beta prior centered on the
// cell baseline p0; disrupted when that posterior sits at/under DISRUPTED_RATIO *
// p0 — a drop relative to the direction's own normal, not a global cutoff.
function classifyDirection(
  advancedN: number,
  stalledN: number,
  cell: AdvanceBaselineCell | null,
): Exclude<MovementCondition, 'suspended'> | null {
  const matched = advancedN + stalledN;
  if (matched < MIN_MATCHED_TRIPS) return null;
  if (!cell) return null;
  const post =
    (CLASSIFY_PRIOR_STRENGTH * cell.p0 + advancedN) / (CLASSIFY_PRIOR_STRENGTH + matched);
  return post <= DISRUPTED_RATIO * cell.p0 ? 'disrupted' : 'normal';
}

export function deriveMovementState(
  routeId: string,
  move: MovementRow | undefined,
  svc: ServiceRow | undefined,
  trained: TrainedParams | null,
  observedAt: number,
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
  // Disrupted/normal: score each direction against its own (route, direction,
  // tod_bin) baseline and take the worse — one frozen direction disrupts the
  // route. Abstain (null) when no direction is judgeable.
  const todBin = tod_bin(observedAt);
  const calls: MovementCondition[] = [];
  for (const dir of ['north', 'south'] as const) {
    const drow = move.by_direction[dir];
    if (!drow) continue; // partial by_direction payload — abstain this direction, mirroring load_r2.py
    const cell = advanceBaselineFor(trained, routeId, dir, todBin);
    const call = classifyDirection(drow.advanced_n, drow.stalled_n, cell);
    if (call !== null) calls.push(call);
  }
  if (calls.length === 0) return null;
  return calls.includes('disrupted') ? 'disrupted' : 'normal';
}

/**
 * Per-route movement-derived condition across all routes either feed saw this
 * tick. Routes whose movement can't be judged are omitted (the caller falls back
 * to the alert/HMM condition for those).
 */
export function deriveMovementStates(
  moveRows: Map<string, MovementRow>,
  svcRows: Map<string, ServiceRow>,
  trained: TrainedParams | null,
  observedAt: number,
): Record<string, MovementCondition> {
  const out: Record<string, MovementCondition> = {};
  const routes = new Set<string>([...moveRows.keys(), ...svcRows.keys()]);
  for (const route of routes) {
    const state = deriveMovementState(
      route,
      moveRows.get(route),
      svcRows.get(route),
      trained,
      observedAt,
    );
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
