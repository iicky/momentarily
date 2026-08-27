/**
 * Determine a route's current condition directly from where its trains physically
 * are, rather than inferring it from the alerts feed. This is the published
 * current-state signal: the alert-derived HMM is good at forecasting (recovery,
 * p_normal_in_H) but weak at "is this route disrupted right now" — and right now
 * is directly observable from movement.
 *
 * Inputs are the two per-route metrics already derived each tick:
 *   - MovementRow (vehicle positions): the cross-tick advance fraction
 *     advanced_n / (advanced_n + stalled_n) — of matched trips at a scheduled
 *     THROUGH stop (see vehicles.ts deriveRouteMovementMetric; terminal
 *     layovers are excluded by design), the share that moved to a new stop.
 *     The disrupted/normal axis.
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

import { schedule_bin, tod_bin } from './hmm';
import type { Observation } from './hmm';
import { advanceBaselineFor, scheduleRateFor, serviceBaselineFor } from './params';
import type { AdvanceBaselineCell, ServiceBaseline, TrainedParams } from './params';
import type { RegimeEntry } from './regime';
import type { MovementMetricDoc, ServiceMetricDoc, ServiceQuantiles } from './state';
import type { MovementRow } from './vehicles';
import type { ServiceRow } from './trip_updates';

// ON: the movement-derived condition is the published current state (movement-
// primary). Each tick's states are written for the next tick's snapshot to read;
// routes movement can't judge fall through to 'unknown', never an alert fallback.
export const MOVEMENT_STATE_PUBLISH = true;

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
// A large posterior drop only reads disrupted when the low advance count is also
// statistically significant against the cell baseline (binomial lower tail at or
// under this). Guards degenerate-low baselines — a short shuttle advances ~0 even
// when healthy, so a normal zero-advance tick would otherwise flip disrupted.
const CLASSIFY_ALPHA = 0.05;
// A route in service (>=1 dispatched train) in fewer than this fraction of usable
// ticks at its schedule bin reads not_scheduled — not suspended — when nothing is
// running now. Applied to the trainer's per-bin in-service rate (schedule_rate).
const NOT_SCHEDULED_MAX = 0.5;
export const MIN_MATCHED_TRIPS = 3; // advanced_n + stalled_n floor to make a cross-tick call

export type MovementCondition = 'normal' | 'disrupted' | 'suspended' | 'not_scheduled';

// P(X <= k) for X ~ Binomial(n, p) via an iterative pmf sum. Exact for the
// tick-level counts here (n well under ~50) and free of special functions, so it
// mirrors 1:1 in Python/viz. p is the cell baseline p0, floored off 0 upstream.
export function binomLowerTail(k: number, n: number, p: number): number {
  if (k >= n) return 1;
  if (k < 0) return 0;
  const q = 1 - p;
  let pmf = q ** n; // P(X = 0)
  let cdf = pmf;
  for (let i = 0; i < k; i++) {
    pmf *= ((n - i) / (i + 1)) * (p / q);
    cdf += pmf;
  }
  return cdf;
}

// P(X <= k) for X ~ Poisson(mu) via the same iterative pmf sum, sibling of
// binomLowerTail and mirrored 1:1 in Python (training/segments.py). Used by the
// segment throughput branch, where mu is expected traversals out of ONE stop on
// ONE route-direction over a ~25-minute window — bounded by physical headway at
// well under 50, so exp(-mu) never approaches underflow and no guard is needed.
export function poisLowerTail(k: number, mu: number): number {
  if (k < 0) return 0;
  if (mu <= 0) return 1;
  let pmf = Math.exp(-mu); // P(X = 0)
  let cdf = pmf;
  for (let i = 1; i <= k; i++) {
    pmf *= mu / i;
    cdf += pmf;
  }
  return Math.min(1, cdf);
}

// Beta-Binomial call against a baseline advance rate p0, three ways:
//   normal    — posterior advance rate above DISRUPTED_RATIO * p0.
//   disrupted — posterior at/under DISRUPTED_RATIO * p0 AND the low advance count
//               is significant against p0 (binomial lower tail <= CLASSIFY_ALPHA).
//   null      — too few matches, or a point-estimate drop not distinguishable from
//               a low-p0 normal fluctuation (a degenerate-baseline zero-advance
//               tick, not a stall).
// The one decision rule shared by the direction classifier and the segment
// classifier (segment_flow.ts), so the two never disagree. NOTE: with smoothed
// (decayed) counts the binomial tail is a tuned score, not a calibrated p-value —
// CLASSIFY_ALPHA is an empirical threshold.
export function classifyAdvance(
  advancedN: number,
  stalledN: number,
  p0: number,
): 'normal' | 'disrupted' | null {
  const matched = advancedN + stalledN;
  if (matched < MIN_MATCHED_TRIPS) return null;
  const post =
    (CLASSIFY_PRIOR_STRENGTH * p0 + advancedN) / (CLASSIFY_PRIOR_STRENGTH + matched);
  if (post > DISRUPTED_RATIO * p0) return 'normal';
  return binomLowerTail(advancedN, matched, p0) <= CLASSIFY_ALPHA ? 'disrupted' : null;
}

// Beta-Binomial call for one (route, direction) at one tick, keyed on the cell's
// own baseline. null cell -> can't judge.
function classifyDirection(
  advancedN: number,
  stalledN: number,
  cell: AdvanceBaselineCell | null,
): 'normal' | 'disrupted' | null {
  if (!cell) return null;
  return classifyAdvance(advancedN, stalledN, cell.p0);
}

export function deriveMovementState(
  routeId: string,
  move: MovementRow | undefined,
  svc: ServiceRow | undefined,
  trained: TrainedParams | null,
  observedAt: number,
): MovementCondition | null {
  // No trains physically present? A route with dispatched trains or vehicles is
  // running — classify it by movement below, whatever the trip-updates lag says.
  if (move === undefined || move.vehicles_n === 0) {
    if (svc === undefined || svc.assigned_n === 0) {
      // Nothing running and nothing dispatched: a planned gap where the route
      // rarely runs at this bin, else a suspension. An unknown or normally-running
      // schedule stays suspended — never downgrade a real outage.
      const rate = scheduleRateFor(trained, routeId, schedule_bin(observedAt));
      return rate !== null && rate < NOT_SCHEDULED_MAX ? 'not_scheduled' : 'suspended';
    }
    // trip-updates shows assigned trains but none in the vehicle feed: a feed
    // inconsistency we can't confirm — abstain.
    return null;
  }
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
 * Per-route movement-derived condition for this tick. Routes either feed saw are
 * judged from their movement/service (deriveMovementState); routes the trainer
 * knows but neither feed saw are emitted as not_scheduled when their in-service
 * rate at this schedule bin is confidently low (a planned gap, e.g. a rush-only
 * line off-peak). Absences at a normally-running or unknown bin are left out —
 * outage vs feed gap is ambiguous there, so the caller falls back to the
 * alert/HMM condition.
 */
export function deriveMovementStates(
  moveRows: Map<string, MovementRow>,
  svcRows: Map<string, ServiceRow>,
  trained: TrainedParams | null,
  observedAt: number,
): Record<string, MovementCondition> {
  const out: Record<string, MovementCondition> = {};
  const seen = new Set<string>([...moveRows.keys(), ...svcRows.keys()]);
  for (const route of seen) {
    const state = deriveMovementState(
      route,
      moveRows.get(route),
      svcRows.get(route),
      trained,
      observedAt,
    );
    if (state !== null) out[route] = state;
  }
  // Routes the trainer knows but neither feed saw this tick: a scheduled-off route
  // (in-service rate below the gate at this bin) reads not_scheduled.
  if (trained?.scheduleRate) {
    const bin = schedule_bin(observedAt);
    for (const route of Object.keys(trained.scheduleRate)) {
      if (seen.has(route)) continue;
      const rate = scheduleRateFor(trained, route, bin);
      if (rate !== null && rate < NOT_SCHEDULED_MAX) out[route] = 'not_scheduled';
    }
  }
  return out;
}

export type ServiceCondition = 'normal' | 'degraded' | 'unknown';

// Hysteresis band and debounce, ported from the offline degradation label
// (load_r2.derive_actual_recovery) so the published axis and the grading truth
// agree: a route degrades below DEGRADE_RATIO and only recovers back above the
// higher RECOVER_RATIO, each confirmed for DEBOUNCE_TICKS consecutive ticks, so
// a route riding the threshold doesn't flap.
export const SERVICE_DEGRADE_RATIO = 0.5;
export const SERVICE_RECOVER_RATIO = 0.8;
export const SERVICE_DEBOUNCE_TICKS = 2;

// assigned_n against its own (route, schedule_bin) baseline for one route at one
// tick — the raw supply level. null when there is no reading or no baseline to
// judge it against.
export function serviceRatioFor(
  routeId: string,
  svc: ServiceRow | undefined,
  baseline: ServiceBaseline | null,
  observedAt: number,
): number | null {
  if (svc === undefined) return null;
  const median = baseline?.[routeId]?.[schedule_bin(observedAt)] ?? null;
  if (median === null || median <= 0) return null;
  return svc.assigned_n / median;
}

// The raw hysteresis call for one route at one tick — the target state fed to the
// regime clock, which then debounces it. Keyed off `priorState` (the route's
// currently committed service regime, undefined when never seen): once degraded,
// only a ratio back above RECOVER_RATIO returns 'normal'; otherwise a strict drop
// below DEGRADE_RATIO is what degrades. 'unknown' when the ratio can't be formed
// — the caller omits it so the clock holds the prior regime across a blind tick.
export function deriveServiceState(
  routeId: string,
  svc: ServiceRow | undefined,
  baseline: ServiceBaseline | null,
  observedAt: number,
  priorState?: ServiceCondition,
): ServiceCondition {
  const ratio = serviceRatioFor(routeId, svc, baseline, observedAt);
  if (ratio === null) return 'unknown';
  if (priorState === 'degraded') {
    return ratio >= SERVICE_RECOVER_RATIO ? 'normal' : 'degraded';
  }
  return ratio < SERVICE_DEGRADE_RATIO ? 'degraded' : 'normal';
}

/**
 * Per-route service-level call for this tick — a SUPPLY axis, orthogonal to
 * deriveMovementStates' FLOW axis: a route can have its trips pulled (degraded
 * here) while the trains still running advance fine (normal there), and the
 * reverse. `priorRegimes` is last tick's committed service regimes, so the
 * hysteresis band keys off the state actually published. Only judgeable routes
 * ('normal'/'degraded') are returned; 'unknown' is omitted so the regime clock
 * holds the prior regime across a blind tick, exactly as deriveMovementStates
 * does. Feed the result to advanceRegimes with SERVICE_DEBOUNCE_TICKS.
 */
export function deriveServiceStates(
  svcRows: Map<string, ServiceRow>,
  baseline: ServiceBaseline | null,
  observedAt: number,
  priorRegimes?: Record<string, { state: ServiceCondition }>,
): Record<string, ServiceCondition> {
  const out: Record<string, ServiceCondition> = {};
  for (const [route, svc] of svcRows) {
    const state = deriveServiceState(route, svc, baseline, observedAt, priorRegimes?.[route]?.state);
    if (state !== 'unknown') out[route] = state;
  }
  return out;
}

/**
 * Per-route raw service ratio for this tick — the magnitude behind the
 * service_condition axis. Judgeable routes only (null ratios omitted), so a
 * route absent here publishes service_ratio null, matching service_condition
 * 'unknown'. Not debounced: this is the latest reading, beside the debounced
 * regime the condition comes from.
 */
export function deriveServiceRatios(
  svcRows: Map<string, ServiceRow>,
  baseline: ServiceBaseline | null,
  observedAt: number,
): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [route, svc] of svcRows) {
    const ratio = serviceRatioFor(route, svc, baseline, observedAt);
    if (ratio !== null) out[route] = ratio;
  }
  return out;
}

// A cell's own {p10, p90} spread of assigned_n, or null when the trainer
// hasn't published quantiles for it (old sidecar, or a cell below its
// min_samples floor). Same schedule_bin keying as serviceRatioFor's baseline
// lookup.
export function serviceQuantileFor(
  routeId: string,
  quantiles: ServiceQuantiles | null,
  observedAt: number,
): { p10: number; p90: number } | null {
  return quantiles?.[routeId]?.[schedule_bin(observedAt)] ?? null;
}

// A cell's spread normalised onto the SAME scale as serviceRatioFor's output
// (p10/median, p90/median), so it can be drawn as ticks on the same meter. null
// when the cell has no published quantiles, or its baseline median is <= 0 —
// mirrors serviceRatioFor's own null gate on the median.
export function serviceQuantileRatiosFor(
  routeId: string,
  baseline: ServiceBaseline | null,
  quantiles: ServiceQuantiles | null,
  observedAt: number,
): { low: number; high: number } | null {
  const median = baseline?.[routeId]?.[schedule_bin(observedAt)] ?? null;
  if (median === null || median <= 0) return null;
  const cell = serviceQuantileFor(routeId, quantiles, observedAt);
  if (cell === null) return null;
  return { low: cell.p10 / median, high: cell.p90 / median };
}

/**
 * Per-route quantile-derived low/high ratios for this tick, mirroring
 * deriveServiceRatios: only routes with both a reading and a quantile cell are
 * returned, so a route absent here publishes service_low_ratio/
 * service_high_ratio null. Persist alongside service_ratios so the snapshot
 * can render the "notably high" mark without a global constant.
 */
export function deriveServiceQuantileRatios(
  svcRows: Map<string, ServiceRow>,
  baseline: ServiceBaseline | null,
  quantiles: ServiceQuantiles | null,
  observedAt: number,
): Record<string, { low: number; high: number }> {
  const out: Record<string, { low: number; high: number }> = {};
  for (const [route] of svcRows) {
    const ratios = serviceQuantileRatiosFor(route, baseline, quantiles, observedAt);
    if (ratios !== null) out[route] = ratios;
  }
  return out;
}

// Where this tick's supply reading sits within its OWN same-daypart baseline
// distribution, as a 0-100 percentile — the "how bad is it right now vs usual"
// gauge. All three inputs are on the ratio scale serviceRatioFor/
// serviceQuantileRatiosFor already publish: `ratio` = assigned_n / median,
// `low` = p10 / median, `high` = p90 / median. It places `ratio` by
// piecewise-linear interpolation through the three anchors the baseline carries
// — (low, 10), (1.0, 50), (high, 90) — so the reading is EXACT at each real
// quantile and approximate only between them.
//
// The tails are handled deliberately, not by extrapolating a fabricated slope
// (journal 2026-08-22: a p50->p90 multiple can't represent a bimodal cell and
// mis-ranks an ordinary second mode above a genuine outlier). Below p10 it
// interpolates toward the one anchor that is not a guess — zero assigned trains,
// the hard floor of a non-negative count, at the 0th percentile — so the "bad"
// direction keeps its resolution. Above p90 it SATURATES at 90 rather than
// projecting past the last observed quantile: the reading is honestly "in this
// cell's own top decile", no finer. A LOW percentile means fewer trains than
// usual for this daypart; it is a percentile of the baseline, never a forecast.
//
// null whenever any input is null (no reading, or no published quantile spread),
// so a route absent here publishes service_percentile null — the same lifecycle
// as service_ratio. null too when the cell is degenerate (p10 >= median or
// p90 <= median): the anchors aren't ordered, so no honest placement exists.
export function servicePercentile(
  ratio: number | null,
  low: number | null,
  high: number | null,
): number | null {
  if (ratio === null || low === null || high === null) return null;
  if (!(low < 1 && high > 1)) return null;
  let pct: number;
  if (ratio <= 0) {
    pct = 0;
  } else if (ratio < low) {
    // (0, 0) -> (low, 10): a real floor, not an extrapolated slope.
    pct = (ratio / low) * 10;
  } else if (ratio < 1) {
    pct = 10 + ((ratio - low) / (1 - low)) * 40;
  } else if (ratio < high) {
    pct = 50 + ((ratio - 1) / (high - 1)) * 40;
  } else {
    // At or above the cell's own p90: saturate, do not project past it.
    pct = 90;
  }
  return Math.round(Math.max(0, Math.min(100, pct)));
}

/**
 * Seed every newly observed route as 'normal' before the service regime clock.
 *
 * derive_actual_recovery starts every route NOT degraded and requires the
 * debounce even for the FIRST drop. advanceRegimes instead commits a brand-new
 * key's first call immediately, so a cold start (a fresh service_regimes doc
 * after deploy, or a regime that expired after a long blind gap) would flip
 * straight to 'degraded' on a single low tick. Seeding 'normal' makes that first
 * degrade go through the same 2-tick debounce as every later one, matching the
 * offline label exactly rather than only after the first commit.
 */
export function seedNormalServiceRegimes(
  prev: Record<string, RegimeEntry<ServiceCondition>> | null | undefined,
  observed: Record<string, ServiceCondition>,
  observedAt: number,
): Record<string, RegimeEntry<ServiceCondition>> {
  const seeded: Record<string, RegimeEntry<ServiceCondition>> = { ...(prev ?? {}) };
  for (const route of Object.keys(observed)) {
    seeded[route] ??= {
      state: 'normal',
      entered_at: observedAt,
      last_seen_at: observedAt,
      pending: null,
      pending_since: 0,
      pending_run: 0,
    };
  }
  return seeded;
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
