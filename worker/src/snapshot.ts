/**
 * Render and publish the public snapshot.
 *
 * Shape matches src/momentarily/schema.py's Snapshot. Surfaces whose upstream
 * source isn't wired yet (observations, stations, bridges, tunnels) emit as
 * empty placeholders so the schema_version=1 contract stays honored. alerts,
 * routes, and equipment are populated from the data already fetched each tick.
 *
 * Output is publicly readable at https://feed.momentarily.nyc/v1/snapshot.json
 * via the R2 custom domain. Cache headers per ADR (max-age=60, s-maxage=300).
 */

import type { RouteRoll } from './alpha';
import type { Provenance } from './buildinfo';
import { codeProvenance } from './buildinfo';
import type { AlertOut, AlertRef, DirectionAlerts, RouteSnapshot } from './derive';
import { buildRoutes, metaForRoute } from './derive';
import { conditionalRecovery, pLeaveBy } from './dwell';
import { HYSTERESIS_TICKS, N_STATES, PUBLISHED_UNKNOWN, STATES, projectForward } from './hmm';
import type { PublishedLabel } from './hmm';
import { NO_ALERTS_FALLBACK, categoryForLabel, coarseStatus } from './mapping';
import type { TrainedParams } from './params';
import { dwellForRouteState, paramsForRoute } from './params';
import type { EquipmentOut, StationStatus } from './stations';
import type { StationOut } from './stations_static';

// Above this, the geometric dwell estimate is uninformative — a trained
// self-loop ≈ 1 means the model has no evidence the regime ever ends (typical
// of open-ended planned work). Clamp + flag rather than publish "34 days".
const MAX_RECOVERY_MINUTES = 1440;

// Fast-attack threshold for surfacing `condition`. When the filter is
// highly confident in a state that disagrees with the hysteresis-gated
// published label, we surface the filter's view instead of the lagged
// label. Hysteresis still protects the underlying publish state machine
// from flapping on ambiguous evidence; this only governs what consumers
// see. See momentarily-8ga.
const FAST_ATTACK_PROB = 0.9;

// Movement state is carried from the prior tick, normally ~5 min old. If the
// vehicle feeds stall, don't keep publishing a frozen reading indefinitely —
// past this age the route falls back to the alert/HMM condition. Six ticks.
const MAX_MOVEMENT_STATE_AGE_SEC = 1800;

const SNAPSHOT_KEY = 'v1/snapshot.json';

export const SCHEMA_VERSION = '1';

export const ATTRIBUTION =
  'Snapshot built from MTA GTFS-RT feeds via api.mta.info. '
  + 'Published by Momentarily (https://feed.momentarily.nyc). '
  + 'Not affiliated with the MTA.';

interface Inference {
  condition: string;
  recovery_minutes: number;
  is_disrupted: boolean;
  p_normal: number;
  p_disrupted: number;
  p_suspended: number;
  regime_entered_at: number;
  regime_age_seconds: number;
  recovery_minutes_low: number;
  recovery_minutes_high: number;
  // True when the dwell estimate saturated MAX_RECOVERY_MINUTES — the regime
  // is so persistent the model can't bound when it ends. recovery_minutes and
  // its bounds are clamped to the ceiling in that case.
  recovery_indeterminate: boolean;
  p_normal_in_30min: number;
  p_normal_in_60min: number;
  p_normal_in_120min: number;
  model_warming_up: boolean;
  // Where recovery_minutes comes from: "schedule" is a deterministic lookup of
  // the planned-work resume time (no model uncertainty); "hmm" is the dwell
  // estimate. The grader excludes "schedule" rows from HMM calibration.
  recovery_source: 'hmm' | 'schedule';
  // Announced resume time (epoch s) for schedule recovery; null for hmm.
  resumes_at: number | null;
  // now has passed resumes_at but the planned alert is still active — recovery
  // is clamped to 0 rather than counting down past the announced time.
  overdue: boolean;
}

interface RouteStatusOut {
  route_id: string;
  alerts: string[];
  // Severity axis — current state, observed from train movement when available,
  // else the hysteresis-stable HMM published label.
  condition: string;
  // Where `condition` came from this tick: 'movement' (observed), 'hmm' (alert-
  // derived fallback), or 'unknown' (no inference yet).
  condition_source: string;
  // Cause axis — our vocabulary, derived from the MTA alert_type.
  category: string;
  primary_alert_type: string | null;
  // Soft-deprecated: now derivable from condition + category. Kept for
  // existing consumers and the compat layer.
  label: string;
  by_direction: {
    northbound: DirectionAlerts;
    southbound: DirectionAlerts;
  };
  inference: Inference | null;
}

interface Freshness {
  subway_alerts: number | null;
  lirr_alerts: number | null;
  mnr_alerts: number | null;
  bus_alerts: number | null;
  path_alerts: number | null;
  ferry_alerts: number | null;
  ene: number | null;
  stations_static: number | null;
}

interface Accessibility {
  elevators_out: number;
  escalators_out: number;
  ada_pathways_degraded: number;
}

interface ModeRollup {
  routes_with_alerts: string[];
  alert_count: number;
  severity_max: number;
}

interface SystemStatus {
  by_mode: Record<string, ModeRollup>;
  accessibility: Accessibility;
  overall_label: string;
  condition: string | null;
  lines_disrupted_count: number;
  most_degraded_line: string | null;
  most_recovered_line: string | null;
}

interface CompatRouteSummary {
  north: string | null;
  south: string | null;
}

interface CompatServiceChangeSummary {
  both: string[];
  north: string[];
  south: string[];
}

interface CompatRoute {
  id: string;
  name: string;
  color: string;
  status: string;
  scheduled: boolean;
  direction_statuses: CompatRouteSummary | null;
  delay_summaries: CompatRouteSummary | null;
  service_irregularity_summaries: CompatRouteSummary | null;
  service_change_summaries: CompatServiceChangeSummary | null;
}

interface Compat {
  subwaynow_routes: Record<string, CompatRoute>;
}

interface Snapshot {
  schema_version: string;
  generated_at: number;
  provenance: Provenance;
  attribution: string;
  supported_modes: string[];
  freshness: Freshness;
  alerts: AlertOut[];
  observations: unknown[];
  routes: Record<string, unknown>;
  stations: Record<string, StationOut>;
  equipment: EquipmentOut[];
  bridges: unknown[];
  tunnels: unknown[];
  route_status: Record<string, RouteStatusOut>;
  station_status: Record<string, unknown>;
  system: SystemStatus;
  compat: Compat;
}

export function buildSnapshot(args: {
  generatedAt: number;
  alertsFreshness: number;
  routeSnapshots: Map<string, RouteSnapshot>;
  rolls: Record<string, RouteRoll>;
  trainedParams: TrainedParams | null;
  tickSeconds: number;
  /** Cached station_status, refreshed on hourly E&E fetches. Empty when
   * E&E hasn't been parsed yet (e.g. before the first hourly tick after
   * deploy). */
  stationStatuses?: Record<string, StationStatus>;
  eneFreshness?: number | null;
  /** Alerts active this tick — the atomic objects route_status IDs resolve
   * against. Empty only on a true alerts-feed gap. */
  alerts?: AlertOut[];
  /** Elevators/escalators currently out, cached from the hourly E&E fetch. */
  equipment?: EquipmentOut[];
  /** Static station metadata, cached from the daily 39hk-dx4f fetch. */
  stations?: Record<string, StationOut>;
  /** Epoch the served station metadata was fetched, or null before first fetch. */
  stationsStaticFreshness?: number | null;
  /** Last tick's movement-derived per-route condition. Routes present here have
   * their published `condition` observed from train movement; absent routes fall
   * back to the alert/HMM condition. Null/undefined before the first vehicle tick
   * after deploy. Lagged one tick (~5 min) — see state.MOVEMENT_STATE_KEY. */
  movementStates?: { observed_at: number; states: Record<string, string> } | null;
}): Snapshot {
  const route_status: Record<string, RouteStatusOut> = {};

  // Use movement state only while it's reasonably fresh; a long vehicle-feed
  // gap shouldn't pin a stale condition on the public surface.
  const movementFresh =
    args.movementStates != null &&
    args.generatedAt - args.movementStates.observed_at <= MAX_MOVEMENT_STATE_AGE_SEC;
  const movementStates = movementFresh ? args.movementStates : null;

  // Publish every route we have alpha for — good-service lines get their
  // inference too. Union with current routeSnapshots in case a route just got
  // its first alert this tick (alpha entry written after buildSnapshot reads).
  const allRouteIds = new Set<string>([
    ...Object.keys(args.rolls),
    ...args.routeSnapshots.keys(),
  ]);

  for (const routeId of allRouteIds) {
    const snap = args.routeSnapshots.get(routeId);
    const roll = args.rolls[routeId];
    const activeAlerts = snap?.active_alert_ids ?? [];
    const schedule: ScheduleFacts = {
      isNotScheduled: snap?.is_not_scheduled ?? false,
      hasRealtimeAlert: snap?.has_realtime_alert ?? false,
      scheduledResumeAt: snap?.scheduled_resume_at ?? null,
    };
    const inference: Inference | null = roll
      ? buildInference(
          roll,
          args.generatedAt,
          args.tickSeconds,
          routeId,
          args.trainedParams,
          activeAlerts.length,
          schedule,
        )
      : null;

    const label = snap?.coarse_label ?? NO_ALERTS_FALLBACK;
    // Current state is observed from train movement when we have a judgeable
    // reading; the alert-derived HMM condition is the fallback (cold start, feed
    // gap, too few cross-tick matches). not_scheduled is a planned non-run and
    // always wins — a route that isn't meant to run now reads neither disrupted
    // nor suspended just because no trains are moving.
    const hmmCondition = inference ? inference.condition : 'unknown';
    const movementCondition = movementStates?.states[routeId];
    const useMovement =
      movementCondition !== undefined && hmmCondition !== 'not_scheduled';
    route_status[routeId] = {
      route_id: routeId,
      alerts: activeAlerts,
      condition: useMovement ? movementCondition : hmmCondition,
      condition_source: useMovement ? 'movement' : inference ? 'hmm' : 'unknown',
      category: categoryForLabel(label),
      primary_alert_type: snap?.primary_alert_type ?? null,
      label,
      by_direction: snap?.by_direction ?? {
        northbound: { alerts: [], primary_alert_type: null },
        southbound: { alerts: [], primary_alert_type: null },
      },
      inference,
    };
  }

  const system = buildSystemStatus(route_status, args.routeSnapshots, args.stationStatuses ?? {});
  const compat = buildCompat(route_status, args.routeSnapshots);

  return {
    schema_version: SCHEMA_VERSION,
    generated_at: args.generatedAt,
    provenance: codeProvenance(),
    attribution: ATTRIBUTION,
    supported_modes: ['subway'],
    freshness: {
      subway_alerts: args.alertsFreshness,
      lirr_alerts: null,
      mnr_alerts: null,
      bus_alerts: null,
      path_alerts: null,
      ferry_alerts: null,
      ene: args.eneFreshness ?? null,
      stations_static: args.stationsStaticFreshness ?? null,
    },
    alerts: args.alerts ?? [],
    observations: [],
    routes: buildRoutes(),
    stations: args.stations ?? {},
    equipment: args.equipment ?? [],
    bridges: [],
    tunnels: [],
    route_status,
    station_status: args.stationStatuses ?? {},
    system,
    compat,
  };
}

function buildSystemStatus(
  routeStatuses: Record<string, RouteStatusOut>,
  routeSnapshots: Map<string, RouteSnapshot>,
  stationStatuses: Record<string, StationStatus>,
): SystemStatus {
  const routes_with_alerts: string[] = [];
  let alert_count = 0;
  let severity_max = 0;
  for (const [routeId, rs] of Object.entries(routeStatuses)) {
    if (rs.alerts.length > 0) {
      routes_with_alerts.push(routeId);
      alert_count += rs.alerts.length;
    }
    // not_scheduled is a planned non-disruption — keep it out of the disruption
    // rollups (severity_max here; lines_disrupted_count/most_degraded_line are
    // gated by is_disrupted, which is already false for it).
    if (rs.condition === 'not_scheduled') continue;
    const snap = routeSnapshots.get(routeId);
    if (snap && snap.severity_max > severity_max) severity_max = snap.severity_max;
  }
  routes_with_alerts.sort();

  let lines_disrupted_count = 0;
  let most_degraded_line: string | null = null;
  let mostDegradedScore = -1;
  let most_recovered_line: string | null = null;
  let mostRecoveredEnteredAt = -1;
  for (const [routeId, rs] of Object.entries(routeStatuses)) {
    const inf = rs.inference;
    // Count what's published: condition is movement-observed when available,
    // HMM-derived otherwise. Rank within (most degraded/recovered) still uses
    // the HMM's continuous probabilities and regime age, which movement lacks.
    const disrupted = rs.condition === 'disrupted' || rs.condition === 'suspended';
    if (disrupted) {
      lines_disrupted_count += 1;
      const score = inf ? inf.p_disrupted + inf.p_suspended : 1;
      if (score > mostDegradedScore) {
        mostDegradedScore = score;
        most_degraded_line = routeId;
      }
    } else if (rs.condition === 'normal' && inf && inf.regime_entered_at > mostRecoveredEnteredAt) {
      mostRecoveredEnteredAt = inf.regime_entered_at;
      most_recovered_line = routeId;
    }
  }

  return {
    by_mode: {
      subway: { routes_with_alerts, alert_count, severity_max },
    },
    accessibility: buildAccessibility(stationStatuses),
    overall_label:
      routes_with_alerts.length === 0
        ? 'All systems normal'
        : `Alerts on ${routes_with_alerts.length} subway lines`,
    condition: null,
    lines_disrupted_count,
    most_degraded_line,
    most_recovered_line,
  };
}

function buildAccessibility(
  stationStatuses: Record<string, StationStatus>,
): Accessibility {
  let elevators_out = 0;
  let escalators_out = 0;
  let ada_pathways_degraded = 0;
  for (const s of Object.values(stationStatuses)) {
    elevators_out += s.elevators_out;
    escalators_out += s.escalators_out;
    if (s.ada_status === 'ada_degraded') ada_pathways_degraded += 1;
  }
  return { elevators_out, escalators_out, ada_pathways_degraded };
}

function buildCompat(
  routeStatuses: Record<string, RouteStatusOut>,
  routeSnapshots: Map<string, RouteSnapshot>,
): Compat {
  const subwaynow_routes: Record<string, CompatRoute> = {};
  for (const [routeId, rs] of Object.entries(routeStatuses)) {
    const meta = metaForRoute(routeId);
    const snap = routeSnapshots.get(routeId);
    const refs = snap?.alerts ?? [];

    const direction_statuses: CompatRouteSummary = {
      north: rs.by_direction.northbound.primary_alert_type
        ? coarseStatus(rs.by_direction.northbound.primary_alert_type)
        : null,
      south: rs.by_direction.southbound.primary_alert_type
        ? coarseStatus(rs.by_direction.southbound.primary_alert_type)
        : null,
    };

    const northRefs = refs.filter((r) => r.direction_id === 0 || r.direction_id === null);
    const southRefs = refs.filter((r) => r.direction_id === 1 || r.direction_id === null);

    const delayKeywords = ['delay'];
    const irregularityKeywords = ['slow', 'reroute', 'skip'];
    const changeKeywords = ['service change', 'suspend', 'express', 'local'];

    const delay_summaries: CompatRouteSummary = {
      north: firstHeaderMatching(northRefs, delayKeywords),
      south: firstHeaderMatching(southRefs, delayKeywords),
    };
    const service_irregularity_summaries: CompatRouteSummary = {
      north: firstHeaderMatching(northRefs, irregularityKeywords),
      south: firstHeaderMatching(southRefs, irregularityKeywords),
    };
    const service_change_summaries: CompatServiceChangeSummary = {
      both: headersMatching(refs, changeKeywords),
      north: [],
      south: [],
    };

    // not_scheduled is a new condition value; render it as a scheduled gap so
    // the HomeAssistant integration doesn't choke on an unknown status.
    const notScheduled = rs.condition === 'not_scheduled';
    subwaynow_routes[routeId] = {
      id: routeId,
      name: meta.name,
      color: meta.color,
      status: notScheduled ? 'Not Scheduled' : rs.label,
      scheduled: !notScheduled,
      direction_statuses,
      delay_summaries,
      service_irregularity_summaries,
      service_change_summaries,
    };
  }
  return { subwaynow_routes };
}

function firstHeaderMatching(refs: AlertRef[], keywords: string[]): string | null {
  for (const r of refs) {
    if (!r.header_text) continue;
    const typeLower = r.alert_type.toLowerCase();
    if (keywords.some((k) => typeLower.includes(k))) return r.header_text;
  }
  return null;
}

function headersMatching(refs: AlertRef[], keywords: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const r of refs) {
    if (!r.header_text) continue;
    const typeLower = r.alert_type.toLowerCase();
    if (!keywords.some((k) => typeLower.includes(k))) continue;
    if (seen.has(r.header_text)) continue;
    seen.add(r.header_text);
    out.push(r.header_text);
  }
  return out;
}

interface ScheduleFacts {
  isNotScheduled: boolean;
  hasRealtimeAlert: boolean;
  scheduledResumeAt: number | null;
}

function buildInference(
  roll: RouteRoll,
  now: number,
  tickSeconds: number,
  routeId: string,
  trained: TrainedParams | null,
  activeAlertCount: number,
  schedule: ScheduleFacts,
): Inference {
  const probs = roll.filter.probabilities;
  const params = paramsForRoute(trained, routeId);

  // p_normal_in_X — project forward to find marginal P(normal in k min)
  const ticksFor = (minutes: number): number =>
    Math.max(1, Math.round((minutes * 60) / tickSeconds));
  const p30 = projectForward(roll.filter, params, ticksFor(30));
  const p60 = projectForward(roll.filter, params, ticksFor(60));
  const p120 = projectForward(roll.filter, params, ticksFor(120));
  // Geometric projection by default; overridden below with the empirical
  // recovery curve when a dwell cell exists (it's cause-aware and heavy-tailed,
  // where the projection is neither — roughly halves 120-min Brier).
  let p_normal_in_30 = p30[0];
  let p_normal_in_60 = p60[0];
  let p_normal_in_120 = p120[0];

  const argmaxIdx = argmaxOf(probs);

  const condition = resolveCondition(roll, activeAlertCount, schedule);

  // Recovery_minutes is "time until back to normal." Two sources, in order
  // of preference:
  //   1. Empirical dwell quantiles from the regime_transitions stream — heavy-
  //      tailed reality, not a geometric approximation. Prefers the cause-
  //      conditioned (route, condition, alert_type_at_entry) cell, falling back
  //      to the (route, condition) aggregate. Only used when the trainer
  //      included the cell (sample size above its floor).
  //   2. Geometric dwell from the trained transition self-loop — works
  //      everywhere but saturates at the clamp ceiling for any route with
  //      sustained planned-work alerts. See momentarily-w97.
  let recovery_minutes = 0;
  let recovery_minutes_low = 0;
  let recovery_minutes_high = 0;
  let recovery_indeterminate = false;
  let recovery_source: 'hmm' | 'schedule' = 'hmm';
  let resumes_at: number | null = null;
  let overdue = false;

  // A planned-work disruption announces its own resume time (the window end),
  // so recovery is a deterministic schedule lookup, not a dwell estimate — for
  // ALL planned_work, not just no-service. Real-time alerts have no trustworthy
  // end and keep HMM recovery; when both are present the real-time alert wins.
  const scheduleRecovery =
    condition !== 'normal'
    && !schedule.hasRealtimeAlert
    && schedule.scheduledResumeAt !== null;

  if (scheduleRecovery) {
    const resume = schedule.scheduledResumeAt!;
    recovery_source = 'schedule';
    resumes_at = resume;
    // now has passed the announced resume but the alert is still active this
    // tick — clamp to 0 rather than count down past it. Next tick an extension
    // or a newly-posted real-time alert takes over via precedence.
    overdue = now >= resume;
    const remaining = Math.max(0, Math.round((resume - now) / 60));
    recovery_minutes = remaining;
    recovery_minutes_low = remaining;
    recovery_minutes_high = remaining;
    // It's back at the announced time: P(normal in k) is 1 once the window end
    // falls within k minutes, else 0.
    const within = (mins: number): number => (resume <= now + mins * 60 ? 1 : 0);
    p_normal_in_30 = within(30);
    p_normal_in_60 = within(60);
    p_normal_in_120 = within(120);
  } else if (condition !== 'normal') {
    const clamp = (m: number): number => Math.min(m, MAX_RECOVERY_MINUTES);
    const empirical = dwellForRouteState(
      trained,
      routeId,
      condition,
      roll.alert_type_at_entry,
    );
    if (empirical !== null) {
      const secToMin = (s: number): number => Math.round(s / 60);
      // Condition on how long the regime has already lasted: for heavy-tailed
      // dwells the unconditional quantiles/fractions are only correct at
      // elapsed=0, so recovery is the *remaining* time.
      const elapsedSec = Math.max(0, now - roll.filter.regime_entered_at);
      // The dwell curve is all-cause — time until the regime ends, whether to
      // normal or by escalating to suspended. p_normal_in_X needs P(normal), not
      // P(exited), so weight the exit probability by the share of exits that go
      // to normal (from the transition matrix). Homogeneous approximation; a
      // competing-risks cumulative-incidence split is the proper version.
      const ci = condition === 'suspended' ? 2 : 1;
      const sl = params.transition[ci]![ci]!;
      const toNormal =
        sl < 1 ? Math.min(1, Math.max(0, params.transition[ci]![0]! / (1 - sl))) : 0;

      if (empirical.curve_sec !== undefined) {
        // p_normal: exit probability (tail-extrapolated past the curve) split to
        // the normal destination — kept meaningful once the regime outlives every
        // observed dwell, where recovery_minutes below goes indeterminate.
        const curve = empirical.curve_sec;
        p_normal_in_30 = pLeaveBy(curve, elapsedSec, 1800) * toNormal;
        p_normal_in_60 = pLeaveBy(curve, elapsedSec, 3600) * toNormal;
        p_normal_in_120 = pLeaveBy(curve, elapsedSec, 7200) * toNormal;
        const conditional = conditionalRecovery(curve, elapsedSec);
        if (conditional !== null) {
          recovery_minutes = clamp(secToMin(conditional.median_sec));
          recovery_minutes_low = clamp(secToMin(conditional.q25_sec));
          recovery_minutes_high = clamp(secToMin(conditional.q75_sec));
          recovery_indeterminate = recovery_minutes >= MAX_RECOVERY_MINUTES;
        } else {
          // Outlived every observed dwell — no trustworthy recovery time.
          recovery_minutes = MAX_RECOVERY_MINUTES;
          recovery_minutes_low = MAX_RECOVERY_MINUTES;
          recovery_minutes_high = MAX_RECOVERY_MINUTES;
          recovery_indeterminate = true;
        }
      } else {
        // Pre-curve params.json: unconditional cell values (legacy behavior
        // until the trainer republishes with curve_sec).
        recovery_minutes = clamp(secToMin(empirical.median_sec));
        recovery_minutes_low = clamp(secToMin(empirical.q25_sec));
        recovery_minutes_high = clamp(secToMin(empirical.q75_sec));
        recovery_indeterminate = recovery_minutes >= MAX_RECOVERY_MINUTES;
        if (empirical.recover_by_30 !== undefined) p_normal_in_30 = empirical.recover_by_30 * toNormal;
        if (empirical.recover_by_60 !== undefined) p_normal_in_60 = empirical.recover_by_60 * toNormal;
        if (empirical.recover_by_120 !== undefined)
          p_normal_in_120 = empirical.recover_by_120 * toNormal;
      }
    } else {
      const selfLoop = params.transition[argmaxIdx]![argmaxIdx]!;
      const dwellTicks = dwellQuantiles(selfLoop);
      const dwellToMinutes = (t: number): number => Math.round((t * tickSeconds) / 60);
      const rawMedian = dwellToMinutes(dwellTicks.median);
      recovery_indeterminate = rawMedian >= MAX_RECOVERY_MINUTES;
      recovery_minutes = clamp(rawMedian);
      recovery_minutes_low = clamp(dwellToMinutes(dwellTicks.q25));
      recovery_minutes_high = clamp(dwellToMinutes(dwellTicks.q75));
    }
  }

  // The filter is still settling when: the route just appeared (regime younger
  // than the hysteresis window), the published label hasn't cleared hysteresis,
  // or we're recovering from a feed gap ("unknown").
  const model_warming_up =
    roll.published.label === PUBLISHED_UNKNOWN
    || roll.published.pending_streak < HYSTERESIS_TICKS
    || now - roll.filter.regime_entered_at < HYSTERESIS_TICKS * tickSeconds;

  return {
    condition,
    recovery_minutes,
    // Tie to the gated condition so a no-alert route never counts as disrupted
    // (keeps lines_disrupted_count consistent with the published condition).
    // not_scheduled is a planned non-disruption — never counts.
    is_disrupted:
      condition !== 'not_scheduled' && activeAlertCount > 0 && probs[1] + probs[2] > 0.7,
    p_normal: probs[0],
    p_disrupted: probs[1],
    p_suspended: probs[2],
    regime_entered_at: roll.filter.regime_entered_at,
    regime_age_seconds: Math.max(0, now - roll.filter.regime_entered_at),
    recovery_minutes_low,
    recovery_minutes_high,
    recovery_indeterminate,
    p_normal_in_30min: p_normal_in_30,
    p_normal_in_60min: p_normal_in_60,
    p_normal_in_120min: p_normal_in_120,
    model_warming_up,
    recovery_source,
    resumes_at,
    overdue,
  };
}

function argmaxOf(v: readonly [number, number, number]): 0 | 1 | 2 {
  if (v[0] >= v[1] && v[0] >= v[2]) return 0;
  if (v[1] >= v[2]) return 1;
  return 2;
}

/**
 * Decide which label to surface to consumers as `condition`.
 *
 *   - "unknown" published label (post-feed-gap) → use filter argmax
 *   - filter very confident (max p ≥ FAST_ATTACK_PROB) and disagrees with
 *     published.label → use filter argmax (skip the hysteresis lag)
 *   - otherwise → use the hysteresis-gated published.label
 *
 * The underlying publish state machine still respects HYSTERESIS_TICKS;
 * this only governs what we render. See momentarily-8ga.
 */
/**
 * Apply the condition precedence on top of the HMM label:
 *   1. real-time disruptive alert (lmm:alert:*) active → HMM condition (live
 *      reality wins, even if a planned alert is also active)
 *   2. else active planned "No Scheduled Service" → not_scheduled (off-timetable,
 *      not broken)
 *   3. else → HMM condition / normal
 */
function resolveCondition(
  roll: RouteRoll,
  activeAlertCount: number,
  schedule: ScheduleFacts,
): string {
  if (schedule.hasRealtimeAlert) return effectiveCondition(roll, activeAlertCount);
  if (schedule.isNotScheduled) return 'not_scheduled';
  return effectiveCondition(roll, activeAlertCount);
}

function effectiveCondition(roll: RouteRoll, activeAlertCount: number): PublishedLabel {
  // Consistency guardrail: every disruption signal the filter sees is derived
  // from alerts, so with zero active alerts the honest condition is `normal`.
  // This stops a stale or over-confident filter from publishing `disrupted`
  // with no alert to explain it, and keeps system.overall_label consistent with
  // lines_disrupted_count. See momentarily-13j.
  if (activeAlertCount === 0) return 'normal';
  const argmaxState = STATES[argmaxOf(roll.filter.probabilities)]!;
  if (roll.published.label === PUBLISHED_UNKNOWN) return argmaxState;
  const peakProb = roll.filter.probabilities[argmaxOf(roll.filter.probabilities)];
  if (peakProb >= FAST_ATTACK_PROB && argmaxState !== roll.published.label) {
    return argmaxState;
  }
  return roll.published.label;
}

function dwellQuantiles(selfLoop: number): {
  median: number;
  q25: number;
  q75: number;
} {
  const LARGE = 10_000;
  if (selfLoop >= 1.0) return { median: LARGE, q25: LARGE, q75: LARGE };
  if (selfLoop <= 0) return { median: 1, q25: 1, q75: 1 };
  const logSelf = Math.log(selfLoop);
  const q = (qv: number): number => {
    const target = 1 - qv;
    return target <= 0 ? LARGE : Math.max(1, Math.ceil(Math.log(target) / logSelf));
  };
  return { median: q(0.5), q25: q(0.25), q75: q(0.75) };
}

/**
 * Document-level corruption that makes the whole snapshot unusable to ANY
 * consumer (missing version, no timestamp, no provenance). These block the
 * publish. Per-route inference problems do NOT belong here — one bad route must
 * never black out the entire feed; those are scrubbed instead (below).
 */
export function snapshotFatalViolations(s: Snapshot): string[] {
  const v: string[] = [];
  if (!s.schema_version) v.push('schema_version is empty');
  if (!Number.isFinite(s.generated_at) || s.generated_at <= 0) {
    v.push(`generated_at invalid: ${s.generated_at}`);
  }
  if (!s.provenance || typeof s.provenance.code_sha !== 'string') {
    v.push('provenance.code_sha missing');
  }
  return v;
}

/**
 * Cross-surface consistency checks that don't corrupt a consumer but signal a
 * wiring regression — the rollups counting things the detail arrays then drop.
 * Warn-only: a self-contradictory feed is worse than a stale one only if it
 * also blacks out, so we log and keep publishing. The class of bug this catches
 * is exactly what shipped `alert_count: 14` next to `alerts: []`.
 */
export function snapshotConsistencyWarnings(s: Snapshot): string[] {
  const w: string[] = [];
  const subwayAlerts = s.system.by_mode.subway?.alert_count ?? 0;
  if (subwayAlerts > 0 && s.alerts.length === 0) {
    w.push(`system.alert_count=${subwayAlerts} but alerts[] is empty`);
  }
  const out =
    s.system.accessibility.elevators_out + s.system.accessibility.escalators_out;
  if (out > 0 && s.equipment.length === 0) {
    w.push(`accessibility reports ${out} units out but equipment[] is empty`);
  }
  return w;
}

/**
 * Null out any route inference carrying a non-finite (NaN/Infinity) number —
 * the only kind of value that genuinely poisons a consumer (it serializes to
 * `null` and breaks a numeric reader). The inference field is already nullable,
 * so a scrubbed route ships in a valid degraded state and the rest of the feed
 * publishes normally. Marginal floats (e.g. 1.0000001) are finite and ship
 * as-is — we do NOT range-check, because that once stalled the whole feed.
 * Mutates `s` in place; returns the route ids scrubbed (for logging).
 */
export function scrubCorruptInferences(s: Snapshot): string[] {
  const scrubbed: string[] = [];
  for (const [routeId, rs] of Object.entries(s.route_status)) {
    const inf = rs.inference;
    if (!inf) continue;
    const allFinite =
      Number.isFinite(inf.p_normal) &&
      Number.isFinite(inf.p_disrupted) &&
      Number.isFinite(inf.p_suspended) &&
      Number.isFinite(inf.p_normal_in_30min) &&
      Number.isFinite(inf.p_normal_in_60min) &&
      Number.isFinite(inf.p_normal_in_120min) &&
      Number.isFinite(inf.recovery_minutes) &&
      Number.isFinite(inf.recovery_minutes_low) &&
      Number.isFinite(inf.recovery_minutes_high);
    if (!allFinite) {
      rs.inference = null;
      scrubbed.push(routeId);
    }
  }
  return scrubbed;
}

export async function publishSnapshot(
  bucket: R2Bucket,
  snapshot: Snapshot,
): Promise<void> {
  // Scoped fail-safe: scrub corrupt per-route inferences (and keep publishing
  // everything else), and only refuse to publish on document-level corruption —
  // so the CDN keeps serving the last-good snapshot in that rare case. A single
  // bad route can never stale the whole feed.
  const scrubbed = scrubCorruptInferences(snapshot);
  if (scrubbed.length > 0) {
    console.warn(
      `publish: scrubbed non-finite inference on ${scrubbed.length} route(s): ${scrubbed.join(', ')}`,
    );
  }
  const fatal = snapshotFatalViolations(snapshot);
  if (fatal.length > 0) {
    throw new Error(`snapshot fatally malformed, not publishing: ${fatal.join('; ')}`);
  }
  const inconsistencies = snapshotConsistencyWarnings(snapshot);
  if (inconsistencies.length > 0) {
    console.warn(`publish: snapshot consistency: ${inconsistencies.join('; ')}`);
  }
  await bucket.put(SNAPSHOT_KEY, JSON.stringify(snapshot), {
    httpMetadata: {
      contentType: 'application/json',
      cacheControl: 'public, max-age=60, s-maxage=300',
    },
  });
}

// Re-export for the entrypoint to use, no need to import N_STATES directly.
export { N_STATES };
export const TICK_SECONDS = 300;
export const NO_ALERTS = NO_ALERTS_FALLBACK;
