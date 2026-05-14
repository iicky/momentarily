/**
 * Render and publish the public snapshot.
 *
 * Shape matches src/momentarily/schema.py's Snapshot. Top-level fields whose
 * data sources aren't wired up yet (alerts, observations, routes, stations,
 * equipment, bridges, tunnels, station_status, compat) are emitted as empty
 * placeholders so the schema_version=1 contract stays honored.
 *
 * Output is publicly readable at https://feed.momentarily.nyc/v1/snapshot.json
 * via the R2 custom domain. Cache headers per ADR (max-age=60, s-maxage=300).
 */

import type { RouteRoll } from './alpha';
import type { AlertRef, DirectionAlerts, RouteSnapshot } from './derive';
import { metaForRoute } from './derive';
import { N_STATES, STATES, projectForward } from './hmm';
import { NO_ALERTS_FALLBACK, coarseStatus } from './mapping';
import type { TrainedParams } from './params';
import { paramsForRoute } from './params';

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
  p_normal_in_30min: number;
  p_normal_in_60min: number;
  p_normal_in_120min: number;
  model_warming_up: boolean;
}

interface RouteStatusOut {
  route_id: string;
  alerts: string[];
  primary_alert_type: string | null;
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
  attribution: string;
  supported_modes: string[];
  freshness: Freshness;
  alerts: unknown[];
  observations: unknown[];
  routes: Record<string, unknown>;
  stations: Record<string, unknown>;
  equipment: unknown[];
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
}): Snapshot {
  const route_status: Record<string, RouteStatusOut> = {};

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
    const inference: Inference | null = roll
      ? buildInference(roll, args.generatedAt, args.tickSeconds, routeId, args.trainedParams)
      : null;

    route_status[routeId] = {
      route_id: routeId,
      alerts: snap?.active_alert_ids ?? [],
      primary_alert_type: snap?.primary_alert_type ?? null,
      label: snap?.coarse_label ?? NO_ALERTS_FALLBACK,
      by_direction: snap?.by_direction ?? {
        northbound: { alerts: [], primary_alert_type: null },
        southbound: { alerts: [], primary_alert_type: null },
      },
      inference,
    };
  }

  const system = buildSystemStatus(route_status, args.routeSnapshots);
  const compat = buildCompat(route_status, args.routeSnapshots);

  return {
    schema_version: SCHEMA_VERSION,
    generated_at: args.generatedAt,
    attribution: ATTRIBUTION,
    supported_modes: ['subway'],
    freshness: {
      subway_alerts: args.alertsFreshness,
      lirr_alerts: null,
      mnr_alerts: null,
      bus_alerts: null,
      path_alerts: null,
      ferry_alerts: null,
      ene: null,
      stations_static: null,
    },
    alerts: [],
    observations: [],
    routes: {},
    stations: {},
    equipment: [],
    bridges: [],
    tunnels: [],
    route_status,
    station_status: {},
    system,
    compat,
  };
}

function buildSystemStatus(
  routeStatuses: Record<string, RouteStatusOut>,
  routeSnapshots: Map<string, RouteSnapshot>,
): SystemStatus {
  const routes_with_alerts: string[] = [];
  let alert_count = 0;
  let severity_max = 0;
  for (const [routeId, rs] of Object.entries(routeStatuses)) {
    if (rs.alerts.length > 0) {
      routes_with_alerts.push(routeId);
      alert_count += rs.alerts.length;
    }
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
    if (!inf) continue;
    if (inf.is_disrupted) {
      lines_disrupted_count += 1;
      const score = inf.p_disrupted + inf.p_suspended;
      if (score > mostDegradedScore) {
        mostDegradedScore = score;
        most_degraded_line = routeId;
      }
    } else if (inf.condition === 'normal' && inf.regime_entered_at > mostRecoveredEnteredAt) {
      mostRecoveredEnteredAt = inf.regime_entered_at;
      most_recovered_line = routeId;
    }
  }

  return {
    by_mode: {
      subway: { routes_with_alerts, alert_count, severity_max },
    },
    accessibility: { elevators_out: 0, escalators_out: 0, ada_pathways_degraded: 0 },
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

    subwaynow_routes[routeId] = {
      id: routeId,
      name: meta.name,
      color: meta.color,
      status: rs.label,
      scheduled: true,
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

function buildInference(
  roll: RouteRoll,
  now: number,
  tickSeconds: number,
  routeId: string,
  trained: TrainedParams | null,
): Inference {
  const probs = roll.filter.probabilities;
  const params = paramsForRoute(trained, routeId);

  // p_normal_in_X — project forward to find marginal P(normal in k min)
  const ticksFor = (minutes: number): number =>
    Math.max(1, Math.round((minutes * 60) / tickSeconds));
  const p30 = projectForward(roll.filter, params, ticksFor(30));
  const p60 = projectForward(roll.filter, params, ticksFor(60));
  const p120 = projectForward(roll.filter, params, ticksFor(120));

  // Dwell math, using the same percentile geometry as Python's expected_dwell_ticks
  const argmaxIdx = argmaxOf(probs);
  const selfLoop = params.transition[argmaxIdx]![argmaxIdx]!;
  const dwellTicks = dwellQuantiles(selfLoop);
  const dwellToMinutes = (t: number): number => Math.round((t * tickSeconds) / 60);

  // Use the published label (hysteresis-stable) for `condition`. If still
  // "unknown" from a feed gap, fall back to argmax — Inference itself isn't
  // gated on hysteresis, that's the publish layer's job.
  const condition =
    roll.published.label === 'unknown' ? STATES[argmaxIdx]! : roll.published.label;

  return {
    condition,
    recovery_minutes: dwellToMinutes(dwellTicks.median),
    is_disrupted: probs[1] + probs[2] > 0.7,
    p_normal: probs[0],
    p_disrupted: probs[1],
    p_suspended: probs[2],
    regime_entered_at: roll.filter.regime_entered_at,
    regime_age_seconds: Math.max(0, now - roll.filter.regime_entered_at),
    recovery_minutes_low: dwellToMinutes(dwellTicks.q25),
    recovery_minutes_high: dwellToMinutes(dwellTicks.q75),
    p_normal_in_30min: p30[0],
    p_normal_in_60min: p60[0],
    p_normal_in_120min: p120[0],
    model_warming_up: false,
  };
}

function argmaxOf(v: readonly [number, number, number]): 0 | 1 | 2 {
  if (v[0] >= v[1] && v[0] >= v[2]) return 0;
  if (v[1] >= v[2]) return 1;
  return 2;
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

export async function publishSnapshot(
  bucket: R2Bucket,
  snapshot: Snapshot,
): Promise<void> {
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
