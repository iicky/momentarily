/**
 * Build per-route observations + derived status from the MTA alerts payload.
 *
 * The Worker fetches the alerts feed each cron tick, computes one Observation
 * per route at that tick, runs the HMM filter, and emits per-route status.
 * Station-level and equipment derivation happens elsewhere (E&E feed).
 */

import type { Observation } from './hmm';
import { tod_bin } from './hmm';
import { coarseStatus, NO_ALERTS_FALLBACK } from './mapping';

interface RouteEntityRef {
  alert_id: string;
  alert_type: string;
  sort_order: number;
  direction_id: number | null;
  active_period: ReadonlyArray<{ start?: number; end?: number }>;
}

export interface RouteSnapshot {
  route_id: string;
  observation: Observation;
  /** alert_ids active for this route at this tick */
  active_alert_ids: string[];
  /** Highest-severity alert_type active on this route, or null if none */
  primary_alert_type: string | null;
  /** coarseStatus(primary_alert_type) — short human label */
  coarse_label: string;
  /** by_direction: northbound/southbound primary types */
  by_direction: {
    northbound: string | null;
    southbound: string | null;
  };
}

/**
 * Walk the alerts payload, group active alerts by route, and produce per-route
 * snapshots at the given tick.
 */
export function deriveRouteSnapshots(
  alertsPayload: unknown,
  observedAt: number,
): Map<string, RouteSnapshot> {
  const byRoute = new Map<string, RouteEntityRef[]>();

  for (const entity of extractEntities(alertsPayload)) {
    const ref = parseAlertEntity(entity);
    if (!ref) continue;
    if (!isActiveAt(ref.active_period, observedAt)) continue;

    for (const route of ref.routes) {
      const arr = byRoute.get(route.route_id);
      const item = {
        alert_id: ref.alert_id,
        alert_type: ref.alert_type,
        sort_order: route.sort_order,
        direction_id: route.direction_id,
        active_period: ref.active_period,
      };
      if (arr) arr.push(item);
      else byRoute.set(route.route_id, [item]);
    }
  }

  const tick = tod_bin(observedAt);
  const out = new Map<string, RouteSnapshot>();
  for (const [routeId, alerts] of byRoute) {
    out.set(routeId, buildRouteSnapshot(routeId, alerts, observedAt, tick));
  }
  return out;
}

function buildRouteSnapshot(
  routeId: string,
  alerts: RouteEntityRef[],
  observedAt: number,
  todBinValue: number,
): RouteSnapshot {
  const primary = pickPrimary(alerts);
  const types = alerts.map((a) => a.alert_type);

  const observation: Observation = {
    alert_count: alerts.length,
    severity_sum: alerts.reduce((acc, a) => acc + a.sort_order, 0),
    has_suspended_alert: anyMatch(types, ['Suspend', 'No Trains', 'No Scheduled Service']),
    has_delays: anyMatch(types, ['Delays', 'Severe Delays']),
    has_service_change: anyMatch(
      types,
      [
        'Service Change',
        'Trains Rerouted',
        'Reroute',
        'Stops Skipped',
        'Express to Local',
        'Local to Express',
      ],
      'Planned -',
    ),
    has_planned: types.some((t) => t.startsWith('Planned -')),
    tod_bin: todBinValue,
  };

  return {
    route_id: routeId,
    observation,
    active_alert_ids: alerts.map((a) => a.alert_id),
    primary_alert_type: primary?.alert_type ?? null,
    coarse_label: primary ? coarseStatus(primary.alert_type) : NO_ALERTS_FALLBACK,
    by_direction: splitByDirection(alerts),
  };
}

function splitByDirection(alerts: RouteEntityRef[]): {
  northbound: string | null;
  southbound: string | null;
} {
  const north: RouteEntityRef[] = [];
  const south: RouteEntityRef[] = [];
  for (const a of alerts) {
    const d = a.direction_id;
    if (d === 0 || d === null) north.push(a);
    if (d === 1 || d === null) south.push(a);
  }
  return {
    northbound: pickPrimary(north)?.alert_type ?? null,
    southbound: pickPrimary(south)?.alert_type ?? null,
  };
}

function pickPrimary(alerts: RouteEntityRef[]): RouteEntityRef | null {
  if (alerts.length === 0) return null;
  return alerts.reduce(
    (best, a) => (a.sort_order > (best?.sort_order ?? -1) ? a : best),
    null as RouteEntityRef | null,
  );
}

function anyMatch(
  types: string[],
  needles: string[],
  excludePrefix?: string,
): boolean {
  for (const at of types) {
    if (excludePrefix && at.startsWith(excludePrefix)) continue;
    if (needles.some((n) => at.includes(n))) return true;
  }
  return false;
}

function isActiveAt(
  periods: ReadonlyArray<{ start?: number; end?: number }>,
  now: number,
): boolean {
  if (periods.length === 0) return true;
  for (const p of periods) {
    const start = p.start ?? 0;
    const end = p.end ?? 9_999_999_999;
    if (start <= now && now <= end) return true;
  }
  return false;
}

// --- payload extraction (loose, defensive) ---

function extractEntities(payload: unknown): unknown[] {
  if (!payload || typeof payload !== 'object') return [];
  const entity = (payload as { entity?: unknown }).entity;
  return Array.isArray(entity) ? entity : [];
}

const SORT_ORDER_RE = /:(\d+)$/;

function parseAlertEntity(entity: unknown): {
  alert_id: string;
  alert_type: string;
  active_period: ReadonlyArray<{ start?: number; end?: number }>;
  routes: Array<{ route_id: string; sort_order: number; direction_id: number | null }>;
} | null {
  if (!entity || typeof entity !== 'object') return null;
  const id = (entity as { id?: unknown }).id;
  if (typeof id !== 'string') return null;
  const inner = (entity as { alert?: unknown }).alert;
  if (!inner || typeof inner !== 'object') return null;

  const mercury = (inner as { 'transit_realtime.mercury_alert'?: unknown })[
    'transit_realtime.mercury_alert'
  ];
  const alertType =
    mercury && typeof mercury === 'object'
      ? ((mercury as { alert_type?: unknown }).alert_type as string | undefined)
      : undefined;
  if (typeof alertType !== 'string') return null;

  const periodsRaw = (inner as { active_period?: unknown }).active_period;
  const active_period: Array<{ start?: number; end?: number }> = [];
  if (Array.isArray(periodsRaw)) {
    for (const p of periodsRaw) {
      if (p && typeof p === 'object') {
        const start = (p as { start?: unknown }).start;
        const end = (p as { end?: unknown }).end;
        const period: { start?: number; end?: number } = {};
        if (typeof start === 'number') period.start = start;
        if (typeof end === 'number') period.end = end;
        active_period.push(period);
      }
    }
  }

  const entitiesRaw = (inner as { informed_entity?: unknown }).informed_entity;
  const routes: Array<{
    route_id: string;
    sort_order: number;
    direction_id: number | null;
  }> = [];
  if (Array.isArray(entitiesRaw)) {
    for (const e of entitiesRaw) {
      if (!e || typeof e !== 'object') continue;
      const routeId = (e as { route_id?: unknown }).route_id;
      if (typeof routeId !== 'string') continue;
      const direction = (e as { direction_id?: unknown }).direction_id;
      const selector = (e as { 'transit_realtime.mercury_entity_selector'?: unknown })[
        'transit_realtime.mercury_entity_selector'
      ];
      let sortOrder = 0;
      if (selector && typeof selector === 'object') {
        const raw = (selector as { sort_order?: unknown }).sort_order;
        if (typeof raw === 'string') {
          const match = SORT_ORDER_RE.exec(raw);
          if (match) sortOrder = Number.parseInt(match[1]!, 10);
        }
      }
      routes.push({
        route_id: routeId,
        sort_order: sortOrder,
        direction_id: typeof direction === 'number' ? direction : null,
      });
    }
  }
  if (routes.length === 0) return null;

  return { alert_id: id, alert_type: alertType, active_period, routes };
}
