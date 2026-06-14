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

/**
 * Canonical NYC subway service route IDs as they appear in MTA GTFS-RT.
 * Inference runs for every route in this set each tick, so good-service lines
 * get a continuous history. New IDs observed in alerts auto-add to alpha state
 * regardless — this list is the lower bound, not the ceiling.
 */
export const SUBWAY_ROUTES: readonly string[] = [
  '1', '2', '3', '4', '5', '6', '7',
  'A', 'B', 'C', 'D', 'E', 'F', 'G',
  'J', 'L', 'M', 'N', 'Q', 'R', 'W', 'Z',
  'GS', 'FS', 'H',
  'SI',
] as const;

/**
 * Subway route metadata for compat.subwaynow_routes. Colors are MTA's official
 * bullet colors. Express variants and one-offs (7X, FX, etc.) inherit from the
 * base route via lookup fallback.
 */
export const SUBWAY_ROUTE_META: Readonly<Record<string, { name: string; color: string }>> = {
  '1': { name: '1', color: '#EE352E' },
  '2': { name: '2', color: '#EE352E' },
  '3': { name: '3', color: '#EE352E' },
  '4': { name: '4', color: '#00933C' },
  '5': { name: '5', color: '#00933C' },
  '6': { name: '6', color: '#00933C' },
  '7': { name: '7', color: '#B933AD' },
  A: { name: 'A', color: '#2850AD' },
  B: { name: 'B', color: '#FF6319' },
  C: { name: 'C', color: '#2850AD' },
  D: { name: 'D', color: '#FF6319' },
  E: { name: 'E', color: '#2850AD' },
  F: { name: 'F', color: '#FF6319' },
  G: { name: 'G', color: '#6CBE45' },
  J: { name: 'J', color: '#996633' },
  L: { name: 'L', color: '#A7A9AC' },
  M: { name: 'M', color: '#FF6319' },
  N: { name: 'N', color: '#FCCC0A' },
  Q: { name: 'Q', color: '#FCCC0A' },
  R: { name: 'R', color: '#FCCC0A' },
  W: { name: 'W', color: '#FCCC0A' },
  Z: { name: 'Z', color: '#996633' },
  GS: { name: 'S', color: '#808183' },
  FS: { name: 'S', color: '#808183' },
  H: { name: 'S', color: '#808183' },
  SI: { name: 'SIR', color: '#1F4F9F' },
};

/** Resolve metadata for a route_id, falling back to the base route for express
 * variants like 7X, FX. Returns a generic black if no match. */
export function metaForRoute(routeId: string): { name: string; color: string } {
  const direct = SUBWAY_ROUTE_META[routeId];
  if (direct) return direct;
  const base = routeId.replace(/X$/, '');
  const fallback = SUBWAY_ROUTE_META[base];
  if (fallback) return { name: routeId, color: fallback.color };
  return { name: routeId, color: '#000000' };
}

/** Quiet (no-alerts) observation for a route at this tick's tod_bin. */
export function quietObservation(observedAt: number): Observation {
  return {
    alert_count: 0,
    severity_sum: 0,
    has_suspended_alert: false,
    has_delays: false,
    has_service_change: false,
    has_planned: false,
    tod_bin: tod_bin(observedAt),
  };
}

export interface AlertRef {
  alert_id: string;
  alert_type: string;
  /** English header_text if present in the alert payload, else null */
  header_text: string | null;
  sort_order: number;
  direction_id: number | null;
}

interface RouteEntityRef extends AlertRef {
  active_period: ReadonlyArray<{ start?: number; end?: number }>;
}

export interface DirectionAlerts {
  alerts: string[];
  primary_alert_type: string | null;
}

export interface RouteSnapshot {
  route_id: string;
  observation: Observation;
  /** alert_ids active for this route at this tick (deduped) */
  active_alert_ids: string[];
  /** Full alert refs incl. header text, for compat-layer summaries */
  alerts: AlertRef[];
  /** Highest sort_order among active alerts on this route (0 if none) */
  severity_max: number;
  /** Highest-severity alert_type active on this route, or null if none */
  primary_alert_type: string | null;
  /** coarseStatus(primary_alert_type) — short human label */
  coarse_label: string;
  /** by_direction: northbound/southbound deduped alert IDs + primary type */
  by_direction: {
    northbound: DirectionAlerts;
    southbound: DirectionAlerts;
  };
  /** A real-time disruptive alert (lmm:alert:*) is active on this route. When
   *  set, the published condition stays HMM-derived even if a planned alert is
   *  also active. */
  has_realtime_alert: boolean;
  /** A planned "No Scheduled Service" alert is active — the line is off its
   *  timetable, not broken. Precedence (realtime first) is applied downstream. */
  is_not_scheduled: boolean;
  /** End of the planned-work window containing `now` (epoch s), or null when no
   *  planned window is active. Latest end among the route's currently-active
   *  planned windows — recomputed each tick, never the max across a recurring
   *  alert's future windows. */
  scheduled_resume_at: number | null;
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
      const item: RouteEntityRef = {
        alert_id: ref.alert_id,
        alert_type: ref.alert_type,
        header_text: ref.header_text,
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
  // Neither "Extra Service" nor "No Scheduled Service" is a disruption to
  // recover from — extra service is good news, no-service is planned absence
  // (overnight/weekend gaps, rush-only lines). Both stay in the display
  // surfaces (alerts list, primary/coarse label) but drop out of the HMM
  // observation so the filter reads quiet and stays normal. Mirrors
  // training/load_r2.py.
  const counted = alerts.filter(
    (a) =>
      !a.alert_type.includes('Extra Service')
      && !a.alert_type.includes('No Scheduled Service'),
  );
  const types = counted.map((a) => a.alert_type);

  const observation: Observation = {
    alert_count: counted.length,
    severity_sum: counted.reduce((acc, a) => acc + a.sort_order, 0),
    // "No Scheduled Service" is scheduled absence (overnight/weekend
    // non-service), not a suspension — keep it out of this flag. Mirrors
    // training/load_r2.py. See momentarily-vk0.3.
    has_suspended_alert: anyMatch(
      types,
      ['Suspend', 'No Trains'],
      'Planned -',
    ),
    has_delays: anyMatch(types, ['Delays', 'Severe Delays'], 'Planned -'),
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
    active_alert_ids: dedupeIds(alerts),
    alerts: dedupeRefs(alerts),
    severity_max: primary?.sort_order ?? 0,
    primary_alert_type: primary?.alert_type ?? null,
    coarse_label: primary ? coarseStatus(primary.alert_type) : NO_ALERTS_FALLBACK,
    by_direction: splitByDirection(alerts),
    has_realtime_alert: alerts.some((a) => isRealtimeId(a.alert_id)),
    is_not_scheduled: alerts.some((a) => a.alert_type.includes('No Scheduled Service')),
    scheduled_resume_at: scheduledResumeAt(alerts, observedAt),
  };
}

// Entity-id namespaces discriminate the two alert kinds more robustly than the
// alert_type string: lmm:alert:* are real-time disruptions (end is a rolling
// display TTL, never a resume time); lmm:planned_work:* carry a bounded
// active_period.end that IS the resume time — and cover Reduced/Extra/No
// Scheduled/Special Schedule, which lack the "Planned -" type prefix.
function isRealtimeId(alertId: string): boolean {
  return alertId.startsWith('lmm:alert:');
}

function isPlannedWorkId(alertId: string): boolean {
  return alertId.startsWith('lmm:planned_work:');
}

/**
 * End of the planned-work window containing `now`, or null. Among the route's
 * planned alerts, take the latest end across windows that contain `now` —
 * "when everything currently planned is done." Recurring alerts carry many
 * windows (months out); only the one bracketing `now` is the resume time, so we
 * never reach for max(end) across the whole alert.
 */
function scheduledResumeAt(alerts: RouteEntityRef[], now: number): number | null {
  let resume: number | null = null;
  for (const a of alerts) {
    if (!isPlannedWorkId(a.alert_id)) continue;
    for (const p of a.active_period) {
      const end = p.end;
      if (end === undefined) continue;
      const start = p.start ?? 0;
      if (start <= now && now <= end && (resume === null || end > resume)) {
        resume = end;
      }
    }
  }
  return resume;
}

function dedupeRefs(refs: RouteEntityRef[]): AlertRef[] {
  const seen = new Set<string>();
  const out: AlertRef[] = [];
  for (const r of refs) {
    if (seen.has(r.alert_id)) continue;
    seen.add(r.alert_id);
    out.push({
      alert_id: r.alert_id,
      alert_type: r.alert_type,
      header_text: r.header_text,
      sort_order: r.sort_order,
      direction_id: r.direction_id,
    });
  }
  return out;
}

function splitByDirection(alerts: RouteEntityRef[]): {
  northbound: DirectionAlerts;
  southbound: DirectionAlerts;
} {
  const north: RouteEntityRef[] = [];
  const south: RouteEntityRef[] = [];
  for (const a of alerts) {
    const d = a.direction_id;
    if (d === 0 || d === null) north.push(a);
    if (d === 1 || d === null) south.push(a);
  }
  return {
    northbound: {
      alerts: dedupeIds(north),
      primary_alert_type: pickPrimary(north)?.alert_type ?? null,
    },
    southbound: {
      alerts: dedupeIds(south),
      primary_alert_type: pickPrimary(south)?.alert_type ?? null,
    },
  };
}

function dedupeIds(refs: RouteEntityRef[]): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const r of refs) {
    if (seen.has(r.alert_id)) continue;
    seen.add(r.alert_id);
    out.push(r.alert_id);
  }
  return out;
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
  header_text: string | null;
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

  const headerText = extractEnglishHeader((inner as { header_text?: unknown }).header_text);

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

  return { alert_id: id, alert_type: alertType, header_text: headerText, active_period, routes };
}

function extractEnglishHeader(raw: unknown): string | null {
  if (!raw || typeof raw !== 'object') return null;
  const translation = (raw as { translation?: unknown }).translation;
  if (!Array.isArray(translation)) return null;
  let fallback: string | null = null;
  for (const t of translation) {
    if (!t || typeof t !== 'object') continue;
    const text = (t as { text?: unknown }).text;
    if (typeof text !== 'string') continue;
    const lang = (t as { language?: unknown }).language;
    if (lang === 'en' || lang === undefined || lang === null) return text;
    if (fallback === null) fallback = text;
  }
  return fallback;
}
