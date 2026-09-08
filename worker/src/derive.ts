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

/**
 * Classify every entity in the alerts payload into one of three per-entity
 * classes, for the system-wide sanity floor in index.ts step 4a:
 *
 *   - recognizedInScope    a structurally recognizable MTA alert that names a
 *                          canonical subway route in its informed_entity.
 *   - recognizedOutOfScope a structurally recognizable MTA alert that names NO
 *                          subway route — a station-scoped elevator/escalator
 *                          notice, an agency-wide notice, a non-subway alert.
 *   - unrecognizable       nothing that looks like an MTA alert at all.
 *
 * "Structurally recognizable" is an id plus an alert object carrying a
 * header_text OR the mercury alert_type — the fields the MTA schema always
 * carries — regardless of whether the alert is active or names a route.
 *
 * The gate degrades only when entities > 0 and recognizedInScope +
 * recognizedOutOfScope === 0 — the payload carried content but nothing in it
 * looks like an MTA alert at all, the structural schema break the flag exists to
 * catch. A feed of only out-of-scope notices, or only planned work whose
 * active_period is future/expired, is recognizable and reads as a quiet system,
 * never drift; only a genuinely unreadable payload abstains.
 */
export function classifyAlertsPayload(alertsPayload: unknown): {
  entities: number;
  recognizedInScope: number;
  recognizedOutOfScope: number;
  unrecognizable: number;
} {
  const entities = extractEntities(alertsPayload);
  let recognizedInScope = 0;
  let recognizedOutOfScope = 0;
  let unrecognizable = 0;
  for (const entity of entities) {
    if (!isRecognizableAlert(entity)) {
      unrecognizable += 1;
    } else if (namesSubwayRoute(entity)) {
      recognizedInScope += 1;
    } else {
      recognizedOutOfScope += 1;
    }
  }
  return { entities: entities.length, recognizedInScope, recognizedOutOfScope, unrecognizable };
}

/**
 * Whether an entity is structurally an MTA GTFS-RT alert, regardless of route
 * scope. Two arms, each looser than parseAlertEntity (which also demands a subway
 * route and an active window) so a route-less station notice still reads as an
 * alert:
 *   - the mercury alert_type is present — the alert-semantic marker the MTA
 *     schema always carries; sufficient on its own (a station elevator notice
 *     has it), so a genuine out-of-scope notice is never mistaken for drift.
 *   - OR a header_text AND a validated informed_entity selector list (an array
 *     with a route_id or stop_id). The selector requirement is what stops a bare
 *     header_text with no alert body and no selectors — a drift that stripped the
 *     alert down to nothing readable — from being blessed as a healthy alert and
 *     suppressing the degraded flag.
 * namesSubwayRoute then splits recognizable entities into in-scope / out-of-scope.
 */
function isRecognizableAlert(entity: unknown): boolean {
  if (!entity || typeof entity !== 'object') return false;
  const id = (entity as { id?: unknown }).id;
  if (typeof id !== 'string') return false;
  const inner = (entity as { alert?: unknown }).alert;
  if (!inner || typeof inner !== 'object') return false;
  const mercury = (inner as { 'transit_realtime.mercury_alert'?: unknown })[
    'transit_realtime.mercury_alert'
  ];
  if (
    !!mercury &&
    typeof mercury === 'object' &&
    typeof (mercury as { alert_type?: unknown }).alert_type === 'string'
  ) {
    return true;
  }
  const header = (inner as { header_text?: unknown }).header_text;
  return !!header && typeof header === 'object' && hasSelectorList(inner);
}

/** Whether an alert object carries a validated informed_entity selector list —
 * a non-empty array with at least one entry naming a route_id or a stop_id. A
 * header_text alone, with no selectors, is not enough to call an entity an
 * alert. */
function hasSelectorList(inner: object): boolean {
  const list = (inner as { informed_entity?: unknown }).informed_entity;
  if (!Array.isArray(list)) return false;
  for (const e of list) {
    if (!e || typeof e !== 'object') continue;
    const routeId = (e as { route_id?: unknown }).route_id;
    const stopId = (e as { stop_id?: unknown }).stop_id;
    if (typeof routeId === 'string' || typeof stopId === 'string') return true;
  }
  return false;
}

/** Whether a recognizable alert names at least one canonical subway route in its
 * informed_entity — the in-scope vs out-of-scope split. A station notice selects
 * a stop_id, never a route_id, so it reads out of scope. */
function namesSubwayRoute(entity: unknown): boolean {
  const inner = (entity as { alert?: unknown }).alert;
  if (!inner || typeof inner !== 'object') return false;
  const list = (inner as { informed_entity?: unknown }).informed_entity;
  if (!Array.isArray(list)) return false;
  for (const e of list) {
    if (!e || typeof e !== 'object') continue;
    const routeId = (e as { route_id?: unknown }).route_id;
    if (typeof routeId === 'string' && SUBWAY_ROUTES.includes(routeId)) return true;
  }
  return false;
}

/** A full Alert object for the snapshot's top-level `alerts` array — the atomic
 * unit consumers resolve the IDs in route_status/station_status against. */
export interface AlertOut {
  id: string;
  alert_type: string;
  source: string;
  sort_order: number | null;
  active_period: Array<{ start?: number; end?: number }>;
  header_text: { translation: Array<{ text: string; language: string }> } | null;
  informed_entities: Array<{ route_id: string; direction_id?: number }>;
}

/** Static per-route metadata for the snapshot's `routes` map. */
export interface RouteOut {
  id: string;
  mode: string;
  short_name: string;
  long_name: string | null;
  color: string | null;
  agency: string;
}

/**
 * Flatten the alerts payload into the deduped set of alerts active at this tick.
 * Same parse + active-window filter the per-route snapshots use, so the IDs they
 * reference always resolve to an object here.
 */
export function buildAlertList(alertsPayload: unknown, observedAt: number): AlertOut[] {
  const out: AlertOut[] = [];
  const seen = new Set<string>();
  for (const entity of extractEntities(alertsPayload)) {
    const ref = parseAlertEntity(entity);
    if (!ref) continue;
    if (!isActiveAt(ref.active_period, observedAt)) continue;
    if (seen.has(ref.alert_id)) continue;
    seen.add(ref.alert_id);
    out.push({
      id: ref.alert_id,
      alert_type: ref.alert_type,
      source: 'subway',
      sort_order: ref.routes.length
        ? Math.max(...ref.routes.map((r) => r.sort_order))
        : null,
      active_period: ref.active_period.map((p) => ({ ...p })),
      header_text: ref.header_text
        ? { translation: [{ text: ref.header_text, language: 'en' }] }
        : null,
      informed_entities: ref.routes.map((r) =>
        r.direction_id !== null
          ? { route_id: r.route_id, direction_id: r.direction_id }
          : { route_id: r.route_id },
      ),
    });
  }
  return out;
}

/** Static metadata for every canonical subway route. */
export function buildRoutes(): Record<string, RouteOut> {
  const out: Record<string, RouteOut> = {};
  for (const id of SUBWAY_ROUTES) {
    const meta = metaForRoute(id);
    out[id] = {
      id,
      mode: 'subway',
      short_name: meta.name,
      long_name: null,
      color: meta.color,
      agency: 'nyct_subway',
    };
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
  // Planned work drops out of the HMM disruption observation; real-time and
  // 'other' alerts count. This reads from the single three-way decision in
  // alertNamespace / countsAsDisruption below (Mirrors training/load.py +
  // load_r2.py is_planned_work_id).
  const counted = alerts.filter((a) => countsAsDisruption(alertNamespace(a.alert_id)));
  const types = counted.map((a) => a.alert_type);

  const observation: Observation = {
    alert_count: counted.length,
    severity_sum: counted.reduce((acc, a) => acc + a.sort_order, 0),
    // "No Scheduled Service" is scheduled absence (overnight/weekend
    // non-service), not a suspension — keep it out of this flag. Mirrors
    // training/load_r2.py.
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
    // Real-time namespace only (gates the schedule-recovery arm, which requires
    // !has_realtime_alert); 'other' is counted but deliberately excluded here.
    has_realtime_alert: alerts.some((a) => alertNamespace(a.alert_id) === 'realtime'),
    is_not_scheduled: alerts.some((a) => a.alert_type.includes('No Scheduled Service')),
    scheduled_resume_at: scheduledResumeAt(alerts, observedAt),
  };
}

// Every alert id falls into exactly one namespace. The partition is defined
// once here so alert_count and has_realtime_alert are derived from the same
// classification and cannot drift into the accidental non-complementarity that
// snapshot.ts's composition guard exists to catch:
//   - 'planned'  lmm:planned_work:* — bounded active_period.end IS the resume
//                time; also covers Reduced/Extra/No Scheduled/Special Schedule,
//                which lack the "Planned -" type prefix. Not a disruption to
//                recover from, so it drops out of the HMM observation.
//   - 'realtime' lmm:alert:* — a live disruption whose end is a rolling display
//                TTL, never a resume time.
//   - 'other'    an id in neither MTA namespace (e.g. lmm:situation:*). A real
//                third category, classified by a deliberate branch below — NOT
//                the residue of negating the planned check.
type AlertNamespace = 'planned' | 'realtime' | 'other';

function alertNamespace(alertId: string): AlertNamespace {
  if (alertId.startsWith('lmm:planned_work:')) return 'planned';
  if (alertId.startsWith('lmm:alert:')) return 'realtime';
  return 'other';
}

// Whether a namespace counts toward the HMM disruption observation
// (alert_count / severity_sum).
function countsAsDisruption(ns: AlertNamespace): boolean {
  switch (ns) {
    case 'planned':
      // Bounded resume window; the line is coming back on a schedule.
      return false;
    case 'realtime':
      return true;
    case 'other':
      // Deliberate: an unknown-namespace alert carries no bounded resume
      // window, so it is a live disruption we must not silence — counting it is
      // the safe read. But we cannot assume the real-time TTL semantics of
      // lmm:alert:*, so it does NOT set has_realtime_alert (the 'realtime'-only
      // check on has_realtime_alert above). That intentionally leaves the
      // schedule-recovery arm reachable for a route carrying both an 'other'
      // alert and a planned window; the resulting cross-arm collision (a
      // determinate overdue zero under a live-disruption read) is handled by
      // snapshot.ts's composition guard, which keys off the answer an arm
      // produced rather than assuming these predicates complement.
      return true;
  }
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
    if (alertNamespace(a.alert_id) !== 'planned') continue;
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
