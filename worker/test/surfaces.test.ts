/**
 * Top-level detail surfaces: alerts, routes, equipment. These used to ship
 * empty while the rollups counted them — a route_status alert id resolved to
 * nothing, accessibility reported outages with no equipment array. Each builder
 * pulls from the data already fetched per tick; the consistency guard catches a
 * regression back to the empty-array state.
 */

import Ajv2020 from 'ajv/dist/2020';
import { describe, expect, test } from 'vitest';

import schema from '../../schema/snapshot.schema.json';
import { buildAlertList, buildRoutes } from '../src/derive';
import type { ActiveOutage, EquipmentCatalogEntry } from '../src/ene';
import { TICK_SECONDS, buildSnapshot, snapshotConsistencyWarnings } from '../src/snapshot';
import { buildEquipmentList } from '../src/stations';

const ajv = new Ajv2020({ allErrors: true, strict: false });
const validate = ajv.compile(schema);

const NOW = 1_700_000_000;

// A single alert entity. In the real Mercury feed one alert (one id) carries
// every route it affects as separate informed_entity rows, so `routes` lets a
// fixture model a multi-route alert without splitting the id across entities.
function entity(opts: {
  id: string;
  alertType: string;
  route: string;
  routes?: Array<{ route: string; directionId?: number }>;
  sortOrder?: number;
  directionId?: number;
  periods?: Array<{ start?: number; end?: number }>;
}): unknown {
  const rows = opts.routes ?? [{ route: opts.route, directionId: opts.directionId }];
  return {
    id: opts.id,
    alert: {
      active_period: opts.periods ?? [{ start: NOW - 100 }],
      informed_entity: rows.map((r) => ({
        agency_id: 'MTASBWY',
        route_id: r.route,
        direction_id: r.directionId,
        'transit_realtime.mercury_entity_selector': {
          sort_order: `MTASBWY:${r.route}:${opts.sortOrder ?? 10}`,
        },
      })),
      header_text: { translation: [{ text: `${opts.alertType} on ${opts.route}`, language: 'en' }] },
      'transit_realtime.mercury_alert': { alert_type: opts.alertType },
    },
  };
}

function cat(
  equipment_id: string,
  station_complex_id: string,
  type: 'elevator' | 'escalator',
  ada = false,
): EquipmentCatalogEntry {
  return {
    equipment_id,
    station: station_complex_id,
    station_complex_id,
    gtfs_stop_id: null,
    type,
    ada_pathway: ada,
    is_active: true,
  };
}

function outage(
  equipment_id: string,
  type: 'elevator' | 'escalator',
  opts: { since?: number; est?: number | null } = {},
): ActiveOutage {
  return {
    equipment_id,
    station: 'Display Name',
    type,
    ada_pathway: false,
    outage: { reason: 'Repair', since: opts.since ?? NOW - 1000, est_return: opts.est ?? NOW + 5000 },
  };
}

describe('buildAlertList', () => {
  test('emits full alert objects, one per id, with every informed route', () => {
    const payload = {
      entity: [
        entity({
          id: 'lmm:alert:1',
          alertType: 'Delays',
          route: 'A',
          routes: [
            { route: 'A', directionId: 0 },
            { route: 'C', directionId: 1 },
          ],
        }),
        entity({ id: 'lmm:planned_work:2', alertType: 'Planned - Reroute', route: 'B' }),
      ],
    };
    const alerts = buildAlertList(payload, NOW);
    expect(alerts.map((a) => a.id).sort()).toEqual(['lmm:alert:1', 'lmm:planned_work:2']);
    const a1 = alerts.find((a) => a.id === 'lmm:alert:1')!;
    expect(a1.alert_type).toBe('Delays');
    expect(a1.source).toBe('subway');
    expect(a1.header_text?.translation[0]?.text).toBe('Delays on A');
    expect(a1.informed_entities).toEqual([
      { route_id: 'A', direction_id: 0 },
      { route_id: 'C', direction_id: 1 },
    ]);
  });

  test('the same id appearing twice collapses to one alert', () => {
    const payload = {
      entity: [
        entity({ id: 'lmm:alert:dup', alertType: 'Delays', route: 'A' }),
        entity({ id: 'lmm:alert:dup', alertType: 'Delays', route: 'A' }),
      ],
    };
    expect(buildAlertList(payload, NOW)).toHaveLength(1);
  });

  test('drops alerts outside their active window', () => {
    const payload = {
      entity: [entity({ id: 'lmm:alert:old', alertType: 'Delays', route: 'A', periods: [{ end: NOW - 10 }] })],
    };
    expect(buildAlertList(payload, NOW)).toEqual([]);
  });

  test('a route_status alert id resolves to an emitted alert', () => {
    const payload = { entity: [entity({ id: 'lmm:alert:9', alertType: 'Delays', route: 'F' })] };
    const ids = new Set(buildAlertList(payload, NOW).map((a) => a.id));
    expect(ids.has('lmm:alert:9')).toBe(true);
  });
});

describe('buildRoutes', () => {
  test('every canonical route gets static metadata', () => {
    const routes = buildRoutes();
    expect(routes['1']).toMatchObject({ id: '1', mode: 'subway', short_name: '1', agency: 'nyct_subway' });
    expect(routes['SI']?.short_name).toBe('SIR');
    expect(Object.keys(routes).length).toBeGreaterThan(20);
  });
});

describe('buildEquipmentList', () => {
  test('emits only units with an active outage, enriched from the catalog', () => {
    const catalog = [cat('EL1', 'S1', 'elevator', true), cat('EL2', 'S1', 'elevator'), cat('ES1', 'S1', 'escalator')];
    const outages = [outage('EL1', 'elevator')];
    const eq = buildEquipmentList(catalog, outages, NOW);
    expect(eq).toHaveLength(1);
    expect(eq[0]).toMatchObject({
      equipment_id: 'EL1',
      type: 'elevator',
      station_complex_id: 'S1',
      ada_pathway: true,
    });
  });

  test('skips outages whose est_return has already passed', () => {
    const outages = [outage('EL1', 'elevator', { est: NOW - 1 })];
    expect(buildEquipmentList([cat('EL1', 'S1', 'elevator')], outages, NOW)).toEqual([]);
  });

  test('an outage with no catalog entry still publishes', () => {
    const eq = buildEquipmentList([], [outage('ELX', 'escalator')], NOW);
    expect(eq).toHaveLength(1);
    expect(eq[0]?.equipment_id).toBe('ELX');
  });
});

describe('snapshotConsistencyWarnings', () => {
  test('flags alert_count > 0 with empty alerts[]', () => {
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: new Map(),
      rolls: {},
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
    });
    snap.system.by_mode.subway = { routes_with_alerts: ['A'], alert_count: 3, severity_max: 5 };
    expect(snapshotConsistencyWarnings(snap)).toEqual([expect.stringContaining('alert_count=3')]);
  });

  test('a populated snapshot raises no warnings and validates', () => {
    const payload = { entity: [entity({ id: 'lmm:alert:1', alertType: 'Delays', route: 'A' })] };
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: new Map(),
      rolls: {},
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      alerts: buildAlertList(payload, NOW),
      equipment: buildEquipmentList([cat('EL1', 'S1', 'elevator')], [outage('EL1', 'elevator')], NOW),
    });
    expect(snap.alerts.length).toBe(1);
    expect(snap.equipment.length).toBe(1);
    expect(Object.keys(snap.routes).length).toBeGreaterThan(20);
    expect(snapshotConsistencyWarnings(snap)).toEqual([]);
    expect(validate(snap), JSON.stringify(validate.errors, null, 2)).toBe(true);
  });
});
