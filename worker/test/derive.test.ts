/**
 * Alert-id namespace partition (planned / realtime / other).
 *
 * derive.ts classifies every alert id into exactly one namespace and derives
 * both the disruption count (observation.alert_count) and has_realtime_alert
 * from that single partition. These tests pin the deliberate treatment of each
 * namespace — in particular the third 'other' category, whose "counted but not
 * real-time" membership must be a documented decision, not the accidental
 * residue of negating the planned-work check.
 */

import { describe, expect, test } from 'vitest';

import { alertsPayloadDegraded, classifyAlertsPayload, deriveRouteSnapshots } from '../src/derive';

const NOW = 1_700_000_000;

/** Build a single GTFS-RT alert entity for one route. */
function entity(opts: {
  id: string;
  alertType: string;
  route: string;
  sortOrder?: number;
  periods: Array<{ start: number; end?: number }>;
}): unknown {
  return {
    id: opts.id,
    alert: {
      active_period: opts.periods,
      informed_entity: [
        {
          agency_id: 'MTASBWY',
          route_id: opts.route,
          'transit_realtime.mercury_entity_selector': {
            sort_order: `MTASBWY:${opts.route}:${opts.sortOrder ?? 10}`,
          },
        },
      ],
      header_text: { translation: [{ text: `${opts.alertType} on ${opts.route}`, language: 'en' }] },
      'transit_realtime.mercury_alert': { alert_type: opts.alertType },
    },
  };
}

function payload(...entities: unknown[]): unknown {
  return { entity: entities };
}

describe('derive: alert-id namespace partition', () => {
  test('realtime id counts and sets the realtime flag', () => {
    const snaps = deriveRouteSnapshots(
      payload(
        entity({
          id: 'lmm:alert:1',
          alertType: 'Delays',
          route: 'A',
          sortOrder: 30,
          periods: [{ start: NOW - 600 }],
        }),
      ),
      NOW,
    );
    const a = snaps.get('A')!;
    expect(a.observation.alert_count).toBe(1);
    expect(a.has_realtime_alert).toBe(true);
  });

  test('planned id neither counts nor sets the realtime flag', () => {
    const snaps = deriveRouteSnapshots(
      payload(
        entity({
          id: 'lmm:planned_work:1',
          alertType: 'Planned - Part Suspended',
          route: 'B',
          sortOrder: 25,
          periods: [{ start: NOW - 3600, end: NOW + 3600 }],
        }),
      ),
      NOW,
    );
    const b = snaps.get('B')!;
    expect(b.observation.alert_count).toBe(0);
    expect(b.has_realtime_alert).toBe(false);
  });

  test("third-namespace id counts as a disruption but is deliberately not real-time", () => {
    // The whole point of the partition: an id in neither MTA namespace is a
    // live disruption we must not silence (it counts), yet we cannot assume the
    // real-time TTL semantics of lmm:alert:*, so it must NOT set the realtime
    // flag. A regression that folds 'other' into either sibling namespace — by
    // reverting to !isPlannedWorkId or by broadening the realtime check — flips
    // exactly one of these expectations.
    const snaps = deriveRouteSnapshots(
      payload(
        entity({
          id: 'lmm:situation:99001',
          alertType: 'Delays',
          route: 'C',
          sortOrder: 30,
          periods: [{ start: NOW - 600 }],
        }),
      ),
      NOW,
    );
    const c = snaps.get('C')!;
    expect(c.observation.alert_count).toBe(1);
    expect(c.has_realtime_alert).toBe(false);
  });

  test('has_realtime_alert reflects only the realtime namespace when kinds mix', () => {
    // planned + other + realtime together: alert_count counts realtime and
    // other (2), while has_realtime_alert stays keyed to the realtime member.
    const snaps = deriveRouteSnapshots(
      payload(
        entity({
          id: 'lmm:planned_work:2',
          alertType: 'Planned - Reroute',
          route: 'D',
          sortOrder: 25,
          periods: [{ start: NOW - 3600, end: NOW + 3600 }],
        }),
        entity({
          id: 'lmm:situation:2',
          alertType: 'Delays',
          route: 'D',
          sortOrder: 20,
          periods: [{ start: NOW - 600 }],
        }),
        entity({
          id: 'lmm:alert:2',
          alertType: 'Severe Delays',
          route: 'D',
          sortOrder: 40,
          periods: [{ start: NOW - 600 }],
        }),
      ),
      NOW,
    );
    const d = snaps.get('D')!;
    expect(d.observation.alert_count).toBe(2);
    expect(d.has_realtime_alert).toBe(true);
  });
});

/** A structurally valid MTA alert that names a station, not a subway route —
 * an elevator/escalator notice. Recognizable (id + header_text + mercury
 * alert_type) but out of scope for routes: informed_entity carries a stop_id. */
function stationNotice(id: string): unknown {
  return {
    id,
    alert: {
      active_period: [{ start: NOW - 600 }],
      informed_entity: [{ agency_id: 'MTASBWY', stop_id: 'A24' }],
      header_text: { translation: [{ text: 'Elevator out at station', language: 'en' }] },
      'transit_realtime.mercury_alert': { alert_type: 'Elevator' },
    },
  };
}

describe('derive: alerts-schema drift floor (classifyAlertsPayload + alertsPayloadDegraded)', () => {
  // The floor (index.ts step 4a) trips when the payload carries entities and
  // either nothing in it is a recognizable MTA alert, or the route-bearing
  // alerts it carries are all ones parseAlertEntity cannot consume.
  const degraded = (p: unknown) => alertsPayloadDegraded(classifyAlertsPayload(p));

  test('a normal route alert is recognized in scope and not degraded', () => {
    const p = payload(
      entity({ id: 'lmm:alert:1', alertType: 'Delays', route: 'A', periods: [{ start: NOW - 600 }] }),
    );
    const health = classifyAlertsPayload(p);
    expect(health).toEqual({
      entities: 1,
      recognizedInScope: 1,
      inScopeUnparseable: 0,
      recognizedOutOfScope: 0,
      unrecognizable: 0,
    });
    expect(degraded(p)).toBe(false);
  });

  test('structural drift: entities present, but none are recognizable MTA alerts', () => {
    // An MTA alerts-schema change that still ships an entity array whose
    // members carry no alert object with a header_text or the mercury
    // alert_type at all.
    const p = {
      entity: [
        { id: 'x1', alert: { some_new_shape: { renamed: 'Delays' } } },
        { id: 'x2', alert: { informed_entity: [{ route_id: 'A' }] } },
      ],
    };
    expect(classifyAlertsPayload(p).unrecognizable).toBe(2);
    expect(degraded(p)).toBe(true);
  });

  test('header_text with no selectors and no alert_type is drift', () => {
    // An entity stripped to a bare header_text — no mercury alert_type, no
    // informed_entity selectors — is not a readable MTA alert and must not
    // bless the payload as healthy.
    const p = { entity: [{ id: 'x', alert: { header_text: { translation: [{ text: 'Delays' }] } } }] };
    expect(classifyAlertsPayload(p).unrecognizable).toBe(1);
    expect(degraded(p)).toBe(true);
  });

  test('mercury extension dropped from route alerts is drift, even beside station notices', () => {
    // The most plausible real drift: the vendor mercury_alert extension (and
    // its alert_type) disappears while standard GTFS-RT fields survive. Such
    // entities still name a route and read as recognizable, but
    // parseAlertEntity rejects every one, so the pipeline would silently see
    // no route alerts. Station notices beside them keep their alert_type and
    // must NOT mask the drift.
    const stripped = (id: string, route: string) => ({
      id,
      alert: {
        header_text: { translation: [{ text: 'Delays', language: 'en' }] },
        informed_entity: [{ route_id: route }],
        active_period: [{ start: NOW - 600 }],
      },
    });
    const p = { entity: [stripped('lmm:alert:1', 'A'), stripped('lmm:alert:2', 'L'), stationNotice('lmm:alert:elev1')] };
    const health = classifyAlertsPayload(p);
    expect(health.recognizedInScope).toBe(0);
    expect(health.inScopeUnparseable).toBe(2);
    expect(health.recognizedOutOfScope).toBe(1);
    expect(degraded(p)).toBe(true);
    expect(deriveRouteSnapshots(p, NOW).size).toBe(0);
  });

  test('a feed of only station notices is out of scope, NOT drift', () => {
    // Elevator/escalator notices name a stop, not a subway route. They are
    // recognizable MTA alerts, so the floor does not trip — even though they
    // derive no subway route snapshot.
    const p = payload(stationNotice('lmm:alert:elev1'), stationNotice('lmm:alert:elev2'));
    const health = classifyAlertsPayload(p);
    expect(health.recognizedInScope).toBe(0);
    expect(health.recognizedOutOfScope).toBe(2);
    expect(degraded(p)).toBe(false);
    expect(deriveRouteSnapshots(p, NOW).size).toBe(0);
  });

  test('a valid feed of only inactive route alerts is in scope, NOT drift', () => {
    // A well-formed route alert whose active_period is expired (or future —
    // planned work published ahead). deriveRouteSnapshots drops it by active
    // window, but it parses, so the floor does not trip.
    const p = payload(
      entity({ id: 'lmm:alert:1', alertType: 'Delays', route: 'A', periods: [{ start: NOW - 7200, end: NOW - 3600 }] }),
    );
    expect(classifyAlertsPayload(p).recognizedInScope).toBe(1);
    expect(degraded(p)).toBe(false);
    expect(deriveRouteSnapshots(p, NOW).size).toBe(0);
  });

  test('mixed parseable route alert and garbage: NOT drift', () => {
    // One real route alert beside unrecognizable garbage: the pipeline still
    // consumes a route alert, so the payload is not degraded.
    const p = {
      entity: [
        entity({ id: 'lmm:alert:1', alertType: 'Delays', route: 'A', periods: [{ start: NOW - 600 }] }),
        { id: 'garbage', alert: { some_new_shape: {} } },
      ],
    };
    const health = classifyAlertsPayload(p);
    expect(health.recognizedInScope).toBe(1);
    expect(health.unrecognizable).toBe(1);
    expect(degraded(p)).toBe(false);
  });

  test('a genuinely empty feed is a quiet system, NOT drift', () => {
    expect(classifyAlertsPayload(payload()).entities).toBe(0);
    expect(degraded(payload())).toBe(false);
  });
});
