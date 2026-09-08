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

import { classifyAlertsPayload, deriveRouteSnapshots } from '../src/derive';

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

describe('derive: classifyAlertsPayload (alerts-schema drift floor)', () => {
  // The floor (index.ts step 4a) trips on: entities > 0 AND
  // recognizedInScope + recognizedOutOfScope === 0.
  test('a normal route alert is recognized in scope', () => {
    const health = classifyAlertsPayload(
      payload(
        entity({
          id: 'lmm:alert:1',
          alertType: 'Delays',
          route: 'A',
          periods: [{ start: NOW - 600 }],
        }),
      ),
    );
    expect(health.entities).toBe(1);
    expect(health.recognizedInScope).toBe(1);
    expect(health.recognizedOutOfScope).toBe(0);
    expect(health.unrecognizable).toBe(0);
  });

  test('structural drift: entities present, but none are recognizable MTA alerts', () => {
    // The failure the floor exists to catch: an MTA alerts-schema change that
    // still ships an entity array whose members carry no alert object with a
    // header_text or the mercury alert_type at all.
    const health = classifyAlertsPayload({
      entity: [
        { id: 'x1', alert: { some_new_shape: { renamed: 'Delays' } } },
        { id: 'x2', alert: { informed_entity: [{ route_id: 'A' }] } },
      ],
    });
    expect(health.entities).toBe(2);
    expect(health.recognizedInScope + health.recognizedOutOfScope).toBe(0);
    expect(health.unrecognizable).toBe(2);
  });

  test('header_text with no selectors and no alert_type is NOT recognizable (drift)', () => {
    // Regression: an entity stripped to a bare header_text — no mercury
    // alert_type, no informed_entity selectors — is not a readable MTA alert.
    // It must count as drift, never bless the payload as healthy and suppress
    // the degraded flag (a header_text can survive a drift that renamed/removed
    // the route selectors and the alert body).
    const health = classifyAlertsPayload({
      entity: [{ id: 'x', alert: { header_text: { translation: [{ text: 'Delays' }] } } }],
    });
    expect(health.entities).toBe(1);
    expect(health.recognizedInScope + health.recognizedOutOfScope).toBe(0);
    expect(health.unrecognizable).toBe(1);
  });

  test('a feed of only station notices is recognized out of scope, NOT drift', () => {
    // A valid feed carrying only out-of-scope alerts: elevator/escalator notices
    // name a stop, not a subway route. They are recognizable MTA alerts, so the
    // floor does not trip — even though they derive no subway route snapshot.
    const p = payload(stationNotice('lmm:alert:elev1'), stationNotice('lmm:alert:elev2'));
    const health = classifyAlertsPayload(p);
    expect(health.entities).toBe(2);
    expect(health.recognizedInScope).toBe(0);
    expect(health.recognizedOutOfScope).toBe(2);
    expect(deriveRouteSnapshots(p, NOW).size).toBe(0);
  });

  test('a valid feed of only inactive route alerts is recognized in scope, NOT drift', () => {
    // A well-formed route alert whose active_period is expired (or future —
    // planned work published ahead). deriveRouteSnapshots drops it by active
    // window, but it is recognizable in scope, so the floor does not trip.
    const p = payload(
      entity({
        id: 'lmm:alert:1',
        alertType: 'Delays',
        route: 'A',
        periods: [{ start: NOW - 7200, end: NOW - 3600 }], // ended an hour ago
      }),
    );
    const health = classifyAlertsPayload(p);
    expect(health.entities).toBe(1);
    expect(health.recognizedInScope).toBe(1);
    expect(deriveRouteSnapshots(p, NOW).size).toBe(0);
  });

  test('mixed recognizable and garbage: still NOT drift', () => {
    // One real route alert beside unrecognizable garbage: the payload plainly
    // still carries MTA alerts (recognizable > 0), so it is not degraded.
    const health = classifyAlertsPayload({
      entity: [
        entity({ id: 'lmm:alert:1', alertType: 'Delays', route: 'A', periods: [{ start: NOW - 600 }] }),
        { id: 'garbage', alert: { some_new_shape: {} } },
      ],
    });
    expect(health.entities).toBe(2);
    expect(health.recognizedInScope).toBe(1);
    expect(health.unrecognizable).toBe(1);
  });

  test('a genuinely empty feed carries no entities', () => {
    const health = classifyAlertsPayload(payload());
    expect(health.entities).toBe(0);
    expect(health.recognizedInScope + health.recognizedOutOfScope).toBe(0);
  });
});
