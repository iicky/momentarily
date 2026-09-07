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

import { deriveRouteSnapshots } from '../src/derive';

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
