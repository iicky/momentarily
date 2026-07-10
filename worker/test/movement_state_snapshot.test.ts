/**
 * Movement-determined current state in the snapshot: the published `condition`
 * is movement-primary — observed train movement drives it when a fresh reading
 * exists, otherwise it's an honest `unknown` (no HMM fallback), and a planned
 * not_scheduled always wins over movement.
 */

import { describe, expect, test } from 'vitest';

import type { RouteRoll } from '../src/alpha';
import { deriveRouteSnapshots } from '../src/derive';
import { TICK_SECONDS, buildSnapshot } from '../src/snapshot';

const NOW = 1_700_000_000;

function entity(opts: { id: string; alertType: string; route: string; periods?: Array<{ start: number; end?: number }> }): unknown {
  return {
    id: opts.id,
    alert: {
      active_period: opts.periods ?? [{ start: NOW - 3600 }],
      informed_entity: [
        {
          agency_id: 'MTASBWY',
          route_id: opts.route,
          'transit_realtime.mercury_entity_selector': { sort_order: `MTASBWY:${opts.route}:10` },
        },
      ],
      header_text: { translation: [{ text: `${opts.alertType} on ${opts.route}`, language: 'en' }] },
      'transit_realtime.mercury_alert': { alert_type: opts.alertType },
    },
  };
}
const payload = (...e: unknown[]): unknown => ({ entity: e });

function roll(state: 'normal' | 'disrupted' | 'suspended'): RouteRoll {
  const probs: [number, number, number] =
    state === 'normal' ? [0.95, 0.04, 0.01] : state === 'disrupted' ? [0.04, 0.95, 0.01] : [0.02, 0.03, 0.95];
  return {
    filter: { probabilities: probs, regime_entered_at: NOW, last_updated_at: NOW },
    published: { label: state, pending_state: state, pending_streak: 5, last_updated_at: NOW },
    alert_type_at_entry: null,
  };
}

describe('buildSnapshot: movement-determined condition', () => {
  test('movement overrides the HMM condition and records the source', () => {
    const snaps = deriveRouteSnapshots(payload(entity({ id: 'a', alertType: 'Delays', route: 'A' })), NOW);
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: snaps,
      rolls: { A: roll('normal') },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      movementStates: { observed_at: NOW - 300, states: { A: 'disrupted' } },
    });
    const a = snap.route_status.A!;
    expect(a.condition).toBe('disrupted');
    expect(a.condition_source).toBe('movement');
    // HMM still recorded under inference for the forecast surfaces.
    expect(a.inference?.condition).toBe('normal');
  });

  test('a route with no movement reading has unknown condition', () => {
    const snaps = deriveRouteSnapshots(payload(entity({ id: 'b', alertType: 'Delays', route: 'B' })), NOW);
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: snaps,
      rolls: { B: roll('normal') },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      movementStates: { observed_at: NOW - 300, states: {} }, // B absent
    });
    const b = snap.route_status.B!;
    expect(b.condition_source).toBe('unknown');
    expect(b.condition).toBe('unknown');
  });

  test('movement is ignored without a movementStates arg (back-compat)', () => {
    const snaps = deriveRouteSnapshots(payload(entity({ id: 'c', alertType: 'Delays', route: 'A' })), NOW);
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: snaps,
      rolls: { A: roll('disrupted') },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
    });
    expect(snap.route_status.A!.condition_source).toBe('unknown');
    expect(snap.route_status.A!.condition).toBe('unknown');
  });

  test('not_scheduled is never overridden by movement', () => {
    // A No Scheduled Service alert with a current gap drives is_not_scheduled.
    const snaps = deriveRouteSnapshots(
      payload(entity({ id: 'z', alertType: 'No Scheduled Service', route: 'Z', periods: [{ start: NOW - 3600, end: NOW + 1800 }] })),
      NOW,
    );
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: snaps,
      rolls: { Z: roll('normal') },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      movementStates: { observed_at: NOW - 300, states: { Z: 'suspended' } },
    });
    const z = snap.route_status.Z!;
    expect(z.condition).toBe('not_scheduled');
    expect(z.condition_source).toBe('schedule');
  });

  test('stale movement state is ignored (condition is unknown)', () => {
    const snaps = deriveRouteSnapshots(payload(entity({ id: 'a', alertType: 'Delays', route: 'A' })), NOW);
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: snaps,
      rolls: { A: roll('normal') },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      movementStates: { observed_at: NOW - 3600, states: { A: 'disrupted' } }, // 1h old
    });
    expect(snap.route_status.A!.condition_source).toBe('unknown');
    expect(snap.route_status.A!.condition).toBe('unknown');
  });

  test('lines_disrupted_count reflects the movement-overridden conditions', () => {
    const snaps = deriveRouteSnapshots(
      payload(entity({ id: 'a', alertType: 'Delays', route: 'A' }), entity({ id: 'b', alertType: 'Delays', route: 'B' })),
      NOW,
    );
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: snaps,
      rolls: { A: roll('normal'), B: roll('normal') },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      movementStates: { observed_at: NOW - 300, states: { A: 'suspended', B: 'normal' } },
    });
    expect(snap.route_status.A!.condition).toBe('suspended');
    expect(snap.system.lines_disrupted_count).toBe(1); // A counted, B normal
  });

  test('a route present only in movementStates.states is published with its movement condition', () => {
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: new Map(),
      rolls: {},
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      movementStates: { observed_at: NOW - 300, states: { Q: 'disrupted' } },
    });
    const q = snap.route_status.Q!;
    expect(q.condition).toBe('disrupted');
    expect(q.condition_source).toBe('movement');
    expect(q.inference).toBeNull();
  });

  test('a movement-only disrupted route (no HMM inference) is counted in lines_disrupted_count', () => {
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: new Map(),
      rolls: {},
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      movementStates: { observed_at: NOW - 300, states: { Q: 'disrupted' } },
    });
    const q = snap.route_status.Q!;
    expect(q.inference).toBeNull();
    expect(q.condition).toBe('disrupted');
    // The system rollup gates on the published `condition`, not on HMM
    // inference, so a movement-only route with no roll/snapshot must still
    // be counted — and score a flat 1 as the most degraded line.
    expect(snap.system.lines_disrupted_count).toBe(1);
    expect(snap.system.most_degraded_line).toBe('Q');
  });
});
