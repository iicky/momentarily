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

function roll(
  state: 'normal' | 'disrupted' | 'suspended',
  onset?: { publishedCondition?: string; conditionEnteredAt?: number | null },
): RouteRoll {
  const probs: [number, number, number] =
    state === 'normal' ? [0.95, 0.04, 0.01] : state === 'disrupted' ? [0.04, 0.95, 0.01] : [0.02, 0.03, 0.95];
  return {
    filter: { probabilities: probs, regime_entered_at: NOW, last_updated_at: NOW },
    published: { label: state, pending_state: state, pending_streak: 5, last_updated_at: NOW },
    alert_type_at_entry: null,
    ...(onset?.publishedCondition !== undefined ? { published_condition: onset.publishedCondition } : {}),
    ...(onset?.conditionEnteredAt !== undefined ? { condition_entered_at: onset.conditionEnteredAt } : {}),
  };
}

/** Settled regimes from a plain condition map — these cases exercise the
 * snapshot's read of the clock, not the debounce that produced it. */
function settled(states: Record<string, string>): Record<string, { state: string; entered_at: number }> {
  return Object.fromEntries(
    Object.entries(states).map(([route, state]) => [route, { state, entered_at: NOW - 3600 }]),
  );
}

describe('buildSnapshot: alert-graded published condition', () => {
  test('a severe alert publishes disrupted from the alert feed, clock off the roll', () => {
    const snaps = deriveRouteSnapshots(payload(entity({ id: 'a', alertType: 'Severe Delays', route: 'A' })), NOW);
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: snaps,
      // index.ts back-dates the onset onto the roll; the snapshot publishes it.
      rolls: { A: roll('normal', { publishedCondition: 'disrupted', conditionEnteredAt: NOW - 3600 }) },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      // A movement 'normal' reading is present and deliberately IGNORED for the
      // condition — alerts decide it now.
      movementStates: { observed_at: NOW - 300, regimes: settled({ A: 'normal' }) },
      vehicleFreshFeeds: [],
      vehicleExpectedFeeds: [],
    });
    const a = snap.route_status.A!;
    expect(a.condition).toBe('disrupted');
    expect(a.condition_source).toBe('alerts');
    // The badge clock is the alert regime's onset carried on the roll, NOT the
    // HMM argmax clock (inference.regime_entered_at === NOW here).
    expect(a.condition_entered_at).toBe(NOW - 3600);
    expect(a.inference?.regime_entered_at).toBe(NOW);
    // The HMM read is still recorded under inference for the grading surfaces.
    expect(a.inference?.condition).toBe('normal');
  });

  test('a sub-floor alert (ordinary Delays) grades normal', () => {
    const snaps = deriveRouteSnapshots(payload(entity({ id: 'a', alertType: 'Delays', route: 'A' })), NOW);
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: snaps,
      rolls: { A: roll('disrupted', { publishedCondition: 'normal', conditionEnteredAt: NOW - 1800 }) },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      vehicleFreshFeeds: [],
      vehicleExpectedFeeds: [],
    });
    const a = snap.route_status.A!;
    // Tier-1 alerts are below the canonical severe-only floor → normal.
    expect(a.condition).toBe('normal');
    expect(a.condition_source).toBe('alerts');
    expect(a.condition_entered_at).toBe(NOW - 1800);
  });

  test('a suspension alert grades suspended', () => {
    const snaps = deriveRouteSnapshots(payload(entity({ id: 'a', alertType: 'Suspended', route: 'A' })), NOW);
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: snaps,
      rolls: { A: roll('normal', { publishedCondition: 'suspended', conditionEnteredAt: NOW - 600 }) },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      vehicleFreshFeeds: [],
      vehicleExpectedFeeds: [],
    });
    expect(snap.route_status.A!.condition).toBe('suspended');
    expect(snap.route_status.A!.condition_source).toBe('alerts');
  });

  test('no active alert grades normal even with a movement-disrupted reading', () => {
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: deriveRouteSnapshots(payload(), NOW),
      rolls: { A: roll('normal', { publishedCondition: 'normal', conditionEnteredAt: NOW - 7200 }) },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      // Movement says disrupted; the alert feed says nothing. Alerts win.
      movementStates: { observed_at: NOW - 300, regimes: settled({ A: 'disrupted' }) },
      vehicleFreshFeeds: [],
      vehicleExpectedFeeds: [],
    });
    const a = snap.route_status.A!;
    expect(a.condition).toBe('normal');
    expect(a.condition_source).toBe('alerts');
  });

  test('a stale/unparsed alert feed abstains to unknown', () => {
    const snaps = deriveRouteSnapshots(payload(entity({ id: 'a', alertType: 'Severe Delays', route: 'A' })), NOW);
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW - 3600,
      routeSnapshots: snaps,
      rolls: { A: roll('normal', { publishedCondition: 'unknown', conditionEnteredAt: null }) },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      // The feed could not be used this tick — the only path to unknown now.
      alertsFeedUsable: false,
      vehicleFreshFeeds: [],
      vehicleExpectedFeeds: [],
    });
    const a = snap.route_status.A!;
    expect(a.condition).toBe('unknown');
    expect(a.condition_source).toBe('unknown');
    expect(a.condition_entered_at).toBeNull();
  });

  test('not_scheduled wins as schedule with no onset clock', () => {
    const snaps = deriveRouteSnapshots(
      payload(entity({ id: 'z', alertType: 'No Scheduled Service', route: 'Z', periods: [{ start: NOW - 3600, end: NOW + 1800 }] })),
      NOW,
    );
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: snaps,
      // Even a non-null onset on the roll is withheld on the schedule arm.
      rolls: { Z: roll('normal', { publishedCondition: 'not_scheduled', conditionEnteredAt: NOW - 900 }) },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      vehicleFreshFeeds: [],
      vehicleExpectedFeeds: [],
    });
    const z = snap.route_status.Z!;
    expect(z.condition).toBe('not_scheduled');
    expect(z.condition_source).toBe('schedule');
    expect(z.condition_entered_at).toBeNull();
  });

  test('the alerts arm withholds the clock when the roll carries no onset', () => {
    const snaps = deriveRouteSnapshots(payload(entity({ id: 'a', alertType: 'Severe Delays', route: 'A' })), NOW);
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: snaps,
      rolls: { A: roll('normal') }, // no condition_entered_at
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      vehicleFreshFeeds: [],
      vehicleExpectedFeeds: [],
    });
    const a = snap.route_status.A!;
    expect(a.condition).toBe('disrupted');
    expect(a.condition_source).toBe('alerts');
    expect(a.condition_entered_at).toBeNull();
  });

  test('lines_disrupted_count reflects the alert-graded conditions', () => {
    const snaps = deriveRouteSnapshots(
      payload(entity({ id: 'a', alertType: 'Suspended', route: 'A' }), entity({ id: 'b', alertType: 'Delays', route: 'B' })),
      NOW,
    );
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: snaps,
      rolls: { A: roll('normal'), B: roll('normal') },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      vehicleFreshFeeds: [],
      vehicleExpectedFeeds: [],
    });
    expect(snap.route_status.A!.condition).toBe('suspended');
    expect(snap.route_status.B!.condition).toBe('normal'); // ordinary Delays is sub-floor
    expect(snap.system.lines_disrupted_count).toBe(1); // A counted, B normal
  });

  test('a route present only in movementStates.regimes is not disrupted off movement', () => {
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: new Map(),
      rolls: {},
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      movementStates: { observed_at: NOW - 300, regimes: settled({ Q: 'disrupted' }) },
      vehicleFreshFeeds: [],
      vehicleExpectedFeeds: [],
    });
    const q = snap.route_status.Q!;
    // No alert and no roll → normal off the alert feed; movement no longer
    // asserts the condition, so nothing is counted disrupted.
    expect(q.condition).toBe('normal');
    expect(q.condition_source).toBe('alerts');
    expect(q.inference).toBeNull();
    expect(snap.system.lines_disrupted_count).toBe(0);
  });
});

describe('buildSnapshot: service_condition (supply axis)', () => {
  test('publishes the service regime state, independent of the movement condition', () => {
    const snaps = deriveRouteSnapshots(payload(entity({ id: 'a', alertType: 'Delays', route: 'A' })), NOW);
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: snaps,
      rolls: { A: roll('normal') },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      // Trains moving fine (movement normal) while trips are pulled (service
      // degraded): the two axes carry different states, as they must.
      movementStates: {
        observed_at: NOW - 300,
        regimes: settled({ A: 'normal' }),
        service_regimes: settled({ A: 'degraded' }),
        service_ratios: { A: 0.3 },
      },
      vehicleFreshFeeds: [],
      vehicleExpectedFeeds: [],
    });
    const a = snap.route_status.A!;
    expect(a.condition).toBe('normal');
    expect(a.service_condition).toBe('degraded');
    expect(a.service_ratio).toBe(0.3);
  });

  test('a route absent from service_regimes reads unknown', () => {
    const snaps = deriveRouteSnapshots(payload(entity({ id: 'b', alertType: 'Delays', route: 'B' })), NOW);
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: snaps,
      rolls: { B: roll('normal') },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      movementStates: { observed_at: NOW - 300, regimes: settled({ B: 'normal' }), service_regimes: settled({}) },
      vehicleFreshFeeds: [],
      vehicleExpectedFeeds: [],
    });
    expect(snap.route_status.B!.service_condition).toBe('unknown');
  });

  test('a doc without service_regimes at all reads unknown (back-compat)', () => {
    const snaps = deriveRouteSnapshots(payload(entity({ id: 'c', alertType: 'Delays', route: 'A' })), NOW);
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: snaps,
      rolls: { A: roll('normal') },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      // Old-shaped doc: no service_regimes, no service_ratios, no
      // service_quantile_ratios — exactly what a sidecar with no quantiles
      // produces. The whole axis, including the two new fields, must behave
      // exactly as before this change: unknown/null, never a thrown parse or a
      // fabricated ratio.
      movementStates: { observed_at: NOW - 300, regimes: settled({ A: 'normal' }) },
      vehicleFreshFeeds: [],
      vehicleExpectedFeeds: [],
    });
    expect(snap.route_status.A!.service_condition).toBe('unknown');
    expect(snap.route_status.A!.service_ratio).toBeNull();
    expect(snap.route_status.A!.service_low_ratio).toBeNull();
    expect(snap.route_status.A!.service_high_ratio).toBeNull();
  });

  test('publishes service_low_ratio/service_high_ratio from service_quantile_ratios', () => {
    const snaps = deriveRouteSnapshots(payload(entity({ id: 'd', alertType: 'Delays', route: 'A' })), NOW);
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: snaps,
      rolls: { A: roll('normal') },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      movementStates: {
        observed_at: NOW - 300,
        regimes: settled({ A: 'normal' }),
        service_regimes: settled({ A: 'normal' }),
        service_ratios: { A: 0.9 },
        service_quantile_ratios: { A: { low: 0.8, high: 1.3 } },
      },
      vehicleFreshFeeds: [],
      vehicleExpectedFeeds: [],
    });
    const a = snap.route_status.A!;
    expect(a.service_low_ratio).toBe(0.8);
    expect(a.service_high_ratio).toBe(1.3);
  });

  test('a route with service_ratio but no quantile cell reads null for both new fields', () => {
    const snaps = deriveRouteSnapshots(payload(entity({ id: 'e', alertType: 'Delays', route: 'A' })), NOW);
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: snaps,
      rolls: { A: roll('normal') },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      movementStates: {
        observed_at: NOW - 300,
        regimes: settled({ A: 'normal' }),
        service_regimes: settled({ A: 'normal' }),
        service_ratios: { A: 0.9 },
        // No quantile cell for A, even though service_quantile_ratios is present.
        service_quantile_ratios: {},
      },
      vehicleFreshFeeds: [],
      vehicleExpectedFeeds: [],
    });
    const a = snap.route_status.A!;
    expect(a.service_ratio).toBe(0.9);
    expect(a.service_low_ratio).toBeNull();
    expect(a.service_high_ratio).toBeNull();
  });
});
