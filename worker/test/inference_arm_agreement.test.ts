/**
 * Cross-arm composition of the published `inference` block.
 *
 * `is_disrupted` is the ALERT arm's read (snapshot.ts's resolveCondition);
 * `recovery_minutes`/`_low`/`_high`/`recovery_indeterminate` come from
 * whichever arm `recovery_source` names. Each arm is self-consistent. Composed
 * where the arms disagree, they were not: observed live on J
 * (generated_at=1788394231) as is_disrupted=true with p_disrupted=0.999992
 * published beside recovery_minutes=0, interval [0,0] and
 * recovery_indeterminate=false, because the movement arm's `normal` branch
 * honestly returns 0/0/0-determinate meaning "nothing to recover from".
 *
 * The fix withholds the number the way the ceiling gate already does for the
 * mirror-image case. These tests pin the disagreement, and pin that the two
 * rows from the same live snapshot that were already correct — H (no movement
 * read, alert-arm estimate) and Z (not_scheduled, ceiling convention) — did
 * not move.
 */

import { describe, expect, test } from 'vitest';

import type { RouteRoll } from '../src/alpha';
import { deriveRouteSnapshots } from '../src/derive';
import type { TrainedParams } from '../src/params';
import { parseTrainedParams } from '../src/params';
import { TICK_SECONDS, buildSnapshot } from '../src/snapshot';

const NOW = 1_700_000_000;
const MIN = 60;
const HOUR = 3600;
const MAX_RECOVERY_MINUTES = 1440;

// Same heavy-tailed movement dwell fixtures as movement_recovery.test.ts.
const DISRUPTED_CURVE = Array.from({ length: 21 }, (_, i) => Math.round(120 * 1.35 ** i));
const NORMAL_CURVE = Array.from({ length: 21 }, (_, i) => Math.round(600 * 1.55 ** i));

function routeParams(): Record<string, unknown> {
  return {
    transition: [
      [0.95, 0.04, 0.01],
      [0.08, 0.9, 0.02],
      [0.02, 0.1, 0.88],
    ],
    initial: [0.9, 0.08, 0.02],
    emissions: {
      poisson_lambda: [0.3, 4.0, 12.0],
      gamma_alpha: [1.0, 3.0, 6.0],
      gamma_beta: [2.0, 0.4, 0.2],
      bernoulli_p: [0.001, 0.05, 0.95],
      bernoulli_p_delays: [0.02, 0.6, 0.35],
      bernoulli_p_service_change: [0.02, 0.6, 0.4],
      bernoulli_p_planned: [0.05, 0.6, 0.35],
    },
  };
}

function trained(routeIds: string[]): TrainedParams | null {
  const routes: Record<string, unknown> = {};
  const dwell_movement: Record<string, unknown> = {};
  for (const id of routeIds) {
    routes[id] = routeParams();
    dwell_movement[id] = {
      normal: { n: 40, q25_sec: 5368, median_sec: 48025, q75_sec: 429662, curve_sec: NORMAL_CURVE },
      disrupted: { n: 30, q25_sec: 538, median_sec: 2413, q75_sec: 10819, curve_sec: DISRUPTED_CURVE },
    };
  }
  return parseTrainedParams({
    schema_version: '1',
    trained_at: NOW,
    routes,
    dwell_movement,
  });
}

/** A GTFS-RT alert entity for one route. */
function entity(opts: {
  id: string;
  alertType: string;
  route: string;
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
          'transit_realtime.mercury_entity_selector': { sort_order: `MTASBWY:${opts.route}:10` },
        },
      ],
      header_text: { translation: [{ text: `${opts.alertType} on ${opts.route}`, language: 'en' }] },
      'transit_realtime.mercury_alert': { alert_type: opts.alertType },
    },
  };
}

/** A roll whose filter sits in `state` with the confidence the live J row had. */
function roll(state: 'normal' | 'disrupted'): RouteRoll {
  const probs: [number, number, number] =
    state === 'normal' ? [0.95, 0.04, 0.01] : [0.000004, 0.999992, 0.000004];
  return {
    filter: { probabilities: probs, regime_entered_at: NOW - 3 * HOUR, last_updated_at: NOW },
    published: { label: state, pending_state: state, pending_streak: 5, last_updated_at: NOW },
    alert_type_at_entry: null,
  };
}

function build(opts: {
  routeId: string;
  roll: RouteRoll;
  alerts?: unknown[];
  movement?: { state: string; entered_at: number } | null;
}) {
  return buildSnapshot({
    generatedAt: NOW,
    alertsFreshness: NOW,
    routeSnapshots: deriveRouteSnapshots({ entity: opts.alerts ?? [] }, NOW),
    rolls: { [opts.routeId]: opts.roll },
    trainedParams: trained([opts.routeId]),
    tickSeconds: TICK_SECONDS,
    movementStates:
      opts.movement == null
        ? null
        : { observed_at: NOW - 300, regimes: { [opts.routeId]: opts.movement } },
  });
}

/** The live J row: real-time Delays alert (so the alert arm is not force-normal)
 * plus a confident `disrupted` filter, while movement reads `normal`. */
function disagreeingArms(routeId: string) {
  return build({
    routeId,
    roll: roll('disrupted'),
    alerts: [
      entity({
        id: 'lmm:alert:535417',
        alertType: 'Delays',
        route: routeId,
        periods: [{ start: NOW - 2 * HOUR }],
      }),
    ],
    movement: { state: 'normal', entered_at: NOW - HOUR },
  });
}

describe('inference: arms that disagree must not compose into a confident zero', () => {
  test('alert arm disrupted + movement arm normal withholds the recovery number (the J row)', () => {
    const snap = disagreeingArms('J');
    const status = snap.route_status.J!;
    const inf = status.inference!;

    // Preconditions: the published badge is movement-primary and reads normal,
    // while the alert shadow reads a confident disruption. This is the
    // disagreement, reproduced.
    expect(status.condition).toBe('normal');
    expect(status.condition_source).toBe('movement');
    expect(inf.condition).toBe('disrupted');
    expect(inf.is_disrupted).toBe(true);
    expect(inf.p_disrupted).toBeGreaterThan(0.999);
    expect(inf.recovery_source).toBe('movement');

    // The defect: a determinate, confident zero on an object that calls itself
    // disrupted. Withheld via the existing ceiling convention instead.
    expect(inf.recovery_indeterminate).toBe(true);
    expect(inf.recovery_minutes).toBe(MAX_RECOVERY_MINUTES);
    expect(inf.recovery_minutes_low).toBe(MAX_RECOVERY_MINUTES);
    expect(inf.recovery_minutes_high).toBe(MAX_RECOVERY_MINUTES);

    // p_normal_in_30min forecasts the PUBLISHED condition, which is normal, so
    // "still normal in 30 min" is a correct and well-measured claim. It is
    // deliberately not withheld with the recovery number.
    expect(inf.p_normal_in_30min).not.toBeNull();
    expect(inf.p_normal_in_30min!).toBeGreaterThan(0.9);
  });

  test('schedule arm clamped to an overdue zero withholds too (the arm-enumeration hole)', () => {
    // The guard must key off the ANSWER the selected arm produced, not off a
    // list of arms assumed to always be timing a disruption. derive.ts:256 is
    // explicit that alert_count counts "real-time alerts and any other id",
    // while has_realtime_alert only matches lmm:alert:*. Those are not
    // complements: a third-namespace alert is counted (so effectiveCondition
    // does NOT force normal, and is_disrupted goes true) yet leaves
    // has_realtime_alert false, which is exactly the scheduleRecovery
    // precondition. Combined with an announced resume already reached, the
    // schedule arm clamps to a determinate 0/[0,0] via `overdue` — the same
    // fabricated confident zero as the J row, on a different arm.
    const snap = build({
      routeId: 'M',
      roll: roll('disrupted'),
      alerts: [
        // Counted (not lmm:planned_work:*) but not real-time (not lmm:alert:*).
        entity({
          id: 'lmm:situation:99001',
          alertType: 'Delays',
          route: 'M',
          periods: [{ start: NOW - 2 * HOUR }],
        }),
        // Active planned window whose end is exactly now: scheduledResumeAt's
        // containment test is `now <= end`, so this is the resume time, and
        // `overdue = now >= resume` fires.
        entity({
          id: 'lmm:planned_work:99002',
          alertType: 'Planned - Part Suspended',
          route: 'M',
          periods: [{ start: NOW - 3 * HOUR, end: NOW }],
        }),
      ],
      movement: { state: 'disrupted', entered_at: NOW - 40 * MIN },
    });
    const status = snap.route_status.M!;
    const inf = status.inference!;

    // Preconditions: the schedule arm was selected and it went overdue.
    expect(inf.is_disrupted).toBe(true);
    expect(inf.recovery_source).toBe('schedule');
    expect(inf.overdue).toBe(true);
    expect(inf.resumes_at).toBe(NOW);
    expect(status.condition).toBe('disrupted');

    // Same invariant as the J row: no determinate zero-width zero under a
    // live disruption claim.
    expect(inf.recovery_indeterminate).toBe(true);
    expect(inf.recovery_minutes).toBe(MAX_RECOVERY_MINUTES);
    expect(inf.recovery_minutes_low).toBe(MAX_RECOVERY_MINUTES);
    expect(inf.recovery_minutes_high).toBe(MAX_RECOVERY_MINUTES);
  });

  test('an imminent FUTURE scheduled resume publishes a truthful 1, never a withheld unknown', () => {
    // This case was originally written asserting the withheld ceiling, which
    // was wrong and is the whole lesson. A resume 20 seconds out is a TRUTHFUL
    // imminent recovery; the old `Math.round` published it as a determinate
    // [0,0], and the composition guard then converted that to
    // 1440/indeterminate — turning "back in under a minute" into a
    // user-visible false unknown, strictly worse than the zero it replaced.
    //
    // The defect was the encoding, not the composition: `round` overloaded 0 to
    // mean both "already over" (the overdue clamp) and "less than 30 seconds
    // away". The countdown now ceils, so a strictly-positive wait is
    // structurally >= 1 and 0 belongs to `overdue` alone. That is what makes
    // the guard's zero-width-zero predicate exactly right rather than
    // approximately right.
    const snap = build({
      routeId: 'M',
      roll: roll('disrupted'),
      alerts: [
        entity({
          id: 'lmm:situation:99001',
          alertType: 'Delays',
          route: 'M',
          periods: [{ start: NOW - 2 * HOUR }],
        }),
        entity({
          id: 'lmm:planned_work:99002',
          alertType: 'Planned - Part Suspended',
          route: 'M',
          periods: [{ start: NOW - 3 * HOUR, end: NOW + 20 }],
        }),
      ],
      movement: { state: 'disrupted', entered_at: NOW - 40 * MIN },
    });
    const inf = snap.route_status.M!.inference!;

    // Strictly inside the window, so this reaches the schedule arm under either
    // end-boundary convention — and it is NOT the overdue clamp.
    expect(inf.is_disrupted).toBe(true);
    expect(inf.recovery_source).toBe('schedule');
    expect(inf.resumes_at).toBe(NOW + 20);
    expect(inf.overdue).toBe(false);

    // The truthful answer, kept: 20 seconds ceils to 1 minute, determinate.
    // Asserting 1 rather than "not 1440" so a regression back to round (0, then
    // withheld to the ceiling) fails here loudly.
    expect(inf.recovery_minutes).toBe(1);
    expect(inf.recovery_minutes_low).toBe(1);
    expect(inf.recovery_minutes_high).toBe(1);
    expect(inf.recovery_indeterminate).toBe(false);
  });

  test('a future resume never publishes zero, across the whole sub-minute window', () => {
    // The property the ceil change establishes, swept rather than sampled: for
    // every strictly-future resume offset, recovery is >= 1 and determinate, so
    // the composition guard can never fire on a resume that has not happened.
    // 0 is reachable from the schedule arm only via `overdue`.
    for (const offsetSec of [1, 20, 29, 30, 31, 59, 60, 61, 90, 119, 120]) {
      const snap = build({
        routeId: 'M',
        roll: roll('disrupted'),
        alerts: [
          entity({
            id: 'lmm:situation:99001',
            alertType: 'Delays',
            route: 'M',
            periods: [{ start: NOW - 2 * HOUR }],
          }),
          entity({
            id: 'lmm:planned_work:99002',
            alertType: 'Planned - Part Suspended',
            route: 'M',
            periods: [{ start: NOW - 3 * HOUR, end: NOW + offsetSec }],
          }),
        ],
        movement: { state: 'disrupted', entered_at: NOW - 40 * MIN },
      });
      const inf = snap.route_status.M!.inference!;
      expect(inf.recovery_source).toBe('schedule');
      expect(inf.overdue).toBe(false);
      expect(inf.recovery_indeterminate).toBe(false);
      expect(inf.recovery_minutes).toBe(Math.ceil(offsetSec / 60));
      expect(inf.recovery_minutes).toBeGreaterThanOrEqual(1);
    }
  });

  test('a near-zero recovery with a non-zero upper bound is a forecast, not a completion claim', () => {
    // Only the DETERMINATE ZERO-WIDTH zero is contradictory. A fitted curve
    // whose median rounds to 0 while the interval still has width says
    // "probably imminent, could be a while" — a legitimate forecast that must
    // survive, so the guard cannot simply test recovery_minutes === 0.
    const snap = build({
      routeId: 'M',
      roll: roll('disrupted'),
      alerts: [
        entity({
          id: 'lmm:situation:99001',
          alertType: 'Delays',
          route: 'M',
          periods: [{ start: NOW - 2 * HOUR }],
        }),
        // Resume 20 minutes out: a real countdown, nothing to withhold.
        entity({
          id: 'lmm:planned_work:99002',
          alertType: 'Planned - Part Suspended',
          route: 'M',
          periods: [{ start: NOW - 3 * HOUR, end: NOW + 20 * MIN }],
        }),
      ],
      movement: { state: 'disrupted', entered_at: NOW - 40 * MIN },
    });
    const inf = snap.route_status.M!.inference!;
    expect(inf.is_disrupted).toBe(true);
    expect(inf.recovery_source).toBe('schedule');
    expect(inf.overdue).toBe(false);
    expect(inf.recovery_indeterminate).toBe(false);
    expect(inf.recovery_minutes).toBe(20);
  });

  test('agreeing arms are untouched: both read disrupted, movement estimate survives', () => {
    const snap = build({
      routeId: 'J',
      roll: roll('disrupted'),
      alerts: [
        entity({
          id: 'lmm:alert:535417',
          alertType: 'Delays',
          route: 'J',
          periods: [{ start: NOW - 2 * HOUR }],
        }),
      ],
      movement: { state: 'disrupted', entered_at: NOW - 30 * MIN },
    });
    const inf = snap.route_status.J!.inference!;
    expect(snap.route_status.J!.condition).toBe('disrupted');
    expect(inf.is_disrupted).toBe(true);
    expect(inf.recovery_source).toBe('movement');
    // A real movement-curve estimate, not the withheld ceiling.
    expect(inf.recovery_indeterminate).toBe(false);
    expect(inf.recovery_minutes).toBeGreaterThan(0);
    expect(inf.recovery_minutes).toBeLessThan(MAX_RECOVERY_MINUTES);
    expect(inf.recovery_minutes_high).toBeGreaterThan(inf.recovery_minutes_low);
  });

  test('agreeing arms are untouched: both read normal, recovery stays a determinate zero', () => {
    const snap = build({
      routeId: 'J',
      roll: roll('normal'),
      movement: { state: 'normal', entered_at: NOW - HOUR },
    });
    const inf = snap.route_status.J!.inference!;
    expect(snap.route_status.J!.condition).toBe('normal');
    expect(inf.is_disrupted).toBe(false);
    expect(inf.recovery_source).toBe('movement');
    // Nothing to recover from, and nothing claiming otherwise: 0 is honest.
    expect(inf.recovery_indeterminate).toBe(false);
    expect(inf.recovery_minutes).toBe(0);
    expect(inf.recovery_minutes_low).toBe(0);
    expect(inf.recovery_minutes_high).toBe(0);
  });

  test('H-style row (no movement read, alert-arm dwell estimate) keeps its estimate', () => {
    // No movement state this tick, so the published condition is an honest
    // 'unknown' and the recovery block falls to the alert-HMM arm — the arm
    // is_disrupted itself comes from, so the arms agree and nothing is
    // withheld. Live H published 80 [55,100] this way.
    const snap = build({
      routeId: 'H',
      roll: roll('disrupted'),
      alerts: [
        entity({
          id: 'lmm:alert:535418',
          alertType: 'Delays',
          route: 'H',
          periods: [{ start: NOW - 2 * HOUR }],
        }),
      ],
      movement: null,
    });
    const status = snap.route_status.H!;
    const inf = status.inference!;
    expect(status.condition).toBe('unknown');
    expect(inf.condition).toBe('disrupted');
    expect(inf.is_disrupted).toBe(true);
    expect(inf.recovery_source).toBe('hmm');
    expect(inf.recovery_indeterminate).toBe(false);
    expect(inf.recovery_minutes).toBeGreaterThan(0);
    expect(inf.recovery_minutes).toBeLessThan(MAX_RECOVERY_MINUTES);
    // Still the alert arm forecasting its own regime, so the probability is
    // withheld exactly as before.
    expect(inf.p_normal_in_30min).toBeNull();
  });

  test('Z-style row (not_scheduled with no announced resume) keeps the 1440 ceiling', () => {
    // A No Scheduled Service window with no end: not_scheduled is published, so
    // the recovery question is live, but no arm that describes the published
    // condition can answer it. The pre-existing ceiling gate withholds. Note
    // is_disrupted is false here — not_scheduled never counts — so the new
    // guard is not what produces this.
    const snap = build({
      routeId: 'Z',
      roll: roll('normal'),
      alerts: [
        entity({
          id: 'lmm:planned_work:19829',
          alertType: 'No Scheduled Service',
          route: 'Z',
          periods: [{ start: NOW - HOUR }],
        }),
      ],
      movement: { state: 'normal', entered_at: NOW - HOUR },
    });
    const status = snap.route_status.Z!;
    const inf = status.inference!;
    expect(status.condition).toBe('not_scheduled');
    expect(inf.condition).toBe('not_scheduled');
    expect(inf.is_disrupted).toBe(false);
    expect(inf.resumes_at).toBeNull();
    expect(inf.recovery_indeterminate).toBe(true);
    expect(inf.recovery_minutes).toBe(MAX_RECOVERY_MINUTES);
    expect(inf.recovery_minutes_low).toBe(MAX_RECOVERY_MINUTES);
    expect(inf.recovery_minutes_high).toBe(MAX_RECOVERY_MINUTES);
  });
});
