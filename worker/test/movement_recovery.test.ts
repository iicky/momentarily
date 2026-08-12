/**
 * Recovery + p_normal_in_H off the MOVEMENT curve, conditioned on the
 * MOVEMENT clock — the fix for the P0 anti-correlation: the alert-HMM path
 * force-returns `normal` whenever disruptiveAlertCount is 0 (see
 * effectiveCondition), so p_normal_in_H described a route's alert history,
 * not the movement-primary condition published right next to it. When the
 * published condition came from movement, recovery/p_normal_in_H now come
 * from the movement curve + clock instead; the alert-HMM path is the
 * fallback only.
 */

import { describe, expect, test } from 'vitest';

import type { RouteRoll } from '../src/alpha';
import type { RouteSnapshot } from '../src/derive';
import { deriveRouteSnapshots } from '../src/derive';
import type { TrainedParams } from '../src/params';
import { movementDwellFor, parseTrainedParams } from '../src/params';
import { TICK_SECONDS, buildSnapshot } from '../src/snapshot';

const NOW = 1_700_000_000;
const MIN = 60;
const HOUR = 3600;

// Heavy-tailed disrupted movement dwell: median ~40min, quartiles 9min/40min/3h.
const DISRUPTED_CURVE = Array.from({ length: 21 }, (_, i) => Math.round(120 * 1.35 ** i));
// Heavy-tailed normal movement dwell, same shape as the real per-route cells
// the trainer emits (mirrors normal_dwell.test.ts's NORMAL_CURVE).
const NORMAL_CURVE = Array.from({ length: 21 }, (_, i) => Math.round(600 * 1.55 ** i));

function emissions(): Record<string, unknown> {
  return {
    poisson_lambda: [0.3, 4.0, 12.0],
    gamma_alpha: [1.0, 3.0, 6.0],
    gamma_beta: [2.0, 0.4, 0.2],
    bernoulli_p: [0.001, 0.05, 0.95],
    bernoulli_p_delays: [0.02, 0.6, 0.35],
    bernoulli_p_service_change: [0.02, 0.6, 0.4],
    bernoulli_p_planned: [0.05, 0.6, 0.35],
  };
}

function routeParams(): Record<string, unknown> {
  return {
    transition: [
      [0.95, 0.04, 0.01],
      [0.08, 0.9, 0.02],
      [0.02, 0.1, 0.88],
    ],
    initial: [0.9, 0.08, 0.02],
    emissions: emissions(),
  };
}

function movementDwellCell(): Record<string, unknown> {
  return {
    normal: { n: 40, q25_sec: 5368, median_sec: 48025, q75_sec: 429662, curve_sec: NORMAL_CURVE },
    disrupted: { n: 30, q25_sec: 538, median_sec: 2413, q75_sec: 10819, curve_sec: DISRUPTED_CURVE },
  };
}

/** trainedParams with `routes` for every id and, unless disabled, a
 * dwell_movement cell (normal + disrupted) for every id too. */
function trained(routeIds: string[], opts?: { dwellMovement?: boolean }): TrainedParams | null {
  const routes: Record<string, unknown> = {};
  const dwell_movement: Record<string, unknown> = {};
  for (const id of routeIds) {
    routes[id] = routeParams();
    dwell_movement[id] = movementDwellCell();
  }
  const doc: Record<string, unknown> = { schema_version: '1', trained_at: NOW, routes };
  if (opts?.dwellMovement !== false) doc['dwell_movement'] = dwell_movement;
  return parseTrainedParams(doc);
}

// The alert-HMM roll every route below carries: confident `normal`, no
// active alerts. This is the exact shape effectiveCondition force-returns
// `normal` for regardless of what movement says — the bug's precondition.
function normalRoll(): RouteRoll {
  return {
    filter: { probabilities: [0.95, 0.04, 0.01], regime_entered_at: NOW - HOUR, last_updated_at: NOW },
    published: { label: 'normal', pending_state: 'normal', pending_streak: 5, last_updated_at: NOW },
    alert_type_at_entry: null,
  };
}

function movementRegime(state: string, elapsedSec: number): { state: string; entered_at: number } {
  return { state, entered_at: NOW - elapsedSec };
}

function build(opts: {
  routeId: string;
  roll?: RouteRoll;
  routeSnapshots?: Map<string, RouteSnapshot>;
  movement?: { state: string; entered_at: number } | null;
  trainedParams: TrainedParams | null;
}) {
  return buildSnapshot({
    generatedAt: NOW,
    alertsFreshness: NOW,
    routeSnapshots: opts.routeSnapshots ?? deriveRouteSnapshots({ entity: [] }, NOW),
    rolls: { [opts.routeId]: opts.roll ?? normalRoll() },
    trainedParams: opts.trainedParams,
    tickSeconds: TICK_SECONDS,
    movementStates:
      opts.movement === undefined || opts.movement === null
        ? null
        : { observed_at: NOW - 300, regimes: { [opts.routeId]: opts.movement } },
  });
}

describe('movement recovery: p_normal_in_H off the movement curve + clock', () => {
  test('movement-disrupted has materially LOWER p_normal_in_30min than movement-normal (the anti-correlation fix)', () => {
    const params = trained(['A', 'B']);
    const disruptedSnap = build({
      routeId: 'A',
      movement: movementRegime('disrupted', 30 * MIN),
      trainedParams: params,
    });
    const normalSnap = build({
      routeId: 'B',
      movement: movementRegime('normal', HOUR),
      trainedParams: params,
    });
    const disruptedInf = disruptedSnap.route_status.A!.inference!;
    const normalInf = normalSnap.route_status.B!.inference!;

    // Both routes read `normal` on the alert-HMM shadow — zero alerts forces
    // it — yet the PUBLISHED condition (movement) disagrees for A. This is
    // exactly the anti-correlation: reading recovery off the shadow ignored
    // that disagreement entirely.
    expect(disruptedInf.condition).toBe('normal');
    expect(normalInf.condition).toBe('normal');
    expect(disruptedSnap.route_status.A!.condition).toBe('disrupted');
    expect(disruptedSnap.route_status.A!.condition_source).toBe('movement');
    expect(normalSnap.route_status.B!.condition).toBe('normal');

    expect(disruptedInf.recovery_source).toBe('movement');
    expect(normalInf.recovery_source).toBe('movement');
    // The pinning assertion: materially lower, not just "less than".
    expect(disruptedInf.p_normal_in_30min).toBeLessThan(normalInf.p_normal_in_30min - 0.3);
    expect(disruptedInf.p_normal_in_30min).toBeLessThan(0.5);
    expect(normalInf.p_normal_in_30min).toBeGreaterThan(0.9);
  });

  test('no dwell_movement in params falls back to unchanged alert-HMM behaviour', () => {
    const withoutMovementCurve = trained(['A'], { dwellMovement: false });
    const withMovement = build({
      routeId: 'A',
      movement: movementRegime('disrupted', 30 * MIN),
      trainedParams: withoutMovementCurve,
    });
    const withoutMovementStates = build({
      routeId: 'A',
      // No movement reading this tick — same runtime effect as omitting the
      // arg entirely (buildSnapshot treats null/undefined identically).
      movement: null,
      trainedParams: withoutMovementCurve,
    });
    const a = withMovement.route_status.A!.inference!;
    const b = withoutMovementStates.route_status.A!.inference!;
    // Published condition still reflects movement (worker/src/snapshot.ts's
    // own condition read doesn't depend on dwell_movement)...
    expect(withMovement.route_status.A!.condition).toBe('disrupted');
    // ...but with no curve to read, recovery falls all the way back to the
    // alert-HMM path — identical output to a tick with no movement reading
    // at all.
    expect(a.recovery_source).toBe('hmm');
    expect(a.p_normal_in_30min).toBe(b.p_normal_in_30min);
    expect(a.p_normal_in_60min).toBe(b.p_normal_in_60min);
    expect(a.recovery_minutes).toBe(b.recovery_minutes);
  });

  test('movement_regime_entered_at of 0 falls back to the alert-HMM path', () => {
    const params = trained(['A']);
    const zeroClock = build({
      routeId: 'A',
      movement: { state: 'disrupted', entered_at: 0 },
      trainedParams: params,
    });
    const noMovement = build({ routeId: 'A', movement: null, trainedParams: params });
    const a = zeroClock.route_status.A!.inference!;
    const b = noMovement.route_status.A!.inference!;
    expect(a.recovery_source).toBe('hmm');
    expect(a.p_normal_in_30min).toBe(b.p_normal_in_30min);
    expect(a.recovery_minutes).toBe(b.recovery_minutes);
  });

  test('a suspended movement regime defers to the alert arm rather than assuming its exit split', () => {
    // Exits from `disrupted` were measured at 100% to normal, so reading the
    // exit probability as p_normal is justified there. `suspended` completed
    // no episode in either archive window, so its split is unmeasured — a
    // route resuming from no trains at all can come back degraded. Serving
    // the raw exit probability would publish a P(normal) nothing supports.
    //
    // The params below DO carry a suspended curve, so the only thing declining
    // to use it is the measured-split guard.
    const params = parseTrainedParams({
      schema_version: '1',
      trained_at: NOW,
      routes: { A: routeParams() },
      dwell_movement: {
        A: { ...movementDwellCell(), suspended: { n: 12, q25_sec: 538, median_sec: 2413, q75_sec: 10819, curve_sec: DISRUPTED_CURVE } },
      },
    });
    expect(params!.dwellMovement.A!.suspended).toBeDefined();

    const suspendedCurve = build({
      routeId: 'A',
      movement: movementRegime('suspended', 30 * MIN),
      trainedParams: params,
    });
    const inf = suspendedCurve.route_status.A!.inference!;
    expect(suspendedCurve.route_status.A!.condition).toBe('suspended');
    expect(suspendedCurve.route_status.A!.condition_source).toBe('movement');
    expect(inf.recovery_source).not.toBe('movement');

    // And it matches the no-movement-reading fallback exactly, so deferring is
    // a real handoff to the alert arm, not a separate half-path.
    const fallback = build({ routeId: 'A', movement: null, trainedParams: params });
    const fb = fallback.route_status.A!.inference!;
    expect(inf.p_normal_in_30min).toBe(fb.p_normal_in_30min);
    expect(inf.recovery_minutes).toBe(fb.recovery_minutes);
  });

  test('recovery_source is "movement" only when the movement curve was actually used', () => {
    const params = trained(['A']);

    // No movement reading at all this tick.
    const noReading = build({ routeId: 'A', movement: null, trainedParams: params });
    expect(noReading.route_status.A!.inference!.recovery_source).toBe('hmm');

    // A movement reading, but not_scheduled wins precedence over movement —
    // buildSnapshot itself never reads condition_source='movement' here, and
    // buildInference must agree, not use the curve either.
    const snaps = new Map<string, RouteSnapshot>([
      [
        'A',
        {
          route_id: 'A',
          observation: {
            alert_count: 0,
            severity_sum: 0,
            has_suspended_alert: false,
            has_delays: false,
            has_service_change: false,
            has_planned: false,
            tod_bin: 0,
          },
          active_alert_ids: ['lmm:planned_work:1'],
          alerts: [],
          severity_max: 0,
          primary_alert_type: 'No Scheduled Service',
          coarse_label: 'No Scheduled Service',
          by_direction: {
            northbound: { alerts: [], primary_alert_type: null },
            southbound: { alerts: [], primary_alert_type: null },
          },
          has_realtime_alert: false,
          is_not_scheduled: true,
          scheduled_resume_at: null,
        },
      ],
    ]);
    const notScheduled = build({
      routeId: 'A',
      routeSnapshots: snaps,
      movement: movementRegime('disrupted', 30 * MIN),
      trainedParams: params,
    });
    expect(notScheduled.route_status.A!.condition).toBe('not_scheduled');
    expect(notScheduled.route_status.A!.condition_source).toBe('schedule');
    expect(notScheduled.route_status.A!.inference!.recovery_source).toBe('hmm');

    // A genuine movement reading with a curve: 'movement'.
    const withReading = build({
      routeId: 'A',
      movement: movementRegime('disrupted', 30 * MIN),
      trainedParams: params,
    });
    expect(withReading.route_status.A!.inference!.recovery_source).toBe('movement');
  });

  test('schedule countdown still wins over the movement curve', () => {
    const params = trained(['S']);
    const roll: RouteRoll = {
      filter: { probabilities: [0.02, 0.95, 0.03], regime_entered_at: NOW - 2 * HOUR, last_updated_at: NOW },
      published: { label: 'disrupted', pending_state: 'disrupted', pending_streak: 5, last_updated_at: NOW },
      alert_type_at_entry: 'Delays',
    };
    const snaps = new Map<string, RouteSnapshot>([
      [
        'S',
        {
          route_id: 'S',
          observation: {
            alert_count: 1,
            severity_sum: 30,
            has_suspended_alert: false,
            has_delays: true,
            has_service_change: false,
            has_planned: false,
            tod_bin: 0,
          },
          active_alert_ids: ['lmm:alert:1'],
          alerts: [],
          severity_max: 30,
          primary_alert_type: 'Delays',
          coarse_label: 'Delays',
          by_direction: {
            northbound: { alerts: [], primary_alert_type: null },
            southbound: { alerts: [], primary_alert_type: null },
          },
          has_realtime_alert: false,
          is_not_scheduled: false,
          scheduled_resume_at: NOW + 30 * MIN,
        },
      ],
    ]);
    const snap = build({
      routeId: 'S',
      roll,
      routeSnapshots: snaps,
      // A movement curve exists and IS eligible (isNotScheduled is false) —
      // schedule must still win.
      movement: movementRegime('disrupted', 45 * MIN),
      trainedParams: params,
    });
    const inf = snap.route_status.S!.inference!;
    expect(inf.recovery_source).toBe('schedule');
    expect(inf.recovery_minutes).toBe(30);
    expect(inf.resumes_at).toBe(NOW + 30 * MIN);
  });
});

describe('movementDwellFor (C3)', () => {
  test('reads a (route, state) cell from dwell_movement', () => {
    const params = trained(['A']);
    const cell = movementDwellFor(params, 'A', 'disrupted');
    expect(cell).not.toBeNull();
    expect(cell!.curve_sec).toEqual(DISRUPTED_CURVE);
  });

  test('missing route, missing state, or null trained returns null (never throws)', () => {
    const params = trained(['A']);
    expect(movementDwellFor(params, 'B', 'disrupted')).toBeNull();
    expect(movementDwellFor(params, 'A', 'suspended')).toBeNull();
    expect(movementDwellFor(null, 'A', 'disrupted')).toBeNull();
  });

  test('absent dwell_movement in params.json yields a clean null, not a throw', () => {
    const params = trained(['A'], { dwellMovement: false });
    expect(params).not.toBeNull();
    expect(params!.dwellMovement).toEqual({});
    expect(movementDwellFor(params, 'A', 'disrupted')).toBeNull();
  });
});
