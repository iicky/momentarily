/**
 * p_normal_in_H for a route whose PUBLISHED condition is movement-sourced and
 * currently normal — movementRecovery()'s `state === 'normal'` branch, off
 * the dwell_movement cell and the movement regime clock (entered_at). The
 * alert-HMM arm no longer computes or publishes a normal-state forecast at
 * all: p_normal_in_30min is populated only when the arm that produced it also
 * produced the published condition (see the Inference interface in
 * snapshot.ts — movement-sourced rows graded AUC 0.856 against the alert
 * arm's 0.261; publishing both in one field scored 0.084, worse than either).
 *
 * The geometric projection decays a per-tick self-loop and has no elapsed
 * term, so it reads P(still normal) down with the horizon regardless of how
 * long the route has already been fine. The empirical movement-dwell curve
 * conditions on elapsed instead. See the by_current decomposition in
 * training/eval.py. hmm.ts's projectForward is unrelated to this path now —
 * snapshot.ts never calls it for a normal-state forecast, curve or no curve —
 * and stays covered on its own by parity.test.ts's `describe('projectForward', ...)`.
 */

import { describe, expect, test } from 'vitest';

import type { RouteRoll } from '../src/alpha';
import { deriveRouteSnapshots } from '../src/derive';
import { pLeaveBy } from '../src/dwell';
import type { TrainedParams } from '../src/params';
import { parseTrainedParams } from '../src/params';
import { TICK_SECONDS, buildSnapshot } from '../src/snapshot';

const NOW = 1_700_000_000;
const HOUR = 3600;

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

// A heavy-tailed normal movement dwell: quartiles at 10h / 53h / 83h, like the
// real per-route dwell_movement cells the trainer emits (mirrors
// movement_recovery.test.ts's NORMAL_CURVE).
const NORMAL_CURVE = Array.from({ length: 21 }, (_, i) => Math.round(600 * 1.55 ** i));

function normalMovementCell(): Record<string, unknown> {
  return {
    n: 8,
    n_censored: 1,
    q25_sec: 36_000,
    median_sec: 191_978,
    q75_sec: 298_540,
    curve_sec: NORMAL_CURVE,
  };
}

// Atom mixture fixture: most normal-regime episodes end on the very first
// movement tick, with the log-logistic tail refit conditional on T > atom_sec.
// curve_sec is deliberately kept as the STALE plain NORMAL_CURVE (not
// recomputed for the mixture) to prove the atom path reads
// tail_ll/atom_p/atom_sec and never falls back to it.
const ATOM_SHAPE = 1.8;
const ATOM_SCALE = 250_000;
const ATOM_P = 0.55;
const ATOM_SEC = TICK_SECONDS;

function normalMovementCellAtom(): Record<string, unknown> {
  return {
    n: 50,
    n_censored: 4,
    q25_sec: ATOM_SEC,
    median_sec: ATOM_SEC,
    q75_sec: 200_000,
    curve_sec: NORMAL_CURVE,
    tail_ll: [ATOM_SHAPE, ATOM_SCALE],
    atom_p: ATOM_P,
    atom_sec: ATOM_SEC,
  };
}

/** trainedParams with route A and, when `normalCell` is given, a
 * dwell_movement.A.normal cell built from it. Omitting it leaves
 * dwell_movement out of the document entirely — the no-curve case. */
function trained(normalCell?: Record<string, unknown>): TrainedParams | null {
  const doc: Record<string, unknown> = {
    schema_version: '1',
    trained_at: NOW,
    routes: { A: routeParams() },
  };
  if (normalCell) doc['dwell_movement'] = { A: { normal: normalCell } };
  return parseTrainedParams(doc);
}

// The alert-HMM roll every route below carries: confident `normal`, no active
// alerts — the shadow condition (inference.condition). The PUBLISHED
// condition below comes from the movement regime instead (see pNormal).
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

function pNormal(
  trainedParams: TrainedParams | null,
  elapsedSec: number,
): { p30: number | null; p60: number | null; p120: number | null } {
  const snap = buildSnapshot({
    generatedAt: NOW,
    alertsFreshness: NOW,
    routeSnapshots: deriveRouteSnapshots({ entity: [] }, NOW),
    rolls: { A: normalRoll() },
    trainedParams,
    tickSeconds: TICK_SECONDS,
    movementStates: { observed_at: NOW - 300, regimes: { A: movementRegime('normal', elapsedSec) } },
  });
  const status = snap.route_status.A!;
  const inf = status.inference!;
  expect(status.condition).toBe('normal');
  expect(status.condition_source).toBe('movement');
  expect(inf.condition).toBe('normal');
  expect(inf.recovery_source).toBe('movement');
  return {
    p30: inf.p_normal_in_30min,
    p60: inf.p_normal_in_60min,
    p120: inf.p_normal_in_120min,
  };
}

describe('p_normal_in_H for a route whose published condition is movement-normal', () => {
  const withCurve = trained(normalMovementCell());
  const withoutCurve = trained();

  test('a long-running movement-normal regime stays near-certain at the published (30min) horizon; 60/120min are withheld', () => {
    const p = pNormal(withCurve, 20 * HOUR);
    expect(p.p30).toBeGreaterThan(0.97);
    expect(p.p60).toBeNull();
    expect(p.p120).toBeNull();
  });

  test('a regime that has already lasted longer is likelier to persist', () => {
    // p_normal_in_120min is withheld on this arm (recovery_source
    // 'movement' is a fitted-curve arm, not 'schedule'), so pin the
    // "survival so far is evidence of more survival" property directly on
    // pLeaveBy — the closed form movementRecovery's normal branch calls for
    // this exact computation.
    const staysNormalFor120 = (elapsedSec: number): number =>
      1 - pLeaveBy(NORMAL_CURVE, elapsedSec, 7200);
    expect(staysNormalFor120(40 * HOUR)).toBeGreaterThan(staysNormalFor120(30 * 60));

    const young = pNormal(withCurve, 30 * 60);
    const old = pNormal(withCurve, 40 * HOUR);
    expect(young.p120).toBeNull();
    expect(old.p120).toBeNull();
  });

  test('every horizon stays a probability', () => {
    for (const elapsed of [0, 60, 5 * HOUR, 500 * HOUR]) {
      const p = pNormal(withCurve, elapsed);
      expect(p.p30).toBeGreaterThanOrEqual(0);
      expect(p.p30).toBeLessThanOrEqual(1);
      expect(Number.isFinite(p.p30)).toBe(true);
      // 60/120min are withheld on this arm (recovery_source 'movement'); the
      // underlying computation still runs every tick, so check its output is
      // a valid probability directly off pLeaveBy instead of via the
      // (now-null) published fields.
      expect(p.p60).toBeNull();
      expect(p.p120).toBeNull();
      for (const horizonSec of [3600, 7200]) {
        const v = 1 - pLeaveBy(NORMAL_CURVE, elapsed, horizonSec);
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
        expect(Number.isFinite(v)).toBe(true);
      }
    }
  });

  test('P(normal) is non-increasing in the horizon', () => {
    // p_normal_in_60min/120min are withheld on this arm (recovery_source
    // 'movement'), so this invariant is no longer observable on the
    // published fields — pin it on pLeaveBy directly, the closed form
    // movementRecovery's normal branch (staysNormalFor) calls for every
    // horizon.
    const elapsed = 6 * HOUR;
    const staysNormalFor = (horizonSec: number): number =>
      1 - pLeaveBy(NORMAL_CURVE, elapsed, horizonSec);
    const p30 = staysNormalFor(1800);
    const p60 = staysNormalFor(3600);
    const p120 = staysNormalFor(7200);
    expect(p30).toBeGreaterThanOrEqual(p60);
    expect(p60).toBeGreaterThanOrEqual(p120);

    const p = pNormal(withCurve, elapsed);
    expect(p.p30).toBeCloseTo(p30, 9);
    expect(p.p60).toBeNull();
    expect(p.p120).toBeNull();
  });

  test('a movement-sourced condition with no dwell_movement cell publishes p_normal_in_30min === null but still reports recovery_source "hmm"', () => {
    // The regression this whole file exists to catch: movementRecovery
    // returns null when the (route, 'normal') cell has no curve_sec, and
    // unlike the disrupted/suspended branches there is no geometric fallback
    // for 'normal' — so the forecast is withheld outright rather than
    // silently describing a different arm's clock.
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: deriveRouteSnapshots({ entity: [] }, NOW),
      rolls: { A: normalRoll() },
      trainedParams: withoutCurve,
      tickSeconds: TICK_SECONDS,
      movementStates: { observed_at: NOW - 300, regimes: { A: movementRegime('normal', 20 * HOUR) } },
    });
    const status = snap.route_status.A!;
    const inf = status.inference!;
    // The published condition is still movement-sourced...
    expect(status.condition).toBe('normal');
    expect(status.condition_source).toBe('movement');
    // ...but with no dwell_movement cell for this route, the forecast is
    // withheld outright, and recovery_source falls back to the untouched
    // alert-HMM default rather than claiming 'movement'.
    expect(inf.recovery_source).toBe('hmm');
    expect(inf.p_normal_in_30min).toBeNull();
    expect(inf.p_normal_in_60min).toBeNull();
    expect(inf.p_normal_in_120min).toBeNull();
  });
});

describe('p_normal_in_H with the atom mixture active on the movement-normal cell', () => {
  const withAtom = trained(normalMovementCellAtom());

  test("matches pLeaveBy's closed-form mixture output directly, not the stale curve_sec", () => {
    const elapsed = 20 * HOUR;
    const p = pNormal(withAtom, elapsed);
    const tail: [number, number] = [ATOM_SHAPE, ATOM_SCALE];
    const atom = { p: ATOM_P, sec: ATOM_SEC };
    // curveSec passed as [] on purpose: the atom closed form must not read it
    // at all, so an empty (unusable) curve still reproduces what buildSnapshot
    // published from the real (stale) NORMAL_CURVE.
    expect(p.p30).toBeCloseTo(1 - pLeaveBy([], elapsed, 1800, tail, atom), 9);
    // 60/120min are withheld on this arm (recovery_source 'movement') — see
    // the "genuinely differs" test below, which pins the same closed form
    // directly at those horizons instead of via the published field.
    expect(p.p60).toBeNull();
    expect(p.p120).toBeNull();
  });

  test('genuinely differs from the legacy curve_sec/tail_ll splice for the same cell', () => {
    // Proof curve_sec was bypassed, not coincidentally reproduced.
    const elapsed = 20 * HOUR;
    const p = pNormal(withAtom, elapsed);
    const tail: [number, number] = [ATOM_SHAPE, ATOM_SCALE];
    const atom = { p: ATOM_P, sec: ATOM_SEC };
    const legacyP30 = 1 - pLeaveBy(NORMAL_CURVE, elapsed, 1800, tail);
    expect(p.p30).not.toBeCloseTo(legacyP30, 6);

    // p_normal_in_60min/120min are withheld on this arm (recovery_source
    // 'movement'), so the same divergence can no longer be read off the
    // published fields — pin it directly on the pLeaveBy closed form the
    // field would have been assembled from.
    const atomP60 = 1 - pLeaveBy([], elapsed, 3600, tail, atom);
    const legacyP60 = 1 - pLeaveBy(NORMAL_CURVE, elapsed, 3600, tail);
    expect(atomP60).not.toBeCloseTo(legacyP60, 6);

    const atomP120 = 1 - pLeaveBy([], elapsed, 7200, tail, atom);
    const legacyP120 = 1 - pLeaveBy(NORMAL_CURVE, elapsed, 7200, tail);
    expect(atomP120).not.toBeCloseTo(legacyP120, 6);
  });
});

describe('p_normal_in_H legacy curve_sec/tail_ll path is unaffected by the atom fields being absent', () => {
  test('matches pLeaveBy(curve, elapsed, horizon, tail) called directly, with no atom', () => {
    const tail: [number, number] = [1.8, 5400];
    const withTail = trained({
      n: 50,
      q25_sec: 30_000,
      median_sec: 190_000,
      q75_sec: 300_000,
      curve_sec: NORMAL_CURVE,
      tail_ll: tail,
    });
    const elapsed = 1200 * HOUR; // past the curve, exercises the tail branch
    const p = pNormal(withTail, elapsed);
    expect(p.p30).toBeCloseTo(1 - pLeaveBy(NORMAL_CURVE, elapsed, 1800, tail), 9);
    // 60min is withheld on this arm (recovery_source 'movement'), same as
    // every other fitted-curve path.
    expect(p.p60).toBeNull();
  });
});
