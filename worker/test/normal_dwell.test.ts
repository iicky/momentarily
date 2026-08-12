/**
 * p_normal_in_H while the route is NORMAL now.
 *
 * The geometric projection decays a per-tick self-loop and has no elapsed term,
 * so it reads P(still normal) down with the horizon regardless of how long the
 * route has already been fine. The empirical normal-dwell curve conditions on
 * elapsed instead. See the by_current decomposition in training/eval.py.
 */

import { describe, expect, test } from 'vitest';

import type { RouteRoll } from '../src/alpha';
import { deriveRouteSnapshots } from '../src/derive';
import { pLeaveBy } from '../src/dwell';
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

function route(dwell?: Record<string, unknown>): Record<string, unknown> {
  const doc: Record<string, unknown> = {
    transition: [
      [0.95, 0.04, 0.01],
      [0.08, 0.9, 0.02],
      [0.02, 0.1, 0.88],
    ],
    initial: [0.9, 0.08, 0.02],
    emissions: emissions(),
  };
  if (dwell) doc['dwell_quantiles'] = dwell;
  return doc;
}

// A heavy-tailed normal regime: quartiles at 10h / 53h / 83h, like the real
// per-route cells the trainer emits.
const NORMAL_CURVE = Array.from({ length: 21 }, (_, i) => Math.round(600 * 1.55 ** i));

function normalDwell(): Record<string, unknown> {
  return {
    normal: {
      n: 8,
      n_censored: 1,
      q25_sec: 36_000,
      median_sec: 191_978,
      q75_sec: 298_540,
      curve_sec: NORMAL_CURVE,
    },
  };
}

// Atom mixture fixture: most normal-regime episodes end on the very first
// tick, with the log-logistic tail refit conditional on T > atom_sec.
// curve_sec is deliberately kept as the STALE plain NORMAL_CURVE (not
// recomputed for the mixture) to prove the atom path reads
// tail_ll/atom_p/atom_sec and never falls back to it.
const ATOM_SHAPE = 1.8;
const ATOM_SCALE = 250_000;
const ATOM_P = 0.55;
const ATOM_SEC = TICK_SECONDS;

function normalDwellAtom(): Record<string, unknown> {
  return {
    normal: {
      n: 50,
      n_censored: 4,
      q25_sec: ATOM_SEC,
      median_sec: ATOM_SEC,
      q75_sec: 200_000,
      curve_sec: NORMAL_CURVE,
      tail_ll: [ATOM_SHAPE, ATOM_SCALE],
      atom_p: ATOM_P,
      atom_sec: ATOM_SEC,
    },
  };
}

function normalRoll(regimeEnteredAt: number): RouteRoll {
  return {
    filter: {
      probabilities: [0.95, 0.04, 0.01],
      regime_entered_at: regimeEnteredAt,
      last_updated_at: NOW,
    },
    published: { label: 'normal', pending_state: 'normal', pending_streak: 5, last_updated_at: NOW },
    alert_type_at_entry: null,
  };
}

function pNormal(
  trainedParams: ReturnType<typeof parseTrainedParams>,
  elapsedSec: number,
): { p30: number; p60: number; p120: number } {
  const snap = buildSnapshot({
    generatedAt: NOW,
    alertsFreshness: NOW,
    routeSnapshots: deriveRouteSnapshots({ entity: [] }, NOW),
    rolls: { A: normalRoll(NOW - elapsedSec) },
    trainedParams,
    tickSeconds: TICK_SECONDS,
  });
  const inf = snap.route_status.A!.inference!;
  expect(inf.condition).toBe('normal');
  return {
    p30: inf.p_normal_in_30min,
    p60: inf.p_normal_in_60min,
    p120: inf.p_normal_in_120min,
  };
}

describe('p_normal_in_H for a route that is normal now', () => {
  const withCurve = parseTrainedParams({
    schema_version: '1',
    trained_at: NOW,
    routes: { A: route(normalDwell()) },
  });
  const withoutCurve = parseTrainedParams({
    schema_version: '1',
    trained_at: NOW,
    routes: { A: route() },
  });

  test('a long-running normal regime stays near-certain across every horizon', () => {
    const p = pNormal(withCurve, 20 * HOUR);
    expect(p.p30).toBeGreaterThan(0.97);
    expect(p.p60).toBeGreaterThan(0.95);
    expect(p.p120).toBeGreaterThan(0.9);
  });

  test('beats the geometric projection it replaces, most at the long horizon', () => {
    const elapsed = 20 * HOUR;
    const empirical = pNormal(withCurve, elapsed);
    const geometric = pNormal(withoutCurve, elapsed);
    expect(empirical.p30).toBeGreaterThan(geometric.p30);
    expect(empirical.p120).toBeGreaterThan(geometric.p120);
    // The geometric decay is the bug: it falls away with the horizon.
    expect(geometric.p120).toBeLessThan(geometric.p30);
    expect(empirical.p120 - geometric.p120).toBeGreaterThan(empirical.p30 - geometric.p30);
  });

  test('a regime that has already lasted longer is likelier to persist', () => {
    // Heavy-tailed dwell: survival so far is evidence of more survival.
    const young = pNormal(withCurve, 30 * 60);
    const old = pNormal(withCurve, 40 * HOUR);
    expect(old.p120).toBeGreaterThan(young.p120);
  });

  test('falls back to the geometric projection when the route has no curve', () => {
    const p = pNormal(withoutCurve, 20 * HOUR);
    // Unchanged legacy behavior: decays with the horizon.
    expect(p.p30).toBeGreaterThan(p.p60);
    expect(p.p60).toBeGreaterThan(p.p120);
  });

  test('every horizon stays a probability', () => {
    for (const elapsed of [0, 60, 5 * HOUR, 500 * HOUR]) {
      const p = pNormal(withCurve, elapsed);
      for (const v of [p.p30, p.p60, p.p120]) {
        expect(v).toBeGreaterThanOrEqual(0);
        expect(v).toBeLessThanOrEqual(1);
        expect(Number.isFinite(v)).toBe(true);
      }
    }
  });

  test('P(normal) is non-increasing in the horizon', () => {
    const p = pNormal(withCurve, 6 * HOUR);
    expect(p.p30).toBeGreaterThanOrEqual(p.p60);
    expect(p.p60).toBeGreaterThanOrEqual(p.p120);
  });
});

describe('p_normal_in_H with the atom mixture active on the normal cell', () => {
  const withAtom = parseTrainedParams({
    schema_version: '1',
    trained_at: NOW,
    routes: { A: route(normalDwellAtom()) },
  });

  test("matches pLeaveBy's closed-form mixture output directly, not the stale curve_sec", () => {
    const elapsed = 20 * HOUR;
    const p = pNormal(withAtom, elapsed);
    const tail: [number, number] = [ATOM_SHAPE, ATOM_SCALE];
    const atom = { p: ATOM_P, sec: ATOM_SEC };
    // curveSec passed as [] on purpose: the atom closed form must not read it
    // at all, so an empty (unusable) curve still reproduces what buildSnapshot
    // published from the real (stale) NORMAL_CURVE.
    expect(p.p30).toBeCloseTo(1 - pLeaveBy([], elapsed, 1800, tail, atom), 9);
    expect(p.p60).toBeCloseTo(1 - pLeaveBy([], elapsed, 3600, tail, atom), 9);
    expect(p.p120).toBeCloseTo(1 - pLeaveBy([], elapsed, 7200, tail, atom), 9);
  });

  test('genuinely differs from the legacy curve_sec/tail_ll splice for the same cell', () => {
    // Proof curve_sec was bypassed, not coincidentally reproduced.
    const elapsed = 20 * HOUR;
    const p = pNormal(withAtom, elapsed);
    const tail: [number, number] = [ATOM_SHAPE, ATOM_SCALE];
    const legacyP30 = 1 - pLeaveBy(NORMAL_CURVE, elapsed, 1800, tail);
    expect(p.p30).not.toBeCloseTo(legacyP30, 6);
  });
});

describe('p_normal_in_H legacy curve_sec/tail_ll path is unaffected by the atom fields being absent', () => {
  test('matches pLeaveBy(curve, elapsed, horizon, tail) called directly, with no atom', () => {
    const tail: [number, number] = [1.8, 5400];
    const withTail = parseTrainedParams({
      schema_version: '1',
      trained_at: NOW,
      routes: {
        A: route({
          normal: {
            n: 50,
            q25_sec: 30_000,
            median_sec: 190_000,
            q75_sec: 300_000,
            curve_sec: NORMAL_CURVE,
            tail_ll: tail,
          },
        }),
      },
    });
    const elapsed = 1200 * HOUR; // past the curve, exercises the tail branch
    const p = pNormal(withTail, elapsed);
    expect(p.p30).toBeCloseTo(1 - pLeaveBy(NORMAL_CURVE, elapsed, 1800, tail), 9);
    expect(p.p60).toBeCloseTo(1 - pLeaveBy(NORMAL_CURVE, elapsed, 3600, tail), 9);
  });
});
