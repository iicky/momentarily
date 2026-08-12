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
import { projectForward } from '../src/hmm';
import { parseTrainedParams, paramsForRoute } from '../src/params';
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
): { p30: number; p60: number | null; p120: number | null } {
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

  test('a long-running normal regime stays near-certain at the published (30min) horizon; 60/120min are withheld', () => {
    const p = pNormal(withCurve, 20 * HOUR);
    expect(p.p30).toBeGreaterThan(0.97);
    expect(p.p60).toBeNull();
    expect(p.p120).toBeNull();
  });

  test('beats the geometric projection it replaces, most at the long horizon', () => {
    const elapsed = 20 * HOUR;
    const empirical = pNormal(withCurve, elapsed);
    const geometric = pNormal(withoutCurve, elapsed);
    // p_normal_in_30min is still published: the win is directly observable there.
    expect(empirical.p30).toBeGreaterThan(geometric.p30);
    expect(empirical.p60).toBeNull();
    expect(empirical.p120).toBeNull();

    // p_normal_in_60min/120min are withheld on this arm (recovery_source
    // 'hmm' — every normal-regime forecast is), so "most at the long
    // horizon" can no longer be read off the published fields. It's still a
    // real property of the math buildInference computes internally —
    // pLeaveBy (dwell.ts) for the empirical curve vs. hmm.ts's
    // projectForward for the geometric fallback — so pin it there directly
    // instead of via the snapshot.
    const geometricParams = paramsForRoute(withoutCurve, 'A');
    const ticksFor = (horizonSec: number): number =>
      Math.max(1, Math.round(horizonSec / TICK_SECONDS));
    const geometric120 = projectForward(
      normalRoll(NOW - elapsed).filter,
      geometricParams,
      ticksFor(7200),
    )[0];
    const empirical120 = 1 - pLeaveBy(NORMAL_CURVE, elapsed, 7200);

    expect(empirical120).toBeGreaterThan(geometric120);
    // The geometric decay is the bug: it falls away with the horizon.
    expect(geometric120).toBeLessThan(geometric.p30);
    expect(empirical120 - geometric120).toBeGreaterThan(empirical.p30 - geometric.p30);
  });

  test('a regime that has already lasted longer is likelier to persist', () => {
    // p_normal_in_120min is withheld on this arm (recovery_source 'hmm'), so
    // pin the "survival so far is evidence of more survival" property
    // directly on pLeaveBy — the closed form buildInference's normal-regime
    // arm calls for this exact computation.
    const staysNormalFor120 = (elapsedSec: number): number =>
      1 - pLeaveBy(NORMAL_CURVE, elapsedSec, 7200);
    expect(staysNormalFor120(40 * HOUR)).toBeGreaterThan(staysNormalFor120(30 * 60));

    const young = pNormal(withCurve, 30 * 60);
    const old = pNormal(withCurve, 40 * HOUR);
    expect(young.p120).toBeNull();
    expect(old.p120).toBeNull();
  });

  test('falls back to the geometric projection when the route has no curve', () => {
    const elapsed = 20 * HOUR;
    const p = pNormal(withoutCurve, elapsed);
    expect(p.p60).toBeNull();
    expect(p.p120).toBeNull();

    // Unchanged legacy behavior: decays with the horizon. p_normal_in_60min/
    // 120min are withheld even on this no-curve arm (still recovery_source
    // 'hmm'), so read the geometric decay off projectForward directly — the
    // same call buildInference makes when there's no dwell curve to
    // override it.
    const params = paramsForRoute(withoutCurve, 'A');
    const filter = normalRoll(NOW - elapsed).filter;
    const ticksFor = (horizonSec: number): number =>
      Math.max(1, Math.round(horizonSec / TICK_SECONDS));
    const p60 = projectForward(filter, params, ticksFor(3600))[0];
    const p120 = projectForward(filter, params, ticksFor(7200))[0];
    expect(p.p30).toBeGreaterThan(p60);
    expect(p60).toBeGreaterThan(p120);
  });

  test('every horizon stays a probability', () => {
    for (const elapsed of [0, 60, 5 * HOUR, 500 * HOUR]) {
      const p = pNormal(withCurve, elapsed);
      expect(p.p30).toBeGreaterThanOrEqual(0);
      expect(p.p30).toBeLessThanOrEqual(1);
      expect(Number.isFinite(p.p30)).toBe(true);
      // 60/120min are withheld on this arm (recovery_source 'hmm'); the
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
    // 'hmm'), so this invariant is no longer observable on the published
    // fields — pin it on pLeaveBy directly, the closed form buildInference's
    // normal-regime arm (staysNormalFor) calls for every horizon.
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
    // 60/120min are withheld on this arm (recovery_source 'hmm') — see the
    // "genuinely differs" test below, which pins the same closed form
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
    // 'hmm'), so the same divergence can no longer be read off the
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
    // 60min is withheld on this arm (recovery_source 'hmm'), same as every
    // other fitted-curve path.
    expect(p.p60).toBeNull();
  });
});
