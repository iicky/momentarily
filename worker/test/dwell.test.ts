/**
 * Conditional survival math over the empirical dwell curve.
 *
 * Mirrors tests/test_dwell.py (Python reference implementation); the fixtures
 * and expected values are intentionally identical so the two implementations
 * can't drift apart silently.
 */

import { describe, expect, test } from 'vitest';

import { conditionalRecovery, dwellCdf, mixtureQuantile, mixtureSurvival, pLeaveBy } from '../src/dwell';

describe('dwellCdf', () => {
  test('uniform two-point curve', () => {
    const curve = [0, 100];
    expect(dwellCdf(curve, -5)).toBe(0.0);
    expect(dwellCdf(curve, 0)).toBe(0.0);
    expect(dwellCdf(curve, 50)).toBe(0.5);
    expect(dwellCdf(curve, 100)).toBe(1.0);
    expect(dwellCdf(curve, 500)).toBe(1.0);
  });

  test('flat curve reads as outlived at its value', () => {
    const curve = Array(21).fill(600);
    expect(dwellCdf(curve, 600)).toBe(1.0);
    expect(dwellCdf(curve, 0)).toBe(0.0);
  });
});

describe('conditionalRecovery', () => {
  test('uniform curve: P(D <= 75 | D > 50) = 0.5 over 25s horizon', () => {
    // Uniform [0, 100]: conditioning on D > 50 leaves uniform on (50, 100].
    const curve = [0, 100];
    const cond = conditionalRecovery(curve, 50);
    expect(cond).not.toBeNull();
    // recover_by_* horizons are fixed (1800s etc.) and exceed this toy curve,
    // so check the quantiles: remaining is uniform on [0, 50].
    expect(cond!.median_sec).toBe(25.0);
    expect(cond!.q25_sec).toBe(12.5);
    expect(cond!.q75_sec).toBe(37.5);
  });

  test('elapsed=0 reduces to the unconditional distribution', () => {
    const curve = [0, 3600];
    const cond = conditionalRecovery(curve, 0);
    expect(cond).not.toBeNull();
    expect(cond!.median_sec).toBe(1800);
    expect(cond!.recover_by_30).toBe(0.5);
    expect(cond!.recover_by_60).toBe(1.0);
  });

  test('outliving every observed dwell returns null', () => {
    const curve = [0, 100];
    expect(conditionalRecovery(curve, 100)).toBeNull();
    expect(conditionalRecovery(curve, 5000)).toBeNull();
    expect(conditionalRecovery(Array(21).fill(600), 600)).toBeNull();
  });

  test('recovery probability decays with elapsed time for heavy tails', () => {
    // Same heavy-tailed sample as the Python test: 20 dwells, nearest-rank
    // quantiles at 5% steps (matches training/dwell.py _quantile).
    const dwells = [
      ...Array(6).fill(300),
      ...Array(6).fill(600),
      ...Array(4).fill(1200),
      ...Array(2).fill(14400),
      ...Array(2).fill(43200),
    ].sort((a, b) => a - b);
    const curve = Array.from({ length: 21 }, (_, i) => {
      const idx = Math.max(0, Math.min(dwells.length - 1, Math.floor((i / 20) * dwells.length)));
      return dwells[idx]!;
    });

    const fresh = conditionalRecovery(curve, 0)!.recover_by_30;
    const aged1h = conditionalRecovery(curve, 3600)!.recover_by_30;
    const aged5h = conditionalRecovery(curve, 18000)!.recover_by_30;
    expect(fresh).toBeGreaterThan(aged1h);
    expect(aged1h).toBeGreaterThan(aged5h);
  });

  test('degenerate curve (fewer than 2 points) returns null', () => {
    expect(conditionalRecovery([], 0)).toBeNull();
    expect(conditionalRecovery([100], 0)).toBeNull();
  });
});

describe('pLeaveBy', () => {
  test('inside the curve it matches conditionalRecovery', () => {
    const curve = [0, 100];
    expect(pLeaveBy(curve, 50, 25)).toBe(0.5);
    expect(pLeaveBy(curve, 0, 25)).toBe(0.25);
  });

  test('extrapolates past the curve where conditionalRecovery gives up', () => {
    const curve = [0, 100];
    expect(conditionalRecovery(curve, 100)).toBeNull();
    const short = pLeaveBy(curve, 100, 600);
    const long = pLeaveBy(curve, 100, 3600);
    expect(short).toBeGreaterThan(0);
    expect(long).toBeGreaterThan(short);
    expect(long).toBeLessThan(1);
  });

  test('degenerate curve (fewer than 2 points) is zero', () => {
    expect(pLeaveBy([], 0, 1800)).toBe(0);
    expect(pLeaveBy([100], 0, 1800)).toBe(0);
  });

  test('log-logistic tail is used only past the curve, not in the body', () => {
    const curve = [0, 100];
    const tail: [number, number] = [2, 100];
    // Inside the curve the splice is inert — body stays empirical.
    expect(pLeaveBy(curve, 50, 25, tail)).toBe(pLeaveBy(curve, 50, 25));
    // Past the curve the tail's decreasing hazard is *less* eager to leave than
    // the constant-hazard exponential patch — a long-calm regime stays confident.
    const ll = pLeaveBy(curve, 100, 3600, tail);
    const exp = pLeaveBy(curve, 100, 3600);
    expect(ll).toBeGreaterThan(0);
    expect(ll).toBeLessThan(exp);
  });

  test('log-logistic tail matches the conditional-survival formula past the curve', () => {
    const curve = [0, 100];
    const [shape, scale] = [1.5, 200];
    const sNow = 1 / (1 + (100 / scale) ** shape);
    const sFut = 1 / (1 + (700 / scale) ** shape);
    expect(pLeaveBy(curve, 100, 600, [shape, scale])).toBeCloseTo(1 - sFut / sNow, 12);
  });
});

describe('dwellCdf lower-boundary fix', () => {
  test('strictly increasing curve matches the pre-fix formula exactly', () => {
    // Oracle: the OLD dwellCdf formula (`x <= curve[0] -> 0`, first-interval
    // match) written inline so this regresses if the fix ever drifts off a
    // strictly increasing curve — the fix must be a no-op there.
    const oldDwellCdf = (curveSec: number[], x: number): number => {
      const k = curveSec.length;
      if (x >= curveSec[k - 1]!) return 1.0;
      if (x <= curveSec[0]!) return 0.0;
      for (let i = 0; i < k - 1; i++) {
        const lo = curveSec[i]!;
        const hi = curveSec[i + 1]!;
        if (lo <= x && x <= hi) {
          const frac = hi === lo ? 0.0 : (x - lo) / (hi - lo);
          return (i + frac) / (k - 1);
        }
      }
      return 1.0;
    };
    const curve = Array.from({ length: 21 }, (_, i) => Math.round(120 * 1.35 ** i));
    const xs = [-10, curve[0]!, curve[0]! + 1, 5000, curve[10]!, curve[10]! + 50, curve[20]!, curve[20]! + 100];
    for (const x of xs) {
      expect(dwellCdf(curve, x)).toBeCloseTo(oldDwellCdf(curve, x), 12);
    }
  });

  test('repeated left knot returns the top of the flat run, not 0', () => {
    // A mixture cell's curve repeats atom_sec across every knot below its
    // cumulative share — e.g. most episodes at one tick means the first knots
    // are all 300. The old `x <= curve[0] -> 0` guard zeroed the CDF at
    // exactly that tick (the PIT=0 defect); the fix returns the CDF at the
    // TOP of the flat run instead.
    const curve = [300, 300, 300, 300, 300, 600, 900, 1200];
    expect(dwellCdf(curve, 300)).toBeCloseTo(4 / 7, 12);
    // Strictly below the flat run is still 0.
    expect(dwellCdf(curve, 299)).toBe(0.0);
  });
});

describe('mixtureSurvival / mixtureQuantile', () => {
  const shape = 1.8;
  const scale = 900;
  const atomP = 0.704;
  const atomSec = 300;

  test('F(atomSec) equals atomP — the atom is inclusive at its own location', () => {
    const s = mixtureSurvival(atomSec, shape, scale, atomP, atomSec);
    expect(1 - s).toBeCloseTo(atomP, 9);
    // Below the atom, survival is still 1 (F = 0): the mixture hasn't started.
    expect(mixtureSurvival(atomSec - 1, shape, scale, atomP, atomSec)).toBe(1.0);
  });

  test('mixtureQuantile inverts the atom boundary at u === atomP', () => {
    expect(mixtureQuantile(atomP, shape, scale, atomP, atomSec)).toBe(atomSec);
    expect(mixtureQuantile(atomP + 1e-6, shape, scale, atomP, atomSec)).toBeGreaterThan(atomSec);
  });

  test('pLeaveBy with atom reduces to the plain log-logistic conditional once elapsed >= atomSec', () => {
    // Sanity property from the contract: past the atom, atomP cancels out of
    // the ratio and the mixture is indistinguishable from the unconditional
    // log-logistic tail. curveSec is passed empty to also prove it is never
    // read on the atom path.
    const elapsed = 600;
    const horizon = 1800;
    const sLL = (t: number): number => 1 / (1 + (t / scale) ** shape);
    const expected = 1 - sLL(elapsed + horizon) / sLL(elapsed);
    const actual = pLeaveBy([], elapsed, horizon, [shape, scale], { p: atomP, sec: atomSec });
    expect(actual).toBeCloseTo(expected, 9);
  });

  test('conditionalRecovery median lands exactly on the atom when atom_p is the majority', () => {
    const cond = conditionalRecovery([], 0, [shape, scale], { p: atomP, sec: atomSec });
    expect(cond).not.toBeNull();
    expect(cond!.median_sec).toBe(atomSec);
  });
});

describe('legacy path is unaffected when atom is absent', () => {
  test('pLeaveBy: an explicit atom=undefined matches the pre-mixture call exactly', () => {
    const curve = [0, 100];
    const tail: [number, number] = [1.5, 200];
    expect(pLeaveBy(curve, 100, 600, tail, undefined)).toBe(pLeaveBy(curve, 100, 600, tail));
  });

  test('conditionalRecovery: explicit tailLl/atom=undefined matches the pre-mixture 2-arg call exactly', () => {
    const curve = [0, 100];
    const withExtraArgs = conditionalRecovery(curve, 50, undefined, undefined);
    const legacy = conditionalRecovery(curve, 50);
    expect(withExtraArgs).toEqual(legacy);
  });
});
