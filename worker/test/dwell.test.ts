/**
 * Conditional survival math over the empirical dwell curve — momentarily-vk0.1.
 *
 * Mirrors tests/test_dwell.py (Python reference implementation); the fixtures
 * and expected values are intentionally identical so the two implementations
 * can't drift apart silently.
 */

import { describe, expect, test } from 'vitest';

import { conditionalRecovery, dwellCdf } from '../src/dwell';

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
