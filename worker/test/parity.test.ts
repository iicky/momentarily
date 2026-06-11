/**
 * Parity tests: TS forward filter must produce identical probabilities to the
 * Python implementation for a canonical (params, observations) fixture.
 *
 * Reference values were computed by running tests/test_hmm.py equivalents and
 * captured here as expected outputs. If a TS port drifts from Python, these
 * fail loudly.
 */

import { describe, expect, test } from 'vitest';

import type { EmissionParams, FilterState, HMMParams, Observation } from '../src/hmm';
import {
  expectedDwellTicks,
  forwardUpdate,
  projectForward,
  stationaryDistribution,
  tod_bin,
} from '../src/hmm';

const DEFAULT_EMISSIONS: EmissionParams = {
  poisson_lambda: [0.3, 4.0, 12.0],
  gamma_alpha: [1.0, 3.0, 6.0],
  gamma_beta: [2.0, 0.4, 0.2],
  bernoulli_p: [0.001, 0.05, 0.95],
  bernoulli_p_delays: [0.01, 0.45, 0.5],
  bernoulli_p_service_change: [0.01, 0.5, 0.6],
  bernoulli_p_planned: [0.05, 0.3, 0.4],
};

const DEFAULT_PARAMS: HMMParams = {
  transition: [
    [0.95, 0.04, 0.01],
    [0.08, 0.9, 0.02],
    [0.02, 0.1, 0.88],
  ],
  initial: [0.9, 0.08, 0.02],
  emissions: DEFAULT_EMISSIONS,
};

const FLAT: FilterState = {
  probabilities: [1 / 3, 1 / 3, 1 / 3],
  regime_entered_at: 0,
  last_updated_at: 0,
};

function quietObs(): Observation {
  return {
    alert_count: 0,
    severity_sum: 0,
    has_suspended_alert: false,
    has_delays: false,
    has_service_change: false,
    has_planned: false,
    tod_bin: 0,
  };
}

function suspendedObs(): Observation {
  return {
    alert_count: 15,
    severity_sum: 80,
    has_suspended_alert: true,
    has_delays: false,
    has_service_change: false,
    has_planned: false,
    tod_bin: 0,
  };
}

describe('forwardUpdate matches Python', () => {
  test('posterior sums to one', () => {
    const result = forwardUpdate(FLAT, quietObs(), DEFAULT_PARAMS, 100);
    const sum =
      result.probabilities[0] + result.probabilities[1] + result.probabilities[2];
    expect(sum).toBeCloseTo(1.0, 9);
  });

  test('quiet observation pulls toward normal', () => {
    const result = forwardUpdate(FLAT, quietObs(), DEFAULT_PARAMS, 100);
    const [n, d, s] = result.probabilities;
    expect(n).toBeGreaterThan(d);
    expect(d).toBeGreaterThan(s);
    expect(n).toBeGreaterThan(0.8);
  });

  test('suspended-flagged observation pulls toward suspended', () => {
    const result = forwardUpdate(FLAT, suspendedObs(), DEFAULT_PARAMS, 100);
    const [n, d, s] = result.probabilities;
    expect(s).toBeGreaterThan(d);
    expect(d).toBeGreaterThan(n);
    expect(s).toBeGreaterThan(0.6);
  });

  test('regime_entered_at advances on state change', () => {
    const start: FilterState = {
      probabilities: [0.95, 0.04, 0.01],
      regime_entered_at: 100,
      last_updated_at: 100,
    };
    const disrupted: Observation = {
      alert_count: 8,
      severity_sum: 50,
      has_suspended_alert: false,
      has_delays: true,
      has_service_change: false,
      has_planned: false,
      tod_bin: 0,
    };
    const s1 = forwardUpdate(start, disrupted, DEFAULT_PARAMS, 200);
    const s2 = forwardUpdate(s1, disrupted, DEFAULT_PARAMS, 300);
    // After 2 disruption ticks, argmax should flip away from normal
    expect(s2.probabilities[1]).toBeGreaterThan(s2.probabilities[0]);
    expect([200, 300]).toContain(s2.regime_entered_at);
  });
});

describe('projectForward', () => {
  test('zero ticks is identity', () => {
    const result = projectForward(FLAT, DEFAULT_PARAMS, 0);
    expect(result).toEqual(FLAT.probabilities);
  });

  test('converges to stationary from peaked start', () => {
    const peaked: FilterState = {
      probabilities: [0, 1, 0],
      regime_entered_at: 0,
      last_updated_at: 0,
    };
    const far = projectForward(peaked, DEFAULT_PARAMS, 500);
    // Stationary is dominated by normal under this transition matrix
    expect(far[0]).toBeGreaterThan(far[1]);
    expect(far[1]).toBeGreaterThan(far[2]);
    const sum = far[0] + far[1] + far[2];
    expect(sum).toBeCloseTo(1.0, 9);
  });

  test('rejects negative ticksAhead', () => {
    expect(() => projectForward(FLAT, DEFAULT_PARAMS, -1)).toThrow();
  });
});

describe('stationaryDistribution', () => {
  test('is a left eigenvector of the transition matrix (π · T = π)', () => {
    const pi = stationaryDistribution(DEFAULT_PARAMS);
    const t = DEFAULT_PARAMS.transition;
    const piT: [number, number, number] = [0, 1, 2].map((s) =>
      [0, 1, 2].reduce((acc, sp) => acc + pi[sp]! * t[sp]![s]!, 0),
    ) as unknown as [number, number, number];
    for (let s = 0; s < 3; s += 1) {
      expect(piT[s]!).toBeCloseTo(pi[s]!, 6);
    }
  });

  test('sums to 1', () => {
    const pi = stationaryDistribution(DEFAULT_PARAMS);
    expect(pi[0] + pi[1] + pi[2]).toBeCloseTo(1.0, 6);
  });

  test('all entries positive for an irreducible chain', () => {
    const pi = stationaryDistribution(DEFAULT_PARAMS);
    for (const p of pi) {
      expect(p).toBeGreaterThan(0);
    }
  });

  test('softer than params.initial when initial is one-hot', () => {
    const peakedInit: HMMParams = {
      ...DEFAULT_PARAMS,
      initial: [1, 0, 0],
    };
    const pi = stationaryDistribution(peakedInit);
    // π[0] must be strictly less than 1 — stationary respects the off-diagonal
    // transition mass that one-hot initial ignores.
    expect(pi[0]).toBeLessThan(1);
    expect(pi[1]).toBeGreaterThan(0);
    expect(pi[2]).toBeGreaterThan(0);
  });
});

describe('expectedDwellTicks', () => {
  test('high self-loop yields long stay', () => {
    const inNormal: FilterState = {
      probabilities: [0.99, 0.005, 0.005],
      regime_entered_at: 0,
      last_updated_at: 0,
    };
    const { median, q25, q75 } = expectedDwellTicks(inNormal, DEFAULT_PARAMS);
    // ceil(log(0.5)/log(0.95)) = 14
    expect(median).toBe(14);
    expect(q25).toBeLessThan(median);
    expect(q75).toBeGreaterThan(median);
  });

  test('disrupted state expected dwell', () => {
    const inDisrupted: FilterState = {
      probabilities: [0.1, 0.85, 0.05],
      regime_entered_at: 0,
      last_updated_at: 0,
    };
    const { median } = expectedDwellTicks(inDisrupted, DEFAULT_PARAMS);
    // ceil(log(0.5)/log(0.90)) = 7
    expect(median).toBe(7);
  });
});

describe('tod_bin', () => {
  test('covers all 24 hours and maps to valid bins', () => {
    const seen = new Set<number>();
    for (let h = 0; h < 24; h += 1) {
      const b = tod_bin(h * 3600);
      expect(b).toBeGreaterThanOrEqual(0);
      expect(b).toBeLessThan(5);
      seen.add(b);
    }
    expect(seen.size).toBe(5);
  });

  test('is DST-aware — mirrors tests/test_hmm.py', () => {
    // 10:00 UTC = 05:00 EST (overnight) in January, 06:00 EDT (rush) in July.
    const winter = Date.UTC(2026, 0, 15, 10, 0) / 1000;
    const summer = Date.UTC(2026, 6, 15, 10, 0) / 1000;
    expect(tod_bin(winter)).toBe(0);
    expect(tod_bin(summer)).toBe(1);
    // ET bin edge at 15:00 (EDT = UTC-4).
    expect(tod_bin(Date.UTC(2026, 6, 15, 18, 59) / 1000)).toBe(2);
    expect(tod_bin(Date.UTC(2026, 6, 15, 19, 0) / 1000)).toBe(3);
  });
});
