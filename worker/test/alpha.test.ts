/**
 * Reseeding across a params swap: posteriors are replaced (they were filtered
 * under old emissions) but the regime clock and cause carry over, so recovery
 * predictions that condition on regime age survive a retrain.
 */

import { describe, expect, test } from 'vitest';

import type { RouteRoll } from '../src/alpha';
import { reseedForNewParams } from '../src/alpha';
import type { EmissionParams, HMMParams, Observation } from '../src/hmm';
import { forwardStep } from '../src/hmm';

const EMISSIONS: EmissionParams = {
  poisson_lambda: [0.3, 4.0, 12.0],
  gamma_alpha: [1.0, 3.0, 6.0],
  gamma_beta: [2.0, 0.4, 0.2],
  bernoulli_p: [0.001, 0.05, 0.95],
  bernoulli_p_delays: [0.01, 0.45, 0.5],
  bernoulli_p_service_change: [0.01, 0.5, 0.6],
  bernoulli_p_planned: [0.05, 0.3, 0.4],
};

const PARAMS: HMMParams = {
  transition: [
    [0.95, 0.04, 0.01],
    [0.08, 0.9, 0.02],
    [0.02, 0.1, 0.88],
  ],
  initial: [0.9, 0.08, 0.02],
  emissions: EMISSIONS,
};

const REGIME_START = 1_700_000_000;

function disruptedRoll(): RouteRoll {
  return {
    filter: {
      probabilities: [0.02, 0.97, 0.01],
      regime_entered_at: REGIME_START,
      last_updated_at: REGIME_START + 36_000,
    },
    published: {
      label: 'disrupted',
      pending_state: 'disrupted',
      pending_streak: 5,
      last_updated_at: REGIME_START + 36_000,
    },
    alert_type_at_entry: 'Delays',
  };
}

function delaysObs(): Observation {
  return {
    alert_count: 4,
    severity_sum: 25,
    has_suspended_alert: false,
    has_delays: true,
    has_service_change: false,
    has_planned: false,
    tod_bin: 0,
  };
}

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

describe('reseedForNewParams', () => {
  test('preserves the regime clock, cause, and published state', () => {
    const reseeded = reseedForNewParams(disruptedRoll());
    expect(reseeded.filter.regime_entered_at).toBe(REGIME_START);
    expect(reseeded.alert_type_at_entry).toBe('Delays');
    expect(reseeded.published.label).toBe('disrupted');
  });

  test('softens the posterior onto the old argmax', () => {
    const reseeded = reseedForNewParams(disruptedRoll());
    const p = reseeded.filter.probabilities;
    expect(p[1]).toBeCloseTo(0.7, 10);
    expect(p[0]).toBeCloseTo(0.15, 10);
    expect(p[2]).toBeCloseTo(0.15, 10);
    expect(p[0] + p[1] + p[2]).toBeCloseTo(1.0, 10);
  });

  test('regime age survives a params swap while the disruption continues', () => {
    const reseeded = reseedForNewParams(disruptedRoll());
    const now = REGIME_START + 36_300;
    const { state } = forwardStep(
      reseeded.filter,
      reseeded.published,
      delaysObs(),
      PARAMS,
      now,
    );
    // Still disrupted under the new params → the 10-hour-old clock holds.
    expect(state.probabilities[1]).toBeGreaterThan(state.probabilities[0]);
    expect(state.regime_entered_at).toBe(REGIME_START);
  });

  test('clock still resets honestly when the regime actually changed', () => {
    const reseeded = reseedForNewParams(disruptedRoll());
    const now = REGIME_START + 36_300;
    const { state } = forwardStep(
      reseeded.filter,
      reseeded.published,
      quietObs(),
      PARAMS,
      now,
    );
    // Alerts cleared → normal wins and the regime clock advances to now.
    expect(state.probabilities[0]).toBeGreaterThan(state.probabilities[1]);
    expect(state.regime_entered_at).toBe(now);
  });
});
