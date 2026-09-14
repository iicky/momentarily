/**
 * Cross-language parity: the TS severity grade must reproduce, case for case,
 * the tiers and graded states Python recorded in
 * tests/fixtures/parity_graded_state.json.
 *
 * This is load-bearing: the Worker now publishes route_status.condition as the
 * severity-graded alert read, and training/review.py grades that published
 * condition against the SAME derive_graded_mta_state rule. If severityTier /
 * deriveGradedMtaState drift from the Python definition, the Worker publishes a
 * condition the review scores as wrong.
 *
 * Regenerate the fixture with:
 *   uv run python -m scripts.gen_graded_state_parity_fixture
 */

import { describe, expect, test } from 'vitest';

import fixture from '../../tests/fixtures/parity_graded_state.json';
import { deriveGradedMtaState, severityTier } from '../src/mapping';

describe('severity grade parity with Python', () => {
  test('severityTier reproduces every recorded tier', () => {
    for (const c of fixture.severity) {
      expect(severityTier(c.alert_type)).toBe(c.tier);
    }
  });

  test('deriveGradedMtaState reproduces every recorded graded state', () => {
    for (const c of fixture.graded) {
      expect(deriveGradedMtaState(c.alert_types, c.floor)).toBe(c.expected);
    }
  });
});
