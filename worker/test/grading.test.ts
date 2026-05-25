import { describe, expect, test } from 'vitest';

import type { RouteRoll } from '../src/alpha';
import { detectTransitions } from '../src/grading';

function roll(
  probs: [number, number, number],
  regimeEnteredAt: number,
  alertTypeAtEntry: string | null = null,
): RouteRoll {
  return {
    filter: {
      probabilities: probs,
      regime_entered_at: regimeEnteredAt,
      last_updated_at: regimeEnteredAt,
    },
    published: {
      label: 'normal',
      pending_state: 'normal',
      pending_streak: 2,
      last_updated_at: regimeEnteredAt,
    },
    alert_type_at_entry: alertTypeAtEntry,
  };
}

describe('detectTransitions threads alert_type_at_entry from prev regime', () => {
  test('emits prev alert_type when regime ended', () => {
    const prev = { '1': roll([0.05, 0.94, 0.01], 1_700_000_000, 'Delays') };
    const next = { '1': roll([0.95, 0.04, 0.01], 1_700_000_300) };
    const out = detectTransitions(prev, next, 1_700_000_300);
    expect(out).toHaveLength(1);
    expect(out[0]!.alert_type_at_entry).toBe('Delays');
    expect(out[0]!.prev_state).toBe('disrupted');
    expect(out[0]!.new_state).toBe('normal');
    expect(out[0]!.dwell_sec).toBe(300);
  });

  test('emits null when no alert was active at regime start', () => {
    const prev = { '1': roll([0.05, 0.94, 0.01], 1_700_000_000, null) };
    const next = { '1': roll([0.95, 0.04, 0.01], 1_700_000_300) };
    const out = detectTransitions(prev, next, 1_700_000_300);
    expect(out[0]!.alert_type_at_entry).toBeNull();
  });

  test('no transition emitted when regime persists', () => {
    const prev = { '1': roll([0.05, 0.94, 0.01], 1_700_000_000, 'Delays') };
    const next = { '1': roll([0.10, 0.89, 0.01], 1_700_000_000, 'Delays') };
    const out = detectTransitions(prev, next, 1_700_000_300);
    expect(out).toHaveLength(0);
  });
});
