import { describe, expect, test } from 'vitest';

import type { RouteRoll } from '../src/alpha';
import { detectTransitions, movementTransitions } from '../src/grading';
import type { RegimeChange } from '../src/regime';

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

describe('movementTransitions', () => {
  const change: RegimeChange = {
    key: 'A',
    prev_state: 'normal',
    new_state: 'disrupted',
    entered_at: 1_700_000_000,
    exited_at: 1_700_000_600,
    dwell_sec: 600,
  };

  test('route scope uses the key as the route', () => {
    expect(movementTransitions([change], 'route', 1_700_000_900)).toEqual([
      {
        ts: 1_700_000_900,
        scope: 'route',
        key: 'A',
        route: 'A',
        prev_state: 'normal',
        new_state: 'disrupted',
        regime_entered_at: 1_700_000_000,
        exited_at: 1_700_000_600,
        dwell_sec: 600,
      },
    ]);
  });

  test('segment scope takes the route from the first key field', () => {
    const seg = { ...change, key: 'Q|north|Q05N' };
    const [out] = movementTransitions([seg], 'segment', 1_700_000_900);
    expect(out!.route).toBe('Q');
    expect(out!.key).toBe('Q|north|Q05N');
    expect(out!.scope).toBe('segment');
  });

  test('carries the clock through unchanged so both streams grade alike', () => {
    const [out] = movementTransitions([change], 'route', 1_700_000_900);
    expect(out!.regime_entered_at).toBe(change.entered_at);
    expect(out!.dwell_sec).toBe(change.exited_at - change.entered_at);
  });
});
