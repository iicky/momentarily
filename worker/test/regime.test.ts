/**
 * Regime clock: debounce, back-dating, abstention, and eviction.
 *
 * The cross-language pin against training/regime.py lives in
 * regime_parity.test.ts.
 */

import { describe, expect, test } from 'vitest';

import { DEBOUNCE_TICKS, MAX_IDLE_SEC, advanceRegimes } from '../src/regime';
import type { RegimeEntry } from '../src/regime';

const TICK = 300;
const T0 = 1_700_000_000;
const t = (i: number) => T0 + i * TICK;

function entry(state: string, enteredAt: number, lastSeenAt = enteredAt): RegimeEntry {
  return {
    state,
    entered_at: enteredAt,
    last_seen_at: lastSeenAt,
    pending: null,
    pending_since: 0,
    pending_run: 0,
  };
}

// The debounce mechanism is exercised at debounceTicks: 2 explicitly. The
// production default is 1 (see DEBOUNCE_TICKS) — these pin the machinery, not
// the shipped setting.
const D2 = { debounceTicks: 2 };

describe('advanceRegimes', () => {
  test('cold start opens a regime without debounce', () => {
    const { entries, changes } = advanceRegimes(null, { A: 'disrupted' }, t(0));
    expect(changes).toEqual([]);
    expect(entries.A).toEqual(entry('disrupted', t(0)));
  });

  test('single-tick blip does not commit', () => {
    const { entries } = advanceRegimes(null, { A: 'normal' }, t(0), D2);
    const step = advanceRegimes(entries, { A: 'disrupted' }, t(1), D2);
    expect(step.changes).toEqual([]);
    expect(step.entries.A!.state).toBe('normal');
    expect(step.entries.A!.entered_at).toBe(t(0));
    expect(step.entries.A!.pending).toBe('disrupted');
  });

  test('commit back-dates to the first tick of the run', () => {
    let entries: Record<string, RegimeEntry> = advanceRegimes(null, { A: 'normal' }, t(0), D2).entries;
    entries = advanceRegimes(entries, { A: 'disrupted' }, t(1), D2).entries;
    const step = advanceRegimes(entries, { A: 'disrupted' }, t(2), D2);
    expect(step.entries.A!.state).toBe('disrupted');
    expect(step.entries.A!.entered_at).toBe(t(1));
    expect(step.changes).toEqual([
      {
        key: 'A',
        prev_state: 'normal',
        new_state: 'disrupted',
        entered_at: t(0),
        exited_at: t(1),
        dwell_sec: TICK,
      },
    ]);
  });

  test('interrupted run restarts the debounce', () => {
    let entries: Record<string, RegimeEntry> = advanceRegimes(null, { A: 'normal' }, t(0), D2).entries;
    entries = advanceRegimes(entries, { A: 'disrupted' }, t(1), D2).entries;
    entries = advanceRegimes(entries, { A: 'normal' }, t(2), D2).entries;
    const step = advanceRegimes(entries, { A: 'disrupted' }, t(3), D2);
    expect(step.changes).toEqual([]);
    expect(step.entries.A!.state).toBe('normal');
    expect(step.entries.A!.pending_run).toBe(1);
  });

  test('abstention resets the run but holds the regime open', () => {
    let entries: Record<string, RegimeEntry> = advanceRegimes(null, { A: 'normal' }, t(0), D2).entries;
    entries = advanceRegimes(entries, { A: 'disrupted' }, t(1), D2).entries;
    const held = advanceRegimes(entries, {}, t(2), D2);
    expect(held.changes).toEqual([]);
    expect(held.entries.A!.state).toBe('normal');
    expect(held.entries.A!.entered_at).toBe(t(0));
    expect(held.entries.A!.pending).toBeNull();

    const again = advanceRegimes(held.entries, { A: 'disrupted' }, t(3), D2);
    expect(again.changes).toEqual([]);
    const committed = advanceRegimes(again.entries, { A: 'disrupted' }, t(4), D2);
    expect(committed.changes.map((c) => c.exited_at)).toEqual([t(3)]);
  });

  test('cell evicts after max idle', () => {
    const entries = advanceRegimes(null, { A: 'normal' }, t(0)).entries;
    const step = advanceRegimes(entries, {}, t(0) + MAX_IDLE_SEC + 1);
    expect(step.entries).toEqual({});
    expect(step.changes).toEqual([]);
  });

  test('regime survives an abstention shorter than max idle', () => {
    const entries = advanceRegimes(null, { A: 'normal' }, t(0)).entries;
    const step = advanceRegimes(entries, {}, t(0) + MAX_IDLE_SEC - 1);
    expect(step.entries.A!.entered_at).toBe(t(0));
  });

  test('debounce of one commits immediately', () => {
    const entries = advanceRegimes(
      null,
      { A: 'normal' },
      t(0),
      { debounceTicks: 1 },
    ).entries;
    const step = advanceRegimes(entries, { A: 'disrupted' }, t(1), { debounceTicks: 1 });
    expect(step.entries.A!.state).toBe('disrupted');
    expect(step.entries.A!.entered_at).toBe(t(1));
    expect(step.changes.map((c) => c.dwell_sec)).toEqual([TICK]);
  });

  test('changes are ordered by key', () => {
    const prev: Record<string, RegimeEntry> = {
      C: entry('normal', t(0)),
      A: entry('normal', t(0)),
      B: entry('normal', t(0)),
    };
    const { changes } = advanceRegimes(
      prev,
      { C: 'disrupted', A: 'disrupted', B: 'disrupted' },
      t(1),
      { debounceTicks: 1 },
    );
    expect(changes.map((c) => c.key)).toEqual(['A', 'B', 'C']);
  });

  test('clock is key-agnostic across route and segment scope', () => {
    const route = 'A';
    const segment = 'Q|north|Q05N';
    let entries: Record<string, RegimeEntry> = advanceRegimes(
      null,
      { [route]: 'normal', [segment]: 'normal' },
      t(0),
      D2,
    ).entries;
    entries = advanceRegimes(
      entries,
      { [route]: 'disrupted', [segment]: 'disrupted' },
      t(1),
      D2,
    ).entries;
    const step = advanceRegimes(
      entries,
      { [route]: 'disrupted', [segment]: 'disrupted' },
      t(2),
      D2,
    );
    expect(step.entries[route]!.entered_at).toBe(t(1));
    expect(step.entries[segment]!.entered_at).toBe(t(1));
    expect(step.changes.map((c) => c.dwell_sec)).toEqual([TICK, TICK]);
  });

  test('shipped debounce commits on the first change', () => {
    // Raising this to 2 measurably erased 76% of the route episode population.
    expect(DEBOUNCE_TICKS).toBe(1);
    const entries = advanceRegimes(null, { A: 'normal' }, t(0)).entries;
    const step = advanceRegimes(entries, { A: 'disrupted' }, t(1));
    expect(step.entries.A!.state).toBe('disrupted');
    expect(step.entries.A!.entered_at).toBe(t(1));
    expect(step.changes.map((c) => c.dwell_sec)).toEqual([TICK]);
  });

  test('abstention holds the regime open at the shipped default', () => {
    const entries = advanceRegimes(null, { A: 'normal' }, t(0)).entries;
    const step = advanceRegimes(entries, {}, t(1));
    expect(step.changes).toEqual([]);
    expect(step.entries.A!.state).toBe('normal');
    expect(step.entries.A!.entered_at).toBe(t(0));
  });
});
