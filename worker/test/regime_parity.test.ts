/**
 * Cross-language parity: the TS regime clock must reproduce, tick for tick,
 * the entries and changes Python recorded in tests/fixtures/parity_regime.json.
 *
 * Drift here is expensive in a specific way. The trainer fits dwell curves on
 * regimes it segmented offline; the Worker projects those curves over regimes
 * it segmented online. Disagree on where a regime starts and the curve
 * describes an episode that never happened.
 *
 * Regenerate the fixture with:
 *   uv run python -m scripts.gen_regime_parity_fixture
 */

import { describe, expect, test } from 'vitest';

import fixture from '../../tests/fixtures/parity_regime.json';
import { advanceRegimes } from '../src/regime';
import type { RegimeEntry } from '../src/regime';

describe('regime clock parity with training/regime.py', () => {
  test('reproduces every recorded tick', () => {
    let entries: Record<string, RegimeEntry> = {};
    for (const step of fixture.steps) {
      const advanced = advanceRegimes(entries, step.observed, step.observed_at, {
        debounceTicks: fixture.debounce_ticks,
        maxIdleSec: fixture.max_idle_sec,
      });
      entries = advanced.entries;
      // Sorted so the comparison does not depend on insertion order.
      expect(Object.fromEntries(Object.entries(entries).sort(([a], [b]) => (a < b ? -1 : 1))))
        .toEqual(step.entries);
      expect(advanced.changes).toEqual(step.changes);
    }
    expect(Object.fromEntries(Object.entries(entries).sort(([a], [b]) => (a < b ? -1 : 1))))
      .toEqual(fixture.final_entries);
  });
});
