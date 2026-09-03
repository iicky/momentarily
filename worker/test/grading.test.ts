import { describe, expect, test } from 'vitest';

import type { RouteRoll } from '../src/alpha';
import {
  detectTransitions,
  movementTransitions,
  writeMovementTransitions,
} from '../src/grading';
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

/**
 * Regression: the two per-tick writes must not overwrite each other.
 *
 * index.ts commits the route clock and then the segment clock with the same
 * observedAt. The key used to be `<ts>.jsonl` for both, and R2 put replaces,
 * so every tick with a segment change destroyed that tick's route records.
 * Route scope was empty for eight days once segment volume rose ~60x
 * (journal.md 2026-09-03).
 */
describe('writeMovementTransitions: both scopes survive one tick', () => {
  const observedAt = 1_700_000_900;

  function change(key: string): RegimeChange {
    return {
      key,
      prev_state: 'normal',
      new_state: 'disrupted',
      entered_at: 1_700_000_000,
      exited_at: 1_700_000_600,
      dwell_sec: 600,
    };
  }

  const routeRecords = movementTransitions([change('A')], 'route', observedAt);
  const segmentRecords = movementTransitions(
    [change('Q|north|Q05N')],
    'segment',
    observedAt,
  );

  // Minimal in-memory R2 — same convention as archive.test.ts. Deliberately a
  // replacing Map, because that is exactly what R2 put does.
  function fakeBucket() {
    const store = new Map<string, string>();
    return {
      bucket: {
        async put(key: string, body: string) {
          store.set(key, body);
          return {} as unknown;
        },
      } as unknown as R2Bucket,
      store,
    };
  }

  // Narrow rather than assert a shape onto JSON.parse: the point of the test is
  // that the archived line really carries a scope, so a missing one must fail.
  function scopeOf(line: string): string {
    const parsed: unknown = JSON.parse(line);
    if (
      parsed === null
      || typeof parsed !== 'object'
      || !('scope' in parsed)
      || typeof parsed.scope !== 'string'
    ) {
      throw new Error(`archived line carries no scope: ${line}`);
    }
    return parsed.scope;
  }

  function scopesIn(store: Map<string, string>): string[] {
    return [...store.values()]
      .flatMap((body) => body.split('\n'))
      .filter((line) => line.trim() !== '')
      .map(scopeOf);
  }

  test('the old unscoped key scheme loses the route write — the bug', async () => {
    const { bucket, store } = fakeBucket();
    // The pre-fix key, reproduced verbatim so the test fails for the same
    // reason production did rather than by construction of the new code.
    const oldKey = (ts: number) => `v1/movement_transitions/2023-11-14/${ts}.jsonl`;
    const put = async (records: typeof routeRecords) => {
      await bucket.put(oldKey(observedAt), records.map((r) => JSON.stringify(r)).join('\n'));
    };
    await put(routeRecords);
    await put(segmentRecords);

    expect(store.size).toBe(1);
    expect(scopesIn(store)).toEqual(['segment']);
  });

  test('the scoped key keeps both writes, in separate objects', async () => {
    const { bucket, store } = fakeBucket();
    await writeMovementTransitions(bucket, observedAt, routeRecords);
    await writeMovementTransitions(bucket, observedAt, segmentRecords);

    expect([...store.keys()].sort()).toEqual([
      `v1/movement_transitions/2023-11-14/${observedAt}-route.jsonl`,
      `v1/movement_transitions/2023-11-14/${observedAt}-segment.jsonl`,
    ]);
    expect(scopesIn(store).sort()).toEqual(['route', 'segment']);
  });

  test('write order does not matter — neither scope can displace the other', async () => {
    const { bucket, store } = fakeBucket();
    await writeMovementTransitions(bucket, observedAt, segmentRecords);
    await writeMovementTransitions(bucket, observedAt, routeRecords);
    expect(scopesIn(store).sort()).toEqual(['route', 'segment']);
  });

  test('a mixed-scope batch partitions rather than mislabelling one key', async () => {
    const { bucket, store } = fakeBucket();
    await writeMovementTransitions(bucket, observedAt, [
      ...routeRecords,
      ...segmentRecords,
    ]);

    expect([...store.keys()].sort()).toEqual([
      `v1/movement_transitions/2023-11-14/${observedAt}-route.jsonl`,
      `v1/movement_transitions/2023-11-14/${observedAt}-segment.jsonl`,
    ]);
    // Each object holds only its own scope, so the key never lies about its
    // contents even though the caller batched.
    for (const [key, body] of store) {
      const scope = key.endsWith('-route.jsonl') ? 'route' : 'segment';
      for (const line of body.split('\n')) expect(scopeOf(line)).toBe(scope);
    }
  });

  test('both keys stay under the date prefix the Python loader lists', async () => {
    const { bucket, store } = fakeBucket();
    await writeMovementTransitions(bucket, observedAt, routeRecords);
    await writeMovementTransitions(bucket, observedAt, segmentRecords);
    for (const key of store.keys()) {
      expect(key.startsWith('v1/movement_transitions/2023-11-14/')).toBe(true);
    }
  });

  test('an empty tick still writes nothing', async () => {
    const { bucket, store } = fakeBucket();
    await writeMovementTransitions(bucket, observedAt, []);
    expect(store.size).toBe(0);
  });
});
