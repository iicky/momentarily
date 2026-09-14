import { describe, expect, test } from 'vitest';

import type { RouteRoll } from '../src/alpha';
import {
  buildMovementCensus,
  detectTransitions,
  movementTransitions,
  writeMovementCensus,
  writeMovementTransitions,
} from '../src/grading';
import { advanceRegimes } from '../src/regime';
import type { RegimeChange, RegimeEntry } from '../src/regime';
import { deriveMovementStates } from '../src/movement_state';
import type { TrainedParams } from '../src/params';
import type { ServiceRow } from '../src/trip_updates';

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

describe('buildMovementCensus', () => {
  const t0 = 1_700_000_000;
  const t1 = 1_700_000_300;

  function entry(state: string, entered_at: number, at: number): RegimeEntry {
    return { state, entered_at, last_seen_at: at, pending: null, pending_since: 0, pending_run: 0 };
  }

  test('abstaining feed route with no prior regime appears as unknown', () => {
    const observed: Record<string, string> = { Y: 'normal' };
    const feedKeys = new Set(['X', 'Y']);
    const entries: Record<string, RegimeEntry> = {
      Y: entry('normal', t0, t1),
    };

    const census = buildMovementCensus(observed, feedKeys, entries, t1);

    expect(census.regimes.X).toEqual({ state: 'unknown', open_state: null, open_since: null });
    expect(census.regimes.Y).toEqual({ state: 'normal', open_state: 'normal', open_since: t0 });
  });

  test('pending transition shows raw state diverging from open regime state', () => {
    const observed: Record<string, string> = { P: 'disrupted' };
    const feedKeys = new Set(['P']);
    const entries: Record<string, RegimeEntry> = {
      P: { state: 'normal', entered_at: t0, last_seen_at: t1, pending: 'disrupted', pending_since: t1, pending_run: 1 },
    };

    const census = buildMovementCensus(observed, feedKeys, entries, t1);

    expect(census.regimes.P).toEqual({ state: 'disrupted', open_state: 'normal', open_since: t0 });
  });

  test('idle-graced routes NOT in feedKeys or observed are excluded', () => {
    const observed: Record<string, string> = { A: 'normal' };
    const feedKeys = new Set(['A']);
    const entries: Record<string, RegimeEntry> = {
      A: entry('normal', t0, t1),
      Z: entry('disrupted', t0, t0),  // idle-graced, not in feeds or observed
    };

    const census = buildMovementCensus(observed, feedKeys, entries, t1);

    expect(Object.keys(census.regimes)).toEqual(['A']);
  });

  test('schedule-rate gate: above-gate feed-absent route excluded, below-gate not_scheduled included', () => {
    // Exercise the real classifier with a scheduleRate fixture so the gate
    // at NOT_SCHEDULED_MAX (0.5) is tested end-to-end, not hand-constructed.
    //
    // observedAt = 1_700_000_300 → schedule_bin = 'wd17' (Tue 17:00 ET).
    // Route NS: rate 0.1 (< 0.5) → deriveMovementStates emits not_scheduled.
    // Route ACTIVE: rate 0.9 (>= 0.5) → NOT emitted, not classified.
    // Route A: in the feed (feedKeys), classified by the movement path.
    const observedAt = 1_700_000_300;
    const trained = {
      scheduleRate: {
        NS: { wd17: 0.1 },       // below gate → not_scheduled
        ACTIVE: { wd17: 0.9 },   // above gate → feed-absent, never classified
      },
    } as unknown as TrainedParams;

    // Only route A is in the feeds; NS and ACTIVE are feed-absent.
    // A has no movement baseline so deriveMovementState returns null → abstains.
    const moveRows = new Map<string, never>();
    const svcRows = new Map([['A', { assigned_n: 5 }]]) as unknown as Map<string, ServiceRow>;

    const observed = deriveMovementStates(moveRows, svcRows, trained, observedAt);
    const { entries } = advanceRegimes(null, observed, observedAt);
    const feedKeys = new Set([...moveRows.keys(), ...svcRows.keys()]);
    const census = buildMovementCensus(observed, feedKeys, entries, observedAt);

    // NS appears via observed (universe = feedKeys ∪ keys(observed)).
    expect(census.regimes.NS).toEqual({
      state: 'not_scheduled',
      open_state: 'not_scheduled',
      open_since: observedAt,
    });
    // ACTIVE is above the gate: absent from observed AND feedKeys → excluded.
    expect(census.regimes.ACTIVE).toBeUndefined();
    // A is in feedKeys but deriveMovementState returned null → 'unknown'.
    expect(census.regimes.A?.state).toBe('unknown');
    expect(Object.keys(census.regimes).sort()).toEqual(['A', 'NS']);
  });
});

describe('writeMovementCensus: multi-tick three-state fixture', () => {
  const t0 = 1_700_000_000;   // regimes opened
  const tick1 = 1_700_000_900;
  const tick2 = 1_700_001_200;

  function entry(state: string, entered_at: number, at: number): RegimeEntry {
    return { state, entered_at, last_seen_at: at, pending: null, pending_since: 0, pending_run: 0 };
  }

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

  // Three routes in three distinct states; Q never transitions between ticks.
  // All three are feed routes (in feedKeys).
  const feedKeys = new Set(['A', 'B', 'Q']);
  const observed1: Record<string, string> = { A: 'normal', B: 'disrupted', Q: 'suspended' };
  const entries1: Record<string, RegimeEntry> = {
    A: entry('normal', t0, tick1),
    B: entry('disrupted', t0 + 300, tick1),
    Q: entry('suspended', t0 + 100, tick1),
  };

  // Tick 2: A flips, B flips, Q stays suspended — the never-transition case.
  const observed2: Record<string, string> = { A: 'disrupted', B: 'normal', Q: 'suspended' };
  const entries2: Record<string, RegimeEntry> = {
    A: entry('disrupted', tick2, tick2),
    B: entry('normal', tick2, tick2),
    Q: entry('suspended', t0 + 100, tick2),  // same entered_at — never transitioned
  };

  test('two ticks write two objects; unchanged route Q appears in both', async () => {
    const { bucket, store } = fakeBucket();

    const c1 = buildMovementCensus(observed1, feedKeys, entries1, tick1);
    const c2 = buildMovementCensus(observed2, feedKeys, entries2, tick2);
    await writeMovementCensus(bucket, c1);
    await writeMovementCensus(bucket, c2);

    expect(store.size).toBe(2);
    const keys = [...store.keys()].sort();
    expect(keys).toEqual([
      `archive/movement_census/2023-11-14/${tick1}.json`,
      `archive/movement_census/2023-11-14/${tick2}.json`,
    ]);

    const p1 = JSON.parse(store.get(keys[0]!)!);
    const p2 = JSON.parse(store.get(keys[1]!)!);

    // All three states present in tick 1.
    expect(Object.keys(p1.regimes).sort()).toEqual(['A', 'B', 'Q']);
    expect(p1.regimes.A.state).toBe('normal');
    expect(p1.regimes.B.state).toBe('disrupted');
    expect(p1.regimes.Q.state).toBe('suspended');

    // Q appears in tick 2 with the same open_since — it never transitioned.
    expect(p2.regimes.Q).toEqual({
      state: 'suspended',
      open_state: 'suspended',
      open_since: t0 + 100,
    });
    // A and B flipped between ticks.
    expect(p2.regimes.A.state).toBe('disrupted');
    expect(p2.regimes.B.state).toBe('normal');
  });

  test('measured three-route object size', async () => {
    const { bucket, store } = fakeBucket();
    const census = buildMovementCensus(observed1, feedKeys, entries1, tick1);
    await writeMovementCensus(bucket, census);

    const body = [...store.values()][0]!;
    const parsed = JSON.parse(body);
    expect(parsed.observed_at).toBe(tick1);
    expect(Object.keys(parsed.regimes)).toHaveLength(3);

    // Exact byte count: 3-route fixture → 257 bytes.
    // Production (~30 routes) scales linearly: ~2.4 KB/tick, ~681 KB/day
    // at 288 ticks.
    const byteLength = new TextEncoder().encode(body).byteLength;
    expect(byteLength).toBe(257);
  });
});
