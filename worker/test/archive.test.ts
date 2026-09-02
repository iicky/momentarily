/**
 * Unit tests for the alert-fetch liveness record (archive.ts). Exists so an
 * offline evaluation can prove the alerts feed was genuinely live during a
 * quiet period, rather than inferring it from archiveNewAlerts writing zero
 * new alert versions — which happens on BOTH a calm night and a feed outage.
 */

import { describe, expect, test } from 'vitest';

import { archiveAlertsLiveness, deriveAlertsLiveness } from '../src/archive';

// Minimal in-memory R2 bucket — just the put this helper touches, same
// convention as state.test.ts's fakeBucket.
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

describe('deriveAlertsLiveness: outcome mapping', () => {
  test('a successful fetch maps to success, fetched_at stamped to this tick', () => {
    expect(deriveAlertsLiveness(true, 1_704_067_200)).toEqual({
      outcome: 'success',
      fetched_at: 1_704_067_200,
    });
  });

  test('a failed fetch maps to fail, NOT to a fabricated success', () => {
    const result = deriveAlertsLiveness(false, 1_704_067_200);
    expect(result.outcome).toBe('fail');
  });

  test('stale-fallback branch: on failure, fetched_at carries the last known-good time forward — never bumped to "now"', () => {
    // Mirrors index.ts: alertsFeedFresh starts as lastSeen.alerts_at and is
    // only advanced to observedAt inside the fetch's success branch. A
    // failed tick therefore calls this with the OLD timestamp, not the
    // current observedAt — that gap (observedAt - fetched_at) is the exact
    // signal an offline reader needs to size an outage.
    const observedAt = 1_704_067_500; // "now"
    const lastKnownGoodAt = 1_704_060_000; // ~2h5m earlier
    const result = deriveAlertsLiveness(false, lastKnownGoodAt);
    expect(result).toEqual({ outcome: 'fail', fetched_at: lastKnownGoodAt });
    expect(result.fetched_at).not.toBe(observedAt);
  });

  test('a cold start with no prior successful fetch reports fetched_at 0, not fail-silently-as-fresh', () => {
    expect(deriveAlertsLiveness(false, 0)).toEqual({ outcome: 'fail', fetched_at: 0 });
  });
});

describe('archiveAlertsLiveness: R2 write shape', () => {
  test('writes a date-partitioned, tick-keyed object with the full record', async () => {
    const { bucket, store } = fakeBucket();
    await archiveAlertsLiveness(
      bucket,
      { outcome: 'success', fetched_at: 1_704_067_200 },
      1_704_067_200,
    );
    const key = 'archive/alerts_liveness/2024-01-01/1704067200.json';
    expect(store.has(key)).toBe(true);
    expect(JSON.parse(store.get(key)!)).toEqual({
      observed_at: 1_704_067_200,
      outcome: 'success',
      fetched_at: 1_704_067_200,
    });
  });

  test('writes the fail outcome too — this is the tick the record exists to capture', async () => {
    const { bucket, store } = fakeBucket();
    await archiveAlertsLiveness(
      bucket,
      { outcome: 'fail', fetched_at: 1_704_060_000 },
      1_704_067_500,
    );
    const key = 'archive/alerts_liveness/2024-01-01/1704067500.json';
    expect(JSON.parse(store.get(key)!)).toEqual({
      observed_at: 1_704_067_500,
      outcome: 'fail',
      fetched_at: 1_704_060_000,
    });
  });

  test('a retried scheduled minute keeps BOTH attempts, because the key is execution wall-clock', async () => {
    // observedAt is Date.now()-derived, not the cron's scheduled minute, and
    // this call is not gated on the alpha CAS winner. So a failed attempt and a
    // succeeding retry a few seconds later land under two keys and both survive
    // — that retained evidence is the whole point of the prefix. A reader takes
    // the attempts in a scheduled minute as a set, not as a single record.
    const { bucket, store } = fakeBucket();
    const firstAttempt = 1_704_067_201;
    const retry = 1_704_067_204; // same scheduled minute, 3s later
    await archiveAlertsLiveness(bucket, { outcome: 'fail', fetched_at: 0 }, firstAttempt);
    await archiveAlertsLiveness(bucket, { outcome: 'success', fetched_at: retry }, retry);

    const keys = [...store.keys()]
      .filter((k) => k.startsWith('archive/alerts_liveness/'))
      .sort();
    expect(keys).toEqual([
      'archive/alerts_liveness/2024-01-01/1704067201.json',
      'archive/alerts_liveness/2024-01-01/1704067204.json',
    ]);
    expect(JSON.parse(store.get(keys[0]!)!).outcome).toBe('fail');
    expect(JSON.parse(store.get(keys[1]!)!).outcome).toBe('success');
  });
});
