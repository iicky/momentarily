/**
 * Unit tests for the alert-fetch liveness record (archive.ts). Exists so an
 * offline evaluation can prove the alerts feed was genuinely live during a
 * quiet period, rather than inferring it from archiveNewAlerts writing zero
 * new alert versions — which happens on BOTH a calm night and a feed outage.
 */

import { describe, expect, test } from 'vitest';

import { archiveAlertsLiveness, archiveTraceRows, deriveAlertsLiveness } from '../src/archive';

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

describe('archiveTraceRows: feed_digest / feed_etag', () => {
  const row = {
    trip_id: 't1',
    route_id: 'A',
    direction: 'north' as const,
    stop_id: 'A09N',
    stop_seq: 1,
    stopped: true,
    vehicle_ts: 1_704_067_200,
  };
  // 2024-01-01 00:00:00 UTC
  const at = 1_704_067_200;

  test('includes both feed_digest and feed_etag when provided', async () => {
    const { bucket, store } = fakeBucket();
    await archiveTraceRows(bucket, [row], ['nqrw'], at, at, 'abc123', '"etag-1"');
    const key = `archive/trace/2024-01-01/${at}.json`;
    const body = JSON.parse(store.get(key)!);
    expect(body.feed_digest).toBe('abc123');
    expect(body.feed_etag).toBe('"etag-1"');
  });

  test('explicit nulls when HEAD failed (not omitted — distinguishes resolution failure from historical absence)', async () => {
    const { bucket, store } = fakeBucket();
    await archiveTraceRows(bucket, [row], ['nqrw'], at, at, null, null);
    const key = `archive/trace/2024-01-01/${at}.json`;
    const body = JSON.parse(store.get(key)!);
    expect(body.feed_digest).toBeNull();
    expect(body.feed_etag).toBeNull();
    expect(body.observed_at).toBe(at);
    expect(body.rows).toHaveLength(1);
  });

  test('feed_etag null while feed_digest is set', async () => {
    const { bucket, store } = fakeBucket();
    await archiveTraceRows(bucket, [row], ['nqrw'], at, at, 'abc123', null);
    const key = `archive/trace/2024-01-01/${at}.json`;
    const body = JSON.parse(store.get(key)!);
    expect(body.feed_digest).toBe('abc123');
    expect(body.feed_etag).toBeNull();
  });

  test('a HEAD/pointer mismatch tick carries feed_etag with feed_digest null — capture is detached and has not finished yet', async () => {
    const { bucket, store } = fakeBucket();
    await archiveTraceRows(bucket, [row], ['nqrw'], at, at, null, '"new-etag"');
    const key = `archive/trace/2024-01-01/${at}.json`;
    const body = JSON.parse(store.get(key)!);
    expect(body.feed_digest).toBeNull();
    expect(body.feed_etag).toBe('"new-etag"');
  });

  test('default feedDigest/feedEtag are both null (explicit in body)', async () => {
    const { bucket, store } = fakeBucket();
    await archiveTraceRows(bucket, [row], ['nqrw'], at, at);
    const key = `archive/trace/2024-01-01/${at}.json`;
    const body = JSON.parse(store.get(key)!);
    expect(body.feed_digest).toBeNull();
    expect(body.feed_etag).toBeNull();
  });
});
