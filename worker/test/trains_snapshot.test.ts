/**
 * trains.json: the published sibling artifact to snapshot.json — aggregated
 * live train positions for the /map overlay, at its own URL rather than a
 * Snapshot field (see snapshot.ts's file header for the size/consumer
 * rationale). This file covers buildTrains/publishTrains in isolation, plus
 * the "snapshot no longer carries trains" contract; the fail-soft publish
 * gate (all-feeds-failed vs. partial) is covered end-to-end through the real
 * scheduled() handler in index.test.ts, since that's where the decision to
 * call publishTrains at all is made.
 */

import { describe, expect, test } from 'vitest';

import { TICK_SECONDS, buildSnapshot, buildTrains, publishTrains } from '../src/snapshot';
import type { TrainPosition } from '../src/vehicles';

const NOW = 1_700_000_000;

function fakeBucket() {
  const store = new Map<string, { body: string; httpMetadata?: R2HTTPMetadata }>();
  return {
    bucket: {
      async put(key: string, body: string, opts?: { httpMetadata?: R2HTTPMetadata }) {
        store.set(key, { body, ...(opts?.httpMetadata ? { httpMetadata: opts.httpMetadata } : {}) });
        return {} as unknown;
      },
    } as unknown as R2Bucket,
    store,
  };
}

describe('Snapshot no longer carries a trains field', () => {
  test('buildSnapshot output has no "trains" key at all — it moved to its own artifact', () => {
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: new Map(),
      rolls: {},
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
    });
    expect(Object.keys(snap)).not.toContain('trains');
    expect(JSON.stringify(snap)).not.toContain('"trains"');
  });
});

describe('buildTrains', () => {
  const positions: TrainPosition[] = [
    { route: 'F', direction: 'north', stop: 'A09N', stopped: false, n: 3 },
  ];

  test('carries its own observed_at and a provenance block, independent of any Snapshot', () => {
    const trains = buildTrains(NOW, positions, ['ace'], ['ace', 'bdfm']);
    expect(trains.observed_at).toBe(NOW);
    expect(trains.provenance).toEqual({ code_sha: 'unknown', dirty: null, producer: 'worker' });
    expect(trains.positions).toEqual(positions);
  });

  test('a complete feed set (fresh_feeds === expected_feeds) with zero positions is a genuine "zero trains" reading', () => {
    const trains = buildTrains(NOW, [], ['ace', 'bdfm'], ['ace', 'bdfm']);
    expect(trains.fresh_feeds).toEqual(trains.expected_feeds);
    expect(trains.positions).toEqual([]);
  });

  test('fresh_feeds shorter than expected_feeds flags positions as a partial, not a complete, read', () => {
    const trains = buildTrains(NOW, positions, ['ace'], ['ace', 'bdfm', 'g']);
    expect(trains.fresh_feeds).toEqual(['ace']);
    expect(trains.expected_feeds).toEqual(['ace', 'bdfm', 'g']);
    expect(trains.fresh_feeds.length).toBeLessThan(trains.expected_feeds.length);
  });
});

describe('publishTrains', () => {
  test('writes to v1/trains.json with the same content-type/cache-control convention as snapshot.json', async () => {
    const { bucket, store } = fakeBucket();
    const trains = buildTrains(NOW, [], ['ace'], ['ace']);
    await publishTrains(bucket, trains);

    const rec = store.get('v1/trains.json');
    expect(rec).toBeDefined();
    expect(JSON.parse(rec!.body)).toEqual(trains);
    expect(rec!.httpMetadata?.contentType).toBe('application/json');
    expect(rec!.httpMetadata?.cacheControl).toBe('public, max-age=60, s-maxage=300');
  });
});
