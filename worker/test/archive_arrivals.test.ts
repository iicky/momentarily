/**
 * Unit tests for the per-minute arrivals sample archive (archive.ts).
 * Tests the archiveArrivalsSample function that writes per-stop countdown
 * ETAs (first 4 arrivals only, gzipped) for grading against realized
 * arrivals.
 */

import { describe, expect, test, vi } from 'vitest';

import { archiveArrivalsSample } from '../src/archive';

// Minimal in-memory R2 bucket — just the put this helper touches. Captures
// httpMetadata too, so the gzip content-type/encoding contract is checkable.
function fakeBucket() {
  const store = new Map<string, Uint8Array>();
  const metadata = new Map<string, { contentType?: string; contentEncoding?: string }>();
  return {
    bucket: {
      async put(key: string, body: Uint8Array, opts?: { httpMetadata?: R2HTTPMetadata }) {
        store.set(key, body);
        if (opts?.httpMetadata) metadata.set(key, opts.httpMetadata);
        return {} as unknown;
      },
    } as unknown as R2Bucket,
    store,
    metadata,
  };
}

// archiveArrivalsSample gzips its body (Workers CompressionStream); tests
// decompress with the Web Platform's DecompressionStream, available in the
// same runtime, then parse.
async function readGzippedJson(bytes: Uint8Array): Promise<unknown> {
  const stream = new DecompressionStream('gzip');
  const writer = stream.writable.getWriter();
  void writer.write(bytes);
  void writer.close();

  const chunks: Uint8Array[] = [];
  const reader = stream.readable.getReader();
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.length;
  }
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return JSON.parse(new TextDecoder().decode(out));
}

describe('archiveArrivalsSample: R2 write shape', () => {
  test('writes a gzipped, date-partitioned, tick-keyed object with schema_version and observed_at', async () => {
    const { bucket, store, metadata } = fakeBucket();
    const arrivals = {
      'Q05S': [
        { route: '2', eta_epoch: 1_704_067_200, seconds_away: 100, trip_id: 'trip1' },
        { route: '3', eta_epoch: 1_704_067_300, seconds_away: 200, trip_id: 'trip2' },
      ],
    };

    await archiveArrivalsSample(
      bucket,
      1_704_067_200,
      arrivals,
      ['GTFSRT_feed1'],
      ['GTFSRT_feed1', 'GTFSRT_feed2'],
    );

    const key = 'archive/arrivals/2024-01-01/1704067200.json.gz';
    expect(store.has(key)).toBe(true);
    expect(metadata.get(key)).toEqual({ contentType: 'application/json', contentEncoding: 'gzip' });
    const body = await readGzippedJson(store.get(key)!);
    expect(body).toEqual({
      schema_version: 1,
      observed_at: 1_704_067_200,
      fresh_feeds: ['GTFSRT_feed1'],
      expected_feeds: ['GTFSRT_feed1', 'GTFSRT_feed2'],
      stops: {
        'Q05S': [
          { route: '2', eta_epoch: 1_704_067_200, trip_id: 'trip1' },
          { route: '3', eta_epoch: 1_704_067_300, trip_id: 'trip2' },
        ],
      },
    });
  });

  test('omits seconds_away field from archived arrivals (accepts full Arrival objects)', async () => {
    const { bucket, store } = fakeBucket();
    // Pass full Arrival objects with seconds_away
    const arrivals = {
      'Q05S': [
        { route: '2', eta_epoch: 1_704_067_200, seconds_away: 100, trip_id: 'trip1' },
      ],
    };

    await archiveArrivalsSample(
      bucket,
      1_704_067_200,
      arrivals,
      ['GTFSRT_feed1'],
      ['GTFSRT_feed1'],
    );

    const body = (await readGzippedJson(store.get('archive/arrivals/2024-01-01/1704067200.json.gz')!)) as {
      stops: Record<string, unknown[]>;
    };
    const arrival = body.stops['Q05S']![0];
    expect(arrival).not.toHaveProperty('seconds_away');
    expect(arrival).toEqual({
      route: '2',
      eta_epoch: 1_704_067_200,
      trip_id: 'trip1',
    });
  });

  test('keeps only the first 4 arrivals per stop when more are provided', async () => {
    const { bucket, store } = fakeBucket();
    const arrivals = {
      'Q05S': [
        { route: '2', eta_epoch: 1_704_067_200, seconds_away: 100, trip_id: 'trip1' },
        { route: '3', eta_epoch: 1_704_067_300, seconds_away: 200, trip_id: 'trip2' },
        { route: '4', eta_epoch: 1_704_067_400, seconds_away: 300, trip_id: 'trip3' },
        { route: '5', eta_epoch: 1_704_067_500, seconds_away: 400, trip_id: 'trip4' },
        { route: '6', eta_epoch: 1_704_067_600, seconds_away: 500, trip_id: 'trip5' },
      ],
    };

    await archiveArrivalsSample(
      bucket,
      1_704_067_200,
      arrivals,
      ['GTFSRT_feed1'],
      ['GTFSRT_feed1'],
    );

    const body = (await readGzippedJson(store.get('archive/arrivals/2024-01-01/1704067200.json.gz')!)) as {
      stops: Record<string, Array<{ trip_id: string | null }>>;
    };
    expect(body.stops['Q05S']).toHaveLength(4);
    expect(body.stops['Q05S']!.map((a) => a.trip_id)).toEqual(['trip1', 'trip2', 'trip3', 'trip4']);
  });

  test('preserves soonest-first ordering per stop', async () => {
    const { bucket, store } = fakeBucket();
    const arrivals = {
      'Q05S': [
        { route: '2', eta_epoch: 100, seconds_away: 50, trip_id: 'early' },
        { route: '3', eta_epoch: 200, seconds_away: 150, trip_id: 'late' },
      ],
    };

    await archiveArrivalsSample(
      bucket,
      1_704_067_200,
      arrivals,
      [],
      [],
    );

    const body = (await readGzippedJson(store.get('archive/arrivals/2024-01-01/1704067200.json.gz')!)) as {
      stops: Record<string, Array<{ eta_epoch: number }>>;
    };
    expect(body.stops['Q05S']![0]!.eta_epoch).toBe(100);
    expect(body.stops['Q05S']![1]!.eta_epoch).toBe(200);
  });

  test('handles multiple stops independently', async () => {
    const { bucket, store } = fakeBucket();
    const arrivals = {
      'Q05S': [
        { route: '2', eta_epoch: 100, seconds_away: 50, trip_id: 'trip1' },
      ],
      'Q05N': [
        { route: '3', eta_epoch: 200, seconds_away: 100, trip_id: 'trip2' },
        { route: '4', eta_epoch: 300, seconds_away: 150, trip_id: 'trip3' },
      ],
    };

    await archiveArrivalsSample(
      bucket,
      1_704_067_200,
      arrivals,
      ['GTFSRT_feed1'],
      ['GTFSRT_feed1'],
    );

    const body = (await readGzippedJson(store.get('archive/arrivals/2024-01-01/1704067200.json.gz')!)) as {
      stops: Record<string, unknown[]>;
    };
    expect(Object.keys(body.stops)).toEqual(['Q05N', 'Q05S']);
    expect(body.stops['Q05S']).toHaveLength(1);
    expect(body.stops['Q05N']).toHaveLength(2);
  });

  test('handles empty arrivals object', async () => {
    const { bucket, store } = fakeBucket();
    await archiveArrivalsSample(
      bucket,
      1_704_067_200,
      {},
      ['GTFSRT_feed1'],
      ['GTFSRT_feed1'],
    );

    const body = (await readGzippedJson(store.get('archive/arrivals/2024-01-01/1704067200.json.gz')!)) as {
      stops: unknown;
    };
    expect(body.stops).toEqual({});
  });

  test('handles null trip_id values', async () => {
    const { bucket, store } = fakeBucket();
    const arrivals = {
      'Q05S': [
        { route: '2', eta_epoch: 1_704_067_200, seconds_away: 100, trip_id: null },
      ],
    };

    await archiveArrivalsSample(
      bucket,
      1_704_067_200,
      arrivals,
      ['GTFSRT_feed1'],
      ['GTFSRT_feed1'],
    );

    const body = (await readGzippedJson(store.get('archive/arrivals/2024-01-01/1704067200.json.gz')!)) as {
      stops: Record<string, Array<{ trip_id: string | null }>>;
    };
    expect(body.stops['Q05S']![0]!.trip_id).toBeNull();
  });

  test('uses deterministic key based on observedAt', async () => {
    const { bucket: bucket1, store: store1 } = fakeBucket();
    const { bucket: bucket2, store: store2 } = fakeBucket();
    const arrivals = {
      'Q05S': [
        { route: '2', eta_epoch: 100, seconds_away: 50, trip_id: 'trip1' },
      ],
    };

    // Same observedAt generates same key, so a retry overwrites
    await archiveArrivalsSample(bucket1, 1_704_067_200, arrivals, ['f1'], ['f1']);
    await archiveArrivalsSample(bucket2, 1_704_067_200, arrivals, ['f1'], ['f1']);

    const key1 = Array.from(store1.keys())[0];
    const key2 = Array.from(store2.keys())[0];
    expect(key1).toBe(key2);
    expect(key1).toBe('archive/arrivals/2024-01-01/1704067200.json.gz');
  });

  test('includes fresh_feeds and expected_feeds arrays in the body', async () => {
    const { bucket, store } = fakeBucket();
    const arrivals = {
      'Q05S': [
        { route: '2', eta_epoch: 1_704_067_200, seconds_away: 100, trip_id: 'trip1' },
      ],
    };

    await archiveArrivalsSample(
      bucket,
      1_704_067_200,
      arrivals,
      ['GTFSRT_feed1', 'GTFSRT_feed2'],
      ['GTFSRT_feed1', 'GTFSRT_feed2', 'GTFSRT_feed3'],
    );

    const body = (await readGzippedJson(store.get('archive/arrivals/2024-01-01/1704067200.json.gz')!)) as {
      fresh_feeds: string[];
      expected_feeds: string[];
    };
    expect(body.fresh_feeds).toEqual(['GTFSRT_feed1', 'GTFSRT_feed2']);
    expect(body.expected_feeds).toEqual(['GTFSRT_feed1', 'GTFSRT_feed2', 'GTFSRT_feed3']);
  });

  test('sorts stop IDs for deterministic JSON key order', async () => {
    const { bucket, store } = fakeBucket();
    const arrivals = {
      'Z05S': [
        { route: '2', eta_epoch: 100, seconds_away: 50, trip_id: 'trip1' },
      ],
      'A05S': [
        { route: '3', eta_epoch: 200, seconds_away: 100, trip_id: 'trip2' },
      ],
      'M05S': [
        { route: '4', eta_epoch: 300, seconds_away: 150, trip_id: 'trip3' },
      ],
    };

    await archiveArrivalsSample(
      bucket,
      1_704_067_200,
      arrivals,
      ['GTFSRT_feed1'],
      ['GTFSRT_feed1'],
    );

    const body = (await readGzippedJson(store.get('archive/arrivals/2024-01-01/1704067200.json.gz')!)) as {
      stops: Record<string, unknown>;
    };
    const stopIds = Object.keys(body.stops);
    // Verify stops are in sorted order (A, M, Z)
    expect(stopIds).toEqual(['A05S', 'M05S', 'Z05S']);
  });

  test('bucket.put failure propagates to the caller for fail-soft handling', async () => {
    const failingBucket = {
      async put() {
        throw new Error('bucket failure: simulated R2 outage');
      },
    } as unknown as R2Bucket;

    const arrivals = {
      'Q05S': [
        { route: '2', eta_epoch: 100, seconds_away: 50, trip_id: 'trip1' },
      ],
    };

    // bucket.put failure should propagate (not be swallowed internally)
    await expect(
      archiveArrivalsSample(failingBucket, 1_704_067_200, arrivals, ['f1'], ['f1']),
    ).rejects.toThrow('bucket failure: simulated R2 outage');
  });
});
