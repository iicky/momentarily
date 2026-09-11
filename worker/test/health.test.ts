/**
 * Pins the per-tick write-failure counter (archive/health/) added in
 * archive.ts/index.ts. Every archive/state write in `scheduled` already
 * degrades fail-soft (log and move on) on its own — correct for a single
 * tick, but it means a PERSISTENT failure on one prefix (say, a bad R2
 * object wedging one write path) never fails a tick or pages anyone: the
 * live snapshot stays fresh from the writes that DO succeed while that one
 * training-input stream quietly stops accruing. These tests exercise the
 * real `scheduled` handler end to end against a fake R2 bucket that can be
 * told to fail puts under a chosen prefix, and assert the failure shows up
 * as a count in that tick's archive/health/ record.
 */

import { beforeEach, describe, expect, test, vi } from 'vitest';

const fetchState = vi.hoisted(() => ({
  jsonByUrl: new Map<string, unknown>(),
  protobufByUrl: new Map<string, Uint8Array>(),
  protobufFailUrls: new Set<string>(),
  jsonFailUrls: new Set<string>(),
}));

vi.mock('../src/fetch', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/fetch')>();
  return {
    ...actual,
    fetchJson: async (url: string) => {
      if (fetchState.jsonFailUrls.has(url)) throw new Error(`mock fetch failure: ${url}`);
      return fetchState.jsonByUrl.get(url) ?? {};
    },
    fetchProtobuf: async (url: string) => {
      if (fetchState.protobufFailUrls.has(url)) throw new Error(`mock fetch failure: ${url}`);
      return fetchState.protobufByUrl.get(url) ?? new Uint8Array();
    },
  };
});

import { FEEDS, STATIONS_FEED, TRIP_UPDATE_FEEDS } from '../src/fetch';
import worker from '../src/index';
import type { Env } from '../src/index';

// --- fake R2 bucket with etag CAS support (matches r2.ts's conditionalPut),
// plus a `failPrefixes` set a test can populate to force `put` to reject for
// any key starting with one of those prefixes — same convention as
// index.test.ts's fakeBucket, extended with the one thing this suite needs
// that no existing test does: a forceable write failure.
interface StoredObject {
  body: string;
  etag: string;
  httpMetadata?: R2HTTPMetadata;
}

function fakeBucket() {
  const store = new Map<string, StoredObject>();
  const failPrefixes = new Set<string>();
  let seq = 0;
  const bucket = {
    async get(key: string) {
      const rec = store.get(key);
      if (!rec) return null;
      return {
        etag: rec.etag,
        httpEtag: rec.etag,
        json: async () => JSON.parse(rec.body) as unknown,
        text: async () => rec.body,
        body: rec.body,
        writeHttpMetadata(headers: Headers) {
          if (rec.httpMetadata?.contentType) headers.set('content-type', rec.httpMetadata.contentType);
        },
      };
    },
    async put(
      key: string,
      body: string,
      opts?: { httpMetadata?: R2HTTPMetadata; onlyIf?: R2Conditional | Headers },
    ) {
      for (const prefix of failPrefixes) {
        if (key.startsWith(prefix)) throw new Error(`forced put failure: ${key}`);
      }
      const existing = store.get(key);
      const onlyIf = opts?.onlyIf;
      if (onlyIf) {
        if (onlyIf instanceof Headers) {
          if (onlyIf.get('If-None-Match') === '*' && existing) return null;
        } else if ('etagMatches' in onlyIf && onlyIf.etagMatches !== undefined) {
          if (!existing || existing.etag !== onlyIf.etagMatches) return null;
        }
      }
      const etag = `etag-${++seq}`;
      store.set(key, { body, etag, ...(opts?.httpMetadata ? { httpMetadata: opts.httpMetadata } : {}) });
      return { etag };
    },
  };
  return { bucket: bucket as unknown as R2Bucket, store, failPrefixes };
}

function keysWithPrefix(store: Map<string, StoredObject>, prefix: string): string[] {
  return [...store.keys()].filter((k) => k.startsWith(prefix));
}

function jsonAt(store: Map<string, StoredObject>, key: string): unknown {
  const rec = store.get(key);
  return rec ? JSON.parse(rec.body) : undefined;
}

function healthKey(observedAt: number): string {
  const date = new Date(observedAt * 1000).toISOString().slice(0, 10);
  return `archive/health/${date}/${observedAt}.json`;
}

function scheduledAt(epochSec: number): ScheduledController {
  return { cron: '* * * * *', scheduledTime: epochSec * 1000 } as unknown as ScheduledController;
}
const execCtx = {} as unknown as ExecutionContext;

async function runTick(env: Env, observedAt: number): Promise<void> {
  const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(observedAt * 1000);
  try {
    await worker.scheduled(scheduledAt(observedAt), env, execCtx);
  } finally {
    nowSpy.mockRestore();
  }
}

beforeEach(() => {
  fetchState.jsonByUrl.clear();
  fetchState.protobufByUrl.clear();
  fetchState.protobufFailUrls.clear();
  fetchState.jsonFailUrls.clear();
});

describe('scheduled: write-failure counter (archive/health/)', () => {
  const BOUNDARY_AT = 1_704_067_200; // 2024-01-01T00:00:00Z, minute 0
  const NEXT_BOUNDARY_AT = BOUNDARY_AT + 300; // +5 minutes

  test('a clean tick archives an empty write_failures record', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    fetchState.jsonByUrl.set(FEEDS.alerts, { entity: [] });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);

    await runTick(env, BOUNDARY_AT);

    expect(jsonAt(store, healthKey(BOUNDARY_AT))).toEqual({
      observed_at: BOUNDARY_AT,
      write_failures: {},
    });
  });

  test('a forced put failure on the predictions prefix shows up as a count in the health record', async () => {
    const { bucket, store, failPrefixes } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    fetchState.jsonByUrl.set(FEEDS.alerts, { entity: [] });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);
    // v1/predictions/ is written by writePredictions (grading.ts), gated
    // behind the alpha CAS winner — forcing it to fail exercises index.ts's
    // predictions-write catch block without touching any other write path.
    failPrefixes.add('v1/predictions/');

    await runTick(env, BOUNDARY_AT);

    // The forced prefix never landed...
    expect(keysWithPrefix(store, 'v1/predictions/')).toHaveLength(0);
    // ...but the tick otherwise completed normally: snapshot and last_seen,
    // on different R2 objects, are untouched by the predictions failure.
    expect(store.has('v1/snapshot.json')).toBe(true);
    expect(store.has('state/last_seen.json')).toBe(true);
    // And the failure is now visible as a count, not just a swallowed log line.
    expect(jsonAt(store, healthKey(BOUNDARY_AT))).toEqual({
      observed_at: BOUNDARY_AT,
      write_failures: { predictions_write: 1 },
    });
  });

  test('a persistent failure across ticks shows up as a count on every tick, not just the first — the corpus-rot signal this record exists to surface', async () => {
    const { bucket, store, failPrefixes } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    fetchState.jsonByUrl.set(FEEDS.alerts, { entity: [] });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);
    failPrefixes.add('v1/predictions/');

    await runTick(env, BOUNDARY_AT);
    await runTick(env, NEXT_BOUNDARY_AT);

    expect(jsonAt(store, healthKey(BOUNDARY_AT))).toEqual({
      observed_at: BOUNDARY_AT,
      write_failures: { predictions_write: 1 },
    });
    expect(jsonAt(store, healthKey(NEXT_BOUNDARY_AT))).toEqual({
      observed_at: NEXT_BOUNDARY_AT,
      write_failures: { predictions_write: 1 },
    });
    // The rest of the pipeline kept publishing fresh data both ticks — this
    // is precisely the "feed stays fresh while training inputs quietly stop
    // accruing" failure mode: without this record it would be invisible.
    expect(jsonAt(store, 'v1/snapshot.json')).toMatchObject({ generated_at: NEXT_BOUNDARY_AT });
  });

  test('a failure on a different prefix (alerts liveness) is counted under its own key, independent of predictions', async () => {
    const { bucket, store, failPrefixes } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    fetchState.jsonByUrl.set(FEEDS.alerts, { entity: [] });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);
    failPrefixes.add('archive/alerts_liveness/');

    await runTick(env, BOUNDARY_AT);

    expect(keysWithPrefix(store, 'archive/alerts_liveness/')).toHaveLength(0);
    expect(jsonAt(store, healthKey(BOUNDARY_AT))).toEqual({
      observed_at: BOUNDARY_AT,
      write_failures: { alerts_liveness: 1 },
    });
    // Predictions, on an unrelated prefix, still wrote fine.
    expect(keysWithPrefix(store, 'v1/predictions/').length).toBeGreaterThan(0);
  });

  test('a forced stations cache write failure is counted as stations_cache_write', async () => {
    const { bucket, store, failPrefixes } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    fetchState.jsonByUrl.set(FEEDS.alerts, { entity: [] });
    // Return a parseable station row so parseStationsFeed yields a non-empty
    // array and writeStationsCache is actually reached (an empty feed skips
    // the write entirely — see index.ts step 8c).
    fetchState.jsonByUrl.set(STATIONS_FEED, [
      { gtfs_stop_id: '101', stop_name: 'Test Station' },
    ]);
    failPrefixes.add('state/stations.json');

    await runTick(env, BOUNDARY_AT);

    // The stations object was never written...
    expect(store.has('state/stations.json')).toBe(false);
    // ...and the failure is visible in the health record.
    const health = jsonAt(store, healthKey(BOUNDARY_AT)) as {
      write_failures: Record<string, number>;
    };
    expect(health.write_failures.stations_cache_write).toBe(1);
  });
});
