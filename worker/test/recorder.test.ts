/**
 * The raw-observation recorder (recorder.ts) against a Miniflare R2 bucket,
 * driven through the real `scheduled` handler wherever the wiring matters.
 *
 * Upstream bodies are one minute of the live MTA feeds captured on
 * 2026-10-20 (test/fixtures/observations/, gzipped): the eight GTFS-RT feeds,
 * the alerts JSON and the ENE outage list. Every record the recorder writes is
 * validated against the JSON Schema exported from the Pydantic contracts;
 * tests/fixtures/recorder_output.json carries one through the Pydantic
 * validators themselves (tests/test_recorder_output.py).
 *
 * Date.now is frozen in every test, so nothing depends on the day it runs.
 */

/// <reference types="node" />

import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync, inflateRawSync } from 'node:zlib';

import Ajv2020 from 'ajv/dist/2020';
import { Miniflare } from 'miniflare';
import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, test, vi } from 'vitest';

import collectorTickSchema from '../../schema/model/s1/observations/collector_tick.schema.json';
import bundleIndexSchema from '../../schema/model/s1/observations/tick_bundle_index.schema.json';
import type { CollectorTick, FeedFetch } from '../../shared/model/s1/observations/collector_tick';
import type { TickBundleIndex } from '../../shared/model/s1/observations/tick_bundle_index';
import keyParity from '../../tests/fixtures/parity_observation_keys.json';
import { ENTRANCES_FEED, FEEDS, STATIONS_FEED, TRIP_UPDATE_FEEDS, fetchJson, fetchProtobuf } from '../src/fetch';
import worker from '../src/index';
import { OBSERVATIONS_PREFIX, blobKey, bundleKey, tickKey, utcDay, type BlobExt } from '../src/observation_keys';
import { Recorder } from '../src/recorder';

const ajv = new Ajv2020({ allErrors: true, strict: false });
const validateTick = ajv.compile<CollectorTick>(collectorTickSchema);
const validateIndex = ajv.compile<TickBundleIndex>(bundleIndexSchema);

function expectValid(validate: typeof validateTick | typeof validateIndex, doc: unknown): void {
  expect(validate(doc), JSON.stringify(validate.errors)).toBe(true);
}

const HERE = dirname(fileURLToPath(import.meta.url));
const OUTPUT_FIXTURE = join(HERE, '..', '..', 'tests', 'fixtures', 'recorder_output.json');
const fixture = (name: string): Uint8Array =>
  new Uint8Array(gunzipSync(readFileSync(join(HERE, 'fixtures', 'observations', `${name}.gz`))));

const GTFSRT_BODIES: Record<string, Uint8Array> = Object.fromEntries(
  TRIP_UPDATE_FEEDS.map(([name]) => [name, fixture(`${name}.pb`)]),
);
const ALERTS_BODY = fixture('alerts.json');
const ENE_BODY = fixture('ene_current.json');
const STATIONS_BODY = new TextEncoder().encode('[{"gtfs_stop_id":"A02","stop_name":"Inwood-207 St"}]');

const sha256 = (bytes: Uint8Array): string => createHash('sha256').update(bytes).digest('hex');
const text = (bytes: Uint8Array): string => new TextDecoder().decode(bytes);

interface InflateInfo {
  buffer: Buffer;
  engine: { bytesWritten: number };
}

/**
 * Split a tick bundle into its gzip members. Each member is a 10-byte header
 * with no optional fields (what CompressionStream writes), raw deflate, and an
 * 8-byte trailer; inflateRaw reports how much deflate input it consumed.
 */
function bundleMembers(bundle: Uint8Array): Uint8Array[] {
  const members: Uint8Array[] = [];
  let offset = 0;
  while (offset < bundle.length) {
    expect([bundle[offset], bundle[offset + 1], bundle[offset + 3]]).toEqual([0x1f, 0x8b, 0]);
    // Node's zlib types omit the `{ info: true }` overload's return shape.
    const inflated = inflateRawSync(bundle.subarray(offset + 10), { info: true }) as unknown as InflateInfo;
    members.push(new Uint8Array(inflated.buffer));
    offset += 10 + inflated.engine.bytesWritten + 8;
  }
  expect(offset).toBe(bundle.length);
  return members;
}

// --- upstream -------------------------------------------------------------------

type Upstream = (url: string, init: RequestInit) => Promise<Response>;

const HEADERS = { 'content-type': 'text/plain', date: 'Tue, 20 Oct 2026 16:42:01 GMT' };

/** The captured minute: every feed answers 200 with its fixture body. */
const liveMinute: Upstream = async (url, init) => {
  if (init.method === 'HEAD') return new Response(null, { status: 200, headers: { etag: '"static"' } });
  const gtfsrt = TRIP_UPDATE_FEEDS.find(([, u]) => u === url);
  if (gtfsrt) return new Response(GTFSRT_BODIES[gtfsrt[0]], { headers: HEADERS });
  if (url === FEEDS.alerts) return new Response(ALERTS_BODY, { headers: { ...HEADERS, 'content-type': 'application/json' } });
  if (url === FEEDS.ene_current) return new Response(ENE_BODY, { headers: HEADERS });
  if (url === STATIONS_FEED) return new Response(STATIONS_BODY, { headers: HEADERS });
  return new Response('[]', { headers: HEADERS });
};

const upstreamCalls: string[] = [];

function stubUpstream(upstream: Upstream): void {
  vi.stubGlobal('fetch', vi.fn((url: string, init?: RequestInit) => {
    upstreamCalls.push(url);
    return upstream(url, init ?? {});
  }));
}

/** The live minute with `url` answered by `answer` instead. */
const replacing = (url: string, answer: Upstream): Upstream =>
  (u, init) => (u === url ? answer(u, init) : liveMinute(u, init));

const timeout: Upstream = async () => {
  throw new DOMException('The operation was aborted due to timeout', 'TimeoutError');
};
const http500: Upstream = async () => new Response('upstream broke', { status: 500, headers: HEADERS });
const empty200: Upstream = async () => new Response(new Uint8Array(), { headers: HEADERS });
const GARBAGE = new TextEncoder().encode('<html><body>Service Unavailable</body></html>');
const garbage200: Upstream = async () => new Response(GARBAGE, { headers: HEADERS });
const bodyFails: Upstream = async () =>
  new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new Uint8Array([0x0a, 0x03]));
        controller.error(new TypeError('connection reset'));
      },
    }),
    { headers: HEADERS },
  );

// --- R2 -------------------------------------------------------------------------

let mf: Miniflare;
let r2: R2Bucket;

interface Faults {
  put?: string;
  head?: string;
  get?: string;
}

interface BucketSpy {
  bucket: R2Bucket;
  puts: string[];
  faults: Faults;
  onGetFault?: () => void;
}

/**
 * The Miniflare bucket, recording PUT keys and failing put/head/get on keys
 * under the prefix set in `faults`. Miniflare's Node-side binding cannot
 * serialize a `Headers` onlyIf (it rejects every non-POJO), so the
 * `If-None-Match: *` form r2.ts's conditionalPut sends is passed on as its
 * object equivalent, `{ etagDoesNotMatch: '*' }`.
 */
function spyBucket(): BucketSpy {
  const spy: BucketSpy = { bucket: r2, puts: [], faults: {} };
  const fail = (op: keyof Faults, key: string): void => {
    const prefix = spy.faults[op];
    if (prefix !== undefined && key.startsWith(prefix)) throw new Error(`injected ${op} failure: ${key}`);
  };
  spy.bucket = new Proxy(r2, {
    get(target, prop, receiver) {
      if (prop === 'put') {
        return async (key: string, body: unknown, opts?: R2PutOptions) => {
          spy.puts.push(key);
          fail('put', key);
          const onlyIf = opts?.onlyIf instanceof Headers
            ? (opts.onlyIf.get('If-None-Match') === '*' ? { etagDoesNotMatch: '*' } : undefined)
            : opts?.onlyIf;
          return target.put(key, body as string, { ...opts, ...(onlyIf ? { onlyIf } : {}) } as R2PutOptions);
        };
      }
      if (prop === 'head') {
        return async (key: string) => {
          fail('head', key);
          return target.head(key);
        };
      }
      if (prop === 'get') {
        return async (key: string, opts?: R2GetOptions) => {
          if (spy.faults.get !== undefined && key.startsWith(spy.faults.get)) spy.onGetFault?.();
          fail('get', key);
          return target.get(key, opts);
        };
      }
      const value = Reflect.get(target, prop, receiver) as unknown;
      return typeof value === 'function' ? value.bind(target) : value;
    },
  });
  return spy;
}

async function keysUnder(prefix: string): Promise<string[]> {
  const listed = await r2.list({ prefix });
  return listed.objects.map((o) => o.key).sort();
}

async function bytesAt(key: string): Promise<Uint8Array> {
  const obj = await r2.get(key);
  expect(obj, key).not.toBeNull();
  return new Uint8Array(await obj!.arrayBuffer());
}

async function tickAt(scheduledAt: number, startedAtMs: number): Promise<CollectorTick> {
  const tick = JSON.parse(text(await bytesAt(tickKey(scheduledAt, startedAtMs)))) as unknown;
  expectValid(validateTick, tick);
  return tick as CollectorTick;
}

async function indexAt(scheduledAt: number, startedAtMs: number): Promise<{ index: TickBundleIndex; members: Uint8Array[] }> {
  const members = bundleMembers(await bytesAt(bundleKey(scheduledAt, startedAtMs)));
  const index = JSON.parse(text(members[0]!)) as unknown;
  expectValid(validateIndex, index);
  return { index: index as TickBundleIndex, members };
}

async function expectHealth(observedAt: number, writeFailures: Record<string, number>): Promise<void> {
  const key = `archive/health/${utcDay(observedAt)}/${observedAt}.json`;
  expect(JSON.parse(text(await bytesAt(key)))).toEqual({ observed_at: observedAt, write_failures: writeFailures });
}

function fetchOf(tick: CollectorTick, source: FeedFetch['source']): FeedFetch {
  const f = tick.fetches.find((x) => x.source === source);
  expect(f, source).toBeDefined();
  return f!;
}

// 2026-10-20T16:43:00Z, minute 43: off the 5-minute boundary.
const SCHEDULED_AT = 1_792_514_580;
const STARTED_AT_MS = SCHEDULED_AT * 1000 + 412;
// 2026-10-20T16:40:00Z, a 5-minute boundary.
const BOUNDARY_AT = 1_792_514_400;

beforeAll(async () => {
  mf = new Miniflare({ modules: true, script: 'export default {}', r2Buckets: ['MOMENTARILY'] });
  r2 = (await mf.getR2Bucket('MOMENTARILY')) as unknown as R2Bucket;
});

afterAll(async () => {
  await mf.dispose();
});

beforeEach(async () => {
  let listed = await r2.list();
  while (listed.objects.length > 0) {
    await r2.delete(listed.objects.map((o) => o.key));
    listed = await r2.list();
  }
  upstreamCalls.length = 0;
  vi.spyOn(Date, 'now').mockReturnValue(STARTED_AT_MS);
});

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

/** One real cron invocation for `scheduledAt`, started at `startedAtMs`. */
async function runScheduled(bucket: R2Bucket, scheduledAt: number, startedAtMs: number): Promise<void> {
  vi.mocked(Date.now).mockReturnValue(startedAtMs);
  await worker.scheduled(
    { cron: '* * * * *', scheduledTime: scheduledAt * 1000 } as unknown as ScheduledController,
    { MOMENTARILY: bucket },
    { waitUntil: () => {} } as unknown as ExecutionContext,
  );
}

/** One recorder over the GTFS-RT feeds and the given JSON sources, the way index.ts drives it. */
async function recordMinute(
  bucket: R2Bucket,
  opts: { startedAtMs?: number; json?: Array<[FeedFetch['source'], string]>; failWrite?: (k: string) => void } = {},
): Promise<void> {
  const recorder = new Recorder(bucket, SCHEDULED_AT, opts.startedAtMs ?? STARTED_AT_MS, opts.failWrite ?? (() => {}));
  for (const [source, url] of opts.json ?? [['alerts', FEEDS.alerts]]) {
    void recorder.attempt(source, (observe) => fetchJson(url, observe)).catch(() => {});
  }
  await Promise.allSettled(
    TRIP_UPDATE_FEEDS.map(([name, url]) => recorder.attempt(name, (observe) => fetchProtobuf(url, observe))),
  );
  await recorder.storeBundle();
  await recorder.finish();
}

// --- tests -------------------------------------------------------------------------

describe('key layout matches keys.py', () => {
  test('tick, bundle and blob keys equal the Python-formatted fixture', () => {
    for (const a of keyParity.attempts) {
      expect(utcDay(a.scheduled_at)).toBe(a.utc_day);
      expect(tickKey(a.scheduled_at, a.started_at_ms)).toBe(a.tick);
      expect(bundleKey(a.scheduled_at, a.started_at_ms)).toBe(a.bundle);
    }
    for (const b of keyParity.blobs) {
      expect(blobKey(utcDay(b.observed_at), b.sha256, b.ext as BlobExt)).toBe(b.key);
    }
  });

  test('instants keys.py cannot hold are refused, not formatted', () => {
    expect(() => tickKey(-1, 0)).toThrow(RangeError);
    expect(() => bundleKey(253_402_300_800, 0)).toThrow(RangeError);
    expect(() => blobKey('2026-02-30', '0'.repeat(64), 'json')).toThrow(RangeError);
  });
});

describe('scheduled: an off-boundary tick', () => {
  test('writes its tick record, a bundle byte-identical to the fetched bodies, and the alerts blob', async () => {
    stubUpstream(liveMinute);
    await runScheduled(r2, SCHEDULED_AT, STARTED_AT_MS);

    const day = utcDay(SCHEDULED_AT);
    const alertsKey = blobKey(day, sha256(ALERTS_BODY), 'json');
    expect(await keysUnder(OBSERVATIONS_PREFIX)).toEqual(
      [alertsKey, bundleKey(SCHEDULED_AT, STARTED_AT_MS), tickKey(SCHEDULED_AT, STARTED_AT_MS)].sort(),
    );
    // The pipeline beside it ran as usual.
    expect(await keysUnder('archive/trace/')).toHaveLength(1);
    await expectHealth(SCHEDULED_AT, {});

    const tick = await tickAt(SCHEDULED_AT, STARTED_AT_MS);
    expect(tick.tick_id).toBe(`${SCHEDULED_AT}-${STARTED_AT_MS}`);
    expect([...tick.planned_sources].sort()).toEqual(
      [...TRIP_UPDATE_FEEDS.map(([n]) => n), 'alerts', 'gtfs_static'].sort(),
    );
    expect(fetchOf(tick, 'gtfs_static')).toMatchObject({ method: 'HEAD', outcome: 'ok', http_etag: '"static"' });

    const { index, members } = await indexAt(SCHEDULED_AT, STARTED_AT_MS);
    expect(members).toHaveLength(1 + TRIP_UPDATE_FEEDS.length);
    for (const [name] of TRIP_UPDATE_FEEDS) {
      const f = fetchOf(tick, name);
      expect(f.outcome).toBe('ok');
      expect(f.entity_count).toBeGreaterThan(0);
      expect(f.header_timestamp).toBeGreaterThan(1_792_000_000);
      expect(f.http_date).toBe(Date.parse(HEADERS.date) / 1000);
      expect(f.payload_key).toBe(bundleKey(SCHEDULED_AT, STARTED_AT_MS));
      const member = members[f.payload_member!]!;
      expect(member).toEqual(GTFSRT_BODIES[name]);
      expect(sha256(member)).toBe(f.payload_sha256);
      expect(index.fetches.find((x) => x.source === name)).toEqual(f);
    }

    expect(fetchOf(tick, 'alerts')).toMatchObject({
      outcome: 'ok',
      payload_key: alertsKey,
      payload_member: null,
      payload_sha256: sha256(ALERTS_BODY),
      body_bytes: ALERTS_BODY.length,
      entity_count: 185,
      header_timestamp: 1_792_518_837,
      content_type: 'application/json',
    });
    expect(new Uint8Array(gunzipSync(await bytesAt(alertsKey)))).toEqual(ALERTS_BODY);
  });

  test('a failed bundle PUT is counted in archive/health/, and the tick record still lands', async () => {
    stubUpstream(liveMinute);
    const spy = spyBucket();
    spy.faults.put = `${OBSERVATIONS_PREFIX}raw/gtfsrt/`;
    await runScheduled(spy.bucket, SCHEDULED_AT, STARTED_AT_MS);

    const tick = await tickAt(SCHEDULED_AT, STARTED_AT_MS);
    for (const [name] of TRIP_UPDATE_FEEDS) {
      expect(fetchOf(tick, name)).toMatchObject({ payload_key: null, payload_member: null });
      expect(fetchOf(tick, name).store_error).toMatch(/injected put failure/);
    }
    await expectHealth(SCHEDULED_AT, { observations_bundle: 1 });
  });

  test('a failed tick PUT is counted in archive/health/', async () => {
    stubUpstream(liveMinute);
    const spy = spyBucket();
    spy.faults.put = `${OBSERVATIONS_PREFIX}ticks/`;
    await runScheduled(spy.bucket, SCHEDULED_AT, STARTED_AT_MS);

    expect(await keysUnder(`${OBSERVATIONS_PREFIX}ticks/`)).toEqual([]);
    expect(await keysUnder(`${OBSERVATIONS_PREFIX}raw/gtfsrt/`)).toHaveLength(1);
    await expectHealth(SCHEDULED_AT, { observations_tick: 1 });
  });

  test('a failed blob HEAD is a miss: the blob is still written and nothing is counted', async () => {
    stubUpstream(liveMinute);
    const spy = spyBucket();
    spy.faults.head = `${OBSERVATIONS_PREFIX}raw/blob/`;
    await runScheduled(spy.bucket, SCHEDULED_AT, STARTED_AT_MS);

    const alertsKey = blobKey(utcDay(SCHEDULED_AT), sha256(ALERTS_BODY), 'json');
    expect(fetchOf(await tickAt(SCHEDULED_AT, STARTED_AT_MS), 'alerts')).toMatchObject({
      payload_key: alertsKey, store_error: null,
    });
    expect(new Uint8Array(gunzipSync(await bytesAt(alertsKey)))).toEqual(ALERTS_BODY);
    await expectHealth(SCHEDULED_AT, {});
  });

  test('overlapping invocations for one cron minute write separate keys', async () => {
    stubUpstream(liveMinute);
    const later = STARTED_AT_MS + 2_000;
    await Promise.all([
      recordMinute(r2, { startedAtMs: STARTED_AT_MS }),
      recordMinute(r2, { startedAtMs: later }),
    ]);
    expect(await keysUnder(`${OBSERVATIONS_PREFIX}ticks/`)).toEqual([
      tickKey(SCHEDULED_AT, STARTED_AT_MS),
      tickKey(SCHEDULED_AT, later),
    ]);
    expect(await keysUnder(`${OBSERVATIONS_PREFIX}raw/gtfsrt/`)).toEqual([
      bundleKey(SCHEDULED_AT, STARTED_AT_MS),
      bundleKey(SCHEDULED_AT, later),
    ]);
    expect(fetchOf(await tickAt(SCHEDULED_AT, later), 'ace').payload_key).toBe(bundleKey(SCHEDULED_AT, later));
  });
});

describe('scheduled: a boundary tick', () => {
  const STARTED = BOUNDARY_AT * 1000 + 250;

  test('fetches alerts once for the recorder and the pipeline, and records the alpha winner\'s ENE and Socrata fetches', async () => {
    stubUpstream(liveMinute);
    const spy = spyBucket();
    await runScheduled(spy.bucket, BOUNDARY_AT, STARTED);

    expect(upstreamCalls.filter((u) => u === FEEDS.alerts)).toHaveLength(1);
    // Step 2 took that one fetch: the pipeline saw a fresh feed this tick.
    expect(JSON.parse(text(await bytesAt('state/last_seen.json')))).toMatchObject({
      alerts_at: BOUNDARY_AT,
      ene_at: BOUNDARY_AT,
    });
    expect(await keysUnder('archive/ene/')).toHaveLength(3);

    const tick = await tickAt(BOUNDARY_AT, STARTED);
    expect([...tick.planned_sources].sort()).toEqual(
      [...TRIP_UPDATE_FEEDS.map(([n]) => n), 'alerts', 'gtfs_static',
        'ene_current', 'ene_upcoming', 'ene_equipments', 'stations', 'entrances'].sort(),
    );
    expect(fetchOf(tick, 'ene_current')).toMatchObject({ outcome: 'ok', header_timestamp: null });
    expect(fetchOf(tick, 'ene_upcoming')).toMatchObject({ outcome: 'empty', entity_count: 0 });
    expect(fetchOf(tick, 'stations')).toMatchObject({ outcome: 'ok', entity_count: 1 });
    expect(fetchOf(tick, 'entrances')).toMatchObject({ outcome: 'empty', url: ENTRANCES_FEED });
    // ENE upcoming and equipments and entrances all answered `[]`: one blob
    // key, which all three point at.
    const day = utcDay(BOUNDARY_AT);
    const emptyKey = blobKey(day, sha256(new TextEncoder().encode('[]')), 'json');
    for (const source of ['ene_upcoming', 'ene_equipments', 'entrances'] as const) {
      expect(fetchOf(tick, source).payload_key).toBe(emptyKey);
    }
    expect(await keysUnder(`${OBSERVATIONS_PREFIX}raw/blob/`)).toEqual([
      blobKey(day, sha256(ALERTS_BODY), 'json'),
      blobKey(day, sha256(ENE_BODY), 'json'),
      blobKey(day, sha256(STATIONS_BODY), 'json'),
      emptyKey,
    ].sort());
    expect(new Uint8Array(gunzipSync(await bytesAt(fetchOf(tick, 'ene_current').payload_key!)))).toEqual(ENE_BODY);
    await expectHealth(BOUNDARY_AT, {});
  });

  test('a throw mid-tick still records the alerts fetch that was in flight, then rethrows', async () => {
    let releaseAlerts!: () => void;
    const alertsHeld = new Promise<void>((resolve) => { releaseAlerts = resolve; });
    stubUpstream(replacing(FEEDS.alerts, async (url, init) => {
      await alertsHeld;
      return liveMinute(url, init);
    }));
    const spy = spyBucket();
    // Step 1's state read fails while the alerts response is still pending.
    // The response is released on the next macrotask: the rejection reaches
    // the finally, and finish() starts waiting, within the current microtask
    // queue, so the bytes arrive only after the tick has begun to close.
    spy.faults.get = 'state/last_seen.json';
    spy.onGetFault = () => setImmediate(releaseAlerts);

    await expect(runScheduled(spy.bucket, BOUNDARY_AT, STARTED)).rejects.toThrow(/injected get failure/);

    const tick = await tickAt(BOUNDARY_AT, STARTED);
    expect(fetchOf(tick, 'alerts')).toMatchObject({
      outcome: 'ok', payload_key: blobKey(utcDay(BOUNDARY_AT), sha256(ALERTS_BODY), 'json'),
    });
    expect(await keysUnder(`${OBSERVATIONS_PREFIX}raw/blob/`)).toHaveLength(1);
  });
});

describe('fetch outcomes', () => {
  const ACE = TRIP_UPDATE_FEEDS[0]![1];

  /** One scheduled tick with `ace` answered by `ace`; its recorded fetch. */
  async function aceFetch(ace: Upstream): Promise<FeedFetch> {
    stubUpstream(replacing(ACE, ace));
    await runScheduled(r2, SCHEDULED_AT, STARTED_AT_MS);
    const tick = await tickAt(SCHEDULED_AT, STARTED_AT_MS);
    const { members } = await indexAt(SCHEDULED_AT, STARTED_AT_MS);
    const f = fetchOf(tick, 'ace');
    if (f.payload_member !== null) expect(sha256(members[f.payload_member]!)).toBe(f.payload_sha256);
    return f;
  }

  test('timeout: no status, no finish time, nothing stored', async () => {
    expect(await aceFetch(timeout)).toMatchObject({
      outcome: 'timeout', finished_at_ms: null, http_status: null, content_type: null,
      body_bytes: null, payload_sha256: null, payload_key: null, store_error: null,
    });
  });

  test('HTTP 500: http_error with the status, body not read', async () => {
    expect(await aceFetch(http500)).toMatchObject({
      outcome: 'http_error', http_status: 500, body_bytes: null, payload_key: null,
    });
  });

  test('empty 200: outcome empty, the zero bytes still stored', async () => {
    expect(await aceFetch(empty200)).toMatchObject({
      outcome: 'empty', http_status: 200, body_bytes: 0, entity_count: 0, header_timestamp: null,
      payload_sha256: sha256(new Uint8Array()), payload_key: bundleKey(SCHEDULED_AT, STARTED_AT_MS),
    });
  });

  test('garbage 200: unparseable, the bytes still stored', async () => {
    const f = await aceFetch(garbage200);
    expect(f).toMatchObject({
      outcome: 'unparseable', http_status: 200, body_bytes: GARBAGE.length, entity_count: null,
      header_timestamp: null, payload_sha256: sha256(GARBAGE),
    });
    expect(f.payload_member).not.toBeNull();
  });

  test('body read fails after a 200: network_error that keeps the status', async () => {
    const f = await aceFetch(bodyFails);
    expect(f).toMatchObject({
      outcome: 'network_error', http_status: 200, content_type: 'text/plain',
      body_bytes: null, payload_sha256: null, payload_key: null, store_error: null,
    });
    expect(f.finished_at_ms).not.toBeNull();
  });

  test('alerts that are not a feed object are unparseable, still stored', async () => {
    stubUpstream(replacing(FEEDS.alerts, async () => new Response('[1,2]', { headers: HEADERS })));
    await runScheduled(r2, SCHEDULED_AT, STARTED_AT_MS);
    const alerts = fetchOf(await tickAt(SCHEDULED_AT, STARTED_AT_MS), 'alerts');
    expect(alerts).toMatchObject({ outcome: 'unparseable', entity_count: null });
    expect(alerts.payload_key).not.toBeNull();
  });

  test('a failed blob PUT sets store_error and counts a write failure; the tick still lands', async () => {
    stubUpstream(liveMinute);
    const spy = spyBucket();
    spy.faults.put = `${OBSERVATIONS_PREFIX}raw/blob/`;
    await runScheduled(spy.bucket, SCHEDULED_AT, STARTED_AT_MS);
    const tick = await tickAt(SCHEDULED_AT, STARTED_AT_MS);
    expect(fetchOf(tick, 'alerts')).toMatchObject({ payload_key: null, payload_sha256: sha256(ALERTS_BODY) });
    expect(fetchOf(tick, 'alerts').store_error).toMatch(/injected put failure/);
    expect(fetchOf(tick, 'ace').payload_key).toBe(bundleKey(SCHEDULED_AT, STARTED_AT_MS));
    await expectHealth(SCHEDULED_AT, { observations_blob: 1 });
  });
});

describe('blob dedup within a UTC day', () => {
  test('the same alerts bytes are stored once a day, and again the next day', async () => {
    stubUpstream(liveMinute);
    const spy = spyBucket();
    const blobPuts = (): string[] => spy.puts.filter((k) => k.includes('/raw/blob/'));

    await recordMinute(spy.bucket, { startedAtMs: STARTED_AT_MS });
    vi.mocked(Date.now).mockReturnValue(STARTED_AT_MS + 60_000);
    await recordMinute(spy.bucket, { startedAtMs: STARTED_AT_MS + 60_000 });
    const today = blobKey(utcDay(SCHEDULED_AT), sha256(ALERTS_BODY), 'json');
    expect(blobPuts()).toEqual([today]);
    expect(fetchOf(await tickAt(SCHEDULED_AT, STARTED_AT_MS + 60_000), 'alerts')).toMatchObject({
      payload_key: today, store_error: null,
    });

    // The blob is filed under the UTC day the fetch finished.
    const nextDayMs = (Math.floor(SCHEDULED_AT / 86_400) + 1) * 86_400_000 + 5_000;
    vi.mocked(Date.now).mockReturnValue(nextDayMs);
    await recordMinute(spy.bucket, { startedAtMs: STARTED_AT_MS + 120_000 });
    const tomorrow = blobKey(utcDay(nextDayMs / 1000), sha256(ALERTS_BODY), 'json');
    expect(blobPuts()).toEqual([today, tomorrow]);
    expect(new Uint8Array(gunzipSync(await bytesAt(tomorrow)))).toEqual(ALERTS_BODY);
  });
});

describe('recorder output fixture for the Pydantic validators', () => {
  // One off-boundary tick with every outcome the recorder can give a GTFS-RT
  // feed. tests/test_recorder_output.py runs it through CollectorTick and
  // TickBundleIndex, whose cross-field rules the JSON Schema cannot express.
  // Regenerate: UPDATE_RECORDER_FIXTURE=1 bun run test test/recorder.test.ts
  test('tests/fixtures/recorder_output.json is what the recorder writes', async () => {
    const answers: Record<string, Upstream> = {
      ace: timeout, bdfm: http500, g: empty200, jz: garbage200, l: bodyFails,
    };
    stubUpstream(async (url, init) => {
      const feed = TRIP_UPDATE_FEEDS.find(([, u]) => u === url);
      const answer = feed ? answers[feed[0]] : undefined;
      return answer ? answer(url, init) : liveMinute(url, init);
    });
    await runScheduled(r2, SCHEDULED_AT, STARTED_AT_MS);
    const output = {
      collector_tick: await tickAt(SCHEDULED_AT, STARTED_AT_MS),
      tick_bundle_index: (await indexAt(SCHEDULED_AT, STARTED_AT_MS)).index,
    };
    const rendered = `${JSON.stringify(output, null, 2)}\n`;
    if (process.env.UPDATE_RECORDER_FIXTURE) writeFileSync(OUTPUT_FIXTURE, rendered);
    expect(readFileSync(OUTPUT_FIXTURE, 'utf8'), 'stale: rerun with UPDATE_RECORDER_FIXTURE=1').toBe(rendered);
  });
});
