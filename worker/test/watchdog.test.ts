/**
 * The backup trigger (watchdog.ts) in real workerd through Miniflare: the
 * production Worker and Durable Object, bundled with test hooks
 * (watchdog_harness.ts), a SQLite-backed object, an R2 bucket, and upstream
 * feeds answered from the 2026-10-20 captured minute (test/fixtures/observations/).
 *
 * The object's clock runs at an offset the test sets, so "now" can sit at any
 * point of a minute; alarms are fired through the handler directly, except in
 * the test that waits for the runtime to fire one. Every minute here runs the
 * whole pipeline, so the assertions read what landed in R2: tick records (one
 * per run, naming its trigger) and the published objects.
 *
 * A cron that is slow, or that stops renewing its lease, is a real `scheduled`
 * run held on one upstream answer (`holdFirst`) while the test moves the
 * object's clock and fires alarms; the harness shortens the lease heartbeat so
 * minutes of renewals take a fraction of a second.
 *
 * MEASURE_WAKE_CPU=1 also measures the CPU of an idle wake; run that test alone,
 * since it reads the workerd process CPU from `ps`.
 */

/// <reference types="node" />

import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gunzipSync } from 'node:zlib';

import Ajv2020 from 'ajv/dist/2020';
import { build } from 'esbuild';
import { Miniflare } from 'miniflare';
import { afterEach, beforeAll, describe, expect, test } from 'vitest';

import collectorTickSchema from '../../schema/model/s1/observations/collector_tick.schema.json';
import type { CollectorTick } from '../../shared/model/s1/observations/collector_tick';
import { FEEDS, STATIONS_FEED, TRIP_UPDATE_FEEDS } from '../src/fetch';
import { OBSERVATIONS_PREFIX } from '../src/observation_keys';
import { CRON_OFFSET_MS, DEADLINE_MS, MINUTE_MS, nextDeadline } from '../src/watchdog';

const HERE = dirname(fileURLToPath(import.meta.url));
const ajv = new Ajv2020({ allErrors: true, strict: false });
const validateTick = ajv.compile<CollectorTick>(collectorTickSchema);

const fixture = (name: string): Uint8Array =>
  new Uint8Array(gunzipSync(readFileSync(join(HERE, 'fixtures', 'observations', `${name}.gz`))));
const GTFSRT: Map<string, Uint8Array> = new Map(TRIP_UPDATE_FEEDS.map(([name, url]) => [url, fixture(`${name}.pb`)]));
const ALERTS = fixture('alerts.json');
const ENE = fixture('ene_current.json');

/** The first request to `url` waits until released; every later one is answered. */
interface Held {
  url: string;
  used: boolean;
  enter: () => void;
  answer: Promise<Response | null>;
}

/** Upstream state: alerts requests hang (never answer) while `hangAlerts` is set. */
const upstream: { hangAlerts: boolean; alertsRequests: number; held: Held | null } = {
  hangAlerts: false,
  alertsRequests: 0,
  held: null,
};

/**
 * Hold the first request to `url` (the cron's: it runs first) until
 * `release`, which answers it as usual, or with `answer`.
 */
function holdFirst(url: string): { entered: Promise<void>; release: (answer?: Response) => void } {
  let enter = (): void => {};
  const entered = new Promise<void>((resolve) => {
    enter = resolve;
  });
  let release: (answer: Response | null) => void = () => {};
  const answer = new Promise<Response | null>((resolve) => {
    release = resolve;
  });
  upstream.held = { url, used: false, enter, answer };
  return { entered, release: (r) => release(r ?? null) };
}

async function answerUpstream(request: Request): Promise<Response> {
  const headers = { date: new Date().toUTCString() };
  if (request.method === 'HEAD') return new Response(null, { headers: { ...headers, etag: '"static"' } });
  const held = upstream.held;
  if (held !== null && !held.used && request.url === held.url) {
    held.used = true;
    held.enter();
    const answer = await held.answer;
    if (answer !== null) return answer;
  }
  const gtfsrt = GTFSRT.get(request.url);
  if (gtfsrt) return new Response(gtfsrt, { headers });
  if (request.url === FEEDS.alerts) {
    upstream.alertsRequests += 1;
    if (upstream.hangAlerts) return new Promise<Response>(() => {});
    return new Response(ALERTS, { headers: { ...headers, 'content-type': 'application/json' } });
  }
  if (request.url === FEEDS.ene_current) return new Response(ENE, { headers });
  return new Response('[]', { headers });
}

let script = '';
beforeAll(async () => {
  const out = await build({
    entryPoints: [join(HERE, 'watchdog_harness.ts')],
    bundle: true,
    format: 'esm',
    platform: 'browser',
    target: 'es2022',
    write: false,
    logLevel: 'error',
  });
  script = out.outputFiles[0]!.text;
});

let mf: Miniflare | null = null;
afterEach(async () => {
  upstream.hangAlerts = false;
  upstream.alertsRequests = 0;
  upstream.held = null;
  await mf?.dispose();
  mf = null;
});

function start(): Miniflare {
  mf = new Miniflare({
    modules: true,
    script,
    // The newest date this Miniflare's workerd knows; production is 2026-05-13.
    compatibilityDate: '2024-12-30',
    durableObjects: { MINUTE_CLOCK: { className: 'HarnessClock', useSQLite: true } },
    r2Buckets: ['MOMENTARILY'],
    outboundService: answerUpstream,
  });
  return mf;
}

// Real waits, only to poll for work running in the workerd process (a hung
// cron storing its bundle, the runtime firing an alarm, a lease heartbeat):
// no fake timer reaches that clock.
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function harness<T = Record<string, unknown>>(m: Miniflare, path: string): Promise<T> {
  const res = await m.dispatchFetch(`http://worker/harness/${path}`);
  expect(res.status, path).toBe(200);
  return (await res.json()) as T;
}

interface StoredClaim {
  trigger: string;
  done: boolean;
  run?: string;
  leaseUntil?: number;
  missed?: string;
}

interface ClockState {
  now: number;
  alarm: number | null;
  claims: Record<string, StoredClaim>;
}

/** A claim as the tests compare it: `missed` is undefined unless marked. */
interface ClaimView {
  trigger: string;
  done: boolean;
  missed?: string | undefined;
}

const view = (held: StoredClaim): ClaimView => ({ trigger: held.trigger, done: held.done, missed: held.missed });

/** Who holds each claimed slot, whether its run ended, and why it was missed, oldest first. */
async function claims(m: Miniflare): Promise<ClaimView[]> {
  const state = await harness<ClockState>(m, 'state');
  return Object.keys(state.claims)
    .sort()
    .map((key) => view(state.claims[key]!));
}

async function claimOf(m: Miniflare, slot: number): Promise<ClaimView | undefined> {
  const held = (await harness<ClockState>(m, 'state')).claims[`slot:${String(slot).padStart(12, '0')}`];
  return held === undefined ? undefined : view(held);
}

const MISSED_NO_CLAIM: ClaimView = { trigger: 'alarm', done: true, missed: 'no_claim' };
const MISSED_LEASE_EXPIRED: ClaimView = { trigger: 'alarm', done: true, missed: 'lease_expired' };

/**
 * A 5-minute boundary slot 6-10 minutes ahead of the real clock, so the whole
 * pipeline runs. The object stores its alarm as an absolute instant, which the
 * runtime reads on the real clock; with every slot in the real future, no
 * alarm the object sets can fire by itself during a test.
 */
function boundarySlot(): number {
  const slot = Math.floor(Date.now() / MINUTE_MS) + 10;
  return slot - (slot % 5);
}

/** Put the object's clock `intoMinute` ms into `slot`. */
async function clockAt(m: Miniflare, slot: number, intoMinute: number): Promise<void> {
  await harness(m, `offset?ms=${slot * MINUTE_MS + intoMinute - Date.now()}`);
}

async function cron(m: Miniflare, slot: number): Promise<void> {
  const worker = await m.getWorker();
  await worker.scheduled({ scheduledTime: new Date(slot * MINUTE_MS + CRON_OFFSET_MS), cron: '* * * * *' });
}

/** Start the cron of `slot` without waiting for it. */
async function cronInBackground(m: Miniflare, slot: number): Promise<void> {
  const worker = await m.getWorker();
  return worker.scheduled({ scheduledTime: new Date(slot * MINUTE_MS + CRON_OFFSET_MS), cron: '* * * * *' }).then(() => {});
}

async function keys(m: Miniflare, prefix: string): Promise<string[]> {
  const r2 = await m.getR2Bucket('MOMENTARILY');
  const out: string[] = [];
  let cursor: string | undefined;
  do {
    const listed = await r2.list({ prefix, ...(cursor ? { cursor } : {}) });
    out.push(...listed.objects.map((o) => o.key));
    cursor = listed.truncated ? listed.cursor : undefined;
  } while (cursor);
  return out;
}

/** Every object in the bucket and its etag. */
async function bucketState(m: Miniflare): Promise<Map<string, string>> {
  const r2 = await m.getR2Bucket('MOMENTARILY');
  const out = new Map<string, string>();
  let cursor: string | undefined;
  do {
    const listed = await r2.list({ ...(cursor ? { cursor } : {}) });
    for (const o of listed.objects) out.set(o.key, o.etag);
    cursor = listed.truncated ? listed.cursor : undefined;
  } while (cursor);
  return out;
}

/** The keys written (created or replaced) since `before` was taken. */
async function writtenSince(m: Miniflare, before: Map<string, string>): Promise<string[]> {
  return [...(await bucketState(m))].filter(([key, etag]) => before.get(key) !== etag).map(([key]) => key);
}

async function text(m: Miniflare, key: string): Promise<string | undefined> {
  return (await (await m.getR2Bucket('MOMENTARILY')).get(key))?.text();
}

const traceKey = (slot: number): string => {
  const scheduledAt = slot * 60 + CRON_OFFSET_MS / 1000;
  return `archive/trace/${new Date(scheduledAt * 1000).toISOString().slice(0, 10)}/${scheduledAt}.json`;
};

/** Every tick record of `slot`, each validated against the exported schema. */
async function ticks(m: Miniflare, slot: number): Promise<CollectorTick[]> {
  const r2 = await m.getR2Bucket('MOMENTARILY');
  const out: CollectorTick[] = [];
  for (const key of await keys(m, `${OBSERVATIONS_PREFIX}ticks/`)) {
    const tick: unknown = JSON.parse(await (await r2.get(key))!.text());
    expect(validateTick(tick), JSON.stringify(validateTick.errors)).toBe(true);
    const t = tick as CollectorTick;
    if (Math.floor(t.scheduled_at / 60) === slot) out.push(t);
  }
  return out;
}

const triggers = async (m: Miniflare, slot: number): Promise<string[]> =>
  (await ticks(m, slot)).map((t) => t.trigger).sort();

/** Poll until `ready` holds (work running in workerd), failing after ~5 s. */
async function until(ready: () => Promise<boolean>): Promise<void> {
  for (let i = 0; i < 100; i += 1) {
    if (await ready()) return;
    await sleep(50);
  }
  throw new Error('timed out waiting for the Worker');
}

describe('watchdog: one run per minute', () => {
  test('a minute only the alarm sees runs once, by the alarm, at the cron offset', async () => {
    const m = start();
    const slot = boundarySlot();
    await clockAt(m, slot, DEADLINE_MS + 5);
    await harness(m, 'alarm');
    await harness(m, 'alarm'); // a retry of the same wake
    const [tick, ...rest] = await ticks(m, slot);
    expect(rest).toEqual([]);
    expect(tick!.trigger).toBe('alarm');
    expect(tick!.scheduled_at).toBe(slot * 60 + CRON_OFFSET_MS / 1000);
    // The whole minute ran: the 5-minute publish as well as the record.
    expect(await keys(m, 'v1/snapshot.json')).toEqual(['v1/snapshot.json']);
    expect(await claims(m)).toEqual([{ trigger: 'alarm', done: true }]);
    // Re-armed for the next minute's deadline.
    expect((await harness<ClockState>(m, 'state')).alarm).toBe((slot + 1) * MINUTE_MS + DEADLINE_MS);
  }, 30_000);

  test('cron and alarm in one minute: the cron runs it, the alarm and a duplicated cron do nothing', async () => {
    const m = start();
    const slot = boundarySlot();
    await clockAt(m, slot, CRON_OFFSET_MS + 2_000);
    await cron(m, slot);
    await cron(m, slot); // a duplicated cron dispatch
    await clockAt(m, slot, DEADLINE_MS + 5);
    await harness(m, 'alarm');
    expect(await triggers(m, slot)).toEqual(['cron']);
  }, 30_000);

  test('a cron arriving after the alarm ran its minute is a no-op', async () => {
    const m = start();
    const slot = boundarySlot();
    await clockAt(m, slot, DEADLINE_MS + 5);
    await harness(m, 'alarm');
    await clockAt(m, slot + 3, 0); // three minutes late
    await cron(m, slot);
    expect(await triggers(m, slot)).toEqual(['alarm']);
  }, 30_000);

  test('a cron that claims and dies before storing anything: its lease runs out and the next wake marks the minute missed, never running it', async () => {
    const m = start();
    const slot = boundarySlot();
    await clockAt(m, slot, CRON_OFFSET_MS + 2_000);
    expect(await harness(m, `claim?slot=${slot}`)).toEqual({ won: true });
    // At its own deadline the lease (claimed at :21, 60 s) still holds.
    await clockAt(m, slot, DEADLINE_MS + 5);
    await harness(m, 'alarm');
    expect(await triggers(m, slot)).toEqual([]);
    expect(await claimOf(m, slot)).toEqual({ trigger: 'cron', done: false });
    // At the next deadline it has expired: the minute is past, so it is
    // marked missed (which fences the dead run), not run with the present feeds.
    await clockAt(m, slot + 1, DEADLINE_MS + 5);
    await harness(m, 'alarm');
    await harness(m, 'alarm');
    expect(await triggers(m, slot)).toEqual([]);
    expect(await claimOf(m, slot)).toEqual(MISSED_LEASE_EXPIRED);
    expect(await keys(m, 'v1/snapshot.json')).toEqual([]);
    // That wake ran its own minute, which nobody claimed.
    expect(await triggers(m, slot + 1)).toEqual(['alarm']);
  }, 30_000);
});

describe('watchdog: a slow cron is never taken over; a stopped one is fenced', () => {
  /**
   * Move the object's clock from :21 of `slot` to `seconds` later, 20 s of
   * object time per step plus every :45 deadline on the way, giving the
   * cron's heartbeat (50 ms) real time to renew at each step, and firing the
   * alarm at each deadline. The real waits are the heartbeat's: it runs on
   * workerd's clock, which no fake timer reaches.
   */
  async function liveFor(m: Miniflare, slot: number, seconds: number): Promise<void> {
    const from = CRON_OFFSET_MS + 2_000;
    const to = from + seconds * 1_000;
    const points = new Set<number>();
    for (let t = from + 20_000; t < to; t += 20_000) points.add(t);
    for (let d = DEADLINE_MS; d <= to; d += MINUTE_MS) if (d > from) points.add(d);
    points.add(to);
    for (const t of [...points].sort((a, b) => a - b)) {
      await clockAt(m, slot, t);
      await sleep(150);
      if (t % MINUTE_MS === DEADLINE_MS) await harness(m, 'alarm');
    }
  }

  test.each([
    [50, 'alerts', FEEDS.alerts],
    [120, 'stations', STATIONS_FEED],
    [240, 'alerts', FEEDS.alerts],
  ])(
    'a 5-minute tick whose cron runs %i s, held on the %s feed: the cron runs it alone, and alpha advances once',
    async (seconds, _feed, url) => {
      const m = start();
      const slot = boundarySlot();
      await harness(m, 'heartbeat?ms=50');
      await clockAt(m, slot, CRON_OFFSET_MS + 2_000);
      const held = holdFirst(url);
      const running = cronInBackground(m, slot);
      await held.entered;
      await liveFor(m, slot, seconds);
      expect(await triggers(m, slot)).toEqual([]); // the cron is still running
      expect(await claimOf(m, slot)).toEqual({ trigger: 'cron', done: false });
      held.release();
      await running;

      expect(await triggers(m, slot)).toEqual(['cron']);
      expect(await claimOf(m, slot)).toEqual({ trigger: 'cron', done: true });
      // One alpha advance, for this tick, and its snapshot published.
      const r2 = await m.getR2Bucket('MOMENTARILY');
      expect((await r2.head('state/alpha.json'))?.customMetadata).toEqual({ tick: String(slot * 60 + 19) });
      expect(await keys(m, 'v1/snapshot.json')).toEqual(['v1/snapshot.json']);
      // The minute's trace is the cron's, complete.
      expect(JSON.parse((await text(m, traceKey(slot)))!).fresh_feeds).toHaveLength(TRIP_UPDATE_FEEDS.length);
    },
    60_000,
  );

  test('a cron that stops renewing after storing its raw feeds: its minute is marked missed at the next deadline, not run again; when it returns it writes nothing', async () => {
    const m = start();
    const slot = boundarySlot();
    await harness(m, 'heartbeat?ms=50');
    await harness(m, 'fail?mode=renew-500'); // its renewals never reach the object
    await clockAt(m, slot, CRON_OFFSET_MS + 2_000);
    // It stores the bundle and the per-minute surfaces, then waits on the
    // alerts feed the 5-minute pipeline needs.
    const held = holdFirst(FEEDS.alerts);
    const stalled = cronInBackground(m, slot);
    await until(async () => (await keys(m, `${OBSERVATIONS_PREFIX}raw/gtfsrt/`)).length === 1);
    await held.entered;
    expect(await keys(m, 'v1/snapshot.json')).toEqual([]);

    await clockAt(m, slot, DEADLINE_MS + 5);
    await harness(m, 'alarm');
    expect(await triggers(m, slot)).toEqual([]); // lease still held at its own deadline
    await clockAt(m, slot + 1, DEADLINE_MS + 5);
    await harness(m, 'alarm');
    await harness(m, 'alarm');
    expect(await triggers(m, slot)).toEqual([]);
    expect(await claimOf(m, slot)).toEqual(MISSED_LEASE_EXPIRED);
    expect(await keys(m, 'v1/snapshot.json')).toEqual([]);
    expect(await triggers(m, slot + 1)).toEqual(['alarm']);
    expect(await keys(m, `${OBSERVATIONS_PREFIX}raw/gtfsrt/`)).toHaveLength(2); // cron (slot), alarm (slot + 1)

    const before = await bucketState(m);
    await harness(m, 'fail?mode=none'); // its next renewal reaches the object: fenced
    held.release();
    await stalled;
    expect(await writtenSince(m, before)).toEqual([]);
    expect(await triggers(m, slot)).toEqual([]);
    expect(await claimOf(m, slot)).toEqual(MISSED_LEASE_EXPIRED);
  }, 60_000);

  test('a cron that stops renewing after committing alpha, during publish: the next wake does not advance alpha for its minute; the cron, back, writes nothing', async () => {
    const m = start();
    const slot = boundarySlot();
    await harness(m, 'heartbeat?ms=50');
    await harness(m, 'fail?mode=renew-500');
    await clockAt(m, slot, CRON_OFFSET_MS + 2_000);
    // Step 8c's daily stations fetch: after alpha (5) and the snapshot (6).
    const held = holdFirst(STATIONS_FEED);
    const stalled = cronInBackground(m, slot);
    await held.entered;
    const r2 = await m.getR2Bucket('MOMENTARILY');
    const alpha = (await r2.head('state/alpha.json'))!.etag;
    const snapshot = (await r2.head('v1/snapshot.json'))!.etag;
    const trace = await text(m, traceKey(slot));

    await clockAt(m, slot + 1, DEADLINE_MS + 5);
    await harness(m, 'alarm');
    expect(await claimOf(m, slot)).toEqual(MISSED_LEASE_EXPIRED);
    expect(await triggers(m, slot)).toEqual([]);
    expect((await r2.head('state/alpha.json'))!.etag).toBe(alpha);
    expect((await r2.head('v1/snapshot.json'))!.etag).toBe(snapshot);
    expect(await text(m, traceKey(slot))).toBe(trace);

    const before = await bucketState(m);
    await harness(m, 'fail?mode=none');
    held.release();
    await stalled;
    expect(await writtenSince(m, before)).toEqual([]);
    expect(await keys(m, 'state/stations.json')).toEqual([]);
  }, 60_000);

  test('the fence is not trusted alone: a cron whose ownership check passes after its minute was marked missed cannot replace a newer run\'s alpha, snapshot, arrivals or platform waits', async () => {
    // The 'renew-lie' fault answers every renewal "held" without extending
    // the lease: the old run believes it owns its minute after the alarm
    // marked it missed, as a run does whose check passed just before the mark.
    const m = start();
    const slot = boundarySlot();
    await harness(m, 'heartbeat?ms=50');
    await harness(m, 'fail?mode=renew-lie');
    await clockAt(m, slot, CRON_OFFSET_MS + 2_000);
    const firstFeed = TRIP_UPDATE_FEEDS[0]![1];
    const held = holdFirst(firstFeed);
    const late = cronInBackground(m, slot);
    await held.entered;
    await clockAt(m, slot, DEADLINE_MS + 5);
    await harness(m, 'alarm'); // its lease still holds

    // Five minutes on, the next 5-minute tick's wake marks the cron's minute
    // missed and runs its own, fetched after the cron's.
    await clockAt(m, slot + 5, DEADLINE_MS + 5);
    await harness(m, 'alarm');
    expect(await claimOf(m, slot)).toEqual(MISSED_LEASE_EXPIRED);
    expect(await triggers(m, slot + 5)).toEqual(['alarm']);
    const r2 = await m.getR2Bucket('MOMENTARILY');
    const alpha = (await r2.head('state/alpha.json'))!;
    expect(alpha.customMetadata).toEqual({ tick: String((slot + 5) * 60 + 19) });
    const guarded = ['v1/snapshot.json', 'state/last_seen.json', 'v1/arrivals.json', 'state/station_wait.json'];
    const etags = new Map<string, string>();
    for (const key of guarded) etags.set(key, (await r2.head(key))!.etag);

    // The late cron gets a failed feed and runs its whole minute.
    held.release(new Response('late upstream failure', { status: 503 }));
    await late;
    expect(await triggers(m, slot)).toEqual(['cron']);
    expect((await r2.head('state/alpha.json'))!.etag).toBe(alpha.etag);
    for (const key of guarded) expect((await r2.head(key))!.etag, key).toBe(etags.get(key));
    expect(JSON.parse((await text(m, 'v1/arrivals.json'))!).fresh_feeds).toHaveLength(TRIP_UPDATE_FEEDS.length);
  }, 60_000);

  test('a cron taken over after its renewals failed for 60 s: when it finishes, its older observation does not replace the alarm\'s arrivals or platform waits', async () => {
    // A late wake (the alarm fires 45 s after the minute's deadline) finds
    // the current minute's cron lease expired and runs the minute itself.
    const m = start();
    const slot = boundarySlot();
    await harness(m, 'heartbeat?ms=50');
    await clockAt(m, slot, CRON_OFFSET_MS + 2_000);
    const held = holdFirst(TRIP_UPDATE_FEEDS[0]![1]);
    const late = cronInBackground(m, slot);
    await held.entered;
    await harness(m, 'fail?mode=renew-500');
    await clockAt(m, slot + 1, 30_000);
    await sleep(300);
    await harness(m, 'alarm');
    expect(await triggers(m, slot)).toEqual(['alarm']);
    const r2 = await m.getR2Bucket('MOMENTARILY');
    const guarded = ['v1/arrivals.json', 'state/station_wait.json', 'state/alpha.json'];
    const etags = new Map<string, string>();
    for (const key of guarded) etags.set(key, (await r2.head(key))!.etag);

    // Its renewals still fail, so it runs on to the end, one feed short.
    held.release(new Response('late upstream failure', { status: 503 }));
    await late;
    await harness(m, 'fail?mode=none');
    expect(await triggers(m, slot)).toEqual(['alarm', 'cron']);
    for (const key of guarded) expect((await r2.head(key))!.etag, key).toBe(etags.get(key));
    expect(JSON.parse((await text(m, 'v1/arrivals.json'))!).fresh_feeds).toHaveLength(TRIP_UPDATE_FEEDS.length);
  }, 60_000);
});

describe('watchdog: the alarm loop', () => {
  test('a cron claim arms a missing alarm; the runtime fires it and it runs the next unclaimed minute', async () => {
    // On the real clock (no offset): the runtime fires alarms by it. Waits up
    // to a minute for the next :45.
    const m = start();
    const now = Date.now();
    expect((await harness<ClockState>(m, 'state')).alarm).toBeNull();
    const previous = Math.floor(now / MINUTE_MS) - 1;
    await cron(m, previous);
    const armed = (await harness<ClockState>(m, 'state')).alarm!;
    expect(armed).toBeGreaterThan(now);
    expect(armed).toBeLessThanOrEqual(nextDeadline(Date.now()));
    expect(armed % MINUTE_MS).toBe(DEADLINE_MS);
    const served = Math.floor((armed - DEADLINE_MS) / MINUTE_MS);
    let ran: string[] = [];
    for (let i = 0; i < 400 && ran.length === 0; i += 1) {
      await sleep(250);
      ran = await triggers(m, served);
    }
    expect(await triggers(m, previous)).toEqual(['cron']);
    expect(ran).toEqual(['alarm']);
    expect(Date.now()).toBeGreaterThanOrEqual(armed);
    // It re-armed itself for the minute after.
    expect((await harness<ClockState>(m, 'state')).alarm).toBe(armed + MINUTE_MS);
  }, 150_000);

  test('a cron claim re-arms an alarm more than a minute overdue', async () => {
    const m = start();
    const slot = boundarySlot();
    await clockAt(m, slot, CRON_OFFSET_MS);
    await harness(m, `claim?slot=${slot}`);
    const stored = (await harness<ClockState>(m, 'state')).alarm!;
    expect(stored).toBe(slot * MINUTE_MS + DEADLINE_MS);
    // The stored alarm never fired (as local workerd does after a restart).
    await clockAt(m, slot + 1, DEADLINE_MS - 1_000);
    await harness(m, `claim?slot=${slot + 1}`);
    expect((await harness<ClockState>(m, 'state')).alarm).toBe(stored); // under a minute overdue: left alone
    await clockAt(m, slot + 2, DEADLINE_MS + 1);
    const { now } = await harness<ClockState>(m, 'state');
    await harness(m, `claim?slot=${slot + 2}`);
    expect((await harness<ClockState>(m, 'state')).alarm).toBe(nextDeadline(now));
  }, 30_000);

  test('an alarm run that throws is marked done; a retry of the wake does not run it again', async () => {
    const m = start();
    const slot = boundarySlot();
    await clockAt(m, slot, DEADLINE_MS + 5);
    await harness(m, 'fail?mode=state-read'); // the 5-minute pipeline throws on its state reads
    expect(await harness(m, 'alarm')).toEqual({ ran: true }); // the handler itself does not throw
    await harness(m, 'fail?mode=none');
    await harness(m, 'alarm');
    expect(await triggers(m, slot)).toEqual(['alarm']); // its finally still wrote the one tick
    expect(await keys(m, 'v1/snapshot.json')).toEqual([]);
    expect(await claims(m)).toEqual([{ trigger: 'alarm', done: true }]);
    expect((await harness<ClockState>(m, 'state')).alarm).toBe((slot + 1) * MINUTE_MS + DEADLINE_MS);
  }, 30_000);

  test('an alarm still running its minute: a second wake for that minute runs nothing', async () => {
    const m = start();
    const slot = boundarySlot();
    await clockAt(m, slot, DEADLINE_MS + 5);
    upstream.hangAlerts = true;
    const first = m.dispatchFetch('http://worker/harness/alarm').catch(() => null);
    for (let i = 0; i < 50 && upstream.alertsRequests === 0; i += 1) await sleep(100);
    expect(upstream.alertsRequests).toBe(1);
    upstream.hangAlerts = false;
    await harness(m, 'alarm');
    expect(upstream.alertsRequests).toBe(1); // the second wake fetched nothing
    expect(await claims(m)).toEqual([{ trigger: 'alarm', done: false }]);
    void first;
  }, 30_000);

  test('a minute whose deadline passed while the alarm was running another is marked missed by the next wake, which runs only its own', async () => {
    const m = start();
    const slot = boundarySlot();
    await clockAt(m, slot, DEADLINE_MS + 5);
    // The alarm's run of `slot` lasts until :50 two minutes on (130 s, inside
    // the recorded run times): the deadline of `slot + 1` passes meanwhile,
    // with no cron to claim it.
    const held = holdFirst(FEEDS.alerts);
    const long = m.dispatchFetch('http://worker/harness/alarm');
    await held.entered;
    await clockAt(m, slot + 2, DEADLINE_MS + 5);
    held.release();
    await long;
    expect(await triggers(m, slot)).toEqual(['alarm']);
    // The runtime fires the next alarm at once. The feeds now show the
    // present, so `slot + 1` cannot be recorded any more: it is marked missed.
    await harness(m, 'alarm');
    expect(await triggers(m, slot + 1)).toEqual([]);
    expect(await claimOf(m, slot + 1)).toEqual(MISSED_NO_CLAIM);
    expect(await triggers(m, slot + 2)).toEqual(['alarm']);
  }, 30_000);

  test('minutes whose deadlines passed while no alarm ran (the object unreachable) are marked missed, not run late; a late cron for one does nothing', async () => {
    const m = start();
    const slot = boundarySlot();
    await clockAt(m, slot, DEADLINE_MS + 5);
    await harness(m, 'alarm');
    await clockAt(m, slot + 3, DEADLINE_MS + 5);
    await harness(m, 'alarm');
    expect(await triggers(m, slot + 1)).toEqual([]);
    expect(await triggers(m, slot + 2)).toEqual([]);
    expect(await triggers(m, slot + 3)).toEqual(['alarm']);
    expect(await claims(m)).toEqual([
      { trigger: 'alarm', done: true },
      MISSED_NO_CLAIM,
      MISSED_NO_CLAIM,
      { trigger: 'alarm', done: true },
    ]);
    await cron(m, slot + 1);
    expect(await triggers(m, slot + 1)).toEqual([]);
  }, 30_000);
});

describe('watchdog: a wake runs the current minute only, never a past one', () => {
  /**
   * Run the alarm of `slot` and hold it on the alerts feed while the object's
   * clock moves to :50 of `until` (no cron runs meanwhile), then release it.
   * The runtime fires the next alarm at once; the caller fires it. The hold
   * lasts over a second of real time, so the next wake's run observes in a
   * later second than this one's, as it would minutes later in production.
   */
  async function longRun(m: Miniflare, slot: number, until: number): Promise<void> {
    await clockAt(m, slot, DEADLINE_MS + 5);
    const held = holdFirst(FEEDS.alerts);
    const long = m.dispatchFetch('http://worker/harness/alarm');
    await held.entered;
    await clockAt(m, until, DEADLINE_MS + 5);
    await sleep(1_100);
    held.release();
    await long;
    expect(await triggers(m, slot)).toEqual(['alarm']);
  }

  test('a wake after a 6-minute outage runs one minute, the current one, and marks the other 5 missed: one of each per-run record, no alpha commit', async () => {
    const m = start();
    const slot = boundarySlot();
    await longRun(m, slot, slot + 6);
    const r2 = await m.getR2Bucket('MOMENTARILY');
    const alpha = (await r2.head('state/alpha.json'))!.etag;
    const before = await bucketState(m);

    await harness(m, 'alarm');
    for (let s = slot + 1; s <= slot + 5; s += 1) {
      expect(await triggers(m, s), `minute +${s - slot}`).toEqual([]);
      expect(await claimOf(m, s), `minute +${s - slot}`).toEqual(MISSED_NO_CLAIM);
    }
    expect(await triggers(m, slot + 6)).toEqual(['alarm']);
    const written = await writtenSince(m, before);
    for (const prefix of [`${OBSERVATIONS_PREFIX}ticks/`, `${OBSERVATIONS_PREFIX}raw/gtfsrt/`, 'archive/arrivals/', 'archive/health/']) {
      expect(written.filter((k) => k.startsWith(prefix)), prefix).toHaveLength(1);
    }
    // `slot + 5` was a 5-minute boundary, now past: its pipeline never ran.
    expect((await r2.head('state/alpha.json'))!.etag).toBe(alpha);
    const fiveMinute = ['archive/movement_census/', 'archive/vehicles/', 'archive/trip_updates/', 'v1/predictions/', 'v1/snapshot.json'];
    expect(written.filter((k) => fiveMinute.some((p) => k.startsWith(p)))).toEqual([]);
  }, 30_000);

  test('a wake spanning two 5-minute boundaries applies the pipeline once, for the current boundary; each census diffs over the real interval between applied ticks', async () => {
    const m = start();
    const slot = boundarySlot();
    await longRun(m, slot, slot + 10);
    const r2 = await m.getR2Bucket('MOMENTARILY');
    const before = await bucketState(m);

    await harness(m, 'alarm');
    expect(await triggers(m, slot + 5)).toEqual([]);
    expect(await claimOf(m, slot + 5)).toEqual(MISSED_NO_CLAIM);
    const [tick, ...rest] = await ticks(m, slot + 10);
    expect(rest).toEqual([]);
    expect(tick!.trigger).toBe('alarm');
    // One alpha advance in the wake, for the current boundary only.
    expect((await r2.head('state/alpha.json'))!.customMetadata).toEqual({ tick: String((slot + 10) * 60 + 19) });
    const written = await writtenSince(m, before);
    for (const prefix of ['archive/movement_census/', 'archive/vehicles/', 'archive/trip_updates/', 'v1/predictions/']) {
      expect(written.filter((k) => k.startsWith(prefix)), prefix).toHaveLength(1);
    }
    // One census per applied tick, each stamped with its own run's
    // observation: the second diffs against the first over the real time
    // between the two runs, never over a replay a second later.
    const [first] = await ticks(m, slot);
    const census: number[] = [];
    for (const key of await keys(m, 'archive/movement_census/')) {
      census.push(JSON.parse((await text(m, key))!).observed_at);
    }
    expect(census.sort((a, b) => a - b)).toEqual([first!, tick!].map((t) => Math.floor(t.started_at_ms / 1000)));
  }, 30_000);
});

describe('watchdog: the cron fails open', () => {
  test.each([
    ['answers 500', 'claim-500'],
    ['does not answer in 3 s', 'claim-hang'],
  ])('a claim the watchdog %s: the cron runs its minute anyway', async (_label, mode) => {
    const m = start();
    const slot = boundarySlot();
    await clockAt(m, slot, CRON_OFFSET_MS + 2_000);
    await harness(m, `fail?mode=${mode}`);
    await cron(m, slot);
    await harness(m, 'fail?mode=none');
    expect(await triggers(m, slot)).toEqual(['cron']);
    expect(await keys(m, 'v1/snapshot.json')).toEqual(['v1/snapshot.json']);
    // The cron still reported its run done, so the alarm does not repeat it.
    await clockAt(m, slot, DEADLINE_MS + 5);
    await harness(m, 'alarm');
    expect(await triggers(m, slot)).toEqual(['cron']);
  }, 30_000);

  test('renewals that cannot reach the watchdog: the cron runs its minute to the end and marks it done', async () => {
    const m = start();
    const slot = boundarySlot();
    await clockAt(m, slot, CRON_OFFSET_MS + 2_000);
    await harness(m, 'fail?mode=renew-500');
    await cron(m, slot);
    await harness(m, 'fail?mode=none');
    expect(await triggers(m, slot)).toEqual(['cron']);
    expect(await keys(m, 'v1/snapshot.json')).toEqual(['v1/snapshot.json']);
    expect(await claimOf(m, slot)).toEqual({ trigger: 'cron', done: true });
  }, 30_000);
});

describe.runIf(process.env.MEASURE_WAKE_CPU === '1')('watchdog: wake cost', () => {
  /** CPU seconds of each workerd process, by pid. */
  function workerdCpu(): Map<string, number> {
    const out = new Map<string, number>();
    for (const line of execFileSync('ps', ['-Ao', 'pid=,time=,comm=']).toString().split('\n')) {
      if (!line.includes('workerd')) continue;
      const [pid, time] = line.trim().split(/\s+/);
      const parts = time!.split(':').map(Number);
      out.set(pid!, parts.reduce((acc, p) => acc * 60 + p, 0));
    }
    return out;
  }

  test('CPU per idle wake (minute already run by the cron)', async () => {
    const before = new Set(workerdCpu().keys());
    const m = start();
    const slot = boundarySlot();
    await clockAt(m, slot, CRON_OFFSET_MS + 2_000);
    await cron(m, slot);
    await clockAt(m, slot, DEADLINE_MS + 5);
    const pids = [...workerdCpu().keys()].filter((p) => !before.has(p));
    const cpu = () => pids.reduce((acc, p) => acc + (workerdCpu().get(p) ?? 0), 0);
    const n = 2_000;
    const run = async (path: string) => {
      const c0 = cpu();
      for (let i = 0; i < n; i += 1) await m.dispatchFetch(`http://worker/harness/${path}`);
      return cpu() - c0;
    };
    await run('noop'); // warm-up
    const dispatch = await run('noop');
    const wakes = await run('alarm');
    const result = {
      wakes: n,
      cpu_s_dispatch_only: dispatch,
      cpu_s_wakes: wakes,
      cpu_ms_per_idle_wake: ((wakes - dispatch) * 1000) / n,
    };
    console.log('idle wake cost', result);
    expect(await triggers(m, slot)).toEqual(['cron']);
  }, 300_000);

  test('CPU per wake that runs a whole 5-minute minute (no cron)', async () => {
    const before = new Set(workerdCpu().keys());
    const m = start();
    const slot = boundarySlot();
    await clockAt(m, slot, DEADLINE_MS + 5);
    await harness(m, 'alarm'); // warm-up, and the first wake's one-time setup
    const pids = [...workerdCpu().keys()].filter((p) => !before.has(p));
    const cpu = () => pids.reduce((acc, p) => acc + (workerdCpu().get(p) ?? 0), 0);
    const n = 30;
    const c0 = cpu();
    for (let i = 1; i <= n; i += 1) {
      await clockAt(m, slot + 5 * i, DEADLINE_MS + 5);
      await harness(m, 'alarm');
    }
    const wakes = cpu() - c0;
    console.log('full-minute wake cost', { wakes: n, cpu_s: wakes, cpu_ms_per_wake: (wakes * 1000) / n });
    expect(await triggers(m, slot + 5 * n)).toEqual(['alarm']);
  }, 300_000);
});
