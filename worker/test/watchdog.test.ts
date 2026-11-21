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
import { FEEDS, TRIP_UPDATE_FEEDS } from '../src/fetch';
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

/** Upstream state: alerts requests hang (never answer) while this is set. */
const upstream = { hangAlerts: false, alertsRequests: 0 };

async function answerUpstream(request: Request): Promise<Response> {
  const headers = { date: new Date().toUTCString() };
  if (request.method === 'HEAD') return new Response(null, { headers: { ...headers, etag: '"static"' } });
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
// cron storing its bundle, the runtime firing an alarm): no fake timer reaches
// that clock.
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function harness<T = Record<string, unknown>>(m: Miniflare, path: string): Promise<T> {
  const res = await m.dispatchFetch(`http://worker/harness/${path}`);
  expect(res.status, path).toBe(200);
  return (await res.json()) as T;
}

interface ClockState {
  now: number;
  alarm: number | null;
  claims: Record<string, { trigger: string; done: boolean }>;
}

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
    const state = await harness<ClockState>(m, 'state');
    expect(Object.values(state.claims)).toEqual([{ trigger: 'alarm', done: true }]);
    // Re-armed for the next minute's deadline.
    expect(state.alarm).toBe((slot + 1) * MINUTE_MS + DEADLINE_MS);
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

  test('a cron that claims and dies before storing anything: the alarm runs the minute once', async () => {
    const m = start();
    const slot = boundarySlot();
    await clockAt(m, slot, CRON_OFFSET_MS + 2_000);
    expect(await harness(m, `claim?slot=${slot}`)).toEqual({ won: true });
    await clockAt(m, slot, DEADLINE_MS + 5);
    await harness(m, 'alarm');
    await harness(m, 'alarm');
    expect(await triggers(m, slot)).toEqual(['alarm']);
  }, 30_000);

  test('a cron that dies after storing the raw feeds, before publishing: the alarm runs the minute once', async () => {
    const m = start();
    const slot = boundarySlot();
    await clockAt(m, slot, CRON_OFFSET_MS + 2_000);
    // The cron stores its GTFS-RT bundle, then hangs on the alerts feed the
    // snapshot needs: it never publishes, never writes its tick, never ends.
    upstream.hangAlerts = true;
    const worker = await m.getWorker();
    const dying = worker
      .scheduled({ scheduledTime: new Date(slot * MINUTE_MS + CRON_OFFSET_MS), cron: '* * * * *' })
      .catch(() => {});
    let bundles: string[] = [];
    for (let i = 0; i < 50 && bundles.length === 0; i += 1) {
      await sleep(100);
      bundles = await keys(m, `${OBSERVATIONS_PREFIX}raw/gtfsrt/`);
    }
    expect(bundles).toHaveLength(1);
    expect(await keys(m, 'v1/snapshot.json')).toEqual([]);
    expect(await ticks(m, slot)).toEqual([]);
    expect(Object.values((await harness<ClockState>(m, 'state')).claims)).toEqual([{ trigger: 'cron', done: false }]);

    upstream.hangAlerts = false;
    await clockAt(m, slot, DEADLINE_MS + 5);
    await harness(m, 'alarm');
    await harness(m, 'alarm');
    // Checked before the hung fetch's 10 s budget runs out.
    expect(await triggers(m, slot)).toEqual(['alarm']);
    expect(await keys(m, 'v1/snapshot.json')).toEqual(['v1/snapshot.json']);
    expect(await keys(m, `${OBSERVATIONS_PREFIX}raw/gtfsrt/`)).toHaveLength(2);
    void dying;
  }, 30_000);
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
    const state = await harness<ClockState>(m, 'state');
    expect(Object.values(state.claims)).toEqual([{ trigger: 'alarm', done: true }]);
    expect(state.alarm).toBe((slot + 1) * MINUTE_MS + DEADLINE_MS);
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
    expect(Object.values((await harness<ClockState>(m, 'state')).claims)).toEqual([{ trigger: 'alarm', done: false }]);
    void first;
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
});

describe.runIf(process.env.MEASURE_WAKE_CPU === '1')('watchdog: idle wake cost', () => {
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
});
