/**
 * The public grading streams through the public Worker route: the production
 * writers put records into a real R2 bucket (Miniflare, inside workerd, via
 * public_streams_harness.ts), then each record is requested the only way a
 * public client can, by exact key. The Worker lists nothing, so a record is
 * public only if its name can be computed from the documented layout
 * (ADR 0001, worker/README.md) or read from the stream's day index.
 */

/// <reference types="node" />

import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { build } from 'esbuild';
import { Miniflare } from 'miniflare';
import { afterAll, beforeAll, beforeEach, describe, expect, test } from 'vitest';

import type { DayIndex } from '../src/grading';
import { dayIndexKey } from '../src/grading';

const HERE = dirname(fileURLToPath(import.meta.url));

/** 2026-11-21T08:00:01Z: a boundary run's execution second. */
const AT = 1_795_248_001;
const DAY = '2026-11-21';
/** Two runs of different boundaries that start in the same second. */
const LATE_RUN = `${AT - 319}-${AT * 1000 + 120}`;
const ON_TIME_RUN = `${AT - 19}-${AT * 1000 + 480}`;

type Writer = 'predictions' | 'transitions' | 'movement';
type Line = Record<string, string | number>;

let mf: Miniflare;

beforeAll(async () => {
  const out = await build({
    entryPoints: [join(HERE, 'public_streams_harness.ts')],
    bundle: true,
    format: 'esm',
    platform: 'browser',
    target: 'es2022',
    write: false,
    logLevel: 'error',
  });
  mf = new Miniflare({
    modules: true,
    script: out.outputFiles[0]!.text,
    compatibilityDate: '2024-12-30',
    r2Buckets: ['MOMENTARILY'],
  });
}, 30_000);

afterAll(async () => {
  await mf.dispose();
});

beforeEach(async () => {
  const r2 = await mf.getR2Bucket('MOMENTARILY');
  const listed = await r2.list();
  await Promise.all(listed.objects.map((o) => r2.delete(o.key)));
});

async function write(writer: Writer, run: string, records: Line[]): Promise<void> {
  const res = await mf.dispatchFetch('http://worker/harness/write', {
    method: 'POST',
    body: JSON.stringify({ writer, at: AT, run, records }),
  });
  expect(res.status, await res.text()).toBe(200);
}

/** GET `key` through the public Worker route. */
async function publicGet(key: string): Promise<{ status: number; body: string }> {
  const res = await mf.dispatchFetch(`http://worker/${key}`);
  return { status: res.status, body: await res.text() };
}

async function listed(prefix: string): Promise<string[]> {
  return (await (await mf.getR2Bucket('MOMENTARILY')).list({ prefix })).objects.map((o) => o.key).sort();
}

const jsonl = (records: Line[]): string => records.map((r) => JSON.stringify(r)).join('\n');
const line = (route: string, extra: Line = {}): Line => ({ ts: AT, route, ...extra });

describe('public grading streams: a new record is served at its documented URL', () => {
  test('predictions, regime transitions and both movement scopes', async () => {
    const predictions = [line('A'), line('C')];
    const transitions = [line('A')];
    const route = [line('A', { scope: 'route', key: 'A' })];
    const segment = [line('A', { scope: 'segment', key: 'A|north|A02N' })];
    await write('predictions', ON_TIME_RUN, predictions);
    await write('transitions', ON_TIME_RUN, transitions);
    await write('movement', ON_TIME_RUN, [...route, ...segment]);

    for (const [key, records] of [
      [`v1/predictions/${DAY}/${AT}.jsonl`, predictions],
      [`v1/regime_transitions/${DAY}/${AT}.jsonl`, transitions],
      [`v1/movement_transitions/${DAY}/${AT}-route.jsonl`, route],
      [`v1/movement_transitions/${DAY}/${AT}-segment.jsonl`, segment],
    ] as const) {
      expect(await publicGet(key), key).toEqual({ status: 200, body: jsonl(records) });
    }
    // No collision, no index: a day index only lists fallbacks.
    for (const stream of ['v1/predictions', 'v1/regime_transitions', 'v1/movement_transitions']) {
      expect((await publicGet(dayIndexKey(stream, DAY))).status, stream).toBe(404);
    }
    expect(await listed('')).toHaveLength(4);
  });
});

describe('public grading streams: two runs on one documented name both stay public', () => {
  test.each([
    ['v1/predictions', `${AT}`, 'predictions', {}],
    ['v1/regime_transitions', `${AT}`, 'transitions', {}],
    ['v1/movement_transitions', `${AT}-route`, 'movement', { scope: 'route', key: 'A' }],
  ] as const)('%s', async (stream, stem, writer, extra) => {
    const first = [line('A', extra)];
    const second = [line('B', extra)];
    await write(writer, ON_TIME_RUN, first);
    await write(writer, LATE_RUN, second);

    const documented = `${stream}/${DAY}/${stem}.jsonl`;
    const fallback = `${stream}/${DAY}/${stem}-${LATE_RUN}.jsonl`;
    // The first writer keeps the documented name; create-once never replaces it.
    expect(await publicGet(documented)).toEqual({ status: 200, body: jsonl(first) });
    // The day index, at a computable URL, names the second record's key.
    const index = await publicGet(dayIndexKey(stream, DAY));
    expect(index.status).toBe(200);
    expect(JSON.parse(index.body) as DayIndex).toEqual({
      day: DAY,
      objects: [{ key: fallback, documented_key: documented }],
    });
    expect(await publicGet(fallback)).toEqual({ status: 200, body: jsonl(second) });
    // Only records in the day folder, so a date-prefix listing reads no index.
    expect(await listed(`${stream}/${DAY}/`)).toEqual([documented, fallback].sort());
  });

  test('concurrent fallbacks of one day are all listed (index compare-and-swap)', async () => {
    await write('predictions', ON_TIME_RUN, [line('A')]);
    const runs = [1, 2, 3, 4].map((i) => `${AT - 19 - 300 * i}-${AT * 1000 + i}`);
    await Promise.all(runs.map((run) => write('predictions', run, [line(run)])));
    const index = JSON.parse((await publicGet(dayIndexKey('v1/predictions', DAY))).body) as DayIndex;
    expect(index.objects.map((o) => o.key).sort()).toEqual(
      runs.map((run) => `v1/predictions/${DAY}/${AT}-${run}.jsonl`).sort(),
    );
    for (const { key } of index.objects) expect((await publicGet(key)).status, key).toBe(200);
  });
});
