/**
 * Intra-day GTFS feed change detection (gtfs_feed.ts). resolveFeedIdentity
 * does a HEAD+ETag compare against archive/gtfs/latest.json on every trace
 * tick: a match resolves the digest with no download, a mismatch returns the
 * fresh etag immediately (digest null for THIS tick) plus a detached
 * `capture` promise the caller waitUntil's, and a HEAD failure degrades to
 * both fields null. These tests pin that the tick's own identity is NEVER
 * blocked on the ~5.6 MB download, that a successful capture writes the same
 * R2 layout training/gtfs_archive.store() uses, and that a failed capture
 * leaves the pointer untouched rather than corrupting it.
 */

import { createHash } from 'node:crypto';

import { afterEach, describe, expect, test, vi } from 'vitest';

import { resolveFeedIdentity } from '../src/gtfs_feed';

interface FakeStoredObject {
  body: ArrayBuffer | string;
  httpMetadata?: R2HTTPMetadata;
}

// Minimal in-memory R2 bucket covering get/head/put — the three gtfs_feed.ts
// touches. Extends archive.test.ts's fakeBucket() convention with head, which
// the capture path uses to skip re-uploading an already-held artifact.
function fakeBucket() {
  const store = new Map<string, FakeStoredObject>();
  const bucket = {
    async get(key: string) {
      const rec = store.get(key);
      if (!rec) return null;
      const text = typeof rec.body === 'string' ? rec.body : new TextDecoder().decode(rec.body);
      return { json: async () => JSON.parse(text) as unknown, text: async () => text };
    },
    async head(key: string) {
      return store.has(key) ? ({} as R2Object) : null;
    },
    async put(key: string, body: ArrayBuffer | string, opts?: { httpMetadata?: R2HTTPMetadata }) {
      store.set(key, { body, ...(opts?.httpMetadata ? { httpMetadata: opts.httpMetadata } : {}) });
      return {} as R2Object;
    },
  };
  return { bucket: bucket as unknown as R2Bucket, store };
}

function jsonAt(store: Map<string, FakeStoredObject>, key: string): unknown {
  const rec = store.get(key);
  if (!rec) return undefined;
  const text = typeof rec.body === 'string' ? rec.body : new TextDecoder().decode(rec.body);
  return JSON.parse(text);
}

function seedLatest(store: Map<string, FakeStoredObject>, pointer: Record<string, unknown>): void {
  store.set('archive/gtfs/latest.json', { body: JSON.stringify(pointer) });
}

interface FakeResponse {
  ok: boolean;
  status?: number;
  headers?: Record<string, string | null>;
  body?: Uint8Array;
}

/** Stubs the global fetch gtfs_feed.ts calls directly (HEAD, then an
 * unconditional-method GET on capture) — mirroring fetch.test.ts's `respond`
 * convention for fetchProtobuf, but dispatching on `init.method` since a
 * single test may see both a HEAD and a GET. */
function stubFetch(handler: (url: string, init?: RequestInit) => FakeResponse): void {
  const fn = vi.fn(async (url: string, init?: RequestInit) => {
    const r = handler(url, init);
    return {
      ok: r.ok,
      status: r.status ?? (r.ok ? 200 : 500),
      headers: { get: (name: string) => r.headers?.[name.toLowerCase()] ?? null },
      arrayBuffer: async () => {
        const b = r.body ?? new Uint8Array();
        return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength);
      },
    } as unknown as Response;
  });
  vi.stubGlobal('fetch', fn);
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('resolveFeedIdentity', () => {
  test('pointer match uses pointer digest without GET', async () => {
    const { bucket, store } = fakeBucket();
    const digest = 'a'.repeat(64);
    seedLatest(store, {
      digest,
      version: '20240101',
      etag: '"etag-1"',
      last_modified: 'Mon, 01 Jan 2024 00:00:00 GMT',
      stored_at: 1_700_000_000,
    });
    let getCalls = 0;
    stubFetch((_url, init) => {
      if (init?.method === 'HEAD') return { ok: true, headers: { etag: '"etag-1"' } };
      getCalls += 1;
      return { ok: true, body: new Uint8Array([1, 2, 3]) };
    });

    const resolution = await resolveFeedIdentity(bucket);

    expect(resolution.identity).toEqual({ digest, etag: '"etag-1"' });
    expect(resolution.capture).toBeNull();
    expect(getCalls).toBe(0);
  });

  test('a weak HEAD ETag never resolves against the pointer, but still captures when the GET is strong', async () => {
    const { bucket, store } = fakeBucket();
    // Pointer holds the strong ETag the Python nightly run stored.
    seedLatest(store, {
      digest: 'b'.repeat(64),
      version: '20240101',
      etag: '"etag-1"',
      last_modified: null,
      stored_at: 1_700_000_000,
    });
    const zipBytes = new TextEncoder().encode('bytes behind a weak validator');
    const expectedDigest = createHash('sha256').update(zipBytes).digest('hex');
    // HEAD carries the same token in weak form: it promises equivalence, not
    // identity, so the pointer's digest must NOT be stamped on this tick.
    stubFetch((_url, init) => {
      if (init?.method === 'HEAD') return { ok: true, headers: { etag: 'W/"etag-1"' } };
      return { ok: true, headers: { etag: '"etag-2"' }, body: zipBytes };
    });

    const resolution = await resolveFeedIdentity(bucket);

    expect(resolution.identity).toEqual({ digest: null, etag: 'W/"etag-1"' });
    expect(resolution.capture).not.toBeNull();
    await resolution.capture;
    // Keyed on the GET's strong validator, never the weak HEAD one.
    expect(jsonAt(store, 'archive/gtfs/by_etag/etag-2.json')).toEqual({ digest: expectedDigest });
    expect(store.has('archive/gtfs/by_etag/etag-1.json')).toBe(false);
    expect(jsonAt(store, 'archive/gtfs/latest.json')).toMatchObject({ digest: expectedDigest, etag: '"etag-2"' });
  });

  test('a weak GET ETag aborts the capture with nothing written', async () => {
    const { bucket, store } = fakeBucket();
    seedLatest(store, {
      digest: 'a'.repeat(64),
      version: null,
      etag: '"old-etag"',
      last_modified: null,
      stored_at: 1_690_000_000,
    });
    stubFetch((_url, init) => {
      if (init?.method === 'HEAD') return { ok: true, headers: { etag: '"new-etag"' } };
      return { ok: true, headers: { etag: 'W/"new-etag"' }, body: new Uint8Array([9, 9, 9]) };
    });

    const resolution = await resolveFeedIdentity(bucket);
    await resolution.capture;

    expect(jsonAt(store, 'archive/gtfs/latest.json')).toMatchObject({ digest: 'a'.repeat(64), etag: '"old-etag"' });
    expect([...store.keys()].some((k) => k.startsWith('archive/gtfs/by_etag/new-etag'))).toBe(false);
  });

  test('mismatch triggers capture and writes latest.json in the Python-compatible shape', async () => {
    const { bucket, store } = fakeBucket();
    seedLatest(store, {
      digest: 'a'.repeat(64),
      version: '20231201',
      etag: '"old-etag"',
      last_modified: null,
      stored_at: 1_690_000_000,
    });
    const zipBytes = new TextEncoder().encode('pretend this is the gtfs_subway.zip bytes');
    const expectedDigest = createHash('sha256').update(zipBytes).digest('hex');
    stubFetch((_url, init) => {
      if (init?.method === 'HEAD') return { ok: true, headers: { etag: '"new-etag"' } };
      return {
        ok: true,
        headers: { etag: '"new-etag"', 'last-modified': 'Tue, 02 Jan 2024 00:00:00 GMT' },
        body: zipBytes,
      };
    });

    const resolution = await resolveFeedIdentity(bucket);

    expect(resolution.identity).toEqual({ digest: null, etag: '"new-etag"' });
    expect(resolution.capture).not.toBeNull();

    await resolution.capture;

    expect(store.has(`archive/gtfs/${expectedDigest}.zip`)).toBe(true);
    expect(jsonAt(store, 'archive/gtfs/by_etag/new-etag.json')).toEqual({ digest: expectedDigest });
    expect(jsonAt(store, 'archive/gtfs/latest.json')).toEqual({
      digest: expectedDigest,
      version: null,
      etag: '"new-etag"',
      last_modified: 'Tue, 02 Jan 2024 00:00:00 GMT',
      stored_at: expect.any(Number),
    });
  });

  test('HEAD failure records nulls without throwing — the tick is never blocked', async () => {
    const { bucket } = fakeBucket();
    stubFetch(() => {
      throw new Error('network error');
    });

    const resolution = await resolveFeedIdentity(bucket);

    expect(resolution.identity).toEqual({ digest: null, etag: null });
    expect(resolution.capture).toBeNull();
  });

  test('concurrent capture is deduplicated: only one GET for two racing ticks', async () => {
    const { bucket, store } = fakeBucket();
    seedLatest(store, {
      digest: 'a'.repeat(64),
      version: null,
      etag: '"old-etag"',
      last_modified: null,
      stored_at: 1_690_000_000,
    });
    let getCalls = 0;
    const zipBytes = new TextEncoder().encode('another fake zip, different bytes');
    stubFetch((_url, init) => {
      if (init?.method === 'HEAD') return { ok: true, headers: { etag: '"new-etag"' } };
      getCalls += 1;
      return { ok: true, headers: { etag: '"new-etag"' }, body: zipBytes };
    });

    const [r1, r2] = await Promise.all([resolveFeedIdentity(bucket), resolveFeedIdentity(bucket)]);
    expect(r1.capture).not.toBeNull();
    expect(r2.capture).not.toBeNull();
    await Promise.all([r1.capture, r2.capture]);

    expect(getCalls).toBe(1);
    expect(store.has('archive/gtfs/latest.json')).toBe(true);
  });

  test('GET throws mid-capture: this tick already carries the etag, latest.json stays untouched', async () => {
    const { bucket, store } = fakeBucket();
    const pointer = {
      digest: 'a'.repeat(64),
      version: '20231201',
      etag: '"old-etag"',
      last_modified: null,
      stored_at: 1_690_000_000,
    };
    seedLatest(store, pointer);
    stubFetch((_url, init) => {
      if (init?.method === 'HEAD') return { ok: true, headers: { etag: '"new-etag"' } };
      throw new Error('network error mid-GET');
    });

    const resolution = await resolveFeedIdentity(bucket);

    expect(resolution.identity).toEqual({ digest: null, etag: '"new-etag"' });
    expect(resolution.capture).not.toBeNull();
    await expect(resolution.capture).resolves.toBeUndefined();

    expect(jsonAt(store, 'archive/gtfs/latest.json')).toEqual(pointer);
    expect([...store.keys()].some((k) => k.startsWith('archive/gtfs/by_etag/'))).toBe(false);
    expect([...store.keys()].some((k) => k.endsWith('.zip'))).toBe(false);
  });

  test('GET ETag differs from HEAD ETag: by_etag/ is keyed on the GET ETag', async () => {
    const { bucket, store } = fakeBucket();
    seedLatest(store, {
      digest: 'a'.repeat(64),
      version: null,
      etag: '"old-etag"',
      last_modified: null,
      stored_at: 1_690_000_000,
    });
    const zipBytes = new TextEncoder().encode('the feed that rolled over between HEAD and GET');
    const expectedDigest = createHash('sha256').update(zipBytes).digest('hex');
    stubFetch((_url, init) => {
      if (init?.method === 'HEAD') return { ok: true, headers: { etag: '"head-etag"' } };
      // GET returns a DIFFERENT ETag — the object rolled over on S3
      return {
        ok: true,
        headers: { etag: '"get-etag"', 'last-modified': 'Wed, 03 Jan 2024 00:00:00 GMT' },
        body: zipBytes,
      };
    });

    const resolution = await resolveFeedIdentity(bucket);
    expect(resolution.identity.etag).toBe('"head-etag"');
    await resolution.capture;

    // by_etag/ is keyed on the GET's etag, NOT the HEAD's
    expect(store.has('archive/gtfs/by_etag/get-etag.json')).toBe(true);
    expect(store.has('archive/gtfs/by_etag/head-etag.json')).toBe(false);
    expect(jsonAt(store, 'archive/gtfs/by_etag/get-etag.json')).toEqual({ digest: expectedDigest });

    // latest.json carries the GET's etag
    const latest = jsonAt(store, 'archive/gtfs/latest.json') as Record<string, unknown>;
    expect(latest.etag).toBe('"get-etag"');
    expect(latest.digest).toBe(expectedDigest);
  });

  test('GET omits ETag: capture is aborted, nothing is written', async () => {
    const { bucket, store } = fakeBucket();
    seedLatest(store, {
      digest: 'a'.repeat(64),
      version: null,
      etag: '"old-etag"',
      last_modified: null,
      stored_at: 1_690_000_000,
    });
    const originalPointer = jsonAt(store, 'archive/gtfs/latest.json');
    stubFetch((_url, init) => {
      if (init?.method === 'HEAD') return { ok: true, headers: { etag: '"new-etag"' } };
      // GET returns no ETag header at all
      return { ok: true, headers: {}, body: new TextEncoder().encode('some bytes') };
    });

    const resolution = await resolveFeedIdentity(bucket);
    expect(resolution.identity).toEqual({ digest: null, etag: '"new-etag"' });
    await resolution.capture;

    // Nothing written: no zip, no by_etag, latest.json unchanged
    expect([...store.keys()].some((k) => k.endsWith('.zip'))).toBe(false);
    expect([...store.keys()].some((k) => k.startsWith('archive/gtfs/by_etag/'))).toBe(false);
    expect(jsonAt(store, 'archive/gtfs/latest.json')).toEqual(originalPointer);
  });
});
