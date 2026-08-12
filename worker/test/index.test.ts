/**
 * Pins the 5-minute pipeline gate in index.ts's `scheduled` handler. The cron
 * now fires every minute (wrangler.toml) to drive the per-minute vehicle
 * trace, but everything from step 1 onward (alerts, the HMM filter, snapshot
 * publish, vehicle_stops.json, the movement/service metrics...) is defined
 * per 5-MINUTE TICK and must keep running at exactly that cadence. These
 * tests exercise the real `scheduled` handler end to end against a fake R2
 * bucket and a mocked feed fetch, so they fail if the gate is ever removed,
 * loosened, or bypassed — not just if the standalone boundary check breaks.
 */

import { beforeEach, describe, expect, test, vi } from 'vitest';

const fetchState = vi.hoisted(() => ({
  jsonByUrl: new Map<string, unknown>(),
  protobufByUrl: new Map<string, Uint8Array>(),
  protobufCalls: [] as string[],
}));

vi.mock('../src/fetch', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/fetch')>();
  return {
    ...actual,
    fetchJson: async (url: string) => fetchState.jsonByUrl.get(url) ?? {},
    fetchProtobuf: async (url: string) => {
      fetchState.protobufCalls.push(url);
      return fetchState.protobufByUrl.get(url) ?? new Uint8Array();
    },
  };
});

import { FEEDS, STATIONS_FEED, TRIP_UPDATE_FEEDS } from '../src/fetch';
import worker, { tickMinute } from '../src/index';
import type { Env } from '../src/index';

// --- tiny protobuf encoder (test-only; mirrors gtfsrt.test.ts's fixtures) ---
// VehiclePosition: trip(1, message), current_stop_sequence(3, varint),
// current_status(4, varint), timestamp(5, varint), stop_id(7, string).
// TripDescriptor: trip_id(1, string), route_id(5, string).
function varint(n: number): number[] {
  const out: number[] = [];
  while (n > 0x7f) {
    out.push((n & 0x7f) | 0x80);
    n >>>= 7;
  }
  out.push(n);
  return out;
}
const tag = (field: number, wire: number): number[] => varint(field * 8 + wire);
const lenField = (field: number, body: number[]): number[] => [
  ...tag(field, 2),
  ...varint(body.length),
  ...body,
];
const strField = (field: number, s: string): number[] =>
  lenField(field, [...new TextEncoder().encode(s)]);
const varField = (field: number, n: number): number[] => [...tag(field, 0), ...varint(n)];

interface FakeVehicle {
  tripId: string;
  routeId: string;
  stopId: string;
  status?: number;
  stopSeq?: number;
  timestamp?: number;
}

function vehiclePosition(v: FakeVehicle): number[] {
  return [
    ...lenField(1, [...strField(1, v.tripId), ...strField(5, v.routeId)]),
    ...(v.stopSeq !== undefined ? varField(3, v.stopSeq) : []),
    ...(v.status !== undefined ? varField(4, v.status) : []),
    ...(v.timestamp !== undefined ? varField(5, v.timestamp) : []),
    ...strField(7, v.stopId),
  ];
}
function vehicleEntity(v: FakeVehicle): number[] {
  return lenField(2, [...strField(1, `${v.tripId}-veh`), ...lenField(4, vehiclePosition(v))]);
}
function vehicleFeed(...vehicles: FakeVehicle[]): Uint8Array {
  return new Uint8Array(vehicles.flatMap((v) => vehicleEntity(v)));
}

// --- fake R2 bucket with etag CAS support, matching r2.ts's conditionalPut ---
interface StoredObject {
  body: string;
  etag: string;
  httpMetadata?: R2HTTPMetadata;
}

function fakeBucket() {
  const store = new Map<string, StoredObject>();
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
  return { bucket: bucket as unknown as R2Bucket, store };
}

function keysWithPrefix(store: Map<string, StoredObject>, prefix: string): string[] {
  return [...store.keys()].filter((k) => k.startsWith(prefix));
}

function jsonAt(store: Map<string, StoredObject>, key: string): unknown {
  const rec = store.get(key);
  return rec ? JSON.parse(rec.body) : undefined;
}

interface TraceArchiveDoc {
  observed_at: number;
  fresh_feeds: string[];
  rows: unknown[];
}

/** Reads back an archive/trace/... object's `rows` array — the shape is
 * exactly what archiveTraceRows (src/archive.ts) writes. */
function traceRowsAt(store: Map<string, StoredObject>, key: string): unknown[] {
  const doc = jsonAt(store, key) as TraceArchiveDoc;
  return doc.rows;
}

/** The handler gates on the cron's SCHEDULED minute, not on Date.now(), so a
 * late-starting boundary run still does its work. Tests therefore have to drive
 * scheduledTime; mocking Date.now() alone would no longer move the gate. */
function scheduledAt(epochSec: number): ScheduledController {
  return { cron: '* * * * *', scheduledTime: epochSec * 1000 } as unknown as ScheduledController;
}
const execCtx = {} as unknown as ExecutionContext;

beforeEach(() => {
  fetchState.jsonByUrl.clear();
  fetchState.protobufByUrl.clear();
  fetchState.protobufCalls = [];
});

describe('tickMinute', () => {
  test('extracts UTC minute-of-hour from a tick\'s observedAt (POSIX seconds)', () => {
    expect(tickMinute(1_704_067_200)).toBe(0); // 2024-01-01T00:00:00Z
    expect(tickMinute(1_704_067_380)).toBe(3); // +3 minutes
    expect(tickMinute(1_704_067_500)).toBe(5); // +5 minutes
    expect(tickMinute(1_704_067_800)).toBe(10); // +10 minutes
  });
});

describe('scheduled: the 5-minute pipeline gate', () => {
  const BOUNDARY_AT = 1_704_067_200; // 2024-01-01T00:00:00Z, minute 0
  const NON_BOUNDARY_AT = 1_704_067_380; // +3 minutes, minute 3

  test('non-boundary minute: the trace runs, and NOTHING else does', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    fetchState.protobufByUrl.set(
      TRIP_UPDATE_FEEDS[0]![1],
      vehicleFeed({ tripId: 'a', routeId: 'A', stopId: 'A01N' }),
    );

    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(NON_BOUNDARY_AT * 1000);
    try {
      await worker.scheduled(scheduledAt(NON_BOUNDARY_AT), env, execCtx);
    } finally {
      nowSpy.mockRestore();
    }

    // The trace ran: it archived its snapshot rows under archive/trace/ only.
    const traceKeys = keysWithPrefix(store, 'archive/trace/');
    expect(traceKeys).toHaveLength(1);
    expect(traceRowsAt(store, traceKeys[0]!)).toEqual([
      expect.objectContaining({ trip_id: 'a', stop_id: 'A01N', stopped: false }),
    ]);

    // Nothing else did: the trace never writes ANY state/ object (it is a
    // pure function of the feed, not a carry), no 5-minute pipeline state,
    // no snapshot, no vehicles/trip-updates archive.
    expect(keysWithPrefix(store, 'state/')).toHaveLength(0);
    expect(store.has('v1/snapshot.json')).toBe(false);
    expect(keysWithPrefix(store, 'archive/vehicles/')).toHaveLength(0);
    expect(keysWithPrefix(store, 'archive/trip_updates/')).toHaveLength(0);
    expect(keysWithPrefix(store, 'v1/')).toHaveLength(0);
  });

  test('boundary minute (minute % 5 === 0): the 5-minute pipeline runs as before, AND the trace also runs', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    fetchState.jsonByUrl.set(FEEDS.alerts, { entity: [] });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);
    fetchState.protobufByUrl.set(
      TRIP_UPDATE_FEEDS[0]![1],
      vehicleFeed({ tripId: 'a', routeId: 'A', stopId: 'A01N' }),
    );

    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(BOUNDARY_AT * 1000);
    try {
      await worker.scheduled(scheduledAt(BOUNDARY_AT), env, execCtx);
    } finally {
      nowSpy.mockRestore();
    }

    // The 5-minute pipeline ran exactly as it did before this change.
    expect(store.has('state/vehicle_stops.json')).toBe(true);
    expect(jsonAt(store, 'state/vehicle_stops.json')).toEqual({ a: 'A01N' });
    expect(keysWithPrefix(store, 'archive/vehicles/')).toHaveLength(1);
    expect(keysWithPrefix(store, 'archive/trip_updates/')).toHaveLength(1);
    expect(store.has('state/alpha.json')).toBe(true);
    expect(store.has('state/last_seen.json')).toBe(true);
    expect(store.has('v1/snapshot.json')).toBe(true);

    // The trace ALSO ran, off its own archive prefix — never touching state/.
    expect(keysWithPrefix(store, 'archive/trace/')).toHaveLength(1);

    // The vehicle-position feed was fetched exactly once per line-group feed
    // this tick — the trace and the 5-minute pipeline share the same fetch,
    // never a double-fetch on a boundary minute.
    expect(fetchState.protobufCalls).toHaveLength(TRIP_UPDATE_FEEDS.length);
  });

  test('REGRESSION: a boundary run that STARTS LATE still runs the 5-minute pipeline', async () => {
    // Cloudflare does not promise punctuality. If the gate read the wall clock
    // at execution instead of the cron's scheduled minute, a boundary run that
    // started 61s late would read as minute 1, fail the `% 5` test, and silently
    // skip everything — no snapshot, no state advance, for five minutes, with
    // only a log line. Scheduled for minute 0, executing during minute 1.
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    fetchState.jsonByUrl.set(FEEDS.alerts, { entity: [] });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);
    fetchState.protobufByUrl.set(
      TRIP_UPDATE_FEEDS[0]![1],
      vehicleFeed({ tripId: 'a', routeId: 'A', stopId: 'A01N' }),
    );

    const lateAt = BOUNDARY_AT + 61;
    expect(new Date(lateAt * 1000).getUTCMinutes() % 5).not.toBe(0);

    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(lateAt * 1000);
    try {
      await worker.scheduled(scheduledAt(BOUNDARY_AT), env, execCtx);
    } finally {
      nowSpy.mockRestore();
    }

    expect(store.has('v1/snapshot.json')).toBe(true);
    expect(store.has('state/vehicle_stops.json')).toBe(true);
    expect(keysWithPrefix(store, 'archive/vehicles/')).toHaveLength(1);
  });

  test('REGRESSION: a retry of the same cron minute overwrites its trace object, never duplicates it', async () => {
    // The trace step runs BEFORE any compare-and-swap winner check, so a retried
    // or overlapping invocation for the same scheduled minute reaches the archive
    // writer twice. Keyed on execution time those two runs land on different
    // seconds and write TWO objects holding the same rows — double-counted
    // arrivals, and no history in a fresh archive to notice it. Keyed on the
    // scheduled second, the retry overwrites.
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    fetchState.jsonByUrl.set(FEEDS.alerts, { entity: [] });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);
    fetchState.protobufByUrl.set(
      TRIP_UPDATE_FEEDS[0]![1],
      vehicleFeed({ tripId: 'a', routeId: 'A', stopId: 'A01N' }),
    );

    // Same scheduled minute, two different execution instants.
    for (const executedAt of [NON_BOUNDARY_AT, NON_BOUNDARY_AT + 7]) {
      const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(executedAt * 1000);
      try {
        await worker.scheduled(scheduledAt(NON_BOUNDARY_AT), env, execCtx);
      } finally {
        nowSpy.mockRestore();
      }
    }

    // Not just one object — the SAME object, still holding the row it recorded
    // on the first pass. The old carry-based delta would have seen its own
    // prior write, computed zero changed rows on the retry, and overwritten
    // this with an empty array — silently destroying the observation.
    const traceKeys = keysWithPrefix(store, 'archive/trace/');
    expect(traceKeys).toHaveLength(1);
    expect(traceRowsAt(store, traceKeys[0]!)).toEqual([
      expect.objectContaining({ trip_id: 'a', stop_id: 'A01N', stopped: false }),
    ]);
  });

  test('two snapshots five minutes apart both carry the trip, with the differing stop_id/stopped that the offline arrival diff consumes', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    fetchState.jsonByUrl.set(FEEDS.alerts, { entity: [] });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);

    // Tick 1 (minute 0): train in transit to A01N.
    fetchState.protobufByUrl.set(
      TRIP_UPDATE_FEEDS[0]![1],
      vehicleFeed({ tripId: 'a', routeId: 'A', stopId: 'A01N' }),
    );
    let nowSpy = vi.spyOn(Date, 'now').mockReturnValue(BOUNDARY_AT * 1000);
    try {
      await worker.scheduled(scheduledAt(BOUNDARY_AT), env, execCtx);
    } finally {
      nowSpy.mockRestore();
    }

    // Tick 2 (minute 5, the next boundary): now stopped at A01N — the
    // arrival. The 5-minute movement carry also advances the same tick.
    fetchState.protobufByUrl.set(
      TRIP_UPDATE_FEEDS[0]![1],
      vehicleFeed({ tripId: 'a', routeId: 'A', stopId: 'A01N', status: 1, stopSeq: 1 }),
    );
    nowSpy = vi.spyOn(Date, 'now').mockReturnValue((BOUNDARY_AT + 300) * 1000);
    try {
      await worker.scheduled(scheduledAt(BOUNDARY_AT + 300), env, execCtx);
    } finally {
      nowSpy.mockRestore();
    }

    const traceKeys = keysWithPrefix(store, 'archive/trace/').sort();
    expect(traceKeys).toHaveLength(2);
    expect(traceRowsAt(store, traceKeys[0]!)).toEqual([
      expect.objectContaining({ trip_id: 'a', stop_id: 'A01N', stopped: false }),
    ]);
    expect(traceRowsAt(store, traceKeys[1]!)).toEqual([
      expect.objectContaining({ trip_id: 'a', stop_id: 'A01N', stopped: true, stop_seq: 1 }),
    ]);
  });
});
