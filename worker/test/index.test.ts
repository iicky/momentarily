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
  // Urls that should reject this test, simulating a real fetchProtobuf
  // network/upstream failure rather than an empty-but-successful feed —
  // Promise.allSettled in index.ts treats these two very differently.
  protobufFailUrls: new Set<string>(),
}));

vi.mock('../src/fetch', async (importOriginal) => {
  const actual = await importOriginal<typeof import('../src/fetch')>();
  return {
    ...actual,
    fetchJson: async (url: string) => fetchState.jsonByUrl.get(url) ?? {},
    fetchProtobuf: async (url: string) => {
      fetchState.protobufCalls.push(url);
      if (fetchState.protobufFailUrls.has(url)) throw new Error(`mock fetch failure: ${url}`);
      return fetchState.protobufByUrl.get(url) ?? new Uint8Array();
    },
  };
});

import { FEEDS, STATIONS_FEED, TRIP_UPDATE_FEEDS } from '../src/fetch';
import { tod_bin } from '../src/hmm';
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
  fetchState.protobufFailUrls.clear();
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

    // Nothing else did — with one deliberate exception: the per-minute
    // platform-wait carry (state/station_wait.json) is allowed
    // off the 5-minute boundary too, for the same reason the trace itself
    // is — 1-minute resolution on when a train cleared a platform. No other
    // state/ object moves: no 5-minute pipeline state (vehicle_stops.json
    // included), no snapshot, no vehicles/trip-updates archive.
    expect(keysWithPrefix(store, 'state/')).toEqual(['state/station_wait.json']);
    expect(store.has('v1/snapshot.json')).toBe(false);
    expect(keysWithPrefix(store, 'archive/vehicles/')).toHaveLength(0);
    expect(keysWithPrefix(store, 'archive/trip_updates/')).toHaveLength(0);
    expect(keysWithPrefix(store, 'v1/')).toHaveLength(0);
  });

  test('a tick with no trace rows leaves the platform-wait carry untouched', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    // A total vehicle-feed outage: every feed fails, so deriveTrace yields
    // zero rows. Folding that in as an observation would prune the whole
    // trip -> stop carry (blinding the departure rule for the tick after the
    // feed returns) and stamp a fresh observed_at over frozen platform
    // timestamps, so the surface would keep publishing an ageing crowd as if
    // it were current. The prior doc must survive byte-for-byte instead.
    const prior = JSON.stringify({
      observed_at: NON_BOUNDARY_AT - 600,
      platforms: { A01N: NON_BOUNDARY_AT - 700 },
      trips: { a: 'A01N' },
    });
    store.set('state/station_wait.json', { body: prior, etag: 'w0' });

    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(NON_BOUNDARY_AT * 1000);
    try {
      await worker.scheduled(scheduledAt(NON_BOUNDARY_AT), env, execCtx);
    } finally {
      nowSpy.mockRestore();
    }

    expect(store.get('state/station_wait.json')?.body).toBe(prior);
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

describe('scheduled: trains.json publish (fail-soft on the vehicle feed)', () => {
  const BOUNDARY_AT = 1_704_067_200; // 2024-01-01T00:00:00Z, minute 0

  test('all vehicle feeds failing: v1/trains.json is left un-rewritten, never published as a fabricated empty read', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    fetchState.jsonByUrl.set(FEEDS.alerts, { entity: [] });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);
    for (const [, url] of TRIP_UPDATE_FEEDS) fetchState.protobufFailUrls.add(url);

    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(BOUNDARY_AT * 1000);
    try {
      await worker.scheduled(scheduledAt(BOUNDARY_AT), env, execCtx);
    } finally {
      nowSpy.mockRestore();
    }

    // The snapshot still publishes — a trains.json failure must never block
    // or fail the tick it rides alongside.
    expect(store.has('v1/snapshot.json')).toBe(true);
    // trains.json is simply absent, not written as {positions: []} — that
    // would assert "zero trains in NYC" when the true state is "unknown".
    expect(store.has('v1/trains.json')).toBe(false);
  });

  test('one of eight vehicle feeds failing: v1/trains.json IS published, flagged partial via fresh_feeds', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    fetchState.jsonByUrl.set(FEEDS.alerts, { entity: [] });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);
    fetchState.protobufFailUrls.add(TRIP_UPDATE_FEEDS[0]![1]); // 'ace' fails
    fetchState.protobufByUrl.set(
      TRIP_UPDATE_FEEDS[1]![1], // 'bdfm' decodes normally
      vehicleFeed({ tripId: 'a', routeId: 'F', stopId: 'A09N' }),
    );

    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(BOUNDARY_AT * 1000);
    try {
      await worker.scheduled(scheduledAt(BOUNDARY_AT), env, execCtx);
    } finally {
      nowSpy.mockRestore();
    }

    expect(store.has('v1/trains.json')).toBe(true);
    const trains = jsonAt(store, 'v1/trains.json') as {
      fresh_feeds: string[];
      expected_feeds: string[];
      positions: unknown[];
    };
    // expected_feeds is the full constant set regardless of what failed —
    // it's what a consumer diffs fresh_feeds against.
    expect(trains.expected_feeds).toEqual(TRIP_UPDATE_FEEDS.map(([name]) => name));
    // fresh_feeds names only the survivors: 'ace' is silently excluded, the
    // other seven decoded and are named.
    expect(trains.fresh_feeds).not.toContain('ace');
    expect(trains.fresh_feeds).toHaveLength(TRIP_UPDATE_FEEDS.length - 1);
    // The published positions still reflect exactly what DID decode — the
    // 'ace' gap doesn't zero out the routes that came through on other feeds.
    expect(trains.positions).toEqual([
      { route: 'F', direction: 'north', stop: 'A09N', stopped: false, n: 1 },
    ]);
  });
});

describe('step 8b: movement_through_stops from params.json', () => {
  const BOUNDARY_AT = 1_704_067_200; // 2024-01-01T00:00:00Z, minute 0

  test('a terminal stall (from_stop outside the trained set) is excluded from advanced_n/stalled_n but still recorded in transitions', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    fetchState.jsonByUrl.set(FEEDS.alerts, { entity: [] });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);
    await bucket.put(
      'state/params.json',
      JSON.stringify({
        schema_version: '1',
        trained_at: 1,
        routes: {},
        movement_through_stops: { A: { north: ['A09N'] } }, // A05N is NOT a through stop
      }),
    );

    // Tick 1: train sitting at A05N (a terminal, out of the trained set).
    fetchState.protobufByUrl.set(
      TRIP_UPDATE_FEEDS[0]![1],
      vehicleFeed({ tripId: 'a', routeId: 'A', stopId: 'A05N' }),
    );
    let nowSpy = vi.spyOn(Date, 'now').mockReturnValue(BOUNDARY_AT * 1000);
    try {
      await worker.scheduled(scheduledAt(BOUNDARY_AT), env, execCtx);
    } finally {
      nowSpy.mockRestore();
    }

    // Tick 2: still at A05N — a stall, but at an excluded from_stop.
    nowSpy = vi.spyOn(Date, 'now').mockReturnValue((BOUNDARY_AT + 300) * 1000);
    try {
      await worker.scheduled(scheduledAt(BOUNDARY_AT + 300), env, execCtx);
    } finally {
      nowSpy.mockRestore();
    }

    const vehicleKeys = keysWithPrefix(store, 'archive/vehicles/').sort();
    expect(vehicleKeys).toHaveLength(2);
    const tick2 = jsonAt(store, vehicleKeys[1]!) as {
      rows: Record<string, { advanced_n: number; stalled_n: number; by_direction: { north: { transitions: Record<string, number> } } }>;
    };
    expect(tick2.rows['A']!.advanced_n).toBe(0);
    expect(tick2.rows['A']!.stalled_n).toBe(0);
    expect(tick2.rows['A']!.by_direction.north.transitions).toEqual({ 'A05N>A05N': 1 });
  });

  test('trainedParams present but with no through-stop set counts every stop (the visible-log fallback path)', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    fetchState.jsonByUrl.set(FEEDS.alerts, { entity: [] });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);
    await bucket.put(
      'state/params.json',
      JSON.stringify({ schema_version: '1', trained_at: 1, routes: {} }), // no movement_through_stops
    );

    // Same terminal-stall setup as the filtered test above, but with no
    // through-stop set published: `trainedParams?.throughStops ?? null` must
    // still fall back to null (count every stop) exactly like a missing
    // params.json, not an empty/all-excluding set. index.ts logs this case
    // visibly (see the `!throughStops` branch at step 8b) — not asserted here
    // since this test env can't intercept the Workers-runtime console, but
    // the counting behaviour it accompanies is directly observable.
    fetchState.protobufByUrl.set(
      TRIP_UPDATE_FEEDS[0]![1],
      vehicleFeed({ tripId: 'a', routeId: 'A', stopId: 'A05N' }),
    );
    let nowSpy = vi.spyOn(Date, 'now').mockReturnValue(BOUNDARY_AT * 1000);
    try {
      await worker.scheduled(scheduledAt(BOUNDARY_AT), env, execCtx);
    } finally {
      nowSpy.mockRestore();
    }

    nowSpy = vi.spyOn(Date, 'now').mockReturnValue((BOUNDARY_AT + 300) * 1000);
    try {
      await worker.scheduled(scheduledAt(BOUNDARY_AT + 300), env, execCtx);
    } finally {
      nowSpy.mockRestore();
    }

    const vehicleKeys = keysWithPrefix(store, 'archive/vehicles/').sort();
    const tick2 = jsonAt(store, vehicleKeys[1]!) as {
      rows: Record<string, { advanced_n: number; stalled_n: number }>;
    };
    // Unfiltered: the terminal stall counts, unlike the filtered test above.
    expect(tick2.rows['A']!.stalled_n).toBe(1);
    expect(tick2.rows['A']!.advanced_n).toBe(0);
  });
});

describe('step 7: the movement channel inputs land on the prediction stream', () => {
  const BOUNDARY_AT = 1_704_067_200; // 2024-01-01T00:00:00Z, minute 0
  const TOD = String(tod_bin(BOUNDARY_AT));

  interface PredRow {
    route: string;
    matched_n: number | null;
    advanced_n: number | null;
  }

  /** The v1/predictions JSONL from the most recent tick, parsed. */
  function predictionRows(store: Map<string, StoredObject>): PredRow[] {
    const keys = keysWithPrefix(store, 'v1/predictions/').sort();
    const body = store.get(keys[keys.length - 1]!)!.body;
    return body
      .trim()
      .split('\n')
      .map((line) => JSON.parse(line) as PredRow);
  }

  /** Three trips on route A, each advancing one stop per tick — over
   *  MIN_MATCHED_TRIPS, so the channel is judgeable.
   *
   *  Three ticks, not two. The movement channel is one tick lagged: the metric
   *  comparing tick 1 to tick 2 is only WRITTEN at tick 2, so the first tick
   *  whose observation can fold it in is tick 3. Asserting on tick 2 would
   *  read null no matter how the gate behaved. */
  async function driveThreeTicks(env: Env) {
    fetchState.jsonByUrl.set(FEEDS.alerts, { entity: [] });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);
    const at = (stop: string) =>
      vehicleFeed(
        { tripId: 'a', routeId: 'A', stopId: stop },
        { tripId: 'b', routeId: 'A', stopId: stop },
        { tripId: 'c', routeId: 'A', stopId: stop },
      );
    const stops = ['A09N', 'A10N', 'A11N'];
    for (let i = 0; i < stops.length; i += 1) {
      fetchState.protobufByUrl.set(TRIP_UPDATE_FEEDS[0]![1], at(stops[i]!));
      const tickAt = BOUNDARY_AT + i * 300;
      const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(tickAt * 1000);
      try {
        await worker.scheduled(scheduledAt(tickAt), env, execCtx);
      } finally {
        nowSpy.mockRestore();
      }
    }
  }

  // A trained route whose emissions carry the fitted advance rate — without it
  // logEmission drops the channel however many trips matched.
  const ROUTE_A_PARAMS = {
    transition: [
      [0.95, 0.04, 0.01],
      [0.08, 0.9, 0.02],
      [0.02, 0.1, 0.88],
    ],
    initial: [0.9, 0.08, 0.02],
    emissions: {
      poisson_lambda: [0.3, 4.0, 12.0],
      gamma_alpha: [1.0, 3.0, 6.0],
      gamma_beta: [2.0, 0.4, 0.2],
      bernoulli_p: [0.001, 0.05, 0.95],
      bernoulli_p_delays: [0.02, 0.6, 0.35],
      bernoulli_p_service_change: [0.02, 0.6, 0.4],
      bernoulli_p_planned: [0.05, 0.6, 0.35],
      advance_rate: [0.6, 0.3, 0.02],
    },
    dwell_quantiles: {},
    dwell_quantiles_by_alert: {},
  };
  const BASELINE = { A: { north: { [TOD]: { p0: 0.6, alpha: 6, beta: 4, n: 50 } } } };

  test('records the counts the binomial was evaluated at when the channel fires', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    await bucket.put(
      'state/params.json',
      JSON.stringify({
        schema_version: '1',
        trained_at: 1,
        routes: { A: ROUTE_A_PARAMS },
        movement_baseline: BASELINE,
      }),
    );
    await driveThreeTicks(env);

    const rowA = predictionRows(store).find((r) => r.route === 'A');
    expect(rowA).toBeDefined();
    expect(rowA!.matched_n).toBe(3);
    expect(rowA!.advanced_n).toBe(3);
  });

  test('leaves the counts null when the params carry no fitted advance_rate, however many trips matched', async () => {
    // The exact divergence a has_movement-only check gets wrong: the baseline
    // gates has_movement on, three trips matched, and the channel STILL
    // contributes 0 because logEmission needs the rate to score against. A
    // count here would attribute nats to a channel that never fired.
    const { advance_rate: _dropped, ...noRate } = ROUTE_A_PARAMS.emissions;
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    await bucket.put(
      'state/params.json',
      JSON.stringify({
        schema_version: '1',
        trained_at: 1,
        routes: { A: { ...ROUTE_A_PARAMS, emissions: noRate } },
        movement_baseline: BASELINE,
      }),
    );
    await driveThreeTicks(env);

    const rowA = predictionRows(store).find((r) => r.route === 'A');
    expect(rowA).toBeDefined();
    expect(rowA!.matched_n).toBeNull();
    expect(rowA!.advanced_n).toBeNull();
  });

  test('leaves the counts null when no movement baseline gates the channel in', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    await bucket.put(
      'state/params.json',
      JSON.stringify({ schema_version: '1', trained_at: 1, routes: {} }), // no movement_baseline
    );
    await driveThreeTicks(env);

    // The trips still moved and the vehicle archive still counts them — what
    // changes is that the channel contributed nothing to this posterior, so
    // there is no count to attribute to it.
    const rowA = predictionRows(store).find((r) => r.route === 'A');
    expect(rowA).toBeDefined();
    expect(rowA!.matched_n).toBeNull();
    expect(rowA!.advanced_n).toBeNull();
  });
});
