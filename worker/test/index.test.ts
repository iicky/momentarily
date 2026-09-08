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
  // Same idea for fetchJson (e.g. FEEDS.alerts) — a rejected promise, not an
  // empty-but-successful `{}` response, so index.ts's try/catch around the
  // alerts fetch takes its catch branch like a real upstream outage would.
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
      fetchState.protobufCalls.push(url);
      if (fetchState.protobufFailUrls.has(url)) throw new Error(`mock fetch failure: ${url}`);
      return fetchState.protobufByUrl.get(url) ?? new Uint8Array();
    },
  };
});

// Toggle to force the real buildSnapshot to throw for one test, exercising the
// fail-soft wrap (a build throw must degrade to "no fresh publish" without
// skipping the downstream last_seen/E&E/transitions steps). Off by default so
// every other test runs the genuine assembler.
const snapshotState = vi.hoisted(() => ({ buildThrows: false }));

vi.mock('../src/snapshot', async (importOriginal) => {
  const actual = await importOriginal<typeof SnapshotModule>();
  return {
    ...actual,
    buildSnapshot: (args: Parameters<typeof SnapshotModule.buildSnapshot>[0]) => {
      if (snapshotState.buildThrows) throw new Error('mock buildSnapshot failure');
      return actual.buildSnapshot(args);
    },
  };
});

import { FEEDS, STATIONS_FEED, TRIP_UPDATE_FEEDS } from '../src/fetch';
import { tod_bin } from '../src/hmm';
import worker, { tickMinute } from '../src/index';
import type { Env } from '../src/index';
import type * as SnapshotModule from '../src/snapshot';

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
  fetchState.jsonFailUrls.clear();
  snapshotState.buildThrows = false;
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

    // Nothing else did — with two deliberate exceptions, both per-minute by
    // design and both allowed off the 5-minute boundary for the same reason
    // the trace itself is: the platform-wait carry
    // (state/station_wait.json), which needs 1-minute resolution on when a
    // train cleared a platform, and the observed-headway carry
    // (state/headway.json), which needs it on when a train cleared a
    // reference stop. headway.json is absent HERE only because this fixture
    // publishes no state/segment_params.json, so there are no scheduled
    // stopping patterns to pick a reference stop from and the surface
    // abstains — see the off-boundary headway test below for the case where
    // it does write. No other state/ object moves: no 5-minute pipeline
    // state (vehicle_stops.json included), no snapshot, no
    // vehicles/trip-updates archive.
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

describe('scheduled: the observed-headway surface (state/headway.json)', () => {
  const BOUNDARY_AT = 1_704_067_200; // 2024-01-01T00:00:00Z, minute 0
  const NON_BOUNDARY_AT = BOUNDARY_AT + 180; // +3 minutes, minute 3

  /** The trainer doc the reference-stop rule reads: a three-stop A-north
   * pattern, so A02N is the one through-stop and becomes the measurement
   * point. `cells`/`adjacency` are empty — the headway surface reads only
   * `route_stops`. */
  function publishSegmentParams(store: Map<string, StoredObject>): void {
    store.set('state/segment_params.json', {
      body: JSON.stringify({
        schema_version: '1',
        trained_at: 1_700_000_000,
        min_share: 0.5,
        topology_source: 'gtfs_static',
        cells: {},
        adjacency: {},
        route_stops: {
          'A|north': [{ stops: ['A01N', 'A02N', 'A03N'], n_trips: 5 }],
        },
      }),
      etag: 'etag-seed',
    });
  }

  interface HeadwayDoc {
    observed_at: number;
    reference_stops: Record<string, string>;
    reference_trained_at: number;
    cells: Record<string, { stop_id: string; passings: { at: number; trip: string }[] }>;
    trips: Record<string, { cell: string; stop: string; at: number }>;
  }

  async function runAt(env: Env, at: number): Promise<void> {
    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(at * 1000);
    try {
      await worker.scheduled(scheduledAt(at), env, execCtx);
    } finally {
      nowSpy.mockRestore();
    }
  }

  test('the carry runs off the 5-minute boundary, at its own reference stop', async () => {
    // A train clears a stop in well under five minutes, so the passing
    // detection has to see every minute — this is the assertion that the
    // surface is genuinely per-minute and not silently gated to the pipeline.
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    publishSegmentParams(store);
    fetchState.protobufByUrl.set(
      TRIP_UPDATE_FEEDS[0]![1],
      vehicleFeed({ tripId: 'a', routeId: 'A', stopId: 'A02N' }),
    );

    await runAt(env, NON_BOUNDARY_AT);

    const doc = jsonAt(store, 'state/headway.json') as HeadwayDoc;
    // The rule picked the only through-stop of the published pattern.
    expect(doc.reference_stops).toEqual({ 'A|north': 'A02N' });
    expect(doc.reference_trained_at).toBe(1_700_000_000);
    expect(doc.observed_at).toBe(NON_BOUNDARY_AT);
    // The train is AT the reference stop, so it is carried awaiting its
    // departure — no passing, and deliberately no cell yet.
    expect(doc.trips).toEqual({ a: { cell: 'A|north', stop: 'A02N', at: NON_BOUNDARY_AT } });
    expect(doc.cells).toEqual({});
    // Still nothing from the 5-minute pipeline.
    expect(store.has('v1/snapshot.json')).toBe(false);
  });

  test('a measured headway reaches the published snapshot as an observation', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    publishSegmentParams(store);

    // Four one-minute polls: train a at the reference stop, a past it (one
    // passing), train b at it, b past it (the passing that closes the gap).
    const polls: [number, FakeVehicle][] = [
      [BOUNDARY_AT, { tripId: 'a', routeId: 'A', stopId: 'A02N' }],
      [BOUNDARY_AT + 60, { tripId: 'a', routeId: 'A', stopId: 'A03N' }],
      [BOUNDARY_AT + 120, { tripId: 'b', routeId: 'A', stopId: 'A02N' }],
      [BOUNDARY_AT + 300, { tripId: 'b', routeId: 'A', stopId: 'A03N' }],
    ];
    for (const [at, vehicle] of polls) {
      fetchState.protobufByUrl.set(TRIP_UPDATE_FEEDS[0]![1], vehicleFeed(vehicle));
      await runAt(env, at);
    }

    const doc = jsonAt(store, 'state/headway.json') as HeadwayDoc;
    // Two passings recorded at the reference stop: a's departure at +60 and
    // b's at +300. The published headway is derived from that pair.
    expect(doc.cells['A|north']?.passings).toEqual([
      { at: BOUNDARY_AT + 60, trip: 'a' },
      { at: BOUNDARY_AT + 300, trip: 'b' },
    ]);

    // The last poll was a 5-minute boundary, so it published — and the
    // measurement is on the public surface, with its measurement point.
    const snapshot = jsonAt(store, 'v1/snapshot.json') as {
      observations: {
        entity_ref: string;
        kind: string;
        value: number;
        unit: string;
        observed_at: number;
        source: string;
        direction: string;
        stop_id: string;
        window: { value: number; observed_at: number }[];
      }[];
      freshness: { vehicle_positions: number | null };
    };
    expect(snapshot.observations).toEqual([
      {
        entity_ref: 'subway_route:A',
        kind: 'headway',
        value: 240,
        unit: 'seconds',
        observed_at: BOUNDARY_AT + 300,
        source: 'gtfs_rt_vehicle_positions',
        direction: 'north',
        stop_id: 'A02N',
        window: [{ value: 240, observed_at: BOUNDARY_AT + 300 }],
        // No scheduled_headway.json seeded in this bucket: observed alone.
        scheduled: null,
        off_reference: false,
      },
    ]);
    expect(snapshot.freshness.vehicle_positions).toBe(BOUNDARY_AT + 300);
  });

  test('with no trainer stopping patterns the surface abstains, and publishes empty', async () => {
    // The observed-adjacency fallback doc carries route_stops: {} — there is
    // no defensible measurement point, so nothing is written and nothing is
    // published. Not a zero, not an arbitrary stop.
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    store.set('state/segment_params.json', {
      body: JSON.stringify({
        schema_version: '1',
        trained_at: 1_700_000_000,
        min_share: 0.5,
        topology_source: 'observed',
        cells: {},
        adjacency: {},
        route_stops: {},
      }),
      etag: 'etag-seed',
    });
    fetchState.protobufByUrl.set(
      TRIP_UPDATE_FEEDS[0]![1],
      vehicleFeed({ tripId: 'a', routeId: 'A', stopId: 'A02N' }),
    );

    await runAt(env, BOUNDARY_AT);

    expect(store.has('state/headway.json')).toBe(false);
    const snapshot = jsonAt(store, 'v1/snapshot.json') as { observations: unknown[] };
    expect(snapshot.observations).toEqual([]);
  });
});

describe('scheduled: alert-fetch liveness record (archive/alerts_liveness/)', () => {
  const BOUNDARY_AT = 1_704_067_200; // 2024-01-01T00:00:00Z, minute 0
  const NEXT_BOUNDARY_AT = BOUNDARY_AT + 300; // +5 minutes

  function livenessKey(observedAt: number): string {
    const date = new Date(observedAt * 1000).toISOString().slice(0, 10);
    return `archive/alerts_liveness/${date}/${observedAt}.json`;
  }

  async function runTick(
    env: Env,
    observedAt: number,
  ): Promise<void> {
    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(observedAt * 1000);
    try {
      await worker.scheduled(scheduledAt(observedAt), env, execCtx);
    } finally {
      nowSpy.mockRestore();
    }
  }

  test('a successful, alert-free tick still writes a liveness record even though archiveNewAlerts writes nothing', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    fetchState.jsonByUrl.set(FEEDS.alerts, { entity: [] });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);

    await runTick(env, BOUNDARY_AT);

    // The quiet-success case this record exists to disambiguate: zero new
    // alert versions archived (nothing changed), same as a total outage
    // would produce — but the liveness record still lands.
    expect(keysWithPrefix(store, 'archive/alerts/')).toHaveLength(0);
    expect(jsonAt(store, livenessKey(BOUNDARY_AT))).toEqual({
      observed_at: BOUNDARY_AT,
      outcome: 'success',
      fetched_at: BOUNDARY_AT,
    });
  });

  test('a failed alerts fetch STILL writes a liveness record, with outcome "fail"', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    fetchState.jsonFailUrls.add(FEEDS.alerts);
    fetchState.jsonByUrl.set(STATIONS_FEED, []);

    await runTick(env, BOUNDARY_AT);

    // Cold start: no prior successful fetch, so the stale fallback is 0 —
    // still recorded honestly rather than omitted or faked fresh.
    expect(jsonAt(store, livenessKey(BOUNDARY_AT))).toEqual({
      observed_at: BOUNDARY_AT,
      outcome: 'fail',
      fetched_at: 0,
    });
    // The 5-minute pipeline otherwise degrades gracefully around the gap —
    // this test only pins the liveness record, not the whole tick.
  });

  test('stale-fallback branch: a fetch failure after a prior success reports the STALE fetched_at, not the current tick', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    fetchState.jsonByUrl.set(FEEDS.alerts, { entity: [] });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);

    // Tick 1: alerts fetch succeeds, alerts_at advances to BOUNDARY_AT.
    await runTick(env, BOUNDARY_AT);
    expect(jsonAt(store, livenessKey(BOUNDARY_AT))).toMatchObject({ outcome: 'success' });

    // Tick 2: alerts fetch now fails. The record must report BOUNDARY_AT —
    // the last known-good fetch — not NEXT_BOUNDARY_AT, which would silently
    // claim the feed was live when it was not.
    fetchState.jsonFailUrls.add(FEEDS.alerts);
    await runTick(env, NEXT_BOUNDARY_AT);

    expect(jsonAt(store, livenessKey(NEXT_BOUNDARY_AT))).toEqual({
      observed_at: NEXT_BOUNDARY_AT,
      outcome: 'fail',
      fetched_at: BOUNDARY_AT,
    });
  });

  test('the published snapshot contract is untouched: alertsFreshness (v1/snapshot.json) still reflects the same stale fallback the archive record now also captures', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    fetchState.jsonByUrl.set(FEEDS.alerts, { entity: [] });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);

    await runTick(env, BOUNDARY_AT);
    fetchState.jsonFailUrls.add(FEEDS.alerts);
    await runTick(env, NEXT_BOUNDARY_AT);

    const snapshot = jsonAt(store, 'v1/snapshot.json') as { freshness: { subway_alerts: number } };
    expect(snapshot.freshness.subway_alerts).toBe(BOUNDARY_AT);
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

describe('freshness.params_stale: schema_version deploy skew', () => {
  const BOUNDARY_AT = 1_704_067_200; // 2024-01-01T00:00:00Z, minute 0

  async function runBoundary(env: Env): Promise<void> {
    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(BOUNDARY_AT * 1000);
    try {
      await worker.scheduled(scheduledAt(BOUNDARY_AT), env, execCtx);
    } finally {
      nowSpy.mockRestore();
    }
  }

  test('a bumped-version params.json still publishes on bootstrap, flagged params_stale', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    fetchState.jsonByUrl.set(FEEDS.alerts, { entity: [] });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);
    fetchState.protobufByUrl.set(
      TRIP_UPDATE_FEEDS[0]![1],
      vehicleFeed({ tripId: 'a', routeId: 'A', stopId: 'A01N' }),
    );
    // JSON-shape-compatible but a version the Worker can't read.
    await bucket.put(
      'state/params.json',
      JSON.stringify({ schema_version: '2', trained_at: 1, routes: {} }),
    );

    await runBoundary(env);

    // The tick still published — the mismatch degrades the model, not the feed.
    const snapshot = jsonAt(store, 'v1/snapshot.json') as {
      freshness: { params_stale: boolean };
      provenance: { params: { trained_at: number | null } };
    };
    expect(store.has('v1/snapshot.json')).toBe(true);
    // Flagged stale, and running on bootstrap: no trained_at behind the model.
    expect(snapshot.freshness.params_stale).toBe(true);
    expect(snapshot.provenance.params.trained_at).toBeNull();
  });

  test('a readable params.json publishes with params_stale false', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    fetchState.jsonByUrl.set(FEEDS.alerts, { entity: [] });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);
    fetchState.protobufByUrl.set(
      TRIP_UPDATE_FEEDS[0]![1],
      vehicleFeed({ tripId: 'a', routeId: 'A', stopId: 'A01N' }),
    );
    await bucket.put(
      'state/params.json',
      JSON.stringify({ schema_version: '1', trained_at: 42, routes: {} }),
    );

    await runBoundary(env);

    const snapshot = jsonAt(store, 'v1/snapshot.json') as {
      freshness: { params_stale: boolean };
      provenance: { params: { trained_at: number | null } };
    };
    expect(snapshot.freshness.params_stale).toBe(false);
    expect(snapshot.provenance.params.trained_at).toBe(42);
  });
});

describe('fail-soft: a buildSnapshot throw degrades the step, not the tick', () => {
  const BOUNDARY_AT = 1_704_067_200; // 2024-01-01T00:00:00Z, minute 0

  test('the snapshot is not published, but downstream last_seen/trip-updates still commit', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    fetchState.jsonByUrl.set(FEEDS.alerts, { entity: [] });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);
    fetchState.protobufByUrl.set(
      TRIP_UPDATE_FEEDS[0]![1],
      vehicleFeed({ tripId: 'a', routeId: 'A', stopId: 'A01N' }),
    );
    snapshotState.buildThrows = true;

    const err = vi.spyOn(console, 'error').mockImplementation(() => {});
    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(BOUNDARY_AT * 1000);
    try {
      await worker.scheduled(scheduledAt(BOUNDARY_AT), env, execCtx);
    } finally {
      nowSpy.mockRestore();
      err.mockRestore();
    }

    // The assembly threw, so no fresh snapshot was published this tick — the CDN
    // keeps serving the last-good one.
    expect(store.has('v1/snapshot.json')).toBe(false);
    // But the tick did NOT die: the alpha CAS (step 5), the trip-updates archive
    // (step 8b) and the last_seen CAS (step 9) — all independent of the snapshot
    // object — still committed. Before the wrap, the throw skipped every one.
    expect(store.has('state/alpha.json')).toBe(true);
    expect(store.has('state/last_seen.json')).toBe(true);
    expect(keysWithPrefix(store, 'archive/vehicles/')).toHaveLength(1);
  });
});

describe('freshness.alerts_parse_degraded: alerts-schema drift abstention', () => {
  const BOUNDARY_AT = 1_704_067_200; // 2024-01-01T00:00:00Z, minute 0

  // Seed last tick's movement regime so route A would otherwise publish a
  // movement-derived 'normal' condition. The whole point of the fix is that a
  // degraded alerts payload abstains OVER this read rather than let it stand.
  function seedMovementNormalA(store: Map<string, StoredObject>): void {
    store.set('state/movement_state.json', {
      body: JSON.stringify({
        observed_at: BOUNDARY_AT - 300,
        regimes: {
          A: {
            state: 'normal',
            entered_at: BOUNDARY_AT - 3600,
            last_seen_at: BOUNDARY_AT - 300,
            pending: null,
            pending_since: 0,
            pending_run: 0,
          },
        },
      }),
      etag: 'seed-movement',
    });
  }

  async function runBoundary(env: Env): Promise<void> {
    const nowSpy = vi.spyOn(Date, 'now').mockReturnValue(BOUNDARY_AT * 1000);
    try {
      await worker.scheduled(scheduledAt(BOUNDARY_AT), env, execCtx);
    } finally {
      nowSpy.mockRestore();
    }
  }

  test('entities in an unrecognised shape: routes abstain to unknown and the flag is set', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    seedMovementNormalA(store);
    // A non-empty entity array whose members carry no alert object with a
    // header_text or the mercury alert_type — nothing recognizable as an MTA
    // alert. The soft-parse drift the floor exists to catch.
    fetchState.jsonByUrl.set(FEEDS.alerts, {
      entity: [
        { id: 'x1', alert: { informed_entity: [{ route_id: 'A' }] } },
        { id: 'x2', alert: { some_new_shape: { renamed: 'Delays' } } },
      ],
    });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);

    await runBoundary(env);

    const snapshot = jsonAt(store, 'v1/snapshot.json') as {
      freshness: { alerts_parse_degraded: boolean };
      alerts: unknown[];
      route_status: Record<string, { condition: string; condition_source: string }>;
    };
    expect(store.has('v1/snapshot.json')).toBe(true);
    expect(snapshot.freshness.alerts_parse_degraded).toBe(true);
    // Nothing recognizable, so no alert surfaced — the empty list that used to
    // read as "calm system".
    expect(snapshot.alerts).toEqual([]);
    // Every route abstains, INCLUDING route A whose seeded movement regime was
    // 'normal': the degraded tick supersedes it rather than assert good service.
    const a = snapshot.route_status['A']!;
    expect(a.condition).toBe('unknown');
    expect(a.condition_source).toBe('unknown');
    for (const rs of Object.values(snapshot.route_status)) {
      expect(rs.condition).toBe('unknown');
    }
  });

  test('header-only entities with no selectors trip the floor: routes abstain to unknown', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    seedMovementNormalA(store);
    // Regression for the codex fixture: a payload of bare header_text entities
    // with no mercury alert_type and no informed_entity selectors. Nothing is a
    // readable MTA alert, so the floor must degrade rather than let route A fall
    // back to its seeded movement 'normal'.
    fetchState.jsonByUrl.set(FEEDS.alerts, {
      entity: [
        { id: 'x1', alert: { header_text: { translation: [{ text: 'Delays' }] } } },
        { id: 'x2', alert: { header_text: { translation: [{ text: 'Suspended' }] } } },
      ],
    });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);

    await runBoundary(env);

    const snapshot = jsonAt(store, 'v1/snapshot.json') as {
      freshness: { alerts_parse_degraded: boolean };
      route_status: Record<string, { condition: string }>;
    };
    expect(snapshot.freshness.alerts_parse_degraded).toBe(true);
    expect(snapshot.route_status['A']!.condition).toBe('unknown');
  });

  test('a feed of only station-scoped notices is NOT degraded: flag false, movement condition stands', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    seedMovementNormalA(store);
    // A valid feed carrying only out-of-scope alerts: an elevator notice names a
    // stop, not a subway route, so it derives no route snapshot. It is a
    // recognizable MTA alert, so the floor must NOT trip — this is a normal
    // quiet system, not schema drift.
    fetchState.jsonByUrl.set(FEEDS.alerts, {
      entity: [
        {
          id: 'lmm:alert:elev1',
          alert: {
            active_period: [{ start: BOUNDARY_AT - 600 }],
            informed_entity: [{ agency_id: 'MTASBWY', stop_id: 'A24' }],
            header_text: { translation: [{ text: 'Elevator out at station', language: 'en' }] },
            'transit_realtime.mercury_alert': { alert_type: 'Elevator' },
          },
        },
      ],
    });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);

    await runBoundary(env);

    const snapshot = jsonAt(store, 'v1/snapshot.json') as {
      freshness: { alerts_parse_degraded: boolean };
      route_status: Record<string, { condition: string }>;
    };
    expect(snapshot.freshness.alerts_parse_degraded).toBe(false);
    // Route A keeps its seeded movement 'normal' — no false abstention.
    expect(snapshot.route_status['A']!.condition).toBe('normal');
  });

  test('a valid feed of only inactive alerts is NOT degraded: flag false, movement condition stands', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    seedMovementNormalA(store);
    // A well-formed alert whose active_period ended an hour ago: it parses fine
    // but deriveRouteSnapshots drops it by active window. This is a valid quiet
    // feed (planned work published ahead / an alert just cleared), NOT drift —
    // the floor keys off parse success, so it must not trip.
    fetchState.jsonByUrl.set(FEEDS.alerts, {
      entity: [
        {
          id: 'lmm:alert:1',
          alert: {
            active_period: [{ start: BOUNDARY_AT - 7200, end: BOUNDARY_AT - 3600 }],
            informed_entity: [
              {
                route_id: 'A',
                'transit_realtime.mercury_entity_selector': {
                  sort_order: 'MTASBWY:A:30',
                },
              },
            ],
            header_text: { translation: [{ text: 'Delays on A', language: 'en' }] },
            'transit_realtime.mercury_alert': { alert_type: 'Delays' },
          },
        },
      ],
    });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);

    await runBoundary(env);

    const snapshot = jsonAt(store, 'v1/snapshot.json') as {
      freshness: { alerts_parse_degraded: boolean };
      route_status: Record<string, { condition: string }>;
    };
    expect(snapshot.freshness.alerts_parse_degraded).toBe(false);
    // Route A keeps its seeded movement 'normal' — no false abstention.
    expect(snapshot.route_status['A']!.condition).toBe('normal');
  });

  test('a well-formed payload with parseable entities is unchanged: flag false, movement condition stands', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    seedMovementNormalA(store);
    // Entities present AND recognizable (recognizable > 0): the floor (entities >
    // 0 && recognizable === 0) must not trip — a healthy feed with an active alert.
    fetchState.jsonByUrl.set(FEEDS.alerts, {
      entity: [
        {
          id: 'lmm:alert:1',
          alert: {
            active_period: [{ start: BOUNDARY_AT - 600 }],
            informed_entity: [
              {
                route_id: 'A',
                'transit_realtime.mercury_entity_selector': {
                  sort_order: 'MTASBWY:A:30',
                },
              },
            ],
            header_text: { translation: [{ text: 'Delays on A', language: 'en' }] },
            'transit_realtime.mercury_alert': { alert_type: 'Delays' },
          },
        },
      ],
    });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);

    await runBoundary(env);

    const snapshot = jsonAt(store, 'v1/snapshot.json') as {
      freshness: { alerts_parse_degraded: boolean };
      alerts: Array<{ id: string }>;
      route_status: Record<string, { condition: string; condition_source: string }>;
    };
    expect(snapshot.freshness.alerts_parse_degraded).toBe(false);
    // The alert parsed through to the published list.
    expect(snapshot.alerts.map((x) => x.id)).toContain('lmm:alert:1');
    // Route A keeps its movement-derived condition — the abstention override is
    // scoped strictly to the degraded tick.
    const a = snapshot.route_status['A']!;
    expect(a.condition).toBe('normal');
    expect(a.condition_source).toBe('movement');
  });

  test('mixed recognizable and garbage: NOT degraded, movement condition stands', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    seedMovementNormalA(store);
    // One real active route alert beside an unrecognizable garbage entity: the
    // payload plainly still carries MTA alerts (recognizable > 0), so the floor
    // must not trip. The real alert flows through; route A keeps movement normal.
    fetchState.jsonByUrl.set(FEEDS.alerts, {
      entity: [
        {
          id: 'lmm:alert:1',
          alert: {
            active_period: [{ start: BOUNDARY_AT - 600 }],
            informed_entity: [
              {
                route_id: 'A',
                'transit_realtime.mercury_entity_selector': {
                  sort_order: 'MTASBWY:A:30',
                },
              },
            ],
            header_text: { translation: [{ text: 'Delays on A', language: 'en' }] },
            'transit_realtime.mercury_alert': { alert_type: 'Delays' },
          },
        },
        { id: 'garbage', alert: { some_new_shape: {} } },
      ],
    });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);

    await runBoundary(env);

    const snapshot = jsonAt(store, 'v1/snapshot.json') as {
      freshness: { alerts_parse_degraded: boolean };
      alerts: Array<{ id: string }>;
      route_status: Record<string, { condition: string }>;
    };
    expect(snapshot.freshness.alerts_parse_degraded).toBe(false);
    expect(snapshot.alerts.map((x) => x.id)).toContain('lmm:alert:1');
    expect(snapshot.route_status['A']!.condition).toBe('normal');
  });

  test('a genuinely empty feed is a real quiet system, not degraded', async () => {
    const { bucket, store } = fakeBucket();
    const env: Env = { MOMENTARILY: bucket };
    seedMovementNormalA(store);
    fetchState.jsonByUrl.set(FEEDS.alerts, { entity: [] });
    fetchState.jsonByUrl.set(STATIONS_FEED, []);

    await runBoundary(env);

    const snapshot = jsonAt(store, 'v1/snapshot.json') as {
      freshness: { alerts_parse_degraded: boolean };
      route_status: Record<string, { condition: string }>;
    };
    expect(snapshot.freshness.alerts_parse_degraded).toBe(false);
    // No entities to parse is honest quiet: route A still reads its movement
    // 'normal', never abstained.
    expect(snapshot.route_status['A']!.condition).toBe('normal');
  });
});
