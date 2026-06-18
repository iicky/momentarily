/**
 * Static station metadata: parsing the 39hk-dx4f Socrata rows, the R2 cache
 * round-trip, and the snapshot embedding it with a stations_static freshness
 * stamp. The heavy payload lives in its own R2 object, so last_seen stays lean.
 */

import Ajv2020 from 'ajv/dist/2020';
import { describe, expect, test } from 'vitest';

import schema from '../../schema/snapshot.schema.json';
import { TICK_SECONDS, buildSnapshot } from '../src/snapshot';
import { parseStationsFeed, readStationsCache, writeStationsCache } from '../src/stations_static';

const ajv = new Ajv2020({ allErrors: true, strict: false });
const validate = ajv.compile(schema);

const NOW = 1_700_000_000;

// One real-shaped 39hk-dx4f row.
function row(over: Record<string, unknown> = {}): Record<string, unknown> {
  return {
    gtfs_stop_id: 'R03',
    complex_id: '2',
    stop_name: 'Astoria Blvd',
    borough: 'Q',
    daytime_routes: 'N W',
    ada: '1',
    ada_northbound: '1',
    ada_southbound: '1',
    ...over,
  };
}

// Minimal in-memory R2 bucket — just the get/put the cache uses.
function fakeBucket() {
  const store = new Map<string, string>();
  return {
    async get(key: string) {
      const body = store.get(key);
      if (body === undefined) return null;
      return { json: async () => JSON.parse(body) } as unknown;
    },
    async put(key: string, body: string) {
      store.set(key, body);
      return {} as unknown;
    },
  } as unknown as R2Bucket;
}

describe('parseStationsFeed', () => {
  test('maps fields, expands borough, splits routes, coerces ada', () => {
    const [s] = parseStationsFeed([row()]);
    expect(s).toEqual({
      gtfs_stop_id: 'R03',
      station_complex_id: '2',
      name: 'Astoria Blvd',
      borough: 'Queens',
      routes_served: ['N', 'W'],
      ada: 1,
      ada_northbound: true,
      ada_southbound: true,
    });
  });

  test('ada=2 (partial) is preserved; non-1 northbound flag is false', () => {
    const [s] = parseStationsFeed([row({ ada: '2', ada_northbound: '0' })]);
    expect(s?.ada).toBe(2);
    expect(s?.ada_northbound).toBe(false);
  });

  test('skips rows missing a stop id or name, and tolerates a non-array payload', () => {
    expect(parseStationsFeed([row(), { complex_id: '9' }, row({ stop_name: '' })])).toHaveLength(1);
    expect(parseStationsFeed({ not: 'an array' })).toEqual([]);
  });

  test('an unknown borough code passes through unchanged', () => {
    const [s] = parseStationsFeed([row({ borough: 'XX' })]);
    expect(s?.borough).toBe('XX');
  });
});

describe('stations cache round-trip', () => {
  test('write then read returns the records keyed by gtfs_stop_id', async () => {
    const bucket = fakeBucket();
    const stations = parseStationsFeed([row(), row({ gtfs_stop_id: 'R01', stop_name: 'Astoria-Ditmars Blvd' })]);
    await writeStationsCache(bucket, stations, NOW);
    const cached = await readStationsCache(bucket);
    expect(cached?.fetched_at).toBe(NOW);
    expect(Object.keys(cached!.stations).sort()).toEqual(['R01', 'R03']);
    expect(cached!.stations.R03?.name).toBe('Astoria Blvd');
  });

  test('a missing cache reads as null', async () => {
    expect(await readStationsCache(fakeBucket())).toBeNull();
  });
});

describe('snapshot stations surface', () => {
  test('embeds stations + stations_static freshness and validates', () => {
    const stations = parseStationsFeed([row()]);
    const byId = Object.fromEntries(stations.map((s) => [s.gtfs_stop_id, s]));
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: new Map(),
      rolls: {},
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      stations: byId,
      stationsStaticFreshness: NOW,
    });
    expect(snap.stations.R03?.borough).toBe('Queens');
    expect(snap.freshness.stations_static).toBe(NOW);
    expect(validate(snap), JSON.stringify(validate.errors, null, 2)).toBe(true);
  });

  test('stations_static stays null when no cache is supplied', () => {
    const snap = buildSnapshot({
      generatedAt: NOW,
      alertsFreshness: NOW,
      routeSnapshots: new Map(),
      rolls: {},
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
    });
    expect(snap.stations).toEqual({});
    expect(snap.freshness.stations_static).toBeNull();
  });
});
