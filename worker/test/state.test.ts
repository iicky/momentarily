/**
 * Rolling state R2 round-trips. Focused on the vehicle-movement carry map,
 * which lives in its own object (state/vehicle_stops.json) rather than
 * last_seen.json — keeping its ~700 entries off the per-tick state parse.
 */

import { describe, expect, test } from 'vitest';

import { readVehicleStops, writeVehicleStops } from '../src/state';

// Minimal in-memory R2 bucket — just the get/put these helpers touch.
function fakeBucket() {
  const store = new Map<string, string>();
  return {
    bucket: {
      async get(key: string) {
        const body = store.get(key);
        if (body === undefined) return null;
        return { json: async () => JSON.parse(body) } as unknown;
      },
      async put(key: string, body: string) {
        store.set(key, body);
        return {} as unknown;
      },
    } as unknown as R2Bucket,
    store,
  };
}

describe('vehicle stops carry map', () => {
  test('round-trips the trip_id -> stop_id map', async () => {
    const { bucket } = fakeBucket();
    const stops = { trip_a: 'R03', trip_b: 'A41', trip_c: 'R03' };
    await writeVehicleStops(bucket, stops);
    expect(await readVehicleStops(bucket)).toEqual(stops);
  });

  test('returns an empty map when the object is absent', async () => {
    const { bucket } = fakeBucket();
    expect(await readVehicleStops(bucket)).toEqual({});
  });

  test('returns an empty map when the object is corrupt', async () => {
    const { bucket, store } = fakeBucket();
    // Wrong shape (values must be strings) — schema parse should reject it and
    // the reader fall back to {} so cross-tick counters just stay 0 that tick.
    store.set('state/vehicle_stops.json', JSON.stringify({ trip_a: 42 }));
    expect(await readVehicleStops(bucket)).toEqual({});
  });
});
