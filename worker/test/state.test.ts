/**
 * Rolling state R2 round-trips. Focused on the vehicle-movement carry map,
 * which lives in its own object (state/vehicle_stops.json) rather than
 * last_seen.json — keeping its ~700 entries off the per-tick state parse.
 */

import { describe, expect, test } from 'vitest';

import {
  readMovementMetric,
  readServiceMetric,
  readVehicleStops,
  writeMovementMetric,
  writeServiceMetric,
  writeVehicleStops,
} from '../src/state';
import type { MovementRow } from '../src/vehicles';
import type { ServiceRow } from '../src/trip_updates';

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

function moveRow(over: Partial<MovementRow>): MovementRow {
  return {
    vehicles_n: 10,
    stopped_n: 4,
    moving_n: 6,
    advanced_n: 8,
    stalled_n: 2,
    by_direction: {
      north: { vehicles_n: 5, advanced_n: 4, stalled_n: 1, transitions: {} },
      south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1, transitions: {} },
    },
    ...over,
  };
}

describe('movement metric carry doc', () => {
  test('round-trips per-route by-direction advanced_n/stalled_n, dropping the unused MovementRow fields', async () => {
    const { bucket } = fakeBucket();
    const moveRows = new Map<string, MovementRow>([
      [
        'A',
        moveRow({
          by_direction: {
            north: { vehicles_n: 5, advanced_n: 9, stalled_n: 1, transitions: {} },
            south: { vehicles_n: 4, advanced_n: 3, stalled_n: 2, transitions: {} },
          },
        }),
      ],
      [
        'F',
        moveRow({
          by_direction: {
            north: { vehicles_n: 2, advanced_n: 0, stalled_n: 2, transitions: {} },
            south: { vehicles_n: 3, advanced_n: 1, stalled_n: 0, transitions: {} },
          },
        }),
      ],
    ]);
    const observedAt = 1700000000;
    await writeMovementMetric(bucket, observedAt, moveRows);
    // toEqual is exact — it also proves vehicles_n/stopped_n/moving_n and the
    // by_direction vehicles_n counts are dropped from the persisted doc.
    expect(await readMovementMetric(bucket)).toEqual({
      observed_at: observedAt,
      rows: {
        A: { north: { advanced_n: 9, stalled_n: 1 }, south: { advanced_n: 3, stalled_n: 2 } },
        F: { north: { advanced_n: 0, stalled_n: 2 }, south: { advanced_n: 1, stalled_n: 0 } },
      },
    });
  });

  test('returns null when the object is absent', async () => {
    const { bucket } = fakeBucket();
    expect(await readMovementMetric(bucket)).toBeNull();
  });

  test('returns null when the object is corrupt', async () => {
    const { bucket, store } = fakeBucket();
    // Wrong shape (counts must be numbers) — schema parse should reject it and
    // the reader return null so the movement channel just drops out that tick.
    store.set(
      'state/movement_metric.json',
      JSON.stringify({
        observed_at: 1700000000,
        rows: { A: { north: { advanced_n: 'nope', stalled_n: 1 }, south: { advanced_n: 1, stalled_n: 1 } } },
      }),
    );
    expect(await readMovementMetric(bucket)).toBeNull();
  });
});

function svcRow(over: Partial<ServiceRow>): ServiceRow {
  return { assigned_n: 10, trips_n: 12, with_movement_n: 9, dir_n: 5, dir_s: 5, ...over };
}

describe('service metric carry doc', () => {
  test('round-trips per-route assigned_n, dropping the unused ServiceRow fields', async () => {
    const { bucket } = fakeBucket();
    const svcRows = new Map<string, ServiceRow>([
      ['A', svcRow({ assigned_n: 9, trips_n: 11, with_movement_n: 8, dir_n: 5, dir_s: 4 })],
      ['F', svcRow({ assigned_n: 0, trips_n: 6, with_movement_n: 0, dir_n: 0, dir_s: 0 })],
    ]);
    const observedAt = 1700000000;
    await writeServiceMetric(bucket, observedAt, svcRows);
    // toEqual is exact — it also proves trips_n/with_movement_n/dir_n/dir_s are
    // dropped from the persisted doc, leaving just the assigned_n numbers.
    expect(await readServiceMetric(bucket)).toEqual({
      observed_at: observedAt,
      rows: { A: 9, F: 0 },
    });
  });

  test('returns null when the object is absent', async () => {
    const { bucket } = fakeBucket();
    expect(await readServiceMetric(bucket)).toBeNull();
  });

  test('returns null when the object is corrupt', async () => {
    const { bucket, store } = fakeBucket();
    // Wrong shape (assigned_n must be a nonnegative int) — schema parse should
    // reject it and the reader return null so the service channel just drops
    // out that tick.
    store.set('state/service_metric.json', JSON.stringify({ observed_at: 1700000000, rows: { A: 'nope' } }));
    expect(await readServiceMetric(bucket)).toBeNull();
  });
});
