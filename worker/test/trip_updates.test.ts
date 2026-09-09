import { describe, expect, test } from 'vitest';

import type { TripLite } from '../src/gtfsrt';
import { decodeTripUpdates } from '../src/gtfsrt';
import { deriveRouteServiceMetric } from '../src/trip_updates';
import { SERVICE_FIXTURE } from './gtfsrt_fixture';

function trip(over: Partial<TripLite>): TripLite {
  return {
    routeId: 'A',
    tripId: '000000_A..N00X000',
    isAssigned: true,
    direction: null,
    stopCount: 5,
    stopTimes: [],
    ...over,
  };
}

describe('deriveRouteServiceMetric', () => {
  test('counts assigned, total, with-movement, and direction split', () => {
    const rows = deriveRouteServiceMetric([
      trip({ routeId: 'A', isAssigned: true, direction: 1, stopCount: 4 }),
      trip({ routeId: 'A', isAssigned: true, direction: 3, stopCount: 0 }), // assigned, parked
      trip({ routeId: 'A', isAssigned: false, stopCount: 2 }), // scheduled, not running
    ]);
    expect(rows.get('A')).toEqual({
      assigned_n: 2,
      trips_n: 3,
      with_movement_n: 1, // only the one with stopCount > 0
      dir_n: 1,
      dir_s: 1,
    });
  });

  test('folds express variants to the base route', () => {
    const rows = deriveRouteServiceMetric([
      trip({ routeId: '6', isAssigned: true }),
      trip({ routeId: '6X', isAssigned: true }),
    ]);
    expect(rows.has('6X')).toBe(false);
    expect(rows.get('6')!.assigned_n).toBe(2);
  });

  test('falls back to trip_id direction char when the enum is absent', () => {
    const rows = deriveRouteServiceMetric([
      trip({ routeId: 'L', isAssigned: true, direction: null, tripId: '012345_L..S01R' }),
    ]);
    expect(rows.get('L')).toMatchObject({ dir_s: 1, dir_n: 0 });
  });

  test('a fully suspended route reads assigned_n 0', () => {
    const rows = deriveRouteServiceMetric([
      trip({ routeId: 'G', isAssigned: false }),
      trip({ routeId: 'G', isAssigned: false }),
    ]);
    expect(rows.get('G')).toMatchObject({ assigned_n: 0, trips_n: 2 });
  });
});

// The metric reads stopCount, not the rows, so materializing stop times must
// not move it. Same feed bytes, same map.
describe('deriveRouteServiceMetric over decoded feed bytes', () => {
  test('is unaffected by materialized stop times', () => {
    const rows = deriveRouteServiceMetric(decodeTripUpdates(SERVICE_FIXTURE));
    expect(Object.fromEntries(rows)).toEqual({
      A: { assigned_n: 2, trips_n: 3, with_movement_n: 1, dir_n: 1, dir_s: 1 },
      '6': { assigned_n: 1, trips_n: 1, with_movement_n: 1, dir_n: 0, dir_s: 1 },
    });
  });

  test('rows are materialized on the very trips the metric counted', () => {
    const trips = decodeTripUpdates(SERVICE_FIXTURE);
    expect(trips.map((t) => t.stopCount)).toEqual([2, 0, 1, 3]);
    expect(trips.map((t) => t.stopTimes.length)).toEqual([2, 0, 1, 3]);
    expect(trips[0]!.stopTimes[0]).toEqual({
      stopId: 'A15N',
      arrival: 1780000010,
      departure: 1780000040,
      scheduleRelationship: 0,
    });
    expect(trips[3]!.stopTimes.map((s) => s.scheduleRelationship)).toEqual([0, 1, 0]);
  });
});
