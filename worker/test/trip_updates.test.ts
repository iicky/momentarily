import { describe, expect, test } from 'vitest';

import type { TripLite } from '../src/gtfsrt';
import { deriveRouteServiceMetric } from '../src/trip_updates';

function trip(over: Partial<TripLite>): TripLite {
  return {
    routeId: 'A',
    tripId: '000000_A..N00X000',
    isAssigned: true,
    direction: null,
    stopCount: 5,
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
