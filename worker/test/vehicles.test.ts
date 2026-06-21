import { describe, expect, test } from 'vitest';

import type { VehicleLite } from '../src/gtfsrt';
import { deriveRouteMovementMetric, stopPositions } from '../src/vehicles';

function veh(over: Partial<VehicleLite>): VehicleLite {
  return {
    routeId: 'A',
    tripId: '000000_A..N00X000',
    stopId: 'A01N',
    status: null,
    stopSeq: null,
    ...over,
  };
}

describe('deriveRouteMovementMetric', () => {
  test('counts vehicles and the stopped/moving split (status 1 = stopped)', () => {
    const rows = deriveRouteMovementMetric([
      veh({ routeId: 'A', tripId: 'a', status: 1 }), // STOPPED_AT
      veh({ routeId: 'A', tripId: 'b', status: null }), // field absent -> moving
      veh({ routeId: 'A', tripId: 'c', status: 2 }), // explicit IN_TRANSIT_TO -> moving
    ]);
    expect(rows.get('A')).toEqual({
      vehicles_n: 3,
      stopped_n: 1,
      moving_n: 2,
      advanced_n: 0,
      stalled_n: 0,
    });
  });

  test('folds express variants to the base route', () => {
    const rows = deriveRouteMovementMetric([
      veh({ routeId: '6', tripId: 'a' }),
      veh({ routeId: '6X', tripId: 'b' }),
    ]);
    expect(rows.has('6X')).toBe(false);
    expect(rows.get('6')!.vehicles_n).toBe(2);
  });

  test('cross-tick: unchanged stop_id is stalled, changed is advanced', () => {
    const prev = stopPositions([
      veh({ tripId: 'a', stopId: 'A01N' }),
      veh({ tripId: 'b', stopId: 'A05N' }),
    ]);
    const rows = deriveRouteMovementMetric(
      [
        veh({ routeId: 'A', tripId: 'a', stopId: 'A01N' }), // unchanged -> stalled
        veh({ routeId: 'A', tripId: 'b', stopId: 'A07N' }), // moved on -> advanced
        veh({ routeId: 'A', tripId: 'c', stopId: 'A02N' }), // new this tick -> neither
      ],
      prev,
    );
    expect(rows.get('A')).toMatchObject({ vehicles_n: 3, advanced_n: 1, stalled_n: 1 });
  });

  test('no previous state leaves cross-tick counters at 0', () => {
    const rows = deriveRouteMovementMetric([veh({ routeId: 'A', tripId: 'a', stopId: 'A01N' })]);
    expect(rows.get('A')).toMatchObject({ advanced_n: 0, stalled_n: 0 });
  });

  test('a frozen route reads all stalled, none advanced', () => {
    const prev = stopPositions([
      veh({ tripId: 'a', stopId: 'F01N' }),
      veh({ tripId: 'b', stopId: 'F02N' }),
    ]);
    const rows = deriveRouteMovementMetric(
      [
        veh({ routeId: 'F', tripId: 'a', stopId: 'F01N', status: 1 }),
        veh({ routeId: 'F', tripId: 'b', stopId: 'F02N', status: 1 }),
      ],
      prev,
    );
    expect(rows.get('F')).toMatchObject({ stalled_n: 2, advanced_n: 0, moving_n: 0 });
  });
});

describe('stopPositions', () => {
  test('maps trip_id to stop_id and drops empty trip_ids', () => {
    const map = stopPositions([
      veh({ tripId: 'a', stopId: 'A01N' }),
      veh({ tripId: '', stopId: 'A02N' }),
    ]);
    expect(map).toEqual({ a: 'A01N' });
  });
});
