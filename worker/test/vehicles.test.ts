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
      by_direction: {
        north: { vehicles_n: 3, advanced_n: 0, stalled_n: 0, transitions: {} }, // default stop_id A01N
        south: { vehicles_n: 0, advanced_n: 0, stalled_n: 0, transitions: {} },
      },
    });
  });

  test('splits advance/stall by direction from the stop_id suffix', () => {
    const prev = stopPositions([
      veh({ tripId: 'n1', stopId: 'A05N' }),
      veh({ tripId: 's1', stopId: 'A05S' }),
      veh({ tripId: 's2', stopId: 'A07S' }),
    ]);
    const rows = deriveRouteMovementMetric(
      [
        veh({ routeId: 'A', tripId: 'n1', stopId: 'A06N' }), // north, advanced
        veh({ routeId: 'A', tripId: 's1', stopId: 'A05S' }), // south, stalled
        veh({ routeId: 'A', tripId: 's2', stopId: 'A09S' }), // south, advanced
      ],
      prev,
    );
    const a = rows.get('A')!;
    expect(a.by_direction.north).toEqual({
      vehicles_n: 1,
      advanced_n: 1,
      stalled_n: 0,
      transitions: { 'A05N>A06N': 1 },
    });
    expect(a.by_direction.south).toEqual({
      vehicles_n: 2,
      advanced_n: 1,
      stalled_n: 1,
      transitions: { 'A05S>A05S': 1, 'A07S>A09S': 1 },
    });
    // route totals still aggregate both directions
    expect(a).toMatchObject({ vehicles_n: 3, advanced_n: 2, stalled_n: 1 });
  });

  test('falls back to the trip_id direction char when stop_id has no suffix', () => {
    const rows = deriveRouteMovementMetric([
      veh({ routeId: 'L', tripId: '012345_L..S01R', stopId: 'L06' }), // no N/S on stop
    ]);
    expect(rows.get('L')!.by_direction.south.vehicles_n).toBe(1);
    expect(rows.get('L')!.by_direction.north.vehicles_n).toBe(0);
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

  test('a matched advance records the from>to transition and increments advanced_n', () => {
    const prev = stopPositions([veh({ tripId: 't1', stopId: 'A09N' })]);
    const rows = deriveRouteMovementMetric([veh({ routeId: 'A', tripId: 't1', stopId: 'A10N' })], prev);
    expect(rows.get('A')!.by_direction.north).toEqual({
      vehicles_n: 1,
      advanced_n: 1,
      stalled_n: 0,
      transitions: { 'A09N>A10N': 1 },
    });
  });

  test('a matched stall records the A>A self-transition and increments stalled_n', () => {
    const prev = stopPositions([veh({ tripId: 't1', stopId: 'A09N' })]);
    const rows = deriveRouteMovementMetric([veh({ routeId: 'A', tripId: 't1', stopId: 'A09N' })], prev);
    expect(rows.get('A')!.by_direction.north).toEqual({
      vehicles_n: 1,
      advanced_n: 0,
      stalled_n: 1,
      transitions: { 'A09N>A09N': 1 },
    });
  });

  test('two trips making the same transition sum their counts', () => {
    const prev = stopPositions([
      veh({ tripId: 't1', stopId: 'A09N' }),
      veh({ tripId: 't2', stopId: 'A09N' }),
    ]);
    const rows = deriveRouteMovementMetric(
      [
        veh({ routeId: 'A', tripId: 't1', stopId: 'A10N' }),
        veh({ routeId: 'A', tripId: 't2', stopId: 'A10N' }),
      ],
      prev,
    );
    expect(rows.get('A')!.by_direction.north.transitions).toEqual({ 'A09N>A10N': 2 });
  });

  test('an unmatched trip (tripId not in prevStops) records no transition', () => {
    const prev = stopPositions([veh({ tripId: 'other', stopId: 'A09N' })]);
    const rows = deriveRouteMovementMetric([veh({ routeId: 'A', tripId: 'new-trip', stopId: 'A10N' })], prev);
    const north = rows.get('A')!.by_direction.north;
    expect(north.transitions).toEqual({});
    expect(north.advanced_n).toBe(0);
    expect(north.stalled_n).toBe(0);
  });

  test('an unknown-direction vehicle records no transition and no dir-row advance/stall', () => {
    const prev = stopPositions([veh({ tripId: 'unknown_trip', stopId: 'R05' })]);
    const rows = deriveRouteMovementMetric([veh({ routeId: 'R', tripId: 'unknown_trip', stopId: 'R06' })], prev);
    const r = rows.get('R')!;
    expect(r.advanced_n).toBe(1); // route-level cross-tick still counts
    expect(r.by_direction.north).toEqual({ vehicles_n: 0, advanced_n: 0, stalled_n: 0, transitions: {} });
    expect(r.by_direction.south).toEqual({ vehicles_n: 0, advanced_n: 0, stalled_n: 0, transitions: {} });
  });

  test('transitions attach to the correct direction and never to route-level', () => {
    const prev = stopPositions([
      veh({ tripId: 'n1', stopId: 'A09N' }),
      veh({ tripId: 's1', stopId: 'A09S' }),
    ]);
    const rows = deriveRouteMovementMetric(
      [
        veh({ routeId: 'A', tripId: 'n1', stopId: 'A10N' }),
        veh({ routeId: 'A', tripId: 's1', stopId: 'A10S' }),
      ],
      prev,
    );
    const a = rows.get('A')!;
    expect(a.by_direction.north.transitions).toEqual({ 'A09N>A10N': 1 });
    expect(a.by_direction.south.transitions).toEqual({ 'A09S>A10S': 1 });
    expect(a).not.toHaveProperty('transitions');
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
