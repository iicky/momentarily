import { describe, expect, test } from 'vitest';

import type { VehicleLite } from '../src/gtfsrt';
import { deriveRouteMovementMetric, deriveTrace, stopPositions } from '../src/vehicles';

function veh(over: Partial<VehicleLite>): VehicleLite {
  return {
    routeId: 'A',
    tripId: '000000_A..N00X000',
    stopId: 'A01N',
    status: null,
    stopSeq: null,
    timestamp: null,
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

  test('an unknown-direction vehicle records no transition and no advance/stall anywhere (route or dir-row)', () => {
    const prev = stopPositions([veh({ tripId: 'unknown_trip', stopId: 'R05' })]);
    const rows = deriveRouteMovementMetric([veh({ routeId: 'R', tripId: 'unknown_trip', stopId: 'R06' })], prev);
    const r = rows.get('R')!;
    // Route-level now requires a known direction too (same as the transitions
    // map always has), so route-level stays in lockstep with north+south
    // instead of counting a trip neither direction saw.
    expect(r.advanced_n).toBe(0);
    expect(r.stalled_n).toBe(0);
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

describe('deriveRouteMovementMetric: through-stop filter', () => {
  test('a stall at an out-of-set from-stop counts toward neither counter, but the transition is still recorded', () => {
    const prev = stopPositions([veh({ tripId: 't1', stopId: 'R05N' })]); // R05N: a terminal, not in the set
    const throughStops = new Set(['R|north|R09N']);
    const rows = deriveRouteMovementMetric(
      [veh({ routeId: 'R', tripId: 't1', stopId: 'R05N' })],
      prev,
      throughStops,
    );
    const r = rows.get('R')!;
    expect(r.advanced_n).toBe(0);
    expect(r.stalled_n).toBe(0);
    expect(r.by_direction.north.advanced_n).toBe(0);
    expect(r.by_direction.north.stalled_n).toBe(0);
    expect(r.by_direction.north.transitions).toEqual({ 'R05N>R05N': 1 });
  });

  test('a stall at an in-set from-stop still counts', () => {
    const prev = stopPositions([veh({ tripId: 't1', stopId: 'R09N' })]);
    const throughStops = new Set(['R|north|R09N']);
    const rows = deriveRouteMovementMetric(
      [veh({ routeId: 'R', tripId: 't1', stopId: 'R09N' })],
      prev,
      throughStops,
    );
    const r = rows.get('R')!;
    expect(r.stalled_n).toBe(1);
    expect(r.by_direction.north.stalled_n).toBe(1);
  });

  test('an advance out of an out-of-set from-stop is not counted; an advance into an out-of-set to-stop is', () => {
    const throughStops = new Set(['R|north|R09N']); // only R09N is a through stop
    const prev = stopPositions([
      veh({ tripId: 'out', stopId: 'R05N' }), // terminal, out of set
      veh({ tripId: 'in', stopId: 'R09N' }), // through stop, in set
    ]);
    const rows = deriveRouteMovementMetric(
      [
        veh({ routeId: 'R', tripId: 'out', stopId: 'R06N' }), // advances OUT of R05N — filtered
        veh({ routeId: 'R', tripId: 'in', stopId: 'R05N' }), // advances INTO R05N — from_stop R09N passes
      ],
      prev,
      throughStops,
    );
    const r = rows.get('R')!;
    expect(r.advanced_n).toBe(1);
    expect(r.by_direction.north.advanced_n).toBe(1);
    // Both transitions are still recorded raw, filter or no filter.
    expect(r.by_direction.north.transitions).toEqual({ 'R05N>R06N': 1, 'R09N>R05N': 1 });
  });

  test('the transitions map always records the excluded trip, whether it stalled or advanced', () => {
    const throughStops = new Set(['R|north|R09N']);
    const prev = stopPositions([veh({ tripId: 'excluded', stopId: 'R05N' })]);
    const rows = deriveRouteMovementMetric(
      [veh({ routeId: 'R', tripId: 'excluded', stopId: 'R06N' })],
      prev,
      throughStops,
    );
    expect(rows.get('R')!.by_direction.north.transitions).toEqual({ 'R05N>R06N': 1 });
  });

  test('a null set (the default) reproduces the pre-filter counts exactly', () => {
    const prev = stopPositions([
      veh({ tripId: 'n1', stopId: 'A05N' }),
      veh({ tripId: 's1', stopId: 'A05S' }),
      veh({ tripId: 's2', stopId: 'A07S' }),
    ]);
    const vehicles = [
      veh({ routeId: 'A', tripId: 'n1', stopId: 'A06N' }), // advance
      veh({ routeId: 'A', tripId: 's1', stopId: 'A05S' }), // stall
      veh({ routeId: 'A', tripId: 's2', stopId: 'A09S' }), // advance
    ];
    const implicit = deriveRouteMovementMetric(vehicles, prev);
    const explicit = deriveRouteMovementMetric(vehicles, prev, null);
    expect(explicit).toEqual(implicit);
    expect(explicit.get('A')).toMatchObject({ vehicles_n: 3, advanced_n: 2, stalled_n: 1 });
  });

  test('route-level counters equal the sum of the two directions even with an out-of-set stop and an unknown-direction trip mixed in', () => {
    const throughStops = new Set(['R|north|R09N']);
    const prev = stopPositions([
      veh({ tripId: 'in', stopId: 'R09N' }), // in-set, north
      veh({ tripId: 'out', stopId: 'R05N' }), // out-of-set, north
      veh({ tripId: 'unk', stopId: 'R12' }), // no N/S suffix, no ..N/..S trip_id -> unknown direction
    ]);
    const rows = deriveRouteMovementMetric(
      [
        veh({ routeId: 'R', tripId: 'in', stopId: 'R10N' }),
        veh({ routeId: 'R', tripId: 'out', stopId: 'R06N' }),
        veh({ routeId: 'R', tripId: 'unk', stopId: 'R13' }),
      ],
      prev,
      throughStops,
    );
    const r = rows.get('R')!;
    const sumAdvanced = r.by_direction.north.advanced_n + r.by_direction.south.advanced_n;
    const sumStalled = r.by_direction.north.stalled_n + r.by_direction.south.stalled_n;
    expect(r.advanced_n).toBe(sumAdvanced);
    expect(r.stalled_n).toBe(sumStalled);
    expect(r.advanced_n).toBe(1); // only 'in' (known direction, in-set from_stop) counts
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

describe('deriveTrace', () => {
  test('a trip whose position is unchanged from the previous call still emits a row — the old delta would have emitted nothing here', () => {
    const vehicles = [veh({ tripId: 'a', stopId: 'A01N', status: null })];
    deriveTrace(vehicles); // simulate a prior poll that already observed this exact position
    const rows = deriveTrace(vehicles);
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ trip_id: 'a', stop_id: 'A01N', stopped: false });
  });

  test('every in-service trip yields exactly one row per call', () => {
    const rows = deriveTrace([
      veh({ tripId: 'a', stopId: 'A01N' }),
      veh({ tripId: 'b', stopId: 'B02S' }),
      veh({ tripId: 'c', stopId: 'C03N' }),
    ]);
    expect(rows.map((r) => r.trip_id)).toEqual(['a', 'b', 'c']);
  });

  test('calling it twice with the same input returns identical output — pure function of the feed, which is what fixes the retry bug', () => {
    const vehicles = [
      veh({
        tripId: 'a', routeId: '6X', stopId: 'A09N', status: 1, stopSeq: 9, timestamp: 1_750_000_000,
      }),
      veh({ tripId: 'b', stopId: 'B02S', status: null }),
    ];
    expect(deriveTrace(vehicles)).toEqual(deriveTrace(vehicles));
  });

  test('a trip absent from this call\'s input simply produces no row — how a disappearance (censored traversal) is represented', () => {
    const atT = deriveTrace([
      veh({ tripId: 'a', stopId: 'A01N' }),
      veh({ tripId: 'b', stopId: 'B01N' }),
    ]);
    expect(atT.map((r) => r.trip_id)).toEqual(['a', 'b']);

    const atT1 = deriveTrace([veh({ tripId: 'a', stopId: 'A02N' })]); // b has vanished from the feed
    expect(atT1.map((r) => r.trip_id)).toEqual(['a']);
  });

  test('stop_seq is populated only when stopped, null while in transit', () => {
    const rows = deriveTrace([
      veh({ tripId: 'a', stopId: 'A09N', status: 1, stopSeq: 9 }),
      veh({ tripId: 'b', stopId: 'B09N', status: null, stopSeq: 9 }), // feed carried one anyway
    ]);
    expect(rows.find((r) => r.trip_id === 'a')!.stop_seq).toBe(9);
    expect(rows.find((r) => r.trip_id === 'b')!.stop_seq).toBeNull();
  });

  test('empty trip_id is skipped (cannot be matched across polls)', () => {
    expect(deriveTrace([veh({ tripId: '', stopId: 'A01N' })])).toEqual([]);
  });

  test('empty stop_id is skipped', () => {
    expect(deriveTrace([veh({ tripId: 'a', stopId: '' })])).toEqual([]);
  });

  test('row shape: route folded, direction derived, vehicle_ts passed through', () => {
    const rows = deriveTrace([veh({
      tripId: 'a', routeId: '6X', stopId: 'A09N', status: 1, stopSeq: 9, timestamp: 1_750_000_000,
    })]);
    expect(rows[0]).toEqual({
      trip_id: 'a',
      route_id: '6',
      direction: 'north',
      stop_id: 'A09N',
      stop_seq: 9,
      stopped: true,
      vehicle_ts: 1_750_000_000,
    });
  });

  test('direction falls back to the trip_id direction char when stop_id has no N/S suffix', () => {
    const rows = deriveTrace([veh({ tripId: '012345_L..S01R', routeId: 'L', stopId: 'L06' })]);
    expect(rows[0]!.direction).toBe('south');
  });

  test('vehicle_ts is null when the feed omits the per-vehicle timestamp', () => {
    const rows = deriveTrace([veh({ tripId: 'a', stopId: 'A01N', timestamp: null })]);
    expect(rows[0]!.vehicle_ts).toBeNull();
  });
});
