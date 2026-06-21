/**
 * Decoder tests for the minimal GTFS-RT reader. Fixtures are hand-encoded
 * protobuf so the exact field numbers (verified against a live ACE feed) are
 * pinned: a wrong tag would silently mis-decode real feeds otherwise.
 */

import { describe, expect, test } from 'vitest';

import { decodeTripUpdates, decodeVehicles } from '../src/gtfsrt';

// --- tiny protobuf encoder (test-only) ---

function varint(n: number): number[] {
  const out: number[] = [];
  while (n > 0x7f) {
    out.push((n & 0x7f) | 0x80);
    n = Math.floor(n / 128);
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

function nyct(isAssigned: boolean, direction?: number): number[] {
  return [
    ...strField(1, 'TRAIN_ID'),
    ...varField(2, isAssigned ? 1 : 0),
    ...(direction !== undefined ? varField(3, direction) : []),
  ];
}
function tripDescriptor(opts: {
  tripId?: string;
  routeId?: string;
  isAssigned?: boolean;
  direction?: number;
  withNyct?: boolean;
}): number[] {
  return [
    ...strField(1, opts.tripId ?? '000000_A..N'),
    ...(opts.routeId !== undefined ? strField(5, opts.routeId) : []),
    ...(opts.withNyct === false
      ? []
      : lenField(1001, nyct(opts.isAssigned ?? false, opts.direction))),
  ];
}
function tripUpdate(desc: number[], nStops: number): number[] {
  const stops: number[] = [];
  for (let i = 0; i < nStops; i++) stops.push(...lenField(2, strField(1, `STOP${i}`)));
  return [...lenField(1, desc), ...stops];
}
function entity(body: number[], field = 3): number[] {
  return lenField(2, [...strField(1, 'entity-id'), ...lenField(field, body)]);
}
function feed(...entities: number[][]): Uint8Array {
  // FeedMessage.header (field 1) + entities (field 2)
  return new Uint8Array([...lenField(1, varField(1, 2)), ...entities.flat()]);
}

describe('decodeTripUpdates', () => {
  test('decodes route, is_assigned, direction, and stop count', () => {
    const buf = feed(
      entity(
        tripUpdate(
          tripDescriptor({ routeId: 'A', isAssigned: true, direction: 1, tripId: 'X..N' }),
          3,
        ),
      ),
    );
    const trips = decodeTripUpdates(buf);
    expect(trips).toHaveLength(1);
    expect(trips[0]).toEqual({
      routeId: 'A',
      tripId: 'X..N',
      isAssigned: true,
      direction: 1,
      stopCount: 3,
    });
  });

  test('unassigned trip decodes with isAssigned false', () => {
    const buf = feed(
      entity(tripUpdate(tripDescriptor({ routeId: 'C', isAssigned: false }), 0)),
    );
    const trips = decodeTripUpdates(buf);
    expect(trips).toHaveLength(1);
    expect(trips[0]!.isAssigned).toBe(false);
    expect(trips[0]!.stopCount).toBe(0);
  });

  test('trip with no NYCT extension defaults to not-assigned, null direction', () => {
    const buf = feed(
      entity(tripUpdate(tripDescriptor({ routeId: 'E', withNyct: false }), 1)),
    );
    const trips = decodeTripUpdates(buf);
    expect(trips[0]).toMatchObject({ routeId: 'E', isAssigned: false, direction: null });
  });

  test('entities without a trip_update are skipped', () => {
    // field 4 = vehicle, not trip_update (3) — must be ignored
    const buf = feed(
      entity(tripUpdate(tripDescriptor({ routeId: 'A', isAssigned: true }), 1)),
      entity(varField(1, 1), 4),
    );
    expect(decodeTripUpdates(buf)).toHaveLength(1);
  });

  test('trip with no route_id is dropped', () => {
    const buf = feed(entity(tripUpdate(tripDescriptor({ isAssigned: true }), 2)));
    expect(decodeTripUpdates(buf)).toHaveLength(0);
  });

  test('decodes multiple entities', () => {
    const buf = feed(
      entity(tripUpdate(tripDescriptor({ routeId: 'N', isAssigned: true, direction: 3 }), 5)),
      entity(tripUpdate(tripDescriptor({ routeId: 'Q', isAssigned: true, direction: 1 }), 4)),
      entity(tripUpdate(tripDescriptor({ routeId: 'R', isAssigned: false }), 0)),
    );
    const trips = decodeTripUpdates(buf);
    expect(trips.map((t) => t.routeId)).toEqual(['N', 'Q', 'R']);
    expect(trips.map((t) => t.direction)).toEqual([3, 1, null]);
  });
});

// VehiclePosition: trip(1), current_stop_sequence(3, varint), current_status(4,
// enum varint), stop_id(7, string). Status/seq omitted when not provided, which
// is how NYCT emits in-transit vehicles.
function vehiclePosition(opts: {
  routeId?: string;
  tripId?: string;
  stopId?: string;
  status?: number;
  stopSeq?: number;
}): number[] {
  return [
    ...lenField(1, tripDescriptor({
      ...(opts.tripId !== undefined ? { tripId: opts.tripId } : {}),
      ...(opts.routeId !== undefined ? { routeId: opts.routeId } : {}),
      withNyct: false,
    })),
    ...(opts.stopSeq !== undefined ? varField(3, opts.stopSeq) : []),
    ...(opts.status !== undefined ? varField(4, opts.status) : []),
    ...(opts.stopId !== undefined ? strField(7, opts.stopId) : []),
  ];
}

describe('decodeVehicles', () => {
  test('decodes route, stop_id, status, and stop_seq', () => {
    const buf = feed(
      entity(
        vehiclePosition({ routeId: 'A', tripId: 'X..N', stopId: 'A09N', status: 1, stopSeq: 31 }),
        4,
      ),
    );
    const v = decodeVehicles(buf);
    expect(v).toHaveLength(1);
    expect(v[0]).toEqual({
      routeId: 'A',
      tripId: 'X..N',
      stopId: 'A09N',
      status: 1,
      stopSeq: 31,
    });
  });

  test('in-transit vehicle (no status/seq field) decodes status and seq null', () => {
    const buf = feed(entity(vehiclePosition({ routeId: 'C', tripId: 'Y..S', stopId: 'A15S' }), 4));
    const v = decodeVehicles(buf);
    expect(v[0]).toMatchObject({ status: null, stopSeq: null, stopId: 'A15S' });
  });

  test('trip_update entities (field 3) are skipped by the vehicle decoder', () => {
    const buf = feed(
      entity(vehiclePosition({ routeId: 'A', tripId: 'X..N', stopId: 'A09N', status: 1 }), 4),
      entity(tripUpdate(tripDescriptor({ routeId: 'E', isAssigned: true }), 2)), // field 3
    );
    expect(decodeVehicles(buf)).toHaveLength(1);
  });

  test('vehicle with no route_id is dropped', () => {
    const buf = feed(entity(vehiclePosition({ tripId: 'Z', stopId: 'A01N', status: 1 }), 4));
    expect(decodeVehicles(buf)).toHaveLength(0);
  });

  test('decodes multiple vehicles', () => {
    const buf = feed(
      entity(vehiclePosition({ routeId: 'A', tripId: 'a', stopId: 'A01N', status: 1 }), 4),
      entity(vehiclePosition({ routeId: 'A', tripId: 'b', stopId: 'A05N' }), 4),
      entity(vehiclePosition({ routeId: 'C', tripId: 'c', stopId: 'C20S', status: 1 }), 4),
    );
    const v = decodeVehicles(buf);
    expect(v.map((x) => x.routeId)).toEqual(['A', 'A', 'C']);
    expect(v.map((x) => x.status)).toEqual([1, null, 1]);
  });
});
