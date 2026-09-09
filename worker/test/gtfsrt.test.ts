/**
 * Decoder tests for the minimal GTFS-RT reader. Fixtures are hand-encoded
 * protobuf so the exact field numbers (verified against a live ACE feed) are
 * pinned: a wrong tag would silently mis-decode real feeds otherwise.
 */

import { describe, expect, test } from 'vitest';

import { decodeTripUpdates, decodeVehicles } from '../src/gtfsrt';
import {
  entity,
  feed,
  lenField,
  strField,
  tripDescriptor,
  tripUpdate,
  varField,
} from './gtfsrt_fixture';

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
      stopTimes: [
        { stopId: 'STOP0', arrival: null, departure: null, scheduleRelationship: 0 },
        { stopId: 'STOP1', arrival: null, departure: null, scheduleRelationship: 0 },
        { stopId: 'STOP2', arrival: null, departure: null, scheduleRelationship: 0 },
      ],
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

describe('decodeTripUpdates stop times', () => {
  test('materializes stop_id, arrival, departure and schedule_relationship', () => {
    const buf = feed(
      entity(
        tripUpdate(tripDescriptor({ routeId: 'Q', isAssigned: true, direction: 3 }), [
          { stopId: 'Q05S', arrival: 1780000001, departure: 1780000031, rel: 0 },
          { stopId: '626S', arrival: 1780000123, departure: 1780000153, rel: 1 },
          // Past 2^31: a shift-based varint would sign-flip this. Not a time a
          // feed emits today, but int64 is what the wire promises.
          { stopId: 'FAR', arrival: 2200000000 },
        ]),
      ),
    );
    const trips = decodeTripUpdates(buf);
    expect(trips[0]!.stopTimes).toEqual([
      { stopId: 'Q05S', arrival: 1780000001, departure: 1780000031, scheduleRelationship: 0 },
      { stopId: '626S', arrival: 1780000123, departure: 1780000153, scheduleRelationship: 1 },
      { stopId: 'FAR', arrival: 2200000000, departure: null, scheduleRelationship: 0 },
    ]);
  });

  test('missing arrival, missing departure, and both missing decode as null', () => {
    const buf = feed(
      entity(
        tripUpdate(tripDescriptor({ routeId: 'F', isAssigned: true }), [
          { stopId: 'DEP-ONLY', departure: 1780000200 }, // terminal-bound: no arrival
          { stopId: 'ARR-ONLY', arrival: 1780000300 }, // last stop: no departure
          { stopId: 'NEITHER' },
          { stopId: 'DELAY-ONLY', arrivalDelayOnly: true }, // event present, time absent
        ]),
      ),
    );
    expect(decodeTripUpdates(buf)[0]!.stopTimes).toEqual([
      { stopId: 'DEP-ONLY', arrival: null, departure: 1780000200, scheduleRelationship: 0 },
      { stopId: 'ARR-ONLY', arrival: 1780000300, departure: null, scheduleRelationship: 0 },
      { stopId: 'NEITHER', arrival: null, departure: null, scheduleRelationship: 0 },
      { stopId: 'DELAY-ONLY', arrival: null, departure: null, scheduleRelationship: 0 },
    ]);
  });

  test('absent stop_id decodes as empty string, absent relationship as SCHEDULED', () => {
    const buf = feed(
      entity(
        tripUpdate(tripDescriptor({ routeId: 'G', isAssigned: true }), [
          { arrival: 1780000400 },
          { stopId: '', arrival: 1780000500 },
        ]),
      ),
    );
    const rows = decodeTripUpdates(buf)[0]!.stopTimes;
    expect(rows.map((s) => s.stopId)).toEqual(['', '']);
    expect(rows.map((s) => s.scheduleRelationship)).toEqual([0, 0]);
  });

  test('unknown StopTimeUpdate fields are skipped without losing the row', () => {
    const buf = feed(
      entity(
        tripUpdate(tripDescriptor({ routeId: 'L', isAssigned: true }), [
          {
            // stop_sequence (1, varint) + an NYCT extension message (1001) we
            // do not read: both must be skipped by wire type, in sync.
            extraField: [...varField(1, 17), ...lenField(1001, strField(1, 'X'))],
            stopId: 'L06N',
            arrival: 1780000600,
            rel: 3,
          },
        ]),
      ),
    );
    expect(decodeTripUpdates(buf)[0]!.stopTimes).toEqual([
      { stopId: 'L06N', arrival: 1780000600, departure: null, scheduleRelationship: 3 },
    ]);
  });

  test('stopCount still equals the materialized row count', () => {
    const buf = feed(
      entity(tripUpdate(tripDescriptor({ routeId: '7', isAssigned: true }), 11)),
      entity(tripUpdate(tripDescriptor({ routeId: 'M', isAssigned: false }), 0)),
    );
    for (const t of decodeTripUpdates(buf)) {
      expect(t.stopCount).toBe(t.stopTimes.length);
    }
    expect(decodeTripUpdates(buf).map((t) => t.stopCount)).toEqual([11, 0]);
  });

  test('multi-byte stop ids decode as UTF-8', () => {
    const buf = feed(
      entity(
        tripUpdate(tripDescriptor({ routeId: 'A', isAssigned: true }), [
          { stopId: 'Ø7Ünion', arrival: 1780000700 },
        ]),
      ),
    );
    expect(decodeTripUpdates(buf)[0]!.stopTimes[0]!.stopId).toBe('Ø7Ünion');
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
  timestamp?: number;
}): number[] {
  return [
    ...lenField(1, tripDescriptor({
      ...(opts.tripId !== undefined ? { tripId: opts.tripId } : {}),
      ...(opts.routeId !== undefined ? { routeId: opts.routeId } : {}),
      withNyct: false,
    })),
    ...(opts.stopSeq !== undefined ? varField(3, opts.stopSeq) : []),
    ...(opts.status !== undefined ? varField(4, opts.status) : []),
    ...(opts.timestamp !== undefined ? varField(5, opts.timestamp) : []),
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
      timestamp: null,
    });
  });

  test('in-transit vehicle (no status/seq field) decodes status and seq null', () => {
    const buf = feed(entity(vehiclePosition({ routeId: 'C', tripId: 'Y..S', stopId: 'A15S' }), 4));
    const v = decodeVehicles(buf);
    expect(v[0]).toMatchObject({ status: null, stopSeq: null, stopId: 'A15S', timestamp: null });
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

  test('decodes the per-vehicle timestamp (field 5) when present', () => {
    const buf = feed(
      entity(
        vehiclePosition({
          routeId: 'A', tripId: 'X..N', stopId: 'A09N', status: 1, stopSeq: 31, timestamp: 1_750_000_000,
        }),
        4,
      ),
    );
    const v = decodeVehicles(buf);
    expect(v[0]).toMatchObject({ timestamp: 1_750_000_000 });
  });

  test('timestamp is null when the field is absent', () => {
    const buf = feed(entity(vehiclePosition({ routeId: 'A', tripId: 'X..N', stopId: 'A09N' }), 4));
    const v = decodeVehicles(buf);
    expect(v[0]!.timestamp).toBeNull();
  });
});
