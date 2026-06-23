import { describe, expect, test } from 'vitest';

import type { MovementRow } from '../src/vehicles';
import type { ServiceRow } from '../src/trip_updates';
import { deriveMovementState, deriveMovementStates } from '../src/movement_state';

function move(over: Partial<MovementRow>): MovementRow {
  return {
    vehicles_n: 10,
    stopped_n: 4,
    moving_n: 6,
    advanced_n: 8,
    stalled_n: 2,
    by_direction: {
      north: { vehicles_n: 5, advanced_n: 4, stalled_n: 1 },
      south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1 },
    },
    ...over,
  };
}
function svc(over: Partial<ServiceRow>): ServiceRow {
  return { assigned_n: 10, trips_n: 12, with_movement_n: 9, dir_n: 5, dir_s: 5, ...over };
}

describe('deriveMovementState', () => {
  test('advancing trains read normal', () => {
    expect(deriveMovementState(move({ advanced_n: 8, stalled_n: 2 }), svc({}))).toBe('normal');
  });

  test('mostly-stalled trains read disrupted (<=25% advancing)', () => {
    expect(deriveMovementState(move({ advanced_n: 1, stalled_n: 7 }), svc({}))).toBe('disrupted');
  });

  test('exactly 25% advancing is disrupted (threshold inclusive)', () => {
    expect(deriveMovementState(move({ advanced_n: 1, stalled_n: 3 }), svc({}))).toBe('disrupted');
  });

  test('assigned_n 0 with trips scheduled reads suspended', () => {
    expect(
      deriveMovementState(move({ advanced_n: 8, stalled_n: 2 }), svc({ assigned_n: 0, trips_n: 5 })),
    ).toBe('suspended');
  });

  test('suspended wins over a would-be normal movement reading', () => {
    // trains technically advancing but nothing is dispatched — service is suspended
    expect(deriveMovementState(move({ advanced_n: 4, stalled_n: 0 }), svc({ assigned_n: 0, trips_n: 3 }))).toBe(
      'suspended',
    );
  });

  test('no vehicles and no assigned trains reads suspended', () => {
    expect(
      deriveMovementState(move({ vehicles_n: 0, advanced_n: 0, stalled_n: 0 }), svc({ assigned_n: 0, trips_n: 4 })),
    ).toBe('suspended');
  });

  test('too few cross-tick matches is unjudgeable (null)', () => {
    expect(deriveMovementState(move({ advanced_n: 1, stalled_n: 1 }), svc({}))).toBeNull();
  });

  test('no movement row is unjudgeable (null)', () => {
    expect(deriveMovementState(undefined, svc({}))).toBeNull();
  });

  test('vehicles_n 0 does not read suspended when trains are assigned (feed inconsistency)', () => {
    // assigned trains but none in the vehicle feed: fall through, no cross-tick matches -> null
    expect(
      deriveMovementState(move({ vehicles_n: 0, advanced_n: 0, stalled_n: 0 }), svc({ assigned_n: 8 })),
    ).toBeNull();
  });

  test('works with service row absent (movement only)', () => {
    expect(deriveMovementState(move({ advanced_n: 1, stalled_n: 9 }), undefined)).toBe('disrupted');
  });
});

describe('deriveMovementStates', () => {
  test('maps each judgeable route and omits unjudgeable ones', () => {
    const moveRows = new Map<string, MovementRow>([
      ['A', move({ advanced_n: 8, stalled_n: 2 })], // normal
      ['F', move({ advanced_n: 0, stalled_n: 6 })], // disrupted
      ['G', move({ advanced_n: 1, stalled_n: 1 })], // too few -> omitted
    ]);
    const svcRows = new Map<string, ServiceRow>([
      ['L', svc({ assigned_n: 0, trips_n: 4 })], // suspended, movement absent
    ]);
    expect(deriveMovementStates(moveRows, svcRows)).toEqual({
      A: 'normal',
      F: 'disrupted',
      L: 'suspended',
    });
  });
});
