import { describe, expect, test } from 'vitest';

import type { MovementRow } from '../src/vehicles';
import type { ServiceRow } from '../src/trip_updates';
import type { AdvanceBaselineCell, MovementBaseline, TrainedParams } from '../src/params';
import type { MovementMetricDoc } from '../src/state';
import { tod_bin } from '../src/hmm';
import {
  deriveMovementState,
  deriveMovementStates,
  MAX_MOVEMENT_METRIC_LAG_SECONDS,
  movementObservationFields,
} from '../src/movement_state';

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

describe('movementObservationFields', () => {
  // 2026-06-15T16:00:00Z = 12:00 ET = tod_bin 2 (midday, 10-15h ET).
  const T0 = Date.parse('2026-06-15T16:00:00Z') / 1000;
  const ROUTE = 'Q';

  type MetricRowEntry = MovementMetricDoc['rows'][string];

  function dirCounts(advanced_n: number, stalled_n: number) {
    return { advanced_n, stalled_n };
  }

  // Defaults sum to matched_n 3 (the MIN_MATCHED_TRIPS floor) across both directions.
  function metricRow(over: Partial<MetricRowEntry>): MetricRowEntry {
    return { north: dirCounts(2, 0), south: dirCounts(1, 0), ...over };
  }

  function metricDoc(observedAt: number, rows: MovementMetricDoc['rows']): MovementMetricDoc {
    return { observed_at: observedAt, rows };
  }

  function baselineCell(over: Partial<AdvanceBaselineCell>): AdvanceBaselineCell {
    return { p0: 0.9, alpha: 9, beta: 1, n: 50, ...over };
  }

  function trainedWithBaseline(movementBaseline: MovementBaseline): TrainedParams {
    return { schema_version: 'test', trained_at: 0, routes: {}, dwell: {}, dwellByAlert: {}, movementBaseline };
  }

  test('aggregates both directions into advanced_n/matched_n when a baseline exists', () => {
    const metric = metricDoc(T0, { [ROUTE]: metricRow({}) });
    const trained = trainedWithBaseline({ [ROUTE]: { north: { [String(tod_bin(T0))]: baselineCell({}) } } });
    expect(movementObservationFields(metric, trained, ROUTE, T0)).toEqual({
      advanced_n: 3,
      matched_n: 3,
      has_movement: true,
    });
  });

  test('null metric -> null', () => {
    const trained = trainedWithBaseline({ [ROUTE]: { north: { [String(tod_bin(T0))]: baselineCell({}) } } });
    expect(movementObservationFields(null, trained, ROUTE, T0)).toBeNull();
  });

  test('metric older than MAX_MOVEMENT_METRIC_LAG_SECONDS is stale -> null', () => {
    const metricAt = T0 - MAX_MOVEMENT_METRIC_LAG_SECONDS - 1;
    const metric = metricDoc(metricAt, { [ROUTE]: metricRow({}) });
    const trained = trainedWithBaseline({ [ROUTE]: { north: { [String(tod_bin(metricAt))]: baselineCell({}) } } });
    expect(movementObservationFields(metric, trained, ROUTE, T0)).toBeNull();
  });

  test('metric age exactly at MAX_MOVEMENT_METRIC_LAG_SECONDS is NOT stale (boundary inclusive)', () => {
    const metricAt = T0 - MAX_MOVEMENT_METRIC_LAG_SECONDS;
    const metric = metricDoc(metricAt, { [ROUTE]: metricRow({}) });
    const trained = trainedWithBaseline({ [ROUTE]: { north: { [String(tod_bin(metricAt))]: baselineCell({}) } } });
    expect(movementObservationFields(metric, trained, ROUTE, T0)).toEqual({
      advanced_n: 3,
      matched_n: 3,
      has_movement: true,
    });
  });

  test('route absent from metric.rows -> null', () => {
    const metric = metricDoc(T0, {});
    const trained = trainedWithBaseline({ [ROUTE]: { north: { [String(tod_bin(T0))]: baselineCell({}) } } });
    expect(movementObservationFields(metric, trained, ROUTE, T0)).toBeNull();
  });

  test('matched_n below MIN_MATCHED_TRIPS is unjudgeable -> null', () => {
    const metric = metricDoc(T0, { [ROUTE]: metricRow({ north: dirCounts(1, 0), south: dirCounts(1, 0) }) }); // matched_n = 2
    const trained = trainedWithBaseline({ [ROUTE]: { north: { [String(tod_bin(T0))]: baselineCell({}) } } });
    expect(movementObservationFields(metric, trained, ROUTE, T0)).toBeNull();
  });

  test('no trainer baseline in either direction -> null', () => {
    const metric = metricDoc(T0, { [ROUTE]: metricRow({}) });
    const trained = trainedWithBaseline({});
    expect(movementObservationFields(metric, trained, ROUTE, T0)).toBeNull();
  });

  test('null trained -> null', () => {
    const metric = metricDoc(T0, { [ROUTE]: metricRow({}) });
    expect(movementObservationFields(metric, null, ROUTE, T0)).toBeNull();
  });

  test('south-only baseline still satisfies the gate (either direction is enough)', () => {
    const metric = metricDoc(T0, { [ROUTE]: metricRow({}) });
    const trained = trainedWithBaseline({ [ROUTE]: { south: { [String(tod_bin(T0))]: baselineCell({}) } } });
    expect(movementObservationFields(metric, trained, ROUTE, T0)).toEqual({
      advanced_n: 3,
      matched_n: 3,
      has_movement: true,
    });
  });

  describe('baseline gate uses the current-tick tod bin, not the metric tod bin', () => {
    // metric.observed_at 09:55 ET (tod_bin 1); the current tick is 10:03 ET
    // (tod_bin 2), 8 minutes later — well within MAX_MOVEMENT_METRIC_LAG_SECONDS,
    // so only the tod bin crosses, not staleness.
    const metricAt = Date.parse('2026-06-15T13:55:00Z') / 1000;
    const tickAt = Date.parse('2026-06-15T14:03:00Z') / 1000;

    test('sanity: metricAt and tickAt fall in different tod bins', () => {
      expect(tod_bin(metricAt)).toBe(1);
      expect(tod_bin(tickAt)).toBe(2);
    });

    test('baseline built for the current-tick bin is found even though the metric sits in a different bin', () => {
      const metric = metricDoc(metricAt, { [ROUTE]: metricRow({}) });
      const trained = trainedWithBaseline({
        [ROUTE]: { north: { [String(tod_bin(tickAt))]: baselineCell({}) } },
      });
      expect(movementObservationFields(metric, trained, ROUTE, tickAt)).toEqual({
        advanced_n: 3,
        matched_n: 3,
        has_movement: true,
      });
    });

    test('baseline built only for the metric bin is NOT found (gate must use the current-tick bin)', () => {
      const metric = metricDoc(metricAt, { [ROUTE]: metricRow({}) });
      const trained = trainedWithBaseline({
        [ROUTE]: { north: { [String(tod_bin(metricAt))]: baselineCell({}) } },
      });
      expect(movementObservationFields(metric, trained, ROUTE, tickAt)).toBeNull();
    });
  });
});
