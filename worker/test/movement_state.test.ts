import { describe, expect, test } from 'vitest';

import type { MovementRow } from '../src/vehicles';
import type { ServiceRow } from '../src/trip_updates';
import type { AdvanceBaselineCell, MovementBaseline, ServiceBaseline, TrainedParams } from '../src/params';
import type { MovementMetricDoc, ServiceMetricDoc } from '../src/state';
import { tod_bin } from '../src/hmm';
import {
  deriveMovementState,
  deriveMovementStates,
  MAX_MOVEMENT_METRIC_LAG_SECONDS,
  MAX_SERVICE_METRIC_LAG_SECONDS,
  movementObservationFields,
  serviceObservationFields,
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

function baselineCell(over: Partial<AdvanceBaselineCell>): AdvanceBaselineCell {
  return { p0: 0.9, alpha: 9, beta: 1, n: 50, ...over };
}

function trainedWithBaseline(movementBaseline: MovementBaseline): TrainedParams {
  return {
    schema_version: 'test',
    trained_at: 0,
    routes: {},
    dwell: {},
    dwellByAlert: {},
    movementBaseline,
    serviceBaseline: {},
  };
}

describe('deriveMovementState', () => {
  // 2026-06-15T16:00:00Z = 12:00 ET = tod_bin 2 (midday, 10-15h ET).
  const T0 = Date.parse('2026-06-15T16:00:00Z') / 1000;
  const ROUTE = 'A';
  const BIN = String(tod_bin(T0));

  test('trunk direction advancing at its own baseline rate reads normal', () => {
    // p0=0.9, advanced=8, stalled=1 (matched=9): post = (8*0.9+8)/(8+9) = 15.2/17 ~ 0.894 > 0.45.
    const move1 = move({
      by_direction: {
        north: { vehicles_n: 5, advanced_n: 8, stalled_n: 1 },
        south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1 },
      },
    });
    const trained = trainedWithBaseline({ [ROUTE]: { north: { [BIN]: baselineCell({}) } } });
    expect(deriveMovementState(ROUTE, move1, svc({}), trained, T0)).toBe('normal');
  });

  test('trunk direction far below its own baseline reads disrupted', () => {
    // p0=0.9, advanced=0, stalled=12 (matched=12): post = 7.2/20 = 0.36 <= 0.45.
    const move1 = move({
      by_direction: {
        north: { vehicles_n: 5, advanced_n: 0, stalled_n: 12 },
        south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1 },
      },
    });
    const trained = trainedWithBaseline({ [ROUTE]: { north: { [BIN]: baselineCell({}) } } });
    expect(deriveMovementState(ROUTE, move1, svc({}), trained, T0)).toBe('disrupted');
  });

  test('shuttle running at its own ~10% normal rate reads normal, not disrupted (debiasing)', () => {
    // The whole point of the rewrite: p0=0.1, advanced=1, stalled=9 (matched=10,
    // raw advance_frac 0.10) — the old fixed-0.25 rule called this disrupted.
    // post = (8*0.1+1)/(8+10) = 1.8/18 = 0.10 > 0.05 (RATIO*p0) -> normal.
    const move1 = move({
      by_direction: {
        north: { vehicles_n: 5, advanced_n: 1, stalled_n: 9 },
        south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1 },
      },
    });
    const trained = trainedWithBaseline({ [ROUTE]: { north: { [BIN]: baselineCell({ p0: 0.1 }) } } });
    expect(deriveMovementState(ROUTE, move1, undefined, trained, T0)).toBe('normal');
  });

  test('too few cross-tick matches is unjudgeable (null), even with a baseline', () => {
    const move1 = move({
      by_direction: {
        north: { vehicles_n: 5, advanced_n: 1, stalled_n: 1 },
        south: { vehicles_n: 5, advanced_n: 0, stalled_n: 1 },
      },
    });
    const trained = trainedWithBaseline({ [ROUTE]: { north: { [BIN]: baselineCell({}) } } });
    expect(deriveMovementState(ROUTE, move1, svc({}), trained, T0)).toBeNull();
  });

  test('no baseline cell for either direction is unjudgeable (null)', () => {
    const move1 = move({
      by_direction: {
        north: { vehicles_n: 5, advanced_n: 8, stalled_n: 1 },
        south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1 },
      },
    });
    expect(deriveMovementState(ROUTE, move1, svc({}), trainedWithBaseline({}), T0)).toBeNull();
  });

  test('no movement row is unjudgeable (null)', () => {
    expect(deriveMovementState(ROUTE, undefined, svc({}), null, T0)).toBeNull();
  });

  test('worst-of: one disrupted direction disrupts the whole route', () => {
    const move1 = move({
      by_direction: {
        north: { vehicles_n: 5, advanced_n: 0, stalled_n: 12 }, // disrupted
        south: { vehicles_n: 5, advanced_n: 8, stalled_n: 1 }, // normal
      },
    });
    const trained = trainedWithBaseline({
      [ROUTE]: { north: { [BIN]: baselineCell({}) }, south: { [BIN]: baselineCell({}) } },
    });
    expect(deriveMovementState(ROUTE, move1, svc({}), trained, T0)).toBe('disrupted');
  });

  test('both directions normal reads normal', () => {
    const move1 = move({
      by_direction: {
        north: { vehicles_n: 5, advanced_n: 8, stalled_n: 1 },
        south: { vehicles_n: 5, advanced_n: 8, stalled_n: 1 },
      },
    });
    const trained = trainedWithBaseline({
      [ROUTE]: { north: { [BIN]: baselineCell({}) }, south: { [BIN]: baselineCell({}) } },
    });
    expect(deriveMovementState(ROUTE, move1, svc({}), trained, T0)).toBe('normal');
  });

  test('assigned_n 0 with trips scheduled reads suspended', () => {
    expect(deriveMovementState(ROUTE, move({}), svc({ assigned_n: 0, trips_n: 5 }), null, T0)).toBe('suspended');
  });

  test('suspended wins over a would-be normal movement reading', () => {
    // trains technically advancing but nothing is dispatched — service is suspended
    expect(deriveMovementState(ROUTE, move({}), svc({ assigned_n: 0, trips_n: 3 }), null, T0)).toBe('suspended');
  });

  test('no vehicles and no assigned trains reads suspended', () => {
    expect(
      deriveMovementState(
        ROUTE,
        move({ vehicles_n: 0, advanced_n: 0, stalled_n: 0 }),
        svc({ assigned_n: 0, trips_n: 4 }),
        null,
        T0,
      ),
    ).toBe('suspended');
  });

  test('vehicles_n 0 does not read suspended when trains are assigned (feed inconsistency)', () => {
    // assigned trains but none in the vehicle feed: fall through; no baseline -> null
    expect(
      deriveMovementState(
        ROUTE,
        move({ vehicles_n: 0, advanced_n: 0, stalled_n: 0 }),
        svc({ assigned_n: 8 }),
        null,
        T0,
      ),
    ).toBeNull();
  });
});

describe('deriveMovementStates', () => {
  const T0 = Date.parse('2026-06-15T16:00:00Z') / 1000;
  const BIN = String(tod_bin(T0));

  test('maps each judgeable route and omits unjudgeable ones (too-few / no-baseline)', () => {
    const moveRows = new Map<string, MovementRow>([
      [
        'A',
        move({
          by_direction: {
            north: { vehicles_n: 5, advanced_n: 8, stalled_n: 1 },
            south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1 },
          },
        }),
      ], // normal
      [
        'F',
        move({
          by_direction: {
            north: { vehicles_n: 5, advanced_n: 0, stalled_n: 12 },
            south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1 },
          },
        }),
      ], // disrupted
      [
        'G',
        move({
          by_direction: {
            north: { vehicles_n: 5, advanced_n: 1, stalled_n: 1 },
            south: { vehicles_n: 5, advanced_n: 0, stalled_n: 1 },
          },
        }),
      ], // too few matches (baseline present) -> omitted
      [
        'N',
        move({
          by_direction: {
            north: { vehicles_n: 5, advanced_n: 8, stalled_n: 1 },
            south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1 },
          },
        }),
      ], // enough matches but no baseline -> omitted
    ]);
    const svcRows = new Map<string, ServiceRow>([
      ['L', svc({ assigned_n: 0, trips_n: 4 })], // suspended, movement absent
    ]);
    const trained = trainedWithBaseline({
      A: { north: { [BIN]: baselineCell({}) } },
      F: { north: { [BIN]: baselineCell({}) } },
      G: { north: { [BIN]: baselineCell({}) } },
    });
    expect(deriveMovementStates(moveRows, svcRows, trained, T0)).toEqual({
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
    return { schema_version: 'test', trained_at: 0, routes: {}, dwell: {}, dwellByAlert: {}, movementBaseline, serviceBaseline: {} };
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

describe('serviceObservationFields', () => {
  // 2026-06-15T16:00:00Z = 12:00 ET = tod_bin 2 (midday, 10-15h ET).
  const T0 = Date.parse('2026-06-15T16:00:00Z') / 1000;
  const ROUTE = 'Q';

  function metricDoc(observedAt: number, rows: ServiceMetricDoc['rows']): ServiceMetricDoc {
    return { observed_at: observedAt, rows };
  }

  function trainedWithServiceBaseline(serviceBaseline: ServiceBaseline): TrainedParams {
    return {
      schema_version: 'test',
      trained_at: 0,
      routes: {},
      dwell: {},
      dwellByAlert: {},
      movementBaseline: {},
      serviceBaseline,
    };
  }

  test('service_ratio = assigned_n / baseline when a baseline exists', () => {
    const metric = metricDoc(T0, { [ROUTE]: 8 });
    const trained = trainedWithServiceBaseline({ [ROUTE]: { [String(tod_bin(T0))]: 10 } });
    expect(serviceObservationFields(metric, trained, ROUTE, T0)).toEqual({
      service_ratio: 0.8,
      has_service: true,
    });
  });

  test('assigned_n 0 -> service_ratio 0 with has_service true (suspension signal, not dropped)', () => {
    const metric = metricDoc(T0, { [ROUTE]: 0 });
    const trained = trainedWithServiceBaseline({ [ROUTE]: { [String(tod_bin(T0))]: 10 } });
    expect(serviceObservationFields(metric, trained, ROUTE, T0)).toEqual({
      service_ratio: 0,
      has_service: true,
    });
  });

  test('null metric -> null', () => {
    const trained = trainedWithServiceBaseline({ [ROUTE]: { [String(tod_bin(T0))]: 10 } });
    expect(serviceObservationFields(null, trained, ROUTE, T0)).toBeNull();
  });

  test('metric older than MAX_SERVICE_METRIC_LAG_SECONDS is stale -> null', () => {
    const metricAt = T0 - MAX_SERVICE_METRIC_LAG_SECONDS - 1;
    const metric = metricDoc(metricAt, { [ROUTE]: 8 });
    const trained = trainedWithServiceBaseline({ [ROUTE]: { [String(tod_bin(metricAt))]: 10 } });
    expect(serviceObservationFields(metric, trained, ROUTE, T0)).toBeNull();
  });

  test('metric age exactly at MAX_SERVICE_METRIC_LAG_SECONDS is NOT stale (boundary inclusive)', () => {
    const metricAt = T0 - MAX_SERVICE_METRIC_LAG_SECONDS;
    const metric = metricDoc(metricAt, { [ROUTE]: 8 });
    const trained = trainedWithServiceBaseline({ [ROUTE]: { [String(tod_bin(metricAt))]: 10 } });
    expect(serviceObservationFields(metric, trained, ROUTE, T0)).toEqual({
      service_ratio: 0.8,
      has_service: true,
    });
  });

  test('route absent from metric.rows -> null', () => {
    const metric = metricDoc(T0, {});
    const trained = trainedWithServiceBaseline({ [ROUTE]: { [String(tod_bin(T0))]: 10 } });
    expect(serviceObservationFields(metric, trained, ROUTE, T0)).toBeNull();
  });

  test('no trainer baseline for the cell -> null', () => {
    const metric = metricDoc(T0, { [ROUTE]: 8 });
    const trained = trainedWithServiceBaseline({});
    expect(serviceObservationFields(metric, trained, ROUTE, T0)).toBeNull();
  });

  test('baseline of exactly 0 is treated as no baseline -> null (unlike assigned_n 0, which stays on)', () => {
    const metric = metricDoc(T0, { [ROUTE]: 8 });
    const trained = trainedWithServiceBaseline({ [ROUTE]: { [String(tod_bin(T0))]: 0 } });
    expect(serviceObservationFields(metric, trained, ROUTE, T0)).toBeNull();
  });

  test('null trained -> null', () => {
    const metric = metricDoc(T0, { [ROUTE]: 8 });
    expect(serviceObservationFields(metric, null, ROUTE, T0)).toBeNull();
  });

  describe('baseline gate uses the current-tick tod bin, not the metric tod bin', () => {
    // metric.observed_at 09:55 ET (tod_bin 1); the current tick is 10:03 ET
    // (tod_bin 2), 8 minutes later — well within MAX_SERVICE_METRIC_LAG_SECONDS,
    // so only the tod bin crosses, not staleness.
    const metricAt = Date.parse('2026-06-15T13:55:00Z') / 1000;
    const tickAt = Date.parse('2026-06-15T14:03:00Z') / 1000;

    test('sanity: metricAt and tickAt fall in different tod bins', () => {
      expect(tod_bin(metricAt)).toBe(1);
      expect(tod_bin(tickAt)).toBe(2);
    });

    test('baseline built for the current-tick bin is found even though the metric sits in a different bin', () => {
      const metric = metricDoc(metricAt, { [ROUTE]: 8 });
      const trained = trainedWithServiceBaseline({
        [ROUTE]: { [String(tod_bin(tickAt))]: 10 },
      });
      expect(serviceObservationFields(metric, trained, ROUTE, tickAt)).toEqual({
        service_ratio: 0.8,
        has_service: true,
      });
    });

    test('baseline built only for the metric bin is NOT found (gate must use the current-tick bin)', () => {
      const metric = metricDoc(metricAt, { [ROUTE]: 8 });
      const trained = trainedWithServiceBaseline({
        [ROUTE]: { [String(tod_bin(metricAt))]: 10 },
      });
      expect(serviceObservationFields(metric, trained, ROUTE, tickAt)).toBeNull();
    });
  });
});
