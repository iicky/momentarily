import { describe, expect, test } from 'vitest';

import type { MovementRow } from '../src/vehicles';
import type { ServiceRow } from '../src/trip_updates';
import type { AdvanceBaselineCell, MovementBaseline, ServiceBaseline, TrainedParams } from '../src/params';
import { scheduleRateFor } from '../src/params';
import type { MovementMetricDoc, ServiceMetricDoc, ServiceQuantiles } from '../src/state';
import { schedule_bin, tod_bin } from '../src/hmm';
import {
  deriveMovementState,
  deriveMovementStates,
  deriveServiceQuantileRatios,
  deriveServiceRatios,
  deriveServiceState,
  deriveServiceStates,
  seedNormalServiceRegimes,
  serviceQuantileFor,
  serviceQuantileRatiosFor,
  serviceRatioFor,
  SERVICE_DEGRADE_RATIO,
  SERVICE_DEBOUNCE_TICKS,
  MAX_MOVEMENT_METRIC_LAG_SECONDS,
  MAX_SERVICE_METRIC_LAG_SECONDS,
  movementObservationFields,
  serviceObservationFields,
} from '../src/movement_state';
import { advanceRegimes, pruneIdleRegimes, MAX_IDLE_SEC } from '../src/regime';
import type { RegimeEntry } from '../src/regime';
import type { ServiceCondition } from '../src/movement_state';

function move(over: Partial<MovementRow>): MovementRow {
  return {
    vehicles_n: 10,
    stopped_n: 4,
    moving_n: 6,
    advanced_n: 8,
    stalled_n: 2,
    by_direction: {
      north: { vehicles_n: 5, advanced_n: 4, stalled_n: 1, transitions: {} },
      south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1, transitions: {} },
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

function trainedWithBaseline(
  movementBaseline: MovementBaseline,
  scheduleRate: TrainedParams['scheduleRate'] = {},
): TrainedParams {
  return {
    schema_version: 'test',
    trained_at: 0,
    routes: {},
    dwell: {},
    dwellByAlert: {},
    dwellMovement: {},
    movementBaseline,
    throughStops: null,
    serviceBaseline: {},
    serviceBaselineHourly: {},
    scheduleRate,
  };
}

// scheduleRate cell for (routeId, schedule_bin(observedAt)) = rate.
function scheduleRateFixture(routeId: string, observedAt: number, rate: number): TrainedParams['scheduleRate'] {
  return { [routeId]: { [schedule_bin(observedAt)]: rate } };
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
        north: { vehicles_n: 5, advanced_n: 8, stalled_n: 1, transitions: {} },
        south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1, transitions: {} },
      },
    });
    const trained = trainedWithBaseline({ [ROUTE]: { north: { [BIN]: baselineCell({}) } } });
    expect(deriveMovementState(ROUTE, move1, svc({}), trained, T0)).toBe('normal');
  });

  test('trunk direction far below its own baseline reads disrupted', () => {
    // p0=0.9, advanced=0, stalled=12 (matched=12): post = 7.2/20 = 0.36 <= 0.45.
    const move1 = move({
      by_direction: {
        north: { vehicles_n: 5, advanced_n: 0, stalled_n: 12, transitions: {} },
        south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1, transitions: {} },
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
        north: { vehicles_n: 5, advanced_n: 1, stalled_n: 9, transitions: {} },
        south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1, transitions: {} },
      },
    });
    const trained = trainedWithBaseline({ [ROUTE]: { north: { [BIN]: baselineCell({ p0: 0.1 }) } } });
    expect(deriveMovementState(ROUTE, move1, undefined, trained, T0)).toBe('normal');
  });

  // --- Three-way significance gate: a degenerate-low
  // baseline no longer misfires disrupted on ordinary low-advance noise unless
  // the drop is also statistically significant against that baseline. ---

  test('shuttle with a degenerate-low baseline and zero advances abstains, not disrupted (THE FIX)', () => {
    // p0=0.125, advanced=0, stalled=8 (matched=8): post = (8*0.125+0)/(8+8) = 0.0625
    // == 0.5*p0 (<=); tail = 0.875**8 ~= 0.3436 > 0.05 -> null, not the old false 'disrupted'.
    const move1 = move({
      by_direction: {
        north: { vehicles_n: 5, advanced_n: 0, stalled_n: 8, transitions: {} },
        south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1, transitions: {} },
      },
    });
    const trained = trainedWithBaseline({ [ROUTE]: { north: { [BIN]: baselineCell({ p0: 0.125 }) } } });
    expect(deriveMovementState(ROUTE, move1, svc({}), trained, T0)).toBeNull();
  });

  test('shuttle freeze with enough matched trips reaches significance and still reads disrupted', () => {
    // p0=0.125, advanced=0, stalled=25 (matched=25): tail = 0.875**25 ~= 0.0356 <= 0.05,
    // post ~= 0.030 <= 0.0625 -> disrupted.
    const move1 = move({
      by_direction: {
        north: { vehicles_n: 5, advanced_n: 0, stalled_n: 25, transitions: {} },
        south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1, transitions: {} },
      },
    });
    const trained = trainedWithBaseline({ [ROUTE]: { north: { [BIN]: baselineCell({ p0: 0.125 }) } } });
    expect(deriveMovementState(ROUTE, move1, svc({}), trained, T0)).toBe('disrupted');
  });

  test('mid-range trunk direction frozen for 17 matches reaches significance and reads disrupted', () => {
    // p0=0.55, advanced=0, stalled=17 (matched=17): post = (8*0.55+0)/(8+17) = 0.176
    // <= 0.275 (0.5*p0); tail = 0.45**17 ~= 1.2e-6 <= 0.05 -> disrupted.
    const move1 = move({
      by_direction: {
        north: { vehicles_n: 5, advanced_n: 0, stalled_n: 17, transitions: {} },
        south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1, transitions: {} },
      },
    });
    const trained = trainedWithBaseline({ [ROUTE]: { north: { [BIN]: baselineCell({ p0: 0.55 }) } } });
    expect(deriveMovementState(ROUTE, move1, svc({}), trained, T0)).toBe('disrupted');
  });

  test('mid-range trunk direction advancing above half its own baseline reads normal', () => {
    // p0=0.55, advanced=8, stalled=9 (matched=17): post = (8*0.55+8)/(8+17) = 0.496 > 0.275 -> normal.
    const move1 = move({
      by_direction: {
        north: { vehicles_n: 5, advanced_n: 8, stalled_n: 9, transitions: {} },
        south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1, transitions: {} },
      },
    });
    const trained = trainedWithBaseline({ [ROUTE]: { north: { [BIN]: baselineCell({ p0: 0.55 }) } } });
    expect(deriveMovementState(ROUTE, move1, svc({}), trained, T0)).toBe('normal');
  });

  test('shuttle below MIN_MATCHED_TRIPS abstains before the significance gate even runs', () => {
    // p0=0.125, advanced=0, stalled=1 (matched=1 < MIN_MATCHED_TRIPS=3): the
    // matched-floor guard short-circuits before the posterior/significance path
    // is ever evaluated, however degenerate the baseline is (unchanged guard).
    const move1 = move({
      by_direction: {
        north: { vehicles_n: 5, advanced_n: 0, stalled_n: 1, transitions: {} },
        south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1, transitions: {} },
      },
    });
    const trained = trainedWithBaseline({ [ROUTE]: { north: { [BIN]: baselineCell({ p0: 0.125 }) } } });
    expect(deriveMovementState(ROUTE, move1, svc({}), trained, T0)).toBeNull();
  });

  test('too few cross-tick matches is unjudgeable (null), even with a baseline', () => {
    const move1 = move({
      by_direction: {
        north: { vehicles_n: 5, advanced_n: 1, stalled_n: 1, transitions: {} },
        south: { vehicles_n: 5, advanced_n: 0, stalled_n: 1, transitions: {} },
      },
    });
    const trained = trainedWithBaseline({ [ROUTE]: { north: { [BIN]: baselineCell({}) } } });
    expect(deriveMovementState(ROUTE, move1, svc({}), trained, T0)).toBeNull();
  });

  test('no baseline cell for either direction is unjudgeable (null)', () => {
    const move1 = move({
      by_direction: {
        north: { vehicles_n: 5, advanced_n: 8, stalled_n: 1, transitions: {} },
        south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1, transitions: {} },
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
        north: { vehicles_n: 5, advanced_n: 0, stalled_n: 12, transitions: {} }, // disrupted
        south: { vehicles_n: 5, advanced_n: 8, stalled_n: 1, transitions: {} }, // normal
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
        north: { vehicles_n: 5, advanced_n: 8, stalled_n: 1, transitions: {} },
        south: { vehicles_n: 5, advanced_n: 8, stalled_n: 1, transitions: {} },
      },
    });
    const trained = trainedWithBaseline({
      [ROUTE]: { north: { [BIN]: baselineCell({}) }, south: { [BIN]: baselineCell({}) } },
    });
    expect(deriveMovementState(ROUTE, move1, svc({}), trained, T0)).toBe('normal');
  });

  test('trains present with assigned_n 0 reads normal (movement wins over dispatch lag)', () => {
    // trains are physically advancing even though trip-updates shows nothing
    // dispatched: movement classification wins over the suspended check.
    const trained = trainedWithBaseline({ [ROUTE]: { north: { [BIN]: baselineCell({}) } } });
    expect(deriveMovementState(ROUTE, move({}), svc({ assigned_n: 0, trips_n: 5 }), trained, T0)).toBe('normal');
  });

  test('movement wins over dispatch even when the movement call is disrupted', () => {
    // same dispatch-lag premise, but the movement call itself is disrupted —
    // proves suspended never overrides movement, whichever way movement calls it.
    const move1 = move({
      by_direction: {
        north: { vehicles_n: 5, advanced_n: 0, stalled_n: 12, transitions: {} },
        south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1, transitions: {} },
      },
    });
    const trained = trainedWithBaseline({ [ROUTE]: { north: { [BIN]: baselineCell({}) } } });
    expect(deriveMovementState(ROUTE, move1, svc({ assigned_n: 0, trips_n: 3 }), trained, T0)).toBe('disrupted');
  });

  test('no vehicles and no assigned trains reads suspended when the schedule rate is unknown', () => {
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
    // assigned trains but none in the vehicle feed: fall through; dispatched -> null
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

  test('no service and a low schedule rate reads not_scheduled', () => {
    const trained = trainedWithBaseline({}, scheduleRateFixture(ROUTE, T0, 0.1));
    expect(deriveMovementState(ROUTE, undefined, svc({ assigned_n: 0 }), trained, T0)).toBe('not_scheduled');
  });

  test('no service and a high schedule rate reads suspended', () => {
    const trained = trainedWithBaseline({}, scheduleRateFixture(ROUTE, T0, 0.9));
    expect(
      deriveMovementState(
        ROUTE,
        move({ vehicles_n: 0, advanced_n: 0, stalled_n: 0 }),
        svc({ assigned_n: 0 }),
        trained,
        T0,
      ),
    ).toBe('suspended');
  });

  test('no service and no schedule-rate cell for this route reads suspended (conservative)', () => {
    const trained = trainedWithBaseline({}, scheduleRateFixture('OTHER', T0, 0.1));
    expect(deriveMovementState(ROUTE, undefined, svc({ assigned_n: 0 }), trained, T0)).toBe('suspended');
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
            north: { vehicles_n: 5, advanced_n: 8, stalled_n: 1, transitions: {} },
            south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1, transitions: {} },
          },
        }),
      ], // normal
      [
        'F',
        move({
          by_direction: {
            north: { vehicles_n: 5, advanced_n: 0, stalled_n: 12, transitions: {} },
            south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1, transitions: {} },
          },
        }),
      ], // disrupted
      [
        'G',
        move({
          by_direction: {
            north: { vehicles_n: 5, advanced_n: 1, stalled_n: 1, transitions: {} },
            south: { vehicles_n: 5, advanced_n: 0, stalled_n: 1, transitions: {} },
          },
        }),
      ], // too few matches (baseline present) -> omitted
      [
        'N',
        move({
          by_direction: {
            north: { vehicles_n: 5, advanced_n: 8, stalled_n: 1, transitions: {} },
            south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1, transitions: {} },
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

describe('deriveMovementStates absent routes', () => {
  const T0 = Date.parse('2026-06-15T16:00:00Z') / 1000;
  const TOD_BIN = String(tod_bin(T0));
  const SCHED_BIN = schedule_bin(T0);
  // A schedule_bin distinct from SCHED_BIN (same wd/we prefix, different hour) so a
  // scheduleRate cell keyed by it never matches this tick's bin.
  const OTHER_BIN = `${SCHED_BIN.slice(0, 2)}${String((Number(SCHED_BIN.slice(2)) + 1) % 24).padStart(2, '0')}`;

  test('absent route with a low schedule rate at this bin reads not_scheduled', () => {
    const trained = trainedWithBaseline({}, { LOW: { [SCHED_BIN]: 0.1 } });
    expect(
      deriveMovementStates(new Map<string, MovementRow>(), new Map<string, ServiceRow>(), trained, T0),
    ).toEqual({ LOW: 'not_scheduled' });
  });

  test('absent route with a high schedule rate at this bin is omitted', () => {
    const trained = trainedWithBaseline({}, { HIGH: { [SCHED_BIN]: 0.9 } });
    expect(
      deriveMovementStates(new Map<string, MovementRow>(), new Map<string, ServiceRow>(), trained, T0),
    ).toEqual({});
  });

  test('absent route with no rate cell for this bin is omitted', () => {
    const trained = trainedWithBaseline({}, { NOCELL: { [OTHER_BIN]: 0.1 } });
    expect(
      deriveMovementStates(new Map<string, MovementRow>(), new Map<string, ServiceRow>(), trained, T0),
    ).toEqual({});
  });

  test('a route present in moveRows is judged normally, not overridden by a low schedule-rate cell', () => {
    const moveRows = new Map<string, MovementRow>([
      [
        'A',
        move({
          by_direction: {
            north: { vehicles_n: 5, advanced_n: 8, stalled_n: 1, transitions: {} },
            south: { vehicles_n: 5, advanced_n: 4, stalled_n: 1, transitions: {} },
          },
        }),
      ],
    ]);
    const trained = trainedWithBaseline(
      { A: { north: { [TOD_BIN]: baselineCell({}) } } },
      { A: { [SCHED_BIN]: 0.1 } },
    );
    expect(deriveMovementStates(moveRows, new Map<string, ServiceRow>(), trained, T0)).toEqual({
      A: 'normal',
    });
  });

  test('trained: null skips the absent pass without throwing', () => {
    expect(() =>
      deriveMovementStates(new Map<string, MovementRow>(), new Map<string, ServiceRow>(), null, T0),
    ).not.toThrow();
    expect(
      deriveMovementStates(new Map<string, MovementRow>(), new Map<string, ServiceRow>(), null, T0),
    ).toEqual({});
  });
});

describe('schedule_bin', () => {
  test('weekday hour maps to a wd-prefixed bin', () => {
    // 2026-06-15T16:00:00Z = Mon 12:00 ET.
    expect(schedule_bin(Date.parse('2026-06-15T16:00:00Z') / 1000)).toBe('wd12');
  });

  test('weekend hour maps to a we-prefixed bin', () => {
    // 2026-06-21T02:00:00Z = Sat 22:00 ET.
    expect(schedule_bin(Date.parse('2026-06-21T02:00:00Z') / 1000)).toBe('we22');
  });
});

describe('scheduleRateFor', () => {
  const T0 = Date.parse('2026-06-15T16:00:00Z') / 1000;
  const BIN = schedule_bin(T0);

  test('returns the trainer-set rate for a known (route, bin) cell', () => {
    const trained = trainedWithBaseline({}, { A: { [BIN]: 0.35 } });
    expect(scheduleRateFor(trained, 'A', BIN)).toBe(0.35);
  });

  test('returns null for an absent cell', () => {
    const trained = trainedWithBaseline({}, { A: { [BIN]: 0.35 } });
    expect(scheduleRateFor(trained, 'B', BIN)).toBeNull();
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
    return {
      schema_version: 'test',
      trained_at: 0,
      routes: {},
      dwell: {},
      dwellByAlert: {},
      dwellMovement: {},
      movementBaseline,
      throughStops: null,
      serviceBaseline: {},
      serviceBaselineHourly: {},
      scheduleRate: {},
    };
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
      dwellMovement: {},
      movementBaseline: {},
      throughStops: null,
      serviceBaseline,
      serviceBaselineHourly: {},
      scheduleRate: {},
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

describe('deriveServiceState / deriveServiceStates (supply axis)', () => {
  const T0 = Date.parse('2026-06-15T16:00:00Z') / 1000; // 12:00 ET
  const ROUTE = 'Q';
  const BIN = schedule_bin(T0);

  const baseline: ServiceBaseline = { [ROUTE]: { [BIN]: 10 } };

  test('assigned_n well below baseline reads degraded', () => {
    expect(deriveServiceState(ROUTE, svc({ assigned_n: 4 }), baseline, T0)).toBe('degraded');
  });

  test('assigned_n near baseline reads normal', () => {
    expect(deriveServiceState(ROUTE, svc({ assigned_n: 8 }), baseline, T0)).toBe('normal');
  });

  test('the degrade threshold is strict: exactly at the floor is not degraded', () => {
    const atFloor = svc({ assigned_n: SERVICE_DEGRADE_RATIO * 10 }); // ratio 0.5
    expect(deriveServiceState(ROUTE, atFloor, baseline, T0)).toBe('normal');
  });

  test('hysteresis: once degraded, a ratio in the [degrade, recover) band holds degraded', () => {
    const mid = svc({ assigned_n: 6 }); // ratio 0.6: above 0.5 floor, below 0.8 ceiling
    expect(deriveServiceState(ROUTE, mid, baseline, T0, 'degraded')).toBe('degraded');
    // From normal the same reading is normal — the band only holds an open one.
    expect(deriveServiceState(ROUTE, mid, baseline, T0, 'normal')).toBe('normal');
  });

  test('hysteresis: a degraded route recovers only at/above the recover ceiling', () => {
    expect(deriveServiceState(ROUTE, svc({ assigned_n: 8 }), baseline, T0, 'degraded')).toBe('normal'); // 0.8
    expect(deriveServiceState(ROUTE, svc({ assigned_n: 7 }), baseline, T0, 'degraded')).toBe('degraded'); // 0.7
  });

  test('deriveServiceStates keys the band off the prior committed regimes', () => {
    const rows = new Map<string, ServiceRow>([[ROUTE, svc({ assigned_n: 6 })]]); // ratio 0.6
    const priorDegraded: Record<string, { state: ServiceCondition }> = { [ROUTE]: { state: 'degraded' } };
    expect(deriveServiceStates(rows, baseline, T0, priorDegraded)).toEqual({ [ROUTE]: 'degraded' });
    expect(deriveServiceStates(rows, baseline, T0)).toEqual({ [ROUTE]: 'normal' });
  });

  test('no hourly baseline for the cell is unknown, never a default call', () => {
    expect(deriveServiceState(ROUTE, svc({ assigned_n: 4 }), {}, T0)).toBe(
      'unknown',
    );
  });

  test('no service row is unknown', () => {
    expect(deriveServiceState(ROUTE, undefined, baseline, T0)).toBe('unknown');
  });

  test('the axis uses the schedule bin, not the coarse tod bin', () => {
    // A baseline keyed only by tod_bin (the emission channel's granularity) is
    // NOT found by the schedule_bin-keyed axis lookup, so it reads unknown.
    const todKeyed: ServiceBaseline = { [ROUTE]: { [String(tod_bin(T0))]: 10 } };
    expect(deriveServiceState(ROUTE, svc({ assigned_n: 4 }), todKeyed, T0)).toBe('unknown');
  });

  test('deriveServiceStates keeps judgeable routes and omits unknown ones', () => {
    const rows = new Map<string, ServiceRow>([
      [ROUTE, svc({ assigned_n: 4 })], // degraded
      ['1', svc({ assigned_n: 12 })], // no baseline -> unknown -> omitted
    ]);
    expect(deriveServiceStates(rows, baseline, T0)).toEqual({ [ROUTE]: 'degraded' });
  });

  test('serviceRatioFor is assigned_n / baseline, null when unjudgeable', () => {
    expect(serviceRatioFor(ROUTE, svc({ assigned_n: 4 }), baseline, T0)).toBe(0.4);
    expect(serviceRatioFor(ROUTE, undefined, baseline, T0)).toBeNull();
    expect(serviceRatioFor(ROUTE, svc({ assigned_n: 4 }), {}, T0)).toBeNull();
  });

  test('deriveServiceRatios returns raw ratios for judgeable routes, omits the rest', () => {
    const rows = new Map<string, ServiceRow>([
      [ROUTE, svc({ assigned_n: 4 })], // 0.4
      ['1', svc({ assigned_n: 12 })], // no baseline -> omitted
    ]);
    expect(deriveServiceRatios(rows, baseline, T0)).toEqual({ [ROUTE]: 0.4 });
  });

  test('cold start + both flips need SERVICE_DEBOUNCE_TICKS, with hysteresis (matches derive_actual_recovery)', () => {
    const STEP = 300;
    let regimes: Record<string, RegimeEntry<ServiceCondition>> | undefined; // fresh doc: no established regime
    const advance = (assigned: number, i: number): string | undefined => {
      const at = T0 + i * STEP;
      const rows = new Map<string, ServiceRow>([[ROUTE, svc({ assigned_n: assigned })]]);
      const live = pruneIdleRegimes(regimes, at);
      const observed = deriveServiceStates(rows, baseline, at, live);
      // Mirror index.ts: expire stale regimes, then seed cold-start routes normal
      // so the FIRST drop debounces too.
      const { entries } = advanceRegimes(
        seedNormalServiceRegimes(live, observed, at),
        observed,
        at,
        { debounceTicks: SERVICE_DEBOUNCE_TICKS },
      );
      regimes = entries;
      return entries[ROUTE]?.state;
    };
    expect(advance(4, 0)).toBe('normal'); // COLD first low tick — seeded normal, pending, NOT degraded
    expect(advance(4, 1)).toBe('degraded'); // second low tick commits
    expect(advance(6, 2)).toBe('degraded'); // 0.6 in the band — held, no flap
    expect(advance(9, 3)).toBe('degraded'); // one high tick — not yet
    expect(advance(9, 4)).toBe('normal'); // second high tick commits
  });

  test('seedNormalServiceRegimes only fills newly observed routes, leaving existing ones intact', () => {
    const existing: Record<string, RegimeEntry<ServiceCondition>> = {
      A: { state: 'degraded', entered_at: 1, last_seen_at: 1, pending: null, pending_since: 0, pending_run: 0 },
    };
    const seeded = seedNormalServiceRegimes(existing, { A: 'normal', B: 'degraded' }, 100);
    expect(seeded.A).toBe(existing.A); // untouched
    expect(seeded.B).toEqual({ state: 'normal', entered_at: 100, last_seen_at: 100, pending: null, pending_since: 0, pending_run: 0 });
  });

  test('MAX_IDLE_SEC expiry resets a returning route to cold (intentional divergence from the offline label)', () => {
    // The offline label has no idle expiry and would hold degraded across the
    // gap; the Worker resets after MAX_IDLE_SEC as a live-feed freshness rule.
    const AT = T0 + MAX_IDLE_SEC + 300; // returns after a blind gap longer than the idle limit
    const baselineGap: ServiceBaseline = {
      [ROUTE]: { [schedule_bin(T0)]: 10, [schedule_bin(AT)]: 10 },
    };
    const stale: Record<string, RegimeEntry<ServiceCondition>> = {
      [ROUTE]: { state: 'degraded', entered_at: T0, last_seen_at: T0, pending: null, pending_since: 0, pending_run: 0 },
    };
    const live = pruneIdleRegimes(stale, AT);
    expect(live[ROUTE]).toBeUndefined(); // stale regime expired before anything reads it
    const rows = new Map<string, ServiceRow>([[ROUTE, svc({ assigned_n: 6 })]]); // ratio 0.6
    const observed = deriveServiceStates(rows, baselineGap, AT, live);
    // With the pre-gap 'degraded' gone, the band no longer holds; 0.6 reads normal.
    expect(observed[ROUTE]).toBe('normal');
    const { entries } = advanceRegimes(
      seedNormalServiceRegimes(live, observed, AT),
      observed,
      AT,
      { debounceTicks: SERVICE_DEBOUNCE_TICKS },
    );
    expect(entries[ROUTE]?.state).toBe('normal');
  });
});

describe('serviceQuantileFor / serviceQuantileRatiosFor / deriveServiceQuantileRatios (supply axis spread)', () => {
  const T0 = Date.parse('2026-06-15T16:00:00Z') / 1000; // 12:00 ET
  const ROUTE = 'Q';
  const BIN = schedule_bin(T0);

  const baseline: ServiceBaseline = { [ROUTE]: { [BIN]: 10 } };
  const quantiles: ServiceQuantiles = { [ROUTE]: { [BIN]: { p10: 8, p90: 13 } } };

  test('serviceQuantileFor returns the cell for the current schedule bin', () => {
    expect(serviceQuantileFor(ROUTE, quantiles, T0)).toEqual({ p10: 8, p90: 13 });
  });

  test('serviceQuantileFor is null when the route has no published quantiles', () => {
    expect(serviceQuantileFor(ROUTE, quantiles, T0)).not.toBeNull();
    expect(serviceQuantileFor('1', quantiles, T0)).toBeNull();
    expect(serviceQuantileFor(ROUTE, null, T0)).toBeNull();
  });

  test('serviceQuantileRatiosFor normalises the cell onto the median scale', () => {
    expect(serviceQuantileRatiosFor(ROUTE, baseline, quantiles, T0)).toEqual({ low: 0.8, high: 1.3 });
  });

  test('serviceQuantileRatiosFor is null on a missing quantile cell', () => {
    expect(serviceQuantileRatiosFor(ROUTE, baseline, {}, T0)).toBeNull();
    expect(serviceQuantileRatiosFor(ROUTE, baseline, null, T0)).toBeNull();
  });

  test('serviceQuantileRatiosFor is null on a zero or absent median', () => {
    expect(serviceQuantileRatiosFor(ROUTE, { [ROUTE]: { [BIN]: 0 } }, quantiles, T0)).toBeNull();
    expect(serviceQuantileRatiosFor(ROUTE, {}, quantiles, T0)).toBeNull();
    expect(serviceQuantileRatiosFor(ROUTE, null, quantiles, T0)).toBeNull();
  });

  test('deriveServiceQuantileRatios returns ratios for judgeable routes with a quantile cell, omits the rest', () => {
    const rows = new Map<string, ServiceRow>([
      [ROUTE, svc({ assigned_n: 9 })], // has both baseline and quantiles
      ['1', svc({ assigned_n: 12 })], // no baseline, no quantiles -> omitted
    ]);
    expect(deriveServiceQuantileRatios(rows, baseline, quantiles, T0)).toEqual({
      [ROUTE]: { low: 0.8, high: 1.3 },
    });
  });

  test('deriveServiceQuantileRatios omits a route with a baseline but no quantile cell', () => {
    const rows = new Map<string, ServiceRow>([[ROUTE, svc({ assigned_n: 9 })]]);
    expect(deriveServiceQuantileRatios(rows, baseline, {}, T0)).toEqual({});
  });
});
