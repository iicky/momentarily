/**
 * Platform-wait carry (state/station_wait.json) and the live per-platform
 * crowding estimate it feeds (platform_crowding). Mirrors segment_flow
 * .test.ts's split: pure state-update behavior, then pure derivation,
 * then (via buildSnapshot) the publish-time freshness gate.
 */

import { describe, expect, test } from 'vitest';

import {
  CROWDING_MAX_GAP_MINUTES,
  WAIT_PRUNE_SECONDS,
  derivePlatformCrowding,
  updateStationWait,
} from '../src/crowding';
import type { PlatformCrowdingResult } from '../src/crowding';
import { loadServiceWeightBaseline } from '../src/params';
import type { RidershipBaselineDoc, ServiceWeightBaselineDoc } from '../src/params';
import { TICK_SECONDS, buildSnapshot } from '../src/snapshot';
import type { StationWaitDoc } from '../src/state';
import type { StationOut } from '../src/stations_static';
import type { TraceRow } from '../src/vehicles';

const NOW = 1_700_000_000;
const MIN = 60;

function row(tripId: string, stopId: string, stopped: boolean): TraceRow {
  return {
    trip_id: tripId,
    route_id: 'A',
    direction: 'north',
    stop_id: stopId,
    stop_seq: stopped ? 5 : null,
    stopped,
    vehicle_ts: null,
  };
}

describe('updateStationWait', () => {
  test('a dwell shorter than the poll interval is still detected as a departure via the carry rule', () => {
    // Tick 1: T1 is heading to/at 127N but the poll never catches it
    // STOPPED_AT there — a dwell shorter than the 1-minute poll interval
    // came and went between two polls. Only the trip carry survives.
    const tick1 = updateStationWait([row('T1', '127N', false)], null, NOW);
    expect(tick1.platforms).toEqual({});
    expect(tick1.trips).toEqual({ T1: '127N' });

    // Tick 2 (60s later): T1's stop_id has moved on to 128N. Even though
    // 127N was never seen stopped=true, the transition alone proves a train
    // was there and has now left.
    const t1 = NOW + 60;
    const tick2 = updateStationWait([row('T1', '128N', false)], tick1, t1);
    expect(tick2.platforms['127N']).toBe(t1);
  });

  test('max semantics make a replayed minute idempotent', () => {
    // T1 was carried at 126N last tick and is now stopped at 127N: this
    // single row both clears 126N (transition) and occupies 127N (stopped).
    const prev: StationWaitDoc = {
      observed_at: NOW - MIN,
      platforms: {},
      trips: { T1: '126N' },
    };
    const rows = [row('T1', '127N', true)];
    const first = updateStationWait(rows, prev, NOW);
    expect(first.platforms).toEqual({ '126N': NOW, '127N': NOW });

    // A retried cron invocation for the SAME minute reads back exactly what
    // the first attempt wrote and reapplies the SAME rows/now. Math.max
    // against already-current values, plus trips already reading '127N' (so
    // no phantom second departure off the stale '126N' carry), reproduces
    // an identical doc.
    const replayed = updateStationWait(rows, first, NOW);
    expect(replayed).toEqual(first);
  });

  test('prunes a platform untouched for over WAIT_PRUNE_SECONDS', () => {
    const staleAt = NOW - WAIT_PRUNE_SECONDS - 1;
    const prev: StationWaitDoc = {
      observed_at: staleAt,
      platforms: { '999N': staleAt },
      trips: {},
    };
    const doc = updateStationWait([], prev, NOW);
    expect(doc.platforms).not.toHaveProperty('999N');
  });
});

function uniformRate(rate: number): { wd: number[]; we: number[] } {
  return { wd: Array(24).fill(rate) as number[], we: Array(24).fill(rate) as number[] };
}

function baselineWith(rates: Record<string, number>): RidershipBaselineDoc {
  const complexes: RidershipBaselineDoc['complexes'] = {};
  for (const [id, rate] of Object.entries(rates)) {
    complexes[id] = {
      name: `Complex ${id}`,
      borough: 'Manhattan',
      entries_per_min: uniformRate(rate),
      entries_total: rate * 100_000,
      transfers_total: 0,
      rank: 1,
      n_cells: 48,
    };
  }
  return {
    schema_version: '1',
    generated_at: NOW - 3600,
    source: {
      dataset: '5wq4-mkjj',
      url: 'https://data.ny.gov/resource/5wq4-mkjj.json',
      transit_mode: 'subway',
      window_start: '2026-05-01T00:00:00',
      window_end: '2026-07-29T00:00:00',
      latest_hour: '2026-07-28T23:00:00',
      weekday_days: 63,
      weekend_days: 26,
    },
    complexes,
    n_complexes: Object.keys(complexes).length,
  };
}

function stationFixture(parentId: string, complexId: string | null): StationOut {
  return {
    gtfs_stop_id: parentId,
    station_complex_id: complexId,
    name: `Station ${parentId}`,
    borough: 'Manhattan',
    routes_served: ['A'],
    ada: 0,
    ada_northbound: false,
    ada_southbound: false,
  };
}

describe('derivePlatformCrowding', () => {
  test('computes waiting_riders = entries_per_min * minutes_since_last_train, rounded', () => {
    const baseline = baselineWith({ C1: 60 });
    const stations = { '127': stationFixture('127', 'C1') };
    const wait: StationWaitDoc = {
      observed_at: NOW,
      platforms: { '127N': NOW - 5 * MIN },
      trips: {},
    };
    const result = derivePlatformCrowding(wait, baseline, null, stations, NOW);
    expect(result.platforms['127N']).toEqual({
      last_train_at: NOW - 5 * MIN,
      entries_per_min: 60,
      waiting_riders: 300,
    });
    expect(result.n_platforms).toBe(1);
    expect(result.abstained).toEqual({});
  });

  test('the complex split divides only across served platforms, so an out-of-service platform does not absorb demand', () => {
    const baseline = baselineWith({ C1: 60 });
    const stations = {
      '127': stationFixture('127', 'C1'),
      '128': stationFixture('128', 'C1'),
    };
    const wait: StationWaitDoc = {
      observed_at: NOW,
      platforms: {
        '127N': NOW - 2 * MIN, // recent: in service
        '128N': NOW - 100 * MIN, // long silent: out of service
      },
      trips: {},
    };
    const result = derivePlatformCrowding(wait, baseline, null, stations, NOW);
    // 127N gets the FULL complex rate, not split against the dark platform.
    expect(result.platforms['127N']?.entries_per_min).toBe(60);
    expect(result.platforms['128N']).toBeUndefined();
  });

  test("abstains 'unknown_stop' when the platform's parent station has no complex id", () => {
    const baseline = baselineWith({ C1: 60 });
    const wait: StationWaitDoc = {
      observed_at: NOW,
      platforms: { '900N': NOW - MIN },
      trips: {},
    };
    const result = derivePlatformCrowding(wait, baseline, null, {}, NOW);
    expect(result.abstained).toEqual({ unknown_stop: 1 });
    expect(result.platforms).toEqual({});
  });

  test("abstains 'no_baseline' when the complex isn't in the ridership baseline", () => {
    const baseline = baselineWith({});
    const stations = { '127': stationFixture('127', 'C1') };
    const wait: StationWaitDoc = {
      observed_at: NOW,
      platforms: { '127N': NOW - MIN },
      trips: {},
    };
    const result = derivePlatformCrowding(wait, baseline, null, stations, NOW);
    expect(result.abstained).toEqual({ no_baseline: 1 });
  });

  test("abstains 'no_recent_train' when the platform's own gap exceeds the served window", () => {
    const baseline = baselineWith({ C1: 60 });
    const stations = { '127': stationFixture('127', 'C1') };
    const wait: StationWaitDoc = {
      observed_at: NOW,
      platforms: { '127N': NOW - 90 * MIN }, // > CROWDING_SERVED_WINDOW_MINUTES (60)
      trips: {},
    };
    const result = derivePlatformCrowding(wait, baseline, null, stations, NOW);
    expect(result.abstained).toEqual({ no_recent_train: 1 });
  });

  test("abstains 'gap_exceeds_cap' when within the served window but past the publish cap", () => {
    const baseline = baselineWith({ C1: 60 });
    const stations = { '127': stationFixture('127', 'C1') };
    // Just past the 30-min publish cap but well inside the 60-min served
    // window, so it still counts toward its own denominator.
    const gapMinutes = CROWDING_MAX_GAP_MINUTES + 15;
    const wait: StationWaitDoc = {
      observed_at: NOW,
      platforms: { '127N': NOW - gapMinutes * MIN },
      trips: {},
    };
    const result = derivePlatformCrowding(wait, baseline, null, stations, NOW);
    expect(result.abstained).toEqual({ gap_exceeds_cap: 1 });
  });

  test('the publish cap is inclusive: exactly CROWDING_MAX_GAP_MINUTES abstains', () => {
    const baseline = baselineWith({ C1: 60 });
    const stations = { '127': stationFixture('127', 'C1') };
    const wait: StationWaitDoc = {
      observed_at: NOW,
      platforms: { '127N': NOW - CROWDING_MAX_GAP_MINUTES * MIN },
      trips: {},
    };
    const result = derivePlatformCrowding(wait, baseline, null, stations, NOW);
    expect(result.abstained).toEqual({ gap_exceeds_cap: 1 });
  });
});

function serviceWeightWith(counts: Record<string, number>): ServiceWeightBaselineDoc {
  const stops: ServiceWeightBaselineDoc['stops'] = {};
  for (const [id, c] of Object.entries(counts)) {
    stops[id] = { wd: Array(24).fill(c) as number[], we: Array(24).fill(c) as number[] };
  }
  return {
    schema_version: '1',
    generated_at: NOW - 7200,
    source: {
      dataset: 'gtfs_subway',
      url: 'https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip',
      feed_version: '20260807',
      weekday_services: ['Weekday'],
      weekend_services: ['Saturday', 'Sunday'],
    },
    stops,
    n_stops: Object.keys(stops).length,
  };
}

describe('derivePlatformCrowding scheduled-service split', () => {
  // Grand Central shape: three parent stops in one complex, the 42 St Shuttle
  // (901) running far less service than the 4/5/6 (631). The whole complex is
  // in service and every platform is covered, so the split is scheduled.
  const gcStations = {
    '631': stationFixture('631', '610'),
    '723': stationFixture('723', '610'),
    '901': stationFixture('901', '610'),
  };
  const gcWait: StationWaitDoc = {
    observed_at: NOW,
    platforms: {
      '631N': NOW - 2 * MIN,
      '631S': NOW - 2 * MIN,
      '723N': NOW - 2 * MIN,
      '723S': NOW - 2 * MIN,
      '901N': NOW - 2 * MIN,
      '901S': NOW - 2 * MIN,
    },
    trips: {},
  };
  // Weights sum to 170 so the complex rate of 170/min makes each platform's
  // scheduled entries_per_min equal to its own weight — easy to read.
  const gcWeights = serviceWeightWith({
    '631N': 40,
    '631S': 40,
    '723N': 27,
    '723S': 27,
    '901N': 18,
    '901S': 18,
  });
  const gcBaseline = baselineWith({ '610': 170 });

  test('splits a complex in proportion to scheduled service, so the low-service shuttle platform gets less than the busy one', () => {
    const result = derivePlatformCrowding(gcWait, gcBaseline, gcWeights, gcStations, NOW);
    expect(result.platforms['631N']?.entries_per_min).toBe(40);
    expect(result.platforms['901S']?.entries_per_min).toBe(18);
    expect(result.platforms['901S']!.entries_per_min).toBeLessThan(
      result.platforms['631N']!.entries_per_min,
    );
  });

  test('the scheduled split cuts the shuttle platform below the even split it would otherwise get', () => {
    const scheduled = derivePlatformCrowding(gcWait, gcBaseline, gcWeights, gcStations, NOW);
    const uniform = derivePlatformCrowding(gcWait, gcBaseline, null, gcStations, NOW);
    // Even split: 170 / 6 served platforms ≈ 28.33 (rounded); scheduled: 18.
    expect(uniform.platforms['901S']?.entries_per_min).toBe(28.33);
    expect(scheduled.platforms['901S']!.entries_per_min).toBeLessThan(
      uniform.platforms['901S']!.entries_per_min,
    );
    // Demand is conserved: reweighting moves it between platforms, never
    // creates or destroys it. Both bases sum to the complex rate, up to the
    // 2-decimal rounding each published rate carries (±0.005 per platform).
    const sum = (r: PlatformCrowdingResult) =>
      Object.values(r.platforms).reduce((a, p) => a + p.entries_per_min, 0);
    expect(sum(scheduled)).toBeCloseTo(170, 1);
    expect(sum(uniform)).toBeCloseTo(170, 1);
  });

  test('a complex falls back to the even split when a served platform is absent from the baseline — no imputed weight', () => {
    const stations = { '127': stationFixture('127', 'C1') };
    const wait: StationWaitDoc = {
      observed_at: NOW,
      platforms: { '127N': NOW - 2 * MIN, '127S': NOW - 2 * MIN },
      trips: {},
    };
    // Only 127N is covered; 127S has no entry, so the whole complex is uniform.
    const weights = serviceWeightWith({ '127N': 90 });
    const result = derivePlatformCrowding(wait, baselineWith({ C1: 60 }), weights, stations, NOW);
    expect(result.platforms['127N']?.entries_per_min).toBe(30);
    expect(result.platforms['127S']?.entries_per_min).toBe(30);
  });

  test('a served platform the schedule gives zero trains this hour forces its complex to the even split', () => {
    const stations = { '127': stationFixture('127', 'C1') };
    const wait: StationWaitDoc = {
      observed_at: NOW,
      platforms: { '127N': NOW - 2 * MIN, '127S': NOW - 2 * MIN },
      trips: {},
    };
    // 127S is in the baseline but scheduled for 0 trains — a hole, not a weight.
    const weights = serviceWeightWith({ '127N': 90, '127S': 0 });
    const result = derivePlatformCrowding(wait, baselineWith({ C1: 60 }), weights, stations, NOW);
    expect(result.platforms['127N']?.entries_per_min).toBe(30);
    expect(result.platforms['127S']?.entries_per_min).toBe(30);
  });

  test('an out-of-service platform contributes no scheduled weight, so a covered running platform still takes the full rate', () => {
    const stations = { '127': stationFixture('127', 'C1') };
    const wait: StationWaitDoc = {
      observed_at: NOW,
      platforms: {
        '127N': NOW - 2 * MIN, // running
        '127S': NOW - 100 * MIN, // dark, outside the served window
      },
      trips: {},
    };
    const weights = serviceWeightWith({ '127N': 40, '127S': 40 });
    const result = derivePlatformCrowding(wait, baselineWith({ C1: 60 }), weights, stations, NOW);
    expect(result.platforms['127N']?.entries_per_min).toBe(60);
    expect(result.platforms['127S']).toBeUndefined();
  });
});

function buildWithCrowding(opts: {
  stationWait?: StationWaitDoc | null;
  ridershipBaseline?: RidershipBaselineDoc | null;
  stations?: Record<string, StationOut>;
}) {
  return buildSnapshot({
    generatedAt: NOW,
    alertsFreshness: NOW,
    routeSnapshots: new Map(),
    rolls: {},
    trainedParams: null,
    tickSeconds: TICK_SECONDS,
    ...opts,
  });
}

describe('platform_crowding freshness gate (buildSnapshot)', () => {
  const baseline = baselineWith({ C1: 60 });
  const stations = { '127': stationFixture('127', 'C1') };
  const freshWait: StationWaitDoc = {
    observed_at: NOW,
    platforms: { '127N': NOW - 5 * MIN },
    trips: {},
  };

  test('a fresh station-wait doc with a baseline publishes the surface', () => {
    const snap = buildWithCrowding({ stationWait: freshWait, ridershipBaseline: baseline, stations });
    expect(snap.platform_crowding).not.toBeNull();
    expect(snap.platform_crowding?.platforms['127N']?.waiting_riders).toBe(300);
    expect(snap.platform_crowding?.method.max_gap_minutes).toBe(30);
    expect(snap.platform_crowding?.method.served_window_minutes).toBe(60);
    expect(snap.platform_crowding?.method.baseline_generated_at).toBe(baseline.generated_at);
    expect(snap.platform_crowding?.method.baseline_window_start).toBe(baseline.source.window_start);
    expect(snap.platform_crowding?.method.baseline_window_end).toBe(baseline.source.window_end);
  });

  test('a stale station-wait doc nulls the surface entirely', () => {
    const staleWait: StationWaitDoc = { ...freshWait, observed_at: NOW - 3600 };
    const snap = buildWithCrowding({ stationWait: staleWait, ridershipBaseline: baseline, stations });
    expect(snap.platform_crowding).toBeNull();
  });

  test('no ridership baseline yet nulls the surface', () => {
    const snap = buildWithCrowding({ stationWait: freshWait, ridershipBaseline: null, stations });
    expect(snap.platform_crowding).toBeNull();
  });

  test('no station-wait doc yet (before the first vehicle tick) nulls the surface', () => {
    const snap = buildWithCrowding({ stationWait: null, ridershipBaseline: baseline, stations });
    expect(snap.platform_crowding).toBeNull();
  });
});

describe('platform_crowding surface size', () => {
  test('stays proportionate to the snapshot at realistic (~900 platform) scale', () => {
    const complexes: Record<string, number> = {};
    const stations: Record<string, StationOut> = {};
    const platforms: Record<string, number> = {};
    const N_COMPLEXES = 450; // 2 platforms each = 900 platforms
    for (let i = 0; i < N_COMPLEXES; i++) {
      const complexId = `C${i}`;
      const parent = `${7000 + i}`;
      complexes[complexId] = 50;
      stations[parent] = stationFixture(parent, complexId);
      platforms[`${parent}N`] = NOW - 3 * MIN;
      platforms[`${parent}S`] = NOW - 4 * MIN;
    }
    const wait: StationWaitDoc = { observed_at: NOW, platforms, trips: {} };
    const baseline = baselineWith(complexes);
    const snap = buildWithCrowding({ stationWait: wait, ridershipBaseline: baseline, stations });
    expect(snap.platform_crowding?.n_platforms).toBe(N_COMPLEXES * 2);

    const bytes = Buffer.byteLength(JSON.stringify(snap.platform_crowding), 'utf8');
    console.log(`platform_crowding at ${N_COMPLEXES * 2} platforms: ${bytes} bytes`);
    // ~71KB measured at this fixture's field widths — well under the ~190KB
    // whole-snapshot budget for one surface among many. A generous ceiling
    // (not the raw measurement) so unrelated fixture tweaks don't flake
    // this, while still catching an accidental field/verbosity regression.
    expect(bytes).toBeLessThan(100_000);
  });
});

describe('loadServiceWeightBaseline', () => {
  test('returns null instead of rejecting when the R2 read throws, so an optional baseline never blocks the tick', async () => {
    const bucket = {
      get: async () => {
        throw new Error('R2 unavailable');
      },
    } as unknown as R2Bucket;
    await expect(loadServiceWeightBaseline(bucket)).resolves.toBeNull();
  });

  test('returns null on a malformed document rather than throwing', async () => {
    const bucket = {
      get: async () => ({ json: async () => ({ schema_version: '1', stops: 'not-an-object' }) }),
    } as unknown as R2Bucket;
    await expect(loadServiceWeightBaseline(bucket)).resolves.toBeNull();
  });
});
