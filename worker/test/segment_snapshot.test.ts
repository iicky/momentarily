/**
 * Segment-flow publish surface: per-segment status + expected recovery off
 * the segment dwell curve, and the same recovery rolled up onto
 * station_flow's already-selected worst_segment. Mirrors movement_recovery
 * .test.ts's coverage of the route-level curve + clock, one level down.
 */

import Ajv2020 from 'ajv/dist/2020';
import { describe, expect, test } from 'vitest';

import { conditionalRecovery } from '../src/dwell';
import type { RegimeEntry } from '../src/regime';
import { TICK_SECONDS, buildSnapshot } from '../src/snapshot';
import type {
  SegmentCondition,
  SegmentDwellDoc,
  SegmentFlowDoc,
  SegmentParamsDoc,
  StationFlowDoc,
} from '../src/state';
import schema from '../../schema/snapshot.schema.json';

const ajv = new Ajv2020({ allErrors: true, strict: false });
const validate = ajv.compile(schema);

function checkSchema(snapshot: unknown): void {
  const ok = validate(snapshot);
  expect(
    ok,
    `snapshot failed schema/snapshot.schema.json:\n${JSON.stringify(validate.errors, null, 2)}`,
  ).toBe(true);
}

const NOW = 1_700_000_000;
const MIN = 60;

const KEY = 'F|south|A09S';

// Heavy-tailed disrupted dwell, same shape as movement_recovery.test.ts's.
const DISRUPTED_CURVE = Array.from({ length: 21 }, (_, i) => Math.round(120 * 1.35 ** i));

const params: SegmentParamsDoc = {
  schema_version: '1',
  trained_at: NOW,
  min_share: 0.5,
  topology_source: 'observed',
  cells: { [KEY]: { p0: 0.9, n: 1000 } },
  adjacency: { [KEY]: { to: 'A10S', source: 'observed', share: 0.9, n: 1000 } },
};

function regime(state: SegmentCondition, enteredAt: number): RegimeEntry<SegmentCondition> {
  return {
    state,
    entered_at: enteredAt,
    last_seen_at: NOW,
    pending: null,
    pending_since: 0,
    pending_run: 0,
  };
}

function flowDoc(
  observedAt: number,
  regimes: Record<string, RegimeEntry<SegmentCondition>>,
): SegmentFlowDoc {
  return { observed_at: observedAt, cells: {}, vehicles: {}, regimes };
}

function dwellDoc(): SegmentDwellDoc {
  return {
    schema_version: '1',
    trained_at: NOW,
    cells: {
      [KEY]: {
        disrupted: {
          n: 30,
          n_censored: 0,
          q25_sec: 538,
          median_sec: 2413,
          q75_sec: 10819,
          recover_by_30: 0.4,
          recover_by_60: 0.6,
          recover_by_120: 0.8,
          curve_sec: DISRUPTED_CURVE,
        },
      },
    },
  };
}

function stationFlowDoc(observedAt: number): StationFlowDoc {
  return {
    observed_at: observedAt,
    stations: {
      A09: {
        status: 'degraded',
        worst_deficit: 0.9,
        worst_segment: ['A09S', 'A10S'],
        routes: ['F'],
        n_segments: 1,
      },
    },
  };
}

function build(opts: {
  segmentFlow?: SegmentFlowDoc | null;
  segmentParams?: SegmentParamsDoc | null;
  segmentDwell?: SegmentDwellDoc | null;
  stationFlow?: StationFlowDoc | null;
}) {
  return buildSnapshot({
    generatedAt: NOW,
    alertsFreshness: NOW,
    routeSnapshots: new Map(),
    rolls: {},
    trainedParams: null,
    tickSeconds: TICK_SECONDS,
    ...opts,
    vehicleFreshFeeds: [],
    vehicleExpectedFeeds: [],
  });
}

describe('segment_flow: per-segment status + recovery', () => {
  test('a segment with a dwell curve publishes a withheld recovery block, not a number', () => {
    const elapsed30 = 30 * MIN;
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, { [KEY]: regime('disrupted', NOW - elapsed30) }),
      segmentParams: params,
      segmentDwell: dwellDoc(),
    });
    const seg = snap.segment_flow!.segments[KEY]!;
    expect(seg.status).toBe('disrupted');
    expect(seg.route).toBe('F');
    expect(seg.direction).toBe('south');
    expect(seg.from_stop).toBe('A09S');
    expect(seg.to).toBe('A10S');
    expect(seg.entered_at).toBe(NOW - elapsed30);

    // Every segment recovery number comes off a fitted dwell curve, and that
    // estimate has not cleared the validation gate (snapshot.ts
    // PUBLISH_FITTED_RECOVERY), so the block is present-but-empty: the cell
    // HAS an estimate and it is deliberately not published. That is a
    // different statement from `recovery: null` (no curve, no clock) below,
    // which is why the block survives at all.
    expect(seg.recovery).not.toBeNull();
    expect(seg.recovery!.recovery_withheld).toBe('pending_validation');
    expect(seg.recovery!.recovery_minutes).toBeNull();
    expect(seg.recovery!.recovery_minutes_low).toBeNull();
    expect(seg.recovery!.recovery_minutes_high).toBeNull();
    expect(seg.recovery!.p_normal_in_30min).toBeNull();
    expect(seg.recovery!.p_normal_in_60min).toBeNull();
    expect(seg.recovery!.p_normal_in_120min).toBeNull();
    checkSchema(snap);

    // The conditioning the block would carry is still exercised, at the
    // dwell.ts helper the Worker calls: elapsed=30min and elapsed=5min give
    // different medians off the same curve.
    const at30 = conditionalRecovery(DISRUPTED_CURVE, elapsed30)!;
    const at5 = conditionalRecovery(DISRUPTED_CURVE, 5 * MIN)!;
    expect(at30).not.toBeNull();
    expect(Math.round(at30.median_sec / 60)).not.toBe(Math.round(at5.median_sec / 60));
  });

  test('a segment without a trained curve publishes status and NO recovery', () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, { [KEY]: regime('disrupted', NOW - 30 * MIN) }),
      segmentParams: params,
      segmentDwell: null, // segment_dwell.json doesn't exist yet in production
    });
    const seg = snap.segment_flow!.segments[KEY]!;
    expect(seg.status).toBe('disrupted');
    expect(seg.entered_at).toBe(NOW - 30 * MIN);
    expect(seg.to).toBe('A10S');
    expect(seg.recovery).toBeNull();
  });

  test('a regime with entered_at=0 (clock never started) publishes NO recovery even with a curve', () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, { [KEY]: regime('disrupted', 0) }),
      segmentParams: params,
      segmentDwell: dwellDoc(),
    });
    expect(snap.segment_flow!.segments[KEY]!.recovery).toBeNull();
  });

  test('missing segment_params (topology) degrades to `to: null`, not a dropped cell', () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, { [KEY]: regime('disrupted', NOW - 30 * MIN) }),
      segmentParams: null,
      segmentDwell: null,
    });
    const seg = snap.segment_flow!.segments[KEY]!;
    expect(seg.status).toBe('disrupted');
    expect(seg.to).toBeNull();
  });

  test('a stale segment doc is dropped exactly like a stale station_flow', () => {
    const STALE = 1801; // 1s past MAX_MOVEMENT_STATE_AGE_SEC (1800)
    const snap = build({
      segmentFlow: flowDoc(NOW - STALE, { [KEY]: regime('disrupted', NOW - 30 * MIN) }),
      segmentParams: params,
      segmentDwell: dwellDoc(),
      stationFlow: stationFlowDoc(NOW - STALE),
    });
    expect(snap.segment_flow).toBeNull();
    expect(snap.station_flow).toBeNull();
  });

  test('a segment doc right at the freshness boundary still publishes', () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 1800, { [KEY]: regime('disrupted', NOW - 30 * MIN) }),
      segmentParams: params,
      segmentDwell: dwellDoc(),
    });
    expect(snap.segment_flow).not.toBeNull();
  });
});

describe('station_flow: worst_recovery roll-up', () => {
  test('every pre-existing StationServiceFlow field is unchanged, plus worst_recovery', () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, { [KEY]: regime('disrupted', NOW - 30 * MIN) }),
      segmentParams: params,
      segmentDwell: dwellDoc(),
      stationFlow: stationFlowDoc(NOW - 300),
    });
    const station = snap.station_flow!.stations['A09']!;
    expect(station.status).toBe('degraded');
    expect(station.worst_deficit).toBe(0.9);
    expect(station.worst_segment).toEqual(['A09S', 'A10S']);
    expect(station.routes).toEqual(['F']);
    expect(station.n_segments).toBe(1);
    // The new field: the worst_segment's own recovery, same numbers as the
    // segment surface computed for the same cell.
    expect(station.worst_recovery).toEqual(snap.segment_flow!.segments[KEY]!.recovery);
    expect(station.worst_recovery).not.toBeNull();
  });

  test('worst_recovery is null when the worst segment has no recovery (no curve)', () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, { [KEY]: regime('disrupted', NOW - 30 * MIN) }),
      segmentParams: params,
      segmentDwell: null,
      stationFlow: stationFlowDoc(NOW - 300),
    });
    expect(snap.station_flow!.stations['A09']!.worst_recovery).toBeNull();
  });

  test('worst_recovery is null when segment_flow itself is absent (station_flow still publishes)', () => {
    const snap = build({
      stationFlow: stationFlowDoc(NOW - 300),
      // no segmentFlow at all this tick
    });
    expect(snap.station_flow).not.toBeNull();
    expect(snap.station_flow!.stations['A09']!.status).toBe('degraded');
    expect(snap.station_flow!.stations['A09']!.worst_recovery).toBeNull();
  });

  test('worst_recovery is null when no live segment matches the worst_segment stop pair', () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, {}), // no cells tracked this tick
      segmentParams: params,
      segmentDwell: dwellDoc(),
      stationFlow: stationFlowDoc(NOW - 300),
    });
    expect(snap.station_flow!.stations['A09']!.worst_recovery).toBeNull();
  });
});

describe('segment_flow / station_flow validate against the Pydantic schema', () => {
  test('the near-term production path — segmentDwell absent — still validates', () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, { [KEY]: regime('disrupted', NOW - 30 * MIN) }),
      segmentParams: params,
      segmentDwell: null,
      stationFlow: stationFlowDoc(NOW - 300),
    });
    expect(snap.segment_flow!.segments[KEY]!.recovery).toBeNull();
    checkSchema(snap);
  });

  test('the full path — every state doc present with a trained curve — validates', () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, { [KEY]: regime('disrupted', NOW - 30 * MIN) }),
      segmentParams: params,
      segmentDwell: dwellDoc(),
      stationFlow: stationFlowDoc(NOW - 300),
    });
    expect(snap.segment_flow!.segments[KEY]!.recovery).not.toBeNull();
    checkSchema(snap);
  });

  test('no segment state at all (pre-deploy) still validates', () => {
    checkSchema(build({}));
  });
});
