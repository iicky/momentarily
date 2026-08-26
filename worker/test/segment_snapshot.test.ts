/**
 * Segment-flow publish surface: per-segment status + expected recovery,
 * keyed the same way as segment_flow.json/segment_params.json, for EVERY
 * judged cell — normal and disrupted alike, in the one `segments` dict — and
 * the same recovery rolled up onto station_flow's already-selected
 * worst_segment. Mirrors movement_recovery.test.ts's coverage of the
 * route-level curve + clock, one level down.
 */

import Ajv2020 from "ajv/dist/2020";
import { describe, expect, test } from "vitest";

import { conditionalRecovery } from '../src/dwell';
import type { RegimeEntry } from '../src/regime';
import { TICK_SECONDS, buildSnapshot } from '../src/snapshot';
import type {
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

const KEY = "F|south|A09S";
// A second, unrelated segment cell used to exercise a NORMAL verdict
// alongside `KEY`'s disrupted one, without touching any existing test.
const NORMAL_KEY = "F|south|B01S";

// Heavy-tailed disrupted dwell, same shape as movement_recovery.test.ts's.
const DISRUPTED_CURVE = Array.from({ length: 21 }, (_, i) =>
  Math.round(120 * 1.35 ** i),
);

const params: SegmentParamsDoc = {
  schema_version: "1",
  trained_at: NOW,
  min_share: 0.5,
  topology_source: "observed",
  cells: { [KEY]: { p0: 0.9, n: 1000 }, [NORMAL_KEY]: { p0: 0.9, n: 1000 } },
  adjacency: {
    [KEY]: { to: "A10S", source: "observed", share: 0.9, n: 1000 },
    [NORMAL_KEY]: { to: "B02S", source: "observed", share: 0.9, n: 1000 },
  },
};

function regime(
  state: "normal" | "disrupted",
  enteredAt: number,
): RegimeEntry<"normal" | "disrupted"> {
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
  regimes: Record<string, RegimeEntry<"normal" | "disrupted">>,
): SegmentFlowDoc {
  return { observed_at: observedAt, cells: {}, regimes };
}

function dwellDoc(): SegmentDwellDoc {
  return {
    schema_version: "1",
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
        status: "degraded",
        worst_deficit: 0.9,
        worst_segment: ["A09S", "A10S"],
        routes: ["F"],
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
  });
}

describe("segment_flow: per-segment status + recovery", () => {
  test("a segment with a dwell curve publishes recovery conditioned on its own elapsed clock", () => {
    const elapsed30 = 30 * MIN;
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, {
        [KEY]: regime("disrupted", NOW - elapsed30),
      }),
      segmentParams: params,
      segmentDwell: dwellDoc(),
    });
    const seg = snap.segment_flow!.segments[KEY]!;
    expect(seg.status).toBe("disrupted");
    expect(seg.route).toBe("F");
    expect(seg.direction).toBe("south");
    expect(seg.from_stop).toBe("A09S");
    expect(seg.to).toBe("A10S");
    expect(seg.entered_at).toBe(NOW - elapsed30);
    expect(seg.recovery).not.toBeNull();

    // Conditioned on elapsed=30min, not the unconditional (elapsed=0) curve —
    // computed straight off the same dwell.ts helper the Worker uses.
    const expected = conditionalRecovery(DISRUPTED_CURVE, elapsed30)!;
    expect(seg.recovery!.recovery_minutes).toBe(
      Math.round(expected.median_sec / 60),
    );

    // A different elapsed clock on the SAME curve must yield a different
    // reading — pins the conditioning, not just presence of a number.
    const elapsed5 = 5 * MIN;
    const snap5 = build({
      segmentFlow: flowDoc(NOW - 300, {
        [KEY]: regime("disrupted", NOW - elapsed5),
      }),
      segmentParams: params,
      segmentDwell: dwellDoc(),
    });
    const seg5 = snap5.segment_flow!.segments[KEY]!;
    expect(seg5.recovery!.recovery_minutes).not.toBe(
      seg.recovery!.recovery_minutes,
    );
    const expected5 = conditionalRecovery(DISRUPTED_CURVE, elapsed5)!;
    expect(seg5.recovery!.recovery_minutes).toBe(
      Math.round(expected5.median_sec / 60),
    );
  });

  test("a segment without a trained curve publishes status and NO recovery", () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, {
        [KEY]: regime("disrupted", NOW - 30 * MIN),
      }),
      segmentParams: params,
      segmentDwell: null, // segment_dwell.json doesn't exist yet in production
    });
    const seg = snap.segment_flow!.segments[KEY]!;
    expect(seg.status).toBe("disrupted");
    expect(seg.entered_at).toBe(NOW - 30 * MIN);
    expect(seg.to).toBe("A10S");
    expect(seg.recovery).toBeNull();
  });

  test("a regime with entered_at=0 (clock never started) publishes NO recovery even with a curve", () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, { [KEY]: regime("disrupted", 0) }),
      segmentParams: params,
      segmentDwell: dwellDoc(),
    });
    expect(snap.segment_flow!.segments[KEY]!.recovery).toBeNull();
  });

  test("missing segment_params (topology) degrades to `to: null`, not a dropped cell", () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, {
        [KEY]: regime("disrupted", NOW - 30 * MIN),
      }),
      segmentParams: null,
      segmentDwell: null,
    });
    const seg = snap.segment_flow!.segments[KEY]!;
    expect(seg.status).toBe("disrupted");
    expect(seg.to).toBeNull();
  });

  test("a stale segment doc is dropped exactly like a stale station_flow", () => {
    const STALE = 1801; // 1s past MAX_MOVEMENT_STATE_AGE_SEC (1800)
    const snap = build({
      segmentFlow: flowDoc(NOW - STALE, {
        [KEY]: regime("disrupted", NOW - 30 * MIN),
      }),
      segmentParams: params,
      segmentDwell: dwellDoc(),
      stationFlow: stationFlowDoc(NOW - STALE),
    });
    expect(snap.segment_flow).toBeNull();
    expect(snap.station_flow).toBeNull();
  });

  test("a segment doc right at the freshness boundary still publishes", () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 1800, {
        [KEY]: regime("disrupted", NOW - 30 * MIN),
      }),
      segmentParams: params,
      segmentDwell: dwellDoc(),
    });
    expect(snap.segment_flow).not.toBeNull();
  });
});

describe("segment_flow: every judged cell publishes a full record in segments", () => {
  test("a NORMAL cell publishes status normal, its successor, and a null recovery", () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, {
        [NORMAL_KEY]: regime("normal", NOW - 30 * MIN),
      }),
      segmentParams: params,
      segmentDwell: dwellDoc(),
    });
    // NORMAL_KEY's adjacency.to is 'B02S' — the successor rides along so a
    // branch-sharing consumer can tell which leg the verdict is about.
    const seg = snap.segment_flow!.segments[NORMAL_KEY]!;
    expect(seg.status).toBe("normal");
    expect(seg.to).toBe("B02S");
    expect(seg.entered_at).toBe(NOW - 30 * MIN);
    // Nothing to forecast on healthy track — never fabricated, never a zero.
    expect(seg.recovery).toBeNull();
  });

  test("a DISRUPTED cell publishes status disrupted alongside a NORMAL cell in the same dict", () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, {
        [KEY]: regime("disrupted", NOW - 30 * MIN),
      }),
      segmentParams: params,
      segmentDwell: dwellDoc(),
    });
    expect(Object.keys(snap.segment_flow!.segments)).toEqual([KEY]);
    expect(snap.segment_flow!.segments[KEY]!.status).toBe("disrupted");
  });

  test("mixed regimes both land in segments: their union is every judged cell, none dropped", () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, {
        [KEY]: regime("disrupted", NOW - 30 * MIN),
        [NORMAL_KEY]: regime("normal", NOW - 10 * MIN),
      }),
      segmentParams: params,
      segmentDwell: dwellDoc(),
    });
    expect(Object.keys(snap.segment_flow!.segments).sort()).toEqual(
      [KEY, NORMAL_KEY].sort(),
    );
    expect(snap.segment_flow!.segments[KEY]!.status).toBe("disrupted");
    expect(snap.segment_flow!.segments[NORMAL_KEY]!.status).toBe("normal");
  });

  test("a NORMAL cell with no topology doc publishes successor null, not a dropped entry", () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, {
        [NORMAL_KEY]: regime("normal", NOW - 30 * MIN),
      }),
      segmentParams: null,
      segmentDwell: dwellDoc(),
    });
    expect(snap.segment_flow!.segments[NORMAL_KEY]!.to).toBeNull();
  });
});

describe("station_flow: worst_recovery roll-up", () => {
  test("every pre-existing StationServiceFlow field is unchanged, plus worst_recovery", () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, {
        [KEY]: regime("disrupted", NOW - 30 * MIN),
      }),
      segmentParams: params,
      segmentDwell: dwellDoc(),
      stationFlow: stationFlowDoc(NOW - 300),
    });
    const station = snap.station_flow!.stations["A09"]!;
    expect(station.status).toBe("degraded");
    expect(station.worst_deficit).toBe(0.9);
    expect(station.worst_segment).toEqual(["A09S", "A10S"]);
    expect(station.routes).toEqual(["F"]);
    expect(station.n_segments).toBe(1);
    // The new field: the worst_segment's own recovery, same numbers as the
    // segment surface computed for the same cell.
    expect(station.worst_recovery).toEqual(
      snap.segment_flow!.segments[KEY]!.recovery,
    );
    expect(station.worst_recovery).not.toBeNull();
  });

  test("worst_recovery is null when the worst segment has no recovery (no curve)", () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, {
        [KEY]: regime("disrupted", NOW - 30 * MIN),
      }),
      segmentParams: params,
      segmentDwell: null,
      stationFlow: stationFlowDoc(NOW - 300),
    });
    expect(snap.station_flow!.stations["A09"]!.worst_recovery).toBeNull();
  });

  test("worst_recovery is null when segment_flow itself is absent (station_flow still publishes)", () => {
    const snap = build({
      stationFlow: stationFlowDoc(NOW - 300),
      // no segmentFlow at all this tick
    });
    expect(snap.station_flow).not.toBeNull();
    expect(snap.station_flow!.stations["A09"]!.status).toBe("degraded");
    expect(snap.station_flow!.stations["A09"]!.worst_recovery).toBeNull();
  });

  test("worst_recovery is null when no live segment matches the worst_segment stop pair", () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, {}), // no cells tracked this tick
      segmentParams: params,
      segmentDwell: dwellDoc(),
      stationFlow: stationFlowDoc(NOW - 300),
    });
    expect(snap.station_flow!.stations["A09"]!.worst_recovery).toBeNull();
  });

  test("worst_recovery is null when the worst-touching segment currently reads NORMAL", () => {
    // KEY (A09S -> A10S) is the exact pair worst_segment names, and it IS
    // live and judged this tick — just normal, so its `segments` entry
    // carries a null recovery. There is nothing to forecast recovery FOR on
    // a segment that isn't disrupted.
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, {
        [KEY]: regime("normal", NOW - 30 * MIN),
      }),
      segmentParams: params,
      segmentDwell: dwellDoc(),
      stationFlow: stationFlowDoc(NOW - 300),
    });
    expect(snap.segment_flow!.segments[KEY]!.status).toBe("normal");
    expect(snap.segment_flow!.segments[KEY]!.to).toBe("A10S");
    expect(snap.station_flow!.stations["A09"]!.worst_recovery).toBeNull();
  });
});

describe("segment_flow / station_flow validate against the Pydantic schema", () => {
  test("the near-term production path — segmentDwell absent — still validates", () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, {
        [KEY]: regime("disrupted", NOW - 30 * MIN),
      }),
      segmentParams: params,
      segmentDwell: null,
      stationFlow: stationFlowDoc(NOW - 300),
    });
    expect(snap.segment_flow!.segments[KEY]!.recovery).toBeNull();
    checkSchema(snap);
  });

  test("the full path — every state doc present with a trained curve — validates", () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, {
        [KEY]: regime("disrupted", NOW - 30 * MIN),
      }),
      segmentParams: params,
      segmentDwell: dwellDoc(),
      stationFlow: stationFlowDoc(NOW - 300),
    });
    expect(snap.segment_flow!.segments[KEY]!.recovery).not.toBeNull();
    checkSchema(snap);
  });

  test("a mix of disrupted and normal cells validates together in one segments dict", () => {
    const snap = build({
      segmentFlow: flowDoc(NOW - 300, {
        [KEY]: regime("disrupted", NOW - 30 * MIN),
        [NORMAL_KEY]: regime("normal", NOW - 10 * MIN),
      }),
      segmentParams: params,
      segmentDwell: dwellDoc(),
      stationFlow: stationFlowDoc(NOW - 300),
    });
    expect(Object.keys(snap.segment_flow!.segments).sort()).toEqual(
      [KEY, NORMAL_KEY].sort(),
    );
    expect(snap.segment_flow!.segments[NORMAL_KEY]!.status).toBe("normal");
    checkSchema(snap);
  });

  test("no segment state at all (pre-deploy) still validates", () => {
    checkSchema(build({}));
  });
});
