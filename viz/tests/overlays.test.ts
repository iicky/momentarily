// What each map overlay is allowed to claim, and what it must refuse to.
//
// The property under test throughout is the one the map exists to hold: an
// absence of evidence never renders as a healthy or a zero reading, and a
// reading taken at a coarse unit is never presented as a fine one.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  OVERLAYS,
  edgeSeconds,
  fmtSeconds,
  markerRadius,
  paintEdges,
  readSupply,
  serviceClassNow,
  timeBin,
  timeScale,
  trainCoverage,
  trainLayer,
} from "../lib/overlays.ts";
import type {
  NoteSpan,
  Overlay,
  OverlayContext,
  OverlayId,
} from "../lib/overlays.ts";
import type { Diagram, DiagramEdge, DiagramStation } from "../lib/diagram.ts";
import type {
  RouteStatus,
  SegmentFlow,
  SegmentStatus,
  Snapshot,
  TrainPosition,
  Trains,
} from "../lib/types.ts";

// The eight NYCT line-group feeds the Worker polls every tick.
const FEEDS = ["ace", "bdfm", "g", "jz", "nqrw", "l", "numbered", "si"];

function edge(over: Partial<DiagramEdge> = {}): DiagramEdge {
  return {
    route: "1",
    a: "101",
    b: "103",
    path: [
      [0, 0],
      [1, 1],
    ],
    keys: { north: "1|north|103N", south: "1|south|101S" },
    ...over,
  };
}

function station(over: Partial<DiagramStation> = {}): DiagramStation {
  return { name: "Test St", x: 0, y: 0, routes: ["1"], ...over };
}

function diagramOf(
  edges: DiagramEdge[],
  stations: Record<string, DiagramStation> = {},
): Diagram {
  return {
    feed_version: { version: "test", start: null, end: null },
    view_box: [0, 0, 10, 10],
    routes: { "1": { name: "test", color: "#fff" } },
    stations,
    edges,
    insets: [],
  };
}

// The overlays read a handful of the published snapshot's fields. Spelling out
// the whole contract per test would bury the two under test, so the fixtures
// narrow — the cast is the test's, not the app's.
function snapOf(over: Partial<Snapshot> = {}): Snapshot {
  return {
    generated_at: 1000,
    route_status: {},
    segment_flow: null,
    ...over,
  } as Snapshot;
}

function routeStatus(over: Partial<RouteStatus> = {}): RouteStatus {
  return {
    route_id: "1",
    service_condition: "normal",
    service_ratio: null,
    ...over,
  } as RouteStatus;
}

/** A judged segment record. Defaults to a disrupted verdict; override
 * `status` for a normal one — `segments` carries both alike. */
function cell(over: Partial<SegmentStatus> = {}): SegmentStatus {
  return {
    route: "1",
    direction: "north",
    from_stop: "103N",
    to: null,
    status: "disrupted",
    entered_at: 100,
    recovery: null,
    ...over,
  };
}

function flow(segments: Record<string, SegmentStatus>): SegmentFlow {
  return { observed_at: 1000, segments };
}

function position(over: Partial<TrainPosition> = {}): TrainPosition {
  return { route: "1", direction: "north", stop: "101N", stopped: true, n: 1, ...over };
}

function trainsOf(over: Partial<Trains> = {}): Trains {
  return {
    observed_at: 900,
    provenance: { code_sha: "abc1234def5678", dirty: false, producer: "worker" },
    fresh_feeds: [...FEEDS],
    expected_feeds: [...FEEDS],
    positions: [],
    ...over,
  };
}

function ctxOf(over: Partial<OverlayContext> & { diagram: Diagram }): OverlayContext {
  return {
    snap: null,
    filter: "both",
    serviceClass: "weekday",
    now: 1000,
    time: null,
    trains: { state: "loading" },
    ...over,
  };
}

function overlay(id: OverlayId): Overlay {
  const found = OVERLAYS.find((o) => o.id === id);
  if (found === undefined) throw new Error(`no overlay registered for ${id}`);
  return found;
}

/** Note and caption prose as one string, so a test can assert on what a reader
 * actually sees rather than on the span structure carrying it. */
function prose(spans: NoteSpan[]): string {
  return spans.map((s) => (typeof s === "string" ? s : s.em)).join("");
}

// --- The registry itself --------------------------------------------------

test("every overlay states its spatial unit and its source", () => {
  for (const o of OVERLAYS) {
    const caption = prose(o.caption);
    assert.match(
      caption,
      /^Unit: /,
      `${o.id} must open its caption by naming its spatial unit`,
    );
    assert.ok(caption.length > 80, `${o.id} caption is too thin to be honest`);
  }
});

test("the direction control is offered only where direction means something", () => {
  // Supply is per route and trains are per stop: a north/south control on
  // either would imply the reading splits by direction, and it does not.
  assert.equal(overlay("supply").filters, null);
  assert.equal(overlay("trains").filters, null);
  assert.equal(overlay("movement").filters?.length, 3);
  assert.equal(overlay("time").filters?.length, 3);
});

test("verdicts paint after the no-reading strokes they cross", () => {
  const measured = edge({ a: "101", b: "103", keys: { north: "1|north|103N" } });
  const ghost = edge({ a: "103", b: "104", keys: { north: "1|north|104N" } });
  const ctx = ctxOf({
    diagram: diagramOf([measured, ghost]),
    filter: "north",
    snap: snapOf({
      segment_flow: flow({ "1|north|103N": cell({ status: "disrupted" }) }),
    }),
  });
  const order = paintEdges(overlay("movement"), ctx).map((p) => p.id);
  assert.deepEqual(order, ["1|103|104", "1|101|103"]);
});

// --- Movement: unchanged by the registry ----------------------------------

test("movement draws nothing where the timetable schedules nothing", () => {
  const oneWay = edge({ keys: { south: "1|south|101S" } });
  const ctx = ctxOf({ diagram: diagramOf([oneWay]), filter: "north" });
  assert.equal(overlay("movement").paint(oneWay, ctx), null);
});

test("an unmeasured segment keeps the route colour instead of the healthy one", () => {
  const ctx = ctxOf({ diagram: diagramOf([edge()]), snap: snapOf() });
  const paint = overlay("movement").paint(edge(), ctx);
  // null colour means "wear the route's own", which is the shared shape of
  // "we don't know" across all four overlays.
  assert.equal(paint?.color, null);
  assert.ok(paint !== null && paint.opacity < 0.5);
});

test("a normal cell in segments paints the healthy colour, not the no-reading ghost", () => {
  // The regression this union shape must never reintroduce: a reader who
  // takes "absent from segments" as the only signal would dim every
  // healthy cell.
  const ctx = ctxOf({
    diagram: diagramOf([edge()]),
    filter: "north",
    snap: snapOf({
      segment_flow: flow({ "1|north|103N": cell({ status: "normal", to: "101N" }) }),
    }),
  });
  const paint = overlay("movement").paint(edge(), ctx);
  assert.equal(paint?.color, "var(--normal)");
});

test("a normal verdict's row reports its clock and successor, but never a recovery", () => {
  // Every judged cell is a full record now, normal or disrupted alike — a
  // healthy one just has nothing to forecast, so recovery is the only thing
  // missing from its row.
  const ctx = ctxOf({
    diagram: diagramOf([edge()], { "101": station({ name: "First St" }) }),
    now: 1000,
    snap: snapOf({
      segment_flow: flow({
        "1|north|103N": cell({ status: "normal", entered_at: 400, to: "101N" }),
      }),
    }),
  });
  const row = overlay("movement").detail(edge(), ctx).find((r) => r.key === "N");
  assert.equal(row?.value, "advancing");
  assert.match(row?.note ?? "", /since/);
  assert.match(row?.note ?? "", /toward First St/);
  assert.doesNotMatch(row?.note ?? "", /recovery/);
});

test("a disrupted row still reports its clock and recovery", () => {
  const ctx = ctxOf({
    diagram: diagramOf([edge()]),
    now: 1000,
    snap: snapOf({
      segment_flow: flow({
        "1|north|103N": cell({
          entered_at: 400,
          recovery: {
            recovery_minutes: 12,
            recovery_minutes_low: 6,
            recovery_minutes_high: 20,
            recovery_indeterminate: false,
            p_normal_in_30min: null,
            p_normal_in_60min: 0.5,
            p_normal_in_120min: 0.8,
          },
        }),
      }),
    }),
  });
  const row = overlay("movement").detail(edge(), ctx).find((r) => r.key === "N");
  assert.equal(row?.value, "not advancing");
  assert.match(row?.note ?? "", /since/);
  assert.match(row?.note ?? "", /recovery ~12m/);
});

test("nothing on the movement overlay is dashed", () => {
  // A dash on this map means exactly one thing: the reading came from a
  // coarser unit than the edge it covers. That is true of supply and of
  // nothing else, so movement must never dash.
  const ctx = ctxOf({
    diagram: diagramOf([edge()]),
    filter: "north",
    snap: snapOf({
      segment_flow: flow({
        "1|north|103N": cell(),
        "1|south|101S": cell({ status: "normal", direction: "south", from_stop: "101S", to: "103S" }),
      }),
    }),
  });
  for (const p of paintEdges(overlay("movement"), ctx)) {
    assert.equal(p.paint.dash, null);
  }
});

test("the coverage note counts every judged cell in segments", () => {
  const ctx = ctxOf({
    diagram: diagramOf([edge()]),
    snap: snapOf({
      segment_flow: flow({
        "1|north|103N": cell(),
        "1|south|101S": cell({ status: "normal", direction: "south", from_stop: "101S", to: "103S" }),
      }),
    }),
  });
  const note = prose(overlay("movement").note(ctx));
  // Two cells judged of two scheduled, one of them disrupted.
  assert.match(note, /2 of 2 scheduled directional segments/);
  assert.match(note, /1 of them not advancing/);
  // And no leftover pooling vocabulary.
  assert.doesNotMatch(note, /pooled|corridor|independent measurements/);
});

// --- Supply: per route ----------------------------------------------------

test("an unknown supply condition is a no-reading state, never a healthy one", () => {
  const snap = snapOf({
    route_status: { "1": routeStatus({ service_condition: "unknown" }) },
  });
  const supply = readSupply("1", snap);
  assert.equal(supply.state, "unknown");
  assert.equal(supply.present, true);

  const ctx = ctxOf({ diagram: diagramOf([edge()]), snap });
  const paint = overlay("supply").paint(edge(), ctx);
  assert.equal(paint?.color, null, "unknown supply must not take a state colour");
  assert.notEqual(paint?.color, "var(--normal)");
});

test("a route absent from route_status reads no-reading and says which absence", () => {
  const snap = snapOf({ route_status: {} });
  const supply = readSupply("1", snap);
  assert.equal(supply.state, "unknown");
  assert.equal(supply.present, false);

  const ctx = ctxOf({ diagram: diagramOf([edge()]), snap });
  assert.equal(overlay("supply").paint(edge(), ctx)?.color, null);
  const [row] = overlay("supply").detail(edge(), ctx);
  assert.match(row.note ?? "", /not in the snapshot's route_status/);
});

test("a supply condition outside the published vocabulary is unknown, not guessed", () => {
  // 'suspended' is the flow axis's word, not the supply axis's. A future or
  // stale Worker sending something unrecognised must not be interpreted.
  const snap = snapOf({
    route_status: { "1": routeStatus({ service_condition: "suspended" }) },
  });
  assert.equal(readSupply("1", snap).state, "unknown");
});

test("one route's supply reading lands on every one of that route's edges", () => {
  const snap = snapOf({
    route_status: {
      "1": routeStatus({ service_condition: "degraded", service_ratio: 0.4 }),
    },
  });
  const a = edge({ a: "101", b: "103" });
  const b = edge({ a: "103", b: "104" });
  const ctx = ctxOf({ diagram: diagramOf([a, b]), snap });
  const painted = paintEdges(overlay("supply"), ctx);
  assert.equal(painted.length, 2);
  for (const p of painted) {
    assert.equal(p.paint.color, "var(--disrupted)");
    // Dashed: the colour came from the route, not from this edge. Without the
    // dash a reader could take it for a per-segment measurement.
    assert.ok(p.paint.dash !== null);
  }
});

test("the supply no-reading stroke is not dashed, so the dash means one thing", () => {
  const ctx = ctxOf({ diagram: diagramOf([edge()]), snap: snapOf() });
  assert.equal(overlay("supply").paint(edge(), ctx)?.dash, null);
});

test("the supply overlay says out loud that its unit is the route", () => {
  const ctx = ctxOf({
    diagram: diagramOf([edge()]),
    snap: snapOf({
      route_status: { "1": routeStatus({ service_condition: "degraded" }) },
    }),
  });
  assert.match(prose(overlay("supply").caption), /route/);
  assert.match(prose(overlay("supply").note(ctx)), /per-route/);
  const [row] = overlay("supply").detail(edge(), ctx);
  assert.match(row.note ?? "", /not measured on this segment/);
});

test("supply reports the published ratio and never invents a missing one", () => {
  const withRatio = snapOf({
    route_status: {
      "1": routeStatus({ service_condition: "degraded", service_ratio: 0.35 }),
    },
  });
  const ctx = ctxOf({ diagram: diagramOf([edge()]), snap: withRatio });
  assert.match(overlay("supply").detail(edge(), ctx)[0].note ?? "", /35%/);

  const noRatio = snapOf({
    route_status: {
      "1": routeStatus({ service_condition: "degraded", service_ratio: null }),
    },
  });
  const bare = ctxOf({ diagram: diagramOf([edge()]), snap: noRatio });
  const note = overlay("supply").detail(edge(), bare)[0].note ?? "";
  assert.match(note, /without a ratio/);
  assert.doesNotMatch(note, /\d+%/);
});

// --- Scheduled time: per hop, per direction, per service class -------------

function timed(
  seconds: DiagramEdge["seconds"],
  over: Partial<DiagramEdge> = {},
): DiagramEdge {
  return edge({ seconds, ...over });
}

test("a direction with no scheduled time renders no timing, not zero", () => {
  const e = timed({ weekday: { north: 120 } });
  const ctx = ctxOf({
    diagram: diagramOf([e]),
    time: timeScale(diagramOf([e]), "weekday"),
  });
  const rows = overlay("time").detail(e, ctx);
  const south = rows.find((r) => r.key === "S");
  assert.equal(south?.value, "no timing");
  assert.equal(south?.color, null);
  assert.doesNotMatch(south?.value ?? "", /0/);
  // And the northbound row still reports its real value.
  assert.equal(rows.find((r) => r.key === "N")?.value, "2:00");
});

test("a service class the timetable doesn't cover reads no timing for the hop", () => {
  const e = timed({ weekday: { north: 120, south: 90 } });
  const diagram = diagramOf([e]);
  assert.equal(edgeSeconds(e, "sunday", "both"), null);
  const ctx = ctxOf({
    diagram,
    serviceClass: "sunday",
    time: timeScale(diagram, "sunday"),
  });
  assert.equal(overlay("time").paint(e, ctx)?.color, null);
  assert.equal(overlay("time").detail(e, ctx)[0].value, "no timing");
});

test("each service class is ranked against its own timetable", () => {
  // 180s is the slowest weekday hop here and the fastest Sunday one. Sharing a
  // scale would rank it against runs it isn't competing with.
  const edges = [
    timed({ weekday: { north: 60 }, sunday: { north: 180 } }, { a: "1", b: "2" }),
    timed({ weekday: { north: 180 }, sunday: { north: 600 } }, { a: "2", b: "3" }),
  ];
  const diagram = diagramOf(edges);
  const weekday = timeScale(diagram, "weekday");
  const sunday = timeScale(diagram, "sunday");
  assert.equal(weekday?.max, 180);
  assert.equal(sunday?.min, 180);
  assert.notDeepEqual(
    weekday?.bins.map((b) => b.max),
    sunday?.bins.map((b) => b.max),
  );
});

test("the combined view shows the slower direction, not a health verdict", () => {
  const e = timed({ weekday: { north: 90, south: 240 } });
  assert.equal(edgeSeconds(e, "weekday", "both"), 240);
  assert.equal(edgeSeconds(e, "weekday", "north"), 90);
  // And it falls back to the only direction the timetable times.
  assert.equal(edgeSeconds(timed({ weekday: { south: 75 } }), "weekday", "both"), 75);
  // The control's label must not import the health vocabulary.
  const both = overlay("time").filters?.find((f) => f.id === "both");
  assert.equal(both?.label, "Slower direction");
});

test("an asset with no timings yields no scale and paints nothing as measured", () => {
  const diagram = diagramOf([edge()]);
  assert.equal(timeScale(diagram, "weekday"), null);
  const ctx = ctxOf({ diagram, time: null });
  assert.equal(overlay("time").paint(edge(), ctx)?.color, null);
  assert.match(prose(overlay("time").note(ctx)), /no weekday timings/);
  assert.equal(overlay("time").legend(ctx).length, 1);
});

test("the time ramp shares no colour with the movement state colours", () => {
  // A magnitude painted in the health palette reads as health. The state trio
  // is off limits to this scale, and so is a reader's learned vocabulary.
  const diagram = diagramOf([
    timed({ weekday: { north: 60, south: 90 } }, { a: "1", b: "2" }),
    timed({ weekday: { north: 240, south: 600 } }, { a: "2", b: "3" }),
  ]);
  const scale = timeScale(diagram, "weekday");
  assert.ok(scale !== null);
  for (const bin of scale.bins) {
    assert.ok(
      !["var(--normal)", "var(--disrupted)", "var(--suspended)"].includes(bin.color),
      `ramp colour ${bin.color} is a state colour`,
    );
  }
});

/** A tie-heavy sample shaped like the real timetable: almost everything at 60,
 * 90 or 120 seconds, with a thin slow tail. */
function roundTimetable(): Diagram {
  const counts: Array<[number, number]> = [
    [60, 5],
    [90, 10],
    [120, 6],
    [180, 2],
    [600, 1],
  ];
  const edges: DiagramEdge[] = [];
  let i = 0;
  for (const [seconds, n] of counts) {
    for (let k = 0; k < n; k += 1) {
      i += 1;
      edges.push(
        timed(
          { weekday: { north: seconds } },
          { a: `s${i}`, b: `s${i + 1}`, keys: { north: `1|north|s${i}N` } },
        ),
      );
    }
  }
  return diagramOf(edges);
}

test("the ramp separates the values a round-number timetable actually uses", () => {
  // Plain quintiles put two boundaries on 90 (40% of hops) and collapse a bin,
  // which loses the 60-vs-90-vs-120 distinction that is the whole signal in
  // dense track. Each dominant value must get a boundary of its own.
  const scale = timeScale(roundTimetable(), "weekday");
  assert.ok(scale !== null);
  const bounds = scale.bins.map((b) => b.max);
  for (const dominant of [60, 90, 120]) {
    assert.ok(bounds.includes(dominant), `no bin boundary at ${dominant}s`);
  }
  assert.equal(scale.bins.find((b) => b.max === 90)?.n, 10);
});

test("the ramp's bins tile the sample without overlapping or losing a hop", () => {
  for (const diagram of [roundTimetable(), manyValues()]) {
    const scale = timeScale(diagram, "weekday");
    assert.ok(scale !== null);
    const bounds = scale.bins.map((b) => b.max);
    assert.ok(bounds.length <= 5, "more bins than the ramp has colours");
    assert.deepEqual(bounds, [...bounds].sort((a, b) => a - b), "bins must ascend");
    assert.equal(new Set(bounds).size, bounds.length, "no duplicate boundaries");
    assert.equal(bounds[bounds.length - 1], scale.max, "top bin must reach the max");
    assert.equal(
      scale.bins.reduce((sum, b) => sum + b.n, 0),
      scale.timed,
      "every timed hop must land in exactly one bin",
    );
  }
});

/** Twelve distinct values, one hop each: forces the thinning path, since the
 * quantile probes find more boundaries than the ramp has colours. */
function manyValues(): Diagram {
  return diagramOf(
    Array.from({ length: 12 }, (_, i) =>
      timed(
        { weekday: { north: 30 * (i + 1) } },
        { a: `m${i}`, b: `m${i + 1}`, keys: { north: `1|north|m${i}N` } },
      ),
    ),
  );
}

test("the legend labels bins with real boundary values, not bin numbers", () => {
  const diagram = roundTimetable();
  const ctx = ctxOf({ diagram, time: timeScale(diagram, "weekday") });
  const labels = overlay("time").legend(ctx).map((i) => i.label);
  assert.ok(labels.includes("1:00"), `expected the 60s endpoint, got ${labels}`);
  assert.ok(
    labels.some((l) => l.includes("10:00")),
    `expected the 600s endpoint, got ${labels}`,
  );
  // The no-data state is in the legend too, or a reader can't decode a ghost.
  assert.equal(labels[labels.length - 1], "no timing");
});

test("the time note counts hops per drawn hop, not per segment cell", () => {
  const diagram = roundTimetable();
  const scale = timeScale(diagram, "weekday");
  const ctx = ctxOf({ diagram, time: scale });
  const note = prose(overlay("time").note(ctx));
  assert.match(note, new RegExp(`${scale?.timed} of ${scale?.slots}`));
  assert.match(note, /per drawn hop rather than per segment cell/);
});

test("a hop slower than the scale's top boundary still gets the top bin", () => {
  const scale = timeScale(roundTimetable(), "weekday");
  assert.ok(scale !== null);
  assert.equal(timeBin(scale, 99_999), scale.bins[scale.bins.length - 1]);
});

test("hop times read as m:ss above a minute and plain seconds below", () => {
  assert.equal(fmtSeconds(30), "30s");
  assert.equal(fmtSeconds(60), "1:00");
  assert.equal(fmtSeconds(105), "1:45");
  assert.equal(fmtSeconds(960), "16:00");
});

test("the service class defaults to today's New York calendar", () => {
  // 2026-08-22 is a Saturday and 2026-08-24 a Monday in New York.
  assert.equal(serviceClassNow(Date.parse("2026-08-22T16:00:00Z")), "saturday");
  assert.equal(serviceClassNow(Date.parse("2026-08-23T16:00:00Z")), "sunday");
  assert.equal(serviceClassNow(Date.parse("2026-08-24T16:00:00Z")), "weekday");
  // Monday 02:00Z is Sunday evening in New York, and Sunday's timetable is
  // still the one running. A naive UTC weekday would call this a weekday.
  assert.equal(serviceClassNow(Date.parse("2026-08-24T02:00:00Z")), "sunday");
});

// --- Trains: per stop, from a separately published object ------------------

test("an unavailable train surface yields an empty layer, not a crash", () => {
  const diagram = diagramOf([edge()], { "101": station() });
  const layer = trainLayer(diagram, null);
  assert.deepEqual(layer.markers, []);
  assert.equal(layer.total, 0);
  assert.equal(layer.unplaced, 0);
  assert.equal(layer.absent, true);
  assert.equal(layer.observed_at, null);

  const ctx = ctxOf({
    diagram,
    trains: { state: "unavailable", reason: "not published at this feed yet" },
  });
  const trains = overlay("trains");
  assert.equal(trains.stamp(ctx)?.at, null);
  assert.equal(trains.caveat?.(ctx), null);
  assert.match(prose(trains.note(ctx)), /not published at this feed yet/);
  assert.match(prose(trains.note(ctx)), /missing report/);
  // And the detail panel refuses to report an empty platform it never heard
  // anything about.
  for (const row of trains.detail(edge(), ctx)) {
    assert.equal(row.note, "no train report to place");
  }
});

test("a loading train surface is not reported as an absence of trains", () => {
  const ctx = ctxOf({ diagram: diagramOf([edge()]), trains: { state: "loading" } });
  const note = prose(overlay("trains").note(ctx));
  assert.match(note, /Fetching/);
  assert.doesNotMatch(note, /0 trains/);
});

test("positions fold onto the parent station, keeping stopped apart from inbound", () => {
  const diagram = diagramOf([edge()], {
    "101": station({ name: "First St", x: 5, y: 7 }),
  });
  const layer = trainLayer(
    diagram,
    trainsOf({
      positions: [
        position({ stop: "101N", stopped: true, n: 2 }),
        position({ stop: "101S", stopped: false, n: 3, direction: "south" }),
        position({ stop: "101N", stopped: false, n: 1, route: "2" }),
      ],
    }),
  );
  assert.equal(layer.markers.length, 1);
  const [marker] = layer.markers;
  assert.equal(marker.station, "101");
  assert.equal(marker.x, 5);
  assert.equal(marker.y, 7);
  assert.equal(marker.stopped, 2);
  assert.equal(marker.inbound, 4);
  assert.equal(layer.total, 6);
  assert.equal(layer.unplaced, 0);
});

test("a stop this diagram can't place is reported rather than dropped", () => {
  const diagram = diagramOf([edge()], { "101": station() });
  const layer = trainLayer(
    diagram,
    trainsOf({
      positions: [position({ stop: "101N", n: 1 }), position({ stop: "999N", n: 4 })],
    }),
  );
  assert.equal(layer.total, 5);
  assert.equal(layer.unplaced, 4);
  assert.equal(layer.markers.length, 1);

  const ctx = ctxOf({
    diagram,
    trains: {
      state: "ready",
      trains: trainsOf({
        positions: [
          position({ stop: "101N", n: 1 }),
          position({ stop: "999N", n: 4 }),
        ],
      }),
    },
  });
  assert.match(prose(overlay("trains").note(ctx)), /4 trains sit at a stop/);
});

test("an incomplete feed set is caveated and refuses a per-route claim", () => {
  const trains = trainsOf({
    fresh_feeds: FEEDS.filter((f) => f !== "nqrw" && f !== "l"),
    positions: [position({ stop: "101N" })],
  });
  const cover = trainCoverage(trains);
  assert.equal(cover.complete, false);
  assert.equal(cover.fresh, 6);
  assert.equal(cover.expected, 8);
  assert.deepEqual(cover.stale, ["nqrw", "l"]);

  const diagram = diagramOf([edge()], { "101": station() });
  const ctx = ctxOf({ diagram, trains: { state: "ready", trains } });
  const caveat = overlay("trains").caveat?.(ctx);
  assert.ok(caveat, "an incomplete read must be caveated where it can't be missed");
  const words = prose(caveat);
  assert.match(words, /6 of 8/);
  assert.match(words, /nqrw, l/);
  // The published object carries no feed-to-route mapping, so naming affected
  // routes would be a fabrication. The caveat says so instead.
  assert.match(words, /cannot say which routes/);
});

test("an incomplete read dims the whole line map instead of implying emptiness", () => {
  const diagram = diagramOf([edge()], { "101": station() });
  const partial = ctxOf({
    diagram,
    trains: {
      state: "ready",
      trains: trainsOf({ fresh_feeds: FEEDS.slice(0, 7) }),
    },
  });
  const whole = ctxOf({
    diagram,
    trains: { state: "ready", trains: trainsOf() },
  });
  const dimmed = overlay("trains").paint(edge(), partial);
  const rested = overlay("trains").paint(edge(), whole);
  assert.ok(dimmed !== null && rested !== null);
  assert.ok(
    dimmed.opacity < rested.opacity,
    "a partial read must not look like the full line map",
  );
  // A station with nothing reported can only be called empty when every feed
  // reported.
  const [row] = overlay("trains").detail(edge(), partial);
  assert.match(row.note ?? "", /read is incomplete/);
  assert.equal(overlay("trains").detail(edge(), whole)[0].note, "no trains reported here");
});

test("a complete feed set with no positions is an observation, not a gap", () => {
  const diagram = diagramOf([edge()], { "101": station() });
  const ctx = ctxOf({
    diagram,
    trains: { state: "ready", trains: trainsOf({ positions: [] }) },
  });
  assert.equal(overlay("trains").caveat?.(ctx), null);
  const note = prose(overlay("trains").note(ctx));
  assert.match(note, /All 8 vehicle feeds reported/);
  assert.match(note, /observed empty system/);
  // And it is drawn as the plain line map, not as the unknown treatment.
  const paint = overlay("trains").paint(edge(), ctx);
  assert.ok(paint !== null && paint.opacity > 0.5);
});

test("the train surface is stamped with its own clock, never the snapshot's", () => {
  const ctx = ctxOf({
    diagram: diagramOf([edge()]),
    snap: snapOf({ generated_at: 5000 }),
    trains: { state: "ready", trains: trainsOf({ observed_at: 1234 }) },
  });
  const stamp = overlay("trains").stamp(ctx);
  assert.equal(stamp?.at, 1234);
  assert.notEqual(stamp?.at, 5000);
});

test("marker radius encodes the count by area", () => {
  // Four trains must read as four times the ink, not four times the width.
  assert.equal(markerRadius(4), markerRadius(1) * 2);
  assert.equal(markerRadius(9), markerRadius(1) * 3);
  // Nothing to draw for a count of nothing.
  assert.equal(markerRadius(0), 0);
});
