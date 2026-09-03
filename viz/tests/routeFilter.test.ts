// The line filter's one job is to hide routes without ever changing what the
// map claims about the ones it keeps. The property under test throughout: a
// selection is a SUBSET of the already-painted geometry, never a re-derivation
// — so a segment verdict attributed to one branch can never be spread onto a
// sibling by filtering, and a transfer stop survives on the union of its lines.

import { test } from "node:test";
import assert from "node:assert/strict";
import {
  compareRoutes,
  drawnRoutes,
  edgeShown,
  isRouteOn,
  parseSelection,
  serializeSelection,
  stationShown,
  toggleRoute,
} from "../lib/routeFilter.ts";
import { OVERLAYS, paintEdges, edgeId } from "../lib/overlays.ts";
import type { Overlay, OverlayContext } from "../lib/overlays.ts";
import type { Diagram, DiagramEdge, DiagramStation } from "../lib/diagram.ts";
import type { SegmentFlow, SegmentStatus, Snapshot } from "../lib/types.ts";

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
    routes: {},
    stations,
    edges,
    insets: [],
    topology_source: "gtfs_static",
    adjacency: [],
    route_stops: {},
  };
}

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

function snapOf(over: Partial<Snapshot> = {}): Snapshot {
  return {
    generated_at: 1000,
    route_status: {},
    segment_flow: null,
    ...over,
  } as Snapshot;
}

function ctxOf(
  over: Partial<OverlayContext> & { diagram: Diagram },
): OverlayContext {
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

function movement(): Overlay {
  const found = OVERLAYS.find((o) => o.id === "movement");
  if (found === undefined) throw new Error("no movement overlay registered");
  return found;
}

// --- The drawn-route roster ------------------------------------------------

test("drawnRoutes lists each drawn route once, in rider order", () => {
  const diagram = diagramOf([
    edge({ route: "2" }),
    edge({ route: "10" }),
    edge({ route: "2" }),
    edge({ route: "A" }),
    edge({ route: "1" }),
  ]);
  // 1,2,10 numerically (not lexically), letters after, no duplicate 2.
  assert.deepEqual(drawnRoutes(diagram), ["1", "2", "10", "A"]);
});

test("compareRoutes orders numbers numerically and letters lexically", () => {
  assert.ok(compareRoutes("2", "10") < 0);
  assert.ok(compareRoutes("A", "B") < 0);
});

// --- Edge visibility: one route per edge, no attribution to get wrong -------

test("a null selection shows every edge; a subset shows only its routes", () => {
  const a = edge({ route: "1" });
  const b = edge({ route: "2" });
  assert.equal(edgeShown(a, null), true);
  assert.equal(edgeShown(b, null), true);
  const sel = new Set(["1"]);
  assert.equal(edgeShown(a, sel), true);
  assert.equal(edgeShown(b, sel), false);
  assert.equal(isRouteOn(sel, "1"), true);
  assert.equal(isRouteOn(sel, "2"), false);
  assert.equal(isRouteOn(null, "2"), true);
});

// --- The union rule lives at the station -----------------------------------

test("a transfer stop survives while any of its lines is selected, and only drops when all are hidden", () => {
  const transfer = station({ routes: ["1", "2", "3"] });
  const single = station({ routes: ["3"] });
  assert.equal(stationShown(transfer, null), true);
  // 1 and 2 hidden, 3 still on: the shared stop stays.
  assert.equal(stationShown(transfer, new Set(["3"])), true);
  // A stop served only by hidden lines drops — no orphan dot.
  assert.equal(stationShown(single, new Set(["1", "2"])), false);
  // Every line through it hidden: it drops.
  assert.equal(stationShown(transfer, new Set(["4"])), false);
});

// --- Filtering never re-attributes a verdict -------------------------------

test("filtering the painted geometry is a pure subset — a branch verdict never spreads to its sibling", () => {
  // Two edges leave station 103 as a branch, both keyed on the same from_stop
  // cell `1|north|103N`. The published reading names successor 104, so the
  // overlay attributes it to edge A (103→104) and leaves the sibling B (103→105)
  // unmeasured. Filtering to route 1 must keep exactly that: A lit, B a ghost.
  const a = edge({
    route: "1",
    a: "103",
    b: "104",
    keys: { north: "1|north|103N" },
  });
  const b = edge({
    route: "1",
    a: "103",
    b: "105",
    keys: { north: "1|north|103N" },
  });
  const ctx = ctxOf({
    diagram: diagramOf([a, b]),
    filter: "north",
    snap: snapOf({
      segment_flow: flow({
        "1|north|103N": cell({ status: "disrupted", to: "104N" }),
      }),
    }),
  });
  const painted = paintEdges(movement(), ctx);
  const byId = Object.fromEntries(painted.map((p) => [p.id, p]));
  // Baseline attribution: A carries the disrupted colour, B stays no-reading.
  assert.equal(byId[edgeId(a)].paint.color, "var(--disrupted)");
  assert.equal(byId[edgeId(b)].paint.color, null);

  const sel = new Set(["1"]);
  const shown = painted.filter((p) => edgeShown(p.edge, sel));
  // Both are route 1, so both survive — and, crucially, each keeps the EXACT
  // paint the overlay attributed. Filtering added nothing and moved nothing.
  assert.equal(shown.length, 2);
  for (const p of shown) assert.equal(p.paint, byId[p.id].paint);
  assert.equal(
    byId[edgeId(b)].paint.color,
    null,
    "the sibling branch never inherits A's verdict",
  );

  // And hiding route 1 removes both without touching any paint object.
  const hidden = painted.filter((p) => edgeShown(p.edge, new Set(["2"])));
  assert.equal(hidden.length, 0);
});

// --- Toggle semantics ------------------------------------------------------

test("toggleRoute canonicalises all-on back to null and round-trips a single line", () => {
  const all = ["1", "2", "3"];
  // From all-on (null), toggling 2 off yields the explicit remaining pair.
  const without2 = toggleRoute(null, "2", all);
  assert.deepEqual([...(without2 ?? [])].sort(compareRoutes), ["1", "3"]);
  // Toggling 2 back on restores the canonical all-on null.
  assert.equal(toggleRoute(without2, "2", all), null);
});

// --- URL round-trip --------------------------------------------------------

test("selection round-trips through the URL, with all-on and none-on as distinct states", () => {
  const all = ["1", "2", "3"];
  // All-on omits the param entirely.
  assert.equal(serializeSelection(null, all), null);
  assert.equal(parseSelection(""), null);
  // A subset serialises in drawn order and parses back to the same set.
  const sel = new Set(["3", "1"]);
  assert.equal(serializeSelection(sel, all), "1,3");
  assert.deepEqual(parseSelection("?routes=1,3"), new Set(["1", "3"]));
  // None-on is a real shareable state, distinct from all-on.
  assert.equal(serializeSelection(new Set(), all), "none");
  assert.deepEqual(parseSelection("?routes=none"), new Set());
  // A full set canonicalises to all-on (no param).
  assert.equal(serializeSelection(new Set(all), all), null);
});
