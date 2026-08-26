import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  enumerateJourneys,
  journeyHopKeys,
  boardStop,
  alightStop,
  type Journey,
} from "../lib/journeys.ts";
import type { AdjEdge, RouteStops } from "../lib/stations.ts";

// Build a synthetic topology: one pattern per route|direction, with an adjacency
// cell backing every consecutive pair unless its `route|direction|from` key is
// listed in `omit` — so a test can prove a ride cannot cross an unbacked pair.
function topo(
  lines: Record<string, string[]>,
  omit: string[] = [],
): { routeStops: RouteStops; edges: AdjEdge[] } {
  const routeStops: RouteStops = {};
  const edges: AdjEdge[] = [];
  const skip = new Set(omit);
  for (const [key, stops] of Object.entries(lines)) {
    routeStops[key] = [{ stops, n_trips: 100 }];
    const bar = key.indexOf("|");
    const route = key.slice(0, bar);
    const direction = key.slice(bar + 1);
    for (let i = 0; i < stops.length - 1; i++) {
      const k = `${route}|${direction}|${stops[i]}`;
      if (skip.has(k)) continue;
      edges.push({
        key: k,
        route,
        direction,
        from: stops[i],
        to: stops[i + 1],
        successors: [{ to: stops[i + 1], n_trips: 100 }],
      });
    }
  }
  return { routeStops, edges };
}

const seq = (j: Journey): string =>
  j.legs.map((l) => `${l.route}|${l.direction}`).join(" / ");

test("direct: a single line boarding at origin and reaching destination", () => {
  const { routeStops, edges } = topo({ "C|north": ["O", "M", "D"] });
  const js = enumerateJourneys(routeStops, edges, ["O"], ["D"]);
  assert.equal(js.length, 1);
  assert.equal(js[0].transfers, 0);
  assert.equal(seq(js[0]), "C|north");
  assert.equal(boardStop(js[0].legs[0]), "O");
  assert.equal(alightStop(js[0].legs[0]), "D");
  // The leg carries the ordered pairwise segments, keyed for live-status joins.
  assert.deepEqual(journeyHopKeys(js[0]), ["C|north|O", "C|north|M"]);
});

test("no journey when the only line runs origin after destination", () => {
  // Wrong direction: D precedes O in the pattern, so no train carries O → D.
  const { routeStops, edges } = topo({ "C|north": ["D", "M", "O"] });
  assert.deepEqual(enumerateJourneys(routeStops, edges, ["O"], ["D"]), []);
});

test("single transfer at a shared stop id, and no needless two-transfer", () => {
  const { routeStops, edges } = topo({
    "A|north": ["O", "X"],
    "B|north": ["X", "D"],
  });
  const js = enumerateJourneys(routeStops, edges, ["O"], ["D"]);
  assert.equal(js.length, 1);
  assert.equal(js[0].transfers, 1);
  assert.equal(seq(js[0]), "A|north / B|north");
  assert.equal(alightStop(js[0].legs[0]), "X");
  assert.equal(boardStop(js[0].legs[1]), "X");
});

test("both a direct and a single transfer are enumerated together", () => {
  const { routeStops, edges } = topo({
    "C|north": ["O", "D"], // direct
    "A|north": ["O", "X"], // \ single transfer at X
    "B|north": ["X", "D"], // /
  });
  const js = enumerateJourneys(routeStops, edges, ["O"], ["D"]);
  assert.deepEqual(
    js.map(seq),
    ["C|north", "A|north / B|north"],
  );
  assert.deepEqual(js.map((j) => j.transfers), [0, 1]);
});

test("one canonical journey per route sequence, transferring earliest", () => {
  // A and B share two complexes (X then Y); the sequence A|north / B|north is
  // one candidate, taken at the first shared complex reached along A.
  const { routeStops, edges } = topo({
    "A|north": ["O", "X", "Y"],
    "B|north": ["X", "Y", "D"],
  });
  const js = enumerateJourneys(routeStops, edges, ["O"], ["D"]);
  assert.equal(js.length, 1);
  assert.equal(seq(js[0]), "A|north / B|north");
  assert.equal(alightStop(js[0].legs[0]), "X"); // earliest shared complex, not Y
  assert.deepEqual(
    js[0].legs[1].segments.map((s) => s.from),
    ["X", "Y"],
  );
});

test("opposite running directions of one route sequence are a single candidate", () => {
  // Route pair A→B connects both northbound (via X) and southbound (via Y). As
  // a candidate it is one sequence "A / B"; the sorted-first running is kept.
  const { routeStops, edges } = topo({
    "A|north": ["O", "X"],
    "B|north": ["X", "D"],
    "A|south": ["O", "Y"],
    "B|south": ["Y", "D"],
  });
  const js = enumerateJourneys(routeStops, edges, ["O"], ["D"]);
  assert.equal(js.length, 1);
  assert.equal(seq(js[0]), "A|north / B|north");
});

test("a transfer needs two different routes, never a self-transfer", () => {
  // Only route A touches both O and D, split across its two directions; a
  // change to the same route is not a transfer, so nothing connects.
  const { routeStops, edges } = topo({
    "A|north": ["O", "X"],
    "A|south": ["X", "D"],
  });
  assert.deepEqual(enumerateJourneys(routeStops, edges, ["O"], ["D"]), []);
});

test("distinct stop ids transfer only when listed as one complex", () => {
  const { routeStops, edges } = topo({
    "A|north": ["O", "X1"],
    "B|north": ["X2", "D"],
  });
  // X1 and X2 are unrelated stations by default → no transfer, no journey.
  assert.deepEqual(enumerateJourneys(routeStops, edges, ["O"], ["D"]), []);
  // Told they are one walkable complex → the single transfer appears.
  const js = enumerateJourneys(routeStops, edges, ["O"], ["D"], {
    complexes: [["X1", "X2"]],
  });
  assert.equal(js.length, 1);
  assert.equal(seq(js[0]), "A|north / B|north");
  assert.equal(alightStop(js[0].legs[0]), "X1");
  assert.equal(boardStop(js[0].legs[1]), "X2");
});

test("a ride cannot cross a pair the adjacency graph does not back", () => {
  // A's pattern lists O,X,D but the X→D cell is missing: A can only be ridden
  // O → X, so it is not a direct, and the trip must transfer to B at X.
  const { routeStops, edges } = topo(
    { "A|north": ["O", "X", "D"], "B|north": ["X", "D"] },
    ["A|north|X"],
  );
  const js = enumerateJourneys(routeStops, edges, ["O"], ["D"]);
  assert.deepEqual(js.map(seq), ["A|north / B|north"]);
  assert.equal(alightStop(js[0].legs[0]), "X");
});

test("two transfers only as a fallback when nothing simpler connects", () => {
  const { routeStops, edges } = topo({
    "A|north": ["O", "X"],
    "B|north": ["X", "Y"],
    "C|north": ["Y", "D"],
  });
  const js = enumerateJourneys(routeStops, edges, ["O"], ["D"]);
  assert.equal(js.length, 1);
  assert.equal(js[0].transfers, 2);
  assert.equal(seq(js[0]), "A|north / B|north / C|north");
  // Capping transfers below two suppresses the only connection.
  assert.deepEqual(enumerateJourneys(routeStops, edges, ["O"], ["D"], { maxTransfers: 1 }), []);
});

test("origin equal to destination yields nothing", () => {
  const { routeStops, edges } = topo({ "C|north": ["O", "D"] });
  assert.deepEqual(enumerateJourneys(routeStops, edges, ["O"], ["O"]), []);
});

// --- Known pair against the committed diagram asset -------------------------
// Brooklyn (Atlantic Av-Barclays Ctr) → the 14 St-Union Sq area. The two
// direct corridors are the Lexington express (4/5) and the Broadway line
// (N/Q/R/W); the 6th/7th-Avenue lines reach it only by transferring.

const diagram = JSON.parse(
  readFileSync(new URL("../public/diagram.json", import.meta.url), "utf8"),
) as { adjacency: AdjEdge[]; route_stops: RouteStops };

const ATLANTIC = ["235", "D24", "R31"]; // 2/3/4/5, B/Q, D/N/R/W platforms
const UNION_SQ = ["635", "L03", "R20"]; // 4/5/6, L, N/Q/R/W platforms

test("Atlantic Av → Union Sq: the direct routes are the Lex and Broadway lines", () => {
  const js = enumerateJourneys(diagram.route_stops, diagram.adjacency, ATLANTIC, UNION_SQ);
  const directs = js.filter((j) => j.transfers === 0).map(seq).sort();
  assert.deepEqual(directs, [
    "4|north",
    "5|north",
    "N|north",
    "Q|north",
    "R|north",
    "W|north",
  ]);
});

test("Atlantic Av → Union Sq: every emitted hop key is a real adjacency cell", () => {
  const js = enumerateJourneys(diagram.route_stops, diagram.adjacency, ATLANTIC, UNION_SQ);
  const cells = new Set(diagram.adjacency.map((e) => e.key));
  for (const j of js) {
    for (const key of journeyHopKeys(j)) {
      assert.ok(cells.has(key), `hop ${key} is not backed by adjacency`);
    }
    // Legs are contiguous: each segment's `to` is the next segment's `from`.
    for (const leg of j.legs) {
      for (let i = 1; i < leg.segments.length; i++) {
        assert.equal(leg.segments[i].from, leg.segments[i - 1].to);
      }
    }
  }
});

test("Atlantic Av → Union Sq: the 4 train rides straight through, in order", () => {
  const js = enumerateJourneys(diagram.route_stops, diagram.adjacency, ATLANTIC, UNION_SQ);
  const four = js.find((j) => j.transfers === 0 && j.legs[0].route === "4");
  assert.ok(four, "the 4 should be a direct route");
  assert.equal(boardStop(four.legs[0]), "235N");
  assert.equal(alightStop(four.legs[0]), "635N");
});

test("Atlantic Av → Union Sq: a single transfer from the 7th-Ave 2 is a candidate", () => {
  const js = enumerateJourneys(diagram.route_stops, diagram.adjacency, ATLANTIC, UNION_SQ);
  const viaTwo = js.filter((j) => j.transfers === 1 && j.legs[0].route === "2");
  assert.ok(viaTwo.length > 0, "the 2 should reach Union Sq with one transfer");
});
