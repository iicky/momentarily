import { test } from "node:test";
import assert from "node:assert/strict";
import {
  mergePatterns,
  patternsForTrip,
  orderTrip,
  undirected,
  type RoutePattern,
  type RouteStops,
} from "../lib/stations.ts";

const pat = (stops: string[], n: number): RoutePattern => ({ stops, n_trips: n });

test("mergePatterns splices a local variant into the express spine", () => {
  // Express is dominant; the local's extra stops land next to their neighbours.
  const merged = mergePatterns([
    pat(["A", "C", "E"], 300),
    pat(["A", "B", "C", "D", "E"], 40),
  ]);
  assert.deepEqual(merged, ["A", "B", "C", "D", "E"]);
});

test("mergePatterns appends a disjoint run as its own block", () => {
  // Three unrelated shuttles share no stops: each stays an ordered block.
  const merged = mergePatterns([
    pat(["A", "B", "C"], 10),
    pat(["X", "Y"], 8),
  ]);
  assert.deepEqual(merged, ["A", "B", "C", "X", "Y"]);
});

test("mergePatterns names every stop exactly once", () => {
  const merged = mergePatterns([
    pat(["A", "B", "C"], 5),
    pat(["A", "B", "D"], 3),
  ]);
  assert.deepEqual([...merged].sort(), ["A", "B", "C", "D"]);
  assert.equal(new Set(merged).size, merged.length);
});

test("patternsForTrip resolves the S badge to its shuttle routes, most-run first", () => {
  const rs: RouteStops = {
    "FS|north": [pat(["a"], 5)],
    "GS|north": [pat(["b"], 9)],
    "H|north": [pat(["c"], 2)],
  };
  assert.deepEqual(
    patternsForTrip(rs, "S", "north").map((p) => p.n_trips),
    [9, 5, 2],
  );
});

test("orderTrip prefers published patterns and falls back to the adjacency walk", () => {
  const edges = [
    { key: "2|south|A", route: "2", direction: "south", from: "A", to: "B", successors: [{ to: "B", n_trips: 9 }] },
    { key: "2|south|B", route: "2", direction: "south", from: "B", to: "C", successors: [{ to: "C", n_trips: 9 }] },
  ];
  // No patterns → walk the adjacency graph from the source.
  assert.deepEqual(orderTrip({}, edges, "2", "south"), ["A", "B", "C"]);
  // Patterns present → they win.
  const rs: RouteStops = { "2|south": [pat(["A", "B", "C", "D"], 100)] };
  assert.deepEqual(orderTrip(rs, edges, "2", "south"), ["A", "B", "C", "D"]);
});

test("undirected strips a trailing N/S", () => {
  assert.equal(undirected("101S"), "101");
  assert.equal(undirected("R30N"), "R30");
  assert.equal(undirected("101"), "101");
});
