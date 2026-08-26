import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { indexComplexes, journeysBetween, type ComplexStations } from "../lib/complexes.ts";
import type { AdjEdge, RouteStops } from "../lib/stations.ts";

// Same synthetic-topology helper as the journeys tests: one pattern per
// route|direction, with an adjacency cell backing every consecutive pair.
function topo(lines: Record<string, string[]>): { routeStops: RouteStops; edges: AdjEdge[] } {
  const routeStops: RouteStops = {};
  const edges: AdjEdge[] = [];
  for (const [key, stops] of Object.entries(lines)) {
    routeStops[key] = [{ stops, n_trips: 100 }];
    const bar = key.indexOf("|");
    const route = key.slice(0, bar);
    const direction = key.slice(bar + 1);
    for (let i = 0; i < stops.length - 1; i++) {
      edges.push({
        key: `${route}|${direction}|${stops[i]}`,
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

const stations = (rows: Record<string, string | null>): ComplexStations =>
  Object.fromEntries(
    Object.entries(rows).map(([id, cid]) => [id, { station_complex_id: cid }]),
  );

test("indexComplexes groups stops by complex id and lists them deterministically", () => {
  const idx = indexComplexes(
    stations({ R20: "602", L03: "602", "635": "602", A01: "700" }),
  );
  assert.deepEqual(idx.stopsOf("602"), ["635", "L03", "R20"]); // sorted
  assert.deepEqual(idx.stopsOf("700"), ["A01"]);
  assert.deepEqual(idx.ids, ["602", "700"]);
  assert.equal(idx.complexOf("R20"), "602");
  assert.equal(idx.complexOf("635N"), "602"); // directional id collapses
  assert.equal(idx.complexOf("Z99"), null);
});

test("indexComplexes keeps same-named but distinct complexes apart", () => {
  // The two "96 St" stations share a name but are different complexes; grouping
  // by complex id must not merge them, or it would invent a transfer.
  const idx = indexComplexes(stations({ "120": "310", "625": "396" }));
  assert.equal(idx.complexOf("120"), "310");
  assert.equal(idx.complexOf("625"), "396");
  assert.deepEqual(idx.transferComplexes, []); // both are singletons
});

test("transferComplexes carries only the multi-stop complexes", () => {
  const idx = indexComplexes(
    stations({ R20: "602", L03: "602", "635": "602", A01: "700", B01: "701", C01: null }),
  );
  assert.deepEqual(idx.transferComplexes, [["635", "L03", "R20"]]);
});

test("journeysBetween resolves a multi-platform origin complex to a direct ride", () => {
  // Origin complex OC spans two platforms; the rider picked the complex, and the
  // route serving its other platform still connects.
  const { routeStops, edges } = topo({ "C|north": ["O2", "D1"] });
  const idx = indexComplexes(stations({ O1: "OC", O2: "OC", D1: "DC" }));
  const js = journeysBetween(routeStops, edges, idx, "OC", "DC");
  assert.equal(js.length, 1);
  assert.equal(js[0].transfers, 0);
  assert.equal(js[0].legs[0].route, "C");
});

test("journeysBetween unifies a multi-id transfer complex across platforms", () => {
  // A alights at T1, B departs from T2; only the shared complex TC makes the
  // transfer legal — proving the index's transferComplexes reaches the enumerator.
  const { routeStops, edges } = topo({ "A|north": ["O1", "T1"], "B|north": ["T2", "D1"] });
  const idx = indexComplexes(
    stations({ O1: "OC", T1: "TC", T2: "TC", D1: "DC" }),
  );
  const js = journeysBetween(routeStops, edges, idx, "OC", "DC");
  assert.equal(js.length, 1);
  assert.equal(js[0].transfers, 1);
  assert.deepEqual(js[0].legs.map((l) => l.route), ["A", "B"]);
  assert.equal(js[0].legs[0].segments.at(-1)?.to, "T1");
  assert.equal(js[0].legs[1].segments[0].from, "T2");
});

test("journeysBetween yields nothing for an unknown complex id", () => {
  const { routeStops, edges } = topo({ "C|north": ["O1", "D1"] });
  const idx = indexComplexes(stations({ O1: "OC", D1: "DC" }));
  assert.deepEqual(journeysBetween(routeStops, edges, idx, "OC", "ZZ"), []);
});

// --- The resolver over the committed topology, with a complex map matching the
// live snapshot's shape (station_complex_id → member stop ids). This is the
// Brooklyn → Union Sq pair with cross-platform transfers actually enabled.

const diagram = JSON.parse(
  readFileSync(new URL("../public/diagram.json", import.meta.url), "utf8"),
) as { adjacency: AdjEdge[]; route_stops: RouteStops };

// Complex ids/members as the public snapshot reports them for these stations.
const SNAPSHOT = stations({
  "235": "617", D24: "617", R31: "617", // Atlantic Av-Barclays Ctr
  "635": "602", R20: "602", L03: "602", // 14 St-Union Sq
});

test("Atlantic Av → Union Sq via the resolver keeps the direct corridors", () => {
  const idx = indexComplexes(SNAPSHOT);
  const js = journeysBetween(diagram.route_stops, diagram.adjacency, idx, "617", "602");
  const directs = js
    .filter((j) => j.transfers === 0)
    .map((j) => `${j.legs[0].route}|${j.legs[0].direction}`)
    .sort();
  assert.deepEqual(directs, ["4|north", "5|north", "N|north", "Q|north", "R|north", "W|north"]);
});
