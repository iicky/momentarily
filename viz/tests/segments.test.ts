// What the snapshot is allowed to say about a segment, and what it must refuse
// to say.
//
// `segment_flow.segments` carries every judged cell — normal and disrupted
// alike — as a full record, keyed `route|direction|from_stop`. The failure
// mode most of these tests exist to pin is reading "absent from segments" as
// "no reading": that would report every healthy cell we specifically
// measured as one we know nothing about, roughly 700 a tick.

import { test } from "node:test";
import assert from "node:assert/strict";
import { coverage, readEdge, selectReading, stationOf } from "../lib/segments.ts";
import type { Diagram, DiagramEdge } from "../lib/diagram.ts";
import type { SegmentFlow, SegmentStatus } from "../lib/types.ts";

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

/** A judged segment record. Defaults to a disrupted verdict; override
 * `status` (and anything else) for a normal one — `segments` carries both
 * alike now, so there is no separate fixture shape needed. */
function cell(over: Partial<SegmentStatus> = {}): SegmentStatus {
  return {
    route: "1",
    direction: "north",
    from_stop: "103N",
    to: "101N",
    status: "disrupted",
    entered_at: 100,
    recovery: null,
    ...over,
  };
}

function flow(segments: Record<string, SegmentStatus>): SegmentFlow {
  return { observed_at: 1000, segments };
}

// --- The collection ---------------------------------------------------------

test("a NORMAL cell in segments reads normal, not unmeasured", () => {
  // The regression this union shape must never reintroduce: a reader that
  // takes "absent from segments" as the only signal would report every
  // healthy cell — roughly 700 a tick — as one it knows nothing about.
  const reading = readEdge(edge(), flow({ "1|north|103N": cell({ status: "normal" }) }));
  assert.equal(reading.north.state, "normal");
  assert.equal(reading.north.cell?.status, "normal");
  assert.equal(reading.north.key, "1|north|103N");
});

test("a key in segments reads disrupted and carries its record", () => {
  const reading = readEdge(edge(), flow({ "1|north|103N": cell({ entered_at: 42 }) }));
  assert.equal(reading.north.state, "disrupted");
  assert.equal(reading.north.cell?.entered_at, 42);
});

test("a key absent from segments reads unmeasured, not normal", () => {
  const reading = readEdge(edge(), flow({}));
  assert.equal(reading.north.state, "unmeasured");
  assert.equal(reading.south.state, "unmeasured");
  // The key survives so the UI can name the cell it has no reading for.
  assert.equal(reading.north.key, "1|north|103N");
  assert.equal(reading.north.cell, null);
});

test("north and south read independently off the same collection", () => {
  const reading = readEdge(
    edge(),
    flow({
      "1|north|103N": cell(),
      "1|south|101S": cell({ status: "normal", direction: "south", from_stop: "101S", to: "103S" }),
    }),
  );
  assert.equal(reading.north.state, "disrupted");
  assert.equal(reading.south.state, "normal");
});

test("a missing segment surface reads unmeasured everywhere, never normal", () => {
  const reading = readEdge(edge(), null);
  assert.equal(reading.north.state, "unmeasured");
  assert.equal(reading.south.state, "unmeasured");
});

test("a direction the timetable doesn't schedule reads unscheduled", () => {
  const reading = readEdge(edge({ keys: { south: "1|south|101S" } }), flow({}));
  assert.equal(reading.north.state, "unscheduled");
  assert.equal(reading.north.key, null);
  assert.equal(reading.south.state, "unmeasured");
});

test("a record's presence in segments is what makes it judged, not any field value", () => {
  // `segments` is a Record, so presence has to be tested with `!== undefined`,
  // not a falsy check — a record whose own fields are all falsy-looking
  // (entered_at: 0, to: null) is still a real, judged reading.
  const present = readEdge(
    edge(),
    flow({ "1|north|103N": cell({ status: "normal", entered_at: 0, to: null }) }),
  );
  assert.equal(present.north.state, "normal");
  const absent = readEdge(edge(), flow({}));
  assert.equal(absent.north.state, "unmeasured");
});

// --- Attribution at a branch ----------------------------------------------

test("a disrupted reading lands only on the successor the record names", () => {
  // 101 splits: the 1 runs 101 -> 103 (local) and 101 -> 199 (express). Both
  // edges carry the same key, because a cell is keyed on its from_stop alone.
  const local = edge({ a: "101", b: "103", keys: { north: "1|north|101N" } });
  const express = edge({ a: "101", b: "199", keys: { north: "1|north|101N" } });
  const measured = flow({
    "1|north|101N": cell({ from_stop: "101N", to: "103N" }),
  });
  assert.equal(readEdge(local, measured).north.state, "disrupted");
  // The express leg is not what was measured — spreading the verdict onto it
  // would claim a reading for track the measurement never covered.
  const sibling = readEdge(express, measured).north;
  assert.equal(sibling.state, "unmeasured");
  assert.equal(sibling.cell, null);
  assert.equal(sibling.key, "1|north|101N");
});

test("a normal verdict lands only on the successor it names", () => {
  // The same rule as a disrupted record: a healthy verdict has to name its
  // successor too, or it would have to be spread across every sibling leg,
  // painting unmeasured track as healthy.
  const local = edge({ a: "101", b: "103", keys: { north: "1|north|101N" } });
  const express = edge({ a: "101", b: "199", keys: { north: "1|north|101N" } });
  const measured = flow({
    "1|north|101N": cell({ status: "normal", from_stop: "101N", to: "103N" }),
  });
  assert.equal(readEdge(local, measured).north.state, "normal");
  assert.equal(readEdge(express, measured).north.state, "unmeasured");
});

test("a disrupted record with no successor still reads on its from_stop's edges", () => {
  // `to` is null when the Worker couldn't read the topology doc. The reading is
  // real and there is no better attribution available, so it isn't discarded.
  const measured = flow({ "1|north|101N": cell({ from_stop: "101N", to: null }) });
  const branch = edge({ a: "101", b: "199", keys: { north: "1|north|101N" } });
  assert.equal(readEdge(branch, measured).north.state, "disrupted");
});

test("a normal verdict with no published successor falls back the same way", () => {
  // Same condition, same fallback: null means the topology doc was missing, not
  // that the record is untrustworthy.
  const measured = flow({
    "1|north|101N": cell({ status: "normal", from_stop: "101N", to: null }),
  });
  const branch = edge({ a: "101", b: "199", keys: { north: "1|north|101N" } });
  assert.equal(readEdge(branch, measured).north.state, "normal");
});

// --- Direction selection ----------------------------------------------------

test("the direction filter selects that direction's reading", () => {
  const reading = readEdge(edge(), flow({ "1|north|103N": cell() }));
  assert.equal(selectReading(reading, "north").state, "disrupted");
  assert.equal(selectReading(reading, "south").state, "unmeasured");
});

test("both-directions takes the disruption over the healthy direction", () => {
  const reading = readEdge(
    edge(),
    flow({
      "1|north|103N": cell(),
      "1|south|101S": cell({ status: "normal", direction: "south", from_stop: "101S", to: "103S" }),
    }),
  );
  const selected = selectReading(reading, "both");
  assert.equal(selected.state, "disrupted");
  assert.equal(selected.direction, "north");
});

test("both-directions refuses to call a half-measured pair healthy", () => {
  const reading = readEdge(edge(), flow({}));
  assert.equal(selectReading(reading, "both").state, "unmeasured");
});

test("both-directions ignores the unscheduled side of a one-way pair", () => {
  const reading = readEdge(
    edge({ keys: { south: "1|south|101S" } }),
    flow({ "1|south|101S": cell({ status: "normal", direction: "south", from_stop: "101S", to: "103S" }) }),
  );
  assert.equal(selectReading(reading, "both").state, "normal");
});

// --- Coverage --------------------------------------------------------------

function diagramOf(edges: DiagramEdge[]): Diagram {
  return {
    feed_version: { version: "test", start: null, end: null },
    view_box: [0, 0, 10, 10],
    routes: { "1": { name: "test", color: "#fff" } },
    stations: {},
    edges,
    insets: [],
  };
}

test("coverage counts scheduled cells, not drawn edges", () => {
  const diagram = diagramOf([edge(), edge({ a: "103", b: "104", keys: {} })]);
  const cover = coverage(diagram, flow({}));
  assert.equal(cover.scheduled, 2);
  assert.equal(cover.measured, 0);
  assert.equal(cover.disrupted, 0);
});

test("the coverage numerator counts every judged cell, normal and disrupted alike", () => {
  // Counting by collection membership would have said two disrupted; counting
  // the published `status` field is what keeps this honest.
  const cover = coverage(
    diagramOf([edge()]),
    flow({
      "1|north|103N": cell(),
      "1|south|101S": cell({ status: "normal", direction: "south", from_stop: "101S", to: "103S" }),
    }),
  );
  assert.equal(cover.scheduled, 2);
  assert.equal(cover.measured, 2);
  assert.equal(cover.disrupted, 1);
  assert.equal(cover.unplaced, 0);
});

test("coverage reports published readings the diagram can't place", () => {
  const cover = coverage(
    diagramOf([edge()]),
    flow({
      "1|north|103N": cell(),
      "1|south|101S": cell({ status: "normal", direction: "south", from_stop: "101S", to: "103S" }),
      // A reroute the static timetable doesn't schedule: a real reading with
      // nowhere to draw it, which has to be reported rather than dropped.
      "7|north|701N": cell({
        status: "normal",
        route: "7",
        direction: "north",
        from_stop: "701N",
        to: "702N",
      }),
    }),
  );
  assert.equal(cover.scheduled, 2);
  assert.equal(cover.measured, 2);
  assert.equal(cover.disrupted, 1);
  assert.equal(cover.unplaced, 1);
});

test("a branch counts once in the denominator and once in the numerator", () => {
  // The unit trap: this key belongs to two drawn edges, so counting (edge,
  // direction) slots would report 2 scheduled against 1 measured and claim 50%
  // coverage of a cell that is fully measured.
  const cover = coverage(
    diagramOf([
      edge({ a: "101", b: "103", keys: { north: "1|north|101N" } }),
      edge({ a: "101", b: "199", keys: { north: "1|north|101N" } }),
    ]),
    flow({ "1|north|101N": cell({ status: "normal", from_stop: "101N", to: "103N" }) }),
  );
  assert.equal(cover.scheduled, 1);
  assert.equal(cover.measured, 1);
  assert.equal(cover.unplaced, 0);
});

// --- Key handling ------------------------------------------------------------

test("stationOf strips only a directional suffix", () => {
  assert.equal(stationOf("A24S"), "A24");
  assert.equal(stationOf("R30N"), "R30");
  assert.equal(stationOf("101"), "101");
  // Not every id ends in a direction letter — a bare station id is unchanged.
  assert.equal(stationOf("H01"), "H01");
});
