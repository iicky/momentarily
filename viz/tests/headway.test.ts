import { test } from "node:test";
import assert from "node:assert/strict";
import {
  fmtGap,
  headwayFor,
  headwayHeadline,
  readHeadway,
  THIN_N_TRIPS,
} from "../lib/headway.ts";
import type { Observation, Snapshot } from "../lib/types.ts";

function obs(o: Partial<Observation>): Observation {
  return {
    entity_ref: "subway_route:1",
    kind: "headway",
    value: 540,
    unit: "seconds",
    observed_at: 1_700_000_000,
    source: "gtfs_rt_vehicle_positions",
    direction: "north",
    stop_id: "121N",
    window: [{ value: 540, observed_at: 1_700_000_000 }],
    scheduled: null,
    off_reference: false,
    ...o,
  };
}

function snapWith(observations: Observation[]): Snapshot {
  return { observations } as unknown as Snapshot;
}

test("headwayFor matches on route, direction and kind", () => {
  const north = obs({ direction: "north", value: 540 });
  const south = obs({ direction: "south", value: 300 });
  const other = obs({ entity_ref: "subway_route:2", direction: "north" });
  const snap = snapWith([south, north, other]);
  assert.equal(headwayFor(snap, "1", "north"), north);
  assert.equal(headwayFor(snap, "1", "south"), south);
  assert.equal(headwayFor(snap, "9", "north"), null);
});

test("fmtGap speaks minutes above a minute and seconds below", () => {
  assert.equal(fmtGap(540), "9 min");
  assert.equal(fmtGap(360), "6 min");
  assert.equal(fmtGap(150), "3 min"); // 2.5 min rounds up
  assert.equal(fmtGap(45), "45 sec");
  assert.equal(fmtGap(12), "10 sec"); // rounds to nearest 5
});

test("a reading longer than scheduled reads as longer gaps than usual", () => {
  const read = readHeadway(obs({ value: 540, scheduled: { median_headway_s: 360, n_trips: 10 } }));
  assert.equal(read.observedOnly, null);
  assert.ok(read.scheduled);
  assert.equal(read.scheduled.tone, "gapped");
  assert.equal(read.scheduled.thin, false);
  assert.equal(headwayHeadline(read), "Trains every 9 min, scheduled 6 min — longer gaps than usual");
});

test("a reading close to scheduled reads as about on schedule", () => {
  const read = readHeadway(obs({ value: 380, scheduled: { median_headway_s: 360, n_trips: 10 } }));
  assert.equal(read.scheduled?.tone, "onschedule");
  assert.equal(headwayHeadline(read), "Trains every 6 min, scheduled 6 min — about on schedule");
});

test("a reading well under scheduled reads as running closer than usual", () => {
  const read = readHeadway(obs({ value: 240, scheduled: { median_headway_s: 360, n_trips: 10 } }));
  assert.equal(read.scheduled?.tone, "bunched");
  assert.match(headwayHeadline(read), /running closer than usual$/);
});

test("a thin timetable cell is flagged but still carries its median", () => {
  const read = readHeadway(
    obs({ value: 900, scheduled: { median_headway_s: 1200, n_trips: THIN_N_TRIPS } }),
  );
  assert.ok(read.scheduled);
  assert.equal(read.scheduled.thin, true);
  assert.equal(read.scheduled.seconds, 1200);
});

test("no scheduled baseline shows observed alone, never a fabricated ratio", () => {
  const read = readHeadway(obs({ value: 540, scheduled: null, off_reference: false }));
  assert.equal(read.scheduled, null);
  assert.equal(read.observedOnly, "none");
  assert.equal(headwayHeadline(read), "Trains every 9 min · no scheduled baseline for this hour");
});

test("an off-reference reroute reading is labelled, not compared", () => {
  // off_reference wins even if a scheduled cell is somehow present: the worker
  // withholds it, and the read must not compare across stops.
  const read = readHeadway(
    obs({ value: 540, off_reference: true, scheduled: { median_headway_s: 360, n_trips: 10 } }),
  );
  assert.equal(read.observedOnly, "off_reference");
  assert.equal(read.scheduled, null);
  assert.equal(headwayHeadline(read), "Trains every 9 min · measured at a reroute stop");
});

test("the window is carried through as the strip series, oldest first", () => {
  const read = readHeadway(
    obs({
      value: 240,
      window: [
        { value: 600, observed_at: 1 },
        { value: 420, observed_at: 2 },
        { value: 240, observed_at: 3 },
      ],
    }),
  );
  assert.deepEqual(read.window, [600, 420, 240]);
});
