import { test, beforeEach } from "node:test";
import assert from "node:assert/strict";
import {
  commuteStatus,
  loadCommutes,
  persistCommutes,
  addCommute,
  newCommuteId,
  type Commute,
  type CommuteLeg,
  type CommuteSegment,
} from "../lib/commutes.ts";
import type { Snapshot } from "../lib/types.ts";

// --- Builders --------------------------------------------------------------

const seg = (route: string, direction: string, from: string, to: string): CommuteSegment => ({
  route,
  direction,
  from,
  to,
  key: `${route}|${direction}|${from}`,
});

const leg = (segments: CommuteSegment[]): CommuteLeg => ({
  route: segments[0].route,
  direction: segments[0].direction,
  segments,
});

const commute = (legs: CommuteLeg[], name = "to work"): Commute => ({
  id: "c1",
  name,
  createdAt: 1_000,
  legs,
});

// A snapshot carrying only the two surfaces commuteStatus reads. Cast through
// unknown: the full Snapshot has ~30 unrelated fields the derivation never
// touches, and fabricating them would obscure what each case actually pins.
function snapshot(opts: {
  cells?: Record<string, { status: string; recovery_minutes?: number }>;
  alerts?: Record<string, { northbound?: string | null; southbound?: string | null }>;
}): Snapshot {
  const segments: Record<string, unknown> = {};
  for (const [key, v] of Object.entries(opts.cells ?? {})) {
    segments[key] = {
      status: v.status,
      recovery: v.recovery_minutes != null ? { recovery_minutes: v.recovery_minutes } : null,
    };
  }
  const route_status: Record<string, unknown> = {};
  for (const [route, dir] of Object.entries(opts.alerts ?? {})) {
    route_status[route] = {
      by_direction: {
        northbound: { alerts: dir.northbound ? [dir.northbound] : [], primary_alert_type: dir.northbound ?? null },
        southbound: { alerts: dir.southbound ? [dir.southbound] : [], primary_alert_type: dir.southbound ?? null },
      },
    };
  }
  return { segment_flow: { segments }, route_status } as unknown as Snapshot;
}

// --- Direction-aware scoping: the load-bearing guarantee -------------------

test("scoping is direction-aware — a southbound disruption never colours a northbound commute", () => {
  // Same route "1", same boarding stop id family, opposite directions. Only the
  // southbound cell is disrupted; the northbound commute must read clean.
  const snap = snapshot({
    cells: {
      "1|north|235N": { status: "normal" },
      "1|south|235S": { status: "disrupted", recovery_minutes: 12 },
    },
  });
  const nb = commuteStatus(snap, commute([leg([seg("1", "north", "235N", "228N")])]));
  assert.equal(nb.rollup, "normal");
  assert.equal(nb.disruptedCount, 0);

  const sb = commuteStatus(snap, commute([leg([seg("1", "south", "235S", "242S")])]));
  assert.equal(sb.rollup, "disrupted");
  assert.equal(sb.disruptedCount, 1);
  assert.equal(sb.worstRecoveryMinutes, 12);
});

// --- Rollup precedence + coverage ------------------------------------------

test("rollup: any disrupted cell dominates and reports worst recovery", () => {
  const snap = snapshot({
    cells: {
      "A|north|1N": { status: "normal" },
      "A|north|2N": { status: "disrupted", recovery_minutes: 8 },
      "A|north|3N": { status: "disrupted", recovery_minutes: 20 },
    },
  });
  const st = commuteStatus(
    snap,
    commute([leg([seg("A", "north", "1N", "2N"), seg("A", "north", "2N", "3N"), seg("A", "north", "3N", "4N")])]),
  );
  assert.equal(st.rollup, "disrupted");
  assert.equal(st.disruptedCount, 2);
  assert.equal(st.worstRecoveryMinutes, 20);
});

test("rollup: normal outranks quiet; unknown cells are counted, never healthy", () => {
  const snap = snapshot({
    cells: {
      "A|north|1N": { status: "normal" },
      "A|north|2N": { status: "quiet" },
      // 3N absent from the surface → not judged this tick.
    },
  });
  const st = commuteStatus(
    snap,
    commute([leg([seg("A", "north", "1N", "2N"), seg("A", "north", "2N", "3N"), seg("A", "north", "3N", "4N")])]),
  );
  assert.equal(st.rollup, "normal");
  assert.equal(st.total, 3);
  assert.equal(st.judged, 2);
  assert.equal(st.unknownCount, 1);
  // The unjudged cell reads null — not "normal" by omission.
  assert.equal(st.readings[2].status, null);
});

test("rollup: only-quiet reads quiet; nothing judged reads unknown", () => {
  const quiet = commuteStatus(
    snapshot({ cells: { "A|north|1N": { status: "quiet" } } }),
    commute([leg([seg("A", "north", "1N", "2N")])]),
  );
  assert.equal(quiet.rollup, "quiet");

  const blank = commuteStatus(
    snapshot({ cells: {} }),
    commute([leg([seg("A", "north", "1N", "2N")])]),
  );
  assert.equal(blank.rollup, "unknown");
  assert.equal(blank.judged, 0);
});

// --- Movement-vs-alert disagreement ----------------------------------------

test("disagreement: alert up but trains moving is flagged alert-only, direction-aware", () => {
  const snap = snapshot({
    cells: { "A|north|1N": { status: "normal" }, "A|south|9S": { status: "normal" } },
    // The advisory is on the SOUTHBOUND side only.
    alerts: { A: { southbound: "Delays" } },
  });
  const nb = commuteStatus(snap, commute([leg([seg("A", "north", "1N", "2N")])]));
  assert.equal(nb.disagreements.length, 0); // northbound is clear of the alert

  const sb = commuteStatus(snap, commute([leg([seg("A", "south", "9S", "8S")])]));
  assert.equal(sb.disagreements.length, 1);
  assert.equal(sb.readings[0].disagreement, "alert-only");
  assert.equal(sb.readings[0].alert, "Delays");
});

test("disagreement: movement disrupted with no advisory is flagged movement-only", () => {
  const snap = snapshot({
    cells: { "A|north|1N": { status: "disrupted", recovery_minutes: 5 } },
    alerts: { A: {} }, // route present, no alert either direction
  });
  const st = commuteStatus(snap, commute([leg([seg("A", "north", "1N", "2N")])]));
  assert.equal(st.readings[0].disagreement, "movement-only");
  assert.equal(st.disagreements.length, 1);
});

test("disagreement: 'No Scheduled Service' is benign — surfaced but never contradicts a normal read", () => {
  const snap = snapshot({
    cells: { "A|north|1N": { status: "normal" } },
    alerts: { A: { northbound: "No Scheduled Service" } },
  });
  const st = commuteStatus(snap, commute([leg([seg("A", "north", "1N", "2N")])]));
  assert.equal(st.readings[0].alert, "No Scheduled Service"); // still shown
  assert.equal(st.readings[0].disagreement, null); // but not a disagreement
});

// --- Persistence -----------------------------------------------------------

// Minimal in-memory localStorage so the SSR-guarded store can round-trip in the
// node test runner, which has no window.
class MemoryStorage {
  private map = new Map<string, string>();
  getItem(k: string): string | null {
    const v = this.map.get(k);
    return v === undefined ? null : v;
  }
  setItem(k: string, v: string): void {
    this.map.set(k, v);
  }
  removeItem(k: string): void {
    this.map.delete(k);
  }
  clear(): void {
    this.map.clear();
  }
}

let storage: MemoryStorage;

beforeEach(() => {
  storage = new MemoryStorage();
  // Install a window with just the localStorage the store touches. Object.assign
  // keeps this off an inline cast of globalThis; node's test env has no window.
  Object.assign(globalThis, { window: { localStorage: storage } });
});

test("persistence: commutes round-trip through localStorage", () => {
  const c = commute([leg([seg("A", "north", "1N", "2N")])], "home");
  addCommute(c);
  const loaded = loadCommutes();
  assert.equal(loaded.length, 1);
  assert.equal(loaded[0].name, "home");
  assert.equal(loaded[0].legs[0].segments[0].key, "A|north|1N");
});

test("persistence: a corrupt or wrong-shaped blob loads as empty, never throws", () => {
  storage.setItem("momentarily.commutes.v1", "{not json");
  assert.deepEqual(loadCommutes(), []);

  // Well-formed JSON, wrong shape: a commute whose leg has no segments is dropped.
  storage.setItem(
    "momentarily.commutes.v1",
    JSON.stringify([{ id: "x", name: "y", createdAt: 1, legs: [{ route: "A", direction: "north", segments: [] }] }]),
  );
  assert.deepEqual(loadCommutes(), []);
});

test("newCommuteId yields distinct ids", () => {
  assert.notEqual(newCommuteId(), newCommuteId());
});

test("persistCommutes replaces the whole store", () => {
  addCommute(commute([leg([seg("A", "north", "1N", "2N")])], "one"));
  persistCommutes([commute([leg([seg("B", "south", "5S", "6S")])], "two")]);
  const loaded = loadCommutes();
  assert.equal(loaded.length, 1);
  assert.equal(loaded[0].name, "two");
});
