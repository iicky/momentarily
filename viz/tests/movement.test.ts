import { test } from "node:test";
import assert from "node:assert/strict";
import {
  movementConfusion,
  snapTick,
  type MovementState,
} from "../lib/movement.ts";

test("movementConfusion ranks disagreements and classifies them", () => {
  const preds = [
    { route: "A", ts: 300, condition: "normal" }, // movement says disrupted
    { route: "A", ts: 600, condition: "normal" }, // movement says disrupted again
    { route: "B", ts: 300, condition: "suspended" }, // movement says trains fine
    { route: "C", ts: 300, condition: "suspended" }, // no movement read at all
  ];
  const truth = new Map<string, MovementState>([
    [`A|${snapTick(300)}`, "disrupted"],
    [`A|${snapTick(600)}`, "disrupted"],
    [`B|${snapTick(300)}`, "normal"],
  ]);

  const r = movementConfusion(preds, truth);

  // C had a known condition but no movement read → unjudged, and it was suspended.
  assert.equal(r.coverage.judged, 3);
  assert.equal(r.coverage.unjudged, 1);
  assert.equal(r.coverage.suspendedUnjudged, 1);

  // Two distinct disagreement cells, ranked by count (A's two first).
  assert.equal(r.disagreements.length, 2);
  const [top] = r.disagreements;
  assert.equal(top.route, "A");
  assert.equal(top.count, 2);
  assert.equal(top.kind, "false-normal");
  assert.equal(top.hmm, "normal");
  assert.equal(top.move, "disrupted");

  const b = r.disagreements.find((d) => d.route === "B");
  assert.equal(b?.kind, "false-disrupted");
  assert.equal(b?.rate, 1); // B's only judged tick disagreed
});
