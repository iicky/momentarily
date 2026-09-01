import { test } from "node:test";
import assert from "node:assert/strict";
import { heldFor } from "../lib/feed.ts";

// heldFor renders route_status.condition_entered_at as an age the drawer shows
// beside the badge ("<state> for X") — the movement regime's own clock, not the
// model's argmax clock. Only the sub-minute floor differs from fmtMinutes.
test("heldFor: sub-minute reads 'under a minute', not fmtMinutes' dash", () => {
  assert.equal(heldFor(0), "under a minute");
  assert.equal(heldFor(59), "under a minute");
  // Clock ahead of generated_at (skew) is a floor case too, never a dash.
  assert.equal(heldFor(-30), "under a minute");
});

test("heldFor: a minute or more rounds to fmtMinutes", () => {
  assert.equal(heldFor(60), "1m");
  assert.equal(heldFor(3600), "1h");
  assert.equal(heldFor(3600 + 1800), "1h 30m");
});
