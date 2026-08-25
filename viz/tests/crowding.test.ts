import { test } from "node:test";
import assert from "node:assert/strict";
import {
  fmtRiders,
  platformCrowding,
  CROWD_TYPICAL_RIDERS,
  CROWD_HEAVY_RIDERS,
  CROWD_EXTREME_RIDERS,
} from "../lib/feed.ts";
import type { PlatformCrowding } from "../lib/types.ts";

const OBSERVED = 1_787_500_000;

/** A surface with one platform, whose last train cleared `gapMin` before the
 * publisher observed it. `waiting_riders` is the publisher's own arithmetic at
 * that instant — the number a naive renderer would print. */
function surface(
  entriesPerMin: number,
  gapMin: number,
  maxGapMinutes = 30,
): PlatformCrowding {
  const lastTrainAt = OBSERVED - gapMin * 60;
  return {
    observed_at: OBSERVED,
    method: {
      formula: "entries_per_min * minutes_since_last_train",
      split_basis: "uniform_over_served_platforms",
      max_gap_minutes: maxGapMinutes,
      served_window_minutes: 60,
      excludes: ["in-system transfers", "exits"],
      baseline_generated_at: OBSERVED - 86_400,
      baseline_window_start: "2026-05-01T00:00:00",
      baseline_window_end: "2026-07-29T00:00:00",
    },
    platforms: {
      "127N": {
        last_train_at: lastTrainAt,
        entries_per_min: entriesPerMin,
        waiting_riders: Math.round(entriesPerMin * gapMin),
      },
    },
    n_platforms: 1,
    abstained: {},
  };
}

test("platformCrowding re-derives against the client clock, not the published number", () => {
  // Published at a 4-minute gap: 40 riders. The reader sees it 3 minutes later,
  // by which time the crowd is 70 — the staleness the recompute exists to fix.
  const pc = surface(10, 4);
  assert.equal(pc.platforms["127N"].waiting_riders, 40);
  const v = platformCrowding(pc, "127N", OBSERVED + 180);
  assert.equal(v.estimated, true);
  assert.equal(v.estimated && v.riders, 70);
  assert.equal(v.estimated && v.minutesSince, 7);
  assert.equal(v.estimated && v.entriesPerMin, 10);
});

test("platformCrowding abstains once OUR clock passes the publisher's cap", () => {
  // Published inside the cap at 29 minutes; two more minutes of polling carry it
  // past 30, where the worker itself would have refused to estimate.
  const pc = surface(10, 29);
  const inside = platformCrowding(pc, "127N", OBSERVED);
  assert.equal(inside.estimated, true);

  const past = platformCrowding(pc, "127N", OBSERVED + 120);
  assert.equal(past.estimated, false);
  assert.equal(!past.estimated && past.reason, "gap_exceeds_cap");
  assert.equal(!past.estimated && Math.round(past.minutesSince ?? 0), 31);
  assert.ok(!("riders" in past));
});

test("platformCrowding takes the cap from the surface, not a local constant", () => {
  // A publisher that widened its own cap is followed, not second-guessed.
  const pc = surface(10, 40, 45);
  const v = platformCrowding(pc, "127N", OBSERVED);
  assert.equal(v.estimated, true);
  assert.equal(v.estimated && v.riders, 400);
  // Exactly at the cap still estimates; the worker abstains only beyond it.
  assert.equal(platformCrowding(surface(10, 45, 45), "127N", OBSERVED).estimated, true);
});

test("platformCrowding abstains rather than reporting an empty platform", () => {
  const pc = surface(10, 4);
  const missing = platformCrowding(pc, "127S", OBSERVED);
  assert.equal(missing.estimated, false);
  assert.equal(!missing.estimated && missing.reason, "unpublished");

  const none = platformCrowding(null, "127N", OBSERVED);
  assert.equal(none.estimated, false);
  assert.equal(!none.estimated && none.reason, "no_surface");
});

test("platformCrowding clamps a client clock running behind the publisher", () => {
  // Skew must not subtract riders off the front of the crowd.
  const pc = surface(10, 4);
  const v = platformCrowding(pc, "127N", OBSERVED - 600);
  assert.equal(v.estimated, true);
  assert.equal(v.estimated && v.riders, 0);
  assert.equal(v.estimated && v.minutesSince, 0);
});

test("platformCrowding bands on the measured quantiles of the estimate", () => {
  // Hold the gap at 10 minutes — well inside the cap — and vary the rate, so
  // every band is reachable: p90 and p99 are more riders than 30 minutes of a
  // one-a-minute platform can ever produce.
  const band = (riders: number) => {
    const v = platformCrowding(surface(riders / 10, 10), "127N", OBSERVED);
    return v.estimated ? v.band : v.reason;
  };
  assert.equal(band(0), "light");
  assert.equal(band(CROWD_TYPICAL_RIDERS - 1), "light");
  assert.equal(band(CROWD_TYPICAL_RIDERS), "typical");
  assert.equal(band(CROWD_HEAVY_RIDERS - 1), "typical");
  assert.equal(band(CROWD_HEAVY_RIDERS), "heavy");
  assert.equal(band(CROWD_EXTREME_RIDERS - 1), "heavy");
  assert.equal(band(CROWD_EXTREME_RIDERS), "extreme");
  assert.equal(band(1302), "extreme"); // the measured maximum
});

test("fmtRiders marks the figure as modelled and counts people", () => {
  assert.equal(fmtRiders(86), "~86 riders");
  assert.equal(fmtRiders(1), "~1 rider");
  assert.equal(fmtRiders(0), "~0 riders");
});
