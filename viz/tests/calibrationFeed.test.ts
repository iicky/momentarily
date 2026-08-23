import { test } from "node:test";
import assert from "node:assert/strict";
import {
  calibrationReliability,
  calibrationForLine,
  calibrationRoutes,
  calibrationHeatmap,
  type CalibrationDoc,
} from "../lib/calibrationFeed.ts";

function doc(over: Partial<CalibrationDoc> = {}): CalibrationDoc {
  return {
    generated_at: 1_700_000_000,
    window: { start: 1_699_000_000, end: 1_700_000_000 },
    predictions_seen: 100,
    transitions_seen: 5,
    calibration: [
      {
        horizon_min: 30,
        n: 42,
        brier: 0.12,
        brier_persistence: 0.2,
        brier_climatology: 0.25,
        bss_persistence: 0.4,
        bss_climatology: 0.52,
        bins: [
          { bin_lo: 0.0, bin_hi: 0.1, n: 3, mean_pred: 0.05, mean_outcome: 0.0 },
          { bin_lo: 0.9, bin_hi: 1.0, n: 7, mean_pred: 0.95, mean_outcome: 1.0 },
        ],
      },
    ],
    recovery: {
      overall: { n: 10, mae_min: 22, rmse_min: 30, iqr_coverage: 0.48 },
      per_regime: { n: 3, mae_min: 18, rmse_min: 25, iqr_coverage: 0.5 },
    },
    transition_matrices: {
      trained_at: 1_699_999_000,
      states: ["normal", "disrupted", "suspended"],
      routes: {
        "1": [
          [0.9, 0.1, 0.0],
          [0.2, 0.7, 0.1],
          [0.0, 0.3, 0.7],
        ],
        // Malformed (not 3x3) — must be dropped, not rendered.
        bad: [[1, 0]],
      },
    },
    ...over,
  };
}

test("calibrationReliability maps bins to midpoint/predicted/observed", () => {
  const [r] = calibrationReliability(doc());
  assert.equal(r.horizonMin, 30);
  assert.equal(r.excludedSchedule, 0);
  // No movement block in the fixture, so the shadow grading is the only arm.
  assert.equal(r.arms.length, 1);
  const [shadow] = r.arms;
  assert.equal(shadow.isForecastTarget, false);
  assert.equal(shadow.n, 42);
  assert.equal(shadow.brier, 0.12);
  assert.deepEqual(shadow.bins[0], {
    p: 0.05,
    predictedMean: 0.05,
    observedFreq: 0.0,
    n: 3,
  });
  assert.equal(shadow.bins[1].p, 0.95);
});

test("calibrationReliability threads skill scores and the state decomposition", () => {
  const d = doc();
  d.calibration[0].excluded_schedule = 9;
  d.calibration[0].auc = 0.41;
  d.calibration[0].by_current = {
    normal_now: {
      n: 30,
      brier: 0.02,
      bss_persistence: 0.6,
      mean_pred: 0.99,
      mean_outcome: 0.98,
      auc: 0.4,
    },
    not_normal_now: {
      n: 12,
      brier: 0.3,
      bss_persistence: -0.45,
      mean_pred: 0.99,
      mean_outcome: 0.5,
      auc: 0.47,
    },
  };
  const [r] = calibrationReliability(d);
  assert.equal(r.excludedSchedule, 9);
  const [shadow] = r.arms;
  assert.equal(shadow.skillPersistence, 0.4);
  assert.equal(shadow.skillClimatology, 0.52);
  assert.equal(shadow.auc, 0.41);
  assert.deepEqual(shadow.decomp?.normalNow, {
    n: 30,
    bss: 0.6,
    meanPred: 0.99,
    meanOutcome: 0.98,
  });
  // The sharpness/realized pair is the degenerate-forecast tell; it must survive
  // the reshape, since only this gap distinguishes "confident" from "constant".
  assert.deepEqual(shadow.decomp?.notNormalNow, {
    n: 12,
    bss: -0.45,
    meanPred: 0.99,
    meanOutcome: 0.5,
  });
});

test("calibrationReliability leaves decomp undefined when the feed omits it", () => {
  const [r] = calibrationReliability(doc());
  assert.equal(r.arms[0].decomp, undefined);
  assert.equal(r.arms[0].skillPersistence, 0.4);
});

test("a feed with no auc reports null, never a fabricated 0.5", () => {
  const [r] = calibrationReliability(doc());
  assert.equal(r.arms[0].auc, null);
});

test("a stratum missing the sharpness fields reports null, not undefined", () => {
  const d = doc();
  d.calibration[0].by_current = {
    normal_now: { n: 30, brier: 0.02, bss_persistence: 0.6 },
  };
  const [r] = calibrationReliability(d);
  assert.deepEqual(r.arms[0].decomp?.normalNow, {
    n: 30,
    bss: 0.6,
    meanPred: null,
    meanOutcome: null,
  });
});

test("the movement grading leads and is flagged as the forecast's target", () => {
  const d = doc();
  d.calibration_arm = "condition (alert-shadow)";
  d.calibration_movement = {
    graded_arm: "published_condition (movement-primary)",
    coverage: {
      n_ticks: 100,
      unknown_share: 0.25,
      gradeable_share: 0.67,
      by_condition: { normal: 60, unknown: 25 },
    },
    horizons: [
      {
        horizon_min: 30,
        n: 40,
        brier: 0.001,
        brier_persistence: 0.002,
        brier_climatology: 0.0009,
        bss_persistence: 0.04,
        bss_climatology: -0.11,
        auc: 0.96,
        bins: [
          { bin_lo: 0.9, bin_hi: 1.0, n: 40, mean_pred: 0.99, mean_outcome: 0.999 },
        ],
      },
    ],
  };
  const [r] = calibrationReliability(d);
  assert.equal(r.arms.length, 2);
  // Movement first: it is what p_normal_in_H forecasts, so it is read first even
  // though it is the thinner sample.
  const [movement, shadow] = r.arms;
  assert.equal(movement.isForecastTarget, true);
  assert.equal(movement.arm, "published_condition (movement-primary)");
  assert.equal(movement.auc, 0.96);
  assert.equal(movement.n, 40);
  // Coverage rides along so the panel can say how much of the window this arm
  // could not judge — 25% unjudgeable is why its n is a slice, not the window.
  assert.equal(movement.unknownShare, 0.25);
  assert.equal(shadow.isForecastTarget, false);
  assert.equal(shadow.arm, "condition (alert-shadow)");
  assert.equal(shadow.auc, null); // fixture omits auc on the shadow block
});

test("a horizon the movement block never graded still renders its shadow arm", () => {
  const d = doc();
  d.calibration_movement = {
    graded_arm: "published_condition (movement-primary)",
    horizons: [],
  };
  const [r] = calibrationReliability(d);
  assert.equal(r.arms.length, 1);
  assert.equal(r.arms[0].isForecastTarget, false);
});

test("calibrationReliability null brier/means become NaN", () => {
  const d = doc();
  d.calibration[0].brier = null;
  d.calibration[0].bins[0].mean_pred = null;
  const [r] = calibrationReliability(d);
  assert.ok(Number.isNaN(r.arms[0].brier));
  assert.ok(Number.isNaN(r.arms[0].bins[0].predictedMean));
});

/** Movement block with its own coverage, used to build per-line fixtures. */
function movementBlock(unknownShare: number) {
  return {
    graded_arm: "published_condition (movement-primary)",
    coverage: {
      n_ticks: 100,
      unknown_share: unknownShare,
      gradeable_share: 1 - unknownShare,
      by_condition: { normal: 60 },
    },
    horizons: [
      {
        horizon_min: 30,
        n: 40,
        brier: 0.001,
        brier_persistence: 0.002,
        brier_climatology: 0.0009,
        bss_persistence: 0.04,
        bss_climatology: -0.11,
        auc: 0.96,
        bins: [
          { bin_lo: 0.9, bin_hi: 1.0, n: 40, mean_pred: 0.99, mean_outcome: 0.999 },
        ],
      },
    ],
  };
}

test("the line view reports THIS route's coverage, not the window aggregate", () => {
  // The aggregate rate can be true of no individual route — a shuttle movement
  // can never judge and a trunk line it always can average to a number that
  // describes neither. Passing the aggregate down made the route view claim a
  // coverage figure that was not about that route.
  const d = doc();
  d.calibration_movement = movementBlock(0.25);
  d.by_line = {
    "1": {
      n_predictions: 1200,
      calibration: d.calibration,
      calibration_movement: d.calibration_movement.horizons,
      movement_coverage: {
        n_ticks: 40,
        unknown_share: 0.02,
        gradeable_share: 0.98,
        by_condition: { normal: 39 },
      },
      recovery: d.recovery,
    },
  };
  const line = calibrationForLine(d, "1")!;
  const movement = line.reliability[0].arms.find((a) => a.isForecastTarget)!;
  assert.equal(movement.unknownShare, 0.02);
  // And the window aggregate still reports its own rate.
  const agg = calibrationReliability(d)[0].arms.find((a) => a.isForecastTarget)!;
  assert.equal(agg.unknownShare, 0.25);
});

test("the line view reports THIS route's incident support, not the aggregate", () => {
  // The caller swaps the whole counts line to the selected route, so leaking the
  // system-wide incident count through here claims those incidents back one line.
  const d = doc();
  d.episode_support = {
    graded_arm: "published_condition (movement-primary)",
    n_episodes: 26,
    n_left_censored: 0,
    n_right_censored: 0,
    n_standing: 0,
    standing_tick_share: 0,
    tick_rows: 58493,
  };
  d.by_line = {
    "1": {
      n_predictions: 1200,
      calibration: d.calibration,
      recovery: d.recovery,
      episode_support: { ...d.episode_support, n_episodes: 2, tick_rows: 1200 },
    },
    // No per-route support published: must report nothing, not the aggregate's 26.
    "7": { n_predictions: 900, calibration: d.calibration, recovery: d.recovery },
  };
  assert.equal(calibrationForLine(d, "1")!.episodeSupport?.n_episodes, 2);
  assert.equal(calibrationForLine(d, "7")!.episodeSupport, undefined);
  // The aggregate block is untouched by any of this.
  assert.equal(d.episode_support.n_episodes, 26);
});

test("a route with no per-route coverage shows no coverage claim at all", () => {
  const d = doc();
  d.calibration_movement = movementBlock(0.25);
  d.by_line = {
    "1": {
      n_predictions: 1200,
      calibration: d.calibration,
      calibration_movement: d.calibration_movement.horizons,
      recovery: d.recovery,
    },
  };
  const line = calibrationForLine(d, "1")!;
  const movement = line.reliability[0].arms.find((a) => a.isForecastTarget)!;
  // Undefined suppresses the chip. Falling back to the aggregate would be a
  // false route-specific claim; falling back to 0 would claim full coverage.
  assert.equal(movement.unknownShare, undefined);
});

test("a route the publisher never movement-graded renders shadow-only", () => {
  const d = doc();
  d.calibration_movement = movementBlock(0.25);
  d.by_line = {
    "7": { n_predictions: 900, calibration: d.calibration, recovery: d.recovery },
  };
  const line = calibrationForLine(d, "7")!;
  assert.equal(line.reliability[0].arms.length, 1);
  assert.equal(line.reliability[0].arms[0].isForecastTarget, false);
});

test("calibrationForLine returns null for a route the feed has no breakdown for", () => {
  assert.equal(calibrationForLine(doc(), "nope"), null);
});

test("calibrationRoutes sorts numerically, not lexically", () => {
  const d = doc();
  const entry = {
    n_predictions: 60,
    calibration: d.calibration,
    recovery: d.recovery,
  };
  d.by_line = { "10": entry, "2": entry, A: entry };
  assert.deepEqual(calibrationRoutes(d), ["2", "10", "A"]);
});

test("calibrationHeatmap keeps only 3x3 matrices, sorted naturally", () => {
  const h = calibrationHeatmap(doc());
  assert.equal(h.length, 1);
  assert.equal(h[0].route, "1");
  assert.equal(h[0].transition.length, 3);
});
