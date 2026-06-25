import { test } from "node:test";
import assert from "node:assert/strict";
import {
  calibrationReliability,
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
  assert.equal(r.n, 42);
  assert.equal(r.brier, 0.12);
  assert.equal(r.excludedSchedule, 0);
  assert.deepEqual(r.bins[0], {
    p: 0.05,
    predictedMean: 0.05,
    observedFreq: 0.0,
    n: 3,
  });
  assert.equal(r.bins[1].p, 0.95);
});

test("calibrationReliability threads skill scores and the state decomposition", () => {
  const d = doc();
  d.calibration[0].excluded_schedule = 9;
  d.calibration[0].by_current = {
    normal_now: { n: 30, brier: 0.02, bss_persistence: 0.6 },
    not_normal_now: { n: 12, brier: 0.3, bss_persistence: -0.45 },
  };
  const [r] = calibrationReliability(d);
  assert.equal(r.skillPersistence, 0.4);
  assert.equal(r.skillClimatology, 0.52);
  assert.equal(r.excludedSchedule, 9);
  assert.deepEqual(r.decomp?.normalNow, { n: 30, bss: 0.6 });
  assert.deepEqual(r.decomp?.notNormalNow, { n: 12, bss: -0.45 });
});

test("calibrationReliability leaves decomp undefined when the feed omits it", () => {
  const [r] = calibrationReliability(doc());
  assert.equal(r.decomp, undefined);
  assert.equal(r.skillPersistence, 0.4);
});

test("calibrationReliability null brier/means become NaN", () => {
  const d = doc();
  d.calibration[0].brier = null;
  d.calibration[0].bins[0].mean_pred = null;
  const [r] = calibrationReliability(d);
  assert.ok(Number.isNaN(r.brier));
  assert.ok(Number.isNaN(r.bins[0].predictedMean));
});

test("calibrationHeatmap keeps only 3x3 matrices, sorted naturally", () => {
  const h = calibrationHeatmap(doc());
  assert.equal(h.length, 1);
  assert.equal(h[0].route, "1");
  assert.equal(h[0].transition.length, 3);
});
