import { test } from "node:test";
import assert from "node:assert/strict";
import {
  buildTimelines,
  reliability,
  recoveryError,
} from "../lib/calibration.ts";
import type { PredictionRecord, TransitionRecord } from "../lib/types.ts";

function pred(p: Partial<PredictionRecord>): PredictionRecord {
  return {
    ts: 0,
    route: "L",
    condition: "suspended",
    p_normal: 0,
    p_disrupted: 0,
    p_suspended: 1,
    regime_entered_at: 0,
    recovery_minutes: 40,
    recovery_minutes_low: 20,
    recovery_minutes_high: 50,
    recovery_indeterminate: false,
    p_normal_in_30min: 0,
    p_normal_in_60min: 0,
    p_normal_in_120min: 0,
    primary_alert_type: "Suspended",
    params_version: 1,
    ...p,
  };
}

// Timeline: normal [1000,2000), suspended [2000,5000), normal [5000, observedUntil)
const transitions: TransitionRecord[] = [
  {
    ts: 2000,
    route: "L",
    prev_state: "normal",
    new_state: "suspended",
    regime_entered_at: 1000,
    exited_at: 2000,
    dwell_sec: 1000,
    alert_type_at_entry: null,
  },
  {
    ts: 5000,
    route: "L",
    prev_state: "suspended",
    new_state: "normal",
    regime_entered_at: 2000,
    exited_at: 5000,
    dwell_sec: 3000,
    alert_type_at_entry: "Suspended",
  },
];

const NOW = 9000;

test("buildTimelines reconstructs segments and normal starts", () => {
  const tls = buildTimelines(transitions, NOW);
  const tl = tls.get("L")!;
  assert.equal(tl.observedUntil, 9000);
  assert.deepEqual(tl.normalStarts, [1000, 5000]);
  assert.equal(tl.segments.length, 3);
  assert.deepEqual(
    tl.segments.map((s) => s.state),
    ["normal", "suspended", "normal"],
  );
  assert.deepEqual(tl.segments[2], { state: "normal", start: 5000, end: 9000 });
});

test("reliability scores forecasts against real recovery; censors the unobservable", () => {
  const tls = buildTimelines(transitions, NOW);
  // During suspended at ts=2300; next normal at 5000 → recovers in 2700s (45m).
  const predictions = [
    pred({ ts: 2300, p_normal_in_30min: 0.2, p_normal_in_60min: 0.7 }),
    // normal-state prediction must be ignored entirely
    pred({ ts: 1500, condition: "normal", p_normal_in_30min: 0.9 }),
    // censored: ts+30m exceeds observedUntil(9000)
    pred({ ts: 8000, p_normal_in_30min: 0.5 }),
  ];

  const r30 = reliability(predictions, tls, 30);
  assert.equal(r30.n, 1); // only the ts=2300 point is observable & non-normal
  assert.equal(r30.excludedSchedule, 0);
  // recovered in 45m > 30m → y=0; p=0.2 → brier=0.04
  assert.ok(Math.abs(r30.brier - 0.04) < 1e-9);
  const bin30 = r30.bins.find((b) => b.n > 0)!;
  assert.equal(bin30.observedFreq, 0);

  const r60 = reliability(predictions, tls, 60);
  assert.equal(r60.n, 1);
  // recovered in 45m <= 60m → y=1; p=0.7 → brier=0.09
  assert.ok(Math.abs(r60.brier - 0.09) < 1e-9);
  assert.equal(r60.bins.find((b) => b.n > 0)!.observedFreq, 1);
});

test("recoveryError compares predicted band to actual time-to-normal", () => {
  const tls = buildTimelines(transitions, NOW);
  const predictions = [
    pred({ ts: 2300 }), // actual 45m, band [20,50] → inside
    pred({ ts: 2300, recovery_indeterminate: true }), // excluded
    pred({ ts: 1500, condition: "normal" }), // excluded
  ];
  const res = recoveryError(predictions, tls);
  assert.equal(res.n, 1);
  assert.equal(res.coverage, 1);
  assert.ok(Math.abs(res.points[0].actualMin - 45) < 1e-9);
  assert.ok(Math.abs(res.medianAbsErrorMin - 5) < 1e-9); // |40 - 45|
});

test("schedule-recovery predictions are excluded from HMM calibration", () => {
  const tls = buildTimelines(transitions, NOW);
  const predictions = [
    pred({ ts: 2300 }), // hmm row, scored
    pred({ ts: 2300, recovery_source: "schedule", resumes_at: 5000 }), // excluded
    pred({ ts: 2400, recovery_source: "schedule", resumes_at: 5000 }), // excluded
  ];
  const r30 = reliability(predictions, tls, 30);
  assert.equal(r30.n, 1);
  assert.equal(r30.excludedSchedule, 2);

  const rec = recoveryError(predictions, tls);
  assert.equal(rec.n, 1);
  assert.equal(rec.excludedSchedule, 2);
});
