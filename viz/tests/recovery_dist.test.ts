import { test } from "node:test";
import assert from "node:assert/strict";
import {
  recoveryDistReport,
  recoveryVerdict,
  type RecoveryDistReport,
  type RecoveryDistSample,
} from "../lib/recovery_dist.ts";

// Minimal report carrying only the fields recoveryVerdict reads.
function report(
  pit: number[],
  meanPit: number,
  regimes: number,
  skill: number,
): RecoveryDistReport {
  return {
    perRegime: { n: regimes, meanCrps: 0, baselineCrps: 0, skill, meanPit },
    pit,
    meanPit,
  } as unknown as RecoveryDistReport;
}

const UNIFORM = new Array(10).fill(10);

// Step CDF at integer minutes 0..4 that jumps to 1 at minute `at`.
function stepCurve(at: number): number[] {
  return [0, 1, 2, 3, 4].map((t) => (t >= at ? 1 : 0));
}

function sample(regimeKey: string, actualMin: number, jumpAt: number): RecoveryDistSample {
  return { regimeKey, actualMin, predCurve: stepCurve(jumpAt) };
}

test("recoveryDistReport separates per-tick from per-regime weighting", () => {
  // One long, well-forecast incident (8 ticks, curve nails the recovery) and one
  // short, badly-forecast incident (2 ticks, curve says 'already back'). Per-tick
  // is dominated by the 8 good ticks; per-regime weights the two incidents equally.
  const samples: RecoveryDistSample[] = [
    ...Array.from({ length: 8 }, () => sample("good:0", 2, 2)),
    ...Array.from({ length: 2 }, () => sample("bad:0", 3, 0)),
  ];
  const r = recoveryDistReport(samples);

  assert.equal(r.perTick.n, 10);
  assert.equal(r.perRegime.n, 2);
  // Top-level headline stays per-tick for the curve view's back-compat.
  assert.equal(r.n, 10);
  assert.equal(r.meanCrps, r.perTick.meanCrps);

  // The bad incident is one tick-heavy regime's worth of error spread across only
  // two ticks, so equal-per-incident weighting must score worse than per-tick.
  assert.ok(
    r.perRegime.meanCrps > r.perTick.meanCrps,
    `per-regime ${r.perRegime.meanCrps} should exceed per-tick ${r.perTick.meanCrps}`,
  );
  assert.ok(Number.isFinite(r.perTick.skill));
  assert.ok(Number.isFinite(r.perRegime.skill));
});

test("recoveryDistReport collapses ticks from one regime into a single incident", () => {
  const samples: RecoveryDistSample[] = Array.from({ length: 12 }, () =>
    sample("solo:100", 2, 2),
  );
  const r = recoveryDistReport(samples);
  assert.equal(r.perTick.n, 12);
  assert.equal(r.perRegime.n, 1);
});

test("recoveryDistReport handles the empty window", () => {
  const r = recoveryDistReport([]);
  assert.equal(r.n, 0);
  assert.equal(r.perTick.n, 0);
  assert.equal(r.perRegime.n, 0);
  assert.ok(Number.isNaN(r.perRegime.meanCrps));
});

test("recoveryVerdict: too few incidents reads inconclusive", () => {
  const v = recoveryVerdict(report(UNIFORM, 0.5, 3, 0.2));
  assert.equal(v.verdict, "Inconclusive");
  assert.equal(v.tone, "muted");
});

test("recoveryVerdict: empty histogram reads no-data", () => {
  const v = recoveryVerdict(report(new Array(10).fill(0), NaN, 0, NaN));
  assert.equal(v.verdict, "Not enough data yet");
});

test("recoveryVerdict: uniform PIT with positive skill is well calibrated", () => {
  const v = recoveryVerdict(report(UNIFORM, 0.5, 20, 0.3));
  assert.equal(v.verdict, "Well calibrated");
  assert.equal(v.tone, "good");
  assert.equal(v.warning, undefined);
});

test("recoveryVerdict: calibrated shape but negative skill warns of the conflict", () => {
  const v = recoveryVerdict(report(UNIFORM, 0.5, 20, -0.3));
  assert.equal(v.verdict, "Well calibrated");
  assert.ok(v.warning && /baseline/.test(v.warning), "expected a skill-vs-shape warning");
});

test("recoveryVerdict: left-piled PIT leans cautious", () => {
  const v = recoveryVerdict(report([30, 25, 20, 15, 5, 2, 1, 1, 1, 0], 0.3, 20, 0.1));
  assert.equal(v.verdict, "Leans cautious");
});

test("recoveryVerdict: U-shaped PIT reads overconfident", () => {
  const v = recoveryVerdict(report([40, 5, 3, 2, 1, 1, 2, 3, 5, 38], 0.5, 20, -0.2));
  assert.equal(v.verdict, "Overconfident");
  assert.equal(v.tone, "warn");
});
