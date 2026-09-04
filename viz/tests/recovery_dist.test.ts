import { test } from "node:test";
import assert from "node:assert/strict";
import {
  jumpFraction,
  recoveryDistReport,
  recoveryVerdict,
  type RecoveryDistReport,
  type RecoveryDistSample,
} from "../lib/recovery_dist.ts";

// Minimal report carrying only the fields recoveryVerdict reads. `skill` lands
// on the oracle column: no training window, so the verdict falls back to it and
// its copy must name it as hindsight.
function report(
  pit: number[],
  meanPit: number,
  regimes: number,
  skill: number,
): RecoveryDistReport {
  return {
    perRegime: {
      n: regimes,
      meanCrps: 0,
      oracleBaselineCrps: 0,
      oracleSkill: skill,
      causalBaselineCrps: null,
      causalSkill: null,
      meanPit,
    },
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
  const r = recoveryDistReport(samples, null);

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
  assert.ok(Number.isFinite(r.perTick.oracleSkill));
  assert.ok(Number.isFinite(r.perRegime.oracleSkill));
  // No training durations were supplied, so the causal column must stay null
  // rather than quietly repeating the hindsight number.
  assert.equal(r.perTick.causalSkill, null);
  assert.equal(r.perRegime.causalSkill, null);
});

test("recoveryDistReport collapses ticks from one regime into a single incident", () => {
  const samples: RecoveryDistSample[] = Array.from({ length: 12 }, () =>
    sample("solo:100", 2, 2),
  );
  const r = recoveryDistReport(samples, null);
  assert.equal(r.perTick.n, 12);
  assert.equal(r.perRegime.n, 1);
});

test("recoveryDistReport handles the empty window", () => {
  const r = recoveryDistReport([], null);
  assert.equal(r.n, 0);
  assert.equal(r.perTick.n, 0);
  assert.equal(r.perRegime.n, 0);
  assert.ok(Number.isNaN(r.perRegime.meanCrps));
});

test("recoveryDistReport rejects an empty training window instead of degrading to oracle-only", () => {
  // null means "no pre-window population, decided"; [] means a causal window
  // was built and came out empty. Silently dropping the causal column here is
  // how hindsight-relative numbers ship (mirrors the Python port's ValueError).
  assert.throws(() => recoveryDistReport([sample("r:1", 2, 2)], []), /empty/);
});

test("recoveryDistReport: the causal baseline is fitted only on the durations handed in", () => {
  // Same graded population both times; only the training window changes. A
  // baseline that peeked at the eval durations could not move here.
  const samples: RecoveryDistSample[] = [
    sample("a:0", 2, 2),
    sample("b:0", 3, 2),
    sample("c:0", 1, 2),
  ];
  const near = recoveryDistReport(samples, [1, 2, 3]);
  const far = recoveryDistReport(samples, [4, 4, 4]);
  assert.equal(near.meanCrps, far.meanCrps);
  assert.equal(near.oracleBaselineCrps, far.oracleBaselineCrps);
  assert.notEqual(near.causalBaselineCrps, far.causalBaselineCrps);
  // A training window that misses the truth is a worse forecast, so the model
  // scores better against it — the causal column moves with the baseline alone.
  assert.ok((far.causalSkill ?? 0) > (near.causalSkill ?? 0));
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

test("recoveryDistReport: absent predLeft leaves the PIT unrandomized (pins the pre-correction numbers)", () => {
  // Three continuous cells, no predLeft on any sample — this must reproduce
  // exactly the histogram/meanPit the un-corrected code computed, byte for byte.
  const samples: RecoveryDistSample[] = [
    sample("r1:0", 1, 1), // idx 1, u = f[1] = 1
    sample("r2:0", 2, 3), // idx 2, u = f[2] = 0
    sample("r3:0", 4, 0), // idx 4, u = f[4] = 1
  ];
  const r = recoveryDistReport(samples, null);
  assert.equal(r.meanPit, 2 / 3);
  assert.deepEqual(r.pit, [1, 0, 0, 0, 0, 0, 0, 0, 0, 2]);
});

test("recoveryDistReport: predLeft < u lands strictly inside [predLeft, u)", () => {
  const s: RecoveryDistSample = {
    regimeKey: "atomic:7",
    actualMin: 2,
    predCurve: stepCurve(2), // f = [0, 0, 1, 1, 1], u = f[2] = 1
    predLeft: 0,
  };
  const r = recoveryDistReport([s], null);
  const frac = jumpFraction("atomic:7");
  // n === 1, so the mean is exactly the one randomized PIT value.
  assert.equal(r.meanPit, frac);
  assert.ok(r.meanPit >= 0 && r.meanPit < 1, `expected pit in [0, 1), got ${r.meanPit}`);
});

test("jumpFraction: deterministic across repeated calls for the same regime key", () => {
  const key = "route-9:1700000000";
  const a = jumpFraction(key);
  const b = jumpFraction(key);
  assert.equal(a, b);
  assert.ok(a >= 0 && a < 1);
});

// Hand-derived from the FNV-1a 32-bit algorithm mirrored from
// training/recovery_dist.py's _jump_fraction (h0 = 0x811c9dc5, prime =
// 0x01000193, folded over the UTF-8 bytes of "route-42:12345") — the loop
// ends at h = 3326733886 (0xc649ee3e), so h / 2**32 = 0.774565591942519.
// If this ever drifts, the JS and Python randomized-PIT placements have
// silently diverged.
test("jumpFraction: pins the FNV-1a value for a known key (Python parity)", () => {
  assert.equal(jumpFraction("route-42:12345"), 0.774565591942519);
});
