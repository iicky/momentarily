import { test } from "node:test";
import assert from "node:assert/strict";
import {
  dwellCdf,
  mixtureSurvival,
  mixtureQuantile,
  pLeaveBy,
  predictedRecoveryCurve,
  RECOVERY_TMAX_MIN,
} from "../lib/dwell.ts";

// The pre-fix dwellCdf, kept here verbatim as the regression oracle: the
// lower guard was inclusive (`x <= curve[0]` -> 0), which is the bug the
// strict version fixes. On a strictly increasing curve the two formulas must
// agree everywhere; they only diverge on a repeated knot.
function oldDwellCdf(curveSec: number[], x: number): number {
  const k = curveSec.length;
  if (x >= curveSec[k - 1]) return 1.0;
  if (x <= curveSec[0]) return 0.0;
  for (let i = 0; i < k - 1; i++) {
    const lo = curveSec[i];
    const hi = curveSec[i + 1];
    if (lo <= x && x <= hi) {
      const frac = hi === lo ? 0.0 : (x - lo) / (hi - lo);
      return (i + frac) / (k - 1);
    }
  }
  return 1.0;
}

test("dwellCdf matches the pre-fix formula on strictly increasing curves", () => {
  const curve = [0, 50, 100, 200, 400, 800, 1600];
  for (const x of [-10, 0, 25, 50, 75, 100, 300, 400, 900, 1600, 5000]) {
    assert.equal(dwellCdf(curve, x), oldDwellCdf(curve, x));
  }
});

test("dwellCdf: repeated left knot returns the top of the flat run, not 0", () => {
  // 14 of 21 knots pinned at one tick (300s) — the shape a mixture cell's
  // published curve_sec actually takes.
  const curve = [...Array(14).fill(300), 400, 600, 900, 1300, 1800, 2400, 3000];
  const k = curve.length;
  assert.equal(dwellCdf(curve, 300), 13 / (k - 1));
  // The pre-fix formula zeroed this out — that's the bug being fixed.
  assert.equal(oldDwellCdf(curve, 300), 0.0);
});

test("dwellCdf: below the first knot is still 0", () => {
  assert.equal(dwellCdf([100, 200], 50), 0.0);
});

test("mixtureSurvival: F(atomSec) === atomP exactly", () => {
  const shape = 1.8,
    scale = 1200,
    atomP = 0.704,
    atomSec = 300;
  const s = mixtureSurvival(atomSec, shape, scale, atomP, atomSec);
  assert.ok(Math.abs(1 - s - atomP) < 1e-12);
});

test("mixtureSurvival: below atomSec the atom hasn't resolved yet, S=1", () => {
  assert.equal(mixtureSurvival(299, 1.8, 1200, 0.7, 300), 1.0);
});

test("mixtureQuantile inverts mixtureSurvival's F above the atom", () => {
  const shape = 1.8,
    scale = 1200,
    atomP = 0.7,
    atomSec = 300;
  for (const u of [0.75, 0.85, 0.95, 0.99]) {
    const t = mixtureQuantile(u, shape, scale, atomP, atomSec);
    const s = mixtureSurvival(t, shape, scale, atomP, atomSec);
    assert.ok(Math.abs(1 - s - u) < 1e-6);
  }
});

test("mixtureQuantile: u <= atomP stays at atomSec — the atom is an interval, not a point", () => {
  assert.equal(mixtureQuantile(0.3, 1.8, 1200, 0.7, 300), 300);
  assert.equal(mixtureQuantile(0.7, 1.8, 1200, 0.7, 300), 300);
});

test("pLeaveBy with atom: elapsed=0 over the atom horizon recovers exactly atomP", () => {
  const shape = 1.8,
    scale = 1200,
    atomP = 0.704,
    atomSec = 300;
  // curveSec is irrelevant on the atom path — pass something the curve path
  // would answer very differently for, to prove it's ignored.
  const p = pLeaveBy([1, 2], 0, atomSec, [shape, scale], { p: atomP, sec: atomSec });
  assert.ok(Math.abs(p - atomP) < 1e-12);
});

test("pLeaveBy with atom reduces to the plain log-logistic conditional once elapsed >= atomSec", () => {
  const shape = 1.8,
    scale = 1200,
    atomP = 0.704,
    atomSec = 300;
  const elapsed = 900,
    horizon = 600;
  const sLl = (t: number) => 1 / (1 + (t / scale) ** shape);
  const expected = 1 - sLl(elapsed + horizon) / sLl(elapsed);
  const got = pLeaveBy([1, 2], elapsed, horizon, [shape, scale], { p: atomP, sec: atomSec });
  assert.ok(Math.abs(got - expected) < 1e-9);
});

test("pLeaveBy with atom ignores curveSec entirely, including inside its range", () => {
  const shape = 1.8,
    scale = 1200,
    atomP = 0.704,
    atomSec = 300;
  const a = pLeaveBy([10, 20], 0, 300, [shape, scale], { p: atomP, sec: atomSec });
  const b = pLeaveBy([99999, 100000, 100001], 0, 300, [shape, scale], {
    p: atomP,
    sec: atomSec,
  });
  assert.equal(a, b);
});

test("pLeaveBy: atom without a usable tail falls back to the curve path unchanged", () => {
  const curve = [0, 100];
  const withAtomNoTail = pLeaveBy(curve, 50, 25, undefined, { p: 0.7, sec: 300 });
  const withoutAtom = pLeaveBy(curve, 50, 25);
  assert.equal(withAtomNoTail, withoutAtom);
});

test("pLeaveBy: atom_p outside (0,1) falls back to the curve/tail path unchanged", () => {
  const curve = [0, 100];
  const tail: [number, number] = [2, 100];
  const invalid = pLeaveBy(curve, 50, 25, tail, { p: 1, sec: 300 });
  const noAtom = pLeaveBy(curve, 50, 25, tail);
  assert.equal(invalid, noAtom);
});

test("pLeaveBy / predictedRecoveryCurve: no-atom cells behave exactly as before", () => {
  const curve = [0, 100];
  const tail: [number, number] = [2, 100];
  assert.equal(pLeaveBy(curve, 50, 25, tail), 0.5); // inside the curve the tail splice is inert
  const curveOut = predictedRecoveryCurve(0, curve, tail);
  assert.equal(curveOut.length, RECOVERY_TMAX_MIN + 1);
  assert.equal(curveOut[10], pLeaveBy(curve, 0, 600, tail));
});

test("predictedRecoveryCurve threads the atom into every sampled minute", () => {
  const shape = 1.8,
    scale = 1200,
    atomP = 0.704,
    atomSec = 300; // 5 minutes
  const out = predictedRecoveryCurve(0, [1, 2], [shape, scale], { p: atomP, sec: atomSec });
  assert.ok(Math.abs(out[5] - atomP) < 1e-12);
  // Monotone non-decreasing — a CDF sampled over an increasing horizon can't drop.
  for (let t = 1; t < out.length; t++) {
    assert.ok(out[t] >= out[t - 1] - 1e-12);
  }
});
