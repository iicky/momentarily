import { test } from "node:test";
import assert from "node:assert/strict";
import {
  movementConfusion,
  snapTick,
  todBin,
  classifyDirection,
  computeAdvanceBaseline,
  deriveMovementState,
  buildMovementTruth,
  parseThroughStops,
  type MovementState,
  type AdvanceBaseline,
  type AdvanceBaselineCell,
  type VehicleBody,
  type ThroughStops,
} from "../lib/movement.ts";

function cell(p0: number): AdvanceBaselineCell {
  // Only p0 drives classifyDirection; alpha/beta/n are for type completeness.
  return { p0, alpha: 50 * p0, beta: 50 * (1 - p0), n: 50 };
}

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

// --- classifyDirection: per-(route,direction) Beta-Binomial call against a cell's own p0 ---

test("classifyDirection: normal trunk posterior stays well above the disrupted cutoff", () => {
  // post = (8*0.9 + 8) / (8 + 9) = 15.2/17 ≈ 0.894 > 0.45 (= 0.5 * p0)
  assert.equal(classifyDirection(8, 1, cell(0.9)), "normal");
});

test("classifyDirection: disrupted trunk posterior falls at/under the cutoff", () => {
  // post = (8*0.9 + 0) / (8 + 12) = 7.2/20 = 0.36 <= 0.45 (= 0.5 * p0)
  assert.equal(classifyDirection(0, 12, cell(0.9)), "disrupted");
});

test("classifyDirection: shuttle debiasing reads normal where a global 0.25 cutoff would flag disrupted", () => {
  // Raw advance frac 1/10 = 0.10 would trip a single global 0.25 threshold as
  // disrupted, but scored against this line's own low baseline (p0=0.1) the
  // posterior clears the line's own cutoff: (8*0.1 + 1) / (8 + 10) = 1.8/18 = 0.10
  // > 0.05 (= 0.5 * p0).
  assert.equal(classifyDirection(1, 9, cell(0.1)), "normal");
});

test("classifyDirection: fewer than MIN_MATCHED_TRIPS matches is unjudgeable", () => {
  assert.equal(classifyDirection(1, 1, cell(0.9)), null);
});

test("classifyDirection: missing baseline cell is unjudgeable", () => {
  assert.equal(classifyDirection(5, 5, undefined), null);
});

// --- Three-way significance gate: a degenerate-low
// baseline no longer misfires disrupted on ordinary low-advance noise unless the
// drop is also statistically significant against that baseline. This exact
// (advanced, stalled, p0) -> expected-label table is reused verbatim in
// Python's classify_direction tests (tests/test_load_r2.py) and the worker's
// deriveMovementState tests (worker/test/movement_state.test.ts) as a
// cross-mirror parity spot-check. ---

test("classifyDirection: shuttle with a degenerate-low baseline and zero advances abstains, not disrupted (THE FIX)", () => {
  // p0=0.125, advanced=0, stalled=8 (matched=8): post = (8*0.125+0)/(8+8) = 0.0625
  // == 0.5*p0 (<=); tail = 0.875**8 ~= 0.3436 > 0.05 -> null.
  assert.equal(classifyDirection(0, 8, cell(0.125)), null);
});

test("classifyDirection: shuttle freeze with enough matched trips reaches significance and reads disrupted", () => {
  // p0=0.125, advanced=0, stalled=25 (matched=25): tail = 0.875**25 ~= 0.0356 <= 0.05,
  // post ~= 0.030 <= 0.0625 -> disrupted.
  assert.equal(classifyDirection(0, 25, cell(0.125)), "disrupted");
});

test("classifyDirection: mid-range trunk freeze reaches significance and reads disrupted", () => {
  // p0=0.55, advanced=0, stalled=17 (matched=17): post = 0.176 <= 0.275 (0.5*p0);
  // tail = 0.45**17 ~= 1.2e-6 <= 0.05.
  assert.equal(classifyDirection(0, 17, cell(0.55)), "disrupted");
});

test("classifyDirection: mid-range trunk advancing above half its own baseline reads normal", () => {
  // p0=0.55, advanced=8, stalled=9 (matched=17): post = (8*0.55+8)/(8+17) = 0.496 > 0.275.
  assert.equal(classifyDirection(8, 9, cell(0.55)), "normal");
});

// --- computeAdvanceBaseline: per-(route,direction,tod) median-of-fractions prior ---

test("computeAdvanceBaseline: cell p0 is the median advance fraction, with a matching Beta prior, excluding matched<3 ticks", () => {
  const route = "Q";
  const observedAt = 1_700_000_000; // arbitrary; every body shares it so all fracs land in one cell
  const tod = todBin(snapTick(observedAt));

  const bodies: VehicleBody[] = [];
  // 20 qualifying ticks with matched=19 and advanced 0..19 -> fracs 0/19..19/19, median = 0.5.
  for (let advanced = 0; advanced <= 19; advanced++) {
    bodies.push({
      observed_at: observedAt,
      rows: {
        [route]: {
          vehicles_n: 5,
          advanced_n: advanced,
          stalled_n: 19 - advanced,
          by_direction: {
            north: { vehicles_n: 5, advanced_n: advanced, stalled_n: 19 - advanced },
          },
        },
      },
    });
  }
  // 5 more ticks with matched < 3: must not affect the median or be counted.
  for (let i = 0; i < 5; i++) {
    bodies.push({
      observed_at: observedAt,
      rows: {
        [route]: {
          vehicles_n: 5,
          advanced_n: 1,
          stalled_n: 1,
          by_direction: {
            north: { vehicles_n: 5, advanced_n: 1, stalled_n: 1 },
          },
        },
      },
    });
  }

  const baseline = computeAdvanceBaseline(bodies);
  const got = baseline.get(`${route}|north|${tod}`);
  assert.ok(got, "expected a baseline cell for the qualifying ticks");
  assert.equal(got!.n, 20); // the 5 matched<3 ticks were excluded from the count
  assert.equal(got!.p0, 0.5);
  assert.equal(got!.alpha, 25); // 50 * p0
  assert.equal(got!.beta, 25); // 50 * (1 - p0)
});

test("computeAdvanceBaseline: a cell short of BASELINE_MIN_SAMPLES qualifying ticks is omitted", () => {
  const route = "R19";
  const observedAt = 1_700_000_000;
  const tod = todBin(snapTick(observedAt));

  const bodies: VehicleBody[] = [];
  for (let i = 0; i < 19; i++) {
    bodies.push({
      observed_at: observedAt,
      rows: {
        [route]: {
          vehicles_n: 5,
          advanced_n: 8,
          stalled_n: 1,
          by_direction: {
            north: { vehicles_n: 5, advanced_n: 8, stalled_n: 1 },
          },
        },
      },
    });
  }

  const baseline = computeAdvanceBaseline(bodies);
  assert.equal(baseline.has(`${route}|north|${tod}`), false);
});

// --- deriveMovementState: suspended precedence, then worst-of directions ---

test("deriveMovementState: vehicles_n <= 0 is suspended regardless of direction data", () => {
  const baseline: AdvanceBaseline = new Map();
  const row = { vehicles_n: 0, advanced_n: 0, stalled_n: 0 };
  assert.equal(deriveMovementState("Q", row, 1_700_000_000, baseline), "suspended");
});

test("deriveMovementState: one disrupted direction drags the route to disrupted (worst-of)", () => {
  const route = "Q";
  const tick = 1_700_000_000;
  const tod = todBin(tick);
  const baseline: AdvanceBaseline = new Map([
    [`${route}|north|${tod}`, cell(0.9)],
    [`${route}|south|${tod}`, cell(0.9)],
  ]);
  const row = {
    vehicles_n: 10,
    advanced_n: 8,
    stalled_n: 13,
    by_direction: {
      north: { vehicles_n: 5, advanced_n: 0, stalled_n: 12 }, // disrupted
      south: { vehicles_n: 5, advanced_n: 8, stalled_n: 1 }, // normal
    },
  };
  assert.equal(deriveMovementState(route, row, tick, baseline), "disrupted");
});

test("deriveMovementState: both directions normal reads normal", () => {
  const route = "Q";
  const tick = 1_700_000_000;
  const tod = todBin(tick);
  const baseline: AdvanceBaseline = new Map([
    [`${route}|north|${tod}`, cell(0.9)],
    [`${route}|south|${tod}`, cell(0.9)],
  ]);
  const row = {
    vehicles_n: 10,
    advanced_n: 16,
    stalled_n: 2,
    by_direction: {
      north: { vehicles_n: 5, advanced_n: 8, stalled_n: 1 },
      south: { vehicles_n: 5, advanced_n: 8, stalled_n: 1 },
    },
  };
  assert.equal(deriveMovementState(route, row, tick, baseline), "normal");
});

test("deriveMovementState: too-few matches on every direction is unjudgeable", () => {
  const route = "Q";
  const tick = 1_700_000_000;
  const tod = todBin(tick);
  const baseline: AdvanceBaseline = new Map([
    [`${route}|north|${tod}`, cell(0.9)],
    [`${route}|south|${tod}`, cell(0.9)],
  ]);
  const row = {
    vehicles_n: 10,
    advanced_n: 2,
    stalled_n: 0,
    by_direction: {
      north: { vehicles_n: 5, advanced_n: 1, stalled_n: 0 },
      south: { vehicles_n: 5, advanced_n: 1, stalled_n: 0 },
    },
  };
  assert.equal(deriveMovementState(route, row, tick, baseline), null);
});

test("deriveMovementState: no baseline cell for any direction is unjudgeable", () => {
  const route = "Q";
  const tick = 1_700_000_000;
  const baseline: AdvanceBaseline = new Map(); // no cells trained for this route/tod
  const row = {
    vehicles_n: 10,
    advanced_n: 16,
    stalled_n: 2,
    by_direction: {
      north: { vehicles_n: 5, advanced_n: 8, stalled_n: 1 },
      south: { vehicles_n: 5, advanced_n: 8, stalled_n: 1 },
    },
  };
  assert.equal(deriveMovementState(route, row, tick, baseline), null);
});

// --- buildMovementTruth: (route|tick) -> state, judgeable ticks only ---

test("buildMovementTruth keeps suspended/normal routes and drops unjudgeable ones", () => {
  const observedAt = 1_700_000_000;
  const tick = snapTick(observedAt);
  const tod = todBin(tick);

  // Baseline only covers N1's north direction, on purpose — B1 is left without
  // a cell to exercise the "no baseline cell" drop path independently of D1's
  // "too-few matches" drop path.
  const baseline: AdvanceBaseline = new Map([[`N1|north|${tod}`, cell(0.9)]]);

  const bodies: VehicleBody[] = [
    {
      observed_at: observedAt,
      rows: {
        S1: { vehicles_n: 0, advanced_n: 0, stalled_n: 0 },
        N1: {
          vehicles_n: 5,
          advanced_n: 8,
          stalled_n: 1,
          by_direction: { north: { vehicles_n: 5, advanced_n: 8, stalled_n: 1 } },
        },
        D1: {
          vehicles_n: 5,
          advanced_n: 1,
          stalled_n: 1,
          by_direction: { north: { vehicles_n: 5, advanced_n: 1, stalled_n: 1 } }, // matched<3
        },
        B1: {
          vehicles_n: 5,
          advanced_n: 8,
          stalled_n: 1,
          by_direction: { north: { vehicles_n: 5, advanced_n: 8, stalled_n: 1 } }, // no baseline cell
        },
      },
    },
  ];

  const truth = buildMovementTruth(bodies, baseline);

  assert.equal(truth.get(`S1|${tick}`), "suspended");
  assert.equal(truth.get(`N1|${tick}`), "normal");
  assert.equal(truth.has(`D1|${tick}`), false);
  assert.equal(truth.has(`B1|${tick}`), false);
  assert.equal(truth.size, 2);
});

// --- parseThroughStops: flatten params.json's movement_through_stops into
// the route|direction|stop key set crossTickCounts filters against ---

test("parseThroughStops: flattens route/direction/stop nesting into route|direction|stop keys", () => {
  const parsed = parseThroughStops({
    A: { north: ["101N", "102N"], south: ["201S"] },
    B: { north: [] },
  });
  assert.ok(parsed);
  assert.equal(parsed!.has("A|north|101N"), true);
  assert.equal(parsed!.has("A|north|102N"), true);
  assert.equal(parsed!.has("A|south|201S"), true);
  assert.equal(parsed!.has("B|north|anything"), false);
  assert.equal(parsed!.size, 3);
});

test("parseThroughStops: absent, malformed, or all-empty input returns null, never an empty set", () => {
  assert.equal(parseThroughStops(undefined), null);
  assert.equal(parseThroughStops(null), null);
  assert.equal(parseThroughStops("not an object"), null);
  assert.equal(parseThroughStops({}), null);
  assert.equal(parseThroughStops({ A: { north: [] } }), null);
});

// --- Through-stop filtering: computeAdvanceBaseline/deriveMovementState must
// recompute advanced_n/stalled_n from `transitions`, restricted to the
// through-stop set, instead of trusting the archived counters (which change
// meaning at the next Worker deploy). ---

test("computeAdvanceBaseline: filtered recompute from transitions reproduces the archived fraction when the set admits every from_stop", () => {
  const route = "Q";
  const observedAt = 1_700_000_000;
  const tod = todBin(snapTick(observedAt));
  const throughStops: ThroughStops = new Set([`${route}|north|A`, `${route}|north|B`]);

  const bodies: VehicleBody[] = [];
  for (let i = 0; i < 20; i++) {
    bodies.push({
      observed_at: observedAt,
      rows: {
        [route]: {
          vehicles_n: 5,
          // archived counters, pre-summed from the same transitions below:
          // A>B (5, advance) + B>B (2, stall) + B>C (3, advance) = 8 advanced, 2 stalled.
          advanced_n: 8,
          stalled_n: 2,
          by_direction: {
            north: {
              vehicles_n: 5,
              advanced_n: 8,
              stalled_n: 2,
              transitions: { "A>B": 5, "B>B": 2, "B>C": 3 },
            },
          },
        },
      },
    });
  }

  const key = `${route}|north|${tod}`;
  const unfiltered = computeAdvanceBaseline(bodies).get(key);
  const filtered = computeAdvanceBaseline(bodies, throughStops).get(key);
  assert.ok(unfiltered && filtered, "expected a baseline cell both ways");
  assert.equal(unfiltered!.p0, 0.8); // 8 / 10
  assert.equal(filtered!.p0, unfiltered!.p0);
  assert.equal(filtered!.n, unfiltered!.n);
});

test("computeAdvanceBaseline: a stall at an out-of-set stop drops out of the fitted fraction", () => {
  const route = "Q";
  const observedAt = 1_700_000_000;
  const tod = todBin(snapTick(observedAt));
  const throughStops: ThroughStops = new Set([`${route}|north|Y`]); // X is excluded

  const bodies: VehicleBody[] = [];
  for (let i = 0; i < 20; i++) {
    bodies.push({
      observed_at: observedAt,
      rows: {
        [route]: {
          vehicles_n: 5,
          advanced_n: 8,
          stalled_n: 12,
          by_direction: {
            north: {
              vehicles_n: 5,
              advanced_n: 8,
              stalled_n: 12,
              transitions: { "Y>Z": 8, "X>X": 12 }, // X's stall must not count
            },
          },
        },
      },
    });
  }

  const key = `${route}|north|${tod}`;
  const unfiltered = computeAdvanceBaseline(bodies).get(key);
  const filtered = computeAdvanceBaseline(bodies, throughStops).get(key);
  assert.equal(unfiltered!.p0, 8 / 20); // archived counters: X's stall counted
  assert.equal(filtered!.p0, 0.999); // X's stall dropped: Y>Z alone is 8/8 = 1.0, clamped to 1 - P0_FLOOR
});

test("deriveMovementState: a stall at an out-of-set stop drops out of the classification, flipping disrupted to normal", () => {
  const route = "Q";
  const tick = 1_700_000_000;
  const tod = todBin(tick);
  const throughStops: ThroughStops = new Set([`${route}|north|Y`]); // X is excluded
  const baseline: AdvanceBaseline = new Map([[`${route}|north|${tod}`, cell(0.9)]]);
  const row = {
    vehicles_n: 5,
    advanced_n: 8,
    stalled_n: 40,
    by_direction: {
      north: {
        vehicles_n: 5,
        advanced_n: 8,
        stalled_n: 40,
        transitions: { "Y>Z": 8, "X>X": 40 },
      },
    },
  };
  // Archived counters, unfiltered: X's 40 stalls read as a real freeze.
  assert.equal(deriveMovementState(route, row, tick, baseline), "disrupted");
  // Filtered: X isn't a through stop, so its stall never enters the count.
  assert.equal(deriveMovementState(route, row, tick, baseline, throughStops), "normal");
});

test("deriveMovementState: an advance out of an out-of-set stop drops out while an advance into one is kept", () => {
  const route = "Q";
  const tick = 1_700_000_000;
  const tod = todBin(tick);
  // Only A is a through stop -- B (an advance target) and C (an advance
  // source) are both excluded.
  const throughStops: ThroughStops = new Set([`${route}|north|A`]);
  const baseline: AdvanceBaseline = new Map([[`${route}|north|${tod}`, cell(0.9)]]);
  const row = {
    vehicles_n: 5,
    advanced_n: 44,
    stalled_n: 0,
    by_direction: {
      north: {
        vehicles_n: 5,
        advanced_n: 44,
        stalled_n: 0,
        // A>B: from a through stop into an excluded one -- kept.
        // C>C: from an excluded stop -- dropped, even though it's a stall
        // large enough to flip the call if it were wrongly counted.
        transitions: { "A>B": 4, "C>C": 40 },
      },
    },
  };
  assert.equal(deriveMovementState(route, row, tick, baseline, throughStops), "normal");
});

test("deriveMovementState: a row with no transitions map abstains under a through-stop filter instead of fabricating a zero advance rate", () => {
  const route = "Q";
  const tick = 1_700_000_000;
  const tod = todBin(tick);
  const throughStops: ThroughStops = new Set([`${route}|north|Y`]);
  const baseline: AdvanceBaseline = new Map([[`${route}|north|${tod}`, cell(0.9)]]);
  const row = {
    vehicles_n: 5,
    advanced_n: 0,
    stalled_n: 12,
    by_direction: {
      // No `transitions` map to recompute from -- an older archive row, or a
      // feed gap that only wrote the summary counters.
      north: { vehicles_n: 5, advanced_n: 0, stalled_n: 12 },
    },
  };
  // Unfiltered trusts the archived counters straight off the row -- disrupted.
  assert.equal(deriveMovementState(route, row, tick, baseline), "disrupted");
  // Filtered has nothing to recompute from, so it must abstain (matched=0 <
  // MIN_MATCHED_TRIPS) rather than read the missing map as zero advances,
  // which would misreport a data gap as a real stall.
  assert.equal(deriveMovementState(route, row, tick, baseline, throughStops), null);
});
