/**
 * Segment-level regime clock: the same debounced clock the route uses
 * (advanceRegimes, regime.ts), fed by deriveSegmentStates and gated back to
 * the live cell set by pruneSegmentRegimes. Generic debounce/back-dating/
 * abstention mechanics are pinned in regime.test.ts; this file covers the
 * segment-specific wiring in segment_flow.ts + step 8b of index.ts.
 */

import { describe, expect, test } from "vitest";

import { movementTransitions } from '../src/grading';
import { advanceRegimes } from '../src/regime';
import type {
  AdvanceRegimesOptions,
  RegimeChange,
  RegimeEntry,
} from '../src/regime';
import {
  deriveSegmentStates,
  MIN_EFF_MATCHED,
  pruneSegmentRegimes,
  updateSegmentFlow,
} from '../src/segment_flow';
import type { SegmentFlowDoc, SegmentParamsDoc } from '../src/state';
import type { MovementRow } from '../src/vehicles';

function moveRow(south: Record<string, number>): MovementRow {
  return {
    vehicles_n: 10,
    stopped_n: 0,
    moving_n: 0,
    advanced_n: 0,
    stalled_n: 0,
    by_direction: {
      north: { vehicles_n: 0, advanced_n: 0, stalled_n: 0, transitions: {} },
      south: { vehicles_n: 5, advanced_n: 0, stalled_n: 0, transitions: south },
    },
  } as MovementRow;
}

const params: SegmentParamsDoc = {
  schema_version: "1",
  trained_at: 1,
  min_share: 0.5,
  topology_source: "observed",
  cells: { "F|south|A09S": { p0: 0.9, n: 1000 } },
  adjacency: {
    "F|south|A09S": { to: "A10S", source: "observed", share: 0.9, n: 1000 },
  },
};

const KEY = "F|south|A09S";
const TICK = 300;
const T0 = 1_700_000_000;
const t = (i: number) => T0 + i * TICK;

// One tick of the step-8b segment block: update the decaying cell, advance the
// regime clock off deriveSegmentStates, then intersect with the live cell set
// exactly like index.ts does. `options` defaults to whatever advanceRegimes
// defaults to (the production wiring passes none); pass debounceTicks
// explicitly to pin a test to the generic multi-tick debounce mechanism
// regardless of where the production default sits.
function tick(
  prevFlow: SegmentFlowDoc | null,
  south: Record<string, number> | null,
  at: number,
  options: AdvanceRegimesOptions = {},
): { flow: SegmentFlowDoc; changes: RegimeChange<"normal" | "disrupted">[] } {
  const rows = new Map<string, MovementRow>();
  if (south) rows.set("F", moveRow(south));
  const flow = updateSegmentFlow(prevFlow, rows, at, params);
  const { entries, changes } = advanceRegimes(
    prevFlow?.regimes,
    deriveSegmentStates(flow, params),
    at,
    options,
  );
  flow.regimes = pruneSegmentRegimes(entries, flow.cells);
  return { flow, changes };
}

describe("segment regime clock (step 8b wiring)", () => {
  test("back-dates entered_at to the first disrupted tick, exactly like a route", () => {
    // Cold-start normal, then two agreeing disrupted ticks (>= DEBOUNCE_TICKS).
    let step = tick(null, { "A09S>A10S": 10 }, t(0), { debounceTicks: 2 });
    expect(step.flow.regimes[KEY]).toEqual(
      expect.objectContaining({ state: "normal", entered_at: t(0) }),
    );

    step = tick(step.flow, { "A09S>A09S": 200 }, t(1), { debounceTicks: 2 }); // candidate run=1, not yet committed
    expect(step.flow.regimes[KEY]!.state).toBe("normal");
    expect(step.changes).toEqual([]);

    step = tick(step.flow, { "A09S>A09S": 200 }, t(2), { debounceTicks: 2 }); // run=2 commits
    expect(step.flow.regimes[KEY]).toEqual(
      expect.objectContaining({
        state: "disrupted",
        entered_at: t(1),
        last_seen_at: t(2),
      }),
    );
    // Back-dated to the FIRST tick of the disrupted run (t(1)), not the tick
    // it actually committed on (t(2)).
    expect(step.changes).toEqual([
      {
        key: KEY,
        prev_state: "normal",
        new_state: "disrupted",
        entered_at: t(0),
        exited_at: t(1),
        dwell_sec: TICK,
      },
    ]);
  });

  test("segment transitions carry scope, key, and the route parsed off the key", () => {
    let step = tick(null, { "A09S>A10S": 10 }, t(0), { debounceTicks: 2 });
    step = tick(step.flow, { "A09S>A09S": 200 }, t(1), { debounceTicks: 2 });
    step = tick(step.flow, { "A09S>A09S": 200 }, t(2), { debounceTicks: 2 });

    const [record] = movementTransitions(step.changes, "segment", t(2));
    expect(record).toEqual({
      ts: t(2),
      scope: "segment",
      key: KEY,
      route: "F",
      prev_state: "normal",
      new_state: "disrupted",
      regime_entered_at: t(0),
      exited_at: t(1),
      dwell_sec: TICK,
    });
  });

  test("a cell too thin to judge abstains: the regime holds open, unlike a genuine reading", () => {
    const prevRegimes: Record<string, RegimeEntry<"normal" | "disrupted">> = {
      [KEY]: {
        state: "disrupted",
        entered_at: t(0),
        last_seen_at: t(1),
        pending: null,
        pending_since: 0,
        pending_run: 0,
      },
    };
    // Below MIN_EFF_MATCHED: still tracked (m > PRUNE_MATCHED), but too
    // thin to judge — deriveSegmentStates must leave it out of the map.
    const flow: SegmentFlowDoc = {
      observed_at: t(2),
      cells: { [KEY]: { a: 0, m: MIN_EFF_MATCHED - 1 } },
      regimes: {},
    };
    const observed = deriveSegmentStates(flow, params);
    expect(observed).toEqual({});

    const { entries, changes } = advanceRegimes(prevRegimes, observed, t(2));
    const pruned = pruneSegmentRegimes(entries, flow.cells);
    // Unchanged: no reading of change, so no debounce candidate and no commit.
    expect(pruned[KEY]).toEqual(prevRegimes[KEY]);
    expect(changes).toEqual([]);
  });

  test("a pruned cell drops its regime, unlike an abstaining one", () => {
    const prevFlow: SegmentFlowDoc = {
      observed_at: t(0),
      cells: { [KEY]: { a: 0, m: 0.31 } }, // just above PRUNE_MATCHED (0.3)
      regimes: {
        [KEY]: {
          state: "disrupted",
          entered_at: t(-10),
          last_seen_at: t(0),
          pending: null,
          pending_since: 0,
          pending_run: 0,
        },
      },
    };
    // A quiet tick decays m to SEGMENT_DECAY(0.94) * 0.31 = 0.2914 <
    // PRUNE_MATCHED: the cell drops out of `cells` outright (gone quiet),
    // not merely unobserved.
    const flow = updateSegmentFlow(prevFlow, new Map(), t(1), params);
    expect(flow.cells[KEY]).toBeUndefined();

    const { entries } = advanceRegimes(
      prevFlow.regimes,
      deriveSegmentStates(flow, params),
      t(1),
    );
    // Raw advanceRegimes alone still holds it open (correct generic
    // abstention behavior) — it has no notion of "pruned".
    expect(entries[KEY]?.state).toBe("disrupted");

    // Intersecting with the live cell set is what actually drops it.
    const pruned = pruneSegmentRegimes(entries, flow.cells);
    expect(pruned[KEY]).toBeUndefined();
  });
});

describe("pruneSegmentRegimes", () => {
  test("keeps only entries whose cell survived this tick", () => {
    const entries: Record<string, RegimeEntry> = {
      A: {
        state: "disrupted",
        entered_at: 1,
        last_seen_at: 1,
        pending: null,
        pending_since: 0,
        pending_run: 0,
      },
      B: {
        state: "normal",
        entered_at: 1,
        last_seen_at: 1,
        pending: null,
        pending_since: 0,
        pending_run: 0,
      },
    };
    const liveCells = { A: { a: 1, m: 1 } }; // B pruned out of cells this tick
    expect(pruneSegmentRegimes(entries, liveCells)).toEqual({ A: entries.A });
  });

  test("adds nothing for a live cell with no regime entry yet", () => {
    expect(pruneSegmentRegimes({}, { A: { a: 1, m: 1 } })).toEqual({});
  });
});
