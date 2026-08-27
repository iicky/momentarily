import { describe, expect, test } from 'vitest';

import { schedule_bin } from '../src/hmm';
import {
  classifyThroughput,
  deriveSegmentStates,
  deriveStationFlow,
  pruneSegmentRegimes,
  SEGMENT_DECAY,
  stationId,
  updateSegmentFlow,
} from '../src/segment_flow';
import type { RegimeEntry } from '../src/regime';
import type { SegmentCondition, SegmentFlowDoc, SegmentParamsDoc } from '../src/state';
import type { MovementRow } from '../src/vehicles';

// Wed 2026-08-19 14:00Z = 10:00 ET, and one hour later — two different
// schedule_bins, so the bin-edge behaviour is exercised without hardcoding a
// bin label or assuming a DST offset.
const NOW = Date.UTC(2026, 7, 19, 14, 0, 0) / 1000;
const NEXT_BIN = NOW + 3600;
const BIN = schedule_bin(NOW);
const BIN_NEXT = schedule_bin(NEXT_BIN);

function moveRow(south: Record<string, number>, vehicles = 5): MovementRow {
  return {
    vehicles_n: 10,
    stopped_n: 0,
    moving_n: 0,
    advanced_n: 0,
    stalled_n: 0,
    by_direction: {
      north: { vehicles_n: 0, advanced_n: 0, stalled_n: 0, transitions: {} },
      south: { vehicles_n: vehicles, advanced_n: 0, stalled_n: 0, transitions: south },
    },
  } as MovementRow;
}

const params: SegmentParamsDoc = {
  schema_version: '1',
  trained_at: 1,
  min_share: 0.5,
  topology_source: 'observed',
  cells: {
    'F|south|A09S': { p0: 0.9, n: 1000 },
    'F|south|A10S': { p0: 0.9, n: 1000 },
  },
  adjacency: {
    'F|south|A09S': { to: 'A10S', source: 'observed', share: 0.9, n: 1000 },
    'F|south|A10S': { to: 'A11S', source: 'observed', share: 0.9, n: 1000 },
  },
};

/** `params` plus a throughput fit: A09S busy at BIN and idle at the next bin,
 * A10S never scheduled. */
const fitted: SegmentParamsDoc = {
  ...params,
  cells: {
    'F|south|A09S': { p0: 0.9, n: 1000, lam: { [BIN]: 1 } },
    'F|south|A10S': { p0: 0.9, n: 1000 },
  },
  throughput: { bin: 'schedule_bin', min_ticks: 20, ticks: { [BIN]: 500, [BIN_NEXT]: 500 } },
};

describe('stationId', () => {
  test('strips a trailing N/S, leaves others', () => {
    expect(stationId('A09S')).toBe('A09');
    expect(stationId('A09N')).toBe('A09');
    expect(stationId('A09')).toBe('A09');
  });
});

describe('classifyThroughput', () => {
  // mu = expected * (1 + SEGMENT_DECAY); at decay 0.94 the power floor
  // -ln(0.05) = 2.996 falls between expected = 1.5 and expected = 1.6.
  test('an expectation too small to test reads quiet, not an abstention', () => {
    expect(classifyThroughput(0, 1.5, true)).toBe('quiet');
    expect(classifyThroughput(0, 0, true)).toBe('quiet');
  });

  test('just past the power floor, an empty window reads disrupted', () => {
    expect(classifyThroughput(0, 1.6, true)).toBe('disrupted');
  });

  test('silence against a real expectation reads disrupted', () => {
    expect(classifyThroughput(0, 5, true)).toBe('disrupted');
  });

  test('traversals near the expectation read normal', () => {
    expect(classifyThroughput(5, 5, true)).toBe('normal');
  });

  test('a route the feed skipped abstains instead of blaming the railway', () => {
    expect(classifyThroughput(0, 5, false)).toBeNull();
    // The quiet call is a statement about the timetable, so a dead feed does
    // not suppress it.
    expect(classifyThroughput(0, 0, false)).toBe('quiet');
  });

  test('the effective count floors, so the call is monotone in the observation', () => {
    // With rounding, a cell whose expectation is fading out flips
    // normal/disrupted purely on which side of .5 the decayed remnant lands.
    const rank = { normal: 0, disrupted: 1 } as const;
    const seen: number[] = [];
    for (let m = 60; m >= 0; m--) {
      const call = classifyThroughput(m / 10, 3, true);
      if (call === 'normal' || call === 'disrupted') seen.push(rank[call]);
    }
    expect(seen).toEqual([...seen].sort((a, b) => a - b));
  });
});

describe('updateSegmentFlow', () => {
  test('accumulates advanced/matched, ignoring stalls in the advance count', () => {
    const rows = new Map([['F', moveRow({ 'A09S>A10S': 10, 'A09S>A09S': 2 })]]);
    const state = updateSegmentFlow(null, rows, NOW, params);
    expect(state.cells['F|south|A09S']).toEqual({ a: 10, m: 12, e: 0 });
  });

  test('decays the carried accumulator by SEGMENT_DECAY', () => {
    const prev: SegmentFlowDoc = {
      observed_at: NOW - 300,
      cells: { 'F|south|A09S': { a: 10, m: 12, e: 0 } },
      vehicles: {},
      regimes: {},
    };
    const rows = new Map([['F', moveRow({ 'A09S>A09S': 5 })]]); // all stalls this tick
    const state = updateSegmentFlow(prev, rows, NOW, params);
    expect(state.cells['F|south|A09S']!.a).toBeCloseTo(SEGMENT_DECAY * 10, 6);
    expect(state.cells['F|south|A09S']!.m).toBeCloseTo(5 + SEGMENT_DECAY * 12, 6);
  });

  test('tracks only segments the trainer baselined', () => {
    const rows = new Map([['F', moveRow({ 'Z99S>Z98S': 20 })]]); // not in params.cells
    const state = updateSegmentFlow(null, rows, NOW, params);
    expect(state.cells['F|south|Z99S']).toBeUndefined();
  });

  test('accumulates the current bin rate as an expectation even with no traffic', () => {
    const rows = new Map([['F', moveRow({})]]);
    const state = updateSegmentFlow(null, rows, NOW, fitted);
    // A09S expects 1 traversal per tick at this bin and saw none: kept, with
    // the expectation carried so the next tick compounds it.
    expect(state.cells['F|south|A09S']).toEqual({ a: 0, m: 0, e: 1 });
    // A10S is not scheduled at this bin, so it has neither an observation nor
    // an expectation and prunes back out.
    expect(state.cells['F|south|A10S']).toBeUndefined();
  });

  test('the expectation carries the previous bin rate across a bin edge', () => {
    const rows = new Map([['F', moveRow({})]]);
    const prev = updateSegmentFlow(null, rows, NOW, fitted);
    // One tick later, in a bin where A09S is scheduled for nothing: the window
    // still holds the busy bin's expectation, decayed — the whole point of
    // running the sum over rates instead of scaling the current rate.
    const next = updateSegmentFlow(prev, rows, NEXT_BIN, fitted);
    expect(next.cells['F|south|A09S']!.e).toBeCloseTo(SEGMENT_DECAY, 6);
  });

  test('records this tick per-route vehicles, dropping routes the feed skipped', () => {
    const withTrains = updateSegmentFlow(null, new Map([['F', moveRow({}, 5)]]), NOW, fitted);
    expect(withTrains.vehicles).toEqual({ F: 10 });
    // Not decayed: a decayed vehicle count falls at the same rate as the decayed
    // matched count and starts an order of magnitude higher, so any floor on it
    // is crossed long after the false disrupted calls have fired.
    const gone = updateSegmentFlow(withTrains, new Map(), NOW + 300, fitted);
    expect(gone.vehicles).toEqual({});
  });
});

describe('deriveStationFlow', () => {
  test('frozen segment degrades both endpoint stations', () => {
    const state: SegmentFlowDoc = {
      observed_at: NOW,
      cells: { 'F|south|A09S': { a: 1, m: 40, e: 0 } }, // ~0.025 vs p0 0.9
      vehicles: {},
      regimes: {},
    };
    const doc = deriveStationFlow(state, params);
    expect(doc.stations['A09']!.status).toBe('degraded');
    expect(doc.stations['A10']!.status).toBe('degraded'); // to_stop endpoint too
    expect(doc.stations['A09']!.worst_segment).toEqual(['A09S', 'A10S']);
    expect(doc.stations['A09']!.routes).toEqual(['F']);
  });

  test('segment advancing near normal reads flowing', () => {
    const state: SegmentFlowDoc = {
      observed_at: NOW,
      cells: { 'F|south|A09S': { a: 38, m: 40, e: 0 } }, // ~0.95 vs p0 0.9
      vehicles: {},
      regimes: {},
    };
    const doc = deriveStationFlow(state, params);
    expect(doc.stations['A09']!.status).toBe('flowing');
  });

  test('with no throughput fit, a thin segment is still skipped entirely', () => {
    const state: SegmentFlowDoc = {
      observed_at: NOW,
      cells: { 'F|south|A09S': { a: 0, m: 2, e: 0 } }, // matched 2 < MIN_EFF_MATCHED
      vehicles: {},
      regimes: {},
    };
    expect(deriveStationFlow(state, params).stations).toEqual({});
  });

  test('a station whose every segment is quiet reads quiet, not flowing', () => {
    const state: SegmentFlowDoc = {
      observed_at: NOW,
      cells: {},
      vehicles: { F: 5 },
      regimes: {},
    };
    const doc = deriveStationFlow(state, fitted);
    // Both cells expect nothing over an empty window, so nothing is degraded
    // and nothing is proven flowing.
    expect(doc.stations['A09']!.status).toBe('quiet');
    expect(doc.stations['A11']!.status).toBe('quiet');
  });

  test('one disrupted segment outranks quiet neighbours', () => {
    const state: SegmentFlowDoc = {
      observed_at: NOW,
      cells: { 'F|south|A09S': { a: 0, m: 0, e: 5 } },
      vehicles: { F: 5 },
      regimes: {},
    };
    const doc = deriveStationFlow(state, fitted);
    expect(doc.stations['A09']!.status).toBe('degraded');
    expect(doc.stations['A09']!.worst_deficit).toBe(1);
  });
});

describe('deriveSegmentStates', () => {
  test('a disrupted cell reads disrupted, keyed the same as cells', () => {
    const state: SegmentFlowDoc = {
      observed_at: NOW,
      cells: { 'F|south|A09S': { a: 1, m: 40, e: 0 } }, // ~0.025 vs p0 0.9
      vehicles: {},
      regimes: {},
    };
    expect(deriveSegmentStates(state, params)).toEqual({ 'F|south|A09S': 'disrupted' });
  });

  test('with no throughput fit, a cell below the matched floor is absent', () => {
    const state: SegmentFlowDoc = {
      observed_at: NOW,
      cells: { 'F|south|A09S': { a: 0, m: 2, e: 0 } }, // matched 2 < MIN_EFF_MATCHED
      vehicles: {},
      regimes: {},
    };
    expect(deriveSegmentStates(state, params)).toEqual({});
  });

  test('a cell with no adjacency entry is absent', () => {
    const orphan: SegmentParamsDoc = {
      ...params,
      cells: { 'F|south|Z99S': { p0: 0.9, n: 1000 } },
    };
    const state: SegmentFlowDoc = {
      observed_at: NOW,
      cells: { 'F|south|Z99S': { a: 40, m: 40, e: 0 } },
      vehicles: {},
      regimes: {},
    };
    expect(deriveSegmentStates(state, orphan)).toEqual({});
  });

  test('every baselined cell gets a call once the throughput fit lands', () => {
    const state: SegmentFlowDoc = {
      observed_at: NOW,
      // A09S is silent against a real expectation; A10S expects nothing here.
      cells: { 'F|south|A09S': { a: 0, m: 0, e: 5 } },
      vehicles: { F: 5 },
      regimes: {},
    };
    expect(deriveSegmentStates(state, fitted)).toEqual({
      'F|south|A09S': 'disrupted',
      'F|south|A10S': 'quiet',
    });
  });

  test('the advance branch wins wherever it has an opinion', () => {
    const state: SegmentFlowDoc = {
      observed_at: NOW,
      // Trains present and moving on, yet far below the expected throughput:
      // flow is what this surface publishes, so it reads normal.
      cells: { 'F|south|A09S': { a: 38, m: 40, e: 500 } },
      vehicles: { F: 5 },
      regimes: {},
    };
    expect(deriveSegmentStates(state, fitted)['F|south|A09S']).toBe('normal');
  });

  test('the throughput branch covers an advance-branch abstention', () => {
    // A degenerate-low p0 makes classifyAdvance abstain on a zero-advance
    // window (it can't distinguish a stall from normal), but the trains are
    // arriving, and the throughput branch can say so.
    const degenerate: SegmentParamsDoc = {
      ...fitted,
      cells: { 'F|south|A09S': { p0: 0.01, n: 1000, lam: { [BIN]: 1 } } },
    };
    const state: SegmentFlowDoc = {
      observed_at: NOW,
      cells: { 'F|south|A09S': { a: 0, m: 10, e: 5 } },
      vehicles: { F: 5 },
      regimes: {},
    };
    expect(deriveSegmentStates(state, degenerate)).toEqual({ 'F|south|A09S': 'normal' });
  });
});

describe('pruneSegmentRegimes', () => {
  function entry(state: SegmentCondition): RegimeEntry<SegmentCondition> {
    return {
      state,
      entered_at: NOW,
      last_seen_at: NOW,
      pending: null,
      pending_since: 0,
      pending_run: 0,
    };
  }

  test('keeps regimes for baselined cells, including ones abstaining this tick', () => {
    const kept = pruneSegmentRegimes(
      { 'F|south|A09S': entry('disrupted'), 'F|south|A10S': entry('quiet') },
      params.cells,
    );
    expect(Object.keys(kept).sort()).toEqual(['F|south|A09S', 'F|south|A10S']);
  });

  test('drops a regime whose cell a retrain removed from the baseline', () => {
    const kept = pruneSegmentRegimes({ 'F|south|Z99S': entry('normal') }, params.cells);
    expect(kept).toEqual({});
  });
});
