import { describe, expect, test } from 'vitest';

import {
  deriveSegmentStates,
  deriveStationFlow,
  SEGMENT_DECAY,
  stationId,
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

describe('stationId', () => {
  test('strips a trailing N/S, leaves others', () => {
    expect(stationId('A09S')).toBe('A09');
    expect(stationId('A09N')).toBe('A09');
    expect(stationId('A09')).toBe('A09');
  });
});

describe('updateSegmentFlow', () => {
  test('accumulates advanced/matched, ignoring stalls in the advance count', () => {
    const rows = new Map([['F', moveRow({ 'A09S>A10S': 10, 'A09S>A09S': 2 })]]);
    const state = updateSegmentFlow(null, rows, 100, params);
    expect(state.cells['F|south|A09S']).toEqual({ a: 10, m: 12 });
  });

  test('decays the carried accumulator by SEGMENT_DECAY', () => {
    const prev: SegmentFlowDoc = {
      observed_at: 95,
      cells: { 'F|south|A09S': { a: 10, m: 12 } },
      regimes: {},
    };
    const rows = new Map([['F', moveRow({ 'A09S>A09S': 5 })]]); // all stalls this tick
    const state = updateSegmentFlow(prev, rows, 100, params);
    expect(state.cells['F|south|A09S']!.a).toBeCloseTo(SEGMENT_DECAY * 10, 6);
    expect(state.cells['F|south|A09S']!.m).toBeCloseTo(5 + SEGMENT_DECAY * 12, 6);
  });

  test('tracks only segments the trainer baselined', () => {
    const rows = new Map([['F', moveRow({ 'Z99S>Z98S': 20 })]]); // not in params.cells
    const state = updateSegmentFlow(null, rows, 100, params);
    expect(state.cells['F|south|Z99S']).toBeUndefined();
  });
});

describe('deriveStationFlow', () => {
  test('frozen segment degrades both endpoint stations', () => {
    const state: SegmentFlowDoc = {
      observed_at: 100,
      cells: { 'F|south|A09S': { a: 1, m: 40 } }, // ~0.025 vs p0 0.9
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
      observed_at: 100,
      cells: { 'F|south|A09S': { a: 38, m: 40 } }, // ~0.95 vs p0 0.9
      regimes: {},
    };
    const doc = deriveStationFlow(state, params);
    expect(doc.stations['A09']!.status).toBe('flowing');
  });

  test('segments below the effective-matched floor are skipped', () => {
    const state: SegmentFlowDoc = {
      observed_at: 100,
      cells: { 'F|south|A09S': { a: 0, m: 3 } }, // matched 3 < MIN_EFF_MATCHED
      regimes: {},
    };
    expect(deriveStationFlow(state, params).stations).toEqual({});
  });
});

describe('deriveSegmentStates', () => {
  test('a disrupted cell reads disrupted, keyed the same as cells', () => {
    const state: SegmentFlowDoc = {
      observed_at: 100,
      cells: { 'F|south|A09S': { a: 1, m: 40 } }, // ~0.025 vs p0 0.9
      regimes: {},
    };
    expect(deriveSegmentStates(state, params)).toEqual({ 'F|south|A09S': 'disrupted' });
  });

  test('a cell below the effective-matched floor is absent, not a reading', () => {
    const state: SegmentFlowDoc = {
      observed_at: 100,
      cells: { 'F|south|A09S': { a: 0, m: 3 } }, // matched 3 < MIN_EFF_MATCHED
      regimes: {},
    };
    expect(deriveSegmentStates(state, params)).toEqual({});
  });

  test('an untracked cell (no baseline/adjacency) is absent', () => {
    const state: SegmentFlowDoc = {
      observed_at: 100,
      cells: { 'F|south|Z99S': { a: 40, m: 40 } },
      regimes: {},
    };
    expect(deriveSegmentStates(state, params)).toEqual({});
  });
});
