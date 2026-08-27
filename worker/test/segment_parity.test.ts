/**
 * Cross-language parity: the Worker's segment classifier must reproduce, tick for
 * tick, the accumulator state and the calls Python recorded in
 * tests/fixtures/parity_segment_flow.json.
 *
 * Drift here is expensive in a specific way. training/segment_replay.py is what
 * grades a change to this classifier BEFORE it ships; a replay that is a
 * near-miss of the Worker produces a grade about a model that never runs, which
 * is worse than no grade at all.
 *
 * Regenerate the fixture with:
 *   uv run python -m scripts.gen_segment_parity_fixture
 */

import { describe, expect, test } from 'vitest';

import fixture from '../../tests/fixtures/parity_segment_flow.json';
import { deriveSegmentStates, updateSegmentFlow } from '../src/segment_flow';
import type { SegmentFlowDoc, SegmentParamsDoc } from '../src/state';
import type { DirMovementRow, MovementRow } from '../src/vehicles';

/**
 * The fixture records reduced per-cell (advanced, matched) counts and per-route
 * vehicle totals. Rebuild the MovementRow shape tickCounts/tickVehicles actually
 * read, so the parity check goes through the production entry points rather than
 * a private one: every count is put on the south direction as a `from>to` pair,
 * with the stalls split back out as `from>from`.
 */
function moveRows(
  counts: Record<string, number[]>,
  vehicles: Record<string, number>,
): Map<string, MovementRow> {
  const byRoute = new Map<string, MovementRow>();
  const ensure = (route: string): MovementRow => {
    let row = byRoute.get(route);
    if (!row) {
      const dir = (): DirMovementRow => ({
        vehicles_n: 0,
        advanced_n: 0,
        stalled_n: 0,
        transitions: {},
      });
      row = {
        vehicles_n: vehicles[route] ?? 0,
        stopped_n: 0,
        moving_n: 0,
        advanced_n: 0,
        stalled_n: 0,
        by_direction: { north: dir(), south: dir() },
      };
      byRoute.set(route, row);
    }
    return row;
  };
  for (const route of Object.keys(vehicles)) ensure(route);
  for (const [key, [advanced = 0, matched = 0]] of Object.entries(counts)) {
    const [route = '', direction = '', from = ''] = key.split('|');
    const row = ensure(route);
    const dir = row.by_direction[direction as 'north' | 'south'];
    if (advanced > 0) dir.transitions[`${from}>DEST`] = advanced;
    const stalled = matched - advanced;
    if (stalled > 0) dir.transitions[`${from}>${from}`] = stalled;
  }
  return byRoute;
}

describe('segment classifier parity with training/segment_replay.py', () => {
  test('reproduces every recorded tick', () => {
    const params = fixture.params as unknown as SegmentParamsDoc;
    let flow: SegmentFlowDoc | null = null;
    for (const step of fixture.steps) {
      flow = updateSegmentFlow(
        flow,
        moveRows(step.counts, step.vehicles),
        step.observed_at,
        params,
      );
      const cells = Object.fromEntries(
        Object.entries(flow.cells).sort(([a], [b]) => (a < b ? -1 : 1)),
      );
      // Python and JS both hold IEEE doubles and both fold the same operations
      // in the same order, so the sums must agree exactly, not approximately.
      expect(cells).toEqual(step.cells);
      expect(
        Object.fromEntries(Object.entries(flow.vehicles).sort(([a], [b]) => (a < b ? -1 : 1))),
      ).toEqual(step.state_vehicles);
      const calls = deriveSegmentStates(flow, params);
      expect(
        Object.fromEntries(Object.entries(calls).sort(([a], [b]) => (a < b ? -1 : 1))),
      ).toEqual(step.calls);
    }
  });
});
