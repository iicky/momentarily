/**
 * Observed headway at the canonical reference stops: the reference-stop rule,
 * the passing detection and headway derivation, and (via buildSnapshot) the
 * publish-time freshness gates. Mirrors crowding.test.ts's split — pure
 * selection, pure state update, pure derivation, then the published surface.
 *
 * The reference-stop parity block reads viz/public/diagram.json, which is the
 * repo's committed copy of exactly the `route_stops` the Worker reads off
 * state/segment_params.json (training/diagram.py builds both from
 * gtfs_static.route_patterns). It is the only place the real scheduled
 * patterns live in-repo, and the point of that block is that this rule agrees
 * with training/headway.select_reference_stops on real data rather than on a
 * fixture shaped to make it agree.
 */

import { describe, expect, test } from 'vitest';

import diagram from '../../viz/public/diagram.json';

import {
  DUP_ARRIVAL_SECONDS,
  FEED_GAP_SECONDS,
  MAX_HEADWAY_SECONDS,
  MAX_READING_AGE_SECONDS,
  MIN_HEADWAY_SECONDS,
  REFERENCE_REFRESH_SECONDS,
  TRIP_GAP_SECONDS,
  detectPassings,
  headwayObservations,
  mergePassings,
  referenceStopsStale,
  resolveReference,
  selectFallbackStops,
  selectReferenceStops,
  updateHeadwayState,
} from '../src/headway';
import type { HeadwayReference, RouteStops } from '../src/headway';
import { TICK_SECONDS, buildSnapshot } from '../src/snapshot';
import type { HeadwayStateDoc } from '../src/state';
import type { TraceRow } from '../src/vehicles';

const NOW = 1_700_000_000;
const MIN = 60;

/** One trace row. `stopped` is irrelevant to the passing rule (which keys on
 * the stop_id transition) and is varied in the tests that say so. */
function row(
  tripId: string,
  stopId: string,
  opts: {
    route?: string;
    direction?: 'north' | 'south' | null;
    stopped?: boolean;
    vehicleTs?: number | null;
  } = {},
): TraceRow {
  const stopped = opts.stopped ?? false;
  return {
    trip_id: tripId,
    route_id: opts.route ?? '1',
    direction: opts.direction === undefined ? 'north' : opts.direction,
    stop_id: stopId,
    stop_seq: stopped ? 20 : null,
    stopped,
    vehicle_ts: opts.vehicleTs ?? null,
  };
}

/** The reference block for a one-cell test: 1|north measured at 121N. */
const REF: HeadwayReference = {
  stops: { '1|north': '121N' },
  at: NOW - 3600,
  trained_at: 1_699_000_000,
};

/** The reading a cell would publish right now, ignoring the age gate — the
 * headway is a DERIVED value over the cell's passing ledger, so tests assert
 * the derivation rather than a stored scalar. */
function reading(doc: HeadwayStateDoc, cell = '1|north'): number | undefined {
  const [route, direction] = cell.split('|');
  return headwayObservations(doc, doc.observed_at).find(
    (o) => o.entity_ref === `subway_route:${route}` && o.direction === direction,
  )?.value;
}

/** The trip/time of the cell's most recent recorded passing. */
function newestTrip(doc: HeadwayStateDoc, cell = '1|north'): string | undefined {
  return doc.cells[cell]?.passings.at(-1)?.trip;
}
function newestAt(doc: HeadwayStateDoc, cell = '1|north'): number | undefined {
  return doc.cells[cell]?.passings.at(-1)?.at;
}

/** Fold a sequence of [pollOffsetSeconds, rows] polls in order. */
function replay(
  polls: [number, TraceRow[]][],
  reference: HeadwayReference = REF,
): HeadwayStateDoc {
  let doc: HeadwayStateDoc | null = null;
  for (const [offset, rows] of polls) {
    doc = updateHeadwayState(rows, reference, doc, NOW + offset);
  }
  return doc!;
}

describe('selectReferenceStops', () => {
  test('picks the max-scheduled-trips through-stop, not a terminal', () => {
    // Two patterns over one line: the local runs the whole thing, a short-turn
    // runs the first half. B and C are on both, so they carry the most trips;
    // A and E are terminals (no predecessor / no successor) and D only carries
    // the local. Between B and C, both at 12 trips, the tie breaks toward the
    // middle of the most-run pattern (index 2 of 5, mid 2.5): C at index 2 is
    // 0.5 away, B at index 1 is 1.5 away.
    const patterns: RouteStops = {
      '1|north': [
        { stops: ['A', 'B', 'C', 'D', 'E'], n_trips: 10 },
        { stops: ['A', 'B', 'C'], n_trips: 2 },
      ],
    };
    const picked = selectReferenceStops(patterns);
    expect(picked['1|north']?.stop_id).toBe('C');
    expect(picked['1|north']?.n_scheduled_trips).toBe(12);
    expect(picked['1|north']?.coverage).toBe(1);
  });

  test('a partially-served stop loses to a stop every pattern reaches', () => {
    // The local skips C, so C carries 4 of the 14 trips while both B and D
    // carry all 14 — and a stop 10 of the cell's trains never call at cannot
    // carry a headway series for the cell. B and D tie on trips, and the
    // tie-break sends it to D: the most-run pattern is [A,B,D,E], mid 2, so D
    // (index 2) sits nearer the middle than B (index 1).
    const patterns: RouteStops = {
      '1|north': [
        { stops: ['A', 'B', 'D', 'E'], n_trips: 10 },
        { stops: ['A', 'B', 'C', 'D', 'E'], n_trips: 4 },
      ],
    };
    const picked = selectReferenceStops(patterns);
    expect(picked['1|north']?.stop_id).toBe('D');
    expect(picked['1|north']?.n_scheduled_trips).toBe(14);
    expect(picked['1|north']?.coverage).toBe(1);
  });

  test('a cell with no through-stop abstains rather than falling back to a terminal', () => {
    const picked = selectReferenceStops({
      'FS|north': [{ stops: ['S01N', 'S02N'], n_trips: 100 }],
      'X|north': [],
    });
    expect(picked['FS|north']).toBeUndefined();
    expect(picked['X|north']).toBeUndefined();
  });

  test('is deterministic in the patterns, whatever order they arrive in', () => {
    const forward: RouteStops = {
      '1|north': [
        { stops: ['A', 'B', 'C', 'D'], n_trips: 5 },
        { stops: ['A', 'B', 'C', 'D'], n_trips: 5 },
      ],
    };
    const a = selectReferenceStops(forward)['1|north'];
    const b = selectReferenceStops(forward)['1|north'];
    expect(a).toEqual(b);
  });
});

describe('selectReferenceStops on the real scheduled patterns', () => {
  const routeStops: RouteStops = diagram.route_stops;
  const picked = selectReferenceStops(routeStops);

  test('resolved cells are measured at a stop every pattern serves', () => {
    for (const [key, ref] of Object.entries(picked)) {
      // Coverage 1.0 is the property that makes the series miss no train: the
      // stop is served by every express, local and branch pattern of the cell.
      expect(ref.coverage, key).toBe(1);
      expect(ref.n_scheduled_trips, key).toBeGreaterThan(0);
    }
  });

  test('the only cells that abstain are the two-stop shuttle, which has no interior stop', () => {
    // GS is 42nd St-Grand Central to Times Sq: two stops, both terminals, so
    // there is no through-stop to measure at and no headway series is
    // published for it. Every other route/direction resolves.
    const unresolved = Object.keys(routeStops)
      .filter((key) => picked[key] === undefined)
      .sort();
    expect(unresolved).toEqual(['GS|north', 'GS|south']);
    for (const key of unresolved) {
      expect(routeStops[key]!.every((p) => p.stops.length <= 2), key).toBe(true);
    }
  });

  test('no pick is a terminal, on the real patterns', () => {
    // The property acceptance turns on: a terminal's layover dwell and
    // repeated re-reporting of the same standing train distort the gap, so the
    // rule gates candidates on being through-stops BEFORE ranking them by
    // scheduled trips. Recomputing the terminal set here independently of the
    // rule — sources and sinks of each cell's dominant-successor skeleton,
    // which is what training/gtfs_static.terminals names — and asserting the
    // pick is never one, rather than trusting the gate by inspection.
    for (const [key, ref] of Object.entries(picked)) {
      const succ = new Map<string, Map<string, number>>();
      for (const pattern of routeStops[key]!) {
        for (let i = 0; i + 1 < pattern.stops.length; i++) {
          const from = pattern.stops[i]!;
          let tos = succ.get(from);
          if (tos === undefined) {
            tos = new Map<string, number>();
            succ.set(from, tos);
          }
          const to = pattern.stops[i + 1]!;
          tos.set(to, (tos.get(to) ?? 0) + pattern.n_trips);
        }
      }
      const dominant = new Map<string, string>();
      for (const [from, tos] of succ) {
        let bestTo = '';
        let bestN = -1;
        for (const [to, n] of tos) {
          if (n > bestN || (n === bestN && to < bestTo)) {
            bestTo = to;
            bestN = n;
          }
        }
        dominant.set(from, bestTo);
      }
      const incoming = new Set(dominant.values());
      // A terminal is a source (no scheduled predecessor) or a sink (no
      // scheduled successor).
      const isTerminal = !dominant.has(ref.stop_id) || !incoming.has(ref.stop_id);
      expect(isTerminal, `${key} picked terminal ${ref.stop_id}`).toBe(false);
    }
  });

  test('agrees with the offline rule on its documented picks', () => {
    // training/headway.select_reference_stops' own output on this feed
    // (diagram.json feed_version 20260807-H-rockaways-extension-removed). If
    // the committed diagram is regenerated from a feed where these stops move,
    // this is the assertion to update — after checking the offline rule moved
    // the same way, which is the whole point of pinning it here.
    expect(diagram.feed_version.version).toBe('20260807-H-rockaways-extension-removed');
    expect(picked['1|north']?.stop_id).toBe('121N');
    expect(picked['2|north']?.stop_id).toBe('120N');
    expect(picked['7|south']?.stop_id).toBe('714S');
    expect(picked['L|north']?.stop_id).toBe('L15N');
    expect(picked['L|south']?.stop_id).toBe('L16S');
    expect(picked['A|north']?.stop_id).toBe('A55N');
    expect(picked['A|south']?.stop_id).toBe('A55S');
  });
});

describe('reference refresh', () => {
  test('recomputes when never computed, or past the refresh cadence', () => {
    expect(referenceStopsStale(null, NOW)).toBe(true);
    const doc: HeadwayStateDoc = {
      observed_at: NOW,
      reference_at: NOW,
      reference_trained_at: 1,
      reference_stops: { '1|north': '121N' },
      reference_fallbacks: {},
      cells: {},
      trips: {},
      gaps: [],
    };
    expect(referenceStopsStale({ ...doc, reference_stops: {} }, NOW)).toBe(true);
    expect(referenceStopsStale(doc, NOW + REFERENCE_REFRESH_SECONDS - 1)).toBe(false);
    expect(referenceStopsStale(doc, NOW + REFERENCE_REFRESH_SECONDS)).toBe(true);
  });

  test('a trainer doc with no stopping patterns never blanks a live map', () => {
    const doc: HeadwayStateDoc = {
      observed_at: NOW,
      reference_at: NOW - 1,
      reference_trained_at: 7,
      reference_stops: { '1|north': '121N' },
      reference_fallbacks: {},
      cells: {},
      trips: {},
      gaps: [],
    };
    // The observed-adjacency fallback publishes route_stops as {}.
    const kept = resolveReference(doc, {}, 9, NOW);
    expect(kept).toEqual({
      stops: { '1|north': '121N' },
      fallbacks: {},
      at: NOW - 1,
      trained_at: 7,
    });
    // Nothing read this poll: also keep.
    expect(resolveReference(doc, null, 9, NOW)).toEqual(kept);
    // Real patterns: adopt them and stamp the provenance.
    const fresh = resolveReference(doc, { '1|north': [{ stops: ['A', 'B', 'C'], n_trips: 4 }] }, 9, NOW);
    expect(fresh).toEqual({
      stops: { '1|north': 'B' },
      fallbacks: {},
      at: NOW,
      trained_at: 9,
    });
  });

  test('with nothing carried and nothing readable, the map is empty', () => {
    expect(resolveReference(null, null, 0, NOW)).toEqual({
      stops: {},
      fallbacks: {},
      at: 0,
      trained_at: 0,
    });
  });
});

describe('passing detection', () => {
  test('a passing is the stop_id transition, so a dwell shorter than the poll still counts', () => {
    // T1 is only ever seen heading to 121N, never STOPPED_AT it — the dwell
    // came and went between two polls. Keying on stopped=true would drop this
    // passing entirely and merge two real headways into one double.
    const doc = replay([
      [0, [row('T1', '121N')]],
      [MIN, [row('T1', '122N')]],
    ]);
    expect(newestTrip(doc, '1|north')).toBe('T1');
    expect(newestAt(doc, '1|north')).toBe(NOW + MIN);
  });

  test('a train standing at the reference stop for several polls is one passing', () => {
    const doc = replay([
      [0, [row('T1', '121N', { stopped: true })]],
      [MIN, [row('T1', '121N', { stopped: true })]],
      [2 * MIN, [row('T1', '121N', { stopped: true })]],
      [3 * MIN, [row('T1', '122N')]],
      // T2 five minutes behind T1.
      [4 * MIN, [row('T2', '121N', { stopped: true })]],
      [8 * MIN, [row('T2', '122N')]],
    ]);
    // One headway, T1's departure to T2's: 8 min - 3 min.
    expect(reading(doc, '1|north')).toBe(5 * MIN);
    expect(newestTrip(doc, '1|north')).toBe('T2');
  });

  test('the first train seen yields no observation, and never a zero', () => {
    const doc = replay([
      [0, [row('T1', '121N')]],
      [MIN, [row('T1', '122N')]],
    ]);
    expect(reading(doc, '1|north')).toBeUndefined();
    expect(headwayObservations(doc, NOW + MIN)).toEqual([]);
  });

  test('a route with a single active train still yields nothing rather than a stale gap', () => {
    // T1 passes, and nothing follows for half an hour. There is no second
    // train, so there is no headway — not a 30-minute one, and not a zero.
    const doc = replay([
      [0, [row('T1', '121N')]],
      [MIN, [row('T1', '122N')]],
      [2 * MIN, []],
      [3 * MIN, []],
    ]);
    expect(reading(doc, '1|north')).toBeUndefined();
    expect(headwayObservations(doc, NOW + 30 * MIN)).toEqual([]);
  });

  test('a trip that vanishes for a few polls keeps its pending departure', () => {
    // T1 is at 121N, then missing from the feed for two polls (an individual
    // vehicle dropping out, not a feed gap), then seen past the stop.
    const doc = replay([
      [0, [row('T2', '121N')]],
      [MIN, [row('T2', '122N')]],
      [2 * MIN, [row('T1', '121N')]],
      [3 * MIN, [row('T9', '104N')]],
      [4 * MIN, [row('T9', '104N')]],
      [5 * MIN, [row('T1', '122N')]],
    ]);
    expect(reading(doc, '1|north')).toBe(4 * MIN);
  });

  test('a carry older than the trip gap is not trusted to have just departed', () => {
    // T1 sits in the carry at 121N, then reappears past it well after
    // TRIP_GAP_SECONDS — NYCT reuses trip_ids, so this may be a different
    // train and crediting it would invent a passing.
    const doc = replay([
      [0, [row('T1', '121N')]],
      [TRIP_GAP_SECONDS + MIN, [row('T1', '122N')]],
    ]);
    expect(doc.cells['1|north']).toBeUndefined();
    expect(doc.trips).toEqual({});
  });

  test('the same trip re-reported within the dup window is not a second train', () => {
    // T0 passes, then T1 passes 5 min later. T1 is then re-reported back at
    // 121N and departs again 90s after its own first departure — inside
    // DUP_ARRIVAL_SECONDS, so it is one train the feed said twice, not a
    // 90-second headway.
    const doc = replay([
      [0, [row('T0', '121N')]],
      [MIN, [row('T0', '122N', { vehicleTs: NOW + MIN })]],
      [5 * MIN, [row('T1', '121N')]],
      [6 * MIN, [row('T1', '122N', { vehicleTs: NOW + 6 * MIN })]],
      [7 * MIN, [row('T1', '121N')]],
      [8 * MIN, [row('T1', '122N', { vehicleTs: NOW + 6 * MIN + 90 })]],
    ]);
    expect(90).toBeLessThan(DUP_ARRIVAL_SECONDS);
    // The real T0 -> T1 headway survives, and the re-report neither
    // overwrites it nor moves the interval's open end.
    expect(reading(doc, '1|north')).toBe(5 * MIN);
    expect(newestTrip(doc, '1|north')).toBe('T1');
    expect(newestAt(doc, '1|north')).toBe(NOW + 6 * MIN);
  });

  test('rows with no direction, and cells with no reference stop, are ignored', () => {
    const doc = replay([
      [0, [row('T1', '121N', { direction: null }), row('T2', 'G22N', { route: 'G' })]],
      [MIN, [row('T1', '122N', { direction: null }), row('T2', 'G23N', { route: 'G' })]],
    ]);
    expect(doc.cells).toEqual({});
  });

  test('north and south are separate cells measured at their own stops', () => {
    const reference: HeadwayReference = {
      stops: { '1|north': '121N', '1|south': '122S' },
      at: NOW - 3600,
      trained_at: 1,
    };
    const doc = replay(
      [
        [0, [row('N1', '121N'), row('S1', '122S', { direction: 'south' })]],
        [MIN, [row('N1', '122N'), row('S1', '121S', { direction: 'south' })]],
        [4 * MIN, [row('N2', '121N'), row('S2', '122S', { direction: 'south' })]],
        [5 * MIN, [row('N2', '122N')]],
        [9 * MIN, [row('S2', '121S', { direction: 'south' })]],
      ],
      reference,
    );
    expect(reading(doc, '1|north')).toBe(4 * MIN);
    expect(reading(doc, '1|south')).toBe(8 * MIN);
  });

  test('two trains clearing the stop in one poll are folded in time order', () => {
    const doc = replay([
      [0, [row('T1', '121N'), row('T2', '121N')]],
      // Both report past the stop on the same poll, with distinct vehicle
      // timestamps 90s apart. T1's departure opens the interval T2 closes.
      [
        3 * MIN,
        [
          row('T2', '122N', { vehicleTs: NOW + 3 * MIN }),
          row('T1', '122N', { vehicleTs: NOW + 3 * MIN - 90 }),
        ],
      ],
    ]);
    expect(reading(doc, '1|north')).toBe(90);
    expect(newestTrip(doc, '1|north')).toBe('T2');
  });
});

describe('headway sanity bounds and feed gaps', () => {
  test('an interval crossing a poll gap is refused, not reported as a long wait', () => {
    const doc = replay([
      [0, [row('T1', '121N')]],
      [MIN, [row('T1', '122N')]],
      // Nothing folded in for well over FEED_GAP_SECONDS: the Worker was not
      // polling, so the interval is unobserved rather than long.
      [MIN + FEED_GAP_SECONDS + MIN, [row('T2', '121N')]],
      [MIN + FEED_GAP_SECONDS + 2 * MIN, [row('T2', '122N')]],
    ]);
    expect(reading(doc, '1|north')).toBeUndefined();
    expect(newestTrip(doc, '1|north')).toBe('T2');
    // The gap is recorded as a window on the document, not a per-cell flag.
    expect(doc.gaps).toHaveLength(1);
    expect(doc.gaps[0]).toEqual({ from: NOW + MIN, until: NOW + MIN + FEED_GAP_SECONDS + MIN });
  });

  test('the gap flag clears, so the next fully-observed interval publishes', () => {
    const doc = replay([
      [0, [row('T1', '121N')]],
      [MIN, [row('T1', '122N')]],
      [MIN + FEED_GAP_SECONDS + MIN, [row('T2', '121N')]],
      [MIN + FEED_GAP_SECONDS + 2 * MIN, [row('T2', '122N')]],
      [MIN + FEED_GAP_SECONDS + 5 * MIN, [row('T3', '121N')]],
      [MIN + FEED_GAP_SECONDS + 6 * MIN, [row('T3', '122N')]],
    ]);
    expect(reading(doc, '1|north')).toBe(4 * MIN);
  });

  test('a below-bound gap is refused — one train under two trip ids', () => {
    // Same physical train re-issued under a new trip_id 20s later: the dup
    // guard cannot see it (the ids differ), so the sanity bound is what
    // catches it.
    expect(MIN_HEADWAY_SECONDS).toBe(30);
    const doc = replay([
      [0, [row('T0', '121N')]],
      [MIN, [row('T0', '122N')]],
      [4 * MIN, [row('T1', '121N'), row('T1b', '121N')]],
      [
        5 * MIN,
        [
          row('T1', '122N', { vehicleTs: NOW + 5 * MIN - 20 }),
          row('T1b', '122N', { vehicleTs: NOW + 5 * MIN }),
        ],
      ],
    ]);
    // The ledger records BOTH sightings — it is the observation log, and it
    // does not editorialise. The derivation is what collapses them: T1b sits
    // 20s after T1, closer than two trains of one route can clear a platform,
    // so it reads as T1 re-reported under a reassigned id and the published
    // reading stays the true T0 -> T1 interval.
    expect(newestTrip(doc, '1|north')).toBe('T1b');
    expect(doc.cells['1|north']?.passings.map((p) => p.trip)).toEqual(['T0', 'T1', 'T1b']);
    expect(reading(doc, '1|north')).toBe(4 * MIN - 20);
  });

  test('an above-bound gap is refused rather than published as a three-hour headway', () => {
    const reference: HeadwayReference = { ...REF };
    // Fold polls every minute so no FEED_GAP flag is raised: the long
    // interval is genuinely observed, and still not a headway anyone waited.
    let doc: HeadwayStateDoc | null = null;
    const span = MAX_HEADWAY_SECONDS + 10 * MIN;
    for (let t = 0; t <= span; t += MIN) {
      const rows: TraceRow[] =
        t === 0
          ? [row('T1', '121N')]
          : t === MIN
            ? [row('T1', '122N')]
            : t === span - MIN
              ? [row('T2', '121N')]
              : t === span
                ? [row('T2', '122N')]
                : [row('T9', '999N')];
      doc = updateHeadwayState(rows, reference, doc, NOW + t);
    }
    expect(reading(doc!, '1|north')).toBeUndefined();
    expect(newestTrip(doc!, '1|north')).toBe('T2');
  });

  test('a measurement point that moves starts a new series', () => {
    const first = replay([
      [0, [row('T1', '121N')]],
      [MIN, [row('T1', '122N')]],
      [4 * MIN, [row('T2', '121N')]],
      [5 * MIN, [row('T2', '122N')]],
    ]);
    expect(reading(first, '1|north')).toBe(4 * MIN);
    // The trainer republishes and the rule now picks 120N. The old cell's
    // readings were taken somewhere else and are dropped, not reinterpreted.
    const moved = updateHeadwayState(
      [row('T3', '120N')],
      { stops: { '1|north': '120N' }, at: NOW + 6 * MIN, trained_at: 2 },
      first,
      NOW + 6 * MIN,
    );
    expect(moved.cells['1|north']).toBeUndefined();
    expect(moved.reference_stops).toEqual({ '1|north': '120N' });
    expect(moved.reference_trained_at).toBe(2);
  });

  test('a frozen vehicle clock falls back to the poll stamp', () => {
    const doc = replay([
      [0, [row('T1', '121N')]],
      [MIN, [row('T1', '122N', { vehicleTs: NOW - 86_400 })]],
      [4 * MIN, [row('T2', '121N')]],
      [5 * MIN, [row('T2', '122N', { vehicleTs: NOW - 86_400 })]],
    ]);
    // Both stamps fall back to the poll, so the headway is the poll delta —
    // not the zero two identical frozen timestamps would have produced.
    expect(reading(doc, '1|north')).toBe(4 * MIN);
  });

  test('a replayed poll reproduces the same document', () => {
    const polls: [number, TraceRow[]][] = [
      [0, [row('T1', '121N')]],
      [MIN, [row('T1', '122N', { vehicleTs: NOW + MIN })]],
      [4 * MIN, [row('T2', '121N')]],
    ];
    const once = replay(polls);
    const twice = updateHeadwayState(
      [row('T2', '121N')],
      REF,
      replay(polls.slice(0, 2)),
      NOW + 4 * MIN,
    );
    expect(twice).toEqual(once);
  });
});

describe('headwayObservations', () => {
  const doc = replay([
    [0, [row('T1', '121N')]],
    [MIN, [row('T1', '122N', { vehicleTs: NOW + MIN })]],
    [4 * MIN, [row('T2', '121N')]],
    [5 * MIN, [row('T2', '122N', { vehicleTs: NOW + 5 * MIN })]],
  ]);

  test('emits the measurement with its point, direction and upstream', () => {
    expect(headwayObservations(doc, NOW + 5 * MIN)).toEqual([
      {
        entity_ref: 'subway_route:1',
        kind: 'headway',
        value: 4 * MIN,
        unit: 'seconds',
        observed_at: NOW + 5 * MIN,
        source: 'gtfs_rt_vehicle_positions',
        direction: 'north',
        stop_id: '121N',
      },
    ]);
  });

  test('a reading past the age bound is dropped, not republished as current', () => {
    const at = NOW + 5 * MIN;
    expect(headwayObservations(doc, at + MAX_READING_AGE_SECONDS).length).toBe(1);
    expect(headwayObservations(doc, at + MAX_READING_AGE_SECONDS + 1)).toEqual([]);
  });

  test('output is sorted by cell, so an unchanged surface is byte-stable', () => {
    const reference: HeadwayReference = {
      stops: { '7|south': '714S', '1|north': '121N' },
      at: NOW - 3600,
      trained_at: 1,
    };
    const both = replay(
      [
        [0, [row('N1', '121N'), row('S1', '714S', { route: '7', direction: 'south' })]],
        [MIN, [row('N1', '122N'), row('S1', '715S', { route: '7', direction: 'south' })]],
        [4 * MIN, [row('N2', '121N'), row('S2', '714S', { route: '7', direction: 'south' })]],
        [5 * MIN, [row('N2', '122N'), row('S2', '715S', { route: '7', direction: 'south' })]],
      ],
      reference,
    );
    expect(headwayObservations(both, NOW + 5 * MIN).map((o) => [o.entity_ref, o.direction]))
      .toEqual([
        ['subway_route:1', 'north'],
        ['subway_route:7', 'south'],
      ]);
  });
});

describe('the published observations surface', () => {
  function snapshotWith(
    headway: HeadwayStateDoc | null,
    generatedAt: number,
    vehiclePositionsFreshness: number | null = generatedAt,
  ) {
    return buildSnapshot({
      generatedAt,
      alertsFreshness: generatedAt,
      routeSnapshots: new Map(),
      rolls: {},
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      headway,
      vehiclePositionsFreshness,
    });
  }

  const measured = replay([
    [0, [row('T1', '121N')]],
    [MIN, [row('T1', '122N', { vehicleTs: NOW + MIN })]],
    [4 * MIN, [row('T2', '121N')]],
    [5 * MIN, [row('T2', '122N', { vehicleTs: NOW + 5 * MIN })]],
  ]);

  test('a fresh measurement reaches the snapshot as an observation', () => {
    const snap = snapshotWith(measured, NOW + 5 * MIN);
    expect(snap.observations).toEqual([
      {
        entity_ref: 'subway_route:1',
        kind: 'headway',
        value: 4 * MIN,
        unit: 'seconds',
        observed_at: NOW + 5 * MIN,
        source: 'gtfs_rt_vehicle_positions',
        direction: 'north',
        stop_id: '121N',
      },
    ]);
    expect(snap.freshness.vehicle_positions).toBe(NOW + 5 * MIN);
  });

  test('no state at all publishes an empty surface, and a null feed stamp', () => {
    const snap = snapshotWith(null, NOW, null);
    expect(snap.observations).toEqual([]);
    expect(snap.freshness.vehicle_positions).toBeNull();
  });

  test('a stale document is aged out, and says so via freshness', () => {
    // The doc's own age is the "is the feed being polled at all" gate, matched
    // to every other vehicle-derived surface's 30 minutes.
    const snap = snapshotWith(measured, NOW + 5 * MIN + 1801, NOW + 5 * MIN);
    expect(snap.observations).toEqual([]);
    expect(snap.freshness.vehicle_positions).toBe(NOW + 5 * MIN);
  });

  test('values move across consecutive ticks under normal service', () => {
    // Trains at 121N every ~4 min, then bunching to ~2 min, then a 9-minute
    // gap. Read the surface at each 5-minute publish tick.
    const arrivals = [0, 4, 8, 10, 12, 21, 25].map((m) => m * MIN);
    let doc: HeadwayStateDoc | null = null;
    const readings: number[] = [];
    let nextTick = TICK_SECONDS;
    for (let t = 0; t <= 30 * MIN; t += MIN) {
      const rows: TraceRow[] = [];
      for (let i = 0; i < arrivals.length; i++) {
        const trip = `T${i}`;
        if (t === arrivals[i]) rows.push(row(trip, '121N'));
        else if (t === arrivals[i]! + MIN) {
          rows.push(row(trip, '122N', { vehicleTs: NOW + t }));
        }
      }
      if (rows.length === 0) rows.push(row('TX', '999N'));
      doc = updateHeadwayState(rows, REF, doc, NOW + t);
      if (t >= nextTick) {
        const obs = headwayObservations(doc, NOW + t);
        if (obs.length > 0) readings.push(obs[0]!.value);
        nextTick += TICK_SECONDS;
      }
    }
    // The whole point of the surface: it is not pinned at one value. Six
    // 5-minute publish ticks over the half hour, reading the last completed
    // gap each time — 4 min while service is even, 2 min through the
    // bunching, 9 min after the gap, back to 4 min. A reading repeats when no
    // train has passed since the previous tick, which is honest: the last
    // measured gap is still the last measured gap.
    expect(readings).toEqual([
      4 * MIN,
      4 * MIN,
      2 * MIN,
      2 * MIN,
      9 * MIN,
      4 * MIN,
    ]);
    expect(new Set(readings).size).toBe(3);
    // Every published value is inside the sanity bound.
    for (const v of readings) {
      expect(v).toBeGreaterThanOrEqual(MIN_HEADWAY_SECONDS);
      expect(v).toBeLessThanOrEqual(MAX_HEADWAY_SECONDS);
    }
  });
});

describe('concurrent writers (overlapping or retried crons)', () => {
  // state/headway.json is a read-modify-write, and index.ts already treats
  // overlapping/retried invocations as real (which is why alpha and last_seen
  // use CAS). A dropped or misordered passing here does not lose a datum — it
  // publishes an interval that spans two real headways, which is the single
  // worst thing this surface can emit, so these are correctness tests.

  /** Base: T0 cleared the stop at -540; T1 and T2 are standing at it at 0.
   * Polled densely throughout — a sparse replay would itself straddle
   * FEED_GAP_SECONDS and open a gap window, which is not what these tests are
   * about. */
  function base(): HeadwayStateDoc {
    return replay([
      [-600, [row('T0', '121N')]],
      [-540, [row('T0', '122N', { vehicleTs: NOW - 540 })]],
      [-360, [row('T9', '999N')]],
      [-180, [row('T9', '999N')]],
      [0, [row('T1', '121N'), row('T2', '121N')]],
    ]);
  }

  test('a passing seen only by the LOSING invocation survives the merge', () => {
    const start = base();
    // A polls at +60 and sees T1 depart.
    const ours = detectPassings(
      [row('T1', '122N', { vehicleTs: NOW + 60 })],
      REF,
      start,
      NOW + 60,
    );
    expect(ours).toHaveLength(1);
    // B polls at +120 from the SAME base, but T1 has vanished from its feed
    // (end of run, or a per-vehicle dropout). B records no passing for T1 and
    // merely carries it, so B's document does not contain the departure.
    const winner = updateHeadwayState([row('T9', '999N')], REF, start, NOW + 120);
    expect(reading(winner)).toBeUndefined();
    // B commits first, so A's CAS fails. Merging recovers A's passing.
    const merged = mergePassings(winner, ours);
    expect(reading(merged)).toBe(600); // -540 -> +60
    expect(merged.trips.T1).toBeUndefined(); // cannot fire twice
    expect(merged.observed_at).toBe(NOW + 120); // winner's clock, never rewound
  });

  test('passings commute: the reading is the same whichever invocation won', () => {
    // THE case a sequential fold over a scalar `last_at` got wrong. A saw T1
    // at +60, B saw T2 at +120, both from the same base. The true latest
    // headway is T1 -> T2 = 60s. A scalar fold that applied T2 first would
    // refuse T1 for being behind it and publish T0 -> T2 = 660s instead — one
    // interval reported where two elapsed.
    const start = base();
    const a = detectPassings([row('T1', '122N', { vehicleTs: NOW + 60 })], REF, start, NOW + 60);
    const b = detectPassings([row('T2', '122N', { vehicleTs: NOW + 120 })], REF, start, NOW + 120);

    const aWon = mergePassings(updateHeadwayState([row('T1', '122N', { vehicleTs: NOW + 60 })], REF, start, NOW + 60), b);
    const bWon = mergePassings(updateHeadwayState([row('T2', '122N', { vehicleTs: NOW + 120 })], REF, start, NOW + 120), a);

    expect(reading(aWon)).toBe(60);
    expect(reading(bWon)).toBe(60);
    expect(aWon.cells['1|north']?.passings.map((p) => p.trip))
      .toEqual(bWon.cells['1|north']?.passings.map((p) => p.trip));
  });

  test('merging is idempotent — a retried invocation changes nothing', () => {
    const start = base();
    const ours = detectPassings([row('T1', '122N', { vehicleTs: NOW + 60 })], REF, start, NOW + 60);
    const once = mergePassings(updateHeadwayState([row('T9', '999N')], REF, start, NOW + 120), ours);
    const twice = mergePassings(mergePassings(once, ours), ours);
    expect(twice.cells).toEqual(once.cells);
    expect(reading(twice)).toBe(reading(once));
  });

  test('a departure the winner already recorded is not double-counted', () => {
    const start = base();
    const ours = detectPassings([row('T1', '122N', { vehicleTs: NOW + 60 })], REF, start, NOW + 60);
    // The winner saw the same departure, at its own later stamp.
    const winner = updateHeadwayState(
      [row('T1', '122N', { vehicleTs: NOW + 120 })],
      REF,
      start,
      NOW + 120,
    );
    const merged = mergePassings(winner, ours);
    // One train, so one ledger entry survives the dup window, and the reading
    // is a single real interval rather than a phantom 60s one.
    expect(merged.cells['1|north']?.passings.filter((p) => p.trip === 'T1')).toHaveLength(1);
    expect(reading(merged)).toBe(660);
  });

  test('the feed-gap refusal survives out-of-order insertion', () => {
    // A poll gap opens while an interval is open. Whichever order the two
    // passings land in, the interval that crossed the gap must not publish —
    // this is why the gap is a document-level window and not a flag consumed
    // by whichever entry happens to be newest.
    const start = base();
    // A 300s silence: longer than FEED_GAP_SECONDS, but the whole scenario
    // stays inside TRIP_GAP_SECONDS of the base carry so T1 and T2 are still
    // trusted as the same trains.
    const gapped = updateHeadwayState([row('T9', '999N')], REF, start, NOW + 300);
    expect(gapped.gaps).toEqual([{ from: NOW, until: NOW + 300 }]);
    const at = NOW + 330;
    const later = NOW + 450;
    const p1 = detectPassings([row('T1', '122N', { vehicleTs: at })], REF, gapped, at);
    const p2 = detectPassings([row('T2', '122N', { vehicleTs: later })], REF, gapped, later);
    expect(p1).toHaveLength(1);
    expect(p2).toHaveLength(1);

    const forward = mergePassings(mergePassings(gapped, p1), p2);
    const backward = mergePassings(mergePassings(gapped, p2), p1);
    // T1 -> T2 sits entirely after the outage, so both orders publish it.
    expect(reading(forward)).toBe(120);
    expect(reading(backward)).toBe(120);
    // And the interval that DID cross the outage never published: with only T1
    // applied, the pair is T0 -> T1, which spans it.
    expect(reading(mergePassings(gapped, p1))).toBeUndefined();
  });

  test('a later gap does not un-refuse an interval an earlier gap spanned', () => {
    // Gap windows accumulate rather than overwrite: keeping only the newest
    // would let a still-current pair publish after a second, later outage
    // replaced the window it had crossed.
    const start = base();
    let doc = updateHeadwayState([row('T9', '999N')], REF, start, NOW + FEED_GAP_SECONDS + 120);
    const at = NOW + FEED_GAP_SECONDS + 180;
    doc = updateHeadwayState([row('T1', '122N', { vehicleTs: at })], REF, doc, at);
    expect(reading(doc)).toBeUndefined(); // T0 -> T1 spans the first gap
    // A second outage, with no new passing in between.
    doc = updateHeadwayState([row('T9', '999N')], REF, doc, at + FEED_GAP_SECONDS + 120);
    expect(doc.gaps).toHaveLength(2);
    // Still refused: the first gap is still on record.
    expect(headwayObservations(doc, doc.observed_at)).toEqual([]);
  });

  test('a passing whose measurement point moved since is dropped, not misfiled', () => {
    const start = replay([[0, [row('T1', '121N')]]]);
    const ours = detectPassings(
      [row('T1', '122N', { vehicleTs: NOW + 60 })],
      REF,
      start,
      NOW + 60,
    );
    expect(ours).toHaveLength(1);
    const winner = updateHeadwayState(
      [row('T5', '120N')],
      { stops: { '1|north': '120N' }, at: NOW + 120, trained_at: 2 },
      null,
      NOW + 120,
    );
    expect(mergePassings(winner, ours).cells['1|north']?.passings ?? []).toHaveLength(0);
  });

  test('merging nothing returns the document untouched', () => {
    const winner = base();
    expect(mergePassings(winner, [])).toBe(winner);
  });
});

// A six-stop line spanning three positional zones, shaped like N/R: an outer
// "borough" at each end and the busy core in the middle. All stops carry the
// same trips, so the tie-break toward the middle picks a core stop as primary
// and the fallbacks are drawn from the outer thirds — the segments a core
// reroute leaves running. B1N/Q2N are terminals and must never be picked.
const LINE: RouteStops = {
  'N|north': [{ stops: ['B1N', 'B2N', 'M1N', 'M2N', 'Q1N', 'Q2N'], n_trips: 10 }],
};

/** Two trains (`t1` then `t2`) passing `stop` and departing to the
 * non-candidate `next`, so the cell gets a completed reading of 2*MIN. Trip ids
 * are explicit so concatenated blocks stay chronological and collision-free. */
function servedTwice(
  route: string,
  direction: 'north' | 'south',
  stop: string,
  next: string,
  base: number,
  t1: string,
  t2: string,
): [number, TraceRow[]][] {
  return [
    [base, [row(t1, stop, { route, direction })]],
    [base + MIN, [row(t1, next, { route, direction })]],
    [base + 2 * MIN, [row(t2, stop, { route, direction })]],
    [base + 3 * MIN, [row(t2, next, { route, direction })]],
  ];
}

function obs(doc: HeadwayStateDoc, entity: string, direction: string) {
  return headwayObservations(doc, doc.observed_at).find(
    (o) => o.entity_ref === entity && o.direction === direction,
  );
}

describe('selectFallbackStops', () => {
  test('draws one spread fallback per outer zone, ordered by rank, terminals out', () => {
    expect(selectReferenceStops(LINE)['N|north']?.stop_id).toBe('M2N'); // core primary
    // Q1N (late zone) outranks B2N (early zone) by the middle tie-break, so it
    // is the preferred fallback; both terminals are excluded by the through-
    // stop gate the primary passes.
    expect(selectFallbackStops(LINE)['N|north']).toEqual(['Q1N', 'B2N']);
  });

  test('a cell with a single through-stop has no fallback', () => {
    const oneThrough: RouteStops = {
      'A|north': [{ stops: ['A', 'B', 'C'], n_trips: 4 }], // B is the only through-stop
    };
    expect(selectReferenceStops(oneThrough)['A|north']?.stop_id).toBe('B');
    expect(selectFallbackStops(oneThrough)['A|north']).toBeUndefined();
  });
});

describe('reroute survival: publish from a served fallback, honestly labelled', () => {
  const N_REF: HeadwayReference = {
    stops: { 'N|north': 'M2N' },
    fallbacks: { 'N|north': ['Q1N', 'B2N'] },
    at: NOW - 3600,
    trained_at: 1_699_000_000,
  };

  test('normal service publishes the primary reference stop', () => {
    const doc = replay(servedTwice('N', 'north', 'M2N', 'M2Nx', 0, 'a', 'b'), N_REF);
    const o = obs(doc, 'subway_route:N', 'north');
    expect(o?.stop_id).toBe('M2N');
    expect(o?.value).toBe(2 * MIN);
  });

  test('N: primary dark in Manhattan, the cell falls back to a served Queens stop', () => {
    // "No N service in Manhattan": no train ever passes the M2N primary, but
    // the route is running — trains clear Q1N. The cell publishes from Q1N
    // instead of going dark, and says so in stop_id.
    const doc = replay(servedTwice('N', 'north', 'Q1N', 'Q1Nx', 0, 'a', 'b'), N_REF);
    const o = obs(doc, 'subway_route:N', 'north');
    expect(o?.stop_id).toBe('Q1N'); // the ACTUAL measurement stop, not M2N
    expect(o?.value).toBe(2 * MIN);
    expect(doc.cells['N|north']).toBeUndefined(); // primary saw nothing
  });

  test('the primary is preferred whenever it too has a reading', () => {
    // Both primary and a fallback served this window: the higher-ranked primary
    // wins, so a fallback never overrides a live primary series.
    const doc = replay(
      [
        ...servedTwice('N', 'north', 'M2N', 'M2Nx', 0, 'm1', 'm2'),
        ...servedTwice('N', 'north', 'Q1N', 'Q1Nx', 4 * MIN, 'q1', 'q2'),
      ],
      N_REF,
    );
    expect(obs(doc, 'subway_route:N', 'north')?.stop_id).toBe('M2N');
  });

  test('a full suspension still abstains — no candidate served, no observation', () => {
    // Route N entirely suspended: the only trains in the feed are another
    // route. No candidate stop is served, so the cell publishes nothing rather
    // than fabricating a value.
    const doc = replay([[0, [row('x', '121N', { route: '1' })]], [MIN, [row('x', '120N', { route: '1' })]]], N_REF);
    expect(obs(doc, 'subway_route:N', 'north')).toBeUndefined();
  });

  test('R: same route, one direction on its primary and the other on a fallback', () => {
    // "R rerouted via D/F in Manhattan": R|north's primary (M2N) is dark and it
    // falls back to a served Brooklyn stop, while R|south's primary (M2S) is
    // still served. One direction dark-then-recovered, one direction native —
    // purely from where each reference sits.
    const R_REF: HeadwayReference = {
      stops: { 'R|north': 'M2N', 'R|south': 'M2S' },
      fallbacks: { 'R|north': ['B2N'], 'R|south': ['B2S'] },
      at: NOW - 3600,
      trained_at: 1_699_000_000,
    };
    const doc = replay(
      [
        ...servedTwice('R', 'north', 'B2N', 'B2Nx', 0, 'rn1', 'rn2'), // north in Brooklyn
        ...servedTwice('R', 'south', 'M2S', 'M2Sx', 4 * MIN, 'rs1', 'rs2'), // south primary
      ],
      R_REF,
    );
    expect(obs(doc, 'subway_route:R', 'north')?.stop_id).toBe('B2N'); // fallback
    expect(obs(doc, 'subway_route:R', 'south')?.stop_id).toBe('M2S'); // primary
  });
});
