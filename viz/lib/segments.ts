// One place where "what does the snapshot say about this segment?" is decided,
// so the map view and the station view cannot drift in how they read it.
//
// The distinction this layer exists to hold: three different reasons a stroke
// has no colour, which a map must not collapse into one.
//
//   unscheduled  the timetable schedules no service on this pair in this
//                direction — there is no cell, so there is nothing to measure.
//   unmeasured   there IS a cell, and `segments` doesn't carry it. The Worker
//                only judges a cell once its decayed matched trip count clears
//                a floor within the accumulation window, and the trainer only
//                baselines cells it has history for.
//   normal /
//   disrupted    an actual published verdict.
//
// Painting `unmeasured` as healthy is the specific failure mode here: it would
// claim a reading over the vast majority of the system that has none.
//
// The published surface carries EVERY judged cell in one collection —
// `segments`, keyed `route|direction|from_stop` — as a full record: clock,
// recovery, and the successor stop the reading was measured toward. A normal
// cell's `recovery` is always null (nothing to forecast on healthy track), but
// it is otherwise a record like any other. A key absent from `segments` is
// `unmeasured`; that single membership check is the whole honesty property
// this module holds.
//
// The record's successor is what makes it placeable. A cell key is keyed on
// its from_stop ALONE, so at a branch or express split one key belongs to
// several drawn edges — 131 of 1797 keys on the current asset, up to 4 edges
// for one key. `to` is what says which of them the reading is about; without
// it a verdict would have to be spread across every sibling leg, painting
// unmeasured track as healthy, which is the one failure this view exists to
// prevent. The only fallback is for a null successor: the topology doc was
// unavailable, the reading is still real, and there is no better attribution
// to be had.

import type { Diagram, DiagramEdge, Direction } from "./diagram";
import type { SegmentFlow, SegmentStatus } from "./types";

export const DIRECTIONS: readonly Direction[] = ["north", "south"];

export type SegmentState = "disrupted" | "normal" | "quiet" | "unmeasured" | "unscheduled";

/** A directional stop id collapsed to its parent station: `A24S` -> `A24`.
 * Mirrors worker/src/segment_flow.ts stationId and training/diagram.py
 * parent_station — the segment surface is keyed on platforms, the diagram on
 * stations, and this is the one place that bridges them. */
export function stationOf(stop: string): string {
  const last = stop.slice(-1);
  return last === "N" || last === "S" ? stop.slice(0, -1) : stop;
}

// Worst-first, and `unmeasured` outranks `normal` deliberately: when one
// direction of a pair has a good reading and the other has none, the pair as a
// whole has not been shown to be healthy, so the combined view says so.
//
// `quiet` is a definite verdict — "the timetable runs too little here right now
// for silence to mean anything, normal for now" — so it outranks `normal`: a
// pair with any quiet direction must not read as a clean advancing all-clear.
// It ranks below `unmeasured`, since no verdict at all is the more conservative
// read than a benign one.
const STATE_RANK: Record<SegmentState, number> = {
  disrupted: 4,
  unmeasured: 3,
  quiet: 2,
  normal: 1,
  unscheduled: 0,
};

export interface DirectionReading {
  direction: Direction;
  state: SegmentState;
  // The segment_flow key for this direction, or null when unscheduled.
  key: string | null;
  // The record `segments` published for this key, or null when unmeasured
  // or unscheduled. Every judged cell is a full record now, normal or
  // disrupted alike — a healthy one just carries a null `recovery`.
  cell: SegmentStatus | null;
}

export interface EdgeReading {
  edge: DiagramEdge;
  north: DirectionReading;
  south: DirectionReading;
}

export type DirectionFilter = Direction | "both";

/** Which drawn end of this edge the cell keyed here was measured toward.
 *
 * A cell is keyed on its from_stop alone, so at a branch or express split one
 * key belongs to two drawn edges (131 of 1797 keys in the current timetable).
 * Whoever published the reading names the successor it was measured against,
 * and the sibling branch is NOT what the reading is about — painting it too
 * would spread one measurement across track no train on it used. */
function towardEnd(edge: DiagramEdge, fromStop: string): string {
  return edge.a === stationOf(fromStop) ? edge.b : edge.a;
}

/** Whether a reading keyed here belongs to this drawn edge.
 *
 * `to` is null only when the Worker couldn't read the topology doc. The
 * reading is real and there is no better attribution to be had, so it isn't
 * discarded. */
function placesOn(edge: DiagramEdge, fromStop: string, to: string | null): boolean {
  return to === null || stationOf(to) === towardEnd(edge, fromStop);
}

function readDirection(
  edge: DiagramEdge,
  direction: Direction,
  flow: SegmentFlow | null,
): DirectionReading {
  const key = edge.keys[direction] ?? null;
  if (key === null) {
    return { direction, state: "unscheduled", key: null, cell: null };
  }
  const blank: DirectionReading = { direction, state: "unmeasured", key, cell: null };
  if (flow === null) return blank;

  const cell = flow.segments[key];
  if (cell === undefined) return blank;
  if (!placesOn(edge, cell.from_stop, cell.to)) return blank;
  // The published `status` IS the verdict — normal or disrupted — since
  // `segments` carries a full record for every judged cell.
  return { direction, state: cell.status, key, cell };
}

export function readEdge(edge: DiagramEdge, flow: SegmentFlow | null): EdgeReading {
  return {
    edge,
    north: readDirection(edge, "north", flow),
    south: readDirection(edge, "south", flow),
  };
}

/** The reading a given direction filter selects. "both" takes the worse of the
 * two by STATE_RANK, so a disruption in either direction shows and a
 * half-measured pair never reads healthy. */
export function selectReading(
  reading: EdgeReading,
  filter: DirectionFilter,
): DirectionReading {
  if (filter !== "both") return reading[filter];
  return STATE_RANK[reading.north.state] >= STATE_RANK[reading.south.state]
    ? reading.north
    : reading.south;
}

export interface Coverage {
  /** Directional CELLS the timetable schedules — the denominator.
   *
   * Counted as distinct segment_flow keys, not as (edge, direction) slots,
   * because the numerator is cells: one key belongs to two drawn edges wherever
   * a from_stop branches, and counting slots here would inflate the
   * denominator by those duplicates and quietly understate coverage. */
  scheduled: number;
  /** Of those, how many carry a published verdict this tick — cells, from
   * `segments`. Same unit as `scheduled`, deliberately. */
  measured: number;
  /** Of `measured`, how many read disrupted. Cells again, so the two numbers
   * sit in the same unit as the ratio they qualify. */
  disrupted: number;
  /** Published cells whose key matches no scheduled edge on the diagram.
   *
   * Not a rounding error: the realtime feed reports the from_stop a train
   * actually ran, and a reroute puts trains on pairs the static timetable for
   * that day doesn't schedule. A non-zero count here means the diagram is
   * older than the service, or service is off-pattern. Either way those
   * readings exist and are not drawn, so the number is shown rather than
   * silently dropped. */
  unplaced: number;
}

export function coverage(diagram: Diagram, flow: SegmentFlow | null): Coverage {
  const placed = new Set<string>();
  for (const edge of diagram.edges) {
    for (const direction of DIRECTIONS) {
      const key = edge.keys[direction];
      if (key !== undefined) placed.add(key);
    }
  }
  let measured = 0;
  let disrupted = 0;
  let unplaced = 0;
  for (const [key, cell] of Object.entries(flow?.segments ?? {})) {
    if (!placed.has(key)) {
      unplaced += 1;
      continue;
    }
    measured += 1;
    if (cell.status === "disrupted") disrupted += 1;
  }
  return { scheduled: placed.size, measured, disrupted, unplaced };
}

// Painting order: ghosts first, verdicts on top, so a disrupted stroke is never
// buried under the unmeasured majority it crosses.
export const PAINT_ORDER: Record<SegmentState, number> = {
  unscheduled: 0,
  unmeasured: 1,
  quiet: 2,
  normal: 3,
  disrupted: 4,
};
