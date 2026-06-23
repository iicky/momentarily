/**
 * Derive a compact per-route "are trains actually moving" metric from the
 * decoded GTFS-RT vehicle positions. Archived for OFFLINE validation and as the
 * basis for the movement-derived current state — orthogonal in DERIVATION to
 * assigned_n (where trains physically are, not how many trips are dispatched).
 *
 * Two signals, with different strengths:
 *   - moving_n / vehicles_n: an INSTANTANEOUS movement fraction. Cheap but
 *     noisy on its own — a train is STOPPED_AT every station dwell, so a single
 *     tick can't tell a normal dwell from a stall.
 *   - advanced_n / stalled_n: the CROSS-TICK signal. Given the previous tick's
 *     stop_id per trip, a trip whose stop_id is unchanged ~5 min later is
 *     stalled; one that moved on has advanced. A route where assigned trains are
 *     dispatched (assigned_n high) but none advance is physically frozen — the
 *     disruption mode assigned_n structurally cannot see.
 *
 * Advance/stall are also split by direction (north/south), because the two
 * directions fail independently and the Bayesian movement model scores each
 * line-direction against its own baseline advance rate. Direction comes from the
 * stop_id N/S suffix, falling back to the trip_id `..N`/`..S` char.
 */

import type { VehicleLite } from './gtfsrt';

export interface DirMovementRow {
  vehicles_n: number;
  advanced_n: number; // present last tick AND stop_id changed
  stalled_n: number; // present last tick AND stop_id identical
}

export interface MovementRow {
  vehicles_n: number; // vehicles referencing this route
  stopped_n: number; // current_status STOPPED_AT
  moving_n: number; // everything else (NYCT omits the field for in-transit)
  // Cross-tick (0 when no previous stop is known for the trip):
  advanced_n: number; // present last tick AND stop_id changed
  stalled_n: number; // present last tick AND stop_id identical
  by_direction: { north: DirMovementRow; south: DirMovementRow };
}

/** Express variants (6X, 7X, FX) fold to their base route, matching derive.ts. */
function baseRoute(routeId: string): string {
  return routeId.replace(/X$/, '');
}

const STOPPED_AT = 1; // GTFS-RT VehicleStopStatus; absence defaults to IN_TRANSIT_TO

/** Direction from the stop_id N/S suffix (e.g. `A09N`), falling back to the
 * trip_id direction char after `..` (e.g. `..N`). null when neither is present. */
function directionOf(v: VehicleLite): 'north' | 'south' | null {
  const last = v.stopId.slice(-1);
  if (last === 'N') return 'north';
  if (last === 'S') return 'south';
  const i = v.tripId.indexOf('..');
  if (i >= 0) {
    const c = v.tripId[i + 2];
    if (c === 'N') return 'north';
    if (c === 'S') return 'south';
  }
  return null;
}

function emptyDir(): DirMovementRow {
  return { vehicles_n: 0, advanced_n: 0, stalled_n: 0 };
}

function emptyRow(): MovementRow {
  return {
    vehicles_n: 0,
    stopped_n: 0,
    moving_n: 0,
    advanced_n: 0,
    stalled_n: 0,
    by_direction: { north: emptyDir(), south: emptyDir() },
  };
}

/**
 * The per-trip stop_id snapshot to carry into the next tick, so cross-tick
 * advance can be computed without re-fetching. Keyed by trip_id (stable across
 * ticks for a given run). Empty trip_ids are dropped — they can't be matched.
 */
export function stopPositions(vehicles: VehicleLite[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (const v of vehicles) {
    if (v.tripId) out[v.tripId] = v.stopId;
  }
  return out;
}

/**
 * Group decoded vehicles (across all fetched feeds) into per-route movement
 * rows. `prevStops` is the previous tick's stopPositions(); pass an empty map on
 * the first tick (or when no prior state exists) and the cross-tick counters
 * stay 0 — the instantaneous counters are always populated.
 */
export function deriveRouteMovementMetric(
  vehicles: VehicleLite[],
  prevStops: Record<string, string> = {},
): Map<string, MovementRow> {
  const out = new Map<string, MovementRow>();
  for (const v of vehicles) {
    const route = baseRoute(v.routeId);
    let row = out.get(route);
    if (!row) {
      row = emptyRow();
      out.set(route, row);
    }
    const dir = directionOf(v);
    const dirRow = dir ? row.by_direction[dir] : null;

    row.vehicles_n += 1;
    if (dirRow) dirRow.vehicles_n += 1;
    if (v.status === STOPPED_AT) row.stopped_n += 1;
    else row.moving_n += 1;

    const prev = v.tripId ? prevStops[v.tripId] : undefined;
    if (prev !== undefined) {
      if (prev === v.stopId) {
        row.stalled_n += 1;
        if (dirRow) dirRow.stalled_n += 1;
      } else {
        row.advanced_n += 1;
        if (dirRow) dirRow.advanced_n += 1;
      }
    }
  }
  return out;
}
