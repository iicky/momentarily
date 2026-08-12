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
  // Raw cross-tick from_stop_id>to_stop_id counts (from==to = a stall in place).
  // The segment-level leaf, archived for later hierarchical baselines; canonical
  // segment mapping is deferred (GTFS-RT stop_ids can skip/express/reverse).
  transitions: Record<string, number>;
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
  return { vehicles_n: 0, advanced_n: 0, stalled_n: 0, transitions: {} };
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
      // Raw per-direction transition for the segment leaf. Skip empty stop_ids
      // so a blank endpoint can't poison a segment cell downstream.
      if (dirRow && prev && v.stopId) {
        const key = `${prev}>${v.stopId}`;
        dirRow.transitions[key] = (dirRow.transitions[key] ?? 0) + 1;
      }
    }
  }
  return out;
}

/**
 * Per-trip, per-minute movement trace — the fine-grained sibling of
 * deriveRouteMovementMetric above. That function's advanced_n/stalled_n are a
 * 5-minute cross-tick signal: at 5-minute polling most observed "moves" span
 * 2+ stations (mean ~2.7 on the existing archive), so the intermediate
 * stations are never actually observed and (from_stop, to_stop) segments are
 * really multi-station jumps. Polling every minute instead — and archiving
 * every (stop_id, stopped) change as its own row — gets close enough to
 * per-station observations to measure real station-to-station traversal
 * time, which the 5-minute signal structurally cannot.
 *
 * This is intentionally a PARALLEL, INDEPENDENT pipeline from
 * deriveRouteMovementMetric/stopPositions above: it never reads or writes
 * state/vehicle_stops.json, and it archives under its own archive/trace/
 * prefix, never archive/vehicles/, at its own cadence (every minute, not
 * gated to the 5-minute boundary). See the scheduled handler in index.ts for
 * how the two are kept apart — because this trace never touches
 * vehicle_stops.json or archive/vehicles/, it cannot perturb
 * deriveRouteMovementMetric's advanced_n/stalled_n, and every dwell/
 * advance-rate param trained on them, which assume a fixed 5-minute tick.
 */
export interface TraceRow {
  trip_id: string;
  route_id: string; // baseRoute()-folded, like MovementRow's keys
  direction: 'north' | 'south' | null; // directionOf(), same as above
  stop_id: string; // the stop this vehicle is at (STOPPED_AT) or heading to
  stop_seq: number | null; // current_stop_sequence; present only when STOPPED_AT
  stopped: boolean; // status === STOPPED_AT
  vehicle_ts: number | null; // the feed's own per-vehicle report timestamp
}

/**
 * One row per in-service trip, as observed this poll. The raw stream the
 * traversal-time model is built from.
 *
 * This is a FULL snapshot, not a delta against the previous poll, and that is a
 * deliberate reversal. Delta-encoding looks like free compression but buys
 * nothing here: a train changes (stop_id, status) roughly once a minute anyway,
 * so at ~700 concurrent trips a delta emits ~760 rows/min against ~700 for the
 * full snapshot. It is not smaller, and it costs three real things:
 *
 *   1. A carry object, which is state, which is another thing to corrupt.
 *   2. Idempotency. The delta is a function of (feed, carry), and the carry is
 *      written by the same step — so a retried invocation for one cron minute
 *      sees its own earlier write, computes zero changed rows, and overwrites a
 *      good archive object with an empty one. That silently DESTROYS the arrival
 *      it just recorded. A full snapshot is a pure function of the feed, so a
 *      retry rewrites byte-identical content and cannot lose anything.
 *   3. Disappearances. A train present at minute t and absent at t+1 is a
 *      censored traversal, and we want it. A delta cannot express absence; a
 *      snapshot gives it for free as a trip that stops appearing.
 *
 * Reconstructing arrivals is then a diff of consecutive snapshots, done offline
 * where it is cheap, inspectable and re-runnable against corrected logic —
 * rather than baked irreversibly into collection.
 *
 * On the two meanings of stop_id, which the offline diff has to honour: NYCT's
 * stop_id is the stop a train is *heading to* while in transit, and the stop it
 * is *at* once STOPPED_AT. So one hop into station N shows up as "stop_id=N,
 * stopped=false" and then "stop_id=N, stopped=true" — same stop, different
 * status. The second is the arrival; stop_seq is populated only then.
 *
 * Skips vehicles with an empty tripId (can't be matched across polls, same
 * caution as stopPositions) or an empty stopId (same caution as the transitions
 * map in deriveRouteMovementMetric above).
 */
export function deriveTrace(vehicles: VehicleLite[]): TraceRow[] {
  const rows: TraceRow[] = [];
  for (const v of vehicles) {
    if (!v.tripId || !v.stopId) continue;
    const stopped = v.status === STOPPED_AT;
    rows.push({
      trip_id: v.tripId,
      route_id: baseRoute(v.routeId),
      direction: directionOf(v),
      stop_id: v.stopId,
      stop_seq: stopped ? v.stopSeq : null,
      stopped,
      vehicle_ts: v.timestamp,
    });
  }
  return rows;
}
