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
 *   - advanced_n / stalled_n: the CROSS-TICK signal, restricted to trips whose
 *     FROM stop is a scheduled THROUGH stop (has both a scheduled predecessor
 *     and successor) when the trainer publishes that set (params.ts
 *     `throughStops`); every stop counts when it hasn't. A terminal/
 *     chain-endpoint stop stalls ~89% of the time by design — a scheduled
 *     layover, not a disruption — and carries 83% of all stall mass measured
 *     on the archive, so counting it blends two physically different
 *     populations into one advance rate. Given the previous tick's stop_id per
 *     trip, a trip whose stop_id is unchanged ~5 min later is stalled; one
 *     that moved on has advanced. A route where assigned trains are dispatched
 *     (assigned_n high) but none advance is physically frozen — the
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
  advanced_n: number; // present last tick, stop_id changed, from_stop through-stop-admitted
  stalled_n: number; // present last tick, stop_id identical, from_stop through-stop-admitted
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
  advanced_n: number; // present last tick, stop_id changed, from_stop through-stop-admitted
  stalled_n: number; // present last tick, stop_id identical, from_stop through-stop-admitted
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
 *
 * `throughStops` is the trainer-published route|direction|stop admission set
 * (params.ts `TrainedParams.throughStops`); null (the default) counts every
 * stop, matching behaviour before the filter existed. When set, a matched
 * trip's advance/stall counts (route-level AND by_direction) require its FROM
 * stop to be in the set. The `transitions` map is never filtered — it stays
 * the raw observation stream so offline recomputation (training/load_r2.py
 * StopFilter) can apply any filter after the fact.
 */
export function deriveRouteMovementMetric(
  vehicles: VehicleLite[],
  prevStops: Record<string, string> = {},
  throughStops: ReadonlySet<string> | null = null,
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
    if (prev === undefined) continue;

    // Raw per-direction transition for the segment leaf, unfiltered — the
    // segment_flow classifier and offline recomputation both need every
    // trip's from_stop, not just the ones the through-stop filter admits.
    // Skip empty stop_ids so a blank endpoint can't poison a segment cell
    // downstream.
    if (dirRow && prev && v.stopId) {
      const key = `${prev}>${v.stopId}`;
      dirRow.transitions[key] = (dirRow.transitions[key] ?? 0) + 1;
    }

    // advanced_n/stalled_n require a known direction and non-empty from/to
    // stop ids, so route-level == north+south == the transitions sum by
    // construction — mirrors training/load_r2.py _cross_tick_counts, which
    // sums an archived row the same three ways. On top of that, when
    // throughStops is given a trip counts only if its FROM stop is in it: a
    // terminal/chain-endpoint stall is a scheduled layover, not signal.
    if (!dir || !dirRow || !prev || !v.stopId) continue;
    if (throughStops && !throughStops.has(`${route}|${dir}|${prev}`)) continue;

    if (prev === v.stopId) {
      row.stalled_n += 1;
      dirRow.stalled_n += 1;
    } else {
      row.advanced_n += 1;
      dirRow.advanced_n += 1;
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

/**
 * Aggregate raw per-vehicle rows into a map-drawable view: how many trains
 * currently share each (route, direction, stop, stopped) tuple. Reuses
 * baseRoute/directionOf/STOPPED_AT above — the same rules deriveTrace uses —
 * so a train's route/direction here never disagrees with its trace row. At
 * ~700 concurrent trips this typically collapses to a few hundred distinct
 * tuples (measured in trainPositions.test.ts against a synthetic feed of
 * that size): trains queued behind each other, or bunched at a terminal,
 * fold into one dot with a count instead of N overlapping markers.
 *
 * Skips vehicles with an empty stop_id — same caution as deriveTrace and the
 * transitions map in deriveRouteMovementMetric: an unplaced train can't be
 * drawn.
 *
 * Deliberately does NOT report which segment a moving train occupies. NYCT's
 * stop_id is the stop a train is *heading to* while in transit (stopped ===
 * false) and the stop it is *at* once STOPPED_AT (stopped === true) — the
 * same duality TraceRow documents above. Turning that into "on segment A->B"
 * would mean guessing a direction of travel at every branch or express
 * point (e.g. a 6 train signed for Pelham Bay Park could still be running
 * express or local past 125th St) where stop_id alone doesn't disambiguate.
 * Consumers place the dot at stationOf(stop) and read `stopped` to tell
 * at-platform from approaching, rather than Momentarily guessing the hop.
 *
 * Output is sorted by (route, direction, stop, stopped) so the published
 * snapshot diffs cleanly tick to tick instead of reordering on Map iteration.
 */
export interface TrainPosition {
  route: string; // baseRoute()-folded, like TraceRow's route_id
  direction: 'north' | 'south' | null; // directionOf(), same as TraceRow
  stop: string; // directional stop_id, exactly as the feed reports it
  stopped: boolean; // true = at the platform, false = heading toward it
  n: number; // how many vehicles share this exact tuple
}

export function trainPositions(vehicles: VehicleLite[]): TrainPosition[] {
  const byKey = new Map<string, TrainPosition>();
  for (const v of vehicles) {
    if (!v.stopId) continue;
    const route = baseRoute(v.routeId);
    const direction = directionOf(v);
    const stopped = v.status === STOPPED_AT;
    const key = `${route}|${direction ?? ''}|${v.stopId}|${stopped}`;
    const existing = byKey.get(key);
    if (existing) {
      existing.n += 1;
    } else {
      byKey.set(key, { route, direction, stop: v.stopId, stopped, n: 1 });
    }
  }
  return [...byKey.values()].sort(compareTrainPositions);
}

function compareTrainPositions(a: TrainPosition, b: TrainPosition): number {
  if (a.route !== b.route) return a.route < b.route ? -1 : 1;
  const ad = a.direction ?? '';
  const bd = b.direction ?? '';
  if (ad !== bd) return ad < bd ? -1 : 1;
  if (a.stop !== b.stop) return a.stop < b.stop ? -1 : 1;
  if (a.stopped !== b.stopped) return a.stopped ? 1 : -1;
  return 0;
}
