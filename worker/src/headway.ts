/**
 * Observed headway at a canonical reference stop -> the published
 * `observations` surface.
 *
 * WHAT THIS MEASURES. Per (route, direction), the time between the two most
 * recent trains to serve that cell's reference stop. A raw measurement of the
 * live vehicle feed: no baseline, no model, no grade. It is published as an
 * Observation (schema.py's peer of Alert) precisely because it is not an
 * inference — nothing here is fitted, and the number means only "these two
 * trains were this far apart at this stop".
 *
 * WHY A REFERENCE STOP. Headway is measured at a point, so every route/
 * direction needs one stable, documented stop or the series is not comparable
 * across ticks, days or routes. The rule (selectReferenceStops) is the same
 * one the offline toolkit uses (training/headway.select_reference_stops): among
 * a route/direction's through-stops — stops with both a scheduled predecessor
 * and a scheduled successor, so terminals and yard leads are excluded — take
 * the one carrying the most scheduled trips, breaking ties toward the middle of
 * the most-run pattern and then the smaller stop id. That stop is the trunk
 * where express/local and every branch overlap, so the series misses no train,
 * and excluding terminals is what keeps a turnback's layover dwell (and a
 * terminal's repeated re-reporting of the same train) out of the gap.
 *
 * The rule is applied here, in the Worker, to the scheduled stopping patterns
 * the trainer already publishes on state/segment_params.json (`route_stops`,
 * from training/gtfs_static.route_patterns). It is not a second source of
 * truth: `route_stops` IS route_patterns(), and the consecutive-stop pairs
 * summed over it reproduce gtfs_static.successors() exactly, so the
 * dominant-successor skeleton the through-stop test reads off is the same
 * graph. Verified against the offline rule's documented output on the current
 * feed: 25 routes x 2 directions, coverage 1.000, 1|north=121N, 2|north=120N,
 * 7|south=714S, L=L15N/L16S, A=A55N/A55S.
 *
 * WHAT COUNTS AS A PASSING. The DEPARTURE from the reference stop: the first
 * poll at which a trip last seen with stop_id = the reference stop reports a
 * different stop_id. NOT the STOPPED_AT sighting the offline arrival
 * reconstruction (training/trace.arrivals_from_trace) keys on, and the
 * difference is measured, not stylistic: replaying the real trace over
 * 2026-08-20 07:00-11:00 ET found that only 92.71% of stop transitions had the
 * trip caught STOPPED_AT on the immediately preceding poll (see
 * crowding.ts's CROWDING_MAX_GAP_MINUTES note) — a dwell shorter than the
 * 1-minute poll comes and goes with the stop_id simply changing underneath it.
 * Keying on STOPPED_AT therefore drops ~7% of passings, and a dropped passing
 * does not lose a datum, it MERGES two real headways into one double-length
 * reading. The transition is observed for every train that served the stop.
 *
 * One definition, not two: mixing arrival-stamped and departure-stamped
 * passings would inject a dwell (~30-60s) of jitter into every other gap.
 * Departure-to-departure at a fixed point is the same quantity as
 * arrival-to-arrival, offset by the dwell difference between the two trains.
 *
 * WHAT IS REFUSED, AND WHY EACH REFUSAL EXISTS. Every one of these produces NO
 * observation for the cell rather than a zero, a fabricated value, or a stale
 * one carried forward:
 *   - no second train seen yet (a cold cell, or the first tick after deploy):
 *     there is no gap to report.
 *   - the interval overlaps a window in which the Worker was not polling (the
 *     document's `gaps`): a gap in the FEED is not a gap in SERVICE, so the
 *     interval is unobserved rather than long.
 *   - the value is above MAX_HEADWAY_SECONDS: the sanity bound, below. (The
 *     lower bound is not a refusal but a COLLAPSE — two entries closer than
 *     MIN_HEADWAY_SECONDS are one train reported under two trip_ids, so the
 *     later one is dropped and the true previous interval stays readable. See
 *     cellHeadway.)
 *   - the same trip_id passes twice within DUP_ARRIVAL_SECONDS: one train
 *     re-reported, not a zero-headway pair.
 *   - the trip's carried position is older than TRIP_GAP_SECONDS: NYCT reuses
 *     trip_ids, so a stale carry may belong to a different train and the
 *     transition means nothing.
 *   - the last completed reading is older than MAX_READING_AGE_SECONDS: a gap
 *     that long means there is no current headway, and republishing the last
 *     one would report a four-minute service that stopped running half an hour
 *     ago.
 */

import type { HeadwayCell, HeadwayStateDoc, HeadwayTrip } from './state';
import type { TraceRow } from './vehicles';

/**
 * The scheduled stopping patterns the through-stop rule reads, as
 * state/segment_params.json carries them: 'route|direction' -> patterns,
 * most-run first. Structurally SegmentParamsDoc['route_stops'], declared here
 * rather than imported so a malformed trainer field degrades this surface
 * only.
 */
export type RouteStops = Record<
  string,
  { stops: string[]; n_trips: number }[]
>;

/** One route/direction's canonical measurement point. Mirrors
 * training/headway.ReferenceStop, minus the fields only the offline
 * defence report prints. `coverage` is the share of the cell's scheduled trips
 * whose pattern includes this stop — 1.0 means every express, local and branch
 * pattern serves it, which is what a series that misses no train needs. */
export interface ReferenceStop {
  route: string;
  direction: string;
  stop_id: string;
  n_scheduled_trips: number;
  coverage: number;
}

/** Two passings of the same trip at the same reference stop closer together
 * than this are one train re-reported by the feed, not two trains a
 * near-zero headway apart. Same constant and reasoning as
 * training/headway.DUP_ARRIVAL_SECONDS. */
export const DUP_ARRIVAL_SECONDS = 120;

/** A trip whose carried reference-stop position is older than this is not
 * trusted to have "just departed": NYCT trip_ids encode origin time, route and
 * direction, so the same id comes back and a stale carry would credit a new
 * train's first sighting as an old train's departure. Same constant and
 * reasoning as training/trace.TRIP_GAP_SECONDS. */
export const TRIP_GAP_SECONDS = 600;

/** A poll gap at least this long makes every currently-open interval
 * feed-uncertain: the missing time may be missing observation rather than
 * missing service. The trace polls every ~60s, so this is several missed polls,
 * not one late one. Same constant and reasoning as
 * training/headway.FEED_GAP_SECONDS. */
export const FEED_GAP_SECONDS = 240;

/**
 * The sanity bounds on a published headway, in seconds. The two act
 * differently, which is deliberate — see cellHeadway.
 *
 * Lower (a COLLAPSE, not a refusal): two trains OF THE SAME ROUTE AND
 * DIRECTION cannot clear one platform 30s apart — station dwell alone is
 * roughly that, and signal blocks enforce far more. So a ledger entry that
 * close to its predecessor is not a second train, it is one train reported
 * under two trip_ids (NYCT reassigns mid-run), which the DUP_ARRIVAL_SECONDS
 * guard cannot see because the ids differ. It is dropped from the derivation
 * and the true previous interval is used, rather than the phantom entry
 * blanking the cell. The bound sits at the physically impossible rather than
 * the merely unusual: genuine bunching down to ~60s is real service and must
 * publish.
 *
 * Upper (a refusal): two hours. Real overnight headways reach ~20 minutes and
 * a genuine disruption can stretch a gap past an hour, so the bound has to sit
 * well above both; past two hours the interval is a state artefact — a cell
 * whose ledger survived a service shutdown — and not a wait anybody
 * experienced. There is nothing to fall back to, so no observation is emitted.
 */
export const MIN_HEADWAY_SECONDS = 30;
export const MAX_HEADWAY_SECONDS = 7200;

/**
 * How old a completed reading may be and still be published as the cell's
 * current headway. Matched to the 30 minutes every other movement-derived
 * surface in this snapshot is aged out at (snapshot.ts's
 * MAX_MOVEMENT_STATE_AGE_SEC), for the same reason: past it the number
 * describes a service that is no longer running. In normal service a cell
 * refreshes every few minutes, so this bound only bites during a real gap —
 * where dropping the observation is the honest reading and republishing the
 * last one is a false claim of current service.
 */
export const MAX_READING_AGE_SECONDS = 1800;

/** Drop a cell whose newest passing is this stale, regardless of the bounds
 * above, purely to keep state/headway.json bounded across a multi-day feed
 * outage. A full service day of silence is past the point the ledger is still
 * useful for anything. (Trip carries are not pruned here — they expire on
 * TRIP_GAP_SECONDS, which is much tighter.) */
export const HEADWAY_PRUNE_SECONDS = 86400;

/** How often the reference-stop map is recomputed from segment_params.json.
 * The map is a function of the MTA timetable, which changes a few times a
 * year, so a daily refresh is already far more often than it can move —
 * and it keeps the per-minute path down to one small R2 read. */
export const REFERENCE_REFRESH_SECONDS = 86400;

/**
 * How many times step 0 will refold this poll's rows onto a concurrent
 * winner's document before giving up. A lost compare-and-swap means an
 * overlapping or retried cron advanced state/headway.json between our read and
 * our write; blindly rewriting would drop whatever passings the winner
 * recorded, and a dropped passing does not lose a datum, it merges two real
 * headways into one doubled reading. So the caller re-reads and folds onto the
 * winner instead. Overlap needs an invocation to outlive its own minute, so
 * one retry is already the unlikely case and a second is the safety margin.
 */
export const HEADWAY_WRITE_ATTEMPTS = 3;

/**
 * How many recent passings a cell keeps. The reading needs two; the rest is
 * headroom so an out-of-order merge (mergePassings) still lands inside the
 * window rather than being pruned away before it can be used.
 */
export const HEADWAY_LEDGER_SIZE = 6;

// 'route|direction' — the key `route_stops`, state/headway.json's cells and the
// offline toolkit all use — is built inline as `${route}|${direction}`.

/**
 * Pick the canonical reference stop for every (route, direction) present in
 * the scheduled stopping patterns.
 *
 * RULE (see the module comment for why): among the cell's through-stops, the
 * one carrying the most scheduled trips; ties break toward the stop nearest the
 * middle of the most-run pattern, then the smaller stop id. Deterministic in
 * `routeStops`, so two Workers reading the same segment_params.json agree.
 *
 * Cells with no through-stop at all — a one- or two-stop pattern, or an
 * observed-adjacency fallback doc with no real stopping patterns — are absent
 * from the result rather than falling back to a terminal.
 */
export function selectReferenceStops(
  routeStops: RouteStops,
): Record<string, ReferenceStop> {
  const out: Record<string, ReferenceStop> = {};
  for (const [key, patterns] of Object.entries(routeStops)) {
    const sep = key.indexOf('|');
    if (sep <= 0) continue;
    const route = key.slice(0, sep);
    const direction = key.slice(sep + 1);
    if (patterns.length === 0) continue;

    // The dominant-successor skeleton: per from_stop, the to_stop the most
    // scheduled trips continue to (ties on the smaller stop id). Same
    // construction as gtfs_static.successors() + dominant_successor().
    const succ = new Map<string, Map<string, number>>();
    for (const pattern of patterns) {
      for (let i = 0; i + 1 < pattern.stops.length; i++) {
        const from = pattern.stops[i]!;
        const to = pattern.stops[i + 1]!;
        let tos = succ.get(from);
        if (tos === undefined) {
          tos = new Map<string, number>();
          succ.set(from, tos);
        }
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

    // A through-stop has both a scheduled predecessor and a scheduled
    // successor in that skeleton.
    const dominantPattern = patterns[0]!.stops;
    const mid = dominantPattern.length / 2;
    const total = patterns.reduce((sum, p) => sum + p.n_trips, 0);
    let best: ReferenceStop | null = null;
    let bestRank: [number, number, string] | null = null;
    for (const stop of dominant.keys()) {
      if (!incoming.has(stop)) continue;
      let trips = 0;
      for (const p of patterns) if (p.stops.includes(stop)) trips += p.n_trips;
      const at = dominantPattern.indexOf(stop);
      const rank: [number, number, string] = [
        -trips,
        Math.abs((at === -1 ? dominantPattern.length : at) - mid),
        stop,
      ];
      if (bestRank === null || rankBefore(rank, bestRank)) {
        bestRank = rank;
        best = {
          route,
          direction,
          stop_id: stop,
          n_scheduled_trips: trips,
          coverage: total > 0 ? trips / total : 0,
        };
      }
    }
    if (best !== null) out[key] = best;
  }
  return out;
}

/** Lexicographic order on the (fewest-trips-first, distance-from-middle,
 * stop_id) rank tuple — the tie-break chain the selection rule documents. */
function rankBefore(
  a: [number, number, string],
  b: [number, number, string],
): boolean {
  if (a[0] !== b[0]) return a[0] < b[0];
  if (a[1] !== b[1]) return a[1] < b[1];
  return a[2] < b[2];
}

// The carried document's own shape (HeadwayStateDoc, HeadwayCell,
// HeadwayTrip) is defined in state.ts, where its zod schema is the single
// parse authority — same split as crowding.ts and StationWaitDoc. Its parts:
//
//   cells[key].passings — the LEDGER: recent passings of that reference stop,
//     ascending by `at`, capped at HEADWAY_LEDGER_SIZE. The published reading
//     is DERIVED from its last two distinct entries (cellHeadway), never
//     stored. A ledger rather than a scalar "last passing" because insertion
//     into it is commutative and idempotent, which is what makes a concurrent
//     merge sound and a retried poll a no-op — see mergePassings.
//   gaps — windows during which the Worker was not polling, on the DOCUMENT
//     rather than per cell: a feed outage is global, and a per-cell flag
//     consumed by whichever passing landed newest would make the refusal
//     depend on insertion order.
//   trips[].at — the last poll a trip was seen at its reference stop,
//     deliberately NOT refreshed while the trip is missing from the feed, so
//     TRIP_GAP_SECONDS can actually expire.

export function emptyHeadwayState(): HeadwayStateDoc {
  return {
    observed_at: 0,
    reference_at: 0,
    reference_trained_at: 0,
    reference_stops: {},
    cells: {},
    trips: {},
    gaps: [],
  };
}

/** Whether the reference-stop map should be recomputed from
 * segment_params.json: never computed, or older than the refresh cadence.
 * Checked BEFORE the trainer doc is read, so it takes no argument about it —
 * a trainer doc that landed inside the cadence is picked up at the next
 * refresh, which is the right latency for a quantity that moves with the
 * printed timetable. */
export function referenceStopsStale(
  doc: HeadwayStateDoc | null,
  now: number,
): boolean {
  if (doc === null) return true;
  if (Object.keys(doc.reference_stops).length === 0) return true;
  return now - doc.reference_at >= REFERENCE_REFRESH_SECONDS;
}

/** The measurement points a poll is folded against, and where they came
 * from. `at`/`trained_at` are provenance: which trainer doc picked these
 * stops, and when. */
export interface HeadwayReference {
  stops: Record<string, string>;
  at: number;
  trained_at: number;
}

/**
 * The reference block to measure this poll against.
 *
 * `routeStops` is segment_params.json's scheduled stopping patterns when the
 * refresh above read them, and null when it didn't. A doc that carries no
 * patterns at all — the trainer's observed-adjacency fallback — yields no
 * picks, and the carried block is kept rather than being blanked: an
 * unpublishable trainer doc must not silently move a live measurement point.
 * With nothing carried either, `stops` is empty and the surface abstains.
 */
export function resolveReference(
  prev: HeadwayStateDoc | null,
  routeStops: RouteStops | null,
  trainedAt: number,
  now: number,
): HeadwayReference {
  const carried: HeadwayReference = {
    stops: prev?.reference_stops ?? {},
    at: prev?.reference_at ?? 0,
    trained_at: prev?.reference_trained_at ?? 0,
  };
  if (routeStops === null) return carried;
  const stops: Record<string, string> = {};
  for (const [key, ref] of Object.entries(selectReferenceStops(routeStops))) {
    stops[key] = ref.stop_id;
  }
  if (Object.keys(stops).length === 0) return carried;
  return { stops, at: now, trained_at: trainedAt };
}

/** A passing of a reference stop, before it is folded into a cell. */
export interface HeadwayPassing {
  cell: string;
  stop: string;
  trip: string;
  at: number;
}

/**
 * The passings this poll's rows evidence, relative to `prev`'s trip carry.
 *
 * Separate from updateHeadwayState because the concurrent-writer merge path
 * (index.ts step 0) needs the passings WITHOUT the trip carry the same rows
 * would rebuild: see mergePassings.
 */
export function detectPassings(
  rows: TraceRow[],
  reference: HeadwayReference,
  prev: HeadwayStateDoc | null,
  now: number,
): HeadwayPassing[] {
  const refStops = reference.stops;
  const carriedTrips = prev?.trips ?? {};
  const out: HeadwayPassing[] = [];
  for (const row of rows) {
    if (row.direction === null) continue;
    const ref = refStops[`${row.route_id}|${row.direction}`];
    // Still at (or heading to) the reference stop: the departure is ahead.
    if (ref !== undefined && row.stop_id === ref) continue;
    const carried = carriedTrips[row.trip_id];
    if (carried === undefined) continue;
    if (now - carried.at > TRIP_GAP_SECONDS) continue; // may be a different train
    if (refStops[carried.cell] !== carried.stop) continue; // point moved
    // Last seen at a reference stop, now reporting somewhere else: it served
    // that stop and has left.
    out.push({
      cell: carried.cell,
      stop: carried.stop,
      trip: row.trip_id,
      at: passingTime(row, now),
    });
  }
  // Time order: two trains can clear the same stop within one poll, and the
  // later one's headway is measured against the earlier one.
  out.sort((a, b) => a.at - b.at || (a.trip < b.trip ? -1 : 1));
  return out;
}

/**
 * Fold ONLY these passings into an existing document, leaving its trip carry
 * alone apart from the trips they resolve.
 *
 * This is the concurrent-writer merge path. When a lost compare-and-swap
 * reveals that an invocation for a LATER minute already committed, our rows
 * are older than the stored document and must NOT be used to rebuild its trip
 * carry: a trip that document already resolved would be re-added, and a later
 * poll could then fire a second passing for it — the dup-window guard only
 * catches that when no other train has passed in between, so it is not a
 * guarantee. What our rows do know that the document does not is the passings
 * themselves, which is all this applies.
 *
 * Dropping each resolved trip from the carry is the other half: without it the
 * document keeps the trip pending and fires the same departure again later.
 * Insertion into the ledger is commutative and idempotent (insertPassing), so
 * a departure the winner already recorded is absorbed rather than
 * double-counted, and the resulting reading is the same whichever invocation
 * committed first.
 */
export function mergePassings(
  doc: HeadwayStateDoc,
  passings: HeadwayPassing[],
): HeadwayStateDoc {
  if (passings.length === 0) return doc;
  const cells: Record<string, HeadwayCell> = {};
  for (const [key, cell] of Object.entries(doc.cells)) {
    cells[key] = { stop_id: cell.stop_id, passings: [...cell.passings] };
  }
  const trips = { ...doc.trips };
  for (const p of passings) {
    if (doc.reference_stops[p.cell] !== p.stop) continue; // point moved since
    insertPassing(cells, p);
    delete trips[p.trip];
  }
  return { ...doc, cells, trips };
}

/**
 * Fold this poll's vehicle trace into the headway carry.
 *
 * `reference` is the measurement points and their provenance (see
 * resolveReference). `now` is the poll's own stamp, used for the
 * carry's age arithmetic; a passing itself is stamped with the departing
 * vehicle's own `vehicle_ts` when the feed supplied a believable one, which is
 * finer than the 1-minute cadence — the same preference
 * training/trace._row_time makes. "Believable" means within FEED_GAP_SECONDS
 * of the poll: a vehicle whose clock is frozen or running ahead would otherwise
 * poison both this headway and the next.
 *
 * Cells whose reference stop is no longer the one in `reference.stops` are
 * dropped rather than reinterpreted — a measurement point that moved starts a
 * new series.
 *
 * Callers must skip a poll that produced no rows (a total feed outage or a
 * decode throw), exactly as index.ts step 0 does for the platform-wait carry:
 * folding zero rows in would stamp a fresh observed_at over frozen state and
 * defeat the freshness gate that ages this surface out.
 */
export function updateHeadwayState(
  rows: TraceRow[],
  reference: HeadwayReference,
  prev: HeadwayStateDoc | null,
  now: number,
): HeadwayStateDoc {
  const refStops = reference.stops;
  const base = prev ?? emptyHeadwayState();

  // Carry cells forward only where the measurement point still agrees.
  const cells: Record<string, HeadwayCell> = {};
  for (const [key, cell] of Object.entries(base.cells)) {
    if (refStops[key] !== cell.stop_id) continue;
    const newest = cell.passings[cell.passings.length - 1];
    if (newest === undefined) continue;
    if (now - newest.at > HEADWAY_PRUNE_SECONDS) continue;
    cells[key] = { stop_id: cell.stop_id, passings: [...cell.passings] };
  }

  // A poll gap means the Worker was not observing between the last fold and
  // this one, so any interval crossing that window may be missing observation
  // rather than missing service. Recorded as a window on the document, never
  // as a per-cell flag — see state.ts.
  const gaps = pruneGaps(base.gaps, now);
  if (base.observed_at > 0 && now - base.observed_at > FEED_GAP_SECONDS) {
    gaps.push({ from: base.observed_at, until: now });
  }

  // Rebuild the trip carry from this poll's own rows: which trips are at a
  // reference stop now, plus those still pending from before.
  const trips: Record<string, HeadwayTrip> = {};
  const seen = new Set<string>();
  for (const row of rows) {
    seen.add(row.trip_id);
    if (row.direction === null) continue;
    const key = `${row.route_id}|${row.direction}`;
    const ref = refStops[key];
    if (ref === undefined || row.stop_id !== ref) continue;
    // At the reference stop, heading to it or standing at it. Either way the
    // departure is still ahead, so refresh the carry and wait for it.
    trips[row.trip_id] = { cell: key, stop: ref, at: now };
  }

  // A trip absent from this poll has not departed anything yet — carry it
  // unrefreshed so TRIP_GAP_SECONDS can expire on it, rather than losing the
  // pending departure to one missed sighting.
  for (const [tripId, carried] of Object.entries(base.trips)) {
    if (seen.has(tripId)) continue;
    if (now - carried.at > TRIP_GAP_SECONDS) continue;
    if (refStops[carried.cell] !== carried.stop) continue;
    trips[tripId] = carried;
  }

  for (const p of detectPassings(rows, reference, prev, now)) insertPassing(cells, p);

  return {
    observed_at: now,
    reference_at: reference.at,
    reference_trained_at: reference.trained_at,
    reference_stops: { ...refStops },
    cells,
    trips,
    gaps,
  };
}

/** The departing vehicle's own report time when the feed supplied a believable
 * one, else the poll's stamp. See updateHeadwayState. */
function passingTime(row: TraceRow, now: number): number {
  const ts = row.vehicle_ts;
  if (ts !== null && ts > 0 && Math.abs(ts - now) <= FEED_GAP_SECONDS) return ts;
  return now;
}

/**
 * Insert a passing into a cell's ledger, in place.
 *
 * COMMUTATIVE AND IDEMPOTENT, which is the whole reason the ledger exists.
 * Applying the same set of passings in any order, or twice, yields the same
 * cell — so a concurrent merge (mergePassings) is sound and a retried poll
 * cannot corrupt anything. The previous shape kept a scalar `last_at` and
 * folded sequentially, which could not do this: two invocations that each saw
 * a different train from the same base produced a cell in which the older
 * passing was refused for being behind the newer one, and the published
 * headway spanned both intervals instead of the last one.
 *
 * The dup-window refusal (one train re-reported under the same trip_id) is the
 * only rejection here; the sanity bounds and the gap refusal belong to the
 * DERIVATION (cellHeadway), because they are statements about an interval
 * rather than about the ledger.
 */
function insertPassing(cells: Record<string, HeadwayCell>, p: HeadwayPassing): void {
  let cell = cells[p.cell];
  if (cell === undefined) {
    cell = { stop_id: p.stop, passings: [] };
    cells[p.cell] = cell;
  }
  for (const existing of cell.passings) {
    if (existing.at === p.at && existing.trip === p.trip) return; // idempotent
    if (existing.trip === p.trip && Math.abs(existing.at - p.at) < DUP_ARRIVAL_SECONDS) {
      return; // one train re-reported, not two trains
    }
  }
  cell.passings.push({ at: p.at, trip: p.trip });
  cell.passings.sort((a, b) => a.at - b.at || (a.trip < b.trip ? -1 : 1));
  if (cell.passings.length > HEADWAY_LEDGER_SIZE) {
    cell.passings.splice(0, cell.passings.length - HEADWAY_LEDGER_SIZE);
  }
}

/**
 * Drop gap windows too old to affect any interval that could still publish.
 *
 * A publishable reading closes no earlier than `now - MAX_READING_AGE_SECONDS`
 * and spans at most MAX_HEADWAY_SECONDS, so nothing before their sum can
 * overlap it. That makes the prune exactly safe rather than a guess, and it
 * bounds the list: every gap needs its own FEED_GAP_SECONDS of silence, so only
 * a few dozen fit inside that horizon.
 */
function pruneGaps(
  gaps: { from: number; until: number }[],
  now: number,
): { from: number; until: number }[] {
  const horizon = now - (MAX_READING_AGE_SECONDS + MAX_HEADWAY_SECONDS);
  return gaps.filter((g) => g.until > horizon);
}

/**
 * The cell's current reading: the interval between its two most recent
 * DISTINCT trains, or null when there is no publishable one.
 *
 * A pure function of the ledger and the document's gap windows, so it is
 * order-independent like the ledger itself.
 *
 * Entries closer together than MIN_HEADWAY_SECONDS are collapsed to the
 * earlier one before the pair is taken. Two trains of one route cannot clear a
 * platform that close, so the later entry is one train reported under a second
 * trip_id — NYCT reassigns ids mid-run, which the dup-window guard cannot see
 * because the ids differ. Collapsing keeps the true previous headway readable
 * instead of letting a phantom entry blank the cell.
 *
 * Refuses, in each case producing no reading rather than a zero or a guess:
 * fewer than two distinct trains (no interval yet); an interval overlapping a
 * window in which the Worker was not polling (missing observation, not missing
 * service); a value above the sanity bound.
 */
function cellHeadway(
  cell: HeadwayCell,
  gaps: { from: number; until: number }[],
): { value: number; at: number } | null {
  const distinct: { at: number; trip: string }[] = [];
  for (const entry of cell.passings) {
    const kept = distinct[distinct.length - 1];
    if (kept !== undefined && entry.at - kept.at < MIN_HEADWAY_SECONDS) continue;
    distinct.push(entry);
  }
  if (distinct.length < 2) return null;
  const last = distinct[distinct.length - 1]!;
  const prev = distinct[distinct.length - 2]!;
  const value = last.at - prev.at;
  if (value > MAX_HEADWAY_SECONDS) return null;
  for (const g of gaps) {
    if (g.from < last.at && g.until > prev.at) return null; // interval is feed-uncertain
  }
  return { value, at: last.at };
}

/** One published headway measurement — structurally the snapshot's
 * ObservationOut (snapshot.ts), kept here so this module never imports back
 * from it. */
export interface HeadwayObservation {
  entity_ref: string;
  kind: 'headway';
  value: number;
  unit: 'seconds';
  observed_at: number;
  source: string;
  // Closed vocabulary, matching schema.py's ObservationDirection Literal —
  // NYCT runs exactly two directions and the repo normalises both to these
  // words (vehicles.ts directionOf). A third would be a schema change.
  direction: 'north' | 'south';
  stop_id: string;
}

/** Names the upstream this measurement came from, as Observation.source. The
 * honest label for the boundary this crosses: the GTFS-RT vehicle-position
 * protobuf now reaches the public snapshot through this surface. */
export const HEADWAY_SOURCE = 'gtfs_rt_vehicle_positions';

/**
 * The publishable headway observations in the carried state, sorted by
 * (route, direction) so the surface is byte-stable across ticks that measure
 * the same thing.
 *
 * A cell with no completed reading, or one older than
 * MAX_READING_AGE_SECONDS, is absent — never zero, never carried forward. The
 * caller is expected to have already aged out the document as a whole (its
 * `observed_at`), which is a different check: the doc's age says whether the
 * feed is being polled at all, a cell's says whether trains are actually
 * running past its stop.
 */
export function headwayObservations(
  doc: HeadwayStateDoc,
  now: number,
): HeadwayObservation[] {
  const out: HeadwayObservation[] = [];
  for (const key of Object.keys(doc.cells).sort()) {
    const cell = doc.cells[key]!;
    const reading = cellHeadway(cell, doc.gaps);
    if (reading === null) continue;
    if (now - reading.at > MAX_READING_AGE_SECONDS) continue;
    const sep = key.indexOf('|');
    if (sep <= 0) continue;
    // Cell keys are only ever built from TraceRow.direction, so this holds by
    // construction; a carried key that says otherwise is corrupt state and is
    // dropped rather than published as a direction no consumer can read.
    const direction = key.slice(sep + 1);
    if (direction !== 'north' && direction !== 'south') continue;
    out.push({
      entity_ref: `subway_route:${key.slice(0, sep)}`,
      kind: 'headway',
      value: reading.value,
      unit: 'seconds',
      observed_at: reading.at,
      source: HEADWAY_SOURCE,
      direction,
      stop_id: cell.stop_id,
    });
  }
  return out;
}
