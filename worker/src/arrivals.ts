/**
 * Per-stop next-N arrivals, derived from the decoded GTFS-RT trip-updates.
 *
 * gtfsrt.ts materializes each StopTimeUpdate (stop_id incl. direction suffix,
 * absolute POSIX arrival/departure times, schedule_relationship). This folds
 * those rows into a ranked list per stop id: the next few trains due, soonest
 * first, one entry per physical trip. Route is folded to its base (6X -> 6)
 * exactly as the service metric does; direction is carried by the stop id's own
 * suffix (Q05S / 626N), so northbound and southbound key separately.
 *
 * No schedule join and no delay math: eta_epoch is the feed's own time,
 * seconds_away is just time - now. Absolute times, carried through untouched.
 */

import type { TripLite } from './gtfsrt';
import { baseRoute } from './trip_updates';

/** One upcoming train at a stop. `trip_id` is the NYCT run id, null when the
 * feed omitted it. Times are absolute POSIX seconds; seconds_away = time - now. */
export interface Arrival {
  route: string;
  eta_epoch: number;
  seconds_away: number;
  trip_id: string | null;
}

/** Keyed by GTFS stop id incl. direction suffix (e.g. 'Q05S'), soonest first. */
export type Arrivals = Record<string, Arrival[]>;

// Don't publish a countdown further out than this — past an hour the trip-
// update times are speculative and just pad the list.
const HORIZON_SECONDS = 60 * 60;
// Most a rider reads off a countdown board is the next few trains.
const MAX_ARRIVALS = 6;

/**
 * Fold decoded trips (across all fetched feeds) into per-stop arrivals.
 *
 * One pass over every stop-time row, pushing into a per-stop array, then a
 * sort + truncate per stop at the end. A row is dropped when it is SKIPPED,
 * carries neither an arrival nor a departure time, is keyed to no stop, is
 * already in the past, or sits beyond the horizon. One entry per trip per stop
 * — the first occurrence wins, so a trip listed twice for a stop counts once.
 */
export function deriveArrivals(trips: TripLite[], now: number): Arrivals {
  const out: Arrivals = {};
  // Per-stop set of trip ids already counted, for the one-entry-per-trip rule.
  const seen = new Map<string, Set<string>>();
  const horizon = now + HORIZON_SECONDS;

  for (const t of trips) {
    const route = baseRoute(t.routeId);
    for (const st of t.stopTimes) {
      if (st.scheduleRelationship === 1) continue; // SKIPPED
      const time = st.arrival ?? st.departure;
      if (time === null) continue;
      const stopId = st.stopId;
      if (stopId === '') continue;
      if (time < now || time > horizon) continue;

      let stopSeen = seen.get(stopId);
      if (stopSeen === undefined) {
        stopSeen = new Set<string>();
        seen.set(stopId, stopSeen);
        out[stopId] = [];
      }
      // An anonymous trip cannot be told from another anonymous trip, so it is
      // never deduped: two trains the feed left unnamed are two arrivals.
      if (t.tripId !== '') {
        if (stopSeen.has(t.tripId)) continue; // one entry per trip per stop
        stopSeen.add(t.tripId);
      }

      out[stopId]!.push({
        route,
        eta_epoch: time,
        seconds_away: time - now,
        trip_id: t.tripId === '' ? null : t.tripId,
      });
    }
  }

  for (const stopId of Object.keys(out)) {
    const list = out[stopId]!;
    list.sort((a, b) => a.eta_epoch - b.eta_epoch);
    if (list.length > MAX_ARRIVALS) list.length = MAX_ARRIVALS;
  }
  return out;
}
