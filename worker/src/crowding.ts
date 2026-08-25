/**
 * Per-platform train-departure tracking -> live crowding estimate.
 *
 * Two stages, matching the two artifacts in the crowding contract:
 *
 *   1. updateStationWait folds this tick's per-minute vehicle trace
 *      (vehicles.ts deriveTrace) into state/station_wait.json: for every
 *      directional platform, the epoch a train was last seen there or
 *      cleared it. Runs every minute at step 0 of index.ts, off the
 *      5-minute pipeline, because the whole point is 1-minute resolution on
 *      when a platform emptied.
 *
 *   2. derivePlatformCrowding turns that wait state into the published
 *      surface at snapshot build time: each platform's assumed share of its
 *      station complex's usual entries/min (training/ridership.py's weekly
 *      baseline) times minutes since a train was last there.
 *
 * Sibling to segment_flow.ts in structure (own small state object, own pure
 * derive function) but not in domain: segment_flow tracks whether a SEGMENT
 * is moving, this tracks how long a PLATFORM has sat empty.
 */

import { ridershipRateFor } from './params';
import type { RidershipBaselineDoc } from './params';
import { stationId } from './segment_flow';
import type { StationWaitDoc } from './state';
import type { StationOut } from './stations_static';
import type { TraceRow } from './vehicles';

// Beyond this, the linear waiting_riders = rate * minutes model stops
// describing a crowd: people who have waited this long give up and leave,
// and a platform silent this long is usually out of service rather than
// jammed. Measured on 2026-08-20 07:00-11:00 ET (982 platforms, 28,670
// gaps): p50 = 6.0 min, p90 = 13, p99 = 31, max = 232 — the cap sits just
// past p99, so it abstains ('gap_exceeds_cap') on only 1.04% of real gaps.
// Inclusive: a gap of exactly this many minutes abstains too.
//
// A gate abstaining any platform some trip claimed to be inbound to (never
// caught STOPPED_AT, stop_id unadvanced) past this cap was tried and
// measured against the real trace (2026-08-20 07:00-09:00 ET, 89,352 rows)
// and rejected: it abstained 23 platforms, of which 25 individually had a
// fresh own gap under the cap (24 of those under 10 minutes) — it was
// overwhelmingly removing platforms with real, recent service, because one
// route's stuck vehicle does not invalidate a departure another route
// genuinely made minutes earlier. It also never caught the reading that
// motivated it: that route's trip/stop_id churns every few minutes, which
// kept resetting the elapsed clock the gate keyed on, so the one platform
// that needed it never crossed the threshold. Don't re-add this gate
// without a carry keyed on something the stalled feed can't reset.
export const CROWDING_MAX_GAP_MINUTES = 30;

// A platform counts toward its complex's served-platform denominator (the
// uniform split across "platforms currently running") only if a train
// passed it within this long. Looser than CROWDING_MAX_GAP_MINUTES on
// purpose: a platform 45 minutes quiet is too stale to publish ITS OWN
// estimate, but a train that recent still means the platform is in service
// and should keep absorbing its share of the complex's demand. Only a
// platform gone dark for a full hour stops counting, so an overnight-closed
// platform can't silently inflate the denominator and steal demand from the
// platforms actually running.
export const CROWDING_SERVED_WINDOW_MINUTES = 60;

// Prune platforms this stale from state/station_wait.json regardless of the
// (much tighter) publish cap above, purely to keep the carried R2 object
// bounded across a multi-day vehicle-feed gap — a full service day of
// silence is well past the point any of this state is still useful for
// anything, including a future re-derivation.
export const WAIT_PRUNE_SECONDS = 86400;

/**
 * Fold this tick's vehicle trace into the platform-wait carry.
 *
 * Per row, in order:
 *   1. If this trip was last seen at platform P and is now at a different
 *      stop C, P just cleared: platforms[P] = max(platforms[P], now).
 *   2. If the vehicle is STOPPED_AT C right now, a train is there:
 *      platforms[C] = max(platforms[C], now).
 *   3. Carry trips[trip_id] = C for next tick's comparison.
 *
 * Step 1 is not optional, and step 2 is not a substitute for it. Deriving "a
 * train left this platform" from `stopped === true` sightings alone misses
 * 7.0% of real station departures (1,078 of 15,382 over a 3-hour window),
 * because a dwell shorter than the 1-minute poll interval can come and go
 * between two ticks without ever being caught STOPPED_AT — the stop_id simply
 * changes underneath it. Replaying 2026-08-20 07:00-11:00 ET puts a direct
 * number on the alternative: of 47,936 stop transitions, only 44,441 (92.71%)
 * had the trip caught STOPPED_AT on the immediately preceding observation, so
 * gating departures on that loses 3,495 of them and freezes those platforms'
 * clocks on trains that already left.
 *
 * The two steps are not in tension. Step 1 stamps the departure poll, which is
 * always at or after any earlier stopped sighting at the same platform, so
 * max() resolves to the departure whenever one is observed. Step 2 only
 * governs the interval while a train is physically standing there, where a
 * zero wait is the reading we want. (The one ordering that could strand a
 * departure — same stop_id going stopped -> moving — occurs 79 times in those
 * 47,936 transitions, 0.16%, and merely defers detection by one poll.)
 *
 * `platforms` values only ever move forward (Math.max against the prior
 * value), so replaying the same tick's rows against the same `prev` — a
 * retried cron minute — reproduces the identical doc. That idempotency is
 * also why the R2 write is a plain put, no CAS (see state.ts).
 *
 * Prunes platforms untouched for WAIT_PRUNE_SECONDS, and drops any trip not
 * present in this tick's rows: it's no longer running, or a feed gap means
 * there's simply nothing to compare it against next tick.
 */
export function updateStationWait(
  rows: TraceRow[],
  prev: StationWaitDoc | null,
  now: number,
): StationWaitDoc {
  const platforms: Record<string, number> = { ...(prev?.platforms ?? {}) };
  const trips: Record<string, string> = {};

  for (const row of rows) {
    const prevStop = prev?.trips[row.trip_id];
    if (prevStop !== undefined && prevStop !== row.stop_id) {
      platforms[prevStop] = Math.max(platforms[prevStop] ?? 0, now);
    }
    if (row.stopped) {
      platforms[row.stop_id] = Math.max(platforms[row.stop_id] ?? 0, now);
    }
    trips[row.trip_id] = row.stop_id;
  }

  const prunedPlatforms: Record<string, number> = {};
  for (const [stopId, at] of Object.entries(platforms)) {
    if (now - at <= WAIT_PRUNE_SECONDS) prunedPlatforms[stopId] = at;
  }

  return { observed_at: now, platforms: prunedPlatforms, trips };
}

/** One platform's crowding estimate — structurally PlatformCrowdingEstimateOut
 * (snapshot.ts), kept as its own type here so this module never has to
 * import back from snapshot.ts. */
export interface PlatformCrowdingEstimate {
  last_train_at: number;
  entries_per_min: number;
  waiting_riders: number;
}

/** derivePlatformCrowding's return: everything the published surface needs
 * except `method`, which snapshot.ts attaches (the constants above plus the
 * baseline document's own provenance). */
export interface PlatformCrowdingResult {
  observed_at: number;
  platforms: Record<string, PlatformCrowdingEstimate>;
  n_platforms: number;
  abstained: Record<string, number>;
}

/**
 * Turn the wait state into the published per-platform crowding estimate.
 *
 * For every platform in `wait.platforms`: resolve its parent station
 * (stationId — strip a trailing N/S), that station's complex id, and the
 * complex's usual entries/min for the current (weekday/weekend, hour) cell.
 * Split that rate evenly across however many of the complex's platforms
 * have themselves seen a train within CROWDING_SERVED_WINDOW_MINUTES — an
 * ASSUMPTION (see PlatformCrowdingMethod.split_basis in the schema): no feed
 * says which platform a rider actually walked to. Multiply the platform's
 * own share by minutes since ITS last train.
 *
 * Abstains rather than fabricates, with the exact reason:
 *   - 'unknown_stop': the platform's parent station isn't in `stations`, or
 *     has no station_complex_id.
 *   - 'no_baseline': the complex isn't in the ridership baseline.
 *   - 'no_recent_train': not even this platform counts toward its own
 *     complex's served-platform denominator — its last train predates
 *     CROWDING_SERVED_WINDOW_MINUTES, and (since a served platform always
 *     counts itself) so does every sibling that also failed the test. The
 *     whole complex reads as out of service.
 *   - 'gap_exceeds_cap': the platform IS within the served window (so it
 *     still absorbs its share of complex demand for its neighbors' splits)
 *     but its own gap is at or past CROWDING_MAX_GAP_MINUTES — too stale to
 *     publish an estimate for.
 */
export function derivePlatformCrowding(
  wait: StationWaitDoc,
  baseline: RidershipBaselineDoc,
  stations: Record<string, StationOut>,
  now: number,
): PlatformCrowdingResult {
  const servedWindowSeconds = CROWDING_SERVED_WINDOW_MINUTES * 60;

  // How many platforms per complex currently sit within the served window —
  // the shared denominator every platform in that complex divides by.
  const servedByComplex: Record<string, number> = {};
  for (const [stopId, lastAt] of Object.entries(wait.platforms)) {
    const complexId = stations[stationId(stopId)]?.station_complex_id;
    if (complexId != null && now - lastAt <= servedWindowSeconds) {
      servedByComplex[complexId] = (servedByComplex[complexId] ?? 0) + 1;
    }
  }

  const platforms: Record<string, PlatformCrowdingEstimate> = {};
  const abstained: Record<string, number> = {};

  for (const [stopId, lastAt] of Object.entries(wait.platforms)) {
    const complexId = stations[stationId(stopId)]?.station_complex_id;
    if (complexId == null) {
      abstained.unknown_stop = (abstained.unknown_stop ?? 0) + 1;
      continue;
    }
    const rate = ridershipRateFor(baseline, complexId, now);
    if (rate === null) {
      abstained.no_baseline = (abstained.no_baseline ?? 0) + 1;
      continue;
    }
    const served = servedByComplex[complexId] ?? 0;
    if (served === 0) {
      abstained.no_recent_train = (abstained.no_recent_train ?? 0) + 1;
      continue;
    }
    const minutesSince = (now - lastAt) / 60;
    if (minutesSince >= CROWDING_MAX_GAP_MINUTES) {
      abstained.gap_exceeds_cap = (abstained.gap_exceeds_cap ?? 0) + 1;
      continue;
    }
    // Round the published rate, but compute the count off the unrounded one.
    // `rate / served` is a mean of hourly sums divided by a small integer, so
    // it lands on values like 50.819800000000004 — float noise in a public
    // contract that ~900 platforms then pay for in bytes. Two decimals is
    // 0.6 riders/hour, already finer than a 90-day mean can support.
    const entriesPerMin = rate / served;
    platforms[stopId] = {
      last_train_at: lastAt,
      entries_per_min: Math.round(entriesPerMin * 100) / 100,
      waiting_riders: Math.round(entriesPerMin * minutesSince),
    };
  }

  return {
    observed_at: now,
    platforms,
    n_platforms: Object.keys(platforms).length,
    abstained,
  };
}
