/**
 * Momentarily publisher — Cloudflare Worker entry point.
 *
 * Each cron tick:
 *   0. Fetch + decode the vehicle-position feeds and run the per-minute
 *      movement trace (every tick — see the hazard note at the top of
 *      `scheduled` for why this is the ONLY thing that runs off the
 *      5-minute boundary)
 *   1. Read rolling state (last_seen, alpha) + trained params from R2
 *   2. Fetch the MTA alerts feed
 *   3. Archive new (alert_id, updated_at) versions
 *   4. Derive per-route observations + advance the forward filter
 *   5. Persist alpha.json via etag CAS — a losing run yields the tick here
 *   6. (Only if alpha persisted) Render + publish snapshot.json to R2
 *   7. (Only if alpha persisted) Write predictions + transitions grading streams
 *   8. Hourly: fetch the 3 E&E feeds and archive snapshots
 *   9. Persist last_seen.json via etag CAS
 *
 * Steps 1-9 are the 5-minute pipeline and only run on a 5-minute boundary
 * (minute % 5 === 0) even though the cron itself now fires every minute — see
 * wrangler.toml and the gate at the top of `scheduled`.
 */

import type { AlphaState, RouteRoll } from './alpha';
import { readAlphaState, reseedForNewParams, writeAlphaState } from './alpha';
import {
  archiveEneSnapshot,
  archiveNewAlerts,
  archiveTraceRows,
  archiveTripUpdateMetric,
  archiveVehicleMetric,
} from './archive';
import { updateStationWait } from './crowding';
import type { RouteSnapshot } from './derive';
import { SUBWAY_ROUTES, buildAlertList, deriveRouteSnapshots, quietObservation } from './derive';
import { parseEquipmentFeed, parseOutageFeed } from './ene';
import {
  FEEDS,
  STATIONS_FEED,
  TRIP_UPDATE_FEEDS,
  TRIP_UPDATE_FEED_NAMES,
  fetchJson,
  fetchProtobuf,
} from './fetch';
import type { TripLite, VehicleLite } from './gtfsrt';
import { decodeTripUpdates, decodeVehicles } from './gtfsrt';
import { deriveRouteServiceMetric } from './trip_updates';
import type { TraceRow } from './vehicles';
import {
  deriveRouteMovementMetric,
  deriveTrace,
  stopPositions,
  trainPositions,
} from './vehicles';
import {
  MOVEMENT_STATE_PUBLISH,
  deriveMovementStates,
  SERVICE_DEBOUNCE_TICKS,
  deriveServiceQuantileRatios,
  deriveServiceRatios,
  deriveServiceStates,
  seedNormalServiceRegimes,
  movementObservationFields,
  serviceObservationFields,
} from './movement_state';
import type { PredictionRecord } from './grading';
import {
  detectTransitions,
  movementTransitions,
  writeMovementTransitions,
  writePredictions,
  writeTransitions,
} from './grading';
import type { FilterState, Observation, PublishedState } from './hmm';
import {
  emissionsFor,
  forwardStep,
  initialPublishedState,
  movementChannelActive,
  stationaryDistribution,
} from './hmm';
import { loadParams, loadRidershipBaseline, loadServiceWeightBaseline, paramsForRoute } from './params';
import { advanceRegimes, pruneIdleRegimes } from './regime';
import { deriveSegmentStates, deriveStationFlow, pruneSegmentRegimes, updateSegmentFlow } from './segment_flow';
import {
  TICK_SECONDS,
  buildSnapshot,
  buildTrains,
  publishSnapshot,
  publishTrains,
} from './snapshot';
import { buildEquipmentList, deriveStationStatuses } from './stations';
import { parseStationsFeed, readStationsCache, writeStationsCache } from './stations_static';
import {
  readLastSeen,
  readMovementMetric,
  readMovementState,
  readSegmentDwell,
  readSegmentFlow,
  readSegmentParams,
  readServiceBaseline,
  readServiceMetric,
  readStationFlow,
  readStationWait,
  readVehicleStops,
  writeLastSeen,
  writeMovementMetric,
  writeMovementState,
  writeSegmentFlow,
  writeServiceMetric,
  writeStationFlow,
  writeStationWait,
  writeVehicleStops,
} from './state';
import type { SegmentDwellDoc, SegmentFlowDoc, SegmentParamsDoc, StationFlowDoc, StationWaitDoc } from './state';

export interface Env {
  MOMENTARILY: R2Bucket;
}

// Only this prefix is served publicly. Everything else in the bucket (state/,
// archive/) stays private — the Worker is the auth boundary, the R2 custom
// domain must NOT be bound directly to the bucket.
const PUBLIC_PREFIX = 'v1/';

const ENE_INTERVAL_SECONDS = 3600;
const STATIONS_INTERVAL_SECONDS = 86_400;

async function handlePublicRead(request: Request, env: Env): Promise<Response> {
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    return new Response('Method Not Allowed', { status: 405 });
  }
  // `new URL` normalizes "..", so a key derived from pathname can't escape the
  // prefix — but the explicit startsWith check below is the real guard.
  const key = new URL(request.url).pathname.replace(/^\/+/, '');
  if (key === '') {
    return new Response(
      'Momentarily publisher. Public snapshot: https://feed.momentarily.nyc/v1/snapshot.json\n',
      { headers: { 'content-type': 'text/plain; charset=utf-8' } },
    );
  }
  if (!key.startsWith(PUBLIC_PREFIX)) {
    return new Response('Not Found', { status: 404 });
  }
  const obj = await env.MOMENTARILY.get(key);
  if (obj === null) {
    return new Response('Not Found', { status: 404 });
  }
  const headers = new Headers();
  obj.writeHttpMetadata(headers); // content-type + cache-control as stored on write
  headers.set('etag', obj.httpEtag);
  headers.set('access-control-allow-origin', '*');
  return new Response(request.method === 'HEAD' ? null : obj.body, { headers });
}

const ENE_SOURCES = [
  ['ene_current', FEEDS.ene_current],
  ['ene_upcoming', FEEDS.ene_upcoming],
  ['ene_equipments', FEEDS.ene_equipments],
] as const;

/** UTC minute-of-hour for a tick's observedAt (POSIX seconds). Broken out as
 * a pure function — rather than reading Date.now() again inside `scheduled`
 * — so the 5-minute boundary gate is unit-testable against an arbitrary
 * timestamp.
 *
 * The gate MUST read the cron's SCHEDULED minute, never the wall clock at
 * execution. Cloudflare does not promise punctuality, and a boundary invocation
 * that starts a few seconds late — enough to tip Date.now() into the next minute
 * — would fail the `% 5` test and silently skip the ENTIRE 5-minute pipeline for
 * that cycle: no snapshot, no state advance, nothing published, for five
 * minutes, with only a log line to show for it. `scheduledTime` is the minute
 * the trigger was meant to fire, so a late start still does its work.
 */
export function tickMinute(observedAt: number): number {
  return new Date(observedAt * 1000).getUTCMinutes();
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    return handlePublicRead(request, env);
  },

  async scheduled(
    event: ScheduledController,
    env: Env,
    _ctx: ExecutionContext,
  ): Promise<void> {
    const observedAt = Math.floor(Date.now() / 1000);
    const t0 = Date.now();
    const step = (label: string): void => {
      console.log(`step ${label} t+${Date.now() - t0}ms`);
    };
    console.log(`tick cron=${event.cron} t=${observedAt}`);

    // --- Step 0: per-minute vehicle trace, and the 5-minute pipeline gate ---
    //
    // THE HAZARD THIS GUARDS AGAINST: every step from here through step 9 is
    // defined per 5-MINUTE TICK. advanced_n/stalled_n (vehicles.ts) mean "did
    // this trip's stop_id change in 5 minutes"; the trained advance-rate
    // baseline (params.ts) is fitted on that cadence; the dwell model's
    // one-tick point mass IS 5 minutes; state/vehicle_stops.json is the
    // 5-minute carry map. The cron now fires every minute (wrangler.toml) so
    // that the trace below can poll at 1-minute resolution — but if steps 1-9
    // ran on every one of those fires, advanced_n/stalled_n etc. would
    // silently become 1-MINUTE quantities while every trained param still
    // assumed 5 minutes. Same code, same shape, quietly wrong numbers. So
    // steps 1-9 are gated to run ONLY on a 5-minute boundary (minute % 5 ===
    // 0), reading/writing vehicle_stops.json exactly as often as before this
    // change and seeing exactly the same inputs.
    //
    // The trace is the one thing allowed to run every minute. It never reads
    // or writes vehicle_stops.json, and it archives under its OWN prefix
    // (archive/trace/, see archive.ts) — never archive/vehicles/ — so it can
    // add finer-grained observations without ever touching the 5-minute
    // signal above. See vehicles.ts's deriveTrace for why it writes a full
    // snapshot every poll rather than a delta.
    // Gate off the cron's SCHEDULED minute, not observedAt. observedAt is the
    // wall clock at execution and is right for stamping data, but wrong for the
    // gate: a boundary run that starts a few seconds late would read as minute 6
    // and skip the whole 5-minute pipeline. See tickMinute.
    const scheduledAt = Math.floor(event.scheduledTime / 1000);
    const minute = tickMinute(scheduledAt);
    const isFiveMinuteBoundary = minute % 5 === 0;

    // Fetch + decode the vehicle-position side of the protobuf feeds
    // unconditionally, every tick — the trace needs it every minute, and step
    // 8b below needs the identical fetch on a boundary tick. Fetching here
    // once and threading `feedResults`/`vehicles`/`vehicleFreshFeeds` through
    // means a boundary tick never fetches the feed twice.
    const feedResults = await Promise.allSettled(
      TRIP_UPDATE_FEEDS.map(([, url]) => fetchProtobuf(url)),
    );
    const vehicles: VehicleLite[] = [];
    const vehicleFreshFeeds: string[] = [];
    for (let i = 0; i < feedResults.length; i++) {
      const r = feedResults[i]!;
      const name = TRIP_UPDATE_FEEDS[i]![0];
      if (r.status === 'fulfilled') {
        vehicleFreshFeeds.push(name);
        vehicles.push(...decodeVehicles(r.value));
      } else {
        console.error(`trip-updates ${name} failed:`, r.reason);
      }
    }
    let rows: TraceRow[] = [];
    try {
      rows = deriveTrace(vehicles);
      await archiveTraceRows(
        env.MOMENTARILY,
        rows,
        vehicleFreshFeeds,
        observedAt,
        scheduledAt,
      );
      console.log(`trace: ${rows.length} rows from ${vehicles.length} vehicles`);
    } catch (err) {
      console.error('trace step failed:', err);
    }

    // Per-platform train-departure tracking for the live crowding surface
    // — see crowding.ts's updateStationWait. Runs every minute,
    // right alongside the trace above, deliberately NOT gated behind the
    // 5-minute boundary below: the whole point is 1-minute resolution on
    // when a train cleared a platform, which is exactly the signal a
    // 5-minute cadence would blur past recognition (see the hazard comment
    // above — vehicle_stops.json's 5-minute stop_id carry serves a
    // different, coarser purpose and must stay untouched by this). Reads
    // and writes ONLY state/station_wait.json, its own R2 object — never
    // vehicle_stops.json, never archive/vehicles/. A read/write failure
    // degrades to publishing without the crowding surface this tick, never
    // a failed tick.
    //
    // A tick that produced NO trace rows is skipped entirely rather than
    // folded in as an empty observation, and that distinction is load-bearing
    // twice over. updateStationWait prunes trips absent from the rows it is
    // given, so folding in zero rows would wipe the trip -> stop carry the
    // departure rule depends on, blinding it for the tick after the feed
    // returns. And it would stamp a fresh observed_at over frozen platform
    // timestamps, which is precisely the state buildSnapshot's freshness gate
    // exists to catch: the surface would keep publishing an ageing crowd as
    // though it were current. Leaving the prior doc untouched lets the gate
    // age the surface out on its own. Zero rows means a total vehicle-feed
    // outage or a throw above, never a real empty system.
    let stationWaitDoc: StationWaitDoc | null = null;
    if (rows.length > 0) {
      try {
        const prevStationWait = await readStationWait(env.MOMENTARILY);
        stationWaitDoc = updateStationWait(rows, prevStationWait, observedAt);
        await writeStationWait(env.MOMENTARILY, stationWaitDoc);
      } catch (err) {
        console.error('station wait update failed; publishing without platform crowding:', err);
      }
    } else {
      // Fall back to whatever is already stored, unmodified. A single missed
      // minute shouldn't blank the surface outright — published with its own
      // (now older) observed_at, buildSnapshot's freshness gate ages it out
      // over ~30 min while the cap retires individual platforms as their gaps
      // grow. Same posture as stationFlow: degrade honestly, don't vanish.
      console.log('trace produced no rows; station wait carried forward untouched');
      try {
        stationWaitDoc = await readStationWait(env.MOMENTARILY);
      } catch (err) {
        console.error('station wait read failed; publishing without platform crowding:', err);
      }
    }
    step('0-trace');

    if (!isFiveMinuteBoundary) {
      console.log(
        `tick ${observedAt}: minute=${minute}, off the 5-minute boundary — `
        + '5-minute pipeline skipped, trace only',
      );
      return;
    }

    // --- Step 1: read state ---
    // Capture etags so the write-back is a compare-and-swap — overlapping or
    // retried cron runs can't silently clobber each other.
    const [
      lastSeenRead,
      alphaRead,
      trainedParams,
      prevMovementMetric,
      prevServiceMetric,
      ridershipBaseline,
      serviceWeightBaseline,
    ] = await Promise.all([
      readLastSeen(env.MOMENTARILY),
      readAlphaState(env.MOMENTARILY),
      loadParams(env.MOMENTARILY),
      readMovementMetric(env.MOMENTARILY),
      readServiceMetric(env.MOMENTARILY),
      loadRidershipBaseline(env.MOMENTARILY),
      loadServiceWeightBaseline(env.MOMENTARILY),
    ]);
    const lastSeen = lastSeenRead.state;
    const alphaState = alphaRead.state;
    step('1-read-state');

    // --- Step 2: fetch alerts feed ---
    let alertsPayload: unknown = null;
    // On failure, fall back to the last successful alerts fetch so the
    // snapshot reports the feed gap honestly.
    let alertsFeedFresh = lastSeen.alerts_at;
    try {
      alertsPayload = await fetchJson(FEEDS.alerts);
      alertsFeedFresh = observedAt;
      lastSeen.alerts_at = observedAt;
    } catch (err) {
      console.error('alerts fetch failed; feed gap this tick:', err);
    }
    step('2-fetch-alerts');

    // --- Step 3: archive new versions ---
    if (alertsPayload !== null) {
      try {
        const written = await archiveNewAlerts(
          env.MOMENTARILY,
          alertsPayload,
          lastSeen,
          observedAt,
        );
        console.log(`archive: ${written} new alert versions`);
      } catch (err) {
        console.error('archive failed:', err);
      }
    }
    step('3-archive');

    // --- Step 4: derive per-route observations + advance filter ---
    // A params change (retrain or rollback) invalidates the accumulated
    // posteriors in alpha.json — they were filtered under the old emission/
    // transition params, so carrying the raw numbers forward can pin routes
    // to stale regimes. Reseed each roll instead of dropping it: posterior
    // softened onto the old argmax, regime clock and cause preserved (the
    // nightly retrain must not zero every regime's age).
    const paramsVersion = trainedParams?.trained_at ?? 0;
    const paramsChanged = alphaState.params_version !== paramsVersion;
    if (paramsChanged) {
      console.log(
        `params version ${alphaState.params_version} -> ${paramsVersion}; reseeding alpha filter state`,
      );
    }
    const carriedRoutes = paramsChanged
      ? Object.fromEntries(
          Object.entries(alphaState.routes).map(([r, roll]) => [
            r,
            reseedForNewParams(roll),
          ]),
        )
      : alphaState.routes;

    const newAlphaState: AlphaState = {
      params_version: paramsVersion,
      updated_at: observedAt,
      routes: { ...carriedRoutes },
    };

    let routeSnapshots = new Map<string, RouteSnapshot>();
    if (alertsPayload !== null) {
      routeSnapshots = deriveRouteSnapshots(alertsPayload, observedAt);
    }
    step('4a-derive');

    // Routes to run inference for: union of (observed this tick, previously
    // known via alpha, canonical subway list when we have a payload).
    //   - alertsPayload present + route in routeSnapshots → use that observation
    //   - alertsPayload present + route not in routeSnapshots → quiet obs (good service)
    //   - alertsPayload null → obs=null for every route (true feed gap)
    const observedRouteIds = new Set(routeSnapshots.keys());
    const knownRouteIds = new Set(Object.keys(carriedRoutes));
    const allRoutes = new Set<string>([
      ...observedRouteIds,
      ...knownRouteIds,
      ...(alertsPayload !== null ? SUBWAY_ROUTES : []),
    ]);
    const quietObs: Observation | null =
      alertsPayload !== null ? quietObservation(observedAt) : null;
    // The trip counts the binomial movement channel was actually evaluated at
    // this tick, for the grading stream. The posterior saturates (journal
    // 2026-08-23) and this is the only channel whose log-likelihood scales with
    // the tick's trip count, so attributing that saturation needs the count,
    // not just the posterior it produced. Populated only when the channel
    // really fired, per movementChannelActive — the one definition of the gate.
    // Tick-local: never folded into AlphaState, which is persisted.
    const movementCounts = new Map<string, { matched_n: number; advanced_n: number }>();

    for (const routeId of allRoutes) {
      const prevRoll: RouteRoll | undefined = carriedRoutes[routeId];
      const params = paramsForRoute(trainedParams, routeId);

      // Fresh-reset seed: the trained params.initial often collapses to a
      // one-hot vector (training corpus starts in normal). Use the stationary
      // distribution of the transition matrix instead — a single tick of
      // evidence then settles smoothly rather than snapping to one-hot.
      const baseFilter: FilterState = prevRoll?.filter ?? {
        probabilities: stationaryDistribution(params),
        regime_entered_at: observedAt,
        last_updated_at: observedAt,
      };
      const basePublished: PublishedState =
        prevRoll?.published ?? initialPublishedState(baseFilter);

      const routeSnap = routeSnapshots.get(routeId);
      let obs: Observation | null = routeSnap ? routeSnap.observation : quietObs;
      // Fold in the previous tick's cross-tick movement (option B lag): an
      // independent "are trains moving" channel the alerts feed can't see. Off
      // (logEmission drops the channel) when there's no usable signal.
      if (obs) {
        const mv = movementObservationFields(
          prevMovementMetric,
          trainedParams,
          routeId,
          observedAt,
        );
        if (mv) obs = { ...obs, ...mv };
        // Fold in the previous tick's service level (assigned_n / baseline) the
        // same way — an orthogonal "are trains dispatched" channel.
        const sv = serviceObservationFields(
          prevServiceMetric,
          trainedParams,
          routeId,
          observedAt,
        );
        if (sv) obs = { ...obs, ...sv };
      }
      if (obs && movementChannelActive(obs, emissionsFor(params, obs))) {
        movementCounts.set(routeId, {
          matched_n: obs.matched_n ?? 0,
          advanced_n: obs.advanced_n ?? 0,
        });
      }
      const result = forwardStep(baseFilter, basePublished, obs, params, observedAt);

      // Carry alert_type_at_entry forward while the regime persists; refresh it
      // when the regime just advanced (or on fresh reset).
      const regimeAdvanced =
        result.state.regime_entered_at > baseFilter.regime_entered_at;
      const alertTypeAtEntry =
        !prevRoll || regimeAdvanced
          ? (routeSnap?.primary_alert_type ?? null)
          : (prevRoll.alert_type_at_entry ?? null);

      newAlphaState.routes[routeId] = {
        filter: result.state,
        published: result.published,
        alert_type_at_entry: alertTypeAtEntry,
      };
    }
    step(`4b-forward(${allRoutes.size}r)`);

    // --- Step 5: persist new alpha state (CAS) ---
    // Write before publishing so a concurrent tick that loses the etag race
    // doesn't ship snapshot.json / predictions / transitions derived from
    // state that never landed in R2.
    let alphaWritten = false;
    try {
      alphaWritten = await writeAlphaState(
        env.MOMENTARILY,
        newAlphaState,
        alphaRead.etag,
      );
      if (!alphaWritten) {
        console.warn(
          'alpha.json write conflict; skipping snapshot/predictions/transitions this tick',
        );
      }
    } catch (err) {
      console.error('alpha write failed; skipping outputs this tick:', err);
    }
    step('5-alpha-write');

    if (alphaWritten) {
      // --- Step 6: render + publish snapshot ---
      // Station metadata lives in its own R2 object (refreshed daily), read here
      // and embedded. A read failure degrades to an empty stations surface, never
      // a failed tick.
      let stationsCache: Awaited<ReturnType<typeof readStationsCache>> = null;
      try {
        stationsCache = await readStationsCache(env.MOMENTARILY);
      } catch (err) {
        console.error('stations cache read failed; publishing without stations:', err);
      }
      // Last tick's movement-derived states drive the published current-state
      // condition (lagged ~5 min — written at step 8b, after this publishes).
      // A read failure degrades to alert/HMM conditions, never a failed tick.
      let movementStates: Awaited<ReturnType<typeof readMovementState>> = null;
      try {
        movementStates = await readMovementState(env.MOMENTARILY);
      } catch (err) {
        console.error('movement_state read failed; publishing without it:', err);
      }
      // Last tick's per-station service flow, same one-tick lag.
      let stationFlow: StationFlowDoc | null = null;
      try {
        stationFlow = await readStationFlow(env.MOMENTARILY);
      } catch (err) {
        console.error('station_flow read failed; publishing without it:', err);
      }
      // Last tick's per-segment regimes + the (mostly static) segment
      // topology, same one-tick lag as stationFlow — the segment_flow
      // surface and the recovery attached to station_flow's worst_segment
      // are both built from these. A read failure degrades to publishing
      // without segment_flow, never a failed tick.
      let segmentFlow: SegmentFlowDoc | null = null;
      let segmentParams: SegmentParamsDoc | null = null;
      try {
        [segmentFlow, segmentParams] = await Promise.all([
          readSegmentFlow(env.MOMENTARILY),
          readSegmentParams(env.MOMENTARILY),
        ]);
      } catch (err) {
        console.error('segment_flow/segment_params read failed; publishing without them:', err);
      }
      // Per-segment dwell curves the segment recovery is conditioned on.
      // Trainer-published, not tick-lagged; null until it exists in R2 —
      // segments then publish status without a fabricated recovery.
      let segmentDwell: SegmentDwellDoc | null = null;
      try {
        segmentDwell = await readSegmentDwell(env.MOMENTARILY);
      } catch (err) {
        console.error('segment_dwell read failed; publishing without segment recovery:', err);
      }
      const snapshot = buildSnapshot({
        generatedAt: observedAt,
        alertsFreshness: alertsFeedFresh,
        routeSnapshots,
        rolls: newAlphaState.routes,
        trainedParams,
        tickSeconds: TICK_SECONDS,
        stationStatuses: lastSeen.station_statuses,
        eneFreshness: lastSeen.ene_at > 0 ? lastSeen.ene_at : null,
        alerts:
          alertsPayload !== null ? buildAlertList(alertsPayload, observedAt) : [],
        equipment: lastSeen.equipment,
        stations: stationsCache?.stations ?? {},
        stationsStaticFreshness: stationsCache?.fetched_at ?? null,
        movementStates,
        stationFlow,
        segmentFlow,
        segmentParams,
        segmentDwell,
        stationWait: stationWaitDoc,
        ridershipBaseline,
        serviceWeightBaseline,
      });
      step('6a-build-snapshot');
      try {
        await publishSnapshot(env.MOMENTARILY, snapshot);
        console.log(
          `snapshot: ${Object.keys(snapshot.route_status).length} routes published`,
        );
      } catch (err) {
        console.error('snapshot publish failed:', err);
      }
      step('6b-publish-snapshot');

      // Aggregated live train positions for the /map overlay, published as
      // its own object (v1/trains.json) rather than embedded in the
      // snapshot — see snapshot.ts's file header for the size/consumer
      // rationale. Built from this tick's already-decoded vehicle-position
      // fetch (step 0 above, `vehicles`/`vehicleFreshFeeds`), so it costs no
      // extra request, and published on the same tick as snapshot.json,
      // right after it. Fully independent of the snapshot publish above: a
      // failure on either side never blocks or fails the other, and never
      // fails the tick.
      //
      // vehicleFreshFeeds can be a STRICT SUBSET of TRIP_UPDATE_FEED_NAMES —
      // Promise.allSettled above logs and SKIPS a rejected feed rather than
      // throwing, so a partial vehicle set never reaches this try/catch as
      // an exception. Two cases, not one:
      //   - NO feed decoded (vehicleFreshFeeds empty): skip the publish
      //     entirely. Calling buildTrains/publishTrains here would write
      //     {positions: []} — indistinguishable from "zero trains in NYC",
      //     exactly the fabrication this surface exists to avoid. The
      //     object is left un-rewritten; a consumer sees the last-good read
      //     with its own observed_at, never a fabricated empty one.
      //   - SOME feeds decoded: publish normally, flagged partial via
      //     fresh_feeds/expected_feeds (see PublishedTrains's doc comment)
      //     rather than withheld — a partial map is still useful, as long
      //     as it says so.
      if (vehicleFreshFeeds.length > 0) {
        try {
          await publishTrains(
            env.MOMENTARILY,
            buildTrains(
              observedAt,
              trainPositions(vehicles),
              vehicleFreshFeeds,
              TRIP_UPDATE_FEED_NAMES,
            ),
          );
        } catch (err) {
          console.error(
            'trains publish failed; leaving v1/trains.json unrewritten this tick:',
            err,
          );
        }
      } else {
        console.warn(
          'trains: no vehicle feed decoded this tick, leaving v1/trains.json unrewritten',
        );
      }
      step('6c-publish-trains');

      // --- Step 7: grading streams ---
      const predictions: PredictionRecord[] = [];
      for (const [routeId, rs] of Object.entries(snapshot.route_status)) {
        const inf = rs.inference;
        if (!inf) continue;
        const mv = movementCounts.get(routeId);
        predictions.push({
          ts: observedAt,
          route: routeId,
          condition: inf.condition,
          regime_entered_at: inf.regime_entered_at,
          p_normal: inf.p_normal,
          p_disrupted: inf.p_disrupted,
          p_suspended: inf.p_suspended,
          p_normal_in_30min: inf.p_normal_in_30min,
          p_normal_in_60min: inf.p_normal_in_60min,
          p_normal_in_120min: inf.p_normal_in_120min,
          recovery_minutes: inf.recovery_minutes,
          recovery_minutes_low: inf.recovery_minutes_low,
          recovery_minutes_high: inf.recovery_minutes_high,
          recovery_indeterminate: inf.recovery_indeterminate,
          recovery_source: inf.recovery_source,
          resumes_at: inf.resumes_at,
          primary_alert_type: rs.primary_alert_type,
          params_version: paramsVersion,
          published_condition: rs.condition,
          condition_source: rs.condition_source,
          // From the same one-tick-lagged doc the snapshot published the
          // condition from, so the row describes the regime consumers saw.
          movement_regime_entered_at: movementStates?.regimes[routeId]?.entered_at ?? 0,
          // Null when the movement channel did not fire this tick — there is no
          // count to attribute, and a number here would imply the binomial
          // contributed when it contributed 0.
          matched_n: mv?.matched_n ?? null,
          advanced_n: mv?.advanced_n ?? null,
        });
      }
      try {
        await writePredictions(env.MOMENTARILY, observedAt, predictions);
      } catch (err) {
        console.error('predictions write failed:', err);
      }

      const transitions = detectTransitions(
        carriedRoutes,
        newAlphaState.routes,
        observedAt,
      );
      if (transitions.length > 0) {
        try {
          await writeTransitions(env.MOMENTARILY, observedAt, transitions);
          console.log(`transitions: ${transitions.length} regime flips this tick`);
        } catch (err) {
          console.error('transitions write failed:', err);
        }
      }
      step('7-grading-writes');
    }

    // --- Step 8: E&E (hourly) ---
    // Only the alpha CAS winner gets to mutate lastSeen, so losing runs skip
    // the E&E fetch too.
    if (alphaWritten && observedAt - lastSeen.ene_at >= ENE_INTERVAL_SECONDS) {
      let eneOk = 0;
      const enePayloads: Record<string, unknown> = {};
      for (const [name, url] of ENE_SOURCES) {
        try {
          const payload = await fetchJson(url);
          enePayloads[name] = payload;
          await archiveEneSnapshot(env.MOMENTARILY, name, payload, observedAt);
          eneOk += 1;
        } catch (err) {
          console.error(`ene ${name} failed:`, err);
        }
      }
      // ene_at only advances when both station_status inputs landed — the
      // published freshness has to describe the data actually being served.
      // An incomplete fetch leaves it alone, so the next tick retries.
      const catalogPayload = enePayloads.ene_equipments;
      const outagesPayload = enePayloads.ene_current;
      if (catalogPayload !== undefined && outagesPayload !== undefined) {
        const catalog = parseEquipmentFeed(catalogPayload);
        const outages = parseOutageFeed(outagesPayload);
        const statuses = deriveStationStatuses(catalog, outages, observedAt);
        lastSeen.station_statuses = Object.fromEntries(statuses);
        lastSeen.equipment = buildEquipmentList(catalog, outages, observedAt);
        lastSeen.ene_at = observedAt;
        console.log(
          `ene: ${eneOk}/${ENE_SOURCES.length} feeds archived, `
          + `${statuses.size} station_status entries derived`,
        );
      } else {
        console.log(
          `ene: ${eneOk}/${ENE_SOURCES.length} feeds archived; `
          + 'station_status inputs incomplete, freshness held — retrying next tick',
        );
      }
    }

    // --- Step 8b: trip-updates + vehicle metrics (every 5-minute tick) ---
    // The protobuf feeds were already fetched and their VehiclePosition side
    // already decoded at step 0 above (shared with the per-minute trace, so a
    // boundary tick never fetches twice) — decode the TripUpdate side from
    // those same buffers (`feedResults`) and reuse `vehicles`/
    // `vehicleFreshFeeds` as-is. Derive each compact per-route metric and
    // archive both for offline validation. Gated on the alpha CAS winner like
    // E&E so losing runs don't double-write. A failed/slow feed is non-fatal —
    // its routes are simply absent this tick, recorded via fresh_feeds.
    if (alphaWritten) {
      try {
        const trips: TripLite[] = [];
        for (const r of feedResults) {
          if (r.status === 'fulfilled') trips.push(...decodeTripUpdates(r.value));
          // Fetch failures for this tick were already logged at step 0.
        }
        const freshFeeds = vehicleFreshFeeds;
        if (freshFeeds.length > 0) {
          const rows = deriveRouteServiceMetric(trips);
          await archiveTripUpdateMetric(
            env.MOMENTARILY,
            rows,
            freshFeeds,
            observedAt,
          );
          lastSeen.trip_updates_at = observedAt;

          // Cross-tick movement: diff this tick's stop_ids against the carry map
          // written last tick, then overwrite it. The map is read/written as its
          // own R2 object, kept out of last_seen.json on purpose.
          const prevStops = await readVehicleStops(env.MOMENTARILY);
          // Absent through-stop set (no static topology yet, or a params.json
          // predating the field) is a real, visible degradation, not a silent
          // fall-soft — the advance counters keep counting terminal layovers.
          const throughStops = trainedParams?.throughStops ?? null;
          if (trainedParams && !throughStops) {
            console.log(
              'movement: no through-stop set in params.json; advance counters include terminal layovers',
            );
          }
          const moveRows = deriveRouteMovementMetric(vehicles, prevStops, throughStops);
          await archiveVehicleMetric(
            env.MOMENTARILY,
            moveRows,
            freshFeeds,
            observedAt,
          );
          await writeVehicleStops(env.MOMENTARILY, stopPositions(vehicles));
          lastSeen.vehicles_at = observedAt;
          // Carry these counts one tick forward: next tick's derive step folds
          // them into each route's Observation as the movement emission channel.
          await writeMovementMetric(env.MOMENTARILY, observedAt, moveRows);
          // Carry assigned_n forward too, for the service emission channel.
          await writeServiceMetric(env.MOMENTARILY, observedAt, rows);

          // Movement-derived current state, read by next tick's snapshot build as
          // the published movement-primary condition (debiased per-direction
          // classifier; suspended/not_scheduled from the schedule rate).
          //
          // The raw per-tick call goes through the regime clock before it is
          // published: a change commits only after DEBOUNCE_TICKS agreeing ticks
          // and back-dates to the first of them, so `entered_at` is when the
          // evidence started rather than when it convinced us. That clock is
          // what the movement dwell curves are conditioned on, and the committed
          // changes are the stream those curves are fitted from.
          if (MOVEMENT_STATE_PUBLISH) {
            const prevMovement = await readMovementState(env.MOMENTARILY);
            // Supply-axis baseline: prefer the standalone sidecar object, fall
            // back to params.json's legacy field. Sourcing it here keeps the axis
            // decoupled from the frozen HMM artifact.
            const serviceBaselineDoc = await readServiceBaseline(env.MOMENTARILY);
            const serviceBaseline =
              serviceBaselineDoc?.baseline ?? trainedParams?.serviceBaselineHourly ?? null;
            // Per-cell p10/p90 spread. Sidecar-only — no params.json fallback,
            // since the legacy field never carried a spread.
            const serviceQuantiles = serviceBaselineDoc?.quantiles ?? null;
            const { entries, changes } = advanceRegimes(
              prevMovement?.regimes,
              deriveMovementStates(moveRows, rows, trainedParams, observedAt),
              observedAt,
            );
            // Service-level regime, the SUPPLY axis (assigned_n vs its hourly
            // baseline), advanced on its own clock beside the movement regimes.
            // The degrade/recover band and the 2-tick debounce match the offline
            // label (derive_actual_recovery). No transition stream — no service
            // dwell curves; it exists only to publish the current service_condition.
            //
            // Expire stale regimes FIRST: the hysteresis band and the cold-start
            // seed both read the prior state, so a regime that went blind past
            // MAX_IDLE_SEC must be gone before they run, or a returning route
            // would inherit its pre-gap degraded state. This idle reset is an
            // INTENTIONAL divergence from the offline label, which has no idle
            // expiry and holds degraded across a gap — MAX_IDLE_SEC freshness (the
            // same Worker semantic the movement axis uses) takes precedence for a
            // live feed, where a regime resuming after an hour blind is not
            // knowably the one that stopped.
            const liveServiceRegimes = pruneIdleRegimes(
              prevMovement?.service_regimes,
              observedAt,
            );
            const observedService = deriveServiceStates(
              rows,
              serviceBaseline,
              observedAt,
              liveServiceRegimes,
            );
            const service = advanceRegimes(
              seedNormalServiceRegimes(liveServiceRegimes, observedService, observedAt),
              observedService,
              observedAt,
              { debounceTicks: SERVICE_DEBOUNCE_TICKS },
            );
            await writeMovementState(env.MOMENTARILY, {
              observed_at: observedAt,
              regimes: entries,
              service_regimes: service.entries,
              service_ratios: deriveServiceRatios(rows, serviceBaseline, observedAt),
              service_quantile_ratios: deriveServiceQuantileRatios(
                rows,
                serviceBaseline,
                serviceQuantiles,
                observedAt,
              ),
            });
            try {
              await writeMovementTransitions(
                env.MOMENTARILY,
                observedAt,
                movementTransitions(changes, 'route', observedAt),
              );
              if (changes.length > 0) {
                console.log(`movement: ${changes.length} regime changes this tick`);
              }
            } catch (err) {
              console.error('movement transitions write failed:', err);
            }
          }

          // Segment-level station service flow: decay-smoothed per-segment
          // advance -> classify -> roll up to stations, PLUS the segment
          // regime clock (same debounce as the route clock, keyed on the
          // segment cell) that the per-segment dwell curves condition on.
          // Its own R2 objects, read off the segment baseline (own object
          // too), so the ~1.8k-cell baseline never touches the hot per-tick
          // params parse. Read next tick by the snapshot build (one-tick
          // lag, like movement_state). Fail-soft.
          try {
            const segParams = await readSegmentParams(env.MOMENTARILY);
            if (segParams) {
              const prevFlow = await readSegmentFlow(env.MOMENTARILY);
              const flow = updateSegmentFlow(prevFlow, moveRows, observedAt, segParams);
              const { entries, changes } = advanceRegimes(
                prevFlow?.regimes,
                deriveSegmentStates(flow, segParams),
                observedAt,
              );
              // Regimes are pruned against the trainer's baselined cell set,
              // not the accumulator: every baselined cell is judged every tick
              // now, so the only stale entry left is one whose cell a retrain
              // dropped. A cell that merely abstains this tick keeps
              // advanceRegimes' idle grace.
              flow.regimes = pruneSegmentRegimes(entries, segParams.cells);
              await writeSegmentFlow(env.MOMENTARILY, flow);
              await writeStationFlow(env.MOMENTARILY, deriveStationFlow(flow, segParams));
              try {
                await writeMovementTransitions(
                  env.MOMENTARILY,
                  observedAt,
                  movementTransitions(changes, 'segment', observedAt),
                );
              } catch (err) {
                console.error('segment movement transitions write failed:', err);
              }
            }
          } catch (err) {
            console.error('station flow update failed; skipping:', err);
          }

          console.log(
            `trip-updates: ${freshFeeds.length}/${TRIP_UPDATE_FEEDS.length} feeds, `
            + `${trips.length} trips, ${rows.size} routes; `
            + `vehicles: ${vehicles.length}, ${moveRows.size} routes`,
          );
        }
      } catch (err) {
        console.error('trip-updates step failed:', err);
      }
      step('8b-trip-updates');
    }

    // --- Step 8c: stations static (daily) ---
    // Writes the parsed metadata to its own R2 object; stations_at advances only
    // on a successful, non-empty fetch so a transient failure retries next tick.
    if (alphaWritten && observedAt - lastSeen.stations_at >= STATIONS_INTERVAL_SECONDS) {
      try {
        const stations = parseStationsFeed(await fetchJson(STATIONS_FEED));
        if (stations.length > 0) {
          await writeStationsCache(env.MOMENTARILY, stations, observedAt);
          lastSeen.stations_at = observedAt;
          console.log(`stations: ${stations.length} static records cached`);
        } else {
          console.warn('stations: feed parsed to zero records, freshness held');
        }
      } catch (err) {
        console.error('stations fetch failed; freshness held:', err);
      }
      step('8c-stations');
    }

    // Only the alpha CAS winner commits last_seen — a losing run's outputs
    // were all discarded above, so it must not race the winner's state here.
    if (alphaWritten) {
      try {
        const written = await writeLastSeen(
          env.MOMENTARILY,
          lastSeen,
          lastSeenRead.etag,
        );
        if (!written) {
          console.warn('last_seen.json write conflict; a concurrent run won this tick');
        }
      } catch (err) {
        console.error('last_seen write failed:', err);
      }
    }
    step('9-last-seen-write');
  },
} satisfies ExportedHandler<Env>;

