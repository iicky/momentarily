/**
 * Momentarily publisher — Cloudflare Worker entry point.
 *
 * Each cron tick:
 *   1. Read rolling state (last_seen, alpha) + trained params from R2
 *   2. Fetch the MTA alerts feed
 *   3. Archive new (alert_id, updated_at) versions
 *   4. Derive per-route observations + advance the forward filter
 *   5. Persist alpha.json via etag CAS — a losing run yields the tick here
 *   6. (Only if alpha persisted) Render + publish snapshot.json to R2
 *   7. (Only if alpha persisted) Write predictions + transitions grading streams
 *   8. Hourly: fetch the 3 E&E feeds and archive snapshots
 *   9. Persist last_seen.json via etag CAS
 */

import type { AlphaState, RouteRoll } from './alpha';
import { readAlphaState, reseedForNewParams, writeAlphaState } from './alpha';
import {
  archiveEneSnapshot,
  archiveNewAlerts,
  archiveTripUpdateMetric,
  archiveVehicleMetric,
} from './archive';
import type { RouteSnapshot } from './derive';
import { SUBWAY_ROUTES, buildAlertList, deriveRouteSnapshots, quietObservation } from './derive';
import { parseEquipmentFeed, parseOutageFeed } from './ene';
import { FEEDS, STATIONS_FEED, TRIP_UPDATE_FEEDS, fetchJson, fetchProtobuf } from './fetch';
import type { TripLite, VehicleLite } from './gtfsrt';
import { decodeTripUpdates, decodeVehicles } from './gtfsrt';
import { deriveRouteServiceMetric } from './trip_updates';
import { deriveRouteMovementMetric, stopPositions } from './vehicles';
import {
  MOVEMENT_STATE_PUBLISH,
  deriveMovementStates,
  movementObservationFields,
  serviceObservationFields,
} from './movement_state';
import type { PredictionRecord } from './grading';
import { detectTransitions, writePredictions, writeTransitions } from './grading';
import type { FilterState, Observation, PublishedState } from './hmm';
import { forwardStep, initialPublishedState, stationaryDistribution } from './hmm';
import { loadParams, paramsForRoute } from './params';
import { TICK_SECONDS, buildSnapshot, publishSnapshot } from './snapshot';
import { buildEquipmentList, deriveStationStatuses } from './stations';
import { parseStationsFeed, readStationsCache, writeStationsCache } from './stations_static';
import {
  readLastSeen,
  readMovementMetric,
  readMovementState,
  readServiceMetric,
  readVehicleStops,
  writeLastSeen,
  writeMovementMetric,
  writeMovementState,
  writeServiceMetric,
  writeVehicleStops,
} from './state';

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

    // --- Step 1: read state ---
    // Capture etags so the write-back is a compare-and-swap — overlapping or
    // retried cron runs can't silently clobber each other. See momentarily-j0c.
    const [
      lastSeenRead,
      alphaRead,
      trainedParams,
      prevMovementMetric,
      prevServiceMetric,
    ] = await Promise.all([
      readLastSeen(env.MOMENTARILY),
      readAlphaState(env.MOMENTARILY),
      loadParams(env.MOMENTARILY),
      readMovementMetric(env.MOMENTARILY),
      readServiceMetric(env.MOMENTARILY),
    ]);
    const lastSeen = lastSeenRead.state;
    const alphaState = alphaRead.state;
    step('1-read-state');

    // --- Step 2: fetch alerts feed ---
    let alertsPayload: unknown = null;
    // On failure, fall back to the last successful alerts fetch so the
    // snapshot reports the feed gap honestly. See momentarily-g24.
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

    for (const routeId of allRoutes) {
      const prevRoll: RouteRoll | undefined = carriedRoutes[routeId];
      const params = paramsForRoute(trainedParams, routeId);

      // Fresh-reset seed: the trained params.initial often collapses to a
      // one-hot vector (training corpus starts in normal). Use the stationary
      // distribution of the transition matrix instead — a single tick of
      // evidence then settles smoothly rather than snapping to one-hot. See
      // momentarily-d78.
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
      const result = forwardStep(baseFilter, basePublished, obs, params, observedAt);

      // Carry alert_type_at_entry forward while the regime persists; refresh it
      // when the regime just advanced (or on fresh reset). See momentarily-22k.
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
    // state that never landed in R2. See momentarily-uc4.
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

      // --- Step 7: grading streams ---
      const predictions: PredictionRecord[] = [];
      for (const [routeId, rs] of Object.entries(snapshot.route_status)) {
        const inf = rs.inference;
        if (!inf) continue;
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

    // --- Step 8b: trip-updates + vehicle metrics (every tick) ---
    // Fetch all line-group protobuf feeds concurrently and decode both the
    // TripUpdate and VehiclePosition entities — same bytes carry both, so the
    // vehicle decode is nearly free. Derive each compact per-route metric and
    // archive both for offline validation. Gated on the alpha CAS winner like
    // E&E so losing runs don't double-write. A failed/slow feed is non-fatal —
    // its routes are simply absent this tick, recorded via fresh_feeds.
    if (alphaWritten) {
      try {
        const results = await Promise.allSettled(
          TRIP_UPDATE_FEEDS.map(([, url]) => fetchProtobuf(url)),
        );
        const trips: TripLite[] = [];
        const vehicles: VehicleLite[] = [];
        const freshFeeds: string[] = [];
        for (let i = 0; i < results.length; i++) {
          const r = results[i]!;
          const name = TRIP_UPDATE_FEEDS[i]![0];
          if (r.status === 'fulfilled') {
            freshFeeds.push(name);
            trips.push(...decodeTripUpdates(r.value));
            vehicles.push(...decodeVehicles(r.value));
          } else {
            console.error(`trip-updates ${name} failed:`, r.reason);
          }
        }
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
          const moveRows = deriveRouteMovementMetric(vehicles, prevStops);
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

          // Movement-derived current state, read by next tick's snapshot build.
          // Gated off: the fixed-threshold derivation is biased per-route and not
          // published; the Bayesian model (momentarily-vhh) replaces it. The
          // direction-split archive above still accrues for baseline training.
          if (MOVEMENT_STATE_PUBLISH) {
            await writeMovementState(env.MOMENTARILY, {
              observed_at: observedAt,
              states: deriveMovementStates(moveRows, rows, trainedParams, observedAt),
            });
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

