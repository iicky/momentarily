/**
 * Momentarily publisher — Cloudflare Worker entry point.
 *
 * Each cron tick:
 *   1. Fetch the MTA alerts feed; archive new (alert_id, updated_at) versions
 *   2. Derive per-route observations from currently-active alerts
 *   3. Read trained HMM params (from R2; bootstrap fallback) and rolling alpha
 *   4. Advance the forward filter per route, with hysteresis + Unknown
 *   5. Render snapshot.json and publish to R2 (public via feed.momentarily.nyc)
 *   6. Write alpha.json + last_seen.json back to R2
 *
 * Hourly: fetch the 3 E&E feeds and archive snapshots. Station status
 * derivation for the snapshot lands in a follow-up iteration.
 */

import type { AlphaState, RouteRoll } from './alpha';
import { readAlphaState, writeAlphaState } from './alpha';
import { archiveEneSnapshot, archiveNewAlerts } from './archive';
import type { RouteSnapshot } from './derive';
import { SUBWAY_ROUTES, deriveRouteSnapshots, quietObservation } from './derive';
import { FEEDS, fetchJson } from './fetch';
import type { PredictionRecord } from './grading';
import { detectTransitions, writePredictions, writeTransitions } from './grading';
import type { FilterState, Observation, PublishedState } from './hmm';
import { forwardStep, initialPublishedState } from './hmm';
import { loadParams, paramsForRoute } from './params';
import { TICK_SECONDS, buildSnapshot, publishSnapshot } from './snapshot';
import { readLastSeen, writeLastSeen } from './state';

export interface Env {
  MOMENTARILY: R2Bucket;
}

const ENE_INTERVAL_SECONDS = 3600;

const ENE_SOURCES = [
  ['ene_current', FEEDS.ene_current],
  ['ene_upcoming', FEEDS.ene_upcoming],
  ['ene_equipments', FEEDS.ene_equipments],
] as const;

export default {
  async fetch(_request: Request, _env: Env): Promise<Response> {
    return new Response(
      'Momentarily publisher Worker. Cron-driven. Snapshot at https://feed.momentarily.nyc/v1/snapshot.json\n',
      { headers: { 'content-type': 'text/plain; charset=utf-8' } },
    );
  },

  async scheduled(
    event: ScheduledController,
    env: Env,
    _ctx: ExecutionContext,
  ): Promise<void> {
    const observedAt = Math.floor(Date.now() / 1000);
    console.log(`tick cron=${event.cron} t=${observedAt}`);

    // --- Step 1: read state ---
    const [lastSeen, alphaState, trainedParams] = await Promise.all([
      readLastSeen(env.MOMENTARILY),
      readAlphaState(env.MOMENTARILY),
      loadParams(env.MOMENTARILY),
    ]);

    // --- Step 2: fetch alerts feed ---
    let alertsPayload: unknown = null;
    let alertsFeedFresh = lastSeen.ene_at; // placeholder before we set it
    try {
      alertsPayload = await fetchJson(FEEDS.alerts);
      alertsFeedFresh = observedAt;
    } catch (err) {
      console.error('alerts fetch failed; feed gap this tick:', err);
    }

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

    // --- Step 4: derive per-route observations + advance filter ---
    const newAlphaState: AlphaState = {
      params_version: trainedParams?.trained_at ?? 0,
      updated_at: observedAt,
      routes: { ...alphaState.routes },
    };

    let routeSnapshots = new Map<string, RouteSnapshot>();
    if (alertsPayload !== null) {
      routeSnapshots = deriveRouteSnapshots(alertsPayload, observedAt);
    }

    // Routes to run inference for: union of (observed this tick, previously
    // known via alpha, canonical subway list when we have a payload).
    //   - alertsPayload present + route in routeSnapshots → use that observation
    //   - alertsPayload present + route not in routeSnapshots → quiet obs (good service)
    //   - alertsPayload null → obs=null for every route (true feed gap)
    const observedRouteIds = new Set(routeSnapshots.keys());
    const knownRouteIds = new Set(Object.keys(alphaState.routes));
    const allRoutes = new Set<string>([
      ...observedRouteIds,
      ...knownRouteIds,
      ...(alertsPayload !== null ? SUBWAY_ROUTES : []),
    ]);
    const quietObs: Observation | null =
      alertsPayload !== null ? quietObservation(observedAt) : null;

    for (const routeId of allRoutes) {
      const prevRoll: RouteRoll | undefined = alphaState.routes[routeId];
      const params = paramsForRoute(trainedParams, routeId);

      const baseFilter: FilterState = prevRoll?.filter ?? {
        probabilities: params.initial,
        regime_entered_at: observedAt,
        last_updated_at: observedAt,
      };
      const basePublished: PublishedState =
        prevRoll?.published ?? initialPublishedState(baseFilter);

      const routeSnap = routeSnapshots.get(routeId);
      const obs: Observation | null = routeSnap ? routeSnap.observation : quietObs;
      const result = forwardStep(baseFilter, basePublished, obs, params, observedAt);

      newAlphaState.routes[routeId] = {
        filter: result.state,
        published: result.published,
      };
    }

    // --- Step 5: render + publish snapshot ---
    const snapshot = buildSnapshot({
      generatedAt: observedAt,
      alertsFreshness: alertsFeedFresh,
      routeSnapshots,
      rolls: newAlphaState.routes,
      trainedParams,
      tickSeconds: TICK_SECONDS,
    });
    try {
      await publishSnapshot(env.MOMENTARILY, snapshot);
      console.log(
        `snapshot: ${Object.keys(snapshot.route_status).length} routes published`,
      );
    } catch (err) {
      console.error('snapshot publish failed:', err);
    }

    // --- Step 6: persist state + grading streams ---
    try {
      await writeAlphaState(env.MOMENTARILY, newAlphaState);
    } catch (err) {
      console.error('alpha write failed:', err);
    }

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
      });
    }
    try {
      await writePredictions(env.MOMENTARILY, observedAt, predictions);
    } catch (err) {
      console.error('predictions write failed:', err);
    }

    const transitions = detectTransitions(
      alphaState.routes,
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

    // --- Step 7: E&E (hourly) ---
    if (observedAt - lastSeen.ene_at >= ENE_INTERVAL_SECONDS) {
      let eneOk = 0;
      for (const [name, url] of ENE_SOURCES) {
        try {
          const payload = await fetchJson(url);
          await archiveEneSnapshot(env.MOMENTARILY, name, payload, observedAt);
          eneOk += 1;
        } catch (err) {
          console.error(`ene ${name} failed:`, err);
        }
      }
      if (eneOk > 0) lastSeen.ene_at = observedAt;
      console.log(`ene: ${eneOk}/${ENE_SOURCES.length} feeds archived`);
    }

    await writeLastSeen(env.MOMENTARILY, lastSeen);
  },
} satisfies ExportedHandler<Env>;

