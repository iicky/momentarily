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

// Only this prefix is served publicly. Everything else in the bucket (state/,
// archive/) stays private — the Worker is the auth boundary, the R2 custom
// domain must NOT be bound directly to the bucket.
const PUBLIC_PREFIX = 'v1/';

const ENE_INTERVAL_SECONDS = 3600;

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
    const [lastSeenRead, alphaRead, trainedParams] = await Promise.all([
      readLastSeen(env.MOMENTARILY),
      readAlphaState(env.MOMENTARILY),
      loadParams(env.MOMENTARILY),
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
    // transition params, so carrying them forward pins routes to stale
    // regimes. On a version mismatch, drop the filter state and re-seed from
    // params.initial. See momentarily-x5b.
    const paramsVersion = trainedParams?.trained_at ?? 0;
    const paramsChanged = alphaState.params_version !== paramsVersion;
    if (paramsChanged) {
      console.log(
        `params version ${alphaState.params_version} -> ${paramsVersion}; resetting alpha filter state`,
      );
    }
    const carriedRoutes = paramsChanged ? {} : alphaState.routes;

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
      const snapshot = buildSnapshot({
        generatedAt: observedAt,
        alertsFreshness: alertsFeedFresh,
        routeSnapshots,
        rolls: newAlphaState.routes,
        trainedParams,
        tickSeconds: TICK_SECONDS,
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
    step('9-last-seen-write');
  },
} satisfies ExportedHandler<Env>;

