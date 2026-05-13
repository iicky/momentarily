/**
 * Momentarily publisher — Cloudflare Worker entry point.
 *
 * v0 scope: COLLECTION ONLY. Each cron tick:
 *   1. Read state/last_seen.json from R2
 *   2. Fetch the MTA alerts feed; for each new (alert_id, updated_at) pair,
 *      write a per-change object to archive/alerts/...
 *   3. Hourly: fetch the 3 E&E feeds, write a snapshot per source to
 *      archive/ene/...
 *   4. Write updated state/last_seen.json
 *
 * Forward filter + snapshot publishing are intentionally not here yet — they
 * land in follow-up iterations once enough corpus has accumulated to train
 * against.
 */

import { archiveEneSnapshot, archiveNewAlerts } from './archive';
import { FEEDS, fetchJson } from './fetch';
import { readLastSeen, writeLastSeen } from './state';

export interface Env {
  MOMENTARILY: R2Bucket;
}

// Hourly E&E cadence: the upstream feed itself doesn't change faster than that.
const ENE_INTERVAL_SECONDS = 3600;

const ENE_SOURCES = [
  ['ene_current', FEEDS.ene_current],
  ['ene_upcoming', FEEDS.ene_upcoming],
  ['ene_equipments', FEEDS.ene_equipments],
] as const;

export default {
  async fetch(_request: Request, _env: Env): Promise<Response> {
    return new Response(
      'Momentarily publisher Worker. Cron-driven; no HTTP surface yet.\n',
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

    const lastSeen = await readLastSeen(env.MOMENTARILY);

    try {
      const payload = await fetchJson(FEEDS.alerts);
      const written = await archiveNewAlerts(
        env.MOMENTARILY,
        payload,
        lastSeen,
        observedAt,
      );
      console.log(`alerts: ${written} new versions archived`);
    } catch (err) {
      console.error('alerts pipeline failed:', err);
    }

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
      if (eneOk > 0) {
        // Only advance freshness if at least one E&E feed succeeded; otherwise
        // we want to retry on the next tick rather than wait another hour.
        lastSeen.ene_at = observedAt;
      }
      console.log(`ene: ${eneOk}/${ENE_SOURCES.length} feeds archived`);
    }

    await writeLastSeen(env.MOMENTARILY, lastSeen);
  },
} satisfies ExportedHandler<Env>;
