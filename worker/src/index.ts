/**
 * Momentarily publisher — Cloudflare Worker entry point.
 *
 * Fires on a Workers Cron Trigger every 5 minutes:
 *   1. Fetch MTA GTFS-RT feeds
 *   2. Read rolling HMM state from R2
 *   3. Run forward filter for each route
 *   4. Render snapshot.json + write to R2 (public via feed.momentarily.nyc)
 *   5. Write updated state.json + archive line to R2
 *
 * Step (1) only is wired up today — the rest land as separate commits as the
 * pieces port over from src/momentarily/*.py.
 */

export interface Env {
  MOMENTARILY: R2Bucket;
}

const MTA_SUBWAY_ALERTS =
  'https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/camsys%2Fsubway-alerts.json';

export default {
  async fetch(_request: Request, _env: Env): Promise<Response> {
    return new Response(
      'Momentarily publisher Worker. Cron-driven; no HTTP surface.\n',
      { headers: { 'content-type': 'text/plain; charset=utf-8' } },
    );
  },

  async scheduled(
    event: ScheduledController,
    env: Env,
    ctx: ExecutionContext,
  ): Promise<void> {
    const startedAt = Math.floor(Date.now() / 1000);
    console.log(`scheduled tick: cron=${event.cron} t=${startedAt}`);

    try {
      const response = await fetch(MTA_SUBWAY_ALERTS, {
        cf: { cacheTtl: 0, cacheEverything: false },
      });
      if (!response.ok) {
        console.error(`subway-alerts fetch failed: HTTP ${response.status}`);
        return;
      }
      const payload = (await response.json()) as { entity?: unknown[] };
      const count = payload.entity?.length ?? 0;
      console.log(`subway-alerts: ${count} entities`);

      // Smoke check: write a tiny status object so we can verify the R2
      // binding end-to-end. Replaced by a real snapshot in the next iteration.
      ctx.waitUntil(
        env.MOMENTARILY.put(
          'health/last_tick.json',
          JSON.stringify({
            started_at: startedAt,
            cron: event.cron,
            subway_alerts_count: count,
          }),
          {
            httpMetadata: {
              contentType: 'application/json',
              cacheControl: 'no-store',
            },
          },
        ),
      );
    } catch (err) {
      console.error('tick failed', err);
    }
  },
} satisfies ExportedHandler<Env>;
