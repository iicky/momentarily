/**
 * MTA gateway feeds we read.
 *
 * All four endpoints serve publicly without authentication (verified 2026-05-11);
 * if we ever add protobuf trip-updates the gateway will need an API key.
 */

const MTA_GATEWAY = 'https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds';

export const FEEDS = {
  alerts: `${MTA_GATEWAY}/camsys%2Fsubway-alerts.json`,
  ene_current: `${MTA_GATEWAY}/nyct%2Fnyct_ene.json`,
  ene_upcoming: `${MTA_GATEWAY}/nyct%2Fnyct_ene_upcoming.json`,
  ene_equipments: `${MTA_GATEWAY}/nyct%2Fnyct_ene_equipments.json`,
} as const;

export type FeedName = keyof typeof FEEDS;

/**
 * Fetch a JSON feed with no Cloudflare edge caching — we always want a fresh
 * pull from origin on each cron tick.
 */
export async function fetchJson(url: string): Promise<unknown> {
  const response = await fetch(url, {
    cf: { cacheTtl: 0, cacheEverything: false },
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} from ${url}`);
  }
  return response.json();
}
