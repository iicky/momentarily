/**
 * MTA gateway feeds we read.
 *
 * All endpoints serve publicly without authentication — the JSON feeds verified
 * 2026-05-11, the protobuf trip-update feeds verified 2026-06-14 (HTTP 200, no
 * API key). The gateway serves GTFS-realtime protobuf at the non-`.json` paths.
 */

const MTA_GATEWAY = 'https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds';

export const FEEDS = {
  alerts: `${MTA_GATEWAY}/camsys%2Fsubway-alerts.json`,
  ene_current: `${MTA_GATEWAY}/nyct%2Fnyct_ene.json`,
  ene_upcoming: `${MTA_GATEWAY}/nyct%2Fnyct_ene_upcoming.json`,
  ene_equipments: `${MTA_GATEWAY}/nyct%2Fnyct_ene_equipments.json`,
} as const;

export type FeedName = keyof typeof FEEDS;

// NYS Open Data — MTA Subway Stations (39hk-dx4f). Socrata JSON on data.ny.gov,
// not the MTA gateway. ~500 rows, under Socrata's default page size, but pin an
// explicit limit so a future row-count bump can't silently truncate the set.
export const STATIONS_FEED = 'https://data.ny.gov/resource/39hk-dx4f.json?$limit=2000';

/**
 * GTFS-realtime trip-update feeds, one per NYCT line group. Protobuf, not JSON.
 * Archived (derived) for offline recovery validation — see trip_updates.ts.
 */
export const TRIP_UPDATE_FEEDS: ReadonlyArray<readonly [string, string]> = [
  ['ace', `${MTA_GATEWAY}/nyct%2Fgtfs-ace`],
  ['bdfm', `${MTA_GATEWAY}/nyct%2Fgtfs-bdfm`],
  ['g', `${MTA_GATEWAY}/nyct%2Fgtfs-g`],
  ['jz', `${MTA_GATEWAY}/nyct%2Fgtfs-jz`],
  ['nqrw', `${MTA_GATEWAY}/nyct%2Fgtfs-nqrw`],
  ['l', `${MTA_GATEWAY}/nyct%2Fgtfs-l`],
  ['numbered', `${MTA_GATEWAY}/nyct%2Fgtfs`],
  ['si', `${MTA_GATEWAY}/nyct%2Fgtfs-si`],
] as const;

// Bound each upstream fetch so a hung feed can't stretch the tick toward the
// cron CPU limit — on timeout the caller's try/catch treats it as a feed gap
// and the next tick retries.
const FETCH_TIMEOUT_MS = 10_000;

/**
 * Fetch a JSON feed with no Cloudflare edge caching — we always want a fresh
 * pull from origin on each cron tick.
 */
export async function fetchJson(url: string): Promise<unknown> {
  const response = await fetch(url, {
    cf: { cacheTtl: 0, cacheEverything: false },
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} from ${url}`);
  }
  return response.json();
}

/**
 * Fetch a binary (protobuf) feed as raw bytes, no edge caching — same fresh-pull
 * policy as fetchJson.
 */
export async function fetchProtobuf(url: string): Promise<Uint8Array> {
  const response = await fetch(url, {
    cf: { cacheTtl: 0, cacheEverything: false },
    signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
  });
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} from ${url}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}
