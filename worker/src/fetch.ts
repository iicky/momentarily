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

// Full expected set, same order as above — published as trains.json's
// `expected_feeds` so a consumer can tell a real-empty read from a partial
// one (fresh_feeds shorter than this) without hardcoding NYCT's feed
// grouping itself.
export const TRIP_UPDATE_FEED_NAMES: readonly string[] = TRIP_UPDATE_FEEDS.map(([name]) => name);

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

// Protobuf length-delimited wire type — the one FeedMessage.entity uses.
const WIRE_LEN = 2;

/**
 * True when the buffer carries at least one top-level GTFS-RT FeedEntity
 * (FeedMessage field 2). A minimal wire scan — read each top-level tag, skip
 * non-entity fields by wire type — enough to tell a real feed from a truncated
 * or garbage 200 that decodes to nothing, without materializing any entity.
 * Kept local so the fetch floor owns its own check rather than reaching into
 * the decoder's internals.
 */
function hasFeedEntity(buf: Uint8Array): boolean {
  let p = 0;
  const varint = (): number => {
    let result = 0;
    let shift = 0;
    let b: number;
    do {
      b = buf[p++] ?? 0;
      result += (b & 0x7f) * 2 ** shift;
      shift += 7;
    } while (b & 0x80 && p < buf.length);
    return result;
  };
  while (p < buf.length) {
    const tag = varint();
    const field = Math.floor(tag / 8);
    const wire = tag & 7;
    if (field === 2 && wire === WIRE_LEN) return true;
    switch (wire) {
      case 0: // varint
        varint();
        break;
      case 1: // 64-bit
        p += 8;
        break;
      case WIRE_LEN: {
        // Read the length FIRST: `p += varint()` would sum the stale pre-read p
        // with the length and land a varint-byte short (see gtfsrt.ts's skip).
        const n = varint();
        p += n;
        break;
      }
      case 5: // 32-bit
        p += 4;
        break;
      default:
        // Unknown wire type — bail rather than desync and misread.
        p = buf.length;
    }
  }
  return false;
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
  const bytes = new Uint8Array(await response.arrayBuffer());
  // A 200 is necessary but not sufficient. The gateway occasionally serves a
  // truncated or empty body that the tolerant GTFS-RT reader decodes to zero
  // entities WITHOUT throwing — which would otherwise be recorded as a fresh
  // feed of partial/garbage rows in vehicleFreshFeeds. Require at least one
  // FeedEntity so an obviously-empty decode is treated as a failed feed (a
  // gap the caller's Promise.allSettled skips), not confident garbage.
  if (!hasFeedEntity(bytes)) {
    throw new Error(`empty GTFS-RT feed (zero entities) from ${url}`);
  }
  return bytes;
}
