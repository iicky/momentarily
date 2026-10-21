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

// NYS Open Data — MTA Subway Entrances and Exits (i9wp-a4ja). Same Socrata
// JSON endpoint. ~2120 rows; pin a limit above real count to guard against
// silent truncation.
export const ENTRANCES_FEED = 'https://data.ny.gov/resource/i9wp-a4ja.json?$limit=5000';

/**
 * GTFS-realtime trip-update feeds, one per NYCT line group. Protobuf, not JSON.
 * Archived (derived) for offline recovery validation — see trip_updates.ts.
 */
export const TRIP_UPDATE_FEEDS = [
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

// Fresh pull from origin on each cron tick: no Cloudflare edge caching.
const NO_EDGE_CACHE = { cf: { cacheTtl: 0, cacheEverything: false } } as const;

/**
 * One HTTP attempt as it happened, before any interpretation of the body.
 * The body is read only for a 2xx GET; `failure` says why the attempt ended
 * without one when headers or body never arrived. A failure after 2xx headers
 * keeps `response`, so the status that did arrive is not lost.
 */
export interface RawFetch {
  url: string;
  method: 'GET' | 'HEAD';
  startedAtMs: number;
  /** null exactly when the attempt timed out. */
  finishedAtMs: number | null;
  response: { status: number; headers: Headers } | null;
  body: Uint8Array | null;
  failure: 'timeout' | 'network_error' | null;
}

/** A JSON body's parse result, handed to an observer with the raw attempt. */
export type JsonParse = { ok: true; value: unknown } | { ok: false };

/**
 * Sees every attempt a fetch helper makes, whatever its outcome, before the
 * helper applies its own success rules. The recorder (recorder.ts) is the
 * only observer; `json` is present only for a JSON fetch whose body was read.
 */
export type FetchObserver = (raw: RawFetch, json?: JsonParse) => void;

/**
 * Make one attempt and report it as data; never throws. `init.signal` is
 * replaced by a `timeoutMs` budget covering headers and body alike.
 */
export async function fetchRaw(
  url: string,
  init: RequestInit & { method?: 'GET' | 'HEAD' },
  timeoutMs: number,
): Promise<RawFetch> {
  const method = init.method ?? 'GET';
  const signal = AbortSignal.timeout(timeoutMs);
  const raw: RawFetch = {
    url,
    method,
    startedAtMs: Date.now(),
    finishedAtMs: null,
    response: null,
    body: null,
    failure: null,
  };
  try {
    const response = await fetch(url, { ...init, signal });
    raw.response = { status: response.status, headers: response.headers };
    if (response.ok && method === 'GET') {
      raw.body = new Uint8Array(await response.arrayBuffer());
    }
    raw.finishedAtMs = Date.now();
  } catch (err) {
    const timedOut = signal.aborted || (err as { name?: unknown } | null)?.name === 'TimeoutError';
    raw.failure = timedOut ? 'timeout' : 'network_error';
    raw.finishedAtMs = timedOut ? null : Date.now();
  }
  return raw;
}

/** Today's success rule shared by both helpers: a 2xx whose body was read. */
function requireBody(raw: RawFetch): Uint8Array {
  if (raw.failure !== null) {
    throw new Error(`${raw.failure} from ${raw.url}`);
  }
  if (raw.body === null) {
    throw new Error(`HTTP ${raw.response?.status} from ${raw.url}`);
  }
  return raw.body;
}

/**
 * Fetch a JSON feed with no Cloudflare edge caching — we always want a fresh
 * pull from origin on each cron tick. `observe` sees the attempt and the parse.
 */
export async function fetchJson(url: string, observe?: FetchObserver): Promise<unknown> {
  const raw = await fetchRaw(url, NO_EDGE_CACHE, FETCH_TIMEOUT_MS);
  if (raw.body === null) {
    observe?.(raw);
    return requireBody(raw);
  }
  // Response.json()'s own steps: UTF-8 decode (BOM stripped), then JSON.parse.
  let value: unknown;
  try {
    value = JSON.parse(new TextDecoder().decode(raw.body));
  } catch (err) {
    observe?.(raw, { ok: false });
    throw err;
  }
  observe?.(raw, { ok: true, value });
  return value;
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
 * policy as fetchJson. `observe` sees the attempt before the entity check.
 */
export async function fetchProtobuf(url: string, observe?: FetchObserver): Promise<Uint8Array> {
  const raw = await fetchRaw(url, NO_EDGE_CACHE, FETCH_TIMEOUT_MS);
  observe?.(raw);
  const bytes = requireBody(raw);
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
