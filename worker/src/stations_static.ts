/**
 * Static station metadata from NYS Open Data 39hk-dx4f (MTA Subway Stations).
 *
 * Refreshed daily and cached in its own R2 object — NOT in last_seen.json. The
 * ~500 station records would bloat that hot per-tick state file enough to risk
 * the cron CPU budget on parse/stringify, so the heavy payload lives apart and
 * last_seen carries only the `stations_at` gate epoch.
 *
 * The snapshot reads this cache each tick and embeds it as the `stations`
 * surface; station_status (keyed by complex id) references the metadata here.
 */

export const STATIONS_KEY = 'state/stations.json';

export interface StationOut {
  gtfs_stop_id: string;
  station_complex_id: string | null;
  name: string;
  borough: string | null;
  routes_served: string[];
  ada: 0 | 1 | 2;
  ada_northbound: boolean;
  ada_southbound: boolean;
}

interface StationsCache {
  /** Epoch seconds of the fetch that produced this cache. */
  fetched_at: number;
  /** Keyed by gtfs_stop_id. */
  stations: Record<string, StationOut>;
}

// 39hk-dx4f borough codes → full names. Unknown codes pass through as-is.
const BOROUGH_NAMES: Record<string, string> = {
  M: 'Manhattan',
  Bx: 'Bronx',
  Bk: 'Brooklyn',
  Q: 'Queens',
  SI: 'Staten Island',
};

function asString(v: unknown): string | null {
  return typeof v === 'string' && v.length > 0 ? v : null;
}

/** Coerce the feed's "0" | "1" | "2" ada string to the schema's literal. */
function parseAda(v: unknown): 0 | 1 | 2 {
  const n = typeof v === 'string' ? Number.parseInt(v, 10) : NaN;
  return n === 1 ? 1 : n === 2 ? 2 : 0;
}

/** Parse the Socrata rows array into station records, skipping malformed rows. */
export function parseStationsFeed(payload: unknown): StationOut[] {
  if (!Array.isArray(payload)) return [];
  const out: StationOut[] = [];
  for (const row of payload) {
    if (!row || typeof row !== 'object') continue;
    const r = row as Record<string, unknown>;
    const gtfs_stop_id = asString(r.gtfs_stop_id);
    const name = asString(r.stop_name);
    if (!gtfs_stop_id || !name) continue;
    const boroughCode = asString(r.borough);
    out.push({
      gtfs_stop_id,
      station_complex_id: asString(r.complex_id),
      name,
      borough: boroughCode ? (BOROUGH_NAMES[boroughCode] ?? boroughCode) : null,
      routes_served: asString(r.daytime_routes)?.split(/\s+/).filter(Boolean) ?? [],
      ada: parseAda(r.ada),
      ada_northbound: r.ada_northbound === '1',
      ada_southbound: r.ada_southbound === '1',
    });
  }
  return out;
}

/** Read the cached station metadata, or null when absent/corrupt. */
export async function readStationsCache(
  bucket: R2Bucket,
): Promise<{ stations: Record<string, StationOut>; fetched_at: number } | null> {
  const obj = await bucket.get(STATIONS_KEY);
  if (!obj) return null;
  try {
    const data = (await obj.json()) as StationsCache;
    if (!data || typeof data.fetched_at !== 'number' || typeof data.stations !== 'object') {
      return null;
    }
    return { stations: data.stations, fetched_at: data.fetched_at };
  } catch (err) {
    console.error('stations.json corrupt; ignoring:', err);
    return null;
  }
}

/** Replace the cached station metadata. Plain put — daily full refresh, no CAS. */
export async function writeStationsCache(
  bucket: R2Bucket,
  stations: StationOut[],
  fetchedAt: number,
): Promise<void> {
  const byId: Record<string, StationOut> = {};
  for (const s of stations) byId[s.gtfs_stop_id] = s;
  const cache: StationsCache = { fetched_at: fetchedAt, stations: byId };
  await bucket.put(STATIONS_KEY, JSON.stringify(cache), {
    httpMetadata: { contentType: 'application/json', cacheControl: 'no-store' },
  });
}
