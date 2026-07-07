// Per-date fetch cache for the archive streams.
//
// The archive is partitioned by UTC date (archive/vehicles/<date>/…,
// v1/predictions/<date>/…). Once a date is in the past it is IMMUTABLE — no new
// objects land under it — so its parsed contents can be cached for the life of
// the process. Only today's partition is still being written, so it's always
// refetched. This is what makes the dominant cost (hundreds of small R2 GETs)
// disappear on window changes and refreshes: a 7-day window after warmup only
// re-reads today.

import { listKeys, getText } from "./r2";

const DAY_CACHE = new Map<string, unknown[]>();

function todayUtc(): string {
  return new Date().toISOString().slice(0, 10);
}

async function poolFetch(keys: string[], limit: number): Promise<string[]> {
  const out: string[] = new Array(keys.length);
  let i = 0;
  const workers = Array.from({ length: Math.min(limit, keys.length) }, async () => {
    while (i < keys.length) {
      const idx = i++;
      out[idx] = await getText(keys[idx]);
    }
  });
  await Promise.all(workers);
  return out;
}

/**
 * Fetch + parse every object under `prefix/<date>/`. Past UTC dates are cached
 * (immutable); today is always live. `parse` turns one object's text into 0+
 * records (one JSON body, or many JSONL rows).
 */
export async function fetchDate<T>(
  prefix: string,
  date: string,
  parse: (text: string) => T[],
  concurrency = 8,
): Promise<T[]> {
  const cacheKey = `${prefix}/${date}`;
  const past = date < todayUtc();
  if (past) {
    const hit = DAY_CACHE.get(cacheKey);
    if (hit) return hit as T[];
  }
  const keys = await listKeys(`${prefix}/${date}/`);
  const blobs = await poolFetch(keys, concurrency);
  const records = blobs.flatMap(parse);
  if (past) DAY_CACHE.set(cacheKey, records);
  return records;
}
