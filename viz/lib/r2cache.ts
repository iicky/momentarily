// Per-key fetch cache for the archive streams.
//
// The archive is partitioned by UTC date, and every object under it is written
// once and never rewritten (one tick's snapshot per key). So a KEY's parsed
// value is cached for the life of the process — including today's, because a
// key that exists is already final; only genuinely new keys (today's latest
// ticks) are ever fetched. This is what makes the dominant cost — hundreds to
// thousands of small R2 GETs — vanish on window changes, line filters, and
// refreshes: a re-list is cheap, and everything already seen comes from memory.
// (Per-key, not per-date: the old per-date cache refetched all of today every
// request, which for the per-minute vehicle archive was the whole cost.)

import { listKeys, getText } from "./r2";

const KEY_CACHE = new Map<string, unknown[]>();

async function pool<T>(items: (() => Promise<T>)[], limit: number): Promise<T[]> {
  const out: T[] = new Array(items.length);
  let i = 0;
  const workers = Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (i < items.length) {
      const idx = i++;
      out[idx] = await items[idx]();
    }
  });
  await Promise.all(workers);
  return out;
}

/** Parse one immutable object key, cached for the process lifetime. `parse`
 * turns the object's text into 0+ records (one JSON body, or many JSONL rows). */
export async function cachedRecords<T>(
  key: string,
  parse: (text: string) => T[],
): Promise<T[]> {
  const hit = KEY_CACHE.get(key);
  if (hit !== undefined) return hit as T[];
  const records = parse(await getText(key));
  KEY_CACHE.set(key, records);
  return records;
}

/**
 * Fetch + parse every object under `prefix/<date>/`, each key cached. Re-lists
 * the date each call (cheap) so new keys are picked up, but only uncached keys
 * are fetched.
 */
export async function fetchDate<T>(
  prefix: string,
  date: string,
  parse: (text: string) => T[],
  concurrency = 8,
): Promise<T[]> {
  const keys = await listKeys(`${prefix}/${date}/`);
  const per = await pool(keys.map((k) => () => cachedRecords<T>(k, parse)), concurrency);
  return per.flat();
}
