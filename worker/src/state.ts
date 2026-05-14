/**
 * Rolling state the Worker persists between cron ticks.
 *
 * Lives at r2://momentarily/state/last_seen.json. Read at the start of each
 * tick, mutated as new alert versions arrive, written back at the end.
 *
 * `alerts` is a map from alert id → the `updated_at` epoch of the most-recent
 * version we've already archived. New (alert_id, updated_at) pairs trigger a
 * write into `archive/alerts/...`.
 *
 * `alerts_at` is the epoch of the last *successful* alerts fetch. Tracked
 * separately from `ene_at` so the snapshot can report alerts-feed freshness
 * honestly when a fetch fails, instead of borrowing the E&E timestamp.
 *
 * `ene_at` is the epoch of the last hourly E&E snapshot we wrote. Compared
 * against `now` to decide whether to fetch the E&E feeds this tick.
 */

import { z } from 'zod';

export const STATE_KEY = 'state/last_seen.json';

export const LastSeenSchema = z.object({
  alerts: z.record(z.string(), z.number()),
  alerts_at: z.number().default(0),
  ene_at: z.number(),
});
export type LastSeen = z.infer<typeof LastSeenSchema>;

export function emptyLastSeen(): LastSeen {
  return { alerts: {}, alerts_at: 0, ene_at: 0 };
}

export async function readLastSeen(bucket: R2Bucket): Promise<LastSeen> {
  const obj = await bucket.get(STATE_KEY);
  if (!obj) return emptyLastSeen();
  try {
    const data = await obj.json();
    return LastSeenSchema.parse(data);
  } catch (err) {
    console.error('last_seen.json corrupt; resetting:', err);
    return emptyLastSeen();
  }
}

export async function writeLastSeen(
  bucket: R2Bucket,
  state: LastSeen,
): Promise<void> {
  await bucket.put(STATE_KEY, JSON.stringify(state), {
    httpMetadata: {
      contentType: 'application/json',
      cacheControl: 'no-store',
    },
  });
}
