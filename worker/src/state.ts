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

import { conditionalPut } from './r2';
import type { VersionedRead } from './r2';

export const STATE_KEY = 'state/last_seen.json';

// Cached station_status derivations, refreshed when E&E fetches succeed.
// Stored here (vs recomputed each tick) because E&E only updates hourly while
// snapshot.json publishes every 5 min — recomputing across 700 catalog entries
// + 80 outages each tick would burn CPU for no new information.
const StationStatusEntrySchema = z.object({
  station_complex_id: z.string(),
  alerts: z.array(z.string()).default([]),
  ada_status: z.enum(['operational', 'ada_degraded', 'non_ada']),
  elevators_total: z.number().int().nonnegative(),
  elevators_out: z.number().int().nonnegative(),
  escalators_total: z.number().int().nonnegative(),
  escalators_out: z.number().int().nonnegative(),
  earliest_elevator_return: z.number().nullable(),
  oldest_outage_since: z.number().nullable(),
});

// Cached equipment-outage list, refreshed alongside station_statuses on the
// hourly E&E fetch and republished each tick for the same reason.
const EquipmentEntrySchema = z.object({
  equipment_id: z.string(),
  type: z.enum(['elevator', 'escalator']),
  station_complex_id: z.string().nullable(),
  location_text: z.string().nullable(),
  ada_pathway: z.boolean(),
  outage: z.object({
    reason: z.string().nullable(),
    est_return: z.number().nullable(),
    since: z.number().nullable(),
  }),
});

export const LastSeenSchema = z.object({
  alerts: z.record(z.string(), z.number()),
  alerts_at: z.number().default(0),
  ene_at: z.number(),
  // Epoch of the last successful trip-updates metric archive. Defaulted for
  // back-compat with last_seen.json written before trip-updates shipped.
  trip_updates_at: z.number().default(0),
  station_statuses: z.record(z.string(), StationStatusEntrySchema).default({}),
  equipment: z.array(EquipmentEntrySchema).default([]),
  // Epoch of the last successful daily stations-static fetch. Gates the daily
  // refresh; the heavy station payload itself lives in its own R2 object, not
  // here, to keep this per-tick state file small.
  stations_at: z.number().default(0),
});
export type LastSeen = z.infer<typeof LastSeenSchema>;

export function emptyLastSeen(): LastSeen {
  return {
    alerts: {},
    alerts_at: 0,
    ene_at: 0,
    trip_updates_at: 0,
    station_statuses: {},
    equipment: [],
    stations_at: 0,
  };
}

export async function readLastSeen(
  bucket: R2Bucket,
): Promise<VersionedRead<LastSeen>> {
  const obj = await bucket.get(STATE_KEY);
  if (!obj) return { state: emptyLastSeen(), etag: null };
  try {
    const data = await obj.json();
    return { state: LastSeenSchema.parse(data), etag: obj.etag };
  } catch (err) {
    console.error('last_seen.json corrupt; resetting:', err);
    return { state: emptyLastSeen(), etag: obj.etag };
  }
}

/**
 * Write last_seen.json with compare-and-swap on `etag` (from readLastSeen).
 * Returns false when a concurrent tick already advanced the object.
 */
export async function writeLastSeen(
  bucket: R2Bucket,
  state: LastSeen,
  etag: string | null,
): Promise<boolean> {
  return conditionalPut(bucket, STATE_KEY, JSON.stringify(state), etag, {
    contentType: 'application/json',
    cacheControl: 'no-store',
  });
}
