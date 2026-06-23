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
  // Epoch of the last successful vehicle-movement metric archive. The per-trip
  // stop_id carry map it depends on lives in its own R2 object, not here, so its
  // ~700 entries don't bloat the per-tick state parse.
  vehicles_at: z.number().default(0),
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
    vehicles_at: 0,
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

// The vehicle-movement cross-tick signal needs last tick's trip_id → stop_id
// map. It lives in its own R2 object, deliberately NOT in last_seen.json: the
// map is ~700 entries and last_seen.json is parsed + stringified on every 5-min
// tick, so folding it in would compound the JSON cost that has caused CPU-limit
// outages before. Plain put (no CAS) — step 8b is already gated on the alpha
// winner, so only one run writes it per tick.
export const VEHICLE_STOPS_KEY = 'state/vehicle_stops.json';

const VehicleStopsSchema = z.record(z.string(), z.string());

/** Read last tick's per-trip stop_id carry map. Returns {} when absent or
 * corrupt — the cross-tick counters just stay 0 that tick. */
export async function readVehicleStops(
  bucket: R2Bucket,
): Promise<Record<string, string>> {
  const obj = await bucket.get(VEHICLE_STOPS_KEY);
  if (!obj) return {};
  try {
    return VehicleStopsSchema.parse(await obj.json());
  } catch (err) {
    console.error('vehicle_stops.json corrupt; resetting:', err);
    return {};
  }
}

export async function writeVehicleStops(
  bucket: R2Bucket,
  stops: Record<string, string>,
): Promise<void> {
  await bucket.put(VEHICLE_STOPS_KEY, JSON.stringify(stops), {
    httpMetadata: { contentType: 'application/json', cacheControl: 'no-store' },
  });
}

// Per-route movement-derived condition, computed at step 8b (post-publish) and
// read by the next tick's snapshot build (pre-publish). Carrying it forward this
// way keeps the vehicle fetch off the time-to-publish path; the route's current
// state is published one tick (~5 min) stale, which a slow freeze/recovery
// regime tolerates. Its own small object (~28 routes), like vehicle_stops.json.
export const MOVEMENT_STATE_KEY = 'state/movement_state.json';

const MovementStateSchema = z.object({
  observed_at: z.number(),
  states: z.record(z.string(), z.enum(['normal', 'disrupted', 'suspended'])),
});
export type MovementStateDoc = z.infer<typeof MovementStateSchema>;

/** Read last tick's movement-derived states. Returns null when absent or corrupt
 * — the snapshot then falls back to the alert/HMM condition for every route. */
export async function readMovementState(
  bucket: R2Bucket,
): Promise<MovementStateDoc | null> {
  const obj = await bucket.get(MOVEMENT_STATE_KEY);
  if (!obj) return null;
  try {
    return MovementStateSchema.parse(await obj.json());
  } catch (err) {
    console.error('movement_state.json corrupt; resetting:', err);
    return null;
  }
}

export async function writeMovementState(
  bucket: R2Bucket,
  doc: MovementStateDoc,
): Promise<void> {
  await bucket.put(MOVEMENT_STATE_KEY, JSON.stringify(doc), {
    httpMetadata: { contentType: 'application/json', cacheControl: 'no-store' },
  });
}
