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

import { z } from "zod";

import { conditionalPut } from './r2';
import type { VersionedRead } from './r2';
import type { MovementRow } from './vehicles';
import type { ServiceRow } from './trip_updates';

export const STATE_KEY = "state/last_seen.json";

// Cached station_status derivations, refreshed when E&E fetches succeed.
// Stored here (vs recomputed each tick) because E&E only updates hourly while
// snapshot.json publishes every 5 min — recomputing across 700 catalog entries
// + 80 outages each tick would burn CPU for no new information.
const StationStatusEntrySchema = z.object({
  station_complex_id: z.string(),
  alerts: z.array(z.string()).default([]),
  ada_status: z.enum(["operational", "ada_degraded", "non_ada"]),
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
  type: z.enum(["elevator", "escalator"]),
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
    console.error("last_seen.json corrupt; resetting:", err);
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
    contentType: "application/json",
    cacheControl: "no-store",
  });
}

// The vehicle-movement cross-tick signal needs last tick's trip_id → stop_id
// map. It lives in its own R2 object, deliberately NOT in last_seen.json: the
// map is ~700 entries and last_seen.json is parsed + stringified on every 5-min
// tick, so folding it in would compound the JSON cost that has caused CPU-limit
// outages before. Plain put (no CAS) — step 8b is already gated on the alpha
// winner, so only one run writes it per tick.
export const VEHICLE_STOPS_KEY = "state/vehicle_stops.json";

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
    console.error("vehicle_stops.json corrupt; resetting:", err);
    return {};
  }
}

export async function writeVehicleStops(
  bucket: R2Bucket,
  stops: Record<string, string>,
): Promise<void> {
  await bucket.put(VEHICLE_STOPS_KEY, JSON.stringify(stops), {
    httpMetadata: { contentType: "application/json", cacheControl: "no-store" },
  });
}

// Per-route movement-derived condition, computed at step 8b (post-publish) and
// read by the next tick's snapshot build (pre-publish). Carrying it forward this
// way keeps the vehicle fetch off the time-to-publish path; the route's current
// state is published one tick (~5 min) stale, which a slow freeze/recovery
// regime tolerates. Its own small object (~28 routes), like vehicle_stops.json.
export const MOVEMENT_STATE_KEY = "state/movement_state.json";

const MovementConditionSchema = z.enum([
  "normal",
  "disrupted",
  "suspended",
  "not_scheduled",
]);

// The debounced regime and its clock, per route. `state` is what the snapshot
// publishes; `entered_at` is how long it has held, which is what the movement
// dwell curves are conditioned on. Shape mirrors RegimeEntry in regime.ts.
const MovementRegimeSchema = z.object({
  state: MovementConditionSchema,
  entered_at: z.number(),
  last_seen_at: z.number(),
  pending: MovementConditionSchema.nullable(),
  pending_since: z.number(),
  pending_run: z.number().int().nonnegative(),
});

// Service-level regime, a SUPPLY axis beside the movement (flow) regimes above.
// 'unknown' is permitted so the type lines up with deriveServiceStates' return,
// but is never actually committed — unknown ticks abstain from the clock.
const ServiceConditionSchema = z.enum(["normal", "degraded", "unknown"]);
const ServiceRegimeSchema = z.object({
  state: ServiceConditionSchema,
  entered_at: z.number(),
  last_seen_at: z.number(),
  pending: ServiceConditionSchema.nullable(),
  pending_since: z.number(),
  pending_run: z.number().int().nonnegative(),
});
export type ServiceRegime = z.infer<typeof ServiceRegimeSchema>;

const MovementStateSchema = z.object({
  observed_at: z.number(),
  regimes: z.record(z.string(), MovementRegimeSchema),
  // Per-route service-level regimes, one-tick lagged like `regimes`. Optional:
  // absent in docs written before the service axis, and on a params set with no
  // hourly service baseline. The snapshot then publishes service_condition
  // 'unknown'.
  service_regimes: z.record(z.string(), ServiceRegimeSchema).optional(),
  // Per-route raw service ratio (assigned_n / hourly baseline) this tick — the
  // magnitude the service_condition axis debounces. Optional, same lifecycle as
  // service_regimes.
  service_ratios: z.record(z.string(), z.number()).optional(),
  // Per-route quantile-derived low/high ratios (cell p10/median, p90/median)
  // this tick — the same-scale spread service_high_ratio/service_low_ratio
  // publish. Optional, same lifecycle as service_ratios, plus one more gate:
  // absent when the trainer hasn't published per-cell quantiles at all.
  service_quantile_ratios: z.record(z.string(), z.object({ low: z.number(), high: z.number() })).optional(),
});
export type MovementRegime = z.infer<typeof MovementRegimeSchema>;
export type MovementStateDoc = z.infer<typeof MovementStateSchema>;

/** Read last tick's movement regimes. Returns null when absent or corrupt — the
 * snapshot then publishes 'unknown' for every route and the regime clocks
 * restart from the next tick. */
export async function readMovementState(
  bucket: R2Bucket,
): Promise<MovementStateDoc | null> {
  const obj = await bucket.get(MOVEMENT_STATE_KEY);
  if (!obj) return null;
  try {
    return MovementStateSchema.parse(await obj.json());
  } catch (err) {
    console.error("movement_state.json corrupt; resetting:", err);
    return null;
  }
}

export async function writeMovementState(
  bucket: R2Bucket,
  doc: MovementStateDoc,
): Promise<void> {
  await bucket.put(MOVEMENT_STATE_KEY, JSON.stringify(doc), {
    httpMetadata: { contentType: "application/json", cacheControl: "no-store" },
  });
}

// Per-route cross-tick movement counts, carried one tick forward to feed the
// HMM movement emission at derive time (step 4). Written at step 8b like
// movement_state.json / vehicle_stops.json and read before the filter next
// tick — a ~5-min lag ("option B") that keeps the vehicle fetch off the
// publish path. By-direction so a future per-direction filter can split it;
// the route-level filter aggregates both. Its own small object (~28 routes).
export const MOVEMENT_METRIC_KEY = "state/movement_metric.json";

const DirCountsSchema = z.object({
  advanced_n: z.number().int().nonnegative(),
  stalled_n: z.number().int().nonnegative(),
});
const MovementMetricSchema = z.object({
  observed_at: z.number(),
  rows: z.record(
    z.string(),
    z.object({ north: DirCountsSchema, south: DirCountsSchema }),
  ),
});
export type MovementMetricDoc = z.infer<typeof MovementMetricSchema>;

/** Read last tick's per-route cross-tick movement counts. Returns null when
 * absent or corrupt — the movement emission channel just drops out that tick. */
export async function readMovementMetric(
  bucket: R2Bucket,
): Promise<MovementMetricDoc | null> {
  const obj = await bucket.get(MOVEMENT_METRIC_KEY);
  if (!obj) return null;
  try {
    return MovementMetricSchema.parse(await obj.json());
  } catch (err) {
    console.error("movement_metric.json corrupt; resetting:", err);
    return null;
  }
}

export async function writeMovementMetric(
  bucket: R2Bucket,
  observedAt: number,
  moveRows: Map<string, MovementRow>,
): Promise<void> {
  const rows: MovementMetricDoc["rows"] = {};
  for (const [route, row] of moveRows) {
    rows[route] = {
      north: {
        advanced_n: row.by_direction.north.advanced_n,
        stalled_n: row.by_direction.north.stalled_n,
      },
      south: {
        advanced_n: row.by_direction.south.advanced_n,
        stalled_n: row.by_direction.south.stalled_n,
      },
    };
  }
  const doc: MovementMetricDoc = { observed_at: observedAt, rows };
  await bucket.put(MOVEMENT_METRIC_KEY, JSON.stringify(doc), {
    httpMetadata: { contentType: "application/json", cacheControl: "no-store" },
  });
}

// Per-route assigned_n (dispatched trains), carried one tick forward to feed the
// HMM service emission at derive time. Written at step 8b, read before the filter
// next tick (option B lag), like movement_metric.json. Its own small object.
export const SERVICE_METRIC_KEY = "state/service_metric.json";

const ServiceMetricSchema = z.object({
  observed_at: z.number(),
  rows: z.record(z.string(), z.number().int().nonnegative()),
});
export type ServiceMetricDoc = z.infer<typeof ServiceMetricSchema>;

/** Read last tick's per-route assigned_n. Returns null when absent or corrupt —
 * the service emission channel just drops out that tick. */
export async function readServiceMetric(
  bucket: R2Bucket,
): Promise<ServiceMetricDoc | null> {
  const obj = await bucket.get(SERVICE_METRIC_KEY);
  if (!obj) return null;
  try {
    return ServiceMetricSchema.parse(await obj.json());
  } catch (err) {
    console.error("service_metric.json corrupt; resetting:", err);
    return null;
  }
}

export async function writeServiceMetric(
  bucket: R2Bucket,
  observedAt: number,
  svcRows: Map<string, ServiceRow>,
): Promise<void> {
  const rows: ServiceMetricDoc["rows"] = {};
  for (const [route, row] of svcRows) {
    rows[route] = row.assigned_n;
  }
  const doc: ServiceMetricDoc = { observed_at: observedAt, rows };
  await bucket.put(SERVICE_METRIC_KEY, JSON.stringify(doc), {
    httpMetadata: { contentType: "application/json", cacheControl: "no-store" },
  });
}

// Segment baseline + adjacency, written by the trainer as its OWN object (not
// folded into params.json — the Worker parses that on the hot per-tick path).
// Read at step 8b to score per-segment movement for the station-flow surface.
// Topology (adjacency) comes from the static GTFS timetable when the trainer's
// feed fetch succeeds, falling back to observed cross-tick adjacency otherwise
// — see training/gtfs_static.py + training/train_em.py write_segment_params.
export const SEGMENT_PARAMS_KEY = "state/segment_params.json";

const SegmentParamsSchema = z.object({
  schema_version: z.literal("1"),
  trained_at: z.number(),
  min_share: z.number().min(0).max(1),
  // Absent on segment_params.json written before the static-GTFS switchover;
  // those docs are entirely observed-adjacency, hence that default.
  topology_source: z.enum(["gtfs_static", "observed"]).default("observed"),
  cells: z.record(
    z.string(),
    z.object({
      p0: z.number().min(0).max(1),
      n: z.number().int().nonnegative(),
    }),
  ),
  adjacency: z.record(
    z.string(),
    z.object({
      to: z.string(),
      // Same default reasoning as topology_source, per entry.
      source: z.enum(["gtfs_static", "observed"]).default("observed"),
      // The observed cross-tick reliability ANNOTATION now, not an existence
      // gate: absent when static topology names a segment the vehicle
      // archive never (or too rarely) saw advance out of.
      share: z.number().min(0).max(1).optional(),
      n: z.number().int().nonnegative().optional(),
      // Full static successor list (a branch/express from_stop has more than
      // one), for path queries — `to` is just the highest-n_trips entry.
      // Absent on observed-sourced (fallback) entries.
      successors: z
        .array(
          z.object({ to: z.string(), n_trips: z.number().int().nonnegative() }),
        )
        .optional(),
    }),
  ),
  // Canonical per-(route, direction) stop order: the scheduled trip patterns,
  // most-run first, keyed 'route|direction'. Written by the trainer from static
  // GTFS (training/gtfs_static.route_patterns) for off-Worker consumers (the
  // viz line/map surfaces) that need line order; the Worker itself doesn't read
  // it. Absent on docs written before this field, and {} on the observed
  // fallback. Read as a whole, so a malformed entry degrades the ordering, not
  // the segment scoring.
  route_stops: z
    .record(
      z.string(),
      z.array(z.object({ stops: z.array(z.string()), n_trips: z.number().int().nonnegative() })),
    )
    .optional(),
  // Which code produced this doc — {code_sha, dirty, producer}, the same block
  // params.json/eval.json carry (training/provenance.py). Absent on docs written
  // before this field. The Worker doesn't read it; it's for off-Worker consumers
  // (viz) that surface the topology's lineage.
  provenance: z
    .object({
      code_sha: z.string(),
      dirty: z.boolean().nullable(),
      producer: z.string(),
    })
    .optional(),
});
export type SegmentParamsDoc = z.infer<typeof SegmentParamsSchema>;

/** Read the segment baseline object. Null when absent or corrupt — the station-
 * flow surface just isn't produced that tick. */
export async function readSegmentParams(
  bucket: R2Bucket,
): Promise<SegmentParamsDoc | null> {
  const obj = await bucket.get(SEGMENT_PARAMS_KEY);
  if (!obj) return null;
  try {
    return SegmentParamsSchema.parse(await obj.json());
  } catch (err) {
    console.error("segment_params.json corrupt; station flow off:", err);
    return null;
  }
}

// Per-(route, schedule_bin) assigned_n baseline — the denominator of the
// published service-degradation (supply) axis. Its OWN object, written by the
// trainer like segment_params.json, so the display axis is decoupled from the
// frozen HMM artifact in params.json: landing or refreshing this baseline never
// changes params.json's trained_at, so it never reseeds the filter or splits the
// grader's params-version window. The Worker prefers this object and falls back
// to params.json's legacy service_baseline_hourly when it's absent.
export const SERVICE_BASELINE_KEY = "state/service_baseline.json";

// route -> schedule_bin -> {p10, p90} of assigned_n at that cell, same units
// and the same min_samples gate as `baseline` — a cell present in `baseline`
// should be present here and vice versa. Sibling to `baseline`, not derived
// from it: the median stays the one baseline value, this is only its spread.
const ServiceQuantilesSchema = z.record(
  z.string(),
  z.record(z.string(), z.object({ p10: z.number().nonnegative(), p90: z.number().nonnegative() })),
);
export type ServiceQuantiles = z.infer<typeof ServiceQuantilesSchema>;

const ServiceBaselineDocSchema = z.object({
  schema_version: z.literal("1"),
  // The sidecar's OWN publication stamp, and its versioned-snapshot key. Kept
  // distinct from params.json's trained_at: each refresh gets a fresh stamp so
  // it never overwrites a prior immutable copy nor moves the model version.
  generated_at: z.number(),
  // The frozen model this baseline was computed to accompany. Provenance only —
  // the Worker never keys off it.
  params_trained_at: z.number().optional(),
  // route -> schedule_bin (stringified) -> median assigned_n.
  baseline: z.record(z.string(), z.record(z.string(), z.number().nonnegative())),
  // Optional: absent on sidecars written before the per-cell spread existed.
  // Old documents still parse; the supply axis's low/high ratios then read
  // null instead of falling back to a global multiple.
  quantiles: ServiceQuantilesSchema.optional(),
});
export type ServiceBaselineDoc = z.infer<typeof ServiceBaselineDocSchema>;

/** Read the service-baseline sidecar. Null when absent or corrupt — the supply
 * axis then falls back to params.json's baseline, else reads 'unknown'. */
export async function readServiceBaseline(
  bucket: R2Bucket,
): Promise<ServiceBaselineDoc | null> {
  const obj = await bucket.get(SERVICE_BASELINE_KEY);
  if (!obj) return null;
  try {
    return ServiceBaselineDocSchema.parse(await obj.json());
  } catch (err) {
    console.error(
      "service_baseline.json corrupt; supply axis falls back to params:",
      err,
    );
    return null;
  }
}

// Decaying per-segment advance/matched accumulator, carried tick to tick so a
// ~1-train-per-tick segment accrues enough to judge. Its own object, step 8b.
export const SEGMENT_FLOW_KEY = "state/segment_flow.json";

// Segment cell's debounced regime and its clock — same shape as
// MovementRegimeSchema (mirrors RegimeEntry in regime.ts), scoped to the two
// calls classifyAdvance can make for a segment cell. No suspended/
// not_scheduled: those come from the route's own schedule, not a segment.
const SegmentConditionSchema = z.enum(["normal", "disrupted"]);

const SegmentRegimeSchema = z.object({
  state: SegmentConditionSchema,
  entered_at: z.number(),
  last_seen_at: z.number(),
  pending: SegmentConditionSchema.nullable(),
  pending_since: z.number(),
  pending_run: z.number().int().nonnegative(),
});
export type SegmentRegime = z.infer<typeof SegmentRegimeSchema>;

const SegmentFlowSchema = z.object({
  observed_at: z.number(),
  cells: z.record(z.string(), z.object({ a: z.number(), m: z.number() })),
  // Absent on a live doc written before the regime clock landed; defaults to
  // {} so it still parses instead of resetting the decayed cell state too.
  regimes: z.record(z.string(), SegmentRegimeSchema).default({}),
});
export type SegmentFlowDoc = z.infer<typeof SegmentFlowSchema>;

export async function readSegmentFlow(
  bucket: R2Bucket,
): Promise<SegmentFlowDoc | null> {
  const obj = await bucket.get(SEGMENT_FLOW_KEY);
  if (!obj) return null;
  try {
    return SegmentFlowSchema.parse(await obj.json());
  } catch (err) {
    console.error("segment_flow.json corrupt; resetting:", err);
    return null;
  }
}

export async function writeSegmentFlow(
  bucket: R2Bucket,
  doc: SegmentFlowDoc,
): Promise<void> {
  await bucket.put(SEGMENT_FLOW_KEY, JSON.stringify(doc), {
    httpMetadata: { contentType: "application/json", cacheControl: "no-store" },
  });
}

// Per-segment dwell curves, hierarchically pooled leaf -> route -> system
// (training/segment_dwell.py) from the segment-scope movement_transitions
// stream. Its own R2 object, written by the trainer like segment_params.json
// — segment episodes are short and the ~1.8k-cell curve set never touches the
// hot per-tick params parse. No writer here: the trainer publishes it
// directly, same as SEGMENT_PARAMS_KEY.
export const SEGMENT_DWELL_KEY = "state/segment_dwell.json";

// Mirrors training.dwell.DwellQuantiles exactly. A fresh object with no
// params.json back-compat baggage, so unlike params.ts's older DwellQuantiles
// mirror every field here is required except tail_ll (absent when no
// log-logistic fit converged — no completed events at all).
const SegmentDwellQuantilesSchema = z.object({
  n: z.number().int().nonnegative(),
  n_censored: z.number().int().nonnegative(),
  q25_sec: z.number().int().nonnegative(),
  median_sec: z.number().int().nonnegative(),
  q75_sec: z.number().int().nonnegative(),
  recover_by_30: z.number().min(0).max(1),
  recover_by_60: z.number().min(0).max(1),
  recover_by_120: z.number().min(0).max(1),
  curve_sec: z.array(z.number().nonnegative()).min(2),
  tail_ll: z.tuple([z.number().positive(), z.number().positive()]).optional(),
});
export type SegmentDwellQuantiles = z.infer<typeof SegmentDwellQuantilesSchema>;

const SegmentDwellSchema = z.object({
  schema_version: z.literal("1"),
  trained_at: z.number(),
  cells: z.record(
    z.string(),
    z.record(z.string(), SegmentDwellQuantilesSchema),
  ),
});
export type SegmentDwellDoc = z.infer<typeof SegmentDwellSchema>;

/** Read the per-segment dwell curves. Null when absent or corrupt — callers
 * fall back to whatever route-level or geometric dwell they already have. */
export async function readSegmentDwell(
  bucket: R2Bucket,
): Promise<SegmentDwellDoc | null> {
  const obj = await bucket.get(SEGMENT_DWELL_KEY);
  if (!obj) return null;
  try {
    return SegmentDwellSchema.parse(await obj.json());
  } catch (err) {
    console.error(
      "segment_dwell.json corrupt; segment dwell curves unavailable:",
      err,
    );
    return null;
  }
}

// Per-station service-flow, computed at step 8b and read by the next tick's
// snapshot build (one tick / ~5-min lag, like movement_state).
export const STATION_FLOW_KEY = "state/station_flow.json";

const StationFlowSchema = z.object({
  observed_at: z.number(),
  stations: z.record(
    z.string(),
    z.object({
      status: z.enum(["flowing", "degraded"]),
      worst_deficit: z.number(),
      worst_segment: z.tuple([z.string(), z.string()]).nullable(),
      routes: z.array(z.string()),
      n_segments: z.number().int().nonnegative(),
    }),
  ),
});
export type StationFlowDoc = z.infer<typeof StationFlowSchema>;

/** Read last tick's per-station service flow. Null when absent or corrupt — the
 * snapshot publishes without the station-flow surface. */
export async function readStationFlow(
  bucket: R2Bucket,
): Promise<StationFlowDoc | null> {
  const obj = await bucket.get(STATION_FLOW_KEY);
  if (!obj) return null;
  try {
    return StationFlowSchema.parse(await obj.json());
  } catch (err) {
    console.error("station_flow.json corrupt; resetting:", err);
    return null;
  }
}

export async function writeStationFlow(
  bucket: R2Bucket,
  doc: StationFlowDoc,
): Promise<void> {
  await bucket.put(STATION_FLOW_KEY, JSON.stringify(doc), {
    httpMetadata: { contentType: "application/json", cacheControl: "no-store" },
  });
}
