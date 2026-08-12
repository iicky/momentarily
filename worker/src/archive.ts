/**
 * Write change events to R2 archive.
 *
 * One R2 object per change. Sortable keys let downstream consumers (training
 * runs, calibration notebooks) reconstruct the corpus by listing prefixes.
 *
 *   archive/alerts/YYYY-MM-DD/<updated_at>-<alert_id>.json
 *   archive/ene/YYYY-MM-DD/HH0000-<source>.json
 *
 * We dedupe alerts by (alert_id, updated_at) — an alert that persists for hours
 * occupies one R2 object until its `updated_at` changes, not 72 copies of the
 * same payload like the legacy Python collector did.
 *
 * Object keys are deterministic (alert version / hour bucket, not wall-clock
 * tick time) so an overlapping or retried scheduled run overwrites the same key
 * instead of producing a duplicate object. The date folder still tracks the
 * observation date so the Python loader's date-prefix listing keeps working.
 *
 * E&E is written as a full snapshot per hourly tick (per feed). Volume is low
 * (~3 PUTs/hour × 24 = 72/day) so no dedupe needed yet.
 */

import type { LastSeen } from './state';
import type { ServiceRow } from './trip_updates';
import type { MovementRow, TraceRow } from './vehicles';

function utcDate(epoch: number): string {
  return new Date(epoch * 1000).toISOString().slice(0, 10);
}

function utcTime(epoch: number): string {
  return new Date(epoch * 1000).toISOString().slice(11, 19).replace(/:/g, '');
}

function safeKey(s: string): string {
  return s.replace(/[^a-zA-Z0-9._-]/g, '_');
}

/**
 * Iterate the alerts payload, write a new R2 object for each
 * (alert_id, updated_at) pair we haven't seen before, and mutate `lastSeen`
 * in place to record the new versions. Returns the count of new writes.
 */
export async function archiveNewAlerts(
  bucket: R2Bucket,
  payload: unknown,
  lastSeen: LastSeen,
  observedAt: number,
): Promise<number> {
  const entities = extractEntities(payload);
  if (!entities) return 0;

  const datePrefix = utcDate(observedAt);
  let written = 0;

  // Rebuilt from this tick's feed only. Replaces lastSeen.alerts at the end so
  // the dedupe map stays bounded to the live alert set instead of growing
  // forever with ids that will never appear again.
  const seen: Record<string, number> = {};

  for (const entity of entities) {
    const parsed = parseAlertEntity(entity);
    if (!parsed) continue;
    const { id, updatedAt } = parsed;
    seen[id] = updatedAt;

    if (lastSeen.alerts[id] === updatedAt) continue;

    // Key on the alert version (updated_at), not the tick wall-clock, so a
    // retried/overlapping run writes the same object instead of a duplicate.
    const key = `archive/alerts/${datePrefix}/${updatedAt}-${safeKey(id)}.json`;
    await bucket.put(
      key,
      JSON.stringify({ observed_at: observedAt, alert: entity }),
      { httpMetadata: { contentType: 'application/json' } },
    );
    written += 1;
  }

  lastSeen.alerts = seen;
  return written;
}

export async function archiveEneSnapshot(
  bucket: R2Bucket,
  source: string,
  payload: unknown,
  observedAt: number,
): Promise<void> {
  // Bucket to the top of the hour so two runs in the same hourly window write
  // the same key rather than two near-identical snapshots.
  const hourEpoch = Math.floor(observedAt / 3600) * 3600;
  const key = `archive/ene/${utcDate(observedAt)}/${utcTime(hourEpoch)}-${safeKey(source)}.json`;
  await bucket.put(
    key,
    JSON.stringify({ observed_at: observedAt, source, payload }),
    { httpMetadata: { contentType: 'application/json' } },
  );
}

/**
 * Archive the derived per-route trip-updates service metric — one compact
 * object per tick. Deliberately a tiny derived snapshot (a handful of ints per
 * route), not the raw protobuf: raw would be ~hundreds of MB/day and blow the
 * R2 free tier; this is ~1 KB/tick, on par with the alerts archive. Keyed on
 * the tick wall-clock so an overlapping/retried run overwrites rather than
 * duplicates. `freshFeeds` records which line-group feeds were decoded this
 * tick so the offline loader can tell a real zero from a missing feed.
 */
export async function archiveTripUpdateMetric(
  bucket: R2Bucket,
  rows: Map<string, ServiceRow>,
  freshFeeds: string[],
  observedAt: number,
): Promise<void> {
  const key = `archive/trip_updates/${utcDate(observedAt)}/${observedAt}.json`;
  await bucket.put(
    key,
    JSON.stringify({
      observed_at: observedAt,
      fresh_feeds: freshFeeds,
      rows: Object.fromEntries(rows),
    }),
    { httpMetadata: { contentType: 'application/json' } },
  );
}

/**
 * Archive the derived per-route vehicle-movement metric — one compact object
 * per tick, mirroring archiveTripUpdateMetric. Same rationale: a handful of
 * ints per route (~1 KB/tick), not the raw protobuf. Keyed on the tick
 * wall-clock so an overlapping/retried run overwrites rather than duplicates.
 * `freshFeeds` records which line-group feeds decoded so the offline loader can
 * tell a real zero from a missing feed.
 */
export async function archiveVehicleMetric(
  bucket: R2Bucket,
  rows: Map<string, MovementRow>,
  freshFeeds: string[],
  observedAt: number,
): Promise<void> {
  const key = `archive/vehicles/${utcDate(observedAt)}/${observedAt}.json`;
  await bucket.put(
    key,
    JSON.stringify({
      observed_at: observedAt,
      fresh_feeds: freshFeeds,
      rows: Object.fromEntries(rows),
    }),
    { httpMetadata: { contentType: 'application/json' } },
  );
}

/**
 * Archive the per-minute vehicle trace rows — the fine-grained sibling of
 * archiveVehicleMetric above, and deliberately kept apart from it: this feeds
 * the per-minute movement trace (vehicles.ts deriveTrace), never the
 * 5-minute advanced_n/stalled_n signal archiveVehicleMetric feeds. Same key/
 * body-shape convention — one object per poll, keyed on the tick wall-clock
 * so an overlapping/retried run overwrites rather than duplicates — but its
 * own archive/trace/ prefix so this additive stream can never collide with,
 * or be mistaken for, archive/vehicles/. `rows` is a full snapshot from
 * deriveTrace (one row per in-service trip, every poll, not a delta), so
 * most minutes write ~700 rows, not a small or empty array.
 * `freshFeeds` mirrors archiveVehicleMetric's: which line-group feeds decoded
 * this poll, so the offline loader can tell a real empty poll from a gap.
 */
export async function archiveTraceRows(
  bucket: R2Bucket,
  rows: TraceRow[],
  freshFeeds: string[],
  observedAt: number,
  scheduledAt: number,
): Promise<void> {
  // Keyed on the SCHEDULED second, not observedAt. The trace step runs before
  // any compare-and-swap winner check, so a retried or overlapping invocation
  // for the same cron minute would otherwise land on a different observedAt
  // second and write a SECOND object rather than overwriting the first —
  // duplicating every row in it. Downstream that is double-counted arrivals and
  // silently wrong traversal times, in a brand-new archive with no history to
  // cross-check against. The scheduled second is stable across retries, so a
  // rerun overwrites. observed_at stays in the body: it is the real execution
  // time, and the per-vehicle feed timestamps on each row are finer still.
  const key = `archive/trace/${utcDate(scheduledAt)}/${scheduledAt}.json`;
  await bucket.put(
    key,
    JSON.stringify({
      observed_at: observedAt,
      scheduled_at: scheduledAt,
      fresh_feeds: freshFeeds,
      rows,
    }),
    { httpMetadata: { contentType: 'application/json' } },
  );
}

// --- internal helpers ---

function extractEntities(payload: unknown): unknown[] | null {
  if (!payload || typeof payload !== 'object') return null;
  const candidate = (payload as { entity?: unknown }).entity;
  return Array.isArray(candidate) ? candidate : null;
}

function parseAlertEntity(entity: unknown): { id: string; updatedAt: number } | null {
  if (!entity || typeof entity !== 'object') return null;
  const id = (entity as { id?: unknown }).id;
  if (typeof id !== 'string') return null;
  const inner = (entity as { alert?: unknown }).alert;
  if (!inner || typeof inner !== 'object') return null;
  const mercury = (inner as { 'transit_realtime.mercury_alert'?: unknown })[
    'transit_realtime.mercury_alert'
  ];
  if (!mercury || typeof mercury !== 'object') return null;
  const updatedAt = (mercury as { updated_at?: unknown }).updated_at;
  if (typeof updatedAt !== 'number') return null;
  return { id, updatedAt };
}
