/**
 * Write change events to R2 archive.
 *
 * One R2 object per change. Sortable keys let downstream consumers (training
 * runs, calibration notebooks) reconstruct the corpus by listing prefixes.
 *
 *   archive/alerts/YYYY-MM-DD/HHMMSS-<alert_id>.json
 *   archive/ene/YYYY-MM-DD/HHMMSS-<source>.json
 *
 * We dedupe alerts by (alert_id, updated_at) — an alert that persists for hours
 * occupies one R2 object until its `updated_at` changes, not 72 copies of the
 * same payload like the legacy Python collector did.
 *
 * E&E is written as a full snapshot per hourly tick (per feed). Volume is low
 * (~3 PUTs/hour × 24 = 72/day) so no dedupe needed yet.
 */

import type { LastSeen } from './state';

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
  const timePrefix = utcTime(observedAt);
  let written = 0;

  for (const entity of entities) {
    const parsed = parseAlertEntity(entity);
    if (!parsed) continue;
    const { id, updatedAt } = parsed;

    if (lastSeen.alerts[id] === updatedAt) continue;
    lastSeen.alerts[id] = updatedAt;

    const key = `archive/alerts/${datePrefix}/${timePrefix}-${safeKey(id)}.json`;
    await bucket.put(
      key,
      JSON.stringify({ observed_at: observedAt, alert: entity }),
      { httpMetadata: { contentType: 'application/json' } },
    );
    written += 1;
  }
  return written;
}

export async function archiveEneSnapshot(
  bucket: R2Bucket,
  source: string,
  payload: unknown,
  observedAt: number,
): Promise<void> {
  const datePrefix = utcDate(observedAt);
  const timePrefix = utcTime(observedAt);
  const key = `archive/ene/${datePrefix}/${timePrefix}-${safeKey(source)}.json`;
  await bucket.put(
    key,
    JSON.stringify({ observed_at: observedAt, source, payload }),
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
