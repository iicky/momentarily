/**
 * R2 keys the recorder writes under the observations layer (schema
 * observations/s1). Mirrors the formatting half of
 * src/momentarily/model/observations/keys.py for the three keys the Worker
 * produces; the Python module stays the parser of record.
 *
 *   ticks/<YYYY-MM-DD>/<HH>/<scheduled_at>-<started_at_ms>.json      CollectorTick
 *   raw/gtfsrt/<YYYY-MM-DD>/<HH>/<scheduled_at>-<started_at_ms>.bin  tick bundle
 *   raw/blob/<YYYY-MM-DD>/<sha256>.<json|pb|zip>.gz                  deduplicated blob
 *
 * Ticks and bundles are filed under the UTC hour of `scheduled_at` (the cron
 * minute). A blob is filed under the UTC day of the fetch that stored it.
 * tests/fixtures/parity_observation_keys.json pins both implementations to
 * the same strings.
 */

export const OBSERVATIONS_PREFIX = 'model/observations/s1/';

export type BlobExt = 'json' | 'pb' | 'zip';

// Python's datetime cannot hold year 10000; keep the two layouts' domains equal.
const MAX_INSTANT = 253_402_300_800;
const SHA256 = /^[0-9a-f]{64}$/;
const DAY = /^\d{4}-\d{2}-\d{2}$/;

function utcIso(instant: number): string {
  if (!Number.isSafeInteger(instant) || instant < 0 || instant >= MAX_INSTANT) {
    throw new RangeError(`instant ${instant} is out of range`);
  }
  return new Date(instant * 1000).toISOString();
}

/** ISO date of the UTC day containing `instant` (epoch seconds). */
export function utcDay(instant: number): string {
  return utcIso(instant).slice(0, 10);
}

function attemptPath(kind: string, ext: string, scheduledAt: number, startedAtMs: number): string {
  if (!Number.isSafeInteger(startedAtMs) || startedAtMs < 0) {
    throw new RangeError(`started_at_ms ${startedAtMs} is out of range`);
  }
  const iso = utcIso(scheduledAt);
  return (
    `${OBSERVATIONS_PREFIX}${kind}/${iso.slice(0, 10)}/${iso.slice(11, 13)}/`
    + `${scheduledAt}-${startedAtMs}.${ext}`
  );
}

/** The CollectorTick record of one Worker invocation. */
export function tickKey(scheduledAt: number, startedAtMs: number): string {
  return attemptPath('ticks', 'json', scheduledAt, startedAtMs);
}

/** One invocation's GTFS-RT bundle: a gzip index member, then one gzip member per feed. */
export function bundleKey(scheduledAt: number, startedAtMs: number): string {
  return attemptPath('raw/gtfsrt', 'bin', scheduledAt, startedAtMs);
}

/** Gzipped payload bytes, deduplicated within one UTC day. */
export function blobKey(day: string, sha256: string, ext: BlobExt): string {
  if (!DAY.test(day) || utcDay(Date.parse(`${day}T00:00:00Z`) / 1000) !== day) {
    throw new RangeError(`not an ISO UTC date: ${day}`);
  }
  if (!SHA256.test(sha256)) {
    throw new RangeError(`not a lowercase sha256 hex digest: ${sha256}`);
  }
  return `${OBSERVATIONS_PREFIX}raw/blob/${day}/${sha256}.${ext}.gz`;
}
