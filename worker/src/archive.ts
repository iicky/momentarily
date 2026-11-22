/**
 * Write change events to R2 archive.
 *
 * One R2 object per change. Sortable keys let downstream consumers (training
 * runs, calibration notebooks) reconstruct the corpus by listing prefixes.
 *
 *   archive/alerts/YYYY-MM-DD/<updated_at>-<alert_id>.json
 *   archive/alerts_liveness/YYYY-MM-DD/<scheduled_at>-<started_at_ms>.json
 *   archive/arrivals/YYYY-MM-DD/<scheduled_at>-<started_at_ms>.json.gz
 *   archive/ene/YYYY-MM-DD/HH0000-<source>.json
 *   archive/health/YYYY-MM-DD/<scheduled_at>-<started_at_ms>.json
 *   archive/trace/YYYY-MM-DD/<scheduled_at>.json
 *   archive/trip_updates/YYYY-MM-DD/<scheduled_at>-<started_at_ms>.json
 *   archive/vehicles/YYYY-MM-DD/<scheduled_at>-<started_at_ms>.json
 *
 * We dedupe alerts by (alert_id, updated_at) — an alert that persists for hours
 * occupies one R2 object until its `updated_at` changes, not 72 copies of the
 * same payload like the legacy Python collector did.
 *
 * Three kinds of key, by what an object stands for:
 *   - per run (`runArchiveKey`): one object for each invocation that writes
 *     it, named by the run's tick id, `<scheduled_at>-<started_at_ms>`, the
 *     same id as the run's CollectorTick. Two runs never share a key, even
 *     two runs of different minutes that start in the same second (a late
 *     cron or alarm beside the next minute's on-time cron), so each run's
 *     record survives. Written create-once.
 *   - per minute (trace): keyed on the scheduled second, create-once, so a
 *     second run of the same minute keeps the first run's object.
 *   - per version or hour (alerts, E&E): deterministic, so a rerun lands on
 *     the same key.
 * The date folder of a per-run key is the UTC day of the run's observed_at
 * (execution wall clock), as it was when these objects were named by
 * observed_at, so date-prefix listing finds them under the same day.
 *
 * Per-run objects written before this layout are named `<observed_at>.json`
 * (`.json.gz` for arrivals), the run's execution second. Readers accept both
 * names; the bodies are unchanged and carry observed_at either way.
 *
 * E&E is written as a full snapshot per hourly tick (per feed). Volume is low
 * (~3 PUTs/hour × 24 = 72/day) so no dedupe needed yet.
 */

import { createOnce } from './r2';
import type { LastSeen } from './state';
import type { ServiceRow } from './trip_updates';
import type { MovementRow, TraceRow } from './vehicles';

const RUN_ID = /^\d+-\d+$/;

/** `runId` itself, after checking it has the tick id's form. */
export function checkRunId(runId: string): string {
  if (!RUN_ID.test(runId)) throw new RangeError(`not a run id: ${runId}`);
  return runId;
}

/**
 * Key of an object one run writes once: `<prefix>/<UTC day of observedAt>/
 * <runId><suffix>`. `runId` is the run's tick id (`Recorder.tickId`,
 * `<scheduled_at>-<started_at_ms>`).
 */
export function runArchiveKey(
  prefix: string,
  observedAt: number,
  runId: string,
  suffix: string,
): string {
  return `${prefix}/${utcDate(observedAt)}/${checkRunId(runId)}${suffix}`;
}

/**
 * Create-once put of a per-run object. Its key names the run, so it can
 * exist already only if this run wrote it; the first write is kept.
 */
export async function putRunObject(
  bucket: R2Bucket,
  key: string,
  body: string | Uint8Array,
  httpMetadata: R2HTTPMetadata,
): Promise<void> {
  if (!(await createOnce(bucket, key, body, httpMetadata))) {
    console.warn(`archive: ${key} already exists; kept`);
  }
}

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
    // retried/overlapping run lands on the same object instead of a duplicate.
    // Create-once: a second run of one minute (or a tick after a lost
    // last_seen write) keeps the version's first observation.
    const key = `archive/alerts/${datePrefix}/${updatedAt}-${safeKey(id)}.json`;
    const created = await createOnce(
      bucket,
      key,
      JSON.stringify({ observed_at: observedAt, alert: entity }),
      { contentType: 'application/json' },
    );
    if (created) written += 1;
  }

  lastSeen.alerts = seen;
  return written;
}

/**
 * This tick's alert-fetch outcome. Two states, not three: a fetch that
 * completes at the HTTP layer is 'success' regardless of what it returns —
 * payload validity is a downstream concern (see archiveNewAlerts /
 * extractEntities below), not a fetch-liveness one. What index.ts's comment
 * calls the "stale fallback" — reusing the prior successful fetch's
 * timestamp instead of stamping observedAt — is not a third outcome; it IS
 * 'fail'. The fallback is exactly what makes 'fail' distinguishable from
 * 'success' in the archived record (see fetched_at below).
 */
export type AlertsFetchOutcome = 'success' | 'fail';

export interface AlertsLiveness {
  outcome: AlertsFetchOutcome;
  /** Epoch of the data this record actually reflects: observedAt on
   * success, or the last known-good fetch's time carried forward on
   * failure — never a fabricated "now" for a tick that fetched nothing. */
  fetched_at: number;
}

/**
 * Map this tick's raw alerts-fetch result to the archived liveness record.
 * Pure and kept apart from archiveAlertsLiveness below so the outcome
 * mapping — including the stale-fallback fetched_at on failure — is
 * testable without an R2 bucket.
 */
export function deriveAlertsLiveness(success: boolean, fetchedAt: number): AlertsLiveness {
  return { outcome: success ? 'success' : 'fail', fetched_at: fetchedAt };
}

/**
 * Archive this tick's alert-fetch liveness — one tiny (~80-byte) object per
 * fetch attempt, independent of archiveNewAlerts above. archiveNewAlerts only
 * writes when an alert's (id, updated_at) version actually changes, so a quiet
 * successful tick and a total feed outage both write zero archive/alerts/
 * objects and are otherwise indistinguishable to an offline reader — this
 * closes that gap.
 *
 * KEYED BY THE RUN (`runArchiveKey`: `<scheduled_at>-<started_at_ms>`), not by
 * the scheduled minute and not by the execution second. Every attempt lands
 * under its own key and BOTH are kept: a failed first attempt and a
 * succeeding retry of one minute, and equally a late run of one minute and an
 * on-time run of the next that start in the same second. The body's
 * observed_at is the execution wall clock (index.ts: `const observedAt =
 * Math.floor(Date.now() / 1000)`).
 *
 * That is deliberate, and it is why this call is never gated on the alpha CAS
 * winner as archiveTripUpdateMetric/archiveVehicleMetric below are: a failed
 * first attempt and a succeeding retry should both be on the record. A
 * reader reconstructing liveness should treat the attempts within a scheduled
 * minute as a set (any success means the alerts were fetched for that minute),
 * not assume a single record. Collapsing them here would silently erase
 * whichever attempt lost, which is exactly the evidence this archive exists to
 * keep.
 */
export async function archiveAlertsLiveness(
  bucket: R2Bucket,
  liveness: AlertsLiveness,
  observedAt: number,
  runId: string,
): Promise<void> {
  await putRunObject(
    bucket,
    runArchiveKey('archive/alerts_liveness', observedAt, runId, '.json'),
    JSON.stringify({ observed_at: observedAt, ...liveness }),
    { contentType: 'application/json' },
  );
}

/**
 * This tick's per-prefix write-failure counts — every archive/state write in
 * index.ts's `scheduled` that currently only console.error's on failure also
 * increments its own key here (see `WriteFailureCounts` callers in index.ts).
 * Plain counts, not booleans: keeps the shape uniform even though today only
 * one write of each kind happens per tick.
 */
export type WriteFailureCounts = Record<string, number>;

/**
 * Archive this tick's write-failure counts — one tiny object per tick,
 * written unconditionally (empty `write_failures` on a clean tick), mirroring
 * archiveAlertsLiveness above. WHY THIS EXISTS: every write it covers already
 * degrades fail-soft (log and move on), which is correct for a single tick
 * but means a PERSISTENT failure on one prefix — say, predictions writes
 * wedged behind a bad object — never fails a tick or pages anyone. The live
 * snapshot stays fresh from the writes that DO succeed, while that one
 * training-input stream quietly stops accruing, invisibly, until an offline
 * consumer notices a gap weeks later. Counting failures here, every tick,
 * turns that silent corpus rot into a queryable time series: a reader sees
 * `write_failures.predictions_write > 0` for consecutive ticks and knows
 * exactly which stream and roughly when it started.
 *
 * Keyed by the run exactly like archiveAlertsLiveness — see that function's
 * comment for why an overlapping/retried tick keeps both records rather than
 * overwriting.
 */
export async function archiveHealth(
  bucket: R2Bucket,
  writeFailures: WriteFailureCounts,
  observedAt: number,
  runId: string,
): Promise<void> {
  await putRunObject(
    bucket,
    runArchiveKey('archive/health', observedAt, runId, '.json'),
    JSON.stringify({ observed_at: observedAt, write_failures: writeFailures }),
    { contentType: 'application/json' },
  );
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
 * R2 free tier; this is ~1 KB/tick, on par with the alerts archive. Keyed by
 * the run (`runArchiveKey`) and written create-once; only the alpha CAS
 * winner writes it. `freshFeeds` records which line-group feeds were decoded
 * this tick so the offline loader can tell a real zero from a missing feed.
 */
export async function archiveTripUpdateMetric(
  bucket: R2Bucket,
  rows: Map<string, ServiceRow>,
  freshFeeds: string[],
  observedAt: number,
  runId: string,
): Promise<void> {
  await putRunObject(
    bucket,
    runArchiveKey('archive/trip_updates', observedAt, runId, '.json'),
    JSON.stringify({
      observed_at: observedAt,
      fresh_feeds: freshFeeds,
      rows: Object.fromEntries(rows),
    }),
    { contentType: 'application/json' },
  );
}

/**
 * Archive the derived per-route vehicle-movement metric — one compact object
 * per tick, mirroring archiveTripUpdateMetric. Same rationale: a handful of
 * ints per route (~1 KB/tick), not the raw protobuf. Keyed by the run and
 * written create-once, like archiveTripUpdateMetric.
 * `freshFeeds` records which line-group feeds decoded so the offline loader can
 * tell a real zero from a missing feed.
 */
export async function archiveVehicleMetric(
  bucket: R2Bucket,
  rows: Map<string, MovementRow>,
  freshFeeds: string[],
  observedAt: number,
  runId: string,
): Promise<void> {
  await putRunObject(
    bucket,
    runArchiveKey('archive/vehicles', observedAt, runId, '.json'),
    JSON.stringify({
      observed_at: observedAt,
      fresh_feeds: freshFeeds,
      rows: Object.fromEntries(rows),
    }),
    { contentType: 'application/json' },
  );
}

/**
 * Archive the per-minute vehicle trace rows — the fine-grained sibling of
 * archiveVehicleMetric above, and deliberately kept apart from it: this feeds
 * the per-minute movement trace (vehicles.ts deriveTrace), never the
 * 5-minute advanced_n/stalled_n signal archiveVehicleMetric feeds. Same
 * body-shape convention, but one object per MINUTE rather than per run: keyed
 * on the tick's scheduled second so an overlapping/retried run of the same
 * minute cannot duplicate its rows — and its
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
  feedDigest: string | null = null,
  feedEtag: string | null = null,
): Promise<void> {
  // Keyed on the SCHEDULED second, not observedAt. The trace step runs before
  // any compare-and-swap winner check, so a retried or overlapping invocation
  // for the same cron minute would otherwise land on a different observedAt
  // second and write a SECOND object rather than overwriting the first —
  // duplicating every row in it. Downstream that is double-counted arrivals and
  // silently wrong traversal times, in a brand-new archive with no history to
  // cross-check against. The scheduled second is stable across retries, so a
  // rerun lands on the same key. The put is create-once (If-None-Match: *): the
  // first run's trace is kept, and a second run of the same minute (an alarm
  // takeover, a late or duplicated cron) cannot replace it, whatever it holds.
  // A takeover runs its minute later than the cron did, so the first trace is
  // also the one nearer the minute it is keyed by. observed_at stays in the
  // body: it is the real execution time, and the per-vehicle feed timestamps
  // on each row are finer still.
  const key = `archive/trace/${utcDate(scheduledAt)}/${scheduledAt}.json`;
  // feed_digest / feed_etag: this tick's GTFS static feed identity from
  // gtfs_feed.resolveFeedIdentity's HEAD+ETag compare against
  // archive/gtfs/latest.json. feed_etag is set whenever the HEAD succeeded —
  // including a mismatch tick where the capture (GET+store) has only just
  // been kicked off and hasn't finished — while feed_digest stays null until
  // a LATER tick's HEAD observes the refreshed pointer. Both null when the
  // HEAD itself failed (network error/timeout) or params are unavailable.
  // Older (pre-change) trace objects omit both fields; new objects always
  // carry them so an explicit null records a resolution failure rather than
  // being indistinguishable from a historical gap.
  const body: Record<string, unknown> = {
    observed_at: observedAt,
    scheduled_at: scheduledAt,
    fresh_feeds: freshFeeds,
    feed_digest: feedDigest,
    feed_etag: feedEtag,
    rows,
  };
  const created = await createOnce(bucket, key, JSON.stringify(body), { contentType: 'application/json' });
  if (!created) console.warn(`trace: ${key} already archived by another run of this minute; kept`);
}

/**
 * Archive this tick's per-minute arrivals sample — one per-stop countdown with
 * only the first ARRIVALS_SAMPLE_KEEP_N upcoming trains (soonest first),
 * fresh_feeds, and expected_feeds. Used for grading the MTA's own countdown
 * ETAs against realized arrivals (training/eta_grade.py's `--source r2`).
 *
 * Gzipped, keep-4: measured against 10 days of real snapshots (a local
 * laptop poller, 2026-09-25..10-04, 10,878 full 6-train
 * snapshots). At the Worker's 1-minute cadence, keep-4 plain JSON runs
 * ~342 MB/day, gzip ~31 MB/day — keep-6 gzip is ~40 MB/day, over the 30 MB/day
 * budget. keep-4 is the smallest tested N (2/3/4/6) that keeps >=90% of the
 * 2-5 and 5-10 min countdown predictions (99.6% / 95.1%) — the horizon
 * eta_grade.py's own per-route/per-hour grade already restricts to (the
 * window "a rider reads walking to the platform") — against keep-2's
 * 92.3%/69.9% and keep-3's 98.2%/87.5%. It only reaches 80.3% of the 10-20 min
 * bin (vs keep-6's 100%), the trade made for the ~10x size cut off keep-6.
 *
 * Keyed by the run (`runArchiveKey`) and written create-once: each run's
 * sample is kept, including two runs of adjacent minutes that start in the
 * same second.
 */
const ARRIVALS_SAMPLE_KEEP_N = 4;

export async function archiveArrivalsSample(
  bucket: R2Bucket,
  observedAt: number,
  runId: string,
  arrivals: Record<string, Array<{ route: string; eta_epoch: number; seconds_away: number; trip_id: string | null }>>,
  freshFeeds: readonly string[],
  expectedFeeds: readonly string[],
): Promise<void> {
  const key = runArchiveKey('archive/arrivals', observedAt, runId, '.json.gz');
  const stops: Record<string, Array<{ route: string; eta_epoch: number; trip_id: string | null }>> = {};

  // Keep only the first ARRIVALS_SAMPLE_KEEP_N arrivals per stop, already
  // sorted by eta_epoch, and omit seconds_away (derivable from eta_epoch and
  // wall-clock time). Sort stop IDs for deterministic key order in the JSON
  // output.
  const sortedStopIds = Object.keys(arrivals).sort();
  for (const stopId of sortedStopIds) {
    const arrivals_list = arrivals[stopId]!;
    stops[stopId] = arrivals_list.slice(0, ARRIVALS_SAMPLE_KEEP_N).map((a) => ({
      route: a.route,
      eta_epoch: a.eta_epoch,
      trip_id: a.trip_id,
    }));
  }

  const body: Record<string, unknown> = {
    schema_version: 1,
    observed_at: observedAt,
    fresh_feeds: [...freshFeeds],
    expected_feeds: [...expectedFeeds],
    stops,
  };

  await putRunObject(
    bucket,
    key,
    await gzip(new TextEncoder().encode(JSON.stringify(body))),
    { contentType: 'application/json', contentEncoding: 'gzip' },
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

/**
 * One gzip member of `bytes` via the Workers runtime's CompressionStream.
 * archiveArrivalsSample is the one archive/ stream that writes gzipped —
 * every other archive/ stream here is small enough that plain JSON was the
 * right trade. The observations recorder gzips its raw payloads with it too.
 */
export async function gzip(bytes: Uint8Array): Promise<Uint8Array> {
  const stream = new CompressionStream('gzip');
  const writer = stream.writable.getWriter();
  void writer.write(bytes);
  void writer.close();

  const chunks: Uint8Array[] = [];
  const reader = stream.readable.getReader();
  let total = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    total += value.length;
  }
  const out = new Uint8Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}
