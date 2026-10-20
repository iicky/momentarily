/**
 * Raw-observation recorder (schema observations/s1, docs/model-observations.md
 * §3.1, §3.3 and §4). Runs beside the live pipeline in every cron invocation
 * and keeps what that pipeline throws away: the untouched bytes of every
 * upstream fetch, plus one `CollectorTick` describing every attempt.
 *
 * Per invocation it writes:
 *   - one GTFS-RT tick bundle (`raw/gtfsrt/...bin`): a gzip member holding the
 *     `TickBundleIndex` JSON, then one gzip member per GTFS-RT body read. It is
 *     PUT before the pipeline decodes a single feed, so a decode failure cannot
 *     lose bytes;
 *   - a gzip blob (`raw/blob/<day>/<sha256>.json.gz`) per JSON body (alerts,
 *     ENE, Socrata) whose digest is new that UTC day;
 *   - the `CollectorTick` (`ticks/...json`), from `finish()`, which the caller
 *     runs in a `finally` so it lands whatever the pipeline did, alpha CAS
 *     included.
 *
 * Blob dedup is a HEAD on the content-addressed key, then a PUT on a miss. A
 * HEAD is one Class B op ($0.36/M): the minute-by-minute alerts poll costs
 * about 43k a month, under two cents, and a hit skips the gzip. A per-day
 * digest index object would cost the same Class B read plus a Class A write
 * ($4.50/M) on every change, and needs a CAS against overlapping invocations;
 * the content-addressed key needs neither, since two racing PUTs of one key
 * write the same bytes.
 *
 * A failed bundle or blob PUT never stops the tick: the affected fetches get a
 * `store_error`, the tick record is still written, and the caller's
 * write-failure counter (archive/health/) goes up.
 *
 * Records are built only through the types generated from the exported JSON
 * Schema; the Worker does not validate them at runtime (ajv compiles with
 * `new Function`, which Workers block). worker/test/recorder.test.ts does.
 */

import type { CollectorTick, FeedFetch } from '../../shared/model/s1/observations/collector_tick';
import type { TickBundleIndex } from '../../shared/model/s1/observations/tick_bundle_index';
import { gzip } from './archive';
import { CODE_DIRTY, CODE_SHA } from './buildinfo';
import type { FetchObserver, JsonParse, RawFetch } from './fetch';
import { blobKey, bundleKey, tickKey, utcDay } from './observation_keys';

export type SourceId = FeedFetch['source'];

const SCHEMA = 'observations/s1';

const GTFSRT_SOURCES: Partial<Record<SourceId, true>> = {
  ace: true, bdfm: true, g: true, jz: true, nqrw: true, l: true, numbered: true, si: true,
};

interface Recorded {
  fetch: FeedFetch;
  /** The body until it is stored; dropped afterwards to free the bytes. */
  body: Uint8Array | null;
  /** Digest (and, for a blob, storage) done; never rejects. */
  ready: Promise<void>;
}

interface Parsed {
  outcome: 'ok' | 'empty' | 'unparseable';
  entityCount: number | null;
  headerTimestamp: number | null;
}

// --- GTFS-RT wire scan ---------------------------------------------------------

const WIRE_VARINT = 0;
const WIRE_I64 = 1;
const WIRE_LEN = 2;
const WIRE_I32 = 5;

type FieldVisitor = (field: number, wire: number, value: number, start: number, end: number) => boolean;

/**
 * Walk the protobuf fields in `buf[from, to)`, calling `visit` with each
 * field's number, wire type, varint value and (for a length-delimited field)
 * its payload bounds. False when the bytes are not well-formed protobuf at
 * this level, or `visit` rejects a field.
 */
function walkFields(buf: Uint8Array, from: number, to: number, visit: FieldVisitor): boolean {
  let p = from;
  const varint = (): number | null => {
    let result = 0;
    for (let shift = 0; shift < 70; shift += 7) {
      if (p >= to) return null;
      const b = buf[p++]!;
      result += (b & 0x7f) * 2 ** shift;
      if ((b & 0x80) === 0) return result;
    }
    return null;
  };
  while (p < to) {
    const tag = varint();
    if (tag === null) return false;
    const field = Math.floor(tag / 8);
    const wire = tag % 8;
    if (field === 0) return false;
    let value = 0;
    let start = p;
    switch (wire) {
      case WIRE_VARINT: {
        const v = varint();
        if (v === null) return false;
        value = v;
        break;
      }
      case WIRE_I64:
        p += 8;
        break;
      case WIRE_LEN: {
        const n = varint();
        if (n === null) return false;
        start = p;
        p += n;
        break;
      }
      case WIRE_I32:
        p += 4;
        break;
      default:
        return false;
    }
    if (p > to || !visit(field, wire, value, start, p)) return false;
  }
  return true;
}

/**
 * Entity count and header timestamp of a GTFS-RT FeedMessage, or null when
 * the bytes are not one. Checks the top level and the FeedHeader only: an
 * entity's inner bytes are left to the offline decoder, which keeps this
 * scan cheap and total (it never throws) ahead of the bundle PUT.
 */
export function scanFeedMessage(buf: Uint8Array): { entities: number; headerTimestamp: number | null } | null {
  let entities = 0;
  let headerTimestamp: number | null = null;
  const ok = walkFields(buf, 0, buf.length, (field, wire, _value, start, end) => {
    if (field === 2) {
      entities += 1;
      return wire === WIRE_LEN;
    }
    if (field !== 1) return true;
    // FeedHeader: field 3 is the uint64 build timestamp.
    return wire === WIRE_LEN && walkFields(buf, start, end, (f, w, v) => {
      if (f !== 3) return true;
      headerTimestamp = v;
      return w === WIRE_VARINT;
    });
  });
  return ok ? { entities, headerTimestamp } : null;
}

function parseGtfsRt(body: Uint8Array): Parsed {
  const scan = scanFeedMessage(body);
  if (scan === null) return { outcome: 'unparseable', entityCount: null, headerTimestamp: null };
  return {
    outcome: scan.entities > 0 ? 'ok' : 'empty',
    entityCount: scan.entities,
    headerTimestamp: scan.headerTimestamp,
  };
}

// --- JSON sources -------------------------------------------------------------

const UNPARSEABLE: Parsed = { outcome: 'unparseable', entityCount: null, headerTimestamp: null };

function epochSeconds(value: unknown): number | null {
  if (typeof value === 'number') return Number.isSafeInteger(value) && value >= 0 ? value : null;
  if (typeof value === 'string' && /^\d{1,15}$/.test(value)) return Number(value);
  return null;
}

/**
 * The alerts feed is a GTFS-RT FeedMessage in JSON: an object whose `entity`
 * array (omitted when empty) holds the alerts. ENE and Socrata feeds are
 * top-level arrays of records. Any other shape is unparseable.
 */
function parseJsonSource(source: SourceId, parse: JsonParse): Parsed {
  if (!parse.ok) return UNPARSEABLE;
  const value = parse.value;
  let count: number;
  let headerTimestamp: number | null = null;
  if (source === 'alerts') {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) return UNPARSEABLE;
    const feed = value as { entity?: unknown; header?: { timestamp?: unknown } | null };
    if (feed.entity === undefined) count = 0;
    else if (Array.isArray(feed.entity)) count = feed.entity.length;
    else return UNPARSEABLE;
    headerTimestamp = epochSeconds(feed.header?.timestamp);
  } else {
    if (!Array.isArray(value)) return UNPARSEABLE;
    count = value.length;
  }
  return { outcome: count > 0 ? 'ok' : 'empty', entityCount: count, headerTimestamp };
}

// --- the recorder ---------------------------------------------------------------

function httpInstant(header: string | null | undefined): number | null {
  if (!header) return null;
  const ms = Date.parse(header);
  return Number.isNaN(ms) ? null : Math.floor(ms / 1000);
}

async function sha256Hex(bytes: Uint8Array): Promise<string> {
  const digest = new Uint8Array(await crypto.subtle.digest('SHA-256', bytes));
  return Array.from(digest, (b) => b.toString(16).padStart(2, '0')).join('');
}

function concat(parts: Uint8Array[]): Uint8Array {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let offset = 0;
  for (const part of parts) {
    out.set(part, offset);
    offset += part.length;
  }
  return out;
}

export class Recorder {
  private readonly planned: SourceId[] = [];
  private readonly recorded: Recorded[] = [];
  /** Every attempt started through `attempt`, settled or not; never rejects. */
  private readonly inflight: Promise<void>[] = [];
  private bundle: Promise<void> | null = null;
  private tick: Promise<void> | null = null;
  /** Set once the tick record is being assembled; later reports are dropped. */
  private closed = false;

  /**
   * @param scheduledAt  floor(event.scheduledTime / 1000): the instant Cloudflare
   *                     dispatched this invocation (epoch seconds), seconds past
   *                     the minute and not the cron minute itself
   * @param startedAtMs  when the invocation started, before any fetch
   * @param failWrite    the tick's write-failure counter (archive/health/)
   */
  constructor(
    private readonly bucket: R2Bucket,
    readonly scheduledAt: number,
    readonly startedAtMs: number,
    private readonly failWrite: (kind: string) => void,
  ) {}

  get tickId(): string {
    return `${this.scheduledAt}-${this.startedAtMs}`;
  }

  /**
   * Plan `source` for this tick and run its fetch with the recorder's
   * observer. Call it when the tick decides to fetch, so `planned_sources`
   * lists what was due even when the attempt never reports back. The returned
   * promise is the fetch's own; `finish` waits for it to settle, so an
   * attempt still in flight when the tick throws is still recorded.
   */
  attempt<T>(source: SourceId, run: (observe: FetchObserver) => Promise<T>): Promise<T> {
    if (!this.planned.includes(source)) this.planned.push(source);
    const pending = run((raw, json) => this.record(source, raw, json));
    this.inflight.push(pending.then(() => undefined, () => undefined));
    return pending;
  }

  private record(source: SourceId, raw: RawFetch, json?: JsonParse): void {
    if (this.closed) {
      // The tick record is being written; an attempt it cannot list is dropped
      // rather than left pointing at bytes nothing references.
      console.warn(`recorder: ${source} reported after the tick record; not recorded`);
      return;
    }
    const bundled = GTFSRT_SOURCES[source] === true;
    // Decided now, not after the digest: storeBundle snapshots its members
    // synchronously, so a GTFS-RT body reported after that call has no member.
    const late = bundled && this.bundle !== null;
    const body = raw.body;
    let parsed: Parsed | null = null;
    if (body !== null) {
      parsed = bundled ? parseGtfsRt(body) : parseJsonSource(source, json ?? { ok: false });
    }
    const status = raw.response?.status ?? null;
    const headers = raw.response?.headers;
    let outcome: FeedFetch['outcome'];
    if (raw.failure !== null) outcome = raw.failure;
    else if (status === null || status < 200 || status >= 300) outcome = 'http_error';
    else outcome = parsed?.outcome ?? 'ok'; // a 2xx HEAD reads no body
    const fetch: FeedFetch = {
      fetch_id: `${source}@${raw.startedAtMs}`,
      source,
      url: raw.url,
      method: raw.method,
      scheduled_at: this.scheduledAt,
      started_at_ms: raw.startedAtMs,
      finished_at_ms: raw.finishedAtMs,
      outcome,
      http_status: status,
      content_type: headers?.get('content-type') ?? null,
      http_date: httpInstant(headers?.get('date')),
      http_last_modified: httpInstant(headers?.get('last-modified')),
      http_etag: headers?.get('etag') ?? null,
      body_bytes: body?.length ?? null,
      payload_sha256: null,
      payload_key: null,
      payload_member: null,
      header_timestamp: parsed?.headerTimestamp ?? null,
      entity_count: parsed?.entityCount ?? null,
      store_error: null,
    };
    const entry: Recorded = { fetch, body, ready: Promise.resolve() };
    if (body !== null) {
      entry.ready = (async () => {
        try {
          fetch.payload_sha256 = await sha256Hex(body);
          if (late) fetch.store_error = 'arrived after the tick bundle was written';
          else if (!bundled) await this.storeBlob(entry);
        } catch (err) {
          fetch.store_error ??= `recording failed: ${String(err)}`;
        }
      })();
    }
    this.recorded.push(entry);
  }

  private async storeBlob(entry: Recorded): Promise<void> {
    const { fetch, body } = entry;
    let key = '';
    try {
      // A body that was read always has a finish time; only a timeout has none.
      key = blobKey(utcDay(Math.floor(fetch.finished_at_ms! / 1000)), fetch.payload_sha256!, 'json');
      // A failed HEAD is treated as a miss: rewriting a content-addressed key
      // costs one PUT, skipping it would lose the bytes.
      let present = false;
      try {
        present = (await this.bucket.head(key)) !== null;
      } catch (err) {
        console.warn(`recorder: blob ${key} HEAD failed; writing it anyway:`, err);
      }
      if (!present) {
        await this.bucket.put(key, await gzip(body!), {
          httpMetadata: { contentType: 'application/gzip' },
        });
      }
      fetch.payload_key = key;
    } catch (err) {
      fetch.store_error = `blob write failed: ${String(err)}`;
      this.failWrite('observations_blob');
      console.error(`recorder: blob ${key} write failed:`, err);
    }
    entry.body = null;
  }

  /**
   * PUT the tick's GTFS-RT bundle. Call it once every GTFS-RT fetch has
   * reported and before any is decoded; later calls return the same write.
   */
  storeBundle(): Promise<void> {
    return (this.bundle ??= this.putBundle());
  }

  private async putBundle(): Promise<void> {
    const entries = this.recorded.filter((r) => GTFSRT_SOURCES[r.fetch.source] === true);
    await Promise.all(entries.map((r) => r.ready));
    const key = bundleKey(this.scheduledAt, this.startedAtMs);
    const stored = entries.filter((r) => r.body !== null);
    stored.forEach((r, i) => {
      r.fetch.payload_key = key;
      r.fetch.payload_member = i + 1;
    });
    const index: TickBundleIndex = {
      schema: SCHEMA,
      tick_id: this.tickId,
      scheduled_at: this.scheduledAt,
      started_at_ms: this.startedAtMs,
      fetches: entries.map((r) => r.fetch),
    };
    try {
      const members = await Promise.all([
        gzip(new TextEncoder().encode(JSON.stringify(index))),
        ...stored.map((r) => gzip(r.body!)),
      ]);
      await this.bucket.put(key, concat(members), {
        httpMetadata: { contentType: 'application/octet-stream' },
      });
    } catch (err) {
      for (const r of stored) {
        r.fetch.payload_key = null;
        r.fetch.payload_member = null;
        r.fetch.store_error = `bundle write failed: ${String(err)}`;
      }
      this.failWrite('observations_bundle');
      console.error(`recorder: bundle ${key} write failed:`, err);
    }
    for (const r of stored) r.body = null;
  }

  /**
   * Wait for every attempt still in flight, store whatever is pending, and
   * PUT the `CollectorTick`. Run it in a `finally`; later calls return the
   * same write. Never rejects.
   */
  finish(): Promise<void> {
    return (this.tick ??= this.putTick());
  }

  private async putTick(): Promise<void> {
    // An attempt can only start before finish() is called, so this list is complete.
    await Promise.all(this.inflight);
    this.closed = true;
    await this.storeBundle();
    await Promise.all(this.recorded.map((r) => r.ready));
    const tick: CollectorTick = {
      schema: SCHEMA,
      tick_id: this.tickId,
      scheduled_at: this.scheduledAt,
      started_at_ms: this.startedAtMs,
      finished_at_ms: Date.now(),
      collector: { code_sha: CODE_SHA, code_dirty: CODE_DIRTY },
      planned_sources: [...this.planned],
      fetches: this.recorded.map((r) => r.fetch),
    };
    const key = tickKey(this.scheduledAt, this.startedAtMs);
    try {
      await this.bucket.put(key, JSON.stringify(tick), {
        httpMetadata: { contentType: 'application/json' },
      });
    } catch (err) {
      this.failWrite('observations_tick');
      console.error(`recorder: tick ${key} write failed:`, err);
    }
  }
}
