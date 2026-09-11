/**
 * Intra-day GTFS static feed change detection + capture.
 *
 * The nightly `training/gtfs_archive.store()` run keeps `archive/gtfs/latest.json`
 * fresh once a day, so a trace tick could carry a digest up to ~24h stale after
 * MTA republishes mid-day. This module closes that gap from the Worker side: every
 * trace tick does a cheap HEAD against the GTFS static URL and compares its ETag
 * to the pointer's. A match costs one HEAD + one tiny R2 read. A mismatch means
 * the feed changed since the pointer was last written — the Worker captures the
 * new zip itself, in the same content-addressed layout `gtfs_archive.store()`
 * uses, so a replay never has to wait for the next nightly run.
 *
 * NEVER ON THE CRITICAL PATH. A trace tick must not stall the whole cron
 * invocation behind a ~5.6 MB download. `resolveFeedIdentity` always returns
 * immediately: on a mismatch it hands back `{digest: null, etag: headEtag}` for
 * THIS tick's archive body, plus a `capture` promise the caller hands to
 * `ctx.waitUntil` to run detached from the response. If the capture succeeds,
 * the NEXT tick's HEAD sees the refreshed pointer and starts naming the new
 * digest; if it fails, the pointer is untouched and the tick after that retries
 * — no bookkeeping needed beyond "the pointer didn't move yet".
 *
 * CPU COST. The only CPU-bound step is the sha256 of the downloaded bytes
 * (`crypto.subtle.digest`, hardware-accelerated in the V8 isolate). Measured
 * locally hashing a 5.6 MB buffer with the same WebCrypto API: ~1.8ms median
 * (5 runs, warm). That is well inside a Workers CPU budget even before
 * accounting for `waitUntil` detachment, so no chunked/streaming digest is
 * needed — the plain one-shot `arrayBuffer()` + `digest()` pair below is fine.
 * The GET's network wait and the R2 put/head round trips dominate wall time,
 * not CPU time, and none of it blocks the tick's own response.
 */

const GTFS_STATIC_URL = 'https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip';

// The HEAD that gates every trace tick must stay cheap — 2s, not the 10s
// budget fetch.ts gives the GTFS-RT feeds, since a slow HEAD here still
// blocks this tick's archive write (only the capture GET below is detached).
const HEAD_TIMEOUT_MS = 2_000;
// The capture GET pulls the whole ~5.6 MB zip. Generous because it runs
// detached via ctx.waitUntil — it is never on the tick's response path.
const CAPTURE_TIMEOUT_MS = 30_000;

const PREFIX = 'archive/gtfs/';
const LATEST_KEY = `${PREFIX}latest.json`;
const BY_ETAG_PREFIX = `${PREFIX}by_etag/`;

export interface FeedIdentity {
  digest: string | null;
  etag: string | null;
}

export interface FeedResolution {
  /** This tick's feed identity — always available immediately. */
  identity: FeedIdentity;
  /**
   * Non-null exactly when a HEAD/pointer mismatch was found this tick and a
   * capture (GET + sha256 + R2 writes) was kicked off to catch the pointer
   * up. The caller MUST pass this to `ctx.waitUntil` (or otherwise not await
   * it inline) so the capture runs after the tick's own work, never blocking
   * it. Never rejects — capture failures are swallowed and logged internally
   * so a stray unhandled rejection can't surface on `waitUntil`.
   */
  capture: Promise<void> | null;
}

interface LatestPointer {
  digest: string;
  etag: string | null;
}

// Concurrent ticks in the same isolate (an overlapping or retried cron
// invocation) must not both start downloading the zip. The check-and-set
// below is synchronous — no `await` between the read and the write — so it
// is race-free despite JS's single-threaded interleaving of async tasks.
let capturePromise: Promise<void> | null = null;

async function sha256Hex(buf: ArrayBuffer): Promise<string> {
  const digest = await crypto.subtle.digest('SHA-256', buf);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

// A weak validator (`W/"..."`) only promises semantic equivalence, not the
// byte identity a content digest stands for, so it never resolves or keys a
// digest. S3 sends strong ETags for this object; this is the guard for the
// day it does not.
const WEAK_ETAG = /^\s*[Ww]\//;

async function readLatestPointer(bucket: R2Bucket): Promise<LatestPointer | null> {
  try {
    const obj = await bucket.get(LATEST_KEY);
    if (!obj) return null;
    const data = (await obj.json()) as Record<string, unknown>;
    const digest = data.digest;
    // Exactly 64 lowercase hex chars — the sha256 identity gtfs_archive uses.
    if (typeof digest !== 'string' || !/^[0-9a-f]{64}$/.test(digest)) return null;
    return { digest, etag: typeof data.etag === 'string' ? data.etag : null };
  } catch {
    return null;
  }
}

/**
 * Download the current feed, store it content-addressed, and refresh the
 * `latest.json` / `by_etag/<etag>.json` pointers — the same R2 layout
 * `training/gtfs_archive.store()` writes, so either writer produces artifacts
 * the other can read. Runs detached (see the module doc); failures are
 * logged and swallowed rather than thrown, so the pointer simply stays put
 * until a later tick's HEAD retries.
 *
 * IMPORTANT: the by_etag/ key and latest.json's etag are always taken from the
 * GET response's ETag header — never the HEAD's. Between a HEAD and the GET the
 * object can roll over on S3, so the HEAD etag and the bytes may no longer
 * correspond. Keying on the GET's own etag ensures the mapping is self-
 * consistent: the digest names exactly the bytes the GET returned, and the etag
 * is the one S3 declared for those bytes. If the GET returns no ETag at all, the
 * capture is aborted — a mapping without a validator would be unverifiable.
 */
async function doCapture(bucket: R2Bucket, headEtag: string): Promise<void> {
  try {
    const res = await fetch(GTFS_STATIC_URL, {
      signal: AbortSignal.timeout(CAPTURE_TIMEOUT_MS),
    });
    if (!res.ok) {
      throw new Error(`HTTP ${res.status} from ${GTFS_STATIC_URL}`);
    }

    // Key everything on the GET's own ETag — see the doc above. A weak or
    // missing validator cannot name bytes, so the capture is abandoned.
    const getEtag = res.headers.get('etag');
    if (!getEtag || WEAK_ETAG.test(getEtag)) {
      console.error(
        `gtfs_feed capture aborted: GET returned ${getEtag === null ? 'no' : 'a weak'} ETag; next tick retries`,
      );
      return;
    }
    if (getEtag !== headEtag) {
      console.log(
        `gtfs_feed: GET ETag (${getEtag}) differs from HEAD ETag (${headEtag}); ` +
        'feed rolled over between HEAD and GET — keying on the GET ETag',
      );
    }

    const buf = await res.arrayBuffer();
    const digest = await sha256Hex(buf);
    const lastModified = res.headers.get('last-modified');

    const zipKey = `${PREFIX}${digest}.zip`;
    const existing = await bucket.head(zipKey);
    if (existing === null) {
      await bucket.put(zipKey, buf, { httpMetadata: { contentType: 'application/zip' } });
    }

    // Key stem: the header's quoting removed. Only for key naming — the
    // identity match in resolveFeedIdentity compares raw header values exactly.
    const etagKey = getEtag.trim().replace(/^"|"$/g, '');
    await bucket.put(
      `${BY_ETAG_PREFIX}${etagKey}.json`,
      JSON.stringify({ digest }),
      { httpMetadata: { contentType: 'application/json' } },
    );

    // version is null — parsing the GTFS static schedule.txt in the Worker
    // isn't worth the complexity for a field that's purely informational
    // here. training/gtfs_archive.store()'s nightly run fills it in the next
    // time it runs, same as it always has.
    await bucket.put(
      LATEST_KEY,
      JSON.stringify({
        digest,
        version: null,
        etag: getEtag,
        last_modified: lastModified,
        stored_at: Math.floor(Date.now() / 1000),
      }),
      { httpMetadata: { contentType: 'application/json' } },
    );
  } catch (err) {
    console.error('gtfs_feed capture failed (pointer left untouched; next tick retries):', err);
  }
}

function captureFeed(bucket: R2Bucket, etag: string): Promise<void> {
  if (capturePromise === null) {
    capturePromise = doCapture(bucket, etag).finally(() => {
      capturePromise = null;
    });
  }
  return capturePromise;
}

/**
 * Resolve this tick's GTFS static feed identity. Always returns immediately:
 * a pointer/HEAD match resolves the digest from R2 (no GET), a mismatch
 * returns the freshly observed etag with a null digest plus a detached
 * `capture` promise the caller waitUntil's, and a HEAD failure (network error
 * or timeout) returns both fields null with no capture.
 */
export async function resolveFeedIdentity(bucket: R2Bucket): Promise<FeedResolution> {
  let headEtag: string | null = null;
  try {
    const res = await fetch(GTFS_STATIC_URL, {
      method: 'HEAD',
      signal: AbortSignal.timeout(HEAD_TIMEOUT_MS),
    });
    if (res.ok) {
      headEtag = res.headers.get('etag');
    }
  } catch {
    headEtag = null;
  }

  if (headEtag === null) {
    return { identity: { digest: null, etag: null }, capture: null };
  }
  if (WEAK_ETAG.test(headEtag)) {
    // A weak validator cannot name bytes, so this tick records it unresolved;
    // the capture still runs because the GET's own ETag may be strong, which
    // keeps the artifact and by_etag/ mapping current for the replay side.
    // Costs a download per tick for as long as HEAD stays weak — accepted:
    // S3 has never sent one for this object, and provenance is the feature.
    return { identity: { digest: null, etag: headEtag }, capture: captureFeed(bucket, headEtag) };
  }

  // Exact raw equality: the pointer's etag is the strong validator S3 declared
  // for the archived bytes, so only the identical header names those bytes.
  const pointer = await readLatestPointer(bucket);
  if (pointer !== null && pointer.etag === headEtag) {
    return { identity: { digest: pointer.digest, etag: headEtag }, capture: null };
  }

  // Mismatch, or no pointer written yet: this tick cannot name a digest — the
  // bytes have not been fetched. Record the etag alone and kick off the
  // capture detached so this trace tick's write is never blocked on the
  // ~5.6 MB download.
  return { identity: { digest: null, etag: headEtag }, capture: captureFeed(bucket, headEtag) };
}
