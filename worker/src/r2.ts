/**
 * Conditional R2 writes for read-modify-write state objects.
 *
 * The cron Worker reads a state object, mutates it across the tick, and writes
 * it back. Overlapping or retried scheduled invocations can interleave those
 * steps and clobber each other's updates. Capturing the etag at read time and
 * writing with `onlyIf` turns the write into compare-and-swap: a stale write
 * is rejected rather than silently overwriting a newer one. See momentarily-j0c.
 */

export interface VersionedRead<T> {
  state: T;
  /** etag observed at read time, or null if the object did not exist. */
  etag: string | null;
}

/**
 * Put `body` only if the object is unchanged since it was read at `etag` — or
 * still absent, when `etag` is null. Returns true on success, false when a
 * concurrent writer won the race (the caller should yield, not retry blindly).
 */
export async function conditionalPut(
  bucket: R2Bucket,
  key: string,
  body: string,
  etag: string | null,
  httpMetadata: R2HTTPMetadata,
): Promise<boolean> {
  const onlyIf: R2Conditional | Headers =
    etag === null ? new Headers({ 'If-None-Match': '*' }) : { etagMatches: etag };
  const result = await bucket.put(key, body, { httpMetadata, onlyIf });
  return result !== null;
}
