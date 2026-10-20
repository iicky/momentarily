/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

/**
 * Member 0 of a GTFS-RT tick bundle: the tick's GTFS-RT fetches, so the
 * bundle describes itself when the tick record is lost.
 */
export interface TickBundleIndex {
  /**
   * Every GTFS-RT attempt of the tick, stored or not.
   */
  fetches: FeedFetch[];
  /**
   * UTC epoch seconds.
   */
  scheduled_at: number;
  schema: 'observations/s1';
  /**
   * UTC epoch milliseconds.
   */
  started_at_ms: number;
  tick_id: string;
}
/**
 * One HTTP attempt against one source, whatever its outcome.
 */
export interface FeedFetch {
  /**
   * Decoded body length; null when no body was read.
   */
  body_bytes: number | null;
  /**
   * null when no headers arrived.
   */
  content_type: string | null;
  /**
   * Top-level entities (GTFS-RT, alerts) or records (ENE, Socrata).
   */
  entity_count: number | null;
  /**
   * f'{source}@{started_at_ms}'; unique per attempt.
   */
  fetch_id: string;
  /**
   * null exactly on timeout.
   */
  finished_at_ms: number | null;
  /**
   * The payload's own build time (GTFS-RT or alerts header).
   */
  header_timestamp: number | null;
  /**
   * The response Date header.
   */
  http_date: number | null;
  http_etag: string | null;
  http_last_modified: number | null;
  /**
   * null when no response headers arrived. A timeout or network error after 2xx headers keeps the status that arrived.
   */
  http_status: number | null;
  /**
   * HEAD only for gtfs_static.
   */
  method: 'GET' | 'HEAD';
  outcome: 'ok' | 'empty' | 'unparseable' | 'http_error' | 'timeout' | 'network_error';
  /**
   * R2 key holding the bytes; null if not stored.
   */
  payload_key: string | null;
  /**
   * Member index inside a tick bundle (0 is the index); null for blobs.
   */
  payload_member: number | null;
  /**
   * sha256 of the decoded body bytes; null when no body was read.
   */
  payload_sha256: string | null;
  /**
   * floor(event.scheduledTime / 1000): the invocation's dispatch instant, not minute-aligned.
   */
  scheduled_at: number;
  source:
    | 'ace'
    | 'bdfm'
    | 'g'
    | 'jz'
    | 'nqrw'
    | 'l'
    | 'numbered'
    | 'si'
    | 'alerts'
    | 'ene_current'
    | 'ene_upcoming'
    | 'ene_equipments'
    | 'gtfs_static'
    | 'stations'
    | 'entrances';
  /**
   * UTC epoch milliseconds.
   */
  started_at_ms: number;
  /**
   * Why a body that was read was not stored; set exactly when payload_key is null.
   */
  store_error: string | null;
  url: string;
}
