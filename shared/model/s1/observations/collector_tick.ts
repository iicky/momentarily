/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

/**
 * One Worker invocation. No tick for a minute means the collector did
 * not run; a failed source shows as a fetch with its outcome.
 */
export interface CollectorTick {
  collector: CollectorBuild;
  fetches: FeedFetch[];
  /**
   * UTC epoch milliseconds.
   */
  finished_at_ms: number;
  /**
   * What this tick was due to fetch.
   */
  planned_sources: (
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
    | 'entrances'
  )[];
  /**
   * UTC epoch seconds.
   */
  scheduled_at: number;
  schema: 'observations/s1';
  /**
   * UTC epoch milliseconds.
   */
  started_at_ms: number;
  /**
   * f'{scheduled_at}-{started_at_ms}'; retries keep both.
   */
  tick_id: string;
}
export interface CollectorBuild {
  /**
   * null when the build did not say.
   */
  code_dirty: boolean | null;
  code_sha: string;
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
   * The cron minute this attempt served.
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
