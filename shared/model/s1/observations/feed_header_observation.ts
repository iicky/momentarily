/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

/**
 * The FeedHeader of one GTFS-RT payload. Raw only.
 */
export interface FeedHeaderObservation {
  fetch_id: string;
  gtfs_realtime_version: string | null;
  header_timestamp: number | null;
  incrementality: number | null;
  lineage: RawLineage;
  nyct_version: string | null;
  /**
   * UTC epoch seconds.
   */
  observed_at: number;
  source: 'ace' | 'bdfm' | 'g' | 'jz' | 'nqrw' | 'l' | 'numbered' | 'si';
  /**
   * route_id -> end of the replacement window, as published.
   */
  trip_replacement_periods: {
    [k: string]: number | null | undefined;
  };
}
/**
 * Decoded from a stored raw payload.
 */
export interface RawLineage {
  kind: 'raw';
  /**
   * The tick bundle or blob holding the bytes.
   */
  payload_key: string;
  /**
   * Bundle member; null for a blob.
   */
  payload_member: number | null;
  /**
   * Lowercase sha256 hex.
   */
  payload_sha256: string;
}
