/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

/**
 * The full alert set at one fetch with outcome ok or empty. Comparing
 * consecutive presences bounds when an alert left the feed. Raw only.
 */
export interface AlertPresence {
  /**
   * Sorted by alert_id then digest; [] = the feed said none (outcome empty).
   */
  entries: AlertEntry[];
  fetch_id: string;
  header_timestamp: number | null;
  lineage: RawLineage;
  /**
   * UTC epoch seconds.
   */
  observed_at: number;
  source: 'alerts' | 'numbered';
}
export interface AlertEntry {
  alert_id: string;
  /**
   * sha256 of the entity's canonical JSON.
   */
  content_digest: string;
  updated_at: number | null;
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
