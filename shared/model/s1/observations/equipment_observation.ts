/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

/**
 * One record of one ENE outage payload, verbatim, with its two published
 * local times parsed. Legacy rows come from ``archive/ene``.
 */
export interface EquipmentObservation {
  equipment_id: string;
  estimated_return: number | Unknown | null;
  /**
   * 'estimatedreturntoservice' as published (ET wall time).
   */
  estimated_return_local: string | null;
  /**
   * Unknown on legacy rows.
   */
  fetch_id: string | Unknown;
  lineage: RawLineage | LegacyLineage;
  /**
   * UTC epoch seconds.
   */
  observed_at: number;
  /**
   * Parsed in America/New_York; Unknown if ambiguous or unparseable.
   */
  outage_start: number | Unknown | null;
  /**
   * 'outagedate' as published (ET wall time).
   */
  outage_start_local: string | null;
  /**
   * The record verbatim.
   */
  record: {
    [k: string]: string | undefined;
  };
  /**
   * Position of the record in the payload.
   */
  record_index: number;
  source: 'ene_current' | 'ene_upcoming';
}
/**
 * A measurement that was expected but could not be taken, and why.
 */
export interface Unknown {
  reason: string;
  unknown: true;
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
/**
 * Read from a legacy archive object. What it lacks is ``Unknown`` on the
 * record; layer 3 labels anything it derives from it as legacy-derived.
 */
export interface LegacyLineage {
  archive:
    | 'archive/trace'
    | 'archive/trip_updates'
    | 'archive/vehicles'
    | 'archive/alerts'
    | 'archive/alerts_liveness'
    | 'archive/ene'
    | 'archive/health'
    | 'v1/predictions';
  kind: 'legacy';
  period: LegacyPeriod;
  /**
   * Position of the row in that object; null when it holds one.
   */
  row_index: number | null;
  /**
   * The legacy object the record came from.
   */
  source_key: string;
}
/**
 * The UTC days from ``start`` for which a legacy archive stands in for
 * the recorder's records. Date-granular: when ``end`` is set, the archive
 * stops at the recorder cutover, 2026-10-21T04:00:00Z, on the day before
 * ``end``.
 */
export interface LegacyPeriod {
  /**
   * Exclusive: the first UTC day the archive no longer covers at all; it stops at the recorder cutover (2026-10-21T04:00:00Z) on the day before. Null for records the recorder does not write.
   */
  end: string | null;
  /**
   * First UTC day the legacy archive covers.
   */
  start: string;
}
