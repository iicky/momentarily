/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

/**
 * What a legacy archive kept about one source's fetch: whether it
 * succeeded, nothing more. The trace, trip-update and vehicle archives list
 * the GTFS-RT feeds that were fresh; ``archive/alerts_liveness`` records
 * success or failure of the alerts fetch; an ``archive/ene`` object exists
 * only for a successful fetch.
 */
export interface LegacyFetchOutcome {
  /**
   * alerts_liveness fetched_at: the last successful fetch's time.
   */
  last_success_at: number | Unknown;
  /**
   * f'{lineage.source_key}#{source}'.
   */
  legacy_fetch_id: string;
  lineage: LegacyLineage;
  /**
   * The legacy record's observed_at.
   */
  observed_at: number;
  /**
   * Kept by archive/trace only.
   */
  scheduled_at: number | Unknown;
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
   * false: failed or stale; the archive kept no cause.
   */
  succeeded: boolean;
}
/**
 * A measurement that was expected but could not be taken, and why.
 */
export interface Unknown {
  reason: string;
  unknown: true;
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
 * The UTC days ``[start, end)`` for which a legacy archive stands in for
 * the recorder's records.
 */
export interface LegacyPeriod {
  /**
   * Exclusive: the first UTC day the recorder wrote the replacing record; null while it has not.
   */
  end: string | null;
  /**
   * First UTC day the legacy archive covers.
   */
  start: string;
}
