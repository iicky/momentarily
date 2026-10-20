/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

/**
 * One VehiclePosition entity. A placeholder for an undispatched trip is
 * recorded as published: no status, future timestamp. Legacy rows come
 * from ``archive/trace``.
 */
export interface VehicleObservation {
  congestion_level: number | Unknown | null;
  /**
   * 0 INCOMING_AT, 1 STOPPED_AT, 2 IN_TRANSIT_TO; null when omitted.
   */
  current_status: (0 | 1 | 2) | Unknown | null;
  current_stop_sequence: number | Unknown | null;
  entity_id: string | Unknown;
  /**
   * Position of the entity in the feed.
   */
  entity_index: number | Unknown;
  /**
   * The FeedFetch decoded; Unknown on legacy rows.
   */
  fetch_id: string | Unknown;
  lineage: RawLineage | LegacyLineage;
  /**
   * UTC epoch seconds.
   */
  observed_at: number;
  source: ('ace' | 'bdfm' | 'g' | 'jz' | 'nqrw' | 'l' | 'numbered' | 'si') | Unknown;
  /**
   * As published, realtime-only ids kept.
   */
  stop_id: string | null;
  /**
   * null when the vehicle has no trip.
   */
  trip: TripDescriptorObs | null;
  /**
   * Field paths present but not decoded, e.g. '2' (position).
   */
  unmodeled_fields: string[] | Unknown;
  /**
   * Verbatim; may be in the future.
   */
  vehicle_timestamp: number | null;
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
/**
 * A GTFS-RT TripDescriptor with the NYCT extension, as published.
 * ``Unknown`` only on legacy rows (the trace kept trip_id alone).
 */
export interface TripDescriptorObs {
  direction_id: (0 | 1) | Unknown | null;
  nyct_direction: (1 | 2 | 3 | 4) | Unknown | null;
  /**
   * null when the extension omits it; never read as false.
   */
  nyct_is_assigned: boolean | Unknown | null;
  nyct_train_id: string | Unknown | null;
  /**
   * As published: '6X', 'SS', never folded.
   */
  route_id: string | Unknown | null;
  schedule_relationship: number | Unknown | null;
  /**
   * Published 'YYYYMMDD'; not a ServiceDate.
   */
  start_date: string | Unknown | null;
  /**
   * Published 'HH:MM:SS'; may exceed 24h.
   */
  start_time: string | Unknown | null;
  trip_id: string | null;
}
