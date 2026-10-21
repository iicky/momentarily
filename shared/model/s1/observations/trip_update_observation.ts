/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

/**
 * One TripUpdate entity. Arrival and departure stay separate; nothing is
 * truncated or filtered. Raw only: no legacy archive keeps trip updates.
 */
export interface TripUpdateObservation {
  /**
   * Seconds, signed: negative is early.
   */
  delay: number | null;
  entity_id: string;
  /**
   * Position of the entity in the feed.
   */
  entity_index: number;
  fetch_id: string;
  lineage: RawLineage;
  /**
   * UTC epoch seconds.
   */
  observed_at: number;
  source: 'ace' | 'bdfm' | 'g' | 'jz' | 'nqrw' | 'l' | 'numbered' | 'si';
  stop_times: StopTimePrediction[];
  trip: TripDescriptorObs;
  trip_update_timestamp: number | null;
  unmodeled_fields: string[];
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
export interface StopTimePrediction {
  /**
   * Seconds, signed: negative is early.
   */
  arrival_delay: number | null;
  arrival_time: number | null;
  arrival_uncertainty: number | null;
  /**
   * Seconds, signed: negative is early.
   */
  departure_delay: number | null;
  departure_occupancy_status: number | null;
  departure_time: number | null;
  departure_uncertainty: number | null;
  /**
   * Position in the trip's stop_time_update list.
   */
  index: number;
  nyct_actual_track: string | null;
  nyct_scheduled_track: string | null;
  schedule_relationship: number | null;
  stop_id: string | null;
  stop_sequence: number | null;
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
/**
 * A measurement that was expected but could not be taken, and why.
 */
export interface Unknown {
  reason: string;
  unknown: true;
}
