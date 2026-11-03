/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

/**
 * Observed trains against schedule at one good fetch, per feed and
 * route (§5.2). Evidence for layer 5, never coverage.
 */
export interface SupplyObservation {
  lineage: 'raw' | 'legacy';
  /**
   * Raw: trip updates listing the route; legacy: Unknown.
   */
  listed_trips: number | Unknown;
  /**
   * UTC epoch seconds.
   */
  observed_at: number;
  route_basis: 'scheduled_trip' | 'trip_id_field';
  /**
   * Network route: the matched trip's, else the trip-id field.
   */
  route_id: string;
  /**
   * Layer-1 trips in service at observed_at.
   */
  scheduled_in_service: number;
  source: 'ace' | 'bdfm' | 'g' | 'jz' | 'nqrw' | 'l' | 'numbered' | 'si';
  source_basis: 'published' | 'route_feed_table';
  /**
   * Certain: real vehicles after §2.2, fresh positions.
   */
  vehicles: number;
  /**
   * Uncertain: runs in a stale_tail block (held train or ghost).
   */
  vehicles_stale_tail: number;
}
/**
 * A measurement that was expected but could not be taken, and why.
 */
export interface Unknown {
  reason: string;
  unknown: true;
}
