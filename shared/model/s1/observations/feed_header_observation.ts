/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

/**
 * The FeedHeader of one GTFS-RT payload. Raw only.
 */
export interface FeedHeaderObservation {
  feed_version: string | null;
  fetch_id: string;
  /**
   * Required upstream.
   */
  gtfs_realtime_version: string;
  header_timestamp: number | null;
  /**
   * None: omitted (the proto default is FULL_DATASET, 0).
   */
  incrementality: number | null;
  lineage: RawLineage;
  /**
   * None: the payload has no NYCT extension. An extension with zero periods is a NyctFeedHeaderObs with an empty list.
   */
  nyct_feed_header: NyctFeedHeaderObs | null;
  /**
   * UTC epoch seconds.
   */
  observed_at: number;
  source: 'ace' | 'bdfm' | 'g' | 'jz' | 'nqrw' | 'l' | 'numbered' | 'si';
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
 * The NYCT ``nyct_feed_header`` extension of a FeedHeader, as published.
 */
export interface NyctFeedHeaderObs {
  /**
   * Required upstream.
   */
  nyct_subway_version: string;
  /**
   * Every published period, in published order; a route may repeat. Empty when the extension carries none.
   */
  trip_replacement_periods: TripReplacementPeriodObs[];
}
/**
 * One NYCT ``TripReplacementPeriod``: within the range, the realtime feed
 * replaces every scheduled trip of the route. Both fields are optional
 * upstream; ``None`` means the payload omitted it.
 */
export interface TripReplacementPeriodObs {
  /**
   * None: no TimeRange published. A TimeRange with neither bound is kept as published, distinct from None.
   */
  replacement_period: TimeRangeObs | null;
  route_id: string | null;
}
/**
 * A GTFS-RT TimeRange as published. Each bound is optional upstream:
 * ``None`` means the payload omitted it (an open bound), never "unknown".
 */
export interface TimeRangeObs {
  /**
   * None: omitted, open end.
   */
  end: number | null;
  /**
   * None: omitted, open start.
   */
  start: number | null;
}
