/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

/**
 * One realtime run and what it is (§2). Scheduled-trip identity and the
 * observed pattern fit are separate results: ``compatible_pattern_ids`` is
 * fitted from observations alone.
 */
export interface TripMatch {
  /**
   * Set exactly when flagged service_date_ambiguous.
   */
  alternative_service_date: string | null;
  /**
   * Patterns whose stop order contains the observed stops; sorted.
   */
  compatible_pattern_ids: string[];
  deviation:
    | (
        | 'consistent'
        | 'skips_only'
        | 'extras_only'
        | 'skips_and_extras'
        | 'off_pattern'
        | 'order_violation'
      )
    | null;
  end_state: 'left_feed' | 'stale_tail' | 'window_end';
  /**
   * UTC epoch seconds.
   */
  first_observed_at: number;
  /**
   * Sorted and distinct.
   */
  flags: (
    | 'deduplicated_by_position'
    | 'duplicate_trip_id'
    | 'joined_split_run'
    | 'service_date_ambiguous'
    | 'start_date_mismatch'
  )[];
  /**
   * Runs with identical positions but a different train_id (§2.3).
   */
  identity_uncertain_with: string[];
  /**
   * UTC epoch seconds.
   */
  last_observed_at: number;
  lineage: 'raw' | 'legacy';
  /**
   * Runs joined into this one (split-run join); [] when not joined.
   */
  member_run_ids: string[];
  /**
   * Why no trip matched; null when matched.
   */
  miss_class: string | null;
  n_observations: number;
  n_old_timestamp_rows: number;
  n_placeholder_rows: number;
  /**
   * For an unmatched run: the trip it nearly matched.
   */
  nearest_scheduled_trip_id: string | null;
  network_build_id: string;
  /**
   * Distinct NYCT train ids seen, sorted; Unknown on legacy runs.
   */
  nyct_train_ids: string[] | Unknown;
  /**
   * Observed minus scheduled at the first common stop.
   */
  offset_s: number | null;
  /**
   * Matched only after adding 24 h to the origin.
   */
  origin_wrapped: boolean;
  outcome: 'matched' | 'unscheduled' | 'unparseable';
  pattern_attribution: 'scheduled_trip' | 'unique_fit' | 'ambiguous' | 'off_pattern';
  /**
   * The matched trip's pattern, or the unique fit.
   */
  pattern_id: string | null;
  realtime_trip_id: string;
  route_basis: 'scheduled_trip' | 'published' | 'trip_id_field';
  /**
   * Network route of the matched trip, else the route_basis route.
   */
  route_id: string | null;
  /**
   * As published ('6X', 'SS'); Unknown on legacy runs.
   */
  route_published: string | Unknown | null;
  /**
   * f'{lineage}:{source}:{realtime_trip_id}@{first_observed_at}#{track}' (§2.3).
   */
  run_id: string;
  scheduled_trip_id: string | null;
  /**
   * ISO date of a GTFS service day (never a UTC date).
   */
  service_date: string;
  service_date_basis: 'scheduled_trip' | 'id_origin' | 'clock_rule';
  /**
   * Another id-tier run holds the same scheduled trip.
   */
  shares_scheduled_trip: boolean;
  source: 'ace' | 'bdfm' | 'g' | 'jz' | 'nqrw' | 'l' | 'numbered' | 'si';
  source_basis: 'published' | 'route_feed_table';
  /**
   * Every row has an old vehicle_timestamp.
   */
  stale_only: boolean;
  /**
   * Published 'YYYYMMDD', not a ServiceDate; Unknown on legacy runs.
   */
  start_date_published: string | Unknown | null;
  /**
   * Seen at one stop only; no traversal.
   */
  stationary: boolean;
  /**
   * Provisional live ids this nightly run absorbed.
   */
  supersedes_run_ids: string[];
  tier: ('exact_id' | 'path_prefix' | 'line_code' | 'path_differs' | 'drift') | null;
  /**
   * 0 for an unsplit run; else the track ordinal.
   */
  track: number;
}
/**
 * A measurement that was expected but could not be taken, and why.
 */
export interface Unknown {
  reason: string;
  unknown: true;
}
