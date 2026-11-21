/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

/**
 * One occurrence of a run at a platform (§3). A service event
 * (``stopped``, ``reported_planned``) has an arrival and a departure; a
 * passage (``reported_not_stopped``, ``inferred_passage``) has only a
 * passage bound. "last"-family departures are bounded only by the next
 * arrival, and their dwell is ``Unknown`` (decision 4).
 */
export interface StopEvent {
  /**
   * Service evidence only.
   */
  arrival: EventBound | null;
  /**
   * Service evidence only; Unknown: censored tail or run end.
   */
  departure: EventBound | Unknown | null;
  /**
   * vehicle ('next') or next_arrival ('last'); service evidence only.
   */
  departure_basis: ('vehicle' | 'next_arrival') | null;
  /**
   * Unknown for the 'last' family and for an Unknown departure.
   */
  dwell: DurationBound | Unknown | null;
  family: 'last' | 'next';
  /**
   * Sorted and distinct. plan_dropped (basis trip_update): the trip update removed this listed stop more than 60 s before its predicted arrival.
   */
  flags: (
    | 'first_of_run'
    | 'held_old_timestamp'
    | 'liveness_uncertain'
    | 'order_violation'
    | 'plan_dropped'
    | 'reported_after_bound'
    | 'reported_widened_after'
  )[];
  lineage: 'raw' | 'legacy';
  /**
   * reported_not_stopped and inferred_passage only.
   */
  passage: EventBound | null;
  pattern_attribution: 'scheduled_trip' | 'unique_fit' | 'ambiguous' | 'off_pattern';
  /**
   * null when ambiguous or off pattern.
   */
  pattern_id: string | null;
  /**
   * null when the stop is off pattern or the run ambiguous.
   */
  pattern_index: number | null;
  run_id: string;
  /**
   * Occurrence of this event within the run.
   */
  seq: number;
  /**
   * ISO date of a GTFS service day (never a UTC date).
   */
  service_date: string;
  service_evidence: 'stopped' | 'reported_planned' | 'reported_not_stopped' | 'inferred_passage';
  source: 'ace' | 'bdfm' | 'g' | 'jz' | 'nqrw' | 'l' | 'numbered' | 'si';
  source_basis: 'published' | 'route_feed_table';
  /**
   * Platform.
   */
  stop_id: string;
}
/**
 * Feed-supported bound: the event happened in ``(after, by]``, integer
 * UTC seconds, exclusive below. ``after`` null is left-censored. ``after <
 * by`` strictly. A kept ``reported`` satisfies ``after < reported <= by``
 * (an earlier report widens ``after`` to ``reported - 1``, §3.2).
 */
export interface EventBound {
  /**
   * Exclusive; null = left-censored.
   */
  after: number | null;
  /**
   * Inclusive.
   */
  by: number;
  /**
   * The feed's own time, kept only inside (after, by].
   */
  reported: number | null;
  /**
   * Seconds of (after, by] in legacy suspect_stale; null when after is null and on raw records.
   */
  suspect_s: number | null;
  /**
   * Seconds of (after, by] not covered; null when after is null.
   */
  uncovered_s: number | null;
}
/**
 * A measurement that was expected but could not be taken, and why.
 */
export interface Unknown {
  reason: string;
  unknown: true;
}
/**
 * Interval-censored duration in seconds; consumers fit distributions
 * across events, never per-event midpoints. Nothing in layer 3 is
 * left-censored: a left-censored start or an open end gives ``censoring =
 * "right"`` with ``hi`` null.
 */
export interface DurationBound {
  censoring: 'none' | 'right';
  /**
   * null exactly when censoring is right.
   */
  hi: number | null;
  /**
   * Clamped at 0.
   */
  lo: number;
  /**
   * Difference of the two events' reported times, when both are kept.
   */
  reported: number | null;
  /**
   * Seconds of the span not covered; null exactly when right-censored.
   */
  uncovered_s: number | null;
}
