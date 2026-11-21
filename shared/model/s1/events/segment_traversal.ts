/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

/**
 * A run between two of its stop events on one pattern. ``run_time`` is
 * departure(from) to arrival(to); ``stop_to_stop`` is arrival to arrival,
 * dwell included, comparable with the old ``archive/traversals``.
 */
export interface SegmentTraversal {
  from_event: EventBound;
  from_index: number;
  /**
   * StopEvent.seq, not a pattern index.
   */
  from_seq: number;
  hidden_realtime_stops: number;
  lineage: 'raw' | 'legacy';
  pattern_id: string;
  run_id: string;
  /**
   * Unknown for the 'last' family.
   */
  run_time: DurationBound | Unknown;
  run_time_basis: 'vehicle' | 'next_arrival';
  /**
   * Layer-1 segments covered, in order.
   */
  segment_ids: string[];
  /**
   * ISO date of a GTFS service day (never a UTC date).
   */
  service_date: string;
  source: 'ace' | 'bdfm' | 'g' | 'jz' | 'nqrw' | 'l' | 'numbered' | 'si';
  stop_to_stop: DurationBound;
  to_arrival: EventBound1;
  /**
   * > from_index; > from_index + 1 only across unreported stops.
   */
  to_index: number;
  to_seq: number;
}
/**
 * Departure ('next') or arrival ('last') at the from stop.
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
/**
 * A measurement that was expected but could not be taken, and why.
 */
export interface Unknown {
  reason: string;
  unknown: true;
}
/**
 * Feed-supported bound: the event happened in ``(after, by]``, integer
 * UTC seconds, exclusive below. ``after`` null is left-censored. ``after <
 * by`` strictly. A kept ``reported`` satisfies ``after < reported <= by``
 * (an earlier report widens ``after`` to ``reported - 1``, §3.2).
 */
export interface EventBound1 {
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
