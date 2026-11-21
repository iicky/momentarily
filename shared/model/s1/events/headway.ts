/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

/**
 * Consecutive passings at a stop (§4), owned by the follower's service
 * date. Adjacency is computed per scope and evidence level, never filtered
 * from another series. ``leader_unknown``: no leader in the window and no
 * valid checkpoint, so every leader field is null and ``headway.hi`` too.
 */
export interface Headway {
  /**
   * Set exactly at pattern scope.
   */
  ambiguity_set: ('certain' | 'with_ambiguous') | null;
  /**
   * The follower's timed end: departure (stopped), arrival (reported_planned), else passage.
   */
  basis: 'departure' | 'arrival' | 'passage';
  direction: 'N' | 'S';
  /**
   * service: stopped or reported_planned ends only, never mixed.
   */
  evidence: 'service' | 'physical';
  follower_passing: EventBound;
  follower_pattern_attribution: 'scheduled_trip' | 'unique_fit' | 'ambiguous' | 'off_pattern';
  follower_run_id: string;
  /**
   * Revisits of a stop differ here.
   */
  follower_seq: number;
  follower_service_evidence:
    'stopped' | 'reported_planned' | 'reported_not_stopped' | 'inferred_passage';
  headway: DurationBound;
  leader_passing: EventBound | null;
  leader_pattern_attribution:
    ('scheduled_trip' | 'unique_fit' | 'ambiguous' | 'off_pattern') | null;
  leader_run_id: string | null;
  leader_seq: number | null;
  leader_service_date: string | null;
  leader_service_evidence:
    ('stopped' | 'reported_planned' | 'reported_not_stopped' | 'inferred_passage') | null;
  leader_unknown: boolean;
  lineage: 'raw' | 'legacy';
  /**
   * Set exactly at pattern scope.
   */
  pattern_id: string | null;
  route_id: string;
  scope: 'route' | 'pattern';
  /**
   * The follower's.
   */
  service_date: string;
  /**
   * The follower's feed; every feed of a route has one family.
   */
  source: 'ace' | 'bdfm' | 'g' | 'jz' | 'nqrw' | 'l' | 'numbered' | 'si';
  /**
   * A scheduled closure lies inside the span.
   */
  spans_no_service: boolean;
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
