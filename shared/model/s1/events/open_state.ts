/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

/**
 * Right-censored state at one instant (the live store, §7; also the
 * window-end tail and the next day's checkpoint of a nightly build). Never
 * a closed record. ``current_gap`` is finite at ``as_of``; ``eventual`` is
 * the still-open headway, right-censored from ``current_gap.lo``.
 *
 * ``current_gap`` is exact (§4): with the last passing in ``(after, by]``,
 * ``lo = max(0, as_of - max(by, e))`` and ``hi = as_of - after - 1``, where
 * ``e`` is ``uncovered_until`` (an uncovered interval of the feed ended
 * after the last passing, so a train may have passed unseen) and drops
 * out when null. ``scheduled_no_observation`` reads ``since`` for ``by``
 * and has no ``hi``. ``uncovered_until`` is set exactly when the flag
 * ``uncovered_since`` is; ``as_of`` while the feed is still uncovered,
 * which gives ``lo = 0``.
 */
export interface OpenState {
  /**
   * Pattern scope: one clock per series.
   */
  ambiguity_set: ('certain' | 'with_ambiguous') | null;
  /**
   * UTC epoch seconds.
   */
  as_of: number;
  /**
   * open_run: the live pattern fit, sorted, which may exclude a matched pattern; [] otherwise.
   */
  compatible_pattern_ids: string[];
  current_gap: GapInterval;
  direction: 'N' | 'S';
  eventual: DurationBound;
  /**
   * null for open_run.
   */
  evidence: ('service' | 'physical') | null;
  /**
   * Sorted and distinct.
   */
  flags: ('ambiguous_patterns_only' | 'possibly_stale' | 'uncovered_since')[];
  kind: 'open_run' | 'open_headway' | 'scheduled_no_observation';
  /**
   * null only for scheduled_no_observation.
   */
  last_event: EventBound | null;
  /**
   * Service evidence of last_event; null exactly when it is.
   */
  last_event_evidence:
    ('stopped' | 'reported_planned' | 'reported_not_stopped' | 'inferred_passage') | null;
  /**
   * open_run only: StopEvent.seq of the last event.
   */
  last_event_seq: number | null;
  lineage: 'raw' | 'legacy';
  /**
   * open_run only; scheduled_trip when matched (tiers 1-4 live).
   */
  pattern_attribution: ('scheduled_trip' | 'unique_fit' | 'ambiguous' | 'off_pattern') | null;
  /**
   * open_run: the matched pattern or the unique fit; clocks: set exactly at pattern scope.
   */
  pattern_id: string | null;
  route_id: string;
  /**
   * open_run only.
   */
  run_id: string | null;
  /**
   * null for open_run.
   */
  scope: ('route' | 'pattern') | null;
  /**
   * Start of the open interval (§4 initialization).
   */
  since: number;
  /**
   * The run's feed (open_run) or the route's feed (clocks).
   */
  source: 'ace' | 'bdfm' | 'g' | 'jz' | 'nqrw' | 'l' | 'numbered' | 'si';
  stop_id: string;
  /**
   * End of the last uncovered interval of the feed after the last passing (after since with no passing), as_of while still uncovered; null when none. Set exactly with the uncovered_since flag.
   */
  uncovered_until: number | null;
}
/**
 * Exactly [as_of - max(by, uncovered_until), as_of - after - 1], lo clamped at 0 (§4); since for by with no observation.
 */
export interface GapInterval {
  hi: number | null;
  lo: number;
}
/**
 * The still-open headway: censoring right, hi null, lo = current_gap.lo.
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
