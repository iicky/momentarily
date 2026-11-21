/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

/**
 * Transport evidence only, keyed to fetch instants (§5.1, combined rule
 * of the 10-21 verdict). Train counts never make an interval uncovered.
 * ``not_stored``: fetched, but the bytes were not stored, so nothing
 * decodes. ``suspect_stale`` (legacy) is uncertain evidence, not a
 * transport failure.
 */
export interface CoverageInterval {
  /**
   * Fetches (raw) or trace objects (legacy) inside.
   */
  evidence_count: number;
  interval: Interval;
  lineage: 'raw' | 'legacy';
  source: 'ace' | 'bdfm' | 'g' | 'jz' | 'nqrw' | 'l' | 'numbered' | 'si';
  state:
    | (
        | 'covered'
        | 'fetch_failed'
        | 'no_invocation'
        | 'not_stored'
        | 'stale_header'
        | 'suspect_stale'
      )
    | Unknown;
}
/**
 * A half-open span of instants, ``[start, end)``.
 */
export interface Interval {
  /**
   * UTC epoch seconds.
   */
  end: number;
  /**
   * UTC epoch seconds.
   */
  start: number;
}
/**
 * A measurement that was expected but could not be taken, and why.
 */
export interface Unknown {
  reason: string;
  unknown: true;
}
