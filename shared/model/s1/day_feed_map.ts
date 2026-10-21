/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

export interface DayFeedMap {
  /**
   * Keyed by UTC date YYYY-MM-DD.
   */
  days: {
    [k: string]: DayFeed | undefined;
  };
}
/**
 * Which feed served one UTC collection day of the traversal archive.
 */
export interface DayFeed {
  /**
   * The single stamped digest; null unless verified.
   */
  feed_id: string | null;
  feed_version: string | null;
  /**
   * Unverified days: the archived feed whose version matches.
   */
  inferred_feed_id: string | null;
  /**
   * Why the day is unverified.
   */
  reason: string | null;
  verified: boolean;
}
