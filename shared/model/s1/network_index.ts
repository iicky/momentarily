/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

/**
 * Every network build in an output directory, keyed by feed_id.
 */
export interface NetworkIndex {
  feeds: {
    [k: string]: IndexEntry | undefined;
  };
  schema: string;
}
export interface IndexEntry {
  applied_retroactively: string[];
  build_id: string;
  /**
   * blake3 of the build's canonical uncompressed JSON.
   */
  content_digest: string;
  feed_version: string | null;
  inputs: {
    [k: string]: string | undefined;
  };
  /**
   * First service day the feed declares (inclusive).
   */
  valid_from: string | null;
  /**
   * Exclusive end: the day after feed_info's feed_end_date.
   */
  valid_to: string | null;
}
