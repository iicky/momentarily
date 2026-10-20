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
  feed_version: string | null;
  inputs: {
    [k: string]: string | undefined;
  };
  valid_from: string | null;
  valid_to: string | null;
}
