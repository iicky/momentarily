/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

/**
 * The parquet key-value metadata of every event file, stored as
 * canonical JSON under ``PARQUET_METADATA_KEY``.
 *
 * ``provenance`` reuses layer 1's: ``inputs`` maps each object read (a
 * recorder tick or bundle, a layer-2 derived file, a packaged table) to its
 * digest, ``captured_at`` maps the same keys to the latest instant of data
 * each holds, and ``built_at`` is the latest of those (never the wall
 * clock), so a rebuild gives the same bytes. ``applied_retroactively``
 * names reference inputs measured after the period starts (the
 * ``RouteFeed`` table on legacy days). A UTC-day file (coverage, supply)
 * has no service day and observes its own day; a service-day file names
 * its service date as ``period_start``.
 */
export interface EventFileMetadata {
  /**
   * Version of the rules that built this record kind (coverage: COVERAGE_VERSION).
   */
  builder_version: number;
  /**
   * The events key of this file.
   */
  file_key: string;
  /**
   * columns.LAYOUT_VERSION it was written with.
   */
  layout_version: number;
  provenance: Provenance;
  row_count: number;
  schema: 'events/s1';
}
/**
 * Which code and which exact inputs produced an artifact.
 *
 * ``built_at`` is a *logical* build time taken from the inputs (for the
 * network, the stations snapshot's capture instant), not the wall clock, so
 * rebuilding from the same inputs yields identical bytes. ``captured_at``
 * records when each live input was captured. The period the artifact
 * describes starts on ``observed_from_utc_day`` (the first UTC archive day
 * observed against it) when known, else on the service day ``period_start``.
 * ``applied_retroactively`` names inputs captured on or after that start,
 * i.e. today's reference data stamped onto an older period.
 */
export interface Provenance {
  applied_retroactively: string[];
  /**
   * UTC epoch seconds.
   */
  built_at: number;
  captured_at: {
    /**
     * UTC epoch seconds.
     */
    [k: string]: number | undefined;
  };
  code_dirty: boolean | null;
  code_sha: string;
  inputs: {
    [k: string]: string | undefined;
  };
  observed_from_utc_day: string | null;
  period_start: string | null;
  schema: string;
}
