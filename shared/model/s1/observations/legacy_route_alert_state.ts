/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

/**
 * Lossy legacy alert state read from ``v1/predictions``. Use only for
 * days without ``AlertPresence``; a bound derived from it is legacy-derived.
 * It can bound a route-level alert end, never an alert version end.
 */
export interface LegacyRouteAlertState {
  granularity: RouteAlertGranularity;
  lineage: LegacyLineage;
  /**
   * The prediction row's ts.
   */
  observed_at: number;
  /**
   * null = the route had an inference row and no active alert.
   */
  primary_alert_type: string | null;
  /**
   * The prediction row's route, as written.
   */
  route_id: string;
}
/**
 * What a ``v1/predictions`` row holds: per route, per 5-minute tick, the
 * single highest-priority alert type, with no alert identity.
 */
export interface RouteAlertGranularity {
  content: 'primary_alert_type_presence';
  scope: 'route';
  tick_seconds: 300;
}
/**
 * Read from a legacy archive object. What it lacks is ``Unknown`` on the
 * record; layer 3 labels anything it derives from it as legacy-derived.
 */
export interface LegacyLineage {
  archive:
    | 'archive/trace'
    | 'archive/trip_updates'
    | 'archive/vehicles'
    | 'archive/alerts'
    | 'archive/alerts_liveness'
    | 'archive/ene'
    | 'archive/health'
    | 'v1/predictions';
  kind: 'legacy';
  period: LegacyPeriod;
  /**
   * Position of the row in that object; null when it holds one.
   */
  row_index: number | null;
  /**
   * The legacy object the record came from.
   */
  source_key: string;
}
/**
 * The UTC days from ``start`` for which a legacy archive stands in for
 * the recorder's records. Date-granular: when ``end`` is set, the archive
 * stops at the recorder cutover, 2026-10-21T04:00:00Z, on the day before
 * ``end``.
 */
export interface LegacyPeriod {
  /**
   * Exclusive: the first UTC day the archive no longer covers at all; it stops at the recorder cutover (2026-10-21T04:00:00Z) on the day before. Null for records the recorder does not write.
   */
  end: string | null;
  /**
   * First UTC day the legacy archive covers.
   */
  start: string;
}
