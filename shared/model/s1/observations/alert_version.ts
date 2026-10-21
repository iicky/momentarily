/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

export type JsonValue = unknown;

/**
 * One distinct alert content, and the window in which it was seen.
 *
 * "Fetch" below means a fetch with outcome ok or empty: an empty feed said
 * there were no alerts, which is evidence, not a gap. The version left the
 * feed in ``(last_seen, gone_by]``. Where ``unobserved`` overlaps that span,
 * the end is only known to lie in the wider span. ``gone_kind``
 * ``superseded`` means the same alert id continued with other content.
 * ``window`` is the presence window the index was built over (a UTC day
 * plus carry). Legacy versions come from ``archive/alerts``.
 */
export interface AlertVersion {
  /**
   * Last fetch before first_seen without this version; null = first fetch of the window.
   */
  absent_before: number | Unknown | null;
  alert_id: string;
  /**
   * The entity's mercury alert_type.
   */
  alert_type: string | null;
  /**
   * alert_content_digest(entity).
   */
  content_digest: string;
  /**
   * The entity's mercury created_at.
   */
  created_at: number | null;
  /**
   * Verbatim feed entity.
   */
  entity: {
    [k: string]: JsonValue | undefined;
  };
  first_seen: number | Unknown;
  /**
   * First fetch after last_seen without this version; null = still present at window end.
   */
  gone_by: number | Unknown | null;
  /**
   * null exactly when gone_by is null. With gone_by Unknown: Unknown, or superseded when a later updated_at of the same id proves it.
   */
  gone_kind: ('removed' | 'superseded') | Unknown | null;
  last_seen: number | Unknown;
  /**
   * Raw: the payload of the first sighting.
   */
  lineage: RawLineage | LegacyLineage;
  source: 'alerts' | 'numbered';
  /**
   * Sorted, disjoint gaps inside [first_seen, gone_by) with no fetch.
   */
  unobserved: Interval[] | Unknown;
  /**
   * The entity's mercury updated_at.
   */
  updated_at: number | null;
  window: Interval | Unknown;
}
/**
 * A measurement that was expected but could not be taken, and why.
 */
export interface Unknown {
  reason: string;
  unknown: true;
}
/**
 * Decoded from a stored raw payload.
 */
export interface RawLineage {
  kind: 'raw';
  /**
   * The tick bundle or blob holding the bytes.
   */
  payload_key: string;
  /**
   * Bundle member; null for a blob.
   */
  payload_member: number | null;
  /**
   * Lowercase sha256 hex.
   */
  payload_sha256: string;
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
 * The UTC days ``[start, end)`` for which a legacy archive stands in for
 * the recorder's records.
 */
export interface LegacyPeriod {
  /**
   * Exclusive: the first UTC day the recorder wrote the replacing record; null while it has not.
   */
  end: string | null;
  /**
   * First UTC day the legacy archive covers.
   */
  start: string;
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
