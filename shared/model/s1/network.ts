/* Generated from schema/model/s1 by worker/scripts/model_types.ts. Do not edit.
 * Regenerate: uv run python -m scripts.export_model_schema && (cd worker && bun run gen:model) */

export interface NetworkSnapshot {
  build_id: string;
  complexes: Complex[];
  feed: Feed;
  pattern_segments: PatternSegment[];
  patterns: Pattern[];
  platforms: Platform[];
  provenance: Provenance;
  routes: Route[];
  segments: Segment[];
  service_days: ServiceDay[];
  services: Service[];
  stations: Station[];
  system: System;
  trips: ScheduledTrip[];
}
/**
 * Stations a rider can transfer between without leaving the system.
 */
export interface Complex {
  complex_id: string;
  name: string;
  station_ids: string[];
}
export interface Feed {
  /**
   * sha256 hex of the GTFS zip bytes.
   */
  feed_id: string;
  feed_version: string | null;
  /**
   * First service day the feed declares (feed_info).
   */
  valid_from: string | null;
  /**
   * Exclusive end: the day after feed_info's feed_end_date.
   */
  valid_to: string | null;
}
export interface PatternSegment {
  pattern_id: string;
  segment_id: string;
  /**
   * 0-based hop index within the pattern.
   */
  seq: number;
}
export interface Pattern {
  direction: Direction;
  /**
   * Most common trip headsign (ties broken lexicographically).
   */
  headsign: string | null;
  n_trips: number;
  pattern_id: string;
  route_id: string;
  shape_ids: string[];
  /**
   * Platform stop_ids in order.
   *
   * @minItems 2
   */
  stops: [string, string, ...string[]];
}
export interface Direction {
  /**
   * The N/S suffix shared by every stop, or null if they differ.
   */
  code: ('N' | 'S') | null;
  /**
   * GTFS direction_id; null when the feed omits it.
   */
  direction_id: (0 | 1) | null;
  /**
   * Pattern-wide label. Null in s1: labels vary along a line, so they live on Platform.
   */
  label: string | null;
}
/**
 * A GTFS child stop: where trains actually stop.
 */
export interface Platform {
  /**
   * NYC N/S suffix; null for a platform serving both directions.
   */
  direction_code: ('N' | 'S') | null;
  /**
   * Rider-facing label for this side of the station, e.g. 'Manhattan'.
   */
  direction_label: string | null;
  name: string;
  station_id: string;
  stop_id: string;
}
/**
 * Which code and which exact inputs produced an artifact.
 *
 * ``built_at`` is a *logical* build time derived from the inputs (the capture
 * instant of the newest input), not the wall clock, so rebuilding from the same
 * inputs yields identical bytes. ``applied_retroactively`` names inputs that
 * were captured after part of the period the artifact describes, i.e. today's
 * reference data stamped onto an older timetable.
 */
export interface Provenance {
  applied_retroactively: string[];
  /**
   * UTC epoch seconds.
   */
  built_at: number;
  code_dirty: boolean | null;
  code_sha: string;
  inputs: {
    [k: string]: string | undefined;
  };
  schema: string;
}
export interface Route {
  color: string | null;
  long_name: string;
  route_id: string;
  route_type: number;
  short_name: string;
  sort_order: number | null;
  text_color: string | null;
}
export interface Segment {
  from_stop: string;
  pattern_ids: string[];
  route_ids: string[];
  /**
   * f'{from_stop}>{to_stop}'.
   */
  segment_id: string;
  to_stop: string;
}
export interface ServiceDay {
  active_service_ids: string[];
  /**
   * ISO date of a GTFS service day (never a UTC date).
   */
  date: string;
  /**
   * weekday/saturday/sunday when every active service's weekly rule is exactly that; none when nothing runs; otherwise mixed.
   */
  day_type: 'weekday' | 'saturday' | 'sunday' | 'mixed' | 'none';
  /**
   * calendar_dates.txt touches this date.
   */
  is_exception: boolean;
}
/**
 * One GTFS service_id, exactly as declared (never collapsed to a day type).
 */
export interface Service {
  /**
   * Items: ISO date of a GTFS service day (never a UTC date).
   */
  added_dates: string[];
  /**
   * Exclusive end: the day after calendar.txt's end_date.
   */
  end: string | null;
  /**
   * Items: ISO date of a GTFS service day (never a UTC date).
   */
  removed_dates: string[];
  service_id: string;
  /**
   * First day of the weekly rule.
   */
  start: string | null;
  weekdays: ('monday' | 'tuesday' | 'wednesday' | 'thursday' | 'friday' | 'saturday' | 'sunday')[];
}
/**
 * A GTFS parent stop (location_type 1).
 */
export interface Station {
  ada: ('none' | 'full' | 'partial') | null;
  borough: string | null;
  complex_id: string | null;
  lat: number;
  lon: number;
  name: string;
  station_id: string;
}
export interface System {
  agency_id: string;
  agency_name: string;
  /**
   * Stable system key, e.g. 'nyct-subway'.
   */
  system_id: string;
  /**
   * IANA timezone of the service days.
   */
  timezone: string;
}
export interface ScheduledTrip {
  direction: Direction;
  headsign: string | null;
  pattern_id: string;
  route_id: string;
  service_id: string;
  shape_id: string | null;
  /**
   * [stop_id, arrival, departure] in stop_sequence order.
   */
  stop_times: [string, number, number][];
  trip_id: string;
}
