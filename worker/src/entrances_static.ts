/**
 * Static subway entrance metadata from NYS Open Data i9wp-a4ja
 * (MTA Subway Entrances and Exits: 2024).
 *
 * Refreshed daily alongside the stations-static fetch (39hk-dx4f) and
 * published as its own v1/entrances.json artifact, NOT embedded in the
 * per-tick snapshot. ~2120 rows.
 *
 * Resolution: each entrance whose gtfs_stop_id appears in the stations
 * catalog lands in `entrances` keyed by that id (direct match, ~97%).
 * Entrances in large complexes whose gtfs_stop_id is a sibling constituent
 * NOT in the catalog land in `complex_entrances` keyed by complex_id — a
 * consumer joins those via Station.station_complex_id. This avoids the
 * ambiguity of arbitrarily assigning an entrance to one station in a
 * multi-station complex (Times Sq has 5 constituent stations).
 */

import type { Provenance } from './buildinfo';
import { codeProvenance } from './buildinfo';
import type { StationOut } from './stations_static';

export const ENTRANCES_KEY = 'v1/entrances.json';

const ENTRANCES_CACHE_CONTROL = 'public, max-age=60, s-maxage=300';

// ---------- output types ----------

export interface EntranceOut {
  /** The entrance's own gtfs_stop_id from the source feed (may differ from
   * the key in `entrances` when this entrance was resolved by complex_id). */
  gtfs_stop_id: string;
  complex_id: string | null;
  station_id: string | null;
  entrance_type: string;
  entry_allowed: boolean;
  exit_allowed: boolean;
  lat: number;
  lon: number;
}

export interface EntrancesCoverage {
  /** Total entrance rows ingested. */
  row_count: number;
  /** Distinct station keys in the `entrances` map. */
  station_keys: number;
  /** Distinct complex keys in the `complex_entrances` map. */
  complex_keys: number;
  /** Entrances matched to a station by their own gtfs_stop_id. */
  direct_match: number;
  /** Entrances keyed by complex_id (sibling constituent not in catalog). */
  complex_fallback: number;
  /** Entrances that could not be resolved by either join. */
  unresolved: number;
  /** gtfs_stop_ids of unresolved entrances, if any. */
  unresolved_ids: string[];
  /** Total stations in the catalog used for resolution. */
  catalog_stations: number;
  /** Stations that have at least one entrance (direct match only). */
  stations_with_entrances: number;
  /** Fraction of catalog stations covered by direct match (0–1). */
  station_coverage: number;
  /** Stations reachable by direct match OR via a matched complex_id. */
  stations_resolved: number;
  /** Fraction of catalog stations reachable by either join (0–1). */
  station_coverage_effective: number;
}

export interface PublishedEntrances {
  fetched_at: number;
  provenance: Provenance;
  coverage: EntrancesCoverage;
  /** Keyed by gtfs_stop_id — entrances that matched a station directly. */
  entrances: Record<string, EntranceOut[]>;
  /** Keyed by complex_id — entrances whose gtfs_stop_id wasn't in the
   * stations catalog. A consumer joins these via Station.station_complex_id. */
  complex_entrances: Record<string, EntranceOut[]>;
}

// ---------- parsing ----------

function asString(v: unknown): string | null {
  return typeof v === 'string' && v.length > 0 ? v : null;
}

function asNumber(v: unknown): number | null {
  if (typeof v === 'number' && Number.isFinite(v)) return v;
  if (typeof v === 'string') {
    const n = Number.parseFloat(v);
    return Number.isFinite(n) ? n : null;
  }
  return null;
}

/** Case-insensitive check for Socrata YES/NO booleans. */
function isYes(v: unknown): boolean {
  return typeof v === 'string' && v.toUpperCase() === 'YES';
}

/** Raw parsed entrance before station resolution. */
export interface EntranceRaw {
  gtfs_stop_id: string;
  complex_id: string | null;
  station_id: string | null;
  entrance_type: string;
  entry_allowed: boolean;
  exit_allowed: boolean;
  lat: number;
  lon: number;
}

/** Parse the Socrata rows array, skipping malformed rows. */
export function parseEntrancesFeed(payload: unknown): EntranceRaw[] {
  if (!Array.isArray(payload)) return [];
  const out: EntranceRaw[] = [];
  for (const row of payload) {
    if (!row || typeof row !== 'object') continue;
    const r = row as Record<string, unknown>;
    const gtfs_stop_id = asString(r.gtfs_stop_id);
    if (!gtfs_stop_id) continue;
    const lat = asNumber(r.entrance_latitude);
    const lon = asNumber(r.entrance_longitude);
    if (lat === null || lon === null) continue;
    out.push({
      gtfs_stop_id,
      complex_id: asString(r.complex_id),
      station_id: asString(r.station_id),
      entrance_type: asString(r.entrance_type) ?? 'Unknown',
      entry_allowed: isYes(r.entry_allowed),
      exit_allowed: isYes(r.exit_allowed),
      lat,
      lon,
    });
  }
  return out;
}

// ---------- resolution ----------

/**
 * Resolve raw entrances against the stations catalog.
 *
 * Direct match: entrance gtfs_stop_id exists in stations → keyed under it.
 * Complex fallback: entrance has a complex_id that any station shares →
 *   keyed under complex_id in `complex_entrances`. This avoids ambiguity
 *   when a complex has multiple stations (e.g. Times Sq-42 St complex 611
 *   has stations 127, 725, R16, 902, A27).
 * Unresolved: neither join succeeds → keyed under source gtfs_stop_id in
 *   `entrances` with the count noted in coverage.
 */
export function resolveEntrances(
  raws: EntranceRaw[],
  stations: Record<string, StationOut>,
): {
  entrances: Record<string, EntranceOut[]>;
  complexEntrances: Record<string, EntranceOut[]>;
  coverage: EntrancesCoverage;
} {
  // Build the set of complex_ids that appear in the stations catalog.
  const knownComplexIds = new Set<string>();
  for (const s of Object.values(stations)) {
    if (s.station_complex_id) knownComplexIds.add(s.station_complex_id);
  }

  const entrances: Record<string, EntranceOut[]> = {};
  const complexEntrances: Record<string, EntranceOut[]> = {};
  let directMatch = 0;
  let complexFallback = 0;
  let unresolved = 0;
  const unresolvedIds = new Set<string>();

  for (const raw of raws) {
    const out: EntranceOut = {
      gtfs_stop_id: raw.gtfs_stop_id,
      complex_id: raw.complex_id,
      station_id: raw.station_id,
      entrance_type: raw.entrance_type,
      entry_allowed: raw.entry_allowed,
      exit_allowed: raw.exit_allowed,
      lat: raw.lat,
      lon: raw.lon,
    };

    if (raw.gtfs_stop_id in stations) {
      (entrances[raw.gtfs_stop_id] ??= []).push(out);
      directMatch++;
    } else if (raw.complex_id && knownComplexIds.has(raw.complex_id)) {
      (complexEntrances[raw.complex_id] ??= []).push(out);
      complexFallback++;
    } else {
      // Unresolved — still publish under source id so no data is lost.
      (entrances[raw.gtfs_stop_id] ??= []).push(out);
      unresolved++;
      unresolvedIds.add(raw.gtfs_stop_id);
    }
  }

  const catalogSize = Object.keys(stations).length;
  const directStations = Object.keys(entrances).filter(k => !unresolvedIds.has(k)).length;

  // Stations reachable via complex fallback: every catalog station whose
  // complex_id appears in complexEntrances.
  const resolvedComplexIds = new Set(Object.keys(complexEntrances));
  const complexReachedStations = new Set<string>();
  for (const s of Object.values(stations)) {
    if (s.station_complex_id && resolvedComplexIds.has(s.station_complex_id)) {
      complexReachedStations.add(s.gtfs_stop_id);
    }
  }
  // Effective = direct + complex-reached (minus any overlap, though direct
  // stations shouldn't also appear in complexReached).
  const stationsResolved = new Set([
    ...Object.keys(entrances).filter(k => !unresolvedIds.has(k)),
    ...complexReachedStations,
  ]).size;

  return {
    entrances,
    complexEntrances,
    coverage: {
      row_count: raws.length,
      station_keys: Object.keys(entrances).length,
      complex_keys: Object.keys(complexEntrances).length,
      direct_match: directMatch,
      complex_fallback: complexFallback,
      unresolved,
      unresolved_ids: [...unresolvedIds].sort(),
      catalog_stations: catalogSize,
      stations_with_entrances: directStations,
      station_coverage: catalogSize > 0 ? directStations / catalogSize : 0,
      stations_resolved: stationsResolved,
      station_coverage_effective: catalogSize > 0 ? stationsResolved / catalogSize : 0,
    },
  };
}

// ---------- publish ----------

export function buildEntrances(
  fetchedAt: number,
  raws: EntranceRaw[],
  stations: Record<string, StationOut>,
): PublishedEntrances {
  const { entrances, complexEntrances, coverage } = resolveEntrances(raws, stations);
  return {
    fetched_at: fetchedAt,
    provenance: codeProvenance(),
    coverage,
    entrances,
    complex_entrances: complexEntrances,
  };
}

export async function publishEntrances(
  bucket: R2Bucket,
  published: PublishedEntrances,
): Promise<void> {
  await bucket.put(ENTRANCES_KEY, JSON.stringify(published), {
    httpMetadata: {
      contentType: 'application/json',
      cacheControl: ENTRANCES_CACHE_CONTROL,
    },
  });
}
