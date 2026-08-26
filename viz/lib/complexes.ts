// Station-complex grouping, derived from the public snapshot, that bridges the
// live station metadata to the topology-pure journey enumerator (viz/lib/journeys.ts).
//
// The committed topology keys stops per route, so one walkable station is several
// ids the enumerator can't tell apart from unrelated stops — 14 St-Union Sq is
// 635 (4/5/6), R20 (N/Q/R/W) and L03 (L). The authoritative grouping is the
// snapshot's `station_complex_id` (Station.station_complex_id), NOT the station
// name: two entirely separate "96 St" stations (Broadway vs Lexington) share a
// name but are different complexes, and grouping by name would invent a transfer
// between them. Every snapshot station carries a complex id; the ids are the same
// undirected parent GTFS ids the topology uses, so no translation is needed.
//
// This is deliberately a separate module from journeys.ts: the enumerator stays a
// pure function of the committed topology, and this snapshot-fed layer feeds it
// the complex groupings a real trip needs. Client- and server-safe (no node builtins).

import { undirected, type AdjEdge, type RouteStops } from "./stations.ts";
import { enumerateJourneys, type Journey, type JourneyOptions } from "./journeys.ts";

// The only station fields this needs — satisfied structurally by the snapshot's
// `stations` map (Record<string, Station>), whose key is the gtfs_stop_id.
export type ComplexStations = Record<string, { station_complex_id: string | null }>;

export interface ComplexIndex {
  /** The undirected GTFS stop ids that make up a complex (empty if unknown). */
  stopsOf(complexId: string): string[];
  /** The complex a stop belongs to, or null if the snapshot doesn't place it. */
  complexOf(stop: string): string | null;
  /** Every station_complex_id present, sorted — e.g. to drive a station picker. */
  ids: string[];
  /** The multi-stop complexes as undirected stop-id sets: the `complexes` option
   * for enumerateJourneys. Singletons are omitted — the enumerator already treats
   * a lone shared stop id as a valid transfer point without being told. */
  transferComplexes: string[][];
}

/** Index the snapshot's stations by their station_complex_id. */
export function indexComplexes(stations: ComplexStations): ComplexIndex {
  const stopToComplex = new Map<string, string>();
  const complexToStops = new Map<string, string[]>();
  for (const [rawId, meta] of Object.entries(stations)) {
    const cid = meta.station_complex_id;
    if (!cid) continue;
    const stop = undirected(rawId);
    if (!stopToComplex.has(stop)) stopToComplex.set(stop, cid);
    let stops = complexToStops.get(cid);
    if (!stops) {
      stops = [];
      complexToStops.set(cid, stops);
    }
    if (!stops.includes(stop)) stops.push(stop);
  }
  for (const stops of complexToStops.values()) stops.sort();
  const ids = [...complexToStops.keys()].sort();
  const transferComplexes = ids
    .map((id) => complexToStops.get(id) as string[])
    .filter((stops) => stops.length > 1)
    .map((stops) => [...stops]);
  return {
    stopsOf: (id) => [...(complexToStops.get(id) ?? [])],
    complexOf: (stop) => stopToComplex.get(undirected(stop)) ?? null,
    ids,
    transferComplexes,
  };
}

/**
 * Enumerate journeys between two complexes named by station_complex_id, over the
 * committed topology. Resolves each endpoint to its member stops and feeds the
 * enumerator the full set of multi-stop complexes so mid-journey transfers can
 * cross platforms within one station. A `complexes` option is ignored — the index
 * supplies it. An unknown complex id resolves to no stops, yielding no journeys.
 */
export function journeysBetween(
  routeStops: RouteStops,
  edges: AdjEdge[],
  index: ComplexIndex,
  originComplexId: string,
  destComplexId: string,
  options: Omit<JourneyOptions, "complexes"> = {},
): Journey[] {
  return enumerateJourneys(
    routeStops,
    edges,
    index.stopsOf(originComplexId),
    index.stopsOf(destComplexId),
    { ...options, complexes: index.transferComplexes },
  );
}
