/**
 * Collapse a directional GTFS stop id to its parent station: strip a trailing
 * N/S suffix (`A24S` -> `A24`), leave a bare station id untouched.
 *
 * The single source shared by worker/src/segment_flow.ts (stationId),
 * viz/lib/stations.ts (undirected) and viz/lib/segments.ts (stationOf), which
 * each carried their own copy of this rule. Also mirrored by
 * training/diagram.py parent_station (a Python port, kept in sync separately).
 */
export function stationId(stop: string): string {
  const last = stop.at(-1);
  return last === 'N' || last === 'S' ? stop.slice(0, -1) : stop;
}
