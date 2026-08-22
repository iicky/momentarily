// Station geography + segment topology helpers, shared by the Lines, Station,
// and Map surfaces. Client- and server-safe (no node builtins here).
//
// Two data sources feed these:
//   - /api/stations  → coordinates + labels from NYS Open Data 39hk-dx4f.
//   - /api/topology  → segment adjacency from state/segment_params.json (R2),
//     the static-GTFS successor graph the segment movement model is built on.
// The public snapshot supplies the rest (station metadata, live flow status).

/** Coordinates + static labels for one station, keyed by undirected GTFS id. */
export interface StationCoord {
  stop_id: string;
  name: string;
  lat: number;
  lon: number;
  borough: string | null;
  structure: string | null;
  line: string | null;
  daytime_routes: string[];
  north_label: string | null;
  south_label: string | null;
}

/** One directed pairwise segment: from_stop → its successors on a route+direction. */
export interface AdjEdge {
  key: string; // route|direction|from
  route: string;
  direction: string; // "north" | "south"
  from: string; // directional stop id, e.g. "101S"
  to: string; // primary (highest-traffic) successor
  successors: { to: string; n_trips: number }[];
}

export interface Topology {
  configured: boolean;
  trained_at?: number;
  topology_source?: string;
  edges: AdjEdge[];
}

/** Collapse a directional stop id to its station: strip a trailing N/S.
 * Mirrors worker/src/segment_flow.ts stationId. */
export function undirected(stop: string): string {
  return /[NS]$/.test(stop) ? stop.slice(0, -1) : stop;
}

// Station metadata labels shuttles as one "S" and the SIR as "SIR"; the segment
// topology keys them by the trainer's route ids. Map a line to the topology
// route(s) it covers so a trip resolves its real segments.
const TOPO_ROUTES: Record<string, string[]> = {
  S: ["FS", "GS", "H"],
  SIR: ["SI"],
};

export function edgesFor(edges: AdjEdge[], route: string, direction: string): AdjEdge[] {
  const routes = TOPO_ROUTES[route] ?? [route];
  return edges.filter((e) => routes.includes(e.route) && e.direction === direction);
}

/** Order the stops of a route+direction. Walks the highest-traffic successor
 * from every terminal (a stop with no predecessor), emitting each connected run
 * in order and concatenating them. Handles branches — and lines like the S
 * badge that cover several disjoint shuttles — as separate ordered runs rather
 * than one mangled chain. Every stop the edges name appears exactly once. */
export function orderStops(rdEdges: AdjEdge[]): string[] {
  if (rdEdges.length === 0) return [];
  const next = new Map<string, string>(); // from → primary successor
  const froms = new Set<string>();
  const tos = new Set<string>();
  const all = new Set<string>();
  for (const e of rdEdges) {
    next.set(e.from, e.to);
    froms.add(e.from);
    tos.add(e.to);
    all.add(e.from);
    all.add(e.to);
    for (const s of e.successors) all.add(s.to);
  }

  const runs: string[][] = [];
  const placed = new Set<string>();
  const runFrom = (seed: string) => {
    const run: string[] = [];
    let cur: string | undefined = seed;
    while (cur && !placed.has(cur)) {
      placed.add(cur);
      run.push(cur);
      cur = next.get(cur);
    }
    if (run.length) runs.push(run);
  };

  const byId = (a: string, b: string) => a.localeCompare(b, undefined, { numeric: true });
  // Terminals first (a stop that is never a successor), then any from left
  // unreached (a pure cycle), then stops that only ever appear as a successor.
  for (const s of [...froms].filter((f) => !tos.has(f)).sort(byId)) runFrom(s);
  for (const f of [...froms].sort(byId)) runFrom(f);
  for (const s of [...all].sort(byId)) if (!placed.has(s)) runFrom(s);
  // Longest run first, so the main trunk leads and branches/shuttles follow.
  runs.sort((a, b) => b.length - a.length);
  return runs.flat();
}

/** Equirectangular projection of lat/lon into an SVG viewport, north up.
 * Scales longitude by cos(mean latitude) so NYC keeps its true aspect. */
export function projector(
  coords: StationCoord[],
  width: number,
  height: number,
  pad: number,
): (c: { lat: number; lon: number }) => [number, number] {
  if (coords.length === 0) return () => [width / 2, height / 2];
  const lats = coords.map((c) => c.lat);
  const lons = coords.map((c) => c.lon);
  const minLat = Math.min(...lats);
  const maxLat = Math.max(...lats);
  const minLon = Math.min(...lons);
  const maxLon = Math.max(...lons);
  const meanLat = ((minLat + maxLat) / 2) * (Math.PI / 180);
  const kx = Math.cos(meanLat);
  const spanX = Math.max((maxLon - minLon) * kx, 1e-6);
  const spanY = Math.max(maxLat - minLat, 1e-6);
  const iw = width - 2 * pad;
  const ih = height - 2 * pad;
  // One scale for both axes preserves shape; center within the viewport.
  const scale = Math.min(iw / spanX, ih / spanY);
  const offX = pad + (iw - spanX * scale) / 2;
  const offY = pad + (ih - spanY * scale) / 2;
  return (c) => {
    const x = offX + (c.lon - minLon) * kx * scale;
    const y = offY + (maxLat - c.lat) * scale; // invert: north up
    return [x, y];
  };
}
