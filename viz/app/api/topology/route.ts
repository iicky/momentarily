import { NextResponse } from "next/server";
import { r2Configured, getJson } from "@/lib/r2";
import type { AdjEdge, RouteStops } from "@/lib/stations";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

// The static-GTFS segment graph the movement model is built on
// (state/segment_params.json). Full per-(route, direction) successor topology —
// the pairwise segments — which the public snapshot only carries for the handful
// of segments live this tick (segment_flow). A credentialed R2 GET, same vault
// the Models view uses; degrades to not-configured rather than erroring.
const SEGMENT_PARAMS_KEY = "state/segment_params.json";

interface SegmentParamsDoc {
  trained_at?: number;
  topology_source?: string;
  adjacency?: Record<
    string,
    { to: string; successors?: { to: string; n_trips: number }[] }
  >;
  route_stops?: RouteStops;
}

export async function GET() {
  if (!(await r2Configured())) {
    return NextResponse.json({ configured: false, edges: [], routeStops: {} });
  }
  try {
    const doc = await getJson<SegmentParamsDoc>(SEGMENT_PARAMS_KEY);
    const edges: AdjEdge[] = [];
    for (const [key, adj] of Object.entries(doc.adjacency ?? {})) {
      const parts = key.split("|");
      if (parts.length !== 3) continue;
      const [route, direction, from] = parts;
      const successors = (adj.successors ?? [{ to: adj.to, n_trips: 0 }])
        .filter((s) => s.to)
        .sort((a, b) => b.n_trips - a.n_trips);
      edges.push({ key, route, direction, from, to: adj.to, successors });
    }
    return NextResponse.json({
      configured: true,
      trained_at: doc.trained_at,
      topology_source: doc.topology_source,
      edges,
      routeStops: doc.route_stops ?? {},
    });
  } catch (e) {
    return NextResponse.json({ configured: true, error: (e as Error).message, edges: [], routeStops: {} });
  }
}
