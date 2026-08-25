"use client";

import { Suspense, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useSnapshot, useCoords, useTopology } from "../../useData";
import { PageHeader, RouteBullet } from "../../ui";
import { undirected, edgesFor, orderTrip, projector } from "@/lib/stations";
import { fmtMinutes } from "@/lib/feed";
import type { Snapshot } from "@/lib/types";
import type { StationCoord, Topology } from "@/lib/stations";

type Dir = "north" | "south";
const W = 760;
const H = 900;

// Pairwise segment as drawn on the map: two adjacent stations plus the live
// verdict the movement model published for that stretch (null = topology only).
interface Seg {
  key: string;
  fromId: string;
  toId: string;
  fromName: string;
  toName: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  status: "normal" | "disrupted" | null;
  recoveryMin: number | null;
}

const STROKE: Record<string, string> = {
  normal: "var(--normal)",
  disrupted: "var(--disrupted)",
  topology: "var(--border)",
};

export default function MapPage() {
  return (
    <Suspense fallback={<div className="wrap"><div className="sub">loading…</div></div>}>
      <MapView />
    </Suspense>
  );
}

function MapView() {
  const router = useRouter();
  const qp = useSearchParams();
  const { data: snap } = useSnapshot();
  const { data: coords } = useCoords();
  const { data: topo } = useTopology();

  const routeList = useMemo(() => {
    if (!snap) return [];
    const seen = new Set<string>();
    for (const st of Object.values(snap.stations)) {
      for (const r of st.routes_served) seen.add(r);
    }
    return [...seen].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  }, [snap]);

  const qpRoute = qp.get("route");
  const route = qpRoute && routeList.includes(qpRoute) ? qpRoute : routeList[0] ?? "";
  const dir = (qp.get("dir") === "south" ? "south" : "north") as Dir;

  const setTrip = (r: string, d: Dir) => {
    const p = new URLSearchParams();
    p.set("route", r);
    p.set("dir", d);
    router.replace(`/map/trip?${p}`);
  };

  const trip = useMemo(
    () => buildTrip(snap, coords, topo, route, dir),
    [snap, coords, topo, route, dir],
  );

  return (
    <div className="wrap">
      <PageHeader subtitle="Select a trip — a line and direction — to see its pairwise segments." />

      <div className="line-head">
        {snap && route && <RouteBullet snap={snap} route={route} size={36} />}
        <select
          className="trip-select"
          value={route}
          onChange={(e) => setTrip(e.target.value, dir)}
        >
          {routeList.map((r) => (
            <option key={r} value={r}>
              {r} line
            </option>
          ))}
        </select>
        <div className="dir-toggle">
          <button className={dir === "north" ? "active" : ""} onClick={() => setTrip(route, "north")}>
            Northbound
          </button>
          <button className={dir === "south" ? "active" : ""} onClick={() => setTrip(route, "south")}>
            Southbound
          </button>
        </div>
      </div>

      {!snap || !coords ? (
        <div className="sub">loading…</div>
      ) : trip.markers.length === 0 ? (
        <div className="note">No stations with coordinates for this trip.</div>
      ) : (
        <div className="map-layout">
          <MapCanvas snap={snap} trip={trip} onPick={(id) => router.push(`/stations/${id}`)} />
          <SegPanel snap={snap} route={route} trip={trip} />
        </div>
      )}

      {topo?.feed_version && (
        <div className="prov-note">
          {/* The timetable, not a code sha: this topology is read off a
              committed asset, so what identifies it is which static feed it was
              built from. MTA republishes on every service change and names the
              change in the version string. */}
          timetable · <code>{topo.feed_version.version}</code>
          {topo.topology_source ? ` · ${topo.topology_source}` : ""}
        </div>
      )}
    </div>
  );
}

interface Marker {
  id: string;
  name: string;
  x: number;
  y: number;
  degraded: boolean;
}

interface Trip {
  segs: Seg[];
  markers: Marker[];
}

function buildTrip(
  snap: Snapshot | null,
  coords: Record<string, StationCoord> | null,
  topo: Topology | null,
  route: string,
  dir: Dir,
): Trip {
  if (!snap || !coords || !route) return { segs: [], markers: [] };

  const rdEdges = topo ? edgesFor(topo.edges, route, dir) : [];
  // Which stations belong to this trip, in order: the diagram asset's
  // canonical patterns when the topology names this route+direction, else
  // every station that serves the line (markers-only fallback, also covering
  // the moment before the asset has loaded).
  const dirSuffix = dir === "north" ? "N" : "S";
  const topoStops = topo ? orderTrip(topo.routeStops, topo.edges, route, dir) : [];
  const stopIds = topoStops.length
    ? topoStops
    : Object.values(snap.stations)
        .filter((s) => s.routes_served.includes(route))
        .map((s) => `${s.gtfs_stop_id}${dirSuffix}`);
  const orderOf: Record<string, number> = {};
  stopIds.forEach((s, i) => (orderOf[undirected(s)] = i));

  const withCoords = stopIds
    .map((s) => ({ stop: s, id: undirected(s), c: coords[undirected(s)] }))
    .filter((x): x is { stop: string; id: string; c: StationCoord } => !!x.c);
  const project = projector(
    withCoords.map((x) => x.c),
    W,
    H,
    28,
  );

  const flow = snap.station_flow?.stations;
  const markers: Marker[] = withCoords.map((x) => {
    const [px, py] = project(x.c);
    return {
      id: x.id,
      name: x.c.name,
      x: px,
      y: py,
      degraded: flow?.[x.id]?.status === "degraded",
    };
  });

  const pos: Record<string, { x: number; y: number; name: string }> = {};
  for (const m of markers) pos[m.id] = { x: m.x, y: m.y, name: m.name };

  const segs: Seg[] = [];
  for (const e of rdEdges) {
    const a = pos[undirected(e.from)];
    const b = pos[undirected(e.to)];
    if (!a || !b) continue;
    const live = snap.segment_flow?.segments[e.key] ?? null;
    segs.push({
      key: e.key,
      fromId: undirected(e.from),
      toId: undirected(e.to),
      fromName: a.name,
      toName: b.name,
      x1: a.x,
      y1: a.y,
      x2: b.x,
      y2: b.y,
      status: live?.status ?? null,
      recoveryMin: live?.recovery?.recovery_minutes ?? null,
    });
  }
  // Read in trip order — the panel lists the segments as you'd ride them.
  segs.sort((a, b) => (orderOf[a.fromId] ?? 1e9) - (orderOf[b.fromId] ?? 1e9));
  return { segs, markers };
}

function MapCanvas({
  snap,
  trip,
  onPick,
}: {
  snap: Snapshot;
  trip: Trip;
  onPick: (id: string) => void;
}) {
  const [hover, setHover] = useState<Marker | null>(null);
  return (
    <div className="map-canvas">
      <svg viewBox={`0 0 ${W} ${H}`} className="map-svg" role="img">
        {trip.segs.map((s) => (
          <line
            key={s.key}
            x1={s.x1}
            y1={s.y1}
            x2={s.x2}
            y2={s.y2}
            stroke={STROKE[s.status ?? "topology"]}
            strokeWidth={s.status ? 4 : 2.5}
            strokeLinecap="round"
          />
        ))}
        {trip.markers.map((m) => (
          <circle
            key={m.id}
            cx={m.x}
            cy={m.y}
            r={hover?.id === m.id ? 6 : 4}
            className={`map-node ${m.degraded ? "degraded" : ""}`}
            onMouseEnter={() => setHover(m)}
            onMouseLeave={() => setHover((h) => (h?.id === m.id ? null : h))}
            onClick={() => onPick(m.id)}
          />
        ))}
        {hover && (
          <text x={hover.x + 9} y={hover.y + 4} className="map-label">
            {hover.name}
          </text>
        )}
      </svg>
      <div className="map-legend">
        <span><i style={{ background: "var(--normal)" }} /> flowing</span>
        <span><i style={{ background: "var(--disrupted)" }} /> disrupted</span>
        <span><i style={{ background: "var(--border)" }} /> not judged</span>
      </div>
    </div>
  );
}

function SegPanel({ snap, route, trip }: { snap: Snapshot; route: string; trip: Trip }) {
  return (
    <div className="seg-panel">
      <div className="section-title">Pairwise segments ({trip.segs.length})</div>
      {trip.segs.length === 0 ? (
        <div className="note muted">No segments for this trip.</div>
      ) : (
        <ul className="seglist">
          {trip.segs.map((s) => (
            <li key={s.key}>
              <RouteBullet snap={snap} route={route} size={16} />
              <span className="seg-path">
                <Link href={`/stations/${s.fromId}`}>{s.fromName}</Link>
                {" → "}
                <Link href={`/stations/${s.toId}`}>{s.toName}</Link>
              </span>
              {s.status ? (
                <span className={`cond ${s.status}`}>
                  {s.status}
                  {s.status === "disrupted" && s.recoveryMin != null
                    ? ` · ~${fmtMinutes(s.recoveryMin)}`
                    : ""}
                </span>
              ) : (
                <span className="cond unknown">—</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
