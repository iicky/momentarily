"use client";

import { Suspense, useMemo } from "react";
import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { useSnapshot, useTopology } from "../../useData";
import { PageHeader, RouteBullet } from "../../ui";
import { undirected, orderTrip } from "@/lib/stations";
import { fmtMinutes } from "@/lib/feed";
import type { Snapshot } from "@/lib/types";

type Dir = "north" | "south";

export default function LinePage() {
  return (
    <Suspense fallback={<div className="wrap"><div className="sub">loading…</div></div>}>
      <LineView />
    </Suspense>
  );
}

function LineView() {
  const params = useParams<{ route: string }>();
  const route = decodeURIComponent(params.route);
  const router = useRouter();
  const qp = useSearchParams();
  const { data: snap } = useSnapshot();
  const { data: topo } = useTopology();
  // Direction lives in the URL, so a bookmark, refresh, or the "view on map"
  // link all agree on it.
  const dir: Dir = qp.get("dir") === "south" ? "south" : "north";
  const setDir = (d: Dir) => router.replace(`/lines/${encodeURIComponent(route)}?dir=${d}`);

  // Canonical order from the trainer's published patterns (falling back to an
  // adjacency walk) when topology is loaded; otherwise every station that names
  // this line, alphabetized, so the page is still useful with no R2.
  const { stops, ordered } = useMemo<{ stops: string[]; ordered: boolean }>(() => {
    if (topo?.configured) {
      const s = orderTrip(topo.routeStops, topo.edges, route, dir);
      if (s.length) return { stops: s, ordered: true };
    }
    if (!snap) return { stops: [], ordered: false };
    const alpha = Object.values(snap.stations)
      .filter((s) => s.routes_served.includes(route))
      .sort((a, b) => a.name.localeCompare(b.name))
      .map((s) => `${s.gtfs_stop_id}${dir === "north" ? "N" : "S"}`);
    return { stops: alpha, ordered: false };
  }, [topo, snap, route, dir]);

  return (
    <div className="wrap">
      <PageHeader
        subtitle={
          <>
            <Link href="/lines">Lines</Link> · {stops.length} stations ·{" "}
            <Link href={`/map?route=${route}&dir=${dir}`}>view on map</Link>
          </>
        }
      />
      <div className="line-head">
        <RouteBullet snap={snap} route={route} size={40} />
        <h2>{route} line</h2>
        <div className="dir-toggle">
          <button className={dir === "north" ? "active" : ""} onClick={() => setDir("north")}>
            Northbound
          </button>
          <button className={dir === "south" ? "active" : ""} onClick={() => setDir("south")}>
            Southbound
          </button>
        </div>
      </div>

      {!snap && <div className="sub">loading…</div>}
      {snap && stops.length === 0 && (
        <div className="note">No stations found for this line and direction.</div>
      )}
      {snap && stops.length > 0 && (
        <ol className="stoplist">
          {stops.map((stop, i) => (
            <StopRow
              key={stop}
              snap={snap}
              route={route}
              dir={dir}
              stop={stop}
              // The segment leaving this stop, when the model judged it live.
              segStatus={snap.segment_flow?.segments[`${route}|${dir}|${stop}`]?.status ?? null}
              last={i === stops.length - 1}
              ordered={!!ordered}
            />
          ))}
        </ol>
      )}
    </div>
  );
}

function StopRow({
  snap,
  route,
  dir,
  stop,
  segStatus,
  last,
  ordered,
}: {
  snap: Snapshot;
  route: string;
  dir: Dir;
  stop: string;
  segStatus: "normal" | "disrupted" | null;
  last: boolean;
  ordered: boolean;
}) {
  const id = undirected(stop);
  const meta = snap.stations[id];
  const name = meta?.name ?? id;
  const flow = snap.station_flow?.stations[id] ?? null;
  const flowClass = flow ? (flow.status === "degraded" ? "disrupted" : "normal") : null;
  const others = (meta?.routes_served ?? []).filter((r) => r !== route);
  const adaLabel = meta?.ada === 1 ? "ADA" : meta?.ada === 2 ? "ADA partial" : null;

  return (
    <li className="stop">
      <span className={`stop-node ${flowClass ?? "unknown"}`} />
      {!last && (
        <span
          className={`stop-conn ${ordered ? segStatus ?? "topology" : "topology"}`}
          aria-hidden
        />
      )}
      <Link href={`/stations/${id}`} className="stop-body">
        <span className="stop-name">{name}</span>
        <span className="stop-meta">
          {meta?.borough && <span>{meta.borough}</span>}
          {adaLabel && <span className="ada-badge">{adaLabel}</span>}
          {others.slice(0, 6).map((r) => (
            <RouteBullet key={r} snap={snap} route={r} size={16} />
          ))}
        </span>
      </Link>
      {flow && (
        <span className={`cond ${flowClass}`}>
          {flow.status}
          {flow.status === "degraded" && flow.worst_recovery
            ? ` · ~${fmtMinutes(flow.worst_recovery.recovery_minutes)}`
            : ""}
        </span>
      )}
    </li>
  );
}
