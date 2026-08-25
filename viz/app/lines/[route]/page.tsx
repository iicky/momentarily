"use client";

import { Suspense, useMemo } from "react";
import Link from "next/link";
import { useParams, useRouter, useSearchParams } from "next/navigation";
import { useSnapshot, useTopology } from "../../useData";
import { PageHeader, RouteBullet } from "../../ui";
import { undirected, orderTrip } from "@/lib/stations";
import { fmtMinutes, fmtRiders, platformCrowding } from "@/lib/feed";
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

  // One clock for the whole rail. Each row re-derives its crowding estimate
  // against it rather than printing the published figure, which is already a
  // poll old by the time it renders and grows by 20-50 riders a minute at a
  // busy platform.
  const now = Math.floor(Date.now() / 1000);

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

      {snap?.platform_crowding && (
        <div className="section-note crowd-note">
          Rider counts are <b>estimates</b>, not measurements: each platform&apos;s
          assumed even share of its complex&apos;s usual entry rate for this hour,
          times how long it has been since a train cleared it. Transfers and exits
          are invisible to it. Full method on any station page.
        </div>
      )}

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
              now={now}
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
  now,
}: {
  snap: Snapshot;
  route: string;
  dir: Dir;
  stop: string;
  segStatus: "normal" | "disrupted" | null;
  last: boolean;
  ordered: boolean;
  now: number;
}) {
  const id = undirected(stop);
  const meta = snap.stations[id];
  const name = meta?.name ?? id;
  const flow = snap.station_flow?.stations[id] ?? null;
  const flowClass = flow ? (flow.status === "degraded" ? "disrupted" : "normal") : null;
  const others = (meta?.routes_served ?? []).filter((r) => r !== route);
  const adaLabel = meta?.ada === 1 ? "ADA" : meta?.ada === 2 ? "ADA partial" : null;
  // Secondary signal, so it rides in front of the flow badge rather than
  // replacing it, and stays muted unless the platform is in the top decile —
  // the badge keeps the only colour on the row. The estimate is for the
  // platform in the direction this page is showing.
  const crowd = platformCrowding(snap.platform_crowding, stop, now);

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
      {snap.platform_crowding &&
        (crowd.estimated ? (
          <span className={`stop-crowd ${crowd.band}`}>{fmtRiders(crowd.riders)}</span>
        ) : (
          <span className="stop-crowd">no estimate</span>
        ))}
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
