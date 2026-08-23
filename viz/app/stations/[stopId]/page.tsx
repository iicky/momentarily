"use client";

import { useMemo } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useSnapshot, useCoords, useTopology } from "../../useData";
import { PageHeader, RouteBullet } from "../../ui";
import { undirected } from "@/lib/stations";
import { fmtAgo, fmtMinutes } from "@/lib/feed";
import type { SegmentRecovery } from "@/lib/types";
import type { StationCoord } from "@/lib/stations";

interface AdjRow {
  route: string;
  direction: string;
  from: string;
  to: string;
  incoming: boolean;
  status: "normal" | "disrupted" | null;
}

export default function StationPage() {
  const params = useParams<{ stopId: string }>();
  const id = undirected(decodeURIComponent(params.stopId));
  const { data: snap } = useSnapshot();
  const { data: coords } = useCoords();
  const { data: topo } = useTopology();

  const meta = snap?.stations[id] ?? null;
  const coord: StationCoord | null = coords?.[id] ?? null;
  const flow = snap?.station_flow?.stations[id] ?? null;
  const status = meta?.station_complex_id
    ? snap?.station_status[meta.station_complex_id] ?? null
    : null;

  // Every judged segment touching this station, in either direction.
  const adj = useMemo<AdjRow[]>(() => {
    if (!topo?.configured || !snap) return [];
    const rows: AdjRow[] = [];
    for (const e of topo.edges) {
      const out = undirected(e.from) === id;
      const inc = undirected(e.to) === id;
      if (!out && !inc) continue;
      const live = snap.segment_flow?.segments[e.key]?.status ?? null;
      rows.push({
        route: e.route,
        direction: e.direction,
        from: e.from,
        to: e.to,
        incoming: inc && !out,
        status: live,
      });
    }
    return rows.sort((a, b) => a.route.localeCompare(b.route, undefined, { numeric: true }));
  }, [topo, snap, id]);

  const name = meta?.name ?? coord?.name ?? id;

  return (
    <div className="wrap">
      <PageHeader subtitle={<Link href="/lines">Lines</Link>} />
      {!snap && <div className="sub">loading…</div>}
      {snap && !meta && !coord && <div className="note">No station with id {id}.</div>}
      {snap && (meta || coord) && (
        <>
          <div className="line-head">
            <h2>{name}</h2>
            {flow && (
              <span className={`cond ${flow.status === "degraded" ? "disrupted" : "normal"}`}>
                {flow.status}
              </span>
            )}
          </div>

          <div className="station-cols">
            <section>
              <div className="section-title">Station</div>
              <div className="kv">
                <span className="k">Lines</span>
                <span className="v station-lines">
                  {(meta?.routes_served ?? coord?.daytime_routes ?? []).map((r) => (
                    <RouteBullet key={r} snap={snap} route={r} size={22} href={`/lines/${r}`} />
                  ))}
                </span>
              </div>
              {(meta?.borough ?? coord?.borough) && (
                <div className="kv">
                  <span className="k">Borough</span>
                  <span className="v">{meta?.borough ?? coord?.borough}</span>
                </div>
              )}
              {coord?.structure && (
                <div className="kv">
                  <span className="k">Structure</span>
                  <span className="v">{coord.structure}</span>
                </div>
              )}
              {coord?.line && (
                <div className="kv">
                  <span className="k">Trunk</span>
                  <span className="v">{coord.line}</span>
                </div>
              )}
              {coord && (
                <div className="kv">
                  <span className="k">Location</span>
                  <span className="v">
                    {coord.lat.toFixed(4)}, {coord.lon.toFixed(4)}
                  </span>
                </div>
              )}
              <div className="kv">
                <span className="k">GTFS id</span>
                <span className="v">{id}</span>
              </div>
            </section>

            <section>
              <div className="section-title">Accessibility</div>
              <div className="kv">
                <span className="k">ADA</span>
                <span className="v">
                  {meta?.ada === 1
                    ? "Fully accessible"
                    : meta?.ada === 2
                      ? "Partially accessible"
                      : "Not accessible"}
                </span>
              </div>
              {coord && (
                <>
                  <div className="kv">
                    <span className="k">{coord.north_label ?? "Northbound"}</span>
                    <span className="v">{meta?.ada_northbound ? "accessible" : "—"}</span>
                  </div>
                  <div className="kv">
                    <span className="k">{coord.south_label ?? "Southbound"}</span>
                    <span className="v">{meta?.ada_southbound ? "accessible" : "—"}</span>
                  </div>
                </>
              )}
              {status && (
                <>
                  <div className="kv">
                    <span className="k">Elevators</span>
                    <span className="v">
                      {status.elevators_out}/{status.elevators_total} out
                    </span>
                  </div>
                  <div className="kv">
                    <span className="k">Escalators</span>
                    <span className="v">
                      {status.escalators_out}/{status.escalators_total} out
                    </span>
                  </div>
                  {status.earliest_elevator_return != null && (
                    <div className="kv">
                      <span className="k">Est. return</span>
                      <span className="v">
                        {fmtAgo(status.earliest_elevator_return, Math.floor(Date.now() / 1000))}
                      </span>
                    </div>
                  )}
                </>
              )}
            </section>
          </div>

          <div className="section-title">Service flow</div>
          {flow ? (
            <div className="kv-block">
              <div className="kv">
                <span className="k">Status</span>
                <span className="v">{flow.status}</span>
              </div>
              <div className="kv">
                <span className="k">Worst deficit</span>
                <span className="v">{(flow.worst_deficit * 100).toFixed(0)}%</span>
              </div>
              {flow.worst_segment && (
                <div className="kv">
                  <span className="k">Worst segment</span>
                  <span className="v">
                    {undirected(flow.worst_segment[0])} → {undirected(flow.worst_segment[1])}
                  </span>
                </div>
              )}
              {flow.worst_recovery && <RecoveryLine rec={flow.worst_recovery} />}
            </div>
          ) : (
            <div className="note muted">No live movement judged through this station right now.</div>
          )}

          <div className="section-title">Segments</div>
          {topo && !topo.configured ? (
            <div className="note muted">
              Segment topology needs the R2 vault — run the viz under <code>murk exec</code>.
            </div>
          ) : adj.length === 0 ? (
            <div className="note muted">No segments recorded for this station.</div>
          ) : (
            <ul className="seglist">
              {adj.map((a, i) => (
                <li key={`${a.route}${a.direction}${a.from}${i}`}>
                  <RouteBullet snap={snap} route={a.route} size={18} href={`/lines/${a.route}`} />
                  <span className="seg-dir">{a.direction}</span>
                  <span className="seg-path">
                    {a.incoming
                      ? `${undirected(a.from)} → ${id}`
                      : `${id} → ${undirected(a.to)}`}
                  </span>
                  {a.status && <span className={`cond ${a.status}`}>{a.status}</span>}
                </li>
              ))}
            </ul>
          )}

          {status && status.alerts.length > 0 && (
            <>
              <div className="section-title">Alerts</div>
              <ul className="alertlist">
                {status.alerts.map((al, i) => (
                  <li key={i}>{al}</li>
                ))}
              </ul>
            </>
          )}
        </>
      )}
    </div>
  );
}

function RecoveryLine({ rec }: { rec: SegmentRecovery }) {
  return (
    <div className="kv">
      <span className="k">Expected recovery</span>
      <span className="v">
        {rec.recovery_indeterminate
          ? "indeterminate"
          : `~${fmtMinutes(rec.recovery_minutes)} (${fmtMinutes(rec.recovery_minutes_low)}–${fmtMinutes(rec.recovery_minutes_high)})`}
      </span>
    </div>
  );
}
