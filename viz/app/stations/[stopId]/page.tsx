"use client";

import { useMemo } from "react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useSnapshot, useCoords, useTopology } from "../../useData";
import { PageHeader, RouteBullet } from "../../ui";
import { undirected } from "@/lib/stations";
import { Chip } from "../../models/ChartFrame";
import { fmtEta, fmtMinutes, fmtRiders, platformCrowding } from "@/lib/feed";
import type { PlatformCrowdingView } from "@/lib/feed";
import type { PlatformCrowding, PlatformCrowdingMethod, SegmentRecovery } from "@/lib/types";
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
    if (!topo || !snap) return [];
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

  // Destination labels for the two platforms, falling back to the compass word.
  const northLabel = coord?.north_label ?? "Northbound";
  const southLabel = coord?.south_label ?? "Southbound";

  // Movement through the station, rolled up per direction from the segments
  // touching it: disrupted if any of that direction's segments is, else moving
  // if any was judged at all, else no live read. This is the detail the header
  // chip rolls up — one place for the verdict, one for the per-direction read.
  const movement = useMemo(() => {
    const forDir = (d: "north" | "south") => {
      const segs = adj.filter((a) => a.direction === d);
      if (segs.length === 0) return null;
      return {
        dir: d,
        disrupted: segs.some((s) => s.status === "disrupted"),
        judged: segs.some((s) => s.status !== null),
      };
    };
    const north = forDir("north");
    const south = forDir("south");
    return [north, south].filter((m): m is NonNullable<typeof m> => m != null);
  }, [adj]);

  // A directional stop id to its station name, preferring live metadata, then
  // the coordinate file, and only falling back to the raw id when neither
  // carries the station — so the segment list and worst-segment line read as
  // places, not GTFS ids.
  const nameOf = (dirStop: string): string => {
    const u = undirected(dirStop);
    return snap?.stations[u]?.name ?? coords?.[u]?.name ?? u;
  };

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

          <div className="section-title">Live status</div>
          {movement.length > 0 ? (
            <ul className="movelist">
              {movement.map((m) => {
                const compass = m.dir === "north" ? "Northbound" : "Southbound";
                const label = m.dir === "north" ? northLabel : southLabel;
                const cls = m.disrupted ? "disrupted" : m.judged ? "normal" : "unknown";
                const text = m.disrupted ? "disrupted" : m.judged ? "moving" : "no live read";
                return (
                  <li key={m.dir} className="move-row">
                    <span className="move-dir">
                      {compass}
                      {label !== compass && <span className="move-to"> · toward {label}</span>}
                    </span>
                    <span className={`cond ${cls}`}>{text}</span>
                  </li>
                );
              })}
            </ul>
          ) : (
            <div className="note muted">No live movement judged through this station right now.</div>
          )}
          {flow && flow.status === "degraded" && (
            <div className="kv-block">
              <div className="kv">
                <span className="k">Worst deficit</span>
                <span className="v">{(flow.worst_deficit * 100).toFixed(0)}%</span>
              </div>
              {flow.worst_segment && (
                <div className="kv">
                  <span className="k">Worst segment</span>
                  <span className="v">
                    {nameOf(flow.worst_segment[0])} → {nameOf(flow.worst_segment[1])}
                  </span>
                </div>
              )}
              {flow.worst_recovery && <RecoveryLine rec={flow.worst_recovery} />}
            </div>
          )}
          {status && (status.elevators_total > 0 || status.escalators_total > 0) && (
            <div className="kv-block">
              {status.elevators_total > 0 && (
                <div className="kv">
                  <span className="k">Elevators out</span>
                  <span className="v">
                    {status.elevators_out}/{status.elevators_total}
                  </span>
                </div>
              )}
              {status.escalators_total > 0 && (
                <div className="kv">
                  <span className="k">Escalators out</span>
                  <span className="v">
                    {status.escalators_out}/{status.escalators_total}
                  </span>
                </div>
              )}
              {status.earliest_elevator_return != null && (
                <div className="kv">
                  <span className="k">Est. return</span>
                  <span className="v">
                    {fmtEta(status.earliest_elevator_return, Math.floor(Date.now() / 1000))}
                  </span>
                </div>
              )}
            </div>
          )}

          <div className="section-title crowd-title">
            Waiting riders
            <Chip title="Modelled from entry counts and train movement. Nothing counts people on a platform.">
              estimate
            </Chip>
          </div>
          <WaitingRiders
            pc={snap.platform_crowding}
            stopId={id}
            northLabel={northLabel}
            southLabel={southLabel}
          />

          <div className="section-title">Segments</div>
          {adj.length === 0 ? (
            <div className="note muted">No segments recorded for this station.</div>
          ) : (
            <ul className="seglist">
              {adj.map((a, i) => (
                <li key={`${a.route}${a.direction}${a.from}${i}`}>
                  <RouteBullet snap={snap} route={a.route} size={18} href={`/lines/${a.route}`} />
                  <span className="seg-dir">{a.direction}</span>
                  <span className="seg-path">
                    {a.incoming
                      ? `${nameOf(a.from)} → ${name}`
                      : `${name} → ${nameOf(a.to)}`}
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

          {/* The reference sheet, demoted below the live answer: identifiers and
              static facts a reader looks up rather than leads with. */}
          <div className="station-facts">
            <div className="station-cols">
              <section>
                <div className="section-title">Station</div>
                <div className="kv">
                  <span className="k">Lines</span>
                  <span className="v station-lines">
                    {(meta?.routes_served ?? coord?.daytime_routes ?? []).map((r) => (
                      <RouteBullet key={r} snap={snap} route={r} size={20} href={`/lines/${r}`} />
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
                      <span className="k">Northbound</span>
                      <span className="v">{meta?.ada_northbound ? "step-free" : "—"}</span>
                    </div>
                    <div className="kv">
                      <span className="k">Southbound</span>
                      <span className="v">{meta?.ada_southbound ? "step-free" : "—"}</span>
                    </div>
                  </>
                )}
              </section>
            </div>
          </div>
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

// Estimated riders waiting on this station's two platforms.
//
// The whole point of the section is that this is modelled, not counted, so the
// assumption it rests on is on the page in plain English rather than behind a
// hover: the ridership feed counts fare swipes per COMPLEX, nothing says which
// platform a rider walked to, and the even split across served platforms is
// therefore ours, not the MTA's. A platform we cannot estimate says so and says
// why — it never renders as an empty platform.
function WaitingRiders({
  pc,
  stopId,
  northLabel,
  southLabel,
}: {
  pc: PlatformCrowding | null;
  stopId: string;
  northLabel: string;
  southLabel: string;
}) {
  if (!pc)
    return (
      <div className="note muted">
        No crowding estimate in this snapshot — the surface is absent for every
        platform, not just this station. It appears once the feed carries both a
        ridership baseline and recent train movement.
      </div>
    );
  const now = Math.floor(Date.now() / 1000);
  const abstained = Object.values(pc.abstained).reduce((a, b) => a + b, 0);
  return (
    <>
      <div className="section-note crowd-note">
        Riders who have entered this complex since a train last cleared the
        platform. Nobody counts people on platforms: this is the complex&apos;s usual
        entry rate for this hour,{" "}
        {pc.method.split_basis === "scheduled_service_over_served_platforms" ? (
          <b>
            split across the platforms in service in proportion to how many trains
            the schedule runs at each (or evenly, at a complex the schedule
            doesn&apos;t fully cover this hour)
          </b>
        ) : (
          <b>split evenly across the platforms in service</b>
        )}{" "}
        — an assumption of ours, because the ridership feed counts fare swipes per
        complex and no feed says which platform a rider walked to. It also cannot
        see {pc.method.excludes.join(" or ")}: anyone transferring in for free, and
        everyone who just stepped off an arriving train, are missing from it.
      </div>
      <div className="kv-block">
        <WaitingRow label={northLabel} view={platformCrowding(pc, `${stopId}N`, now)} method={pc.method} />
        <WaitingRow label={southLabel} view={platformCrowding(pc, `${stopId}S`, now)} method={pc.method} />
      </div>
      <div className="chart-meta">
        <Chip>unit: riders</Chip>
        <Chip tone="muted" title="Recomputed on this page against your clock; the published figure is only true as of the snapshot.">
          {pc.method.formula}
        </Chip>
        <Chip tone="muted">split: {pc.method.split_basis.replace(/_/g, " ")}</Chip>
        <Chip tone="muted">excludes: {pc.method.excludes.join(", ")}</Chip>
        <Chip tone="muted">
          cap: {pc.method.max_gap_minutes}m gap · {pc.method.served_window_minutes}m service window
        </Chip>
        <Chip tone="muted" title={`${pc.method.baseline_window_start} → ${pc.method.baseline_window_end}`}>
          baseline {pc.method.baseline_window_start.slice(0, 10)} →{" "}
          {pc.method.baseline_window_end.slice(0, 10)}
        </Chip>
        <Chip tone="muted">{pc.n_platforms.toLocaleString()} platforms estimated</Chip>
        {abstained > 0 && <Chip tone="muted">{abstained.toLocaleString()} abstained</Chip>}
      </div>
    </>
  );
}

function WaitingRow({
  label,
  view,
  method,
}: {
  label: string;
  view: PlatformCrowdingView;
  method: PlatformCrowdingMethod;
}) {
  // The band is emphasis, so say in words what the emphasis means. Only the top
  // decile and percentile earn a chip; below that the number speaks for itself.
  const rank =
    view.estimated && view.band === "extreme"
      ? "busier than 99% of platforms"
      : view.estimated && view.band === "heavy"
        ? "busier than 90% of platforms"
        : null;
  return (
    <div className="crowd-row">
      <div className="crowd-head">
        <span className="crowd-dir">{label}</span>
        {rank && <Chip title="Against the measured spread of this estimate across the system: p50 = 28 riders, p90 = 86, p99 = 270.">{rank}</Chip>}
        {view.estimated ? (
          <span className={`crowd-count ${view.band}`}>{fmtRiders(view.riders)}</span>
        ) : (
          <span className="crowd-count none">no estimate</span>
        )}
      </div>
      <div className="crowd-inputs">
        {view.estimated
          ? `${Math.round(view.minutesSince)} min since a train cleared · assumed share of usual entries ${view.entriesPerMin.toFixed(1)}/min`
          : view.reason === "gap_exceeds_cap"
            ? `last train ${Math.round(view.minutesSince ?? 0)} min ago, past the ${method.max_gap_minutes}-minute cap — beyond it the count stops describing a crowd`
            : `no ridership baseline for this complex, or no train seen on this platform in the last ${method.served_window_minutes} min`}
      </div>
    </div>
  );
}
