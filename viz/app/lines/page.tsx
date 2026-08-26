"use client";

import { useMemo } from "react";
import Link from "next/link";
import { useSnapshot } from "../useData";
import { PageHeader, RouteBullet } from "../ui";
import {
  conditionRank,
  serviceLead,
  conditionLabel,
  conditionClass,
  isServiceFlagged,
  isRunningHigh,
  supplyBand,
  fmtMinutes,
} from "@/lib/feed";
import type { Snapshot, RouteStatus } from "@/lib/types";

export default function LinesPage() {
  const { data: snap, error } = useSnapshot();

  // The station-serving lines are exactly the routes named in station metadata —
  // this drops the express/rerouted route ids (6X, 7X, FX, SS) that never own a
  // station of their own.
  const routes = useMemo(() => {
    if (!snap) return [];
    const seen = new Set<string>();
    for (const st of Object.values(snap.stations)) {
      for (const r of st.routes_served) seen.add(r);
    }
    return [...seen].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
  }, [snap]);

  // Split the board: lines the model is flagging right now lead as full-width
  // rows; everything else collapses into the bullet grid below. A line the
  // snapshot carries no status for (the S/SIR shuttle badges) can't be judged,
  // so it is never flagged — it rides in the grid with the healthy lines.
  const { flagged, healthy } = useMemo(() => {
    if (!snap) return { flagged: [] as string[], healthy: [] as string[] };
    const flagged: string[] = [];
    const healthy: string[] = [];
    for (const r of routes) {
      const rs = snap.route_status[r];
      if (rs && isServiceFlagged(rs)) flagged.push(r);
      else healthy.push(r);
    }
    // Worst first: suspended above disrupted above supply-only degradation, then
    // by line id so the order is stable poll to poll.
    flagged.sort((a, b) => {
      const ra = snap.route_status[a];
      const rb = snap.route_status[b];
      const d = conditionRank(rb.condition) - conditionRank(ra.condition);
      if (d !== 0) return d;
      return a.localeCompare(b, undefined, { numeric: true });
    });
    return { flagged, healthy };
  }, [snap, routes]);

  return (
    <div className="wrap">
      <PageHeader subtitle="Lines the model is flagging lead the board; everything else is running normally." />
      {error && <div className="error">Failed to load feed: {error}</div>}
      {!snap && !error && <div className="sub">loading…</div>}
      {snap && (
        <>
          {flagged.length > 0 && (
            <div className="triage">
              {flagged.map((r) => (
                <TriageRow key={r} snap={snap} route={r} r={snap.route_status[r]} />
              ))}
            </div>
          )}
          <div className="line-grid">
            {healthy.map((r) => (
              <RouteBullet key={r} snap={snap} route={r} size={44} href={`/lines/${r}`} />
            ))}
          </div>
        </>
      )}
    </div>
  );
}

// One flagged line, full width: bullet, the plain movement sentence, and the
// two numbers behind the flag — supply against this hour's usual, and the
// recovery estimate when the flow is disrupted. The whole row links to the line
// page. Supply is what the flag rests on when the flow itself reads normal (a
// line running well under the trains it usually would), so it always shows.
function TriageRow({ snap, route, r }: { snap: Snapshot; route: string; r: RouteStatus }) {
  const cls = conditionClass(r.condition);
  const inf = r.inference;
  const supplyTone = isRunningHigh(r) ? "high" : supplyBand(r);
  const recovery =
    inf && inf.is_disrupted && !inf.recovery_indeterminate
      ? `~${fmtMinutes(inf.recovery_minutes)} to recover`
      : null;
  return (
    <Link href={`/lines/${route}`} className={`triage-row ${cls}`}>
      <RouteBullet snap={snap} route={route} size={34} />
      <span className="triage-body">
        <span className="triage-lead">
          <span className={`cond ${cls}`}>{conditionLabel(r.condition)}</span>
          {serviceLead(r)}
        </span>
        <span className="triage-stats">
          {r.service_ratio != null && (
            <span>
              <b className={supplyTone}>{(r.service_ratio * 100).toFixed(0)}%</b> of usual trains
            </span>
          )}
          {recovery && <span>{recovery}</span>}
        </span>
      </span>
    </Link>
  );
}
