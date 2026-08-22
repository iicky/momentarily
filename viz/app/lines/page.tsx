"use client";

import { useMemo } from "react";
import { useSnapshot } from "../useData";
import { PageHeader, RouteBullet } from "../ui";

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

  return (
    <div className="wrap">
      <PageHeader subtitle="Pick a line to see its stations and live service flow." />
      {error && <div className="error">Failed to load feed: {error}</div>}
      {!snap && !error && <div className="sub">loading…</div>}
      {snap && (
        <div className="line-grid">
          {routes.map((r) => (
            <RouteBullet key={r} snap={snap} route={r} size={44} href={`/lines/${r}`} />
          ))}
        </div>
      )}
    </div>
  );
}
