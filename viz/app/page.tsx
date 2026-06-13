"use client";

import { useEffect, useMemo, useState } from "react";
import Nav from "./Nav";
import {
  fetchSnapshot,
  conditionRank,
  routeColor,
  routeLabel,
  fmtAgo,
  fmtMinutes,
} from "@/lib/feed";
import type { Snapshot, RouteStatus } from "@/lib/types";

const POLL_MS = 60_000;

export default function StatusPage() {
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [fetchedAt, setFetchedAt] = useState<number>(0);
  const [sel, setSel] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    const load = async () => {
      try {
        const s = await fetchSnapshot();
        if (!alive) return;
        setSnap(s);
        setErr(null);
        setFetchedAt(Math.floor(Date.now() / 1000));
      } catch (e) {
        if (alive) setErr((e as Error).message);
      }
    };
    load();
    const id = setInterval(load, POLL_MS);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  const routes = useMemo(() => {
    if (!snap) return [];
    return Object.values(snap.route_status).sort((a, b) => {
      const r = conditionRank(b.condition) - conditionRank(a.condition);
      if (r !== 0) return r;
      return a.route_id.localeCompare(b.route_id, undefined, { numeric: true });
    });
  }, [snap]);

  return (
    <div className="wrap">
      <div className="topbar">
        <h1>Momentarily</h1>
        <Nav />
      </div>
      <div className="sub">
        Live NYC MTA service status + HMM inference ·{" "}
        {snap ? (
          <>
            snapshot {fmtAgo(snap.generated_at, fetchedAt)} · refreshes every 60s
          </>
        ) : (
          "loading…"
        )}
      </div>

      {err && <div className="error">Failed to load feed: {err}</div>}

      {snap && (
        <>
          <SystemBanner snap={snap} />
          <FreshnessStrip snap={snap} now={fetchedAt} />
          <div className="grid">
            {routes.map((r) => (
              <RouteCard
                key={r.route_id}
                snap={snap}
                r={r}
                selected={sel === r.route_id}
                onClick={() => setSel(r.route_id)}
              />
            ))}
          </div>
        </>
      )}

      {snap && sel && snap.route_status[sel] && (
        <RouteDrawer
          snap={snap}
          r={snap.route_status[sel]}
          onClose={() => setSel(null)}
        />
      )}
    </div>
  );
}

function SystemBanner({ snap }: { snap: Snapshot }) {
  const s = snap.system;
  return (
    <div className="banner">
      <div className="label">{s.overall_label}</div>
      <div className="stat">
        <span className="k">Lines disrupted</span>
        <span className="v">{s.lines_disrupted_count}</span>
      </div>
      <div className="stat">
        <span className="k">Most degraded</span>
        <span className="v">{s.most_degraded_line ?? "—"}</span>
      </div>
      <div className="stat">
        <span className="k">Most recovered</span>
        <span className="v">{s.most_recovered_line ?? "—"}</span>
      </div>
      <div className="stat">
        <span className="k">Elevators out</span>
        <span className="v">{s.accessibility.elevators_out}</span>
      </div>
      <div className="stat">
        <span className="k">Escalators out</span>
        <span className="v">{s.accessibility.escalators_out}</span>
      </div>
    </div>
  );
}

const FRESH_FIELDS: [keyof Snapshot["freshness"], string][] = [
  ["subway_alerts", "Subway alerts"],
  ["ene", "Elevators/escalators"],
];

function FreshnessStrip({ snap, now }: { snap: Snapshot; now: number }) {
  return (
    <div className="freshness">
      {FRESH_FIELDS.map(([key, label]) => {
        const ts = snap.freshness[key];
        const age = ts == null ? null : now - ts;
        let cls = "off";
        if (age != null) {
          // alerts tick every 5m, E&E hourly — grade generously.
          cls = age < 600 ? "ok" : age < 3 * 3600 ? "warn" : "stale";
        }
        return (
          <span key={key}>
            <span className={`dot ${cls}`} />
            {label}: {fmtAgo(ts, now)}
          </span>
        );
      })}
    </div>
  );
}

function condClass(r: RouteStatus): string {
  if (!r.inference || r.inference.model_warming_up) return "warming";
  return r.condition || "unknown";
}

function RouteCard({
  snap,
  r,
  selected,
  onClick,
}: {
  snap: Snapshot;
  r: RouteStatus;
  selected: boolean;
  onClick: () => void;
}) {
  const inf = r.inference;
  const warming = !inf || inf.model_warming_up;
  return (
    <div className={`card${selected ? " sel" : ""}`} onClick={onClick}>
      <div className="card-head">
        <span
          className="bullet"
          style={{ background: routeColor(snap, r.route_id) }}
        >
          {routeLabel(snap, r.route_id)}
        </span>
        <span className={`cond ${condClass(r)}`}>
          {warming ? "warming up" : r.condition}
        </span>
      </div>

      {inf && !warming && (
        <div
          className="pbar"
          title={`normal ${(inf.p_normal * 100).toFixed(1)}% · disrupted ${(
            inf.p_disrupted * 100
          ).toFixed(1)}% · suspended ${(inf.p_suspended * 100).toFixed(1)}%`}
        >
          <span className="pn" style={{ width: `${inf.p_normal * 100}%` }} />
          <span className="pd" style={{ width: `${inf.p_disrupted * 100}%` }} />
          <span className="ps" style={{ width: `${inf.p_suspended * 100}%` }} />
        </div>
      )}

      <div className="meta">
        <span>
          {r.primary_alert_type ?? (r.alerts.length ? "alert" : "good service")}
        </span>
        <span>
          {inf && !warming && inf.is_disrupted
            ? inf.recovery_indeterminate
              ? "recovery: indeterminate"
              : `~${fmtMinutes(inf.recovery_minutes)}`
            : ""}
        </span>
      </div>
    </div>
  );
}

function RouteDrawer({
  snap,
  r,
  onClose,
}: {
  snap: Snapshot;
  r: RouteStatus;
  onClose: () => void;
}) {
  const inf = r.inference;
  return (
    <aside className="drawer">
      <button className="close" onClick={onClose} aria-label="close">
        ×
      </button>
      <h2>
        <span
          className="bullet"
          style={{ background: routeColor(snap, r.route_id) }}
        >
          {routeLabel(snap, r.route_id)}
        </span>
        <span className={`cond ${condClass(r)}`}>
          {!inf || inf.model_warming_up ? "warming up" : r.condition}
        </span>
      </h2>

      <div className="kv">
        <span className="k">Label</span>
        <span className="v">{r.label}</span>
        <span className="k">Category</span>
        <span className="v">{r.category}</span>
        <span className="k">Primary alert</span>
        <span className="v">{r.primary_alert_type ?? "—"}</span>
      </div>

      {inf && inf.model_warming_up && (
        <div className="warnbox">Model warming up — inference not yet reliable.</div>
      )}

      {inf && !inf.model_warming_up && (
        <>
          <div className="section-title">Regime probabilities</div>
          <div
            className="pbar"
            style={{ height: 10 }}
            title="normal / disrupted / suspended"
          >
            <span className="pn" style={{ width: `${inf.p_normal * 100}%` }} />
            <span className="pd" style={{ width: `${inf.p_disrupted * 100}%` }} />
            <span className="ps" style={{ width: `${inf.p_suspended * 100}%` }} />
          </div>
          <div className="kv">
            <span className="k">P(normal)</span>
            <span className="v">{(inf.p_normal * 100).toFixed(2)}%</span>
            <span className="k">P(disrupted)</span>
            <span className="v">{(inf.p_disrupted * 100).toFixed(2)}%</span>
            <span className="k">P(suspended)</span>
            <span className="v">{(inf.p_suspended * 100).toFixed(2)}%</span>
            <span className="k">Regime age</span>
            <span className="v">
              {fmtMinutes(inf.regime_age_seconds / 60)}
            </span>
          </div>

          <div className="section-title">Recovery forecast</div>
          {inf.recovery_indeterminate ? (
            <div className="warnbox">
              Indeterminate — regime too persistent to bound recovery.
            </div>
          ) : (
            <div className="kv">
              <span className="k">Median</span>
              <span className="v">{fmtMinutes(inf.recovery_minutes)}</span>
              <span className="k">IQR (25–75%)</span>
              <span className="v">
                {fmtMinutes(inf.recovery_minutes_low)} –{" "}
                {fmtMinutes(inf.recovery_minutes_high)}
              </span>
              <span className="k">P(normal in 30m)</span>
              <span className="v">{(inf.p_normal_in_30min * 100).toFixed(0)}%</span>
              <span className="k">P(normal in 60m)</span>
              <span className="v">{(inf.p_normal_in_60min * 100).toFixed(0)}%</span>
              <span className="k">P(normal in 120m)</span>
              <span className="v">
                {(inf.p_normal_in_120min * 100).toFixed(0)}%
              </span>
            </div>
          )}
        </>
      )}

      <div className="section-title">By direction</div>
      <div className="kv">
        <span className="k">Northbound</span>
        <span className="v">
          {r.by_direction.northbound.primary_alert_type ??
            (r.by_direction.northbound.alerts.length ? "alert" : "good")}
        </span>
        <span className="k">Southbound</span>
        <span className="v">
          {r.by_direction.southbound.primary_alert_type ??
            (r.by_direction.southbound.alerts.length ? "alert" : "good")}
        </span>
      </div>

      {r.alerts.length > 0 && (
        <>
          <div className="section-title">Active alerts ({r.alerts.length})</div>
          <div className="alertlist">{r.alerts.join(", ")}</div>
        </>
      )}
    </aside>
  );
}
