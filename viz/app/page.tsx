"use client";

import { useEffect, useMemo, useState } from "react";
import Nav from "./Nav";
import {
  fetchSnapshot,
  conditionRank,
  routeColor,
  routeLabel,
  alertHeadline,
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
        <h1>
          <span className="brand">
            <StateMark kind="logo" size={18} /> Momentarily
          </span>
        </h1>
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
  return r.condition || "unknown";
}

// Human-readable badge text for a route's published condition. Raw codes like
// "not_scheduled" would render with an underscore under the capitalize style.
function condLabel(r: RouteStatus): string {
  switch (r.condition) {
    case "normal":
      return "Normal";
    case "disrupted":
      return "Disrupted";
    case "suspended":
      return "Suspended";
    case "not_scheduled":
      return "Not scheduled";
    default:
      return "No live signal";
  }
}

// Two-car brand mark. The gap between the cars encodes delay and the bars drop
// height when suspended — geometry from docs/brand/assets/mark-*.svg. Colour is
// the state colour, inherited via currentColor (see .mark.* in globals.css).
type MarkKind = "normal" | "disrupted" | "suspended" | "muted" | "logo";

const MARK_BARS: Record<MarkKind, [number, number, number][]> = {
  // [x, y, height]; each bar is width 5, rx 2.5, on a 24 grid.
  normal: [[5, 3, 18], [14, 3, 18]],
  disrupted: [[3.02, 3, 18], [15.98, 3, 18]],
  suspended: [[1.5, 7, 10], [17.5, 7, 10]],
  muted: [[5, 3, 18], [14, 3, 18]],
  logo: [[5, 3, 18], [14, 3, 18]],
};

function StateMark({ kind, size = 20 }: { kind: MarkKind; size?: number }) {
  return (
    <svg
      className={`mark ${kind}`}
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="currentColor"
      aria-hidden="true"
    >
      {MARK_BARS[kind].map(([x, y, h], i) => (
        <rect key={i} x={x} y={y} width={5} height={h} rx={2.5} />
      ))}
    </svg>
  );
}

function markKind(r: RouteStatus): MarkKind {
  if (r.condition === "disrupted") return "disrupted";
  if (r.condition === "suspended") return "suspended";
  if (r.condition === "normal") return "normal";
  return "muted"; // not_scheduled / unknown
}

// Where the published condition came from — the fact that resolves "Normal
// despite a Delays alert": the status is observed from train movement, not the
// alert feed.
function sourceTag(r: RouteStatus): string {
  switch (r.condition_source) {
    case "movement":
      return "observed · train movement";
    case "schedule":
      return "from schedule";
    case "unknown":
      return "no live signal";
    default:
      return "model · HMM";
  }
}

// One plain-English line: lead with the observed movement status; the MTA alert
// is a second, separate clause, never a competing headline.
function headline(r: RouteStatus): { lead: string; alt?: string } {
  const alert =
    r.category !== "none" && r.primary_alert_type
      ? `The MTA has a “${r.primary_alert_type}” advisory up.`
      : undefined;
  switch (r.condition) {
    case "normal":
      return { lead: "Trains are moving normally.", alt: alert };
    case "disrupted":
      return { lead: "Trains are moving slowly or stalling.", alt: alert };
    case "suspended":
      return { lead: "No trains are running on this line.", alt: alert };
    case "not_scheduled":
      return { lead: "Not scheduled to run right now." };
    default:
      return {
        lead: "No live movement signal to confirm status.",
        alt: alert,
      };
  }
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
  return (
    <div className={`card${selected ? " sel" : ""}`} onClick={onClick}>
      <div className="card-head">
        <span
          className="bullet"
          style={{ background: routeColor(snap, r.route_id) }}
        >
          {routeLabel(snap, r.route_id)}
        </span>
        <StateMark kind={markKind(r)} />
        <span className={`cond ${condClass(r)}`}>
          {condLabel(r)}
        </span>
        {r.service_condition === "degraded" && (
          <span
            className="svc degraded"
            title={
              r.service_ratio != null
                ? `Assigned trains ${(r.service_ratio * 100).toFixed(
                    0,
                  )}% of the normal level for this hour`
                : "Fewer trains assigned than normal"
            }
          >
            supply low
          </span>
        )}
      </div>

      {inf && (
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
          {inf && inf.is_disrupted
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
        <StateMark kind={markKind(r)} size={22} />
        <span className={`cond ${condClass(r)}`}>
          {condLabel(r)}
        </span>
        <span className="src-tag">{sourceTag(r)}</span>
      </h2>

      {(() => {
        const h = headline(r);
        return (
          <div className="headline">
            {h.lead}
            {h.alt && <span className="alt"> {h.alt}</span>}
          </div>
        );
      })()}

      <div className="section-title">Alert (MTA)</div>
      <div className="kv">
        <span className="k">Status</span>
        <span className="v">{r.label}</span>
        <span className="k">Category</span>
        <span className="v">{r.category}</span>
        <span className="k">Primary alert</span>
        <span className="v">{r.primary_alert_type ?? "—"}</span>
      </div>

      <div className="section-title">Service (supply)</div>
      <div className="kv">
        <span className="k">Level</span>
        <span className="v">
          {r.service_condition === "degraded"
            ? "Degraded — fewer trains than normal"
            : r.service_condition === "normal"
              ? "Normal"
              : "Unknown"}
        </span>
        <span className="k">Assigned vs normal</span>
        <span className="v">
          {r.service_ratio != null
            ? `${(r.service_ratio * 100).toFixed(0)}%`
            : "—"}
        </span>
      </div>
      {r.service_condition === "degraded" && (
        <div className="note">
          Supply is a different axis from the status above: it counts how many
          trains are <b>assigned</b> vs normal for this hour. A line can run
          fewer trains (supply low) while the ones running still move fine, and
          the reverse.
        </div>
      )}

      {inf &&
        r.primary_alert_type === "No Scheduled Service" && (
          <div className="note">
            “No Scheduled Service” means this line just isn&apos;t scheduled to
            run right now (it doesn&apos;t run 24/7) — nothing&apos;s broken.
          </div>
        )}

      {inf && (
        <>
          <div className="section-title">Regime probabilities</div>
          <div className="section-note">
            The model&apos;s alert-aware read. The status above follows train
            movement, so when an advisory is up but trains keep moving, the two
            differ.
          </div>
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
              {inf.p_normal_in_30min == null ? (
                <span
                  className="v muted"
                  title="Withheld — this forecast came from a different arm than the one that set the published condition, so it's deliberately not shown rather than plotted on a mismatched scale."
                >
                  not forecast
                </span>
              ) : (
                <span className="v">{(inf.p_normal_in_30min * 100).toFixed(0)}%</span>
              )}
              <span className="k">P(normal in 60m)</span>
              {inf.p_normal_in_60min == null ? (
                <span
                  className="v muted"
                  title="Withheld — this horizon measured worse than naive persistence, so it's deliberately not forecast rather than shown as a number we know is wrong."
                >
                  not forecast
                </span>
              ) : (
                <span className="v">{(inf.p_normal_in_60min * 100).toFixed(0)}%</span>
              )}
              <span className="k">P(normal in 120m)</span>
              {inf.p_normal_in_120min == null ? (
                <span
                  className="v muted"
                  title="Withheld — this horizon measured worse than naive persistence, so it's deliberately not forecast rather than shown as a number we know is wrong."
                >
                  not forecast
                </span>
              ) : (
                <span className="v">
                  {(inf.p_normal_in_120min * 100).toFixed(0)}%
                </span>
              )}
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
          <ul className="alertlist">
            {r.alerts.map((id) => {
              const { type, text } = alertHeadline(snap, id);
              return (
                <li key={id}>
                  {text ? (
                    <>
                      {type && <span className="alert-type">{type}</span>}
                      {text}
                    </>
                  ) : (
                    id
                  )}
                </li>
              );
            })}
          </ul>
        </>
      )}
    </aside>
  );
}
