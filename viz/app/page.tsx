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
  fmtProb,
  supplyBand,
  supplyBars,
  SUPPLY_DEGRADE_RATIO,
  SUPPLY_RECOVER_RATIO,
  isRunningHigh,
  conditionLabel,
  conditionLead,
  conditionClass,
} from "@/lib/feed";
import type { SupplyBand } from "@/lib/feed";
import type { Snapshot, RouteStatus, Inference } from "@/lib/types";

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

// Standard subway-car silhouette, drawn on the same 24 grid and currentColor as
// the flow mark so the two sit together without a font or emoji dependency. One
// even-odd path: rounded car body, windshield knocked out of it, two headlights
// knocked out below, then a pair of feet.
function TrainIcon({ size = 13 }: { size?: number }) {
  return (
    <svg
      className="tmark"
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="currentColor"
      aria-hidden="true"
    >
      <path
        fillRule="evenodd"
        d="M8.5 1.5h7A4.5 4.5 0 0 1 20 6v10a4.5 4.5 0 0 1-4.5 4.5h-7A4.5 4.5 0 0 1 4 16V6a4.5 4.5 0 0 1 4.5-4.5Zm.5 4h6a1.5 1.5 0 0 1 1.5 1.5v3.5a1.5 1.5 0 0 1-1.5 1.5H9a1.5 1.5 0 0 1-1.5-1.5V7A1.5 1.5 0 0 1 9 5.5Zm.6 9.4a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5Zm4.8 0a1.25 1.25 0 1 0 0 2.5 1.25 1.25 0 0 0 0-2.5Z"
      />
      <rect x={6.5} y={21} width={3} height={1.8} rx={0.9} />
      <rect x={14.5} y={21} width={3} height={1.8} rx={0.9} />
    </svg>
  );
}

// Three-bar level glyph — how many trains are out, at a glance, deliberately a
// different shape from the two-car flow mark so the two axes are never read as
// one. Bars light 1/2/3 by band; heights rise left to right so a single lit bar
// reads as "few trains" without needing colour. Over the norm, all three light
// and the tallest overshoots its slot, so the meter reads as driven past full
// rather than merely full — the one thing "3 of 3" could not say on its own. A
// capping spike was tried first and is invisible at a 13px card glyph.
const SUPPLY_BAR_GEOM: [number, number][] = [
  // [y, height] on a 24 grid; x is 2 + i * 8, width 4.
  [15, 7],
  [10, 12],
  [5, 17],
];
const SUPPLY_BAR_HIGH: [number, number] = [0, 22];

function SupplyMark({
  band,
  runningHigh,
  size = 14,
}: {
  band: SupplyBand;
  runningHigh: boolean;
  size?: number;
}) {
  const lit = runningHigh ? 3 : supplyBars(band);
  return (
    <svg
      className={`smark ${band}${runningHigh ? " high" : ""}`}
      viewBox="0 0 24 24"
      width={size}
      height={size}
      aria-hidden="true"
    >
      {SUPPLY_BAR_GEOM.map((slot, i) => {
        const [y, h] =
          runningHigh && i === SUPPLY_BAR_GEOM.length - 1
            ? SUPPLY_BAR_HIGH
            : slot;
        return (
          <rect
            key={i}
            className={i < lit ? "lit" : "dim"}
            x={2 + i * 8}
            y={y}
            width={4}
            height={h}
            rx={2}
          />
        );
      })}
    </svg>
  );
}

// Train icon beside the bars. This pair is the axis's whole identity on a card —
// no pill, no wording — which is what lets it ride every card without competing
// with the condition badge, and lets the word "supply" leave the copy entirely.
function SupplyGlyph({
  band,
  runningHigh,
  size = 14,
}: {
  band: SupplyBand;
  runningHigh: boolean;
  size?: number;
}) {
  return (
    <span className={`supply-glyph ${runningHigh ? "high" : band}`}>
      <TrainIcon size={size - 1} />
      <SupplyMark band={band} runningHigh={runningHigh} size={size} />
    </span>
  );
}

// What the glyph means, spelled out on hover — the only place the numbers appear
// on a card now that the pill is gone.
function supplyTitle(ratio: number | null): string {
  if (ratio == null) return "No reading for how many trains are running";
  return `${(ratio * 100).toFixed(0)}% of the trains this line usually runs at this hour`;
}

// Fixed 0–200% scale. The ratio routinely lands well above 100% (a route can be
// assigned twice its hourly median), and a scale that stretched to fit would
// move the threshold ticks from route to route — the one thing the ticks exist
// to hold still. Anything past the ceiling shows an overflow arrow instead.
const SUPPLY_SCALE_MAX = 2;

// The whole axis in one block: the magnitude, where it sits against the two
// thresholds that actually flip it, and what "usual" is. Everything a reader
// needs is stated exactly once — the earlier version said "fewer trains than
// usual" in four places and still never said where the denominator came from.
function SupplyMeter({
  ratio,
  band,
  runningHigh,
  lowRatio,
  highRatio,
}: {
  ratio: number;
  band: SupplyBand;
  runningHigh: boolean;
  lowRatio: number | null;
  highRatio: number | null;
}) {
  const pos = (v: number): string =>
    `${Math.min(v / SUPPLY_SCALE_MAX, 1) * 100}%`;
  // Running notably high, the whole block — glyph, number, fill — takes the
  // accent, so the three never disagree about what the reading means.
  const tone = runningHigh ? "high" : band;
  return (
    <div className="supply">
      <div className="supply-head">
        <SupplyGlyph band={band} runningHigh={runningHigh} size={17} />
        <b className={`supply-pct ${tone}`}>{(ratio * 100).toFixed(0)}%</b>
        <span className="supply-of">of usual</span>
      </div>
      <div className="supply-track">
        <span className={`supply-fill ${tone}`} style={{ width: pos(ratio) }} />
        <i className="supply-tick" style={{ left: pos(SUPPLY_DEGRADE_RATIO) }} />
        <i className="supply-tick" style={{ left: pos(SUPPLY_RECOVER_RATIO) }} />
        <i className="supply-tick norm" style={{ left: pos(1) }} />
        {lowRatio != null && (
          <i className="supply-tick range" style={{ left: pos(lowRatio) }} />
        )}
        {highRatio != null && (
          <i className="supply-tick range" style={{ left: pos(highRatio) }} />
        )}
        {ratio > SUPPLY_SCALE_MAX && <span className="supply-over">›</span>}
      </div>
      <div className="supply-scale">
        <span style={{ left: pos(SUPPLY_DEGRADE_RATIO) }}>50%</span>
        <span style={{ left: pos(SUPPLY_RECOVER_RATIO) }}>80%</span>
        <span style={{ left: pos(1) }}>100%</span>
      </div>
      <div className="supply-note">
        <b>Usual</b> is the median number of trains this line runs in this hour,
        taken from our archive of the MTA feed. Weekdays and weekends count
        separately. The marked band is where this hour normally sits. Two
        readings under 50% flag the line, and two over 80% clear it.
      </div>
    </div>
  );
}

// One plain-English line: lead with the observed movement status; the MTA alert
// is a second, separate clause, never a competing headline.
function headline(r: RouteStatus): { lead: string; alt?: string } {
  const alert =
    r.category !== "none" && r.primary_alert_type
      ? `The MTA has a “${r.primary_alert_type}” advisory up.`
      : undefined;
  return {
    lead: conditionLead(r.condition),
    // Not-scheduled is a plan, not an incident, so it never carries the alert.
    alt: r.condition === "not_scheduled" ? undefined : alert,
  };
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
  const band = supplyBand(r);
  const runningHigh = isRunningHigh(r);
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
        <span className={`cond ${conditionClass(r.condition)}`}>{conditionLabel(r.condition)}</span>
      </div>

      {inf && (
        <div
          className="pbar"
          title={`normal ${fmtProb(inf.p_normal)} · disrupted ${fmtProb(
            inf.p_disrupted
          )} · suspended ${fmtProb(inf.p_suspended)}`}
        >
          <span className="pn" style={{ width: `${inf.p_normal * 100}%` }} />
          <span className="pd" style={{ width: `${inf.p_disrupted * 100}%` }} />
          <span className="ps" style={{ width: `${inf.p_suspended * 100}%` }} />
        </div>
      )}

      <div className="meta">
        <span className="meta-label">
          {r.primary_alert_type ?? (r.alerts.length ? "alert" : "good service")}
        </span>
        <span className="meta-right">
          {inf && inf.is_disrupted
            ? inf.recovery_indeterminate
              ? "recovery: indeterminate"
              : `~${fmtMinutes(inf.recovery_minutes)}`
            : ""}
          <span className="card-trains" title={supplyTitle(r.service_ratio)}>
            <SupplyGlyph band={band} runningHigh={runningHigh} size={13} />
            {runningHigh && r.service_ratio != null && (
              <b className="trains-pct">
                {(r.service_ratio * 100).toFixed(0)}%
              </b>
            )}
          </span>
        </span>
      </div>
    </div>
  );
}

// The recovery numbers mean four different things depending on which arm set the
// published condition, and one heading asked the wrong question for three of
// them:
//   - published normal: movementRecovery's normal branch returns 0/0/0 for the
//     median/IQR and p_normal_in_30 is P(STAYS normal). "Median —" beside
//     "P(normal in 30m) 93%" read as a broken forecast; it's a hold-time.
//   - published unknown: we declined to judge the condition, so the worker
//     withholds every number. Dashes read as failure, not as abstention.
//   - schedule arm: a deterministic countdown to the MTA's announced window
//     end, so median == q25 == q75 by construction. Printing it as an IQR
//     implied a distribution that doesn't exist.
//   - otherwise: a real time-to-normal estimate off the dwell curve.
function RecoveryBlock({ r, inf }: { r: RouteStatus; inf: Inference }) {
  if (r.condition === "unknown") {
    return (
      <>
        <div className="section-title">Outlook</div>
        <div className="section-note">
          No forecast. We have no live read on this line right now, so we do not
          call its status or its return.
        </div>
      </>
    );
  }

  if (r.condition === "normal") {
    return (
      <>
        <div className="section-title">Stability outlook</div>
        <div className="section-note">
          Nothing to recover from. This is the chance the line keeps moving
          normally.
        </div>
        {inf.p_normal_in_30min == null ? (
          <div className="section-note">
            No reading yet for how long this line normally holds up.
          </div>
        ) : (
          <div className="kv">
            <span className="k">P(stays normal in 30m)</span>
            <span className="v">{fmtProb(inf.p_normal_in_30min)}</span>
          </div>
        )}
      </>
    );
  }

  // The schedule arm answers "when does the announced window end" — for planned
  // work mid-service and for a line that simply isn't running right now. Its
  // median/q25/q75 are the same number by construction, so it's a countdown.
  if (r.condition === "not_scheduled" || inf.recovery_source === "schedule") {
    const announced = inf.recovery_source === "schedule";
    return (
      <>
        <div className="section-title">Scheduled return</div>
        <div className="section-note">
          {announced
            ? "The MTA announced when this work ends. This is a countdown, not an estimate."
            : "The alert does not say when this line starts running again."}
        </div>
        {announced && (
          <div className="kv">
            <span className="k">
              {inf.overdue ? "Announced end" : "Resumes in"}
            </span>
            <span className="v">
              {inf.overdue
                ? "passed, alert still up"
                : fmtMinutes(inf.recovery_minutes)}
            </span>
          </div>
        )}
      </>
    );
  }

  // Two different silences, and one message claimed both. The movement arm sets
  // indeterminate when the regime outlived every dwell it measured OR when the
  // fitted curve's median hits the clamp (snapshot.ts:974), so the copy names
  // the ceiling that recovery_minutes carries instead of asserting which one
  // fired. The hmm arm is not a bounding failure at all: the forecast is
  // withheld because it describes a different regime than the published one
  // (snapshot.ts:855). Three separate things send us here — no movement clock,
  // a state the movement split was never measured for (suspended), or no
  // trained curve for the cell — so the copy names none of them.
  if (inf.recovery_indeterminate) {
    return (
      <>
        <div className="section-title">Recovery forecast</div>
        <div className="warnbox">
          {inf.recovery_source === "movement"
            ? `No estimate. Recovery runs past the ${fmtMinutes(
                inf.recovery_minutes,
              )} we forecast ahead.`
            : "No estimate. The model has no forecast that matches the status above."}
        </div>
      </>
    );
  }

  return (
    <>
      <div className="section-title">Recovery forecast</div>
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
            title="Not shown. This number came from a different part of the model than the status above, so the two are not on the same scale."
          >
            not forecast
          </span>
        ) : (
          <span className="v">{fmtProb(inf.p_normal_in_30min)}</span>
        )}
        <span className="k">P(normal in 60m)</span>
        {inf.p_normal_in_60min == null ? (
          <span
            className="v muted"
            title="Not shown. This far out the model scored worse than simply assuming nothing changes."
          >
            not forecast
          </span>
        ) : (
          <span className="v">{fmtProb(inf.p_normal_in_60min)}</span>
        )}
        <span className="k">P(normal in 120m)</span>
        {inf.p_normal_in_120min == null ? (
          <span
            className="v muted"
            title="Not shown. This far out the model scored worse than simply assuming nothing changes."
          >
            not forecast
          </span>
        ) : (
          <span className="v">{fmtProb(inf.p_normal_in_120min)}</span>
        )}
      </div>
    </>
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
  const band = supplyBand(r);
  // The alert feed's per-direction split, as the rows would render it, next to
  // the route-level value they are compared against. Compared as displayed, so
  // a row survives only when it puts something new on screen. "no alerts"
  // rather than "good" deliberately: these read the alert feed, which cannot
  // see whether trains are moving on that side.
  const { northbound, southbound } = r.by_direction;
  const primary = r.primary_alert_type ?? "—";
  const nb =
    northbound.primary_alert_type ??
    (northbound.alerts.length ? "alert" : "no alerts");
  const sb =
    southbound.primary_alert_type ??
    (southbound.alerts.length ? "alert" : "no alerts");
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
        <span className={`cond ${conditionClass(r.condition)}`}>
          {conditionLabel(r.condition)}
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

      <div className="section-title">MTA alerts</div>
      {r.alerts.length === 0 ? (
        <div className="section-note">
          The MTA has posted nothing for this line.
        </div>
      ) : (
        <div className="kv">
          <span className="k">Primary alert</span>
          <span className="v">{primary}</span>
          {/* Alert feed only. A direction shows up only when it differs from
              the route-level value above — a side that just repeats the primary
              alert adds nothing, so what survives here is the news: which
              direction is clear, or carrying something else. These rows never
              see train movement, so the badge can say disrupted while the MTA
              has posted nothing on that side. A real per-direction movement
              read needs segment_flow coverage, which currently publishes only a
              handful of cells system-wide. */}
          {nb !== primary && (
            <>
              <span className="k">Northbound</span>
              <span className="v">{nb}</span>
            </>
          )}
          {sb !== primary && (
            <>
              <span className="k">Southbound</span>
              <span className="v">{sb}</span>
            </>
          )}
        </div>
      )}

      <div className="section-title">Trains running</div>
      <div className="section-note">
        How many trains are out, not how well they move. A line can run few
        trains that all move fine, or a full set that crawls.
      </div>
      {r.service_ratio != null ? (
        <SupplyMeter
          ratio={r.service_ratio}
          band={band}
          runningHigh={isRunningHigh(r)}
          lowRatio={r.service_low_ratio}
          highRatio={r.service_high_ratio}
        />
      ) : (
        <div className="section-note">
          No reading. We have no recent train count for this line, or nothing to
          compare it against for this hour.
        </div>
      )}

      {inf &&
        r.primary_alert_type === "No Scheduled Service" && (
          <div className="note">
            “No Scheduled Service” means this line does not run at this hour.
            Nothing is broken.
          </div>
        )}

      {inf && (
        <>
          <div className="section-title">Predicted status</div>
          <div className="section-note">
            What the model infers is happening right now. It weighs alerts,
            train movement and train counts together, so it can differ from the
            status above, which follows train movement alone.
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
            <span className="v">{fmtProb(inf.p_normal)}</span>
            <span className="k">P(disrupted)</span>
            <span className="v">{fmtProb(inf.p_disrupted)}</span>
            <span className="k">P(suspended)</span>
            <span className="v">{fmtProb(inf.p_suspended)}</span>
            {/* This clock belongs to the model above: it restarts whenever the
                model's top state changes, which is often. The badge runs on the
                movement arm's own clock, so the label has to say whose age this
                is — swapping in the movement clock would leave this section
                timing a regime it does not show. */}
            <span className="k">Model regime age</span>
            <span className="v">
              {fmtMinutes(inf.regime_age_seconds / 60)}
            </span>
          </div>

          <RecoveryBlock r={r} inf={inf} />
        </>
      )}

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
