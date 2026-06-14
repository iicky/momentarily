// Schedule-reliability metrics for the deterministic planned-work resume path.
//
// Two instruments, neither overlapping the HMM calibration:
//
//   1. resume-churn — do announced planned-work windows hold? Across an alert's
//      archived versions we track each window's end (the resume time) and flag
//      windows whose end moved. The MTA almost never pulls a resume EARLIER, so
//      a pushed end is the main way "it's back" can be announced too soon.
//
//   2. adherence — did the line actually return to normal at the announced
//      resumes_at? We join schedule predictions (announced resume) to the
//      regime-transition stream (when the line truly went normal). This is the
//      only instrument for the silent-overrun case: the schedule said "back at
//      T" but reality dragged later.

import type { RouteTimeline } from "./calibration";
import type { PredictionRecord } from "./types";

/** First time strictly after `ts` that the route entered a normal regime.
 *  Local copy so this module only type-imports calibration (keeps the lib
 *  modules free of runtime cross-imports). */
function nextNormalStart(tl: RouteTimeline, ts: number): number | null {
  for (const s of tl.normalStarts) if (s > ts) return s;
  return null;
}

// --- resume-churn ---

/** One archived planned-work alert version, flattened for churn analysis. */
export interface AlertVersion {
  id: string;
  route: string; // representative (first informed) route
  alertType: string;
  observedAt: number;
  windows: { start: number; end: number | null }[];
}

/**
 * Flatten one archived alert object — `{observed_at, alert: <entity>}`, the
 * nested-Mercury shape the Worker writes to archive/alerts/ — into an
 * AlertVersion. Returns null for non-planned-work alerts (resume-churn only
 * concerns planned windows) or unparseable bodies.
 */
export function parseAlertVersion(body: unknown): AlertVersion | null {
  if (!body || typeof body !== "object") return null;
  const observedAt = (body as { observed_at?: unknown }).observed_at;
  const entity = (body as { alert?: unknown }).alert;
  if (typeof observedAt !== "number" || !entity || typeof entity !== "object") return null;

  const id = (entity as { id?: unknown }).id;
  if (typeof id !== "string" || !id.startsWith("lmm:planned_work:")) return null;

  const inner = (entity as { alert?: unknown }).alert;
  if (!inner || typeof inner !== "object") return null;

  const mercury = (inner as { "transit_realtime.mercury_alert"?: unknown })[
    "transit_realtime.mercury_alert"
  ];
  const alertType =
    mercury && typeof mercury === "object"
      ? (mercury as { alert_type?: unknown }).alert_type
      : undefined;
  if (typeof alertType !== "string") return null;

  const periodsRaw = (inner as { active_period?: unknown }).active_period;
  const windows: { start: number; end: number | null }[] = [];
  if (Array.isArray(periodsRaw)) {
    for (const p of periodsRaw) {
      if (!p || typeof p !== "object") continue;
      const start = (p as { start?: unknown }).start;
      const end = (p as { end?: unknown }).end;
      if (typeof start !== "number") continue;
      windows.push({ start, end: typeof end === "number" ? end : null });
    }
  }
  if (windows.length === 0) return null;

  const informed = (inner as { informed_entity?: unknown }).informed_entity;
  let route = "?";
  if (Array.isArray(informed)) {
    for (const e of informed) {
      const r = e && typeof e === "object" ? (e as { route_id?: unknown }).route_id : undefined;
      if (typeof r === "string") {
        route = r;
        break;
      }
    }
  }

  return { id, route, alertType, observedAt, windows };
}

export interface ResumeChurnWindow {
  id: string;
  route: string;
  alertType: string;
  start: number;
  firstEnd: number;
  lastEnd: number;
  deltaMin: number; // (lastEnd - firstEnd)/60; >0 = pushed later
  status: "pushed" | "pulled" | "stable";
}

export interface ResumeChurnResult {
  windows: number; // windows observed in ≥2 versions (assessable)
  pushed: number;
  pulled: number;
  stable: number;
  pushedPct: number;
  pulledPct: number;
  pushMagnitudesMin: number[]; // sorted ascending, for the distribution
  byRoute: { route: string; windows: number; pushed: number }[];
  byAlertType: { alertType: string; windows: number; pushed: number }[];
}

// Match a window to a slot across versions if its start is within this of the
// slot's start — versions occasionally nudge the start by a few minutes.
const START_TOLERANCE_SEC = 1800;
// Ignore sub-minute end jitter when classifying push/pull.
const END_EPS_SEC = 60;

interface Slot {
  start: number;
  ends: number[]; // in version (observed_at) order
}

/**
 * Analyze how planned-work resume times (window ends) drift across an alert's
 * versions. Only windows seen in ≥2 versions are assessable for churn.
 */
export function resumeChurn(versions: AlertVersion[]): ResumeChurnResult {
  const byId = new Map<string, AlertVersion[]>();
  for (const v of versions) {
    const arr = byId.get(v.id) ?? [];
    arr.push(v);
    byId.set(v.id, arr);
  }

  const out: ResumeChurnWindow[] = [];
  for (const [id, vers] of byId) {
    vers.sort((a, b) => a.observedAt - b.observedAt);
    const route = vers[0].route;
    const alertType = vers[0].alertType;
    const slots: Slot[] = [];
    for (const v of vers) {
      for (const w of v.windows) {
        if (w.end == null) continue;
        let slot = slots.find(
          (s) => Math.abs(s.start - w.start) <= START_TOLERANCE_SEC,
        );
        if (!slot) {
          slot = { start: w.start, ends: [] };
          slots.push(slot);
        }
        slot.ends.push(w.end);
      }
    }
    for (const s of slots) {
      if (s.ends.length < 2) continue; // can't assess churn from one observation
      const firstEnd = s.ends[0];
      const lastEnd = s.ends[s.ends.length - 1];
      const delta = lastEnd - firstEnd;
      const status =
        delta > END_EPS_SEC ? "pushed" : delta < -END_EPS_SEC ? "pulled" : "stable";
      out.push({
        id,
        route,
        alertType,
        start: s.start,
        firstEnd,
        lastEnd,
        deltaMin: delta / 60,
        status,
      });
    }
  }

  const pushed = out.filter((w) => w.status === "pushed");
  const pulled = out.filter((w) => w.status === "pulled");
  const stable = out.filter((w) => w.status === "stable");
  const windows = out.length;

  const tally = (key: (w: ResumeChurnWindow) => string) => {
    const m = new Map<string, { windows: number; pushed: number }>();
    for (const w of out) {
      const k = key(w);
      const e = m.get(k) ?? { windows: 0, pushed: 0 };
      e.windows += 1;
      if (w.status === "pushed") e.pushed += 1;
      m.set(k, e);
    }
    return m;
  };

  const byRoute = [...tally((w) => w.route).entries()]
    .map(([route, v]) => ({ route, ...v }))
    .sort((a, b) => b.pushed - a.pushed || b.windows - a.windows);
  const byAlertType = [...tally((w) => w.alertType).entries()]
    .map(([alertType, v]) => ({ alertType, ...v }))
    .sort((a, b) => b.pushed - a.pushed || b.windows - a.windows);

  return {
    windows,
    pushed: pushed.length,
    pulled: pulled.length,
    stable: stable.length,
    pushedPct: windows ? pushed.length / windows : NaN,
    pulledPct: windows ? pulled.length / windows : NaN,
    pushMagnitudesMin: pushed.map((w) => w.deltaMin).sort((a, b) => a - b),
    byRoute,
    byAlertType,
  };
}

// --- adherence ---

export interface AdherencePoint {
  route: string;
  resumeAt: number; // announced resumes_at
  actualNormalAt: number; // observed return-to-normal
  errorMin: number; // (actual - announced)/60; >0 = overran (back late)
}

export interface AdherenceResult {
  points: AdherencePoint[];
  n: number;
  medianErrorMin: number; // signed — positive median = systematic overrun
  overrunPct: number; // fraction back later than announced (beyond tolerance)
  onTimePct: number; // fraction within tolerance
  censored: number; // schedule resumes whose actual return isn't observable yet
}

const ADHERENCE_TOLERANCE_MIN = 10;

/**
 * Join schedule predictions to actual returns-to-normal. Dedupe to one point
 * per (route, announced resume) — the latest prediction before the resume, the
 * most current announcement — then compare to when the line actually went
 * normal. Resumes whose actual return isn't yet in the transition stream are
 * censored, not scored.
 */
export function adherence(
  predictions: PredictionRecord[],
  timelines: Map<string, RouteTimeline>,
): AdherenceResult {
  // Latest prediction per (route, resumes_at).
  const latest = new Map<string, PredictionRecord>();
  for (const pr of predictions) {
    if (pr.recovery_source !== "schedule" || pr.resumes_at == null) continue;
    const k = `${pr.route}|${pr.resumes_at}`;
    const cur = latest.get(k);
    if (!cur || pr.ts > cur.ts) latest.set(k, pr);
  }

  const points: AdherencePoint[] = [];
  let censored = 0;
  for (const pr of latest.values()) {
    const tl = timelines.get(pr.route);
    const resumeAt = pr.resumes_at as number;
    if (!tl) {
      censored += 1;
      continue;
    }
    // Actual return: next normal regime start after the prediction tick. null
    // when the line never flipped back in-window (e.g. not_scheduled kept the
    // filter normal the whole time — no transition to read).
    const actual = nextNormalStart(tl, pr.ts);
    if (actual == null || actual > tl.observedUntil) {
      censored += 1;
      continue;
    }
    points.push({
      route: pr.route,
      resumeAt,
      actualNormalAt: actual,
      errorMin: (actual - resumeAt) / 60,
    });
  }

  points.sort((a, b) => a.errorMin - b.errorMin);
  const n = points.length;
  const overrun = points.filter((p) => p.errorMin > ADHERENCE_TOLERANCE_MIN).length;
  const onTime = points.filter(
    (p) => Math.abs(p.errorMin) <= ADHERENCE_TOLERANCE_MIN,
  ).length;
  return {
    points,
    n,
    medianErrorMin: n ? points[Math.floor(n / 2)].errorMin : NaN,
    overrunPct: n ? overrun / n : NaN,
    onTimePct: n ? onTime / n : NaN,
    censored,
  };
}
