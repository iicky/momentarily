import type { PlatformCrowding, RouteStatus, Snapshot, Trains } from "./types";

// The public snapshot. Override with NEXT_PUBLIC_FEED_BASE to point at a local
// Worker or a staging feed.
export const FEED_BASE =
  process.env.NEXT_PUBLIC_FEED_BASE ?? "https://feed.momentarily.nyc";

export async function fetchSnapshot(): Promise<Snapshot> {
  const res = await fetch(`${FEED_BASE}/v1/snapshot.json`, { cache: "no-store" });
  if (!res.ok) throw new Error(`snapshot fetch failed: ${res.status}`);
  return res.json();
}

/** The train position surface, published beside the snapshot rather than in it.
 *
 * Three states, because the three are genuinely different and collapsing them
 * would put words in the feed's mouth:
 *   loading      not fetched yet. Only fetched when something asks for it —
 *                it is the largest object the dashboard reads.
 *   unavailable  a 404 (the object isn't published yet) or a transport
 *                failure. NOT an error banner and NOT an empty list: "no
 *                trains reported" and "no train report" are not the same
 *                claim.
 *   ready        a body, with its own observed_at. The Worker skips rewriting
 *                the object on a tick whose vehicle feed didn't decode, so a
 *                200 can still be stale and only observed_at says by how much.
 */
export type TrainsFeed =
  | { state: "loading" }
  | { state: "unavailable"; reason: string }
  | { state: "ready"; trains: Trains };

export async function fetchTrains(): Promise<TrainsFeed> {
  try {
    const res = await fetch(`${FEED_BASE}/v1/trains.json`, { cache: "no-store" });
    if (res.status === 404) {
      return { state: "unavailable", reason: "not published at this feed yet" };
    }
    if (!res.ok) {
      return { state: "unavailable", reason: `feed returned ${res.status}` };
    }
    return { state: "ready", trains: (await res.json()) as Trains };
  } catch (e) {
    // A 404 whose response carries no CORS header — which is what the live feed
    // serves today for an object that isn't published — rejects the fetch
    // before the status is readable. So this branch covers "not there yet" and
    // "couldn't get there" alike, and says only what it knows.
    return { state: "unavailable", reason: `not reachable (${(e as Error).message})` };
  }
}

const CONDITION_RANK: Record<string, number> = {
  suspended: 3,
  disrupted: 2,
  normal: 1,
  unknown: 0,
};

export function conditionRank(c: string | null | undefined): number {
  return CONDITION_RANK[c ?? "unknown"] ?? 0;
}

// The severity the MTA alert *cause* implies, independent of the HMM. Lets us
// flag when the model's condition (severity axis) diverges from the alert's
// label (cause axis) — e.g. planned "No Scheduled Service" reads as a
// suspension for display but the HMM treats it as disrupted.
export function impliedCondition(category: string | null | undefined): string {
  if (category === "service_suspension") return "suspended";
  if (!category || category === "none") return "normal";
  return "disrupted";
}

// Plain-English badge text for a route's published condition. Raw codes like
// "not_scheduled" would render with an underscore under the capitalize style.
// Shared by the Status cards, the line page verdict, and the lines triage board
// so the three never word the same state differently.
export function conditionLabel(condition: string): string {
  switch (condition) {
    case "normal":
      return "Normal";
    case "disrupted":
      return "Disrupted";
    case "suspended":
      return "Suspended";
    case "not_scheduled":
      return "Not scheduled";
    default:
      return "No signal";
  }
}

// One plain sentence for how a line is moving right now — the movement status,
// never the MTA alert (that is a separate clause a caller may add). Shared with
// the Status cards' headline so the verdict header reads identically.
export function conditionLead(condition: string): string {
  switch (condition) {
    case "normal":
      return "Trains are moving normally.";
    case "disrupted":
      return "Trains are moving slowly or stalling.";
    case "suspended":
      return "No trains are running on this line.";
    case "not_scheduled":
      return "Not scheduled to run right now.";
    default:
      return "No live signal from this line's trains.";
  }
}

// The state class name a condition paints with. Maps not_scheduled/unknown onto
// the muted "unknown" swatch the .cond badge already defines.
export function conditionClass(condition: string): string {
  if (condition === "disrupted" || condition === "suspended" || condition === "normal") {
    return condition;
  }
  return "unknown";
}

// Whether a route deserves the reader's attention right now: its flow is
// disrupted or suspended, or its supply has dropped below the degrade floor.
// The triage board leads with exactly these — a normally-running, normally-
// supplied line is not one, and an unknown-signal line is an absence of a
// reading, not a disruption to surface.
export function isServiceFlagged(r: RouteStatus): boolean {
  return (
    r.condition === "disrupted" ||
    r.condition === "suspended" ||
    r.service_condition === "degraded"
  );
}

// The one-sentence takeaway for a line, resolving the two axes the model keeps
// separate: a disrupted or suspended FLOW leads with how trains are moving; a
// line whose flow is fine but whose SUPPLY has dropped below the floor says
// that instead, so a "Normal" badge never sits over a red supply number with no
// account of why the line is flagged. Every other line gets its plain flow lead.
export function serviceLead(r: RouteStatus): string {
  if (r.condition === "disrupted" || r.condition === "suspended") {
    return conditionLead(r.condition);
  }
  if (r.service_condition === "degraded") {
    return "Trains are moving, but the line is running fewer than usual.";
  }
  return conditionLead(r.condition);
}

// Fallback colors for routes the compat layer doesn't carry. Standard MTA hues.
const FALLBACK_COLOR = "#6e6e73";

export function routeColor(snap: Snapshot, routeId: string): string {
  return snap.compat?.subwaynow_routes?.[routeId]?.color ?? FALLBACK_COLOR;
}

export function routeLabel(snap: Snapshot, routeId: string): string {
  return snap.compat?.subwaynow_routes?.[routeId]?.name ?? routeId;
}

// Resolve an alert id (as carried on RouteStatus.alerts) against snap.alerts to
// its human headline. Returns the alert_type and English header text; either can
// be null when the id isn't in the published set or carries no header.
export function alertHeadline(
  snap: Snapshot,
  id: string,
): { type: string | null; text: string | null } {
  const a = snap.alerts?.find((x) => x.id === id);
  if (!a) return { type: null, text: null };
  const tr = a.header_text?.translation ?? [];
  const en = tr.find((t) => t.language === "en") ?? tr[0];
  return { type: a.alert_type ?? null, text: en?.text ?? null };
}

export function fmtAgo(epochSec: number | null | undefined, nowSec: number): string {
  if (epochSec == null) return "—";
  const d = Math.max(0, nowSec - epochSec);
  if (d < 90) return `${d}s ago`;
  if (d < 5400) return `${Math.round(d / 60)}m ago`;
  if (d < 172800) return `${Math.round(d / 3600)}h ago`;
  return `${Math.round(d / 86400)}d ago`;
}

// Countdown to a future epoch — the mirror of fmtAgo, for a time that has not
// happened yet (an announced elevator return). fmtAgo clamped every future
// instant to "0s ago"; a return estimate is always ahead of now, so it needs
// its own direction. A time already past means the estimate lapsed with the
// outage still up, which is a fact worth naming rather than rounding to zero.
export function fmtEta(epochSec: number | null | undefined, nowSec: number): string {
  if (epochSec == null) return "—";
  const d = epochSec - nowSec;
  if (d <= 0) return "overdue";
  if (d < 5400) return `in ${Math.round(d / 60)}m`;
  if (d < 172800) return `in ${Math.round(d / 3600)}h`;
  return `in ${Math.round(d / 86400)}d`;
}

export function fmtMinutes(min: number): string {
  if (min <= 0) return "—";
  if (min < 60) return `${Math.round(min)}m`;
  // Round to whole minutes BEFORE splitting: rounding the remainder on its own
  // lets 22h 59.7m print as "22h 60m".
  const total = Math.round(min);
  const h = Math.floor(total / 60);
  const m = total % 60;
  return m ? `${h}h ${m}m` : `${h}h`;
}

// Probabilities, clamped to the resolution they can actually support.
//
// The filter saturates hard: a settled route publishes p_normal exactly 1.0
// with p_disrupted around 1e-17, and the 30-minute horizon lands at 0.9989.
// Printed naively those read "100.00%" and "100%" — certainty no filter earns,
// and the saturation itself is a known open problem with the posterior. One
// decimal everywhere, with both ends clamped, so no surface ever claims a
// probability of 1 or 0.
//
// The bounds are strict: an exact 0.999 is 99.9% and prints as such, because
// ">99.9%" has to mean strictly more. That leaves the clamps covering exactly
// the values one decimal cannot render honestly — above 0.999 would round up
// to "100.0%", below 0.001 down to "0.0%" — and nothing else.
export function fmtProb(p: number): string {
  if (p > 0.999) return ">99.9%";
  if (p < 0.001) return "<0.1%";
  return `${(p * 100).toFixed(1)}%`;
}

// Supply-axis thresholds, mirroring worker/src/movement_state.ts: a route
// degrades below DEGRADE and only recovers back above the higher RECOVER, each
// confirmed for two ticks. Both are marked on the drawer meter so a reader can
// see where a route sits relative to what actually flips the axis, rather than
// guessing from a bare percentage.
export const SUPPLY_DEGRADE_RATIO = 0.5;
export const SUPPLY_RECOVER_RATIO = 0.8;

// Whether a route is running notably more trains than usual, compared against
// that (route, hour) cell's OWN spread rather than one global multiple: across
// 1,129 cells, measured p90/median spans 1.09x to 12.0x, and single cells are
// bimodal (an ordinary second mode can sit at a cell's 88th percentile while a
// genuinely rare spike in another cell sits at its 96th) — no single cutoff
// separates noise from a real high mark on both. Exported so the card glyph,
// meter, and drawer never disagree about what counts as notable.
export function isRunningHigh(r: RouteStatus): boolean {
  return (
    r.service_ratio != null &&
    r.service_high_ratio != null &&
    r.service_ratio > r.service_high_ratio
  );
}

// How full the level glyph reads. "low" is either the published degraded regime
// or a raw reading under the degrade floor; "thin" is the 50-80% band, which is
// noticeably below usual but is NOT a suppressed flag — the 80% recover line
// only governs a route that is already degraded. Running high is deliberately
// NOT a band: it is the isRunningHigh predicate above, called once at each
// render site, so there is exactly one notion of "notably more trains than
// usual" rather than a band and a marker that can drift apart.
export type SupplyBand = "low" | "thin" | "normal" | "unknown";

export function supplyBand(r: RouteStatus): SupplyBand {
  if (r.service_condition === "degraded") return "low";
  if (r.service_condition !== "normal" || r.service_ratio == null) return "unknown";
  // The ratio is this tick's raw reading; the regime behind service_condition is
  // debounced, so a route can read below the floor while still published normal.
  if (r.service_ratio < SUPPLY_DEGRADE_RATIO) return "low";
  if (r.service_ratio < SUPPLY_RECOVER_RATIO) return "thin";
  return "normal";
}

// The Gauge's colour, kept in lockstep with the glyph/meter so a route's dial
// never disagrees with its other supply surfaces. Running notably high takes the
// "high" accent; otherwise it follows the band. Mirrors GaugeTone in app/Gauge.tsx.
export function gaugeTone(r: RouteStatus): "low" | "thin" | "normal" | "high" | "unknown" {
  return isRunningHigh(r) ? "high" : supplyBand(r);
}

// Bars lit on the 3-bar level glyph. Unknown lights none.
export function supplyBars(band: SupplyBand): number {
  switch (band) {
    case "low":
      return 1;
    case "thin":
      return 2;
    case "normal":
      return 3;
    default:
      return 0;
  }
}

// --- Platform crowding ----------------------------------------------------

// Emphasis bands for a waiting-rider count, cut at the measured quantiles of
// the published estimate itself (2026-08-20 07:00-11:00 ET, cap applied):
// p50 = 28 riders, p90 = 86, p99 = 270, max 1302. Quantiles rather than round
// numbers because the distribution is long-tailed — any invented cutoff either
// fires on nearly every platform or on almost none — so "heavy" always means
// "busier than nine platforms in ten" and "extreme" means the top percentile.
// The unit is people: at p99 the platform holds a quarter of one train load,
// and even the maximum is about one, so this is never a count of trains.
export const CROWD_TYPICAL_RIDERS = 28;
export const CROWD_HEAVY_RIDERS = 86;
export const CROWD_EXTREME_RIDERS = 270;

export type CrowdBand = "light" | "typical" | "heavy" | "extreme";

function crowdBand(riders: number): CrowdBand {
  if (riders >= CROWD_EXTREME_RIDERS) return "extreme";
  if (riders >= CROWD_HEAVY_RIDERS) return "heavy";
  if (riders >= CROWD_TYPICAL_RIDERS) return "typical";
  return "light";
}

// What a render site gets. Abstention is a value carrying its reason, not a
// null and never a zero: "we have no estimate" and "the platform is empty" are
// different facts and the surface refuses to conflate them.
//
// `unpublished` covers every reason the worker abstained — no ridership
// baseline for the complex, no train seen inside the served window, or a stop
// it doesn't know. Which one is only counted system-wide in `abstained`, so a
// single platform can never be told which applied to it.
export type PlatformCrowdingView =
  | {
      estimated: true;
      riders: number;
      band: CrowdBand;
      minutesSince: number;
      entriesPerMin: number;
    }
  | {
      estimated: false;
      reason: "no_surface" | "unpublished" | "gap_exceeds_cap";
      minutesSince: number | null;
    };

// One platform's estimate, re-derived for right now.
//
// The published `waiting_riders` is only true as of `observed_at`. The snapshot
// is polled every 60s and a busy platform gains 20-50 riders a minute, so
// rendering the published integer as-is is the largest error this surface can
// make — bigger than anything in the baseline behind it. The publisher ships
// `last_train_at` and `entries_per_min` for exactly this reason: we redo its
// arithmetic (entries per minute times minutes since the last train) against
// the client clock instead.
//
// The cap is the publisher's own, read off `method.max_gap_minutes` rather than
// hardcoded here, and applied to OUR elapsed time. Past it the linear
// accumulation stops describing a crowd — people give up and leave, and the
// platform is usually out of service rather than jammed — which is why the
// worker abstains on 1.04% of gaps. Extrapolating past a line the publisher
// itself refused to cross would be this page inventing data.
export function platformCrowding(
  pc: PlatformCrowding | null | undefined,
  platformId: string,
  nowSec: number,
): PlatformCrowdingView {
  if (!pc) return { estimated: false, reason: "no_surface", minutesSince: null };
  const est = pc.platforms[platformId];
  if (!est) return { estimated: false, reason: "unpublished", minutesSince: null };
  // Clamped: a client clock running behind the publisher's must not subtract
  // riders off the front of the crowd.
  const minutesSince = Math.max(0, (nowSec - est.last_train_at) / 60);
  if (minutesSince > pc.method.max_gap_minutes)
    return { estimated: false, reason: "gap_exceeds_cap", minutesSince };
  const riders = Math.round(est.entries_per_min * minutesSince);
  return {
    estimated: true,
    riders,
    band: crowdBand(riders),
    minutesSince,
    entriesPerMin: est.entries_per_min,
  };
}

// Riders, with the tilde that marks the whole figure as modelled. Singular is
// worth the branch: the count is genuinely 0 or 1 on a platform a train has
// just cleared, which is most platforms most of the time.
export function fmtRiders(riders: number): string {
  return `~${riders} ${riders === 1 ? "rider" : "riders"}`;
}
