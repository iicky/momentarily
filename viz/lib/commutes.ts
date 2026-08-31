// Saved commutes: NAMED collections of segment cells the site remembers between
// visits. A commute is built from a chosen /trip journey (or a hand-picked
// subset of its segments) and persisted to localStorage only — no accounts, no
// server. The /commutes view reads live status scoped to EXACTLY these
// (route, direction, from_stop) cells, direction-aware by construction: each
// cell's key carries its running direction, so a southbound problem can never
// colour a northbound commute, and vice versa.
//
// This module is pure and React-free. It owns the persisted shape, its
// localStorage read/write (guarded for SSR), and the two derivations the view
// needs: the movement status scoped to a commute's cells, and where that
// movement read disagrees with the alert feed for the same route+direction.

import type { Snapshot } from "./types.ts";

// One saved segment cell. `key` is `${route}|${direction}|${from}` — identical
// to JourneySegment.key, AdjEdge.key, and the segment_flow cell key, so a saved
// segment joins directly to live status with no re-derivation. `to` rides along
// for display only (the strip needs the next stop's id to name it).
export interface CommuteSegment {
  route: string;
  direction: string; // "north" | "south"
  from: string; // directional stop id, e.g. "235N"
  to: string; // directional stop id, e.g. "228N"
  key: string; // `${route}|${direction}|${from}`
}

// One ride within a commute: the contiguous hops on one route+direction,
// mirroring a JourneyLeg so the saved collection renders as the same strip the
// trip view draws.
export interface CommuteLeg {
  route: string;
  direction: string;
  segments: CommuteSegment[];
}

// A named commute. `legs` preserve travel order and the transfer breaks between
// them; `id` is stable across renames so React keys and status caches survive an
// edit.
export interface Commute {
  id: string;
  name: string;
  createdAt: number;
  legs: CommuteLeg[];
}

// Bumped only if the persisted shape changes incompatibly; a version mismatch is
// treated as "no saved commutes" rather than crashing on a stale blob.
const STORAGE_KEY = "momentarily.commutes.v1";

// --- Persistence -----------------------------------------------------------

/** Every saved commute, oldest first. Returns [] under SSR (no localStorage) or
 * when the stored blob is missing, unparseable, or the wrong shape — a corrupt
 * store never throws into a render. */
export function loadCommutes(): Commute[] {
  if (typeof window === "undefined") return [];
  let raw: string | null;
  try {
    raw = window.localStorage.getItem(STORAGE_KEY);
  } catch {
    return [];
  }
  if (!raw) return [];
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return [];
  }
  if (!Array.isArray(parsed)) return [];
  return parsed.filter(isCommute);
}

/** Overwrite the store with `list`. A quota or serialization failure is
 * swallowed — persistence is best-effort convenience, never load-bearing. */
export function persistCommutes(list: Commute[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  } catch {
    /* ignore: private mode / quota — the in-memory list still stands */
  }
}

/** Append one commute and persist, returning the new full list. */
export function addCommute(commute: Commute): Commute[] {
  const next = [...loadCommutes(), commute];
  persistCommutes(next);
  return next;
}

/** A monotonic-ish unique id: crypto.randomUUID when present, else a
 * timestamp+random fallback (older WebViews). Only needs per-store uniqueness. */
export function newCommuteId(): string {
  const c = typeof crypto !== "undefined" ? crypto : undefined;
  if (c && typeof c.randomUUID === "function") return c.randomUUID();
  return `c-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

// Structural guard: enough to reject a stale/foreign blob without pretending to
// validate every field. A leg with no segments, or a segment missing its key,
// is dropped rather than trusted.
function isCommute(v: unknown): v is Commute {
  if (!v || typeof v !== "object") return false;
  const c = v as Record<string, unknown>;
  if (typeof c.id !== "string" || typeof c.name !== "string") return false;
  if (typeof c.createdAt !== "number" || !Array.isArray(c.legs)) return false;
  return c.legs.every(isLeg);
}

function isLeg(v: unknown): v is CommuteLeg {
  if (!v || typeof v !== "object") return false;
  const l = v as Record<string, unknown>;
  if (typeof l.route !== "string" || typeof l.direction !== "string") return false;
  return Array.isArray(l.segments) && l.segments.length > 0 && l.segments.every(isSegment);
}

function isSegment(v: unknown): v is CommuteSegment {
  if (!v || typeof v !== "object") return false;
  const s = v as Record<string, unknown>;
  return (
    typeof s.route === "string" &&
    typeof s.direction === "string" &&
    typeof s.from === "string" &&
    typeof s.to === "string" &&
    typeof s.key === "string"
  );
}

// --- Live status, scoped to the commute's cells ----------------------------

// The movement verdict published for one cell right now: "normal" | "quiet" |
// "disrupted", or null when the cell was not judged this tick. null is never a
// healthy read by omission — an absent cell is no evidence, not a clean bill.
export type CellStatus = "normal" | "quiet" | "disrupted";

// The two ways the movement read and the alert feed can disagree for one
// route+direction. Mirrors the archival movement/alert confusion (lib/movement.ts)
// at the live-snapshot level, from the alert feed's point of view:
//   alert-only    — an advisory is up here, but trains are moving normally
//                   (the alert feed's "false-disrupted").
//   movement-only — movement reads disrupted, yet no advisory is posted
//                   (the alert feed's "false-normal").
// Only the crisp cases are flagged: a "quiet" (thinly-scheduled) or unjudged
// cell has no movement verdict firm enough to contradict an advisory.
export type Disagreement = "alert-only" | "movement-only";

export interface SegmentReading {
  segment: CommuteSegment;
  status: CellStatus | null;
  // Recovery estimate for a disrupted cell, when the model has one; else null.
  recoveryMinutes: number | null;
  // The alert feed's primary advisory for this segment's route+direction, or
  // null when that direction carries none. Shown verbatim; "No Scheduled
  // Service" is a benign off-hours state and never drives a disagreement.
  alert: string | null;
  disagreement: Disagreement | null;
}

// The whole collection's headline movement verdict. Precedence: any disrupted
// cell dominates; otherwise any judged-normal cell reports the commute as
// moving; a commute that is only thinly-scheduled reads "quiet"; and one with no
// judged cell at all reads "unknown". Coverage counts ride alongside so the view
// never lets a headline overclaim health for the cells it could not judge.
export type CommuteRollup = "disrupted" | "normal" | "quiet" | "unknown";

export interface CommuteStatus {
  rollup: CommuteRollup;
  readings: SegmentReading[]; // one per segment, in travel order
  total: number;
  judged: number; // cells with a movement reading this tick
  disruptedCount: number;
  unknownCount: number;
  // The worst recovery estimate across disrupted cells, for the headline; null
  // when nothing disrupted carries one.
  worstRecoveryMinutes: number | null;
  disagreements: SegmentReading[]; // the subset whose read contradicts the feed
}

// The alert feed's primary advisory for a route+direction, direction-aware.
// Falls back to a bare "alert" label when the direction carries alerts but no
// typed primary. null when the route is unknown or that direction is clear.
function alertFor(snap: Snapshot, route: string, direction: string): string | null {
  const bound =
    direction === "north" ? "northbound" : direction === "south" ? "southbound" : null;
  if (!bound) return null;
  const da = snap.route_status?.[route]?.by_direction?.[bound];
  if (!da) return null;
  return da.primary_alert_type ?? (da.alerts.length > 0 ? "alert" : null);
}

/** The commute's live status, scoped to exactly its saved cells and nothing
 * else. Every reading is keyed by the segment's own route|direction|from_stop,
 * so the scoping is direction-aware by construction. */
export function commuteStatus(snap: Snapshot, commute: Commute): CommuteStatus {
  const readings: SegmentReading[] = [];
  for (const leg of commute.legs) {
    for (const segment of leg.segments) {
      const raw = snap.segment_flow?.segments[segment.key] ?? null;
      // A cell key names only its from_stop, so at a branch or express split
      // several drawn edges claim it — `to` says which hop the reading is
      // about. A reading about a different successor, or one that cannot name
      // its successor (to: null), is no evidence for this commute: read it as
      // unknown rather than misattribute. Mirrors rankJourneys.ts.
      const cell = raw != null && raw.to === segment.to ? raw : null;
      const status = cell?.status ?? null;
      const alert = alertFor(snap, segment.route, segment.direction);
      // "No Scheduled Service" is an expected off-hours state, not a disruption
      // to hold the movement read against, so it never drives a disagreement.
      const disruptive = alert != null && alert !== "No Scheduled Service";
      let disagreement: Disagreement | null = null;
      if (status === "normal" && disruptive) disagreement = "alert-only";
      else if (status === "disrupted" && !disruptive) disagreement = "movement-only";
      readings.push({
        segment,
        status,
        recoveryMinutes: cell?.recovery?.recovery_minutes ?? null,
        alert,
        disagreement,
      });
    }
  }

  let disruptedCount = 0;
  let unknownCount = 0;
  let anyNormal = false;
  let anyQuiet = false;
  let worstRecoveryMinutes: number | null = null;
  for (const r of readings) {
    switch (r.status) {
      case "disrupted":
        disruptedCount++;
        if (r.recoveryMinutes != null)
          worstRecoveryMinutes = Math.max(worstRecoveryMinutes ?? 0, r.recoveryMinutes);
        break;
      case "normal":
        anyNormal = true;
        break;
      case "quiet":
        anyQuiet = true;
        break;
      default:
        unknownCount++;
    }
  }

  const rollup: CommuteRollup =
    disruptedCount > 0 ? "disrupted" : anyNormal ? "normal" : anyQuiet ? "quiet" : "unknown";

  return {
    rollup,
    readings,
    total: readings.length,
    judged: readings.length - unknownCount,
    disruptedCount,
    unknownCount,
    worstRecoveryMinutes,
    disagreements: readings.filter((r) => r.disagreement != null),
  };
}
