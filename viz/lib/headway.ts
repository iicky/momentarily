// The observed-vs-scheduled time-between-trains read, in plain terms.
//
// One headway Observation is published per (route, direction) at that cell's
// canonical reference stop (worker/src/headway.ts), each carrying the last hour
// of gaps as `window` and — when the timetable baseline is available and the
// reading is on-reference — a `scheduled` median for the tick's hour-of-week.
// This module turns that raw measurement into the lay reading the site speaks:
// "trains every 9 min, scheduled 6", with the degraded cases stated honestly
// rather than papered over with a fabricated ratio.

import type { Observation, Snapshot } from "./types";

// A cell counted this many scheduled trains or fewer in the hour is a thin,
// wide-headway cell: the median is still shown, but a consumer should not read
// a hard ratio off it. Matches the "distrust thin n_trips" contract on the
// scheduled artifact.
export const THIN_N_TRIPS = 3;

// How far the observed gap must sit from scheduled before the read calls it —
// below these it reads "about on schedule". A gap 25% longer than the timetable
// is a real wait; one 20% shorter is trains running closer than booked.
export const GAPPED_RATIO = 1.25;
export const BUNCHED_RATIO = 0.8;

// The worker caps the rolling window at this many gaps — a full hour of history
// (worker/src/headway.ts HEADWAY_WINDOW_SIZE). A shorter window is a cell that
// has seen fewer trains recently, worth saying so the strip is not read as if
// it covered the whole hour.
export const HEADWAY_WINDOW_SIZE = 12;

/** The headway Observation for one (route, direction), or null when the surface
 * carries none this tick — a cold cell, a feed outage, or a full suspension. */
export function headwayFor(
  snap: Snapshot,
  route: string,
  direction: "north" | "south",
): Observation | null {
  const ref = `subway_route:${route}`;
  return (
    snap.observations.find(
      (o) => o.kind === "headway" && o.entity_ref === ref && o.direction === direction,
    ) ?? null
  );
}

/** A gap in seconds as lay copy: "9 min", "1 min", "40 sec". Sub-minute gaps
 * stay in seconds (rounded to 5s) rather than rounding to "0 min"; everything
 * else rounds to the nearest whole minute, which is how a rider reads a
 * timetable. */
export function fmtGap(seconds: number): string {
  if (seconds < 60) return `${Math.max(5, Math.round(seconds / 5) * 5)} sec`;
  return `${Math.round(seconds / 60)} min`;
}

export type HeadwayTone = "gapped" | "bunched" | "onschedule";

// Why a reading stands alone with no scheduled comparison — the three honest
// degraded cases, kept distinct so the copy can say which one it is.
//   none         — the timetable schedules no service for this cell/hour, or the
//                  baseline artifact is not published yet.
//   off_reference — the reading came from a reroute fallback stop, not the
//                  canonical one the baseline is keyed on: labelled, not compared.
export type ObservedOnlyReason = "none" | "off_reference";

export interface HeadwayRead {
  observedSeconds: number;
  observedAt: number;
  // The last hour of gaps, oldest-first, for the strip. Always at least the one
  // reading. Fewer than a full hour when the cell has seen fewer trains.
  window: number[];
  // Present only when the reading is compared against the timetable.
  scheduled: {
    seconds: number;
    nTrips: number;
    ratio: number; // observed / scheduled
    tone: HeadwayTone;
    thin: boolean; // nTrips <= THIN_N_TRIPS — the median is soft
  } | null;
  // Present only when there is no comparison, saying which degraded case it is.
  observedOnly: ObservedOnlyReason | null;
}

/** Reduce a headway Observation to the reading the views render. Pure — no copy,
 * only the classification the copy is drawn from, so it can be unit-tested and
 * shared by the route and commute surfaces. */
export function readHeadway(obs: Observation): HeadwayRead {
  const observedSeconds = obs.value;
  const window = (obs.window ?? []).map((s) => s.value);
  if (obs.off_reference) {
    return { observedSeconds, observedAt: obs.observed_at, window, scheduled: null, observedOnly: "off_reference" };
  }
  // `!obs.scheduled` covers both an explicit null and a snapshot published
  // before the Worker embedded the field — until that deploy lands, every
  // reading is observed-only, which is exactly the honest degraded state.
  if (!obs.scheduled) {
    return { observedSeconds, observedAt: obs.observed_at, window, scheduled: null, observedOnly: "none" };
  }
  const scheduledSeconds = obs.scheduled.median_headway_s;
  const ratio = scheduledSeconds > 0 ? observedSeconds / scheduledSeconds : 1;
  const tone: HeadwayTone =
    ratio >= GAPPED_RATIO ? "gapped" : ratio <= BUNCHED_RATIO ? "bunched" : "onschedule";
  return {
    observedSeconds,
    observedAt: obs.observed_at,
    window,
    scheduled: {
      seconds: scheduledSeconds,
      nTrips: obs.scheduled.n_trips,
      ratio,
      tone,
      thin: obs.scheduled.n_trips <= THIN_N_TRIPS,
    },
    observedOnly: null,
  };
}

/** The one-line headline the views print, drawn from the read. Lay register,
 * no model jargon: the observed gap first, the scheduled gap when there is one,
 * and the bunched/gapped qualifier the timetable comparison licenses. */
export function headwayHeadline(read: HeadwayRead): string {
  const observed = `Trains every ${fmtGap(read.observedSeconds)}`;
  if (read.scheduled === null) {
    if (read.observedOnly === "off_reference") {
      return `${observed} · measured at a reroute stop`;
    }
    return `${observed} · no scheduled baseline for this hour`;
  }
  const sched = `scheduled ${fmtGap(read.scheduled.seconds)}`;
  const qualifier =
    read.scheduled.tone === "gapped"
      ? "longer gaps than usual"
      : read.scheduled.tone === "bunched"
        ? "running closer than usual"
        : "about on schedule";
  return `${observed}, ${sched} — ${qualifier}`;
}
