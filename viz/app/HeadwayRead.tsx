"use client";

// The observed-vs-scheduled time-between-trains read, as a card row: a plain
// headline ("trains every 9 min, scheduled 6") over a strip of the last hour's
// gaps. The strip is the historical N-car chain from .3 — one bar per gap
// between successive trains, so regular service reads as an even shape and
// bunching or a widening gap reads as a jagged one, with no axis and no legend.
// A strip is a card/row component, never an icon (the parent epic's rendering
// note); the mark in a status column stays the two-car badge elsewhere.

import type { Observation } from "@/lib/types";
import {
  fmtGap,
  headwayHeadline,
  readHeadway,
  HEADWAY_WINDOW_SIZE,
  type HeadwayTone,
} from "@/lib/headway";

// Green when service is on schedule or running closer than booked; amber when
// the gap has opened past the timetable. Neutral when there is nothing to
// compare against, so an uncompared reading never borrows a healthy green.
const TONE_CLASS: Record<HeadwayTone, string> = {
  onschedule: "hw-ok",
  bunched: "hw-ok",
  gapped: "hw-gap",
};

/** One directional headway read for a route, or null to render nothing when the
 * surface carries no measurement for this (route, direction) this tick. */
export function HeadwayRead({
  obs,
}: {
  obs: Observation | null;
}) {
  if (obs === null) return null;
  const read = readHeadway(obs);
  const tone = read.scheduled?.tone ?? null;
  const cls = tone ? TONE_CLASS[tone] : "hw-muted";

  // Per-bar colouring against the scheduled median: a gap at or past the gapped
  // threshold is amber, one at or under scheduled is green, between is neutral.
  // Without a baseline every bar is neutral — the shape still reads, the health
  // claim does not.
  const sched = read.scheduled?.seconds ?? null;
  const maxGap = Math.max(...read.window, read.observedSeconds, 1);

  const shortWindow = read.window.length > 0 && read.window.length < HEADWAY_WINDOW_SIZE;

  return (
    <div className={`hw-read ${cls}`}>
      <p className="hw-headline">{headwayHeadline(read)}</p>
      {read.window.length > 0 && (
        <div className="hw-strip" role="img" aria-label={stripLabel(read.window)}>
          {read.window.map((gap, i) => (
            <span
              key={i}
              className={`hw-bar ${barTone(gap, sched)}`}
              style={{ height: `${Math.round((gap / maxGap) * 100)}%` }}
              title={fmtGap(gap)}
            />
          ))}
        </div>
      )}
      <p className="hw-note">
        {read.scheduled?.thin && (
          <span className="hw-flag">sparse timetable this hour — read the schedule loosely</span>
        )}
        {shortWindow && (
          <span className="hw-flag">
            last {read.window.length} train{read.window.length === 1 ? "" : "s"}
          </span>
        )}
        {read.observedOnly === "off_reference" && (
          <span className="hw-flag">
            this line is running off its usual stop — not compared to the timetable
          </span>
        )}
      </p>
    </div>
  );
}

// Which colour a single gap bar takes against the scheduled median.
function barTone(gap: number, scheduled: number | null): string {
  if (scheduled === null || scheduled <= 0) return "hw-bar-neutral";
  const ratio = gap / scheduled;
  if (ratio >= 1.25) return "hw-bar-gap";
  if (ratio <= 1.0) return "hw-bar-ok";
  return "hw-bar-neutral";
}

// A spoken description of the strip for a screen reader: the gaps in order,
// so the shape is not lost to a non-visual read.
function stripLabel(window: number[]): string {
  return `Last ${window.length} gaps between trains: ${window.map(fmtGap).join(", ")}`;
}
