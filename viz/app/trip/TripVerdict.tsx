"use client";

// The verdict banner for the trip view: given the enumerated candidates and the
// live snapshot, it ranks them (lib/rankJourneys.ts) and prints ONE sentence
// that recommends the best and names the evidence — "Take the A: the F is
// disrupted at Bergen St" — rather than leaving the reader to eyeball a sorted
// list. Clicking it selects the recommended journey so the strip-map follows.
//
// This is deliberately its own component (and the ranking its own module) so it
// stays diff-separable from the picker/strip-map and from the saved-commutes
// surface being built alongside it.

import { useMemo } from "react";
import { RouteBullet } from "../ui";
import { journeyVerdict, type JourneyScore, type Verdict } from "@/lib/rankJourneys";
import type { Journey } from "@/lib/journeys";
import type { Snapshot } from "@/lib/types";

// A route sequence as inline bullets: "A" or "A → C" for a transfer journey.
function RouteRun({ snap, routes, size = 18 }: { snap: Snapshot; routes: string[]; size?: number }) {
  return (
    <span className="verdict-run">
      {routes.map((r, i) => (
        <span className="verdict-run-leg" key={i}>
          {i > 0 && <span className="verdict-run-arrow">→</span>}
          <RouteBullet snap={snap} route={r} size={size} />
        </span>
      ))}
    </span>
  );
}

export function TripVerdict({
  snap,
  journeys,
  onPick,
}: {
  snap: Snapshot;
  journeys: Journey[];
  onPick: (id: string) => void;
}) {
  const verdict = useMemo<Verdict | null>(
    () => journeyVerdict(snap, journeys),
    [snap, journeys],
  );

  if (!verdict) return null;
  const { best, tone, culpritRoute, reason, caveat, singleCandidate } = verdict;

  return (
    <button
      type="button"
      className={`trip-verdict tone-${tone}`}
      onClick={() => onPick(best.id)}
      title="Follow the recommended journey"
    >
      <span className="verdict-tag">{singleCandidate ? "Only route" : "Best right now"}</span>
      <span className="verdict-line">
        {singleCandidate ? (
          <>
            <RouteRun snap={snap} routes={best.routes} />
            <span className="verdict-reason"> {reason}.</span>
          </>
        ) : (
          <>
            <span className="verdict-take">Take</span>
            <RouteRun snap={snap} routes={best.routes} />
            <span className="verdict-sep">—</span>
            {culpritRoute ? (
              <>
                <RouteBullet snap={snap} route={culpritRoute} size={18} />
                <span className="verdict-reason"> {reason}.</span>
              </>
            ) : (
              <span className="verdict-reason">{reason}.</span>
            )}
          </>
        )}
      </span>
      {caveat && <span className="verdict-caveat">{caveat}.</span>}
      <ScoreCrumb best={best} />
    </button>
  );
}

// The compact status tally behind the verdict, so the sentence is auditable at a
// glance: how many hops on the recommended ride are disrupted / quiet / unread.
function ScoreCrumb({ best }: { best: JourneyScore }) {
  const parts: string[] = [];
  if (best.disrupted) parts.push(`${best.disrupted} disrupted`);
  if (best.quiet) parts.push(`${best.quiet} quiet`);
  if (best.unknown) parts.push(`${best.unknown} unread`);
  const clean = parts.length === 0;
  return (
    <span className="verdict-crumb">
      {clean ? "all segments moving" : parts.join(" · ")}
      {` of ${best.journey.segments.length} on this ride`}
    </span>
  );
}
