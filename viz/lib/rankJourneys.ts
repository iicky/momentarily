// Rank enumerated candidate journeys by their LIVE quality and produce a verdict
// that names its evidence — "Take the A: the F is disrupted at Bergen St" — not a
// bare sorted list. Sits beside journeys.ts (which proves a route sequence
// connects) and complexes.ts (which supplies the candidates): this owns "given
// the live snapshot, which of these candidates is moving best right now, and
// why".
//
// INPUTS ARE THE PUBLIC SNAPSHOT ONLY. Three axes, all already published:
//   - per-segment status + recovery: snap.segment_flow.segments[route|direction|from]
//     (SegmentStatus.status ∈ normal|quiet|disrupted, SegmentStatus.recovery)
//   - route flow condition: snap.route_status[route].condition
//   - supply band: supplyBand(snap.route_status[route])  (assigned vs baseline)
//
// WHAT IS DELIBERATELY NOT USED: per-segment expected TRAVERSAL TIMES. They are
// not in the published snapshot (segment_flow carries a status verdict per cell,
// not a scheduled or observed traversal duration), so a total journey time is not
// derivable here and is NOT invented. Ranking is therefore on status SEVERITY —
// disrupted > quiet > normal hop counts, the worst disrupted hop, and its
// recovery estimate — with route flow and supply as secondary axes, and transfer
// count / hop count only as final tie-breaks. Handing the ranker real per-segment
// traversal times is the single change that would sharpen it: it would let a
// journey that is one hop longer but all-normal be weighed against a shorter one
// with a brief disruption on time rather than on hop counts alone.

import { supplyBand, fmtMinutes } from "./feed.ts";
import { undirected } from "./stations.ts";
import { journeyId, type Journey } from "./journeys.ts";
import type { Snapshot } from "./types.ts";

// LIVE-QUALITY penalty weights (lower total = better journey). This score is
// built ONLY from live-status axes off the public snapshot — it never folds in
// transfer or hop count, which are topology, not live quality, and would be a
// back-door proxy for the traversal time the snapshot does not publish. Ordered
// so each tier strictly dominates the ones below it across the range it can
// reach, giving a single number that reads as a severity lexicography:
//   suspended route ≫ disrupted hops ≫ disrupted flow ≫ low supply ≫ thin supply
//   ≫ unread hops ≫ quiet hops.
// Transfer/hop count enter ONLY as a deterministic tie-break in rankJourneys,
// never here and never as a verdict reason.
const W = {
  // A route with no trains running makes the whole journey unusable — nothing
  // else on any candidate can outweigh one suspended line on the path.
  suspended: 100_000,
  // One segment actively degraded on the path. The dominant severity signal.
  disrupted: 1_000,
  // Per-disrupted-hop recovery bump, capped strictly BELOW W.disrupted so the
  // COUNT of disrupted hops always outranks their duration: a journey with two
  // brief disruptions is worse than one with a single long one.
  recoveryCap: 200,
  // The route's flow is disrupted overall even where its named hops on this path
  // were not individually flagged this tick.
  flow: 300,
  // Supply axis: far fewer trains than usual (longer, less certain waits)…
  low: 120,
  // …or somewhat fewer than usual.
  thin: 40,
  // A hop with no read this tick — uncertainty, not a fault, so it is mild.
  unknown: 12,
  // A quiet-normal hop: little scheduled through it right now. Healthy, but
  // ranked above a plainly-normal hop per the severity brief (disrupted > quiet
  // > normal).
  quiet: 6,
} as const;

// The one degraded hop a verdict points at: which route, where, and how long it
// is expected to take to clear (null minutes when disrupted but not yet
// forecastable; indeterminate when recovery runs past the model's horizon).
export interface SegmentEvidence {
  route: string;
  stop: string; // undirected GTFS id, for linking
  stopName: string;
  recoveryMinutes: number | null;
  recoveryIndeterminate: boolean;
}

// One candidate's live score. Carries the tallies and named evidence the verdict
// draws on, not just the final penalty, so the reason it prints is exactly the
// thing that moved the ranking.
export interface JourneyScore {
  journey: Journey;
  id: string;
  routes: string[]; // leg route ids in travel order
  disrupted: number;
  quiet: number;
  normal: number;
  unknown: number;
  worst: SegmentEvidence | null; // worst disrupted hop on the path, if any
  suspendedRoutes: string[];
  disruptedFlowRoutes: string[];
  lowSupplyRoutes: string[];
  thinSupplyRoutes: string[];
  transfers: number;
  recoveryPenalty: number; // summed, capped per-hop recovery contribution
  penalty: number;
}

const stopName = (snap: Snapshot, stop: string): string => {
  const id = undirected(stop);
  return snap.stations[id]?.name ?? id;
};

// Recovery ordering value for picking the worst disrupted hop: an unbounded
// (indeterminate) recovery is worse than any finite estimate; a disrupted hop
// with no forecast yet counts as zero so a forecast one always outranks it.
const recValue = (e: SegmentEvidence): number =>
  e.recoveryIndeterminate ? Number.POSITIVE_INFINITY : e.recoveryMinutes ?? 0;

/** Score one candidate against the live snapshot. Pure and side-effect free. */
export function scoreJourney(snap: Snapshot, journey: Journey): JourneyScore {
  const cells = snap.segment_flow?.segments;
  let disrupted = 0;
  let quiet = 0;
  let normal = 0;
  let unknown = 0;
  let recoveryPenalty = 0;
  let worst: SegmentEvidence | null = null;

  for (const seg of journey.segments) {
    const cell = cells?.[seg.key] ?? null;
    if (!cell) {
      unknown++;
      continue;
    }
    // A cell key names only its from_stop, so at a branch or express split
    // several drawn edges claim it — `to` says which hop the reading is about.
    // A reading about a different successor, or one that cannot name its
    // successor at all (to: null, topology doc missing that tick), is no
    // evidence for this journey: abstain rather than misattribute.
    if (cell.to !== seg.to) {
      unknown++;
      continue;
    }
    if (cell.status === "disrupted") {
      disrupted++;
      const rec = cell.recovery;
      const capped = rec
        ? rec.recovery_indeterminate
          ? W.recoveryCap
          : Math.min(rec.recovery_minutes, W.recoveryCap)
        : 0;
      recoveryPenalty += capped;
      const ev: SegmentEvidence = {
        route: seg.route,
        stop: undirected(seg.to),
        stopName: stopName(snap, seg.to),
        recoveryMinutes: rec && !rec.recovery_indeterminate ? rec.recovery_minutes : null,
        recoveryIndeterminate: !!rec?.recovery_indeterminate,
      };
      if (!worst || recValue(ev) > recValue(worst)) worst = ev;
    } else if (cell.status === "quiet") {
      quiet++;
    } else {
      normal++;
    }
  }

  const suspendedRoutes: string[] = [];
  const disruptedFlowRoutes: string[] = [];
  const lowSupplyRoutes: string[] = [];
  const thinSupplyRoutes: string[] = [];
  const seen = new Set<string>();
  for (const leg of journey.legs) {
    if (seen.has(leg.route)) continue;
    seen.add(leg.route);
    const r = snap.route_status?.[leg.route];
    if (!r) continue;
    // "suspended" and "not_scheduled" are one bucket here: either way this
    // route has no trains to ride right now, the heaviest possible penalty.
    if (r.condition === "suspended" || r.condition === "not_scheduled")
      suspendedRoutes.push(leg.route);
    else if (r.condition === "disrupted") disruptedFlowRoutes.push(leg.route);
    const band = supplyBand(r);
    if (band === "low") lowSupplyRoutes.push(leg.route);
    else if (band === "thin") thinSupplyRoutes.push(leg.route);
  }

  const penalty =
    suspendedRoutes.length * W.suspended +
    disrupted * W.disrupted +
    recoveryPenalty +
    disruptedFlowRoutes.length * W.flow +
    lowSupplyRoutes.length * W.low +
    thinSupplyRoutes.length * W.thin +
    unknown * W.unknown +
    quiet * W.quiet;

  return {
    journey,
    id: journeyId(journey),
    routes: journey.legs.map((l) => l.route),
    disrupted,
    quiet,
    normal,
    unknown,
    worst,
    suspendedRoutes,
    disruptedFlowRoutes,
    lowSupplyRoutes,
    thinSupplyRoutes,
    transfers: journey.transfers,
    recoveryPenalty,
    penalty,
  };
}

/** Score and rank every candidate by live quality, best (lowest penalty) first.
 * `penalty` is live status ONLY. Genuine live-status ties break deterministically
 * — fewer transfers, then fewer hops, then route-sequence id — purely to fix a
 * stable default selection; this ordering is NOT a quality judgement and never
 * becomes a verdict reason. */
export function rankJourneys(snap: Snapshot, journeys: Journey[]): JourneyScore[] {
  return journeys
    .map((j) => scoreJourney(snap, j))
    .sort((a, b) => {
      if (a.penalty !== b.penalty) return a.penalty - b.penalty;
      if (a.transfers !== b.transfers) return a.transfers - b.transfers;
      const la = a.journey.segments.length;
      const lb = b.journey.segments.length;
      if (la !== lb) return la - lb;
      return a.id.localeCompare(b.id, undefined, { numeric: true });
    });
}

// The severity axis a verdict is speaking to, for tone/colour on the UI side.
export type VerdictTone =
  | "clear"
  | "disrupted"
  | "suspended"
  | "supply"
  | "unknown"
  | "quiet"
  | "even";

// Severity order for the tone the banner paints its border with. "even"/"clear"
// are not health states, so they rank at the floor.
const TONE_SEVERITY: Record<VerdictTone, number> = {
  suspended: 5,
  disrupted: 4,
  supply: 3,
  unknown: 2,
  quiet: 1,
  clear: 0,
  even: 0,
};

// The winner's OWN worst live state, independent of why it out-ranked the
// runner-up. Used to floor the banner tone: a recommendation whose ride carries
// disruption must never be painted "clear" just because a structural tie-break
// (fewer transfers, shorter) is what decided it.
const ownTone = (s: JourneyScore): VerdictTone => {
  if (s.suspendedRoutes.length) return "suspended";
  if (s.disrupted > 0 || s.disruptedFlowRoutes.length) return "disrupted";
  if (s.lowSupplyRoutes.length || s.thinSupplyRoutes.length) return "supply";
  if (s.unknown > 0) return "unknown";
  if (s.quiet > 0) return "quiet";
  return "clear";
};

// The banner tone: the more severe of the reason's own tone and the winner's
// own worst state. The reason TEXT still speaks to the differentiator; only the
// colour escalates, so the border never under-states the recommended ride.
const escalate = (reasonTone: VerdictTone, best: JourneyScore): VerdictTone => {
  const own = ownTone(best);
  return TONE_SEVERITY[own] > TONE_SEVERITY[reasonTone] ? own : reasonTone;
};

export interface Verdict {
  ranked: JourneyScore[];
  best: JourneyScore;
  runnerUp: JourneyScore | null;
  // The route the evidence clause is about — the culprit on the runner-up (or
  // the best's own trouble when it is the only candidate). null for a purely
  // structural or even verdict, where the clause is about the journey, not a line.
  culpritRoute: string | null;
  // The evidence clause, already worded: "is disrupted at Bergen St · ~20m to
  // clear", "is suspended — no trains running", "it's a direct ride". Never empty.
  reason: string;
  // A hedge when the winner is not itself clean ("Both routes are disrupted"),
  // so a recommendation never over-promises. null when the best is clean.
  caveat: string | null;
  tone: VerdictTone;
  singleCandidate: boolean;
}

// The named categories, in strict severity order. Each contributes an additive
// slice of a journey's penalty; comparing two journeys slice by slice finds the
// single category that most drove them apart — which is exactly the evidence to
// cite, guaranteed consistent with the ranking itself.
const CATEGORIES = [
  "suspended",
  "disrupted",
  "flow",
  "low",
  "thin",
  "unknown",
  "quiet",
] as const;
type Category = (typeof CATEGORIES)[number];

const contribution = (s: JourneyScore, cat: Category): number => {
  switch (cat) {
    case "suspended":
      return s.suspendedRoutes.length * W.suspended;
    case "disrupted":
      return s.disrupted * W.disrupted + s.recoveryPenalty;
    case "flow":
      return s.disruptedFlowRoutes.length * W.flow;
    case "low":
      return s.lowSupplyRoutes.length * W.low;
    case "thin":
      return s.thinSupplyRoutes.length * W.thin;
    case "unknown":
      return s.unknown * W.unknown;
    case "quiet":
      return s.quiet * W.quiet;
  }
};

// A route in `worse`'s list that `better` does not share — the true
// differentiator to name — falling back to `worse`'s first when they overlap.
const differ = (worse: string[], better: string[]): string | null => {
  const set = new Set(better);
  return worse.find((r) => !set.has(r)) ?? worse[0] ?? null;
};

const disruptPhrase = (e: SegmentEvidence): string => {
  if (e.recoveryIndeterminate) return `is disrupted at ${e.stopName} — no clear recovery yet`;
  if (e.recoveryMinutes != null)
    return `is disrupted at ${e.stopName} · ~${fmtMinutes(e.recoveryMinutes)} to clear`;
  return `is disrupted at ${e.stopName}`;
};

// The verdict for the sole candidate: no comparison to make, so the clause is
// the journey's own live status. Still names its evidence.
function describeSingle(best: JourneyScore): Verdict {
  const base = {
    ranked: [best],
    best,
    runnerUp: null,
    caveat: null,
    singleCandidate: true,
  };
  if (best.suspendedRoutes.length)
    return {
      ...base,
      culpritRoute: best.suspendedRoutes[0],
      reason: "has no trains running right now",
      tone: "suspended",
    };
  if (best.worst)
    return {
      ...base,
      culpritRoute: best.worst.route,
      reason: disruptPhrase(best.worst),
      tone: "disrupted",
    };
  if (best.disruptedFlowRoutes.length)
    return {
      ...base,
      culpritRoute: best.disruptedFlowRoutes[0],
      reason: "is running disrupted right now",
      tone: "disrupted",
    };
  if (best.lowSupplyRoutes.length)
    return {
      ...base,
      culpritRoute: best.lowSupplyRoutes[0],
      reason: "is running far fewer trains than usual",
      tone: "supply",
    };
  if (best.thinSupplyRoutes.length)
    return {
      ...base,
      culpritRoute: best.thinSupplyRoutes[0],
      reason: "is running fewer trains than usual",
      tone: "supply",
    };
  if (best.unknown > 0)
    return {
      ...base,
      culpritRoute: null,
      reason: "has stretches with no live read right now",
      tone: "unknown",
    };
  if (best.quiet > 0)
    return {
      ...base,
      culpritRoute: null,
      reason: "is running but sparse right now — longer waits between trains",
      tone: "quiet",
    };
  return {
    ...base,
    culpritRoute: null,
    reason: "is running normally end to end",
    tone: "clear",
  };
}

// Word the decisive category's evidence, drawn from the runner-up (the loser),
// plus a caveat drawn from the winner when it shares the same class of trouble.
function reasonFor(
  cat: Category,
  best: JourneyScore,
  runnerUp: JourneyScore,
): { culpritRoute: string | null; reason: string; caveat: string | null; tone: VerdictTone } {
  switch (cat) {
    case "suspended": {
      const route = differ(runnerUp.suspendedRoutes, best.suspendedRoutes);
      return {
        culpritRoute: route,
        reason: "is suspended — no trains running",
        caveat: best.suspendedRoutes.length ? "both routes have a suspended line" : null,
        tone: "suspended",
      };
    }
    case "disrupted": {
      const ev = runnerUp.worst;
      const caveat =
        best.worst != null
          ? `the ${best.worst.route} is disrupted at ${best.worst.stopName} too`
          : null;
      return {
        culpritRoute: ev?.route ?? null,
        reason: ev ? disruptPhrase(ev) : "runs through more disruption",
        caveat,
        tone: "disrupted",
      };
    }
    case "flow": {
      const route = differ(runnerUp.disruptedFlowRoutes, best.disruptedFlowRoutes);
      return {
        culpritRoute: route,
        reason: "is running disrupted right now",
        caveat: best.disruptedFlowRoutes.length ? "neither line is running clean" : null,
        tone: "disrupted",
      };
    }
    case "low": {
      const route = differ(runnerUp.lowSupplyRoutes, best.lowSupplyRoutes);
      return {
        culpritRoute: route,
        reason: "is running far fewer trains than usual",
        caveat: null,
        tone: "supply",
      };
    }
    case "thin": {
      const route = differ(runnerUp.thinSupplyRoutes, best.thinSupplyRoutes);
      return {
        culpritRoute: route,
        reason: "is running fewer trains than usual",
        caveat: null,
        tone: "supply",
      };
    }
    case "unknown":
      return {
        culpritRoute: null,
        reason: "the other route has stretches with no live read right now",
        caveat: null,
        tone: "unknown",
      };
    case "quiet":
      return {
        culpritRoute: null,
        reason: "the other route runs sparse right now — longer waits between trains",
        caveat: null,
        tone: "quiet",
      };
  }
}

/**
 * Rank the candidates and word a verdict recommending the best, naming the one
 * piece of live evidence that most separates it from the runner-up. Returns null
 * only when there are no candidates to compare.
 */
export function journeyVerdict(snap: Snapshot, journeys: Journey[]): Verdict | null {
  const ranked = rankJourneys(snap, journeys);
  if (ranked.length === 0) return null;
  const best = ranked[0];
  if (ranked.length === 1) return describeSingle(best);
  const runnerUp = ranked[1];

  // The decisive category is the one where the runner-up's penalty slice most
  // exceeds the best's. Walked in severity order so exact ties in magnitude
  // resolve toward the more serious axis.
  let decisive: Category | null = null;
  let bestDelta = 0;
  for (const cat of CATEGORIES) {
    const delta = contribution(runnerUp, cat) - contribution(best, cat);
    if (delta > bestDelta) {
      bestDelta = delta;
      decisive = cat;
    }
  }

  if (!decisive) {
    // No live-status category separates them: on the published status they are
    // identical. Which one shows as "best" was fixed by rankJourneys' ordering
    // tie-break (fewer transfers, etc.), which is NOT a quality claim — so the
    // verdict says only what is true, that live status gives no edge.
    const clean = ownTone(best) === "clear";
    return {
      ranked,
      best,
      runnerUp,
      culpritRoute: null,
      reason: clean
        ? "every candidate is running clean right now — no live edge between them"
        : "the candidates are running the same on live status right now",
      caveat: null,
      tone: escalate("even", best),
      singleCandidate: false,
    };
  }

  const worded = reasonFor(decisive, best, runnerUp);
  return {
    ranked,
    best,
    runnerUp,
    culpritRoute: worded.culpritRoute,
    reason: worded.reason,
    caveat: worded.caveat,
    tone: escalate(worded.tone, best),
    singleCandidate: false,
  };
}
