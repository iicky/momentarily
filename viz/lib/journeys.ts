// Candidate journey enumeration between two station complexes, built on the
// same committed segment topology the spatial views read (viz/lib/diagram.ts →
// adjacency + route_stops, the trainer's per-route ordered stopping patterns and
// directed successor graph). Sits beside stations.ts, which owns canonical stop
// ordering; this owns "how do you get from A to B".
//
// Deliberately NOT a trip planner. There are no schedules, no timings, no
// shortest-path search over the whole graph. It enumerates a small candidate
// set — every direct route, then every single transfer at a shared complex,
// and only when nothing simpler connects does it fall back to two transfers.
// Ranking those candidates on live status and traversal time is a separate
// concern (the trip-comparison surface); this only says which rides plausibly
// join the two complexes, in a deterministic order.
//
// Reachability is proven inside one RoutePattern at a time, hop by hop against
// the adjacency graph. A pattern is one real stopping sequence of one
// route+direction, so origin-before-destination within a single pattern means a
// single train actually carries you; requiring every consecutive pair to be a
// real `edges` successor cell means every segment the result names is one a
// consumer can key live status and scheduled timings by. The merged display
// order (stations.mergePatterns) must NOT be used here: it interleaves
// express/local variants and appends disjoint branch runs for drawing, which can
// place two stops in an order no train ever runs.

import { undirected, type AdjEdge, type RoutePattern, type RouteStops } from "./stations.ts";
/** One traversed pairwise hop: `from` → `to` on a route+direction. `key` is
 * `route|direction|from`, identical to AdjEdge.key and the segment_flow cell
 * key, so a segment joins directly to live status and scheduled timings. */
export interface JourneySegment {
  route: string;
  direction: string; // "north" | "south"
  from: string; // directional stop id, e.g. "235N"
  to: string; // directional stop id, e.g. "228N"
  key: string; // `${route}|${direction}|${from}`
}

/** One ride on one route+direction: the ordered hops actually traversed,
 * boarding at segments[0].from and alighting at the last segment's `to`. */
export interface JourneyLeg {
  route: string;
  direction: string;
  segments: JourneySegment[]; // ≥ 1, contiguous
}

/** An ordered list of legs joining the origin complex to the destination, with
 * a transfer (a walk within one shared complex) between each consecutive pair.
 * `segments` is every leg's hops concatenated in travel order — the ordered
 * (route, direction, from_stop) list the result is specified to hand back. */
export interface Journey {
  legs: JourneyLeg[];
  segments: JourneySegment[];
  transfers: number; // legs.length - 1
}

export interface JourneyOptions {
  // Extra walkable complexes: each a set of undirected GTFS stop ids a rider can
  // change trains between for free. The topology keys stops per route, so a
  // single multi-line station like Union Sq is several ids (635, R20, L03) that
  // only these groups can tell apart from unrelated stations. Two routes sharing
  // one undirected stop id are always a valid transfer without listing it here;
  // groups are only needed to unify distinct ids under one complex.
  complexes?: ReadonlyArray<ReadonlyArray<string>>;
  // Most transfers to consider. Capped at 2 by design — this is candidate
  // enumeration, not a planner. Two-transfer journeys are only ever produced
  // when no direct route and no single transfer connect the complexes.
  maxTransfers?: 0 | 1 | 2;
}

/** The directional stop a leg boards at / alights at. */
export const boardStop = (leg: JourneyLeg): string => leg.segments[0].from;
export const alightStop = (leg: JourneyLeg): string =>
  leg.segments[leg.segments.length - 1].to;

/** Every pairwise segment key traversed across the whole journey, in order. */
export const journeyHopKeys = (journey: Journey): string[] =>
  journey.segments.map((s) => s.key);

/** One journey's stable identity: its route sequence alone, matching the
 * enumerator's dedup key (opposite directions of one sequence collapse to a
 * single candidate). Route ids never contain a hyphen, so this round-trips as
 * a URL param. Shared by the trip page's selection and the ranking surface so
 * both address a candidate by exactly the same key. */
export const journeyId = (journey: Journey): string =>
  journey.legs.map((l) => l.route).join("-");

// A group is identified by the lexicographically-smallest undirected stop id it
// contains — stable, and independent of the order stops were listed in.
const groupKeyOf = (stops: Iterable<string>): string => {
  let min: string | null = null;
  for (const s of stops) if (min === null || s < min) min = s;
  return min ?? "";
};

// Resolve every stop to the key of the walkable complex it belongs to. Stops
// named by no group are their own singleton complex, so a plain shared stop id
// is still a transfer point. Origin and destination are groups too, so their
// several platforms collapse to one key and never masquerade as a transfer.
function buildGroups(
  origin: ReadonlyArray<string>,
  destination: ReadonlyArray<string>,
  extra: ReadonlyArray<ReadonlyArray<string>>,
): {
  keyOf: (stop: string) => string;
  stopsOf: (key: string) => Set<string>;
  originKey: string;
  destKey: string;
} {
  const norm = (g: ReadonlyArray<string>) => [...new Set(g.map(undirected))];
  const groups = [norm(origin), norm(destination), ...extra.map(norm)].filter(
    (g) => g.length > 0,
  );
  const stopToKey = new Map<string, string>();
  const keyToStops = new Map<string, Set<string>>();
  for (const g of groups) {
    const key = groupKeyOf(g);
    const set = keyToStops.get(key) ?? new Set<string>();
    for (const s of g) {
      if (!stopToKey.has(s)) stopToKey.set(s, key);
      set.add(s);
    }
    keyToStops.set(key, set);
  }
  const keyOf = (stop: string): string => stopToKey.get(undirected(stop)) ?? undirected(stop);
  const stopsOf = (key: string): Set<string> => keyToStops.get(key) ?? new Set([key]);
  return {
    keyOf,
    stopsOf,
    originKey: groupKeyOf(norm(origin)),
    destKey: groupKeyOf(norm(destination)),
  };
}

// The route+direction lines that have any published pattern, in sorted key
// order so enumeration is deterministic regardless of object insertion order.
const lineKeys = (routeStops: RouteStops): string[] =>
  Object.keys(routeStops)
    .filter((k) => (routeStops[k]?.length ?? 0) > 0)
    .sort();

const parseLine = (key: string): { route: string; direction: string } => {
  const bar = key.indexOf("|");
  return { route: key.slice(0, bar), direction: key.slice(bar + 1) };
};

// `route|direction|from` → the set of directional successors the adjacency graph
// backs. A pattern hop absent here is a stop pair no segment cell measures, so a
// ride cannot legitimately claim it — the scan breaks the pattern there.
const buildSuccessors = (edges: AdjEdge[]): Map<string, Set<string>> => {
  const m = new Map<string, Set<string>>();
  for (const e of edges) {
    let set = m.get(e.key);
    if (!set) {
      set = new Set<string>();
      m.set(e.key, set);
    }
    for (const s of e.successors) set.add(s.to);
  }
  return m;
};

// From a boarding complex, every complex reachable on this line, with the exact
// ordered segments ridden to get there. Proven per pattern: within one pattern
// we find the earliest stop in the boarding set, then walk forward one adjacency
// hop at a time. A hop the successor graph doesn't back stops that pattern's ride
// (you cannot continue past a pair no cell measures). The most-run pattern is
// scanned first, so the dominant service wins when several reach the same
// complex. Keyed by destination complex; the boarding complex is never one.
function reachFrom(
  route: string,
  direction: string,
  patterns: RoutePattern[],
  boardSet: Set<string>,
  keyOf: (stop: string) => string,
  succ: Map<string, Set<string>>,
): Map<string, JourneySegment[]> {
  const out = new Map<string, JourneySegment[]>();
  for (const pat of patterns) {
    const stops = pat.stops;
    let bi = -1;
    for (let i = 0; i < stops.length; i++) {
      if (boardSet.has(undirected(stops[i]))) {
        bi = i;
        break;
      }
    }
    if (bi < 0) continue;
    const boardKey = keyOf(stops[bi]);
    const segs: JourneySegment[] = [];
    for (let i = bi + 1; i < stops.length; i++) {
      const from = stops[i - 1];
      const to = stops[i];
      const key = `${route}|${direction}|${from}`;
      if (!succ.get(key)?.has(to)) break; // unbacked pair → ride ends here
      segs.push({ route, direction, from, to, key });
      const ck = keyOf(to);
      if (ck === boardKey) continue; // still inside the boarding complex
      if (!out.has(ck)) out.set(ck, [...segs]);
    }
  }
  return out;
}

const makeLeg = (segments: JourneySegment[]): JourneyLeg => ({
  route: segments[0].route,
  direction: segments[0].direction,
  segments,
});

// Dedup + ordering key: the route sequence alone, directions excluded. One
// canonical journey is emitted per sequence of routes — the first the
// enumeration encounters, which for a single transfer is the shared complex
// reached earliest along leg A's ride, boarded in sorted line order. Opposite
// running directions of the same route sequence are the same candidate here;
// the concrete directions still travel in each leg's segments. Which running of
// a sequence, and which transfer point, is preferable is a scoring question for
// the comparison surface — this only proves the sequence connects and hands
// back one concrete instance of it.
const signature = (legs: JourneyLeg[]): string => legs.map((l) => l.route).join(" / ");

/**
 * Enumerate the plausible journeys from the origin complex to the destination
 * complex over the committed topology's stopping patterns and successor graph.
 *
 * @param routeStops  route|direction → ordered patterns (Topology.routeStops).
 * @param edges       directed successor cells (Topology.edges / diagram.adjacency).
 * @param origin      undirected (or directional) GTFS stop ids of the origin complex.
 * @param destination undirected (or directional) GTFS stop ids of the destination complex.
 *
 * Returns direct routes and single transfers always; two-transfer journeys only
 * when neither connects. Deterministic order: fewest transfers first, then by
 * leg signature. Never a self-transfer (a leg to the same route).
 */
export function enumerateJourneys(
  routeStops: RouteStops,
  edges: AdjEdge[],
  origin: ReadonlyArray<string>,
  destination: ReadonlyArray<string>,
  options: JourneyOptions = {},
): Journey[] {
  const maxTransfers = options.maxTransfers ?? 2;
  const { keyOf, stopsOf, originKey, destKey } = buildGroups(
    origin,
    destination,
    options.complexes ?? [],
  );
  if (originKey === "" || destKey === "" || originKey === destKey) return [];

  const succ = buildSuccessors(edges);
  const lines = lineKeys(routeStops);

  // reachFrom is called repeatedly from the same boarding complexes; memoize by
  // (line, complex key) so the two-transfer fallback stays cheap.
  const reachCache = new Map<string, Map<string, JourneySegment[]>>();
  const reach = (line: string, complexKey: string): Map<string, JourneySegment[]> => {
    const cacheKey = `${line}\u0000${complexKey}`;
    let m = reachCache.get(cacheKey);
    if (!m) {
      const { route, direction } = parseLine(line);
      m = reachFrom(route, direction, routeStops[line], stopsOf(complexKey), keyOf, succ);
      reachCache.set(cacheKey, m);
    }
    return m;
  };

  const seen = new Set<string>();
  const journeys: Journey[] = [];
  const add = (legs: JourneyLeg[]) => {
    const sig = signature(legs);
    if (seen.has(sig)) return;
    seen.add(sig);
    journeys.push({ legs, segments: legs.flatMap((l) => l.segments), transfers: legs.length - 1 });
  };

  // Direct: one line that boards at the origin and later stops at the destination.
  for (const line of lines) {
    const segs = reach(line, originKey).get(destKey);
    if (segs) add([makeLeg(segs)]);
  }

  // Single transfer: line A origin → shared complex T, line B (different route)
  // T → destination. T must be a genuine intermediate complex, not an endpoint.
  if (maxTransfers >= 1) {
    for (const a of lines) {
      const routeA = parseLine(a).route;
      for (const [tKey, segsA] of reach(a, originKey)) {
        if (tKey === originKey || tKey === destKey) continue;
        for (const b of lines) {
          if (parseLine(b).route === routeA) continue;
          const segsB = reach(b, tKey).get(destKey);
          if (segsB) add([makeLeg(segsA), makeLeg(segsB)]);
        }
      }
    }
  }

  // Two transfers, only as a fallback: A origin → T1, B T1 → T2, C T2 → dest,
  // with consecutive routes distinct and T1 ≠ T2, neither being an endpoint.
  if (maxTransfers >= 2 && journeys.length === 0) {
    for (const a of lines) {
      const routeA = parseLine(a).route;
      for (const [t1, segsA] of reach(a, originKey)) {
        if (t1 === originKey || t1 === destKey) continue;
        for (const b of lines) {
          const routeB = parseLine(b).route;
          if (routeB === routeA) continue;
          for (const [t2, segsB] of reach(b, t1)) {
            if (t2 === originKey || t2 === destKey || t2 === t1) continue;
            for (const c of lines) {
              if (parseLine(c).route === routeB) continue;
              const segsC = reach(c, t2).get(destKey);
              if (segsC) add([makeLeg(segsA), makeLeg(segsB), makeLeg(segsC)]);
            }
          }
        }
      }
    }
  }

  journeys.sort((x, y) =>
    x.transfers !== y.transfers
      ? x.transfers - y.transfers
      : signature(x.legs).localeCompare(signature(y.legs)),
  );
  return journeys;
}
