import { test } from "node:test";
import assert from "node:assert/strict";
import { journeyId, type Journey, type JourneyLeg, type JourneySegment } from "../lib/journeys.ts";
import { rankJourneys, journeyVerdict, scoreJourney } from "../lib/rankJourneys.ts";
import type { RouteStatus, SegmentRecovery, SegmentStatus, Snapshot } from "../lib/types.ts";

// --- Builders --------------------------------------------------------------
// Minimal, structurally-valid inputs: the ranker reads only segment_flow,
// route_status and stations off the snapshot, and legs/segments off a journey.

const seg = (route: string, dir: string, from: string, to: string): JourneySegment => ({
  route,
  direction: dir,
  from,
  to,
  key: `${route}|${dir}|${from}`,
});

// A leg from an ordered list of directional stops on one route+direction.
const leg = (route: string, dir: string, stops: string[]): JourneyLeg => ({
  route,
  direction: dir,
  segments: stops.slice(0, -1).map((s, i) => seg(route, dir, s, stops[i + 1])),
});

const journey = (...legs: JourneyLeg[]): Journey => ({
  legs,
  segments: legs.flatMap((l) => l.segments),
  transfers: legs.length - 1,
});

const recovery = (minutes: number, indeterminate = false): SegmentRecovery => ({
  recovery_minutes: minutes,
  recovery_minutes_low: minutes,
  recovery_minutes_high: minutes,
  recovery_indeterminate: indeterminate,
  p_normal_in_30min: 0.5,
  p_normal_in_60min: 0.7,
  p_normal_in_120min: 0.9,
});

const cell = (
  key: string,
  status: SegmentStatus["status"],
  rec: SegmentRecovery | null = null,
  to?: string | null,
): SegmentStatus => {
  const [route, direction, from_stop] = key.split("|");
  // Fixture stops are `<letter><n>` and `leg` chains them in order, so the
  // drawn successor of `a1` is `a2`. Pass `to` explicitly to exercise a
  // wrong-successor or attribution-less (null) reading.
  const derived = from_stop.replace(/\d+$/, (n) => String(Number(n) + 1));
  return {
    route,
    direction,
    from_stop,
    to: to === undefined ? derived : to,
    status,
    entered_at: 0,
    recovery: rec,
  };
};

const routeStatus = (
  condition: string,
  service_condition = "normal",
  service_ratio = 1,
): RouteStatus =>
  // Only condition + supply fields drive scoring; the rest are type completeness.
  ({
    condition,
    service_condition,
    service_ratio,
    service_low_ratio: null,
    service_high_ratio: null,
  }) as unknown as RouteStatus;

function snapshot(opts: {
  segments?: Record<string, SegmentStatus>;
  routes?: Record<string, RouteStatus>;
  names?: Record<string, string>;
}): Snapshot {
  const stations: Record<string, { name: string }> = {};
  for (const [id, name] of Object.entries(opts.names ?? {})) stations[id] = { name };
  // Partial fixture: the ranker reads only these three fields off a snapshot.
  return {
    segment_flow: opts.segments ? { observed_at: 0, segments: opts.segments } : null,
    route_status: opts.routes ?? {},
    stations,
  } as unknown as Snapshot;
}

// --- journeyId -------------------------------------------------------------

test("journeyId is the hyphen-joined route sequence and round-trips", () => {
  const j = journey(leg("A", "north", ["a1", "a2"]), leg("F", "north", ["f1", "f2"]));
  assert.equal(journeyId(j), "A-F");
});

// --- Core severity ordering ------------------------------------------------

test("a clean journey outranks one with a disrupted hop, and the verdict names it", () => {
  const clean = journey(leg("A", "north", ["a1", "a2", "a3"]));
  const bad = journey(leg("F", "north", ["f1", "f2"]));
  const snap = snapshot({
    segments: {
      "A|north|a1": cell("A|north|a1", "normal"),
      "A|north|a2": cell("A|north|a2", "normal"),
      "F|north|f1": cell("F|north|f1", "disrupted", recovery(20)),
    },
    routes: { A: routeStatus("normal"), F: routeStatus("disrupted") },
    names: { f2: "Bergen St" },
  });

  const v = journeyVerdict(snap, [bad, clean])!;
  assert.equal(v.best.id, "A");
  assert.equal(v.tone, "disrupted");
  assert.equal(v.culpritRoute, "F");
  assert.match(v.reason, /disrupted at Bergen St/);
  assert.match(v.reason, /20m to clear/);
});

test("a suspended line dominates: any candidate touching it ranks last", () => {
  // The kept candidate has THREE disrupted hops; the loser has none but rides a
  // suspended line. No amount of disruption outweighs a line with no trains.
  const disrupted = journey(leg("A", "north", ["a1", "a2", "a3", "a4"]));
  const suspended = journey(leg("R", "north", ["r1", "r2"]));
  const snap = snapshot({
    segments: {
      "A|north|a1": cell("A|north|a1", "disrupted", recovery(5)),
      "A|north|a2": cell("A|north|a2", "disrupted", recovery(5)),
      "A|north|a3": cell("A|north|a3", "disrupted", recovery(5)),
      "R|north|r1": cell("R|north|r1", "normal"),
    },
    routes: { A: routeStatus("disrupted"), R: routeStatus("suspended") },
  });

  const v = journeyVerdict(snap, [disrupted, suspended])!;
  assert.equal(v.best.id, "A");
  assert.equal(v.tone, "suspended");
  assert.equal(v.culpritRoute, "R");
  assert.match(v.reason, /suspended/);
});

test("disrupted-hop COUNT dominates recovery duration", () => {
  // One long (indeterminate) disruption beats two short ones: fewer bad hops.
  const oneLong = journey(leg("A", "north", ["a1", "a2"]));
  const twoShort = journey(leg("B", "north", ["b1", "b2", "b3"]));
  const snap = snapshot({
    segments: {
      "A|north|a1": cell("A|north|a1", "disrupted", recovery(999, true)),
      "B|north|b1": cell("B|north|b1", "disrupted", recovery(3)),
      "B|north|b2": cell("B|north|b2", "disrupted", recovery(3)),
    },
    routes: { A: routeStatus("disrupted"), B: routeStatus("disrupted") },
  });
  const ranked = rankJourneys(snap, [twoShort, oneLong]);
  assert.equal(ranked[0].id, "A");
  assert.equal(ranked[0].disrupted, 1);
  assert.equal(ranked[1].disrupted, 2);
});

test("with equal disrupted counts, longer recovery ranks worse and is cited", () => {
  const quick = journey(leg("A", "north", ["a1", "a2"]));
  const slow = journey(leg("F", "north", ["f1", "f2"]));
  const snap = snapshot({
    segments: {
      "A|north|a1": cell("A|north|a1", "disrupted", recovery(5)),
      "F|north|f1": cell("F|north|f1", "disrupted", recovery(45)),
    },
    routes: { A: routeStatus("disrupted"), F: routeStatus("disrupted") },
    names: { f2: "Jay St" },
  });
  const v = journeyVerdict(snap, [slow, quick])!;
  assert.equal(v.best.id, "A");
  assert.equal(v.culpritRoute, "F");
  assert.match(v.reason, /45m to clear/);
});

test("low supply is penalised and surfaced as the reason", () => {
  const good = journey(leg("A", "north", ["a1", "a2"]));
  const thin = journey(leg("C", "north", ["c1", "c2"]));
  const snap = snapshot({
    segments: {
      "A|north|a1": cell("A|north|a1", "normal"),
      "C|north|c1": cell("C|north|c1", "normal"),
    },
    routes: { A: routeStatus("normal"), C: routeStatus("normal", "degraded", 0.4) },
  });
  const v = journeyVerdict(snap, [thin, good])!;
  assert.equal(v.best.id, "A");
  assert.equal(v.tone, "supply");
  assert.equal(v.culpritRoute, "C");
  assert.match(v.reason, /fewer trains/);
});

// --- Ties: live status only, structure never a verdict reason --------------

test("a live-status tie yields an 'even' verdict, never a 'shorter/direct' claim", () => {
  // Two all-normal candidates that differ ONLY in transfers. The transfer count
  // fixes which shows as best, but is NOT a quality reason.
  const direct = journey(leg("A", "north", ["a1", "a2"]));
  const twoLeg = journey(leg("B", "north", ["b1", "b2"]), leg("C", "north", ["c1", "c2"]));
  const snap = snapshot({
    segments: {
      "A|north|a1": cell("A|north|a1", "normal"),
      "B|north|b1": cell("B|north|b1", "normal"),
      "C|north|c1": cell("C|north|c1", "normal"),
    },
    routes: { A: routeStatus("normal"), B: routeStatus("normal"), C: routeStatus("normal") },
  });
  const v = journeyVerdict(snap, [twoLeg, direct])!;
  assert.equal(v.best.id, "A"); // fewer transfers wins the deterministic tie-break
  assert.equal(v.tone, "even");
  assert.doesNotMatch(v.reason, /transfer|shorter|direct/i);
  assert.match(v.reason, /running clean/);
});

test("transfer/hop count never enter the quality penalty", () => {
  // A direct clean ride and a two-transfer clean ride have identical penalties:
  // structure is not a quality axis.
  const direct = scoreJourney(
    snapshot({ segments: { "A|north|a1": cell("A|north|a1", "normal") }, routes: { A: routeStatus("normal") } }),
    journey(leg("A", "north", ["a1", "a2"])),
  );
  const longer = scoreJourney(
    snapshot({
      segments: {
        "A|north|a1": cell("A|north|a1", "normal"),
        "B|north|b1": cell("B|north|b1", "normal"),
        "C|north|c1": cell("C|north|c1", "normal"),
      },
      routes: { A: routeStatus("normal"), B: routeStatus("normal"), C: routeStatus("normal") },
    }),
    journey(leg("A", "north", ["a1", "a2"]), leg("B", "north", ["b1", "b2"]), leg("C", "north", ["c1", "c2"])),
  );
  assert.equal(direct.penalty, 0);
  assert.equal(longer.penalty, 0);
});

// --- Tone escalation -------------------------------------------------------

test("banner tone escalates to the winner's own worst state, not just the reason", () => {
  // Best and runner-up both carry one disrupted hop, so the DIFFERENTIATOR is
  // the runner-up's extra quiet hops. The reason speaks to that, but the tone
  // must still be 'disrupted' because the recommended ride itself is disrupted.
  const best = journey(leg("A", "north", ["a1", "a2"]));
  const worse = journey(leg("B", "north", ["b1", "b2", "b3", "b4"]));
  const snap = snapshot({
    segments: {
      "A|north|a1": cell("A|north|a1", "disrupted", recovery(5)),
      "B|north|b1": cell("B|north|b1", "disrupted", recovery(5)),
      "B|north|b2": cell("B|north|b2", "quiet"),
      "B|north|b3": cell("B|north|b3", "quiet"),
    },
    routes: { A: routeStatus("disrupted"), B: routeStatus("disrupted") },
  });
  const v = journeyVerdict(snap, [worse, best])!;
  assert.equal(v.best.id, "A");
  assert.equal(v.tone, "disrupted"); // escalated, though the reason is about quiet hops
  assert.match(v.reason, /sparse/);
});

// --- Single candidate ------------------------------------------------------

test("a sole candidate is described by its own live status", () => {
  const only = journey(leg("A", "north", ["a1", "a2"]));
  const snap = snapshot({
    segments: { "A|north|a1": cell("A|north|a1", "normal") },
    routes: { A: routeStatus("normal") },
  });
  const v = journeyVerdict(snap, [only])!;
  assert.equal(v.singleCandidate, true);
  assert.equal(v.tone, "clear");
  assert.match(v.reason, /running normally/);
});

test("a sole candidate with quiet hops is NOT reported as running normally", () => {
  const only = journey(leg("A", "north", ["a1", "a2", "a3"]));
  const snap = snapshot({
    segments: {
      "A|north|a1": cell("A|north|a1", "normal"),
      "A|north|a2": cell("A|north|a2", "quiet"),
    },
    routes: { A: routeStatus("normal") },
  });
  const v = journeyVerdict(snap, [only])!;
  assert.equal(v.singleCandidate, true);
  assert.equal(v.tone, "quiet");
  assert.match(v.reason, /sparse/);
  assert.doesNotMatch(v.reason, /normally/);
});

test("an unread hop counts as uncertainty, not as normal", () => {
  const j = journey(leg("A", "north", ["a1", "a2", "a3"]));
  const snap = snapshot({
    segments: { "A|north|a1": cell("A|north|a1", "normal") }, // a2 hop unjudged
    routes: { A: routeStatus("normal") },
  });
  const s = scoreJourney(snap, j);
  assert.equal(s.normal, 1);
  assert.equal(s.unknown, 1);
  assert.ok(s.penalty > 0);
});

// --- Attribution: `to` is load-bearing at branches ---------------------------

test("a disrupted reading about a different successor is no evidence against this journey", () => {
  const j = journey(leg("A", "north", ["a1", "a2"]));
  const snap = snapshot({
    // The cell key matches, but the reading is about the OTHER branch (a9).
    segments: { "A|north|a1": cell("A|north|a1", "disrupted", recovery(30), "a9") },
    routes: { A: routeStatus("normal") },
  });
  const s = scoreJourney(snap, j);
  assert.equal(s.disrupted, 0);
  assert.equal(s.unknown, 1);
  assert.equal(s.worst, null);
});

test("a reading that cannot name its successor abstains instead of attributing", () => {
  const j = journey(leg("A", "north", ["a1", "a2"]));
  const snap = snapshot({
    segments: { "A|north|a1": cell("A|north|a1", "disrupted", recovery(30), null) },
    routes: { A: routeStatus("normal") },
  });
  const s = scoreJourney(snap, j);
  assert.equal(s.disrupted, 0);
  assert.equal(s.unknown, 1);
});

test("a not-scheduled route is as unusable as a suspended one and the verdict says why", () => {
  const running = journey(leg("A", "north", ["a1", "a2"]));
  const ghost = journey(leg("F", "north", ["f1", "f2"]));
  const snap = snapshot({
    segments: {},
    routes: { A: routeStatus("normal"), F: routeStatus("not_scheduled") },
  });
  const v = journeyVerdict(snap, [ghost, running])!;
  assert.equal(v.best.routes[0], "A");
  assert.equal(v.culpritRoute, "F");
  assert.match(v.reason, /no trains running/);
});
