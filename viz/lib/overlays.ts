// The map's overlay registry: four different questions asked of one piece of
// geometry. They are not interchangeable, and the thing this module exists to
// keep straight is that each answers at a DIFFERENT SPATIAL UNIT:
//
//   movement   per (route, direction, station pair) — a segment was measured.
//   supply     per ROUTE — every edge of a route wears the route's reading, so
//              a lit edge here was never measured on its own.
//   time       per drawn hop and direction — the static timetable's scheduled
//              run time, not an observation of today's trains at all.
//   trains     per STOP — where trains are (or are heading), with no claim
//              about which segment a moving one occupies.
//
// Collapsing those units is the failure mode: a reader who takes a supply-lit
// edge for a measured segment has been lied to by the map, not by the feed. So
// every entry carries its own caption naming its unit and its source, and the
// page is required to show it.
//
// The second invariant, inherited from ./segments: no-reading is never painted
// as healthy. Each overlay has its own flavour of "no reading" — an unmeasured
// cell, a route the feed didn't judge, a hop the timetable doesn't time, a
// missing surface — and all of them render as the dimmed route colour.

// Explicit .ts specifiers: this is the first lib module to import VALUES from
// its siblings rather than types, and `node --test` strips types without
// resolving extensionless specifiers. tsconfig has allowImportingTsExtensions
// for the same reason the test files do.
import { edgePath } from "./diagram.ts";
import type { Diagram, DiagramEdge, Direction, ServiceClass } from "./diagram.ts";
import { fmtAgo, fmtMinutes } from "./feed.ts";
import type { TrainsFeed } from "./feed.ts";
import {
  DIRECTIONS,
  PAINT_ORDER,
  coverage,
  readEdge,
  selectReading,
  stationOf,
} from "./segments.ts";
import type { DirectionFilter, SegmentState } from "./segments.ts";
import type { Snapshot, TrainPosition, Trains } from "./types.ts";

/** The station name a directional stop id resolves to on the diagram, falling
 * back to the raw id when the asset predates the stop. Both a disrupted cell's
 * successor and a normal verdict's successor need it in lockstep. */
function stopName(diagram: Diagram, stop: string): string {
  return diagram.stations[stationOf(stop)]?.name ?? stop;
}

export type OverlayId = "movement" | "supply" | "time" | "trains";

/** A run of caption/note prose. `em` marks the phrase the sentence turns on,
 * which keeps the registry React-free without losing the emphasis. */
export type NoteSpan = string | { em: string };

/** How one edge should be stroked. `null` from `paint` means "draw nothing
 * here": the overlay has no business on this edge at all. */
export interface EdgePaint {
  /** null = wear the route's own colour. Only a reading earns a scale colour,
   * so the no-reading majority reads as the line map at rest. */
  color: string | null;
  width: number;
  opacity: number;
  /** Dashed = the colour came from a COARSER unit than this edge. Echoes the
   * dashed supply chip on the route cards, where the same distinction between
   * the flow axis and the supply axis is already drawn. */
  dash: string | null;
  /** Lower paints first, so a verdict lands on top of the ghosts it crosses
   * instead of under them. */
  order: number;
}

/** The plain line map: the route's own colour, undimmed, no claim attached.
 * What an overlay draws where it has nothing to add to the geometry. */
export const REST_PAINT: EdgePaint = {
  color: null,
  width: 1.9,
  opacity: 0.9,
  dash: null,
  order: 1,
};

/** The one shape of "we don't know" in this whole view: the route's own
 * colour, dimmed and thinned. Every overlay's no-reading state uses it — an
 * unmeasured segment, an unjudged route, an untimed hop, a route whose vehicle
 * feed didn't report — so a reader learns the vocabulary once and it holds. */
export const UNKNOWN_PAINT: EdgePaint = {
  color: null,
  width: 1.7,
  opacity: 0.45,
  dash: null,
  order: 0,
};

/** One line of the hover/pin detail panel. `key` is a short gutter label — the
 * panel's first column is narrow, so anything longer belongs in `value`. */
export interface DetailRow {
  key: string;
  value: string;
  /** Colour for `value`; null takes the muted default. */
  color: string | null;
  /** Provenance or detail under it. */
  note: string | null;
}

export interface LegendItem {
  label: string;
  /** CSS colour for the swatch, or null for the dimmed route-colour stand-in
   * every overlay's no-reading state uses. */
  color: string | null;
  shape: "line" | "dash" | "dot" | "ring";
  /** Hover text for a swatch whose label can't carry everything — the ramp's
   * bin populations, which explain why the ranges are the widths they are. */
  title?: string;
}

/** The freshness stamp for the overlay's data. null for an overlay that reads
 * a static asset, which has a timetable version rather than an observation
 * time — claiming an age for it would be a fabricated timestamp. */
export interface OverlayStamp {
  label: string;
  at: number | null;
}

export interface FilterOption {
  id: DirectionFilter;
  label: string;
}

export interface ServiceClassOption {
  id: ServiceClass;
  label: string;
}

export const SERVICE_CLASSES: readonly ServiceClassOption[] = [
  { id: "weekday", label: "Weekday" },
  { id: "saturday", label: "Saturday" },
  { id: "sunday", label: "Sunday" },
];

/** Which timetable today is, by the calendar in New York — the sensible thing
 * to open the overlay on, since a reader looking at a live map means today.
 *
 * It is a default, not a claim: NYCT runs a weekend timetable on some
 * holidays, the static asset says nothing about holidays, and nothing here
 * pretends to know. The class is always named in the control and the caption,
 * so a reader can see which schedule they are looking at and change it. */
export function serviceClassNow(nowMs: number): ServiceClass {
  const day = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/New_York",
    weekday: "short",
  }).format(new Date(nowMs));
  if (day === "Sat") return "saturday";
  if (day === "Sun") return "sunday";
  return "weekday";
}

export interface OverlayContext {
  diagram: Diagram;
  snap: Snapshot | null;
  filter: DirectionFilter;
  /** Wall clock the relative timestamps are measured against — the moment the
   * snapshot was fetched, not `Date.now()`, so the panel and the header agree. */
  now: number;
  /** Which of NYCT's timetables the scheduled-time overlay reads. */
  serviceClass: ServiceClass;
  /** The scheduled-time bins for this asset and service class, or null when
   * that class carries no timings. Derived once per (asset, class) and passed
   * in, because it is a sort over every hop and `paint` runs 988 times. */
  time: TimeScale | null;
  /** The train surface. Published as its own object beside the snapshot, with
   * its own clock and its own failure mode, so it is fetched separately and
   * only while the overlay that needs it is selected. */
  trains: TrainsFeed;
}

export interface Overlay {
  id: OverlayId;
  label: string;
  /** Names the spatial unit and the source. Shown under the map, always. */
  caption: NoteSpan[];
  /** What this overlay covers right now, in prose. */
  note(ctx: OverlayContext): NoteSpan[];
  legend(ctx: OverlayContext): LegendItem[];
  detail(edge: DiagramEdge, ctx: OverlayContext): DetailRow[];
  stamp(ctx: OverlayContext): OverlayStamp | null;
  /** Direction filter choices, or null when the overlay isn't directional.
   * The control is hidden rather than disabled for those: a filter that does
   * nothing is worse than no filter. */
  filters: readonly FilterOption[] | null;
  /** Service-class choices, for the overlay that reads a timetable. Same
   * contract as `filters`: absent hides the control entirely. */
  classes?: readonly ServiceClassOption[];
  /** A caveat about the reading itself, rendered as a warning box above the
   * map rather than a tooltip: it changes what the picture means, so a reader
   * must not be able to miss it. Omitted by overlays that never have one. */
  caveat?(ctx: OverlayContext): NoteSpan[] | null;
  /** How each edge is stroked. `null` = draw nothing here, the overlay has no
   * business on this edge at all.
   *
   * Every overlay paints edges, including the ones whose real subject is a
   * point layer: the train overlay has to dim the routes whose vehicle feed
   * didn't report, and "unknown" is an edge-level statement. A layer overlay
   * additionally registers a component in the page's LAYERS table. */
  paint(edge: DiagramEdge, ctx: OverlayContext): EdgePaint | null;
  // Every measurement on this map is about exactly one edge or one route, so
  // there is no grouping hook: a stroke is an edge. If a surface ever
  // publishes a measurement spanning several hops again, it needs one — one
  // reading drawn as N coloured hops presents it as N.
}

// --- Movement: per (route, direction, station pair) ------------------------

// Stroke weight and opacity per state. `unmeasured` keeps the route's own
// colour, dimmed: the no-reading majority should read as the line map at rest
// so a published verdict is the thing that lights up. It must never take the
// healthy colour.
const STROKE: Record<SegmentState, { width: number; opacity: number }> = {
  unscheduled: { width: 0, opacity: 0 },
  unmeasured: { width: 1.7, opacity: 0.45 },
  normal: { width: 2.4, opacity: 0.95 },
  disrupted: { width: 3.6, opacity: 1 },
};

// Null means "wear the route colour" — only a verdict gets a state colour.
export const STATE_VAR: Record<SegmentState, string | null> = {
  unscheduled: null,
  unmeasured: null,
  normal: "var(--normal)",
  disrupted: "var(--disrupted)",
};

const STATE_LABEL: Record<SegmentState, string> = {
  unscheduled: "not scheduled here",
  unmeasured: "no reading",
  normal: "advancing",
  disrupted: "not advancing",
};

const MOVEMENT: Overlay = {
  id: "movement",
  label: "Movement",
  filters: [
    { id: "both", label: "Worst of both" },
    { id: "north", label: "Northbound" },
    { id: "south", label: "Southbound" },
  ],
  caption: [
    "Unit: one directional segment — a (route, direction, station pair) cell " +
      "of the published movement surface, drawn only on the successor the cell " +
      "names. Every judged cell publishes a full record; a healthy one carries " +
      "a null recovery, since there's nothing to forecast on track that isn't " +
      "disrupted.",
  ],
  paint(edge, ctx) {
    const side = selectReading(
      readEdge(edge, ctx.snap?.segment_flow ?? null),
      ctx.filter,
    );
    if (side.state === "unscheduled") return null;
    return {
      color: STATE_VAR[side.state],
      width: STROKE[side.state].width,
      opacity: STROKE[side.state].opacity,
      // Every reading on this overlay is about exactly one cell, so nothing
      // here is dashed. On this map a dash means one thing and one thing only:
      // the reading came from a coarser unit than the edge it covers, which is
      // true of the supply overlay and of nothing else.
      dash: null,
      order: PAINT_ORDER[side.state],
    };
  },
  detail(edge, ctx) {
    const reading = readEdge(edge, ctx.snap?.segment_flow ?? null);
    return DIRECTIONS.map((direction) => {
      const side = reading[direction];
      const cell = side.cell;
      const parts: string[] = [];
      if (cell !== null) {
        parts.push(`since ${fmtAgo(cell.entered_at, ctx.now)}`);
        if (cell.to !== null) {
          parts.push(`toward ${stopName(ctx.diagram, cell.to)}`);
        }
        // Recovery is the answer to "when does this come back", which only
        // means something while the segment is down — a normal cell's
        // recovery is always null, so this never shows on a healthy row.
        const recovery =
          cell.recovery?.recovery_indeterminate === false ? cell.recovery : null;
        if (recovery !== null) {
          parts.push(`recovery ~${fmtMinutes(recovery.recovery_minutes)}`);
        }
      } else {
        parts.push(side.key ?? "no cell in this direction");
      }
      return {
        key: direction === "north" ? "N" : "S",
        value: STATE_LABEL[side.state],
        color: STATE_VAR[side.state],
        note: parts.join(" · "),
      };
    });
  },
  legend: () => [
    { label: "not advancing", color: "var(--disrupted)", shape: "line" },
    { label: "advancing", color: "var(--normal)", shape: "line" },
    { label: "no reading — route colour, dimmed", color: null, shape: "line" },
  ],
  stamp(ctx) {
    return {
      label: "segment surface",
      at: ctx.snap?.segment_flow?.observed_at ?? null,
    };
  },
  note(ctx) {
    const cover = coverage(ctx.diagram, ctx.snap?.segment_flow ?? null);
    // Cells on both sides of the ratio. The denominator counts distinct cells
    // rather than (edge, direction) slots, because a branching from_stop's key
    // belongs to several drawn edges and counting slots would inflate it.
    const spans: NoteSpan[] = [
      `${cover.measured} of ${cover.scheduled} scheduled directional segments ` +
        `carry a reading right now, ${cover.disrupted} of them not advancing. ` +
        "A segment is judged only once enough matched trips accumulate in the " +
        "window, so the rest reads ",
      { em: "no reading" },
      " — an absence of evidence, not a clean bill of health.",
    ];
    if (cover.unplaced > 0) {
      spans.push(
        cover.unplaced === 1
          ? ` ${cover.unplaced} published reading sits on a pair this ` +
            "timetable doesn't schedule, so it isn't drawn."
          : ` ${cover.unplaced} published readings sit on pairs this ` +
            "timetable doesn't schedule, so they aren't drawn.",
      );
    }
    return spans;
  },
};

// --- Supply: per route ----------------------------------------------------

export type SupplyState = "normal" | "degraded" | "unknown";

export interface SupplyReading {
  route: string;
  state: SupplyState;
  /** assigned_n / the hourly baseline, or null when the Worker couldn't judge
   * it. A degraded route can still be missing a ratio. */
  ratio: number | null;
  /** false when `route_status` carries no entry for this route at all. Paints
   * the same as a published "unknown" — both are no reading — but the panel
   * says which, because a route missing from the feed is a different problem
   * from a route the Worker declined to judge. */
  present: boolean;
}

/** What the snapshot says about a route's supply. The published vocabulary is
 * 'normal' | 'degraded' | 'unknown'; anything else (an older or newer Worker)
 * is treated as unknown rather than guessed at. */
export function readSupply(route: string, snap: Snapshot | null): SupplyReading {
  const status = snap?.route_status?.[route];
  if (!status) return { route, state: "unknown", ratio: null, present: false };
  const raw = status.service_condition;
  const state: SupplyState =
    raw === "normal" || raw === "degraded" ? raw : "unknown";
  return { route, state, ratio: status.service_ratio ?? null, present: true };
}

// Same colours as the movement verdicts, because normal/degraded IS a state
// axis and the route cards already speak this trio. The dash is what separates
// them: it marks a colour that came from the route, not from this edge.
const SUPPLY_PAINT: Record<SupplyState, EdgePaint> = {
  degraded: {
    color: "var(--disrupted)",
    width: 3.2,
    opacity: 1,
    dash: "7 4",
    order: 3,
  },
  normal: {
    color: "var(--normal)",
    width: 2.4,
    opacity: 0.95,
    dash: "7 4",
    order: 2,
  },
  // No reading looks the same in every overlay: the route's own colour,
  // dimmed, undashed. `unknown` must never inherit the healthy colour just
  // because the route exists and nothing was said about it.
  unknown: { ...UNKNOWN_PAINT, order: 1 },
};

// Short, because the detail panel renders these in a pill that capitalises
// them; the sentence explaining the axis belongs in the row's note.
const SUPPLY_LABEL: Record<SupplyState, string> = {
  degraded: "supply low",
  normal: "supply normal",
  unknown: "no supply reading",
};

const SUPPLY: Overlay = {
  id: "supply",
  label: "Supply",
  // Per-route: a direction filter would imply the reading splits by direction,
  // and it does not.
  filters: null,
  caption: [
    "Unit: the whole ",
    { em: "route" },
    ". Every edge of a route wears that route's supply reading (trains " +
      "assigned vs the normal level for this hour), so a lit edge here was " +
      "never measured on its own — dashed strokes mark that coarser unit. " +
      "Source: route_status.service_condition.",
  ],
  paint(edge, ctx) {
    return SUPPLY_PAINT[readSupply(edge.route, ctx.snap).state];
  },
  detail(edge, ctx) {
    const supply = readSupply(edge.route, ctx.snap);
    const parts: string[] = [];
    if (!supply.present) {
      parts.push("this route is not in the snapshot's route_status");
    } else if (supply.ratio !== null) {
      parts.push(
        `${Math.round(supply.ratio * 100)}% of the trains normally assigned this hour`,
      );
    } else {
      parts.push("condition published without a ratio");
    }
    parts.push("route-level — not measured on this segment");
    return [
      {
        key: edge.route,
        value: SUPPLY_LABEL[supply.state],
        color: SUPPLY_PAINT[supply.state].color,
        note: parts.join(" · "),
      },
    ];
  },
  legend: () => [
    { label: "supply low", color: "var(--disrupted)", shape: "dash" },
    { label: "supply normal", color: "var(--normal)", shape: "dash" },
    { label: "no reading — route colour, dimmed", color: null, shape: "line" },
  ],
  stamp(ctx) {
    return { label: "route status", at: ctx.snap?.generated_at ?? null };
  },
  note(ctx) {
    let judged = 0;
    let degraded = 0;
    const routes = new Set(ctx.diagram.edges.map((edge) => edge.route));
    for (const route of routes) {
      const supply = readSupply(route, ctx.snap);
      if (supply.state === "unknown") continue;
      judged += 1;
      if (supply.state === "degraded") degraded += 1;
    }
    return [
      `${judged} of ${routes.size} drawn routes carry a supply reading, ` +
        `${degraded} of them low. This is a `,
      { em: "per-route" },
      " reading painted onto every one of that route's segments: it says how " +
        "many trains the schedule has out, not whether the trains on any " +
        "particular stretch of track are moving.",
    ];
  },
};

// --- Scheduled time: per drawn hop and direction ---------------------------

// A sequential ramp, deliberately none of the state colours: this is a
// magnitude, and reusing --normal/--disrupted would read as health. Violet ->
// sky, monotonic in lightness so it survives red-green colour blindness on
// lightness alone, and stroke width double-encodes the same magnitude.
const RAMP = [
  "var(--ramp-1)",
  "var(--ramp-2)",
  "var(--ramp-3)",
  "var(--ramp-4)",
  "var(--ramp-5)",
];
const RAMP_WIDTH = [1.8, 2.2, 2.6, 3.0, 3.4];

export interface TimeBin {
  /** Inclusive upper bound, in seconds. */
  max: number;
  /** Timed hops that land in this bin — what the bin is worth. */
  n: number;
  color: string;
  width: number;
}

export interface TimeScale {
  /** Ascending, at most RAMP.length entries. */
  bins: TimeBin[];
  min: number;
  max: number;
  /** Directional hops the timetable times — the ramp's sample size. */
  timed: number;
  /** Directional hops the diagram draws. Counted as (edge, direction) slots
   * and NOT as segment cells, unlike ./segments coverage: a timing belongs to
   * the hop, so the two legs of a branch have two different timings even
   * though they share one cell key. */
  slots: number;
}

/** Bins for the scheduled-time ramp, over one service class of the asset in
 * hand. Saturday and Sunday are separate timetables from the weekday one and
 * from each other, so they get separate scales: sharing one would rank a
 * Sunday hop against weekday hops it is not competing with.
 *
 * Rank bins, not equal intervals. The measurements behind that, on the weekday
 * class of the current asset: 1923 timed directional hops spanning 0:30 to
 * 16:00, but across only 28 distinct values — timetables are written in round
 * numbers, and 90 seconds alone is 40% of all hops, with 60/90/120 together
 * 76%. An equal-interval scale over that range therefore drops three quarters
 * of the network into the bottom fifth of the ramp, and plain quintiles
 * degenerate the other way: two adjacent quintile boundaries both land on 90.
 *
 * So: probe quantiles at twice the ramp's resolution, drop the boundaries the
 * ties collapse, then fold the thinnest surviving bin into a neighbour until
 * the ramp fits. That gives each of the three dominant values a bin of its own.
 *
 * Returns null when this class carries no timings at all — true of every class
 * on an asset generated before timings shipped, and the honest answer for a
 * class the timetable doesn't cover. */
export function timeScale(diagram: Diagram, cls: ServiceClass): TimeScale | null {
  const values: number[] = [];
  let slots = 0;
  for (const edge of diagram.edges) {
    for (const direction of DIRECTIONS) {
      if (edge.keys[direction] !== undefined) slots += 1;
      const seconds = edge.seconds?.[cls]?.[direction];
      if (seconds !== undefined) values.push(seconds);
    }
  }
  if (values.length === 0) return null;
  values.sort((a, b) => a - b);

  // Candidate boundaries: nearest-rank quantiles, deduped. A boundary a tie
  // already claimed would draw an empty range in the legend.
  const probes = RAMP.length * 2;
  const cut: Array<{ max: number; n: number }> = [];
  for (let i = 1; i <= probes; i += 1) {
    const at = Math.min(
      values.length - 1,
      Math.ceil((i / probes) * values.length) - 1,
    );
    if (cut.length > 0 && cut[cut.length - 1].max === values[at]) continue;
    cut.push({ max: values[at], n: 0 });
  }
  // Every value belongs to the first bin whose bound it clears. The last
  // boundary is the sample's own maximum, so the walk can't run off the end.
  let bin = 0;
  for (const value of values) {
    while (value > cut[bin].max) bin += 1;
    cut[bin].n += 1;
  }
  // Thin to the ramp by folding the emptiest bin into a neighbour. Folding
  // upward drops that bin's boundary and keeps the ones that separate the most
  // hops — an even thin would instead drop whichever boundary happened to sit
  // in the middle, which on this asset is the 2:00 one carrying 24% of hops.
  while (cut.length > RAMP.length) {
    let worst = 0;
    for (let i = 1; i < cut.length; i += 1) {
      if (cut[i].n < cut[worst].n) worst = i;
    }
    if (worst === cut.length - 1) {
      // The top bin has no upper neighbour, so extend the one below it.
      cut[worst - 1].max = cut[worst].max;
      cut[worst - 1].n += cut[worst].n;
    } else {
      cut[worst + 1].n += cut[worst].n;
    }
    cut.splice(worst, 1);
  }
  // Spread the ramp across however many bins survived, so the fastest and
  // slowest hops always take the ramp's own extremes.
  const step = cut.length === 1 ? 0 : (RAMP.length - 1) / (cut.length - 1);
  return {
    bins: cut.map((b, i) => ({
      max: b.max,
      n: b.n,
      color: RAMP[Math.round(i * step)],
      width: RAMP_WIDTH[Math.round(i * step)],
    })),
    min: values[0],
    max: values[values.length - 1],
    timed: values.length,
    slots,
  };
}

/** The bin a hop time falls in. Values above the top boundary can only come
 * from a scale built on a different asset, and land in the top bin. */
export function timeBin(scale: TimeScale, seconds: number): TimeBin {
  for (const bin of scale.bins) {
    if (seconds <= bin.max) return bin;
  }
  return scale.bins[scale.bins.length - 1];
}

/** The scheduled seconds a service class and direction filter select, or null
 * when the timetable times nothing there. "both" takes the slower direction:
 * the two directions of a hop are genuinely different runs, and the slower one
 * is the one worth seeing on a single stroke. */
export function edgeSeconds(
  edge: DiagramEdge,
  cls: ServiceClass,
  filter: DirectionFilter,
): number | null {
  const timings = edge.seconds?.[cls];
  if (timings === undefined) return null;
  if (filter !== "both") return timings[filter] ?? null;
  const north = timings.north;
  const south = timings.south;
  if (north === undefined) return south ?? null;
  if (south === undefined) return north;
  return Math.max(north, south);
}

/** m:ss above a minute, plain seconds below it. Hop times cluster around one
 * to three minutes, where "1:45" reads faster than "105s". */
export function fmtSeconds(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

const TIME: Overlay = {
  id: "time",
  label: "Scheduled time",
  classes: SERVICE_CLASSES,
  filters: [
    // Not "worst of both": a long hop is not a bad hop, and this control must
    // not import the health vocabulary into a magnitude scale.
    { id: "both", label: "Slower direction" },
    { id: "north", label: "Northbound" },
    { id: "south", label: "Southbound" },
  ],
  caption: [
    "Unit: one drawn hop and direction, from the ",
    { em: "static timetable" },
    " — the scheduled run time between the two stations, for one of NYCT's " +
      "three timetables. Nothing here is an observation of today's trains, and " +
      "the class defaults to today's day of the week in New York without " +
      "knowing about holidays, which run a weekend timetable. Colour is rank " +
      "within the selected class, not magnitude: bins hold roughly equal " +
      "numbers of hops, because the timetable writes almost every hop as 60, " +
      "90 or 120 seconds. The legend gives the real boundaries.",
  ],
  paint(edge, ctx) {
    const scale = ctx.time;
    if (scale === null) return UNKNOWN_PAINT;
    const seconds = edgeSeconds(edge, ctx.serviceClass, ctx.filter);
    if (seconds === null) return UNKNOWN_PAINT;
    const bin = timeBin(scale, seconds);
    return {
      color: bin.color,
      width: bin.width,
      opacity: 0.95,
      dash: null,
      // Slower hops on top: at overview scale the long express and Staten
      // Island runs are the shape of the scale, and they are also the rarest.
      order: 1 + scale.bins.indexOf(bin),
    };
  },
  detail(edge, ctx) {
    const timings = edge.seconds?.[ctx.serviceClass];
    return DIRECTIONS.map((direction) => {
      const seconds = timings?.[direction];
      if (seconds === undefined) {
        return {
          key: direction === "north" ? "N" : "S",
          value: "no timing",
          color: null,
          note:
            edge.keys[direction] === undefined
              ? "the timetable schedules nothing in this direction"
              : `no ${ctx.serviceClass} run time for this direction`,
        };
      }
      const bin = ctx.time === null ? null : timeBin(ctx.time, seconds);
      const parts = [`scheduled ${ctx.serviceClass} run time`, `${seconds}s`];
      if (bin !== null) parts.push(`one of ${bin.n} hops in its colour bin`);
      return {
        key: direction === "north" ? "N" : "S",
        value: fmtSeconds(seconds),
        color: bin?.color ?? null,
        note: parts.join(" · "),
      };
    });
  },
  legend(ctx) {
    const scale = ctx.time;
    if (scale === null) {
      return [
        {
          label: `no ${ctx.serviceClass} timings in this asset`,
          color: null,
          shape: "line",
        },
      ];
    }
    let low = scale.min;
    const items: LegendItem[] = scale.bins.map((bin) => {
      // Real endpoints, not bin indices: a rank scale has to show the values it
      // ranked or it is unreadable.
      const label =
        low === bin.max
          ? fmtSeconds(bin.max)
          : `${fmtSeconds(low)}–${fmtSeconds(bin.max)}`;
      low = bin.max + 1;
      return {
        label,
        color: bin.color,
        shape: "line",
        title: `${bin.n} of ${scale.timed} timed hops`,
      };
    });
    items.push({ label: "no timing", color: null, shape: "line" });
    return items;
  },
  // A static asset has a timetable version, not an observation time. The
  // version is already on the map's footer chips; inventing an age here would
  // be a fabricated timestamp.
  stamp: () => null,
  note(ctx) {
    const scale = ctx.time;
    if (scale === null) {
      return [
        `This diagram asset carries no ${ctx.serviceClass} timings, so every ` +
          "hop reads ",
        { em: "no timing" },
        ". Regenerate the asset from the static feed to populate them.",
      ];
    }
    return [
      `${scale.timed} of ${scale.slots} drawn directional hops carry a ` +
        `${ctx.serviceClass} run time, from ${fmtSeconds(scale.min)} to ` +
        `${fmtSeconds(scale.max)}. Weekend classes cover fewer hops than the ` +
        "weekday one because fewer routes and patterns run, and an untimed hop " +
        "reads as no timing rather than as a fast one. Timings are counted per " +
        "drawn hop rather than per segment cell, because the two legs of a " +
        "branch share one cell key but have two different scheduled times.",
    ];
  },
};

// --- Trains: per stop -----------------------------------------------------

export interface TrainMarker {
  /** Diagram station id — the parent of the directional stop the feed named. */
  station: string;
  x: number;
  y: number;
  /** Trains standing at this platform. */
  stopped: number;
  /** Trains the feed reports as heading toward it. Not placed on a segment:
   * see the module note on why guessing which one would be a fabrication. */
  inbound: number;
  /** The published entries folded into this marker, for the detail panel. */
  positions: TrainPosition[];
}

export interface TrainLayer {
  observed_at: number | null;
  markers: TrainMarker[];
  byStation: Record<string, TrainMarker>;
  /** Trains whose stop maps to no station on this diagram. Reported rather
   * than dropped: the realtime feed names stops a stale asset can lack, and a
   * silent drop would understate the count on the map. */
  unplaced: number;
  /** Trains counted, placed or not. */
  total: number;
  /** True when there is no train surface to place at all — not published, not
   * reachable, or not fetched yet. An absent report is not a report of zero
   * trains, so nothing may be drawn and the note has to say which it is. */
  absent: boolean;
}

/** Fold the published positions onto diagram stations.
 *
 * `trains` is null whenever the surface isn't in hand — the object isn't
 * published, the fetch failed, or it hasn't been fetched yet. Either way the
 * layer is empty and flagged absent, never an implied zero. */
export function trainLayer(diagram: Diagram, trains: Trains | null): TrainLayer {
  const byStation: Record<string, TrainMarker> = {};
  if (!trains) {
    return {
      observed_at: null,
      markers: [],
      byStation,
      unplaced: 0,
      total: 0,
      absent: true,
    };
  }
  let unplaced = 0;
  let total = 0;
  for (const position of trains.positions) {
    total += position.n;
    const id = stationOf(position.stop);
    const station = diagram.stations[id];
    if (!station) {
      unplaced += position.n;
      continue;
    }
    let marker = byStation[id];
    if (marker === undefined) {
      marker = {
        station: id,
        x: station.x,
        y: station.y,
        stopped: 0,
        inbound: 0,
        positions: [],
      };
      byStation[id] = marker;
    }
    if (position.stopped) marker.stopped += position.n;
    else marker.inbound += position.n;
    marker.positions.push(position);
  }
  const markers = Object.values(byStation);
  return {
    observed_at: trains.observed_at,
    markers,
    byStation,
    unplaced,
    total,
    absent: false,
  };
}

/** Marker radii in diagram units, before the zoom divide the page applies.
 * Area encodes the count, so the radius goes as sqrt — a station with four
 * trains reads as four times the ink, not four times the width. */
export const MARKER_R = 4.5;

export function markerRadius(count: number): number {
  return count <= 0 ? 0 : MARKER_R * Math.sqrt(count);
}

/** How complete this tick's read is.
 *
 * NYCT splits vehicles across line-group feeds and the published object names
 * which of them decoded. When one is missing, the routes behind it have no
 * positions — silence, not an empty platform. The object carries no
 * feed-name -> route-id mapping (see ./types Trains for why nobody will guess
 * one), so the only honest statement available is about the whole overlay:
 * this many of that many feeds reported, and these are the names missing. */
export interface TrainCoverage {
  fresh: number;
  expected: number;
  /** Expected feeds that didn't decode this tick, in the published order. */
  stale: string[];
  complete: boolean;
}

export function trainCoverage(trains: Trains): TrainCoverage {
  const stale = trains.expected_feeds.filter(
    (feed) => !trains.fresh_feeds.includes(feed),
  );
  return {
    fresh: trains.fresh_feeds.length,
    expected: trains.expected_feeds.length,
    stale,
    complete: stale.length === 0,
  };
}

function trainSummary(
  marker: TrainMarker | undefined,
  name: string,
  feed: TrainsFeed,
): DetailRow {
  // "No report" and "reported, nothing here" are different claims, and only
  // one of them is about this station.
  if (feed.state !== "ready") {
    return { key: "", value: name, color: null, note: "no train report to place" };
  }
  if (marker === undefined) {
    // Only sayable when every feed reported. Otherwise this station might be
    // served by a route whose feed went missing, and "no trains reported here"
    // would read as an empty platform.
    return {
      key: "",
      value: name,
      color: null,
      note: trainCoverage(feed.trains).complete
        ? "no trains reported here"
        : "nothing reported here, and this tick's read is incomplete",
    };
  }
  const parts: string[] = [];
  if (marker.stopped > 0) parts.push(`${marker.stopped} at the platform`);
  if (marker.inbound > 0) parts.push(`${marker.inbound} inbound`);
  const routes = [...new Set(marker.positions.map((p) => p.route))].sort();
  parts.push(`${routes.length === 1 ? "route" : "routes"} ${routes.join(", ")}`);
  return { key: "", value: name, color: "var(--text)", note: parts.join(" · ") };
}

const TRAINS: Overlay = {
  id: "trains",
  label: "Trains",
  // Positions carry a direction, but the marker's unit is a stop, and a
  // north/south filter over stops would suggest the map knows which track a
  // train is on. The panel breaks the routes down instead.
  filters: null,
  caption: [
    "Unit: one ",
    { em: "stop" },
    ", from the realtime vehicle feed — published as its own object at " +
      "v1/trains.json, beside the snapshot and on its own clock. NYCT names " +
      "the stop a train is heading TO while it is moving and the stop it is AT " +
      "once it has stopped, so a marker means \u201cat this platform\u201d or " +
      "\u201cinbound to it\u201d. The feed does not say which segment a moving " +
      "train occupies, and this map does not guess.",
  ],
  legend: () => [
    { label: "at the platform", color: "var(--text)", shape: "dot" },
    { label: "at the platform or inbound", color: "var(--text)", shape: "ring" },
  ],
  // The line map underneath. It goes to the shared unknown treatment whenever
  // the read isn't a complete one, because with no feed-name -> route mapping
  // the incompleteness can't be attributed to particular routes: a reader must
  // not be able to take any marker-free line for an empty one. Dimming the
  // whole map is the coarse statement that is true.
  paint(_edge, ctx) {
    const feed = ctx.trains;
    const whole = feed.state === "ready" && trainCoverage(feed.trains).complete;
    return whole ? REST_PAINT : UNKNOWN_PAINT;
  },
  caveat(ctx) {
    const feed = ctx.trains;
    if (feed.state !== "ready") return null;
    const cover = trainCoverage(feed.trains);
    if (cover.complete) return null;
    return [
      `Partial read: ${cover.fresh} of ${cover.expected} vehicle feeds ` +
        `reporting (missing ${cover.stale.join(", ")}). `,
      "The routes on a missing feed have no positions in this tick, and the " +
        "published object carries no feed-to-route mapping, so this map cannot " +
        "say which routes those are. Every line is drawn dimmed for that " +
        "reason: a station with no marker may have no train, or may be on a " +
        "line we didn't hear from.",
    ];
  },
  detail(edge, ctx) {
    const feed = ctx.trains;
    const layer = trainLayer(
      ctx.diagram,
      feed.state === "ready" ? feed.trains : null,
    );
    return [edge.a, edge.b].map((id) =>
      trainSummary(layer.byStation[id], ctx.diagram.stations[id]?.name ?? id, feed),
    );
  },
  stamp(ctx) {
    // Its own clock, deliberately not the snapshot's: the object is republished
    // on its own cadence and skipped entirely on a tick whose vehicle feed
    // didn't decode, so borrowing generated_at would overstate its freshness.
    return {
      label: "train positions",
      at: ctx.trains.state === "ready" ? ctx.trains.trains.observed_at : null,
    };
  },
  note(ctx) {
    const feed = ctx.trains;
    if (feed.state === "loading") {
      return [
        "Fetching v1/trains.json — the train surface is a separate object from " +
          "the snapshot, and it is read only while this overlay is selected.",
      ];
    }
    if (feed.state === "unavailable") {
      return [
        `No train surface: ${feed.reason}. It is published as its own object ` +
          "at v1/trains.json — separately from the snapshot, so the rest of " +
          "the map is unaffected — and it may simply not exist yet. Nothing is " +
          "drawn, and that is a ",
        { em: "missing report" },
        " — not a report of no trains running.",
      ];
    }
    const layer = trainLayer(ctx.diagram, feed.trains);
    const cover = trainCoverage(feed.trains);
    const spans: NoteSpan[] = [];
    if (cover.complete && layer.total === 0) {
      // A real observation, and a different claim from an incomplete read:
      // every feed reported and none of them had a train to report.
      spans.push(
        `All ${cover.expected} vehicle feeds reported and none carried a ` +
          "train — an observed empty system, not a missing reading. ",
      );
    } else {
      const sha = feed.trains.provenance.code_sha.slice(0, 7);
      spans.push(
        `${layer.total - layer.unplaced} trains at ${layer.markers.length} ` +
          `stations, as of ${sha}. A filled disc counts trains standing at ` +
          "the platform, the ring around it adds the ones the feed reports " +
          "heading there; both scale by area. ",
      );
    }
    spans.push(
      "Nothing is drawn on the track between stations, because the feed names " +
        "a stop rather than a segment and a moving train's segment is " +
        "ambiguous at a branch.",
    );
    if (layer.unplaced > 0) {
      const one = layer.unplaced === 1;
      spans.push(
        ` ${layer.unplaced} ${one ? "train sits" : "trains sit"} at a stop ` +
          "this diagram has no station for, so " +
          `${one ? "it isn't" : "they aren't"} drawn.`,
      );
    }
    return spans;
  },
};

export const OVERLAYS: readonly Overlay[] = [MOVEMENT, SUPPLY, TIME, TRAINS];

/** Stable per-edge identity: a route crosses a station pair at most once. */
export function edgeId(edge: DiagramEdge): string {
  return `${edge.route}|${edge.a}|${edge.b}`;
}

/** One edge ready to draw: its identity, its geometry, and what the active
 * overlay makes of it. Built once per set of context inputs so hovering
 * doesn't re-derive 988 readings. */
export interface PaintedEdge {
  id: string;
  edge: DiagramEdge;
  d: string;
  paint: EdgePaint;
}

/** Every drawn edge under one overlay, ordered so verdicts paint over the
 * ghosts they cross rather than under them. */
export function paintEdges(overlay: Overlay, ctx: OverlayContext): PaintedEdge[] {
  const out: PaintedEdge[] = [];
  for (const edge of ctx.diagram.edges) {
    const paint = overlay.paint(edge, ctx);
    if (paint === null) continue;
    out.push({ id: edgeId(edge), edge, d: edgePath(edge), paint });
  }
  return out.sort((a, b) => a.paint.order - b.paint.order);
}
