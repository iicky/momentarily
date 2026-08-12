// Movement-vs-alerts analytics — pure functions over the archived vehicle
// metrics, mirroring the debiased movement classifier in training/load_r2.py
// (_cross_tick_counts, compute_advance_baseline, classify_direction,
// derive_movement_state, build_movement_truth), the through-stop topology in
// training/gtfs_static.py (through_stops / terminals), and
// worker/src/movement_state.ts + worker/src/params.ts (TrainedParams.throughStops).
//
// Two artifacts:
//   1. A confusion matrix of the alert-derived HMM condition against the
//      independent movement-derived state — "where do trains-on-the-ground and
//      the alert feed disagree about right now?" The movement state is a
//      per-(route,direction) Beta-Binomial call scored against each cell's own
//      advance baseline, so a line is judged against its own normal rather than
//      one global cutoff.
//   2. Per-route advance-rate baselines plus the share a single global threshold
//      would flag, which make the per-route bias of that threshold visible
//      (shuttles sit low, trunk lines high) — the motivation for the per-
//      direction model.
//
// Independent in DERIVATION from alerts (physical train positions vs. authored
// alert text), not in source (same upstream MTA gateway).
//
// Counting rule: advanced_n/stalled_n only count trips whose FROM stop has a
// scheduled predecessor AND a scheduled successor in the static timetable
// (through_stops) — never a chain endpoint and never a stop the timetable
// skeleton omits entirely. A terminal layover isn't a stall, it's a train
// dwelling out its scheduled turnaround; measured on the real archive, chain
// endpoints and off-skeleton stops "stall" 89%/78% of the time by design and
// together carry 83% of all observed stall mass, so an unfiltered advance
// signal blends two physically different populations. Restricting to through
// stops drops held-out thin-leaf Brier from 0.14857 to 0.08082. The set is
// threaded in from params.json's movement_through_stops (parseThroughStops
// below); null/absent means "count every stop" — pre-through-stops behavior,
// and the fallback for any tick the static topology couldn't be fetched for.

export const TICK_SECONDS = 300;
// Thresholds mirror training/load_r2.py + worker/src/movement_state.ts so the
// offline series and the live signal agree on the movement call.
export const MIN_MATCHED_TRIPS = 3; // advanced_n + stalled_n floor to judge a tick
export const CLASSIFY_PRIOR_STRENGTH = 8; // pseudo-trials regularizing a single tick toward the cell baseline
export const DISRUPTED_RATIO = 0.5; // disrupted when posterior advance rate <= this * baseline p0
export const CLASSIFY_ALPHA = 0.05; // disrupted only when the low advance count is significant (binomial lower tail <= this)
export const ADVANCE_PRIOR_STRENGTH = 50; // Beta pseudo-trials behind the baseline prior
export const P0_FLOOR = 1e-3; // keep p0 off the degenerate Beta endpoints
export const BASELINE_MIN_SAMPLES = 20; // ticks needed to back a cell baseline
// Legacy global-threshold reference: the advance frac a single cutoff would call
// frozen. Retained only for the baseline strip's bias illustration, NOT the
// classifier — the live call is baseline-relative (DISRUPTED_RATIO * p0).
export const FROZEN_ADVANCE_FRAC = 0.25;

export const STATES = ["normal", "disrupted", "suspended"] as const;
export type MovementState = (typeof STATES)[number];
export const DIRECTIONS = ["north", "south"] as const;
export type Direction = (typeof DIRECTIONS)[number];
export const TOD_LABELS = ["overnight", "am_rush", "midday", "pm_rush", "evening"];

export interface DirRow {
  vehicles_n: number;
  advanced_n: number;
  stalled_n: number;
  // Raw cross-tick from_stop>to_stop counts (from==to = a stall in place),
  // archived verbatim from worker/src/vehicles.ts's DirMovementRow.transitions.
  // RAW and unfiltered regardless of movement_through_stops — the one source
  // crossTickCounts recomputes advanced_n/stalled_n from, so a topology change
  // never rewrites archive history, only how it's read.
  transitions?: Record<string, number>;
}
export interface MovementRow {
  vehicles_n: number;
  stopped_n?: number;
  moving_n?: number;
  advanced_n: number;
  stalled_n: number;
  by_direction?: { north?: DirRow; south?: DirRow };
}
export interface VehicleBody {
  observed_at: number;
  fresh_feeds?: string[];
  rows?: Record<string, MovementRow>;
}

// route|direction|stop membership set, the same flat key shape as
// worker/src/params.ts's TrainedParams.throughStops. null means "no
// restriction" — the same fallback StopFilter=None gives _cross_tick_counts
// in load_r2.py, and what every tick got before movement_through_stops existed.
export type ThroughStops = ReadonlySet<string> | null;

/** Flatten params.json's movement_through_stops ({route: {direction:
 * [stop_id, ...]}}, published from training.gtfs_static.through_stops) into
 * the route|direction|stop key set crossTickCounts filters on. Malformed or
 * absent input returns null, never an empty set — an empty set would exclude
 * every stop instead of leaving the signal unfiltered. Mirrors the flattening
 * worker/src/params.ts does for TrainedParams.throughStops. */
export function parseThroughStops(raw: unknown): ThroughStops {
  if (!raw || typeof raw !== "object") return null;
  const out = new Set<string>();
  for (const [route, byDir] of Object.entries(raw as Record<string, unknown>)) {
    if (!byDir || typeof byDir !== "object") continue;
    for (const [dir, stops] of Object.entries(byDir as Record<string, unknown>)) {
      if (!Array.isArray(stops)) continue;
      for (const stop of stops) {
        if (typeof stop === "string" && stop) out.add(`${route}|${dir}|${stop}`);
      }
    }
  }
  return out.size > 0 ? out : null;
}

/** One archived direction row's (advanced, stalled). Unfiltered (throughStops
 * null), straight off the archived counters. Filtered, recomputed from the
 * row's raw `transitions` map — the archived counters are already summed and
 * can't be narrowed after the fact, while `transitions` still carries each
 * trip's from_stop. Only the FROM stop is tested; a trip landing ON an
 * excluded stop still counts as an advance out of wherever it started. Mirrors
 * _cross_tick_counts in training/load_r2.py exactly, including the unfiltered
 * fallback. */
function crossTickCounts(
  d: DirRow,
  route: string,
  dir: Direction,
  throughStops: ThroughStops,
): { advanced: number; stalled: number } {
  if (!throughStops) {
    return { advanced: d.advanced_n ?? 0, stalled: d.stalled_n ?? 0 };
  }
  let advanced = 0;
  let stalled = 0;
  for (const [pair, count] of Object.entries(d.transitions ?? {})) {
    const gt = pair.indexOf(">");
    if (gt < 0) continue;
    const frm = pair.slice(0, gt);
    const to = pair.slice(gt + 1);
    if (!frm || !to) continue;
    const n = count ?? 0;
    if (n <= 0 || !throughStops.has(`${route}|${dir}|${frm}`)) continue;
    if (frm === to) stalled += n;
    else advanced += n;
  }
  return { advanced, stalled };
}

/** One archived row's route-level (advanced, stalled): the by_direction sum
 * of crossTickCounts when filtered, the archived route counters when not.
 * Mirrors the per-row branch in build_movement_series (load_r2.py). */
function routeCrossTickCounts(
  route: string,
  row: MovementRow,
  throughStops: ThroughStops,
): { advanced: number; stalled: number } {
  if (!throughStops) {
    return { advanced: row.advanced_n ?? 0, stalled: row.stalled_n ?? 0 };
  }
  let advanced = 0;
  let stalled = 0;
  for (const dir of DIRECTIONS) {
    const d = row.by_direction?.[dir];
    if (!d) continue;
    const c = crossTickCounts(d, route, dir, throughStops);
    advanced += c.advanced;
    stalled += c.stalled;
  }
  return { advanced, stalled };
}

export function snapTick(epoch: number): number {
  return Math.floor(epoch / TICK_SECONDS) * TICK_SECONDS;
}

const _hourFmt = new Intl.DateTimeFormat("en-US", {
  timeZone: "America/New_York",
  hour: "numeric",
  hour12: false,
});

/** ET-local TOD bin, matching src/momentarily/hmm.py tod_bin / worker hmm.ts. */
export function todBin(epochSec: number): number {
  // hourCycle quirk: "24" can come back for midnight; normalize to 0-23.
  const h = Number(_hourFmt.format(new Date(epochSec * 1000))) % 24;
  if (h < 6) return 0;
  if (h < 10) return 1;
  if (h < 15) return 2;
  if (h < 20) return 3;
  return 4;
}

function median(xs: number[]): number {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

export interface AdvanceBaselineCell {
  p0: number; // baseline (normal) advance rate for the cell
  alpha: number; // Beta prior successes: ADVANCE_PRIOR_STRENGTH * p0
  beta: number; // Beta prior failures: ADVANCE_PRIOR_STRENGTH * (1 - p0)
  n: number; // ticks contributing to the cell
}
// route|direction|tod_bin -> baseline cell.
export type AdvanceBaseline = Map<string, AdvanceBaselineCell>;

function baselineKey(route: string, dir: Direction, tod: number): string {
  return `${route}|${dir}|${tod}`;
}

/** Per-(route,direction,tod) advance-rate baseline as a Beta prior, mirroring
 * compute_advance_baseline in load_r2.py. Train it on a window that precedes the
 * scored window so a sustained outage can't lower its own reference.
 * `throughStops` restricts each tick's advanced_n/stalled_n to trips whose
 * from_stop it admits (see crossTickCounts); null/omitted counts every stop. */
export function computeAdvanceBaseline(
  bodies: VehicleBody[],
  throughStops: ThroughStops = null,
): AdvanceBaseline {
  const buckets = new Map<string, number[]>();
  for (const body of bodies) {
    const tod = todBin(snapTick(body.observed_at ?? 0));
    for (const [route, row] of Object.entries(body.rows ?? {})) {
      const bd = row.by_direction;
      if (!bd) continue;
      for (const dir of DIRECTIONS) {
        const d = bd[dir];
        if (!d) continue;
        const { advanced, stalled } = crossTickCounts(d, route, dir, throughStops);
        const matched = advanced + stalled;
        if (matched < MIN_MATCHED_TRIPS) continue;
        const key = baselineKey(route, dir, tod);
        const arr = buckets.get(key);
        if (arr) arr.push(advanced / matched);
        else buckets.set(key, [advanced / matched]);
      }
    }
  }
  const out: AdvanceBaseline = new Map();
  for (const [key, fracs] of buckets.entries()) {
    if (fracs.length < BASELINE_MIN_SAMPLES) continue;
    const p0 = Math.min(Math.max(median(fracs), P0_FLOOR), 1 - P0_FLOOR);
    out.set(key, {
      p0,
      alpha: ADVANCE_PRIOR_STRENGTH * p0,
      beta: ADVANCE_PRIOR_STRENGTH * (1 - p0),
      n: fracs.length,
    });
  }
  return out;
}

/** P(X <= k) for X ~ Binomial(n, p) via an iterative pmf sum. Exact for the
 * tick-level counts here and free of special functions. Mirrors binomLowerTail
 * in worker/src/movement_state.ts and _binom_lower_tail in load_r2.py. */
function binomLowerTail(k: number, n: number, p: number): number {
  if (k >= n) return 1;
  if (k < 0) return 0;
  const q = 1 - p;
  let pmf = q ** n; // P(X = 0)
  let cdf = pmf;
  for (let i = 0; i < k; i++) {
    pmf *= ((n - i) / (i + 1)) * (p / q);
    cdf += pmf;
  }
  return cdf;
}

/** Beta-Binomial call for one (route,direction) at one tick, three ways:
 * normal (posterior above DISRUPTED_RATIO * p0); disrupted (posterior at/under it
 * AND the low advance count is significant, binomial lower tail <= CLASSIFY_ALPHA);
 * null when it can't be judged — too few matches, no baseline cell, or a
 * point-estimate drop indistinguishable from a low-p0 normal fluctuation. Mirrors
 * classify_direction in load_r2.py. */
export function classifyDirection(
  advancedN: number,
  stalledN: number,
  cell: AdvanceBaselineCell | undefined,
): "normal" | "disrupted" | null {
  const matched = advancedN + stalledN;
  if (matched < MIN_MATCHED_TRIPS) return null;
  if (!cell) return null;
  const post =
    (CLASSIFY_PRIOR_STRENGTH * cell.p0 + advancedN) / (CLASSIFY_PRIOR_STRENGTH + matched);
  if (post > DISRUPTED_RATIO * cell.p0) return "normal";
  return binomLowerTail(advancedN, matched, cell.p0) <= CLASSIFY_ALPHA ? "disrupted" : null;
}

/** Independent current-state for one route at one tick, or null when movement
 * can't support a call. Suspended when no trains are present; otherwise each
 * direction is scored against its own (route,direction,tod) baseline and the
 * route takes the worse. Vehicle-only, mirroring derive_movement_state in
 * load_r2.py. `throughStops` restricts the per-direction counts the same way
 * as computeAdvanceBaseline — pass the same set (or null) used to train
 * `baseline`, or the live call and its prior disagree about which stops count. */
export function deriveMovementState(
  route: string,
  row: MovementRow,
  tick: number,
  baseline: AdvanceBaseline,
  throughStops: ThroughStops = null,
): MovementState | null {
  if ((row.vehicles_n ?? 0) <= 0) return "suspended";
  const tod = todBin(tick);
  const bd = row.by_direction;
  const calls: MovementState[] = [];
  for (const dir of DIRECTIONS) {
    const d = bd?.[dir];
    if (!d) continue;
    const { advanced, stalled } = crossTickCounts(d, route, dir, throughStops);
    const call = classifyDirection(advanced, stalled, baseline.get(baselineKey(route, dir, tod)));
    if (call) calls.push(call);
  }
  if (!calls.length) return null;
  return calls.includes("disrupted") ? "disrupted" : "normal";
}

/** (route|tick) -> independent movement-derived state, judgeable ticks only.
 * `baseline` is the per-cell advance prior; pass one trained on a preceding
 * window (see computeAdvanceBaseline). `throughStops` must be the same set
 * (or null) that trained `baseline` — see deriveMovementState. */
export function buildMovementTruth(
  bodies: VehicleBody[],
  baseline: AdvanceBaseline,
  throughStops: ThroughStops = null,
): Map<string, MovementState> {
  const out = new Map<string, MovementState>();
  for (const body of bodies) {
    const tick = snapTick(body.observed_at ?? 0);
    for (const [route, row] of Object.entries(body.rows ?? {})) {
      const state = deriveMovementState(route, row, tick, baseline, throughStops);
      if (state) out.set(`${route}|${tick}`, state);
    }
  }
  return out;
}

// --- Confusion: alert/HMM condition vs movement-derived state ---

export type DisagreementKind =
  | "false-normal" // published normal, movement says it isn't
  | "false-disrupted" // published disrupted/suspended, trains moving fine
  | "state-mismatch"; // disrupted↔suspended disagreement

export interface ConfusionResult {
  states: string[];
  // matrix[hmmRow][moveCol] counts
  matrix: number[][];
  rowTotals: number[];
  total: number;
  agreement: number; // share on the diagonal
  perRoute: { route: string; n: number; agree: number; agreePct: number }[];
  // Top off-diagonal cells across routes, ranked by count — the actionable list.
  disagreements: {
    route: string;
    hmm: string;
    move: string;
    count: number;
    rate: number; // count / the route's judged ticks
    kind: DisagreementKind;
  }[];
  // Coverage: ticks we could vs couldn't judge against movement.
  coverage: {
    judged: number; // condition ticks matched to a movement read (== total)
    unjudged: number; // condition ticks with no movement truth in window
    suspendedUnjudged: number; // subset: published suspended, no vehicles to judge
  };
}

interface PredLike {
  route: string;
  ts: number;
  condition: string;
}

const STATE_INDEX: Record<string, number> = { normal: 0, disrupted: 1, suspended: 2 };

export function movementConfusion(
  predictions: PredLike[],
  truth: Map<string, MovementState>,
): ConfusionResult {
  const matrix = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const per = new Map<string, { n: number; agree: number; cells: number[][] }>();
  let unjudged = 0;
  let suspendedUnjudged = 0;
  const newCells = () => [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (const p of predictions) {
    const row = STATE_INDEX[p.condition];
    if (row === undefined) continue; // skip unknown / not_scheduled
    const ms = truth.get(`${p.route}|${snapTick(p.ts)}`);
    if (!ms) {
      unjudged += 1;
      if (row === STATE_INDEX.suspended) suspendedUnjudged += 1;
      continue;
    }
    const col = STATE_INDEX[ms];
    matrix[row][col] += 1;
    const r = per.get(p.route) ?? { n: 0, agree: 0, cells: newCells() };
    r.n += 1;
    r.cells[row][col] += 1;
    if (row === col) r.agree += 1;
    per.set(p.route, r);
  }
  const rowTotals = matrix.map((r) => r[0] + r[1] + r[2]);
  const total = rowTotals.reduce((a, b) => a + b, 0);
  const diag = matrix[0][0] + matrix[1][1] + matrix[2][2];
  const perRoute = [...per.entries()]
    .map(([route, v]) => ({
      route,
      n: v.n,
      agree: v.agree,
      agreePct: v.n ? v.agree / v.n : 0,
    }))
    .sort((a, b) => a.agreePct - b.agreePct);

  const kindOf = (row: number, col: number): DisagreementKind =>
    row === STATE_INDEX.normal
      ? "false-normal"
      : col === STATE_INDEX.normal
        ? "false-disrupted"
        : "state-mismatch";
  const disagreements = [...per.entries()]
    .flatMap(([route, v]) =>
      v.cells.flatMap((cols, row) =>
        cols.flatMap((count, col) =>
          row === col || count === 0
            ? []
            : [
                {
                  route,
                  hmm: STATES[row],
                  move: STATES[col],
                  count,
                  rate: v.n ? count / v.n : 0,
                  kind: kindOf(row, col),
                },
              ],
        ),
      ),
    )
    .sort((a, b) => b.count - a.count);

  return {
    states: [...STATES],
    matrix,
    rowTotals,
    total,
    agreement: total ? diag / total : 0,
    perRoute,
    disagreements,
    coverage: { judged: total, unjudged, suspendedUnjudged },
  };
}

// --- Per-route advance-rate baselines ---

export interface DirBaseline {
  p0: number;
  n: number;
}
export interface RouteBaseline {
  route: string;
  p0: number; // median advance frac over all judgeable ticks (both directions)
  n: number; // judgeable ticks
  disruptedShare: number; // legacy global-threshold reference: share of judgeable ticks <= FROZEN_ADVANCE_FRAC
  fracs: number[]; // downsampled advance-fraction distribution for the strip
  north: DirBaseline | null;
  south: DirBaseline | null;
}

const STRIP_CAP = 140; // advance fractions returned per route for the strip plot

function dirP0(fracs: number[]): DirBaseline | null {
  if (fracs.length < BASELINE_MIN_SAMPLES) return null;
  const p0 = Math.min(Math.max(median(fracs), P0_FLOOR), 1 - P0_FLOOR);
  return { p0, n: fracs.length };
}

function downsample(xs: number[], cap: number): number[] {
  if (xs.length <= cap) return xs;
  const stride = Math.ceil(xs.length / cap);
  return xs.filter((_, i) => i % stride === 0);
}

/** Per-route advance-rate baseline + distribution, mirroring
 * compute_advance_baseline but rolled up to the route for the panel.
 * `throughStops` restricts advanced_n/stalled_n the same way as
 * computeAdvanceBaseline; null/omitted counts every stop. */
export function advanceBaselines(
  bodies: VehicleBody[],
  throughStops: ThroughStops = null,
): RouteBaseline[] {
  const all = new Map<string, number[]>();
  const north = new Map<string, number[]>();
  const south = new Map<string, number[]>();
  const push = (m: Map<string, number[]>, route: string, frac: number) => {
    const arr = m.get(route);
    if (arr) arr.push(frac);
    else m.set(route, [frac]);
  };

  for (const body of bodies) {
    for (const [route, row] of Object.entries(body.rows ?? {})) {
      const routeCounts = routeCrossTickCounts(route, row, throughStops);
      const matched = routeCounts.advanced + routeCounts.stalled;
      if (matched >= MIN_MATCHED_TRIPS) push(all, route, routeCounts.advanced / matched);
      const bd = row.by_direction;
      if (!bd) continue;
      for (const [dir, m] of [
        ["north", north],
        ["south", south],
      ] as const) {
        const d = bd[dir];
        if (!d) continue;
        const dirCounts = crossTickCounts(d, route, dir, throughStops);
        const dm = dirCounts.advanced + dirCounts.stalled;
        if (dm >= MIN_MATCHED_TRIPS) push(m, route, dirCounts.advanced / dm);
      }
    }
  }

  const out: RouteBaseline[] = [];
  for (const [route, fracs] of all.entries()) {
    if (fracs.length < BASELINE_MIN_SAMPLES) continue;
    const p0 = Math.min(Math.max(median(fracs), P0_FLOOR), 1 - P0_FLOOR);
    const disrupted = fracs.filter((f) => f <= FROZEN_ADVANCE_FRAC).length;
    out.push({
      route,
      p0,
      n: fracs.length,
      disruptedShare: disrupted / fracs.length,
      fracs: downsample(fracs, STRIP_CAP),
      north: dirP0(north.get(route) ?? []),
      south: dirP0(south.get(route) ?? []),
    });
  }
  out.sort((a, b) => a.p0 - b.p0);
  return out;
}
