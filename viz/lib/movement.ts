// Movement-vs-alerts analytics — pure functions over the archived vehicle
// metrics, mirroring training/load_r2.py (compute_advance_baseline,
// derive_movement_state, build_movement_truth) and worker/src/movement_state.ts.
//
// Two artifacts:
//   1. A confusion matrix of the alert-derived HMM condition against the
//      independent movement-derived state — "where do trains-on-the-ground and
//      the alert feed disagree about right now?"
//   2. Per-(route,direction,tod) baseline advance rates, which make the
//      per-route bias of a single global threshold visible (shuttles sit near
//      ~0.3 normal, trunk lines near ~0.7) — the motivation for the Bayesian
//      per-direction model (momentarily-vhh).
//
// Independent in DERIVATION from alerts (physical train positions vs. authored
// alert text), not in source (same upstream MTA gateway).

export const TICK_SECONDS = 300;
// Thresholds mirror training/load_r2.py + worker/src/movement_state.ts so the
// offline series and the live signal agree on what "frozen" means.
export const MIN_MATCHED_TRIPS = 3; // advanced_n + stalled_n floor to judge a tick
export const FROZEN_ADVANCE_FRAC = 0.25; // advance frac at/under which a tick reads frozen
export const ADVANCE_PRIOR_STRENGTH = 50; // Beta pseudo-trials behind the baseline
export const P0_FLOOR = 1e-3; // keep p0 off the degenerate Beta endpoints
export const BASELINE_MIN_SAMPLES = 20; // ticks needed to back a cell baseline

export const STATES = ["normal", "disrupted", "suspended"] as const;
export type MovementState = (typeof STATES)[number];
export const DIRECTIONS = ["north", "south"] as const;
export type Direction = (typeof DIRECTIONS)[number];
export const TOD_LABELS = ["overnight", "am_rush", "midday", "pm_rush", "evening"];

export interface DirRow {
  vehicles_n: number;
  advanced_n: number;
  stalled_n: number;
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

/** Independent current-state from one tick's movement row, or null when the row
 * can't support a call (caller treats null as "can't judge"). Vehicle-only, to
 * match build_movement_truth in load_r2.py. */
export function deriveMovementState(row: MovementRow): MovementState | null {
  if ((row.vehicles_n ?? 0) <= 0) return "suspended";
  const matched = (row.advanced_n ?? 0) + (row.stalled_n ?? 0);
  if (matched < MIN_MATCHED_TRIPS) return null;
  return row.advanced_n / matched <= FROZEN_ADVANCE_FRAC ? "disrupted" : "normal";
}

/** (route|tick) -> independent movement-derived state, judgeable ticks only. */
export function buildMovementTruth(bodies: VehicleBody[]): Map<string, MovementState> {
  const out = new Map<string, MovementState>();
  for (const body of bodies) {
    const tick = snapTick(body.observed_at ?? 0);
    for (const [route, row] of Object.entries(body.rows ?? {})) {
      const state = deriveMovementState(row);
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
  disruptedShare: number; // share of judgeable ticks reading <= FROZEN_ADVANCE_FRAC
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
 * compute_advance_baseline but rolled up to the route for the panel. */
export function advanceBaselines(bodies: VehicleBody[]): RouteBaseline[] {
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
      const matched = (row.advanced_n ?? 0) + (row.stalled_n ?? 0);
      if (matched >= MIN_MATCHED_TRIPS) push(all, route, row.advanced_n / matched);
      const bd = row.by_direction;
      if (!bd) continue;
      for (const [dir, m] of [
        ["north", north],
        ["south", south],
      ] as const) {
        const d = bd[dir];
        if (!d) continue;
        const dm = (d.advanced_n ?? 0) + (d.stalled_n ?? 0);
        if (dm >= MIN_MATCHED_TRIPS) push(m, route, d.advanced_n / dm);
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
