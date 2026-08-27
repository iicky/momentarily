/**
 * Segment-level movement -> per-station service flow: "is MY station
 * moving", the question riders actually ask.
 *
 * A single 5-minute tick sees ~1 tracked train per segment — far too few to judge.
 * So we keep a DECAYING SUM of advanced/matched per segment (decayed = this tick +
 * DECAY * previous), an O(1)-per-segment accumulator whose effective window is
 * ~1/(1-DECAY) ticks. Two branches then read that window, and every baselined
 * cell goes through one of them every tick:
 *
 *   advance-rate — MIN_EFF_MATCHED or more matched trips: the same Beta-Binomial
 *     call the direction classifier uses (movement_state.classifyAdvance) against
 *     the cell's own trainer baseline p0. "Of the trains that were here, did they
 *     move on."
 *   throughput — fewer than that, which is most cells most of the time: a Poisson
 *     test of the window's matched count against the cell's expected traversal
 *     rate for this time bin (segment_params.json `lam`). "Did the trains the
 *     timetable promised actually show up." An empty window is evidence here,
 *     not an abstention, which is the whole reason this branch exists — sub-5-min
 *     headways are the only way to clear MIN_EFF_MATCHED, so the advance branch
 *     alone leaves nearly every cell permanently unjudged.
 *
 * The advance branch wins wherever it has an opinion: trains present and moving
 * on is a statement about flow, and flow is what this surface publishes. The
 * throughput branch fills in beneath it, and its 'disrupted' means "the trains
 * are not arriving" — which reaches a rider's station the same way a stall does.
 *
 * Incident segments then roll up to their endpoint stations.
 *
 * Everything here runs at step 8b (post-publish, off the time-to-publish path) and
 * reads/writes its own R2 objects, so the ~1.8k segment baseline never touches the
 * hot per-tick params parse. NOTE: with overlapping decayed samples neither tail
 * is a calibrated p-value — DECAY, CLASSIFY_ALPHA and THROUGHPUT_ALPHA are
 * empirical knobs.
 */

import { schedule_bin } from './hmm';
import { classifyAdvance, poisLowerTail } from './movement_state';
import type { RegimeEntry } from './regime';
import type { MovementRow } from './vehicles';
import type {
  SegmentCondition,
  SegmentFlowDoc,
  SegmentParamsDoc,
  StationFlowDoc,
} from './state';

// this tick + DECAY * previous; ~1/(1-DECAY) ≈ 17-tick (~83 min) effective
// window.
//
// 0.94 rather than the original 0.8 (~25 min), on a graded bakeoff of both
// coverage axes crossed (training/segment_coverage.py). Against the independent
// assigned_n episode label this pairing separates real service collapses from
// healthy service by 5.3x, where the throughput branch at 25 min manages 3.6x
// and a wider window with no throughput fit manages 1.9x and is not significant
// at all. It also alarms LESS on healthy routes (6.4% of testable cells against
// 13.4%): the wider window is smoothing noise, not reaching further.
//
// The obvious objection to a longer window is staleness — a verdict lagging its
// evidence, which `entered_at` and the recovery forecast conditioned on it would
// inherit. That objection is structurally void once the throughput branch runs:
// staleness enters through the regime clock HOLDING a cell's last state while the
// cell abstains, and a cell with a published rate never abstains. Graded on the
// published (debounced) surface rather than the raw calls, the two throughput
// arms score identically to their own call surface, while the two arms without it
// lose 14-16% of their separation to held-open verdicts.
//
// Latency points the same way, for a mechanical reason worth stating: on the
// ABSENCE question a longer window is faster, not slower. The expected count it
// accumulates is ~3.3x larger, so an empty window crosses the Poisson threshold
// sooner. Measured median onset latency is 10 min here against 35 for the
// narrow-window throughput arm — on 5 and 7 detections respectively, so a
// direction rather than a measurement, but a direction that agrees with the
// mechanism and with the separation.
export const SEGMENT_DECAY = 0.94;
// Effective (decayed) matched trips a segment needs before the advance-rate
// branch judges it. Under it the throughput branch takes over — it used to be
// the point where the cell dropped out entirely. 3 rather than 5 so this floor
// stops shadowing classifyAdvance's own MIN_MATCHED_TRIPS, which already rejects
// anything thinner; the sweep behind the bakeoff measured 1, 2 and 3 tying
// exactly at every window width for that reason.
export const MIN_EFF_MATCHED = 3;
// Drop a segment from the carried accumulator once BOTH its decayed matched and
// its decayed expectation fall below this — nothing observed and nothing
// expected, so the entry carries no information and the state object stays
// bounded. Both halves matter: dropping a cell that still expects traffic would
// restart its expectation from zero and bias it toward normal, and dropping one
// that saw traffic would do the same to the observation. Re-entry from zero
// discards at most this much of either tail.
const PRUNE_MATCHED = 0.3;

// Poisson lower-tail threshold for the throughput branch — the role
// CLASSIFY_ALPHA plays on the advance branch, and deliberately the same number,
// so both branches call a cell disrupted at one nominal strictness.
export const THROUGHPUT_ALPHA = 0.05;

// A decayed sum of per-tick Poisson counts is not itself Poisson: with weights
// DECAY^i its mean is sum(DECAY^i * lambda_i) but its variance only
// sum(DECAY^2i * lambda_i), so a raw Poisson tail on it would be badly
// over-strict. Scaling by (1+DECAY) — the standard weighted-Poisson effective
// count — puts mean and variance back on each other, and one Poisson tail then
// covers both sides:
//     k_eff = matched * (1+DECAY),   mu = expected * (1+DECAY)
// where `expected` is the SAME decaying sum run over the per-tick rates rather
// than the per-tick counts. Running it that way instead of scaling the current
// bin's rate by 1/(1-DECAY) is what keeps an hourly or weekday/weekend bin edge
// honest: for the ~5 ticks after 06:00 the window still holds overnight
// traffic, and comparing it against a rush-hour rate would read as a collapse
// on every cell at every bin edge. With a constant rate the two agree exactly.
const EFF_COUNT_SCALE = 1 + SEGMENT_DECAY;

// Expected effective traversals a window needs before absence can be judged at
// all. Below -ln(THROUGHPUT_ALPHA) even a completely empty window sits above the
// tail threshold, so the test provably has no power there. A cell under it reads
// 'quiet' rather than abstaining: too little runs here right now for silence to
// carry information, and saying so IS the opinion that cell warrants.
const QUIET_MAX_EXPECTED = -Math.log(THROUGHPUT_ALPHA);

// THE OUTAGE GUARD, and what it deliberately does not cover.
//
// A route the vehicle feed said nothing about this tick has no transitions for
// any of its cells, so every one of them would read disrupted at once. That is
// what `vehicles` (route -> this tick's tracked vehicles, omitted at zero) is
// for: no entry, no throughput call.
//
// Whole-ROUTE only. A single direction going dark while the route is otherwise
// reported IS evidence and is judged, because that is the case this branch
// exists for. The cost is that a route which genuinely stops running in both
// directions abstains here instead of reading disrupted — and that is the right
// division of labour, not a gap: whole-route service is the route classifier's
// question, and deriveMovementStates already answers it with suspended /
// not_scheduled off the schedule rate. The segment surface exists for the
// sub-route granularity that one cannot see.
//
// Presence, not a threshold: one decoded vehicle is the difference between "the
// feed told us about this route" and "it did not". Residual it cannot cover: a
// feed that still decodes vehicles but stops matching them across ticks would
// depress matched without depressing this.

/** Collapse a directional stop id to its station: strip a trailing N/S. */
export function stationId(stop: string): string {
  const last = stop.at(-1);
  return last === 'N' || last === 'S' ? stop.slice(0, -1) : stop;
}

/** This tick's advanced/matched per (route|dir|from) from the raw transitions. */
function tickCounts(
  moveRows: Map<string, MovementRow>,
): Map<string, { adv: number; matched: number }> {
  const out = new Map<string, { adv: number; matched: number }>();
  for (const [routeId, row] of moveRows) {
    for (const dir of ['north', 'south'] as const) {
      const trans = row.by_direction[dir]?.transitions;
      if (!trans) continue;
      for (const [pair, n] of Object.entries(trans)) {
        const gt = pair.indexOf('>');
        if (gt < 0 || n <= 0) continue;
        const frm = pair.slice(0, gt);
        const to = pair.slice(gt + 1);
        if (!frm || !to) continue;
        const key = `${routeId}|${dir}|${frm}`;
        const acc = out.get(key) ?? { adv: 0, matched: 0 };
        if (frm !== to) acc.adv += n;
        acc.matched += n;
        out.set(key, acc);
      }
    }
  }
  return out;
}

/** This tick's tracked vehicles per route, omitting routes the feed said nothing
 * about — the feed-liveness input to the outage guard, independent of whether
 * any trip was matched across ticks. */
function tickVehicles(moveRows: Map<string, MovementRow>): Record<string, number> {
  const out: Record<string, number> = {};
  for (const [routeId, row] of moveRows) {
    if (row.vehicles_n > 0) out[routeId] = row.vehicles_n;
  }
  return out;
}

/** The published bin whose rates apply at `observedAt`, or null when the trainer
 * published no throughput fit covering it — a params doc predating the fit, or a
 * bin that never cleared its exposure floor. Null means the throughput branch
 * has nothing to test against and abstains. */
function fittedBin(params: SegmentParamsDoc, observedAt: number): string | null {
  const tp = params.throughput;
  if (tp == null) return null;
  const bin = schedule_bin(observedAt);
  return bin in tp.ticks ? bin : null;
}

/** Advance the decaying per-segment accumulator with this tick's counts and this
 * tick's expected rate, and record this tick's per-route vehicle counts for the
 * outage guard.
 *
 * Iteration is over the trainer's baselined cells, not over what moved: a cell
 * that saw no train still has to accrue its expectation, because the gap between
 * the two is what the throughput branch reads. Cells with nothing observed and
 * nothing expected prune back out, which keeps the doc to the cells that are
 * actually in play. */
export function updateSegmentFlow(
  prev: SegmentFlowDoc | null,
  moveRows: Map<string, MovementRow>,
  observedAt: number,
  params: SegmentParamsDoc,
): SegmentFlowDoc {
  const counts = tickCounts(moveRows);
  const prevCells = prev?.cells ?? {};
  const bin = fittedBin(params, observedAt);
  const cells: Record<string, { a: number; m: number; e: number }> = {};
  for (const [key, cell] of Object.entries(params.cells)) {
    const { a: pa = 0, m: pm = 0, e: pe = 0 } = prevCells[key] ?? {};
    const t = counts.get(key) ?? { adv: 0, matched: 0 };
    const a = t.adv + SEGMENT_DECAY * pa;
    const m = t.matched + SEGMENT_DECAY * pm;
    // An unfitted bin contributes nothing to the expectation, which understates
    // it for the following window and biases those ticks toward normal — the
    // safe direction, and unreachable at any sane trainer window width, where
    // every bin clears its exposure floor.
    const e = (bin === null ? 0 : (cell.lam?.[bin] ?? 0)) + SEGMENT_DECAY * pe;
    if (m < PRUNE_MATCHED && e < PRUNE_MATCHED) continue;
    cells[key] = { a, m, e };
  }

  const vehicles = tickVehicles(moveRows);

  // Regimes are advanced by the caller (step 8b, alongside advanceRegimes) and
  // written back onto this doc once computed; carry the previous map through
  // so the type is whole in between.
  return { observed_at: observedAt, cells, vehicles, regimes: prev?.regimes ?? {} };
}

/** Clamped relative shortfall of `observed` against `expected`, in [0, 1] — the
 * magnitude the station roll-up ranks segments by. Both branches feed it: the
 * advance branch in advance-rate units, the throughput branch in effective
 * traversal counts. Zero when there is nothing to fall short of. */
function deficitOf(expected: number, observed: number): number {
  if (expected <= 0) return 0;
  return Math.max(0, Math.min(1, (expected - observed) / expected));
}

interface SegmentCall {
  key: string;
  call: SegmentCondition;
  route: string;
  seg: [string, string];
  deficit: number;
}

/**
 * The throughput call for one cell: is the window's traversal count consistent
 * with what the timetable expected over that same window?
 *
 * Both arguments are decayed sums over the accumulator's window — `matched` of
 * the observed traversals, `expected` of the per-tick rates that applied when
 * each of them could have happened — so a bin edge inside the window is already
 * accounted for. They go onto the effective-Poisson scale together.
 *
 *   quiet     — the window expected less than QUIET_MAX_EXPECTED traversals, so
 *               no observation could reach the tail. Normal for now, by
 *               timetable, and saying so beats abstaining.
 *   null      — the expectation is real but the vehicle feed said nothing about
 *               this route at all, so the silence is unattributable.
 *   disrupted — Poisson lower tail at or under THROUGHPUT_ALPHA: the trains the
 *               timetable promised are not arriving.
 *   normal    — enough of them are.
 *
 * The effective count FLOORS rather than rounds: it is "how many complete
 * traversals the window can account for", and rounding a fractional decayed
 * remnant up to a whole traversal credits evidence that was never observed.
 * Flooring also makes the call monotone in `matched`, which matters at the edge
 * of the quiet band — with rounding, a cell whose expectation is fading out
 * flips normal/disrupted purely on which side of .5 the remnant lands.
 */
export function classifyThroughput(
  matched: number,
  expected: number,
  routeSeen: boolean,
): SegmentCondition | null {
  const mu = expected * EFF_COUNT_SCALE;
  if (mu < QUIET_MAX_EXPECTED) return 'quiet';
  if (!routeSeen) return null;
  const k = Math.floor(matched * EFF_COUNT_SCALE);
  return poisLowerTail(k, mu) <= THROUGHPUT_ALPHA ? 'disrupted' : 'normal';
}

/** A call for every cell the trainer baselined — iteration is over
 * `params.cells`, not the accumulator, because a cell that saw no train has no
 * accumulator entry and that silence is exactly what the throughput branch
 * reads. A cell both branches abstain on is simply absent: the shared basis for
 * both deriveStationFlow's incident roll-up and deriveSegmentStates' regime-clock
 * feed, so the two never disagree about which cells were judged. */
function classifySegments(state: SegmentFlowDoc, params: SegmentParamsDoc): SegmentCall[] {
  const out: SegmentCall[] = [];
  // The accumulated expectation is only meaningful while the fit covers the
  // current bin; outside that the throughput branch has nothing to test.
  const binFitted = fittedBin(params, state.observed_at) !== null;
  for (const [key, cell] of Object.entries(params.cells)) {
    const adj = params.adjacency[key];
    if (!adj) continue;
    const parts = key.split('|');
    const route = parts[0] ?? '';
    const frm = parts[2] ?? '';
    const { a = 0, m = 0, e = 0 } = state.cells[key] ?? {};
    const matched = Math.round(m);
    const advanced = Math.min(Math.round(a), matched);
    let call: SegmentCondition | null = null;
    let deficit = 0;
    if (matched >= MIN_EFF_MATCHED) {
      call = classifyAdvance(advanced, matched - advanced, cell.p0);
      if (call !== null) deficit = deficitOf(cell.p0, advanced / matched);
    }
    if (call === null && binFitted) {
      call = classifyThroughput(m, e, route in state.vehicles);
      if (call === 'disrupted') deficit = deficitOf(e, m);
    }
    if (call === null) continue;
    out.push({ key, call, route, seg: [frm, adj.to], deficit });
  }
  return out;
}

/** Classify each smoothed segment and roll incident segments up to per-station
 * service flow: a station is degraded when any segment touching it reads
 * disrupted, flowing when any reads normal, and quiet when every one of them is
 * quiet — nothing scheduled here right now is neither of the other two. */
export function deriveStationFlow(
  state: SegmentFlowDoc,
  params: SegmentParamsDoc,
): StationFlowDoc {
  interface Incident {
    deficit: number;
    call: SegmentCondition;
    route: string;
    seg: [string, string];
  }
  const byStation = new Map<string, Incident[]>();
  for (const c of classifySegments(state, params)) {
    const incident: Incident = {
      deficit: c.deficit,
      call: c.call,
      route: c.route,
      seg: c.seg,
    };
    for (const sid of new Set([stationId(c.seg[0]), stationId(c.seg[1])])) {
      const arr = byStation.get(sid) ?? [];
      arr.push(incident);
      byStation.set(sid, arr);
    }
  }

  const stations: StationFlowDoc['stations'] = {};
  for (const [sid, incs] of byStation) {
    const worst = incs.reduce((w, c) => (c.deficit > w.deficit ? c : w));
    // Status follows the shared classifier only, so the station surface never
    // contradicts the segment calls; worst_deficit rides along as magnitude.
    const status = incs.some((c) => c.call === 'disrupted')
      ? 'degraded'
      : incs.some((c) => c.call === 'normal')
        ? 'flowing'
        : 'quiet';
    stations[sid] = {
      status,
      worst_deficit: worst.deficit,
      worst_segment: worst.seg,
      routes: [...new Set(incs.map((c) => c.route))].sort(),
      n_segments: incs.length,
    };
  }
  return { observed_at: state.observed_at, stations };
}

/** Per-tick classification call for every judged segment cell, keyed the
 * same way as segment_params.json's cells (`route|direction|from_stop`). A cell
 * both branches abstain on is absent from the map — same contract as
 * movement_state.deriveMovementStates — so advanceRegimes treats "can't
 * judge this tick" as an abstention that holds the open regime, not a
 * reading of change. */
export function deriveSegmentStates(
  state: SegmentFlowDoc,
  params: SegmentParamsDoc,
): Record<string, SegmentCondition> {
  const out: Record<string, SegmentCondition> = {};
  for (const c of classifySegments(state, params)) out[c.key] = c.call;
  return out;
}

/** Keep only regime entries for cells the trainer still baselines. A retrained
 * segment_params.json that drops a cell (its from_stop left the timetable's
 * through-stop set) must not leave that cell's regime alive in the carried doc,
 * where nothing will ever judge or expire it again. Cells that merely abstain
 * this tick are NOT pruned here — advanceRegimes' idle grace (up to
 * MAX_IDLE_SEC) is the right expiry for those. Call against the same params
 * doc classifySegments read, every tick. */
export function pruneSegmentRegimes<C extends string>(
  entries: Record<string, RegimeEntry<C>>,
  baselined: SegmentParamsDoc['cells'],
): Record<string, RegimeEntry<C>> {
  const out: Record<string, RegimeEntry<C>> = {};
  for (const key of Object.keys(baselined)) {
    const entry = entries[key];
    if (entry) out[key] = entry;
  }
  return out;
}
