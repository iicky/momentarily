/**
 * Segment-level movement -> per-station service flow: "is MY station
 * moving", the question riders actually ask.
 *
 * A single 5-minute tick sees ~1 tracked train per segment — far too few to judge.
 * So we keep a DECAYING SUM of advanced/matched per segment (decayed = this tick +
 * DECAY * previous), an O(1)-per-segment accumulator whose effective window is
 * ~1/(1-DECAY) ticks. The smoothed counts feed the same Beta-Binomial call the
 * direction classifier uses (movement_state.classifyAdvance) against the segment's
 * own trainer baseline, then incident segments roll up to their endpoint stations.
 *
 * Everything here runs at step 8b (post-publish, off the time-to-publish path) and
 * reads/writes its own R2 objects, so the ~1.8k segment baseline never touches the
 * hot per-tick params parse. NOTE: with overlapping decayed samples the binomial
 * tail is a tuned score, not a calibrated p-value — DECAY and CLASSIFY_ALPHA are
 * empirical knobs, and a minimum effective matched count guards the thinnest cells.
 */

import { classifyAdvance } from './movement_state';
import type { RegimeEntry } from './regime';
import type { MovementRow } from './vehicles';
import type { SegmentFlowDoc, SegmentParamsDoc, StationFlowDoc } from './state';

// this tick + DECAY * previous; ~1/(1-DECAY) ≈ 5-tick (~25 min) effective window.
export const SEGMENT_DECAY = 0.8;
// Effective (decayed) matched trips a segment needs before it's judged at all.
export const MIN_EFF_MATCHED = 5;
// Drop a segment from the carried state once its decayed matched falls below this
// (gone quiet) so the state object stays bounded.
const PRUNE_MATCHED = 0.3;

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

/** Advance the decaying per-segment accumulator with this tick's counts. Only
 * segments the trainer baselined are tracked; quiet ones prune out. */
export function updateSegmentFlow(
  prev: SegmentFlowDoc | null,
  moveRows: Map<string, MovementRow>,
  observedAt: number,
  params: SegmentParamsDoc,
): SegmentFlowDoc {
  const counts = tickCounts(moveRows);
  const prevCells = prev?.cells ?? {};
  const cells: Record<string, { a: number; m: number }> = {};
  const keys = new Set<string>([...Object.keys(prevCells), ...counts.keys()]);
  for (const key of keys) {
    if (!(key in params.cells)) continue;
    const p = prevCells[key] ?? { a: 0, m: 0 };
    const t = counts.get(key) ?? { adv: 0, matched: 0 };
    const a = t.adv + SEGMENT_DECAY * p.a;
    const m = t.matched + SEGMENT_DECAY * p.m;
    if (m < PRUNE_MATCHED) continue;
    cells[key] = { a, m };
  }
  // Regimes are advanced by the caller (step 8b, alongside advanceRegimes) and
  // written back onto this doc once computed; carry the previous map through
  // so the type is whole in between.
  return { observed_at: observedAt, cells, regimes: prev?.regimes ?? {} };
}

function deficitOf(p0: number, rate: number): number {
  if (p0 <= 0) return 0;
  return Math.max(0, Math.min(1, (p0 - rate) / p0));
}

interface SegmentCall {
  key: string;
  call: 'normal' | 'disrupted';
  route: string;
  seg: [string, string];
  deficit: number;
}

/** Beta-Binomial call for every segment cell with enough effective matched
 * trips this tick, via the one decision rule shared with the direction
 * classifier (movement_state.classifyAdvance). A cell classifyAdvance
 * abstains on — too few matches, or a point-estimate drop indistinguishable
 * from a low-p0 fluctuation — is simply absent: the shared basis for both
 * deriveStationFlow's incident roll-up and deriveSegmentStates' regime-clock
 * feed, so the two never disagree about which cells were judged. */
function classifySegments(state: SegmentFlowDoc, params: SegmentParamsDoc): SegmentCall[] {
  const out: SegmentCall[] = [];
  for (const [key, { a, m }] of Object.entries(state.cells)) {
    const cell = params.cells[key];
    const adj = params.adjacency[key];
    if (!cell || !adj) continue;
    const matched = Math.round(m);
    if (matched < MIN_EFF_MATCHED) continue;
    const advanced = Math.min(Math.round(a), matched);
    const call = classifyAdvance(advanced, matched - advanced, cell.p0);
    if (call === null) continue;
    const parts = key.split('|');
    const route = parts[0] ?? '';
    const frm = parts[2] ?? '';
    out.push({
      key,
      call,
      route,
      seg: [frm, adj.to],
      deficit: deficitOf(cell.p0, advanced / matched),
    });
  }
  return out;
}

/** Classify each smoothed segment and roll incident segments up to per-station
 * service flow: a station is degraded when any segment touching it reads disrupted. */
export function deriveStationFlow(
  state: SegmentFlowDoc,
  params: SegmentParamsDoc,
): StationFlowDoc {
  interface Incident {
    deficit: number;
    disrupted: boolean;
    route: string;
    seg: [string, string];
  }
  const byStation = new Map<string, Incident[]>();
  for (const c of classifySegments(state, params)) {
    const incident: Incident = {
      deficit: c.deficit,
      disrupted: c.call === 'disrupted',
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
    // contradicts the segment call; worst_deficit rides along as magnitude.
    const degraded = incs.some((c) => c.disrupted);
    stations[sid] = {
      status: degraded ? 'degraded' : 'flowing',
      worst_deficit: worst.deficit,
      worst_segment: worst.seg,
      routes: [...new Set(incs.map((c) => c.route))].sort(),
      n_segments: incs.length,
    };
  }
  return { observed_at: state.observed_at, stations };
}

/** Per-tick classification call for every judged segment cell, keyed the
 * same way as SegmentFlowDoc.cells (`route|direction|from_stop`). A cell
 * classifyAdvance abstains on is absent from the map — same contract as
 * movement_state.deriveMovementStates — so advanceRegimes treats "can't
 * judge this tick" as an abstention that holds the open regime, not a
 * reading of change. */
export function deriveSegmentStates(
  state: SegmentFlowDoc,
  params: SegmentParamsDoc,
): Record<string, 'normal' | 'disrupted'> {
  const out: Record<string, 'normal' | 'disrupted'> = {};
  for (const c of classifySegments(state, params)) out[c.key] = c.call;
  return out;
}

/** Keep only regime entries for cells updateSegmentFlow still tracks this
 * tick. A cell's decayed matched count falling below PRUNE_MATCHED drops it
 * from `cells` outright (gone quiet); without this, advanceRegimes would
 * hold its regime open through the idle-abstention grace (up to
 * MAX_IDLE_SEC) as if it were merely unheard from this tick, when pruning
 * already means gone. Call against the SAME tick's updated cell set, every
 * tick. */
export function pruneSegmentRegimes<C extends string>(
  entries: Record<string, RegimeEntry<C>>,
  liveCells: SegmentFlowDoc['cells'],
): Record<string, RegimeEntry<C>> {
  const out: Record<string, RegimeEntry<C>> = {};
  for (const key of Object.keys(liveCells)) {
    const entry = entries[key];
    if (entry) out[key] = entry;
  }
  return out;
}
