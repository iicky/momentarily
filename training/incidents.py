"""Incident-level duration for clustered contiguous disrupted segments.

NOT wired into anything published. The premise this module was built on —
that adjacent disrupted segments overwhelmingly share ONE cause, so a
path's duration should compose per INCIDENT rather than per segment — did
not survive measurement. Two runs against the real archive, at increasing
statistical power, agree on the verdict:

  - 8 days (2026-08-04..08-11, journal.md 2026-08-11 "clustered incidents"
    and its "revises the entry above" follow-up): 0/1260 candidate ticks
    produced a multi-segment cluster, on a baseline self-trained on that
    same 8-day window. A per-day robustness check found the "zero" was
    sensitive to how little data trained each day's baseline (4/8 days
    reproduced it, 4/8 didn't, 3.8-14.7% adjacent share) — inconclusive by
    itself.
  - 53 days (2026-06-21..08-12, journal.md 2026-08-12), scored against
    BOTH a baseline self-trained on the full 53 days AND the live
    published `state/segment_params.json` baseline: the two agree almost
    exactly. 5257-5489 candidate ticks (>=2 disrupted segments somewhere
    on the network) produced only 21 multi-segment-cluster ticks each —
    max_gap=0 and max_gap=1 give the IDENTICAL 21, so loosening the gap
    adds nothing — and only 0.27-0.29% of disrupted-segment observations
    belong to a multi-segment incident. Tracing identity through the whole
    window (`advance_incidents`) resolves that into exactly 3 DISTINCT
    real multi-segment incidents network-wide in 53 days, agreed on by
    both baselines down to the segment keys and timestamps: two
    occurrences of the same 7|south 701S/702S pair (2026-07-16,
    2026-07-27) and one 3-segment A|north cluster (2026-07-24). Real, but
    rare — nowhere close to "overwhelming."

The longer window resolved the earlier ambiguity — self-trained and
published baselines converge, so the answer isn't a baseline-choice
artifact — but it didn't reverse the direction: adjacent disrupted
segments are the exception, not the rule. `path_incident_durations`
remains unwired for that reason; per-segment duration composition is the
right default until the data looks different.

Three pieces, in the order a caller would use them if that changed:

1. `cluster_disrupted` — pure, one tick: group this tick's disrupted segments
   into incidents via the static successor graph, scoped within one
   (route, direction). Hermetic, no I/O.
2. `advance_incidents` / `replay_incidents` — track an incident's identity
   across ticks and debounce its open/close through `training.regime`, the
   SAME clock every other cell in this codebase uses. No separate hysteresis.
3. `fit_incident_duration` — reuse `training.dwell` (empirical Kaplan-Meier)
   and `training.pooled_dwell` (partially-pooled AFT fallback) to fit
   duration at the incident level, not new survival math.

`path_incident_durations` is the composition function a path query would
need: one duration distribution per DISTINCT incident touching the path,
never one per disrupted segment. It is exercised by the test suite and
correct — nothing calls it, per the measurement above.

Everything above is pure and hermetically testable. The R2-touching pieces —
fetching archives, reconstructing a real disrupted-segment tick history via
`training.movement_backfill` (this epic's canonical offline segment
reconstruction), and reporting the measured clustering — live at the bottom
behind `main()` and are never exercised by the test suite.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from types import MappingProxyType
from typing import Any, cast

from training.dwell import (
    MIN_SAMPLES_FOR_EMPIRICAL,
    DwellQuantiles,
    OpenRegimes,
    compute_dwell_quantiles,
)
from training.eval import TransitionRecord
from training.gtfs_static import load_successors
from training.hierarchical import PooledCell
from training.load import TICK_SECONDS
from training.load_r2 import (
    ADVANCE_PRIOR_STRENGTH,
    build_segment_series,
    fetch_vehicle_metrics,
)
from training.movement_backfill import (
    SEGMENT_WINDOW_TICKS,
    segment_ticks_from_vehicle_bodies,
)
from training.pooled_dwell import pooled_dwell_cells
from training.r2_client import get_object_bytes, load_config, make_client
from training.regime import (
    DEBOUNCE_TICKS,
    MAX_IDLE_SEC,
    RegimeChange,
    RegimeEntry,
    advance_regimes,
)
from training.segments import classify_segment

# route|direction|from_stop — the same segment cell key regime.py and
# segment_flow.ts already use; a segment's key IS a node in the topology
# graph below, its to_stop is the next segment's from_stop.
SegmentKey = str

# state/segment_params.json's `adjacency` dict, keyed on SegmentKey, one entry
# per from_stop: {"to": str, "successors": [{"to": str, "n_trips": int}, ...]}
# (successors present when topology_source == "gtfs_static") or just
# {"to": str, "share": float, "n": int} (the "observed" fallback shape).
AdjacencyDoc = Mapping[str, Mapping[str, Any]]

Topology = Mapping[SegmentKey, list[SegmentKey]]

# Two disrupted segments separated by exactly this many non-disrupted
# segments still cluster into one incident. Default 0: only segments that
# DIRECTLY touch merge. A segment that itself reads normal has, by
# definition, let a train through it — that is evidence AGAINST a blockage
# there, not silence, so bridging across it risks lumping two genuinely
# separate incidents (one at each end of an unaffected stretch) into one.
# `measure_premise`'s gap-1 sensitivity check, run against the real 53-day
# archive (journal.md 2026-08-12), found gap=1 widens NOTHING over this
# default: self-trained and published baselines both produced the IDENTICAL
# 21 multi-segment-cluster ticks at gap=0 and gap=1. Loosening the default
# has no support in the data measured so far.
DEFAULT_MAX_GAP = 0

_ACTIVE = "active"
_ENDED = "ended"


def route_of(key: SegmentKey) -> str:
    return key.split("|", 1)[0]


# --- Topology ---------------------------------------------------------


def parse_topology(adjacency: AdjacencyDoc) -> dict[SegmentKey, list[SegmentKey]]:
    """`state/segment_params.json`'s `adjacency` dict -> a directed successor
    graph over segment keys. A segment's to_stop IS the next segment's
    from_stop, so `key`'s neighbors are `route|direction|<to>` for every `to`
    in its full static successor list (topology_source == "gtfs_static"), or
    just the single dominant `to` when only the observed fallback shape is
    present (no `successors` field)."""
    graph: dict[SegmentKey, list[SegmentKey]] = {}
    for key, entry in adjacency.items():
        route, direction, _from = key.split("|", 2)
        successors = entry.get("successors")
        to_stops: list[str] = (
            [s["to"] for s in successors] if successors else [entry["to"]]
        )
        graph[key] = [f"{route}|{direction}|{to}" for to in to_stops]
    return graph


def topology_from_successors(
    succ: Mapping[tuple[str, str, str], list[tuple[str, int]]],
) -> dict[SegmentKey, list[SegmentKey]]:
    """`gtfs_static.load_successors`' raw shape -> this module's segment-key
    graph, for a caller that already has the static feed fetched (this
    module's own `main`) rather than the persisted adjacency doc."""
    graph: dict[SegmentKey, list[SegmentKey]] = {}
    for (route, direction, frm), succs in succ.items():
        if not succs:
            continue
        graph[f"{route}|{direction}|{frm}"] = [
            f"{route}|{direction}|{to}" for to, _n_trips in succs
        ]
    return graph


def _undirected(topology: Topology) -> dict[SegmentKey, set[SegmentKey]]:
    edges: dict[SegmentKey, set[SegmentKey]] = defaultdict(set)
    for key, succs in topology.items():
        for to_key in succs:
            edges[key].add(to_key)
            edges[to_key].add(key)
    return edges


# --- Clustering ---------------------------------------------------------


def _bridge(
    node: SegmentKey,
    edges: Mapping[SegmentKey, set[SegmentKey]],
    disrupted: set[SegmentKey],
    max_gap: int,
) -> set[SegmentKey]:
    """Disrupted segments reachable from `node` through at most `max_gap`
    non-disrupted segments. BFS with a best-budget-seen cache per healthy
    node so a cyclic topology (a shuttle loop) still terminates."""
    out: set[SegmentKey] = set()
    best: dict[SegmentKey, int] = {}
    frontier: deque[tuple[SegmentKey, int]] = deque([(node, max_gap)])
    while frontier:
        cur, budget = frontier.popleft()
        for nxt in edges.get(cur, ()):
            if nxt in disrupted:
                if nxt != node:
                    out.add(nxt)
                continue
            if budget <= 0:
                continue
            remaining = budget - 1
            if best.get(nxt, -1) >= remaining:
                continue
            best[nxt] = remaining
            frontier.append((nxt, remaining))
    return out


def cluster_disrupted(
    disrupted: Iterable[SegmentKey],
    topology: Topology,
    *,
    max_gap: int = DEFAULT_MAX_GAP,
) -> list[frozenset[SegmentKey]]:
    """Group one tick's disrupted segments into incidents: connected
    components of the static successor graph, where up to `max_gap`
    consecutive non-disrupted segments may bridge two disrupted ones without
    themselves joining the incident. Two segments on different routes never
    merge — the topology graph has no edges between routes, so they can't be
    reachable from each other regardless of `max_gap`. A segment absent from
    `topology` (no known static adjacency) has no neighbors and clusters
    alone. Pure — no I/O, safe to call every tick."""
    disrupted_set = set(disrupted)
    if not disrupted_set:
        return []
    edges = _undirected(topology)
    visited: set[SegmentKey] = set()
    clusters: list[frozenset[SegmentKey]] = []
    for start in sorted(disrupted_set):
        if start in visited:
            continue
        cluster = {start}
        visited.add(start)
        stack = [start]
        while stack:
            node = stack.pop()
            for reached in _bridge(node, edges, disrupted_set, max_gap):
                if reached not in cluster:
                    cluster.add(reached)
                    visited.add(reached)
                    stack.append(reached)
        clusters.append(frozenset(cluster))
    return clusters


# --- Tracking incident identity across ticks -----------------------------


def _empty_regimes() -> dict[str, RegimeEntry]:
    return {}


def _empty_footprints() -> dict[str, frozenset[SegmentKey]]:
    return {}


@dataclass(frozen=True)
class IncidentState:
    """Threaded across ticks by `replay_incidents`, mirroring the
    (entries, changes) fold `training.regime.replay_regimes` itself uses.
    `regimes` is the debounced active/ended clock, keyed on incident id,
    running through `advance_regimes` unmodified. `footprints` is the side
    channel this module needs on top of it: a bare regime state doesn't
    carry which segments currently belong to an incident, and the next
    tick's identity match needs that.
    """

    regimes: Mapping[str, RegimeEntry] = field(default_factory=_empty_regimes)
    footprints: Mapping[str, frozenset[SegmentKey]] = field(
        default_factory=_empty_footprints
    )
    next_seq: int = 0


def _resolve_ids(
    state: IncidentState, clusters: list[frozenset[SegmentKey]]
) -> tuple[dict[str, frozenset[SegmentKey]], set[str], int]:
    """Match this tick's clusters to tracked incident ids by footprint
    overlap. Identity rule: an id keeps whichever cluster shares the MOST
    segments with its last known footprint (ties go to the older incident,
    then the smaller id, then the cluster whose smallest segment key sorts
    first — deterministic across ties without depending on iteration order).
    One rule resolves both directions of ambiguity: a MERGE (two ids both
    overlap one bigger cluster) leaves the id with the smaller overlap
    unmatched. A SPLIT (one id's footprint now overlaps two disjoint
    clusters) leaves its smaller piece unmatched. An unmatched id is
    returned separately so the caller can start debouncing its close; an
    unmatched cluster is a genuinely new incident and gets a fresh id. A
    footprint that merely grows or shrinks without merging or splitting is
    the single-candidate case and keeps its id automatically.
    """
    alive = {
        iid: fp
        for iid, fp in state.footprints.items()
        if state.regimes.get(iid) is not None and state.regimes[iid].state == _ACTIVE
    }
    candidates: list[tuple[int, int, str, str, int]] = [
        (
            len(alive[iid] & cluster),
            state.regimes[iid].entered_at,
            iid,
            min(cluster),
            ci,
        )
        for ci, cluster in enumerate(clusters)
        for iid in alive
        if alive[iid] & cluster
    ]
    candidates.sort(key=lambda c: (-c[0], c[1], c[2], c[3]))

    claimed_ids: set[str] = set()
    claimed_clusters: set[int] = set()
    resolved: dict[str, frozenset[SegmentKey]] = {}
    for _overlap, _entered_at, iid, _min_seg, ci in candidates:
        if iid in claimed_ids or ci in claimed_clusters:
            continue
        resolved[iid] = clusters[ci]
        claimed_ids.add(iid)
        claimed_clusters.add(ci)

    next_seq = state.next_seq
    for ci, cluster in enumerate(clusters):
        if ci in claimed_clusters:
            continue
        route, direction, _from = next(iter(cluster)).split("|", 2)
        resolved[f"{route}|{direction}#{next_seq}"] = cluster
        next_seq += 1

    ended_ids = set(alive) - claimed_ids
    return resolved, ended_ids, next_seq


def advance_incidents(
    prev: IncidentState | None,
    disrupted: Iterable[SegmentKey],
    topology: Topology,
    observed_at: int,
    *,
    max_gap: int = DEFAULT_MAX_GAP,
    debounce_ticks: int = DEBOUNCE_TICKS,
    max_idle_sec: int = MAX_IDLE_SEC,
) -> tuple[IncidentState, list[RegimeChange]]:
    """Advance every tracked incident by one tick.

    Cluster this tick's disrupted segments, resolve identity against last
    tick's footprints, then run the result through `advance_regimes` exactly
    like a route or segment cell: 'active' while a matching cluster exists,
    'ended' once one stops matching, committed after `debounce_ticks` like
    everything else — a growing or shrinking footprint that never loses
    overlap keeps calling 'active' and never touches the debounce at all.

    Silence (an id absent from `observed`) is reserved for ids the clock has
    already closed and this module is done with. A footprint that merely
    stops matching this tick is an explicit 'ended' call so the close can
    debounce at the SAME speed as an open, rather than waiting on
    `advance_regimes`'s much slower idle eviction — that eviction exists for
    routes/segments where losing track for an hour is a real "we don't know
    any more", not for a candidate close that should resolve in minutes.
    """
    state = prev or IncidentState()
    clusters = cluster_disrupted(disrupted, topology, max_gap=max_gap)
    resolved, ended_ids, next_seq = _resolve_ids(state, clusters)

    observed: dict[str, str] = dict.fromkeys(resolved, _ACTIVE)
    observed.update(dict.fromkeys(ended_ids, _ENDED))

    new_regimes, changes = advance_regimes(
        state.regimes,
        observed,
        observed_at,
        debounce_ticks=debounce_ticks,
        max_idle_sec=max_idle_sec,
    )

    committed = {c.key for c in changes}
    footprints = dict(state.footprints)
    footprints.update(resolved)
    for iid in list(footprints):
        if iid not in new_regimes or iid in committed:
            del footprints[iid]

    return (
        IncidentState(regimes=new_regimes, footprints=footprints, next_seq=next_seq),
        changes,
    )


def replay_incidents(
    ticks: list[tuple[int, Iterable[SegmentKey]]],
    topology: Topology,
    *,
    max_gap: int = DEFAULT_MAX_GAP,
    debounce_ticks: int = DEBOUNCE_TICKS,
    max_idle_sec: int = MAX_IDLE_SEC,
) -> tuple[IncidentState, list[RegimeChange]]:
    """Fold `advance_incidents` over a whole (observed_at, disrupted_segments)
    history — the offline mirror of `regime.replay_regimes`, for the same
    reason: a backfill goes through the tick-by-tick clock, it doesn't
    reimplement it. Every tick in the window belongs here, including ticks
    with zero disrupted segments — those are what let an open incident's
    close actually debounce."""
    state = IncidentState()
    changes: list[RegimeChange] = []
    for observed_at, disrupted in sorted(ticks, key=lambda t: t[0]):
        state, tick_changes = advance_incidents(
            state,
            disrupted,
            topology,
            observed_at,
            max_gap=max_gap,
            debounce_ticks=debounce_ticks,
            max_idle_sec=max_idle_sec,
        )
        changes.extend(tick_changes)
    return state, changes


def open_incident_regimes(state: IncidentState) -> OpenRegimes:
    """Routes with an incident regime still open at the end of a replay, for
    right-censoring — one entry per route (`OpenRegimes`'s shape, matching
    `dwell.py`'s own open-regime censoring). Two incidents open on the same
    route at once collapse to the older (longer-running) one: duration
    fitting only needs SOME censored sample per route, not every concurrent
    one, and the older incident is the more informative censored draw."""
    best: dict[str, tuple[str, int]] = {}
    for iid, entry in state.regimes.items():
        if entry.state != _ACTIVE:
            continue
        route = route_of(iid)
        current = best.get(route)
        if current is None or entry.entered_at < current[1]:
            best[route] = (_ACTIVE, entry.entered_at)
    return best


# --- Fitting duration at the incident level -------------------------------


def _incident_transitions(changes: Sequence[RegimeChange]) -> list[TransitionRecord]:
    """Completed incident episodes as `TransitionRecord`, so `dwell.py`'s
    grouping/censoring/curve machinery applies unmodified. `route` is the
    incident's line; `prev_state` is the constant `_ACTIVE`, so the fitted
    cell lands in the same `{route: {state: DwellQuantiles}}` shape every
    other dwell cell uses, just with one trivial state."""
    return [
        TransitionRecord(
            ts=c.exited_at,
            route=route_of(c.key),
            prev_state=_ACTIVE,
            new_state=_ENDED,
            regime_entered_at=c.entered_at,
            exited_at=c.exited_at,
            dwell_sec=c.dwell_sec,
        )
        for c in changes
        if c.new_state == _ENDED
    ]


def fit_incident_duration(
    changes: Sequence[RegimeChange],
    *,
    window_end: int | None = None,
    still_open: OpenRegimes = MappingProxyType({}),
    min_samples: int = MIN_SAMPLES_FOR_EMPIRICAL,
) -> dict[str, DwellQuantiles]:
    """Per-route incident-duration cell: `dwell.py`'s empirical Kaplan-Meier
    curve where a route has >= `min_samples` completed incidents, else
    `pooled_dwell.py`'s partially-pooled AFT fit — the same empirical-first,
    pooled-fallback split those two modules already use for the normal-regime
    cell, because most routes will land in the sparse case (incidents are
    rarer than ordinary regime transitions by construction: they only exist
    where segments were ALREADY disrupted and adjacent).

    `still_open` (from `open_incident_regimes`) right-censors incidents still
    running at `window_end` — the same Kaplan-Meier treatment any other open
    regime gets, so a marathon incident in progress pushes the tail up
    instead of being invisible until it clears, and is never counted as a
    completed observation.
    """
    transitions = _incident_transitions(changes)
    pooled = pooled_dwell_cells(
        transitions, state=_ACTIVE, window_end=window_end, open_regimes=still_open
    )
    empirical = compute_dwell_quantiles(
        transitions,
        min_samples=min_samples,
        window_end=window_end,
        open_regimes=still_open,
    )
    out = dict(pooled)
    out.update(
        {
            route: cells[_ACTIVE]
            for route, cells in empirical.items()
            if _ACTIVE in cells
        }
    )
    return out


def path_incident_durations(
    path_segments: Iterable[SegmentKey],
    disrupted: Iterable[SegmentKey],
    topology: Topology,
    route_quantiles: Mapping[str, DwellQuantiles],
    *,
    max_gap: int = DEFAULT_MAX_GAP,
) -> list[DwellQuantiles]:
    """The duration distribution(s) relevant to one path query: one entry per
    DISTINCT incident currently touching the path, never one per disrupted
    segment.

    Adjacent disrupted segments on the path that cluster into the same
    incident collapse to a SINGLE distribution here — summing or multiplying
    their per-segment dwells would price one delay as though it were several
    independent draws, which is exactly the composition error this module
    exists to avoid: adjacent segments are strongly correlated (one cause),
    not independent. A path with two genuinely separate incidents (a healthy
    gap wider than `max_gap` between them) correctly returns two
    distributions; how a routing layer combines multiple concurrent
    incidents into one path-level estimate is that layer's decision, not
    this function's.
    """
    path_set = set(path_segments)
    disrupted_set = set(disrupted)
    if not (disrupted_set & path_set):
        return []
    clusters = cluster_disrupted(disrupted_set, topology, max_gap=max_gap)
    out: list[DwellQuantiles] = []
    for cluster in clusters:
        touching = cluster & path_set
        if not touching:
            continue
        quantiles = route_quantiles.get(route_of(next(iter(touching))))
        if quantiles is not None:
            out.append(quantiles)
    return out


# --- Measuring the premise against real archived data ---------------------
#
# `main` fetches from R2 and the static GTFS feed and is never exercised by
# the test suite. The pure helpers around it — `_disrupted_ticks_from_calls`,
# `published_baseline_cells`, `segment_ticks_with_baseline`, `measure_premise`,
# `_week_windows`, and `premise_report` — take/return plain data and ARE unit
# tested; only the fetch itself needs a live R2 credential.


def _disrupted_ticks_from_calls(
    ticks: list[tuple[int, Mapping[str, str]]],
) -> list[tuple[int, list[SegmentKey]]]:
    """`movement_backfill.segment_ticks_from_vehicle_bodies`'s raw per-tick
    classifier calls, filtered down to the keys that called 'disrupted' —
    this module's clustering input shape. Reused rather than reimplemented:
    that module already owns the canonical offline segment reconstruction
    (the same 6-tick/30-min trailing pre-accumulation `segment_flow.ts` uses
    online), and `DEBOUNCE_TICKS` now defaults to 1 (training/regime.py) —
    a segment's regime state IS its raw call with no further smoothing on
    top, so there is nothing left for this module to reproduce."""
    return [
        (tick, [key for key, call in calls.items() if call == "disrupted"])
        for tick, calls in ticks
    ]


# state/segment_params.json's own key, so `main` can fetch the published
# baseline the Worker classifies against, not just the self-trained one
# `segment_ticks_from_vehicle_bodies` builds from the measured window itself.
SEGMENT_PARAMS_KEY = "state/segment_params.json"


def published_baseline_cells(
    doc: Mapping[str, Any],
) -> dict[tuple[str, str, str], PooledCell]:
    """`state/segment_params.json`'s `cells` dict -> the same
    `{(route, direction, from_stop): PooledCell}` shape `load_r2.
    build_segment_baseline` returns, so `segment_ticks_with_baseline` can
    score against either interchangeably.

    `write_segment_params` (train_em.py) only publishes `p0`/`n` per cell —
    the trainer keeps the leaf's raw rate, pooling `source`, and fitted
    alpha/beta to itself. `classify_segment` (via `classify_direction`)
    reads only `.p0` off the cell it's given, so reconstructing the rest as
    a p0-anchored Beta prior at `ADVANCE_PRIOR_STRENGTH` pseudo-trials — the
    same convention `compute_advance_baseline` anchors its own prior with —
    is exact for classification, even though it can't recover the leaf's
    actual fitted concentration.
    """
    out: dict[tuple[str, str, str], PooledCell] = {}
    for raw_key, cell in doc.get("cells", {}).items():
        parts = raw_key.split("|")
        if len(parts) != 3:
            continue
        p0 = float(cell["p0"])
        out[(parts[0], parts[1], parts[2])] = PooledCell(
            p0=p0,
            raw=p0,
            n=int(cell.get("n", 0)),
            alpha=p0 * ADVANCE_PRIOR_STRENGTH,
            beta=(1.0 - p0) * ADVANCE_PRIOR_STRENGTH,
            source="published",
        )
    return out


def segment_ticks_with_baseline(
    bodies: list[dict[str, Any]],
    baseline: Mapping[tuple[str, str, str], PooledCell],
    *,
    window_ticks: int = SEGMENT_WINDOW_TICKS,
) -> list[tuple[int, Mapping[str, str]]]:
    """The same per-tick segment classification `movement_backfill.
    segment_ticks_from_vehicle_bodies` does, scored against a `baseline`
    the caller supplies instead of one self-trained on `bodies` — lets
    `main` classify the vehicle archive against the published
    `state/segment_params.json` baseline (`published_baseline_cells`)
    instead of the window it's measuring, which is exactly the
    self-training confound that left the first 8-day run inconclusive
    (journal.md 2026-08-11, "revises the entry above").

    Duplicates that function's trailing-window accumulation rather than
    parameterizing it — movement_backfill.py is a sibling module this task
    doesn't touch, and its self-trained default is still the right choice
    for its own callers.
    """
    series = build_segment_series(bodies)

    per_leaf: dict[tuple[str, str, str], dict[int, tuple[int, int]]] = defaultdict(dict)
    for (route, direction, frm, to, tick), n in series.items():
        leaf = (route, direction, frm)
        adv, stall = per_leaf[leaf].get(tick, (0, 0))
        if frm == to:
            stall += n
        else:
            adv += n
        per_leaf[leaf][tick] = (adv, stall)

    window_sec = window_ticks * TICK_SECONDS
    by_tick: dict[int, dict[str, str]] = {}
    for leaf, tick_counts in per_leaf.items():
        cell = baseline.get(leaf)
        if cell is None:
            continue
        route, direction, frm = leaf
        key = f"{route}|{direction}|{frm}"
        window: deque[tuple[int, int, int]] = deque()
        adv_sum = stall_sum = 0
        for tick in sorted(tick_counts):
            a, s = tick_counts[tick]
            window.append((tick, a, s))
            adv_sum += a
            stall_sum += s
            while window and tick - window[0][0] >= window_sec:
                _, old_adv, old_stall = window.popleft()
                adv_sum -= old_adv
                stall_sum -= old_stall
            call = classify_segment(adv_sum, stall_sum, cell)
            if call is not None:
                by_tick.setdefault(tick, {})[key] = call
    return cast("list[tuple[int, Mapping[str, str]]]", sorted(by_tick.items()))


def measure_premise(
    ticks: list[tuple[int, list[SegmentKey]]],
    topology: Topology,
) -> dict[str, Any]:
    """Pure: the adjacency-vs-scattered stats the clustering premise needs, from
    an already-reconstructed tick history. Hermetic — no R2 involved, safe to
    unit test, kept separate from the R2/archive-fetching code in `main` for
    exactly that reason. Reports max_gap=0 (the shipped default) and
    max_gap=1 (the sensitivity check DEFAULT_MAX_GAP's docstring cites) side
    by side, so a caller never has to rerun this to check the other gap."""
    considered_ticks = 0
    ticks_with_adjacent_gap0 = 0
    ticks_with_adjacent_gap1 = 0
    sizes_gap0: list[int] = []
    sizes_gap1: list[int] = []
    disrupted_total = 0
    disrupted_in_multi_gap0 = 0
    disrupted_in_multi_gap1 = 0

    for _observed_at, disrupted in ticks:
        if len(disrupted) < 2:
            continue
        considered_ticks += 1
        clusters0 = cluster_disrupted(disrupted, topology, max_gap=0)
        clusters1 = cluster_disrupted(disrupted, topology, max_gap=1)
        sizes_gap0.extend(len(c) for c in clusters0)
        sizes_gap1.extend(len(c) for c in clusters1)
        if any(len(c) > 1 for c in clusters0):
            ticks_with_adjacent_gap0 += 1
        if any(len(c) > 1 for c in clusters1):
            ticks_with_adjacent_gap1 += 1
        disrupted_total += len(disrupted)
        disrupted_in_multi_gap0 += sum(len(c) for c in clusters0 if len(c) > 1)
        disrupted_in_multi_gap1 += sum(len(c) for c in clusters1 if len(c) > 1)

    return {
        "ticks_with_2plus_disrupted": considered_ticks,
        "ticks_with_adjacent_cluster_gap0": ticks_with_adjacent_gap0,
        "ticks_with_adjacent_cluster_gap1": ticks_with_adjacent_gap1,
        "share_ticks_adjacent_gap0": (
            ticks_with_adjacent_gap0 / considered_ticks if considered_ticks else None
        ),
        "share_ticks_adjacent_gap1": (
            ticks_with_adjacent_gap1 / considered_ticks if considered_ticks else None
        ),
        "mean_cluster_size_gap0": (statistics.mean(sizes_gap0) if sizes_gap0 else None),
        "mean_cluster_size_gap1": (statistics.mean(sizes_gap1) if sizes_gap1 else None),
        "median_cluster_size_gap0": (
            statistics.median(sizes_gap0) if sizes_gap0 else None
        ),
        "median_cluster_size_gap1": (
            statistics.median(sizes_gap1) if sizes_gap1 else None
        ),
        "share_disrupted_segments_in_multi_segment_incident_gap0": (
            disrupted_in_multi_gap0 / disrupted_total if disrupted_total else None
        ),
        "share_disrupted_segments_in_multi_segment_incident_gap1": (
            disrupted_in_multi_gap1 / disrupted_total if disrupted_total else None
        ),
    }


def _week_windows(start: date, end: date, *, days: int = 7) -> list[tuple[date, date]]:
    """[start, end] split into `days`-day bins for a stability check — the
    last bin is whatever's left over, never padded to a full window."""
    windows: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        bin_end = min(cursor + timedelta(days=days - 1), end)
        windows.append((cursor, bin_end))
        cursor = bin_end + timedelta(days=1)
    return windows


def premise_report(
    ticks: list[tuple[int, list[SegmentKey]]],
    topology: Topology,
    start: date,
    end: date,
) -> dict[str, Any]:
    """`measure_premise` pooled over [start, end] plus a per-week breakdown,
    so a multi-week pooled number can be checked against week-to-week
    stability instead of trusted on its own — the same worry that left the
    first 8-day run (journal.md 2026-08-11) inconclusive."""
    weekly: list[dict[str, Any]] = []
    for week_start, week_end in _week_windows(start, end):
        lo = int(
            datetime.combine(week_start, datetime.min.time(), tzinfo=UTC).timestamp()
        )
        hi = int(
            datetime.combine(
                week_end + timedelta(days=1), datetime.min.time(), tzinfo=UTC
            ).timestamp()
        )
        week_ticks = [(ts, d) for ts, d in ticks if lo <= ts < hi]
        weekly.append(
            {
                "start": week_start.isoformat(),
                "end": week_end.isoformat(),
                **measure_premise(week_ticks, topology),
            }
        )
    return {"overall": measure_premise(ticks, topology), "weekly": weekly}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Cluster disrupted segments into incidents against real "
        "archived movement data and report the measured clustering."
    )
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument("--days", type=int, default=8, help="trailing window")
    parser.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP)
    parser.add_argument(
        "--baseline",
        choices=("self", "published", "both"),
        default="self",
        help="self: build_segment_baseline over the fetched window (this "
        "module's original behavior). published: score against the live "
        "state/segment_params.json baseline instead. both: run and print "
        "each, so the two can be compared directly.",
    )
    args = parser.parse_args(argv)

    today = datetime.now(UTC).date()
    end = args.end_date or today
    start = args.start_date or (end - timedelta(days=args.days - 1))

    cfg = load_config()
    client = make_client(cfg)
    print(f"fetching archived vehicle metrics {start}..{end}", file=sys.stderr)
    bodies = fetch_vehicle_metrics(cfg, start_date=start, end_date=end, client=client)
    print(f"{len(bodies)} archived vehicle-metric ticks", file=sys.stderr)

    print("fetching static GTFS topology", file=sys.stderr)
    topology = topology_from_successors(load_successors())
    print(f"{len(topology)} segments in topology", file=sys.stderr)

    variants: dict[str, list[tuple[int, Mapping[str, str]]]] = {}
    if args.baseline in ("self", "both"):
        variants["self"] = segment_ticks_from_vehicle_bodies(bodies)
    if args.baseline in ("published", "both"):
        published_doc = json.loads(
            get_object_bytes(client, cfg.bucket, SEGMENT_PARAMS_KEY)
        )
        published = published_baseline_cells(published_doc)
        print(
            f"{len(published)} published baseline cells "
            f"(trained_at={published_doc.get('trained_at')})",
            file=sys.stderr,
        )
        variants["published"] = segment_ticks_with_baseline(bodies, published)

    for name, raw_calls in variants.items():
        ticks = _disrupted_ticks_from_calls(raw_calls)
        print(f"baseline={name}: {len(ticks)} reconstructed ticks", file=sys.stderr)
        report = premise_report(ticks, topology, start, end)
        print(json.dumps({"baseline": name, **report}, indent=2))

        state: IncidentState | None = None
        changes: list[RegimeChange] = []
        max_size: dict[str, int] = defaultdict(int)
        for observed_at, disrupted in ticks:
            state, tick_changes = advance_incidents(
                state, disrupted, topology, observed_at, max_gap=args.max_gap
            )
            changes.extend(tick_changes)
            for iid, fp in state.footprints.items():
                entry = state.regimes.get(iid)
                if entry is not None and entry.state == _ACTIVE:
                    max_size[iid] = max(max_size[iid], len(fp))
        final_state = state or IncidentState()

        completed = [c for c in changes if c.new_state == _ENDED]
        durations = [c.dwell_sec for c in completed]
        still_open = open_incident_regimes(final_state)
        incident_ids = {c.key for c in changes if c.new_state == _ENDED}
        incident_ids |= {
            iid for iid, entry in final_state.regimes.items() if entry.state == _ACTIVE
        }

        incident_report = {
            "baseline": name,
            "incident_count": len(incident_ids),
            "completed_incidents": len(completed),
            "still_open_incidents": len(still_open),
            "median_duration_sec": (
                statistics.median(durations) if durations else None
            ),
            "median_incident_size_segments": (
                statistics.median(max_size.values()) if max_size else None
            ),
        }
        print(json.dumps(incident_report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
