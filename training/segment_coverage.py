"""Sweep the segment-flow accumulator's coverage/false-alarm frontier.

`worker/src/segment_flow.ts` judges a segment cell only once its decayed
matched-trip count clears `MIN_EFF_MATCHED`. Measured on the live state
(`state/segment_flow.json`, 2026-08-23): median decayed matched is 1.03
against a floor of 5, so only 11 of 1658 baselined cells ever get a verdict on
a given tick — the accumulation POLICY is the bottleneck, not the trainer
baseline (only 842 of 1658 cells even have a measured baseline to begin with,
which this module cannot fix; it only asks how much of the other axis is
recoverable).

This module reproduces the Worker's accumulator exactly —
`updateSegmentFlow` + `classifySegments` (segment_flow.ts), whose decision
rule is `classifyAdvance` (movement_state.ts) — and replays it against the
archived vehicle stream tick by tick, scored against the PUBLISHED baseline
(`state/segment_params.json`), never one self-trained on the measured replay
window (a confound this repo has hit before: training.incidents.py's
docstring is the record of it). It sweeps three independent levers:

  - `decay` (SEGMENT_DECAY): the accumulator's effective window,
    ~1/(1-decay) ticks. Trades TEMPORAL resolution for coverage — a verdict
    can be stale by up to the window's length.
  - `min_eff_matched` (MIN_EFF_MATCHED): the floor a cell's decayed matched
    count must clear before it is judged at all.
  - `expand`: whether an observed multi-station jump (a train seen at stop A
    one tick and stop C the next, skipping B) credits every hop it provably
    crossed, not just the A-keyed cell it's recorded under. Valid ONLY for
    the advance question — a jump proves every intervening hop was crossed
    SOMEWHERE inside the tick, never which hop the train occupied when, so
    this must never feed a timing/traversal measure. "Provably" is a
    pattern-level test (`PatternIndex`): every scheduled stopping pattern
    that could have produced the jump must agree on which stops it passed
    through, or the observation doesn't determine it and only the from_stop
    cell is credited, exactly as today. An EARLIER version of this rule
    walked the successor GRAPH instead of scheduled PATTERNS and measured a
    2.37x credit multiplier at 0.9% ambiguous — wrong, because base routes
    fold express/local variants (6X/7X/FX -> 6/7/F) onto one graph, so a
    graph-shortest-path silently credits hops an express run skipped. The
    pattern-level test corrected that to 2.30x at ~16-17% ambiguous (cross-
    checked independently twice against 2026-08-21, a weekday, in this
    session — see the frontier run's stderr log for this run's own figure).

...and ALSO measures one structurally different variant, corridor pooling: a
cell under the floor pools forward along its successor chain (the static
adjacency graph, not scheduled patterns) until the POOLED decayed count
clears the floor, and the resulting verdict is attributed to every cell the
corridor spans. This trades SPATIAL resolution for coverage the same way
`decay` trades temporal resolution — see `classify_with_corridor`.

Every policy is scored on:
  - `judged` / `judged_share` — the coverage numerator, against the 1658-cell
    published baseline.
  - `disrupted_share` — of judged cells, the share reading disrupted.
  - `quiet_disrupted_rate` — disrupted share restricted to route-ticks with
    NO active severe alert (Severe Delays or a suspension) — the false-alarm
    proxy, via `training.review.mta_truth` at the canonical severity floor
    (the repo's one severity-graded truth rule; not reimplemented here).
  - `route_agreement` — of judged cells, the share agreeing with the raw
    (undecayed) route+direction `classify_direction` call for the same tick.

Selection rule, decided before any number was measured: maximise
`judged_share` subject to `quiet_disrupted_rate` no worse than the status quo
(decay=0.8, floor=5) plus 1 percentage point. See `select_policy`.

OUTCOME (2026-08-23, pinned 7-day replay 2026-08-17..08-23,
n_alert=944808 route-ticks): the mechanical rule above picked decay=0.98
(~250-min window) -- coverage 72.30% of baselined cells/tick vs the 1.26%
status quo, alert_disrupted_rate (recall, relative only -- see caveat)
0.22% vs the status quo's 0.05%. THAT PICK WAS NOT SHIPPED: at a 250-minute
window `entered_at` can start up to four hours after a disruption's true
onset and `recovery` is a forecast conditioned on that same clock, and
training/episodes.py's own 21-day sample puts the MEDIAN alert episode (the
truth this is graded against) at 45 minutes -- 5.6x shorter than that
window. SHIPPED INSTEAD: decay=0.94 (~83-min window, ~1.8x the median
episode), a JUDGEMENT CALL against incident duration overriding the
mechanical rule's own answer -- captures 77% of 0.98's recall (0.17% vs
0.22%) at 42.28% coverage (701/1658 cells/tick), quiet_disrupted_rate 0.49%
(ceiling was 6.12%). floor held at 3: 1/2/3 tie exactly at every decay
tested (movement_state.classifyAdvance's own MIN_MATCHED_TRIPS=3 already
rejects anything thinner, so a lower floor is cosmetic).

CAVEAT: alert_disrupted_rate and quiet_disrupted_rate are tiny in absolute
terms -- the truth column is ROUTE-level severity while the judged column
is PER-SEGMENT, so only the RELATIVE ordering across policies here is
meaningful; none of these percentages is an accuracy claim on its own.

TWO VARIANTS WERE MEASURED AND REJECTED, real negative results kept here
rather than omitted:
  - CORRIDOR POOLING (`classify_with_corridor`, `--max-corridor`): at
    decay=0.80's native 25-min window, pooling reached 45.61% coverage but
    only 0.06% recall -- barely above the FLAT classifier's own 0.05% at
    the same window, and under a third of decay=0.94's 0.17% at similar
    coverage. Shipped to the Worker once (2026-08-23), reverted the same
    day once this wider-window comparison landed. Space was never where
    the leverage was: a spatially pooled measurement dilutes a spatially
    sharp signal the same way a wide time window dilutes a temporally
    sharp one, and the temporal knob had far more headroom. This module
    keeps the sweep so the comparison stays reproducible; the Worker does
    not carry the corridor classifier.
  - EXPAND: costs recall at every decay tested (0.80: 0.05% -> 0.03%;
    0.94: 0.17% -> 0.10%; 0.98: 0.22% -> 0.13%) in exchange for coverage --
    the opposite of what the retune was for. Not shipped for that reason,
    not for the ~685KB trainer-side hop map it would additionally need.

FOLLOW-UP (not done here): the metric that should govern SEGMENT_DECAY is
detection LATENCY against the movement truth -- how long after a real
onset the surface flips -- not a route-level recall proxy with a ceiling
far below 1.0. training/scorecard.py's `onset_latency` and
training/review.py's `changepoint_alignment` already do this for the
route-level HMM; extending either to segment-level regimes would let this
knob be picked directly against latency instead of against incident
duration as a stand-in, which is what the 2026-08-23 choice actually is.

Run with:
    uv run python -m training.segment_coverage [--days 7]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo

from momentarily.hmm import tod_bin
from training.gtfs_static import Pattern, Timetable, base_route, load_timetable
from training.hierarchical import PooledCell
from training.incidents import (
    SEGMENT_PARAMS_KEY,
    SegmentKey,
    Topology,
    parse_topology,
    published_baseline_cells,
)
from training.load import TICK_SECONDS
from training.load_r2 import (
    ADVANCE_PRIOR_STRENGTH,
    AdvanceBaseline,
    build_movement_series_by_direction,
    classify_direction,
    fetch_vehicle_metrics,
)
from training.r2_client import get_object_bytes, load_config, make_client
from training.review import CANONICAL_SEVERITY_FLOOR, load_truth_observations, mta_truth
from training.segments import classify_segment

# --- Mirrors of the Worker's live constants (segment_flow.ts) --------------

# Drop a cell from the carried accumulator once its decayed matched falls
# below this (gone quiet), so the working set stays bounded. Never swept —
# it only prunes bookkeeping between ticks, it never gates a verdict.
PRUNE_MATCHED = 0.3

# The Worker's live policy today — the frontier's status-quo row and the
# negative-result yardstick `select_policy` measures every candidate against.
STATUS_QUO_DECAY = 0.8
STATUS_QUO_FLOOR = 5

DECAY_GRID: tuple[float, ...] = (0.8, 0.9, 0.94, 0.96, 0.98)
FLOOR_GRID: tuple[int, ...] = (1, 2, 3, 5)
EXPAND_GRID: tuple[bool, ...] = (False, True)

# Selection rule v2 (2026-08-23): the original rule (quiet_disrupted_rate
# ceiling alone) had no sensitivity term, so a policy that simply detects
# less of everything — real disruptions and false alarms together — scored
# as an improvement. QUIET_CEILING_SLACK is the false-alarm side (a swept
# policy may not read more disrupted than the status quo on QUIET route-
# ticks by more than this many points); RECALL_FLOOR_RATIO is the new
# sensitivity side (a swept policy must still read disrupted on ALERT-
# active route-ticks at least this fraction as often as the status quo
# does). Both floors are fixed here, before any candidate is measured
# against them — see select_policy.
QUIET_CEILING_SLACK = 0.01
RECALL_FLOOR_RATIO = 0.75

# A feed glitch or a direction reversal can nominally produce an A>C jump the
# timetable would need a dozen hops to explain. This refuses to credit any of
# it once the (already-unanimous) implied hop count exceeds the cap, rather
# than smearing one tick's observation across that much of the line.
MAX_EXPAND_HOPS = 8

# Safety bound for corridor pooling, not a measured constant: without it a
# chronically thin chain near a terminal (where the successor chain runs out,
# or every downstream cell is just as quiet) would pool indefinitely and
# never actually clear the floor, burning cycles for no verdict.
DEFAULT_MAX_CORRIDOR = 12

# `state/params.json` — the trainer's published HMM + movement-baseline
# doc. Every module that reads it (train_em.py, eval.py, worker/params.ts)
# defines its own copy of this key rather than sharing one; matched here.
PARAMS_KEY = "state/params.json"

# A service day is a local-time question (gtfs_static._ET, private there).
_ET = ZoneInfo("America/New_York")


def _snap_tick(epoch: int) -> int:
    return (epoch // TICK_SECONDS) * TICK_SECONDS


def _service_day(tick: int) -> date:
    return datetime.fromtimestamp(tick, tz=_ET).date()


# --- Tick replay input -------------------------------------------------


@dataclass(frozen=True)
class TickTransitions:
    """One archived tick's raw by_direction transitions, close to the
    archived shape (route -> direction -> "from>to" -> n) rather than
    pre-aggregated to from_stop, because path expansion needs the individual
    (from, to) pairs before they collapse into a from_stop-only count."""

    tick: int
    service_day: date
    routes: Mapping[str, Mapping[str, Mapping[str, int]]]


def ticks_from_bodies(bodies: Sequence[Mapping[str, Any]]) -> list[TickTransitions]:
    """Archived vehicle-metric bodies -> tick-ordered replay input. Sums
    transitions across bodies that snap to the same tick (the archive is one
    object per tick in practice; summing rather than assuming uniqueness
    costs nothing and stays correct either way)."""
    by_tick: dict[int, dict[str, dict[str, dict[str, int]]]] = {}
    for body in bodies:
        tick = _snap_tick(int(body.get("observed_at") or 0))
        rows = cast("dict[str, Any]", body.get("rows") or {})
        dest = by_tick.setdefault(tick, {})
        for route, row in rows.items():
            if not isinstance(row, dict):
                continue
            by_dir = cast(
                "dict[str, Any]", cast("dict[str, Any]", row).get("by_direction") or {}
            )
            for direction in ("north", "south"):
                drow = by_dir.get(direction)
                if not isinstance(drow, dict):
                    continue
                trans = cast(
                    "dict[str, Any]",
                    cast("dict[str, Any]", drow).get("transitions") or {},
                )
                if not trans:
                    continue
                dest_dir = dest.setdefault(route, {}).setdefault(direction, {})
                for pair, n in trans.items():
                    if ">" not in pair:
                        continue
                    nn = int(n or 0)
                    if nn <= 0:
                        continue
                    dest_dir[pair] = dest_dir.get(pair, 0) + nn
    return [
        TickTransitions(tick=t, service_day=_service_day(t), routes=routes)
        for t, routes in sorted(by_tick.items())
    ]


# --- Path expansion: pattern-level "which hops did this jump provably cross" --


def _route_direction_of_code(code: str) -> tuple[str, str] | None:
    """Base route + direction a scheduled path code names, from NYCT's own
    "<route>..<dir><variant>" convention — the same parse
    gtfs_static.is_express_variant (route field) and direction_of's trip_id
    fallback (direction char) already rely on. A Pattern carries no route or
    direction field of its own (DayTimetable.patterns is keyed on path code
    alone), so this is how candidates get bucketed by (route, direction)."""
    route_field, sep, rest = code.partition("..")
    if not sep or not rest:
        return None
    direction = {"N": "north", "S": "south"}.get(rest[0])
    if direction is None:
        return None
    return base_route(route_field), direction


def _intervening_stops(
    patterns: Sequence[Pattern], frm: str, to: str, *, max_hops: int
) -> tuple[str, ...] | None:
    """Every hop an observed frm>to jump provably crossed, or None when the
    observation doesn't determine it.

    Requires EVERY candidate pattern containing frm before to to agree on the
    intervening stop list — the same unanimity DayTimetable.span applies per
    trip ("a disagreement between candidates is not a reading of the
    timetable"), reused here at route+direction granularity because the
    archived transitions are trip-anonymous (aggregated off
    by_direction.transitions, no trip_id survives). An express pattern
    skipping a stop a local pattern serves is exactly the disagreement this
    must catch: crediting the skipped hop to a run that never made it would
    be a fabricated observation, not a read of more evidence.

    Valid for the ADVANCE question only. The jump proves the train crossed
    every returned hop SOMEWHERE inside the tick, never which hop it
    occupied when — never feed this into a timing/traversal measure.
    """
    answers: set[tuple[str, ...]] = set()
    for pattern in patterns:
        i = pattern.index.get(frm)
        j = pattern.index.get(to)
        if i is None or j is None or j <= i:
            continue
        answers.add(pattern.stops[i:j])
    if len(answers) != 1:
        return None
    hops = answers.pop()
    return hops if len(hops) <= max_hops else None


class PatternIndex:
    """Read-side path-expansion oracle for one archive replay.

    `day_patterns` resolves a service day to that day's {path_code: patterns}
    map — normally `Timetable.day(d).patterns` — injected so this class stays
    testable without a static-feed fetch. Both the (route, direction)
    grouping and the per-jump agreement check are static for the life of a
    replay (the timetable doesn't change tick to tick), so both are
    memoized: the same jump pair recurs every tick a corridor is busy.
    """

    def __init__(
        self, day_patterns: Callable[[date], Mapping[str, tuple[Pattern, ...]]]
    ) -> None:
        self._day_patterns = day_patterns
        self._by_route_dir: dict[date, dict[tuple[str, str], list[Pattern]]] = {}
        self._cache: dict[tuple[date, str, str, str, str], tuple[str, ...] | None] = {}

    def _grouped(self, day: date) -> dict[tuple[str, str], list[Pattern]]:
        cached = self._by_route_dir.get(day)
        if cached is not None:
            return cached
        grouped: dict[tuple[str, str], list[Pattern]] = {}
        for code, patterns in self._day_patterns(day).items():
            rd = _route_direction_of_code(code)
            if rd is None:
                continue
            grouped.setdefault(rd, []).extend(patterns)
        self._by_route_dir[day] = grouped
        return grouped

    def implied_hops(
        self, day: date, route: str, direction: str, frm: str, to: str
    ) -> tuple[str, ...] | None:
        key = (day, route, direction, frm, to)
        if key in self._cache:
            return self._cache[key]
        candidates = self._grouped(day).get((route, direction), [])
        result = _intervening_stops(candidates, frm, to, max_hops=MAX_EXPAND_HOPS)
        self._cache[key] = result
        return result


def tick_counts(
    tick: TickTransitions, *, expand: bool, patterns: PatternIndex | None
) -> dict[str, tuple[float, float]]:
    """This tick's advanced/matched increment per segment key
    (`route|direction|from_stop`), mirroring segment_flow.ts's tickCounts.

    `expand=False` reproduces the Worker's live behaviour exactly: a jump
    credits only its own from_stop, a stall credits only itself. `expand=True`
    additionally credits every hop a jump provably crossed (PatternIndex) —
    stalls (frm == to) are never expandable, they are evidence about exactly
    one cell.
    """
    out: dict[str, list[float]] = {}

    def bump(key: str, adv: float, matched: float) -> None:
        acc = out.setdefault(key, [0.0, 0.0])
        acc[0] += adv
        acc[1] += matched

    for route, by_dir in tick.routes.items():
        for direction, trans in by_dir.items():
            for pair, n in trans.items():
                frm, sep, to = pair.partition(">")
                if not sep or not frm or not to:
                    continue
                nf = float(n)
                if frm == to:
                    bump(f"{route}|{direction}|{frm}", 0.0, nf)
                    continue
                hops: tuple[str, ...] | None = None
                if expand and patterns is not None:
                    hops = patterns.implied_hops(
                        tick.service_day, route, direction, frm, to
                    )
                if hops is None:
                    bump(f"{route}|{direction}|{frm}", nf, nf)
                else:
                    for stop in hops:
                        bump(f"{route}|{direction}|{stop}", nf, nf)
    return {k: (v[0], v[1]) for k, v in out.items()}


# --- Sizing a trainer-published hop map (the shippable form of `expand`) ---
#
# The Worker cannot run the pattern-unanimity test live: segment_params.json
# carries `adjacency` (successors), not scheduled stopping patterns, and the
# Worker has no stop_times to derive them from. If `expand`'s coverage gain
# is worth shipping, the trainer would need to precompute
# PatternIndex.implied_hops for every OBSERVED (route, direction, from, to)
# jump pair — bounded by what the archive actually sees, not the full
# stop-pair cross product — and publish the resulting hop lists alongside
# segment_params.json. Sized here, not implemented as a live-path fallback.


def observed_jump_pairs(
    ticks: Sequence[TickTransitions],
) -> set[tuple[str, str, str, str]]:
    """Every distinct (route, direction, from, to) NON-STALL pair seen
    anywhere in the replay window — the key space a hop map would actually
    need, not every stop pair the topology could theoretically name."""
    out: set[tuple[str, str, str, str]] = set()
    for t in ticks:
        for route, by_dir in t.routes.items():
            for direction, trans in by_dir.items():
                for pair in trans:
                    frm, sep, to = pair.partition(">")
                    if sep and frm and to and frm != to:
                        out.add((route, direction, frm, to))
    return out


# Stable print/publish order: a map keyed by service class needs one row per
# class it actually distinguishes.
_SERVICE_CLASS_ORDER = ("weekday", "saturday", "sunday")


def representative_service_dates(
    ticks: Sequence[TickTransitions],
) -> list[tuple[str, date]]:
    """One representative date per weekday/Saturday/Sunday service class
    actually present in the replay window — the MOST RECENT occurrence of
    each, so a map sized off it uses a date the fetched archive covers
    rather than an arbitrary hardcoded one. NYCT's calendar runs materially
    different service weekday vs. weekend (Timetable's own docstring: pooling
    them gets 26% of hops wrong by >10%), so a hop map keyed on a single
    flat date would silently mis-serve whichever classes it wasn't built
    from — this is why the map has to be per service class, not one map."""
    by_class: dict[str, date] = {}
    for t in ticks:
        weekday = t.service_day.weekday()  # Monday=0 .. Sunday=6
        label = "saturday" if weekday == 5 else "sunday" if weekday == 6 else "weekday"
        prev = by_class.get(label)
        if prev is None or t.service_day > prev:
            by_class[label] = t.service_day
    return [
        (label, by_class[label]) for label in _SERVICE_CLASS_ORDER if label in by_class
    ]


def size_hop_map(
    pairs: set[tuple[str, str, str, str]],
    pattern_index: PatternIndex,
    service_dates: Sequence[tuple[str, date]],
) -> dict[str, Any]:
    """Size the trainer-published hop map exactly as it would be published:
    one JSON object per service class, `"route|direction|from|to": [hops]`
    for every observed pair that resolves unambiguously on that class's
    representative date (an ambiguous/off-pattern pair gets no entry — the
    Worker's fallback, crediting the from_stop cell alone, is unaffected and
    needs no explicit "no data" marker). `json_bytes` is measured by actually
    serializing the map, not estimated from an average entry size."""
    doc: dict[str, dict[str, list[str]]] = {label: {} for label, _ in service_dates}
    for label, day in service_dates:
        for route, direction, frm, to in pairs:
            hops = pattern_index.implied_hops(day, route, direction, frm, to)
            if hops is not None:
                doc[label][f"{route}|{direction}|{frm}|{to}"] = list(hops)
    counts = {label: len(doc[label]) for label, _ in service_dates}
    payload = json.dumps(doc, separators=(",", ":"))
    return {
        "observed_pairs": len(pairs),
        "entries_per_class": counts,
        "total_entries": sum(counts.values()),
        "json_bytes": len(payload.encode("utf-8")),
    }


# --- The accumulator (updateSegmentFlow) and classifier (classifySegments) --


def advance_accumulator(
    prev: Mapping[str, tuple[float, float]],
    counts: Mapping[str, tuple[float, float]],
    *,
    decay: float,
    tracked: frozenset[str],
) -> dict[str, tuple[float, float]]:
    """One tick of the decayed accumulator, reproduced exactly from
    updateSegmentFlow: this tick's count plus `decay` times the previous
    decayed value, pruned once decayed matched falls under PRUNE_MATCHED so
    the carried state stays bounded the same way the Worker's does.
    `tracked` (the published baseline's cell keys) mirrors
    `if (!(key in params.cells)) continue;` — a key the trainer never
    baselined is never accumulated, no matter how it presents in the feed."""
    keys = (set(prev) | set(counts)) & tracked
    out: dict[str, tuple[float, float]] = {}
    for key in keys:
        pa, pm = prev.get(key, (0.0, 0.0))
        ta, tm = counts.get(key, (0.0, 0.0))
        a = ta + decay * pa
        m = tm + decay * pm
        if m < PRUNE_MATCHED:
            continue
        out[key] = (a, m)
    return out


def replay_states(
    counts_by_tick: Sequence[Mapping[str, tuple[float, float]]],
    *,
    decay: float,
    tracked: frozenset[str],
) -> list[dict[str, tuple[float, float]]]:
    """The accumulator's state after each tick, in order — the input every
    classifier (plain or corridor) reads from for that tick."""
    states: list[dict[str, tuple[float, float]]] = []
    state: dict[str, tuple[float, float]] = {}
    for counts in counts_by_tick:
        state = advance_accumulator(state, counts, decay=decay, tracked=tracked)
        states.append(state)
    return states


def classify_cells(
    state: Mapping[str, tuple[float, float]],
    baseline: Mapping[str, PooledCell],
    adjacency_keys: frozenset[str],
    *,
    min_eff_matched: int,
) -> tuple[dict[str, str], list[int]]:
    """One tick's segment verdicts, mirroring classifySegments exactly:
    round the decayed counts, gate on the floor, then the shared
    Beta-Binomial call (training.segments.classify_segment ==
    movement_state.classifyAdvance). Only cells with BOTH a published
    baseline AND an adjacency entry are eligible, matching
    `if (!cell || !adj) continue;`. Returns an empty span list — spans are
    only a corridor-pooling concept — so this shares a call signature with
    `classify_with_corridor`."""
    out: dict[str, str] = {}
    for key, (a, m) in state.items():
        cell = baseline.get(key)
        if cell is None or key not in adjacency_keys:
            continue
        matched = round(m)
        if matched < min_eff_matched:
            continue
        advanced = min(round(a), matched)
        call = classify_segment(advanced, matched - advanced, cell)
        if call is not None:
            out[key] = call
    return out, []


def _pooled_reference(
    chain: Sequence[str],
    state: Mapping[str, tuple[float, float]],
    baseline: Mapping[str, PooledCell],
) -> PooledCell:
    """The corridor's judged-against rate: the DECAYED-m-WEIGHTED mean of its
    members' own baseline p0 — sum(p0_i * m_i) / sum(m_i), the SAME
    weighting `advanced`/`matched` already carry, since both are sums of the
    members' own decayed a/m. An UNWEIGHTED mean compares a weighted
    observation against an unweighted null: traffic volume and baseline
    advance rate correlate across the network, so a busy trunk hop and a
    quiet branch hop are not statistically interchangeable, and judging
    against the wrong null systematically suppresses or fabricates
    disruption calls depending on which way the correlation runs — not
    noise, a bias (2026-08-23, caught after the unweighted version had
    already shipped a live recommendation off it once). A member with m=0
    (including one never observed this tick) naturally contributes nothing
    to either sum, no special case needed. `baseline[k]` is never defaulted
    on a miss: every chain member is guaranteed present by construction
    (classify_with_corridor only ever extends into a key that is already in
    `baseline`), so a missing entry raises KeyError instead of silently
    contributing p0=0 — the invariant is enforced, not papered over.
    Re-anchored at ADVANCE_PRIOR_STRENGTH pseudo-trials the same way
    incidents.published_baseline_cells reconstructs a Beta prior from a bare
    p0 — classify_segment reads only `.p0` off what it's given, so this is
    exact for classification even though `n` doesn't mean what a single
    leaf's `n` would."""
    numerator = 0.0
    denominator = 0.0
    for k in chain:
        _, m = state.get(k, (0.0, 0.0))
        numerator += baseline[k].p0 * m
        denominator += m
    # denominator > 0 is guaranteed here: the caller only reaches this once
    # the SAME sum (mSum) has already cleared min_eff_matched >= 1.
    p0 = numerator / denominator
    n = sum(baseline[k].n for k in chain)
    return PooledCell(
        p0=p0,
        raw=p0,
        n=n,
        alpha=p0 * ADVANCE_PRIOR_STRENGTH,
        beta=(1.0 - p0) * ADVANCE_PRIOR_STRENGTH,
        source="corridor",
    )


def classify_with_corridor(
    state: Mapping[str, tuple[float, float]],
    baseline: Mapping[str, PooledCell],
    adjacency_keys: frozenset[str],
    topology: Topology,
    *,
    min_eff_matched: int,
    max_corridor: int = DEFAULT_MAX_CORRIDOR,
) -> tuple[dict[str, str], list[int]]:
    """Corridor-pooling variant of classify_cells: a cell under the floor
    pools forward along its successor chain (the static adjacency graph
    parse_topology already builds for incident clustering) until the pooled
    decayed matched count clears `min_eff_matched`, then the WHOLE corridor
    is judged as one segment and its verdict attributed to every cell it
    spans — trading spatial resolution for coverage the way `decay` trades
    temporal resolution.

    The chain only extends through a cell with EXACTLY ONE successor in the
    STATIC topology: a branch point (express/local split, a merge) has no
    single "next" cell to pool with, so the corridor stops there rather than
    guessing which branch the evidence belongs to — even when only one of
    the branches happens to carry a published baseline, continuing down it
    would still be treating a genuinely forked stretch of track as a simple
    corridor. `max_corridor` bounds how far a chronically thin cell may
    reach before giving up unjudged, so a chain that never clears the floor
    doesn't silently absorb the whole route.

    Allocation is greedy in `baseline` iteration order: the first eligible
    cell to start a corridor claims every cell it pools, so a later cell that
    would have formed a different (possibly better) corridor on its own
    never gets the chance. Deterministic and simple to reason about; not
    claimed to be optimal.

    Returns (verdicts keyed the same as classify_cells, corridor span in
    cells for every corridor that WAS judged, including spans of 1 — cells
    that cleared the floor alone, no pooling needed)."""
    out: dict[str, str] = {}
    spans: list[int] = []
    judged_already: set[str] = set()
    eligible = [k for k in baseline if k in adjacency_keys]
    for key in eligible:
        if key in judged_already:
            continue
        chain = [key]
        a_sum, m_sum = state.get(key, (0.0, 0.0))
        node = key
        while round(m_sum) < min_eff_matched and len(chain) < max_corridor:
            raw_successors = topology.get(node, ())
            if len(raw_successors) != 1:
                break  # branch point or dead end in the static topology
            nxt = raw_successors[0]
            if nxt not in baseline or nxt not in adjacency_keys:
                break  # the one physical successor isn't judgeable
            if nxt in chain or nxt in judged_already:
                # `nxt in chain` guards a cycle (shouldn't occur on real
                # topology); `nxt in judged_already` guards a MERGE point —
                # two different anchors can each have exactly one raw
                # successor and still both point at the same downstream
                # cell. Without this, a cell already committed to an
                # earlier corridor could be pooled into a later one too,
                # double-counting its evidence and silently overwriting its
                # published verdict with whichever corridor processed it
                # last.
                break
            a2, m2 = state.get(nxt, (0.0, 0.0))
            a_sum += a2
            m_sum += m2
            chain.append(nxt)
            node = nxt
        matched = round(m_sum)
        if matched < min_eff_matched:
            continue
        advanced = min(round(a_sum), matched)
        call = classify_segment(
            advanced, matched - advanced, _pooled_reference(chain, state, baseline)
        )
        if call is None:
            continue
        for k in chain:
            out[k] = call
            judged_already.add(k)
        spans.append(len(chain))
    return out, spans


# --- Published baselines -----------------------------------------------


def movement_baseline_from_doc(
    doc: Mapping[str, Any],
) -> dict[tuple[str, str, int], AdvanceBaseline]:
    """`state/params.json`'s `movement_baseline` field -> the same
    {(route, direction, tod_bin): AdvanceBaseline} shape load_r2.
    compute_advance_baseline returns. Used only for `route_agreement`'s
    route+direction call — the PUBLISHED prior, matching the segment side's
    use of `published_baseline_cells` rather than a baseline self-trained on
    the measured replay window."""
    out: dict[tuple[str, str, int], AdvanceBaseline] = {}
    raw = cast("dict[str, Any]", doc.get("movement_baseline") or {})
    for route, by_dir_any in raw.items():
        if not isinstance(by_dir_any, dict):
            continue
        by_dir = cast("dict[str, Any]", by_dir_any)
        for direction, by_tod_any in by_dir.items():
            if not isinstance(by_tod_any, dict):
                continue
            by_tod = cast("dict[str, Any]", by_tod_any)
            for tod_str, cell_any in by_tod.items():
                if not isinstance(cell_any, dict):
                    continue
                cell = cast("dict[str, Any]", cell_any)
                try:
                    out[(route, direction, int(tod_str))] = AdvanceBaseline(
                        p0=float(cell["p0"]),
                        n=int(cell.get("n", 0)),
                        alpha=float(cell["alpha"]),
                        beta=float(cell["beta"]),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
    return out


def route_direction_calls_by_tick(
    dir_series: Mapping[tuple[str, str, int], Mapping[str, int]],
    movement_baseline: Mapping[tuple[str, str, int], AdvanceBaseline],
) -> dict[int, dict[tuple[str, str], str]]:
    """Per-tick {(route, direction): call} — the raw (undecayed)
    classify_direction read `route_agreement` compares judged segments
    against. Independent of every swept policy (it never touches the segment
    accumulator), so it is computed once and shared across the whole sweep."""
    out: dict[int, dict[tuple[str, str], str]] = {}
    for (route, direction, tick), row in dir_series.items():
        baseline = movement_baseline.get((route, direction, tod_bin(tick)))
        call = classify_direction(
            int(row.get("advanced_n", 0)), int(row.get("stalled_n", 0)), baseline
        )
        if call is not None:
            out.setdefault(tick, {})[(route, direction)] = call
    return out


def baselined_cells_by_route(baseline: Mapping[str, PooledCell]) -> dict[str, int]:
    """Baselined cell count per route — the static per-route denominator
    basis `alert_disrupted_rate` needs: ALL of a route's segments while it
    has an active severe alert, not just the ones a policy happened to
    judge. Independent of every swept policy."""
    out: dict[str, int] = {}
    for key in baseline:
        route = key.split("|", 1)[0]
        out[route] = out.get(route, 0) + 1
    return out


def alerted_routes_by_tick(
    truth: Mapping[tuple[str, int], str],
) -> dict[int, list[str]]:
    """tick -> every route with an active severe alert that tick (severity-
    graded truth != "normal"). Independent of every swept policy — it never
    touches the segment accumulator, only the alert archive."""
    out: dict[int, list[str]] = {}
    for (route, tick), state in truth.items():
        if state != "normal":
            out.setdefault(tick, []).append(route)
    return out


# --- Scoring one policy over the whole replay ---------------------------


def _percentile(values: Sequence[float], pct: float) -> float:
    """pct in (0, 100]. A single-sample series has no distribution to
    interpolate, so it stands in for its own percentile."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    qs = statistics.quantiles(sorted(values), n=100, method="inclusive")
    idx = min(max(round(pct), 1), 99) - 1
    return qs[idx]


@dataclass(frozen=True)
class PolicyResult:
    label: str
    decay: float
    min_eff_matched: int
    expand: bool
    corridor: bool
    n_ticks: int
    baselined_cells: int
    judged_total: int
    judged_per_tick_mean: float
    judged_share: float  # pooled: judged_total / (baselined_cells * n_ticks)
    judged_share_median_tick: float
    judged_share_p90_tick: float
    disrupted_share: float
    window_min: (
        float  # ~1/(1-decay) ticks, in minutes — the temporal cost, visible up front
    )
    quiet_disrupted_rate: float  # false-alarm proxy: disrupted share of JUDGED cells with NO active severe alert
    quiet_denominator: int
    # Recall/sensitivity proxy, denominated over every baselined cell of a
    # route while that route has an active severe alert — NOT just the
    # judged ones, so an abstention counts as a miss the same as a wrong
    # verdict would (see evaluate_policy). alert_disrupted_rate is the
    # product of the two components below; they're kept separate because a
    # low product from a collapsed alert_judged_share (went blind) and one
    # from a collapsed alert_disrupted_given_judged (calls alert routes
    # normal) are different defects with different fixes.
    #
    # CAVEAT: truth is ROUTE-level (a severe alert doesn't mean every
    # segment of that route is stalled), but the denominator is PER-SEGMENT.
    # Even a perfect segment classifier reads well under 1.0 here, so this
    # is only meaningful as a RELATIVE comparison ACROSS policies (which is
    # all the recall floor needs) — never as an absolute skill claim.
    alert_disrupted_rate: float
    alert_judged_share: (
        float  # judged / all baselined cells on alert route-ticks — did we even look?
    )
    alert_disrupted_given_judged: (
        float  # disrupted / judged there — what we said when we looked
    )
    alert_denominator: (
        int  # support behind the alert-stratum metrics; same across every policy
    )
    route_agreement: float
    route_agreement_denominator: int
    corridor_span_median: float | None
    corridor_span_p90: float | None
    # Independent PUBLISHED RECORDS, not cells covered: one per judged
    # verdict, whether self-measured (span 1, a `segments` entry) or pooled
    # (span >= 2, a `corridors` entry). For corridor=False every judged cell
    # is trivially its own span-1 record. This is what the published-payload
    # projection needs — record count, not cell count.
    measurement_count: int
    single_measurement_count: int  # span == 1 -> `segments` entries
    corridor_measurement_count: int  # span >= 2 -> `corridors` entries


def evaluate_policy(
    label: str,
    ticks: Sequence[TickTransitions],
    states_by_tick: Sequence[Mapping[str, tuple[float, float]]],
    *,
    decay: float,
    min_eff_matched: int,
    expand: bool,
    corridor: bool,
    baseline: Mapping[str, PooledCell],
    adjacency_keys: frozenset[str],
    topology: Topology,
    rd_calls_by_tick: Mapping[int, Mapping[tuple[str, str], str]],
    truth: Mapping[tuple[str, int], str],
    route_cell_counts: Mapping[str, int],
    alerted_routes: Mapping[int, Sequence[str]],
    max_corridor: int = DEFAULT_MAX_CORRIDOR,
) -> PolicyResult:
    """Score one policy over the whole replay: pool judged/disrupted counts
    tick by tick into the coverage, false-alarm (quiet_disrupted_rate), and
    recall (alert_disrupted_rate) metrics on PolicyResult. `route_cell_counts`
    and `alerted_routes` fix the recall denominator to every baselined cell
    of a route while it has an active severe alert — computed once, shared
    across the whole sweep, independent of `min_eff_matched`/`decay`/
    `expand`/`corridor` — so an abstaining policy's recall collapses the same
    way a wrong verdict's would, rather than shrinking numerator and
    denominator together and reading as flawless."""
    judged_total = 0
    disrupted_total = 0
    quiet_den = 0
    quiet_disrupted = 0
    alert_denominator = 0
    alert_judged = 0
    alert_disrupted = 0
    agree = 0
    agree_den = 0
    judged_per_tick: list[int] = []
    all_spans: list[int] = []

    for t, state in zip(ticks, states_by_tick, strict=True):
        if corridor:
            calls, spans = classify_with_corridor(
                state,
                baseline,
                adjacency_keys,
                topology,
                min_eff_matched=min_eff_matched,
                max_corridor=max_corridor,
            )
            all_spans.extend(spans)
        else:
            calls, _ = classify_cells(
                state, baseline, adjacency_keys, min_eff_matched=min_eff_matched
            )

        judged_per_tick.append(len(calls))
        judged_total += len(calls)
        rd_calls = rd_calls_by_tick.get(t.tick, {})

        # The recall denominator is fixed by which routes have an active
        # severe alert this tick — every one of that route's baselined
        # cells counts, whether this policy judged it or not, so an
        # abstained cell drags the rate down exactly like a wrong verdict
        # would (a policy that abstains straight through an alert must not
        # score as if it had nothing to detect).
        for route in alerted_routes.get(t.tick, ()):
            alert_denominator += route_cell_counts.get(route, 0)

        for key, verdict in calls.items():
            route, direction, _frm = key.split("|", 2)
            if verdict == "disrupted":
                disrupted_total += 1
            # `truth` carries only severity-graded severe/suspended states; a
            # route-tick absent from it (or explicitly "normal") had no
            # active severe alert — quiet.
            truth_state = truth.get((route, t.tick), "normal")
            if truth_state == "normal":
                quiet_den += 1
                if verdict == "disrupted":
                    quiet_disrupted += 1
            else:
                alert_judged += 1
                if verdict == "disrupted":
                    alert_disrupted += 1
            rd_call = rd_calls.get((route, direction))
            if rd_call is not None:
                agree_den += 1
                if rd_call == verdict:
                    agree += 1

    n_ticks = len(ticks)
    baselined = len(baseline)
    denom = baselined * n_ticks
    judged_shares = [j / baselined for j in judged_per_tick] if baselined else []
    true_corridor_spans = [s for s in all_spans if s >= 2]
    return PolicyResult(
        label=label,
        decay=decay,
        min_eff_matched=min_eff_matched,
        expand=expand,
        corridor=corridor,
        n_ticks=n_ticks,
        baselined_cells=baselined,
        judged_total=judged_total,
        judged_per_tick_mean=(judged_total / n_ticks) if n_ticks else 0.0,
        judged_share=(judged_total / denom) if denom else 0.0,
        judged_share_median_tick=statistics.median(judged_shares)
        if judged_shares
        else 0.0,
        judged_share_p90_tick=_percentile(judged_shares, 90),
        disrupted_share=(disrupted_total / judged_total) if judged_total else 0.0,
        window_min=(1.0 / (1.0 - decay)) * (TICK_SECONDS / 60.0),
        quiet_disrupted_rate=(quiet_disrupted / quiet_den) if quiet_den else 0.0,
        quiet_denominator=quiet_den,
        alert_disrupted_rate=(alert_disrupted / alert_denominator)
        if alert_denominator
        else 0.0,
        alert_judged_share=(alert_judged / alert_denominator)
        if alert_denominator
        else 0.0,
        alert_disrupted_given_judged=(alert_disrupted / alert_judged)
        if alert_judged
        else 0.0,
        alert_denominator=alert_denominator,
        route_agreement=(agree / agree_den) if agree_den else 0.0,
        route_agreement_denominator=agree_den,
        # Only genuine pooled corridors (span >= 2) — the population that
        # actually appears in the published `corridors` collection. A span-1
        # entry never pooled at all (it cleared the floor alone and
        # publishes in `segments`), so mixing it in would understate what a
        # real corridor costs.
        corridor_span_median=(
            statistics.median(true_corridor_spans) if true_corridor_spans else None
        ),
        corridor_span_p90=(
            _percentile([float(s) for s in true_corridor_spans], 90)
            if true_corridor_spans
            else None
        ),
        measurement_count=(len(all_spans) if corridor else judged_total),
        single_measurement_count=(
            sum(1 for s in all_spans if s == 1) if corridor else judged_total
        ),
        corridor_measurement_count=(len(true_corridor_spans) if corridor else 0),
    )


def select_policy(results: Sequence[PolicyResult]) -> PolicyResult | None:
    """The pre-committed selection rule (v2): maximise judged_share subject
    to BOTH (a) quiet_disrupted_rate — the false-alarm proxy — no worse than
    the status quo's own rate plus QUIET_CEILING_SLACK, and (b)
    alert_disrupted_rate — the sensitivity/recall proxy, read on route-ticks
    where a severe alert IS active — at least RECALL_FLOOR_RATIO times the
    status quo's own rate. (a) alone cannot tell "cleaner" from "detects
    less of everything": smoothing suppresses genuine detections and false
    ones together, so a policy that has gone blind reads as an improvement
    under (a) alone. Returns None when no swept policy clears BOTH bars with
    a strict coverage improvement — a negative result, not an error."""
    status_quo = next(
        r
        for r in results
        if not r.expand
        and not r.corridor
        and r.decay == STATUS_QUO_DECAY
        and r.min_eff_matched == STATUS_QUO_FLOOR
    )
    quiet_ceiling = status_quo.quiet_disrupted_rate + QUIET_CEILING_SLACK
    recall_floor = status_quo.alert_disrupted_rate * RECALL_FLOOR_RATIO
    candidates = [
        r
        for r in results
        if r is not status_quo
        and r.quiet_disrupted_rate <= quiet_ceiling
        and r.alert_disrupted_rate >= recall_floor
        and r.judged_share > status_quo.judged_share
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.judged_share)


# --- Reporting -----------------------------------------------------------


def format_frontier(results: Sequence[PolicyResult]) -> str:
    """The whole coverage/false-alarm frontier as one fixed-width table —
    every swept policy, not just the winner, so a reader can see what was
    traded for what."""
    head_cells = [
        "policy",
        "decay",
        "floor",
        "win_min",
        "ticks",
        "judg/tk",
        "judg_sh",
        "disr_sh",
        "quiet_dis",
        "alert_rec",
        "alert_judg",
        "alert_p|j",
        "n_alert",
        "rt_agree",
        "span_md",
        "span_p90",
    ]
    widths = [34, 6, 6, 8, 7, 9, 10, 9, 10, 10, 11, 10, 8, 9, 8, 9]
    lines = [
        "".join(
            c.rjust(w) if i else c.ljust(w)
            for i, (c, w) in enumerate(zip(head_cells, widths, strict=True))
        )
    ]
    lines.append("-" * sum(widths))
    for r in results:
        span_md = (
            f"{r.corridor_span_median:.1f}"
            if r.corridor_span_median is not None
            else "-"
        )
        span_p90 = (
            f"{r.corridor_span_p90:.1f}" if r.corridor_span_p90 is not None else "-"
        )
        cells = [
            r.label,
            f"{r.decay:.2f}",
            str(r.min_eff_matched),
            f"{r.window_min:.0f}",
            str(r.n_ticks),
            f"{r.judged_per_tick_mean:.1f}",
            f"{r.judged_share:.2%}",
            f"{r.disrupted_share:.2%}",
            f"{r.quiet_disrupted_rate:.2%}",
            f"{r.alert_disrupted_rate:.2%}",
            f"{r.alert_judged_share:.2%}",
            f"{r.alert_disrupted_given_judged:.2%}",
            str(r.alert_denominator),
            f"{r.route_agreement:.2%}",
            span_md,
            span_p90,
        ]
        lines.append(
            "".join(
                c.rjust(w) if i else c.ljust(w)
                for i, (c, w) in enumerate(zip(cells, widths, strict=True))
            )
        )
    return "\n".join(lines)


def policy_label(*, decay: float, floor: int, expand: bool, corridor: bool) -> str:
    tag = "corridor" if corridor else "grid"
    return f"{tag} decay={decay:.2f} floor={floor} expand={'Y' if expand else 'N'}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sweep the segment-flow accumulator's decay/floor/expand "
        "policy space against the archived vehicle stream and the published "
        "baseline, and report the coverage/false-alarm frontier."
    )
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument("--days", type=int, default=3, help="trailing window")
    parser.add_argument(
        "--max-corridor",
        type=int,
        default=DEFAULT_MAX_CORRIDOR,
        help="cells a chronically-thin corridor may pool before giving up unjudged",
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
    ticks = ticks_from_bodies(bodies)
    print(f"{len(ticks)} distinct replay ticks", file=sys.stderr)
    if not ticks:
        print("no vehicle-metric ticks in window; nothing to sweep", file=sys.stderr)
        return 1

    print(
        f"fetching published segment baseline ({SEGMENT_PARAMS_KEY})", file=sys.stderr
    )
    segment_doc = cast(
        "dict[str, Any]",
        json.loads(get_object_bytes(client, cfg.bucket, SEGMENT_PARAMS_KEY)),
    )
    baseline_tuples = published_baseline_cells(segment_doc)
    baseline: dict[SegmentKey, PooledCell] = {
        f"{r}|{d}|{f}": cell for (r, d, f), cell in baseline_tuples.items()
    }
    adjacency_doc = cast("dict[str, Any]", segment_doc.get("adjacency") or {})
    adjacency_keys: frozenset[str] = frozenset(adjacency_doc.keys())
    topology = parse_topology(adjacency_doc)
    print(
        f"{len(baseline)} published baseline cells, {len(adjacency_keys)} adjacency "
        f"entries (trained_at={segment_doc.get('trained_at')})",
        file=sys.stderr,
    )

    print(f"fetching published movement baseline ({PARAMS_KEY})", file=sys.stderr)
    params_doc = cast(
        "dict[str, Any]", json.loads(get_object_bytes(client, cfg.bucket, PARAMS_KEY))
    )
    movement_baseline = movement_baseline_from_doc(params_doc)
    print(
        f"{len(movement_baseline)} published (route,direction,tod) movement cells",
        file=sys.stderr,
    )

    print("fetching static GTFS timetable for path expansion", file=sys.stderr)
    table: Timetable = load_timetable()
    pattern_index = PatternIndex(day_patterns=lambda d: table.day(d).patterns)

    print(
        f"fetching alert archive {start}..{end} for severe-alert truth", file=sys.stderr
    )
    truth_obs = load_truth_observations(client, cfg.bucket, start, end)
    truth = mta_truth(truth_obs, severity_floor=CANONICAL_SEVERITY_FLOOR)
    print(f"{len(truth)} (route, tick) severity-graded truth cells", file=sys.stderr)

    dir_series = build_movement_series_by_direction(bodies)
    rd_calls_by_tick = route_direction_calls_by_tick(dir_series, movement_baseline)
    route_cell_counts = baselined_cells_by_route(baseline)
    alert_routes_by_tick = alerted_routes_by_tick(truth)

    tracked = frozenset(baseline)

    print("computing tick counts (expand=False, expand=True)...", file=sys.stderr)
    counts_false = [tick_counts(t, expand=False, patterns=None) for t in ticks]
    counts_true = [tick_counts(t, expand=True, patterns=pattern_index) for t in ticks]
    credits_false = sum(m for c in counts_false for _, m in c.values())
    credits_true = sum(m for c in counts_true for _, m in c.values())
    cells_false = len({k for c in counts_false for k in c})
    cells_true = len({k for c in counts_true for k in c})
    multiplier = credits_true / credits_false if credits_false else float("nan")
    print(
        f"path expansion: credits {credits_false:.0f} -> {credits_true:.0f} "
        f"({multiplier:.2f}x), distinct cells touched {cells_false} -> {cells_true}",
        file=sys.stderr,
    )

    print(
        "sizing a trainer-published hop map (the shippable form of expand)...",
        file=sys.stderr,
    )
    observed_pairs = observed_jump_pairs(ticks)
    service_dates = representative_service_dates(ticks)
    hop_map = size_hop_map(observed_pairs, pattern_index, service_dates)
    print(
        f"hop map: {hop_map['observed_pairs']} observed jump pairs, "
        f"entries per service class {hop_map['entries_per_class']}, "
        f"{hop_map['total_entries']} entries total, "
        f"{hop_map['json_bytes'] / 1024:.1f} KB as published JSON "
        f"(service classes: {[label for label, _ in service_dates]})",
        file=sys.stderr,
    )

    results: list[PolicyResult] = []
    for expand, counts_by_tick in ((False, counts_false), (True, counts_true)):
        for decay in DECAY_GRID:
            states = replay_states(counts_by_tick, decay=decay, tracked=tracked)
            for floor in FLOOR_GRID:
                results.append(
                    evaluate_policy(
                        policy_label(
                            decay=decay, floor=floor, expand=expand, corridor=False
                        ),
                        ticks,
                        states,
                        decay=decay,
                        min_eff_matched=floor,
                        expand=expand,
                        corridor=False,
                        baseline=baseline,
                        adjacency_keys=adjacency_keys,
                        topology=topology,
                        rd_calls_by_tick=rd_calls_by_tick,
                        truth=truth,
                        route_cell_counts=route_cell_counts,
                        alerted_routes=alert_routes_by_tick,
                    )
                )
            if decay == STATUS_QUO_DECAY:
                for corridor_floor in FLOOR_GRID:
                    results.append(
                        evaluate_policy(
                            policy_label(
                                decay=decay,
                                floor=corridor_floor,
                                expand=expand,
                                corridor=True,
                            ),
                            ticks,
                            states,
                            decay=decay,
                            min_eff_matched=corridor_floor,
                            expand=expand,
                            corridor=True,
                            baseline=baseline,
                            adjacency_keys=adjacency_keys,
                            topology=topology,
                            rd_calls_by_tick=rd_calls_by_tick,
                            truth=truth,
                            route_cell_counts=route_cell_counts,
                            alerted_routes=alert_routes_by_tick,
                            max_corridor=args.max_corridor,
                        )
                    )

    print()
    print(format_frontier(results))
    print()

    status_quo = next(
        r
        for r in results
        if not r.expand
        and not r.corridor
        and r.decay == STATUS_QUO_DECAY
        and r.min_eff_matched == STATUS_QUO_FLOOR
    )
    print(
        f"status quo: {status_quo.label} -> {format_frontier([status_quo]).splitlines()[-1]}"
    )
    print(
        f"  status quo alert-stratum support: {status_quo.alert_denominator} "
        f"baselined cell-ticks under an active severe alert "
        f"(alert_judged_share={status_quo.alert_judged_share:.2%}, "
        f"alert_disrupted_given_judged={status_quo.alert_disrupted_given_judged:.2%}, "
        f"alert_disrupted_rate={status_quo.alert_disrupted_rate:.2%})"
    )
    if status_quo.alert_denominator < 200:
        print(
            "  WARNING: alert-stratum support is thin (< 200 baselined cell-ticks) — "
            "the 0.75x recall floor may be measuring noise, not signal. Widen --days "
            "and re-run before trusting a selection made against it."
        )

    def _report(title: str, chosen: PolicyResult | None) -> None:
        if chosen is None:
            print(
                f"{title}: no swept policy beats the status quo's judged_share "
                "within both the quiet_disrupted_rate ceiling and the alert_disrupted_rate "
                "recall floor; recommend no change."
            )
            return
        print(f"{title}: {chosen.label}")
        print(
            f"  judged_share {status_quo.judged_share:.2%} -> {chosen.judged_share:.2%}, "
            f"judged/tick {status_quo.judged_per_tick_mean:.1f} -> "
            f"{chosen.judged_per_tick_mean:.1f} of {chosen.baselined_cells} baselined cells"
        )
        print(
            f"  quiet_disrupted_rate {status_quo.quiet_disrupted_rate:.2%} -> "
            f"{chosen.quiet_disrupted_rate:.2%} (ceiling was "
            f"{status_quo.quiet_disrupted_rate + QUIET_CEILING_SLACK:.2%})"
        )
        print(
            f"  alert_disrupted_rate (recall, relative only — see caveat) "
            f"{status_quo.alert_disrupted_rate:.2%} -> {chosen.alert_disrupted_rate:.2%} "
            f"(floor was {status_quo.alert_disrupted_rate * RECALL_FLOOR_RATIO:.2%}); "
            f"alert_judged_share {status_quo.alert_judged_share:.2%} -> "
            f"{chosen.alert_judged_share:.2%}, alert_disrupted_given_judged "
            f"{status_quo.alert_disrupted_given_judged:.2%} -> "
            f"{chosen.alert_disrupted_given_judged:.2%}"
        )
        print(
            f"  effective window ~{status_quo.window_min:.0f} min -> ~{chosen.window_min:.0f} min "
            "(worst-case verdict staleness moves with it)"
        )

    # `expand` needs scheduled stopping patterns (gtfs_static.Pattern), which
    # segment_params.json does not carry and the Worker has no stop_times to
    # derive live — see the module docstring. So the winner across the WHOLE
    # grid (below) is a ceiling on what's achievable once/if the trainer
    # publishes the hop map sized above; only the expand=False subset is
    # actually shippable today, and that's what decides the Worker change.
    _report(
        "selected (shippable today: decay/floor/corridor only)",
        select_policy([r for r in results if not r.expand]),
    )
    _report(
        "selected (whole grid, including expand — NOT shippable as-is)",
        select_policy(results),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
