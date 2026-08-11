"""Per-segment dwell curves, hierarchically pooled leaf -> route -> system.

Segment episodes are far shorter and far sparser than route episodes: most
segment cells accumulate only a handful of completed regime exits before the
usable movement-history window runs out (see load_r2/train_em route-level
dwell, which needs a route-wide corpus for the same reason). A per-cell
log-logistic fit (pooled_dwell.partially_pooled_dwell) is the right survival
math for a thin, right-censored sample, but most leaves lack even that.

Which level a leaf draws its curve from is decided the same way the
segment-level advance-rate baseline already is (training.hierarchical): pool
each leaf's completed-vs-still-open fraction hierarchically leaf ->
(route,dir) -> route -> system via the EXISTING partially_pool, and read off
which level had enough voting support. That walk is a genuine Beta-Binomial
question (the leaf's completion fraction is a bounded [0,1] proportion) and
partially_pool answers it unmodified. The *curve* at whatever level is chosen
is a different, unbounded (duration) question, answered exclusively by
pooled_dwell.partially_pooled_dwell / dwell.dwell_samples_by_cell — this
module fits no new survival math of its own.

A leaf votes for itself (gets its own standalone fit) once it clears
MIN_LEAF_N total observations, the same floor partially_pool uses to decide
whether a leaf's own rate is trustworthy enough to vote for its parent's
centre. Below that, the leaf borrows partially_pool's resolved level:
(route,dir)/route collapse to one "route" aggregate (every segment on that
route pooled into one curve for the state), and "system" falls back to one
curve pooled across every segment in the state, system-wide.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from training.dwell import DwellQuantiles, DwellSample, dwell_samples_by_cell
from training.eval import MovementTransitionRecord, TransitionRecord
from training.hierarchical import MIN_LEAF_N, partially_pool
from training.pooled_dwell import PooledDwellFit, cell_from_fit, partially_pooled_dwell

# Pseudo-key for the flat, system-wide pool (every segment, one state) — never
# collides with a real `route|direction|from_stop` segment key or a bare
# route id.
_SYSTEM_KEY = "__system__"


@dataclass(frozen=True)
class SegmentDwellStats:
    """How many (segment, state) cells landed at each level of the hierarchy."""

    n_cells_own: int
    n_cells_route: int
    n_cells_system: int
    n_cells_skipped: int

    @property
    def n_cells_total(self) -> int:
        return self.n_cells_own + self.n_cells_route + self.n_cells_system


def _segment_key_parts(key: str) -> tuple[str, str, str]:
    route, direction, from_stop = key.split("|", 2)
    return route, direction, from_stop


def _adapt(records: list[MovementTransitionRecord]) -> list[TransitionRecord]:
    """Relabel segment-scope movement transitions as TransitionRecords keyed
    on the segment cell rather than the route, so dwell.dwell_samples_by_cell
    (which groups on `.route`) does the leaf-level split for free. Silently
    drops route-scope records — callers may pass the merged stream."""
    return [
        TransitionRecord(
            ts=r.ts,
            route=r.key,
            prev_state=r.prev_state,
            new_state=r.new_state,
            regime_entered_at=r.regime_entered_at,
            exited_at=r.exited_at,
            dwell_sec=r.dwell_sec,
        )
        for r in records
        if r.scope == "segment"
    ]


def build_segment_dwell(
    transitions: list[MovementTransitionRecord],
    *,
    window_end: int | None = None,
    min_leaf_n: int = MIN_LEAF_N,
) -> tuple[dict[str, dict[str, DwellQuantiles]], SegmentDwellStats]:
    """Per-(segment cell, state) DwellQuantiles, hierarchically pooled
    leaf -> route -> system.

    `window_end` right-censors regimes still open at that boundary (Kaplan-
    Meier), inferred from the last transition per cell — same convention as
    dwell.dwell_samples_by_cell without an explicit `open_regimes` map (there
    is no per-segment prediction stream to source one from).

    Returns ({segment_key: {state: DwellQuantiles}}, stats).
    """
    by_cell = dwell_samples_by_cell(_adapt(transitions), window_end=window_end)

    by_state: dict[str, dict[str, list[DwellSample]]] = defaultdict(dict)
    for (seg_key, state), samples in by_cell.items():
        by_state[state][seg_key] = samples

    out: dict[str, dict[str, DwellQuantiles]] = defaultdict(dict)
    n_own = n_route = n_system = n_skipped = 0

    for state, samples_by_segment in by_state.items():
        leaves: dict[tuple[str, str, str], tuple[int, int]] = {}
        for seg_key, samples in samples_by_segment.items():
            n_events = sum(1 for _duration, completed in samples if completed)
            leaves[_segment_key_parts(seg_key)] = (n_events, len(samples) - n_events)
        levels = partially_pool(leaves, min_leaf_n=min_leaf_n)

        all_samples = [s for ss in samples_by_segment.values() for s in ss]
        system_fit = partially_pooled_dwell({_SYSTEM_KEY: all_samples}).get(_SYSTEM_KEY)

        by_route: dict[str, dict[str, list[DwellSample]]] = defaultdict(dict)
        for seg_key, samples in samples_by_segment.items():
            route, _direction, _from_stop = _segment_key_parts(seg_key)
            by_route[route][seg_key] = samples
        route_fits: dict[str, PooledDwellFit | None] = {}

        for seg_key, samples in samples_by_segment.items():
            route, direction, from_stop = _segment_key_parts(seg_key)
            level = levels[(route, direction, from_stop)]

            if level.n >= min_leaf_n:
                leaf_fit = partially_pooled_dwell({seg_key: samples}).get(seg_key)
                if leaf_fit is None:
                    n_skipped += 1
                    continue
                out[seg_key][state] = cell_from_fit(leaf_fit)
                n_own += 1
                continue

            if level.source in ("route_dir", "route"):
                if route not in route_fits:
                    agg = [s for ss in by_route[route].values() for s in ss]
                    route_fits[route] = partially_pooled_dwell({route: agg}).get(route)
                fit = route_fits[route]
                if fit is not None:
                    out[seg_key][state] = cell_from_fit(fit)
                    n_route += 1
                    continue

            if system_fit is not None:
                out[seg_key][state] = cell_from_fit(system_fit)
                n_system += 1
            else:
                n_skipped += 1

    stats = SegmentDwellStats(
        n_cells_own=n_own,
        n_cells_route=n_route,
        n_cells_system=n_system,
        n_cells_skipped=n_skipped,
    )
    return dict(out), stats
