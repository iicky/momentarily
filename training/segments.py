"""Segment-aware classification and canonical adjacency for the movement leaf.

Two Phase-3 primitives on top of the hierarchical segment baseline
(training.hierarchical / load_r2.build_segment_baseline):

- classify_segment: the segment-aware current-state call. It is the exact
  Beta-Binomial decision the direction-level classifier uses (load_r2.
  classify_direction), applied to a segment leaf's pooled baseline — so a segment
  is judged against its OWN normal advance rate, not the line average.

  Power note: a single 5-minute tick sees ~1 tracked train per segment, far too
  few for the significance gate to fire, so this classifier is meant for
  ACCUMULATED counts (an offline reliability window, or a future rolling per-
  segment worker state), not a raw per-tick call. The direction-level classifier
  stays the live per-tick path.

- canonical_adjacency: the "next stop" for each (route, direction, from_stop),
  taken as the modal to_stop in the observed cross-tick transitions. This gives a
  segment its human identity ("59 St -> 125 St") for localizing a stall, without
  static GTFS: the line ordering is already latent in which stop pairs trains
  actually traverse. `share` (modal count / all advances out of the stop) flags
  express/reroute ambiguity — a clean through-stop is ~1.0, a branch/express point
  lower.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from training.hierarchical import PooledCell
from training.load_r2 import (
    CLASSIFY_ALPHA,
    CLASSIFY_PRIOR_STRENGTH,
    DISRUPTED_RATIO,
    MIN_MATCHED_TRIPS,
    AdvanceBaseline,
    build_segment_series,
    classify_direction,
)


def classify_segment(
    advanced: int,
    stalled: int,
    cell: PooledCell,
    *,
    prior_strength: float = CLASSIFY_PRIOR_STRENGTH,
    disrupted_ratio: float = DISRUPTED_RATIO,
    min_matched: int = MIN_MATCHED_TRIPS,
    alpha: float = CLASSIFY_ALPHA,
) -> str | None:
    """normal / disrupted / None for one segment's accumulated advance vs stall
    counts, judged against its pooled baseline. Reuses classify_direction (which
    reads only the baseline's p0), so the segment and direction calls share one
    decision rule and stay in lockstep."""
    baseline = AdvanceBaseline(p0=cell.p0, n=cell.n, alpha=cell.alpha, beta=cell.beta)
    return classify_direction(
        advanced,
        stalled,
        baseline,
        prior_strength=prior_strength,
        disrupted_ratio=disrupted_ratio,
        min_matched=min_matched,
        alpha=alpha,
    )


@dataclass(frozen=True)
class Adjacency:
    """The canonical next stop out of a from_stop on one (route, direction).

    to_stop is the modal observed successor; share is its fraction of all advances
    out of the stop (1.0 = one clean successor; lower = express/branch ambiguity);
    n is the total advances the estimate rests on.
    """

    to_stop: str
    share: float
    n: int


def canonical_adjacency(
    bodies: list[dict[str, Any]],
    *,
    min_advances: int = MIN_MATCHED_TRIPS,
) -> dict[tuple[str, str, str], Adjacency]:
    """Modal next stop for each (route, direction, from_stop) from the observed
    cross-tick transitions. Stalls (from==to) are ignored; from_stops with fewer
    than min_advances total advances are omitted (too thin to name a successor).
    Ties break on the smaller to_stop id so the mapping is deterministic."""
    counts: dict[tuple[str, str, str], dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for (route, direction, frm, to, _tick), n in build_segment_series(bodies).items():
        if frm != to:
            counts[(route, direction, frm)][to] += n

    out: dict[tuple[str, str, str], Adjacency] = {}
    for key, tos in counts.items():
        total = sum(tos.values())
        if total < min_advances:
            continue
        best = max(tos.values())
        to_stop = min(t for t, c in tos.items() if c == best)  # smallest id on ties
        out[key] = Adjacency(to_stop=to_stop, share=best / total, n=total)
    return out
