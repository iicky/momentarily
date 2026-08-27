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

- canonical_adjacency: the modal to_stop for each (route, direction, from_stop)
  in the observed cross-tick transitions. training.gtfs_static.load_successors
  (the static timetable) is the primary segment topology source now — this is
  the fallback for when that fetch fails, and elsewhere the `share` (modal
  count / all advances out of the stop) rides along as a reliability
  annotation: a clean through-stop is ~1.0, a branch/express point lower.
"""

from __future__ import annotations

import math
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


# --- Throughput branch -----------------------------------------------------
#
# Mirrors worker/src/segment_flow.ts. classify_segment above answers "of the
# trains that were here, did they move on" and needs accumulated counts to say
# anything; this branch answers "did the trains the timetable promised show up",
# which an empty window answers on its own. Keep the two files in lockstep —
# tests/fixtures/parity_segment_flow.json pins them.

# Poisson lower-tail threshold, deliberately the same number as CLASSIFY_ALPHA
# so both branches call a cell disrupted at one nominal strictness.
THROUGHPUT_ALPHA = 0.05

# Expected effective traversals a window needs before absence can be judged at
# all. Below -ln(THROUGHPUT_ALPHA) even a completely empty window sits above the
# tail threshold, so the test provably has no power there.
QUIET_MAX_EXPECTED = -math.log(THROUGHPUT_ALPHA)


def pois_lower_tail(k: int, mu: float) -> float:
    """P(X <= k) for X ~ Poisson(mu) via an iterative pmf sum, sibling of
    _binom_lower_tail and mirroring the Worker's poisLowerTail 1:1. mu here is
    expected traversals out of ONE stop on ONE route-direction over a ~25-minute
    window, bounded by physical headway at well under 50, so exp(-mu) never
    approaches underflow."""
    if k < 0:
        return 0.0
    if mu <= 0:
        return 1.0
    pmf = math.exp(-mu)
    cdf = pmf
    for i in range(1, k + 1):
        pmf *= mu / i
        cdf += pmf
    return min(1.0, cdf)


def classify_throughput(
    matched: float,
    expected: float,
    route_seen: bool,
    *,
    eff_count_scale: float,
    alpha: float = THROUGHPUT_ALPHA,
    quiet_max_expected: float = QUIET_MAX_EXPECTED,
) -> str | None:
    """quiet / normal / disrupted / None for one segment's decayed window,
    judged against the decayed expectation over that same window.

      quiet     — the window expected less than `quiet_max_expected` traversals,
                  so no observation could reach the tail. Normal for now, by
                  timetable, and saying so beats abstaining.
      None      — the expectation is real but the vehicle feed said nothing about
                  this route at all, so the silence is unattributable.
      disrupted — Poisson lower tail at or under `alpha`.
      normal    — enough of the promised trains arrived.

    `eff_count_scale` converts the decayed sums onto the effective-Poisson
    scale; it is a property of the accumulator's decay, so the caller that owns
    the accumulator owns it (training.segment_replay.EFF_COUNT_SCALE for the
    Worker's EWMA).

    The effective count FLOORS: it is "how many complete traversals the window
    can account for", and rounding a fractional decayed remnant up to a whole
    traversal credits evidence never observed. Flooring also makes the call
    monotone in `matched`, which matters at the edge of the quiet band.
    """
    mu = expected * eff_count_scale
    if mu < quiet_max_expected:
        return "quiet"
    if not route_seen:
        return None
    k = math.floor(matched * eff_count_scale)
    return "disrupted" if pois_lower_tail(k, mu) <= alpha else "normal"


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
