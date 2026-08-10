"""Peer-comparison reliability scorecard from the movement baselines.

Descriptive, not predictive. It answers "which line-directions and which places on
a line ran worse than they normally do" — the reliability question — rather than
"which places are slowest", which just rediscovers terminals and shuttle endpoints
(structurally low advance rates that are perfectly normal for them).

So the score is a held-out ADVANCE DEFICIT, not the raw rate: compare each cell's
observed advance rate over a recent window against its own baseline (fit on an
earlier, causal window). A terminal that dwells as usual scores ~0 deficit; a
through-segment that froze below its own normal scores high. p0 (the structural
normal rate) rides along as context, never as the ranking key.

Two views, both terminal-robust:
  - direction_reliability: (route, direction) ranked by aggregate deficit.
  - segment_reliability: individual segments, labelled with the canonical next
    stop, ranked by deficit — the chokepoints that degraded recently.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from training.hierarchical import PooledCell
from training.load_r2 import AdvanceBaseline
from training.segments import Adjacency

# Observed matched trips a cell needs over the recent window to be ranked; thinner
# cells are omitted so a couple of samples can't top the leaderboard.
MIN_OBS_N = 50

# Baseline advance rate at or below this is a structural dwell/terminal point;
# flagged for context (the deficit already keeps it off the top on its own).
TERMINAL_P0 = 0.15

# A through-segment needs a dominant canonical successor (>= this share) to be
# ranked. Below it the from_stop is a branch/express/reversal point where advance
# to "the" next stop is ambiguous, so the deficit would be an artifact.
MIN_SHARE = 0.5


def advance_deficit(p0: float, observed_rate: float) -> float:
    """Relative shortfall of observed advance vs the cell's own normal, in [0, 1].
    0 = ran at or above normal; 1 = fully frozen relative to normal. Terminal-safe:
    a low-p0 cell running at its normal low rate scores ~0."""
    if p0 <= 0:
        return 0.0
    return max(0.0, min(1.0, (p0 - observed_rate) / p0))


@dataclass(frozen=True)
class DirectionScore:
    route: str
    direction: str
    deficit: float  # ranking key: held-out shortfall vs own normal
    p0: float  # baseline (normal) advance rate, context
    observed_rate: float
    n: int  # observed matched trips
    percentile: float  # 0=best (no deficit), 1=worst, among ranked peers


@dataclass(frozen=True)
class SegmentScore:
    route: str
    direction: str
    from_stop: str
    to_stop: str
    deficit: float
    p0: float
    observed_rate: float
    n: int
    share: float
    is_terminal_like: bool
    percentile: float


def _percentiles(values: list[float]) -> dict[int, float]:
    """Index -> percentile by ascending value (0=lowest). Ties share the mean rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    pct: dict[int, float] = {}
    denom = max(len(values) - 1, 1)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0
        for k in range(i, j + 1):
            pct[order[k]] = avg_rank / denom
        i = j + 1
    return pct


def _observed_rate(counts: tuple[int, int]) -> tuple[float, int]:
    advanced, matched = counts
    return (advanced / matched if matched else 0.0), matched


def direction_reliability(
    baseline: Mapping[tuple[str, str, int], AdvanceBaseline],
    observed: Mapping[tuple[str, str], tuple[int, int]],
    *,
    min_n: int = MIN_OBS_N,
) -> list[DirectionScore]:
    """(route, direction) leaderboard by held-out advance deficit. `baseline` is the
    causal per-(route,direction,tod) baseline; `observed` is (advanced, matched)
    aggregated over the recent window per (route, direction)."""
    # Collapse the causal baseline to one n-weighted p0 per (route, direction).
    agg: dict[tuple[str, str], tuple[float, int]] = {}
    for (route, direction, _tod), cell in baseline.items():
        num, den = agg.get((route, direction), (0.0, 0))
        agg[(route, direction)] = (num + cell.p0 * cell.n, den + cell.n)

    rows: list[tuple[str, str, float, float, float, int]] = []
    for (route, direction), obs in observed.items():
        base = agg.get((route, direction))
        if base is None or base[1] == 0:
            continue
        rate, n = _observed_rate(obs)
        if n < min_n:
            continue
        p0 = base[0] / base[1]
        rows.append((route, direction, advance_deficit(p0, rate), p0, rate, n))

    pct = _percentiles([r[2] for r in rows])
    scores = [
        DirectionScore(route, direction, deficit, p0, rate, n, pct[i])
        for i, (route, direction, deficit, p0, rate, n) in enumerate(rows)
    ]
    scores.sort(key=lambda s: s.deficit, reverse=True)  # worst deficit first
    return scores


def segment_reliability(
    baseline: Mapping[tuple[str, str, str], PooledCell],
    observed: Mapping[tuple[str, str, str], tuple[int, int]],
    adjacency: Mapping[tuple[str, str, str], Adjacency],
    *,
    min_n: int = MIN_OBS_N,
) -> list[SegmentScore]:
    """Per-segment leaderboard by held-out advance deficit against the causal pooled
    baseline. `observed` is (advanced, matched) per (route, direction, from_stop)
    over the recent window."""
    rows: list[
        tuple[tuple[str, str, str], PooledCell, Adjacency, float, float, int]
    ] = []
    for key, obs in observed.items():
        cell = baseline.get(key)
        adj = adjacency.get(key)
        # Only clean through-segments are rankable: skip a from_stop with no
        # canonical next stop (a terminal, advance is ill-defined) or an ambiguous
        # one (branch/express/reversal, share below MIN_SHARE) — either would rank
        # an artifact rather than a real chokepoint.
        if cell is None or adj is None or adj.share < MIN_SHARE:
            continue
        rate, n = _observed_rate(obs)
        if n < min_n:
            continue
        rows.append((key, cell, adj, advance_deficit(cell.p0, rate), rate, n))

    pct = _percentiles([r[3] for r in rows])
    scores: list[SegmentScore] = []
    for i, ((route, direction, frm), cell, adj, deficit, rate, n) in enumerate(rows):
        scores.append(
            SegmentScore(
                route=route,
                direction=direction,
                from_stop=frm,
                to_stop=adj.to_stop,
                deficit=deficit,
                p0=cell.p0,
                observed_rate=rate,
                n=n,
                share=adj.share,
                is_terminal_like=cell.p0 <= TERMINAL_P0,
                percentile=pct[i],
            )
        )
    scores.sort(key=lambda s: s.deficit, reverse=True)
    return scores
