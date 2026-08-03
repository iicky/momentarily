"""Peer-comparison reliability scorecard (training/reliability.py)."""

from __future__ import annotations

from training.hierarchical import PooledCell
from training.load_r2 import AdvanceBaseline
from training.reliability import (
    advance_deficit,
    direction_reliability,
    segment_reliability,
)
from training.segments import Adjacency


def _ab(p0: float) -> AdvanceBaseline:
    return AdvanceBaseline(p0=p0, n=1000, alpha=30 * p0, beta=30 * (1 - p0))


def _cell(p0: float) -> PooledCell:
    return PooledCell(
        p0=p0, raw=p0, n=1000, alpha=30 * p0, beta=30 * (1 - p0), source="t"
    )


def test_deficit_zero_when_running_at_or_above_normal():
    assert advance_deficit(0.9, 0.9) == 0.0
    assert advance_deficit(0.9, 0.95) == 0.0  # above normal clamps to 0


def test_deficit_positive_and_clamped_when_below_normal():
    assert advance_deficit(0.9, 0.45) == (0.9 - 0.45) / 0.9
    assert advance_deficit(0.9, 0.0) == 1.0
    assert advance_deficit(0.0, 0.0) == 0.0  # degenerate baseline -> no deficit


def test_direction_ranks_by_deficit_not_raw_rate():
    baseline = {("F", "south", 0): _ab(0.9), ("G", "south", 0): _ab(0.9)}
    observed = {("F", "south"): (450, 1000), ("G", "south"): (900, 1000)}
    scores = direction_reliability(baseline, observed)
    assert [s.route for s in scores] == ["F", "G"]  # F degraded, ranked worst first
    assert scores[0].deficit == (0.9 - 0.45) / 0.9
    assert scores[1].deficit == 0.0


def test_direction_terminal_running_normally_is_not_flagged_worst():
    # A structurally-slow direction (p0 low) running at its own normal has ~0
    # deficit, so it does NOT top the leaderboard over a genuinely degraded peer.
    baseline = {("SLOW", "south", 0): _ab(0.10), ("A", "south", 0): _ab(0.95)}
    observed = {("SLOW", "south"): (100, 1000), ("A", "south"): (500, 1000)}
    scores = direction_reliability(baseline, observed)
    assert scores[0].route == "A"  # A degraded (0.95 -> 0.50); SLOW ran at normal
    slow = next(s for s in scores if s.route == "SLOW")
    assert slow.deficit == 0.0


def test_direction_omits_thin_observed():
    baseline = {("F", "south", 0): _ab(0.9)}
    observed = {("F", "south"): (0, 10)}  # matched 10 < MIN_OBS_N
    assert direction_reliability(baseline, observed) == []


def test_segment_ranks_by_deficit_and_labels_next_stop():
    baseline = {
        ("F", "south", "A"): _cell(0.9),
        ("F", "south", "T"): _cell(0.08),  # terminal-like but has a successor
    }
    observed = {
        ("F", "south", "A"): (300, 1000),  # deficit ~0.667
        ("F", "south", "T"): (80, 1000),  # ran at its own normal -> ~0
    }
    adj = {
        ("F", "south", "A"): Adjacency(to_stop="B", share=0.9, n=1000),
        ("F", "south", "T"): Adjacency(to_stop="U", share=0.7, n=1000),
    }
    scores = segment_reliability(baseline, observed, adj)
    assert scores[0].from_stop == "A"
    assert scores[0].to_stop == "B"
    assert scores[0].deficit > 0.6
    term = next(s for s in scores if s.from_stop == "T")
    assert term.deficit == 0.0
    assert term.is_terminal_like is True
    assert term.to_stop == "U"


def test_segment_omits_endpoints_without_a_successor():
    # A from_stop with no canonical next stop is a line terminal — advance is
    # ill-defined there, so it's excluded rather than ranked as a fake chokepoint.
    baseline = {("F", "south", "T"): _cell(0.5)}
    observed = {("F", "south", "T"): (0, 500)}  # would be deficit 1.0 if ranked
    assert segment_reliability(baseline, observed, {}) == []


def test_segment_omits_thin_and_baseline_missing():
    baseline = {("F", "south", "A"): _cell(0.9)}
    observed = {
        ("F", "south", "A"): (0, 20),  # thin (< MIN_OBS_N)
        ("F", "south", "Z"): (0, 500),  # no baseline
    }
    adj = {
        ("F", "south", "A"): Adjacency("B", 1.0, 100),
        ("F", "south", "Z"): Adjacency("Y", 1.0, 100),
    }
    assert segment_reliability(baseline, observed, adj) == []


def test_segment_omits_ambiguous_low_share_successor():
    # A branch/express/reversal point (share below MIN_SHARE) is excluded even
    # though it has a modal successor — its advance semantics are ambiguous.
    baseline = {("F", "south", "A"): _cell(0.9)}
    observed = {("F", "south", "A"): (0, 500)}  # would be deficit 1.0 if ranked
    adj = {("F", "south", "A"): Adjacency(to_stop="B", share=0.33, n=1000)}
    assert segment_reliability(baseline, observed, adj) == []
