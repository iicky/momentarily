"""Segment-aware classifier and canonical adjacency (training/segments.py)."""

from __future__ import annotations

from training.hierarchical import PooledCell
from training.segments import Adjacency, canonical_adjacency, classify_segment


def _cell(p0: float) -> PooledCell:
    return PooledCell(
        p0=p0, raw=p0, n=1000, alpha=30 * p0, beta=30 * (1 - p0), source="test"
    )


def test_classify_segment_normal_when_advancing():
    assert classify_segment(18, 2, _cell(0.9)) == "normal"


def test_classify_segment_disrupted_when_frozen_and_significant():
    assert classify_segment(0, 20, _cell(0.9)) == "disrupted"


def test_classify_segment_abstains_below_min_matched():
    assert classify_segment(1, 1, _cell(0.9)) is None


def test_classify_segment_abstains_on_degenerate_baseline():
    # Low baseline + thin sample: the drop isn't significant against a stop that
    # barely advances when healthy, so abstain rather than false-flag.
    assert classify_segment(0, 8, _cell(0.3)) is None


def _body(transitions: dict[str, int], route: str = "F", direction: str = "south"):
    return {
        "observed_at": 1_700_000_100,
        "rows": {route: {"by_direction": {direction: {"transitions": transitions}}}},
    }


def test_canonical_adjacency_picks_modal_successor():
    adj = canonical_adjacency([_body({"A>B": 10, "A>C": 2, "A>A": 5, "B>C": 4})])
    a = adj[("F", "south", "A")]
    assert a == Adjacency(to_stop="B", share=10 / 12, n=12)  # stall A>A ignored
    assert adj[("F", "south", "B")] == Adjacency(to_stop="C", share=1.0, n=4)


def test_canonical_adjacency_omits_thin_from_stops():
    # 'A' has only 2 advances (< MIN_MATCHED_TRIPS=3) -> no named successor.
    adj = canonical_adjacency([_body({"A>B": 2})])
    assert ("F", "south", "A") not in adj


def test_canonical_adjacency_ignores_stall_only_stops():
    adj = canonical_adjacency([_body({"Q>Q": 9})])
    assert adj == {}


def test_canonical_adjacency_tie_breaks_on_smaller_stop_id():
    adj = canonical_adjacency([_body({"X>Z": 3, "X>Y": 3})])
    assert adj[("F", "south", "X")].to_stop == "Y"  # smaller id wins the tie
    assert adj[("F", "south", "X")].share == 0.5


def test_canonical_adjacency_sums_across_ticks():
    bodies = [_body({"A>B": 5}), _body({"A>B": 7, "A>C": 1})]
    a = canonical_adjacency(bodies)[("F", "south", "A")]
    assert a.n == 13  # 5 + 7 + 1
    assert a.to_stop == "B"
    assert a.share == 12 / 13
