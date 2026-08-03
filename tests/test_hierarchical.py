"""Partial-pooling behaviour for the segment advance-rate baseline."""

from __future__ import annotations

from training.hierarchical import (
    KAPPA_MAX,
    KAPPA_MIN,
    MIN_PARENT_LEAVES,
    P0_FLOOR,
    partially_pool,
    robust_concentration,
)
from training.load_r2 import build_segment_baseline

Leaves = dict[tuple[str, str, str], tuple[int, int]]


def _rd_leaves(
    direction: str, rates_ns: list[tuple[float, int]], route: str = "F"
) -> Leaves:
    """Build leaves for one (route, direction), one synthetic stop per rate."""
    leaves: Leaves = {}
    for i, (rate, n) in enumerate(rates_ns):
        adv = round(rate * n)
        leaves[(route, direction, f"S{i:02d}")] = (adv, n - adv)
    return leaves


def test_rich_leaf_keeps_its_own_rate():
    # A leaf with abundant data barely moves off its raw fraction, even when the
    # rest of the line runs faster.
    leaves = _rd_leaves("south", [(0.98, 2000)] * 6 + [(0.30, 5000)])
    pooled = partially_pool(leaves)
    terminal = pooled[("F", "south", "S06")]
    assert abs(terminal.p0 - 0.30) < 0.05
    assert terminal.raw == 0.30


def test_sparse_leaf_pools_toward_parent():
    # A thin leaf (few trials) lands near the line-direction normal, not its own
    # noisy fraction.
    leaves = _rd_leaves("south", [(0.97, 2000)] * 6)
    leaves[("F", "south", "SPARSE")] = (0, 3)  # raw 0.0, n=3
    pooled = partially_pool(leaves)
    cell = pooled[("F", "south", "SPARSE")]
    assert cell.raw == 0.0
    assert cell.p0 > 0.8  # pulled up toward the ~0.97 parent
    assert cell.source == "route_dir"


def test_terminal_low_rate_survives_high_rate_siblings():
    # The terminal dwell case from real data: low rate, huge support, high-rate
    # siblings. It must keep its low rate (this is what makes localization work).
    leaves = _rd_leaves("south", [(0.99, 3000)] * 8)
    leaves[("F", "south", "TERM")] = (2145, 24174)  # ~0.082, n~26k
    pooled = partially_pool(leaves)
    term = pooled[("F", "south", "TERM")]
    assert abs(term.p0 - 0.082) < 0.02


def test_concentration_higher_when_tightly_clustered():
    tight = robust_concentration([0.90, 0.91, 0.90, 0.905, 0.895, 0.90])
    spread = robust_concentration([0.60, 0.95, 0.40, 0.99, 0.75, 0.20])
    assert tight is not None
    assert spread is not None
    assert tight > spread
    assert KAPPA_MIN <= spread <= KAPPA_MAX
    assert KAPPA_MIN <= tight <= KAPPA_MAX


def test_concentration_none_below_min_leaves():
    assert robust_concentration([0.9] * (MIN_PARENT_LEAVES - 1)) is None


def test_backoff_to_route_when_direction_thin():
    # 'north' has only one well-sampled leaf (below MIN_PARENT_LEAVES) so its sparse
    # leaf backs off to the route-level parent, which pools both directions.
    leaves = _rd_leaves("south", [(0.95, 2000)] * 6)
    leaves[("F", "north", "N00")] = (1900, 2000)  # one rich north leaf
    leaves[("F", "north", "NSP")] = (0, 2)  # sparse north leaf
    pooled = partially_pool(leaves)
    assert pooled[("F", "north", "NSP")].source == "route"
    assert pooled[("F", "north", "NSP")].p0 > 0.8


def test_backoff_to_system_when_route_thin():
    # A route with no well-sampled leaves of its own falls through to the system
    # centre, estimated from the other routes' rich leaves.
    leaves: Leaves = {
        (f"R{i}", "south", "X"): (round(0.9 * 2000), 200) for i in range(6)
    }
    leaves[("LONE", "south", "SP")] = (0, 2)
    pooled = partially_pool(leaves)
    assert pooled[("LONE", "south", "SP")].source == "system"


def test_pooled_rate_clipped_off_boundary():
    leaves = _rd_leaves("south", [(1.0, 5000)] * 6)
    pooled = partially_pool(leaves)
    for cell in pooled.values():
        assert P0_FLOOR <= cell.p0 <= 1.0 - P0_FLOOR
        assert cell.alpha > 0
        assert cell.beta > 0


def test_alpha_beta_track_pooled_rate():
    leaves = _rd_leaves("south", [(0.9, 3000)] * 6)
    pooled = partially_pool(leaves, prior_strength=30.0)
    cell = next(iter(pooled.values()))
    assert abs(cell.alpha / (cell.alpha + cell.beta) - cell.p0) < 1e-9


def test_build_segment_baseline_aggregates_advance_and_stall():
    # from!=to counts as an advance out of from_stop; from==to counts as a stall.
    body = {
        "observed_at": 1_700_000_100,
        "rows": {
            "F": {
                "by_direction": {
                    "south": {"transitions": {"A>B": 5, "A>A": 3, "B>C": 2}}
                }
            }
        },
    }
    cells = build_segment_baseline([body])
    a = cells[("F", "south", "A")]
    assert a.n == 8  # 5 advanced (A>B) + 3 stalled (A>A)
    assert a.raw == 5 / 8
    b = cells[("F", "south", "B")]
    assert b.n == 2  # 2 advanced (B>C), no stall
    assert b.raw == 1.0
