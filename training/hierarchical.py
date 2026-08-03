"""Hierarchical partial-pooling for the segment-level advance-rate baseline.

The flat movement baseline (load_r2.compute_advance_baseline) estimates one normal
advance rate per (route, direction, tod_bin). The segment leaf pushes that finer —
one rate per (route, direction, from_stop) — so a stall can be localized to a place
on the line ("between 59 St and 125 St") instead of averaged across the whole route.

Segment leaves are sparse (about one tracked train per leaf per tick), so a raw
per-leaf fraction is far too noisy to serve. This module shrinks each leaf toward a
robust parent rate via an empirical-Bayes Beta-Binomial: the leaf's own data
dominates when it is plentiful (a terminal with tens of thousands of dwell samples
keeps its true low rate), and a thin leaf borrows its line-direction's normal rate.

The pooling strength (concentration) is estimated per parent from the spread of its
well-sampled children, using a robust centre (median) and spread (MAD) so a handful
of structurally different stops — terminals, transfer points — don't distort the
estimate for the ordinary through-stops that actually need pooling.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping
from dataclasses import dataclass

# A leaf needs at least this many matched trials to contribute to a parent's centre
# and concentration estimate; below it the leaf is pooled but does not vote.
MIN_LEAF_N = 20

# A parent needs at least this many voting leaves to estimate its own centre and
# concentration; below it, it backs off to the next level up.
MIN_PARENT_LEAVES = 4

# Concentration (pseudo-trials of prior) is clamped to this range. The floor keeps a
# genuinely dispersed parent from disabling pooling on its sparse leaves; the ceiling
# keeps a tight parent from overwhelming a leaf that does have real data.
KAPPA_MIN = 10.0
KAPPA_MAX = 300.0
# Default concentration when a parent can't estimate its own (too few leaves).
KAPPA_DEFAULT = 50.0

# Rates are held off the 0/1 boundary so a Beta prior stays proper.
P0_FLOOR = 1e-3

# MAD -> standard-deviation scale factor for a normal distribution.
_MAD_TO_STD = 1.4826


@dataclass(frozen=True)
class PooledCell:
    """A partially-pooled advance-rate cell for one segment leaf.

    p0 is the pooled normal advance rate the Worker scores a live tick against;
    raw/n expose the unpooled fraction and its support for transparency; alpha/beta
    carry p0 as a Beta prior (matching load_r2.AdvanceBaseline); source records which
    level supplied the prior mean (self when the leaf pooled toward its own rate).
    """

    p0: float
    raw: float
    n: int
    alpha: float
    beta: float
    source: str


def _clip01(p: float) -> float:
    return min(max(p, P0_FLOOR), 1.0 - P0_FLOOR)


def robust_concentration(rates: list[float]) -> float | None:
    """Empirical-Bayes concentration from a parent's well-sampled child rates.

    Models the child rates as Beta(mu*kappa, (1-mu)*kappa) and inverts the Beta
    mean/variance identity kappa = mu(1-mu)/var - 1. Uses a robust centre (median)
    and spread (scaled MAD) so a minority of structurally different children —
    terminals that dwell, transfer points — inflates neither, leaving a
    concentration that reflects how tightly the ordinary children cluster. Returns
    None when there are too few children to estimate a spread.
    """
    if len(rates) < MIN_PARENT_LEAVES:
        return None
    mu = statistics.median(rates)
    mad = statistics.median([abs(r - mu) for r in rates])
    var = (mad * _MAD_TO_STD) ** 2
    spread = mu * (1.0 - mu)
    if var <= 0.0:
        # All well-sampled children agree — pool hard, but not unboundedly.
        return KAPPA_MAX
    if var >= spread:
        # Children more dispersed than a point-mass Beta allows — pool weakly.
        return KAPPA_MIN
    kappa = spread / var - 1.0
    return min(max(kappa, KAPPA_MIN), KAPPA_MAX)


def _parent_rate(rates: list[float]) -> float | None:
    """Robust centre of a parent's well-sampled child rates, or None below support."""
    if len(rates) < MIN_PARENT_LEAVES:
        return None
    return statistics.median(rates)


def partially_pool(
    leaves: Mapping[tuple[str, str, str], tuple[int, int]],
    *,
    prior_strength: float = 30.0,
    min_leaf_n: int = MIN_LEAF_N,
) -> dict[tuple[str, str, str], PooledCell]:
    """Partially pool per-leaf advance rates up a (route, direction, from_stop)
    hierarchy: leaf -> (route, direction) -> route -> system.

    leaves maps (route, direction, from_stop) -> (advanced, stalled). Each leaf's
    pooled rate is (kappa*mu_parent + advanced) / (kappa + matched), where mu_parent
    is the nearest level with enough well-sampled children and kappa is that level's
    estimated concentration. A data-rich leaf barely moves off its own rate; a thin
    leaf lands near its parent's normal. prior_strength anchors the emitted Beta
    prior (alpha/beta) at the pooled rate, matching the flat baseline's convention.
    """
    # Well-sampled child rates grouped at each hierarchy level.
    rd_rates: dict[tuple[str, str], list[float]] = {}
    route_rates: dict[str, list[float]] = {}
    system_rates: list[float] = []
    for (route, direction, _stop), (adv, stall) in leaves.items():
        n = adv + stall
        if n < min_leaf_n:
            continue
        rate = adv / n
        rd_rates.setdefault((route, direction), []).append(rate)
        route_rates.setdefault(route, []).append(rate)
        system_rates.append(rate)

    system_mu = statistics.median(system_rates) if system_rates else 0.5
    system_kappa = robust_concentration(system_rates) or KAPPA_DEFAULT

    out: dict[tuple[str, str, str], PooledCell] = {}
    for (route, direction, stop), (adv, stall) in leaves.items():
        n = adv + stall
        raw = adv / n if n > 0 else system_mu

        # Nearest parent level with enough support supplies the prior mean + kappa.
        rd_rate = _parent_rate(rd_rates.get((route, direction), []))
        if rd_rate is not None:
            mu = rd_rate
            kappa = robust_concentration(rd_rates[(route, direction)]) or KAPPA_DEFAULT
            source = "route_dir"
        elif (r := _parent_rate(route_rates.get(route, []))) is not None:
            mu = r
            kappa = robust_concentration(route_rates[route]) or KAPPA_DEFAULT
            source = "route"
        else:
            mu = system_mu
            kappa = system_kappa
            source = "system"

        pooled = _clip01((kappa * mu + adv) / (kappa + n)) if n > 0 else _clip01(mu)
        out[(route, direction, stop)] = PooledCell(
            p0=pooled,
            raw=raw,
            n=n,
            alpha=prior_strength * pooled,
            beta=prior_strength * (1.0 - pooled),
            source=source,
        )
    return out
