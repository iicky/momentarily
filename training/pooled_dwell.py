"""Partially-pooled normal-regime dwell.

The empirical dwell cells in dwell.py gate on completed exits: a (route, state)
cell needs MIN_SAMPLES_FOR_EMPIRICAL of them or the Worker falls back to a
geometric projection off the transition self-loop. For the disrupted states that
gate is nearly always met — disruptions end. For `normal` it is a trap, because
a route only accumulates completed normal regimes by *leaving* normal. The gate
therefore selects for exactly the routes whose normal regimes are shortest.

Measured over a 28-day transition sample:

  * Spearman(completed exits, median normal dwell) = -0.81.
  * Routes clearing the gate ran a median 8.9h between exits; routes failing it
    ran 119h.
  * An omnibus log-rank test rejects a shared hazard across routes outright
    (chi2 = 57.5, df = 23, p = 8.7e-05).

So neither escape works. Falling back to geometric leaves the steadiest routes
on a memoryless projection with no elapsed term. Pooling completed events into
one system-wide curve weights each route by how often it flips, importing the
flappiest routes' hazard onto the steadiest ones — and the log-rank test says a
single shared curve is wrong regardless of how it is weighted.

What works is pooling at the route level with the censored observations doing
real work: a shared-shape log-logistic AFT with a per-route scale. Each route's
log-scale is a MAP estimate under its own right-censored likelihood plus a
Normal prior centred on the population median. A route that has been normal for
625h and counting lands long off that censored observation alone, rather than
collapsing to the population centre, while a route with 29 completed exits
barely moves off its own MLE.

Shape is shared because it is not identifiable from one or two observations, and
the scale carries the between-route variation the log-rank test detected.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from training.dwell import (
    CURVE_POINTS,
    DwellQuantiles,
    DwellSample,
    OpenRegimes,
    RegimeTransition,
    dwell_samples_by_cell,
)
from training.survival import (
    ParametricFit,
    fit_loglogistic,
    km_sup_distance,
    loglogistic_loglik,
    loglogistic_quantile,
    loglogistic_survival,
    parametric_curve_sec,
)

# A route needs this many completed exits before its own scale estimate is
# trusted enough to vote on the population centre. Voting and being pooled are
# separate: every route gets a fitted cell, only well-observed ones set the prior.
MIN_VOTER_EVENTS = 3

# Prior spread on log(scale), as a robust MAD-derived standard deviation of the
# voters' log-scales.
#
# The floor matters more than it looks. A MAD taken over a handful of voters is
# a noisy and downward-biased estimate of the true between-route spread, and it
# collapses to zero outright when the voters happen to agree. Too tight a floor
# does not "keep some pooling" — it pins every sparse route to the median and
# throws away exactly the censored evidence this estimator exists to use. A
# 28-day sample measured the real spread at ~1.5 log units, so a floor of 1.0
# stays below anything observed (it never binds on real data) while still
# representing genuine ignorance about a route we have never seen leave normal.
#
# The ceiling keeps a dispersed population from making the prior so weak that a
# single observation runs away with the fit.
TAU_FLOOR = 1.0
TAU_CEIL = 2.0

# Search bounds for log(scale): one minute to ninety days. Wide enough that the
# prior, not the bound, is what stops a heavily censored route.
_LOG_SCALE_LO = math.log(60.0)
_LOG_SCALE_HI = math.log(90.0 * 86400.0)

# Golden-section tolerance in log units; 1e-6 is far below the resolution any
# downstream horizon can distinguish.
_TOL = 1e-6

_INV_PHI = (math.sqrt(5.0) - 1.0) / 2.0

# MAD -> standard-deviation scale factor for a normal distribution.
_MAD_TO_STD = 1.4826


@dataclass(frozen=True)
class PooledDwellFit:
    """One route's partially-pooled dwell fit, with the provenance to audit it.

    `scale_sec` is the route's own MAP scale; `parent_scale_sec` is the
    population centre it was shrunk toward. Comparing the two shows how much the
    route's own data moved it, which is the whole question this estimator exists
    to answer.
    """

    route: str
    fit: ParametricFit
    n_events: int
    n_censored: int
    parent_scale_sec: float
    # "own" when the route cleared MIN_VOTER_EVENTS and its likelihood dominates,
    # "pooled" when the prior is carrying the estimate.
    source: str

    @property
    def scale_sec(self) -> float:
        return self.fit.scale


def _golden_section_max(
    objective: Callable[[float], float],
    lo: float,
    hi: float,
) -> float:
    """Argmax of a unimodal 1-D objective by golden-section search.

    Deterministic and derivative-free. The penalised log-likelihood below is
    smooth and unimodal in log(scale) for a fixed shape, so bracketing is safe.
    """
    a, b = lo, hi
    c, d = b - _INV_PHI * (b - a), a + _INV_PHI * (b - a)
    fc, fd = objective(c), objective(d)
    while b - a > _TOL:
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - _INV_PHI * (b - a)
            fc = objective(c)
        else:
            a, c, fc = c, d, fd
            d = a + _INV_PHI * (b - a)
            fd = objective(d)
    return (a + b) / 2.0


def _mle_log_scale(samples: list[DwellSample], shape: float) -> float:
    """Unpenalised MLE of log(scale) at fixed shape. Runs to the upper bound for
    a route with no completed exits — which is precisely why the MAP variant
    below is what we publish."""
    return _golden_section_max(
        lambda ls: loglogistic_loglik(samples, shape, math.exp(ls)),
        _LOG_SCALE_LO,
        _LOG_SCALE_HI,
    )


def _map_log_scale(
    samples: list[DwellSample], shape: float, parent_ls: float, tau: float
) -> float:
    """MAP log(scale): the route's right-censored log-likelihood plus a Normal
    log-prior centred on the population log-scale."""
    return _golden_section_max(
        lambda ls: (
            loglogistic_loglik(samples, shape, math.exp(ls))
            - 0.5 * ((ls - parent_ls) / tau) ** 2
        ),
        _LOG_SCALE_LO,
        _LOG_SCALE_HI,
    )


def partially_pooled_dwell(
    samples_by_route: dict[str, list[DwellSample]],
    *,
    min_voter_events: int = MIN_VOTER_EVENTS,
) -> dict[str, PooledDwellFit]:
    """Fit a shared-shape log-logistic with a per-route partially-pooled scale.

    Returns one fit per route that has any observation at all, censored ones
    included. Empty when the input carries no samples.
    """
    pooled = [s for samples in samples_by_route.values() for s in samples]
    if not pooled:
        return {}

    # Shape is estimated once on everything. A per-route shape is unidentifiable
    # at one or two observations, and the log-rank test located the between-route
    # difference in level, not curvature.
    shared = fit_loglogistic(pooled)
    if shared is None:
        return {}
    shape = shared.shape

    leaf_ls = {r: _mle_log_scale(s, shape) for r, s in samples_by_route.items()}
    n_events = {
        r: sum(1 for _d, completed in s if completed)
        for r, s in samples_by_route.items()
    }

    # One vote per route, not per event: the population centre must not be
    # dragged toward whichever routes happen to churn most.
    voters = [leaf_ls[r] for r in samples_by_route if n_events[r] >= min_voter_events]
    if len(voters) >= 2:
        parent_ls = statistics.median(voters)
        mad = statistics.median([abs(v - parent_ls) for v in voters]) * _MAD_TO_STD
        tau = min(max(mad, TAU_FLOOR), TAU_CEIL)
    else:
        # Too few well-observed routes to estimate a population spread. Centre on
        # the pooled fit and use the widest admissible prior, so each route's own
        # data (including its censoring) still dominates.
        parent_ls = math.log(shared.scale)
        tau = TAU_CEIL

    out: dict[str, PooledDwellFit] = {}
    for route, samples in samples_by_route.items():
        ls = _map_log_scale(samples, shape, parent_ls, tau)
        scale = math.exp(ls)
        loglik = loglogistic_loglik(samples, shape, scale)
        n_ev = n_events[route]
        n_cens = len(samples) - n_ev
        out[route] = PooledDwellFit(
            route=route,
            fit=ParametricFit(
                family="loglogistic",
                shape=shape,
                scale=scale,
                n_events=n_ev,
                n_censored=n_cens,
                loglik=loglik,
                # Shape is shared and scale is shrunk, so this route spends about
                # one free parameter, not two.
                aic=2.0 - 2.0 * loglik,
                km_sup_distance=km_sup_distance(
                    samples,
                    lambda t, _s=scale: loglogistic_survival(t, shape, _s),
                ),
            ),
            n_events=n_ev,
            n_censored=n_cens,
            parent_scale_sec=math.exp(parent_ls),
            source="own" if n_ev >= min_voter_events else "pooled",
        )
    return out


def cell_from_fit(pooled: PooledDwellFit) -> DwellQuantiles:
    """Render a fit into the DwellQuantiles shape the Worker already consumes.

    Same contract as dwell._make_cell: a curve plus a log-logistic tail. The body
    is parametric here rather than Kaplan-Meier, because for these cells the KM
    body is built from too few events to be worth preserving.
    """
    fit = pooled.fit

    def cdf_at(t: float) -> float:
        return 1.0 - loglogistic_survival(t, fit.shape, fit.scale)

    # n stays completed-exits-only, matching dwell._make_cell. It is the key the
    # min-samples floor reads, and these cells deliberately bypass that floor: a
    # route whose normal regime never ended publishes n=0, n_censored=1. That is
    # the case this module exists for, not an empty cell — don't re-gate on n.
    return DwellQuantiles(
        n=pooled.n_events,
        n_censored=pooled.n_censored,
        q25_sec=round(loglogistic_quantile(0.25, fit.shape, fit.scale)),
        median_sec=round(loglogistic_quantile(0.50, fit.shape, fit.scale)),
        q75_sec=round(loglogistic_quantile(0.75, fit.shape, fit.scale)),
        recover_by_30=cdf_at(1800),
        recover_by_60=cdf_at(3600),
        recover_by_120=cdf_at(7200),
        curve_sec=parametric_curve_sec(fit, CURVE_POINTS),
        tail_ll=[fit.shape, fit.scale],
    )


def pooled_dwell_cells(
    transitions: Sequence[RegimeTransition],
    *,
    state: str,
    window_end: int | None = None,
    min_voter_events: int = MIN_VOTER_EVENTS,
    open_regimes: OpenRegimes | None = None,
) -> dict[str, DwellQuantiles]:
    """Partially-pooled {route: DwellQuantiles} for one regime state.

    Unlike compute_dwell_quantiles there is no min-samples gate: every route with
    an observation gets a cell, because for `normal` the gate is what produced
    the bias in the first place. With `window_end`, each route's still-open
    regime joins its cell as a right-censored observation — the load-bearing
    input for a route that has not left normal inside the window.

    Without `open_regimes` that input is inferred from transition records, so a
    route with no transitions in the window supplies nothing and never reaches
    the pooling step at all — the steadiest routes, which are the ones this
    estimator exists to serve. Pass the prediction-derived map to cover them.
    """
    by_cell = dwell_samples_by_cell(
        transitions, window_end=window_end, open_regimes=open_regimes
    )
    samples_by_route = {r: s for (r, st), s in by_cell.items() if st == state}
    fits = partially_pooled_dwell(samples_by_route, min_voter_events=min_voter_events)
    return {route: cell_from_fit(f) for route, f in fits.items()}
