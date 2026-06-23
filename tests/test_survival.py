"""Parametric survival fits (Weibull, log-logistic) under right-censoring."""

from __future__ import annotations

import math
import random

from training.dwell import DwellSample
from training.survival import (
    fit_loglogistic,
    fit_weibull,
    loglogistic_loglik,
    loglogistic_survival,
    parametric_curve_sec,
    select_parametric,
    weibull_loglik,
    weibull_survival,
)


def _weibull_samples(
    n: int, shape: float, scale: float, *, seed: int, censor_at: float | None = None
) -> list[DwellSample]:
    rng = random.Random(seed)
    out: list[DwellSample] = []
    for _ in range(n):
        u = rng.random()
        t = scale * (-math.log(1.0 - u)) ** (1.0 / shape)
        if censor_at is not None and t > censor_at:
            out.append((int(censor_at), False))
        else:
            out.append((int(t), True))
    return out


def _loglogistic_samples(
    n: int, shape: float, scale: float, *, seed: int, censor_at: float | None = None
) -> list[DwellSample]:
    rng = random.Random(seed)
    out: list[DwellSample] = []
    for _ in range(n):
        u = rng.random()
        t = scale * (u / (1.0 - u)) ** (1.0 / shape)
        if censor_at is not None and t > censor_at:
            out.append((int(censor_at), False))
        else:
            out.append((int(t), True))
    return out


# --- Survival/quantile invariants ----------------------------------------------


def test_weibull_survival_is_monotone_and_bounded():
    s_prev = weibull_survival(0.0, 1.5, 1800.0)
    assert math.isclose(s_prev, 1.0)
    for t in range(60, 36000, 600):
        s = weibull_survival(float(t), 1.5, 1800.0)
        assert 0.0 <= s <= s_prev
        s_prev = s
    assert s_prev < 0.01  # decays to ~0 in the far tail


def test_loglogistic_survival_is_monotone_and_bounded():
    s_prev = loglogistic_survival(0.0, 2.0, 1800.0)
    assert math.isclose(s_prev, 1.0)
    for t in range(60, 36000, 600):
        s = loglogistic_survival(float(t), 2.0, 1800.0)
        assert 0.0 <= s <= s_prev
        s_prev = s


# --- MLE parameter recovery ----------------------------------------------------


def test_weibull_mle_recovers_known_params():
    samples = _weibull_samples(4000, shape=1.4, scale=1800.0, seed=1)
    fit = fit_weibull(samples)
    assert fit is not None
    assert abs(fit.shape - 1.4) / 1.4 < 0.1
    assert abs(fit.scale - 1800.0) / 1800.0 < 0.1
    assert fit.n_censored == 0


def test_loglogistic_mle_recovers_known_params():
    samples = _loglogistic_samples(4000, shape=2.0, scale=1800.0, seed=2)
    fit = fit_loglogistic(samples)
    assert fit is not None
    assert abs(fit.shape - 2.0) / 2.0 < 0.12
    assert abs(fit.scale - 1800.0) / 1800.0 < 0.12


def test_weibull_mle_handles_right_censoring():
    # ~20% of the mass sits past the censor time for these params.
    samples = _weibull_samples(4000, shape=1.4, scale=1800.0, seed=3, censor_at=3000.0)
    assert any(not c for _t, c in samples)
    fit = fit_weibull(samples)
    assert fit is not None
    assert fit.n_censored > 0
    # Censoring is handled (not treated as events), so the fit stays near truth
    # instead of collapsing the scale toward the censor time.
    assert abs(fit.shape - 1.4) / 1.4 < 0.15
    assert abs(fit.scale - 1800.0) / 1800.0 < 0.15


def test_ignoring_censoring_would_bias_the_scale_down():
    # Same draws, but pretend every censored obs completed at the censor time.
    censored = _weibull_samples(4000, shape=1.4, scale=1800.0, seed=3, censor_at=3000.0)
    naive = [(t, True) for t, _c in censored]
    honest = fit_weibull(censored)
    biased = fit_weibull(naive)
    assert honest is not None
    assert biased is not None
    # Treating censored regimes as if they ended understates dwell.
    assert biased.scale < honest.scale


# --- Likelihood sanity ---------------------------------------------------------


def test_weibull_loglik_peaks_at_the_mle():
    samples = _weibull_samples(2000, shape=1.4, scale=1800.0, seed=4)
    fit = fit_weibull(samples)
    assert fit is not None
    here = weibull_loglik(samples, fit.shape, fit.scale)
    for ds, dl in ((1.3, 1.0), (0.7, 1.0), (1.0, 1.3), (1.0, 0.7)):
        off = weibull_loglik(samples, fit.shape * ds, fit.scale * dl)
        assert off < here


def test_loglogistic_loglik_peaks_at_the_mle():
    samples = _loglogistic_samples(2000, shape=2.0, scale=1800.0, seed=5)
    fit = fit_loglogistic(samples)
    assert fit is not None
    here = loglogistic_loglik(samples, fit.shape, fit.scale)
    for ds, dl in ((1.3, 1.0), (0.7, 1.0), (1.0, 1.3), (1.0, 0.7)):
        off = loglogistic_loglik(samples, fit.shape * ds, fit.scale * dl)
        assert off < here


# --- Goodness of fit / selection ------------------------------------------------


def test_fit_matches_km_on_its_own_data():
    samples = _weibull_samples(3000, shape=1.6, scale=1800.0, seed=6)
    fit = fit_weibull(samples)
    assert fit is not None
    # The fitted curve tracks the empirical KM estimate closely on its own draws.
    assert fit.km_sup_distance < 0.05


def test_aic_selects_the_generating_family_when_tails_differ():
    # k=3 Weibull has a very light (super-exponential) tail; log-logistic always
    # carries a heavier polynomial tail, so AIC should prefer Weibull here.
    wb = _weibull_samples(4000, shape=3.0, scale=1800.0, seed=7)
    best, fits = select_parametric(wb)
    assert best is not None
    assert {f.family for f in fits} == {"weibull", "loglogistic"}
    assert best.family == "weibull"

    # beta=1 log-logistic is heavy-tailed (infinite mean); Weibull's exponential
    # tail fits it poorly, so AIC should prefer log-logistic.
    ll = _loglogistic_samples(4000, shape=1.0, scale=1800.0, seed=8)
    best_ll, _ = select_parametric(ll)
    assert best_ll is not None
    assert best_ll.family == "loglogistic"


def test_no_events_yields_no_fit():
    censored_only: list[DwellSample] = [(1800, False)] * 10
    assert fit_weibull(censored_only) is None
    assert fit_loglogistic(censored_only) is None
    best, fits = select_parametric(censored_only)
    assert best is None
    assert fits == []


# --- Curve emission ------------------------------------------------------------


def test_parametric_curve_sec_is_nondecreasing_and_finite():
    samples = _loglogistic_samples(3000, shape=1.3, scale=1800.0, seed=9)
    fit = fit_loglogistic(samples)
    assert fit is not None
    curve = parametric_curve_sec(fit)
    assert all(isinstance(x, int) for x in curve)
    assert curve == sorted(curve)
    assert curve[0] == 0
    assert math.isfinite(curve[-1])
    # The capped p=1 endpoint stays finite even for this heavy tail.
    assert curve[-1] > curve[-2]
