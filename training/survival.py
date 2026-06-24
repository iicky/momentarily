"""Parametric survival fits (Weibull, log-logistic) for disrupted dwell.

The empirical KM curve (training/dwell.py) resolves the heavy upper tail only as
a single linear segment between the 95th percentile and the largest observed
dwell — exactly the region that governs long-disruption recovery. A parametric
model gives a smooth, monotone tail there and extrapolates past the largest
observed dwell without the curve's coarse exponential patch.

This module fits both families under right-censoring by maximum likelihood,
scores each against the nonparametric KM estimate (a KS-style supremum distance),
and selects by AIC. Pure stdlib to match dwell.py; no scipy.

Weibull MLE uses the closed-form profile: the scale solves in one step given the
shape, and the shape is a 1-D root of the profile score. Log-logistic has no such
reduction, so it goes through a small Nelder-Mead simplex on the log-parameters.

It also carries a censored k-group log-rank test, for asking whether the dwell
hazard genuinely differs across strata (tod_bin, route, alert_type) before
keying curves on a covariate.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

from training.dwell import CURVE_POINTS, DwellSample, km_cdf_points

# Durations are seconds; floor at 1 so ln(t) is finite for an instant regime.
_MIN_DURATION = 1.0


@dataclass(frozen=True)
class ParametricFit:
    """An MLE fit of one survival family to a set of (duration, completed) dwells.

    `shape`/`scale` parameterize the family (Weibull: k, lambda; log-logistic:
    beta, alpha). `km_sup_distance` is sup_t |S_param(t) - S_km(t)| over the KM
    event times — a goodness-of-fit measure against the empirical curve. `aic`
    ranks families (lower is better); both families have 2 parameters, so AIC
    ordering is just the log-likelihood ordering, but it's reported in the
    conventional scale.
    """

    family: str  # "weibull" | "loglogistic"
    shape: float
    scale: float
    loglik: float
    n_events: int
    n_censored: int
    aic: float
    km_sup_distance: float


# --- Survival / quantile functions ---------------------------------------------


def weibull_survival(t: float, shape: float, scale: float) -> float:
    """S(t) = exp(-(t/scale)^shape)."""
    if t <= 0:
        return 1.0
    return math.exp(-((t / scale) ** shape))


def loglogistic_survival(t: float, shape: float, scale: float) -> float:
    """S(t) = 1 / (1 + (t/scale)^shape)."""
    if t <= 0:
        return 1.0
    return 1.0 / (1.0 + (t / scale) ** shape)


def weibull_quantile(p: float, shape: float, scale: float) -> float:
    """Inverse CDF: smallest t with F(t) = p. Diverges as p -> 1."""
    p = min(max(p, 0.0), 1.0 - 1e-12)
    return scale * (-math.log(1.0 - p)) ** (1.0 / shape)


def loglogistic_quantile(p: float, shape: float, scale: float) -> float:
    """Inverse CDF: t = scale * (p/(1-p))^(1/shape). Diverges as p -> 1."""
    p = min(max(p, 0.0), 1.0 - 1e-12)
    return scale * (p / (1.0 - p)) ** (1.0 / shape)


# --- Log-likelihoods (right-censored) ------------------------------------------


def weibull_loglik(samples: list[DwellSample], shape: float, scale: float) -> float:
    """Right-censored Weibull log-likelihood. Events contribute log f(t);
    censored observations contribute log S(t)."""
    if shape <= 0 or scale <= 0:
        return -math.inf
    ll = 0.0
    for raw_t, completed in samples:
        t = max(float(raw_t), _MIN_DURATION)
        z = (t / scale) ** shape
        if completed:
            ll += math.log(shape) - math.log(scale) + (shape - 1.0) * math.log(t / scale) - z
        else:
            ll -= z
    return ll


def loglogistic_loglik(samples: list[DwellSample], shape: float, scale: float) -> float:
    """Right-censored log-logistic log-likelihood."""
    if shape <= 0 or scale <= 0:
        return -math.inf
    ll = 0.0
    for raw_t, completed in samples:
        t = max(float(raw_t), _MIN_DURATION)
        z = (t / scale) ** shape
        if completed:
            # log f = log(shape) + log z - log t - 2 log(1+z)
            ll += math.log(shape) + math.log(z) - math.log(t) - 2.0 * math.log1p(z)
        else:
            ll -= math.log1p(z)
    return ll


# --- Fitting -------------------------------------------------------------------


def _km_sup_distance(
    samples: list[DwellSample], survival: Callable[[float], float]
) -> float:
    """sup_t |S_param(t) - S_km(t)| over the KM event times."""
    points = km_cdf_points(samples)
    worst = 0.0
    for t, f_km in points:
        s_km = 1.0 - f_km
        worst = max(worst, abs(survival(float(t)) - s_km))
    return worst


def fit_weibull(samples: list[DwellSample]) -> ParametricFit | None:
    """MLE Weibull fit via the profile score. None if there are no events
    (uncensored observations) to anchor the shape."""
    events = [max(float(t), _MIN_DURATION) for t, c in samples if c]
    all_t = [max(float(t), _MIN_DURATION) for t, _c in samples]
    d = len(events)
    if d == 0 or len(all_t) < 2:
        return None

    mean_ln_events = sum(math.log(t) for t in events) / d

    def score(k: float) -> float:
        # 1/k + mean_ln_events - (sum t^k ln t)/(sum t^k) = 0 at the MLE.
        s0 = sum(t**k for t in all_t)
        s1 = sum(t**k * math.log(t) for t in all_t)
        return 1.0 / k + mean_ln_events - s1 / s0

    # score -> +inf as k -> 0 and decreases through a single root; bracket then bisect.
    k_lo, k_hi = 1e-4, 1.0
    while score(k_hi) > 0 and k_hi < 1e4:
        k_hi *= 2.0
    if score(k_hi) > 0:  # degenerate (e.g. all durations equal) — no finite root
        return None
    for _ in range(200):
        k_mid = 0.5 * (k_lo + k_hi)
        if score(k_mid) > 0:
            k_lo = k_mid
        else:
            k_hi = k_mid
    shape = 0.5 * (k_lo + k_hi)
    scale = (sum(t**shape for t in all_t) / d) ** (1.0 / shape)

    loglik = weibull_loglik(samples, shape, scale)
    return ParametricFit(
        family="weibull",
        shape=shape,
        scale=scale,
        loglik=loglik,
        n_events=d,
        n_censored=len(samples) - d,
        aic=2.0 * 2 - 2.0 * loglik,
        km_sup_distance=_km_sup_distance(
            samples, lambda t: weibull_survival(t, shape, scale)
        ),
    )


def _nelder_mead(
    objective: Callable[[list[float]], float],
    start: list[float],
    *,
    iters: int = 400,
    tol: float = 1e-8,
) -> list[float]:
    """Minimal Nelder-Mead simplex minimizer over R^n. Deterministic; used for
    the log-logistic fit where no closed-form profile exists."""
    n = len(start)
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    simplex = [list(start)]
    for i in range(n):
        pt = list(start)
        pt[i] += 0.5 if pt[i] == 0 else 0.5 * abs(pt[i])
        simplex.append(pt)
    fvals = [objective(p) for p in simplex]
    for _ in range(iters):
        order = sorted(range(n + 1), key=lambda j: fvals[j])
        simplex = [simplex[j] for j in order]
        fvals = [fvals[j] for j in order]
        if abs(fvals[-1] - fvals[0]) < tol:
            break
        centroid = [sum(simplex[j][i] for j in range(n)) / n for i in range(n)]
        # Reflection
        refl = [centroid[i] + alpha * (centroid[i] - simplex[-1][i]) for i in range(n)]
        f_refl = objective(refl)
        if fvals[0] <= f_refl < fvals[-2]:
            simplex[-1], fvals[-1] = refl, f_refl
            continue
        if f_refl < fvals[0]:
            exp = [centroid[i] + gamma * (refl[i] - centroid[i]) for i in range(n)]
            f_exp = objective(exp)
            if f_exp < f_refl:
                simplex[-1], fvals[-1] = exp, f_exp
            else:
                simplex[-1], fvals[-1] = refl, f_refl
            continue
        # Contraction
        contr = [centroid[i] + rho * (simplex[-1][i] - centroid[i]) for i in range(n)]
        f_contr = objective(contr)
        if f_contr < fvals[-1]:
            simplex[-1], fvals[-1] = contr, f_contr
            continue
        # Shrink toward the best vertex
        best = simplex[0]
        for j in range(1, n + 1):
            simplex[j] = [best[i] + sigma * (simplex[j][i] - best[i]) for i in range(n)]
            fvals[j] = objective(simplex[j])
    best_idx = min(range(n + 1), key=lambda j: fvals[j])
    return simplex[best_idx]


def fit_loglogistic(samples: list[DwellSample]) -> ParametricFit | None:
    """MLE log-logistic fit via Nelder-Mead on (log scale, log shape). None if
    there are no events to anchor the fit."""
    events = [max(float(t), _MIN_DURATION) for t, c in samples if c]
    d = len(events)
    if d == 0 or len(samples) < 2:
        return None

    # Median event time is a robust scale start; shape 1 is the neutral start.
    ordered = sorted(events)
    scale0 = ordered[len(ordered) // 2]

    def neg_ll(theta: list[float]) -> float:
        scale, shape = math.exp(theta[0]), math.exp(theta[1])
        return -loglogistic_loglik(samples, shape, scale)

    theta = _nelder_mead(neg_ll, [math.log(scale0), 0.0])
    scale, shape = math.exp(theta[0]), math.exp(theta[1])

    loglik = loglogistic_loglik(samples, shape, scale)
    return ParametricFit(
        family="loglogistic",
        shape=shape,
        scale=scale,
        loglik=loglik,
        n_events=d,
        n_censored=len(samples) - d,
        aic=2.0 * 2 - 2.0 * loglik,
        km_sup_distance=_km_sup_distance(
            samples, lambda t: loglogistic_survival(t, shape, scale)
        ),
    )


def survival_of(fit: ParametricFit, t: float) -> float:
    """S(t) for whichever family `fit` holds."""
    if fit.family == "weibull":
        return weibull_survival(t, fit.shape, fit.scale)
    return loglogistic_survival(t, fit.shape, fit.scale)


def quantile_of(fit: ParametricFit, p: float) -> float:
    """Inverse CDF for whichever family `fit` holds."""
    if fit.family == "weibull":
        return weibull_quantile(p, fit.shape, fit.scale)
    return loglogistic_quantile(p, fit.shape, fit.scale)


def loglogistic_tail(samples: list[DwellSample]) -> list[float] | None:
    """[shape, scale] of the log-logistic fit, or None if it doesn't converge.
    The compact tail descriptor stored on each dwell cell for the Worker's
    past-the-curve splice (dwell.p_leave_by / worker pLeaveBy). See
    momentarily-gtq.5."""
    fit = fit_loglogistic(samples)
    return [fit.shape, fit.scale] if fit is not None else None


def fit_all(samples: list[DwellSample]) -> list[ParametricFit]:
    """Both families, fitted; failed fits dropped. Empty if nothing fits."""
    return [f for f in (fit_weibull(samples), fit_loglogistic(samples)) if f is not None]


def select_parametric(
    samples: list[DwellSample],
) -> tuple[ParametricFit | None, list[ParametricFit]]:
    """Fit both families and pick the lower-AIC one. Returns (best, all_fits);
    best is None when neither family fits (no events)."""
    fits = fit_all(samples)
    if not fits:
        return None, []
    best = min(fits, key=lambda f: f.aic)
    return best, fits


# --- Log-rank: does the dwell hazard differ across strata? ----------------------


def chi2_sf(x: float, df: int) -> float:
    """Upper-tail P(chi2_df > x) via the regularized incomplete gamma Q(df/2, x/2).
    Hand-rolled (no scipy): series for x < a+1, continued fraction otherwise."""
    if x <= 0:
        return 1.0
    a = df / 2.0
    z = x / 2.0
    gln = math.lgamma(a)
    if z < a + 1.0:
        # Series for the lower regularized gamma P(a, z); Q = 1 - P.
        term = 1.0 / a
        total = term
        n = a
        for _ in range(1000):
            n += 1.0
            term *= z / n
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
        p = total * math.exp(-z + a * math.log(z) - gln)
        return 1.0 - p
    # Lentz continued fraction for Q(a, z) directly.
    tiny = 1e-300
    b = z + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h * math.exp(-z + a * math.log(z) - gln)


@dataclass(frozen=True)
class LogRankResult:
    """Omnibus k-group log-rank test of equal hazard across strata."""

    statistic: float
    df: int
    p_value: float
    observed: dict[str, float]  # observed events per group
    expected: dict[str, float]  # expected events under the null


def _solve(matrix: list[list[float]], rhs: list[float]) -> list[float] | None:
    """Gaussian elimination with partial pivoting; None if singular."""
    n = len(rhs)
    aug = [[*matrix[r], rhs[r]] for r in range(n)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col] / aug[col][col]
            for k in range(col, n + 1):
                aug[r][k] -= factor * aug[col][k]
    return [aug[r][n] / aug[r][r] for r in range(n)]


def logrank_test(groups: dict[str, list[DwellSample]]) -> LogRankResult | None:
    """Omnibus log-rank test that the dwell hazard is equal across `groups`.

    Each group is a list of (duration, completed) dwells, right-censoring honored
    (censored observations leave the risk set without an event). Returns None when
    fewer than two non-empty groups or no events exist. The statistic is
    chi-square with (k-1) df under the null of equal hazards.
    """
    labels = [g for g, s in groups.items() if s]
    if len(labels) < 2:
        return None
    k = len(labels)
    event_times = sorted(
        {int(t) for g in labels for t, c in groups[g] if c}
    )
    if not event_times:
        return None

    observed = dict.fromkeys(labels, 0.0)
    expected = dict.fromkeys(labels, 0.0)
    # Covariance accumulated only over the first k-1 groups (the last is redundant).
    cov = [[0.0 for _ in range(k - 1)] for _ in range(k - 1)]

    def at_risk(g: str, t: int) -> int:
        return sum(1 for dur, _c in groups[g] if dur >= t)

    for t in event_times:
        n_g = [at_risk(labels[j], t) for j in range(k)]
        d_g = [sum(1 for dur, c in groups[labels[j]] if c and dur == t) for j in range(k)]
        n = sum(n_g)
        d = sum(d_g)
        if n <= 1 or d == 0:
            continue
        for j in range(k):
            e = d * n_g[j] / n
            observed[labels[j]] += d_g[j]
            expected[labels[j]] += e
        var_common = d * (n - d) / (n - 1) / (n * n)
        for a in range(k - 1):
            cov[a][a] += var_common * n_g[a] * (n - n_g[a])
            for b in range(a + 1, k - 1):
                term = -var_common * n_g[a] * n_g[b]
                cov[a][b] += term
                cov[b][a] += term

    diff = [observed[labels[j]] - expected[labels[j]] for j in range(k - 1)]
    solved = _solve(cov, diff)
    if solved is None:
        return None
    statistic = sum(diff[a] * solved[a] for a in range(k - 1))
    statistic = max(0.0, statistic)
    return LogRankResult(
        statistic=statistic,
        df=k - 1,
        p_value=chi2_sf(statistic, k - 1),
        observed=observed,
        expected=expected,
    )


def parametric_curve_sec(fit: ParametricFit, points: int = CURVE_POINTS) -> list[int]:
    """Emit a curve_sec-style quantile array from a parametric fit, drop-in
    comparable to dwell.py's empirical curve. The p=1 endpoint is capped at the
    0.999 quantile so the (unbounded) parametric tail stays finite."""
    out: list[int] = []
    for i in range(points):
        p = i / (points - 1)
        p = min(p, 0.999)
        out.append(round(quantile_of(fit, p)))
    # Keep it nondecreasing (rounding at the dense low-probability end can tie).
    for i in range(1, len(out)):
        out[i] = max(out[i], out[i - 1])
    return out
