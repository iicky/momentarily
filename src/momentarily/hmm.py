"""Per-line Hidden Markov Model over transit service state.

Three hidden states (normal, disrupted, suspended). Observations at each cron tick:
  - alert_count        (Poisson per state)
  - severity_sum       (Gamma per state)  — sum of sort_order across active alerts
  - has_suspended_alert (Bernoulli per state)

Hand-rolled — no extra deps. Forward algorithm for filtering, Baum-Welch for the
weekly refit (training loop will live separately and call into here).

Methodology
-----------
The regime-switching framing follows Cheng & Sun (2024), "Conditional forecasting
of bus travel time and passenger occupancy with Bayesian Markov regime-switching
VAR" (arXiv:2401.17387), adapted for the GTFS-RT Mercury alerts feed rather than
travel-time signals. The recovery-prediction framing — modeling expected
time-to-clear from the current regime — borrows from Liu et al. (2022),
"Detecting metro service disruptions via large-scale vehicle location data"
(Transportation Research Part C, 145), which used GMM on vehicle headways
rather than alerts but established the recovery-aware probabilistic framing for
metro state.

See docs/papers.md for the full prior-art survey.

This is the engine that backs the user-facing `condition` and `recovery_minutes`
fields in the snapshot. Outputs are shadow-logged only during Phase 1 of the
rollout; they graduate to public snapshot fields after calibration review.

NOT yet wired into the publisher — this module is scaffolding. Under the Path 2
architecture the forward filter is ported to TypeScript for the live Worker;
this Python implementation becomes the reference for that port and the engine
for offline Baum-Welch training.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

State = Literal["normal", "disrupted", "suspended"]
STATES: tuple[State, ...] = ("normal", "disrupted", "suspended")
N_STATES = len(STATES)


@dataclass(frozen=True)
class Observation:
    """One cron-tick observation for a single entity."""

    alert_count: int
    severity_sum: int  # sum of sort_order across active alerts
    has_suspended_alert: bool


@dataclass(frozen=True)
class EmissionParams:
    """Per-state emission parameters for one entity.

    Each state has its own Poisson rate for alert count, Gamma shape/rate for
    severity sum, and Bernoulli probability for the has-suspended flag.
    """

    # Indexed parallel to STATES (normal, disrupted, suspended)
    poisson_lambda: tuple[float, float, float]
    gamma_alpha: tuple[float, float, float]
    gamma_beta: tuple[float, float, float]
    bernoulli_p: tuple[float, float, float]


@dataclass(frozen=True)
class HMMParams:
    """Trained per-entity HMM parameters."""

    transition: tuple[tuple[float, float, float], ...]  # 3x3 matrix
    initial: tuple[float, float, float]
    emissions: EmissionParams


@dataclass(frozen=True)
class FilterState:
    """Posterior over states for one entity at the current tick.

    Probabilities sum to 1.0 across STATES.
    """

    probabilities: tuple[float, float, float]
    regime_entered_at: int  # epoch when the most-likely state last changed
    last_updated_at: int  # epoch of the observation that produced this state


# -----------------------------------------------------------------------------
# Forward algorithm (filtering) — per-tick update of FilterState
# -----------------------------------------------------------------------------


def _log_poisson(k: int, lam: float) -> float:
    if lam <= 0:
        return 0.0 if k == 0 else -math.inf
    return -lam + k * math.log(lam) - math.lgamma(k + 1)


def _log_gamma(x: float, alpha: float, beta: float) -> float:
    # Gamma(shape=α, rate=β): pdf = β^α · x^(α-1) · exp(-β·x) / Γ(α)
    # Severity_sum is integer ≥ 0; we shift x by 0.5 so x=0 doesn't blow up log(x).
    shifted = max(x + 0.5, 1e-9)
    return (
        alpha * math.log(beta)
        + (alpha - 1) * math.log(shifted)
        - beta * shifted
        - math.lgamma(alpha)
    )


def _log_bernoulli(value: bool, p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p) if value else math.log1p(-p)


def _log_emission(
    obs: Observation, params: EmissionParams
) -> tuple[float, float, float]:
    """Per-state log P(obs | state)."""
    out = [
        _log_poisson(obs.alert_count, params.poisson_lambda[i])
        + _log_gamma(
            float(obs.severity_sum), params.gamma_alpha[i], params.gamma_beta[i]
        )
        + _log_bernoulli(obs.has_suspended_alert, params.bernoulli_p[i])
        for i in range(N_STATES)
    ]
    return (out[0], out[1], out[2])


def forward_update(
    state: FilterState,
    obs: Observation,
    params: HMMParams,
    now: int,
) -> FilterState:
    """Advance the filter one tick given a new observation.

    Algorithm:
      1. predict:  predicted[s] = Σ_s' prior[s'] * A[s', s]
      2. update:   posterior[s] ∝ predicted[s] · P(obs | s) (log-space, then normalize)
      3. track regime entry: if argmax(posterior) changed, set regime_entered_at = now.
    """
    prior = state.probabilities
    a = params.transition

    predicted = [
        sum(prior[sp] * a[sp][s] for sp in range(N_STATES)) for s in range(N_STATES)
    ]

    log_emis = _log_emission(obs, params.emissions)
    log_post_unnorm = [
        (math.log(predicted[s]) if predicted[s] > 0 else -math.inf) + log_emis[s]
        for s in range(N_STATES)
    ]

    max_log = max(log_post_unnorm)
    if max_log == -math.inf:
        # No state can explain this observation under the current params — keep prior.
        post = list(prior)
    else:
        scaled = [math.exp(lp - max_log) for lp in log_post_unnorm]
        total = sum(scaled)
        post = [s / total for s in scaled]

    prev_argmax = max(range(N_STATES), key=lambda i: prior[i])
    new_argmax = max(range(N_STATES), key=lambda i: post[i])
    regime_entered_at = (
        now if new_argmax != prev_argmax else state.regime_entered_at
    )

    return FilterState(
        probabilities=(post[0], post[1], post[2]),
        regime_entered_at=regime_entered_at,
        last_updated_at=now,
    )


# -----------------------------------------------------------------------------
# Forward marginal projection — "P(state = normal in k ticks)"
# -----------------------------------------------------------------------------


def project_forward(
    state: FilterState,
    params: HMMParams,
    ticks_ahead: int,
) -> tuple[float, float, float]:
    """Marginal P(state in ticks_ahead steps) starting from `state`.

    Iterates predicted[s] = Σ_s' current[s'] · A[s', s]. For small ticks_ahead
    (≤ a few dozen) this is cheaper than matrix exponentiation and stays in
    probability space without log-space gymnastics.
    """
    if ticks_ahead < 0:
        raise ValueError("ticks_ahead must be >= 0")
    if ticks_ahead == 0:
        return state.probabilities

    a = params.transition
    current = list(state.probabilities)
    for _ in range(ticks_ahead):
        current = [
            sum(current[sp] * a[sp][s] for sp in range(N_STATES))
            for s in range(N_STATES)
        ]
    return (current[0], current[1], current[2])


# -----------------------------------------------------------------------------
# Expected dwell time — "how long will we stay in the current regime"
# -----------------------------------------------------------------------------


def expected_dwell_ticks(
    state: FilterState, params: HMMParams
) -> tuple[int, int, int]:
    """Expected remaining dwell in the most-likely state, plus low/high bounds.

    Treats remaining dwell as geometric with leave-probability (1 − A[s*, s*])
    where s* is the most-likely state. The q-th quantile of P(leave-by-k) ≥ q is
    the smallest k where (1 − A[s*, s*])^k ≥ 1 − q, i.e.
    k_q = ⌈log(1 − q) / log(A[s*, s*])⌉.

    Returns (median, 25th_percentile, 75th_percentile) in ticks. Caller converts
    ticks → minutes via the cron interval.
    """
    argmax = max(range(N_STATES), key=lambda i: state.probabilities[i])
    self_loop = params.transition[argmax][argmax]

    LARGE = 10_000
    if self_loop >= 1.0:
        return (LARGE, LARGE, LARGE)
    if self_loop <= 0:
        return (1, 1, 1)

    log_self = math.log(self_loop)

    def quantile(q: float) -> int:
        target = 1 - q
        if target <= 0:
            return LARGE
        return max(1, math.ceil(math.log(target) / log_self))

    return (quantile(0.5), quantile(0.25), quantile(0.75))


# -----------------------------------------------------------------------------
# Baum-Welch EM — periodic refit (weekly) from history
# -----------------------------------------------------------------------------


def fit_em(
    observations: list[Observation],
    max_iterations: int = 50,
    tolerance: float = 1e-4,
) -> HMMParams:
    """Train per-entity HMM parameters from a sequence of observations.

    NOT YET IMPLEMENTED — scaffold only.

    Uses Baum-Welch (forward-backward + parameter re-estimation). Convergence
    criterion: log-likelihood change below tolerance.

    Cold-start (this entity has < N days of history) is handled at the caller:
    publisher initializes with a weakly-informative prior and sets
    `model_warming_up=True` in the published Inference object.
    """
    raise NotImplementedError("fit_em is scaffold-only; implementation lands in 5w0.5")
