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


def forward_update(
    state: FilterState,
    obs: Observation,
    params: HMMParams,
    now: int,
) -> FilterState:
    """Advance the filter one tick given a new observation.

    NOT YET IMPLEMENTED — scaffold only. Full implementation lands with 5w0.5.

    Algorithm:
      1. predict:  predicted[s] = Σ_s' prior[s'] * A[s', s]
      2. update:   posterior[s] = predicted[s] * P(obs | s); normalize.
      3. track regime entry: if argmax changed, set regime_entered_at = now.

    Numerically stable variant works in log-space to avoid underflow on long runs.
    """
    raise NotImplementedError(
        "forward_update is scaffold-only; implementation lands in 5w0.5"
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

    NOT YET IMPLEMENTED — scaffold only.

    Computes A^ticks_ahead applied to current posterior. Used for
    p_normal_in_30min / 60min / 120min fields.
    """
    raise NotImplementedError(
        "project_forward is scaffold-only; implementation lands in 5w0.5"
    )


# -----------------------------------------------------------------------------
# Expected dwell time — "how long will we stay in the current regime"
# -----------------------------------------------------------------------------


def expected_dwell_ticks(state: FilterState, params: HMMParams) -> tuple[int, int, int]:
    """Expected remaining dwell in the most-likely state, plus low/high bounds.

    NOT YET IMPLEMENTED — scaffold only.

    Returns (median, 25th_percentile, 75th_percentile) in ticks. Caller converts
    ticks → minutes via the cron interval.
    """
    raise NotImplementedError(
        "expected_dwell_ticks is scaffold-only; implementation lands in 5w0.5"
    )


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
