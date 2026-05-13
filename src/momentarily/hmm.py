"""Per-line Hidden Markov Model over transit service state.

Three hidden states (normal, disrupted, suspended). Observations at each cron tick:
  - alert_count          (Poisson per state)
  - severity_sum         (Gamma per state)  — sum of sort_order across active alerts
  - has_suspended_alert  (Bernoulli per state) — any "Suspended" / "No Trains"
  - has_delays           (Bernoulli per state) — any "Delays" / "Severe Delays"
  - has_service_change   (Bernoulli per state) — any non-planned "Service Change" /
                                                 "Trains Rerouted" / "Stops Skipped"
  - has_planned          (Bernoulli per state) — any alert_type starting "Planned -"

Hand-rolled — no extra deps. Forward algorithm for filtering, Baum-Welch for the
weekly refit (training loop will live separately and call into here).

This implementation backs the user-facing `condition` and `recovery_minutes`
fields in the snapshot. Outputs are shadow-logged during Phase 1 of the
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

# Time-of-day bins for emission conditioning. UTC-based for simplicity (no DST
# branching in the loader). NYC subway operational rhythms broadly align with
# ET, which is UTC-4/5 — these bins approximate (UTC-4):
#   0 overnight     00-05 UTC  ≈ 20-01 ET — late-night planned work peak
#   1 morning_rush  05-13 UTC  ≈ 01-09 ET — wind-down + morning ramp
#   2 midday        13-17 UTC  ≈ 09-13 ET — morning peak into midday
#   3 evening_rush  17-23 UTC  ≈ 13-19 ET — midday through evening peak
#   4 late          23-24 UTC  ≈ 19-20 ET — evening into overnight
N_TOD_BINS = 5


def tod_bin(epoch_seconds: int) -> int:
    """Map UTC epoch seconds to a TOD bin index in [0, N_TOD_BINS)."""
    hour = (epoch_seconds // 3600) % 24
    if hour < 5:
        return 0
    if hour < 13:
        return 1
    if hour < 17:
        return 2
    if hour < 23:
        return 3
    return 4


@dataclass(frozen=True)
class Observation:
    """One cron-tick observation for a single entity."""

    alert_count: int
    severity_sum: int  # sum of sort_order across active alerts
    has_suspended_alert: bool
    has_delays: bool = False
    has_service_change: bool = False
    has_planned: bool = False
    tod_bin: int = 0  # TOD bin index; if HMMParams.emissions_by_bin unset, ignored


@dataclass(frozen=True)
class EmissionParams:
    """Per-state emission parameters for one entity.

    Each state has its own Poisson rate for alert count, Gamma shape/rate for
    severity sum, and Bernoulli probabilities for each alert-family indicator.
    """

    # Indexed parallel to STATES (normal, disrupted, suspended)
    poisson_lambda: tuple[float, float, float]
    gamma_alpha: tuple[float, float, float]
    gamma_beta: tuple[float, float, float]
    # bernoulli_p is the "suspended-alert present" channel; kept name-stable.
    bernoulli_p: tuple[float, float, float]
    bernoulli_p_delays: tuple[float, float, float] = (0.01, 0.3, 0.5)
    bernoulli_p_service_change: tuple[float, float, float] = (0.01, 0.4, 0.6)
    bernoulli_p_planned: tuple[float, float, float] = (0.05, 0.3, 0.5)


@dataclass(frozen=True)
class HMMParams:
    """Trained per-entity HMM parameters.

    `emissions` is the unconditioned (single) emission set — used when
    `emissions_by_bin` is None. When `emissions_by_bin` is provided (length
    N_TOD_BINS), forward/EM look up per-bin emissions via obs.tod_bin and
    `emissions` is ignored.
    """

    transition: tuple[tuple[float, float, float], ...]  # 3x3 matrix
    initial: tuple[float, float, float]
    emissions: EmissionParams
    emissions_by_bin: tuple[EmissionParams, ...] | None = None


def _emissions_for(params: HMMParams, obs: Observation) -> EmissionParams:
    """Pick the right EmissionParams for this observation's TOD bin."""
    if params.emissions_by_bin is None:
        return params.emissions
    bin_idx = max(0, min(N_TOD_BINS - 1, obs.tod_bin))
    return params.emissions_by_bin[bin_idx]


@dataclass(frozen=True)
class FilterState:
    """Posterior over states for one entity at the current tick.

    Probabilities sum to 1.0 across STATES.
    """

    probabilities: tuple[float, float, float]
    regime_entered_at: int  # epoch when the most-likely state last changed
    last_updated_at: int  # epoch of the observation that produced this state


# Published-state vocabulary — includes the HMM hidden states plus Unknown,
# which is a publish-layer concept (used when upstream feed is unavailable).
PUBLISHED_UNKNOWN = "unknown"
PublishedLabel = Literal["normal", "disrupted", "suspended", "unknown"]

# Consecutive ticks the argmax must hold before a state change is published.
# 2 = 10 min at the 5-min cron cadence — kills single-tick blips while still
# tracking real regime changes within one cron period.
HYSTERESIS_TICKS = 2


@dataclass(frozen=True)
class PublishedState:
    """What consumers see — lags raw argmax(alpha) via hysteresis, surfaces
    Unknown for feed gaps. Independent from FilterState (the posterior is
    always advanced honestly; PublishedState just decides what to expose).
    """

    label: PublishedLabel
    pending_state: State  # current argmax candidate (or last published if unknown)
    pending_streak: int  # consecutive ticks argmax has held at pending_state
    last_updated_at: int


def initial_published_state(state: FilterState) -> PublishedState:
    """Seed a PublishedState aligned to the filter's current argmax."""
    argmax_idx = max(range(N_STATES), key=lambda i: state.probabilities[i])
    return PublishedState(
        label=STATES[argmax_idx],
        pending_state=STATES[argmax_idx],
        pending_streak=HYSTERESIS_TICKS,  # start already-published
        last_updated_at=state.last_updated_at,
    )


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
    """Per-state log P(obs | state).

    Channels treated as conditionally independent given state. Real-world
    independence is imperfect (planned + delays correlate), but with 3 states
    the bias is small relative to the signal gain from the extra channels.
    """
    out = [
        _log_poisson(obs.alert_count, params.poisson_lambda[i])
        + _log_gamma(
            float(obs.severity_sum), params.gamma_alpha[i], params.gamma_beta[i]
        )
        + _log_bernoulli(obs.has_suspended_alert, params.bernoulli_p[i])
        + _log_bernoulli(obs.has_delays, params.bernoulli_p_delays[i])
        + _log_bernoulli(obs.has_service_change, params.bernoulli_p_service_change[i])
        + _log_bernoulli(obs.has_planned, params.bernoulli_p_planned[i])
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

    log_emis = _log_emission(obs, _emissions_for(params, obs))
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


def forward_step(
    state: FilterState,
    published: PublishedState,
    obs: Observation | None,
    params: HMMParams,
    now: int,
) -> tuple[FilterState, PublishedState]:
    """Advance one tick with hysteresis + Unknown handling.

    obs=None signals an upstream feed gap (fetch failure, malformed payload,
    etc.). In that case the posterior is preserved (no forward update) and the
    published label flips to "unknown". When obs returns, hysteresis resumes
    from the surviving posterior.
    """
    if obs is None:
        # Feed gap: don't corrupt alpha; surface Unknown to consumers but keep
        # pending_state where it was so the next real obs resumes normally.
        return state, PublishedState(
            label=PUBLISHED_UNKNOWN,
            pending_state=published.pending_state,
            pending_streak=published.pending_streak,
            last_updated_at=now,
        )

    new_state = forward_update(state, obs, params, now)
    new_argmax_idx = max(
        range(N_STATES), key=lambda i: new_state.probabilities[i]
    )
    new_argmax = STATES[new_argmax_idx]

    if new_argmax == published.pending_state:
        streak = published.pending_streak + 1
    else:
        streak = 1

    # Promote pending to published only when it's been stable long enough.
    # Special case: when coming back from Unknown, the first real observation
    # should publish immediately (we already lost ticks; don't compound the lag).
    coming_from_unknown = published.label == PUBLISHED_UNKNOWN
    if coming_from_unknown or streak >= HYSTERESIS_TICKS:
        new_label: PublishedLabel = new_argmax
    else:
        new_label = published.label

    return new_state, PublishedState(
        label=new_label,
        pending_state=new_argmax,
        pending_streak=streak,
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


def _per_tick_emissions(
    obs_seq: list[Observation], params: HMMParams
) -> tuple[list[tuple[float, float, float]], list[float]]:
    """For numerical stability, rescale emissions per tick so the max across
    states is 1.0. The forward scaling absorbs the per-tick rescale; we just
    have to add the rescale offsets back when computing log P(o | θ).

    Each tick looks up its TOD-bin's emissions via _emissions_for; with
    emissions_by_bin=None this collapses to a single emission set everywhere.
    """
    emis: list[tuple[float, float, float]] = []
    offsets: list[float] = []
    for obs in obs_seq:
        log_e = _log_emission(obs, _emissions_for(params, obs))
        max_log = max(log_e)
        offsets.append(max_log)
        emis.append(
            (
                math.exp(log_e[0] - max_log),
                math.exp(log_e[1] - max_log),
                math.exp(log_e[2] - max_log),
            )
        )
    return emis, offsets


def _forward_scaled(
    emis: list[tuple[float, float, float]], params: HMMParams
) -> tuple[list[list[float]], list[float]]:
    """Scaled forward pass. Returns (alpha[t][s], scales[t]) where alpha sums to 1
    across states at every t."""
    t_max = len(emis)
    alpha: list[list[float]] = [[0.0] * N_STATES for _ in range(t_max)]
    scales: list[float] = [0.0] * t_max
    a = params.transition

    for s in range(N_STATES):
        alpha[0][s] = params.initial[s] * emis[0][s]
    s0 = sum(alpha[0]) or 1e-300
    scales[0] = s0
    for s in range(N_STATES):
        alpha[0][s] /= s0

    for t in range(1, t_max):
        for s in range(N_STATES):
            alpha[t][s] = (
                sum(alpha[t - 1][sp] * a[sp][s] for sp in range(N_STATES))
                * emis[t][s]
            )
        st = sum(alpha[t]) or 1e-300
        scales[t] = st
        for s in range(N_STATES):
            alpha[t][s] /= st
    return alpha, scales


def _backward_scaled(
    emis: list[tuple[float, float, float]],
    scales: list[float],
    params: HMMParams,
) -> list[list[float]]:
    """Scaled backward pass using the same per-tick scaling as the forward pass."""
    t_max = len(emis)
    beta: list[list[float]] = [[0.0] * N_STATES for _ in range(t_max)]
    a = params.transition

    for s in range(N_STATES):
        beta[t_max - 1][s] = 1.0 / scales[t_max - 1]

    for t in reversed(range(t_max - 1)):
        for s in range(N_STATES):
            beta[t][s] = (
                sum(
                    a[s][sp] * emis[t + 1][sp] * beta[t + 1][sp]
                    for sp in range(N_STATES)
                )
                / scales[t]
            )
    return beta


def _estimate_emissions(
    gamma: list[list[float]],
    observations: list[Observation],
    indices,
    fallback: EmissionParams,
) -> EmissionParams:
    """Weighted-MLE M-step over a subset of tick indices. When a subset has too
    little posterior mass for a given state, fall back to that state's existing
    params (don't overwrite with noise from 1-2 observations).
    """
    idx_list = list(indices)
    state_weight = [
        sum(gamma[t][s] for t in idx_list) for s in range(N_STATES)
    ]

    poisson_lambda: list[float] = []
    bernoulli_p: list[float] = []
    bernoulli_p_delays: list[float] = []
    bernoulli_p_service_change: list[float] = []
    bernoulli_p_planned: list[float] = []
    gamma_alpha: list[float] = []
    gamma_beta: list[float] = []

    def weighted_bernoulli(s: int, w: float, indicator) -> float:
        p = (
            sum(
                gamma[t][s] * (1.0 if indicator(observations[t]) else 0.0)
                for t in idx_list
            )
            / w
        )
        return min(max(p, 1e-6), 1 - 1e-6)

    # If the whole subset has < this many effective observations across all
    # states, the slice is too thin to fit reliably — keep the fallback.
    MIN_EFFECTIVE_OBS = 5
    total_weight = sum(state_weight)
    if total_weight < MIN_EFFECTIVE_OBS:
        return fallback

    for s in range(N_STATES):
        w = state_weight[s]
        if w <= 0:
            poisson_lambda.append(fallback.poisson_lambda[s])
            bernoulli_p.append(fallback.bernoulli_p[s])
            bernoulli_p_delays.append(fallback.bernoulli_p_delays[s])
            bernoulli_p_service_change.append(fallback.bernoulli_p_service_change[s])
            bernoulli_p_planned.append(fallback.bernoulli_p_planned[s])
            gamma_alpha.append(fallback.gamma_alpha[s])
            gamma_beta.append(fallback.gamma_beta[s])
            continue

        lam = (
            sum(gamma[t][s] * observations[t].alert_count for t in idx_list) / w
        )
        poisson_lambda.append(max(lam, 1e-6))

        bernoulli_p.append(weighted_bernoulli(s, w, lambda o: o.has_suspended_alert))
        bernoulli_p_delays.append(weighted_bernoulli(s, w, lambda o: o.has_delays))
        bernoulli_p_service_change.append(
            weighted_bernoulli(s, w, lambda o: o.has_service_change)
        )
        bernoulli_p_planned.append(weighted_bernoulli(s, w, lambda o: o.has_planned))

        x = [observations[t].severity_sum + 0.5 for t in idx_list]
        mu = sum(gamma[t][s] * xt for t, xt in zip(idx_list, x)) / w
        var = sum(gamma[t][s] * (xt - mu) ** 2 for t, xt in zip(idx_list, x)) / w
        if var <= 0:
            var = 1e-6
        gamma_alpha.append(max(mu * mu / var, 1e-3))
        gamma_beta.append(max(mu / var, 1e-6))

    return EmissionParams(
        poisson_lambda=(poisson_lambda[0], poisson_lambda[1], poisson_lambda[2]),
        gamma_alpha=(gamma_alpha[0], gamma_alpha[1], gamma_alpha[2]),
        gamma_beta=(gamma_beta[0], gamma_beta[1], gamma_beta[2]),
        bernoulli_p=(bernoulli_p[0], bernoulli_p[1], bernoulli_p[2]),
        bernoulli_p_delays=(
            bernoulli_p_delays[0],
            bernoulli_p_delays[1],
            bernoulli_p_delays[2],
        ),
        bernoulli_p_service_change=(
            bernoulli_p_service_change[0],
            bernoulli_p_service_change[1],
            bernoulli_p_service_change[2],
        ),
        bernoulli_p_planned=(
            bernoulli_p_planned[0],
            bernoulli_p_planned[1],
            bernoulli_p_planned[2],
        ),
    )


def _em_iteration(
    observations: list[Observation], params: HMMParams
) -> tuple[HMMParams, float]:
    """One E-step + M-step. Returns (new_params, log_likelihood_under_old_params)."""
    emis, offsets = _per_tick_emissions(observations, params)
    alpha, scales = _forward_scaled(emis, params)
    beta = _backward_scaled(emis, scales, params)
    t_max = len(observations)
    a = params.transition

    gamma: list[list[float]] = [[0.0] * N_STATES for _ in range(t_max)]
    for t in range(t_max):
        z = sum(alpha[t][s] * beta[t][s] for s in range(N_STATES)) or 1e-300
        for s in range(N_STATES):
            gamma[t][s] = alpha[t][s] * beta[t][s] / z

    xi_sum: list[list[float]] = [[0.0] * N_STATES for _ in range(N_STATES)]
    for t in range(t_max - 1):
        for s in range(N_STATES):
            for sp in range(N_STATES):
                xi_sum[s][sp] += (
                    alpha[t][s] * a[s][sp] * emis[t + 1][sp] * beta[t + 1][sp]
                )

    # ----- M-step -----
    new_pi = (gamma[0][0], gamma[0][1], gamma[0][2])

    new_a_rows: list[tuple[float, float, float]] = []
    for s in range(N_STATES):
        denom = sum(gamma[t][s] for t in range(t_max - 1))
        if denom <= 0:
            new_a_rows.append(params.transition[s])
            continue
        row = [xi_sum[s][sp] / denom for sp in range(N_STATES)]
        rsum = sum(row) or 1.0
        row = [r / rsum for r in row]
        new_a_rows.append((row[0], row[1], row[2]))
    new_a = tuple(new_a_rows)

    if params.emissions_by_bin is None:
        new_emissions = _estimate_emissions(
            gamma, observations, range(t_max), params.emissions
        )
        new_emissions_by_bin = None
    else:
        # Bucket tick indices by tod_bin
        buckets: dict[int, list[int]] = {b: [] for b in range(N_TOD_BINS)}
        for t, obs in enumerate(observations):
            bin_idx = max(0, min(N_TOD_BINS - 1, obs.tod_bin))
            buckets[bin_idx].append(t)
        # Re-estimate emissions per bin against its existing prior
        new_emissions_by_bin = tuple(
            _estimate_emissions(
                gamma, observations, buckets[b], params.emissions_by_bin[b]
            )
            for b in range(N_TOD_BINS)
        )
        # Keep .emissions in sync with bin 0 so legacy consumers don't see stale data
        new_emissions = new_emissions_by_bin[0]

    new_params = HMMParams(
        transition=new_a,
        initial=new_pi,
        emissions=new_emissions,
        emissions_by_bin=new_emissions_by_bin,
    )

    log_lik = sum(math.log(c) for c in scales) + sum(offsets)
    return new_params, log_lik


def _sort_states_by_lambda(params: HMMParams) -> HMMParams:
    """Reorder states so poisson_lambda is ascending: state 0 = quietest.

    EM is invariant to state labels (label-switching), so re-sort after fitting
    to keep "normal/disrupted/suspended" semantically consistent across runs.
    With emissions_by_bin set, ordering is by the SUM of poisson_lambda across
    bins — a state that's quietest overall stays state 0 even if its rank
    flips in one bin.
    """
    if params.emissions_by_bin is None:
        rank_lambda = params.emissions.poisson_lambda
    else:
        rank_lambda = tuple(
            sum(em.poisson_lambda[s] for em in params.emissions_by_bin)
            for s in range(N_STATES)
        )
    order = sorted(range(N_STATES), key=lambda s: rank_lambda[s])
    if order == [0, 1, 2]:
        return params

    def reorder3(t: tuple[float, ...]) -> tuple[float, float, float]:
        return (t[order[0]], t[order[1]], t[order[2]])

    def reorder_emissions(em: EmissionParams) -> EmissionParams:
        return EmissionParams(
            poisson_lambda=reorder3(em.poisson_lambda),
            gamma_alpha=reorder3(em.gamma_alpha),
            gamma_beta=reorder3(em.gamma_beta),
            bernoulli_p=reorder3(em.bernoulli_p),
            bernoulli_p_delays=reorder3(em.bernoulli_p_delays),
            bernoulli_p_service_change=reorder3(em.bernoulli_p_service_change),
            bernoulli_p_planned=reorder3(em.bernoulli_p_planned),
        )

    new_emissions = reorder_emissions(params.emissions)
    new_emissions_by_bin = (
        tuple(reorder_emissions(em) for em in params.emissions_by_bin)
        if params.emissions_by_bin is not None
        else None
    )
    new_initial = reorder3(params.initial)
    new_transition = tuple(
        reorder3(tuple(params.transition[order[s]])) for s in range(N_STATES)
    )
    return HMMParams(
        transition=new_transition,
        initial=new_initial,
        emissions=new_emissions,
        emissions_by_bin=new_emissions_by_bin,
    )


def fit_em(
    observations: list[Observation],
    initial_params: HMMParams,
    max_iterations: int = 50,
    tolerance: float = 1e-4,
) -> tuple[HMMParams, list[float]]:
    """Train per-entity HMM parameters from a sequence of observations.

    Uses Baum-Welch: forward-backward (scaled for numerical stability) for the
    E-step, weighted-MLE for the M-step. Converges when relative change in
    log-likelihood drops below `tolerance` or `max_iterations` is reached.

    Returns (fitted_params, log_likelihoods) where log_likelihoods is the
    sequence across iterations — useful for verifying monotonicity.

    The output is re-sorted so state 0 has the smallest poisson_lambda, giving
    stable "normal/disrupted/suspended" semantics across runs.
    """
    if not observations:
        raise ValueError("fit_em requires at least one observation")

    params = initial_params
    log_liks: list[float] = []
    prev: float | None = None

    for _ in range(max_iterations):
        params, log_lik = _em_iteration(observations, params)
        log_liks.append(log_lik)
        if prev is not None:
            denom = max(abs(prev), 1e-12)
            if abs(log_lik - prev) / denom < tolerance:
                break
        prev = log_lik

    return _sort_states_by_lambda(params), log_liks
