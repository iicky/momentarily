"""Per-line Hidden Markov Model over transit service state.

Three hidden states (normal, disrupted, suspended). Observations at each cron tick:
  - alert_count          (Poisson per state)
  - has_suspended_alert  (Bernoulli per state) — any "Suspended" / "No Trains"
  - has_delays           (Bernoulli per state) — any "Delays" / "Severe Delays"
  - has_service_change   (Bernoulli per state) — any non-planned "Service Change" /
                                                 "Trains Rerouted" / "Stops Skipped"
  - has_planned          (Bernoulli per state) — any alert_type starting "Planned -"
  - advanced_n of matched_n (Binomial per state) - of the trips seen both this
                            tick and last, how many advanced a stop. Per-state
                            advance rate (normal~baseline, disrupted<baseline,
                            suspended~0). Gated off when matched_n==0 or the
                            derivation has no baseline (has_movement=False).

severity_sum is still carried on Observation but is NOT a likelihood channel:
it's a near-deterministic function of the same alert list as alert_count and
the flags, so treating it as an independent Gamma channel double-counted the
count evidence and saturated the posterior (reliability mass piled at the
extremes). The gamma_alpha/gamma_beta params remain in the schema for
back-compat but are vestigial. See momentarily-vk0.8.

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
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from zoneinfo import ZoneInfo

State = Literal["normal", "disrupted", "suspended"]
STATES: tuple[State, ...] = ("normal", "disrupted", "suspended")
N_STATES = len(STATES)

# Time-of-day bins for emission conditioning, in America/New_York local time
# (DST-aware — the old UTC bins drifted an hour for the EST half of the year
# and their labels never matched; see momentarily-vk0.10):
#   0 overnight      00-06 ET — late-night planned work window
#   1 morning_rush   06-10 ET
#   2 midday         10-15 ET
#   3 evening_rush   15-20 ET
#   4 evening        20-24 ET
N_TOD_BINS = 5

_NYC_TZ = ZoneInfo("America/New_York")


def tod_bin(epoch_seconds: int) -> int:
    """Map epoch seconds to a TOD bin index in [0, N_TOD_BINS), by ET local hour.

    Mirrors worker/src/hmm.ts tod_bin (Intl-based); keep the bin edges in sync.
    """
    hour = datetime.fromtimestamp(epoch_seconds, tz=_NYC_TZ).hour
    if hour < 6:
        return 0
    if hour < 10:
        return 1
    if hour < 15:
        return 2
    if hour < 20:
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
    # Train-movement channel: of matched_n trips seen both this tick and last,
    # advanced_n moved up a stop. has_movement gates the channel out (no
    # baseline, feed gap, or matched_n==0) so the tick scores on alerts alone.
    advanced_n: int = 0
    matched_n: int = 0
    has_movement: bool = False
    # Service-level channel: assigned_n / (route, tod) baseline. has_service gates
    # it out (no baseline or feed gap) so the tick scores on the other channels.
    service_ratio: float | None = None
    has_service: bool = False


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
    # Per-state probability a matched trip advances a stop in one tick. Normal
    # sits near the route-direction baseline; disrupted below it; suspended ~0.
    advance_rate: tuple[float, float, float] = (0.6, 0.3, 0.02)
    # Per-state service-ratio (assigned_n / baseline) Gaussian: normal ~1.0,
    # disrupted below it, suspended ~0. Optional for back-compat with params
    # written before the service channel.
    service_mu: tuple[float, float, float] = (1.0, 0.6, 0.05)
    service_sigma: tuple[float, float, float] = (0.3, 0.3, 0.15)


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


def _reorder_emissions(
    em: EmissionParams, perm: tuple[int, int, int]
) -> EmissionParams:
    """Reindex every per-state channel of `em` by `perm` (new index -> old)."""

    def r(t: tuple[float, float, float]) -> tuple[float, float, float]:
        return (t[perm[0]], t[perm[1]], t[perm[2]])

    return EmissionParams(
        poisson_lambda=r(em.poisson_lambda),
        gamma_alpha=r(em.gamma_alpha),
        gamma_beta=r(em.gamma_beta),
        bernoulli_p=r(em.bernoulli_p),
        bernoulli_p_delays=r(em.bernoulli_p_delays),
        bernoulli_p_service_change=r(em.bernoulli_p_service_change),
        bernoulli_p_planned=r(em.bernoulli_p_planned),
        advance_rate=r(em.advance_rate),
        service_mu=r(em.service_mu),
        service_sigma=r(em.service_sigma),
    )


def canonicalize_states(params: HMMParams) -> HMMParams:
    """Permute the three states into canonical label order so the state index
    matches its semantics: normal < disrupted < suspended in disruption.

    EM is unsupervised — it finds three clusters but the cluster-to-index
    assignment is arbitrary, so a route can converge with the *quiet* cluster
    sitting on the `disrupted` index. Then a route with no active alerts (an
    all-quiet observation) matches `disrupted` best and latches there with no
    alert to explain it. We anchor identity by the emissions:

      - normal    = the lowest alert-rate (Poisson lambda) state — good service
                    is the quietest cluster.
      - suspended = of the remaining two, the higher suspended-alert
                    probability (bernoulli_p) — suspension is flag-defined, not
                    just count-defined.
      - disrupted = the remaining state.

    With emissions_by_bin set, ranking uses the per-state sums across bins so
    one unusual TOD bin (e.g. overnight, which is bin 0 and aliases
    `.emissions`) can't misrank the whole route.

    The same permutation is applied to the initial vector, the transition matrix
    (rows and columns), and every per-bin emission set. The relabeled model is
    statistically identical — only the index<->label mapping changes. This is
    the single state-ordering rule — fit_em applies it before returning, and
    any post-processing keyed by state index (e.g. per-state self-loop caps)
    must run after it. See momentarily-13j, momentarily-vk0.7.
    """
    em = params.emissions
    if params.emissions_by_bin is None:
        rank_lambda: tuple[float, ...] = em.poisson_lambda
        rank_p: tuple[float, ...] = em.bernoulli_p
        rank_advance: tuple[float, ...] = em.advance_rate
        rank_service: tuple[float, ...] = em.service_mu
    else:
        rank_lambda = tuple(
            sum(e.poisson_lambda[s] for e in params.emissions_by_bin)
            for s in range(N_STATES)
        )
        rank_p = tuple(
            sum(e.bernoulli_p[s] for e in params.emissions_by_bin)
            for s in range(N_STATES)
        )
        rank_advance = tuple(
            sum(e.advance_rate[s] for e in params.emissions_by_bin)
            for s in range(N_STATES)
        )
        rank_service = tuple(
            sum(e.service_mu[s] for e in params.emissions_by_bin)
            for s in range(N_STATES)
        )
    # Movement + service reinforce identity (normal runs most trains and moves
    # most, suspended least) but only break ties: lowest alert rate wins normal,
    # then highest advance rate, then highest service ratio; highest suspended-flag
    # wins suspended, then lowest advance rate, then lowest service ratio.
    normal_idx = min(
        range(N_STATES),
        key=lambda s: (rank_lambda[s], -rank_advance[s], -rank_service[s]),
    )
    rest = [s for s in range(N_STATES) if s != normal_idx]
    suspended_idx = max(
        rest, key=lambda s: (rank_p[s], -rank_advance[s], -rank_service[s])
    )
    disrupted_idx = next(s for s in rest if s != suspended_idx)
    perm = (normal_idx, disrupted_idx, suspended_idx)

    if perm == (0, 1, 2):
        return params  # already canonical

    new_transition = tuple(
        (
            params.transition[perm[i]][perm[0]],
            params.transition[perm[i]][perm[1]],
            params.transition[perm[i]][perm[2]],
        )
        for i in range(N_STATES)
    )
    new_initial = (
        params.initial[perm[0]],
        params.initial[perm[1]],
        params.initial[perm[2]],
    )
    new_by_bin = (
        tuple(_reorder_emissions(e, perm) for e in params.emissions_by_bin)
        if params.emissions_by_bin is not None
        else None
    )
    return HMMParams(
        transition=new_transition,
        initial=new_initial,
        emissions=_reorder_emissions(em, perm),
        emissions_by_bin=new_by_bin,
    )


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


def _log_bernoulli(value: bool, p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p) if value else math.log1p(-p)


def _log_binomial(k: int, n: int, p: float) -> float:
    """log P(k of n advanced | Binomial(n, p)). The n-choose-k coefficient is
    constant across states so it cancels in the per-tick posterior, but it's
    kept so the EM log-likelihood is a true data likelihood (mirrors how the
    Poisson term carries its log k! factor)."""
    if n <= 0:
        return 0.0
    p = min(max(p, 1e-12), 1 - 1e-12)
    log_coef = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    return log_coef + k * math.log(p) + (n - k) * math.log1p(-p)


# Floor on the service-ratio Gaussian's std so a state whose fitted ratios were
# near-constant (e.g. suspended ~0) doesn't collapse to a delta and reject any
# deviation. Mirrors the spirit of BERNOULLI_FLOOR on the other channels.
SERVICE_SIGMA_FLOOR = 0.05


def _log_gauss(x: float, mu: float, sigma: float) -> float:
    s = max(sigma, SERVICE_SIGMA_FLOOR)
    return -0.5 * math.log(2.0 * math.pi * s * s) - (x - mu) ** 2 / (2.0 * s * s)


def _log_emission(
    obs: Observation, params: EmissionParams
) -> tuple[float, float, float]:
    """Per-state log P(obs | state).

    Channels treated as conditionally independent given state. Real-world
    independence is imperfect (planned + delays correlate), but with 3 states
    the bias is small relative to the signal gain from the extra channels.
    severity_sum is deliberately absent — see the module docstring
    (momentarily-vk0.8). The movement channel drops out (contributes 0) when
    has_movement is False or no trips matched; the service channel drops out when
    has_service is False or the ratio is unavailable.
    """
    has_movement = obs.has_movement and obs.matched_n > 0
    has_service = obs.has_service and obs.service_ratio is not None
    out: list[float] = []
    for i in range(N_STATES):
        log_lik = (
            _log_poisson(obs.alert_count, params.poisson_lambda[i])
            + _log_bernoulli(obs.has_suspended_alert, params.bernoulli_p[i])
            + _log_bernoulli(obs.has_delays, params.bernoulli_p_delays[i])
            + _log_bernoulli(
                obs.has_service_change, params.bernoulli_p_service_change[i]
            )
            + _log_bernoulli(obs.has_planned, params.bernoulli_p_planned[i])
        )
        if has_movement:
            log_lik += _log_binomial(
                obs.advanced_n, obs.matched_n, params.advance_rate[i]
            )
        if has_service and obs.service_ratio is not None:
            log_lik += _log_gauss(
                obs.service_ratio, params.service_mu[i], params.service_sigma[i]
            )
        out.append(log_lik)
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
    regime_entered_at = now if new_argmax != prev_argmax else state.regime_entered_at

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
    new_argmax_idx = max(range(N_STATES), key=lambda i: new_state.probabilities[i])
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


def expected_dwell_ticks(state: FilterState, params: HMMParams) -> tuple[int, int, int]:
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
                sum(alpha[t - 1][sp] * a[sp][s] for sp in range(N_STATES)) * emis[t][s]
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


# Floor and ceiling on Bernoulli emissions. 1e-6 was too aggressive: an EM
# fit that drove a flag's normal-state probability to ~1e-6 made the normal
# state reject any observation where that flag was True, even briefly, so
# the forward filter never recovered to "normal" once it had ever been
# tripped. 1e-3 still says "rare" without being numerically degenerate.
BERNOULLI_FLOOR = 1e-3


def _estimate_emissions(
    gamma: list[list[float]],
    observations: list[Observation],
    indices: Iterable[int],
    fallback: EmissionParams,
    *,
    prior: EmissionParams | None = None,
    prior_strength: float = 0.0,
) -> EmissionParams:
    """Weighted-MLE M-step over a subset of tick indices. When a subset has too
    little posterior mass for a given state, fall back to that state's existing
    params (don't overwrite with noise from 1-2 observations).

    With `prior` and `prior_strength > 0`, blend MLE with the prior using
    conjugate posteriors (Gamma/Beta pseudo-counts; convex combo for the Gamma
    α/β where there's no clean conjugate). `prior_strength` is in units of
    effective observations.
    """
    idx_list = list(indices)
    state_weight = [sum(gamma[t][s] for t in idx_list) for s in range(N_STATES)]

    poisson_lambda: list[float] = []
    bernoulli_p: list[float] = []
    bernoulli_p_delays: list[float] = []
    bernoulli_p_service_change: list[float] = []
    bernoulli_p_planned: list[float] = []
    advance_rate: list[float] = []
    gamma_alpha: list[float] = []
    gamma_beta: list[float] = []
    service_mu: list[float] = []
    service_sigma: list[float] = []

    kappa = max(prior_strength, 0.0)
    use_prior = prior is not None and kappa > 0.0

    def posterior_bernoulli(
        s: int,
        w: float,
        indicator: Callable[[Observation], bool],
        prior_p: float,
    ) -> float:
        successes = sum(
            gamma[t][s] * (1.0 if indicator(observations[t]) else 0.0) for t in idx_list
        )
        p = (kappa * prior_p + successes) / (kappa + w) if use_prior else successes / w
        return min(max(p, BERNOULLI_FLOOR), 1.0 - BERNOULLI_FLOOR)

    # If the whole subset has < this many effective observations across all
    # states, the slice is too thin to fit reliably — keep the fallback (or
    # the prior, when one is set).
    MIN_EFFECTIVE_OBS = 5
    total_weight = sum(state_weight)
    if total_weight < MIN_EFFECTIVE_OBS:
        if use_prior:
            assert prior is not None
            return prior
        return fallback

    for s in range(N_STATES):
        w = state_weight[s]
        if w <= 0 and not use_prior:
            poisson_lambda.append(fallback.poisson_lambda[s])
            bernoulli_p.append(fallback.bernoulli_p[s])
            bernoulli_p_delays.append(fallback.bernoulli_p_delays[s])
            bernoulli_p_service_change.append(fallback.bernoulli_p_service_change[s])
            bernoulli_p_planned.append(fallback.bernoulli_p_planned[s])
            advance_rate.append(fallback.advance_rate[s])
            gamma_alpha.append(fallback.gamma_alpha[s])
            gamma_beta.append(fallback.gamma_beta[s])
            service_mu.append(fallback.service_mu[s])
            service_sigma.append(fallback.service_sigma[s])
            continue

        # Poisson λ: Gamma(κ·λ_prior, κ) prior → posterior mean below.
        sum_alerts = sum(gamma[t][s] * observations[t].alert_count for t in idx_list)
        if use_prior:
            assert prior is not None
            lam = (kappa * prior.poisson_lambda[s] + sum_alerts) / (kappa + w)
        else:
            lam = sum_alerts / w
        poisson_lambda.append(max(lam, 1e-6))

        if use_prior:
            assert prior is not None
            bernoulli_p.append(
                posterior_bernoulli(
                    s, w, lambda o: o.has_suspended_alert, prior.bernoulli_p[s]
                )
            )
            bernoulli_p_delays.append(
                posterior_bernoulli(
                    s, w, lambda o: o.has_delays, prior.bernoulli_p_delays[s]
                )
            )
            bernoulli_p_service_change.append(
                posterior_bernoulli(
                    s,
                    w,
                    lambda o: o.has_service_change,
                    prior.bernoulli_p_service_change[s],
                )
            )
            bernoulli_p_planned.append(
                posterior_bernoulli(
                    s, w, lambda o: o.has_planned, prior.bernoulli_p_planned[s]
                )
            )
        else:
            bernoulli_p.append(
                posterior_bernoulli(s, w, lambda o: o.has_suspended_alert, 0.0)
            )
            bernoulli_p_delays.append(
                posterior_bernoulli(s, w, lambda o: o.has_delays, 0.0)
            )
            bernoulli_p_service_change.append(
                posterior_bernoulli(s, w, lambda o: o.has_service_change, 0.0)
            )
            bernoulli_p_planned.append(
                posterior_bernoulli(s, w, lambda o: o.has_planned, 0.0)
            )

        # Advance rate: responsibility-weighted pooled Binomial rate over ticks
        # where movement is available, k = Σ γ·advanced_n, n = Σ γ·matched_n.
        # The prior acts as κ pseudo-trials at prior.advance_rate (Beta-style),
        # so a thin-movement state leans on the baseline prior instead of noise.
        mov_k = sum(
            gamma[t][s] * observations[t].advanced_n
            for t in idx_list
            if observations[t].has_movement
        )
        mov_n = sum(
            gamma[t][s] * observations[t].matched_n
            for t in idx_list
            if observations[t].has_movement
        )
        if use_prior:
            assert prior is not None
            rate = (kappa * prior.advance_rate[s] + mov_k) / (kappa + mov_n)
        elif mov_n > 0:
            rate = mov_k / mov_n
        else:
            rate = fallback.advance_rate[s]
        advance_rate.append(min(max(rate, BERNOULLI_FLOOR), 1.0 - BERNOULLI_FLOOR))

        # Service ratio: responsibility-weighted Gaussian over ticks where the
        # service level is available. The prior acts as κ pseudo-observations at
        # prior.service_mu/sigma so a state with little service data leans on it.
        svc: list[tuple[float, float]] = []
        for t in idx_list:
            r = observations[t].service_ratio
            if observations[t].has_service and r is not None:
                svc.append((gamma[t][s], r))
        svc_w = sum(g for g, _ in svc)
        svc_sum = sum(g * r for g, r in svc)
        if use_prior:
            assert prior is not None
            mu = (kappa * prior.service_mu[s] + svc_sum) / (kappa + svc_w)
        elif svc_w > 0:
            mu = svc_sum / svc_w
        else:
            mu = fallback.service_mu[s]
        service_mu.append(mu)
        if svc_w > 0:
            var = sum(g * (r - mu) ** 2 for g, r in svc) / svc_w
            sd = max(var**0.5, SERVICE_SIGMA_FLOOR)
            if use_prior:
                assert prior is not None
                sd = (kappa * prior.service_sigma[s] + svc_w * sd) / (kappa + svc_w)
        elif use_prior:
            assert prior is not None
            sd = prior.service_sigma[s]
        else:
            sd = fallback.service_sigma[s]
        service_sigma.append(max(sd, SERVICE_SIGMA_FLOOR))

        # Gamma α/β are vestigial — severity_sum is no longer a likelihood
        # channel (momentarily-vk0.8) — so pass them through unchanged for
        # schema back-compat rather than fitting dead parameters.
        if use_prior:
            assert prior is not None
            gamma_alpha.append(prior.gamma_alpha[s])
            gamma_beta.append(prior.gamma_beta[s])
        else:
            gamma_alpha.append(fallback.gamma_alpha[s])
            gamma_beta.append(fallback.gamma_beta[s])

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
        advance_rate=(advance_rate[0], advance_rate[1], advance_rate[2]),
        service_mu=(service_mu[0], service_mu[1], service_mu[2]),
        service_sigma=(service_sigma[0], service_sigma[1], service_sigma[2]),
    )


def _em_iteration(
    observations: list[Observation],
    params: HMMParams,
    *,
    prior_params: HMMParams | None = None,
    prior_strength: float = 0.0,
) -> tuple[HMMParams, float]:
    """One E-step + M-step. Returns (new_params, log_likelihood_under_old_params).

    With `prior_params` and `prior_strength > 0`, the M-step uses Dirichlet
    pseudo-counts for the transition matrix + initial distribution, and the
    same prior threads into `_estimate_emissions`. Pure MLE when off.
    """
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
    kappa = max(prior_strength, 0.0)
    use_prior = prior_params is not None and kappa > 0.0

    # Initial distribution. With prior: π = (κ·prior_π + γ[0]) / (κ + 1).
    if use_prior:
        assert prior_params is not None
        pi_prior = prior_params.initial
        denom_pi = kappa + 1.0
        new_pi = (
            (kappa * pi_prior[0] + gamma[0][0]) / denom_pi,
            (kappa * pi_prior[1] + gamma[0][1]) / denom_pi,
            (kappa * pi_prior[2] + gamma[0][2]) / denom_pi,
        )
    else:
        new_pi = (gamma[0][0], gamma[0][1], gamma[0][2])

    # Transition rows. With prior: a[s][sp] = (κ·a_prior[s][sp] + ξ_sum[s][sp]) / (κ + Σ_t γ[t][s]).
    new_a_rows: list[tuple[float, float, float]] = []
    for s in range(N_STATES):
        denom = sum(gamma[t][s] for t in range(t_max - 1))
        if use_prior:
            assert prior_params is not None
            prior_row = prior_params.transition[s]
            total = kappa + denom
            row = [
                (kappa * prior_row[sp] + xi_sum[s][sp]) / total
                for sp in range(N_STATES)
            ]
        elif denom <= 0:
            new_a_rows.append(params.transition[s])
            continue
        else:
            row = [xi_sum[s][sp] / denom for sp in range(N_STATES)]
            rsum = sum(row) or 1.0
            row = [r / rsum for r in row]
        new_a_rows.append((row[0], row[1], row[2]))
    new_a = tuple(new_a_rows)

    if params.emissions_by_bin is None:
        new_emissions = _estimate_emissions(
            gamma,
            observations,
            range(t_max),
            params.emissions,
            prior=prior_params.emissions if prior_params else None,
            prior_strength=kappa,
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
                gamma,
                observations,
                buckets[b],
                params.emissions_by_bin[b],
                prior=prior_params.emissions_by_bin[b]
                if (prior_params and prior_params.emissions_by_bin)
                else None,
                prior_strength=kappa,
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


def fit_em(
    observations: list[Observation],
    initial_params: HMMParams,
    max_iterations: int = 50,
    tolerance: float = 1e-4,
    *,
    prior_params: HMMParams | None = None,
    prior_strength: float = 0.0,
) -> tuple[HMMParams, list[float]]:
    """Train per-entity HMM parameters from a sequence of observations.

    Uses Baum-Welch: forward-backward (scaled for numerical stability) for the
    E-step, weighted-MLE for the M-step. Converges when relative change in
    log-likelihood drops below `tolerance` or `max_iterations` is reached.

    Returns (fitted_params, log_likelihoods) where log_likelihoods is the
    sequence across iterations — useful for verifying monotonicity.

    When `prior_params` + `prior_strength > 0` are supplied, the M-step runs
    MAP estimation with conjugate priors (Dirichlet/Gamma/Beta pseudo-counts).
    Useful for empirical-Bayes per-entity training where a global fit acts as
    a regularizer for thin-data entities. `prior_strength` is in units of
    effective observations — `~100` is a sensible default for a daily corpus.

    The output is passed through canonicalize_states so the index<->label
    mapping (normal/disrupted/suspended) is stable across runs — there is
    exactly one ordering rule; don't add another. See momentarily-vk0.7.
    """
    if not observations:
        raise ValueError("fit_em requires at least one observation")

    params = initial_params
    log_liks: list[float] = []
    prev: float | None = None

    for _ in range(max_iterations):
        params, log_lik = _em_iteration(
            observations,
            params,
            prior_params=prior_params,
            prior_strength=prior_strength,
        )
        log_liks.append(log_lik)
        if prev is not None:
            denom = max(abs(prev), 1e-12)
            if abs(log_lik - prev) / denom < tolerance:
                break
        prev = log_lik

    return canonicalize_states(params), log_liks
