"""Tests for the HMM forward filter, projection, dwell prediction, and EM training.

Verifies the math against hand-constructed sequences where we know the answer.
"""

from __future__ import annotations

import math
import random
from itertools import pairwise

import pytest

from momentarily.hmm import (
    HYSTERESIS_TICKS,
    N_STATES,
    N_TOD_BINS,
    PUBLISHED_UNKNOWN,
    EmissionParams,
    FilterState,
    HMMParams,
    Observation,
    PublishedState,
    canonicalize_states,
    expected_dwell_ticks,
    fit_em,
    forward_step,
    forward_update,
    initial_published_state,
    project_forward,
    tod_bin,
)


def _default_params() -> HMMParams:
    """Hand-picked parameters with the regime separation we expect from real data.

    normal:    almost no alerts, severity ~0, all flags rare
    disrupted: a handful of alerts, moderate severity, delays/changes common
    suspended: many alerts, high severity, suspended/no-service near-certain
    """
    return HMMParams(
        transition=(
            (0.95, 0.04, 0.01),
            (0.08, 0.90, 0.02),
            (0.02, 0.10, 0.88),
        ),
        initial=(0.9, 0.08, 0.02),
        emissions=EmissionParams(
            poisson_lambda=(0.3, 4.0, 12.0),
            gamma_alpha=(1.0, 3.0, 6.0),
            gamma_beta=(2.0, 0.4, 0.2),
            bernoulli_p=(0.001, 0.05, 0.95),
            bernoulli_p_delays=(0.01, 0.45, 0.5),
            bernoulli_p_service_change=(0.01, 0.5, 0.6),
            bernoulli_p_planned=(0.05, 0.3, 0.4),
        ),
    )


def _flat_state() -> FilterState:
    return FilterState(
        probabilities=(1.0 / 3, 1.0 / 3, 1.0 / 3),
        regime_entered_at=0,
        last_updated_at=0,
    )


def test_posterior_sums_to_one() -> None:
    params = _default_params()
    state = _flat_state()
    obs = Observation(alert_count=2, severity_sum=10, has_suspended_alert=False)
    updated = forward_update(state, obs, params, now=100)
    assert math.isclose(sum(updated.probabilities), 1.0, abs_tol=1e-9)


def test_quiet_observation_pulls_toward_normal() -> None:
    """No alerts, no severity, no suspended → posterior heavily favors normal."""
    params = _default_params()
    state = _flat_state()
    obs = Observation(alert_count=0, severity_sum=0, has_suspended_alert=False)
    updated = forward_update(state, obs, params, now=100)
    p_normal, p_disrupted, p_suspended = updated.probabilities
    assert p_normal > p_disrupted > p_suspended
    assert p_normal > 0.8


def test_suspended_alert_pulls_toward_suspended() -> None:
    """High alert count + high severity + suspended flag → posterior favors suspended."""
    params = _default_params()
    state = _flat_state()
    obs = Observation(alert_count=15, severity_sum=80, has_suspended_alert=True)
    updated = forward_update(state, obs, params, now=100)
    p_normal, p_disrupted, p_suspended = updated.probabilities
    assert p_suspended > p_disrupted > p_normal
    assert p_suspended > 0.6


def test_regime_entered_at_advances_on_state_change() -> None:
    """When argmax shifts between ticks, regime_entered_at moves to `now`."""
    params = _default_params()
    state = FilterState(
        probabilities=(0.95, 0.04, 0.01),
        regime_entered_at=100,
        last_updated_at=100,
    )
    # A disruption-favoring observation
    obs = Observation(alert_count=8, severity_sum=50, has_suspended_alert=False)
    # Two consecutive disruption ticks should flip argmax to disrupted.
    s1 = forward_update(state, obs, params, now=200)
    s2 = forward_update(s1, obs, params, now=300)
    assert s2.probabilities[1] > s2.probabilities[0]  # disrupted > normal
    # regime_entered_at moved to the tick where argmax changed.
    assert s1.regime_entered_at == 200 or s2.regime_entered_at == 300


def test_regime_entered_at_holds_when_state_unchanged() -> None:
    """Consecutive ticks in the same regime don't bump regime_entered_at."""
    params = _default_params()
    state = FilterState(
        probabilities=(0.95, 0.04, 0.01),
        regime_entered_at=100,
        last_updated_at=100,
    )
    quiet = Observation(alert_count=0, severity_sum=0, has_suspended_alert=False)
    s1 = forward_update(state, quiet, params, now=200)
    s2 = forward_update(s1, quiet, params, now=300)
    assert s2.regime_entered_at == 100


def test_project_forward_zero_ticks_is_identity() -> None:
    params = _default_params()
    state = _flat_state()
    assert project_forward(state, params, 0) == state.probabilities


def test_project_forward_converges_to_stationary() -> None:
    """A long projection lands near the chain's stationary distribution."""
    params = _default_params()
    state = FilterState(
        probabilities=(0.0, 1.0, 0.0),  # start fully in disrupted
        regime_entered_at=0,
        last_updated_at=0,
    )
    far = project_forward(state, params, ticks_ahead=500)
    # Stationary is dominated by normal because A favors returning to it.
    assert far[0] > far[1] > far[2]
    assert math.isclose(sum(far), 1.0, abs_tol=1e-9)


def test_project_forward_negative_ticks_rejected() -> None:
    with pytest.raises(ValueError, match="ticks_ahead must be >= 0"):
        project_forward(_flat_state(), _default_params(), ticks_ahead=-1)


def test_expected_dwell_high_self_loop_yields_long_stay() -> None:
    """Strong self-loop in 'normal' (0.95) → expected dwell well above 1 tick."""
    params = _default_params()
    state = FilterState(
        probabilities=(0.99, 0.005, 0.005),  # firmly in normal
        regime_entered_at=0,
        last_updated_at=0,
    )
    median, low, high = expected_dwell_ticks(state, params)
    # ceil(log(0.5)/log(0.95)) = 14
    assert median == 14
    # 25th and 75th bracket the median
    assert low < median < high


def test_expected_dwell_disrupted_state() -> None:
    params = _default_params()
    state = FilterState(
        probabilities=(0.1, 0.85, 0.05),
        regime_entered_at=0,
        last_updated_at=0,
    )
    median, _, _ = expected_dwell_ticks(state, params)
    # ceil(log(0.5)/log(0.90)) = 7
    assert median == 7


def test_state_dimensionality() -> None:
    """Tuple lengths must match N_STATES — guard against silent schema drift."""
    state = _flat_state()
    assert len(state.probabilities) == N_STATES
    params = _default_params()
    assert len(params.transition) == N_STATES
    assert all(len(row) == N_STATES for row in params.transition)


# ---------------------------------------------------------------------------
# Baum-Welch EM
# ---------------------------------------------------------------------------


def _generate_synthetic_sequence(
    true_params: HMMParams, length: int, seed: int = 42
) -> list[Observation]:
    """Sample a length-T observation sequence from a known HMM.

    Deterministic given a seed so test failures are reproducible.
    """
    rng = random.Random(seed)
    # Sample state sequence by transition Markov chain
    states: list[int] = []
    weights = list(true_params.initial)
    states.append(rng.choices(range(N_STATES), weights=weights, k=1)[0])
    for _ in range(length - 1):
        prev = states[-1]
        row = list(true_params.transition[prev])
        states.append(rng.choices(range(N_STATES), weights=row, k=1)[0])

    em = true_params.emissions
    obs: list[Observation] = []
    for s in states:
        alert_count = _sample_poisson(rng, em.poisson_lambda[s])
        severity = round(_sample_gamma(rng, em.gamma_alpha[s], em.gamma_beta[s]))
        suspended = rng.random() < em.bernoulli_p[s]
        obs.append(
            Observation(
                alert_count=alert_count,
                severity_sum=max(0, int(severity)),
                has_suspended_alert=suspended,
            )
        )
    return obs


def _sample_poisson(rng: random.Random, lam: float) -> int:
    """Knuth's algorithm; fine for the small λ we use in tests."""
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p < L:
            return k - 1


def _sample_gamma(rng: random.Random, alpha: float, beta: float) -> float:
    """Marsaglia–Tsang for shape ≥ 1, Ahrens–Dieter for shape < 1.
    Python's random.gammavariate uses shape & scale; we use shape & rate.
    """
    if alpha <= 0 or beta <= 0:
        return 0.0
    # gammavariate(alpha, scale) — scale = 1 / rate
    return rng.gammavariate(alpha, 1.0 / beta)


def test_em_likelihood_improves_overall() -> None:
    """EM improves the model overall. Strict per-step monotonicity doesn't hold
    because we use method-of-moments for Gamma (not the true MLE), making this
    a generalized EM. Tiny step-to-step wiggles are expected; we assert overall
    improvement and no catastrophic regression.
    """
    true_params = _default_params()
    obs = _generate_synthetic_sequence(true_params, length=300, seed=0)
    init = HMMParams(
        transition=(
            (0.5, 0.3, 0.2),
            (0.3, 0.4, 0.3),
            (0.2, 0.3, 0.5),
        ),
        initial=(1.0 / 3, 1.0 / 3, 1.0 / 3),
        emissions=EmissionParams(
            poisson_lambda=(1.0, 5.0, 10.0),
            gamma_alpha=(2.0, 2.0, 2.0),
            gamma_beta=(1.0, 1.0, 1.0),
            bernoulli_p=(0.1, 0.3, 0.7),
        ),
    )
    _fitted, log_liks = fit_em(obs, init, max_iterations=20, tolerance=1e-8)
    assert log_liks[-1] > log_liks[0], (
        f"EM did not improve likelihood: {log_liks[0]} → {log_liks[-1]}"
    )
    # No single step should regress by more than a tiny amount (Gamma MoM noise).
    for prev, curr in pairwise(log_liks):
        assert curr >= prev - 1e-2, f"catastrophic regression: {prev} → {curr}"


def test_em_recovers_state_ordering_on_synthetic_data() -> None:
    """EM recovers the qualitative regime structure: quiet/medium/noisy ordering."""
    true_params = _default_params()
    obs = _generate_synthetic_sequence(true_params, length=2000, seed=7)
    init = HMMParams(
        transition=(
            (0.8, 0.15, 0.05),
            (0.15, 0.7, 0.15),
            (0.05, 0.15, 0.8),
        ),
        initial=(0.6, 0.3, 0.1),
        emissions=EmissionParams(
            poisson_lambda=(1.0, 5.0, 10.0),
            gamma_alpha=(1.5, 2.5, 4.0),
            gamma_beta=(1.0, 0.5, 0.3),
            bernoulli_p=(0.05, 0.2, 0.7),
        ),
    )
    fitted, _ = fit_em(obs, init, max_iterations=40, tolerance=1e-5)

    # After canonicalize_states, state 0 is the quietest; this synthetic data
    # aligns the suspended flag with the busiest cluster, so lambda is monotone.
    lam = fitted.emissions.poisson_lambda
    assert lam[0] < lam[1] < lam[2], f"states not sorted by quietness: {lam}"

    # Bernoulli p should increase with state index (suspended ↔ noisy regime).
    p = fitted.emissions.bernoulli_p
    assert p[0] < p[2], f"suspended probability not increasing: {p}"

    # Transition rows must be valid stochastic — each sums to 1.
    for row in fitted.transition:
        assert math.isclose(sum(row), 1.0, abs_tol=1e-6)


def test_em_converges_within_max_iterations() -> None:
    """With a reasonable init, EM stops before hitting max_iterations."""
    true_params = _default_params()
    obs = _generate_synthetic_sequence(true_params, length=500, seed=3)
    _fitted, log_liks = fit_em(obs, true_params, max_iterations=100, tolerance=1e-4)
    assert len(log_liks) < 100, (
        f"expected convergence well before max iter, took {len(log_liks)}"
    )


def test_em_single_observation_doesnt_crash() -> None:
    """Edge case: training on one tick should still produce valid params."""
    obs = [Observation(alert_count=5, severity_sum=30, has_suspended_alert=False)]
    fitted, _ = fit_em(obs, _default_params(), max_iterations=5)
    # Just verify shape integrity — no specific param values are meaningful.
    assert math.isclose(sum(fitted.initial), 1.0, abs_tol=1e-6)
    for row in fitted.transition:
        assert math.isclose(sum(row), 1.0, abs_tol=1e-6)


def test_em_empty_observations_rejected() -> None:
    with pytest.raises(ValueError, match="requires at least one observation"):
        fit_em([], _default_params())


def test_em_quiet_corpus_does_not_collapse_normal_emission() -> None:
    """Regression: a quiet corpus (severity_sum=0 in most ticks) used to
    drive var → 0 and gamma_alpha → 250k, turning the normal state into a
    delta function. The forward filter then refused any deviating
    observation. The variance + Bernoulli floors must keep both bounded.
    See momentarily-p8y.
    """
    quiet = Observation(
        alert_count=0,
        severity_sum=0,
        has_suspended_alert=False,
        has_delays=False,
        has_service_change=False,
        has_planned=False,
    )
    burst = Observation(
        alert_count=5,
        severity_sum=120,
        has_suspended_alert=True,
        has_delays=True,
        has_service_change=False,
        has_planned=False,
    )
    obs = [quiet] * 280 + [burst] * 8 + [quiet] * 280
    fitted, _ = fit_em(obs, _default_params(), max_iterations=30)

    em = fitted.emissions
    assert max(em.gamma_alpha) <= 100.0 + 1e-6, (
        f"gamma_alpha unbounded: {em.gamma_alpha}"
    )
    for p in (
        em.bernoulli_p,
        em.bernoulli_p_delays,
        em.bernoulli_p_service_change,
        em.bernoulli_p_planned,
    ):
        assert min(p) >= 1e-3 - 1e-9, f"Bernoulli below floor: {p}"
        assert max(p) <= 1.0 - 1e-3 + 1e-9, f"Bernoulli above ceiling: {p}"


# ---------------------------------------------------------------------------
# Prior-anchored EM (empirical-Bayes)
# ---------------------------------------------------------------------------


def test_em_prior_strength_zero_matches_pure_mle() -> None:
    """prior_strength=0 must give identical result to no-prior call."""
    true_params = _default_params()
    obs = _generate_synthetic_sequence(true_params, length=200, seed=42)
    no_prior, _ = fit_em(obs, true_params, max_iterations=10, tolerance=1e-9)
    with_prior_strength_zero, _ = fit_em(
        obs,
        true_params,
        max_iterations=10,
        tolerance=1e-9,
        prior_params=true_params,
        prior_strength=0.0,
    )
    for s in range(3):
        assert math.isclose(
            no_prior.emissions.poisson_lambda[s],
            with_prior_strength_zero.emissions.poisson_lambda[s],
            rel_tol=1e-9,
        )
        for sp in range(3):
            assert math.isclose(
                no_prior.transition[s][sp],
                with_prior_strength_zero.transition[s][sp],
                rel_tol=1e-9,
            )


def test_em_strong_prior_pulls_emissions_toward_prior() -> None:
    """Heavy prior strength on a short series should leave emissions close to
    the prior — the prior's pseudo-counts dominate the data's actual counts."""
    prior = _default_params()
    # Generate a sequence from a *different* true model so MLE drifts away.
    drift = HMMParams(
        transition=prior.transition,
        initial=prior.initial,
        emissions=EmissionParams(
            poisson_lambda=(0.05, 1.0, 3.0),  # all states quieter than prior
            gamma_alpha=prior.emissions.gamma_alpha,
            gamma_beta=prior.emissions.gamma_beta,
            bernoulli_p=prior.emissions.bernoulli_p,
        ),
    )
    obs = _generate_synthetic_sequence(drift, length=50, seed=1)

    no_prior_fit, _ = fit_em(obs, prior, max_iterations=20, tolerance=1e-6)
    heavy_prior_fit, _ = fit_em(
        obs,
        prior,
        max_iterations=20,
        tolerance=1e-6,
        prior_params=prior,
        prior_strength=10_000.0,
    )

    # Distance from prior in λ space — heavy prior should be much closer.
    def lam_dist(p: HMMParams) -> float:
        return sum(
            abs(p.emissions.poisson_lambda[s] - prior.emissions.poisson_lambda[s])
            for s in range(3)
        )

    assert lam_dist(heavy_prior_fit) < lam_dist(no_prior_fit) * 0.1, (
        f"heavy prior didn't anchor: heavy={lam_dist(heavy_prior_fit)}, "
        f"none={lam_dist(no_prior_fit)}"
    )


def test_em_prior_returns_prior_when_data_thin() -> None:
    """One observation with prior → emissions stay at the prior (MIN_EFFECTIVE_OBS guard)."""
    prior = _default_params()
    obs = [Observation(alert_count=0, severity_sum=0, has_suspended_alert=False)]
    fitted, _ = fit_em(
        obs, prior, max_iterations=5, prior_params=prior, prior_strength=100.0
    )
    # canonicalize_states may reorder; just compare the sorted tuples.
    assert sorted(fitted.emissions.poisson_lambda) == sorted(
        prior.emissions.poisson_lambda
    )


# ---------------------------------------------------------------------------
# Hysteresis + Unknown (forward_step)
# ---------------------------------------------------------------------------


def _quiet_obs() -> Observation:
    return Observation(alert_count=0, severity_sum=0, has_suspended_alert=False)


def _suspended_obs() -> Observation:
    return Observation(alert_count=15, severity_sum=80, has_suspended_alert=True)


def test_hysteresis_suppresses_single_tick_flicker() -> None:
    """A one-tick blip mid-quiet shouldn't bump the published state."""
    params = _default_params()
    state = FilterState(
        probabilities=(0.95, 0.04, 0.01),
        regime_entered_at=0,
        last_updated_at=0,
    )
    published = initial_published_state(state)
    assert published.label == "normal"

    # Long quiet streak — published stays normal
    for t in range(1, 5):
        state, published = forward_step(
            state, published, _quiet_obs(), params, now=t * 300
        )
    assert published.label == "normal"

    # One-tick blip: looks suspended, but only one tick
    state, published = forward_step(
        state, published, _suspended_obs(), params, now=5 * 300
    )
    # Posterior almost certainly argmaxes to suspended now, but published stays
    # normal because pending_streak just reset to 1
    assert published.label == "normal", (
        f"single-tick blip bumped published state: {published}"
    )

    # Back to quiet — pending resets to normal
    state, published = forward_step(state, published, _quiet_obs(), params, now=6 * 300)
    assert published.label == "normal"


def test_hysteresis_publishes_after_sustained_change() -> None:
    """Two consecutive suspended ticks should flip the published label."""
    params = _default_params()
    state = FilterState(
        probabilities=(0.95, 0.04, 0.01),
        regime_entered_at=0,
        last_updated_at=0,
    )
    published = initial_published_state(state)

    # First suspended tick — pending advances but published holds
    state, published = forward_step(state, published, _suspended_obs(), params, now=300)
    assert published.label == "normal"
    assert published.pending_state == "suspended"
    assert published.pending_streak == 1

    # Second suspended tick — published flips
    state, published = forward_step(state, published, _suspended_obs(), params, now=600)
    assert published.label == "suspended"
    assert published.pending_streak >= HYSTERESIS_TICKS


def test_feed_gap_publishes_unknown_without_corrupting_alpha() -> None:
    """obs=None preserves the posterior and surfaces Unknown."""
    params = _default_params()
    state = FilterState(
        probabilities=(0.95, 0.04, 0.01),
        regime_entered_at=0,
        last_updated_at=0,
    )
    published = initial_published_state(state)

    pre_alpha = state.probabilities
    new_state, published = forward_step(state, published, None, params, now=300)

    assert published.label == PUBLISHED_UNKNOWN
    assert new_state.probabilities == pre_alpha, "alpha was modified during gap"


def test_publish_immediately_after_unknown() -> None:
    """First real obs after a gap publishes immediately (no extra hysteresis lag)."""
    params = _default_params()
    state = FilterState(
        probabilities=(0.95, 0.04, 0.01),
        regime_entered_at=0,
        last_updated_at=0,
    )
    published = initial_published_state(state)

    # Feed gap
    state, published = forward_step(state, published, None, params, now=300)
    assert published.label == PUBLISHED_UNKNOWN

    # Real suspended observation — should publish "suspended" right away
    state, published = forward_step(state, published, _suspended_obs(), params, now=600)
    assert published.label == "suspended"


def test_published_state_does_not_mutate_filter_math() -> None:
    """forward_step on a normal observation produces same FilterState as forward_update."""
    params = _default_params()
    state = FilterState(
        probabilities=(0.95, 0.04, 0.01),
        regime_entered_at=0,
        last_updated_at=0,
    )
    obs = _suspended_obs()

    direct = forward_update(state, obs, params, now=300)
    via_step, _published = forward_step(
        state, initial_published_state(state), obs, params, now=300
    )

    assert direct.probabilities == via_step.probabilities
    assert direct.regime_entered_at == via_step.regime_entered_at
    assert direct.last_updated_at == via_step.last_updated_at


def test_published_state_type_safety() -> None:
    """Initial PublishedState has the right shape."""
    state = _flat_state()
    p = initial_published_state(state)
    assert isinstance(p, PublishedState)
    assert p.label in ("normal", "disrupted", "suspended", PUBLISHED_UNKNOWN)


# ---------------------------------------------------------------------------
# Per-alert-type Bernoulli emissions
# ---------------------------------------------------------------------------


def test_planned_alerts_can_distinguish_overnight_from_real_disruption() -> None:
    """Two routes with identical alert_count and severity but different alert
    types — one planned, one real disruption — should produce different posteriors."""
    params = _default_params()
    state = _flat_state()

    # Overnight planned work: lots of alerts but all Planned
    planned = Observation(
        alert_count=8,
        severity_sum=60,
        has_suspended_alert=False,
        has_delays=False,
        has_service_change=False,
        has_planned=True,
    )
    p_state = forward_update(state, planned, params, now=300)

    # Real-time disruption: same shape but no planned flag, with delays + suspension
    real = Observation(
        alert_count=8,
        severity_sum=60,
        has_suspended_alert=True,
        has_delays=True,
        has_service_change=True,
        has_planned=False,
    )
    r_state = forward_update(state, real, params, now=300)

    # Posteriors should differ — the channels carry signal
    assert p_state.probabilities != r_state.probabilities

    # Real disruption should pull harder toward suspended than planned does
    assert r_state.probabilities[2] > p_state.probabilities[2]


def test_em_recovers_distinct_alert_type_profiles() -> None:
    """EM should learn distinct Bernoulli p's per state when synthetic data
    encodes the asymmetry."""
    true_params = HMMParams(
        transition=(
            (0.95, 0.04, 0.01),
            (0.08, 0.90, 0.02),
            (0.02, 0.10, 0.88),
        ),
        initial=(0.5, 0.3, 0.2),
        emissions=EmissionParams(
            poisson_lambda=(0.2, 3.0, 9.0),
            gamma_alpha=(1.0, 3.0, 6.0),
            gamma_beta=(2.0, 0.5, 0.3),
            bernoulli_p=(0.01, 0.10, 0.80),  # suspended-alert
            bernoulli_p_delays=(0.05, 0.60, 0.30),  # delays peak in disrupted
            bernoulli_p_service_change=(0.02, 0.40, 0.20),
            bernoulli_p_planned=(0.10, 0.20, 0.10),
        ),
    )
    obs = _generate_synthetic_sequence(true_params, length=1500, seed=11)
    init = HMMParams(
        transition=(
            (0.8, 0.15, 0.05),
            (0.15, 0.7, 0.15),
            (0.05, 0.15, 0.8),
        ),
        initial=(0.6, 0.3, 0.1),
        emissions=EmissionParams(
            poisson_lambda=(1.0, 4.0, 8.0),
            gamma_alpha=(1.5, 2.5, 4.0),
            gamma_beta=(1.0, 0.5, 0.3),
            bernoulli_p=(0.1, 0.3, 0.7),
            bernoulli_p_delays=(0.1, 0.4, 0.4),
            bernoulli_p_service_change=(0.1, 0.4, 0.4),
            bernoulli_p_planned=(0.1, 0.2, 0.2),
        ),
    )
    fitted, _ = fit_em(obs, init, max_iterations=40, tolerance=1e-5)

    # After sort-by-lambda, suspended-state probability should be highest in state 2
    assert fitted.emissions.bernoulli_p[2] > fitted.emissions.bernoulli_p[0]
    # Delays should peak in the disrupted state (highest in middle, not extremes)
    # — at minimum, it shouldn't be lowest in state 1 (disrupted)
    delays = fitted.emissions.bernoulli_p_delays
    assert delays[1] >= delays[0] - 0.1, (
        f"delays in disrupted dropped below normal: {delays}"
    )


# ---------------------------------------------------------------------------
# Time-of-day conditioning
# ---------------------------------------------------------------------------


def test_tod_bin_covers_full_24_hours() -> None:
    """Every UTC hour maps to some valid bin in [0, N_TOD_BINS)."""
    seen: set[int] = set()
    for hour in range(24):
        epoch = hour * 3600  # midnight + N hours UTC
        b = tod_bin(epoch)
        assert 0 <= b < N_TOD_BINS
        seen.add(b)
    # All bins should be exercised across the 24 hours
    assert seen == set(range(N_TOD_BINS))


def test_emissions_by_bin_path_routes_correctly() -> None:
    """Forward filter uses the bin's EmissionParams when emissions_by_bin is set."""
    quiet = EmissionParams(
        poisson_lambda=(0.1, 0.2, 0.3),
        gamma_alpha=(1.0, 1.0, 1.0),
        gamma_beta=(2.0, 2.0, 2.0),
        bernoulli_p=(0.01, 0.05, 0.10),
    )
    busy = EmissionParams(
        poisson_lambda=(5.0, 8.0, 12.0),
        gamma_alpha=(2.0, 4.0, 6.0),
        gamma_beta=(0.5, 0.3, 0.2),
        bernoulli_p=(0.10, 0.50, 0.95),
    )
    per_bin = tuple([quiet] + [busy] * (N_TOD_BINS - 1))
    params = HMMParams(
        transition=(
            (0.95, 0.04, 0.01),
            (0.08, 0.90, 0.02),
            (0.02, 0.10, 0.88),
        ),
        initial=(1 / 3, 1 / 3, 1 / 3),
        emissions=quiet,
        emissions_by_bin=per_bin,
    )

    # bin=0 (quiet emissions) — many alerts should look anomalous → pull to non-normal
    obs_busy_in_quiet_bin = Observation(
        alert_count=10, severity_sum=80, has_suspended_alert=True, tod_bin=0
    )
    s = forward_update(_flat_state(), obs_busy_in_quiet_bin, params, now=100)
    # bin=1 (busy emissions) — same observation should look normal-for-bin → less extreme
    obs_busy_in_busy_bin = Observation(
        alert_count=10, severity_sum=80, has_suspended_alert=True, tod_bin=1
    )
    t = forward_update(_flat_state(), obs_busy_in_busy_bin, params, now=100)

    # Posteriors must differ — confirms the bin lookup actually changes behavior
    assert s.probabilities != t.probabilities


def test_em_learns_per_bin_emissions() -> None:
    """EM with emissions_by_bin re-estimates each bin from the observations it saw."""
    # Synthetic data: TOD 0 is dominated by state 2 (busy), TOD 1 by state 0 (quiet)
    rng = random.Random(42)
    obs: list[Observation] = []
    for _ in range(800):
        bin_idx = rng.choice([0, 1])
        if bin_idx == 0:
            obs.append(
                Observation(
                    alert_count=rng.randint(8, 15),
                    severity_sum=rng.randint(50, 150),
                    has_suspended_alert=True,
                    has_planned=True,
                    tod_bin=0,
                )
            )
        else:
            obs.append(
                Observation(
                    alert_count=0,
                    severity_sum=0,
                    has_suspended_alert=False,
                    tod_bin=1,
                )
            )

    seed_em = EmissionParams(
        poisson_lambda=(1.0, 4.0, 8.0),
        gamma_alpha=(1.5, 2.5, 4.0),
        gamma_beta=(1.0, 0.5, 0.3),
        bernoulli_p=(0.1, 0.3, 0.7),
    )
    init = HMMParams(
        transition=(
            (0.8, 0.15, 0.05),
            (0.15, 0.7, 0.15),
            (0.05, 0.15, 0.8),
        ),
        initial=(1 / 3, 1 / 3, 1 / 3),
        emissions=seed_em,
        emissions_by_bin=tuple([seed_em] * N_TOD_BINS),
    )
    fitted, _ = fit_em(obs, init, max_iterations=30, tolerance=1e-5)

    assert fitted.emissions_by_bin is not None
    bin0 = fitted.emissions_by_bin[0]
    bin1 = fitted.emissions_by_bin[1]
    # Bin 0 saw busy data → its high-lambda state should be MUCH higher than
    # bin 1's, because bin 1 saw only quiet observations.
    assert bin0.poisson_lambda[2] > bin1.poisson_lambda[2] + 5.0


def test_observation_defaults_back_compat() -> None:
    """Old call sites without the new boolean flags still work."""
    obs = Observation(alert_count=3, severity_sum=20, has_suspended_alert=False)
    assert obs.has_delays is False
    assert obs.has_service_change is False
    assert obs.has_planned is False
    # forward_update accepts it
    params = _default_params()
    state = _flat_state()
    new_state = forward_update(state, obs, params, now=100)
    assert math.isclose(sum(new_state.probabilities), 1.0, abs_tol=1e-9)


def _scrambled_params() -> HMMParams:
    """EM label-switch: the quiet cluster sits on the `disrupted` index (idx1),
    the busiest on `normal` (idx0). Mirrors real routes 1/3/D/Q. Channels are
    ordered (idx0=mid, idx1=quiet, idx2=busy).
    """
    return HMMParams(
        transition=(
            (0.97, 0.02, 0.01),
            (0.03, 0.97, 0.0),
            (0.02, 0.01, 0.97),
        ),
        initial=(0.8, 0.15, 0.05),
        emissions=EmissionParams(
            poisson_lambda=(10.0, 0.02, 34.0),
            gamma_alpha=(3.0, 1.0, 6.0),
            gamma_beta=(0.4, 2.0, 0.2),
            bernoulli_p=(0.03, 0.01, 0.9),
        ),
    )


def test_canonicalize_puts_quiet_cluster_on_normal() -> None:
    canon = canonicalize_states(_scrambled_params())
    lam = canon.emissions.poisson_lambda
    # normal is now the quietest, suspended the busiest, monotonic in between.
    assert lam[0] < lam[1] < lam[2]
    assert math.isclose(lam[0], 0.02)  # the old idx1 quiet cluster
    # suspended owns the suspended-alert flag.
    assert canon.emissions.bernoulli_p[2] == max(canon.emissions.bernoulli_p)


def test_canonicalize_is_a_pure_relabel() -> None:
    """An all-quiet observation should land on `normal` after canonicalization,
    and probabilities still sum to 1 (it's a relabeling, not a refit)."""
    canon = canonicalize_states(_scrambled_params())
    quiet = Observation(alert_count=0, severity_sum=0, has_suspended_alert=False)
    state = FilterState(
        probabilities=(1 / 3, 1 / 3, 1 / 3), regime_entered_at=0, last_updated_at=0
    )
    post = forward_update(state, quiet, canon, now=100)
    assert math.isclose(sum(post.probabilities), 1.0, abs_tol=1e-9)
    assert post.probabilities[0] == max(post.probabilities)  # quiet -> normal


def test_canonicalize_noop_when_already_ordered() -> None:
    params = _default_params()  # normal<disrupted<suspended already
    assert canonicalize_states(params) is params


def test_canonicalize_quiet_but_flagged_cluster_is_normal() -> None:
    """The case where the two pre-consolidation ordering rules disagreed: the
    quietest cluster carries a small suspended-flag rate (overnight blips)
    while a busy planned-spam cluster never trips the flag. Sorting by
    bernoulli_p first put the spam cluster on `normal`; the consolidated rule
    keys normal on alert rate. See momentarily-vk0.7."""
    params = HMMParams(
        transition=(
            (0.97, 0.02, 0.01),
            (0.03, 0.97, 0.0),
            (0.02, 0.01, 0.97),
        ),
        initial=(0.8, 0.15, 0.05),
        emissions=EmissionParams(
            poisson_lambda=(8.0, 0.05, 3.0),  # idx0 = planned spam, idx1 = quiet
            gamma_alpha=(3.0, 1.0, 6.0),
            gamma_beta=(0.4, 2.0, 0.2),
            bernoulli_p=(0.02, 0.10, 0.9),  # quiet cluster flags more than spam
        ),
    )
    canon = canonicalize_states(params)
    assert math.isclose(canon.emissions.poisson_lambda[0], 0.05)  # quiet → normal
    assert math.isclose(canon.emissions.bernoulli_p[2], 0.9)  # flag → suspended
    assert math.isclose(canon.emissions.poisson_lambda[1], 8.0)  # spam → disrupted


def test_canonicalize_ranks_across_tod_bins() -> None:
    """With emissions_by_bin set, ranking must use per-state sums across bins —
    bin 0 alone (which aliases .emissions) can mislead."""
    # Bin 0 says idx0 is busier than idx1; the other four bins say the
    # opposite, loudly. Summed: idx1 is the quiet cluster.
    bin0 = EmissionParams(
        poisson_lambda=(5.0, 2.0, 9.0),
        gamma_alpha=(1.0, 1.0, 1.0),
        gamma_beta=(1.0, 1.0, 1.0),
        bernoulli_p=(0.9, 0.01, 0.4),
    )
    rest = EmissionParams(
        poisson_lambda=(5.0, 0.1, 9.0),
        gamma_alpha=(1.0, 1.0, 1.0),
        gamma_beta=(1.0, 1.0, 1.0),
        bernoulli_p=(0.9, 0.01, 0.4),
    )
    params = HMMParams(
        transition=(
            (0.97, 0.02, 0.01),
            (0.03, 0.97, 0.0),
            (0.02, 0.01, 0.97),
        ),
        initial=(0.8, 0.15, 0.05),
        emissions=bin0,
        emissions_by_bin=(bin0, rest, rest, rest, rest),
    )
    canon = canonicalize_states(params)
    assert canon.emissions_by_bin is not None
    # normal = summed-quietest (old idx1); suspended = summed-highest flag (old idx0).
    assert math.isclose(canon.emissions_by_bin[1].poisson_lambda[0], 0.1)
    assert math.isclose(canon.emissions_by_bin[1].bernoulli_p[2], 0.9)
