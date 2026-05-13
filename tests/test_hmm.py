"""Tests for the HMM forward filter, projection, dwell prediction, and EM training.

Verifies the math against hand-constructed sequences where we know the answer.
"""

from __future__ import annotations

import math
import random

import pytest

from momentarily.hmm import (
    HYSTERESIS_TICKS,
    N_STATES,
    PUBLISHED_UNKNOWN,
    EmissionParams,
    FilterState,
    HMMParams,
    Observation,
    PublishedState,
    expected_dwell_ticks,
    fit_em,
    forward_step,
    forward_update,
    initial_published_state,
    project_forward,
)


def _default_params() -> HMMParams:
    """Hand-picked parameters with the regime separation we expect from real data.

    normal:    almost no alerts, severity ~0, suspended very rare
    disrupted: a handful of alerts, moderate severity, suspended uncommon
    suspended: many alerts, high severity, suspended-alert near-certain
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
    with pytest.raises(ValueError):
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
    for prev, curr in zip(log_liks, log_liks[1:]):
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

    # After _sort_states_by_lambda, state 0 < state 1 < state 2 in poisson_lambda.
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
    with pytest.raises(ValueError):
        fit_em([], _default_params())


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
    state, published = forward_step(
        state, published, _quiet_obs(), params, now=6 * 300
    )
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
    state, published = forward_step(
        state, published, _suspended_obs(), params, now=300
    )
    assert published.label == "normal"
    assert published.pending_state == "suspended"
    assert published.pending_streak == 1

    # Second suspended tick — published flips
    state, published = forward_step(
        state, published, _suspended_obs(), params, now=600
    )
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
    state, published = forward_step(
        state, published, _suspended_obs(), params, now=600
    )
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
