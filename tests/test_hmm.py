"""Tests for the HMM forward filter, projection, and dwell prediction.

Verifies the math against hand-constructed sequences where we know the answer.
"""

from __future__ import annotations

import math

import pytest

from momentarily.hmm import (
    N_STATES,
    EmissionParams,
    FilterState,
    HMMParams,
    Observation,
    expected_dwell_ticks,
    forward_update,
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
