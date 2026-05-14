"""Unit tests for the per-route trainer (training/train_em.py).

Pure math — no R2. Verifies pooling + prior anchoring behave as advertised.
"""

from __future__ import annotations

from momentarily.hmm import EmissionParams, HMMParams, Observation
from training.train_em import _params_to_json, train


def _quiet(n: int) -> list[Observation]:
    return [
        Observation(
            alert_count=0,
            severity_sum=0,
            has_suspended_alert=False,
            has_delays=False,
            has_service_change=False,
            has_planned=False,
            tod_bin=0,
        )
    ] * n


def _noisy(n: int) -> list[Observation]:
    return [
        Observation(
            alert_count=8,
            severity_sum=40,
            has_suspended_alert=False,
            has_delays=True,
            has_service_change=False,
            has_planned=False,
            tod_bin=0,
        )
    ] * n


def test_train_assigns_global_prior_to_thin_routes() -> None:
    series = {
        "FAT": _quiet(200) + _noisy(200) + _quiet(200),  # rich enough
        "THIN": _quiet(5),  # under MIN_TICKS — should fall back to prior
    }
    global_prior, per_route = train(series, min_ticks=100, prior_strength=10.0)
    assert per_route["THIN"] == global_prior
    assert per_route["FAT"] is not global_prior  # was actually fitted


def test_train_pools_observations_for_global() -> None:
    """The global prior should reflect both routes — neither extreme."""
    series = {
        "QUIET": _quiet(500),
        "NOISY": _noisy(500),
    }
    global_prior, _per_route = train(series, min_ticks=100, prior_strength=10.0)
    # Global λ in the "active" state should sit somewhere between pure quiet (0)
    # and pure noisy (~8) — i.e. it actually learned from both.
    active_lams = sorted(global_prior.emissions.poisson_lambda)
    assert active_lams[-1] > 1.0, f"top λ unrealistically low: {active_lams}"


def test_params_to_json_round_trip_shape() -> None:
    params = HMMParams(
        transition=((0.9, 0.08, 0.02), (0.1, 0.85, 0.05), (0.02, 0.13, 0.85)),
        initial=(0.8, 0.15, 0.05),
        emissions=EmissionParams(
            poisson_lambda=(0.3, 4.0, 12.0),
            gamma_alpha=(1.0, 3.0, 6.0),
            gamma_beta=(2.0, 0.4, 0.2),
            bernoulli_p=(0.001, 0.05, 0.95),
        ),
    )
    body = _params_to_json(params)
    assert body["transition"] == [[0.9, 0.08, 0.02], [0.1, 0.85, 0.05], [0.02, 0.13, 0.85]]
    assert body["initial"] == [0.8, 0.15, 0.05]
    assert body["emissions"]["poisson_lambda"] == (0.3, 4.0, 12.0)
    # emissions_by_bin omitted when params.emissions_by_bin is None
    assert "emissions_by_bin" not in body
