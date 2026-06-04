"""Unit tests for the per-route trainer (training/train_em.py).

Covers pooling + prior anchoring, the self-loop cap, and the R2 write paths
(live pointer + versioned snapshot) via a fake S3 client — no real R2.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import pytest

from momentarily.hmm import EmissionParams, HMMParams, Observation
from training.train_em import (
    MAX_SELF_LOOP,
    PARAMS_KEY,
    SCHEMA_VERSION,
    VERSIONED_PARAMS_PREFIX,
    CorpusStats,
    _cap_self_loops,  # pyright: ignore[reportPrivateUsage]
    _params_to_json,  # pyright: ignore[reportPrivateUsage]
    train,
    write_params,
)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


def _approx(expected: float) -> object:
    """Typed wrapper around ``pytest.approx``.

    pytest's ``approx`` leaks ``Unknown`` through its ``ApproxBase`` return type
    under strict mode, so we pin the boundary to ``object`` once here.
    """
    return pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]


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


def _params_with_transition(
    transition: tuple[tuple[float, float, float], ...],
) -> HMMParams:
    return HMMParams(
        transition=transition,
        initial=(0.9, 0.08, 0.02),
        emissions=EmissionParams(
            poisson_lambda=(0.3, 4.0, 12.0),
            gamma_alpha=(1.0, 3.0, 6.0),
            gamma_beta=(2.0, 0.4, 0.2),
            bernoulli_p=(0.001, 0.05, 0.95),
        ),
    )


def test_cap_self_loops_clamps_and_renormalizes() -> None:
    params = _params_with_transition(
        ((0.999, 0.0008, 0.0002), (0.08, 0.9, 0.02), (0.003, 0.002, 0.995))
    )
    capped = _cap_self_loops(params)
    for s in range(3):
        row = capped.transition[s]
        assert row[s] <= MAX_SELF_LOOP[s] + 1e-9, f"row {s} self-loop not capped: {row}"
        assert abs(sum(row) - 1.0) < 1e-9, f"row {s} not normalized: {row}"
    # Untouched row passes through unchanged (0.9 is below the disrupted cap).
    assert capped.transition[1] == (0.08, 0.9, 0.02)
    # Off-diagonal proportions are preserved when redistributing freed mass.
    assert capped.transition[0][1] / capped.transition[0][2] == _approx(4.0)


def test_cap_self_loops_handles_zero_off_diagonal() -> None:
    # A degenerate row [0, 0, 1] has no off-diagonal mass to scale — freed mass
    # must spread evenly instead of dividing by zero.
    capped = _cap_self_loops(
        _params_with_transition(
            ((0.95, 0.03, 0.02), (0.05, 0.93, 0.02), (0.0, 0.0, 1.0))
        )
    )
    row = capped.transition[2]
    assert row[2] == _approx(MAX_SELF_LOOP[2])
    assert row[0] == _approx((1.0 - MAX_SELF_LOOP[2]) / 2)
    assert row[1] == _approx((1.0 - MAX_SELF_LOOP[2]) / 2)


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
    assert body["transition"] == [
        [0.9, 0.08, 0.02],
        [0.1, 0.85, 0.05],
        [0.02, 0.13, 0.85],
    ]
    assert body["initial"] == [0.8, 0.15, 0.05]
    assert body["emissions"]["poisson_lambda"] == (0.3, 4.0, 12.0)
    # emissions_by_bin omitted when params.emissions_by_bin is None
    assert "emissions_by_bin" not in body


class _FakeS3:
    """Minimal stand-in for the boto3 S3 client — captures put_object calls."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_: object) -> None:
        self.objects[Key] = Body


def _two_route_params() -> dict[str, HMMParams]:
    return {
        "1": _params_with_transition(
            ((0.9, 0.08, 0.02), (0.1, 0.85, 0.05), (0.02, 0.13, 0.85))
        ),
        "A": _params_with_transition(
            ((0.92, 0.06, 0.02), (0.12, 0.83, 0.05), (0.03, 0.12, 0.85))
        ),
    }


def test_write_params_writes_live_and_versioned_keys() -> None:
    fake = _FakeS3()
    corpus = CorpusStats(
        start_tick=1_700_000_000, end_tick=1_701_209_600, n_observations=512
    )
    versioned_key = write_params(
        cast("S3Client", fake),
        "test-bucket",
        _two_route_params(),
        corpus=corpus,
        n_routes_trained=2,
        trained_at=1_701_300_000,
    )
    assert versioned_key == f"{VERSIONED_PARAMS_PREFIX}v1701300000.json"
    assert set(fake.objects) == {PARAMS_KEY, versioned_key}
    # Live pointer and versioned snapshot are byte-identical.
    assert fake.objects[PARAMS_KEY] == fake.objects[versioned_key]


def test_write_params_doc_shape_round_trips() -> None:
    fake = _FakeS3()
    corpus = CorpusStats(start_tick=100, end_tick=200, n_observations=7)
    write_params(
        cast("S3Client", fake),
        "test-bucket",
        _two_route_params(),
        corpus=corpus,
        n_routes_trained=1,
        trained_at=42,
    )
    doc = json.loads(fake.objects[PARAMS_KEY])
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["trained_at"] == 42
    assert doc["training_corpus"] == {
        "start_tick": 100,
        "end_tick": 200,
        "n_routes_trained": 1,
        "n_observations": 7,
    }
    assert set(doc["routes"]) == {"1", "A"}
    # Each route round-trips to the loose HMMParams shape the Worker reads.
    route = doc["routes"]["1"]
    assert len(route["transition"]) == 3
    assert len(route["initial"]) == 3
    assert "poisson_lambda" in route["emissions"]
