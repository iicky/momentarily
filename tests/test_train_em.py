"""Unit tests for the per-route trainer (training/train_em.py).

Covers pooling + prior anchoring, the self-loop cap, and the R2 write paths
(live pointer + versioned snapshot) via a fake S3 client — no real R2.
"""

from __future__ import annotations

import json
from datetime import date
from typing import TYPE_CHECKING, Any, cast

import pytest

from momentarily.hmm import EmissionParams, HMMParams, Observation
from training.eval import TransitionRecord
from training.r2_client import R2Config
from training.train_em import (
    MAX_SELF_LOOP,
    MIN_DATA_DAYS,
    PARAMS_KEY,
    SCHEMA_VERSION,
    VERSIONED_PARAMS_PREFIX,
    CorpusStats,
    _cap_self_loops,  # pyright: ignore[reportPrivateUsage]
    _movement_baseline,  # pyright: ignore[reportPrivateUsage]
    _params_to_json,  # pyright: ignore[reportPrivateUsage]
    main,
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
    corpus = CorpusStats(
        start_tick=100,
        end_tick=200,
        n_observations=7,
        n_input_versions=5,
        input_blake3="deadbeef",
    )
    hyperparams = {
        "window_start": "2026-06-01",
        "window_end": "2026-06-14",
        "prior_strength": 100.0,
        "min_ticks": 288,
        "routes": None,
    }
    write_params(
        cast("S3Client", fake),
        "test-bucket",
        _two_route_params(),
        corpus=corpus,
        n_routes_trained=1,
        hyperparams=hyperparams,
        trained_at=42,
    )
    doc = json.loads(fake.objects[PARAMS_KEY])
    assert doc["schema_version"] == SCHEMA_VERSION
    assert doc["trained_at"] == 42
    assert doc["hyperparams"] == hyperparams
    assert doc["training_corpus"] == {
        "start_tick": 100,
        "end_tick": 200,
        "n_routes_trained": 1,
        "n_observations": 7,
        "n_input_versions": 5,
        "input_blake3": "deadbeef",
    }
    assert set(doc["routes"]) == {"1", "A"}
    # Each route round-trips to the loose HMMParams shape the Worker reads.
    route = doc["routes"]["1"]
    assert len(route["transition"]) == 3
    assert len(route["initial"]) == 3
    assert "poisson_lambda" in route["emissions"]


def _r2_config() -> R2Config:
    return R2Config(
        account_id="acct",
        access_key_id="key",
        secret_access_key="secret",
        bucket="test-bucket",
    )


def test_movement_baseline_uses_explicit_window_and_threads_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_movement_baseline must fetch the *explicit* training window (not
    fetch_vehicle_metrics' yesterday..today default) and thread the
    build/compute/serialize chain's output straight through."""
    fetch_calls: list[dict[str, Any]] = []
    bodies_seen: list[list[dict[str, Any]]] = []
    series_seen: list[dict[tuple[str, str, int], dict[str, int]]] = []
    baseline_seen: list[dict[tuple[str, str, int], object]] = []

    sentinel_bodies: list[dict[str, Any]] = [{"marker": "body"}]
    sentinel_series: dict[tuple[str, str, int], dict[str, int]] = {
        ("A", "north", 0): {"vehicles_n": 1}
    }
    sentinel_baseline: dict[tuple[str, str, int], object] = {
        ("A", "north", 0): object(),
        ("A", "south", 300): object(),
    }
    sentinel_json: dict[str, Any] = {"A": {"north": {"0": {"p0": 0.9}}}}

    def _fake_fetch(
        cfg: R2Config,
        *,
        start_date: date,
        end_date: date,
        client: object,
    ) -> list[dict[str, Any]]:
        fetch_calls.append(
            {"start_date": start_date, "end_date": end_date, "client": client}
        )
        return sentinel_bodies

    def _fake_build_series(
        bodies: list[dict[str, Any]],
    ) -> dict[tuple[str, str, int], dict[str, int]]:
        bodies_seen.append(bodies)
        return sentinel_series

    def _fake_compute_baseline(
        series: dict[tuple[str, str, int], dict[str, int]],
    ) -> dict[tuple[str, str, int], object]:
        series_seen.append(series)
        return sentinel_baseline

    def _fake_to_json(
        baseline: dict[tuple[str, str, int], object],
    ) -> dict[str, Any]:
        baseline_seen.append(baseline)
        return sentinel_json

    monkeypatch.setattr("training.train_em.fetch_vehicle_metrics", _fake_fetch)
    monkeypatch.setattr(
        "training.train_em.build_movement_series_by_direction", _fake_build_series
    )
    monkeypatch.setattr(
        "training.train_em.compute_advance_baseline", _fake_compute_baseline
    )
    monkeypatch.setattr("training.train_em.advance_baseline_to_json", _fake_to_json)

    cfg = _r2_config()
    client = cast("S3Client", _FakeS3())
    start = date(2026, 6, 1)
    end = date(2026, 6, 14)

    result, n_cells = _movement_baseline(cfg, client, start, end)

    assert fetch_calls == [{"start_date": start, "end_date": end, "client": client}]
    assert bodies_seen == [sentinel_bodies]
    assert series_seen == [sentinel_series]
    assert baseline_seen == [sentinel_baseline]
    assert result == sentinel_json
    assert n_cells == 2


def test_movement_baseline_fails_soft_on_archive_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Any exception in the vehicle-archive chain must not propagate — the
    movement channel is optional and a hiccup there can't block params publish."""

    def _raise_fetch(
        cfg: R2Config,
        *,
        start_date: date,
        end_date: date,
        client: object,
    ) -> list[dict[str, Any]]:
        raise RuntimeError("vehicle archive unavailable")

    monkeypatch.setattr("training.train_em.fetch_vehicle_metrics", _raise_fetch)

    result, n_cells = _movement_baseline(
        _r2_config(), cast("S3Client", _FakeS3()), date(2026, 6, 1), date(2026, 6, 14)
    )

    assert result == {}
    assert n_cells == 0
    assert "movement baseline skipped" in capsys.readouterr().err


def test_main_passes_movement_baseline_through_to_write_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for compute-but-forget-to-pass: main() must thread the
    baseline it computes into the write_params call, not just log it."""
    cfg = _r2_config()
    fake_client = cast("S3Client", _FakeS3())
    series = {"R1": _quiet(10)}
    corpus = CorpusStats(
        start_tick=0,
        end_tick=MIN_DATA_DAYS * 86_400 + 1,
        n_observations=10,
    )
    sentinel_baseline: dict[str, Any] = {
        "SENTINEL_ROUTE": {
            "north": {"0": {"p0": 0.9, "alpha": 1.0, "beta": 1.0, "n": 1}}
        }
    }
    captured_kwargs: dict[str, Any] = {}

    def _fake_load_config() -> R2Config:
        return cfg

    def _fake_make_client(config: R2Config | None = None) -> S3Client:
        return fake_client

    def _fake_load_series_by_route(
        cfg_arg: R2Config, start: date, end: date
    ) -> tuple[dict[str, list[Observation]], CorpusStats, dict[str, Any]]:
        return series, corpus, {}

    def _fake_load_transitions(
        client: S3Client, bucket: str, start_date: date, end_date: date
    ) -> list[TransitionRecord]:
        return []

    def _fake_movement_baseline(
        cfg_arg: R2Config, client: S3Client, start_date: date, end_date: date
    ) -> tuple[dict[str, Any], int]:
        return sentinel_baseline, 3

    def _fake_write_params(*args: Any, **kwargs: Any) -> str:
        captured_kwargs.update(kwargs)
        return "state/params/v1.json"

    monkeypatch.setattr("training.train_em.load_config", _fake_load_config)
    monkeypatch.setattr("training.train_em.make_client", _fake_make_client)
    monkeypatch.setattr(
        "training.train_em.load_series_by_route", _fake_load_series_by_route
    )
    monkeypatch.setattr("training.eval.load_transitions", _fake_load_transitions)
    monkeypatch.setattr("training.train_em._movement_baseline", _fake_movement_baseline)
    monkeypatch.setattr("training.train_em.write_params", _fake_write_params)

    exit_code = main(["--start", "2026-06-01", "--end", "2026-06-14"])

    assert exit_code == 0
    assert captured_kwargs["movement_baseline"] == sentinel_baseline
