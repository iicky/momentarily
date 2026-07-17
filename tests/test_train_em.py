"""Unit tests for the per-route trainer (training/train_em.py).

Covers pooling + prior anchoring, the self-loop cap, and the R2 write paths
(live pointer + versioned snapshot) via a fake S3 client — no real R2.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo

import pytest

from momentarily.hmm import EmissionParams, HMMParams, Observation, schedule_bin
from training.eval import TransitionRecord
from training.load_r2 import (
    MIN_MATCHED_TRIPS,
    MIN_SCHEDULE_TICKS,
    P0_FLOOR,
    compute_schedule_rate,
    schedule_rate_to_json,
)
from training.r2_client import R2Config
from training.train_em import (
    MAX_SELF_LOOP,
    MIN_DATA_DAYS,
    PARAMS_KEY,
    SCHEMA_VERSION,
    VERSIONED_PARAMS_PREFIX,
    CorpusStats,
    _apply_advance_prior,  # pyright: ignore[reportPrivateUsage]
    _cap_self_loops,  # pyright: ignore[reportPrivateUsage]
    _movement_baseline,  # pyright: ignore[reportPrivateUsage]
    _params_to_json,  # pyright: ignore[reportPrivateUsage]
    _service_baseline,  # pyright: ignore[reportPrivateUsage]
    compute_advance_baseline_by_route,
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


def test_train_advance_priors_anchor_fitted_route_normal_state() -> None:
    """A fitted route with a measured baseline anchors advance_rate[0] to the
    fed rate. Observations carry has_movement=False, so EM never updates
    advance_rate from data — this is a pure pass-through of the fed prior."""
    fed_rate = 0.42
    series = {
        "OTHER": _quiet(200) + _noisy(200) + _quiet(200),
        "FED": _quiet(150),
    }
    _global_prior, per_route = train(
        series, min_ticks=100, prior_strength=10.0, advance_priors={"FED": fed_rate}
    )
    fitted = per_route["FED"]
    assert abs(fitted.emissions.advance_rate[0] - fed_rate) < 1e-9
    assert fitted.emissions.advance_rate[1] == _approx(0.3)
    assert fitted.emissions.advance_rate[2] == _approx(0.02)


def test_train_without_advance_priors_keeps_default_advance_rate() -> None:
    """A fitted route with no measured baseline keeps the hardcoded 0.6 default."""
    series = {
        "OTHER": _quiet(200) + _noisy(200) + _quiet(200),
        "FED": _quiet(150),
    }
    _global_prior, per_route = train(series, min_ticks=100, prior_strength=10.0)
    assert per_route["FED"].emissions.advance_rate == (0.6, 0.3, 0.02)


def test_train_advance_priors_thin_route_inherits_global_but_carries_fed_rate() -> None:
    """A thin route (< min_ticks) with a measured baseline inherits the global
    prior's shape but is a distinct object carrying the fed advance rate."""
    fed_rate = 0.42
    series = {
        "FAT": _quiet(200) + _noisy(200) + _quiet(200),
        "THIN": _quiet(5),
    }
    global_prior, per_route = train(
        series, min_ticks=100, prior_strength=10.0, advance_priors={"THIN": fed_rate}
    )
    thin = per_route["THIN"]
    assert thin is not global_prior
    assert abs(thin.emissions.advance_rate[0] - fed_rate) < 1e-9
    assert thin.transition == global_prior.transition
    assert thin.initial == global_prior.initial
    assert thin.emissions.advance_rate[1:] == global_prior.emissions.advance_rate[1:]


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


def test_apply_advance_prior_overrides_only_normal_state() -> None:
    """Only the normal (index 0) advance_rate changes; disrupted/suspended and
    every other emission channel pass through untouched."""
    params = _params_with_transition(
        ((0.95, 0.04, 0.01), (0.08, 0.90, 0.02), (0.02, 0.10, 0.88))
    )
    updated = _apply_advance_prior(params, 0.42)
    assert updated.emissions.advance_rate == (0.42, 0.3, 0.02)
    assert updated.emissions.poisson_lambda == params.emissions.poisson_lambda
    assert updated.emissions.gamma_alpha == params.emissions.gamma_alpha
    assert updated.emissions.gamma_beta == params.emissions.gamma_beta
    assert updated.emissions.bernoulli_p == params.emissions.bernoulli_p
    assert updated.transition == params.transition


def test_apply_advance_prior_overrides_every_bin_when_present() -> None:
    """emissions_by_bin, when present, gets the same normal-only override in
    every bin — the prior survives a TOD-conditioned model."""
    base = _params_with_transition(
        ((0.95, 0.04, 0.01), (0.08, 0.90, 0.02), (0.02, 0.10, 0.88))
    )
    by_bin = (
        replace(base.emissions, advance_rate=(0.5, 0.25, 0.01)),
        replace(base.emissions, advance_rate=(0.7, 0.35, 0.03)),
    )
    params = replace(base, emissions_by_bin=by_bin)
    updated = _apply_advance_prior(params, 0.42)
    assert updated.emissions_by_bin is not None
    assert [e.advance_rate for e in updated.emissions_by_bin] == [
        (0.42, 0.25, 0.01),
        (0.42, 0.35, 0.03),
    ]


def test_apply_advance_prior_leaves_missing_emissions_by_bin_as_none() -> None:
    """emissions_by_bin is None (no per-bin fit yet) stays None."""
    params = _params_with_transition(
        ((0.95, 0.04, 0.01), (0.08, 0.90, 0.02), (0.02, 0.10, 0.88))
    )
    updated = _apply_advance_prior(params, 0.42)
    assert updated.emissions_by_bin is None


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
    schedule_rate = {"A": {"wd06": 0.5}}
    write_params(
        cast("S3Client", fake),
        "test-bucket",
        _two_route_params(),
        corpus=corpus,
        n_routes_trained=1,
        hyperparams=hyperparams,
        schedule_rate=schedule_rate,
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
    assert doc["schedule_rate"] == schedule_rate


def test_compute_advance_baseline_by_route_pools_direction_and_tod() -> None:
    """Two directions collapse into one per-route median, distinct from either
    direction's own median — confirms pooling, not per-direction grain."""
    series: dict[tuple[str, str, int], dict[str, int]] = {}
    for t in range(15):
        series[("A", "north", t * 300)] = {"advanced_n": 9, "stalled_n": 1}
    for t in range(15):
        series[("A", "south", t * 300 + 500_000)] = {"advanced_n": 5, "stalled_n": 5}
    # north medians 0.9, south medians 0.5 -- pooled median is 0.7, neither.
    rates = compute_advance_baseline_by_route(series)
    assert set(rates) == {"A"}
    assert rates["A"] == _approx(0.7)


def test_compute_advance_baseline_by_route_drops_route_below_min_samples() -> None:
    """Fewer than min_samples qualifying ticks -> route omitted, no fabricated prior."""
    series: dict[tuple[str, str, int], dict[str, int]] = {
        ("THIN", "north", t * 300): {"advanced_n": 9, "stalled_n": 1} for t in range(19)
    }
    assert compute_advance_baseline_by_route(series) == {}


def test_compute_advance_baseline_by_route_clamps_to_p0_floor() -> None:
    """A route where every matched trip advanced clamps below 1.0, not exactly 1.0."""
    series: dict[tuple[str, str, int], dict[str, int]] = {
        ("PERFECT", "north", t * 300): {"advanced_n": 10, "stalled_n": 0}
        for t in range(20)
    }
    rates = compute_advance_baseline_by_route(series)
    assert set(rates) == {"PERFECT"}
    assert rates["PERFECT"] == _approx(1.0 - P0_FLOOR)


def test_compute_advance_baseline_by_route_ignores_ticks_below_min_matched() -> None:
    """Ticks with fewer than min_matched cross-tick matches are excluded from
    the median, even though they'd otherwise skew it toward 1.0."""
    series: dict[tuple[str, str, int], dict[str, int]] = {}
    for t in range(20):
        series[("A", "north", t * 300)] = {
            "advanced_n": 5,
            "stalled_n": 5,
        }  # matched=10
    for t in range(20, 40):
        # matched = MIN_MATCHED_TRIPS - 1 < min_matched — must not count.
        series[("A", "north", t * 300)] = {
            "advanced_n": MIN_MATCHED_TRIPS - 1,
            "stalled_n": 0,
        }
    rates = compute_advance_baseline_by_route(series)
    assert set(rates) == {"A"}
    assert rates["A"] == _approx(0.5)


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
    route_series_seen: list[dict[tuple[str, str, int], dict[str, int]]] = []

    sentinel_bodies: list[dict[str, Any]] = [{"marker": "body"}]
    sentinel_series: dict[tuple[str, str, int], dict[str, int]] = {
        ("A", "north", 0): {"vehicles_n": 1}
    }
    sentinel_baseline: dict[tuple[str, str, int], object] = {
        ("A", "north", 0): object(),
        ("A", "south", 300): object(),
    }
    sentinel_json: dict[str, Any] = {"A": {"north": {"0": {"p0": 0.9}}}}
    sentinel_route_rates: dict[str, float] = {"A": 0.9}

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

    def _fake_compute_baseline_by_route(
        series: dict[tuple[str, str, int], dict[str, int]],
    ) -> dict[str, float]:
        route_series_seen.append(series)
        return sentinel_route_rates

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
    monkeypatch.setattr(
        "training.train_em.compute_advance_baseline_by_route",
        _fake_compute_baseline_by_route,
    )

    cfg = _r2_config()
    client = cast("S3Client", _FakeS3())
    start = date(2026, 6, 1)
    end = date(2026, 6, 14)

    result, n_cells, route_rates = _movement_baseline(cfg, client, start, end)

    assert fetch_calls == [{"start_date": start, "end_date": end, "client": client}]
    assert bodies_seen == [sentinel_bodies]
    assert series_seen == [sentinel_series]
    assert route_series_seen == [sentinel_series]
    assert baseline_seen == [sentinel_baseline]
    assert result == sentinel_json
    assert n_cells == 2
    assert route_rates == sentinel_route_rates


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

    result, n_cells, route_rates = _movement_baseline(
        _r2_config(), cast("S3Client", _FakeS3()), date(2026, 6, 1), date(2026, 6, 14)
    )

    assert result == {}
    assert n_cells == 0
    assert route_rates == {}
    assert "movement baseline skipped" in capsys.readouterr().err


def test_service_baseline_uses_explicit_window_and_threads_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_service_baseline must fetch the *explicit* training window (not
    fetch_trip_update_metrics' yesterday..today default) and thread the
    build/compute/serialize chain's output straight through — including the
    schedule-rate chain, from that same single trip-updates fetch."""
    fetch_calls: list[dict[str, Any]] = []
    bodies_seen: list[list[dict[str, Any]]] = []
    series_seen: list[dict[tuple[str, int], int]] = []
    baseline_seen: list[dict[tuple[str, int], float]] = []
    schedule_bodies_seen: list[list[dict[str, Any]]] = []
    schedule_rate_seen: list[dict[tuple[str, str], float]] = []

    sentinel_bodies: list[dict[str, Any]] = [{"marker": "body"}]
    sentinel_series: dict[tuple[str, int], int] = {("A", 0): 6}
    sentinel_baseline: dict[tuple[str, int], float] = {
        ("A", 0): 6.0,
        ("A", 1): 5.0,
    }
    sentinel_json: dict[str, Any] = {"A": {"0": 6.0, "1": 5.0}}
    sentinel_schedule_rate: dict[tuple[str, str], float] = {("A", "wd06"): 0.5}
    sentinel_schedule_json: dict[str, Any] = {"A": {"wd06": 0.5}}

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
    ) -> dict[tuple[str, int], int]:
        bodies_seen.append(bodies)
        return sentinel_series

    def _fake_compute_baseline(
        series: dict[tuple[str, int], int],
    ) -> dict[tuple[str, int], float]:
        series_seen.append(series)
        return sentinel_baseline

    def _fake_to_json(
        baseline: dict[tuple[str, int], float],
    ) -> dict[str, Any]:
        baseline_seen.append(baseline)
        return sentinel_json

    def _fake_compute_schedule_rate(
        bodies: list[dict[str, Any]],
    ) -> dict[tuple[str, str], float]:
        schedule_bodies_seen.append(bodies)
        return sentinel_schedule_rate

    def _fake_schedule_rate_to_json(
        rate: dict[tuple[str, str], float],
    ) -> dict[str, Any]:
        schedule_rate_seen.append(rate)
        return sentinel_schedule_json

    monkeypatch.setattr("training.train_em.fetch_trip_update_metrics", _fake_fetch)
    monkeypatch.setattr("training.train_em.build_service_series", _fake_build_series)
    monkeypatch.setattr("training.train_em.compute_baseline", _fake_compute_baseline)
    monkeypatch.setattr("training.train_em.service_baseline_to_json", _fake_to_json)
    monkeypatch.setattr(
        "training.train_em.compute_schedule_rate", _fake_compute_schedule_rate
    )
    monkeypatch.setattr(
        "training.train_em.schedule_rate_to_json", _fake_schedule_rate_to_json
    )

    cfg = _r2_config()
    client = cast("S3Client", _FakeS3())
    start = date(2026, 6, 1)
    end = date(2026, 6, 14)

    result, n_cells, schedule_result, n_schedule_cells = _service_baseline(
        cfg, client, start, end
    )

    assert fetch_calls == [{"start_date": start, "end_date": end, "client": client}]
    assert bodies_seen == [sentinel_bodies]
    assert series_seen == [sentinel_series]
    assert baseline_seen == [sentinel_baseline]
    assert result == sentinel_json
    assert n_cells == 2
    # Schedule chain reuses the SAME fetch — not a second trip-updates call.
    assert schedule_bodies_seen == [sentinel_bodies]
    assert schedule_rate_seen == [sentinel_schedule_rate]
    assert schedule_result == sentinel_schedule_json
    assert n_schedule_cells == 1


def test_service_baseline_fails_soft_on_archive_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Any exception in the trip-updates archive chain must not propagate —
    the service channel is optional and a hiccup there can't block params
    publish."""

    def _raise_fetch(
        cfg: R2Config,
        *,
        start_date: date,
        end_date: date,
        client: object,
    ) -> list[dict[str, Any]]:
        raise RuntimeError("trip update archive unavailable")

    monkeypatch.setattr("training.train_em.fetch_trip_update_metrics", _raise_fetch)

    result, n_cells, schedule_result, n_schedule_cells = _service_baseline(
        _r2_config(), cast("S3Client", _FakeS3()), date(2026, 6, 1), date(2026, 6, 14)
    )

    assert result == {}
    assert n_cells == 0
    assert schedule_result == {}
    assert n_schedule_cells == 0
    assert "service baseline skipped" in capsys.readouterr().err


def _et_epoch(year: int, month: int, day: int, hour: int) -> int:
    return int(
        datetime(year, month, day, hour, tzinfo=ZoneInfo("America/New_York"))
        .astimezone(UTC)
        .timestamp()
    )


def _weekday_epochs(hour: int, n: int, *, start: date = date(2026, 7, 1)) -> list[int]:
    """n distinct epochs at `hour` ET on n different weekdays (Mon-Fri)."""
    out: list[int] = []
    day = start
    while len(out) < n:
        if day.weekday() < 5:
            out.append(_et_epoch(day.year, day.month, day.day, hour))
        day += timedelta(days=1)
    return out


def test_schedule_bin_weekday_hour() -> None:
    """Weekday ET hour maps to the wd{HH} bin — the brief's worked example."""
    assert schedule_bin(_et_epoch(2026, 7, 15, 6)) == "wd06"  # Wednesday 6am ET


def test_schedule_bin_weekend_hour() -> None:
    """Weekend (Sat/Sun) ET hour maps to the we{HH} bin."""
    assert schedule_bin(_et_epoch(2026, 7, 18, 22)) == "we22"  # Saturday 10pm ET


def test_compute_schedule_rate_is_ran_over_usable_ticks() -> None:
    """A route's in-service rate is exactly ran-ticks / usable-ticks."""
    epochs = _weekday_epochs(6, MIN_SCHEDULE_TICKS)
    bin_key = schedule_bin(epochs[0])
    bodies = [
        {"observed_at": ep, "rows": {"A": {"assigned_n": 1 if i < 15 else 0}}}
        for i, ep in enumerate(epochs)
    ]
    rate = compute_schedule_rate(bodies)
    assert rate[("A", bin_key)] == _approx(15 / MIN_SCHEDULE_TICKS)


def test_compute_schedule_rate_present_but_never_running_is_zero() -> None:
    """A route that's present in the feed all bin but never dispatches a
    train gets an explicit 0.0 rate, not omission — the cell exists because
    the route is scheduled there, it just never actually runs."""
    epochs = _weekday_epochs(7, MIN_SCHEDULE_TICKS)
    bin_key = schedule_bin(epochs[0])
    bodies = [{"observed_at": ep, "rows": {"B": {"assigned_n": 0}}} for ep in epochs]
    rate = compute_schedule_rate(bodies)
    assert rate[("B", bin_key)] == 0.0


def test_compute_schedule_rate_excludes_feed_outage_ticks_from_denominator() -> None:
    """A globally-empty tick (`rows: {}`) is a feed outage, not evidence the
    route wasn't running — it must not inflate the bin's denominator."""
    epochs = _weekday_epochs(8, MIN_SCHEDULE_TICKS + 5)
    bin_key = schedule_bin(epochs[0])
    usable, outage = epochs[:MIN_SCHEDULE_TICKS], epochs[MIN_SCHEDULE_TICKS:]
    bodies = [
        {"observed_at": ep, "rows": {"C": {"assigned_n": 1 if i < 10 else 0}}}
        for i, ep in enumerate(usable)
    ] + [{"observed_at": ep, "rows": {}} for ep in outage]
    rate = compute_schedule_rate(bodies)
    assert rate[("C", bin_key)] == _approx(10 / MIN_SCHEDULE_TICKS)


def test_compute_schedule_rate_omits_bin_below_min_ticks() -> None:
    """A (route, bin) with fewer than MIN_SCHEDULE_TICKS usable ticks is
    omitted entirely — callers must treat a missing rate as unknown, not 0."""
    epochs = _weekday_epochs(9, MIN_SCHEDULE_TICKS - 1)
    bin_key = schedule_bin(epochs[0])
    bodies = [{"observed_at": ep, "rows": {"D": {"assigned_n": 1}}} for ep in epochs]
    rate = compute_schedule_rate(bodies)
    assert ("D", bin_key) not in rate


def test_compute_schedule_rate_output_is_sorted() -> None:
    """Output keys come out `(route, bin)`-sorted regardless of dict/set
    iteration order, so params.json delivery is stable across runs."""
    epochs = _weekday_epochs(10, MIN_SCHEDULE_TICKS)
    bodies = [
        {
            "observed_at": ep,
            "rows": {
                "Z": {"assigned_n": 1},
                "A": {"assigned_n": 1},
                "M": {"assigned_n": 1},
            },
        }
        for ep in epochs
    ]
    rate = compute_schedule_rate(bodies)
    assert {route for route, _ in rate} == {"A", "M", "Z"}
    assert list(rate.keys()) == sorted(rate.keys())


def test_compute_schedule_rate_absent_bin_gets_explicit_zero() -> None:
    """A route that never appears in a bin's rows at all — not just present
    but idle — still gets an explicit rate-0 cell for that bin, not an
    omission. Route X only runs at wd06, route Y only at wd07; each must
    read 0.0 (not missing) at the other's bin, alongside a real 1.0 where it
    actually ran every tick."""
    wd06_epochs = _weekday_epochs(6, MIN_SCHEDULE_TICKS)
    wd07_epochs = _weekday_epochs(7, MIN_SCHEDULE_TICKS)
    bin06 = schedule_bin(wd06_epochs[0])
    bin07 = schedule_bin(wd07_epochs[0])
    bodies = [
        {"observed_at": ep, "rows": {"X": {"assigned_n": 1}}} for ep in wd06_epochs
    ] + [{"observed_at": ep, "rows": {"Y": {"assigned_n": 1}}} for ep in wd07_epochs]
    rate = compute_schedule_rate(bodies)
    assert rate[("X", bin07)] == 0.0
    assert rate[("Y", bin06)] == 0.0
    assert rate[("X", bin06)] == 1.0
    assert rate[("Y", bin07)] == 1.0


def test_schedule_rate_to_json_nests_route_then_bin() -> None:
    rate = {("A", "wd06"): 0.75, ("A", "we22"): 0.2, ("B", "wd06"): 0.1}
    assert schedule_rate_to_json(rate) == {
        "A": {"wd06": 0.75, "we22": 0.2},
        "B": {"wd06": 0.1},
    }


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
    ) -> tuple[dict[str, Any], int, dict[str, float]]:
        return sentinel_baseline, 3, {}

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


def test_main_passes_service_baseline_through_to_write_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for compute-but-forget-to-pass: main() must thread
    both the service baseline and the schedule rate it computes into the
    write_params call, not just log them. The service fake returns a
    4-tuple (movement's is a 3-tuple)."""
    cfg = _r2_config()
    fake_client = cast("S3Client", _FakeS3())
    series = {"R1": _quiet(10)}
    corpus = CorpusStats(
        start_tick=0,
        end_tick=MIN_DATA_DAYS * 86_400 + 1,
        n_observations=10,
    )
    sentinel_baseline: dict[str, Any] = {"SENTINEL_ROUTE": {"0": 6.0}}
    sentinel_schedule_rate: dict[str, Any] = {"SENTINEL_ROUTE": {"wd06": 0.5}}
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

    def _fake_service_baseline(
        cfg_arg: R2Config, client: S3Client, start_date: date, end_date: date
    ) -> tuple[dict[str, Any], int, dict[str, Any], int]:
        return sentinel_baseline, 4, sentinel_schedule_rate, 6

    def _fake_write_params(*args: Any, **kwargs: Any) -> str:
        captured_kwargs.update(kwargs)
        return "state/params/v1.json"

    monkeypatch.setattr("training.train_em.load_config", _fake_load_config)
    monkeypatch.setattr("training.train_em.make_client", _fake_make_client)
    monkeypatch.setattr(
        "training.train_em.load_series_by_route", _fake_load_series_by_route
    )
    monkeypatch.setattr("training.eval.load_transitions", _fake_load_transitions)
    monkeypatch.setattr("training.train_em._service_baseline", _fake_service_baseline)
    monkeypatch.setattr("training.train_em.write_params", _fake_write_params)

    exit_code = main(
        ["--start", "2026-06-01", "--end", "2026-06-14", "--allow-empty-baseline"]
    )

    assert exit_code == 0
    assert captured_kwargs["service_baseline"] == sentinel_baseline
    assert captured_kwargs["schedule_rate"] == sentinel_schedule_rate


def test_main_passes_advance_priors_through_to_train(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression guard for compute-but-forget-to-pass: main() must thread the
    per-route advance rates it measures into train(), not just log them."""
    cfg = _r2_config()
    fake_client = cast("S3Client", _FakeS3())
    series = {"R1": _quiet(10)}
    corpus = CorpusStats(
        start_tick=0,
        end_tick=MIN_DATA_DAYS * 86_400 + 1,
        n_observations=10,
    )
    sentinel_route_rates: dict[str, float] = {"R1": 0.42}
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
    ) -> tuple[dict[str, Any], int, dict[str, float]]:
        return {}, 0, sentinel_route_rates

    def _fake_train(
        series_by_route: dict[str, list[Observation]],
        *,
        prior_strength: float = 100.0,
        min_ticks: int = 288,
        advance_priors: dict[str, float] | None = None,
    ) -> tuple[HMMParams, dict[str, HMMParams]]:
        captured_kwargs["advance_priors"] = advance_priors
        global_prior = _params_with_transition(
            ((0.9, 0.08, 0.02), (0.1, 0.85, 0.05), (0.02, 0.13, 0.85))
        )
        return global_prior, dict.fromkeys(series_by_route, global_prior)

    def _fake_write_params(*args: Any, **kwargs: Any) -> str:
        return "state/params/v1.json"

    monkeypatch.setattr("training.train_em.load_config", _fake_load_config)
    monkeypatch.setattr("training.train_em.make_client", _fake_make_client)
    monkeypatch.setattr(
        "training.train_em.load_series_by_route", _fake_load_series_by_route
    )
    monkeypatch.setattr("training.eval.load_transitions", _fake_load_transitions)
    monkeypatch.setattr("training.train_em._movement_baseline", _fake_movement_baseline)
    monkeypatch.setattr("training.train_em.train", _fake_train)
    monkeypatch.setattr("training.train_em.write_params", _fake_write_params)

    exit_code = main(
        ["--start", "2026-06-01", "--end", "2026-06-14", "--allow-empty-baseline"]
    )

    assert exit_code == 0
    assert captured_kwargs["advance_priors"] == sentinel_route_rates


def test_main_refuses_empty_movement_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """0 baseline cells silently disables the movement-primary condition, so
    main() refuses to publish unless --allow-empty-baseline is passed."""
    cfg = _r2_config()
    fake_client = cast("S3Client", _FakeS3())
    series = {"R1": _quiet(10)}
    corpus = CorpusStats(
        start_tick=0, end_tick=MIN_DATA_DAYS * 86_400 + 1, n_observations=10
    )
    published: list[str] = []

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
    ) -> tuple[dict[str, Any], int, dict[str, float]]:
        return {}, 0, {}

    def _fake_write_params(*args: Any, **kwargs: Any) -> str:
        published.append("wrote")
        return "state/params/v1.json"

    monkeypatch.setattr("training.train_em.load_config", _fake_load_config)
    monkeypatch.setattr("training.train_em.make_client", _fake_make_client)
    monkeypatch.setattr(
        "training.train_em.load_series_by_route", _fake_load_series_by_route
    )
    monkeypatch.setattr("training.eval.load_transitions", _fake_load_transitions)
    monkeypatch.setattr("training.train_em._movement_baseline", _fake_movement_baseline)
    monkeypatch.setattr("training.train_em.write_params", _fake_write_params)

    assert main(["--start", "2026-06-01", "--end", "2026-06-14"]) == 1
    assert published == []
    assert (
        main(["--start", "2026-06-01", "--end", "2026-06-14", "--allow-empty-baseline"])
        == 0
    )
    assert published == ["wrote"]


def test_main_passes_dwell_by_cause_through_to_write_params(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """main() threads cause-conditioned dwell quantiles into write_params so the
    episode-recovery grader's dwell_quantiles_by_cause lookup resolves (1a6)."""
    cfg = _r2_config()
    fake_client = cast("S3Client", _FakeS3())
    series = {"R1": _quiet(10)}
    corpus = CorpusStats(
        start_tick=0, end_tick=MIN_DATA_DAYS * 86_400 + 1, n_observations=10
    )
    transitions = [
        TransitionRecord(
            ts=i * 10_000,
            route="R1",
            prev_state="disrupted",
            new_state="normal",
            regime_entered_at=i * 10_000 - 300 * i,
            exited_at=i * 10_000,
            dwell_sec=300 * i,
            alert_type_at_entry="Delays",
        )
        for i in range(1, 7)
    ]
    captured: dict[str, Any] = {}

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
        return transitions

    def _fake_movement_baseline(
        cfg_arg: R2Config, client: S3Client, start_date: date, end_date: date
    ) -> tuple[dict[str, Any], int, dict[str, float]]:
        return {}, 0, {}

    def _fake_write_params(*args: Any, **kwargs: Any) -> str:
        captured.update(kwargs)
        return "state/params/v1.json"

    monkeypatch.setattr("training.train_em.load_config", _fake_load_config)
    monkeypatch.setattr("training.train_em.make_client", _fake_make_client)
    monkeypatch.setattr(
        "training.train_em.load_series_by_route", _fake_load_series_by_route
    )
    monkeypatch.setattr("training.eval.load_transitions", _fake_load_transitions)
    monkeypatch.setattr("training.train_em._movement_baseline", _fake_movement_baseline)
    monkeypatch.setattr("training.train_em.write_params", _fake_write_params)

    exit_code = main(
        ["--start", "2026-06-01", "--end", "2026-06-14", "--allow-empty-baseline"]
    )
    assert exit_code == 0
    by_cause = captured["dwell_quantiles_by_cause"]
    assert "R1" in by_cause
    causes = by_cause["R1"]["disrupted"]
    assert len(causes) == 1
    assert next(iter(causes.values()))["n"] == 6
