"""Transactional publish ordering + the params plausibility gate (train_em.py).

Covers the two-phase publish (params.json flips last; a PROV write failure aborts
before any live pointer moves) and the plausibility gate (a degenerate params blob
is refused against the currently-live one, with a named reason).
"""

from __future__ import annotations

import json
from datetime import date
from typing import TYPE_CHECKING, Any, cast

import pytest
from botocore.exceptions import ClientError

from momentarily.hmm import EmissionParams, HMMParams, Observation
from training.eval import PredictionRecord, TransitionRecord
from training.load import TICK_SECONDS
from training.publish_params import (
    PARAMS_KEY,
    PROV_KEY,
    PUBLIC_PROV_KEY,
    SERVICE_BASELINE_KEY,
    VERSIONED_PARAMS_PREFIX,
    VERSIONED_PROV_PREFIX,
    CorpusStats,
    build_params_doc,
    implausible_params,
)
from training.r2_client import R2Config
from training.segment_dwell import SegmentDwellStats
from training.train_em import (
    MIN_DATA_DAYS,
    MovementInputs,
    ServiceInputs,
    main,
)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

_NORMAL = (
    (0.9, 0.08, 0.02),
    (0.1, 0.85, 0.05),
    (0.02, 0.13, 0.85),
)
_IDENTITY = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def _params(
    transition: tuple[tuple[float, float, float], ...],
    *,
    poisson_lambda: tuple[float, float, float] = (0.3, 4.0, 12.0),
) -> HMMParams:
    return HMMParams(
        transition=transition,
        initial=(0.9, 0.08, 0.02),
        emissions=EmissionParams(
            poisson_lambda=poisson_lambda,
            gamma_alpha=(1.0, 3.0, 6.0),
            gamma_beta=(2.0, 0.4, 0.2),
            bernoulli_p=(0.001, 0.05, 0.95),
        ),
    )


def _doc(per_route: dict[str, HMMParams], *, n_routes_trained: int) -> dict[str, Any]:
    return build_params_doc(
        per_route,
        corpus=CorpusStats(start_tick=100, end_tick=200, n_observations=10),
        n_routes_trained=n_routes_trained,
        trained_at=42,
    )


# --- plausibility gate ---------------------------------------------------


def test_collapsed_transition_matrix_refused_with_named_reason() -> None:
    """A degenerate, non-empty fit — a route whose transition matrix has collapsed
    to an absorbing state — is refused, and the reason names the check and route."""
    live = _doc({"1": _params(_NORMAL), "A": _params(_NORMAL)}, n_routes_trained=2)
    new = _doc({"1": _params(_IDENTITY), "A": _params(_NORMAL)}, n_routes_trained=2)
    assert implausible_params(new, live) == "collapsed_transition:1"


def test_normal_refit_passes() -> None:
    """A well-formed fit that matches the live regime mix is admitted."""
    live = _doc({"1": _params(_NORMAL), "A": _params(_NORMAL)}, n_routes_trained=2)
    new = _doc({"1": _params(_NORMAL), "A": _params(_NORMAL)}, n_routes_trained=2)
    assert implausible_params(new, live) is None


def test_non_finite_emission_refused() -> None:
    """A non-finite emission (a NaN Poisson rate) is a value-bound violation,
    refused regardless of what was serving before."""
    live = _doc({"1": _params(_NORMAL)}, n_routes_trained=1)
    new = _doc(
        {"1": _params(_NORMAL, poisson_lambda=(float("nan"), 4.0, 12.0))},
        n_routes_trained=1,
    )
    assert implausible_params(new, live) == "non_finite:1:poisson_lambda"


def test_stationary_distribution_shift_refused() -> None:
    """A well-formed matrix whose stationary regime mix lurches away from the live
    one (here: from mostly-normal to mostly-suspended) is refused as a different
    model, not a refit."""
    live = _doc({"1": _params(_NORMAL), "A": _params(_NORMAL)}, n_routes_trained=2)
    suspended_heavy = (
        (0.02, 0.13, 0.85),
        (0.02, 0.13, 0.85),
        (0.02, 0.13, 0.85),
    )
    new = _doc(
        {"1": _params(suspended_heavy), "A": _params(_NORMAL)}, n_routes_trained=2
    )
    assert implausible_params(new, live) == "stationary_shift:1"


def test_prior_fallback_surge_refused() -> None:
    """A run where the routes that inherited the global prior surge (the window
    went thin) is refused even when each route is individually well-formed."""
    live = _doc({"1": _params(_NORMAL), "A": _params(_NORMAL)}, n_routes_trained=2)
    new = _doc({"1": _params(_NORMAL), "A": _params(_NORMAL)}, n_routes_trained=0)
    assert implausible_params(new, live) == "prior_fallback_surge"


# --- transactional publish ordering --------------------------------------


class _RecordingS3:
    """A fake S3 client that records put order and can seed a live params.json."""

    def __init__(self, seed: dict[str, bytes] | None = None) -> None:
        self.objects: dict[str, bytes] = dict(seed or {})
        self.order: list[str] = []

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_: object) -> None:
        self.objects[Key] = Body
        self.order.append(Key)

    def get_object(self, *, Bucket: str, Key: str, **_: object) -> dict[str, Any]:
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        body = self.objects[Key]

        class _Body:
            def read(self) -> bytes:
                return body

        return {"Body": _Body()}

    def list_objects_v2(self, **_: object) -> dict[str, object]:
        return {"Contents": [], "IsTruncated": False}


def _r2_config() -> R2Config:
    return R2Config(
        account_id="acct",
        access_key_id="key",
        secret_access_key="secret",
        bucket="test-bucket",
    )


def _quiet(n: int) -> list[Observation]:
    return [
        Observation(
            alert_count=0,
            severity_sum=0,
            has_suspended_alert=False,
            tod_bin=(i * TICK_SECONDS // 3600) % 24,
        )
        for i in range(n)
    ]


def _install_publish_stubs(
    monkeypatch: pytest.MonkeyPatch, client: _RecordingS3
) -> None:
    """Wire main() onto the fake client with the archive/topology reads stubbed
    out, leaving write_params, write_service_baseline and write_prov real so the
    two-phase ordering is exercised end to end. The other sidecars are stubbed to
    no-ops (they neither publish nor defer a pointer)."""
    series: dict[str, list[Observation]] = {"R1": _quiet(10)}
    corpus = CorpusStats(
        start_tick=0, end_tick=MIN_DATA_DAYS * 86_400 + 1, n_observations=10
    )

    def _make_client(config: R2Config | None = None) -> S3Client:
        return cast("S3Client", client)

    def static_topology() -> tuple[None, None, str]:
        return None, None, "observed"

    def _load_series(
        cfg_arg: R2Config, start: date, end: date, **_: object
    ) -> tuple[dict[str, list[Observation]], CorpusStats, dict[str, Any]]:
        return series, corpus, {}

    def _fetch_feed() -> tuple[bytes, Any] | None:
        return None

    def _load_transitions(
        s3: S3Client, bucket: str, start_date: date, end_date: date
    ) -> list[TransitionRecord]:
        return []

    def _load_predictions(
        s3: S3Client, bucket: str, start_date: date, end_date: date
    ) -> list[PredictionRecord]:
        return []

    def _fake_service_baseline(
        cfg_arg: R2Config, s3: S3Client, start_date: date, end_date: date
    ) -> ServiceInputs:
        return ServiceInputs(
            baseline_json={"R1": {"0": 6.0}},
            n_cells=1,
            schedule_json={},
            n_schedule=0,
            hourly_json={"R1": {"wd06": 6.0}},
            n_hourly=1,
            hourly_quantiles_json={},
            n_hourly_quantiles=0,
            baseline_by_cell={},
            service_by_tick={},
        )

    def _fake_movement_baseline(
        cfg_arg: R2Config,
        s3: S3Client,
        start_date: date,
        end_date: date,
        through: frozenset[tuple[str, str, str]] | None,
    ) -> MovementInputs:
        return MovementInputs({}, 0, {}, set(), {})

    def _skip_segment_params(*_a: Any, **_k: Any) -> int:
        return 0

    def _skip_segment_dwell(*_a: Any, **_k: Any) -> tuple[int, SegmentDwellStats]:
        return 0, SegmentDwellStats(
            n_cells_own=0, n_cells_route=0, n_cells_system=0, n_cells_skipped=0
        )

    def _skip_scheduled_headway(*_a: Any, **_k: Any) -> int:
        return 0

    monkeypatch.setattr("training.train_em.load_config", _r2_config)
    monkeypatch.setattr("training.train_em.make_client", _make_client)
    monkeypatch.setattr("training.train_em.static_topology", static_topology)
    monkeypatch.setattr("training.train_em.load_series_by_route", _load_series)
    monkeypatch.setattr("training.train_em.fetch_gtfs_feed", _fetch_feed)
    monkeypatch.setattr("training.eval.load_transitions", _load_transitions)
    monkeypatch.setattr("training.eval.load_predictions", _load_predictions)
    monkeypatch.setattr("training.train_em._service_baseline", _fake_service_baseline)
    monkeypatch.setattr("training.train_em._movement_baseline", _fake_movement_baseline)
    monkeypatch.setattr("training.train_em.write_segment_params", _skip_segment_params)
    monkeypatch.setattr("training.train_em.write_segment_dwell", _skip_segment_dwell)
    monkeypatch.setattr(
        "training.train_em.write_scheduled_headway", _skip_scheduled_headway
    )


def test_params_pointer_flips_after_sidecars_and_prov() -> None:
    """state/params.json is the last live pointer written, and the immutable PROV
    doc it references is durable before it flips — so a reader mid-publish never
    sees params.json newer than its sidecars or pointing at a missing lineage."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        client = _RecordingS3()
        _install_publish_stubs(monkeypatch, client)
        exit_code = main(
            ["--start", "2026-06-01", "--end", "2026-06-14", "--allow-empty-baseline"]
        )

    assert exit_code == 0
    # The live params pointer is the final write of the whole run.
    assert client.order[-1] == PARAMS_KEY
    # The service sidecar's live pointer flipped before params.json.
    assert client.order.index(SERVICE_BASELINE_KEY) < client.order.index(PARAMS_KEY)
    # The immutable versioned params snapshot and the versioned PROV doc are both
    # durable before the params pointer flips.
    versioned_params = next(
        k for k in client.order if k.startswith(VERSIONED_PARAMS_PREFIX)
    )
    versioned_prov = next(
        k for k in client.order if k.startswith(VERSIONED_PROV_PREFIX) and k != PROV_KEY
    )
    assert client.order.index(versioned_params) < client.order.index(PARAMS_KEY)
    assert client.order.index(versioned_prov) < client.order.index(PARAMS_KEY)


def test_prov_write_failure_aborts_before_any_pointer_flips() -> None:
    """A PROV write failure aborts the publish before any live pointer moves: the
    previous run stays live and no published artifact points at a lineage doc that
    was never written."""
    with pytest.MonkeyPatch.context() as monkeypatch:
        client = _RecordingS3()
        _install_publish_stubs(monkeypatch, client)

        def _boom(*_a: object, **_k: object) -> str:
            raise RuntimeError("simulated prov outage")

        monkeypatch.setattr("training.train_em.write_prov", _boom)
        exit_code = main(
            ["--start", "2026-06-01", "--end", "2026-06-14", "--allow-empty-baseline"]
        )

    assert exit_code == 1
    # No live pointer was flipped — every published artifact still names the prior
    # run, so none carries a prov_ref to the lineage doc that was never written.
    for live_key in (PARAMS_KEY, SERVICE_BASELINE_KEY, PROV_KEY, PUBLIC_PROV_KEY):
        assert live_key not in client.objects


def test_gate_refuses_collapsed_fit_against_live_blob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end: with a live params.json present and the gate on, a run whose fit
    collapsed is refused and the live blob is left untouched (no versioned snapshot
    written either)."""
    live_doc_bytes = json.dumps(
        _doc({"R1": _params(_NORMAL)}, n_routes_trained=1)
    ).encode()
    client = _RecordingS3(seed={PARAMS_KEY: live_doc_bytes})
    _install_publish_stubs(monkeypatch, client)

    def _fake_train(
        series_by_route: dict[str, list[Observation]], **_k: Any
    ) -> tuple[HMMParams, dict[str, HMMParams]]:
        return _params(_NORMAL), dict.fromkeys(series_by_route, _params(_IDENTITY))

    monkeypatch.setattr("training.train_em.train", _fake_train)

    exit_code = main(
        ["--start", "2026-06-01", "--end", "2026-06-14", "--allow-empty-baseline"]
    )

    assert exit_code == 1
    # The live pointer is untouched and no versioned snapshot was written.
    assert client.objects[PARAMS_KEY] == live_doc_bytes
    assert not any(k.startswith(VERSIONED_PARAMS_PREFIX) for k in client.objects)
