"""Cross-language contract guards.

Two things are mirrored by hand between Python and the TypeScript Worker and
will drift without a test pinning them:

  * the published Snapshot shape (Pydantic in src/momentarily/schema.py vs the
    hand-written interfaces in worker/src/snapshot.ts)
  * the HMM forward filter (src/momentarily/hmm.py vs worker/src/hmm.ts)

This module guards the *Python* side: the committed schema must match what
Pydantic emits today, and the committed parity fixture must match what the
forward filter produces today. worker/test/parity_python.test.ts guards the
TS side against the same fixture.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from momentarily.hmm import (
    EmissionParams,
    FilterState,
    HMMParams,
    Observation,
    expected_dwell_ticks,
    forward_step,
    initial_published_state,
    project_forward,
)
from scripts.export_schema import SCHEMA_PATH, render_schema
from scripts.gen_parity_fixture import FIXTURE_PATH

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_committed_schema_matches_pydantic() -> None:
    """schema/snapshot.schema.json must be regenerated when the Pydantic model
    changes. If this fails, run: uv run python -m scripts.export_schema"""
    committed = SCHEMA_PATH.read_text()
    assert committed == render_schema(), (
        "schema/snapshot.schema.json is stale — "
        "run `uv run python -m scripts.export_schema`"
    )


def _params_from_json(d: dict) -> HMMParams:
    return HMMParams(
        transition=tuple(tuple(row) for row in d["transition"]),
        initial=tuple(d["initial"]),
        emissions=EmissionParams(**d["emissions"]),
    )


def _obs_from_json(d: dict | None) -> Observation | None:
    return None if d is None else Observation(**d)


def test_parity_fixture_reproduces_python_forward_filter() -> None:
    """The committed parity fixture must match what hmm.py produces today.
    If this fails, run: uv run python -m scripts.gen_parity_fixture"""
    fixture = json.loads(FIXTURE_PATH.read_text())
    params = _params_from_json(fixture["params"])
    observations = [_obs_from_json(o) for o in fixture["observations"]]

    init = fixture["initial_filter"]
    state = FilterState(
        probabilities=tuple(init["probabilities"]),
        regime_entered_at=init["regime_entered_at"],
        last_updated_at=init["last_updated_at"],
    )
    published = initial_published_state(state)

    tick_seconds = fixture["tick_seconds"]
    start_now = fixture["start_now"]

    for i, (obs, expected) in enumerate(
        zip(observations, fixture["expected_steps"], strict=True)
    ):
        now = start_now + (i + 1) * tick_seconds
        state, published = forward_step(state, published, obs, params, now)
        ef = expected["filter"]
        assert state.probabilities == pytest.approx(ef["probabilities"], abs=1e-15), (
            f"step {i}: filter probabilities drifted"
        )
        assert state.regime_entered_at == ef["regime_entered_at"], f"step {i}"
        assert state.last_updated_at == ef["last_updated_at"], f"step {i}"
        ep = expected["published"]
        assert published.label == ep["label"], f"step {i}: published label"
        assert published.pending_state == ep["pending_state"], f"step {i}"
        assert published.pending_streak == ep["pending_streak"], f"step {i}"
        assert published.last_updated_at == ep["last_updated_at"], f"step {i}"

    for ticks, expected_probs in fixture["expected_projections"].items():
        projected = project_forward(state, params, int(ticks))
        assert projected == pytest.approx(expected_probs, abs=1e-15), (
            f"projection at {ticks} ticks drifted"
        )

    median, q25, q75 = expected_dwell_ticks(state, params)
    assert (median, q25, q75) == (
        fixture["expected_dwell"]["median"],
        fixture["expected_dwell"]["q25"],
        fixture["expected_dwell"]["q75"],
    )
