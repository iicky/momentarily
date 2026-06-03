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
from typing import Any, cast

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


def _approx(expected: object, *, abs: float) -> object:
    """Typed wrapper around ``pytest.approx``.

    pytest ships ``approx`` as a partially-untyped helper (its ``ApproxBase``
    return type leaks ``Unknown`` under strict mode), so we pin a concrete
    ``object`` boundary here once instead of at every comparison site.
    """
    return pytest.approx(expected, abs=abs)  # pyright: ignore[reportUnknownMemberType]


def test_committed_schema_matches_pydantic() -> None:
    """schema/snapshot.schema.json must be regenerated when the Pydantic model
    changes. If this fails, run: uv run python -m scripts.export_schema"""
    committed = SCHEMA_PATH.read_text()
    assert committed == render_schema(), (
        "schema/snapshot.schema.json is stale — "
        "run `uv run python -m scripts.export_schema`"
    )


Triple = tuple[float, float, float]


def _triple(values: list[float]) -> Triple:
    """Coerce a 3-element JSON list into the fixed-width tuple the model expects."""
    a, b, c = values
    return (a, b, c)


def _params_from_json(d: dict[str, Any]) -> HMMParams:
    transition_rows = cast("list[list[float]]", d["transition"])
    initial = cast("list[float]", d["initial"])
    emissions = cast("dict[str, list[float]]", d["emissions"])
    return HMMParams(
        transition=tuple(_triple(row) for row in transition_rows),
        initial=_triple(initial),
        emissions=EmissionParams(
            **{field: _triple(values) for field, values in emissions.items()}
        ),
    )


def _obs_from_json(d: dict[str, Any] | None) -> Observation | None:
    return None if d is None else Observation(**d)


def test_parity_fixture_reproduces_python_forward_filter() -> None:
    """The committed parity fixture must match what hmm.py produces today.
    If this fails, run: uv run python -m scripts.gen_parity_fixture"""
    fixture = cast("dict[str, Any]", json.loads(FIXTURE_PATH.read_text()))
    params = _params_from_json(fixture["params"])
    observations = [
        _obs_from_json(o)
        for o in cast("list[dict[str, Any] | None]", fixture["observations"])
    ]

    init = cast("dict[str, Any]", fixture["initial_filter"])
    state = FilterState(
        probabilities=_triple(cast("list[float]", init["probabilities"])),
        regime_entered_at=init["regime_entered_at"],
        last_updated_at=init["last_updated_at"],
    )
    published = initial_published_state(state)

    tick_seconds = cast("int", fixture["tick_seconds"])
    start_now = cast("int", fixture["start_now"])

    expected_steps = cast("list[dict[str, Any]]", fixture["expected_steps"])
    for i, (obs, expected) in enumerate(zip(observations, expected_steps, strict=True)):
        now = start_now + (i + 1) * tick_seconds
        state, published = forward_step(state, published, obs, params, now)
        ef = cast("dict[str, Any]", expected["filter"])
        assert state.probabilities == _approx(
            cast("list[float]", ef["probabilities"]), abs=1e-15
        ), f"step {i}: filter probabilities drifted"
        assert state.regime_entered_at == ef["regime_entered_at"], f"step {i}"
        assert state.last_updated_at == ef["last_updated_at"], f"step {i}"
        ep = cast("dict[str, Any]", expected["published"])
        assert published.label == ep["label"], f"step {i}: published label"
        assert published.pending_state == ep["pending_state"], f"step {i}"
        assert published.pending_streak == ep["pending_streak"], f"step {i}"
        assert published.last_updated_at == ep["last_updated_at"], f"step {i}"

    projections = cast("dict[str, list[float]]", fixture["expected_projections"])
    for ticks, expected_probs in projections.items():
        projected = project_forward(state, params, int(ticks))
        assert projected == _approx(expected_probs, abs=1e-15), (
            f"projection at {ticks} ticks drifted"
        )

    median, q25, q75 = expected_dwell_ticks(state, params)
    expected_dwell = cast("dict[str, int]", fixture["expected_dwell"])
    assert (median, q25, q75) == (
        expected_dwell["median"],
        expected_dwell["q25"],
        expected_dwell["q75"],
    )
