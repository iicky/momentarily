"""Cross-language contract guard for the movement classifier.

training/load_r2.py's classify_direction is the source of truth for the
per-(route, direction) Beta-Binomial movement call. worker/src/movement_state.ts
(classifyAdvance) and viz/lib/movement.ts (classifyDirection) are hand-ports of
the same arithmetic and will drift the moment a Python constant moves —
DISRUPTED_RATIO, CLASSIFY_ALPHA, CLASSIFY_PRIOR_STRENGTH, MIN_MATCHED_TRIPS —
without a shared fixture pinning them.

This module guards the *Python* side: the committed fixture
(tests/fixtures/parity_movement.json) must match what classify_direction
produces today. worker/test/movement_parity.test.ts and
viz/tests/movement_parity.test.ts guard the TS side against the same fixture.
"""

from __future__ import annotations

import json
from typing import Any, cast

from scripts.gen_movement_parity_fixture import FIXTURE_PATH
from training.load_r2 import AdvanceBaseline, classify_direction


def _baseline(p0: float | None) -> AdvanceBaseline | None:
    """The cell every consumer rebuilds from a case's p0 (see the generator).
    classify_direction reads only p0; alpha/beta/n complete the Beta prior."""
    if p0 is None:
        return None
    return AdvanceBaseline(p0=p0, n=50, alpha=50 * p0, beta=50 * (1.0 - p0))


def _call_case(case: dict[str, Any]) -> object:
    fn = cast("str", case["fn"])
    inputs = cast("dict[str, Any]", case["inputs"])
    if fn == "classify_direction":
        return classify_direction(
            inputs["advanced"],
            inputs["stalled"],
            _baseline(inputs["p0"]),
        )
    raise ValueError(f"unknown fixture fn: {fn}")


def test_movement_parity_fixture_reproduces_classify_direction() -> None:
    """The committed fixture must match what classify_direction produces today.
    If this fails, run: uv run python -m scripts.gen_movement_parity_fixture"""
    fixture = cast("dict[str, Any]", json.loads(FIXTURE_PATH.read_text()))
    cases = cast("list[dict[str, Any]]", fixture["cases"])
    assert cases, "fixture has no cases"
    for case in cases:
        actual = _call_case(case)
        label = case["label"]
        assert actual == case["expected"], f"{label} drifted: got {actual!r}"
