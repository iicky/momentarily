"""Cross-language contract guard for the severity-graded published condition.

src/momentarily/mapping.py severity_tier and training/recovery_baseline.py
derive_graded_mta_state are the source of truth for the alert-derived
route_status.condition the Worker publishes. worker/src/mapping.ts
(severityTier / deriveGradedMtaState) is a hand-port pinned against the same
fixture; this module guards the Python side so the committed fixture cannot go
stale relative to the definition the review grades against.

worker/test/graded_state_parity.test.ts guards the TS side.
"""

from __future__ import annotations

import json
from typing import Any, cast

from momentarily.mapping import severity_tier
from scripts.gen_graded_state_parity_fixture import FIXTURE_PATH
from training.recovery_baseline import derive_graded_mta_state


def _fixture() -> dict[str, Any]:
    return cast("dict[str, Any]", json.loads(FIXTURE_PATH.read_text()))


def test_graded_state_fixture_reproduces_severity_tier() -> None:
    """The committed fixture must match severity_tier today.
    If this fails, run: uv run python -m scripts.gen_graded_state_parity_fixture"""
    cases = cast("list[dict[str, Any]]", _fixture()["severity"])
    assert cases, "fixture has no severity cases"
    for case in cases:
        assert severity_tier(case["alert_type"]) == case["tier"], case["alert_type"]


def test_graded_state_fixture_reproduces_derive_graded_mta_state() -> None:
    """The committed fixture must match derive_graded_mta_state today.
    If this fails, run: uv run python -m scripts.gen_graded_state_parity_fixture"""
    cases = cast("list[dict[str, Any]]", _fixture()["graded"])
    assert cases, "fixture has no graded cases"
    for case in cases:
        actual = derive_graded_mta_state(
            tuple(case["alert_types"]), floor=case["floor"]
        )
        assert actual == case["expected"], case


def test_canonical_floor_matches_fixture() -> None:
    from momentarily.mapping import CANONICAL_SEVERITY_FLOOR

    assert _fixture()["canonical_floor"] == CANONICAL_SEVERITY_FLOOR
