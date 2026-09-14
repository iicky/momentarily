"""Python half of the recovery-climatology cross-language parity pin.

Replays tests/fixtures/parity_recovery.json against training.recovery_baseline
so a change to empirical_quantile / conditional_remaining_quantiles that forgets
to regenerate the fixture fails here; worker/test/recovery_parity.test.ts is the
TypeScript half, replaying the same fixture against worker/src/recovery.ts.

Regenerate the fixture with: uv run python -m scripts.gen_recovery_parity_fixture
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from training.recovery_baseline import (
    conditional_remaining_quantiles,
    empirical_quantile,
)

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "parity_recovery.json"
CASES: list[dict[str, Any]] = json.loads(FIXTURE_PATH.read_text())["cases"]

TOL = 1e-9


def _approx(expected: object, *, abs: float) -> object:
    """Typed wrapper around pytest.approx (its ApproxBase is partially untyped)."""
    return pytest.approx(expected, abs=abs)  # pyright: ignore[reportUnknownMemberType]


def _ids(cases: list[dict[str, Any]]) -> list[str]:
    return [f"{c['fn']}: {c['label']}" for c in cases]


@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_recovery_parity(case: dict[str, Any]) -> None:
    inputs = case["inputs"]
    expected = case["expected"]
    if case["fn"] == "empirical_quantile":
        got = empirical_quantile(inputs["sorted"], inputs["q"])
        assert isinstance(expected, (int, float))
        assert got == _approx(expected, abs=TOL)
    elif case["fn"] == "conditional_remaining_quantiles":
        rq = conditional_remaining_quantiles(
            inputs["samples_min"],
            inputs["elapsed_min"],
            min_samples=inputs["min_samples"],
        )
        if expected is None:
            assert rq is None
        else:
            assert rq is not None
            assert rq.p25 == _approx(expected["p25"], abs=TOL)
            assert rq.p50 == _approx(expected["p50"], abs=TOL)
            assert rq.p75 == _approx(expected["p75"], abs=TOL)
            assert rq.n == expected["n"]
    else:  # pragma: no cover - guards a malformed fixture
        pytest.fail(f"unknown fixture fn {case['fn']!r}")


def test_fixture_covers_both_functions_and_the_abstain_branch() -> None:
    """A regenerated fixture that silently dropped the indeterminate case would
    let the outlived-the-population path rot untested — pin that it is present."""
    fns = {c["fn"] for c in CASES}
    assert fns == {"empirical_quantile", "conditional_remaining_quantiles"}
    assert any(
        c["fn"] == "conditional_remaining_quantiles" and c["expected"] is None
        for c in CASES
    )
