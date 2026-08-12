"""Cross-language contract guard for the dwell mixture.

training/dwell.py is the source of truth for the atom + left-truncated
log-logistic mixture (point mass at one publisher tick, mixed with a
log-logistic tail refit conditional on T > 1 tick) and for the dwell_cdf
lower-boundary fix that goes with it. worker/src/dwell.ts and viz/lib/dwell.ts
are hand-ports of the same math and will drift without a test pinning them.

This module guards the *Python* side: the committed fixture
(tests/fixtures/parity_dwell.json) must match what training.dwell produces
today. worker/test/dwell_parity.test.ts and viz/tests/dwell_parity.test.ts
guard the TS side against the same fixture.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from scripts.gen_dwell_parity_fixture import (
    ATOM_P,
    ATOM_SCALE,
    ATOM_SEC,
    ATOM_SHAPE,
    FIXTURE_PATH,
)
from training.dwell import (
    conditional_remaining_quantile,
    dwell_cdf,
    mixture_quantile,
    mixture_survival,
    p_leave_by,
)


def _approx(expected: object, *, abs: float) -> object:
    """Typed wrapper around ``pytest.approx``.

    pytest ships ``approx`` as a partially-untyped helper (its ``ApproxBase``
    return type leaks ``Unknown`` under strict mode), so we pin a concrete
    ``object`` boundary here once instead of at every comparison site.
    """
    return pytest.approx(expected, abs=abs)  # pyright: ignore[reportUnknownMemberType]


def _atom_tuple(d: dict[str, Any] | None) -> tuple[float, float] | None:
    return None if d is None else (d["p"], d["sec"])


def _call_case(case: dict[str, Any]) -> object:
    fn = cast("str", case["fn"])
    inputs = cast("dict[str, Any]", case["inputs"])
    if fn == "dwell_cdf":
        return dwell_cdf(inputs["curve_sec"], inputs["x"])
    if fn == "mixture_survival":
        return mixture_survival(
            inputs["t"],
            inputs["shape"],
            inputs["scale"],
            inputs["atom_p"],
            inputs["atom_sec"],
        )
    if fn == "mixture_quantile":
        return mixture_quantile(
            inputs["u"],
            inputs["shape"],
            inputs["scale"],
            inputs["atom_p"],
            inputs["atom_sec"],
        )
    if fn == "p_leave_by":
        return p_leave_by(
            inputs["curve_sec"],
            inputs["elapsed_sec"],
            inputs["horizon_sec"],
            tail_ll=inputs["tail_ll"],
            atom=_atom_tuple(inputs["atom"]),
        )
    if fn == "conditional_remaining_quantile":
        return conditional_remaining_quantile(
            inputs["curve_sec"],
            inputs["elapsed_sec"],
            inputs["q"],
            tail_ll=inputs["tail_ll"],
            atom=_atom_tuple(inputs["atom"]),
        )
    raise ValueError(f"unknown fixture fn: {fn}")


def test_dwell_parity_fixture_reproduces_python_dwell_module() -> None:
    """The committed fixture must match what training.dwell produces today.
    If this fails, run: uv run python -m scripts.gen_dwell_parity_fixture"""
    fixture = cast("dict[str, Any]", json.loads(FIXTURE_PATH.read_text()))
    cases = cast("list[dict[str, Any]]", fixture["cases"])
    for case in cases:
        actual = _call_case(case)
        expected = case["expected"]
        label = case["label"]
        if expected is None:
            assert actual is None, f"{label}: expected None, got {actual!r}"
        else:
            assert actual == _approx(expected, abs=1e-9), f"{label} drifted"


def test_atom_boundary_survival_equals_one_minus_atom_p_exactly() -> None:
    """F(atom_sec) == atom_p is the entire point of the closed form over the
    quantile-curve splice, which can only ever reach the mass as
    F(atom_sec + eps). Called fresh against training.dwell (not read back off
    the fixture) so a regenerated fixture can't launder a regression that
    breaks the invariant on both sides identically.
    """
    survival_at_atom_sec = mixture_survival(
        float(ATOM_SEC), ATOM_SHAPE, ATOM_SCALE, ATOM_P, float(ATOM_SEC)
    )
    assert (1.0 - survival_at_atom_sec) == _approx(ATOM_P, abs=1e-9)


def test_dwell_cdf_repeated_knot_regression() -> None:
    """The bug this contract fixes: curve[0] is exactly one tick for any
    disrupted cell, and the old `x <= curve[0] -> 0` guard graded the
    one-tick majority at P=0. At a flat run the CDF must land on the TOP of
    the run, not 0."""
    flat = [300] * 14 + [400, 550, 700, 900, 1200, 1600, 2100]
    k = len(flat)
    assert dwell_cdf(flat, 300.0) == 13 / (k - 1)  # largest j with curve[j]<=300 is 13
    assert dwell_cdf(flat, 299.0) == 0.0  # strictly below the first knot only
