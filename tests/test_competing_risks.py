"""Tests for the Aalen-Johansen competing-risks CIF estimator
(training/competing_risks.py).
"""

from __future__ import annotations

import json

import pytest

from training.competing_risks import (
    CIFResult,
    CompetingSample,
    cif_at,
    cif_curves,
    conditional_cif,
    result_as_dict,
    survival_at,
)

# Hand-verified oracle: two completed exits tied at t=10 (one to each cause),
# one further completed "normal" exit at t=20, one right-censored observation
# at t=30. All fractions here are dyadic (halves/quarters), so the expected
# values are exact in binary floating point — no pytest.approx needed.
ORACLE_SAMPLES: list[CompetingSample] = [
    (10, "normal"),
    (10, "suspended"),
    (20, "normal"),
    (30, None),
]


def _oracle_result() -> CIFResult:
    return cif_curves(ORACLE_SAMPLES)


def test_cif_curves_matches_hand_verified_oracle() -> None:
    result = _oracle_result()
    assert result.causes == ("normal", "suspended")
    assert result.cif["normal"] == [(10, 0.25), (20, 0.5)]
    assert result.cif["suspended"] == [(10, 0.25)]
    assert result.survival == [(10, 0.5), (20, 0.25)]


@pytest.mark.parametrize("t", [10, 20])
def test_cif_and_survival_sum_to_one(t: int) -> None:
    # At every event time, mass has either exited to a cause or is still
    # surviving — the finer-grained CIF split loses nothing relative to
    # plain KM survival.
    result = _oracle_result()
    total = (
        cif_at(result.cif["normal"], t)
        + cif_at(result.cif["suspended"], t)
        + survival_at(result.survival, t)
    )
    assert total == 1.0


@pytest.mark.parametrize(
    ("cause", "elapsed_sec", "horizon_sec", "expected"),
    [
        pytest.param("normal", 5, 10, 0.25, id="normal_before_first_event"),
        pytest.param("normal", 15, 10, 0.5, id="normal_after_first_event"),
        pytest.param("suspended", 15, 10, 0.0, id="suspended_no_further_events"),
    ],
)
def test_conditional_cif_matches_oracle(
    cause: str, elapsed_sec: int, horizon_sec: int, expected: float
) -> None:
    result = _oracle_result()
    assert conditional_cif(result, cause, elapsed_sec, horizon_sec) == expected


def test_three_way_tie_uses_survival_at_t_prev_not_t() -> None:
    # Three samples tie at t=7 (a normal exit, a suspended exit, and a
    # censoring), then a further suspended exit at t=9 that empties the risk
    # set entirely (survival(9) == 0). The AJ increment for the t=9 exit must
    # be charged against S(7) = 0.5, the survival just BEFORE t=9 — not
    # S(9) = 0.0. Using S(t) instead of S(t_prev) would credit that exit with
    # zero incidence and break the CIF+survival=1 identity at t=9.
    samples: list[CompetingSample] = [
        (7, "normal"),
        (7, "suspended"),
        (7, None),
        (9, "suspended"),
    ]
    result = cif_curves(samples)
    assert result.cif["normal"] == [(7, 0.25)]
    assert result.cif["suspended"] == [(7, 0.25), (9, 0.75)]
    assert result.survival == [(7, 0.5), (9, 0.0)]
    total_at_9 = (
        cif_at(result.cif["normal"], 9)
        + cif_at(result.cif["suspended"], 9)
        + survival_at(result.survival, 9)
    )
    assert total_at_9 == 1.0


def test_conditional_cif_zero_when_survival_at_elapsed_is_zero() -> None:
    # Reuses the three-way-tie shape: by t=9 the whole risk set has exited
    # (S(9) == 0), so the conditional probability is undefined and must read
    # 0.0 rather than divide by zero.
    samples: list[CompetingSample] = [
        (7, "normal"),
        (7, "suspended"),
        (7, None),
        (9, "suspended"),
    ]
    result = cif_curves(samples)
    assert conditional_cif(result, "suspended", 9, 5) == 0.0


def test_conditional_cif_zero_for_unknown_cause() -> None:
    result = _oracle_result()
    assert conditional_cif(result, "does_not_exist", 5, 10) == 0.0


def test_empty_samples_yields_empty_result() -> None:
    result = cif_curves([])
    assert result.causes == ()
    assert result.cif == {}
    assert result.survival == []


def test_all_censored_samples_yields_empty_result() -> None:
    samples: list[CompetingSample] = [(10, None), (20, None)]
    result = cif_curves(samples)
    assert result.causes == ()
    assert result.cif == {}
    assert result.survival == []


def test_censored_tail_after_a_real_event_holds_flat() -> None:
    # One completed "normal" exit at t=10, then a censored observation at
    # t=20 that removes mass from the risk set but supplies no event of its
    # own — the curves must hold their last value past it rather than drop or
    # reset, exactly like the plain KM curve treats censoring.
    samples: list[CompetingSample] = [(10, "normal"), (20, None)]
    result = cif_curves(samples)
    assert result.cif["normal"] == [(10, 0.5)]
    assert result.survival == [(10, 0.5)]
    assert cif_at(result.cif["normal"], 1000) == 0.5
    assert survival_at(result.survival, 1000) == 0.5


def test_cif_at_and_survival_at_default_before_first_step() -> None:
    points: list[tuple[int, float]] = [(10, 0.4)]
    assert cif_at(points, 5) == 0.0
    assert survival_at(points, 5) == 1.0


def test_result_as_dict_is_json_serializable() -> None:
    result = _oracle_result()
    round_tripped = json.loads(json.dumps(result_as_dict(result)))
    assert round_tripped["causes"] == ["normal", "suspended"]
    assert round_tripped["cif"]["normal"] == [[10, 0.25], [20, 0.5]]
    assert round_tripped["cif"]["suspended"] == [[10, 0.25]]
    assert round_tripped["survival"] == [[10, 0.5], [20, 0.25]]
