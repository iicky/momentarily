"""Division map completeness/consistency, and the variance-decomposition
math on synthetic inputs with a known answer.
"""

from __future__ import annotations

from training.divisions import (
    ALL_KNOWN_ROUTES,
    DIVISION_BY_ROUTE,
    ROUTES_WITHOUT_DIVISION,
    decompose_variance,
    permutation_p_value,
)


def test_division_map_covers_every_known_route_exactly_once():
    """Every route the trainer knows is either divisioned or explicitly
    excluded, never both, never neither."""
    divisioned = set(DIVISION_BY_ROUTE)
    assert divisioned & ROUTES_WITHOUT_DIVISION == set()
    assert divisioned | ROUTES_WITHOUT_DIVISION == ALL_KNOWN_ROUTES


def test_division_map_only_uses_the_three_named_divisions():
    assert set(DIVISION_BY_ROUTE.values()) == {"IRT", "BMT", "IND"}


def test_division_counts_are_28_routes_roughly_9_per_division():
    """The measured shape: ~9 per division on 28 divisioned
    routes (plus SI, excluded)."""
    counts = {
        division: sum(1 for d in DIVISION_BY_ROUTE.values() if d == division)
        for division in ("IRT", "BMT", "IND")
    }
    assert len(DIVISION_BY_ROUTE) == 28
    assert counts == {"IRT": 10, "BMT": 9, "IND": 9}
    assert ALL_KNOWN_ROUTES - set(DIVISION_BY_ROUTE) == {"SI"}


def test_express_variants_inherit_their_base_routes_division():
    assert DIVISION_BY_ROUTE["6X"] == DIVISION_BY_ROUTE["6"]
    assert DIVISION_BY_ROUTE["7X"] == DIVISION_BY_ROUTE["7"]
    assert DIVISION_BY_ROUTE["FX"] == DIVISION_BY_ROUTE["F"]


def test_shuttles_carry_their_documented_division():
    """The three shuttles are the routes most likely to be guessed wrong --
    each sits on a different division's trackage. See module docstring for
    citations."""
    assert DIVISION_BY_ROUTE["GS"] == "IRT"  # Times Sq-Grand Central Shuttle
    assert DIVISION_BY_ROUTE["FS"] == "BMT"  # Franklin Avenue Shuttle
    assert DIVISION_BY_ROUTE["H"] == "IND"  # Rockaway Park Shuttle


# --- decompose_variance: known-answer synthetic cases -----------------------
#
# Three groups of three integers each, {1,2,3} / {4,5,6} / {7,8,9}. Grand
# mean 5.0. SS_between = 3*(2-5)^2 + 3*(5-5)^2 + 3*(8-5)^2 = 54.
# SS_within = 3 groups * ((x-mean)^2 summed to 2 each) = 6. Hand-checkable.
_SEPARATED_DIVISIONS = ("IRT", "IRT", "IRT", "BMT", "BMT", "BMT", "IND", "IND", "IND")


def _synthetic(values: list[int]) -> tuple[dict[str, float], dict[str, str]]:
    routes = [f"r{i}" for i in range(len(values))]
    return (
        dict(zip(routes, values, strict=True)),
        dict(zip(routes, _SEPARATED_DIVISIONS, strict=True)),
    )


def test_decompose_variance_recovers_a_hand_computed_separation():
    values, divisions = _synthetic([1, 2, 3, 4, 5, 6, 7, 8, 9])
    result = decompose_variance(values, divisions)

    assert result.n_total == 9
    assert result.grand_mean == 5.0
    assert result.df_between == 2
    assert result.df_within == 6
    assert result.ms_between == 27.0
    assert result.ms_within == 1.0
    assert result.f_ratio == 27.0
    assert result.eta_squared == 0.9
    by_division = {g.division: g for g in result.groups}
    assert by_division["IRT"].mean == 2.0
    assert by_division["BMT"].mean == 5.0
    assert by_division["IND"].mean == 8.0
    assert all(g.n == 3 for g in result.groups)


def test_decompose_variance_finds_zero_effect_when_group_means_match():
    """Same {1,2,3} pattern repeated in every division: means agree, so
    division explains none of the variance."""
    values, divisions = _synthetic([1, 2, 3, 1, 2, 3, 1, 2, 3])
    result = decompose_variance(values, divisions)

    assert result.ms_between == 0.0
    assert result.eta_squared == 0.0
    assert result.f_ratio == 0.0


def test_decompose_variance_drops_routes_without_a_known_division():
    """SI (or any unmapped route) must not silently join a group, however
    extreme its value."""
    values, divisions = _synthetic([1, 2, 3, 4, 5, 6, 7, 8, 9])
    values["SI"] = 1_000_000.0  # would dominate every sum of squares if kept
    with_si = decompose_variance(values, {**divisions})  # SI absent from divisions
    without_si = decompose_variance(
        {k: v for k, v in values.items() if k != "SI"}, divisions
    )
    assert with_si == without_si


def test_decompose_variance_raises_without_any_mapped_routes():
    try:
        decompose_variance({"SI": 1.0}, {})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


# --- permutation_p_value -----------------------------------------------------


def test_permutation_p_value_is_small_for_a_real_separation():
    values, divisions = _synthetic([1, 2, 3, 4, 5, 6, 7, 8, 9])
    p = permutation_p_value(values, divisions, n_permutations=2000, seed=0)
    assert p == 0.0029985007496251873


def test_permutation_p_value_is_1_when_the_null_is_exactly_true():
    """eta-squared 0.0 is the minimum possible value, so every relabeling is
    'at least as extreme' -- the p-value must be exactly 1, not just large."""
    values, divisions = _synthetic([1, 2, 3, 1, 2, 3, 1, 2, 3])
    p = permutation_p_value(values, divisions, n_permutations=2000, seed=0)
    assert p == 1.0


def test_permutation_p_value_requires_at_least_three_mapped_routes():
    try:
        permutation_p_value({"A": 1.0, "B": 2.0}, {"A": "IRT", "B": "BMT"})
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
