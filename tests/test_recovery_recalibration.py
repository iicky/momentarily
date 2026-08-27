"""Tests for the grade-driven recovery recalibration.

Covers the CDF-warp transform (dwell.recalibrate_curve / recalibrate_cell), the
factor fit (fit_recovery_gamma), the block application (recalibrate_dwell_block),
and the held-out episode population builder (severe_recovery_episodes) whose whole
job is to honour the two hard methodology rules — full-label boundaries and a
severe-peak criterion applied after segmentation with a severe-derived selector.
"""

from __future__ import annotations

import random

import pytest

from training.dwell import (
    CURVE_POINTS,
    DwellQuantiles,
    dwell_cdf,
    recalibrate_cell,
    recalibrate_curve,
)
from training.eval import TICK_SECONDS
from training.recovery_dist import RecoveryDistSample, predicted_recovery_curve
from training.recovery_recalibration import (
    RECALIBRATED_STATES,
    episode_samples,
    fit_recovery_gamma,
    recalibrate_dwell_cells,
    recalibrate_dwell_cells_by_key,
    severe_recovery_episodes,
)

# A plausible heavy-tailed disrupted-dwell curve: 21 knots, seconds.
CURVE = [
    0,
    60,
    120,
    180,
    240,
    300,
    420,
    600,
    780,
    1020,
    1320,
    1680,
    2100,
    2640,
    3300,
    4200,
    5400,
    7200,
    10800,
    18000,
    43200,
]


def _cell(
    curve: list[int] = CURVE, tail: tuple[float, float] = (1.5, 900.0)
) -> DwellQuantiles:
    return {
        "n": 50,
        "n_censored": 5,
        "q25_sec": 180,
        "median_sec": 600,
        "q75_sec": 2100,
        "recover_by_30": dwell_cdf(curve, 1800),
        "recover_by_60": dwell_cdf(curve, 3600),
        "recover_by_120": dwell_cdf(curve, 7200),
        "curve_sec": list(curve),
        "tail_ll": list(tail),
    }


def _approx(expected: object, *, abs: float | None = None) -> object:
    return pytest.approx(expected, abs=abs)  # pyright: ignore[reportUnknownMemberType]


def _quantile(curve: list[int], p: float) -> float:
    """Local inverse of the quantile curve (avoids importing dwell._dwell_quantile)."""
    k = len(curve)
    pos = min(max(p, 0.0), 1.0) * (k - 1)
    i = min(int(pos), k - 2)
    frac = pos - i
    return curve[i] + frac * (curve[i + 1] - curve[i])


# --- transform -----------------------------------------------------------------


def test_recalibrate_curve_identity_at_gamma_one():
    assert recalibrate_curve(CURVE, 1.0) == CURVE


def test_recalibrate_curve_is_monotone_with_fixed_endpoints():
    warped = recalibrate_curve(CURVE, 1.5)
    assert warped[0] == CURVE[0]
    assert warped[-1] == CURVE[-1]
    assert all(warped[i] <= warped[i + 1] for i in range(len(warped) - 1))
    assert len(warped) == CURVE_POINTS


def test_recalibrate_curve_gamma_gt_one_lowers_cdf_everywhere_interior():
    """gamma > 1 corrects optimism: at every interior time the recalibrated CDF
    is at or below the original (recovery predicted no sooner than before)."""
    warped = recalibrate_curve(CURVE, 1.6)
    for t in (600, 1800, 3600, 7200, 10800):
        assert dwell_cdf(warped, t) <= dwell_cdf(CURVE, t) + 1e-9
    # and strictly lower somewhere in the diseased band
    assert dwell_cdf(warped, 7200) < dwell_cdf(CURVE, 7200) - 1e-6


def test_recalibrate_cell_recomputes_summaries_and_leaves_tail_untouched():
    cell = _cell()
    out = recalibrate_cell(cell, 1.5)
    assert out["recover_by_120"] < cell["recover_by_120"]
    assert out["recover_by_60"] < cell["recover_by_60"]
    assert out["median_sec"] >= cell["median_sec"]
    # the fitted tail is left as-is: the warp fixes curve endpoints, so the
    # past-the-curve splice point (curve_sec[-1]) does not move.
    assert out.get("tail_ll") == cell.get("tail_ll")
    assert out["curve_sec"][-1] == cell["curve_sec"][-1]
    # recover_by fields stay consistent with the reshaped curve the Worker reads
    assert out["recover_by_120"] == _approx(dwell_cdf(out["curve_sec"], 7200))


def test_recalibrate_cell_leaves_mixture_cells_untouched():
    """A point-mass mixture is graded through its closed form, never curve_sec,
    so reshaping the curve could not reach it — it is returned unchanged."""
    cell = _cell()
    cell["atom_p"] = 0.7
    cell["atom_sec"] = TICK_SECONDS
    assert recalibrate_cell(cell, 1.5) is cell


def test_recalibrate_cell_identity_at_gamma_one():
    cell = _cell()
    assert recalibrate_cell(cell, 1.0) is cell


# --- fit -----------------------------------------------------------------------


def test_fit_recovery_gamma_empty_is_identity():
    assert fit_recovery_gamma([]) == 1.0


def test_fit_recovery_gamma_recovers_a_known_optimistic_factor():
    """Draw outcomes from F**gamma_true while the model still publishes F; the
    fit should recover gamma_true (the model is optimistic by exactly that warp)."""
    gamma_true = 1.5
    rng = random.Random(7)
    samples: list[RecoveryDistSample] = []
    for i in range(4000):
        u = rng.random()
        actual_sec = _quantile(CURVE, u ** (1.0 / gamma_true))
        samples.append(
            RecoveryDistSample(
                pred_curve=predicted_recovery_curve(0.0, CURVE, [1.5, 900.0], None),
                actual_min=actual_sec / 60.0,
                regime_key=f"r:{i}",
            )
        )
    assert fit_recovery_gamma(samples) == _approx(gamma_true, abs=0.15)


def test_fit_recovery_gamma_clamps_pessimistic_window_to_one():
    """A window where the model already recovers slower than reality asks for no
    correction, not for extra optimism: gamma floors at 1.0."""
    rng = random.Random(3)
    samples: list[RecoveryDistSample] = []
    for i in range(2000):
        u = rng.random()
        actual_sec = _quantile(CURVE, u ** (1.0 / 0.6))  # pessimistic model
        samples.append(
            RecoveryDistSample(
                pred_curve=predicted_recovery_curve(0.0, CURVE, [1.5, 900.0], None),
                actual_min=actual_sec / 60.0,
                regime_key=f"r:{i}",
            )
        )
    assert fit_recovery_gamma(samples) == 1.0


# --- block application ---------------------------------------------------------


def test_recalibrate_dwell_block_flat_and_nested_shapes():
    flat = {"A": {"disrupted": _cell()}}
    nested = {"A": {"disrupted": {"delays": _cell()}}}
    rf = recalibrate_dwell_cells(flat, 1.5)
    rn = recalibrate_dwell_cells_by_key(nested, 1.5)
    assert (
        rf["A"]["disrupted"]["recover_by_120"]
        < flat["A"]["disrupted"]["recover_by_120"]
    )
    assert (
        rn["A"]["disrupted"]["delays"]["recover_by_120"]
        < nested["A"]["disrupted"]["delays"]["recover_by_120"]
    )


def test_recalibrate_dwell_block_skips_normal_state():
    """gamma was fit on disruption recovery; normal-dwell is a different forecast
    and must pass through untouched."""
    block = {"A": {"normal": _cell(), "disrupted": _cell()}}
    out = recalibrate_dwell_cells(block, 1.5)
    assert out["A"]["normal"] is block["A"]["normal"]
    assert out["A"]["disrupted"] is not block["A"]["disrupted"]
    assert "normal" not in RECALIBRATED_STATES


def test_recalibrate_dwell_block_identity_at_gamma_one():
    block = {"A": {"disrupted": _cell()}}
    assert recalibrate_dwell_cells(block, 1.0) is block


# --- held-out episode population ------------------------------------------------

SEVERE = "Severe Delays"  # tier 2
MINOR = "Delays"  # tier 1
SUSP = "Suspended"  # tier 3


def _grid(
    route: str, states_types: list[tuple[str, list[str]]], *, start_tick: int
) -> tuple[dict[tuple[str, int], str], dict[tuple[str, int], tuple[str, ...]]]:
    """states_types: list of (state, alert_types) per consecutive grid tick.
    Returns (full_truth, types) additions for one route."""
    full_truth: dict[tuple[str, int], str] = {}
    types: dict[tuple[str, int], tuple[str, ...]] = {}
    for i, (state, ats) in enumerate(states_types):
        tick = start_tick + i * TICK_SECONDS
        if state != "normal":
            full_truth[(route, tick)] = state
        if ats:
            types[(route, tick)] = tuple(ats)
    return full_truth, types


def test_full_label_boundaries_merge_across_a_minor_dip():
    """A severe incident that downgrades to a minor delay and back is ONE episode
    under the full label, not two with a fake recovery between them."""
    start = 100 * TICK_SECONDS
    seq = (
        [("disrupted", [SEVERE])] * 4
        + [("disrupted", [MINOR])] * 2  # minor dip: still not-normal under full label
        + [("disrupted", [SEVERE])] * 3
    )
    full, types = _grid("1", seq, start_tick=start)
    eps = severe_recovery_episodes(
        full,
        types,
        window_start=start - 5 * TICK_SECONDS,
        window_end=start + 50 * TICK_SECONDS,
        onset_from=start,
        onset_to=start + 50 * TICK_SECONDS,
    )
    assert len(eps) == 1
    assert eps[0].duration_sec == 9 * TICK_SECONDS  # onset..first normal tick


def test_minor_only_incident_dropped_by_severe_peak_criterion():
    start = 100 * TICK_SECONDS
    full, types = _grid("2", [("disrupted", [MINOR])] * 5, start_tick=start)
    eps = severe_recovery_episodes(
        full,
        types,
        window_start=start - 5 * TICK_SECONDS,
        window_end=start + 50 * TICK_SECONDS,
        onset_from=start,
        onset_to=start + 50 * TICK_SECONDS,
    )
    assert eps == []


def test_selector_is_severe_derived_not_diluted_by_minor_majority():
    """A run dominated by minor delays but carrying a suspension must select the
    suspension cell, not be relabeled 'delays' by the tier-1 majority."""
    start = 100 * TICK_SECONDS
    seq = [("disrupted", [MINOR])] * 8 + [("suspended", [SUSP])] * 2
    full, types = _grid("3", seq, start_tick=start)
    eps = severe_recovery_episodes(
        full,
        types,
        window_start=start - 5 * TICK_SECONDS,
        window_end=start + 50 * TICK_SECONDS,
        onset_from=start,
        onset_to=start + 50 * TICK_SECONDS,
    )
    assert len(eps) == 1
    assert eps[0].peak_state == "suspended"
    assert eps[0].cause == "service_suspension"


def test_onset_window_and_censoring_filters():
    start = 100 * TICK_SECONDS
    # one incident onsets before onset_from -> excluded; one inside -> kept
    full, types = _grid(
        "4", [("disrupted", [SEVERE])] * 3 + [("normal", [])], start_tick=start
    )
    later = start + 20 * TICK_SECONDS
    f2, t2 = _grid(
        "4", [("disrupted", [SEVERE])] * 3 + [("normal", [])], start_tick=later
    )
    full.update(f2)
    types.update(t2)
    eps = severe_recovery_episodes(
        full,
        types,
        window_start=start - 5 * TICK_SECONDS,
        window_end=start + 50 * TICK_SECONDS,
        onset_from=later,
        onset_to=start + 50 * TICK_SECONDS,
    )
    assert len(eps) == 1
    assert eps[0].onset == later


def test_episode_samples_use_the_baked_severe_selector():
    """severe_recovery_episodes bakes the severe selector into peak_state/cause,
    so episode_samples looks up the suspension cell for a suspension incident."""
    start = 100 * TICK_SECONDS
    seq = [("disrupted", [MINOR])] * 6 + [("suspended", [SUSP])] * 2
    full, types = _grid("5", seq, start_tick=start)
    eps = severe_recovery_episodes(
        full,
        types,
        window_start=start - 5 * TICK_SECONDS,
        window_end=start + 50 * TICK_SECONDS,
        onset_from=start,
        onset_to=start + 50 * TICK_SECONDS,
    )
    seen: dict[str, str] = {}

    def lookup(
        route: str, state: str, cause: str
    ) -> tuple[list[int], list[float] | None, tuple[float, float] | None] | None:
        seen["state"], seen["cause"] = state, cause
        return CURVE, [1.5, 900.0], None

    samples = episode_samples(eps, lookup)
    assert len(samples) == 1
    assert seen == {"state": "suspended", "cause": "service_suspension"}
