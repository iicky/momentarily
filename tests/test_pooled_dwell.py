"""Partial-pooling behaviour for normal-regime dwell.

The estimator exists because the min-samples gate on empirical dwell cells is
length-biased for `normal`: a route only completes a normal regime by leaving
normal, so the gate keeps the flappiest routes and drops the steadiest ones.
These tests pin the properties that make the replacement safe — censoring has to
carry a sparse route, a well-observed route has to keep its own estimate, and the
population centre must not be dragged around by whichever route churns most.
"""

from __future__ import annotations

from training.dwell import CURVE_POINTS
from training.eval import MovementTransitionRecord
from training.pooled_dwell import (
    MIN_VOTER_EVENTS,
    cell_from_fit,
    partially_pooled_dwell,
    pooled_dwell_cells,
)

HOUR = 3600
DAY = 24 * HOUR

# Deterministic multiplicative spread around a centre. Real dwells are
# heavy-tailed and never identical; fixtures built from a single repeated
# duration collapse the fitted shape to a near-step function and stop
# exercising anything the estimator actually does.
_SPREAD = (0.25, 0.5, 0.8, 1.0, 1.3, 2.0, 3.5, 6.0)


def _events(centre_sec: int, n: int) -> list[tuple[int, bool]]:
    """n completed dwells scattered around `centre_sec`."""
    return [
        (max(60, round(centre_sec * _SPREAD[i % len(_SPREAD)])), True) for i in range(n)
    ]


def test_route_with_no_completed_exits_is_carried_by_its_censoring():
    """The load-bearing case. A route that never left normal inside the window has
    zero events, so its own MLE is unbounded and a naive pooled curve would hand
    it the population centre. The censored observation says the regime already
    outlived that centre many times over, and the fit has to respect it."""
    samples = {f"R{i}": _events(2 * HOUR, 8) for i in range(5)}
    samples["STEADY"] = [(30 * DAY, False)]  # normal for 30 days, still going

    fits = partially_pooled_dwell(samples)
    steady = fits["STEADY"]

    assert steady.n_events == 0
    assert steady.n_censored == 1
    assert steady.source == "pooled"
    # A single censored observation is weak evidence on its own — it says only
    # "dwell > 30 days", which under a heavy tail is not that surprising — so the
    # move off the parent is a few-fold, not order-of-magnitude. What must hold
    # is that it moves at all (complete pooling would hand it the parent) and
    # that it ends up ranked as the steadiest route in the population.
    assert steady.scale_sec > 3 * steady.parent_scale_sec
    assert steady.scale_sec == max(f.scale_sec for f in fits.values())


def test_well_observed_route_keeps_its_own_scale():
    """A route with plenty of completed exits must not be dragged to the parent
    just because its neighbours differ."""
    samples = {f"R{i}": _events(40 * HOUR, 6) for i in range(5)}
    samples["FLAPPY"] = _events(30 * 60, 40)  # 30-minute normal regimes, n=40

    fits = partially_pooled_dwell(samples)
    flappy = fits["FLAPPY"]

    assert flappy.source == "own"
    # Within a factor of two of its own 30-minute truth, nowhere near the
    # ~40h population centre.
    assert 15 * 60 < flappy.scale_sec < 60 * 60


def test_ordering_survives_pooling():
    """What naive pooling destroys: a steady route must still rank above a flappy
    one after shrinkage, or the published p_normal_in_H is worse than useless."""
    samples = {
        "FLAPPY": _events(20 * 60, 30),
        "MIDDLE": _events(6 * HOUR, 8),
        "STEADY": [*_events(5 * DAY, 4), (20 * DAY, False)],
    }
    fits = partially_pooled_dwell(samples)
    assert (
        fits["FLAPPY"].scale_sec < fits["MIDDLE"].scale_sec < fits["STEADY"].scale_sec
    )


def test_parent_is_route_weighted_not_event_weighted():
    """The core design claim. One route that churns constantly contributes far
    more completed events than all the others combined; it must still get exactly
    one vote on the population centre."""
    steady = {f"S{i}": _events(48 * HOUR, 5) for i in range(6)}
    with_churn = dict(steady)
    with_churn["CHURN"] = _events(5 * 60, 500)  # 500 events at 5 minutes

    parent_without = next(iter(partially_pooled_dwell(steady).values()))
    parent_with = next(iter(partially_pooled_dwell(with_churn).values()))

    # 500 five-minute events against 30 two-day events would dominate any
    # event-weighted centre. One vote out of seven barely moves the median.
    assert parent_with.parent_scale_sec > 0.5 * parent_without.parent_scale_sec


def test_every_route_gets_a_cell_with_no_min_samples_gate():
    """Contrast with compute_dwell_quantiles, which drops sub-threshold cells and
    leaves the Worker on its geometric fallback."""
    samples = {f"R{i}": _events(3 * HOUR, 6) for i in range(4)}
    samples["THIN"] = [(90 * HOUR, False)]
    samples["ONE"] = _events(HOUR, 1)

    fits = partially_pooled_dwell(samples)
    assert set(fits) == set(samples)
    assert fits["THIN"].n_events < MIN_VOTER_EVENTS
    assert fits["ONE"].source == "pooled"


def test_cell_matches_the_worker_contract():
    """The emitted cell has to be a drop-in for the Kaplan-Meier one the Worker
    already reads: a nondecreasing curve of the right length plus a tail."""
    samples = {f"R{i}": _events(4 * HOUR, 6) for i in range(4)}
    fits = partially_pooled_dwell(samples)
    cell = cell_from_fit(fits["R0"])

    assert len(cell["curve_sec"]) == CURVE_POINTS
    assert cell["curve_sec"] == sorted(cell["curve_sec"])
    assert cell["curve_sec"][0] >= 0
    tail = cell.get("tail_ll")
    assert tail is not None
    shape, scale = tail
    assert shape > 0
    assert scale > 0
    assert cell["q25_sec"] <= cell["median_sec"] <= cell["q75_sec"]
    for key in ("recover_by_30", "recover_by_60", "recover_by_120"):
        assert 0.0 <= cell[key] <= 1.0
    assert cell["recover_by_30"] <= cell["recover_by_60"] <= cell["recover_by_120"]


def test_empty_input_is_empty_output():
    assert partially_pooled_dwell({}) == {}
    assert pooled_dwell_cells([], state="normal") == {}


def test_single_voter_falls_back_to_the_pooled_centre():
    """Below two well-observed routes there is no population spread to estimate.
    The estimator must still return usable cells rather than dividing by a
    dispersion it cannot measure."""
    samples = {"ONLY": _events(2 * HOUR, 9), "THIN": [(HOUR, False)]}
    fits = partially_pooled_dwell(samples)

    assert set(fits) == {"ONLY", "THIN"}
    for fit in fits.values():
        assert fit.scale_sec > 0
        assert fit.parent_scale_sec > 0


def test_longer_censoring_pushes_the_scale_up():
    """Monotonicity in the evidence: two routes identical except for how long
    they have been normal must not receive the same forecast."""
    base = {f"R{i}": _events(3 * HOUR, 6) for i in range(5)}
    short = partially_pooled_dwell({**base, "X": [(6 * HOUR, False)]})["X"]
    long_ = partially_pooled_dwell({**base, "X": [(40 * DAY, False)]})["X"]

    assert long_.scale_sec > short.scale_sec


def _movement_tr(
    route: str, prev: str, dwell_sec: int, ts: int = 0, new_state: str = "normal"
) -> MovementTransitionRecord:
    return MovementTransitionRecord(
        ts=ts,
        scope="route",
        key=route,
        route=route,
        prev_state=prev,
        new_state=new_state,
        regime_entered_at=ts - dwell_sec,
        exited_at=ts,
        dwell_sec=dwell_sec,
    )


def test_pooled_dwell_cells_keys_on_movement_state_not_alert_regime():
    """C2 sources dwell_movement from MovementTransitionRecord, keyed on the
    movement clock's own prev_state -- pooled_dwell_cells runs directly off it,
    no TransitionRecord (the alert-regime record) anywhere in the call, and the
    `state` kwarg must select disjoint movement-regime episodes."""
    transitions = [
        _movement_tr("A", "disrupted", 300, ts=1000),
        _movement_tr("A", "disrupted", 900, ts=2000),
        _movement_tr("A", "suspended", 5400, ts=3000),
        _movement_tr("B", "suspended", 2700, ts=3000),
    ]
    disrupted = pooled_dwell_cells(transitions, state="disrupted")
    suspended = pooled_dwell_cells(transitions, state="suspended")
    assert disrupted["A"]["n"] == 2
    assert suspended["A"]["n"] == 1
    assert disrupted["A"]["median_sec"] != suspended["A"]["median_sec"]


def test_movement_open_regime_yields_censored_not_dropped_or_completed():
    """A route still sitting in `disrupted` at window_end must contribute a
    right-censored observation: present in the output, not silently dropped,
    and not folded into the completed-event count."""
    transitions = [_movement_tr("A", "disrupted", 300, ts=1000)]
    open_regimes = {"A": ("disrupted", 100_000)}
    cells = pooled_dwell_cells(
        transitions, state="disrupted", window_end=130_000, open_regimes=open_regimes
    )
    assert "A" in cells  # not dropped
    assert cells["A"]["n"] == 1  # the completed exit, unaffected
    assert cells["A"]["n_censored"] == 1  # the open regime, not a completed event


def test_movement_route_with_zero_episodes_still_gets_a_pooled_cell():
    """A route that has simply held `normal` for the whole window -- zero
    completed movement episodes -- must still surface a cell off its open
    regime alone, mirroring the alert-arm's normal-state estimator."""
    transitions = [
        _movement_tr(f"R{i}", "normal", 3600, ts=10_000 * (i + 1)) for i in range(4)
    ]
    open_regimes = {"QUIET": ("normal", 0)}
    cells = pooled_dwell_cells(
        transitions, state="normal", window_end=1_000_000, open_regimes=open_regimes
    )
    assert "QUIET" in cells
    assert cells["QUIET"]["n"] == 0
    assert cells["QUIET"]["n_censored"] == 1
