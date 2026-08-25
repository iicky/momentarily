"""The offline replay of the Worker's segment classifier.

The parity test at the bottom is the load-bearing one: `training.segment_replay`
exists so a change to the classifier can be graded BEFORE it ships, and a replay
that is a near-miss of the Worker produces a grade about a model that never runs.
The unit tests above it pin the pieces the fixture exercises only in combination.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from momentarily.hmm import schedule_bin
from training.segment_replay import (
    EFF_COUNT_SCALE,
    MIN_EFF_MATCHED,
    PRUNE_MATCHED,
    SEGMENT_DECAY,
    SHIPPED,
    Cell,
    FlowState,
    Policy,
    TickInput,
    classify,
    js_round,
    replay,
    tick_inputs,
    update_flow,
)
from training.segments import (
    QUIET_MAX_EXPECTED,
    THROUGHPUT_ALPHA,
    classify_throughput,
    pois_lower_tail,
)

TICK = 300
# Wed 2026-08-19 09:15 ET, the parity fixture's own start.
T0 = 1787145300
BIN = schedule_bin(T0)

KEY = "F|south|A09S"
IDLE_KEY = "F|south|A11S"


def _params(**over: Any) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "schema_version": "1",
        "trained_at": T0,
        "min_share": 0.5,
        "topology_source": "gtfs_static",
        "cells": {
            KEY: {"p0": 0.9, "n": 1000, "lam": {BIN: 2.0}},
            IDLE_KEY: {"p0": 0.5, "n": 20},
        },
        "adjacency": {
            KEY: {"to": "A10S", "source": "gtfs_static"},
            IDLE_KEY: {"to": "A12S", "source": "gtfs_static"},
        },
        "throughput": {"bin": "schedule_bin", "min_ticks": 20, "ticks": {BIN: 600}},
    }
    doc.update(over)
    return doc


def test_js_round_matches_javascript_not_python() -> None:
    """Python's round() is banker's rounding; the Worker uses Math.round. Every
    exact .5 in a decayed sum would desync the replay otherwise."""
    assert round(2.5) == 2  # the trap
    assert js_round(2.5) == 3
    assert js_round(3.5) == 4
    assert js_round(-2.5) == -2  # Math.round(-2.5) === -2, half UP
    assert js_round(2.4999) == 2


def test_shipped_policy_is_the_worker_constants() -> None:
    """The parity fixture is replayed with SHIPPED, so if these drift apart the
    fixture stops describing the Worker while still passing."""
    assert SHIPPED.decay == SEGMENT_DECAY
    assert SHIPPED.min_eff_matched == MIN_EFF_MATCHED
    assert SHIPPED.prune_matched == PRUNE_MATCHED
    assert SHIPPED.eff_count_scale == EFF_COUNT_SCALE
    # ~25 minutes at a 5-minute tick.
    assert math.isclose(SHIPPED.window_ticks, 5.0)


def test_a_wider_window_judges_more_cells_on_the_same_data() -> None:
    """The competing axis, replayed: a longer accumulator window clears the
    matched floor on cells the shipped policy leaves unjudged — and costs onset
    latency to do it, which is why the two are worth comparing rather than
    assuming."""
    params = _params()
    del params["throughput"]  # advance branch only, so the window is the only lever
    ticks = [TickInput(T0 + i * TICK, {KEY: (1, 1)}, {"F": 20}) for i in range(6)]
    shipped = replay(ticks, params, SHIPPED)
    wide = replay(ticks, params, Policy(decay=0.94, min_eff_matched=3))
    assert shipped[-1][1] == {}  # 5 matched over the window, never clears 5
    assert wide[-1][1] == {KEY: "normal"}


def test_pois_lower_tail_endpoints() -> None:
    assert pois_lower_tail(-1, 5.0) == 0.0
    assert pois_lower_tail(0, 0.0) == 1.0
    assert math.isclose(pois_lower_tail(0, 3.0), math.exp(-3.0))
    # P(X<=1) = e^-mu (1 + mu)
    assert math.isclose(pois_lower_tail(1, 3.0), math.exp(-3.0) * 4.0)
    assert math.isclose(pois_lower_tail(400, 3.0), 1.0)


def test_quiet_floor_is_where_the_tail_loses_all_power() -> None:
    """The quiet call is a derivation, not a tuned threshold: below
    -ln(alpha) even an empty window cannot reach the tail."""
    assert math.isclose(math.exp(-QUIET_MAX_EXPECTED), THROUGHPUT_ALPHA)
    just_under = (QUIET_MAX_EXPECTED - 1e-9) / EFF_COUNT_SCALE
    just_over = (QUIET_MAX_EXPECTED + 1e-9) / EFF_COUNT_SCALE
    assert (
        classify_throughput(0, just_under, True, eff_count_scale=EFF_COUNT_SCALE)
        == "quiet"
    )
    assert (
        classify_throughput(0, just_over, True, eff_count_scale=EFF_COUNT_SCALE)
        == "disrupted"
    )


def test_throughput_is_monotone_in_the_observation() -> None:
    """Flooring the effective count, not rounding it: as the observed window
    shrinks the call may only get worse, never bounce back to normal."""
    calls = [
        classify_throughput(m / 10, 3.0, True, eff_count_scale=EFF_COUNT_SCALE)
        for m in range(60, -1, -1)
    ]
    rank = {"normal": 0, "disrupted": 1}
    ranks = [rank[c] for c in calls if c is not None]
    assert ranks == sorted(ranks)


def test_route_the_feed_skipped_abstains_but_quiet_still_speaks() -> None:
    assert classify_throughput(0, 5.0, False, eff_count_scale=EFF_COUNT_SCALE) is None
    assert (
        classify_throughput(0, 0.0, False, eff_count_scale=EFF_COUNT_SCALE) == "quiet"
    )


def test_expectation_accrues_for_a_cell_that_saw_no_train() -> None:
    """The accumulator iterates the baselined cells, not what moved — the cell
    the whole branch exists for has no observations at all."""
    state = update_flow(FlowState(), TickInput(T0, {}, {"F": 20}), _params())
    assert state.cells[KEY].m == 0
    assert state.cells[KEY].e == 2.0
    # Nothing observed and nothing expected: pruned, and read back as zeros.
    assert IDLE_KEY not in state.cells
    assert classify(state, _params())[IDLE_KEY] == "quiet"


def test_expectation_carries_across_a_schedule_bin_edge() -> None:
    """A rate published for this bin only must not vanish from the window the
    instant the clock ticks over, or every bin edge reads as a collapse."""
    params = _params()
    state = update_flow(FlowState(), TickInput(T0, {}, {"F": 20}), params)
    # 09:15 + 45min = 10:00, the next bin, where this cell has no published rate.
    later = T0 + 9 * TICK
    assert schedule_bin(later) != BIN
    after = update_flow(state, TickInput(later, {}, {"F": 20}), params)
    assert math.isclose(after.cells[KEY].e, SEGMENT_DECAY * 2.0)


def test_a_params_doc_with_no_fit_never_reaches_the_throughput_branch() -> None:
    """The pre-branch behaviour, and what a Worker sees before the trainer has
    published a fit: thin cells abstain, exactly as they used to."""
    params = _params()
    del params["throughput"]
    state = update_flow(FlowState(), TickInput(T0, {KEY: (0, 3)}, {"F": 20}), params)
    assert state.cells[KEY].e == 0
    assert classify(state, params) == {}


def test_advance_branch_wins_where_it_has_an_opinion() -> None:
    """Trains present and moving on is a statement about flow, and flow is what
    this surface publishes — even far below the expected throughput."""
    state = FlowState(
        observed_at=T0, cells={KEY: Cell(a=38, m=40, e=500)}, vehicles={"F": 20}
    )
    assert js_round(state.cells[KEY].m) >= MIN_EFF_MATCHED
    assert classify(state, _params())[KEY] == "normal"


def test_tick_inputs_drops_feed_outage_ticks_and_keeps_per_route_vehicles() -> None:
    bodies: list[dict[str, Any]] = [
        {
            "observed_at": T0 + 7,  # snaps back to the tick boundary
            "rows": {
                "F": {
                    "vehicles_n": 12,
                    "by_direction": {
                        "south": {
                            "vehicles_n": 7,
                            "transitions": {"A09S>A10S": 3, "A09S>A09S": 1},
                        },
                    },
                },
                "G": {"vehicles_n": 0, "by_direction": {}},
            },
        },
        {"observed_at": T0 + TICK, "rows": {}},  # feed outage: no tick at all
    ]
    ticks = tick_inputs(bodies)
    assert [t.observed_at for t in ticks] == [T0]
    assert ticks[0].counts == {KEY: (3, 4)}
    # Per route, and a route the feed reported nothing for is simply absent.
    assert ticks[0].vehicles == {"F": 12}


def test_prune_keeps_a_cell_that_still_expects_traffic() -> None:
    """Both halves of the prune matter: dropping a cell that still expects
    traffic would restart its expectation from zero and bias it toward normal."""
    params = _params()
    state = update_flow(FlowState(), TickInput(T0, {}, {"F": 20}), params)
    assert state.cells[KEY].e > PRUNE_MATCHED
    # Same doc with the rate removed: nothing observed, nothing expected, gone.
    bare = _params(cells={KEY: {"p0": 0.9, "n": 1000}})
    assert KEY not in update_flow(FlowState(), TickInput(T0, {}, {"F": 20}), bare).cells


def test_replay_carries_state_across_ticks() -> None:
    """Both sums have to survive the tick boundary: traffic at the published rate
    must keep reading normal however long it runs, and silence must read
    disrupted from the first tick the window can judge."""
    params = _params()
    at_rate = replay(
        [TickInput(T0 + i * TICK, {KEY: (2, 2)}, {"F": 20}) for i in range(8)], params
    )
    assert [t for t, _ in at_rate] == [T0 + i * TICK for i in range(8)]
    assert {calls[KEY] for _, calls in at_rate} == {"normal"}

    silent = replay([TickInput(T0 + i * TICK, {}, {"F": 20}) for i in range(8)], params)
    assert {calls[KEY] for _, calls in silent} == {"disrupted"}

    # And the expectation is the geometric sum of the per-tick rate, so it
    # approaches lam / (1 - decay) from below. Kept inside one schedule_bin —
    # crossing into a bin this cell has no published rate for would (correctly)
    # start decaying it instead.
    state = FlowState()
    for i in range(8):
        state = update_flow(state, TickInput(T0 + i * TICK, {}, {"F": 20}), params)
    closed_form = 2.0 * (1 - SEGMENT_DECAY**8) / (1 - SEGMENT_DECAY)
    assert math.isclose(state.cells[KEY].e, closed_form, rel_tol=1e-9)
    assert state.cells[KEY].e < 2.0 / (1 - SEGMENT_DECAY)


def test_parity_fixture_matches_this_implementation() -> None:
    """The TS port replays the same fixture in worker/test/segment_parity.test.ts.
    Regenerate with: uv run python -m scripts.gen_segment_parity_fixture"""
    path = Path(__file__).parent / "fixtures" / "parity_segment_flow.json"
    fixture = json.loads(path.read_text())
    params = fixture["params"]
    state = FlowState()
    for step in fixture["steps"]:
        tick = TickInput(
            observed_at=step["observed_at"],
            counts={k: (v[0], v[1]) for k, v in step["counts"].items()},
            vehicles=step["vehicles"],
        )
        state = update_flow(state, tick, params)
        assert {k: vars(v) for k, v in sorted(state.cells.items())} == step["cells"]
        assert dict(sorted(state.vehicles.items())) == step["state_vehicles"]
        assert dict(sorted(classify(state, params).items())) == step["calls"]
