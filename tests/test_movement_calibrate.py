"""Movement cut/debounce calibration: the sweep must pick a deterministic
operating point on a fixed archive slice.

training.movement_calibrate has no importers and its constants ship straight
into worker/src/movement_state.ts (via load_r2.classify_direction), so a
regression here silently republishes the wrong disrupted-arm cut. This builds a
tiny synthetic CalibrationWindow whose one route runs normal -> frozen ->
normal, then asserts the sweep row at the published operating point
(prior_strength=8, disrupted_ratio=0.5, alpha=0.05, debounce=1 — the
CLASSIFY_* / DEBOUNCE_TICKS defaults) is exactly reproduced, and that the sweep
genuinely discriminates: raising the posterior gate off that point drops the
single episode entirely.
"""

from __future__ import annotations

from momentarily.hmm import tod_bin
from training.load_r2 import AdvanceBaseline, Disruption
from training.movement_calibrate import (
    TICK_SECONDS,
    CalibrationWindow,
    SweepRow,
    sweep,
)

T0 = 1_700_000_000
# A frozen (route, direction) baseline that advances 90% of matched trips in a
# healthy tick. classify_direction scores against disrupted_ratio * p0.
P0 = 0.9


def _tick(i: int) -> int:
    return T0 - T0 % TICK_SECONDS + i * TICK_SECONDS


def _window() -> CalibrationWindow:
    """One route "A", north only, over eight adjacent ticks: two healthy, three
    frozen (all matched trips stall against a 0.9 baseline), three healthy — one
    completed disrupted episode with a 15-minute dwell. A trip-updates
    disruption is laid over the frozen span so the corroboration reads 1.0."""
    ticks = [_tick(i) for i in range(8)]
    frozen = {2, 3, 4}
    route_series: dict[tuple[str, int], dict[str, int]] = {}
    dir_series: dict[tuple[str, str, int], dict[str, int]] = {}
    for i, tk in enumerate(ticks):
        route_series[("A", tk)] = {"vehicles_n": 10}
        dir_series[("A", "north", tk)] = (
            {"advanced_n": 0, "stalled_n": 10}
            if i in frozen
            else {"advanced_n": 10, "stalled_n": 0}
        )
    advance_baseline = {
        ("A", "north", tb): AdvanceBaseline(
            p0=P0, n=30, alpha=8 * P0, beta=8 * (1 - P0)
        )
        for tb in {tod_bin(tk) for tk in ticks}
    }
    tu_disruptions = [
        Disruption(route="A", start_tick=ticks[2], recovered_tick=ticks[5])
    ]
    return CalibrationWindow(
        route_series=route_series,
        dir_series=dir_series,
        advance_baseline=advance_baseline,
        tu_disruptions=tu_disruptions,
        eval_ticks=ticks,
        n_routes=1,
    )


def test_sweep_pins_the_published_operating_point() -> None:
    rows = sweep(_window())

    # 3 prior_strengths x 3 disrupted_ratios x 3 alphas x 1 debounce.
    assert len(rows) == 27

    op = next(
        r
        for r in rows
        if (r.prior_strength, r.disrupted_ratio, r.alpha, r.debounce_ticks)
        == (8, 0.5, 0.05, 1)
    )
    # base_rate 3/8 frozen ticks; stickiness 2/3 (two disrupted->disrupted of
    # three disrupted->next pairs); one 15-min episode; single_tick_frac 0 (the
    # dwell clears one tick); trip-updates overlap is perfect on this fixture.
    assert op == SweepRow(
        prior_strength=8,
        disrupted_ratio=0.5,
        alpha=0.05,
        debounce_ticks=1,
        base_rate=0.375,
        stickiness=2 / 3,
        n_episodes=1,
        mean_dwell_min=15.0,
        median_dwell_min=15.0,
        single_tick_frac=0.0,
        tu_precision=1.0,
        tu_recall=1.0,
        churn=0,
    )


def test_sweep_discriminates_off_the_operating_point() -> None:
    rows = {(r.prior_strength, r.disrupted_ratio, r.alpha): r for r in sweep(_window())}
    # At prior_strength=8 the frozen tick's posterior is exactly 0.4 (= 7.2/18).
    # A 0.5 ratio gate (0.45) calls it disrupted; a slacker 0.4 gate (0.36) does
    # not, so the episode and its corroboration vanish. The sweep is sensitive
    # to the very constant it exists to choose.
    assert rows[(8, 0.5, 0.05)].n_episodes == 1
    off = rows[(8, 0.4, 0.05)]
    assert off.n_episodes == 0
    assert off.mean_dwell_min is None
    assert off.tu_precision is None
