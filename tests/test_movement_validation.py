"""The pure half of the movement-condition validation (training/movement_validation.py).

Every R2 read lives behind `main`; everything here runs on synthetic per-route
calls and a synthetic assigned_n label.
"""

from __future__ import annotations

from collections.abc import Mapping

from training.degradation_label import BIN_FN as _bin_fn
from training.load import TICK_SECONDS
from training.load_r2 import Disruption
from training.movement_validation import (
    DISRUPTED,
    NOT_NORMAL,
    SEVERE_RATIO,
    Unit,
    _boot_rates,  # pyright: ignore[reportPrivateUsage]
    _onset_latency,  # pyright: ignore[reportPrivateUsage]
    _state_at,  # pyright: ignore[reportPrivateUsage]
    episode_min_ratio,
    grade,
    published_states,
    split_severity,
)

T0 = 1_700_000_100


def _t(i: int) -> int:
    return T0 + i * TICK_SECONDS


def _bin(tick: int) -> str:
    """schedule_bin, the bucket degradation_label judges supply against."""
    return _bin_fn(tick)


def _calls(
    per_tick: dict[int, dict[str, str]],
) -> list[tuple[int, Mapping[str, str]]]:
    return [(tick, calls) for tick, calls in sorted(per_tick.items())]


# --- _boot_rates ------------------------------------------------------------


def test_boot_rates_reports_counts_and_no_rates_when_empty() -> None:
    """An empty arm has to say so; a zero rate there would read as a measurement."""
    assert _boot_rates([]) == {"n_units": 0, "n_ticks": 0}


def test_boot_rates_brackets_the_point_estimates() -> None:
    units = [Unit(2, 4), Unit(0, 6), Unit(1, 2), Unit(0, 8)]
    out = _boot_rates(units, n=500)
    assert out["n_units"] == 4
    assert out["n_ticks"] == 20
    assert out["alarmed_units"] == 2
    assert out["alarmed_ticks"] == 3
    assert out["tick_rate"] == 3 / 20
    assert out["unit_rate"] == 0.5
    assert out["tick_rate_ci_low"] <= 3 / 20 <= out["tick_rate_ci_high"]
    assert out["unit_rate_ci_low"] <= 0.5 <= out["unit_rate_ci_high"]


def test_boot_rates_tick_rate_is_exposure_matched_where_unit_rate_is_not() -> None:
    """The reason tick_rate leads: a rare firing in a long window reads as a 100%
    unit rate while its tick rate is a hundredth of that."""
    sparse = _boot_rates([Unit(1, 100)] * 20, n=200)
    assert sparse["unit_rate"] == 1.0
    assert sparse["tick_rate"] == 0.01


def test_boot_rates_is_deterministic_for_a_seed() -> None:
    units = [Unit(1, 3), Unit(0, 5), Unit(2, 2)]
    assert _boot_rates(units, n=200, seed=3) == _boot_rates(units, n=200, seed=3)


def test_boot_rates_on_a_unanimous_arm_has_no_spread() -> None:
    out = _boot_rates([Unit(4, 4)] * 12, n=200)
    assert out["tick_rate_ci_low"] == out["tick_rate_ci_high"] == 1.0
    assert out["unit_rate_ci_low"] == out["unit_rate_ci_high"] == 1.0


# --- published_states -------------------------------------------------------


def test_published_states_holds_a_regime_across_an_abstention() -> None:
    """A route absent from a tick keeps its last committed state — the stale
    published disrupted the raw calls never show."""
    calls = _calls({_t(0): {"A": "disrupted"}, _t(1): {}, _t(2): {}})
    pub = _state_at(published_states(calls))
    assert pub[_t(0)]["A"] == "disrupted"
    assert pub[_t(1)]["A"] == "disrupted"  # held open, though A did not report
    assert pub[_t(2)]["A"] == "disrupted"


def test_published_states_debounce_one_commits_immediately() -> None:
    calls = _calls({_t(0): {"A": "normal"}, _t(1): {"A": "disrupted"}})
    pub = _state_at(published_states(calls))
    assert pub[_t(1)]["A"] == "disrupted"


# --- detection / false alarms ----------------------------------------------


def test_grade_detects_a_firing_inside_an_episode() -> None:
    calls = _calls(
        {
            _t(0): {"A": "normal"},
            _t(1): {"A": "disrupted"},
            _t(2): {"A": "disrupted"},
        }
    )
    ep = Disruption("A", _t(1), _t(3))
    out = grade(calls, [ep], [ep], [], [], alarm_states=DISRUPTED, bootstrap=100)
    det = out["calls"]["detection"]["all"]
    assert det["n_ticks"] == 2
    assert det["alarmed_ticks"] == 2
    assert det["tick_rate"] == 1.0
    assert det["gradeable"] == 1
    assert det["offered"] == 1


def test_grade_charges_a_unit_only_for_ticks_the_stream_scored() -> None:
    """The label clock and the vehicle archive clock can disagree about which
    ticks exist; a tick the route was never judged on is not charged."""
    calls = _calls({_t(1): {"A": "disrupted"}})
    ep = Disruption("A", _t(0), _t(6))
    out = grade(calls, [ep], [ep], [], [], alarm_states=DISRUPTED, bootstrap=100)
    det = out["calls"]["detection"]["all"]
    assert det["n_ticks"] == 1  # not 6
    assert det["tick_rate"] == 1.0


def test_grade_drops_an_episode_whose_route_was_never_judged() -> None:
    """A route with no judged tick in its window is an absence of evidence, not a
    miss — counting it would charge the classifier's blind spots to detection."""
    calls = _calls({_t(1): {"B": "disrupted"}})
    ep = Disruption("A", _t(0), _t(4))
    out = grade(calls, [ep], [], [ep], [], alarm_states=DISRUPTED, bootstrap=100)
    det = out["calls"]["detection"]["all"]
    assert det == {"gradeable": 0, "offered": 1, "n_units": 0, "n_ticks": 0}


def test_grade_counts_a_false_alarm_on_a_normal_run() -> None:
    calls = _calls(
        {_t(i): {"A": "normal", "B": "normal"} for i in range(6)}
        | {_t(2): {"A": "disrupted", "B": "normal"}}
    )
    out = grade(
        calls,
        [],
        [],
        [],
        [("A", _t(0), _t(6)), ("B", _t(0), _t(6))],
        alarm_states=DISRUPTED,
        bootstrap=100,
    )
    fa = out["calls"]["false_alarms"]
    assert fa["alarmed_units"] == 1
    assert fa["n_units"] == 2
    assert fa["alarmed_ticks"] == 1
    assert fa["n_ticks"] == 12


def test_not_normal_arm_folds_in_suspended_but_disrupted_arm_does_not() -> None:
    """The disrupted arm is the flow signal independent of assigned_n; suspended
    only counts under the not_normal arm."""
    calls = _calls({_t(0): {"A": "suspended"}, _t(1): {"A": "suspended"}})
    ep = Disruption("A", _t(0), _t(2))
    dis = grade(calls, [ep], [ep], [], [], alarm_states=DISRUPTED, bootstrap=50)
    nn = grade(calls, [ep], [ep], [], [], alarm_states=NOT_NORMAL, bootstrap=50)
    assert dis["calls"]["detection"]["all"]["alarmed_ticks"] == 0
    assert nn["calls"]["detection"]["all"]["alarmed_ticks"] == 2


# --- onset latency ----------------------------------------------------------


def test_latency_is_measured_from_the_onset_itself() -> None:
    onset = _t(6)
    state_at = _state_at(
        _calls({_t(i): {"A": "disrupted" if i == 8 else "normal"} for i in range(12)})
    )
    out = _onset_latency(state_at, [Disruption("A", onset, _t(11))], DISRUPTED)
    assert out["n_detected"] == 1
    assert out["median_latency_min"] == 10.0  # two ticks after onset


def test_latency_counts_a_never_fired_episode_as_missed_not_slow() -> None:
    state_at = _state_at(_calls({_t(i): {"A": "normal"} for i in range(12)}))
    out = _onset_latency(state_at, [Disruption("A", _t(6), _t(11))], DISRUPTED)
    assert out["n_detected"] == 0
    assert out["n_missed"] == 1
    assert out["median_latency_min"] is None


def test_latency_excludes_an_episode_whose_route_was_never_scored() -> None:
    """An episode whose route has no judged movement tick in its span is absence
    of evidence, not a miss — counting it would charge an archive-clock gap (or a
    near-suspension too sparse to judge) to the detection rate."""
    state_at = _state_at(_calls({_t(i): {"B": "normal"} for i in range(12)}))
    out = _onset_latency(state_at, [Disruption("A", _t(6), _t(11))], DISRUPTED)
    assert out["n_offered"] == 1
    assert out["n_ungradeable"] == 1
    assert out["n_episodes"] == 0  # not counted as a miss
    assert out["n_missed"] == 0
    assert out["detection_rate"] is None
    assert out["fresh"]["n_episodes"] == 0


def test_latency_excludes_an_episode_already_firing_at_onset() -> None:
    onset = _t(20)
    state_at = _state_at(_calls({_t(i): {"A": "disrupted"} for i in range(30)}))
    out = _onset_latency(
        state_at, [Disruption("A", onset, onset + 5 * TICK_SECONDS)], DISRUPTED
    )
    assert out["n_offered"] == 1
    assert out["n_alarming_at_onset"] == 1
    assert out["n_episodes"] == 0  # nothing left to measure latency on
    assert out["fresh"]["n_episodes"] == 0


def test_latency_fresh_cohort_keeps_an_alarm_that_arrives() -> None:
    onset = _t(20)
    state_at = _state_at(
        _calls({_t(i): {"A": "disrupted" if i >= 21 else "normal"} for i in range(30)})
    )
    out = _onset_latency(
        state_at, [Disruption("A", onset, onset + 5 * TICK_SECONDS)], DISRUPTED
    )
    assert out["n_alarming_at_onset"] == 0
    assert out["fresh"]["n_episodes"] == 1
    assert out["fresh"]["n_detected"] == 1
    assert out["fresh"]["median_latency_min"] == 5.0


# --- severity split ---------------------------------------------------------


def test_episode_min_ratio_takes_the_worst_supply_tick() -> None:
    series = {("A", _t(0)): 10, ("A", _t(1)): 2, ("A", _t(2)): 8}
    baseline = {("A", _bin(_t(i))): 20.0 for i in range(3)}
    d = Disruption("A", _t(0), _t(3))
    assert episode_min_ratio(d, series, baseline) == 2 / 20.0


def test_split_severity_buckets_by_worst_ratio_and_counts_unrateable() -> None:
    baseline = {("A", _bin(_t(i))): 20.0 for i in range(4)}
    baseline |= {("B", _bin(_t(i))): 20.0 for i in range(4)}
    # A collapses to 1/20 = 0.05 <= SEVERE_RATIO; B dips to 12/20 = 0.6 (partial);
    # C has no baseline anywhere -> unrateable.
    series = {
        ("A", _t(0)): 20,
        ("A", _t(1)): 1,
        ("B", _t(0)): 20,
        ("B", _t(1)): 12,
        ("C", _t(0)): 1,
    }
    disruptions = [
        Disruption("A", _t(0), _t(2)),
        Disruption("B", _t(0), _t(2)),
        Disruption("C", _t(0), _t(2)),
    ]
    severe, partial, n_unrateable = split_severity(disruptions, series, baseline)
    assert [d.route for d in severe] == ["A"]
    assert [d.route for d in partial] == ["B"]
    assert n_unrateable == 1
    assert SEVERE_RATIO < 0.6  # B must not be mistaken for severe
