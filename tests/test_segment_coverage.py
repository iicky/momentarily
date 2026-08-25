"""The pure half of the throughput-branch grade (training/segment_coverage.py).

Every R2 read lives behind `main`; everything here runs on synthetic calls.
"""

from __future__ import annotations

from collections.abc import Mapping

from training.load import TICK_SECONDS
from training.load_r2 import Disruption
from training.segment_coverage import (
    MIN_NORMAL_RUN_TICKS,
    Unit,
    _boot_rates,  # pyright: ignore[reportPrivateUsage]
    grade,
    normal_runs,
)

T0 = 1_700_000_100


def _t(i: int) -> int:
    return T0 + i * TICK_SECONDS


def _calls(
    per_tick: dict[int, dict[str, str]],
) -> list[tuple[int, Mapping[str, str]]]:
    return [(tick, calls) for tick, calls in sorted(per_tick.items())]


def test_boot_rates_reports_counts_and_no_rates_when_empty() -> None:
    """An empty arm has to say so. A zero rate there would read as a
    measurement, and a CI on it would be fabricated."""
    assert _boot_rates([]) == {"n_units": 0, "n_ticks": 0}


def test_boot_rates_brackets_both_point_estimates() -> None:
    units = [Unit(2, 4, 0.5), Unit(0, 6, 0.0), Unit(1, 2, 0.25), Unit(0, 8, 0.0)]
    out = _boot_rates(units, n=500)
    assert out["n_units"] == 4
    assert out["n_ticks"] == 20
    assert out["alarmed_units"] == 2
    assert out["alarmed_ticks"] == 3
    assert out["tick_rate"] == 3 / 20
    assert out["unit_rate"] == 0.5
    assert out["tick_rate_ci_low"] <= 3 / 20 <= out["tick_rate_ci_high"]
    assert out["unit_rate_ci_low"] <= 0.5 <= out["unit_rate_ci_high"]


def test_tick_rate_is_exposure_matched_where_the_unit_rate_is_not() -> None:
    """The whole reason tick_rate leads: two populations firing at the same rate
    per tick get very different unit rates when one is far longer."""
    short = _boot_rates([Unit(1, 10, 1.0)] * 20, n=200)
    long_ = _boot_rates([Unit(10, 100, 10.0)] * 20, n=200)
    assert short["tick_rate"] == long_["tick_rate"] == 0.1
    assert short["unit_rate"] == long_["unit_rate"] == 1.0
    # ...and a population that fires once somewhere in a long window reads as a
    # 100% unit rate while its tick rate is a hundredth of that.
    sparse = _boot_rates([Unit(1, 100, 1.0)] * 20, n=200)
    assert sparse["unit_rate"] == 1.0
    assert sparse["tick_rate"] == 0.01


def test_boot_rates_is_deterministic_for_a_seed() -> None:
    units = [Unit(1, 3, 0.3), Unit(0, 5, 0.0), Unit(2, 2, 1.0)]
    assert _boot_rates(units, n=200, seed=3) == _boot_rates(units, n=200, seed=3)


def test_boot_rates_on_a_unanimous_arm_has_no_spread() -> None:
    out = _boot_rates([Unit(4, 4, 4.0)] * 12, n=200)
    assert out["tick_rate_ci_low"] == out["tick_rate_ci_high"] == 1.0
    assert out["unit_rate_ci_low"] == out["unit_rate_ci_high"] == 1.0


def test_normal_runs_splits_on_not_normal_and_on_missing_ticks() -> None:
    labels = {
        # A: 8 normal ticks, then acute, then 7 more normal — two runs.
        **{("A", _t(i)): "normal" for i in range(8)},
        ("A", _t(8)): "acute",
        **{("A", _t(i)): "normal" for i in range(9, 16)},
        # B: 7 normal, a GAP at tick 7 (no label at all), then 7 more. A gap is
        # not evidence that normal service continued across it, so it splits.
        **{("B", _t(i)): "normal" for i in range(7)},
        **{("B", _t(i)): "normal" for i in range(8, 15)},
    }
    runs = normal_runs(labels)
    assert sorted(runs) == [
        ("A", _t(0), _t(8)),
        ("A", _t(9), _t(16)),
        ("B", _t(0), _t(7)),
        ("B", _t(8), _t(15)),
    ]


def test_normal_runs_drops_slivers_between_events() -> None:
    """A two-tick gap between two events is not a quiet period."""
    labels = {
        ("A", _t(0)): "acute",
        ("A", _t(1)): "normal",
        ("A", _t(2)): "normal",
        ("A", _t(3)): "chronic",
    }
    assert normal_runs(labels) == []
    assert MIN_NORMAL_RUN_TICKS > 2


def test_grade_scores_the_route_share_not_a_single_cell() -> None:
    """The assigned_n label speaks in routes, so the per-cell calls are lifted to
    the route — as a SHARE of its judged cells, because "any cell disrupted"
    saturates by base rate once every cell is judged."""
    calls = _calls(
        {
            _t(0): {"A|south|A09S": "normal", "A|south|A10S": "normal"},
            _t(1): {"A|south|A09S": "disrupted", "A|south|A10S": "normal"},
            _t(2): {"A|south|A09S": "disrupted", "A|south|A10S": "disrupted"},
        }
    )
    out = grade(
        calls, [Disruption("A", _t(1), _t(3))], [], n_baselined=2, bootstrap=100
    )
    det = out["calls"]["episode_detection"]
    assert det["n_ticks"] == 2
    assert det["alarmed_ticks"] == 2
    # Half the route at t1, all of it at t2.
    assert det["mean_share"] == 0.75
    assert out["calls"]["normal_run_false_alarms"] == {"n_units": 0, "n_ticks": 0}


def test_grade_charges_a_unit_only_for_ticks_the_replay_scored() -> None:
    """The label's clock and the vehicle archive's clock can disagree about which
    ticks exist. Charging an episode for a tick that was never classified would
    depress the tick rate by however much the two archives differ."""
    calls = _calls({_t(1): {"A|south|A09S": "disrupted"}})
    out = grade(
        calls, [Disruption("A", _t(0), _t(6))], [], n_baselined=1, bootstrap=100
    )
    assert out["calls"]["episode_detection"]["n_ticks"] == 1  # not 6
    assert out["calls"]["episode_detection"]["tick_rate"] == 1.0


def test_grade_drops_a_unit_whose_route_was_never_judged() -> None:
    """A route with no judged cells in the window is not a miss, it is an
    absence of evidence — counting it as a miss would credit the classifier's
    blind spots to its false-alarm rate and charge them to its detection rate."""
    calls = _calls({_t(1): {"B|south|B01S": "disrupted"}})
    out = grade(
        calls, [Disruption("A", _t(0), _t(4))], [], n_baselined=2, bootstrap=100
    )
    assert out["calls"]["episode_detection"] == {"n_units": 0, "n_ticks": 0}


def test_grade_ignores_a_disrupted_call_outside_the_episode() -> None:
    calls = _calls(
        {
            _t(0): {"A|south|A09S": "disrupted"},
            _t(5): {"A|south|A09S": "normal"},
            _t(6): {"A|south|A09S": "normal"},
        }
    )
    out = grade(
        calls, [Disruption("A", _t(5), _t(7))], [], n_baselined=1, bootstrap=100
    )
    assert out["calls"]["episode_detection"]["n_ticks"] == 2
    assert out["calls"]["episode_detection"]["alarmed_ticks"] == 0


def test_grade_counts_a_false_alarm_on_a_normal_run() -> None:
    calls = _calls(
        {_t(i): {"A|south|A09S": "normal", "B|south|B01S": "normal"} for i in range(6)}
        | {_t(2): {"A|south|A09S": "disrupted", "B|south|B01S": "normal"}}
    )
    out = grade(
        calls,
        [],
        [("A", _t(0), _t(6)), ("B", _t(0), _t(6))],
        n_baselined=2,
        bootstrap=100,
    )
    fa = out["calls"]["normal_run_false_alarms"]
    assert fa["alarmed_units"] == 1
    assert fa["n_units"] == 2
    assert fa["alarmed_ticks"] == 1
    assert fa["n_ticks"] == 12


def test_grade_coverage_counts_cells_and_the_call_mix() -> None:
    calls = _calls(
        {
            _t(0): {"A|south|A09S": "normal", "A|south|A10S": "quiet"},
            _t(1): {"A|south|A09S": "disrupted"},
        }
    )
    coverage = grade(calls, [], [], n_baselined=4, bootstrap=10)["coverage"]
    assert coverage["n_ticks"] == 2
    assert coverage["judged_per_tick_mean"] == 1.5
    assert coverage["cells_judged_at_least_once"] == 2
    assert coverage["share_of_baselined_cells_judged"] == 0.5
    assert coverage["call_mix_cell_ticks"] == {"disrupted": 1, "normal": 1, "quiet": 1}


def test_latency_is_measured_from_the_onset_itself() -> None:
    """No lead window by default. With one, a surface that is usually already
    firing has every latency pin to exactly -lead_sec and the median reports the
    boundary instead of the classifier — measured at 30 minutes on real data, all
    four policies returned -25 to -30 against a -30 floor."""
    onset = _t(6)
    calls = _calls(
        {
            _t(i): {"A|south|A09S": "disrupted" if i == 8 else "normal"}
            for i in range(12)
        }
    )
    out = grade(calls, [Disruption("A", onset, _t(11))], [], 1, bootstrap=50)["calls"][
        "onset_latency"
    ]
    assert out["lead_tolerance_min"] == 0
    assert out["n_detected"] == 1
    assert out["median_latency_min"] == 10.0  # two ticks after the onset


def test_latency_counts_an_episode_never_alarmed_as_missed_not_as_slow() -> None:
    """Folding a miss in as a large latency would let a blind policy report a
    flattering median. Misses are counted, never imputed."""
    calls = _calls({_t(i): {"A|south|A09S": "normal"} for i in range(12)})
    out = grade(calls, [Disruption("A", _t(6), _t(11))], [], 1, bootstrap=50)["calls"][
        "onset_latency"
    ]
    assert out["n_detected"] == 0
    assert out["n_missed"] == 1
    assert out["median_latency_min"] is None


def test_an_episode_already_alarming_at_onset_is_excluded_not_scored_as_zero() -> None:
    """Its latency is zero for a reason that has nothing to do with detection.
    Counting it would let a constantly-alarming policy report perfect latency."""
    onset = _t(20)
    always = _calls({_t(i): {"A|south|A09S": "disrupted"} for i in range(30)})
    out = grade(
        always, [Disruption("A", onset, onset + 5 * TICK_SECONDS)], [], 1, bootstrap=50
    )["calls"]["onset_latency"]
    assert out["n_offered"] == 1
    assert out["n_alarming_at_onset"] == 1
    assert out["n_episodes"] == 0  # nothing left to measure latency on
    assert out["median_latency_min"] is None
    assert out["fresh"]["n_episodes"] == 0


def test_fresh_cohort_keeps_an_alarm_that_actually_arrives() -> None:
    onset = _t(20)
    calls = _calls(
        {
            _t(i): {"A|south|A09S": "disrupted" if i >= 21 else "normal"}
            for i in range(30)
        }
    )
    out = grade(
        calls, [Disruption("A", onset, onset + 5 * TICK_SECONDS)], [], 1, bootstrap=50
    )["calls"]["onset_latency"]
    assert out["n_alarming_at_onset"] == 0
    assert out["fresh"]["n_episodes"] == 1
    assert out["fresh"]["n_detected"] == 1
    assert out["fresh"]["median_latency_min"] == 5.0


def test_published_surface_holds_a_verdict_the_classifier_stopped_making() -> None:
    """The reason both surfaces are scored. A cell the classifier abstains on
    keeps its last published state, so the snapshot can read disrupted long after
    the evidence stopped — a false alarm a rider sees and the raw calls do not
    show. This is where a wider accumulator window's staleness lands."""
    # Disrupted once, then nothing at all: no call, so no raw alarm...
    calls = _calls(
        {_t(0): {"A|south|A09S": "disrupted"}, **{_t(i): {} for i in range(1, 8)}}
    )
    run = [("A", _t(1), _t(8))]
    out = grade(calls, [], run, n_baselined=1, bootstrap=50)
    assert out["calls"]["normal_run_false_alarms"]["n_ticks"] == 0  # nothing judged
    # ...but the published surface still shows the stale disrupted regime.
    pub = out["published"]["normal_run_false_alarms"]
    assert pub["n_ticks"] == 7
    assert pub["alarmed_ticks"] == 7
    assert pub["tick_rate"] == 1.0
