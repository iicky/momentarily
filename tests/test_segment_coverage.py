"""Segment-flow accumulator sweep replay (training/segment_coverage.py).
Hermetic — no R2/network; the archive-touching pieces live behind main()."""

from __future__ import annotations

from datetime import date

import pytest

from training.gtfs_static import Pattern
from training.hierarchical import PooledCell
from training.incidents import parse_topology
from training.segment_coverage import (
    PRUNE_MATCHED,
    RECALL_FLOOR_RATIO,
    STATUS_QUO_DECAY,
    STATUS_QUO_FLOOR,
    PatternIndex,
    PolicyResult,
    TickTransitions,
    _intervening_stops,  # pyright: ignore[reportPrivateUsage]
    _route_direction_of_code,  # pyright: ignore[reportPrivateUsage]
    advance_accumulator,
    alerted_routes_by_tick,
    baselined_cells_by_route,
    classify_cells,
    classify_with_corridor,
    evaluate_policy,
    observed_jump_pairs,
    representative_service_dates,
    select_policy,
    size_hop_map,
    tick_counts,
    ticks_from_bodies,
)


def _approx(expected: float) -> object:
    """Typed wrapper around ``pytest.approx``; see tests/test_dwell.py."""
    return pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]


def _cell(p0: float, n: int = 1000) -> PooledCell:
    return PooledCell(
        p0=p0, raw=p0, n=n, alpha=30 * p0, beta=30 * (1 - p0), source="test"
    )


def _pattern(stops: tuple[str, ...], hop_seconds: int = 60) -> Pattern:
    offsets = tuple(i * hop_seconds for i in range(len(stops)))
    return Pattern(
        stops=stops, offsets=offsets, index={s: i for i, s in enumerate(stops)}
    )


def _tick(
    transitions: dict[str, int],
    *,
    route: str = "F",
    direction: str = "south",
    tick: int = 0,
    day: date = date(2026, 8, 21),
) -> TickTransitions:
    return TickTransitions(
        tick=tick, service_day=day, routes={route: {direction: transitions}}
    )


# --- ticks_from_bodies: parsing the archive into replay ticks --------------


def test_ticks_from_bodies_sums_transitions_for_the_same_snapped_tick():
    bodies = [
        {
            "observed_at": 300_000,
            "rows": {"F": {"by_direction": {"south": {"transitions": {"A>B": 3}}}}},
        },
        {
            "observed_at": 300_010,  # same 300s tick as above
            "rows": {"F": {"by_direction": {"south": {"transitions": {"A>B": 2}}}}},
        },
    ]
    ticks = ticks_from_bodies(bodies)
    assert len(ticks) == 1
    assert ticks[0].routes["F"]["south"]["A>B"] == 5


def test_ticks_from_bodies_drops_non_positive_and_malformed_pairs():
    bodies = [
        {
            "observed_at": 300_000,
            "rows": {
                "F": {
                    "by_direction": {
                        "south": {"transitions": {"A>B": 0, "noarrow": 5, "A>C": 4}}
                    }
                }
            },
        }
    ]
    ticks = ticks_from_bodies(bodies)
    assert ticks[0].routes["F"]["south"] == {"A>C": 4}


# --- advance_accumulator: the decayed EWMA arithmetic -----------------------


def test_advance_accumulator_decays_previous_and_adds_this_ticks_count():
    prev = {"F|south|A": (2.0, 4.0)}
    counts = {"F|south|A": (1.0, 1.0)}
    out = advance_accumulator(prev, counts, decay=0.8, tracked=frozenset({"F|south|A"}))
    assert out["F|south|A"] == (_approx(1.0 + 0.8 * 2.0), _approx(1.0 + 0.8 * 4.0))


def test_advance_accumulator_prunes_a_cell_once_decayed_matched_goes_negligible():
    prev = {"F|south|A": (0.1, 0.2)}  # 0.2 * decay < PRUNE_MATCHED for any decay <= 1
    out = advance_accumulator(prev, {}, decay=0.8, tracked=frozenset({"F|south|A"}))
    assert PRUNE_MATCHED > 0.2 * 0.8
    assert out == {}


def test_advance_accumulator_ignores_a_key_the_baseline_never_tracked():
    counts = {"F|south|A": (5.0, 5.0), "F|south|UNTRACKED": (5.0, 5.0)}
    out = advance_accumulator({}, counts, decay=0.8, tracked=frozenset({"F|south|A"}))
    assert set(out) == {"F|south|A"}


def test_advance_accumulator_compounds_a_steady_signal_into_a_geometric_series():
    tracked = frozenset({"F|south|A"})
    state: dict[str, tuple[float, float]] = {}
    for _ in range(3):
        state = advance_accumulator(
            state, {"F|south|A": (1.0, 1.0)}, decay=0.5, tracked=tracked
        )
    # 1 + 0.5 + 0.25 = 1.75, matched == advanced since every tick fully advances.
    a, m = state["F|south|A"]
    assert a == _approx(1.75)
    assert m == _approx(1.75)


# --- classify_cells: the floor gate -----------------------------------------


def test_classify_cells_does_not_judge_a_cell_below_the_floor():
    state = {"F|south|A": (4.0, 4.0)}
    baseline = {"F|south|A": _cell(0.9)}
    calls, spans = classify_cells(
        state, baseline, frozenset({"F|south|A"}), min_eff_matched=5
    )
    assert calls == {}
    assert spans == []


def test_classify_cells_judges_a_cell_exactly_at_the_floor():
    state = {"F|south|A": (4.5, 5.0)}
    baseline = {"F|south|A": _cell(0.9)}
    calls, _ = classify_cells(
        state, baseline, frozenset({"F|south|A"}), min_eff_matched=5
    )
    assert calls == {"F|south|A": "normal"}


def test_classify_cells_requires_an_adjacency_entry_even_with_a_baseline():
    state = {"F|south|A": (5.0, 5.0)}
    baseline = {"F|south|A": _cell(0.9)}
    calls, _ = classify_cells(state, baseline, frozenset(), min_eff_matched=5)
    assert calls == {}


# --- tick_counts: single-hop credit vs path expansion -----------------------


def test_tick_counts_credits_only_the_from_stop_when_not_expanded():
    out = tick_counts(_tick({"A>C": 4}), expand=False, patterns=None)
    assert out == {"F|south|A": (4.0, 4.0)}


def test_tick_counts_stall_credits_matched_but_not_advanced():
    out = tick_counts(_tick({"A>A": 3}), expand=False, patterns=None)
    assert out == {"F|south|A": (0.0, 3.0)}


def test_tick_counts_expands_a_jump_along_a_unique_scheduled_pattern():
    local = _pattern(("A", "B", "C"))
    idx = PatternIndex(day_patterns=lambda _d: {"F..S01R": (local,)})
    out = tick_counts(_tick({"A>C": 4}), expand=True, patterns=idx)
    assert out == {"F|south|A": (4.0, 4.0), "F|south|B": (4.0, 4.0)}


def test_tick_counts_falls_back_to_from_stop_when_candidate_patterns_disagree():
    local = _pattern(("A", "B", "C"))
    express = _pattern(("A", "C"))  # skips B: an express variant
    idx = PatternIndex(
        day_patterns=lambda _d: {"F..S01R": (local,), "FX..S02R": (express,)}
    )
    out = tick_counts(_tick({"A>C": 4}), expand=True, patterns=idx)
    assert out == {"F|south|A": (4.0, 4.0)}


def test_tick_counts_never_expands_a_stall_even_with_patterns_available():
    local = _pattern(("A", "B", "C"))
    idx = PatternIndex(day_patterns=lambda _d: {"F..S01R": (local,)})
    out = tick_counts(_tick({"A>A": 3}), expand=True, patterns=idx)
    assert out == {"F|south|A": (0.0, 3.0)}


# --- _route_direction_of_code / _intervening_stops: the pattern-level rule --


def test_route_direction_of_code_folds_express_and_reads_the_direction_char():
    assert _route_direction_of_code("7X..N30R") == ("7", "north")
    assert _route_direction_of_code("A..S55R") == ("A", "south")


def test_route_direction_of_code_none_when_the_code_names_neither():
    assert _route_direction_of_code("garbage") is None
    assert _route_direction_of_code("A..") is None


def test_intervening_stops_returns_the_hop_list_when_every_candidate_agrees():
    full = _pattern(("A", "B", "C", "D"))
    short_turn = _pattern(("A", "B", "C"))  # a candidate that still names A..C
    assert _intervening_stops([full, short_turn], "A", "C", max_hops=8) == ("A", "B")


def test_intervening_stops_none_when_candidates_disagree():
    local = _pattern(("A", "B", "C"))
    express = _pattern(("A", "C"))
    assert _intervening_stops([local, express], "A", "C", max_hops=8) is None


def test_intervening_stops_none_with_no_candidate_pattern():
    assert _intervening_stops([], "A", "C", max_hops=8) is None


def test_intervening_stops_none_when_the_unanimous_answer_exceeds_the_cap():
    long = _pattern(tuple("ABCDEFGHIJ"))  # A(0)..J(9): 9 hops
    assert _intervening_stops([long], "A", "J", max_hops=8) is None
    assert _intervening_stops([long], "A", "I", max_hops=8) == tuple(
        "ABCDEFGH"
    )  # 8 hops, at the cap


def test_pattern_index_memoizes_the_grouping_per_service_day():
    calls: list[date] = []

    def day_patterns(d: date) -> dict[str, tuple[Pattern, ...]]:
        calls.append(d)
        return {"F..S01R": (_pattern(("A", "B", "C")),)}

    idx = PatternIndex(day_patterns=day_patterns)
    day = date(2026, 8, 21)
    idx.implied_hops(day, "F", "south", "A", "C")
    idx.implied_hops(day, "F", "south", "A", "C")
    assert calls == [day]  # fetched once; the second lookup hit the cache


# --- classify_with_corridor: spatial pooling ---------------------------

_CORRIDOR_ADJACENCY = {
    "F|south|A": {"to": "B", "successors": [{"to": "B", "n_trips": 10}]},
    "F|south|B": {"to": "C", "successors": [{"to": "C", "n_trips": 10}]},
    "F|south|C": {"to": "D", "successors": [{"to": "D", "n_trips": 10}]},
    "F|south|D": {"to": "E", "successors": [{"to": "E", "n_trips": 10}]},
}
_CORRIDOR_TOPOLOGY = parse_topology(_CORRIDOR_ADJACENCY)
_CORRIDOR_KEYS = ("F|south|A", "F|south|B", "F|south|C", "F|south|D")
_CORRIDOR_ADJ_KEYS = frozenset(_CORRIDOR_ADJACENCY)


def _corridor_baseline() -> dict[str, PooledCell]:
    return {k: _cell(0.9) for k in _CORRIDOR_KEYS}


def test_classify_with_corridor_pools_forward_until_the_floor_clears():
    # Each cell alone reads decayed matched 2 (< floor 5); A pools A+B+C to 6.
    state: dict[str, tuple[float, float]] = dict.fromkeys(_CORRIDOR_KEYS, (1.8, 2.0))
    calls, spans = classify_with_corridor(
        state,
        _corridor_baseline(),
        _CORRIDOR_ADJ_KEYS,
        _CORRIDOR_TOPOLOGY,
        min_eff_matched=5,
    )
    assert calls["F|south|A"] == calls["F|south|B"] == calls["F|south|C"] == "normal"
    assert spans == [3]  # one pooled 3-cell corridor judged


def test_classify_with_corridor_attributes_the_pooled_verdict_to_every_cell_in_it():
    state: dict[str, tuple[float, float]] = dict.fromkeys(_CORRIDOR_KEYS, (1.8, 2.0))
    calls, _ = classify_with_corridor(
        state,
        _corridor_baseline(),
        _CORRIDOR_ADJ_KEYS,
        _CORRIDOR_TOPOLOGY,
        min_eff_matched=5,
    )
    verdicts = {calls[k] for k in ("F|south|A", "F|south|B", "F|south|C")}
    assert verdicts == {
        "normal"
    }  # identical verdict, not independently re-derived per cell


def test_classify_with_corridor_never_pools_a_cell_that_can_already_stand_alone():
    state = {"F|south|A": (4.5, 5.0), "F|south|B": (1.0, 1.0)}
    calls, spans = classify_with_corridor(
        state,
        _corridor_baseline(),
        _CORRIDOR_ADJ_KEYS,
        _CORRIDOR_TOPOLOGY,
        min_eff_matched=5,
    )
    assert calls == {"F|south|A": "normal"}
    assert spans == [1]  # judged alone; B's thin count was never asked to pool with it
    assert (
        "F|south|B" not in calls
    )  # B's own 3-cell attempt (B+C+D) never clears the floor


def test_classify_with_corridor_stops_at_a_branch_point_instead_of_guessing():
    branching = {
        "F|south|A": {
            "to": "B",
            "successors": [{"to": "B", "n_trips": 5}, {"to": "X", "n_trips": 5}],
        },
        "F|south|B": {"to": "C", "successors": [{"to": "C", "n_trips": 10}]},
    }
    topology = parse_topology(branching)
    baseline = {"F|south|A": _cell(0.9), "F|south|B": _cell(0.9)}
    adj_keys = frozenset(branching)
    state = {"F|south|A": (1.0, 1.0), "F|south|B": (10.0, 10.0)}
    calls, spans = classify_with_corridor(
        state, baseline, adj_keys, topology, min_eff_matched=5
    )
    # A has two successors (B and X): the chain must stop at A rather than
    # guess which branch the evidence belongs to, so A alone never clears 5.
    assert "F|south|A" not in calls
    assert calls["F|south|B"] == "normal"
    assert spans == [1]


def test_classify_with_corridor_gives_up_at_the_max_corridor_cap():
    keys = [f"F|south|{c}" for c in "ABCDE"]
    adjacency = {
        keys[i]: {
            "to": keys[i + 1].rsplit("|", 1)[-1],
            "successors": [{"to": keys[i + 1].rsplit("|", 1)[-1], "n_trips": 1}],
        }
        for i in range(len(keys) - 1)
    }
    topology = parse_topology(adjacency)
    baseline = {k: _cell(0.9) for k in keys}
    adj_keys = frozenset(adjacency)
    state = dict.fromkeys(keys, (0.1, 0.1))  # deeply under the floor even pooled
    calls, spans = classify_with_corridor(
        state, baseline, adj_keys, topology, min_eff_matched=5, max_corridor=3
    )
    assert calls == {}
    assert spans == []


# --- evaluate_policy: pooled aggregation over a replay -----------------------


def test_evaluate_policy_counts_quiet_disrupted_and_route_agreement():
    baseline = {"F|south|A": _cell(0.5)}
    adj_keys = frozenset({"F|south|A"})
    ticks = [_tick({}, tick=0)]
    states = [{"F|south|A": (0.0, 10.0)}]  # 0/10 advance against p0=0.5: disrupted
    result = evaluate_policy(
        "test",
        ticks,
        states,
        decay=STATUS_QUO_DECAY,
        min_eff_matched=STATUS_QUO_FLOOR,
        expand=False,
        corridor=False,
        baseline=baseline,
        adjacency_keys=adj_keys,
        topology={},
        rd_calls_by_tick={0: {("F", "south"): "disrupted"}},
        truth={},  # no entry anywhere => every route-tick reads "normal" (quiet)
        route_cell_counts={"F": 1},
        alerted_routes={},
    )
    assert result.judged_total == 1
    assert result.disrupted_share == 1.0
    assert result.quiet_denominator == 1
    assert result.quiet_disrupted_rate == 1.0  # quiet tick, disrupted verdict
    assert result.route_agreement_denominator == 1
    assert result.route_agreement == 1.0  # segment call matches the route call
    assert (
        result.alert_denominator == 0
    )  # no alert anywhere -> the recall stratum is empty


def test_evaluate_policy_excludes_ticks_with_an_active_severe_alert_from_the_quiet_rate():
    baseline = {"F|south|A": _cell(0.5)}
    adj_keys = frozenset({"F|south|A"})
    ticks = [_tick({}, tick=0)]
    states = [{"F|south|A": (0.0, 10.0)}]
    result = evaluate_policy(
        "test",
        ticks,
        states,
        decay=STATUS_QUO_DECAY,
        min_eff_matched=STATUS_QUO_FLOOR,
        expand=False,
        corridor=False,
        baseline=baseline,
        adjacency_keys=adj_keys,
        topology={},
        rd_calls_by_tick={},
        truth={("F", 0): "disrupted"},  # a severe alert IS active this route-tick
        route_cell_counts={"F": 1},
        alerted_routes={0: ["F"]},
    )
    assert result.judged_total == 1
    assert result.quiet_denominator == 0  # excluded, not counted as quiet-and-normal
    assert result.quiet_disrupted_rate == 0.0


def test_evaluate_policy_recall_denominator_includes_abstained_cells_not_just_judged_ones():
    """The defect this metric exists to catch: a policy that judges only ONE
    of a route's two segments during an active alert must not read a
    perfect recall just because the one cell it looked at happened to read
    disrupted — the segment it abstained on has to drag the rate down, the
    same way a wrong verdict would."""
    baseline = {"F|south|A": _cell(0.5), "F|south|B": _cell(0.5)}
    adj_keys = frozenset({"F|south|A", "F|south|B"})
    ticks = [_tick({}, tick=0)]
    # A never accumulated anything (absent from state -> abstains). B is
    # judged and correctly reads disrupted.
    states = [{"F|south|B": (0.0, 10.0)}]
    result = evaluate_policy(
        "test",
        ticks,
        states,
        decay=STATUS_QUO_DECAY,
        min_eff_matched=STATUS_QUO_FLOOR,
        expand=False,
        corridor=False,
        baseline=baseline,
        adjacency_keys=adj_keys,
        topology={},
        rd_calls_by_tick={},
        truth={("F", 0): "disrupted"},
        route_cell_counts={"F": 2},
        alerted_routes={0: ["F"]},
    )
    assert result.judged_total == 1  # only B
    assert (
        result.alert_denominator == 2
    )  # BOTH of route F's cells, not just the judged one
    assert result.alert_judged_share == 0.5  # looked at half the route
    assert (
        result.alert_disrupted_given_judged == 1.0
    )  # the one it looked at read disrupted
    assert result.alert_disrupted_rate == 0.5  # NOT 1.0 — A's abstention drags it down


def test_evaluate_policy_recall_is_perfect_when_every_alerted_cell_reads_disrupted():
    baseline = {"F|south|A": _cell(0.5), "F|south|B": _cell(0.5)}
    adj_keys = frozenset({"F|south|A", "F|south|B"})
    ticks = [_tick({}, tick=0)]
    states = [{"F|south|A": (0.0, 10.0), "F|south|B": (0.0, 10.0)}]
    result = evaluate_policy(
        "test",
        ticks,
        states,
        decay=STATUS_QUO_DECAY,
        min_eff_matched=STATUS_QUO_FLOOR,
        expand=False,
        corridor=False,
        baseline=baseline,
        adjacency_keys=adj_keys,
        topology={},
        rd_calls_by_tick={},
        truth={("F", 0): "disrupted"},
        route_cell_counts={"F": 2},
        alerted_routes={0: ["F"]},
    )
    assert result.alert_disrupted_rate == 1.0
    assert result.alert_judged_share == 1.0
    assert result.alert_disrupted_given_judged == 1.0


# --- baselined_cells_by_route / alerted_routes_by_tick: the recall inputs ---


def test_baselined_cells_by_route_counts_per_route_not_per_direction():
    baseline = {
        "F|south|A": _cell(0.9),
        "F|north|A": _cell(0.9),
        "Q|south|X": _cell(0.9),
    }
    assert baselined_cells_by_route(baseline) == {"F": 2, "Q": 1}


def test_alerted_routes_by_tick_excludes_normal_and_groups_by_tick():
    truth = {("F", 0): "disrupted", ("Q", 0): "normal", ("F", 300): "suspended"}
    assert alerted_routes_by_tick(truth) == {0: ["F"], 300: ["F"]}


# --- select_policy: the pre-committed frontier rule --------------------------


def _result(**over: object) -> PolicyResult:
    base: dict[str, object] = {
        "label": "status quo",
        "decay": STATUS_QUO_DECAY,
        "min_eff_matched": STATUS_QUO_FLOOR,
        "expand": False,
        "corridor": False,
        "n_ticks": 10,
        "baselined_cells": 100,
        "judged_total": 10,
        "judged_per_tick_mean": 1.0,
        "judged_share": 0.01,
        "judged_share_median_tick": 0.01,
        "judged_share_p90_tick": 0.01,
        "disrupted_share": 0.1,
        "window_min": 25.0,
        "quiet_disrupted_rate": 0.02,
        "quiet_denominator": 100,
        "alert_disrupted_rate": 0.4,
        "alert_judged_share": 0.5,
        "alert_disrupted_given_judged": 0.8,
        "alert_denominator": 100,
        "route_agreement": 0.9,
        "route_agreement_denominator": 10,
        "corridor_span_median": None,
        "corridor_span_p90": None,
        "measurement_count": 10,
        "single_measurement_count": 10,
        "corridor_measurement_count": 0,
    }
    base.update(over)
    return PolicyResult(**base)  # pyright: ignore[reportArgumentType]


def test_select_policy_picks_the_best_coverage_within_both_floors():
    status_quo = _result()
    within_budget = _result(
        label="better", judged_share=0.5, quiet_disrupted_rate=0.025
    )
    breaches_quiet_ceiling = _result(
        label="noisy", judged_share=0.9, quiet_disrupted_rate=0.05
    )
    chosen = select_policy([status_quo, within_budget, breaches_quiet_ceiling])
    assert chosen is not None
    assert chosen.label == "better"


def test_select_policy_rejects_a_policy_that_clears_the_quiet_ceiling_by_going_blind():
    """A policy can read perfectly clean (low quiet_disrupted_rate) simply by
    abstaining or calling everything normal — the recall floor exists to
    catch exactly that, even when coverage and the false-alarm proxy both
    look like wins."""
    status_quo = (
        _result()
    )  # alert_disrupted_rate=0.4, so the floor is 0.4 * 0.75 = 0.30
    went_blind = _result(
        label="blind",
        judged_share=0.9,
        quiet_disrupted_rate=0.001,  # clears the false-alarm ceiling easily
        alert_disrupted_rate=0.01,  # far under the 0.30 recall floor
    )
    assert select_policy([status_quo, went_blind]) is None


def test_select_policy_returns_none_when_nothing_beats_the_status_quos_coverage():
    status_quo = _result()
    worse_coverage = _result(
        label="worse", judged_share=0.005, quiet_disrupted_rate=0.02
    )
    assert select_policy([status_quo, worse_coverage]) is None


def test_recall_floor_ratio_is_the_documented_075():
    assert RECALL_FLOOR_RATIO == 0.75


# --- observed_jump_pairs / representative_service_dates / size_hop_map: ---
# --- sizing the trainer-published hop map (the shippable form of expand) --


def test_observed_jump_pairs_excludes_stalls_and_dedupes_across_ticks():
    ticks = [
        _tick({"A>B": 3, "A>A": 5}, tick=0),
        _tick({"A>B": 2, "B>C": 1}, tick=300),
    ]
    assert observed_jump_pairs(ticks) == {
        ("F", "south", "A", "B"),
        ("F", "south", "B", "C"),
    }


def test_representative_service_dates_picks_the_most_recent_of_each_class():
    ticks = [
        _tick({}, tick=0, day=date(2026, 8, 17)),  # Monday -> weekday
        _tick({}, tick=1, day=date(2026, 8, 19)),  # Wednesday -> weekday, more recent
        _tick({}, tick=2, day=date(2026, 8, 22)),  # Saturday
        _tick({}, tick=3, day=date(2026, 8, 23)),  # Sunday
    ]
    assert representative_service_dates(ticks) == [
        ("weekday", date(2026, 8, 19)),
        ("saturday", date(2026, 8, 22)),
        ("sunday", date(2026, 8, 23)),
    ]


def test_representative_service_dates_omits_a_class_never_seen():
    ticks = [_tick({}, tick=0, day=date(2026, 8, 19))]  # weekday only
    assert representative_service_dates(ticks) == [("weekday", date(2026, 8, 19))]


def test_size_hop_map_counts_only_unambiguous_pairs_and_reports_real_serialized_bytes():
    local = _pattern(("A", "B", "C"))
    idx = PatternIndex(day_patterns=lambda _d: {"F..S01R": (local,)})
    pairs = {("F", "south", "A", "C"), ("F", "south", "X", "Y")}  # X>Y never resolves
    service_dates = [("weekday", date(2026, 8, 19))]
    result = size_hop_map(pairs, idx, service_dates)
    assert result["observed_pairs"] == 2
    assert result["entries_per_class"] == {"weekday": 1}  # only A>C resolves
    assert result["total_entries"] == 1
    # The byte count is a real serialization, not an average-entry estimate.
    import json

    expected = {"weekday": {"F|south|A|C": ["A", "B"]}}
    assert result["json_bytes"] == len(
        json.dumps(expected, separators=(",", ":")).encode("utf-8")
    )


def test_classify_with_corridor_pooled_null_is_m_weighted_not_a_plain_average():
    """A is far below the floor alone (m rounds to 2) and has a HIGH
    baseline (0.95); B carries almost all the pooled evidence (m=17.6) with
    a LOW baseline (0.05). The correctly weighted null (~0.158) reads
    "normal" at advanced=3/matched=20; the WRONG plain-average null (0.5)
    would read "disrupted" for the IDENTICAL observation — this is the case
    that would have caught the unweighted-mean bug (2026-08-23)."""
    adjacency = {
        "F|south|A": {"to": "B", "successors": [{"to": "B", "n_trips": 10}]},
        "F|south|B": {"to": "C", "successors": [{"to": "C", "n_trips": 10}]},
    }
    topology = parse_topology(adjacency)
    baseline = {
        "F|south|A": _cell(0.95),
        "F|south|B": _cell(0.05),
    }
    adj_keys = frozenset(adjacency)
    state = {"F|south|A": (0.0, 2.4), "F|south|B": (3.0, 17.6)}
    calls, _ = classify_with_corridor(
        state, baseline, adj_keys, topology, min_eff_matched=3
    )
    assert calls == {"F|south|A": "normal", "F|south|B": "normal"}


def test_classify_with_corridor_zero_m_member_contributes_nothing_to_the_pooled_null():
    """A(p0 0.9, m=1) -> B(p0 0.9, ABSENT from state -> m=0) -> C(p0 0.05,
    m=15). B mirrors A's p0 exactly, so a plain (unweighted) average of the
    three p0s barely moves off 0.9 — reading "disrupted" at
    advanced=1/matched=16. The correctly weighted null excludes B entirely
    (0 weight) and is dominated by C's low baseline instead, reading
    "normal" for the identical observation."""
    adjacency = {
        "F|south|A": {"to": "B", "successors": [{"to": "B", "n_trips": 10}]},
        "F|south|B": {"to": "C", "successors": [{"to": "C", "n_trips": 10}]},
        "F|south|C": {"to": "D", "successors": [{"to": "D", "n_trips": 10}]},
    }
    topology = parse_topology(adjacency)
    baseline = {
        "F|south|A": _cell(0.9),
        "F|south|B": _cell(0.9),
        "F|south|C": _cell(0.05),
    }
    adj_keys = frozenset(adjacency)
    # "F|south|B" deliberately absent — never observed this tick.
    state = {"F|south|A": (1.0, 1.0), "F|south|C": (0.0, 15.0)}
    calls, _ = classify_with_corridor(
        state, baseline, adj_keys, topology, min_eff_matched=3
    )
    assert calls == {
        "F|south|A": "normal",
        "F|south|B": "normal",
        "F|south|C": "normal",
    }


def test_classify_with_corridor_never_double_claims_a_cell_two_chains_converge_on():
    """Two different anchors (A, B) can each have exactly one raw successor
    and still both point at the same downstream cell (C) — a merge, not a
    branch (branches are ruled out by the raw-successor-COUNT check, which
    only looks at outgoing edges; it says nothing about incoming ones).
    Without guarding against this, C's evidence would get pooled into BOTH
    corridors and its published verdict would depend on iteration order —
    whichever corridor committed last would silently overwrite the other's
    reading of the same cell."""
    adjacency = {
        "F|south|A": {"to": "C", "successors": [{"to": "C", "n_trips": 5}]},
        "F|south|B": {"to": "C", "successors": [{"to": "C", "n_trips": 5}]},
        "F|south|C": {"to": "D", "successors": [{"to": "D", "n_trips": 5}]},
    }
    topology = parse_topology(adjacency)
    baseline = {k: _cell(0.9) for k in adjacency}
    adj_keys = frozenset(adjacency)
    # A alone (m=2) needs C's evidence to clear floor 3; so does B, if it
    # were allowed to reach C first.
    state = {"F|south|A": (1.8, 2.0), "F|south|B": (1.8, 2.0), "F|south|C": (1.8, 2.0)}
    calls, spans = classify_with_corridor(
        state, baseline, adj_keys, topology, min_eff_matched=3
    )
    # Baseline (and eligible) order is A, B, C: A claims C first.
    assert calls["F|south|A"] == calls["F|south|C"]
    assert (
        "F|south|B" not in calls
    )  # B could not double-claim C; alone it never clears 3
    assert spans == [2]  # exactly one corridor, not two sharing a cell
