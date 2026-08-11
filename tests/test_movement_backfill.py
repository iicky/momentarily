"""Offline movement-transition reconstruction (training/movement_backfill.py).

Synthetic predictions/vehicle-archive bodies and raw (tick, calls) sequences —
no R2 access. The R2 fetch (`_fetch_vehicle_bodies`, `load_predictions`, and
the client/bucket entry points) is exercised only via monkeypatched fakes.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import TYPE_CHECKING, Any, cast

import pytest

from training.eval import MovementTransitionRecord, PredictionRecord
from training.movement_backfill import (
    EpisodeCensus,
    episode_census,
    matched_transition_count,
    movement_open_regimes,
    open_regimes_from_ticks,
    reconstruct_movement_transitions,
    route_ticks_from_vehicle_bodies,
    segment_ticks_from_vehicle_bodies,
    ticks_from_predictions,
    transitions_from_ticks,
)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

TICK = 300
T0 = 1_700_000_100  # tick-aligned, ET tod_bin=3 for T0..T0+29*TICK
_D0 = date(2026, 8, 4)
# Placeholder for tests that raise before ever touching the client (validation
# errors, or with load_predictions/_fetch_vehicle_bodies monkeypatched out).
_DUMMY_CLIENT = cast("S3Client", None)

# Explicit alias for hand-built ticks literals: transitions_from_ticks/
# open_regimes_from_ticks accept Mapping (covariant); a bare dict literal
# stored in a local var infers list[tuple[int, dict[str, str]]], which is
# invariant against list[tuple[int, Mapping[str, str]]].
Ticks = list[tuple[int, Mapping[str, str]]]


def t(i: int) -> int:
    return T0 + i * TICK


def _pred(
    *, ts: int, route: str = "A", published_condition: str | None = "normal"
) -> PredictionRecord:
    return PredictionRecord(
        ts=ts,
        route=route,
        condition="normal",
        regime_entered_at=ts,
        p_normal=0.9,
        p_disrupted=0.05,
        p_suspended=0.05,
        p_normal_in_30min=0.9,
        p_normal_in_60min=0.9,
        p_normal_in_120min=0.9,
        recovery_minutes=0,
        recovery_minutes_low=0,
        recovery_minutes_high=0,
        published_condition=published_condition,
    )


def _dir_row(vehicles_n: int, advanced_n: int, stalled_n: int) -> dict[str, int]:
    return {"vehicles_n": vehicles_n, "advanced_n": advanced_n, "stalled_n": stalled_n}


def _route_body(
    tick: int,
    route: str = "A",
    *,
    vehicles_n: int = 10,
    north: tuple[int, int, int] = (5, 5, 0),
    south: tuple[int, int, int] = (5, 5, 0),
) -> dict[str, Any]:
    """north/south = (vehicles_n, advanced_n, stalled_n)."""
    return {
        "observed_at": tick,
        "rows": {
            route: {
                "vehicles_n": vehicles_n,
                "by_direction": {"north": _dir_row(*north), "south": _dir_row(*south)},
            }
        },
    }


def _suspended_body(tick: int, route: str = "A") -> dict[str, Any]:
    return {"observed_at": tick, "rows": {route: {"vehicles_n": 0}}}


def _segment_body(
    tick: int, transitions: dict[str, int], route: str = "F", direction: str = "south"
) -> dict[str, Any]:
    return {
        "observed_at": tick,
        "rows": {route: {"by_direction": {direction: {"transitions": transitions}}}},
    }


# --- ticks_from_predictions -------------------------------------------------


def test_unknown_is_an_abstention_not_a_state() -> None:
    predictions = [
        _pred(ts=t(0), route="A", published_condition="normal"),
        _pred(ts=t(0), route="B", published_condition="unknown"),
    ]
    ticks = ticks_from_predictions(predictions)
    assert dict(ticks[0][1]) == {"A": "normal"}


def test_rows_where_the_arm_never_shipped_are_excluded() -> None:
    predictions = [_pred(ts=t(0), route="A", published_condition=None)]
    assert ticks_from_predictions(predictions) == []


def test_not_scheduled_is_a_real_call_not_excluded() -> None:
    """Unlike `unknown`, `not_scheduled` is a genuine state the Worker's own
    regime clock (deriveMovementStates) feeds — only `unknown` is special-cased."""
    predictions = [_pred(ts=t(0), route="A", published_condition="not_scheduled")]
    ticks = ticks_from_predictions(predictions)
    assert dict(ticks[0][1]) == {"A": "not_scheduled"}


def test_ticks_group_by_ts_across_routes() -> None:
    predictions = [
        _pred(ts=t(0), route="A", published_condition="normal"),
        _pred(ts=t(0), route="B", published_condition="disrupted"),
        _pred(ts=t(1), route="A", published_condition="disrupted"),
    ]
    ticks = ticks_from_predictions(predictions)
    by_tick = dict(ticks)
    assert by_tick[t(0)] == {"A": "normal", "B": "disrupted"}
    assert by_tick[t(1)] == {"A": "disrupted"}


# --- route_ticks_from_vehicle_bodies ----------------------------------------


def test_route_ticks_from_vehicle_bodies_groups_by_tick_and_classifies() -> None:
    """20 clean ticks clear compute_advance_baseline's min_samples=20 gate in one
    tod_bin, then a vehicles_n=0 tick reads suspended without needing a baseline
    at all — proves the (route, tick) -> state grouping wires baseline + classify
    correctly end to end."""
    baseline_ticks = [_route_body(t(i), "A") for i in range(20)]
    suspended_tick = _suspended_body(t(20), "A")
    ticks = route_ticks_from_vehicle_bodies([*baseline_ticks, suspended_tick])
    by_tick = dict(ticks)
    assert by_tick[t(0)]["A"] == "normal"
    assert by_tick[t(20)]["A"] == "suspended"


def test_route_ticks_below_min_samples_abstains() -> None:
    """Fewer than compute_advance_baseline's 20-tick floor leaves no baseline, so
    a moving route can't be judged and is dropped rather than guessed at."""
    ticks = route_ticks_from_vehicle_bodies([_route_body(t(0), "A")])
    assert ticks == []


# --- segment_ticks_from_vehicle_bodies --------------------------------------


def test_segment_key_is_route_pipe_direction_pipe_from_stop() -> None:
    bodies = [_segment_body(t(i), {"A>B": 2}) for i in range(2)]
    ticks = segment_ticks_from_vehicle_bodies(bodies, window_ticks=2)
    keys = {k for _, calls in ticks for k in calls}
    assert keys == {"F|south|A"}


def test_segment_rolling_window_unlocks_a_call_a_single_tick_cannot() -> None:
    """A single tick sees matched=2 < MIN_MATCHED_TRIPS=3 (segments.py) and
    abstains; the 2-tick trailing window accumulates matched=4 and calls it.
    All-advance counts (0 stalled) always classify `normal` regardless of the
    baseline's exact pooled p0 (the posterior's likelihood term dominates any
    p0 in (0, 1] once every matched trip advanced) — see classify_direction."""
    bodies = [_segment_body(t(i), {"A>B": 2}) for i in range(2)]
    ticks = segment_ticks_from_vehicle_bodies(bodies, window_ticks=2)
    by_tick = dict(ticks)
    assert "F|south|A" not in by_tick.get(t(0), {})
    assert by_tick[t(1)]["F|south|A"] == "normal"


def test_segment_narrow_window_never_accumulates_enough() -> None:
    """window_ticks=1 is a no-op trailing window: each tick is judged alone, so
    the same matched=2-per-tick data never clears MIN_MATCHED_TRIPS."""
    bodies = [_segment_body(t(i), {"A>B": 2}) for i in range(3)]
    ticks = segment_ticks_from_vehicle_bodies(bodies, window_ticks=1)
    assert ticks == []


# --- transitions_from_ticks / open_regimes_from_ticks -----------------------


def test_debounce_is_honoured() -> None:
    """A 1-tick blip (normal -> disrupted -> normal) commits at debounce_ticks=1
    (two transitions: the blip's entry and its immediate revert) and is fully
    invisible at debounce_ticks=2 (the candidate run never reaches 2 agreeing
    ticks before it reverts) — mirrors training.regime's own debounce contract,
    threaded through this module's wrapper."""
    ticks: Ticks = [
        (t(0), {"A": "normal"}),
        (t(1), {"A": "disrupted"}),
        (t(2), {"A": "normal"}),
    ]
    fast = transitions_from_ticks(ticks, "route", debounce_ticks=1)
    slow = transitions_from_ticks(ticks, "route", debounce_ticks=2)
    assert [c.new_state for c in fast] == ["disrupted", "normal"]
    assert slow == []


def test_route_scope_route_field_is_the_key() -> None:
    ticks: Ticks = [
        (t(0), {"A": "normal"}),
        (t(1), {"A": "disrupted"}),
        (t(2), {"A": "disrupted"}),
    ]
    transitions = transitions_from_ticks(ticks, "route", debounce_ticks=2)
    assert len(transitions) == 1
    tr = transitions[0]
    assert tr.scope == "route"
    assert tr.key == "A"
    assert tr.route == "A"
    assert tr.prev_state == "normal"
    assert tr.new_state == "disrupted"
    assert tr.regime_entered_at == t(0)


def test_segment_scope_route_field_parses_off_the_key() -> None:
    """`route|direction|from_stop` -> route is the first field, matching
    worker/src/grading.ts's movementTransitions."""
    ticks: Ticks = [
        (t(0), {"F|south|A": "normal"}),
        (t(1), {"F|south|A": "disrupted"}),
        (t(2), {"F|south|A": "disrupted"}),
    ]
    transitions = transitions_from_ticks(ticks, "segment", debounce_ticks=2)
    assert len(transitions) == 1
    tr = transitions[0]
    assert tr.scope == "segment"
    assert tr.key == "F|south|A"
    assert tr.route == "F"


def test_transitions_from_ticks_rejects_unknown_scope() -> None:
    with pytest.raises(ValueError, match="scope"):
        transitions_from_ticks([(t(0), {"A": "normal"})], "line")


def test_still_open_regime_appears_in_open_regimes_not_in_transitions() -> None:
    """A regime that never closes by the window boundary is a censored
    observation (dwell.py's OpenRegimes), not a completed transition."""
    ticks: Ticks = [(t(0), {"A": "disrupted"}), (t(1), {"A": "disrupted"})]
    transitions = transitions_from_ticks(ticks, "route")
    open_regimes = open_regimes_from_ticks(ticks)
    assert transitions == []
    assert open_regimes == {"A": ("disrupted", t(0))}


def test_open_regimes_omits_a_key_evicted_by_idle_timeout() -> None:
    ticks: Ticks = [(t(0), {"A": "disrupted"}), (t(0) + 4000, {"B": "normal"})]
    open_regimes = open_regimes_from_ticks(ticks, max_idle_sec=3600)
    assert "A" not in open_regimes
    assert open_regimes["B"] == ("normal", t(0) + 4000)


# --- C1 contract: source dispatch + validation ------------------------------


def test_reconstruct_movement_transitions_rejects_bad_scope() -> None:
    with pytest.raises(ValueError, match="scope"):
        reconstruct_movement_transitions(
            client=_DUMMY_CLIENT,
            bucket="b",
            start_date=_D0,
            end_date=_D0,
            scope="line",
        )


def test_reconstruct_movement_transitions_rejects_bad_source() -> None:
    with pytest.raises(ValueError, match="source"):
        reconstruct_movement_transitions(
            client=_DUMMY_CLIENT,
            bucket="b",
            start_date=_D0,
            end_date=_D0,
            source="carrier_pigeon",
        )


def test_segment_scope_has_no_published_condition_source() -> None:
    with pytest.raises(ValueError, match="segment"):
        reconstruct_movement_transitions(
            client=_DUMMY_CLIENT,
            bucket="b",
            start_date=_D0,
            end_date=_D0,
            scope="segment",
            source="predictions",
        )


def test_auto_source_resolves_route_to_predictions_and_segment_to_vehicles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_load_predictions(
        client: Any, bucket: Any, start: Any, end: Any
    ) -> list[Any]:
        calls.append("predictions")
        return []

    def fake_fetch_vehicle_bodies(
        client: Any, bucket: Any, start: Any, end: Any
    ) -> list[Any]:
        calls.append("vehicles")
        return []

    monkeypatch.setattr(
        "training.movement_backfill.load_predictions", fake_load_predictions
    )
    monkeypatch.setattr(
        "training.movement_backfill._fetch_vehicle_bodies", fake_fetch_vehicle_bodies
    )

    reconstruct_movement_transitions(
        client=_DUMMY_CLIENT, bucket="b", start_date=_D0, end_date=_D0, scope="route"
    )
    movement_open_regimes(
        client=_DUMMY_CLIENT, bucket="b", start_date=_D0, end_date=_D0, scope="segment"
    )
    assert calls == ["predictions", "vehicles"]


# --- episode_census / matched_transition_count ------------------------------


def _mtr(
    *,
    key: str = "A",
    route: str = "A",
    prev_state: str = "disrupted",
    new_state: str = "normal",
    entered_at: int = t(0),
    exited_at: int = t(2),
    scope: str = "route",
) -> MovementTransitionRecord:
    return MovementTransitionRecord(
        ts=exited_at,
        scope=scope,
        key=key,
        route=route,
        prev_state=prev_state,
        new_state=new_state,
        regime_entered_at=entered_at,
        exited_at=exited_at,
        dwell_sec=exited_at - entered_at,
    )


def test_episode_census_excludes_normal_exits_and_counts_min_voter_cells() -> None:
    transitions = [
        _mtr(key="A", prev_state="disrupted", entered_at=t(0), exited_at=t(2)),
        _mtr(key="A", prev_state="suspended", entered_at=t(2), exited_at=t(4)),
        _mtr(key="A", prev_state="disrupted", entered_at=t(4), exited_at=t(6)),
        _mtr(key="B", prev_state="disrupted", entered_at=t(0), exited_at=t(1)),
        # A normal exit is a real transition (feeds dwell_movement's own-state
        # curve) but is not an incident episode.
        _mtr(key="A", prev_state="normal", entered_at=t(6), exited_at=t(8)),
    ]
    census = episode_census(transitions)
    assert census == EpisodeCensus(
        n_episodes=4,
        median_duration_min=10.0,
        min_duration_min=5.0,
        max_duration_min=10.0,
        distinct_cells=2,
        cells_with_ge_min_voter_events=1,
    )


def test_matched_transition_count_is_route_state_and_one_tick_entered_at() -> None:
    a = [
        _mtr(route="A", new_state="disrupted", entered_at=t(0)),
        _mtr(route="B", new_state="disrupted", entered_at=t(10)),
    ]
    b = [
        _mtr(route="A", new_state="disrupted", entered_at=t(1)),  # within one tick
        _mtr(route="C", new_state="disrupted", entered_at=t(0)),  # different route
    ]
    assert matched_transition_count(a, b) == 1
    assert matched_transition_count(b, a) == 1


def test_matched_transition_count_is_one_to_one() -> None:
    """Each `b` transition can satisfy at most one `a` match, even if it would
    also match a second."""
    a = [
        _mtr(route="A", new_state="disrupted", entered_at=t(0)),
        _mtr(route="A", new_state="disrupted", entered_at=t(0)),
    ]
    b = [_mtr(route="A", new_state="disrupted", entered_at=t(0))]
    assert matched_transition_count(a, b) == 1
