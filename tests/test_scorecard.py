"""Synthetic-fixture tests for the event-based eval scorecard (training/scorecard.py).

Builds truth episodes via training.episodes.extract_episodes and model episodes
via training.scorecard.model_episodes on the same 5-min grid convention as
tests/test_episodes.py (WS grid origin, g(k) the k-th tick). No MTA key, no R2,
no network — everything here is a hand-built fixture.
"""

from __future__ import annotations

from typing import Any

import pytest

from training.episodes import extract_episodes
from training.eval import TICK_SECONDS, PredictionRecord
from training.scorecard import (
    dwell_lookup_from_params,
    episode_recovery,
    episode_scorecard,
    false_alarms,
    model_episodes,
    onset_latency,
)

WS = 1_700_000_100  # grid-aligned: snap_tick(WS) == WS


def g(k: int) -> int:
    """The k-th 5-min grid tick starting at WS."""
    return WS + TICK_SECONDS * k


def _pred(route: str, ts: int, condition: str) -> PredictionRecord:
    """Minimal PredictionRecord — model_episodes only reads ts/route/condition,
    so every other required field gets an arbitrary placeholder value."""
    return PredictionRecord(
        ts=ts,
        route=route,
        condition=condition,
        regime_entered_at=ts,
        p_normal=0.0,
        p_disrupted=0.0,
        p_suspended=0.0,
        p_normal_in_30min=0.0,
        p_normal_in_60min=0.0,
        p_normal_in_120min=0.0,
        recovery_minutes=0,
        recovery_minutes_low=0,
        recovery_minutes_high=0,
    )


def _approx(expected: float) -> object:
    """Typed wrapper around ``pytest.approx``.

    pytest's ``approx`` leaks ``Unknown`` through its ``ApproxBase`` return type
    under strict mode, so we pin the boundary to ``object`` once here.
    """
    return pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]


# --- model_episodes: segmenting the published-condition stream -------------------


def test_model_episodes_opens_only_on_disrupted_or_suspended() -> None:
    """A run only opens on 'disrupted'/'suspended'; every other condition
    ('normal', 'unknown') reads as normal and, if a run is active, ends it
    without opening one of its own."""
    preds = [
        _pred("A", g(1), "unknown"),
        _pred("A", g(2), "normal"),
        _pred("A", g(3), "disrupted"),
        _pred("A", g(4), "suspended"),
        _pred("A", g(5), "unknown"),  # ends the run
    ]
    eps = model_episodes(preds, window_start=g(0), window_end=g(6))

    assert len(eps) == 1
    ep = eps[0]
    assert ep.onset == g(3)
    assert ep.recovery == g(5)
    assert ep.peak_state == "suspended"


def test_model_episodes_adjacent_runs_split_by_one_normal_tick() -> None:
    """Two disrupted runs separated by exactly one normal-condition tick are
    two distinct episodes, not merged into one."""
    preds = [
        _pred("A", g(1), "disrupted"),
        _pred("A", g(2), "disrupted"),
        _pred("A", g(3), "normal"),
        _pred("A", g(4), "disrupted"),
        _pred("A", g(5), "disrupted"),
    ]
    eps = model_episodes(preds, window_start=g(0), window_end=g(7))

    assert len(eps) == 2
    first, second = eps
    assert (first.onset, first.recovery) == (g(1), g(3))
    assert (second.onset, second.recovery) == (g(4), g(6))


# --- onset_latency -----------------------------------------------------------------


def test_onset_latency_overlapping_model_episode_detects_with_signed_latency() -> None:
    """A model episode overlapping the truth episode detects it; latency is
    signed minutes (model onset minus truth onset), positive when the model
    lags."""
    truth_eps = extract_episodes(
        {("A", g(2)): "disrupted", ("A", g(3)): "disrupted"},
        {},
        window_start=g(0),
        window_end=g(6),
    )
    model_eps = model_episodes(
        [_pred("A", g(3), "disrupted"), _pred("A", g(4), "disrupted")],
        window_start=g(0),
        window_end=g(6),
    )
    result = onset_latency(truth_eps, model_eps)

    assert result["n_episodes"] == 1
    assert result["n_detected"] == 1
    assert result["n_missed"] == 0
    assert result["detection_rate"] == 1.0
    assert result["median_latency_min"] == _approx(5.0)
    assert result["mean_latency_min"] == _approx(5.0)


def test_onset_latency_no_overlapping_model_episode_counts_as_missed() -> None:
    """A truth episode with no overlapping model episode at all counts as
    missed, dragging detection_rate to 0."""
    truth_eps = extract_episodes(
        {("A", g(2)): "disrupted", ("A", g(3)): "disrupted"},
        {},
        window_start=g(0),
        window_end=g(6),
    )
    result = onset_latency(truth_eps, [])

    assert result["n_episodes"] == 1
    assert result["n_detected"] == 0
    assert result["n_missed"] == 1
    assert result["detection_rate"] == 0.0
    assert result["median_latency_min"] is None
    assert result["mean_latency_min"] is None


def test_onset_latency_model_episode_before_truth_onset_does_not_detect() -> None:
    """A model episode that recovers strictly before the truth episode's onset
    has no time overlap — it does not count as a detection."""
    truth_eps = extract_episodes(
        {("A", g(5)): "disrupted", ("A", g(6)): "disrupted"},
        {},
        window_start=g(0),
        window_end=g(9),
    )
    model_eps = model_episodes(
        [_pred("A", g(1), "disrupted"), _pred("A", g(2), "disrupted")],
        window_start=g(0),
        window_end=g(9),
    )
    result = onset_latency(truth_eps, model_eps)

    assert result["n_detected"] == 0
    assert result["n_missed"] == 1


# --- the key invariant: detection and false-alarm share one overlap predicate ----


def test_detection_and_false_alarm_partition_the_model_episodes() -> None:
    """onset_latency and false_alarms use the identical overlap predicate: a
    model episode overlapping the truth is a detection and never a false
    alarm; one overlapping nothing is a false alarm and never a detection.
    With one truth episode and two model episodes (one overlapping it, one
    not), n_detected and n_false_alarm partition the model episodes exactly."""
    truth_eps = extract_episodes(
        {("A", g(2)): "disrupted", ("A", g(3)): "disrupted"},
        {},
        window_start=g(0),
        window_end=g(10),
    )
    model_eps = model_episodes(
        [
            _pred("A", g(3), "disrupted"),  # overlaps the truth episode
            _pred("A", g(8), "disrupted"),  # overlaps nothing
        ],
        window_start=g(0),
        window_end=g(10),
    )
    assert len(model_eps) == 2

    latency = onset_latency(truth_eps, model_eps)
    fa = false_alarms(model_eps, truth_eps, {})

    assert latency["n_detected"] == 1
    assert fa["n_false_alarm"] == 1
    assert latency["n_detected"] + fa["n_false_alarm"] == len(model_eps)


# --- false_alarms: movement-truth classification ------------------------------------


def test_false_alarms_movement_all_normal_contradicts() -> None:
    """A false-alarm episode whose every movement-truth tick reads 'normal' is
    a genuine over-call: movement_contradicted, not confirmed."""
    model_eps = model_episodes(
        [_pred("A", g(1), "disrupted"), _pred("A", g(2), "disrupted")],
        window_start=g(0),
        window_end=g(5),
    )
    movement_truth = {("A", g(1)): "normal", ("A", g(2)): "normal"}
    result = false_alarms(model_eps, [], movement_truth)

    assert result["n_false_alarm"] == 1
    assert result["movement_contradicted"] == 1
    assert result["movement_confirmed"] == 0
    assert result["movement_unjudgeable"] == 0


def test_false_alarms_movement_all_not_normal_confirms() -> None:
    """A false-alarm episode whose movement-truth ticks are all not-normal is
    a real incident the alert truth missed: movement_confirmed."""
    model_eps = model_episodes(
        [_pred("A", g(1), "disrupted"), _pred("A", g(2), "disrupted")],
        window_start=g(0),
        window_end=g(5),
    )
    movement_truth = {("A", g(1)): "disrupted", ("A", g(2)): "suspended"}
    result = false_alarms(model_eps, [], movement_truth)

    assert result["movement_confirmed"] == 1
    assert result["movement_contradicted"] == 0
    assert result["movement_unjudgeable"] == 0


def test_false_alarms_movement_no_ticks_is_unjudgeable() -> None:
    """A false-alarm episode with no movement-truth entries for any of its
    ticks cannot be judged either way."""
    model_eps = model_episodes(
        [_pred("A", g(1), "disrupted"), _pred("A", g(2), "disrupted")],
        window_start=g(0),
        window_end=g(5),
    )
    result = false_alarms(model_eps, [], {})

    assert result["movement_unjudgeable"] == 1
    assert result["movement_confirmed"] == 0
    assert result["movement_contradicted"] == 0


@pytest.mark.parametrize(
    ("min_frac", "confirmed", "contradicted"),
    [
        pytest.param(0.5, 1, 0, id="ratio_at_boundary_confirms"),
        pytest.param(0.51, 0, 1, id="ratio_just_below_boundary_contradicts"),
    ],
)
def test_false_alarms_min_frac_boundary_is_inclusive(
    min_frac: float, confirmed: int, contradicted: int
) -> None:
    """not_normal/judged is compared with `>=`, so a ratio exactly at
    min_frac confirms; a min_frac just past that same ratio contradicts."""
    model_eps = model_episodes(
        [
            _pred("A", g(1), "disrupted"),
            _pred("A", g(2), "disrupted"),
            _pred("A", g(3), "disrupted"),
            _pred("A", g(4), "disrupted"),
        ],
        window_start=g(0),
        window_end=g(6),
    )
    movement_truth = {
        ("A", g(1)): "disrupted",
        ("A", g(2)): "disrupted",
        ("A", g(3)): "normal",
        ("A", g(4)): "normal",
    }
    result = false_alarms(model_eps, [], movement_truth, min_frac=min_frac)

    assert result["movement_confirmed"] == confirmed
    assert result["movement_contradicted"] == contradicted


# --- episode_recovery: censoring and curve availability gate scoring ---------------


def test_episode_recovery_excludes_censored_and_curve_less_episodes() -> None:
    """Right/left-censored episodes are excluded from scoring and counted in
    n_censored_excluded; an uncensored episode whose peak_state has no dwell
    curve is counted in n_no_curve; only the remaining uncensored, curve-
    backed episode is actually scored."""
    truth = {
        ("R", g(8)): "disrupted",
        ("R", g(9)): "disrupted",  # right-censored: active at the last tick
        ("L", g(0)): "disrupted",
        ("L", g(1)): "disrupted",  # left-censored: active at the first tick
        ("N", g(3)): "disrupted",
        ("N", g(4)): "disrupted",  # uncensored, no dwell curve for N
        ("S", g(5)): "disrupted",
        ("S", g(6)): "disrupted",  # uncensored, has a dwell curve
    }
    truth_eps = extract_episodes(truth, {}, window_start=g(0), window_end=g(9))
    assert len(truth_eps) == 4

    def lookup(route: str, state: str) -> tuple[list[int], list[float] | None] | None:
        if route == "S" and state == "disrupted":
            return [300, 600, 900], None
        return None

    result = episode_recovery(truth_eps, lookup)

    assert result["n_censored_excluded"] == 2
    assert result["n_no_curve"] == 1
    assert result["n_scored"] == 1
    assert result["report"]["n"] == 1


# --- dwell_lookup_from_params --------------------------------------------------------


def test_dwell_lookup_from_params_reads_curve_and_tail_or_none() -> None:
    """A present (route, state) cell resolves to (curve_sec, tail_ll); a
    missing route, missing state, or a cell with no usable curve all read
    as None."""
    params: dict[str, Any] = {
        "routes": {
            "A": {
                "dwell_quantiles": {
                    "disrupted": {
                        "curve_sec": [300, 600, 900],
                        "tail_ll": [1.5, 400.0],
                    },
                    "suspended": {"curve_sec": [300]},  # too short: no curve
                    "unknown": {},  # empty cell: no curve
                }
            }
        }
    }
    lookup = dwell_lookup_from_params(params)

    assert lookup("A", "disrupted") == ([300, 600, 900], [1.5, 400.0])
    assert lookup("A", "suspended") is None
    assert lookup("A", "unknown") is None
    assert lookup("A", "missing_state") is None
    assert lookup("missing_route", "disrupted") is None


# --- episode_scorecard: the verified oracle -----------------------------------------


def test_episode_scorecard_matches_the_verified_oracle() -> None:
    """End-to-end cross-check against the hand-verified scenario: truth
    incident A at g2..g5 (one truth episode, onset g2); predictions put A
    disrupted g3..g6 (detected, 5-min lag) and B disrupted g8-g9 with no
    matching truth episode (a false alarm the movement truth contradicts);
    a dwell curve for A/disrupted lets the recovery scorer grade it."""
    truth = {
        ("A", g(2)): "disrupted",
        ("A", g(3)): "disrupted",
        ("A", g(4)): "disrupted",
        ("A", g(5)): "disrupted",
    }
    truth_eps = extract_episodes(truth, {}, window_start=g(0), window_end=g(11))
    assert len(truth_eps) == 1
    assert (truth_eps[0].onset, truth_eps[0].recovery) == (g(2), g(6))

    predictions = [
        _pred("A", g(3), "disrupted"),
        _pred("A", g(4), "disrupted"),
        _pred("A", g(5), "disrupted"),
        _pred("A", g(6), "disrupted"),
        _pred("B", g(8), "disrupted"),
        _pred("B", g(9), "disrupted"),
    ]
    movement_truth = {("B", g(8)): "normal", ("B", g(9)): "normal"}

    def lookup(route: str, state: str) -> tuple[list[int], list[float] | None] | None:
        if route == "A" and state == "disrupted":
            return [300, 600, 900], None
        return None

    card = episode_scorecard(
        truth_eps,
        predictions,
        movement_truth,
        lookup,
        window_start=g(0),
        window_end=g(11),
    )

    assert set(card) == {
        "n_truth_episodes",
        "n_model_episodes",
        "onset_latency",
        "recovery",
        "false_alarms",
    }
    assert card["n_truth_episodes"] == 1
    assert card["n_model_episodes"] == 2
    assert card["onset_latency"]["n_detected"] == 1
    assert card["onset_latency"]["median_latency_min"] == _approx(5.0)
    assert card["false_alarms"]["n_model_episodes"] == 2
    assert card["false_alarms"]["n_false_alarm"] == 1
    assert card["false_alarms"]["movement_contradicted"] == 1
    assert card["recovery"]["n_scored"] == 1
