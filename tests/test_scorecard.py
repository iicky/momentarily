"""Synthetic-fixture tests for the event-based eval scorecard (training/scorecard.py).

Builds truth episodes via training.episodes.extract_episodes and model episodes
via training.scorecard.model_episodes on the same 5-min grid convention as
tests/test_episodes.py (WS grid origin, g(k) the k-th tick). No MTA key, no R2,
no network — everything here is a hand-built fixture.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import pytest

from training.episodes import Episode, extract_episodes
from training.eval import (
    MOVEMENT_ARM_LABEL,
    SHADOW_ARM_LABEL,
    TICK_SECONDS,
    PredictionRecord,
)
from training.scorecard import (
    cause_dwell_lookup,
    dwell_lookup_from_params,
    episode_recovery,
    episode_scorecard,
    false_alarms,
    model_episodes,
    movement_dwell_lookup_from_params,
    onset_latency,
    published_condition_coverage,
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


def test_onset_latency_credits_a_model_episode_that_led_the_alert() -> None:
    """A model episode that closed before the alert landed is a lead, not a miss.

    Movement calls the stall first and the MTA posts minutes later; requiring
    bare overlap scored that as a miss AND a false alarm. Within the lead
    tolerance it counts as a detection with negative latency.
    """
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

    assert result["n_detected"] == 1
    assert result["n_missed"] == 0
    # Model onset g1, truth onset g5 -> 20 minutes of lead on a 5-min grid.
    assert result["median_latency_min"] == -20.0
    assert result["n_detected_leading"] == 1
    assert result["median_lead_min"] == 20.0


def test_onset_latency_does_not_credit_a_lead_beyond_the_tolerance() -> None:
    """The lead window is bounded — an unrelated earlier call is still a miss."""
    truth_eps = extract_episodes(
        {("A", g(20)): "disrupted", ("A", g(21)): "disrupted"},
        {},
        window_start=g(0),
        window_end=g(30),
    )
    model_eps = model_episodes(
        [_pred("A", g(1), "disrupted"), _pred("A", g(2), "disrupted")],
        window_start=g(0),
        window_end=g(30),
    )
    # Model recovers at g3; truth onsets at g20 — 85 minutes later.
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

    def lookup(
        route: str, state: str, _cause: str
    ) -> tuple[list[int], list[float] | None, tuple[float, float] | None] | None:
        if route == "S" and state == "disrupted":
            return [300, 600, 900], None, None
        return None

    result = episode_recovery(truth_eps, lookup, baseline_durations_min=None)

    assert result["n_censored_excluded"] == 2
    assert result["n_no_curve"] == 1
    assert result["n_scored"] == 1
    assert result["report"]["n"] == 1


def test_episode_recovery_uses_cause_curve_then_falls_back_to_state_curve() -> None:
    """episode_recovery looks up the dwell curve by the episode's own cause
    first; when the lookup has no cell for that cause, it falls back to the
    state-level curve. Two 2-point curves are shaped so the conditional-exit
    probability at the same realized duration is unmistakably different (0.5
    vs 1.0), so the resulting per-episode PIT proves which curve fed the
    score — a cause/state mix-up would collapse both episodes into the same
    PIT bin."""
    cause_ep = Episode(
        route="R",
        onset=WS,
        recovery=WS + 600,
        peak_state="disrupted",
        cause="signal_failure",
        n_ticks=2,
        left_censored=False,
        right_censored=False,
    )
    fallback_ep = Episode(
        route="R",
        onset=WS + 10_000,
        recovery=WS + 10_000 + 600,
        peak_state="disrupted",
        cause="weather",
        n_ticks=2,
        left_censored=False,
        right_censored=False,
    )

    def lookup(
        route: str, state: str, cause: str
    ) -> tuple[list[int], list[float] | None, tuple[float, float] | None] | None:
        if route != "R" or state != "disrupted":
            return None
        if cause == "signal_failure":
            return [0, 1200], None, None  # half-life at 600s -> PIT 0.5
        # state-level fallback: outlived by 600s -> PIT 1.0
        return [0, 100], None, None

    result = episode_recovery(
        [cause_ep, fallback_ep], lookup, baseline_durations_min=None
    )

    assert result["n_scored"] == 2
    assert result["n_no_curve"] == 0
    assert result["report"]["pit"][5] == 1
    assert result["report"]["pit"][9] == 1
    assert result["report"]["mean_pit"] == _approx(0.75)


# --- dwell_lookup_from_params --------------------------------------------------------


def test_dwell_lookup_from_params_reads_curve_and_tail_or_none() -> None:
    """A (route, state, cause) cell in dwell_quantiles_by_cause resolves first;
    a cause absent from that cause-conditioned table falls back to the
    (route, state) aggregate in dwell_quantiles; a cell with no usable curve,
    a missing state, or a missing route all read as None."""
    params: dict[str, Any] = {
        "routes": {
            "A": {
                "dwell_quantiles_by_cause": {
                    "disrupted": {
                        "signal_failure": {
                            "curve_sec": [100, 200, 300],
                            "tail_ll": [1.2, 250.0],
                        },
                    },
                },
                "dwell_quantiles": {
                    "disrupted": {
                        "curve_sec": [300, 600, 900],
                        "tail_ll": [1.5, 400.0],
                    },
                    "suspended": {"curve_sec": [300]},  # too short: no curve
                    "unknown": {},  # empty cell: no curve
                },
            }
        }
    }
    lookup = dwell_lookup_from_params(params)

    assert lookup("A", "disrupted", "signal_failure") == (
        [100, 200, 300],
        [1.2, 250.0],
        None,
    )
    assert lookup("A", "disrupted", "weather") == ([300, 600, 900], [1.5, 400.0], None)
    assert lookup("A", "suspended", "weather") is None
    assert lookup("A", "unknown", "weather") is None
    assert lookup("A", "missing_state", "weather") is None
    assert lookup("missing_route", "disrupted", "weather") is None


# --- cause_dwell_lookup: train-derived cause -> state -> pooled fallback -----------


def test_cause_dwell_lookup_fallback_chain_cause_then_state_then_pooled() -> None:
    """cause_dwell_lookup checks train-derived cells in priority order: an
    exact (route, state, cause) cell wins over the (route, state) cell for
    the same route/state, which wins over the state-pooled cell, which wins
    over nothing at all. Four distinct 2-point curves make the fallback
    level that actually resolved unmistakable."""
    by_cause: dict[str, Any] = {
        "R": {"disrupted": {"signal_failure": {"curve_sec": [1, 2]}}},
    }
    by_state: dict[str, Any] = {
        "R": {"disrupted": {"curve_sec": [3, 4]}},
    }
    pooled: dict[str, Any] = {
        "disrupted": {"curve_sec": [5, 6]},
        "suspended": {"curve_sec": [7, 8]},
    }
    lookup = cause_dwell_lookup(by_cause, by_state, pooled)

    assert lookup("R", "disrupted", "signal_failure") == ([1, 2], None, None)
    assert lookup("R", "disrupted", "weather") == ([3, 4], None, None)
    assert lookup("R", "suspended", "signal_failure") == ([7, 8], None, None)
    assert lookup("Q", "totally_missing", "signal_failure") is None


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

    def lookup(
        route: str, state: str, _cause: str
    ) -> tuple[list[int], list[float] | None, tuple[float, float] | None] | None:
        if route == "A" and state == "disrupted":
            return [300, 600, 900], None, None
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
        "n_standing_excluded",
        "n_model_episodes_in_standing",
        "graded_arm",
        "published_coverage",
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


def test_episode_scorecard_forwards_a_causal_baseline_to_the_shadow_recovery_arm() -> (
    None
):
    """The review supplies a pre-window duration climatology; the scorecard must
    thread it into the shadow `recovery` arm so its report carries a real
    `causal_skill`. The default (None) leaves that column null rather than
    letting the hindsight oracle number pass for a climatology comparison."""
    # Two distinct-duration incidents (A 20min, B 30min) so the oracle baseline
    # is non-degenerate and its skill is finite (a lone self-graded duration
    # gives CRPS 0 -> nan skill, which would compare unequal to itself).
    truth = {("A", g(k)): "disrupted" for k in range(2, 6)}
    truth |= {("B", g(k)): "disrupted" for k in range(2, 8)}
    truth_eps = extract_episodes(truth, {}, window_start=g(0), window_end=g(11))
    assert len(truth_eps) == 2

    def lookup(
        route: str, state: str, _cause: str
    ) -> tuple[list[int], list[float] | None, tuple[float, float] | None] | None:
        if state == "disrupted":
            return [300, 600, 900], None, None
        return None

    bare = episode_scorecard(
        truth_eps, [], {}, lookup, window_start=g(0), window_end=g(11)
    )
    assert bare["recovery"]["n_scored"] == 2
    assert bare["recovery"]["report"]["causal_skill"] is None
    assert bare["recovery"]["report"]["causal_baseline_crps"] is None
    assert math.isfinite(bare["recovery"]["report"]["oracle_skill"])

    causal = episode_scorecard(
        truth_eps,
        [],
        {},
        lookup,
        window_start=g(0),
        window_end=g(11),
        baseline_durations_min=[10.0, 20.0, 30.0],
    )
    assert causal["recovery"]["report"]["causal_skill"] is not None
    assert causal["recovery"]["report"]["causal_baseline_crps"] is not None
    # The oracle column is unaffected — same graded population, same hindsight CDF.
    assert (
        causal["recovery"]["report"]["oracle_skill"]
        == bare["recovery"]["report"]["oracle_skill"]
    )


# --- standing advisories are held out of grading, not silently dropped ----------


def _standing_truth(route: str, n_ticks: int) -> dict[tuple[str, int], str]:
    return {(route, g(k)): "disrupted" for k in range(n_ticks)}


def test_a_standing_advisory_is_excluded_from_onset_and_recovery() -> None:
    """A severe-tier alert held past a day measures the alert feed, not the model.

    It stays in n_truth_episodes and is counted in n_standing_excluded, but the
    model is not scored as having missed it.
    """
    # 24h at 5-min ticks = 288 ticks; go one past the threshold.
    ticks = (24 * 3600) // TICK_SECONDS + 1
    truth_eps = extract_episodes(
        _standing_truth("A", ticks), {}, window_start=g(0), window_end=g(ticks + 10)
    )
    assert len(truth_eps) == 1
    assert truth_eps[0].standing

    card = episode_scorecard(
        truth_eps,
        [],
        {},
        lambda _r, _s, _c: None,
        window_start=g(0),
        window_end=g(ticks + 10),
    )
    assert card["n_truth_episodes"] == 1
    assert card["n_standing_excluded"] == 1
    # Not graded as a miss — the denominator is empty, not 1.
    assert card["onset_latency"]["n_episodes"] == 0
    assert card["recovery"]["n_scored"] == 0


def test_an_acute_episode_is_still_graded_alongside_a_standing_one() -> None:
    ticks = (24 * 3600) // TICK_SECONDS + 1
    truth = _standing_truth("A", ticks)
    truth[("B", g(3))] = "disrupted"
    truth[("B", g(4))] = "disrupted"
    truth_eps = extract_episodes(truth, {}, window_start=g(0), window_end=g(ticks + 10))
    card = episode_scorecard(
        truth_eps,
        [_pred("B", g(3), "disrupted"), _pred("B", g(4), "disrupted")],
        {},
        lambda _r, _s, _c: None,
        window_start=g(0),
        window_end=g(ticks + 10),
    )
    assert card["n_standing_excluded"] == 1
    assert card["onset_latency"]["n_episodes"] == 1
    assert card["onset_latency"]["n_detected"] == 1


def test_a_model_call_inside_a_standing_advisory_is_not_a_false_alarm() -> None:
    """The alert feed says that route is impaired, so calling it is not an
    over-call — it just isn't gradeable as an acute detection either."""
    ticks = (24 * 3600) // TICK_SECONDS + 1
    truth_eps = extract_episodes(
        _standing_truth("A", ticks), {}, window_start=g(0), window_end=g(ticks + 10)
    )
    card = episode_scorecard(
        truth_eps,
        [_pred("A", g(5), "disrupted"), _pred("A", g(6), "disrupted")],
        {("A", g(5)): "normal", ("A", g(6)): "normal"},
        lambda _r, _s, _c: None,
        window_start=g(0),
        window_end=g(ticks + 10),
    )
    assert card["n_model_episodes"] == 1
    assert card["n_model_episodes_in_standing"] == 1
    assert card["false_alarms"]["n_false_alarm"] == 0


def test_an_episode_just_under_the_threshold_is_still_graded() -> None:
    ticks = (24 * 3600) // TICK_SECONDS - 2
    truth_eps = extract_episodes(
        _standing_truth("A", ticks), {}, window_start=g(0), window_end=g(ticks + 10)
    )
    assert not truth_eps[0].standing
    card = episode_scorecard(
        truth_eps,
        [],
        {},
        lambda _r, _s, _c: None,
        window_start=g(0),
        window_end=g(ticks + 10),
    )
    assert card["n_standing_excluded"] == 0
    assert card["onset_latency"]["n_episodes"] == 1


# --- the graded arm is the published one, not the alert shadow -------------------


def _pred_arms(
    route: str, ts: int, condition: str, published: str | None
) -> PredictionRecord:
    p = _pred(route, ts, condition)
    return replace(p, published_condition=published)


def test_model_episodes_grade_the_published_arm_not_the_shadow() -> None:
    """The shadow `condition` and the movement-primary `published_condition` are
    different arms; consumers read the published one, so that is what is graded."""
    preds = [
        _pred_arms("A", g(1), "disrupted", "normal"),
        _pred_arms("A", g(2), "disrupted", "normal"),
        _pred_arms("B", g(1), "normal", "disrupted"),
        _pred_arms("B", g(2), "normal", "disrupted"),
    ]
    eps = model_episodes(preds, window_start=g(0), window_end=g(6))

    assert [e.route for e in eps] == ["B"]


def test_model_episodes_fall_back_to_condition_for_pre_published_rows() -> None:
    """Rows written before published_condition existed carry None; those still
    grade off `condition` rather than vanishing."""
    preds = [
        _pred_arms("A", g(1), "disrupted", None),
        _pred_arms("A", g(2), "disrupted", None),
    ]
    eps = model_episodes(preds, window_start=g(0), window_end=g(6))

    assert [e.route for e in eps] == ["A"]


def test_unknown_published_condition_closes_a_run() -> None:
    """`unknown` is no reading, not a disruption — it ends a run like normal."""
    preds = [
        _pred_arms("A", g(1), "normal", "disrupted"),
        _pred_arms("A", g(2), "normal", "unknown"),
        _pred_arms("A", g(3), "normal", "disrupted"),
    ]
    eps = model_episodes(preds, window_start=g(0), window_end=g(6))

    assert len(eps) == 2


def test_published_coverage_reports_the_unknown_share() -> None:
    preds = [
        _pred_arms("A", g(1), "normal", "unknown"),
        _pred_arms("A", g(2), "normal", "normal"),
        _pred_arms("A", g(3), "normal", "disrupted"),
        _pred_arms("A", g(4), "normal", "normal"),
    ]
    cov = published_condition_coverage(preds)

    assert cov["n_ticks"] == 4
    assert cov["unknown_share"] == 0.25
    assert cov["gradeable_share"] == 0.75
    assert cov["by_condition"]["disrupted"] == 1


# --- movement_dwell_lookup_from_params: C4, the movement-regime dwell block ------


def test_movement_dwell_lookup_from_params_reads_curve_and_tail_ignoring_cause() -> (
    None
):
    """(route, state) resolves against params['dwell_movement'] (contract C2);
    `cause` is accepted only for DwellLookup shape compatibility and has no
    effect -- the movement clock has no cause dimension. A too-short curve, a
    missing state, and a missing route all read as None."""
    params: dict[str, Any] = {
        "dwell_movement": {
            "A": {
                "disrupted": {"curve_sec": [100, 200, 300], "tail_ll": [1.2, 250.0]},
                "suspended": {"curve_sec": [300]},  # too short: no curve
            }
        }
    }
    lookup = movement_dwell_lookup_from_params(params)

    assert lookup("A", "disrupted", "signal_failure") == (
        [100, 200, 300],
        [1.2, 250.0],
        None,
    )
    assert lookup("A", "disrupted", "weather") == ([100, 200, 300], [1.2, 250.0], None)
    assert lookup("A", "suspended", "weather") is None
    assert lookup("A", "missing_state", "weather") is None
    assert lookup("missing_route", "disrupted", "weather") is None


def test_movement_dwell_lookup_from_params_absent_block_lands_in_n_no_curve() -> None:
    """dwell_movement not yet present in params (contract C2 has no producer in
    live params yet) resolves every lookup to None; episode_recovery counts
    the episode in n_no_curve rather than crashing."""
    lookup = movement_dwell_lookup_from_params({})
    assert lookup("A", "disrupted", "x") is None

    truth = {("A", g(2)): "disrupted", ("A", g(3)): "disrupted"}
    truth_eps = extract_episodes(truth, {}, window_start=g(0), window_end=g(6))
    assert len(truth_eps) == 1

    result = episode_recovery(truth_eps, lookup, baseline_durations_min=None)

    assert result["n_no_curve"] == 1
    assert result["n_scored"] == 0
    assert result["report"]["n"] == 0


# --- episode_recovery: the graded_arm label -----------------------------------------


def test_episode_recovery_defaults_to_the_alert_shadow_label() -> None:
    """Existing callers (training.backtest.grade_recovery_timing) that don't
    pass graded_arm still get a correctly labelled payload: SHADOW_ARM_LABEL,
    the alert-condition dwell population they've always graded against."""
    truth = {("S", g(5)): "disrupted", ("S", g(6)): "disrupted"}
    truth_eps = extract_episodes(truth, {}, window_start=g(0), window_end=g(9))

    def lookup(
        route: str, state: str, _cause: str
    ) -> tuple[list[int], list[float] | None, tuple[float, float] | None] | None:
        return (
            ([300, 600, 900], None, None)
            if route == "S" and state == "disrupted"
            else None
        )

    result = episode_recovery(truth_eps, lookup, baseline_durations_min=None)

    assert result["graded_arm"] == SHADOW_ARM_LABEL


def test_episode_recovery_grades_a_movement_episode_against_a_movement_curve() -> None:
    """A short movement episode (minutes, not the alert shadow's hours) scored
    against a params['dwell_movement'] curve produces a finite CRPS/PIT, tagged
    MOVEMENT_ARM_LABEL so it can never be misread as an alert-shadow number."""
    truth = {("M", g(2)): "disrupted", ("M", g(3)): "disrupted"}  # 10-min episode
    truth_eps = extract_episodes(truth, {}, window_start=g(0), window_end=g(6))
    assert truth_eps[0].duration_sec == 2 * TICK_SECONDS

    params: dict[str, Any] = {
        "dwell_movement": {"M": {"disrupted": {"curve_sec": [300, 600, 900]}}}
    }
    lookup = movement_dwell_lookup_from_params(params)

    result = episode_recovery(
        truth_eps, lookup, graded_arm=MOVEMENT_ARM_LABEL, baseline_durations_min=None
    )

    assert result["graded_arm"] == MOVEMENT_ARM_LABEL
    assert result["n_scored"] == 1
    assert result["n_no_curve"] == 0
    assert math.isfinite(result["report"]["mean_crps"])
    assert math.isfinite(result["report"]["mean_pit"])


# --- the CRPS baselines: the graded population's own, and the caller's train window ---


def test_movement_recovery_baseline_is_drawn_from_movement_durations_only() -> None:
    """recovery_dist_report's ORACLE baseline (`oracle_baseline_crps`) is built
    from THIS call's samples alone. Two movement episodes (10 and 20 minutes)
    give an exact, hand-computed empirical CDF at t=10min: 1 of 2 durations
    <= 10 -> 0.5. Had an alert-shadow episode (hours, not minutes) leaked into
    this baseline the value could not land on this exact fraction."""
    truth = {
        ("M1", g(2)): "disrupted",
        ("M1", g(3)): "disrupted",  # 10-min episode
        ("M2", g(10)): "disrupted",
        ("M2", g(11)): "disrupted",
        ("M2", g(12)): "disrupted",
        ("M2", g(13)): "disrupted",  # 20-min episode
    }
    truth_eps = extract_episodes(truth, {}, window_start=g(0), window_end=g(20))
    assert sorted(e.duration_sec for e in truth_eps) == [
        2 * TICK_SECONDS,
        4 * TICK_SECONDS,
    ]

    params: dict[str, Any] = {
        "dwell_movement": {
            "M1": {"disrupted": {"curve_sec": [300, 600, 900]}},
            "M2": {"disrupted": {"curve_sec": [300, 600, 900]}},
        }
    }
    lookup = movement_dwell_lookup_from_params(params)

    result = episode_recovery(
        truth_eps, lookup, graded_arm=MOVEMENT_ARM_LABEL, baseline_durations_min=None
    )

    assert result["n_scored"] == 2
    grid = result["report"]["grid"]
    assert grid[2] == 10  # GRID_STEP=5 -> grid[2] is the 10-minute mark
    assert result["report"]["empirical_curve"][2] == _approx(0.5)


def test_episode_recovery_publishes_a_causal_skill_only_when_given_a_train_window() -> (
    None
):
    """The published scorecard must never let the hindsight number pass for a
    climatology comparison. Without training durations `causal_skill` is null —
    a visible omission — and with them it is a real number scored against a
    baseline built from those durations alone."""
    truth = {
        ("M1", g(2)): "disrupted",
        ("M1", g(3)): "disrupted",
        ("M2", g(10)): "disrupted",
        ("M2", g(11)): "disrupted",
        ("M2", g(12)): "disrupted",
        ("M2", g(13)): "disrupted",
    }
    truth_eps = extract_episodes(truth, {}, window_start=g(0), window_end=g(20))
    params: dict[str, Any] = {
        "dwell_movement": {
            "M1": {"disrupted": {"curve_sec": [300, 600, 900]}},
            "M2": {"disrupted": {"curve_sec": [300, 600, 900]}},
        }
    }
    lookup = movement_dwell_lookup_from_params(params)

    bare = episode_recovery(
        truth_eps, lookup, graded_arm=MOVEMENT_ARM_LABEL, baseline_durations_min=None
    )
    assert bare["report"]["causal_skill"] is None
    assert bare["report"]["causal_baseline_crps"] is None
    assert math.isfinite(bare["report"]["oracle_skill"])

    causal = episode_recovery(
        truth_eps,
        lookup,
        graded_arm=MOVEMENT_ARM_LABEL,
        baseline_durations_min=[45.0, 60.0],
    )
    # Same graded population, so the model's own score cannot move; only the
    # rival forecast changed.
    assert causal["report"]["mean_crps"] == bare["report"]["mean_crps"]
    assert causal["report"]["oracle_skill"] == bare["report"]["oracle_skill"]
    assert math.isfinite(causal["report"]["causal_skill"])


# --- episode_scorecard: the movement arm is additive, never replaces the shadow -----


def test_episode_scorecard_omits_recovery_movement_without_a_lookup() -> None:
    """Back-compat: review.py's existing call (no movement_dwell_lookup, since
    contract C2 has no producer in live params yet) gets exactly the pre-
    movement-arm payload shape -- no recovery_movement key."""
    truth = {
        ("A", g(2)): "disrupted",
        ("A", g(3)): "disrupted",
        ("A", g(4)): "disrupted",
        ("A", g(5)): "disrupted",
    }
    truth_eps = extract_episodes(truth, {}, window_start=g(0), window_end=g(11))
    predictions = [_pred("A", g(3), "disrupted")]

    def lookup(
        route: str, state: str, _cause: str
    ) -> tuple[list[int], list[float] | None, tuple[float, float] | None] | None:
        return (
            ([300, 600, 900], None, None)
            if route == "A" and state == "disrupted"
            else None
        )

    card = episode_scorecard(
        truth_eps, predictions, {}, lookup, window_start=g(0), window_end=g(11)
    )

    assert "recovery_movement" not in card
    assert card["recovery"]["graded_arm"] == SHADOW_ARM_LABEL


def test_episode_scorecard_grades_both_arms_when_given_a_movement_lookup() -> None:
    """movement_dwell_lookup, when passed, re-segments movement_truth into its
    own episodes and grades a second, explicitly-labelled recovery_movement
    block alongside (not instead of) the alert-shadow recovery block."""
    truth = {  # alert truth: one episode on A
        ("A", g(2)): "disrupted",
        ("A", g(3)): "disrupted",
        ("A", g(4)): "disrupted",
        ("A", g(5)): "disrupted",
    }
    truth_eps = extract_episodes(truth, {}, window_start=g(0), window_end=g(11))
    predictions = [_pred("A", g(3), "disrupted")]
    movement_truth = {  # movement truth: a shorter, distinct episode on Z
        ("Z", g(6)): "disrupted",
        ("Z", g(7)): "disrupted",
    }

    def alert_lookup(
        route: str, state: str, _cause: str
    ) -> tuple[list[int], list[float] | None, tuple[float, float] | None] | None:
        return (
            ([300, 600, 900], None, None)
            if route == "A" and state == "disrupted"
            else None
        )

    movement_params: dict[str, Any] = {
        "dwell_movement": {"Z": {"disrupted": {"curve_sec": [300, 600, 900]}}}
    }
    movement_lookup = movement_dwell_lookup_from_params(movement_params)

    card = episode_scorecard(
        truth_eps,
        predictions,
        movement_truth,
        alert_lookup,
        window_start=g(0),
        window_end=g(11),
        movement_dwell_lookup=movement_lookup,
    )

    assert card["recovery"]["graded_arm"] == SHADOW_ARM_LABEL
    assert card["recovery"]["n_scored"] == 1
    assert card["recovery_movement"]["graded_arm"] == MOVEMENT_ARM_LABEL
    assert card["recovery_movement"]["n_scored"] == 1
    assert math.isfinite(card["recovery_movement"]["report"]["mean_crps"])
