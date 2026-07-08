"""Synthetic-fixture tests for incident-episode extraction (training/episodes.py).

Builds truth/types dicts directly on the 5-min grid — no MTA key, no R2, no
network. `WS` is the grid origin (already a multiple of TICK_SECONDS) and
`g(k)` is the k-th tick from there, matching the ws / g(k) convention used to
describe the module's contract.
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

from training.episodes import (
    Episode,
    disruptive_types_by_key,
    episode_as_dict,
    episodes_summary,
    extract_episodes,
)
from training.eval import TICK_SECONDS, snap_tick

WS = 1_700_000_100  # grid-aligned: snap_tick(WS) == WS


def g(k: int) -> int:
    """The k-th 5-min grid tick starting at WS."""
    return WS + TICK_SECONDS * k


def by_route(episodes: list[Episode], route: str) -> list[Episode]:
    return [ep for ep in episodes if ep.route == route]


# --- Segmentation: contiguity, merging, censoring -------------------------------


def test_adjacent_runs_split_by_one_normal_tick_are_two_episodes():
    """A single normal tick between two not-normal runs splits them — they are
    NOT merged into one episode."""
    truth = {
        ("Z", g(1)): "disrupted",
        ("Z", g(2)): "disrupted",
        # g(3) absent -> normal, splits the runs
        ("Z", g(4)): "disrupted",
        ("Z", g(5)): "disrupted",
    }
    eps = extract_episodes(truth, {}, window_start=g(0), window_end=g(7))

    assert len(eps) == 2
    first, second = eps
    assert first.onset == g(1)
    assert first.recovery == g(3)
    assert first.n_ticks == 2
    assert second.onset == g(4)
    assert second.recovery == g(6)
    assert second.n_ticks == 2
    assert not first.right_censored
    assert not second.left_censored


def test_escalation_within_a_run_stays_one_episode():
    """A contiguous disrupted -> suspended run is ONE episode, peak_state
    reflects the escalation."""
    truth = {
        ("Y", g(1)): "disrupted",
        ("Y", g(2)): "suspended",
        ("Y", g(3)): "disrupted",
    }
    eps = extract_episodes(truth, {}, window_start=g(0), window_end=g(5))

    assert len(eps) == 1
    ep = eps[0]
    assert ep.onset == g(1)
    assert ep.recovery == g(4)
    assert ep.peak_state == "suspended"
    assert ep.n_ticks == 3


def test_right_censored_run_recovery_is_last_tick_plus_one_grid_step():
    """A run still active at the last grid tick is right-censored; recovery is
    one tick past the window end (a lower-bound duration), not an observed
    recovery."""
    truth = {("R", g(2)): "disrupted", ("R", g(3)): "disrupted"}
    window_start, window_end = g(0), g(3)
    eps = extract_episodes(truth, {}, window_start=window_start, window_end=window_end)

    assert len(eps) == 1
    ep = eps[0]
    assert ep.right_censored is True
    assert ep.left_censored is False
    assert ep.recovery == snap_tick(window_end) + TICK_SECONDS == g(4)


def test_left_censored_run_is_active_at_the_first_grid_tick():
    """A run already active at the first grid tick is left-censored — the true
    onset precedes the window."""
    truth = {("L", g(0)): "disrupted", ("L", g(1)): "disrupted"}
    eps = extract_episodes(truth, {}, window_start=g(0), window_end=g(5))

    assert len(eps) == 1
    ep = eps[0]
    assert ep.left_censored is True
    assert ep.right_censored is False
    assert ep.onset == g(0)
    assert ep.recovery == g(2)


def test_types_without_a_matching_truth_entry_yield_no_episode():
    """`types` alone never manufactures an episode — a tick only opens a run
    when `truth` marks it not-normal; absent truth reads normal regardless of
    what alert types are on file for that tick."""
    types = {("N", g(1)): ("Severe Delays",), ("N", g(2)): ("Delays",)}
    assert extract_episodes({}, types, window_start=g(0), window_end=g(3)) == []
    assert extract_episodes({}, {}, window_start=g(0), window_end=g(3)) == []


# --- Cause attribution -----------------------------------------------------------


def test_cause_is_majority_vote_not_peak_severity():
    """Key invariant: a delays-dominated run (3 Severe Delays ticks) with ONE
    suspension tick has peak_state='suspended' but cause stays 'delays' — the
    lone higher-severity suspension tick does not relabel the cause.

    Mutation check: if cause were instead taken from peak severity (the
    category with the highest per-tick severity tier, ignoring vote counts),
    the suspension tick (tier 3) would outrank the delays ticks (tier 2) and
    this assertion (cause == 'delays') would fail.
    """
    truth = {
        ("M", g(1)): "disrupted",
        ("M", g(2)): "disrupted",
        ("M", g(3)): "disrupted",
        ("M", g(4)): "suspended",
    }
    types = {
        ("M", g(1)): ("Severe Delays",),
        ("M", g(2)): ("Severe Delays",),
        ("M", g(3)): ("Severe Delays",),
        ("M", g(4)): ("Suspended",),
    }
    eps = extract_episodes(truth, types, window_start=g(0), window_end=g(6))

    assert len(eps) == 1
    ep = eps[0]
    assert ep.peak_state == "suspended"
    assert ep.cause == "delays"
    assert ep.n_ticks == 4


def test_cause_of_pure_suspension_run_via_explicit_alert_type():
    """A run whose ticks all carry a 'Suspended' alert votes to
    'service_suspension', and peak_state is 'suspended'."""
    truth = {("T", g(2)): "suspended", ("T", g(3)): "suspended"}
    types = {("T", g(2)): ("Suspended",), ("T", g(3)): ("Suspended",)}
    eps = extract_episodes(truth, types, window_start=g(0), window_end=g(5))

    assert len(eps) == 1
    assert eps[0].cause == "service_suspension"
    assert eps[0].peak_state == "suspended"


def test_cause_of_pure_suspension_run_falls_back_without_recorded_types():
    """With no disruptive alert types on file at all (no votes cast), the
    fallback still resolves a suspended run to 'service_suspension'."""
    truth = {("S", g(2)): "suspended", ("S", g(3)): "suspended"}
    eps = extract_episodes(truth, {}, window_start=g(0), window_end=g(5))

    assert len(eps) == 1
    assert eps[0].cause == "service_suspension"
    assert eps[0].peak_state == "suspended"


# --- Combined oracle scenario ------------------------------------------------------


def _oracle_scenario() -> tuple[
    dict[tuple[str, int], str], dict[tuple[str, int], tuple[str, ...]], int, int
]:
    """Three-route scenario: A escalates across two separate episodes, B is
    left-censored, C is right-censored. Window is g(0)..g(11)."""
    truth = {
        ("A", g(1)): "disrupted",
        ("A", g(2)): "disrupted",
        # g(3) normal -> splits A's two runs
        ("A", g(4)): "disrupted",
        ("A", g(5)): "disrupted",
        ("A", g(6)): "disrupted",
        ("A", g(7)): "suspended",
        ("B", g(0)): "disrupted",
        ("B", g(1)): "disrupted",
        ("C", g(10)): "suspended",
        ("C", g(11)): "suspended",
    }
    types = {
        ("A", g(1)): ("Severe Delays",),
        ("A", g(2)): ("Severe Delays",),
        ("A", g(4)): ("Severe Delays",),
        ("A", g(5)): ("Severe Delays",),
        ("A", g(6)): ("Severe Delays",),
        ("A", g(7)): ("Suspended", "Severe Delays"),
    }
    return truth, types, g(0), g(11)


def test_oracle_scenario_matches_the_verified_episode_table():
    """End-to-end cross-check against a hand-verified three-route scenario
    combining splitting, escalation, and both censoring directions."""
    truth, types, window_start, window_end = _oracle_scenario()
    eps = extract_episodes(
        truth, types, window_start=window_start, window_end=window_end
    )
    assert len(eps) == 4

    a1, a2 = by_route(eps, "A")
    assert (a1.onset, a1.recovery, a1.n_ticks) == (g(1), g(3), 2)
    assert a1.peak_state == "disrupted"
    assert a1.cause == "delays"
    assert not a1.left_censored
    assert not a1.right_censored

    assert (a2.onset, a2.recovery, a2.n_ticks) == (g(4), g(8), 4)
    assert a2.peak_state == "suspended"
    assert a2.cause == "delays"
    assert not a2.left_censored
    assert not a2.right_censored

    (b,) = by_route(eps, "B")
    assert b.left_censored is True
    assert (b.onset, b.recovery, b.n_ticks) == (g(0), g(2), 2)

    (c,) = by_route(eps, "C")
    assert c.right_censored is True
    assert c.recovery == g(12) == snap_tick(window_end) + TICK_SECONDS
    assert c.cause == "service_suspension"
    assert c.peak_state == "suspended"


# --- episodes_summary --------------------------------------------------------------


def test_episodes_summary_agrees_with_the_extracted_list():
    """n / n_left_censored / n_right_censored / by_cause / by_peak_state and the
    table are all derived from — and stay consistent with — the episode list
    extract_episodes actually returned."""
    truth = {
        ("P", g(1)): "disrupted",
        ("P", g(2)): "disrupted",
        ("Q", g(0)): "disrupted",  # left-censored
        ("Q", g(1)): "disrupted",
        ("R", g(8)): "suspended",  # right-censored (window ends at g(9))
        ("R", g(9)): "suspended",
    }
    types = {("P", g(1)): ("Severe Delays",), ("P", g(2)): ("Severe Delays",)}
    eps = extract_episodes(truth, types, window_start=g(0), window_end=g(9))
    summary = episodes_summary(eps)

    assert summary["n"] == len(eps) == 3
    assert summary["n_left_censored"] == sum(ep.left_censored for ep in eps) == 1
    assert summary["n_right_censored"] == sum(ep.right_censored for ep in eps) == 1
    assert summary["by_cause"] == dict(Counter(ep.cause for ep in eps))
    assert summary["by_peak_state"] == dict(Counter(ep.peak_state for ep in eps))
    assert sum(summary["by_cause"].values()) == summary["n"]
    assert sum(summary["by_peak_state"].values()) == summary["n"]
    assert len(summary["table"]) == summary["n"]
    assert summary["table"] == [episode_as_dict(ep) for ep in eps]


# --- disruptive_types_by_key --------------------------------------------------------


def test_disruptive_types_by_key_maps_route_and_tick_to_types():
    """Builds a (route, tick) -> disruptive_types map from a flat observation
    list, keyed by the observation's own route_id/tick, not position."""
    obs = [
        SimpleNamespace(route_id="A", tick=100, disruptive_types=("Severe Delays",)),
        SimpleNamespace(route_id="A", tick=400, disruptive_types=()),
        SimpleNamespace(route_id="B", tick=100, disruptive_types=("Suspended",)),
    ]
    result = disruptive_types_by_key(obs)

    assert result == {
        ("A", 100): ("Severe Delays",),
        ("A", 400): (),
        ("B", 100): ("Suspended",),
    }
    assert ("C", 100) not in result


def test_disruptive_types_by_key_last_observation_wins_on_duplicate_key():
    """Two observations sharing (route_id, tick) collapse to one entry — the
    later observation in the list overwrites the earlier one."""
    obs = [
        SimpleNamespace(route_id="A", tick=100, disruptive_types=("Delays",)),
        SimpleNamespace(route_id="A", tick=100, disruptive_types=("Suspended",)),
    ]
    result = disruptive_types_by_key(obs)

    assert result == {("A", 100): ("Suspended",)}
