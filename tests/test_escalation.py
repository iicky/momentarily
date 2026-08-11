"""Alert corroboration for movement-disrupted episodes (training/escalation.py).

Movement calls a route disrupted from vehicle positions; the alert feed comes
from MTA dispatch. Corroboration is directional: an alert near a movement
episode's onset IS evidence the disruption was real, but alert absence is
NOT evidence against it (the MTA may simply never post one). These tests
build small Episode fixtures by hand on a 300s tick grid and check window
search, nearest-alert selection, boundaries, and the confirmation_rate /
lead_time / n_unconfirmed summary — and that no precision / false-positive /
specificity / "accuracy" number ever appears in the output.
"""

from __future__ import annotations

from typing import cast

from training.episodes import Episode
from training.escalation import (
    confirmation_summary,
    corroborate_episodes,
)

TICK = 300
T0 = 1_700_000_100  # tick-aligned


def _episode(route: str, onset: int, *, n_ticks: int = 1) -> Episode:
    return Episode(
        route=route,
        onset=onset,
        recovery=onset + n_ticks * TICK,
        peak_state="disrupted",
        cause="other",
        n_ticks=n_ticks,
        left_censored=False,
        right_censored=False,
    )


def _alert_at(route: str, offset: int, start: int = T0) -> tuple[str, int]:
    return (route, start + offset * TICK)


# --- Window search & nearest-alert selection ---------------------------------


def test_forward_alert_confirms_with_positive_lead():
    ep = _episode("A", T0)
    corr = corroborate_episodes([ep], {_alert_at("A", 3)})
    assert corr[0].confirmed is True
    assert corr[0].lead_minutes == 15.0


def test_backward_alert_confirms_with_negative_lead():
    # Alert posted 10 minutes BEFORE movement onset: the alert led.
    ep = _episode("A", T0)
    corr = corroborate_episodes([ep], {_alert_at("A", -2)})
    assert corr[0].confirmed is True
    assert corr[0].lead_minutes == -10.0


def test_alert_at_onset_tick_is_zero_lead_confirmation():
    ep = _episode("A", T0)
    corr = corroborate_episodes([ep], {_alert_at("A", 0)})
    assert corr[0].confirmed is True
    assert corr[0].lead_minutes == 0.0


def test_nearest_alert_wins_over_farther_alert_on_either_side():
    ep = _episode("A", T0)
    alert_disrupted = {_alert_at("A", -1), _alert_at("A", 4)}  # -5min vs +20min
    corr = corroborate_episodes([ep], alert_disrupted)
    assert corr[0].lead_minutes == -5.0


def test_no_alert_in_window_is_unconfirmed_not_disconfirmed():
    ep = _episode("A", T0)
    corr = corroborate_episodes([ep], set())
    assert corr[0].confirmed is False
    assert corr[0].lead_minutes is None


def test_corroboration_is_independent_per_route():
    ep_a = _episode("A", T0)
    ep_b = _episode("B", T0)
    # Alert only on route B must not confirm route A's episode.
    corr = corroborate_episodes([ep_a, ep_b], {_alert_at("B", 2)})
    by_route = {c.episode.route: c for c in corr}
    assert by_route["A"].confirmed is False
    assert by_route["B"].confirmed is True


# --- Window boundaries ---------------------------------------------------------


def test_forward_window_boundary():
    horizon_min = 15  # 3 ticks
    ep = _episode("AT", T0)
    ep_past = _episode("PAST", T0)
    corr = corroborate_episodes(
        [ep, ep_past],
        {_alert_at("AT", 3), _alert_at("PAST", 4)},
        forward_minutes=horizon_min,
    )
    by_route = {c.episode.route: c for c in corr}
    assert by_route["AT"].confirmed is True
    assert by_route["AT"].lead_minutes == 15.0
    assert by_route["PAST"].confirmed is False


def test_backward_window_boundary():
    lookback_min = 15  # 3 ticks
    ep = _episode("IN", T0)
    ep_out = _episode("OUT", T0)
    corr = corroborate_episodes(
        [ep, ep_out],
        {_alert_at("IN", -3), _alert_at("OUT", -4)},
        backward_minutes=lookback_min,
    )
    by_route = {c.episode.route: c for c in corr}
    assert by_route["IN"].confirmed is True
    assert by_route["IN"].lead_minutes == -15.0
    assert by_route["OUT"].confirmed is False


# --- confirmation_summary: rate, lead time, unconfirmed ------------------------


def test_empty_episode_list_yields_none_rate_and_zero_counts():
    summary = confirmation_summary([])
    assert summary["confirmation_rate"] is None
    assert summary["n_episodes"] == 0
    assert summary["n_confirmed"] == 0
    assert summary["n_unconfirmed"] == 0
    assert summary["lead_time_minutes"] == {"n": 0, "median": None, "iqr": None}


def test_unconfirmed_episode_lands_in_count_not_a_rate():
    # Baseline: one confirmed episode -> confirmation_rate is 1.0, no
    # unconfirmed. Adding one unconfirmed episode must only ever show up as a
    # count (n_unconfirmed) and must leave the confirmed cohort's lead-time
    # distribution completely untouched -- it is not folded into any rate that
    # could read as evidence against the episode.
    confirmed_ep = _episode("A", T0)
    baseline = confirmation_summary(
        corroborate_episodes([confirmed_ep], {_alert_at("A", 2)})
    )
    assert baseline["confirmation_rate"] == 1.0
    assert baseline["n_unconfirmed"] == 0

    unconfirmed_ep = _episode("B", T0)
    corr = corroborate_episodes([confirmed_ep, unconfirmed_ep], {_alert_at("A", 2)})
    summary = confirmation_summary(corr)
    assert summary["n_episodes"] == 2
    assert summary["n_confirmed"] == 1
    assert summary["n_unconfirmed"] == 1
    assert summary["confirmation_rate"] == 0.5
    # The confirmed episode's lead time is unaffected by the unconfirmed one.
    assert summary["lead_time_minutes"] == baseline["lead_time_minutes"]


def test_corroboration_window_reported_explicitly():
    summary = confirmation_summary([], forward_minutes=45, backward_minutes=20)
    assert summary["corroboration_window_minutes"] == {
        "forward": 45,
        "backward": 20,
    }


def test_lead_time_median_iqr_and_n_over_confirmed_only():
    # Leads: -10, 0, 10, 20 minutes (route C is unconfirmed and must not
    # contribute to n/median/iqr).
    episodes = [
        _episode("A", T0),
        _episode("B", T0 + 100 * TICK),
        _episode("D", T0 + 200 * TICK),
        _episode("E", T0 + 300 * TICK),
        _episode("C", T0 + 400 * TICK),
    ]
    alert_disrupted = {
        _alert_at("A", -2, start=T0),  # -10 min
        _alert_at("B", 0, start=T0 + 100 * TICK),  # 0 min
        _alert_at("D", 2, start=T0 + 200 * TICK),  # +10 min
        _alert_at("E", 4, start=T0 + 300 * TICK),  # +20 min
        # C: no corroborating alert.
    }
    corr = corroborate_episodes(episodes, alert_disrupted)
    summary = confirmation_summary(corr)
    assert summary["n_episodes"] == 5
    assert summary["n_confirmed"] == 4
    assert summary["n_unconfirmed"] == 1
    assert summary["confirmation_rate"] == 4 / 5
    lt = summary["lead_time_minutes"]
    assert lt["n"] == 4
    assert lt["median"] == 5.0
    assert lt["iqr"] == [-2.5, 12.5]


def test_no_invalid_metric_names_anywhere_in_output():
    ep = _episode("A", T0)
    corr = corroborate_episodes([ep], set())
    summary = confirmation_summary(corr)
    forbidden = (
        "precision",
        "false_positive",
        "false positive",
        "specificity",
        "accuracy",
    )

    def _walk(value: object) -> None:
        if isinstance(value, dict):
            for key, sub in cast("dict[object, object]", value).items():
                assert not any(f in str(key).lower() for f in forbidden), key
                _walk(sub)
        elif isinstance(value, str):
            assert not any(f in value.lower() for f in forbidden), value

    _walk(summary)
