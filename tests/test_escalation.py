"""Escalation-arm leading-indicator validation (training/escalation.py).

Movement calls a route disrupted while the MTA alert feed reads normal; because
the disrupted arm is itself derived from vehicle positions, the only honest
adjudication is temporal — score the escalation against later alerts. These
tests build small (route, tick) movement/alert grids by hand on a 300s tick
grid and check onset detection, gap tolerance, post-alert tails, forward
confirmation, persistence/evaporation, and summary aggregation.
"""

from __future__ import annotations

from collections.abc import Sequence

from training.escalation import (
    SUSTAINED_TICKS,
    escalation_events,
    escalation_summary,
)

TICK = 300
T0 = 1_700_000_100  # tick-aligned


def _grid(
    route: str, states: Sequence[str | None], start: int = T0
) -> dict[tuple[str, int], str]:
    """Build (route, tick) -> state entries on the TICK grid. `None` marks a
    feed gap: the tick key is simply omitted, not stored as a value."""
    return {
        (route, start + i * TICK): state
        for i, state in enumerate(states)
        if state is not None
    }


def _at(route: str, offset: int, start: int = T0) -> tuple[str, int]:
    """A (route, tick) key `offset` ticks from `start`."""
    return (route, start + offset * TICK)


# --- 1. Cohort filter ------------------------------------------------------


def test_cohort_filter_only_disrupted_without_alert_escalates():
    movement_state = {
        ("A", T0): "disrupted",  # escalation: disrupted, no alert
        ("B", T0): "normal",  # never escalates
        ("C", T0): "suspended",  # never escalates
        ("D", T0): "not_scheduled",  # never escalates
        ("E", T0): "disrupted",  # agreement with alert -> no escalation
    }
    alert_disrupted = {("E", T0)}
    events = escalation_events(movement_state, alert_disrupted)
    assert [e.route for e in events] == ["A"]


# --- 2. Onset dedup ----------------------------------------------------------


def test_contiguous_disrupted_run_yields_one_onset():
    movement_state = _grid("A", ["disrupted"] * 5)
    events = escalation_events(movement_state, set())
    assert len(events) == 1
    assert events[0].tick == T0
    assert events[0].persisted_ticks == 5


# --- 3. Gap tolerance boundary ----------------------------------------------


def test_gap_within_tolerance_bridges_to_a_single_onset():
    # 1 missing tick in the middle, gap_tolerance_ticks=1 -> tolerated.
    states = ["disrupted", "disrupted", None, "disrupted", "disrupted"]
    movement_state = _grid("A", states)
    events = escalation_events(movement_state, set(), gap_tolerance_ticks=1)
    assert [e.tick for e in events] == [T0]
    # persistence is a same-signal forward walk and is NOT bridged by the gap
    # tolerance that onset detection uses: it stops the instant it hits the gap.
    assert events[0].persisted_ticks == 2


def test_gap_beyond_tolerance_creates_a_new_onset():
    # 2 missing ticks in the middle, gap_tolerance_ticks=1 -> exceeds tolerance.
    states = ["disrupted", "disrupted", None, None, "disrupted", "disrupted"]
    movement_state = _grid("A", states)
    events = escalation_events(movement_state, set(), gap_tolerance_ticks=1)
    assert [e.tick for e in events] == [T0, T0 + 4 * TICK]
    assert events[0].persisted_ticks == 2
    assert events[1].persisted_ticks == 2


# --- 4. Post-alert tail ------------------------------------------------------


def test_post_alert_tail_flagged_and_excluded_from_summary_headline():
    # Alert covers T0 (agreement), clears from T0+TICK on; movement stays
    # disrupted right through the clearance -> the onset at T0+TICK is a tail
    # of the just-cleared alert, not movement leading one.
    movement_state = {
        ("A", T0): "disrupted",
        ("A", T0 + TICK): "disrupted",
        ("A", T0 + 2 * TICK): "disrupted",
    }
    alert_disrupted = {("A", T0)}
    events = escalation_events(movement_state, alert_disrupted)
    assert len(events) == 1
    onset = events[0]
    assert onset.tick == T0 + TICK
    assert onset.post_alert_tail is True
    assert onset.persisted_ticks == 2

    summary = escalation_summary(events)
    assert summary["n_escalations"] == 0
    assert summary["n_post_alert_tail"] == 1


def test_post_alert_tail_lookback_boundary():
    # Alert exactly `lookback` ticks back is inside the window (tail); one
    # tick further back is outside it (not a tail).
    lookback = 3
    movement_state = {
        ("IN", T0): "disrupted",
        ("OUT", T0): "disrupted",
    }
    alert_disrupted = {
        ("IN", T0 - lookback * TICK),
        ("OUT", T0 - (lookback + 1) * TICK),
    }
    events = escalation_events(
        movement_state, alert_disrupted, prior_alert_lookback_ticks=lookback
    )
    by_route = {e.route: e for e in events}
    assert by_route["IN"].post_alert_tail is True
    assert by_route["OUT"].post_alert_tail is False


# --- 5. Forward confirmation & lead time ------------------------------------


def test_forward_confirmation_horizon_boundary():
    horizon = 3
    movement_state = {
        ("AT", T0): "disrupted",  # alert lands exactly at t+horizon
        ("PAST", T0): "disrupted",  # alert lands one tick beyond horizon
    }
    alert_disrupted = {
        ("AT", T0 + horizon * TICK),
        ("PAST", T0 + (horizon + 1) * TICK),
    }
    events = escalation_events(movement_state, alert_disrupted, horizon_ticks=horizon)
    by_route = {e.route: e for e in events}
    assert by_route["AT"].lead_ticks == horizon
    assert by_route["AT"].alert_confirmed is True
    assert by_route["PAST"].lead_ticks is None
    assert by_route["PAST"].alert_confirmed is False


def test_alert_at_onset_tick_precludes_the_escalation_entirely():
    # An alert at the SAME tick as the movement disruption is agreement, not
    # escalation (contract #1) -- no event is emitted at all, so a same-tick
    # alert can never register as "confirmation".
    movement_state = {("A", T0): "disrupted"}
    alert_disrupted = {("A", T0)}
    events = escalation_events(movement_state, alert_disrupted)
    assert events == []


# --- 6. Persistence ----------------------------------------------------------


def test_persistence_stops_at_first_normal_state():
    movement_state = _grid("A", ["disrupted", "disrupted", "disrupted", "normal"])
    events = escalation_events(movement_state, set())
    assert len(events) == 1
    assert events[0].persisted_ticks == 3


# --- 7. Evaporated ------------------------------------------------------------


def test_evaporated_true_for_unconfirmed_single_tick_blip():
    movement_state = {("A", T0): "disrupted", ("A", T0 + TICK): "normal"}
    events = escalation_events(movement_state, set())
    assert len(events) == 1
    assert events[0].persisted_ticks == 1
    assert events[0].lead_ticks is None
    assert events[0].evaporated is True


def test_evaporated_false_for_sustained_unconfirmed_run():
    movement_state = _grid("A", ["disrupted"] * 3 + ["normal"])
    events = escalation_events(movement_state, set())
    assert len(events) == 1
    assert events[0].persisted_ticks == 3
    assert events[0].lead_ticks is None
    assert events[0].evaporated is False


def test_evaporated_false_when_blip_is_later_confirmed():
    # persisted_ticks alone does not decide evaporation: a 1-tick blip that is
    # later confirmed by an alert is NOT evaporated.
    movement_state = {("A", T0): "disrupted", ("A", T0 + TICK): "normal"}
    alert_disrupted = {("A", T0 + 2 * TICK)}
    events = escalation_events(movement_state, alert_disrupted)
    assert len(events) == 1
    assert events[0].persisted_ticks == 1
    assert events[0].lead_ticks == 2
    assert events[0].evaporated is False


# --- 8. Summary math ----------------------------------------------------------


def test_summary_aggregates_confirmation_lead_and_per_route_breakdown():
    # Route A: confirmed at 15min, confirmed at 30min, an unconfirmed blip
    #          (evaporated).
    # Route B: confirmed at 60min, an unconfirmed sustained run (persisted ==
    #          SUSTAINED_TICKS, not evaporated).
    movement_state = {
        **_grid("A", ["disrupted"], start=T0),
        **_grid("A", ["disrupted"], start=T0 + 20 * TICK),
        **_grid("A", ["disrupted"], start=T0 + 40 * TICK),
        **_grid("B", ["disrupted"], start=T0),
        **_grid("B", ["disrupted"] * SUSTAINED_TICKS, start=T0 + 20 * TICK),
    }
    alert_disrupted = {
        _at("A", 3),  # confirms the A onset at offset 0 -> 15 min
        _at("A", 26),  # confirms the A onset at offset 20 -> 30 min
        _at("B", 12),  # confirms the B onset at offset 0 -> 60 min
    }
    events = escalation_events(
        movement_state, alert_disrupted, horizon_ticks=12, gap_tolerance_ticks=2
    )
    assert len(events) == 5
    assert not any(e.post_alert_tail for e in events)

    by_key = {(e.route, e.tick): e for e in events}
    e1 = by_key[_at("A", 0)]
    e2 = by_key[_at("A", 20)]
    e5 = by_key[_at("A", 40)]
    e3 = by_key[_at("B", 0)]
    e4 = by_key[_at("B", 20)]

    assert (e1.lead_ticks, e2.lead_ticks, e3.lead_ticks) == (3, 6, 12)
    assert e5.lead_ticks is None
    assert e5.evaporated is True
    assert e4.lead_ticks is None
    assert e4.persisted_ticks == SUSTAINED_TICKS
    assert e4.evaporated is False

    summary = escalation_summary(events)
    assert summary["n_escalations"] == 5
    assert summary["n_post_alert_tail"] == 0
    assert summary["n_alert_confirmed"] == 3
    assert summary["alert_confirmed_rate"] == 3 / 5
    # buckets are cumulative ("<= minutes"), not exclusive counts per bucket.
    assert summary["confirmed_within_15min"] == 1
    assert summary["confirmed_within_30min"] == 2
    assert summary["confirmed_within_60min"] == 3
    assert summary["lead_minutes_median"] == 30.0
    assert summary["lead_minutes_mean"] == 35.0
    assert summary["n_evaporated"] == 1
    assert summary["evaporated_rate"] == 1 / 5
    assert summary["n_sustained_unconfirmed"] == 1
    assert summary["per_route"] == {
        "A": {"n": 3, "alert_confirmed": 2, "evaporated": 1},
        "B": {"n": 2, "alert_confirmed": 1, "evaporated": 0},
    }


def test_summary_source_and_horizon_minutes_pass_through():
    movement_state = {("A", T0): "disrupted"}
    events = escalation_events(movement_state, set())
    summary = escalation_summary(events, source="archived_published", horizon_ticks=6)
    assert summary["source"] == "archived_published"
    assert summary["horizon_minutes"] == 30


# --- 9. Empty / degenerate ----------------------------------------------------


def test_empty_movement_state_yields_no_events_and_none_rates():
    events = escalation_events({}, set())
    assert events == []
    summary = escalation_summary(events)
    assert summary["n_escalations"] == 0
    assert summary["n_post_alert_tail"] == 0
    assert summary["n_alert_confirmed"] == 0
    assert summary["alert_confirmed_rate"] is None
    assert summary["evaporated_rate"] is None
    assert summary["lead_minutes_median"] is None
    assert summary["lead_minutes_mean"] is None
    assert summary["per_route"] == {}


def test_summary_with_only_tails_has_zero_escalations_but_reports_tail_count():
    movement_state = {
        ("A", T0): "disrupted",
        ("B", T0): "disrupted",
    }
    alert_disrupted = {
        ("A", T0 - TICK),
        ("B", T0 - TICK),
    }
    events = escalation_events(movement_state, alert_disrupted)
    assert len(events) == 2
    assert all(e.post_alert_tail for e in events)

    summary = escalation_summary(events)
    assert summary["n_escalations"] == 0
    assert summary["n_post_alert_tail"] == 2
    assert summary["alert_confirmed_rate"] is None
    assert summary["per_route"] == {}


# --- 10. Multi-route independence --------------------------------------------


def test_events_are_independent_per_route():
    # Route A: a clean, isolated onset. Route B: agreement first, then its own
    # onset one tick later. Neither route's scan should leak into the other's
    # onset detection, persistence, or forward confirmation.
    movement_state = {
        **_grid("A", ["disrupted", "disrupted"]),
        ("B", T0): "disrupted",  # agreement, no event
        ("B", T0 + TICK): "disrupted",  # onset for B
        ("B", T0 + 2 * TICK): "disrupted",
    }
    alert_disrupted = {("B", T0)}
    events = escalation_events(movement_state, alert_disrupted)
    by_route = {e.route: e for e in events}

    assert set(by_route) == {"A", "B"}
    assert by_route["A"].tick == T0
    assert by_route["A"].persisted_ticks == 2
    assert by_route["B"].tick == T0 + TICK
    assert by_route["B"].persisted_ticks == 2
