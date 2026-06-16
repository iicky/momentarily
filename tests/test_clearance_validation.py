"""Independent recovery validation from alert-feed clearance (training/review.py).

Synthetic prediction streams — no R2. The disruption interval comes from the raw
feed (primary_alert_type), independent of the HMM argmax; grading reuses
independent_recovery_metrics. A feed-clearance proxy, not true service recovery
(that's the trip-updates signal). See momentarily-up0 / momentarily-xum.
"""

from __future__ import annotations

from training.eval import PredictionRecord, independent_recovery_metrics
from training.review import clearance_disruptions, is_disruptive

TICK = 300
T0 = 1_700_000_100  # tick-aligned


def _pred(
    ts: int,
    *,
    route: str = "A",
    condition: str = "disrupted",
    alert_type: str | None = "Delays",
    recovery_minutes: int = 30,
    low: int = 0,
    high: int = 60,
    indeterminate: bool = False,
    source: str | None = "hmm",
) -> PredictionRecord:
    return PredictionRecord(
        ts=ts,
        route=route,
        condition=condition,
        regime_entered_at=T0,
        p_normal=0.2,
        p_disrupted=0.8,
        p_suspended=0.0,
        p_normal_in_30min=0.5,
        p_normal_in_60min=0.6,
        p_normal_in_120min=0.7,
        recovery_minutes=recovery_minutes,
        recovery_minutes_low=low,
        recovery_minutes_high=high,
        recovery_indeterminate=indeterminate,
        primary_alert_type=alert_type,
        recovery_source=source,
    )


def _stream(route: str, alert_types: list[str | None]) -> list[PredictionRecord]:
    """One prediction per tick; condition tracks whether an alert is present."""
    out: list[PredictionRecord] = []
    for i, at in enumerate(alert_types):
        cond = "disrupted" if is_disruptive(at) else "normal"
        out.append(_pred(T0 + i * TICK, route=route, condition=cond, alert_type=at))
    return out


def test_is_disruptive_classifies_by_category():
    assert is_disruptive("Delays")
    assert is_disruptive("Trains Rerouted")  # service_change
    assert is_disruptive("Suspended")
    assert not is_disruptive(None)
    assert not is_disruptive("Planned - Service Change")
    assert not is_disruptive("Station Notice")  # information


def test_one_clearance_disruption_detected():
    # clear x3, Delays x4, clear x5  → one interval [onset, first-clear)
    preds = _stream("A", [None, None, None] + ["Delays"] * 4 + [None] * 5)
    out = clearance_disruptions(preds, debounce=2)
    assert len(out) == 1
    d = out[0]
    assert d.route == "A"
    assert d.start_tick == T0 + 3 * TICK  # first alert tick
    assert d.recovered_tick == T0 + 7 * TICK  # first clear tick after the alert


def test_open_disruption_at_window_end_is_censored():
    # alert never clears for `debounce` ticks → no interval emitted
    preds = _stream("A", [None, None] + ["Delays"] * 4)
    assert clearance_disruptions(preds, debounce=2) == []


def test_flapping_clearance_is_debounced():
    # a single clear tick between alerts must not close the disruption
    preds = _stream("A", ["Delays", "Delays", None, "Delays", "Delays"] + [None] * 3)
    out = clearance_disruptions(preds, debounce=2)
    assert len(out) == 1
    assert out[0].recovered_tick == T0 + 5 * TICK  # the real, sustained clear


def test_planned_alert_is_not_a_disruption():
    # a lingering planned/info alert keeps primary_alert_type non-null but is not
    # a disruptive category, so it does not extend the interval
    preds = _stream("A", [None] + ["Delays"] * 3 + ["Station Notice"] * 3)
    out = clearance_disruptions(preds, debounce=2)
    assert len(out) == 1
    assert out[0].recovered_tick == T0 + 4 * TICK  # when Delays cleared, not later


def test_grading_against_clearance_truth():
    # disrupted T0+3..T0+6, recovers at T0+7 (=35 min after onset). The prediction
    # at the onset tick has 4 ticks (20 min) of real remaining time.
    preds = _stream("A", [None, None, None] + ["Delays"] * 4 + [None] * 5)
    disruptions = clearance_disruptions(preds, debounce=2)
    result = independent_recovery_metrics(preds, disruptions)
    assert result.overall.n == 4  # the four disrupted ticks are graded
    # onset tick: predicted 30 min, actual remaining = (T0+7 - (T0+3))/60 = 20 min
    assert result.overall.mae_min is not None
