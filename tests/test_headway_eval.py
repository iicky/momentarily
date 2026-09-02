"""Wait-signal evaluation (training/headway_eval.py): the confirmed-normal
reference, the night-clustered bootstrap's withholding rule, the flags, and the
severity report's disagreement/feed-context accounting. Synthetic data only."""

from __future__ import annotations

from datetime import date, datetime
from typing import cast
from zoneinfo import ZoneInfo

from training.eval_common import service_night
from training.headway import TickWait, WaitCell
from training.headway_eval import (
    Unit,
    above_p90,
    confirmed_normal,
    night_bootstrap,
    severity_report,
    severity_tier,
    twice_typical,
)

_ET = ZoneInfo("America/New_York")


def _epoch(h: int, mi: int = 0, day: int = 20) -> int:
    return int(datetime(2026, 8, day, h, mi, tzinfo=_ET).timestamp())


def _cell(p50: float, p90: float) -> WaitCell:
    return WaitCell(
        p10=p50 / 2, p50=p50, p90=p90, cv_p50=0.5, cv_p90=1.0, n_ticks=100, n_nights=8
    )


# --- confirmed-normal is the AND of two axes, then two liveness witnesses ---


def _witnessed(*ticks: int) -> tuple[dict[int, set[str]], set[date]]:
    """Both witnesses satisfied for `ticks`: route A's line-group feed fresh at
    each tick, and an alert version archived on each tick's service night."""
    return (
        {t: {"ace"} for t in ticks},
        {service_night(t) for t in ticks},
    )


def test_confirmed_normal_requires_both_axes():
    t1, t2, t3 = _epoch(10), _epoch(11), _epoch(12)
    mv = {("A", t1): "normal", ("A", t2): "normal", ("A", t3): "disrupted"}
    sup = {("A", t1): "normal", ("A", t2): "disrupted"}  # t3 absent = abstain
    fresh, nights = _witnessed(t1, t2, t3)
    got = confirmed_normal(mv, sup, fresh=fresh, alert_nights=nights)
    assert got == {("A", t1)}


def test_confirmed_normal_excludes_ticks_the_archive_cannot_witness():
    """Both axes read the same archive, so a collection gap makes them agree on
    "nothing wrong" for want of any evidence. An unwitnessed tick is dropped."""
    t1, t2 = _epoch(10), _epoch(11)
    mv = {("A", t1): "normal", ("A", t2): "normal"}
    sup = {("A", t1): "normal", ("A", t2): "normal"}
    fresh, nights = _witnessed(t1)  # t2 has no collected body
    assert confirmed_normal(mv, sup, fresh=fresh, alert_nights=nights) == {("A", t1)}


def test_confirmed_normal_excludes_ticks_whose_own_feed_group_stalled():
    """A tick can be collected while the route's own line-group feed is stale —
    its trains simply vanish from the trace, which is common-mode blindness, not
    calm service. Route A is governed by the 'ace' group."""
    t1, t2 = _epoch(10), _epoch(11)
    mv = {("A", t1): "normal", ("A", t2): "normal"}
    sup = {("A", t1): "normal", ("A", t2): "normal"}
    fresh = {t1: {"ace"}, t2: {"bdfm"}}  # t2 collected, but not A's group
    nights = {service_night(t1), service_night(t2)}
    assert confirmed_normal(mv, sup, fresh=fresh, alert_nights=nights) == {("A", t1)}


def test_confirmed_normal_excludes_nights_with_no_alert_version_archived():
    """The alerts feed is a separate fetch with a stale-fallback, so cron
    liveness alone cannot prove quiet. A night with no alert archived anywhere is
    an outage reading as calm, and is excluded even though both axes say normal
    and the vehicle feed was fresh."""
    t1 = _epoch(10, day=20)
    t2 = _epoch(10, day=21)  # a different service night
    mv = {("A", t1): "normal", ("A", t2): "normal"}
    sup = {("A", t1): "normal", ("A", t2): "normal"}
    fresh = {t1: {"ace"}, t2: {"ace"}}
    nights = {service_night(t1)}  # t2's night saw no alert version at all
    assert confirmed_normal(mv, sup, fresh=fresh, alert_nights=nights) == {("A", t1)}


# --- night bootstrap withholds the interval below two clusters ---


def test_night_bootstrap_withholds_below_two_units():
    r = night_bootstrap([Unit(alarmed=1, total=10)])
    assert r.rate == 0.1
    assert r.lo is None
    assert r.hi is None
    assert r.n_units == 1


def test_night_bootstrap_gives_interval_with_enough_units():
    units = [Unit(alarmed=1, total=10) for _ in range(20)]
    r = night_bootstrap(units, n_boot=500)
    assert r.rate == 0.1
    assert r.rate is not None
    assert r.lo is not None
    assert r.hi is not None
    assert r.lo <= r.rate <= r.hi


def test_night_bootstrap_no_units_no_rate():
    r = night_bootstrap([])
    assert r.rate is None


# --- flags ---


def test_flags_fire_on_the_right_side_of_the_cell():
    cell = _cell(p50=300, p90=600)
    tw_hi = TickWait("A", "north", _epoch(12), awt_sec=700, cv=1.0, n_headways=5)
    tw_lo = TickWait("A", "north", _epoch(12), awt_sec=500, cv=0.5, n_headways=5)
    assert above_p90(tw_hi, cell)
    assert not above_p90(tw_lo, cell)
    assert twice_typical(tw_hi, cell)  # 700 > 2*300
    assert not twice_typical(tw_lo, cell)  # 500 < 600
    assert severity_tier(tw_hi, cell) == 2
    assert severity_tier(tw_lo, cell) == 0


def test_severity_tier_degraded_band():
    # 2*p50 (800) above p90 (650): the band between p90 and 2*p50 is degraded
    cell = _cell(p50=400, p90=650)
    tw = TickWait("A", "north", _epoch(12), awt_sec=700, cv=0.8, n_headways=5)
    assert severity_tier(tw, cell) == 1  # > p90 but not > 2*p50, so not severe


# --- severity report handles an empty baseline without dividing by zero ---


def test_severity_report_empty_baseline():
    waits = {("A", "north"): [TickWait("A", "north", _epoch(12), 400, 0.6, 5)]}
    rep = severity_report(waits, {}, {}, {}, {})
    assert rep == {"n_ticks": 0}


# --- severity report separates functional disagreement from feed blindness ---


def test_severity_report_disagreement_and_feed_context():
    cell = _cell(p50=200, p90=300)
    baseline = {("A", "north", "wd12"): cell}
    # three consecutive severe ticks, movement silent, no Delays -> headway-only
    ticks = [_epoch(12, m) for m in (0, 5, 10)]
    waits = {
        ("A", "north"): [
            TickWait("A", "north", t, awt_sec=900, cv=1.5, n_headways=6) for t in ticks
        ]
    }
    fresh = {t: {"ace"} for t in ticks}  # A's line-group present and fresh
    rep = severity_report(waits, baseline, {}, {}, fresh)
    assert rep["severe_ticks"] == 3
    assert rep["severe_headway_only"] == 3
    windows = cast(list[dict[str, object]], rep["disagreement_windows"])
    assert windows
    assert windows[0]["route"] == "A"
    assert windows[0]["coverage"] == 1.0
    assert windows[0]["group_fresh_share"] == 1.0
