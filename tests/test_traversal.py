"""Per-segment traversal baselines (training/traversal.py).

Synthetic Traversal lists against a synthetic timetable — no R2, no trace
reconstruction. Each case pins one rule about which observations reach a cell,
where a cell's level comes from, and what the timetable is and is not allowed
to touch.
"""

from __future__ import annotations

import zipfile
from dataclasses import replace
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from training.gtfs_static import HopKey, Timetable, timetable
from training.trace import EXACT, INTERVAL, RIGHT, Traversal
from training.traversal import (
    TraversalCell,
    deviation,
    hop_samples,
    schedule_drift,
    traversal_baseline,
)

from .conftest import FEED_VERSION, make_gtfs_zip

KEY: HopKey = ("A", "south", "A1S", "A2S")
OTHER: HopKey = ("A", "south", "A2S", "A3S")

# A weekday noon, and a trip whose id says it left at noon — so the observation
# lands on the weekday timetable by the same rule the live pipeline uses.
WEEKDAY = date(2026, 8, 12)
AT = int(
    datetime.combine(
        WEEKDAY, time(12, 5), tzinfo=ZoneInfo("America/New_York")
    ).timestamp()
)
TRIP = "072000_A..S01R"
UNKNOWN_TRIP = "072000_A..S99R"  # no such pattern: the day median has to carry it


def _feed() -> zipfile.ZipFile:
    """One southbound weekday pattern: A1S -> A2S is 90 scheduled seconds,
    A2S -> A3S is 400."""
    return make_gtfs_zip(
        [f"A,{TRIP},Weekday,X,1,A..S01R"],
        [
            f"{TRIP},A1S,12:00:00,12:00:00,1",
            f"{TRIP},A2S,12:01:30,12:01:30,2",
            f"{TRIP},A3S,12:08:10,12:08:10,3",
        ],
    )


def _timetable() -> Timetable:
    return timetable(_feed())


def _exact(
    seconds: int, *, key: HopKey = KEY, trip: str = TRIP, n_hops: int = 1
) -> Traversal:
    route, direction, frm, to = key
    return Traversal(
        trip_id=trip,
        route_id=route,
        direction=direction,
        from_stop=frm,
        to_stop=to,
        at=AT,
        seconds=seconds,
        moving_seconds=None,
        n_hops=n_hops,
        censoring=EXACT,
    )


def _many(
    seconds: int, n: int, *, key: HopKey = KEY, trip: str = TRIP
) -> list[Traversal]:
    return [_exact(seconds + i % 7, key=key, trip=trip) for i in range(n)]


def _right() -> Traversal:
    return Traversal(
        trip_id=TRIP,
        route_id="A",
        direction="south",
        from_stop="A1S",
        to_stop=None,
        at=AT,
        seconds=400,
        moving_seconds=None,
        n_hops=None,
        censoring=RIGHT,
    )


def _interval() -> Traversal:
    return Traversal(
        trip_id=TRIP,
        route_id="A",
        direction="south",
        from_stop="A1S",
        to_stop="A3S",
        at=AT,
        seconds=500,
        moving_seconds=None,
        n_hops=2,
        censoring=INTERVAL,
    )


def test_hop_samples_keeps_only_exact_single_hops():
    """A right-censored traversal has no destination and an interval span
    covers several hops, so neither can be attributed to one segment."""
    got = hop_samples([_exact(90), _right(), _interval()], _timetable())
    assert list(got.by_key) == [KEY]
    assert [(s.seconds, s.scheduled_sec) for s in got.by_key[KEY]] == [(90, 90)]


def test_hop_samples_rejects_a_multi_hop_span_however_it_is_labelled():
    """The single-hop requirement is enforced on n_hops, not inferred from the
    censoring kind: a span that crossed a station in between is not a
    measurement of (from_stop, to_stop) even if both ends were seen."""
    assert hop_samples([_exact(500, n_hops=2)], _timetable()).by_key == {}


def test_a_bypass_is_counted_but_kept_in_the_fit():
    """The feed reports its own stop sequence, so a train that skipped A2S calls
    A1S -> A3S one hop. It really did run that stretch nonstop, which is exactly
    what an express does on the same pair, so it is a measurement of it. Letting
    the timetable throw it out would put the schedule in charge of the training
    set and drop movement that clusters on bad days."""
    bypass = _exact(490, key=("A", "south", "A1S", "A3S"))
    got = hop_samples([_exact(90), bypass], _timetable())
    assert got.n_bypass == 1
    assert set(got.by_key) == {KEY, ("A", "south", "A1S", "A3S")}


def test_hop_samples_separates_successors_from_the_same_platform():
    """A branch point serves two successors from one from_stop; pooling them
    would average a short hop with a long one."""
    got = hop_samples([_exact(60), _exact(240, key=OTHER)], _timetable())
    assert set(got.by_key) == {KEY, OTHER}


def test_a_trip_the_timetable_does_not_name_gets_the_day_median_instead():
    """A fifth of live trip ids never match a pattern — rerouted paths, dispatch
    origins off the timetable, Staten Island. The comparison still resolves, off
    the median for the pair on the day they ran."""
    got = hop_samples([_exact(120, trip=UNKNOWN_TRIP)], _timetable())
    assert [s.scheduled_sec for s in got.by_key[KEY]] == [90]


def test_an_observation_outside_the_feed_window_is_compared_against_nothing():
    """The vehicle archive reaches back further than any one GTFS snapshot. A
    traversal from before this feed took effect still fits its own cell — the
    trains ran — but carries no scheduled time, because the schedule it would be
    measured against was not in force."""
    stale = replace(_exact(100), at=AT - 400 * 86_400)
    got = hop_samples([stale], _timetable())
    assert got.n_outside_feed_window == 1
    assert [s.scheduled_sec for s in got.by_key[KEY]] == [None]


def test_a_well_observed_hop_is_fitted_from_its_own_movement():
    cells, stats = traversal_baseline(_many(100, 40), _timetable())
    cell = cells[KEY]
    assert cell.n == 40
    # Fitted from samples spanning 100..106s, so the median lands in that band
    # and p90 sits above it.
    assert 100 <= cell.median_sec <= 107
    assert cell.p90_sec > cell.median_sec
    assert stats.n_cells == 1
    assert stats.n_keys_thin == 0


def test_a_thin_hop_abstains_rather_than_borrowing_a_level():
    """The timetable names this hop's scheduled time, and it is still not used
    to invent a level. A reference that fed the fit could not then measure the
    fit drifting, so a segment with too little movement of its own says nothing.
    """
    traversals = [*_many(100, 40), *_many(410, 3, key=OTHER)]
    cells, stats = traversal_baseline(traversals, _timetable())

    assert OTHER not in cells
    assert stats.n_keys == 2
    assert stats.n_cells == 1
    assert stats.n_keys_thin == 1


def test_a_fitted_cell_still_records_what_the_timetable_allowed():
    """Carried for schedule_drift, never fitted from."""
    cells, _stats = traversal_baseline(
        [_exact(180) for _ in range(30)], _timetable(), min_hop_samples=30
    )
    assert cells[KEY].scheduled_sec == 90
    assert 170 <= cells[KEY].median_sec <= 190


def test_stats_report_what_the_fit_discarded():
    _cells, stats = traversal_baseline(
        [*_many(100, 40), _right(), _interval()], _timetable()
    )
    assert stats.n_traversals == 42
    assert stats.n_fitted == 40
    assert stats.n_dropped_right == 1
    assert stats.n_dropped_interval == 1
    assert stats.n_bypass == 0
    assert stats.n_outside_feed_window == 0


def test_schedule_drift_measures_the_fit_against_the_timetable_it_ignored():
    """Forty traversals of a 90s hop that all took 180s. The cells know nothing
    of the schedule, so the drift measure is free to say the fleet is running at
    twice its own timetable — which is the whole point of keeping it outside."""
    tt = _timetable()
    cells, _stats = traversal_baseline(_many(180, 40), tt)
    drift = schedule_drift(cells, tt)
    assert drift is not None
    assert drift.n_cells == 1
    assert 1.9 <= drift.median <= 2.1
    assert drift.share_slow == 1.0
    assert drift.feed_version == FEED_VERSION


def test_schedule_drift_abstains_when_no_cell_has_a_scheduled_time():
    """Nothing to measure against is not the same as measuring zero drift."""
    tt = _timetable()
    stale = [replace(t, at=AT - 400 * 86_400) for t in _many(100, 40)]
    cells, stats = traversal_baseline(stale, tt)
    assert stats.n_outside_feed_window == 40
    assert cells[KEY].scheduled_sec is None
    assert schedule_drift(cells, tt) is None


def test_deviation_is_relative_to_the_segments_own_median():
    """The point of a per-segment baseline: doubling a 60s hop and doubling a
    400s hop read the same."""
    short = TraversalCell(n=50, median_sec=60.0, p90_sec=80.0, scheduled_sec=60)
    long = TraversalCell(n=50, median_sec=400.0, p90_sec=520.0, scheduled_sec=400)
    assert deviation(short, 120) == 2.0
    assert deviation(long, 800) == 2.0
    assert deviation(short, 60) == 1.0
