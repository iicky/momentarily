"""Per-segment traversal baselines (training/traversal.py).

Synthetic Traversal lists only — no R2, no trace reconstruction. Each case pins
one rule about which observations reach a cell and where a cell's level comes
from.
"""

from __future__ import annotations

from training.gtfs_static import HopKey
from training.trace import EXACT, INTERVAL, RIGHT, Traversal
from training.traversal import (
    OWN,
    SCHEDULED,
    TraversalCell,
    deviation,
    hop_samples,
    traversal_baseline,
)

KEY: HopKey = ("A", "south", "A1S", "A2S")
OTHER: HopKey = ("A", "south", "A2S", "A3S")


def _exact(seconds: int, *, key: HopKey = KEY, trip: str = "t1") -> Traversal:
    route, direction, frm, to = key
    return Traversal(
        trip_id=trip,
        route_id=route,
        direction=direction,
        from_stop=frm,
        to_stop=to,
        seconds=seconds,
        moving_seconds=None,
        n_hops=1,
        censoring=EXACT,
    )


def _many(seconds: int, n: int, *, key: HopKey = KEY) -> list[Traversal]:
    return [_exact(seconds + i % 7, key=key, trip=f"t{i}") for i in range(n)]


def test_hop_samples_keeps_only_exact_single_hops():
    """A right-censored traversal has no destination and an interval span
    covers several hops, so neither can be attributed to one segment."""
    right = Traversal(
        trip_id="t2",
        route_id="A",
        direction="south",
        from_stop="A1S",
        to_stop=None,
        seconds=400,
        moving_seconds=None,
        n_hops=None,
        censoring=RIGHT,
    )
    interval = Traversal(
        trip_id="t3",
        route_id="A",
        direction="south",
        from_stop="A1S",
        to_stop="A3S",
        seconds=500,
        moving_seconds=None,
        n_hops=2,
        censoring=INTERVAL,
    )
    samples = hop_samples([_exact(90), right, interval])
    assert samples == {KEY: [(90, True)]}


def test_hop_samples_rejects_a_multi_hop_span_however_it_is_labelled():
    """The single-hop requirement is enforced on n_hops, not inferred from the
    censoring kind: a span that crossed a station in between is not a
    measurement of (from_stop, to_stop) even if both ends were seen."""
    two_hops = Traversal(
        trip_id="t4",
        route_id="A",
        direction="south",
        from_stop="A1S",
        to_stop="A2S",
        seconds=500,
        moving_seconds=None,
        n_hops=2,
        censoring=EXACT,
    )
    assert hop_samples([two_hops]) == {}


def test_hop_samples_separates_successors_from_the_same_platform():
    """A branch point serves two successors from one from_stop; pooling them
    would average a short hop with a long one."""
    samples = hop_samples([_exact(60), _exact(240, key=OTHER)])
    assert set(samples) == {KEY, OTHER}


def test_a_well_observed_hop_is_fitted_on_its_own():
    cells, stats = traversal_baseline(_many(100, 40), {KEY: 90})
    cell = cells[KEY]
    assert cell.source == OWN
    assert cell.n == 40
    # Fitted from samples spanning 100..106s, so the median lands in that band
    # and p90 sits above it.
    assert 100 <= cell.median_sec <= 107
    assert cell.p90_sec > cell.median_sec
    assert stats.n_cells_own == 1
    assert stats.n_cells_scheduled == 0


def test_a_thin_hop_takes_its_level_from_its_own_scheduled_time():
    """Not from the population's raw seconds: a thin 400s hop must not inherit
    a 100s hop's curve just because most segments are short."""
    traversals = [*_many(100, 40), *_many(410, 3, key=OTHER)]
    cells, stats = traversal_baseline(traversals, {KEY: 90, OTHER: 400})

    thin = cells[OTHER]
    assert thin.source == SCHEDULED
    assert thin.n == 3
    # The population ratio is ~100/90 = 1.11, applied to this hop's own 400s.
    assert 400 <= thin.median_sec <= 480
    assert thin.p90_sec > thin.median_sec
    assert stats.n_cells_scheduled == 1


def test_a_thin_hop_the_timetable_does_not_name_is_omitted():
    """Neither its own data nor the timetable can supply a level, so the cell
    is absent rather than guessed -- callers read that as "can't judge"."""
    traversals = [*_many(100, 40), *_many(410, 3, key=OTHER)]
    cells, stats = traversal_baseline(traversals, {KEY: 90})

    assert OTHER not in cells
    assert stats.n_keys_unjudgeable == 1
    assert stats.n_keys == 2


def test_stats_report_what_the_fit_discarded():
    right = Traversal(
        trip_id="t9",
        route_id="A",
        direction="south",
        from_stop="A1S",
        to_stop=None,
        seconds=400,
        moving_seconds=None,
        n_hops=None,
        censoring=RIGHT,
    )
    interval = Traversal(
        trip_id="t8",
        route_id="A",
        direction="south",
        from_stop="A1S",
        to_stop="A3S",
        seconds=500,
        moving_seconds=None,
        n_hops=2,
        censoring=INTERVAL,
    )
    _cells, stats = traversal_baseline([*_many(100, 40), right, interval], {KEY: 90})
    assert stats.n_traversals == 42
    assert stats.n_fitted == 40
    assert stats.n_dropped_right == 1
    assert stats.n_dropped_interval == 1


def test_population_ratio_median_is_measured_against_the_timetable():
    cells, stats = traversal_baseline(
        [_exact(180) for _ in range(30)], {KEY: 90}, min_hop_samples=30
    )
    assert stats.population_ratio_median == 2.0
    assert cells[KEY].scheduled_sec == 90


def test_deviation_is_relative_to_the_segments_own_median():
    """The point of a per-segment baseline: doubling a 60s hop and doubling a
    400s hop read the same."""
    short = TraversalCell(
        n=50, median_sec=60.0, p90_sec=80.0, scheduled_sec=60, source=OWN
    )
    long = TraversalCell(
        n=50, median_sec=400.0, p90_sec=520.0, scheduled_sec=400, source=OWN
    )
    assert deviation(short, 120) == 2.0
    assert deviation(long, 800) == 2.0
    assert deviation(short, 60) == 1.0
