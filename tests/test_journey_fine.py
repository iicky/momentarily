"""Fine-grained historical forecasters (training/journey_fine.py).

Synthetic spans throughout; the date -> day-type map is injected so no test
fetches the static feed.
"""

from __future__ import annotations

import math
from datetime import date as date_t
from datetime import datetime, time
from zoneinfo import ZoneInfo

from training.journey_baselines import MIN_N, HistoricalBaseline
from training.journey_fine import FORECASTERS, FineHistorical
from training.journey_types import QUANTILES, SpanSample

NY = ZoneInfo("America/New_York")
MONDAY = date_t(2026, 9, 7)
TUESDAY = date_t(2026, 9, 8)
SUNDAY = date_t(2026, 9, 6)

# Every synthetic date resolves through this injected map; MONDAY runs Sunday
# service, the way a holiday does.
DAY_TYPES = {
    MONDAY.isoformat(): "Sunday",
    TUESDAY.isoformat(): "Weekday",
    SUNDAY.isoformat(): "Sunday",
}


def _at(day: date_t, hour: int, minute: int = 0) -> int:
    return int(datetime.combine(day, time(hour, minute), tzinfo=NY).timestamp())


def _span(
    *,
    seconds: int,
    day: date_t = TUESDAY,
    hour: int = 12,
    minute: int = 0,
    origin: str = "Q05S",
    destination: str = "Q06S",
    scheduled: int | None = 100,
    n_hops: int = 1,
    trip: str = "t_Q..S01R",
) -> SpanSample:
    return SpanSample(
        trip_id=trip,
        route_id="Q",
        direction="S",
        origin=origin,
        destination=destination,
        depart_at=_at(day, hour, minute),
        seconds=seconds,
        n_hops=n_hops,
        scheduled_sec=scheduled,
        service_day=day.isoformat(),
    )


def _fine(
    name: str = "fine",
    *,
    hour_resolution: int = 1,
    use_pattern: bool = False,
    shrink: bool = False,
) -> FineHistorical:
    return FineHistorical(
        name,
        hour_resolution=hour_resolution,
        use_pattern=use_pattern,
        shrink=shrink,
        day_types=DAY_TYPES,
    )


def test_registry_names_and_configuration() -> None:
    assert set(FORECASTERS) == {
        "fine_4h_daytype",
        "fine_1h_daytype",
        "fine_1h_daytype_pattern",
        "fine_1h_daytype_shrunk",
        "fine_1h_daytype_pattern_shrunk",
    }
    for name, factory in FORECASTERS.items():
        f = factory()
        assert f.name == name


def test_day_type_follows_the_calendar_not_the_clock() -> None:
    # A Monday mapped to Sunday service uses the Sunday leaf: train holds
    # MIN_N Sunday spans at one value and MIN_N Tuesday (Weekday) spans at
    # another, all in the same clock hour.
    f = _fine()
    f.fit(
        [_span(seconds=100, day=SUNDAY, trip=f"s{i}") for i in range(MIN_N)]
        + [_span(seconds=500, day=TUESDAY, trip=f"w{i}") for i in range(MIN_N)]
    )
    q = f.predict(_span(seconds=0, day=MONDAY), [])
    assert q is not None
    assert math.isclose(q[0.50], 100.0)
    weekday = f.predict(_span(seconds=0, day=TUESDAY), [])
    assert weekday is not None
    assert math.isclose(weekday[0.50], 500.0)


def test_unmapped_date_falls_back_to_clock_weekday_and_is_counted() -> None:
    f = _fine()
    other_tuesday = date_t(2026, 9, 15)  # not in DAY_TYPES
    f.fit(
        [_span(seconds=100, day=TUESDAY, trip=f"w{i}") for i in range(MIN_N)]
        + [_span(seconds=500, day=SUNDAY, trip=f"s{i}") for i in range(MIN_N)]
    )
    q = f.predict(_span(seconds=0, day=other_tuesday), [])
    assert q is not None
    assert math.isclose(q[0.50], 100.0)  # clock Tuesday -> Weekday leaf
    assert f.n_clock_fallback_days == 1


def test_one_hour_bins_separate_759_from_800() -> None:
    f = _fine()
    f.fit(
        [_span(seconds=100, hour=7, minute=i % 60, trip=f"a{i}") for i in range(MIN_N)]
        + [
            _span(seconds=500, hour=8, minute=i % 60, trip=f"b{i}")
            for i in range(MIN_N)
        ]
    )
    early = f.predict(_span(seconds=0, hour=7, minute=59), [])
    late = f.predict(_span(seconds=0, hour=8, minute=0), [])
    assert early is not None
    assert math.isclose(early[0.50], 100.0)
    assert late is not None
    assert math.isclose(late[0.50], 500.0)


def test_four_hour_variant_pools_the_clock_hours() -> None:
    f = _fine(hour_resolution=4)
    f.fit(
        [_span(seconds=100, hour=7, trip=f"a{i}") for i in range(MIN_N // 2)]
        + [_span(seconds=500, hour=5, trip=f"b{i}") for i in range(MIN_N // 2)]
    )
    q = f.predict(_span(seconds=0, hour=6), [])
    assert q is not None
    assert 100.0 < q[0.50] < 500.0


def test_pattern_level_separates_two_path_codes_on_the_same_od() -> None:
    f = _fine(use_pattern=True)
    f.fit(
        [_span(seconds=100, trip=f"{i}_Q..S01R") for i in range(MIN_N)]
        + [_span(seconds=500, trip=f"{i}_Q..S02R") for i in range(MIN_N)]
    )
    local = f.predict(_span(seconds=0, trip="x_Q..S01R"), [])
    express = f.predict(_span(seconds=0, trip="x_Q..S02R"), [])
    assert local is not None
    assert math.isclose(local[0.50], 100.0)
    assert express is not None
    assert math.isclose(express[0.50], 500.0)


def test_each_fallback_step_fires_at_exactly_min_n() -> None:
    # Pattern variant, MIN_N spans of path S01R at 100 s and MIN_N of S02R at
    # 500 s, same OD and hour: the S01R pattern leaf fires at exactly MIN_N
    # and its median is 100. With MIN_N - 1 S01R spans the pattern leaf is
    # thin and predict falls to the mixed hour leaf, whose median is not 100.
    f = _fine(use_pattern=True)
    f.fit(
        [_span(seconds=100, hour=8, trip=f"{i}_Q..S01R") for i in range(MIN_N)]
        + [_span(seconds=500, hour=8, trip=f"{i}_Q..S02R") for i in range(MIN_N)]
    )
    q = f.predict(_span(seconds=0, hour=8, trip="y_Q..S01R"), [])
    assert q is not None
    assert math.isclose(q[0.50], 100.0)
    f2 = _fine(use_pattern=True)
    f2.fit(
        [_span(seconds=100, hour=8, trip=f"{i}_Q..S01R") for i in range(MIN_N - 1)]
        + [_span(seconds=500, hour=8, trip=f"{i}_Q..S02R") for i in range(MIN_N)]
    )
    q = f2.predict(_span(seconds=0, hour=8, trip="y_Q..S01R"), [])
    assert q is not None
    assert q[0.50] > 100.0

    # Hour leaf thin, (4h, day type) level at exactly MIN_N.
    f3 = _fine()
    f3.fit([_span(seconds=200, hour=8 + i % 4, trip=f"t{i}") for i in range(MIN_N)])
    q = f3.predict(_span(seconds=0, hour=11), [])
    assert q is not None
    assert math.isclose(q[0.50], 200.0)

    # (4h, day type) thin, plain 4h bin at exactly MIN_N: split the same 4h
    # bin across Weekday and Sunday service days.
    half = MIN_N // 2
    f4 = _fine()
    f4.fit(
        [
            _span(seconds=200, hour=8 + i % 4, day=TUESDAY, trip=f"t{i}")
            for i in range(half)
        ]
        + [
            _span(seconds=400, hour=8 + i % 4, day=SUNDAY, trip=f"s{i}")
            for i in range(half)
        ]
    )
    q = f4.predict(_span(seconds=0, hour=9, day=TUESDAY), [])
    assert q is not None
    assert 200.0 < q[0.50] < 400.0

    # Every time bin thin, OD pair at exactly MIN_N.
    f5 = _fine()
    f5.fit(
        [
            _span(
                seconds=300, hour=i % 20, day=TUESDAY if i % 2 else SUNDAY, trip=f"t{i}"
            )
            for i in range(MIN_N)
        ]
    )
    q = f5.predict(_span(seconds=0, hour=23), [])
    assert q is not None
    assert math.isclose(q[0.50], 300.0)

    # OD thin, (route, direction, n_hops) ratio at exactly MIN_N, scaled by
    # the sample's schedule.
    f6 = _fine()
    f6.fit(
        [
            _span(
                seconds=200,
                scheduled=100,
                origin=f"Q{i:02d}S",
                destination=f"Q{i + 1:02d}S",
                trip=f"t{i}",
            )
            for i in range(MIN_N)
        ]
    )
    q = f6.predict(
        _span(seconds=0, scheduled=50, origin="Q90S", destination="Q91S"), []
    )
    assert q is not None
    assert math.isclose(q[0.50], 100.0)
    # One fewer and the same sample abstains.
    f7 = _fine()
    f7.fit(
        [
            _span(
                seconds=200,
                scheduled=100,
                origin=f"Q{i:02d}S",
                destination=f"Q{i + 1:02d}S",
                trip=f"t{i}",
            )
            for i in range(MIN_N - 1)
        ]
    )
    assert (
        f7.predict(
            _span(seconds=0, scheduled=50, origin="Q90S", destination="Q91S"), []
        )
        is None
    )


def test_abstention_set_matches_the_baseline() -> None:
    # MIN_N - 1 spans on one OD with no usable schedule: every level of both
    # chains is thin, so the baseline abstains — and so must every variant,
    # hard-cutoff and shrunk alike.
    train = [_span(seconds=100, scheduled=None, trip=f"t{i}") for i in range(MIN_N - 1)]
    probes = [
        _span(seconds=0, scheduled=None),
        _span(seconds=0, scheduled=100),
        _span(seconds=0, origin="A01N", destination="A02N"),
    ]
    h0 = HistoricalBaseline()
    h0.fit(train)
    for shrink in (False, True):
        for use_pattern in (False, True):
            f = _fine(use_pattern=use_pattern, shrink=shrink)
            f.fit(train)
            for probe in probes:
                assert (f.predict(probe, []) is None) == (h0.predict(probe, []) is None)


def test_answers_wherever_the_baseline_answers() -> None:
    # The baseline reaches MIN_N only at its ratio level; the fine chain must
    # answer the same sample rather than abstain.
    train = [
        _span(
            seconds=200,
            scheduled=100,
            origin=f"Q{i:02d}S",
            destination=f"Q{i + 1:02d}S",
            trip=f"t{i}",
        )
        for i in range(MIN_N)
    ]
    probe = _span(seconds=0, scheduled=80, origin="Q90S", destination="Q91S")
    h0 = HistoricalBaseline()
    h0.fit(train)
    assert h0.predict(probe, []) is not None
    for shrink in (False, True):
        for use_pattern in (False, True):
            f = _fine(use_pattern=use_pattern, shrink=shrink)
            f.fit(train)
            assert f.predict(probe, []) is not None


def test_shrinkage_blends_leaf_and_parent_with_the_expected_weight() -> None:
    # Hour-8 leaf: 5 spans at 100 s. Hour-9 spans push every coarser level to
    # MIN_N with the same mixed distribution, so the cascade reduces to one
    # blend with w = 5/25.
    n_leaf = 5
    leaf_spans = [_span(seconds=100, hour=8, trip=f"a{i}") for i in range(n_leaf)]
    parent_spans = [
        _span(seconds=500, hour=9, trip=f"b{i}") for i in range(MIN_N - n_leaf)
    ]
    f = _fine(shrink=True)
    f.fit(leaf_spans + parent_spans)
    q = f.predict(_span(seconds=0, hour=8), [])
    assert q is not None
    # Reproduce the parent's empirical quantiles from the hard-cutoff
    # forecaster fitted on the same data, then check the convex blend.
    hard = _fine(hour_resolution=4)
    hard.fit(leaf_spans + parent_spans)
    parent_q = hard.predict(_span(seconds=0, hour=8), [])
    assert parent_q is not None
    w = n_leaf / (n_leaf + MIN_N)
    for tau in QUANTILES:
        assert math.isclose(q[tau], w * 100.0 + (1 - w) * parent_q[tau])
    values = [q[tau] for tau in QUANTILES]
    assert values == sorted(values)  # convex blend of monotone vectors


def _emp(xs: list[float]) -> dict[float, float]:
    """The baseline's linear-interpolated empirical quantile rule, restated
    independently as the test's expectation."""
    xs = sorted(xs)
    out: dict[float, float] = {}
    for tau in QUANTILES:
        pos = tau * (len(xs) - 1)
        lo = int(pos)
        hi = min(lo + 1, len(xs) - 1)
        out[tau] = xs[lo] + (pos - lo) * (xs[hi] - xs[lo])
    return out


def test_shrinkage_cascades_through_every_nonempty_level() -> None:
    # Two thin levels stacked on one thick one: pattern leaf n=5 (100 s),
    # hour leaf n=10 (patterns A+B), every coarser level n=20 with one shared
    # distribution. Expected: fold the hour leaf into the pooled estimate
    # with w=10/30, then the pattern leaf with w=5/25.
    spans = (
        [_span(seconds=100, hour=8, trip=f"{i}_Q..S01R") for i in range(5)]
        + [_span(seconds=900, hour=8, trip=f"{i}_Q..S02R") for i in range(5)]
        + [_span(seconds=500, hour=9, trip=f"{i}_Q..S03R") for i in range(10)]
    )
    f = _fine(use_pattern=True, shrink=True)
    f.fit(spans)
    q = f.predict(_span(seconds=0, hour=8, trip="y_Q..S01R"), [])
    assert q is not None
    pooled = _emp([100.0] * 5 + [900.0] * 5 + [500.0] * 10)
    hour = _emp([100.0] * 5 + [900.0] * 5)
    w_hour = 10 / (10 + MIN_N)
    w_leaf = 5 / (5 + MIN_N)
    for tau in QUANTILES:
        running = w_hour * hour[tau] + (1 - w_hour) * pooled[tau]
        assert math.isclose(q[tau], w_leaf * 100.0 + (1 - w_leaf) * running)
    values = [q[tau] for tau in QUANTILES]
    assert values == sorted(values)


def test_quantiles_are_keyed_exactly_by_the_grid() -> None:
    f = _fine()
    f.fit([_span(seconds=100 + i, trip=f"t{i}") for i in range(MIN_N)])
    q = f.predict(_span(seconds=0), [])
    assert q is not None
    assert tuple(q) == QUANTILES
    fs = _fine(shrink=True)
    fs.fit([_span(seconds=100 + i, trip=f"t{i}") for i in range(MIN_N)])
    qs = fs.predict(_span(seconds=0), [])
    assert qs is not None
    assert tuple(qs) == QUANTILES
