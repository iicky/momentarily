"""Journey-time baselines (training/journey_baselines.py).

Synthetic train spans, no R2. Each case pins one rule: the schedule point
forecast and its abstention, and the historical baseline's pooling ladder —
leaf, drop-weekend, drop-tod, ratio-to-schedule — with the MIN_N floor at
every rung.
"""

from __future__ import annotations

import math
from datetime import date as date_t
from datetime import datetime, time
from zoneinfo import ZoneInfo

from training.journey_baselines import (
    MIN_N,
    HistoricalBaseline,
    ScheduleBaseline,
    is_weekend,
    tod_bin,
)
from training.journey_types import QUANTILES, SpanSample

NY = ZoneInfo("America/New_York")
WEEKDAY = date_t(2026, 8, 12)  # Wednesday
SATURDAY = date_t(2026, 8, 15)


def _at(day: date_t, hour: int) -> int:
    return int(datetime.combine(day, time(hour), tzinfo=NY).timestamp())


def _span(
    *,
    seconds: int,
    day: date_t = WEEKDAY,
    hour: int = 12,
    origin: str = "Q05S",
    destination: str = "Q06S",
    scheduled: int | None = 100,
    n_hops: int = 1,
    trip: str = "t",
) -> SpanSample:
    return SpanSample(
        trip_id=trip,
        route_id="Q",
        direction="S",
        origin=origin,
        destination=destination,
        depart_at=_at(day, hour),
        seconds=seconds,
        n_hops=n_hops,
        scheduled_sec=scheduled,
        service_day=day.isoformat(),
    )


def test_tod_bin_and_weekend_read_local_new_york_time_of_depart() -> None:
    assert tod_bin(_at(WEEKDAY, 0)) == 0
    assert tod_bin(_at(WEEKDAY, 12)) == 3
    assert tod_bin(_at(WEEKDAY, 23)) == 5
    assert not is_weekend(_at(WEEKDAY, 12))
    assert is_weekend(_at(SATURDAY, 12))
    assert is_weekend(_at(date_t(2026, 8, 16), 12))  # Sunday
    # Shared bin convention: weekend follows the LOCAL DATE of depart_at,
    # not the service day — a post-midnight Saturday depart is weekend even
    # when its service_day is Friday.
    friday_night_service = _at(SATURDAY, 1)  # 01:00 NY Saturday
    assert is_weekend(friday_night_service)


def test_schedule_baseline_is_a_point_forecast_and_abstains_without_one() -> None:
    b = ScheduleBaseline()
    b.fit([])
    q = b.predict(_span(seconds=1, scheduled=240), [])
    assert q is not None
    assert set(q) == set(QUANTILES)
    assert all(v == 240.0 for v in q.values())
    assert b.predict(_span(seconds=1, scheduled=None), []) is None


def test_historical_uses_the_leaf_when_it_is_thick() -> None:
    train = [_span(seconds=100 + i, trip=f"t{i}") for i in range(MIN_N)]
    b = HistoricalBaseline()
    b.fit(train)
    q = b.predict(_span(seconds=0), [])
    assert q is not None
    assert q[0.05] >= 100.0
    assert q[0.95] <= 100.0 + MIN_N - 1
    assert [q[tau] for tau in QUANTILES] == sorted(q[tau] for tau in QUANTILES)


def test_historical_pools_across_weekend_when_the_leaf_is_thin() -> None:
    half = MIN_N // 2
    train = [_span(seconds=100, day=WEEKDAY, trip=f"w{i}") for i in range(half)] + [
        _span(seconds=300, day=SATURDAY, trip=f"s{i}") for i in range(half)
    ]
    b = HistoricalBaseline()
    b.fit(train)
    q = b.predict(_span(seconds=0, day=WEEKDAY), [])
    assert q is not None
    # The pooled cell mixes both day kinds: the median sits between them.
    assert 100.0 < q[0.50] < 300.0


def test_historical_pools_across_tod_then_falls_to_ratio() -> None:
    # MIN_N same-pair spans spread over hours 6..17 (bins 1..4): every
    # (pair, tod) cell is thin, the pair cell is thick.
    train = [_span(seconds=200, hour=6 + (i % 12), trip=f"t{i}") for i in range(MIN_N)]
    b = HistoricalBaseline()
    b.fit(train)
    q = b.predict(_span(seconds=0, hour=8), [])
    assert q is not None
    assert q[0.50] == 200.0

    # A DIFFERENT pair on the same route/direction/n_hops: only the ratio
    # level can carry it. Train ratio is 200/100 = 2.0, so a 500 s schedule
    # forecasts 1000 s.
    other = _span(seconds=0, origin="Q07S", destination="Q08S", scheduled=500)
    q2 = b.predict(other, [])
    assert q2 is not None
    assert math.isclose(q2[0.50], 1000.0)
    # Without a schedule the ratio level cannot scale: abstain.
    assert (
        b.predict(
            _span(seconds=0, origin="Q07S", destination="Q08S", scheduled=None), []
        )
        is None
    )


def test_historical_abstains_when_every_level_is_thin() -> None:
    b = HistoricalBaseline()
    b.fit([_span(seconds=100, trip=f"t{i}") for i in range(MIN_N - 1)])
    assert b.predict(_span(seconds=0), []) is None
