"""Journey-time scoring (training/journey_score.py).

Synthetic quantile grids against known outcomes — each case pins one property
the gate leans on: the point-forecast/MAE identity, sharpness ordering under
CRPS, PIT calibration at the knots, and abstention bookkeeping.
"""

from __future__ import annotations

import math

from training.journey_score import Report, crps_from_quantiles, pit, score
from training.journey_types import QUANTILES, SpanSample


def _grid(values: list[float]) -> dict[float, float]:
    return dict(zip(QUANTILES, values, strict=True))


def _point(v: float) -> dict[float, float]:
    return dict.fromkeys(QUANTILES, v)


def _sample(seconds: int, trip: str = "t1") -> SpanSample:
    return SpanSample(
        trip_id=trip,
        route_id="Q",
        direction="S",
        origin="Q05S",
        destination="Q06S",
        depart_at=1_760_000_000,
        seconds=seconds,
        n_hops=1,
        scheduled_sec=90,
        service_day="2026-09-15",
    )


def test_point_forecast_crps_is_absolute_error() -> None:
    assert crps_from_quantiles(_point(100.0), 137.0) == 37.0
    assert crps_from_quantiles(_point(100.0), 60.0) == 40.0
    assert crps_from_quantiles(_point(100.0), 100.0) == 0.0


def test_sharper_interval_around_truth_scores_better() -> None:
    y = 100.0
    narrow = _grid([90, 92, 96, 100, 104, 108, 110])
    wide = _grid([40, 55, 80, 100, 120, 145, 160])
    assert crps_from_quantiles(narrow, y) < crps_from_quantiles(wide, y)


def test_pit_hits_the_knots_and_clamps() -> None:
    q = _grid([50, 60, 80, 100, 120, 140, 150])
    assert pit(q, 100.0) == 0.5
    assert pit(q, 49.0) == 0.0
    assert pit(q, 151.0) == 1.0
    assert pit(q, 50.0) == 0.05
    assert pit(q, 150.0) == 0.95
    # Halfway between the q25 and q50 values interpolates halfway in tau.
    assert math.isclose(pit(q, 90.0), 0.375)


def test_pit_of_point_forecast_at_its_value_is_half() -> None:
    assert pit(_point(100.0), 100.0) == 0.5


def test_score_aggregates_and_counts_abstentions() -> None:
    q = _grid([50, 60, 80, 100, 120, 140, 150])
    pairs = [
        (_sample(100), q),  # inside [q10, q90], |y - q50| = 0
        (_sample(200), q),  # above q90 and q95
        (_sample(70), None),  # abstained: must not touch any mean
    ]
    r = score("x", pairs)
    assert r.n_scored == 2
    assert r.n_abstained == 1
    assert r.mae_median == 50.0  # (0 + 100) / 2
    assert r.coverage_10_90 == 0.5
    assert sum(r.pit_bins) == 2
    assert r.pit_bins[9] == 1  # pit(200) == 1.0 folds into the last bin
    assert r.pit_bins[5] == 1  # pit(100) == 0.5


def test_skill_vs_is_relative_crps_improvement() -> None:
    def report(crps: float) -> Report:
        return Report(
            name="r",
            n_scored=1,
            n_abstained=0,
            crps_mean=crps,
            mae_median=crps,
            coverage_10_90=1.0,
            pit_bins=(0,) * 10,
        )

    assert math.isclose(report(75.0).skill_vs(report(100.0)), 0.25)
    assert report(100.0).skill_vs(report(100.0)) == 0.0
    assert report(120.0).skill_vs(report(100.0)) < 0.0
