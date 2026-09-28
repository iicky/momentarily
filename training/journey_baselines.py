"""The two baselines the journey-time gate must beat.

Measurement contract: both are Forecasters over journey_types.SpanSample.
ScheduleBaseline is the null model — the GTFS timetable as a point forecast,
every quantile equal to scheduled_sec, so its CRPS is its MAE by construction
(see journey_score.crps_from_quantiles). HistoricalBaseline is the
unconditional empirical distribution of realized seconds from the TRAIN window
only, keyed by (route, direction, origin, destination, four-hour local
time-of-day bin, weekend), pooling down when a leaf is thin: drop weekend,
then drop tod_bin, then fall back to a per-(route, direction, n_hops) RATIO to
scheduled_sec. Bin convention shared with the state model: tod_bin is
New York local hour // 4 of depart_at (edges 00/04/08/12/16/20) and weekend is
Sat/Sun of the LOCAL DATE of depart_at, not the service day — a post-midnight
span bins with its clock date. Below MIN_N observations at every level it
abstains — a thin cell reads "can't judge", the same floor contract the
traversal baselines use.
Neither baseline reads `history`; both are frozen at fit time.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from zoneinfo import ZoneInfo

from training.journey_types import QUANTILES, Quantiles, SpanSample

# Observations a key needs before its quantiles are trusted, at every pooling
# level. Under it the baseline abstains rather than guess.
MIN_N = 20

_NY = ZoneInfo("America/New_York")


def tod_bin(depart_at: int) -> int:
    """Four-hour bin (0..5) of the local New York clock hour of depart_at."""
    return datetime.fromtimestamp(depart_at, tz=_NY).hour // 4


def is_weekend(depart_at: int) -> bool:
    """Whether the local New York date of depart_at is a Saturday or Sunday."""
    return datetime.fromtimestamp(depart_at, tz=_NY).weekday() >= 5


def _empirical(sorted_xs: Sequence[float]) -> dict[float, float]:
    """Linear-interpolated empirical quantiles of an ascending sequence."""
    n = len(sorted_xs)
    out: dict[float, float] = {}
    for tau in QUANTILES:
        pos = tau * (n - 1)
        lo = int(pos)
        frac = pos - lo
        hi = min(lo + 1, n - 1)
        out[tau] = sorted_xs[lo] + frac * (sorted_xs[hi] - sorted_xs[lo])
    return out


class ScheduleBaseline:
    """The GTFS timetable as a degenerate point forecast."""

    name = "schedule"

    def fit(self, train: Sequence[SpanSample]) -> None:
        pass

    def index_history(self, completed: Sequence[SpanSample]) -> None:
        """Reads no history; declared so the harness skips the per-sample slice."""

    def predict(
        self, sample: SpanSample, history: Sequence[SpanSample]
    ) -> Quantiles | None:
        if sample.scheduled_sec is None:
            return None
        v = float(sample.scheduled_sec)
        return dict.fromkeys(QUANTILES, v)


class HistoricalBaseline:
    """Empirical train-window quantiles with hierarchical pooling.

    Levels, tried in order, each requiring MIN_N observations:
      0. (route, direction, origin, destination, tod_bin, weekend)
      1. (route, direction, origin, destination, tod_bin)
      2. (route, direction, origin, destination)
      3. (route, direction, n_hops) as quantiles of seconds/scheduled_sec,
         scaled by the sample's scheduled_sec — abstains when that is None.
    """

    name = "historical"

    def __init__(self) -> None:
        self._leaf: dict[tuple[str, str, str, str, int, bool], list[float]] = {}
        self._no_weekend: dict[tuple[str, str, str, str, int], list[float]] = {}
        self._pair: dict[tuple[str, str, str, str], list[float]] = {}
        self._ratio: dict[tuple[str, str, int], list[float]] = {}

    def index_history(self, completed: Sequence[SpanSample]) -> None:
        """Reads no history; declared so the harness skips the per-sample slice."""

    def fit(self, train: Sequence[SpanSample]) -> None:
        leaf: dict[tuple[str, str, str, str, int, bool], list[float]] = {}
        no_weekend: dict[tuple[str, str, str, str, int], list[float]] = {}
        pair: dict[tuple[str, str, str, str], list[float]] = {}
        ratio: dict[tuple[str, str, int], list[float]] = {}
        for s in train:
            y = float(s.seconds)
            tod = tod_bin(s.depart_at)
            leaf.setdefault(
                (
                    s.route_id,
                    s.direction,
                    s.origin,
                    s.destination,
                    tod,
                    is_weekend(s.depart_at),
                ),
                [],
            ).append(y)
            no_weekend.setdefault(
                (s.route_id, s.direction, s.origin, s.destination, tod), []
            ).append(y)
            pair.setdefault(
                (s.route_id, s.direction, s.origin, s.destination), []
            ).append(y)
            if s.scheduled_sec is not None and s.scheduled_sec > 0:
                ratio.setdefault((s.route_id, s.direction, s.n_hops), []).append(
                    y / s.scheduled_sec
                )
        for table in (leaf, no_weekend, pair, ratio):
            for xs in table.values():
                xs.sort()
        self._leaf = leaf
        self._no_weekend = no_weekend
        self._pair = pair
        self._ratio = ratio

    def predict(
        self, sample: SpanSample, history: Sequence[SpanSample]
    ) -> Quantiles | None:
        tod = tod_bin(sample.depart_at)
        key = (sample.route_id, sample.direction, sample.origin, sample.destination)
        xs = self._leaf.get((*key, tod, is_weekend(sample.depart_at)))
        if xs is None or len(xs) < MIN_N:
            xs = self._no_weekend.get((*key, tod))
        if xs is None or len(xs) < MIN_N:
            xs = self._pair.get(key)
        if xs is not None and len(xs) >= MIN_N:
            return _empirical(xs)
        if sample.scheduled_sec is None:
            return None
        ratios = self._ratio.get((sample.route_id, sample.direction, sample.n_hops))
        if ratios is None or len(ratios) < MIN_N:
            return None
        return {tau: r * sample.scheduled_sec for tau, r in _empirical(ratios).items()}
