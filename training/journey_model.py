"""Causal state-conditioned journey-time forecasters.

Measurement contract: fit() sees only training-window SpanSamples; predict()
sees the sample's identifying fields (never `sample.seconds`) plus a `history`
of spans that COMPLETED before `sample.depart_at`. Completion is re-checked
here (depart_at + seconds <= sample.depart_at, within the trailing
HISTORY_SECONDS) so a prediction cannot leak from the future even if a caller
hands over an unfiltered history.

Two registered forecasters:

* "state" — per-leaf empirical quantiles keyed by (route_id, direction,
  origin, destination, tod_bin, weekend), with pooled fallbacks (drop weekend,
  then tod_bin, then fall to a (route_id, direction, n_hops) ratio-to-schedule
  distribution), multiplied by a live state factor
  f = exp(n / (n + K) * mean(log(seconds / normal))) over same-route+direction
  spans in the trailing hour, clamped to [0.5, 3.0]. f -> 1 when support is
  thin, so the model degrades to the historical leaf, never past it.
* "state_hist_only" — identical leaves, f forced to 1. The gate diffs the two
  to isolate the value of the state term; their abstention sets are identical
  by construction (abstention depends only on the fitted leaves).

Both abstain (return None) when no fallback level has MIN_N support — the
harness scores only answered samples.

Time-of-day bins are six four-hour America/New_York blocks (local hour // 4);
weekend is Sat/Sun of the local date — the same axes the historical journey
baseline cuts on.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import datetime
from zoneinfo import ZoneInfo

from training.eval_common import nearest_rank
from training.journey_types import (
    HISTORY_SECONDS,
    QUANTILES,
    Forecaster,
    Quantiles,
    SpanSample,
)

# Observations a leaf (or pooled fallback cell) needs before it is trusted.
MIN_N = 20

# Shrinkage constant for the state factor: with n supporting spans the log
# state estimate is weighted n / (n + K), so one noisy span moves f little.
K = 5

# The state factor never scales a leaf below half or above triple its
# historical quantiles — beyond that the leaf shape itself is wrong and a
# multiplicative correction would just be confidently wrong.
F_MIN = 0.5
F_MAX = 3.0

_ET = ZoneInfo("America/New_York")

# (route_id, direction, origin, destination)
_ODKey = tuple[str, str, str, str]
# (route_id, direction, origin, destination, tod_bin, weekend)
_LeafKey = tuple[str, str, str, str, int, bool]
# (route_id, direction, origin, destination, tod_bin)
_ODTodKey = tuple[str, str, str, str, int]
# (route_id, direction, n_hops)
_ShapeKey = tuple[str, str, int]


def _tod_bin(epoch_seconds: int) -> int:
    """Which of the six four-hour ET blocks `epoch_seconds` falls in (0..5)."""
    return datetime.fromtimestamp(epoch_seconds, tz=_ET).hour // 4


def _weekend(epoch_seconds: int) -> bool:
    """Whether the ET local date is a Saturday or Sunday."""
    return datetime.fromtimestamp(epoch_seconds, tz=_ET).weekday() >= 5


def _empirical(values: list[float]) -> dict[float, float]:
    """Nearest-rank quantiles on the fixed grid; non-decreasing in q because
    the input is sorted and nearest-rank indexes are monotone in q."""
    values.sort()
    return {q: nearest_rank(values, q) for q in QUANTILES}


def shrunk(n: int, mean_log_ratio: float) -> float:
    """The state factor: mean log ratio weighted n / (n + K), clamped."""
    f = math.exp(n / (n + K) * mean_log_ratio)
    return min(F_MAX, max(F_MIN, f))


class StateConditionedForecaster:
    """Historical leaf quantiles times a live same-route state factor."""

    name = "state"

    # Forces f = 1: the "state_hist_only" ablation flips this in a subclass.
    _use_state = True

    _index: dict[tuple[str, str], tuple[list[int], list[float]]] | None = None
    _leaf: dict[_LeafKey, dict[float, float]]
    _od_tod: dict[_ODTodKey, dict[float, float]]
    _od: dict[_ODKey, dict[float, float]]
    _shape: dict[_ShapeKey, dict[float, float]]
    _normal: dict[_ODKey, float]

    def fit(self, train: Sequence[SpanSample]) -> None:
        by_leaf: dict[_LeafKey, list[float]] = defaultdict(list)
        by_od_tod: dict[_ODTodKey, list[float]] = defaultdict(list)
        by_od: dict[_ODKey, list[float]] = defaultdict(list)
        by_shape: dict[tuple[str, str, int], list[float]] = defaultdict(list)
        for s in train:
            od: _ODKey = (s.route_id, s.direction, s.origin, s.destination)
            tod = _tod_bin(s.depart_at)
            by_leaf[(*od, tod, _weekend(s.depart_at))].append(float(s.seconds))
            by_od_tod[(*od, tod)].append(float(s.seconds))
            by_od[od].append(float(s.seconds))
            if s.scheduled_sec is not None and s.scheduled_sec > 0:
                by_shape[(s.route_id, s.direction, s.n_hops)].append(
                    s.seconds / s.scheduled_sec
                )

        self._leaf = {k: _empirical(v) for k, v in by_leaf.items() if len(v) >= MIN_N}
        self._od_tod = {
            k: _empirical(v) for k, v in by_od_tod.items() if len(v) >= MIN_N
        }
        self._od = {k: _empirical(v) for k, v in by_od.items() if len(v) >= MIN_N}
        self._shape = {k: _empirical(v) for k, v in by_shape.items() if len(v) >= MIN_N}
        # "Normal" per OD for the state term: the pooled OD median, kept even
        # for thin ODs — a normal only anchors a ratio, it is never published.
        self._normal = {k: nearest_rank(sorted(v), 0.50) for k, v in by_od.items() if v}

    def normal(self, od: _ODKey) -> float | None:
        """Public read-only accessor for the pooled OD median anchoring the
        state factor's log ratio — for callers (tests) that need to reason
        about a "normal" without reaching into a private attribute."""
        return self._normal.get(od)

    def _leaf_quantiles(self, sample: SpanSample) -> Quantiles | None:
        """Fallback chain: full leaf, drop weekend, drop tod_bin, then the
        (route, direction, n_hops) ratio-to-schedule shape scaled by the
        sample's own scheduled_sec. None when nothing has support."""
        od: _ODKey = (
            sample.route_id,
            sample.direction,
            sample.origin,
            sample.destination,
        )
        tod = _tod_bin(sample.depart_at)
        cell = self._leaf.get((*od, tod, _weekend(sample.depart_at)))
        if cell is None:
            cell = self._od_tod.get((*od, tod))
        if cell is None:
            cell = self._od.get(od)
        if cell is not None:
            return cell
        if sample.scheduled_sec is None or sample.scheduled_sec <= 0:
            return None
        shape = self._shape.get((sample.route_id, sample.direction, sample.n_hops))
        if shape is None:
            return None
        return {q: r * sample.scheduled_sec for q, r in shape.items()}

    def index_history(self, completed: Sequence[SpanSample]) -> None:
        """Per (route, direction): completion times ascending, beside prefix
        sums of the log ratios and their counts, so a window query is two
        bisects. Same admission rule as `_state_factor`'s scan: a span needs a
        stored normal and a positive realized time. Sorts its input, so any
        caller order gives the same forecasts."""
        times: dict[tuple[str, str], list[int]] = defaultdict(list)
        sums: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0])
        for h in sorted(completed, key=lambda s: s.depart_at + s.seconds):
            normal = self._normal.get(
                (h.route_id, h.direction, h.origin, h.destination)
            )
            if normal is None or normal <= 0 or h.seconds <= 0:
                continue
            key = (h.route_id, h.direction)
            times[key].append(h.depart_at + h.seconds)
            sums[key].append(sums[key][-1] + math.log(h.seconds / normal))
        self._index = {k: (times[k], sums[k]) for k in times}

    def _state_factor(self, sample: SpanSample, history: Sequence[SpanSample]) -> float:
        if self._index is not None:
            entry = self._index.get((sample.route_id, sample.direction))
            if entry is None:
                return 1.0
            times, sums = entry
            lo = bisect_right(times, sample.depart_at - HISTORY_SECONDS)
            hi = bisect_right(times, sample.depart_at)
            n = hi - lo
            if n == 0:
                return 1.0
            return shrunk(n, (sums[hi] - sums[lo]) / n)
        log_ratios: list[float] = []
        for h in history:
            done = h.depart_at + h.seconds
            if done > sample.depart_at or done <= sample.depart_at - HISTORY_SECONDS:
                continue  # not completed yet, or outside the trailing window
            if h.route_id != sample.route_id or h.direction != sample.direction:
                continue
            normal = self._normal.get(
                (h.route_id, h.direction, h.origin, h.destination)
            )
            if normal is None or normal <= 0 or h.seconds <= 0:
                continue
            log_ratios.append(math.log(h.seconds / normal))
        n = len(log_ratios)
        if n == 0:
            return 1.0
        return shrunk(n, sum(log_ratios) / n)

    def predict(
        self, sample: SpanSample, history: Sequence[SpanSample]
    ) -> Quantiles | None:
        cell = self._leaf_quantiles(sample)
        if cell is None:
            return None
        f = self._state_factor(sample, history) if self._use_state else 1.0
        return {q: v * f for q, v in cell.items()}


class _HistOnlyForecaster(StateConditionedForecaster):
    """The same leaves with the state factor forced to 1 — the ablation the
    gate diffs against "state". Abstention sets are identical because
    abstention is decided by the leaves alone."""

    name = "state_hist_only"
    _use_state = False


FORECASTERS: dict[str, Callable[[], Forecaster]] = {
    "state": StateConditionedForecaster,
    "state_hist_only": _HistOnlyForecaster,
}
