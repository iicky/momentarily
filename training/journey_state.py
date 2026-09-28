"""Segment-local state-conditioned journey forecasters.

The route-wide state factor in journey_model applies live evidence at the
wrong resolution: a delay on one branch scales every OD on the route. These
forecasters keep the same historical leaves and pooling (they subclass
StateConditionedForecaster) but compute the state factor from spans that
describe the sample's own segment:

1. Exact-OD spans: completed spans in the trailing window with the same
   (route_id, direction, origin, destination).
2. Containment fallback, only when no exact-OD span is in the window: spans
   whose trip ran the SAME stopping pattern as the sample's trip, and whose
   origin and destination both lie within the sample's origin..destination
   along that pattern.
3. Neither: f = 1.0.

A stopping pattern is the trailing field of the trip_id (the path code).
Its stop order is derived from the TRAIN data at fit time: every 1-hop span
of that pattern contributes an origin -> destination edge, and the order is
used only when those edges form one unbranched chain. A pattern whose edges
branch, merge, or cycle gets no order, so it gets no containment at all:
ambiguity never widens a match. Because the pattern is a single path, a span
on another branch, or on the local track beside an express trip, never
counts as inside the sample's segment.

Three class-attribute switches define the registered variants:

- `_normal_mode`: what "normal" divides a history span's realized seconds.
  "pooled" uses the pooled OD median (parent behaviour); "leaf" uses the
  median of the leaf quantiles the history span itself resolves to, falling
  back to the pooled median, skipping the span when neither exists.
- `_gate`: optional (min_n, threshold) — the factor stays 1.0 unless
  n >= min_n and |mean log ratio| > threshold.
- `_window`: trailing window in seconds (open lower bound, closed upper:
  sample.depart_at - window < completion <= sample.depart_at).

Per-span normals are computed once in `index_history`, not at predict time.
"""

from __future__ import annotations

import math
from array import array
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Callable, Sequence

from training.gtfs_static import path_code
from training.journey_model import StateConditionedForecaster, shrunk
from training.journey_types import HISTORY_SECONDS, Forecaster, SpanSample

# (route_id, direction, origin, destination)
_ODKey = tuple[str, str, str, str]
# (route_id, direction, stopping-pattern code from the trip_id)
_PKey = tuple[str, str, str]


def _pattern_key(s: SpanSample) -> _PKey:
    return (s.route_id, s.direction, path_code(s.trip_id))


def _chain_positions(edges: dict[str, set[str]]) -> dict[str, int] | None:
    """Stop -> position along one stopping pattern, or None when the observed
    edges are not a single unbranched chain (a stop with two successors or two
    predecessors, more than one head, or a cycle). An ambiguous pattern gets no
    order at all, so ambiguity never widens a containment match."""
    preds: dict[str, int] = defaultdict(int)
    for nexts in edges.values():
        if len(nexts) != 1:
            return None
        for b in nexts:
            preds[b] += 1
    if any(n > 1 for n in preds.values()):
        return None
    heads = [a for a in edges if a not in preds]
    if len(heads) != 1:
        return None
    pos: dict[str, int] = {}
    stop: str | None = heads[0]
    while stop is not None:
        if stop in pos:
            return None
        pos[stop] = len(pos)
        nexts = edges.get(stop)
        stop = next(iter(nexts)) if nexts else None
    return pos


class SegmentStateForecaster(StateConditionedForecaster):
    """Historical leaf quantiles times a segment-local live state factor."""

    name = "state_od"

    # What divides a history span's realized seconds: "pooled" | "leaf".
    _normal_mode = "pooled"
    # (min_n, threshold) or None; see the module docstring.
    _gate: tuple[int, float] | None = None
    # Trailing window in seconds.
    _window: int = HISTORY_SECONDS

    _od_index: dict[_ODKey, tuple[array[int], array[float]]] | None = None
    _pattern_index: (
        dict[_PKey, tuple[array[int], list[str], list[str], array[float]]] | None
    ) = None

    _order: dict[_PKey, dict[str, int]]

    def fit(self, train: Sequence[SpanSample]) -> None:
        super().fit(train)
        succ: dict[_PKey, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for s in train:
            if s.n_hops == 1:
                succ[_pattern_key(s)][s.origin].add(s.destination)
        self._order = {}
        for key, edges in succ.items():
            pos = _chain_positions(dict(edges))
            if pos is not None:
                self._order[key] = pos

    def _span_normal(self, h: SpanSample) -> float | None:
        """The normal anchoring `h`'s log ratio, per `_normal_mode`."""
        if self._normal_mode == "leaf":
            cell = self._leaf_quantiles(h)
            if cell is not None:
                return cell[0.50]
        return self._normal.get((h.route_id, h.direction, h.origin, h.destination))

    def index_history(self, completed: Sequence[SpanSample]) -> None:
        """Two indexes over the admitted spans (positive seconds and a usable
        normal), both ascending by completion time: per exact OD, completion
        times beside prefix sums of the log ratios; per stopping pattern,
        parallel columns of completion time, origin, destination, and log
        ratio for the containment fallback. Sorts its input, so any caller
        order gives the same forecasts."""
        od_times: dict[_ODKey, array[int]] = {}
        od_sums: dict[_ODKey, array[float]] = {}
        pattern_index: dict[
            _PKey, tuple[array[int], list[str], list[str], array[float]]
        ] = {}
        for h in sorted(completed, key=lambda s: s.depart_at + s.seconds):
            if h.seconds <= 0:
                continue
            normal = self._span_normal(h)
            if normal is None or normal <= 0:
                continue
            ratio = math.log(h.seconds / normal)
            done = h.depart_at + h.seconds
            od: _ODKey = (h.route_id, h.direction, h.origin, h.destination)
            times = od_times.get(od)
            if times is None:
                times = od_times[od] = array("q")
                od_sums[od] = array("d", (0.0,))
            times.append(done)
            sums = od_sums[od]
            sums.append(sums[-1] + ratio)
            key = _pattern_key(h)
            entry = pattern_index.get(key)
            if entry is None:
                entry = pattern_index[key] = (array("q"), [], [], array("d"))
            entry[0].append(done)
            entry[1].append(h.origin)
            entry[2].append(h.destination)
            entry[3].append(ratio)
        self._od_index = {k: (od_times[k], od_sums[k]) for k in od_times}
        self._pattern_index = pattern_index

    def _factor(self, n: int, mean_log_ratio: float) -> float:
        if self._gate is not None:
            min_n, threshold = self._gate
            if n < min_n or abs(mean_log_ratio) <= threshold:
                return 1.0
        return shrunk(n, mean_log_ratio)

    def _inside(
        self, sample: SpanSample, h_pattern: _PKey, origin: str, destination: str
    ) -> bool:
        """Whether a history span of stopping pattern `h_pattern` covering
        origin..destination lies between the sample's endpoints along the
        sample's own stopping pattern. Different patterns never match."""
        key = _pattern_key(sample)
        if h_pattern != key:
            return False
        pos = self._order.get(key)
        if pos is None:
            return False
        so, sd = pos.get(sample.origin), pos.get(sample.destination)
        ho, hd = pos.get(origin), pos.get(destination)
        if so is None or sd is None or ho is None or hd is None:
            return False
        return so <= ho < hd <= sd

    def _state_factor(self, sample: SpanSample, history: Sequence[SpanSample]) -> float:
        lower = sample.depart_at - self._window
        if self._od_index is not None and self._pattern_index is not None:
            od: _ODKey = (
                sample.route_id,
                sample.direction,
                sample.origin,
                sample.destination,
            )
            exact = self._od_index.get(od)
            if exact is not None:
                times, sums = exact
                lo = bisect_right(times, lower)
                hi = bisect_right(times, sample.depart_at)
                if hi > lo:
                    return self._factor(hi - lo, (sums[hi] - sums[lo]) / (hi - lo))
            key = _pattern_key(sample)
            entry = self._pattern_index.get(key)
            if entry is None:
                return 1.0
            times, origins, dests, ratios = entry
            lo = bisect_right(times, lower)
            hi = bisect_right(times, sample.depart_at)
            n, total = 0, 0.0
            for i in range(lo, hi):
                if self._inside(sample, key, origins[i], dests[i]):
                    n += 1
                    total += ratios[i]
            if n == 0:
                return 1.0
            return self._factor(n, total / n)
        # Per-sample scan over the provided history; same admission and
        # window rule as the indexes.
        exact_ratios: list[float] = []
        contained: list[float] = []
        for h in history:
            done = h.depart_at + h.seconds
            if done > sample.depart_at or done <= lower:
                continue
            if h.route_id != sample.route_id or h.direction != sample.direction:
                continue
            if h.seconds <= 0:
                continue
            normal = self._span_normal(h)
            if normal is None or normal <= 0:
                continue
            ratio = math.log(h.seconds / normal)
            if (h.origin, h.destination) == (sample.origin, sample.destination):
                exact_ratios.append(ratio)
            elif self._inside(sample, _pattern_key(h), h.origin, h.destination):
                contained.append(ratio)
        ratios_used = exact_ratios or contained
        if not ratios_used:
            return 1.0
        return self._factor(len(ratios_used), sum(ratios_used) / len(ratios_used))


class _LeafNormalForecaster(SegmentStateForecaster):
    """Segment-local factor with the leaf-resolved normal."""

    name = "state_od_leaf"
    _normal_mode = "leaf"


class _GatedForecaster(_LeafNormalForecaster):
    """Leaf normal, factor applied only on n >= 3 and |mean log ratio| > 0.05."""

    name = "state_od_gated"
    _gate = (3, 0.05)


class _ShortWindowForecaster(_LeafNormalForecaster):
    """Leaf normal with a 20-minute trailing window."""

    name = "state_od_20m"
    _window = 1200


FORECASTERS: dict[str, Callable[[], Forecaster]] = {
    "state_od": SegmentStateForecaster,
    "state_od_leaf": _LeafNormalForecaster,
    "state_od_gated": _GatedForecaster,
    "state_od_20m": _ShortWindowForecaster,
}
