"""Finer-grained historical journey-time forecasters.

One configurable class behind five registered variants. All refine
HistoricalBaseline's leaf along three axes and keep its estimator
(`_empirical`), its MIN_N floor, and its two lowest pooling levels, so any
difference in score is the leaf definition, not the machinery:

- Service day type: "Weekday" | "Saturday" | "Sunday", read from the static
  feed calendar for the span's `service_day` (active service_ids start with
  one of those words), NOT the clock date — a holiday running Sunday service
  bins with Sundays. The date -> day-type map is built lazily on first fit
  and cached per forecaster; a date whose active services name no single day
  type falls back to the clock weekday and is counted in
  `n_clock_fallback_days`. Tests inject the map instead of fetching the feed.
- Hour bin: the New York local clock hour of depart_at, either the raw hour
  (0..23) or the four-hour bin shared with the baseline.
- Stopping pattern (optional): `gtfs_static.path_code(trip_id)` as the top
  level of the key.

Fallback chain, finest first, each level requiring MIN_N observations in the
hard-cutoff variants:

  1. (route, direction, origin, destination, hour bin, day type[, pattern])
  2. drop the pattern (when used)
  3. (route, direction, origin, destination, 4h bin, day type)
  4. (route, direction, origin, destination, 4h bin)   -- baseline level 1
  5. (route, direction, origin, destination)           -- baseline level 2
  6. (route, direction, n_hops) ratio-to-schedule scaled by the sample's
     scheduled_sec                                     -- baseline level 3

Levels 4..6 are exactly the baseline's, so no variant abstains on a sample
the baseline answers.

Shrinkage variants replace the hard cutoff with a bottom-up cascade of
quantile-by-quantile blends. Walking the chain coarsest-first, the running
estimate starts at the coarsest level with n >= MIN_N; then EVERY finer
level with n >= 1 is folded in as q = w*q_level + (1-w)*q_running with
w = n/(n + MIN_N), so each nonempty level shrinks toward the pooled estimate
beneath it, weighted by its own evidence. A convex blend of two
non-decreasing quantile vectors is non-decreasing, so the forecast grid
stays monotone at every step. When no level reaches MIN_N the forecaster
abstains — the same abstention set as the hard cutoff and the baseline.

Memory: fit accumulates raw seconds per key in `array("f")`, converts every
table to fixed-grid quantiles at the end, and drops the raw arrays.

These forecasters ignore `history`; `index_history` is a declared no-op so
the gate skips the per-sample trailing-window slice.
"""

from __future__ import annotations

import io
import zipfile
from array import array
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from zoneinfo import ZoneInfo

from training.gtfs_static import Calendar, fetch_gtfs_zip, path_code, read_calendar
from training.journey_baselines import (
    MIN_N,
    _empirical,  # pyright: ignore[reportPrivateUsage] -- shared estimator by design
)
from training.journey_types import QUANTILES, Forecaster, Quantiles, SpanSample

_NY = ZoneInfo("America/New_York")

_DAY_WORDS = ("Weekday", "Saturday", "Sunday")

# Clock-weekday fallback: Mon-Fri -> Weekday, Sat, Sun.
_CLOCK_DAY = ("Weekday",) * 5 + ("Saturday", "Sunday")

# A finalized table cell: observation count and the QUANTILES-aligned values.
_Cell = tuple[int, tuple[float, ...]]


def _finalize(
    raw: dict[tuple[str | int, ...], array[float]],
) -> dict[tuple[str | int, ...], _Cell]:
    """Sorted empirical quantiles per key; the raw arrays are dropped."""
    out: dict[tuple[str | int, ...], _Cell] = {}
    for key, xs in raw.items():
        ordered = sorted(xs)
        q = _empirical(ordered)
        out[key] = (len(ordered), tuple(q[tau] for tau in QUANTILES))
    return out


class FineHistorical:
    """Empirical train-window quantiles on a finer leaf. See module docstring
    for the key axes, the fallback chain, and the shrinkage rule."""

    def __init__(
        self,
        name: str,
        *,
        hour_resolution: int,
        use_pattern: bool,
        shrink: bool,
        day_types: Mapping[str, str] | None = None,
    ) -> None:
        if hour_resolution not in (1, 4):
            raise ValueError(f"hour_resolution must be 1 or 4, got {hour_resolution}")
        self.name = name
        self._hour_resolution = hour_resolution
        self._use_pattern = use_pattern
        self._shrink = shrink
        self._injected = day_types
        self._calendar: Calendar | None = None  # loaded lazily on first use
        self._day_cache: dict[str, str] = {}
        self.n_clock_fallback_days = 0
        self._pattern: dict[tuple[str | int, ...], _Cell] = {}
        self._fine: dict[tuple[str | int, ...], _Cell] = {}
        self._tod4_dt: dict[tuple[str | int, ...], _Cell] = {}
        self._tod4: dict[tuple[str | int, ...], _Cell] = {}
        self._pair: dict[tuple[str | int, ...], _Cell] = {}
        self._ratio: dict[tuple[str | int, ...], _Cell] = {}

    def index_history(self, completed: Sequence[SpanSample]) -> None:
        """Reads no history; declared so the harness skips the per-sample slice."""

    def _day_type(self, service_day: str) -> str:
        cached = self._day_cache.get(service_day)
        if cached is not None:
            return cached
        day_type: str | None = None
        if self._injected is not None:
            day_type = self._injected.get(service_day)
        else:
            if self._calendar is None:
                with zipfile.ZipFile(io.BytesIO(fetch_gtfs_zip())) as zf:
                    self._calendar = read_calendar(zf)
            active = self._calendar.active(date.fromisoformat(service_day))
            words = {w for w in _DAY_WORDS if any(sid.startswith(w) for sid in active)}
            if len(words) == 1:
                day_type = next(iter(words))
        if day_type is None:
            day_type = _CLOCK_DAY[date.fromisoformat(service_day).weekday()]
            self.n_clock_fallback_days += 1
        self._day_cache[service_day] = day_type
        return day_type

    def fit(self, train: Sequence[SpanSample]) -> None:
        pattern: dict[tuple[str | int, ...], array[float]] = {}
        fine: dict[tuple[str | int, ...], array[float]] = {}
        tod4_dt: dict[tuple[str | int, ...], array[float]] = {}
        tod4: dict[tuple[str | int, ...], array[float]] = {}
        pair: dict[tuple[str | int, ...], array[float]] = {}
        ratio: dict[tuple[str | int, ...], array[float]] = {}
        one_hour = self._hour_resolution == 1
        use_pattern = self._use_pattern
        for s in train:
            y = float(s.seconds)
            hour = datetime.fromtimestamp(s.depart_at, tz=_NY).hour
            tod = hour // 4
            day_type = self._day_type(s.service_day)
            base = (s.route_id, s.direction, s.origin, s.destination)
            if use_pattern:
                pattern.setdefault(
                    (*base, hour, day_type, path_code(s.trip_id)), array("f")
                ).append(y)
            if one_hour:
                fine.setdefault((*base, hour, day_type), array("f")).append(y)
            tod4_dt.setdefault((*base, tod, day_type), array("f")).append(y)
            tod4.setdefault((*base, tod), array("f")).append(y)
            pair.setdefault(base, array("f")).append(y)
            if s.scheduled_sec is not None and s.scheduled_sec > 0:
                ratio.setdefault(
                    (s.route_id, s.direction, s.n_hops), array("f")
                ).append(y / s.scheduled_sec)
        self._pattern = _finalize(pattern)
        self._fine = _finalize(fine) if one_hour else {}
        self._tod4_dt = _finalize(tod4_dt)
        self._tod4 = _finalize(tod4)
        self._pair = _finalize(pair)
        self._ratio = _finalize(ratio)

    def _chain(self, sample: SpanSample) -> list[_Cell | None]:
        """The fallback chain's cells, finest first; ratio cells arrive scaled
        to seconds by the sample's scheduled_sec, None when that is None."""
        hour = datetime.fromtimestamp(sample.depart_at, tz=_NY).hour
        tod = hour // 4
        day_type = self._day_type(sample.service_day)
        base = (sample.route_id, sample.direction, sample.origin, sample.destination)
        cells: list[_Cell | None] = []
        if self._use_pattern:
            cells.append(
                self._pattern.get((*base, hour, day_type, path_code(sample.trip_id)))
            )
        if self._hour_resolution == 1:
            cells.append(self._fine.get((*base, hour, day_type)))
        cells.append(self._tod4_dt.get((*base, tod, day_type)))
        cells.append(self._tod4.get((*base, tod)))
        cells.append(self._pair.get(base))
        scaled: _Cell | None = None
        if sample.scheduled_sec is not None:
            cell = self._ratio.get((sample.route_id, sample.direction, sample.n_hops))
            if cell is not None:
                n, values = cell
                scaled = (n, tuple(v * sample.scheduled_sec for v in values))
        cells.append(scaled)
        return cells

    def predict(
        self, sample: SpanSample, history: Sequence[SpanSample]
    ) -> Quantiles | None:
        cells = self._chain(sample)
        if not self._shrink:
            for cell in cells:
                if cell is not None and cell[0] >= MIN_N:
                    return dict(zip(QUANTILES, cell[1], strict=True))
            return None
        running: tuple[float, ...] | None = None
        for cell in reversed(cells):
            if cell is None:
                continue
            n, values = cell
            if running is None:
                if n >= MIN_N:
                    running = values
                continue
            w = n / (n + MIN_N)
            running = tuple(
                w * a + (1.0 - w) * b for a, b in zip(values, running, strict=True)
            )
        if running is None:
            return None
        return dict(zip(QUANTILES, running, strict=True))


def _factory(
    name: str, *, hour_resolution: int, use_pattern: bool, shrink: bool
) -> Callable[[], FineHistorical]:
    def make() -> FineHistorical:
        return FineHistorical(
            name,
            hour_resolution=hour_resolution,
            use_pattern=use_pattern,
            shrink=shrink,
        )

    return make


FORECASTERS: dict[str, Callable[[], Forecaster]] = {
    "fine_4h_daytype": _factory(
        "fine_4h_daytype", hour_resolution=4, use_pattern=False, shrink=False
    ),
    "fine_1h_daytype": _factory(
        "fine_1h_daytype", hour_resolution=1, use_pattern=False, shrink=False
    ),
    "fine_1h_daytype_pattern": _factory(
        "fine_1h_daytype_pattern", hour_resolution=1, use_pattern=True, shrink=False
    ),
    "fine_1h_daytype_shrunk": _factory(
        "fine_1h_daytype_shrunk", hour_resolution=1, use_pattern=False, shrink=True
    ),
    "fine_1h_daytype_pattern_shrunk": _factory(
        "fine_1h_daytype_pattern_shrunk",
        hour_resolution=1,
        use_pattern=True,
        shrink=True,
    ),
}
