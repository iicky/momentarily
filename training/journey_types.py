"""Shared types for the journey-time experiment.

A SpanSample is one realized rider journey on one train: the trip's arrival at
`origin` to its arrival at `destination`, chained from consecutive traversal
rows (training.trace.Traversal) of the same trip. `depart_at` is Traversal.at
of the first hop — the arrival at origin, which is when a rider standing there
would board — so `seconds` includes the dwell at origin, matching what the
traversal baselines already fit.

Causality contract for Forecaster.predict: `history` holds only spans that
COMPLETED (depart_at + seconds <= sample.depart_at) within the trailing 3600 s,
across all routes, in ascending completion order. A forecaster may read
nothing else about the present. Returning None means abstain; the harness
reports the abstention rate and scores only answered samples, on the SAME
samples for every forecaster.

JSONL layout (journey.py writes, journey_gate.py reads): one JSON object per
line with exactly the SpanSample field names; `.gz` suffix means gzip.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

# Fixed forecast grid. A Quantiles mapping has exactly these keys, values in
# seconds, non-decreasing in q.
QUANTILES: tuple[float, ...] = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

# Longest span (in realtime hops) the experiment scores.
MAX_HOPS = 8

# Train/holdout split by service_day (inclusive). Both windows are closed days.
# 08-12..08-26 ran under a different static feed version and are excluded rather than scored against a schedule that was not in force.
TRAIN_START = "2026-08-27"
TRAIN_END = "2026-09-13"
HOLDOUT_START = "2026-09-14"
HOLDOUT_END = "2026-09-20"

# Deterministic holdout subsample: score a trip when
# int(hashlib.sha1(trip_id.encode()).hexdigest(), 16) % HOLDOUT_TRIP_MODULUS == 0.
HOLDOUT_TRIP_MODULUS = 10

# Trailing window of completed spans a forecaster may see.
HISTORY_SECONDS = 3600


@dataclass(frozen=True, slots=True)
class SpanSample:
    trip_id: str
    route_id: str
    direction: str  # "N" | "S" as the trace reports it
    origin: str  # GTFS stop id incl. direction suffix, e.g. "Q05S"
    destination: str
    depart_at: int  # epoch s, arrival of the trip at origin
    seconds: int  # realized arrival(destination) - depart_at
    n_hops: int  # realtime hops covered, 1..MAX_HOPS
    scheduled_sec: (
        int | None
    )  # sum of scheduled hop seconds along the trip's observed hops; None if any hop unknown
    service_day: str  # YYYY-MM-DD the span's first hop belongs to


Quantiles = Mapping[float, float]


class Forecaster(Protocol):
    name: str

    def fit(self, train: Sequence[SpanSample]) -> None: ...

    def predict(
        self, sample: SpanSample, history: Sequence[SpanSample]
    ) -> Quantiles | None: ...


class IndexedForecaster(Forecaster, Protocol):
    """A Forecaster that indexes the history once instead of scanning a
    trailing-hour slice per sample. The harness calls `index_history` with
    every span that may appear in any sample's history, ascending by
    completion (depart_at + seconds), and then passes an EMPTY `history` to
    `predict`; the forecaster applies the contract's window rule itself:
    sample.depart_at - HISTORY_SECONDS < completion <= sample.depart_at.
    Same observable inputs as the scan, without the per-sample slice."""

    def index_history(self, completed: Sequence[SpanSample]) -> None: ...
