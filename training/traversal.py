"""Per-segment traversal-time baselines from the reconstructed minute trace.

`training.trace` turns the per-minute vehicle census into per-(trip, hop)
traversals. This module turns those into a per-(route, direction, from_stop,
to_stop) baseline: what this hop normally takes, and how far a live hop sits
from it. That is the segment-level signal the 5-minute advance rate could never
reach, because at 5-minute polling 81.8% of observed moves span two or more
stations.

WHICH CLOCK. The fitted quantity is ARRIVAL-TO-ARRIVAL hop time — from the
train's arrival at from_stop to its arrival at to_stop, station dwell included.
Not travel time. Three reasons, all measured over 2026-08-12 16:30Z..08-13
00:14Z (73,801 exact single hops):

  * Coverage. Only 19,559 of those hops (26.5%) ever caught the train in
    transit and so carry a departure at all. A departure-to-arrival fit throws
    away three quarters of the data.
  * Selection. The hops that do get caught in transit are the slow ones — still
    moving when the next minute's poll lands. That sample is biased toward
    exactly the tail a baseline is meant to detect deviations from.
  * It is the rider's quantity. "How long from this station to the next"
    includes standing at the platform.

The timetable is comparable to it: NYCT sets arrival_time == departure_time on
95.7% of stop_times rows, so gtfs_static.hop_seconds allocates essentially no
dwell, and the observed arrival-to-arrival time runs a few percent over it.

WHAT IS OBSERVABLE. Only EXACT single-hop traversals carry a to_stop, so only
they can be attributed to a hop. A RIGHT-censored traversal was last seen
heading somewhere it never reached — there is no destination to file it under,
and filing it by from_stop alone would pool a hop with whichever successors
happen to share that platform. INTERVAL spans cover several hops and are known
only by their sum. Both are counted in TraversalStats and excluded from the fit.

WHERE A CELL'S LEVEL COMES FROM. A hop with enough observations of its own gets
its own fit. A thin one borrows the population's ratio-to-timetable curve,
rescaled by its own scheduled hop time — the level comes from the timetable, the
shape and the dwell allowance from the population. Pooling raw seconds instead
would put a 60-second hop and a 400-second hop in one distribution, which is why
the ratio is the thing pooled. A thin hop the timetable does not name is omitted
entirely: "can't judge", the same contract compute_baseline and the movement
baseline already use.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import date

from training.dwell import DwellSample
from training.gtfs_static import HopKey, load_hop_seconds
from training.hierarchical import MIN_LEAF_N
from training.survival import fit_loglogistic, loglogistic_quantile
from training.trace import (
    EXACT,
    RIGHT,
    Traversal,
    fetch_trace_bodies,
    traversals_from_trace,
)

# Observations a hop needs before it is fitted on its own rather than borrowing
# the timetable-anchored population curve. Same floor the segment advance-rate
# hierarchy uses to decide a leaf can speak for itself.
MIN_HOP_SAMPLES = MIN_LEAF_N

# Ratio samples are carried as integer parts-per-thousand: DwellSample is
# (int, bool) and the survival fitters are scale-free, so a fixed multiplier
# costs nothing and leaves the sample type alone.
RATIO_SCALE = 1000

# Where a cell's level came from.
OWN = "own"  # the hop's own observations
SCHEDULED = "scheduled"  # population ratio curve x this hop's scheduled time


@dataclass(frozen=True)
class TraversalCell:
    """One hop's normal arrival-to-arrival time.

    `median_sec` is the level a live hop is judged against; `p90_sec` is that
    hop's own idea of slow, which is what keeps a deviation comparable across
    segments of very different lengths.
    """

    n: int  # traversals behind the fit
    median_sec: float
    p90_sec: float
    scheduled_sec: int | None
    source: str


@dataclass(frozen=True)
class TraversalStats:
    """The honest denominator for anything read off the baseline."""

    n_traversals: int
    n_fitted: int  # exact single hops that reached a cell
    n_dropped_right: int  # in progress when last seen: no destination to file under
    n_dropped_interval: int  # several hops known only by their sum
    n_keys: int
    n_cells_own: int
    n_cells_scheduled: int
    # Thin AND absent from the timetable, so neither source can supply a level.
    n_keys_unjudgeable: int
    # Median observed/scheduled over every hop that has both — the standing time
    # the timetable does not allocate, measured rather than assumed.
    population_ratio_median: float | None


def hop_samples(traversals: list[Traversal]) -> dict[HopKey, list[DwellSample]]:
    """Arrival-to-arrival (seconds, completed) per hop key, from the EXACT
    single-hop traversals only — see the module docstring on what the other two
    censoring kinds cannot say about a specific hop.

    `n_hops == 1` is checked rather than inferred from the censoring kind. Today
    trace.py only labels a span EXACT when it covers one hop, but nothing in the
    Traversal type says so, and a span filed under (from_stop, to_stop) that
    actually crossed a station in between would land in a cell as if it were a
    direct hop.
    """
    out: dict[HopKey, list[DwellSample]] = defaultdict(list)
    for t in traversals:
        if t.censoring != EXACT or t.to_stop is None or t.n_hops != 1:
            continue
        out[(t.route_id, t.direction or "", t.from_stop, t.to_stop)].append(
            (t.seconds, True)
        )
    return dict(out)


def _quantiles(shape: float, scale: float) -> tuple[float, float]:
    return (
        loglogistic_quantile(0.5, shape, scale),
        loglogistic_quantile(0.9, shape, scale),
    )


def _population_ratio_fit(
    samples_by_key: Mapping[HopKey, list[DwellSample]],
    scheduled: Mapping[HopKey, int],
) -> tuple[float, float] | None:
    """Log-logistic fit of observed/scheduled over every hop that has both, as
    (shape, scale) in parts-per-thousand of the scheduled time.

    One pooled fit rather than one per route: the ratio is the thing the routes
    have in common, and splitting it per route would hand the thin hops back the
    sparsity the pooling exists to fix."""
    ratios: list[DwellSample] = []
    for key, samples in samples_by_key.items():
        want = scheduled.get(key)
        if not want or want <= 0:
            continue
        ratios.extend(
            (max(round(seconds * RATIO_SCALE / want), 1), completed)
            for seconds, completed in samples
        )
    fit = fit_loglogistic(ratios)
    return None if fit is None else (fit.shape, fit.scale)


def traversal_baseline(
    traversals: list[Traversal],
    scheduled: Mapping[HopKey, int],
    *,
    min_hop_samples: int = MIN_HOP_SAMPLES,
) -> tuple[dict[HopKey, TraversalCell], TraversalStats]:
    """Per-hop arrival-to-arrival baselines over whatever window `traversals`
    covers, plus what had to be discarded to build them."""
    samples_by_key = hop_samples(traversals)
    population = _population_ratio_fit(samples_by_key, scheduled)

    cells: dict[HopKey, TraversalCell] = {}
    ratios: list[float] = []
    n_own = n_scheduled = n_unjudgeable = 0
    for key, samples in samples_by_key.items():
        want = scheduled.get(key)
        if want and want > 0:
            ratios.extend(seconds / want for seconds, _completed in samples)

        fit = fit_loglogistic(samples) if len(samples) >= min_hop_samples else None
        if fit is not None:
            median, p90 = _quantiles(fit.shape, fit.scale)
            cells[key] = TraversalCell(
                n=len(samples),
                median_sec=median,
                p90_sec=p90,
                scheduled_sec=want,
                source=OWN,
            )
            n_own += 1
        elif population is not None and want and want > 0:
            median, p90 = _quantiles(*population)
            cells[key] = TraversalCell(
                n=len(samples),
                median_sec=median * want / RATIO_SCALE,
                p90_sec=p90 * want / RATIO_SCALE,
                scheduled_sec=want,
                source=SCHEDULED,
            )
            n_scheduled += 1
        else:
            n_unjudgeable += 1

    stats = TraversalStats(
        n_traversals=len(traversals),
        n_fitted=sum(len(s) for s in samples_by_key.values()),
        n_dropped_right=sum(1 for t in traversals if t.censoring == RIGHT),
        n_dropped_interval=sum(
            1 for t in traversals if t.censoring not in (EXACT, RIGHT)
        ),
        n_keys=len(samples_by_key),
        n_cells_own=n_own,
        n_cells_scheduled=n_scheduled,
        n_keys_unjudgeable=n_unjudgeable,
        population_ratio_median=(
            round(statistics.median(ratios), 4) if ratios else None
        ),
    )
    return cells, stats


def deviation(cell: TraversalCell, seconds: float) -> float:
    """How far a live hop sits from its own segment's normal: observed over the
    segment's median. 1.0 is exactly normal, 2.0 is twice as long as usual.

    A ratio rather than a difference so a 60-second hop and a 400-second hop are
    on the same scale — the whole reason the baseline is per-segment."""
    return seconds / cell.median_sec


def main(argv: list[str] | None = None) -> int:
    """Fit the baseline over a window and report what it covers."""
    parser = argparse.ArgumentParser(description="Per-segment traversal baseline")
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    args = parser.parse_args(argv)

    bodies = fetch_trace_bodies(start_date=args.start_date, end_date=args.end_date)
    print(f"{len(bodies)} trace snapshots", file=sys.stderr)
    traversals, trace_stats = traversals_from_trace(bodies)
    cells, stats = traversal_baseline(traversals, load_hop_seconds())

    own = [c for c in cells.values() if c.source == OWN and c.scheduled_sec]
    slowest = sorted(own, key=lambda c: c.median_sec / (c.scheduled_sec or 1))[-5:]
    print(
        json.dumps(
            {
                "trace": asdict(trace_stats),
                "baseline": asdict(stats),
                "own_cells": {
                    "median_sec": round(
                        statistics.median([c.median_sec for c in own]), 1
                    ),
                    "median_spread_p90_over_median": round(
                        statistics.median([c.p90_sec / c.median_sec for c in own]), 3
                    ),
                    "median_ratio_to_schedule": round(
                        statistics.median(
                            [c.median_sec / (c.scheduled_sec or 1) for c in own]
                        ),
                        3,
                    ),
                }
                if own
                else None,
                "slowest_vs_schedule": [
                    {
                        "median_sec": round(c.median_sec, 1),
                        "scheduled_sec": c.scheduled_sec,
                        "ratio": round(c.median_sec / (c.scheduled_sec or 1), 2),
                        "n": c.n,
                    }
                    for c in reversed(slowest)
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
