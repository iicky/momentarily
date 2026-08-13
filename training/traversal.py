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

WHERE A CELL'S LEVEL COMES FROM: MOVEMENT, AND ONLY MOVEMENT. Every level here
is fitted from that segment's own observed traversals. A hop under
MIN_HOP_SAMPLES gets no cell and the caller reads "can't judge" — the same
contract compute_baseline and the movement baseline already use. It does NOT
borrow a level from the timetable, and that is a deliberate constraint rather
than an oversight: see below.

WHAT THE TIMETABLE IS FOR, THEN. It is the instrument, not an input. A baseline
fitted on the vehicle archive and graded against the vehicle archive cannot see
its own drift — a system slow for a month teaches that to the baseline as
normal. `schedule_drift` is the one claim in this module the archive does not
get a vote on, and it only means anything because nothing here was fitted from
the schedule. gtfs_static.Timetable is also read for two jobs no amount of
movement data can do: naming the scheduled seconds a cell is later compared
against, and catching bypasses. It carries its own feed version, and a
traversal outside that feed's validity window is compared against nothing.

That reference took three fixes to be worth trusting: pooling weekday and
weekend run times gets 26% of observed hops wrong by more than 10%, a
departure-to-arrival clock another 4.9%, and the route's modal chain credits a
bypassing train with the time of every stop it skipped, so it reads as running
fast exactly when service is worst.

WHAT IS OBSERVABLE, decided from the realtime trace alone. Only EXACT
single-hop traversals carry a to_stop, so only they can be attributed to a hop.
A RIGHT-censored traversal was last seen heading somewhere it never reached —
there is no destination to file it under, and filing it by from_stop alone would
pool a hop with whichever successors happen to share that platform. INTERVAL
spans cover several hops and are known only by their sum. Both are counted in
TraversalStats and excluded.

A BYPASS — one realtime hop with a station inside it per the trip's own
timetable — is NOT excluded. The train covered that stretch without stopping,
which is the same thing an express does on the same pair, so it is a real
measurement of it. It is only counted. Dropping it would have let the schedule
choose the training set, and would have thrown out movement that clusters on
bad days, teaching the baseline a cleaner normal than the one riders got.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import date

from training.gtfs_static import HopKey, Timetable, load_timetable
from training.hierarchical import MIN_LEAF_N
from training.survival import fit_loglogistic, loglogistic_quantile
from training.trace import (
    EXACT,
    RIGHT,
    Traversal,
    fetch_trace_bodies,
    scheduled_for,
    traversals_from_trace,
)

# Observations a hop needs before it can be fitted at all. Under it the cell is
# absent and the caller reads "can't judge" — the same floor and the same
# contract the segment advance-rate hierarchy uses.
MIN_HOP_SAMPLES = MIN_LEAF_N


@dataclass(frozen=True)
class TraversalCell:
    """One hop's normal arrival-to-arrival time, fitted from that hop's own
    observed traversals and nothing else.

    `median_sec` is the level a live hop is judged against; `p90_sec` is that
    hop's own idea of slow, which is what keeps a deviation comparable across
    segments of very different lengths.

    `scheduled_sec` is carried but NEVER fitted from — see `schedule_drift`. The
    timetable is the instrument this baseline is measured with, deliberately not
    an input to it: a reference that fed the fit could not then detect the fit
    drifting.
    """

    n: int  # traversals behind the fit
    median_sec: float
    p90_sec: float
    scheduled_sec: int | None


@dataclass(frozen=True)
class TraversalStats:
    """The honest denominator for anything read off the baseline."""

    n_traversals: int
    n_fitted: int  # exact single hops that reached a cell
    n_dropped_right: int  # in progress when last seen: no destination to file under
    n_dropped_interval: int  # several hops known only by their sum
    # Reported as one hop by the feed, but the trip's own timetable puts a
    # station inside: the train skipped a stop. Counted, not excluded — it
    # really did run that stretch nonstop.
    n_bypass: int
    # Observations whose service day the loaded feed does not claim to describe,
    # so nothing was compared against a schedule that was not in force. They
    # still fit their cell; they just carry no scheduled time.
    n_outside_feed_window: int
    n_keys: int
    n_cells: int
    # Seen, but under MIN_HOP_SAMPLES, so the segment cannot speak for itself
    # yet. It abstains rather than borrowing a level from anywhere.
    n_keys_thin: int


@dataclass(frozen=True)
class HopSample:
    """One traversal attributed to a hop, beside what the timetable allowed it.

    `scheduled_sec` is None when neither the trip's own pattern nor the service
    day's median for the pair reaches it.
    """

    seconds: int
    scheduled_sec: int | None


@dataclass(frozen=True)
class HopSamples:
    """Per-hop observations, and what the timetable had to say about them.

    Both counts are REPORTS. Nothing here decides which movement enters a fit:
    that is settled by the realtime trace alone, so the timetable can still be
    used to measure the result.
    """

    by_key: dict[HopKey, list[HopSample]]
    # Reported as one hop by the feed, but the trip's own timetable puts a
    # station inside: the train skipped a stop. Kept in the fit — it physically
    # ran that stretch without stopping, the same thing an express does on the
    # same pair — and counted here because bypasses cluster on bad days, so
    # excluding them would quietly teach the baseline a cleaner normal than the
    # one riders got.
    n_bypass: int
    # Observations on a service day the loaded feed does not claim to describe.
    # The vehicle archive reaches back further than any one GTFS snapshot, so a
    # replay can land outside it. Those carry no scheduled time at all rather
    # than a wrong one, which keeps them out of the drift measure by
    # construction.
    n_outside_feed_window: int


def hop_samples(traversals: list[Traversal], timetable: Timetable) -> HopSamples:
    """Arrival-to-arrival observations per hop key, each beside what the
    timetable allowed the trip that ran it.

    WHICH OBSERVATIONS COUNT IS A REALTIME QUESTION and is answered only from
    the trace: an EXACT traversal, with a destination, that the feed's own stop
    sequence says covered one hop. The timetable is read here to attach a
    comparison to each observation, never to admit or reject one — a schedule
    that chose the training set could not then be used to measure the result.

    `n_hops == 1` is checked rather than inferred from the censoring kind. Today
    trace.py only labels a span EXACT when it covers one hop, but nothing in the
    Traversal type says so, and a span that crossed a station in between is not
    a measurement of the pair it names.

    A traversal outside the feed's validity window keeps its place in the fit
    and carries no scheduled time: the schedule it would be compared against was
    not in force when it ran.
    """
    out: dict[HopKey, list[HopSample]] = defaultdict(list)
    bypass = outside = 0
    for t in traversals:
        if t.censoring != EXACT or t.to_stop is None or t.n_hops != 1:
            continue
        covered = timetable.covers(t.at, t.trip_id)
        want = scheduled_for(t, timetable) if covered else None
        if not covered:
            outside += 1
        elif want is not None and want.n_hops not in (None, 1):
            bypass += 1
        out[(t.route_id, t.direction or "", t.from_stop, t.to_stop)].append(
            HopSample(
                seconds=t.seconds,
                scheduled_sec=None if want is None else want.seconds,
            )
        )
    return HopSamples(by_key=dict(out), n_bypass=bypass, n_outside_feed_window=outside)


def _quantiles(shape: float, scale: float) -> tuple[float, float]:
    return (
        loglogistic_quantile(0.5, shape, scale),
        loglogistic_quantile(0.9, shape, scale),
    )


def _scheduled_level(samples: Sequence[HopSample]) -> int | None:
    """The hop's scheduled time, as the median of what the timetable allowed the
    trips that actually ran it.

    Not one number looked up per key: a window can straddle service days, and a
    hop the static feed schedules differently for an express and a local pattern has
    no single scheduled time. Taking the median over the observations weights it
    by the service that ran.
    """
    scheduled = [s.scheduled_sec for s in samples if s.scheduled_sec]
    return int(statistics.median(scheduled)) if scheduled else None


def traversal_baseline(
    traversals: list[Traversal],
    timetable: Timetable,
    *,
    min_hop_samples: int = MIN_HOP_SAMPLES,
) -> tuple[dict[HopKey, TraversalCell], TraversalStats]:
    """Per-hop arrival-to-arrival baselines over whatever window `traversals`
    covers, plus what had to be discarded to build them.

    Every level here comes from the segment's own observed traversals. A hop
    under `min_hop_samples` gets no cell — it abstains rather than borrowing a
    level from the timetable, so that the timetable stays outside the model and
    can be used to measure it (`schedule_drift`). The timetable is still read,
    for two things it alone can do: name the scheduled time each cell is later
    compared against, and catch the bypasses that are not measurements of the
    pair they appear to name.
    """
    samples = hop_samples(traversals, timetable)
    by_key = samples.by_key

    cells: dict[HopKey, TraversalCell] = {}
    for key, hops in by_key.items():
        if len(hops) < min_hop_samples:
            continue
        fit = fit_loglogistic([(s.seconds, True) for s in hops])
        if fit is None:
            continue
        median, p90 = _quantiles(fit.shape, fit.scale)
        cells[key] = TraversalCell(
            n=len(hops),
            median_sec=median,
            p90_sec=p90,
            scheduled_sec=_scheduled_level(hops),
        )

    stats = TraversalStats(
        n_traversals=len(traversals),
        n_fitted=sum(len(s) for s in by_key.values()),
        n_dropped_right=sum(1 for t in traversals if t.censoring == RIGHT),
        n_dropped_interval=sum(
            1 for t in traversals if t.censoring not in (EXACT, RIGHT)
        ),
        n_bypass=samples.n_bypass,
        n_outside_feed_window=samples.n_outside_feed_window,
        n_keys=len(by_key),
        n_cells=len(cells),
        n_keys_thin=len(by_key) - len(cells),
    )
    return cells, stats


@dataclass(frozen=True)
class ScheduleDrift:
    """The outside view of a baseline that was fitted from the inside.

    Every level in `traversal_baseline` comes from the vehicle archive, and is
    then graded against the vehicle archive. A system that has been slow for a
    month teaches that to its own baseline as normal and reports good service;
    nothing in the archive can see it. The timetable can, because it did not
    move. This is the only claim in the module the archive does not get a vote
    on.

    Read it as a level and a shape, not a pass/fail. `median` near 1.0 says the
    fleet is running its own timetable. A rising `median` across successive fits
    is either the system slowing or the timetable changing under it — which is
    why `feed_version` is on the record, and is the only way to tell the two
    apart after the fact.

    `share_slow` is the tail: cells whose own normal is a quarter over schedule.
    Some of those are real permanent slow orders, so the level is not the alarm
    — the change in it is.
    """

    feed_version: str
    n_cells: int  # own-fitted cells the timetable also names
    n_cells_unscheduled: int  # fitted, but the timetable never named the hop
    median: float
    p10: float
    p90: float
    share_slow: float  # fitted median at or over 1.25x scheduled


def schedule_drift(
    cells: Mapping[HopKey, TraversalCell], timetable: Timetable
) -> ScheduleDrift | None:
    """Measure the fitted baselines against the timetable they were not fitted
    from. None when no fitted cell has a scheduled time to compare against."""
    ratios = sorted(
        c.median_sec / c.scheduled_sec for c in cells.values() if c.scheduled_sec
    )
    if not ratios:
        return None
    return ScheduleDrift(
        feed_version=timetable.version.version,
        n_cells=len(ratios),
        n_cells_unscheduled=sum(1 for c in cells.values() if not c.scheduled_sec),
        median=round(statistics.median(ratios), 4),
        p10=round(ratios[len(ratios) // 10], 4),
        p90=round(ratios[int(len(ratios) * 0.9)], 4),
        share_slow=round(sum(1 for r in ratios if r >= 1.25) / len(ratios), 4),
    )


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
    timetable = load_timetable()
    cells, stats = traversal_baseline(traversals, timetable)

    scheduled = [c for c in cells.values() if c.scheduled_sec]
    slowest = sorted(scheduled, key=lambda c: c.median_sec / (c.scheduled_sec or 1))
    print(
        json.dumps(
            {
                "trace": asdict(trace_stats),
                "baseline": asdict(stats),
                "cells": {
                    "median_sec": round(
                        statistics.median([c.median_sec for c in cells.values()]), 1
                    ),
                    "median_spread_p90_over_median": round(
                        statistics.median(
                            [c.p90_sec / c.median_sec for c in cells.values()]
                        ),
                        3,
                    ),
                }
                if cells
                else None,
                "schedule_drift": (
                    asdict(drift)
                    if (drift := schedule_drift(cells, timetable))
                    else None
                ),
                "slowest_vs_schedule": [
                    {
                        "median_sec": round(c.median_sec, 1),
                        "scheduled_sec": c.scheduled_sec,
                        "ratio": round(c.median_sec / (c.scheduled_sec or 1), 2),
                        "n": c.n,
                    }
                    for c in reversed(slowest[-5:])
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
