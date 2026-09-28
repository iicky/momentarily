"""How much of a real end-to-end trip the journey-span dataset can see, and
whether what it cannot see is the slow part.

The span dataset (training/journey.py) chains consecutive EXACT/INTERVAL
traversals of one trip and scores windows of up to MAX_HOPS realtime hops. Two
things limit what that says about a rider's whole trip:

- Censoring. A RIGHT-censored traversal (the trip was last seen in transit) or
  a gap in stop sequence ends a chain. A trip that vanishes mid-run is more
  likely a disrupted one, so observed spans may be optimistic.
- Length. A rider's trip is often longer than MAX_HOPS stops; the dataset
  scores nothing past that.

Measured per trip over a window of the derived archive:

- chain structure: number of chains per trip, longest chain in realtime hops,
  and the share of the trip's traversals that sit in its longest chain;
- against the timetable: the trip's scheduled stop count on its service day
  (from the static feed's pattern for its path code, when exactly one is
  known), so "longest observed chain / scheduled hops" says how much of the
  trip one window could ever cover;
- censoring bias: the median realized/scheduled ratio of EXACT hops on trips
  that end in a RIGHT-censored traversal versus trips that do not. A gap here
  is the optimism the span dataset carries by construction.

Operator tool; reads archive/traversals/ and the live static feed. Not wired
into any workflow.
"""

from __future__ import annotations

import argparse
import bisect
import json
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import date
from typing import Any

from training.gtfs_static import (
    DayTimetable,
    Pattern,
    Timetable,
    load_timetable,
    path_code,
)
from training.journey_types import MAX_HOPS
from training.trace import EXACT, INTERVAL, RIGHT, TRIP_GAP_SECONDS, Traversal
from training.traversal_archive import read_days


@dataclass(frozen=True)
class TripShape:
    trip_id: str
    route_id: str
    n_traversals: int
    n_chains: int
    longest_chain_hops: int  # realtime hops in the longest EXACT/INTERVAL chain
    longest_chain_share: float  # traversals in that chain / all traversals
    ends_right_censored: bool
    scheduled_hops: (
        int | None
    )  # pattern stop count - 1, when exactly one pattern is known
    exact_ratio_median: float | None  # median realized/scheduled over EXACT hops


def _chains(rows: Sequence[Traversal]) -> list[list[Traversal]]:
    """Consecutive chainable traversals, the same rule journey.py uses."""
    out: list[list[Traversal]] = []
    cur: list[Traversal] = []
    for t in rows:
        ok = (
            t.censoring in (EXACT, INTERVAL)
            and t.direction is not None
            and t.to_stop is not None
            and t.n_hops is not None
        )
        if ok and cur and cur[-1].to_stop == t.from_stop:
            cur.append(t)
        elif ok:
            if cur:
                out.append(cur)
            cur = [t]
        else:
            if cur:
                out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _pattern_candidates(table: DayTimetable, code: str) -> tuple[Pattern, ...]:
    """Same lookup as DayTimetable._candidates (exact key, else prefix match
    over sorted codes): that method is private to gtfs_static, so this
    derives the same result from the public patterns/codes attributes."""
    exact = table.patterns.get(code)
    if exact is not None:
        return exact
    out: list[Pattern] = []
    i = bisect.bisect_left(table.codes, code)
    while i < len(table.codes) and table.codes[i].startswith(code):
        out.extend(table.patterns[table.codes[i]])
        i += 1
    return tuple(out)


def _scheduled_hops(timetable: Timetable, t: Traversal) -> int | None:
    table = timetable.day_for(t.at, t.trip_id)
    cands = _pattern_candidates(table, path_code(t.trip_id))
    lengths = {len(p.stops) for p in cands}
    return lengths.pop() - 1 if len(lengths) == 1 else None


def split_runs(traversals: Iterable[Traversal]) -> list[list[Traversal]]:
    """One list per continuous run of a trip, each ascending by time.

    A trip_id recurs (the next day, or sooner), and a run can cross UTC
    midnight. So runs split on a time gap, the same rule the trace uses: a
    traversal that starts more than TRIP_GAP_SECONDS after the previous one
    ended belongs to a different run. Calendar days play no part."""
    by_trip: dict[str, list[Traversal]] = defaultdict(list)
    for t in traversals:
        by_trip[t.trip_id].append(t)
    runs: list[list[Traversal]] = []
    for rows in by_trip.values():
        rows.sort(key=lambda r: r.at)
        run: list[Traversal] = [rows[0]]
        for t in rows[1:]:
            prev = run[-1]
            if t.at - (prev.at + prev.seconds) > TRIP_GAP_SECONDS:
                runs.append(run)
                run = []
            run.append(t)
        runs.append(run)
    return runs


def trip_shapes(
    traversals: Iterable[Traversal], timetable: Timetable
) -> list[TripShape]:
    out: list[TripShape] = []
    for rows in split_runs(traversals):
        trip_id = rows[0].trip_id
        chains = _chains(rows)
        empty_chain: list[Traversal] = []
        longest = max(
            chains, key=lambda c: sum(r.n_hops or 0 for r in c), default=empty_chain
        )
        longest_hops = sum(r.n_hops or 0 for r in longest)
        ratios: list[float] = []
        for r in rows:
            if r.censoring != EXACT or r.to_stop is None:
                continue
            day = timetable.day_for(r.at, r.trip_id)
            sched = day.hops.get(
                (r.route_id, r.direction or "", r.from_stop, r.to_stop)
            )
            if sched:
                ratios.append(r.seconds / sched)
        out.append(
            TripShape(
                trip_id=trip_id,
                route_id=rows[0].route_id,
                n_traversals=len(rows),
                n_chains=len(chains),
                longest_chain_hops=longest_hops,
                longest_chain_share=(len(longest) / len(rows)) if rows else 0.0,
                ends_right_censored=rows[-1].censoring == RIGHT,
                scheduled_hops=_scheduled_hops(timetable, rows[0]),
                exact_ratio_median=statistics.median(ratios) if ratios else None,
            )
        )
    return out


def _quantile[T: (int, float)](xs: Sequence[T], p: float) -> T:
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def summarize(shapes: Sequence[TripShape]) -> dict[str, Any]:
    if not shapes:
        raise ValueError("summarize needs at least one trip")
    n = len(shapes)
    longest = sorted(s.longest_chain_hops for s in shapes)
    with_sched = [s for s in shapes if s.scheduled_hops]
    frac = sorted(
        min(1.0, s.longest_chain_hops / sched)
        for s in with_sched
        if (sched := s.scheduled_hops) is not None
    )
    fits = sum(
        1
        for s in with_sched
        if (sched := s.scheduled_hops) is not None and sched <= MAX_HOPS
    )
    censored = [
        s.exact_ratio_median
        for s in shapes
        if s.ends_right_censored and s.exact_ratio_median
    ]
    clean = [
        s.exact_ratio_median
        for s in shapes
        if not s.ends_right_censored and s.exact_ratio_median
    ]
    return {
        "n_trips": n,
        "share_single_chain": sum(1 for s in shapes if s.n_chains == 1) / n,
        "chains_per_trip": dict(
            sorted(Counter(min(s.n_chains, 5) for s in shapes).items())
        ),
        "longest_chain_hops_p10_p50_p90": [
            _quantile(longest, 0.1),
            _quantile(longest, 0.5),
            _quantile(longest, 0.9),
        ],
        "share_longest_chain_ge_max_hops": sum(1 for h in longest if h >= MAX_HOPS) / n,
        "n_trips_with_scheduled_hops": len(with_sched),
        "scheduled_hops_p10_p50_p90": [
            _quantile(
                sorted(
                    sched for s in with_sched if (sched := s.scheduled_hops) is not None
                ),
                p,
            )
            for p in (0.1, 0.5, 0.9)
        ]
        if with_sched
        else None,
        "share_trip_shorter_than_max_hops": fits / len(with_sched)
        if with_sched
        else None,
        "longest_chain_over_scheduled_p10_p50_p90": [
            _quantile(frac, 0.1),
            _quantile(frac, 0.5),
            _quantile(frac, 0.9),
        ]
        if frac
        else None,
        "share_ends_right_censored": sum(1 for s in shapes if s.ends_right_censored)
        / n,
        "exact_ratio_median_censored_trips": statistics.median(censored)
        if censored
        else None,
        "exact_ratio_median_clean_trips": statistics.median(clean) if clean else None,
        "exact_ratio_p90_censored_trips": _quantile(sorted(censored), 0.9)
        if censored
        else None,
        "exact_ratio_p90_clean_trips": _quantile(sorted(clean), 0.9) if clean else None,
    }


def main(argv: list[str] | None = None) -> int:
    assert __doc__ is not None
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--route", default=None, help="one route_id; default all")
    parser.add_argument(
        "--trips-out", default=None, help="write per-trip shapes JSONL here"
    )
    args = parser.parse_args(argv)
    result = read_days(args.start, args.end)
    rows = result.traversals
    if args.route:
        rows = [t for t in rows if t.route_id == args.route]
    if not rows:
        scope = f" for route {args.route}" if args.route else ""
        print(
            f"journey_coverage: no traversals{scope} in {args.start}..{args.end}",
            file=sys.stderr,
        )
        return 1
    shapes = trip_shapes(rows, load_timetable())
    if args.trips_out:
        with open(args.trips_out, "w", encoding="utf-8") as f:
            for s in shapes:
                f.write(json.dumps(asdict(s)) + "\n")
    json.dump(summarize(shapes), sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
