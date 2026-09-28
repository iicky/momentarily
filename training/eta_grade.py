"""Grade the MTA's own countdown predictions against the arrivals the trace
actually observed.

Predictions are the public v1/arrivals.json snapshots saved by a local poller
(one gzipped file per published snapshot under --snapshots/<UTC date>/): per
stop, the next few trains with the MTA's `eta_epoch` and `trip_id`, stamped
with the snapshot's `observed_at`. Truth is training.trace: the realized
arrival of the SAME trip_id at the SAME stop (the first sighting stopped
there, timed by the feed's own vehicle clock).

Per prediction:
    horizon = eta_epoch - observed_at          (how far ahead it predicted)
    error   = actual_arrival - eta_epoch        (+ = train later than promised)

Matching: the first arrival of (trip_id, stop) whose time falls in
[observed_at, observed_at + MATCH_AFTER]. trip_ids recur on later days, so the
window is bounded. A prediction with no matched arrival is classified:
- `already_arrived`: the trip arrived at (or left) that stop in the
  MATCH_BEFORE seconds BEFORE the snapshot — the countdown was still listing a
  train that had already come; stale, never scored as an error;
and, only once the trace runs at least GRACE past its eta:
- `served_unobserved`: the trace has a PASSING of that trip at that stop in the
  window, so the train served the stop but was never caught standing there
  (dwell shorter than the 1-minute poll) — a trace limitation, not an MTA miss;
- `never_seen`: neither — the train never reported that stop in the window
  (skipped, cancelled, rerouted, or dropped from the feed).
Predictions whose outcome window runs past the trace's end are `pending` and
excluded from everything.

Operator tool: reads saved snapshots locally and archive/trace/ from R2.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import statistics
import sys
from array import array
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from training.load_r2 import fetch_objects, list_keys
from training.r2_client import load_config, make_client
from training.trace import Arrival, Passing, arrivals_from_trace, passings_from_trace

MATCH_BEFORE = 120
MATCH_AFTER = 2 * 3600
GRACE = 1800

# Trace minutes borrowed from the previous day so a trip already standing at a
# stop at 00:00 UTC is seen standing there, not credited with a fresh arrival.
# Longer than trace.TRIP_GAP_SECONDS so the run state is continuous.
LEAD_IN = 20 * 60

HORIZON_BINS: tuple[tuple[str, int, int], ...] = (
    ("0-2 min", 0, 120),
    ("2-5 min", 120, 300),
    ("5-10 min", 300, 600),
    ("10-20 min", 600, 1200),
    ("20-60 min", 1200, 3601),
)

_ET = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class Prediction:
    observed_at: int
    stop_id: str
    route: str
    trip_id: str
    eta_epoch: int


def read_snapshots(root: str, start: date, end: date) -> Iterator[Prediction]:
    """Every prediction in every saved snapshot whose UTC date is in [start, end]."""
    d = start
    while d <= end:
        folder = os.path.join(root, d.isoformat())
        if os.path.isdir(folder):
            for name in sorted(os.listdir(folder)):
                if not name.endswith(".json.gz"):
                    continue
                with gzip.open(os.path.join(folder, name), "rt", encoding="utf-8") as f:
                    doc: dict[str, Any] = json.load(f)
                observed_at = int(doc["observed_at"])
                arrivals: dict[str, list[dict[str, Any]]] = doc.get("arrivals") or {}
                for stop_id, rows in arrivals.items():
                    for r in rows:
                        if not r.get("trip_id"):
                            continue
                        yield Prediction(
                            observed_at=observed_at,
                            stop_id=stop_id,
                            route=str(r.get("route") or ""),
                            trip_id=str(r["trip_id"]),
                            eta_epoch=int(r["eta_epoch"]),
                        )
        d += timedelta(days=1)


class _Index:
    """Event times per (trip_id, stop_id), ascending."""

    def __init__(self, events: Iterable[Arrival | Passing]) -> None:
        by: dict[tuple[str, str], list[int]] = defaultdict(list)
        for e in events:
            by[(e.trip_id, e.stop_id)].append(e.at)
        for v in by.values():
            v.sort()
        self._by = by

    def first_in(self, trip_id: str, stop_id: str, lo: int, hi: int) -> int | None:
        times = self._by.get((trip_id, stop_id))
        if not times:
            return None
        i = bisect_left(times, lo)
        return times[i] if i < len(times) and times[i] <= hi else None


def arrivals_and_passings_by_day(
    start: date, end: date
) -> tuple[list[Arrival], list[Passing], int]:
    """Trace arrivals and passings over [start, end], one UTC day at a time so
    memory holds one day of trace bodies, each day read with LEAD_IN of the
    previous day and only its own events kept. Returns the last poll second
    seen as the trace's end."""
    cfg = load_config()
    client = make_client(cfg)
    arrivals: list[Arrival] = []
    passings: list[Passing] = []
    trace_end = 0
    d = start
    while d <= end:
        day_start = int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp())
        keys = list_keys(client, cfg.bucket, f"archive/trace/{d.isoformat()}/")
        prev = list_keys(
            client, cfg.bucket, f"archive/trace/{(d - timedelta(days=1)).isoformat()}/"
        )
        keys += [
            k
            for k in prev
            if int(k.rsplit("/", 1)[1].split(".")[0]) >= day_start - LEAD_IN
        ]
        if keys:
            bodies = fetch_objects(client, cfg.bucket, keys)
            trace_end = max(
                trace_end,
                max(
                    int(b.get("scheduled_at") or b.get("observed_at") or 0)
                    for b in bodies
                ),
            )
            arrivals += [a for a in arrivals_from_trace(bodies) if a.at >= day_start]
            passings += [p for p in passings_from_trace(bodies) if p.at >= day_start]
            del bodies
        print(
            f"trace {d}: {len(keys)} bodies, {len(arrivals)} arrivals so far",
            file=sys.stderr,
        )
        d += timedelta(days=1)
    return arrivals, passings, trace_end


def _bin(horizon: int) -> str | None:
    for name, lo, hi in HORIZON_BINS:
        if lo <= horizon < hi:
            return name
    return None


@dataclass
class Grade:
    """Errors (seconds) per cell, plus outcome counts."""

    errors: dict[str, array[int]] = field(
        default_factory=lambda: defaultdict(lambda: array("i"))
    )
    outcomes: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def add(self, key: str, err: int) -> None:
        self.errors[key].append(err)


def grade(
    predictions: Iterable[Prediction],
    arrivals: Iterable[Arrival],
    passings: Iterable[Passing],
    trace_end: int,
) -> Grade:
    arr = _Index(arrivals)
    pas = _Index(passings)
    g = Grade()
    for p in predictions:
        horizon = p.eta_epoch - p.observed_at
        b = _bin(horizon)
        if b is None:
            g.outcomes["horizon_out_of_range"] += 1
            continue
        # Only an arrival at or after the snapshot is an outcome of it. One in the
        # MATCH_BEFORE seconds before means the countdown was still listing a
        # train that had already arrived: counted as stale, never scored.
        before, lo, hi = (
            p.observed_at - MATCH_BEFORE,
            p.observed_at,
            p.observed_at + MATCH_AFTER,
        )
        actual = arr.first_in(p.trip_id, p.stop_id, lo, hi)
        if actual is None:
            if arr.first_in(p.trip_id, p.stop_id, before, lo - 1) is not None or (
                pas.first_in(p.trip_id, p.stop_id, before, lo - 1) is not None
            ):
                g.outcomes[f"already_arrived|{b}"] += 1
            elif p.eta_epoch + GRACE > trace_end:
                g.outcomes["pending"] += 1
            elif pas.first_in(p.trip_id, p.stop_id, lo, hi) is not None:
                g.outcomes[f"served_unobserved|{b}"] += 1
            else:
                g.outcomes[f"never_seen|{b}"] += 1
            continue
        if actual > trace_end:
            g.outcomes["pending"] += 1
            continue
        g.outcomes[f"matched|{b}"] += 1
        err = actual - p.eta_epoch
        g.add(f"h|{b}", err)
        if 120 <= horizon < 600:  # the countdown a rider reads walking to the platform
            g.add(f"route|{p.route}", err)
            g.add(f"hour|{datetime.fromtimestamp(p.observed_at, _ET).hour:02d}", err)
    return g


def _stats(xs: array[int]) -> dict[str, float | int]:
    s = sorted(xs)
    n = len(s)

    def q(p: float) -> int:
        return s[min(n - 1, int(p * n))]

    return {
        "n": n,
        "bias_mean": round(statistics.fmean(s), 1),
        "median": q(0.5),
        "mae": round(statistics.fmean(abs(x) for x in s), 1),
        "p10": q(0.10),
        "p90": q(0.90),
        "within_60s": round(sum(1 for x in s if abs(x) <= 60) / n, 3),
        "late_over_120s": round(sum(1 for x in s if x > 120) / n, 3),
        "early_over_60s": round(sum(1 for x in s if x < -60) / n, 3),
    }


def summarize(g: Grade) -> dict[str, Any]:
    by_h = {
        b: _stats(g.errors[f"h|{b}"])
        for b, _, _ in HORIZON_BINS
        if g.errors.get(f"h|{b}")
    }
    outcome_by_h: dict[str, dict[str, float | int]] = {}
    for b, _, _ in HORIZON_BINS:
        m = g.outcomes.get(f"matched|{b}", 0)
        aa = g.outcomes.get(f"already_arrived|{b}", 0)
        su = g.outcomes.get(f"served_unobserved|{b}", 0)
        ns = g.outcomes.get(f"never_seen|{b}", 0)
        tot = m + aa + su + ns
        if tot:
            outcome_by_h[b] = {
                "resolved": tot,
                "matched": round(m / tot, 3),
                "already_arrived": round(aa / tot, 3),
                "served_unobserved": round(su / tot, 3),
                "never_seen": round(ns / tot, 3),
            }
    routes = {
        k.split("|", 1)[1]: _stats(v)
        for k, v in g.errors.items()
        if k.startswith("route|") and len(v) >= 500
    }
    hours = {
        k.split("|", 1)[1]: _stats(v)
        for k, v in sorted(g.errors.items())
        if k.startswith("hour|")
    }
    return {
        "by_horizon": by_h,
        "outcomes_by_horizon": outcome_by_h,
        "pending": g.outcomes.get("pending", 0),
        "by_route_2_10min": dict(sorted(routes.items(), key=lambda kv: -kv[1]["mae"])),
        "by_hour_et_2_10min": hours,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Grade MTA countdown ETAs against the trace"
    )
    parser.add_argument(
        "--snapshots",
        required=True,
        help="directory of saved arrivals snapshots, one folder per UTC date",
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        required=True,
        help="first snapshot UTC date",
    )
    parser.add_argument(
        "--end", type=date.fromisoformat, required=True, help="last snapshot UTC date"
    )
    parser.add_argument("--out", default=None, help="write the summary JSON here too")
    args = parser.parse_args(argv)
    # The trace must run past the last prediction's outcome window.
    arrivals, passings, trace_end = arrivals_and_passings_by_day(
        args.start, args.end + timedelta(days=1)
    )
    g = grade(
        read_snapshots(args.snapshots, args.start, args.end),
        arrivals,
        passings,
        trace_end,
    )
    summary = summarize(g)
    summary["window"] = {
        "snapshots": f"{args.start}..{args.end}",
        "trace_end": datetime.fromtimestamp(trace_end, UTC).isoformat(),
        "n_arrivals": len(arrivals),
        "n_passings": len(passings),
    }
    text = json.dumps(summary, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
