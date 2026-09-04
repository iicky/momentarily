"""Replay harness for the transition-keyed headway reconstruction.

Task A acceptance: re-run a window of the real trace archive and report, side by
side, what the old STOPPED_AT arrival rule and the new transition-keyed passing
rule each yield. The two numbers that matter:

  * the caught-fraction — of every stop departure in the window, the share the
    old rule also saw (immediately preceding sighting STOPPED_AT). Its
    complement is the departures the old rule missed, each of which merged two
    real headways into one double. This reproduces the crowding measure
    (journal 2026-08-24: 44,441 / 47,936 = 92.71% over 2026-08-20 07:00-11:00
    ET) from the headway path directly.

  * the headway-count delta at the reference stops — the doubles that collapse
    once the missed departures are restored, and what that does to the pooled
    mean headway and CV (the bunching measure the doubles inflated).

Reads the R2 trace archive, so it needs R2 credentials in the environment or a
murk grant (R2_ACCESS_KEY_ID, R2_ACCOUNT_ID, R2_BUCKET, R2_SECRET_ACCESS_KEY).
Default window is the 2026-08-20 07:00-11:00 ET replay the bug was measured on.

    uv run python -m scripts.replay_headway_reconstruction
    uv run python -m scripts.replay_headway_reconstruction \
        --start 2026-08-20T07:00 --end 2026-08-20T11:00
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from training.headway import (
    ReferenceStop,
    headway_events,
    load_gtfs_zip,
    select_reference_stops,
)
from training.load_r2 import date_range, fetch_objects, list_keys
from training.r2_client import load_config, make_client
from training.trace import (
    Arrival,
    Passing,
    arrivals_from_trace,
    passings_and_catch_stats,
)

_ET = ZoneInfo("America/New_York")


def _parse_et(s: str) -> int:
    """An ET wall-clock 'YYYY-MM-DDTHH:MM' to an epoch second."""
    return int(datetime.strptime(s, "%Y-%m-%dT%H:%M").replace(tzinfo=_ET).timestamp())


def _fetch_window(start: int, end: int) -> tuple[list[dict[str, Any]], list[int]]:
    """Every trace object whose scheduled second falls in [start, end], with the
    sorted seconds actually present (for the feed-gap flags)."""
    cfg = load_config()
    client = make_client(cfg)
    start_d = datetime.fromtimestamp(start, _ET).date()
    end_d = datetime.fromtimestamp(end, _ET).date()
    bodies: list[dict[str, Any]] = []
    covered: list[int] = []
    for d in date_range(start_d, end_d):
        keys = list_keys(client, cfg.bucket, f"archive/trace/{d.isoformat()}/")
        want: list[str] = []
        for k in keys:
            sec = int(k.rsplit("/", 1)[-1].split(".")[0])
            if start <= sec <= end:
                want.append(k)
                covered.append(sec)
        bodies.extend(fetch_objects(client, cfg.bucket, want))
    return bodies, sorted(covered)


def _headways_by_cell(
    series: dict[tuple[str, str], list[Any]],
    covered: list[int],
) -> dict[tuple[str, str], list[int]]:
    """Per-cell headways with feed-gap intervals dropped — the real-service
    gaps the wait statistics are fitted on."""
    events = headway_events(series, covered)
    return {
        key: [e.headway_sec for e in evs if not e.feed_gap]
        for key, evs in events.items()
    }


def _group(
    items: list[Any],
    reference_stops: dict[tuple[str, str], ReferenceStop],
) -> dict[tuple[str, str], list[Any]]:
    """Group arrivals OR passings onto their route/direction reference stop.
    (reference_arrivals typed on Passing; arrivals are structurally identical
    for the fields it reads, so this local grouping keeps the old rule honest
    without a cast.)"""
    want = {
        (rs.route, rs.direction, rs.stop_id): key for key, rs in reference_stops.items()
    }
    out: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for it in items:
        key = want.get((it.route_id, it.direction or "", it.stop_id))
        if key is not None:
            out[key].append(it)
    for s in out.values():
        s.sort(key=lambda x: x.at)
    return dict(out)


def _summary(
    by_cell: dict[tuple[str, str], list[int]],
) -> dict[str, float | int | None]:
    """Headway shape. `cv_pooled` is over all cells' headways at once and is
    dominated by BETWEEN-cell mean differences, so it is not the bunching
    measure; `cv_percell_mean` — the mean of each cell's own CV over cells with
    at least three headways — is the WITHIN-cell bunching the doubles were said
    to inflate."""
    pooled = [h for hs in by_cell.values() for h in hs]
    if len(pooled) < 2:
        return {"n_headways": len(pooled), "n_cells": len(by_cell)}
    mean = statistics.fmean(pooled)
    percell = [
        statistics.stdev(hs) / statistics.fmean(hs)
        for hs in by_cell.values()
        if len(hs) >= 3 and statistics.fmean(hs) > 0
    ]
    return {
        "n_headways": len(pooled),
        "n_cells": len(by_cell),
        "mean_sec": round(mean, 1),
        "median_sec": round(statistics.median(pooled), 1),
        "cv_pooled": round(statistics.stdev(pooled) / mean, 4),
        "cv_percell_mean": round(statistics.fmean(percell), 4) if percell else None,
        "n_cells_cv": len(percell),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2026-08-20T07:00", help="ET wall clock")
    parser.add_argument("--end", default="2026-08-20T11:00", help="ET wall clock")
    args = parser.parse_args(argv)

    start, end = _parse_et(args.start), _parse_et(args.end)
    bodies, covered = _fetch_window(start, end)

    reference_stops = select_reference_stops(load_gtfs_zip())

    old_arr: list[Arrival] = arrivals_from_trace(bodies)
    new_pass: list[Passing]
    new_pass, catch = passings_and_catch_stats(bodies)

    old_series = _group(old_arr, reference_stops)
    new_series = _group(new_pass, reference_stops)
    old = _summary(_headways_by_cell(old_series, covered))
    new = _summary(_headways_by_cell(new_series, covered))

    report = {
        "window_et": [args.start, args.end],
        "n_bodies": len(bodies),
        "caught_fraction": {
            "n_transitions": catch.n_transitions,
            "caught": catch.caught,
            "missed": catch.missed,
            "missed_never_stopped": catch.missed_never_stopped,
            "missed_stopped_then_moved": catch.missed_stopped_then_moved,
            "caught_fraction": round(catch.caught_fraction, 4),
        },
        "reference_stop_headways": {
            "old_stopped_at_rule": old,
            "new_transition_rule": new,
            "doubles_collapsed": int(new.get("n_headways") or 0)
            - int(old.get("n_headways") or 0),
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
