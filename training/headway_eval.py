"""Evaluation of the observed-headway wait signal against the independent
movement and supply axes and the alert feed.

Four questions, all resampled by night or episode (never by tick — the ~5-min
readings inside a night are one autocorrelated draw, not many) with the cluster
count reported beside the tick count and the interval withheld below two
clusters:

  a. false_alarm_gate — how often the wait flags fire on service both the
     movement and supply axes call normal. The discipline to land against is the
     movement arm's certified 0.00101/tick false-alarm bound.
  b. schedule_vs_typical / saturday_head_to_head — how much MORE a schedule
     (SWT) reference flags on normal service than the own-cell typical-actual
     reference, and the 1/2-on-a-Saturday instance of it.
  c. severity_tiers / severity_report — a first-pass ordinal severity from
     own-cell quantile exceedance and dwell, its prevalence and episodes, and
     where it agrees and DISAGREES with the movement condition and the alert
     feed's Delays.

A caution that rides on all of this: the observed headways are reconstructed
from the same vehicle-position (ATS) stream the movement detector reads, so a
common-mode failure — a tracking outage, a feed stall — blinds both at once.
Movement-confirmed-normal is therefore a consistency reference, not an
independent one; the false-alarm rate is a bound on how often the wait signal
contradicts the movement signal on shared-normal service, which is what it is
reported as. Miss adjudication (what BOTH missed) needs an external source and
is out of scope here.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from itertools import groupby
from typing import TYPE_CHECKING, cast
from zoneinfo import ZoneInfo

from momentarily.hmm import schedule_bin
from training.headway import (
    HEADWAY_MIN_NIGHTS,
    TICK_SECONDS,
    ReferenceStop,
    TickWait,
    WaitCell,
    headway_events,
    load_gtfs_zip,
    reference_arrivals,
    scheduled_swt,
    select_reference_stops,
    tick_aligned_waits,
    typical_actual_baseline,
)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

    from training.r2_client import R2Config

_NYC_TZ = ZoneInfo("America/New_York")


def _et_date(epoch_seconds: int):
    """ET calendar date of a tick — the night unit clusters resample on."""
    return datetime.fromtimestamp(epoch_seconds, tz=_NYC_TZ).date()


# route -> the GTFS-RT line-group feed whose freshness governs it, so a
# per-group stall (trains vanish from the trace) can be told from real absence.
ROUTE_FEED_GROUP: dict[str, str] = {
    **dict.fromkeys(("1", "2", "3", "4", "5", "6", "7", "GS"), "numbered"),
    **dict.fromkeys(("A", "C", "E", "H", "FS"), "ace"),
    **dict.fromkeys(("B", "D", "F", "M"), "bdfm"),
    "G": "g",
    **dict.fromkeys(("J", "Z"), "jz"),
    **dict.fromkeys(("N", "Q", "R", "W"), "nqrw"),
    "L": "l",
    **dict.fromkeys(("SI", "SS"), "si"),
}

# --- confirmed-normal reference ---


def confirmed_normal(
    movement: Mapping[tuple[str, int], str],
    supply: Mapping[tuple[str, int], str],
) -> set[tuple[str, int]]:
    """(route, tick) pairs BOTH axes independently judged normal. A tick either
    axis abstains on (absent from its map) or calls not-normal is excluded, so
    the reference is only service two orthogonal-ish signals agree is ordinary."""
    return {
        k for k, v in movement.items() if v == "normal" and supply.get(k) == "normal"
    }


# --- night-clustered bootstrap ---


@dataclass(frozen=True)
class Unit:
    """One resampling cluster's exposure: how many readings it carried and how
    many fired. The cluster is a (route, night), so a whole night moves together
    in a resample rather than each autocorrelated tick counting on its own."""

    alarmed: int
    total: int


@dataclass(frozen=True)
class Rate:
    rate: float | None
    lo: float | None
    hi: float | None
    n_units: int
    n_ticks: int
    n_alarmed: int


def night_bootstrap(
    units: Sequence[Unit], *, n_boot: int = 2000, seed: int = 0
) -> Rate:
    """Pooled firing rate with a night-clustered percentile CI. Resamples whole
    units with replacement; withholds the interval below two units (one cluster
    resampled with replacement returns itself every draw, so the band would be a
    false point). No units -> no rate at all, rather than a fabricated zero."""
    total = sum(u.total for u in units)
    alarmed = sum(u.alarmed for u in units)
    if not units or total == 0:
        return Rate(None, None, None, len(units), total, alarmed)
    point = alarmed / total
    if len(units) < 2:
        return Rate(point, None, None, len(units), total, alarmed)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        pick = [units[rng.randrange(len(units))] for _ in range(len(units))]
        t = sum(u.total for u in pick)
        if t:
            draws.append(sum(u.alarmed for u in pick) / t)
    draws.sort()
    lo = draws[int(0.025 * (len(draws) - 1))]
    hi = draws[int(0.975 * (len(draws) - 1))]
    return Rate(point, lo, hi, len(units), total, alarmed)


# --- flags on a tick ---


def _cell_of(tw: TickWait) -> tuple[str, str, str]:
    return (tw.route, tw.direction, schedule_bin(tw.tick))


def above_p90(tw: TickWait, cell: WaitCell) -> bool:
    return tw.awt_sec > cell.p90


def twice_typical(tw: TickWait, cell: WaitCell) -> bool:
    return tw.awt_sec > 2 * cell.p50


def _sustained_ticks(
    series: Sequence[TickWait],
    baseline: Mapping[tuple[str, str, str], WaitCell],
    *,
    min_run: int,
) -> set[int]:
    """Ticks inside a run of at least `min_run` consecutive above-p90 readings on
    a contiguous 5-min grid — the sustained-excursion form of the flag, closer to
    a real alarm than a single tick clearing p90."""
    flagged = {
        tw.tick
        for tw in series
        if (c := baseline.get(_cell_of(tw))) is not None and above_p90(tw, c)
    }
    out: set[int] = set()
    ordered = sorted(flagged)
    run: list[int] = []
    for t in ordered:
        if run and t - run[-1] == TICK_SECONDS:
            run.append(t)
        else:
            if len(run) >= min_run:
                out.update(run)
            run = [t]
    if len(run) >= min_run:
        out.update(run)
    return out


# --- gate (a): false alarms on confirmed-normal service ---


def false_alarm_gate(
    waits: Mapping[tuple[str, str], Sequence[TickWait]],
    baseline: Mapping[tuple[str, str, str], WaitCell],
    normal: set[tuple[str, int]],
    *,
    sustained_run: int = 6,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Rate]:
    """Per-tick false-alarm rate of each wait flag on confirmed-normal ticks,
    night-bootstrapped. A tick counts only where its cell published a baseline
    (else the signal abstains). `above_p90` is ~0.10 by construction — a
    percentile threshold flags a tenth of its own distribution — and is reported
    as the identity it is; `twice_typical` and the sustained variant are the
    numbers to weigh against the movement bound."""
    per_flag_units: dict[str, dict[tuple[str, str], list[int]]] = {
        "above_p90": {},
        "twice_typical": {},
        "sustained_above_p90": {},
    }
    per_flag_total: dict[str, dict[tuple[str, str], int]] = {
        k: {} for k in per_flag_units
    }

    for (route, _direction), series in waits.items():
        sustained = _sustained_ticks(series, baseline, min_run=sustained_run)
        for tw in series:
            if (route, tw.tick) not in normal:
                continue
            cell = baseline.get(_cell_of(tw))
            if cell is None:
                continue
            night = (route, _et_date(tw.tick).isoformat())
            for flag, fired in (
                ("above_p90", above_p90(tw, cell)),
                ("twice_typical", twice_typical(tw, cell)),
                ("sustained_above_p90", tw.tick in sustained),
            ):
                per_flag_total[flag][night] = per_flag_total[flag].get(night, 0) + 1
                per_flag_units[flag].setdefault(night, []).append(1 if fired else 0)

    out: dict[str, Rate] = {}
    for flag in per_flag_units:
        units = [
            Unit(alarmed=sum(hits), total=per_flag_total[flag][night])
            for night, hits in per_flag_units[flag].items()
        ]
        out[flag] = night_bootstrap(units, n_boot=n_boot, seed=seed)
    return out


# --- gate (b): schedule reference vs typical-actual reference ---


def schedule_vs_typical(
    waits: Mapping[tuple[str, str], Sequence[TickWait]],
    baseline: Mapping[tuple[str, str, str], WaitCell],
    swt: Mapping[tuple[str, str, str], float],
    normal: set[tuple[str, int]],
    *,
    schedule_excess: float = 1.25,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Rate]:
    """On confirmed-normal ticks that have BOTH a typical-actual cell and a
    scheduled SWT, the firing rate of a schedule-referenced flag (AWT over
    `schedule_excess` x SWT — the delivered wait running long against the
    timetable) against the own-cell flag (AWT over own p90). A route whose
    delivered headways chronically differ from its schedule flags against the
    timetable on ordinary service; against its own history it does not."""
    sched_units: dict[tuple[str, str], list[int]] = {}
    own_units: dict[tuple[str, str], list[int]] = {}
    totals: dict[tuple[str, str], int] = {}
    for (route, _direction), series in waits.items():
        for tw in series:
            if (route, tw.tick) not in normal:
                continue
            cell = baseline.get(_cell_of(tw))
            sched = swt.get(_cell_of(tw))
            if cell is None or sched is None:
                continue
            night = (route, _et_date(tw.tick).isoformat())
            totals[night] = totals.get(night, 0) + 1
            sched_units.setdefault(night, []).append(
                1 if tw.awt_sec > schedule_excess * sched else 0
            )
            own_units.setdefault(night, []).append(1 if above_p90(tw, cell) else 0)

    def rate(u: dict[tuple[str, str], list[int]]) -> Rate:
        return night_bootstrap(
            [Unit(sum(h), totals[n]) for n, h in u.items()],
            n_boot=n_boot,
            seed=seed,
        )

    return {"schedule_referenced": rate(sched_units), "own_cell_p90": rate(own_units)}


def saturday_head_to_head(
    waits: Mapping[tuple[str, str], Sequence[TickWait]],
    baseline: Mapping[tuple[str, str, str], WaitCell],
    swt: Mapping[tuple[str, str, str], float],
    day: str,
    routes: Sequence[str],
) -> list[dict[str, object]]:
    """The specific instance: for each (route, direction, schedule_bin) touched
    on `day`, the delivered AWT beside the own-cell p90 and the scheduled SWT.
    Reports, per cell, whether a schedule reference and the own-cell reference
    each flag it — the concrete case where the two disagree on one day of atypical
    (weekday-level) Saturday service."""
    out: list[dict[str, object]] = []
    for (route, direction), series in waits.items():
        if route not in routes:
            continue
        by_cell: dict[str, list[float]] = {}
        for tw in series:
            if _et_date(tw.tick).isoformat() != day:
                continue
            by_cell.setdefault(schedule_bin(tw.tick), []).append(tw.awt_sec)
        for sbin, awts in sorted(by_cell.items()):
            cell = baseline.get((route, direction, sbin))
            sched = swt.get((route, direction, sbin))
            observed = statistics.median(awts)
            out.append(
                {
                    "route": route,
                    "direction": direction,
                    "bin": sbin,
                    "observed_awt_s": round(observed, 1),
                    "own_p50_s": round(cell.p50, 1) if cell else None,
                    "own_p90_s": round(cell.p90, 1) if cell else None,
                    "scheduled_swt_s": round(sched, 1) if sched else None,
                    "flag_own_p90": bool(cell and observed > cell.p90),
                    "flag_schedule_125": bool(sched and observed > 1.25 * sched),
                    "n_ticks": len(awts),
                }
            )
    return out


# --- gate (c): severity tiers ---


# First-pass ordinal cutpoints on own-cell quantile exceedance. Deliberately
# coarse and not tuned to any outcome: normal at or under the cell's typical high
# (p90), severe past twice the cell's typical wait, degraded between. Dwell (an
# episode's length) rides alongside as the second axis rather than being folded
# into the tier.
def severity_tier(tw: TickWait, cell: WaitCell) -> int:
    if tw.awt_sec > 2 * cell.p50 and tw.awt_sec > cell.p90:
        return 2
    if tw.awt_sec > cell.p90:
        return 1
    return 0


@dataclass(frozen=True)
class Episode:
    route: str
    direction: str
    start: int
    end: int
    peak_tier: int
    n_ticks: int


def severity_series(
    waits: Mapping[tuple[str, str], Sequence[TickWait]],
    baseline: Mapping[tuple[str, str, str], WaitCell],
) -> dict[tuple[str, int], tuple[str, int]]:
    """(route, tick) -> (direction, tier) for every tick with a published cell,
    keeping the worse of the two directions when both are read at that tick."""
    out: dict[tuple[str, int], tuple[str, int]] = {}
    for (route, direction), series in waits.items():
        for tw in series:
            cell = baseline.get(_cell_of(tw))
            if cell is None:
                continue
            tier = severity_tier(tw, cell)
            key = (route, tw.tick)
            prev = out.get(key)
            if prev is None or tier > prev[1]:
                out[key] = (direction, tier)
    return out


def episodes(
    waits: Mapping[tuple[str, str], Sequence[TickWait]],
    baseline: Mapping[tuple[str, str, str], WaitCell],
    *,
    min_tier: int = 1,
) -> list[Episode]:
    """Maximal runs of tier >= min_tier on one (route, direction) on a contiguous
    5-min grid — a break in tier or a gap in coverage ends an episode."""
    out: list[Episode] = []
    for (route, direction), series in waits.items():
        tiered = [
            (tw.tick, severity_tier(tw, c))
            for tw in series
            if (c := baseline.get(_cell_of(tw))) is not None
        ]
        run: list[tuple[int, int]] = []
        for tick, tier in tiered:
            hot = tier >= min_tier
            if run and hot and tick - run[-1][0] == TICK_SECONDS:
                run.append((tick, tier))
            else:
                if run:
                    out.append(_episode(route, direction, run))
                run = [(tick, tier)] if hot else []
        if run:
            out.append(_episode(route, direction, run))
    return out


def _episode(route: str, direction: str, run: list[tuple[int, int]]) -> Episode:
    return Episode(
        route=route,
        direction=direction,
        start=run[0][0],
        end=run[-1][0],
        peak_tier=max(t for _, t in run),
        n_ticks=len(run),
    )


def _feed_context(
    route: str,
    ticks: Sequence[int],
    fresh: Mapping[int, set[str]],
) -> dict[str, object]:
    """The shared-failure surface for a window: how present the trace was and
    whether this route's own line-group feed stayed fresh across it. A window
    with a stale/absent group is common-mode blindness, not a functional
    disagreement between the wait signal and the others."""
    group = ROUTE_FEED_GROUP.get(route)
    present = [t for t in ticks if t in fresh]
    group_fresh = [t for t in present if group is None or group in fresh[t]]
    return {
        "coverage": round(len(present) / len(ticks), 3) if ticks else 0.0,
        "group_fresh_share": round(len(group_fresh) / len(present), 3)
        if present
        else 0.0,
        "group": group,
    }


def severity_report(
    waits: Mapping[tuple[str, str], Sequence[TickWait]],
    baseline: Mapping[tuple[str, str, str], WaitCell],
    movement: Mapping[tuple[str, int], str],
    has_delays: Mapping[tuple[str, int], bool],
    fresh: Mapping[int, set[str]],
) -> dict[str, object]:
    """Tier prevalence, per-route episode counts, dwell distribution, and the
    overlap/disagreement of the wait severity with the movement condition and the
    alert feed's Delays — with the feed context on the disagreements, so
    common-mode blindness is separated from real functional disagreement."""
    tiers = severity_series(waits, baseline)
    n = len(tiers)
    if n == 0:
        return {"n_ticks": 0}  # no cell published a baseline (too few nights)
    prevalence = {
        t: round(sum(1 for _d, tier in tiers.values() if tier == t) / n, 4)
        for t in (0, 1, 2)
    }

    eps1 = episodes(waits, baseline, min_tier=1)
    eps2 = episodes(waits, baseline, min_tier=2)
    per_route: dict[str, dict[str, int]] = {}
    for e in eps1:
        per_route.setdefault(e.route, {"tier1_episodes": 0, "tier2_episodes": 0})
        per_route[e.route]["tier1_episodes"] += 1
    for e in eps2:
        per_route.setdefault(e.route, {"tier1_episodes": 0, "tier2_episodes": 0})
        per_route[e.route]["tier2_episodes"] += 1
    dwell1 = sorted(e.n_ticks * TICK_SECONDS // 60 for e in eps1)
    dwell2 = sorted(e.n_ticks * TICK_SECONDS // 60 for e in eps2)

    # overlap on tier>=2 ticks
    severe = [(r, t) for (r, t), (_d, tier) in tiers.items() if tier >= 2]
    mv_not_normal = sum(1 for k in severe if movement.get(k) not in (None, "normal"))
    delays = sum(1 for k in severe if has_delays.get(k))
    headway_only = [
        k
        for k in severe
        if movement.get(k) in (None, "normal") and not has_delays.get(k)
    ]

    # feed context for the headway-only (disagreement) windows, grouped into
    # contiguous per-route runs so a window is one episode not many ticks
    disagreement_windows: list[dict[str, object]] = []
    for route, keys in groupby(sorted(headway_only), key=lambda k: k[0]):
        ticks = sorted(t for _r, t in keys)
        for _, run in groupby(
            enumerate(ticks), key=lambda it: it[1] - it[0] * TICK_SECONDS
        ):
            span = [t for _i, t in run]
            disagreement_windows.append(
                {
                    "route": route,
                    "start": span[0],
                    "n_ticks": len(span),
                    **_feed_context(route, span, fresh),
                }
            )

    return {
        "n_ticks": n,
        "prevalence": prevalence,
        "per_route_episodes": per_route,
        "dwell_min_tier1": {
            "n": len(dwell1),
            "median": dwell1[len(dwell1) // 2] if dwell1 else None,
            "p90": dwell1[int(len(dwell1) * 0.9)] if dwell1 else None,
            "max": dwell1[-1] if dwell1 else None,
        },
        "dwell_min_tier2": {
            "n": len(dwell2),
            "median": dwell2[len(dwell2) // 2] if dwell2 else None,
            "p90": dwell2[int(len(dwell2) * 0.9)] if dwell2 else None,
            "max": dwell2[-1] if dwell2 else None,
        },
        "severe_ticks": len(severe),
        "severe_overlap_movement_not_normal": mv_not_normal,
        "severe_overlap_alert_delays": delays,
        "severe_headway_only": len(headway_only),
        "disagreement_windows": sorted(
            disagreement_windows, key=lambda w: -cast(int, w["n_ticks"])
        )[:20],
    }


# --- end-to-end run over the R2 archive ---


def _reconstruct_waits(
    cfg: R2Config,
    client: S3Client,
    reference_stops: Mapping[tuple[str, str], ReferenceStop],
    start: date,
    end: date,
) -> dict[tuple[str, str], list[TickWait]]:
    """Stream the trace archive day by day (~1440 objects/day) into per-reference-
    stop arrivals, then headways, then tick-aligned waits. Per-day so the full
    window's rows never sit in memory at once."""
    from training.load_r2 import date_range, fetch_objects, list_keys
    from training.trace import Arrival, arrivals_from_trace

    accum: dict[tuple[str, str], list[Arrival]] = defaultdict(list)
    covered: list[int] = []
    for d in date_range(start, end):
        keys = list_keys(client, cfg.bucket, f"archive/trace/{d.isoformat()}/")  # type: ignore[attr-defined]
        covered.extend(int(k.rsplit("/", 1)[-1].split(".")[0]) for k in keys)
        bodies = fetch_objects(client, cfg.bucket, keys)  # type: ignore[attr-defined]
        for key, arrs in reference_arrivals(
            arrivals_from_trace(bodies), reference_stops
        ).items():
            accum[key].extend(arrs)
    for series in accum.values():
        series.sort(key=lambda a: a.at)
    events = headway_events(accum, sorted(covered))
    return {k: tick_aligned_waits(v) for k, v in events.items()}


def _truths(
    cfg: R2Config,
    client: S3Client,
    start: date,
    end: date,
    fit_days: int,
) -> tuple[
    dict[tuple[str, int], str],
    dict[tuple[str, int], str],
    dict[tuple[str, int], bool],
    dict[int, set[str]],
]:
    """Movement, supply, and alert truths over the window, movement fit on a
    leading window that ends before it (held out)."""
    from training.degradation_label import BIN_FN, degraded_now_truth
    from training.gtfs_static import load_topology, through_stops
    from training.load_r2 import (
        SERVICE_MIN_NIGHTS,
        build_movement_series_by_direction,
        build_movement_truth,
        build_service_series,
        build_tick_observations,
        compute_advance_baseline,
        compute_baseline,
        fetch_alert_versions,
        fetch_trip_update_metrics,
        fetch_vehicle_metrics,
    )

    successors, _ = load_topology()
    through = through_stops(successors)

    def stopf(r: str, di: str, s: str) -> bool:
        return (r, di, s) in through

    fit_start = start - timedelta(days=fit_days)
    fit = fetch_vehicle_metrics(
        cfg, start_date=fit_start, end_date=start - timedelta(days=1), client=client
    )  # type: ignore[arg-type]
    adv = compute_advance_baseline(
        build_movement_series_by_direction(fit, counts_from_stop=stopf)
    )
    score = fetch_vehicle_metrics(cfg, start_date=start, end_date=end, client=client)  # type: ignore[arg-type]
    movement = build_movement_truth(
        score, movement_baseline=adv, counts_from_stop=stopf
    )
    fresh: dict[int, set[str]] = {}
    for b in score:
        fresh[(int(b.get("observed_at") or 0) // TICK_SECONDS) * TICK_SECONDS] = set(
            b.get("fresh_feeds") or []
        )

    svc = fetch_trip_update_metrics(
        cfg, start_date=fit_start, end_date=end, client=client
    )  # type: ignore[arg-type]
    all_series = build_service_series(svc)
    supply_base = compute_baseline(
        all_series, bin_fn=BIN_FN, min_nights=SERVICE_MIN_NIGHTS
    )
    boundary = int(datetime.combine(start, datetime.min.time(), tzinfo=UTC).timestamp())
    supply = degraded_now_truth(
        {k: v for k, v in all_series.items() if k[1] >= boundary}, supply_base
    )

    obs = build_tick_observations(
        fetch_alert_versions(cfg, start_date=start, end_date=end, client=client)  # type: ignore[arg-type]
    )
    has_delays = {(o.route_id, o.tick): o.observation.has_delays for o in obs}
    return movement, supply, has_delays, fresh


def _rate_json(r: Rate) -> dict[str, object]:
    return asdict(r)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate the observed-headway wait signal: false alarms on "
        "confirmed-normal service, schedule vs typical-actual reference, and a "
        "first-pass severity tier — all night/episode bootstrapped."
    )
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument("--days", type=int, default=21)
    parser.add_argument("--fit-days", type=int, default=28)
    parser.add_argument("--gtfs-zip", default=None)
    parser.add_argument("--saturday", default=None, help="ET date for the 1/2 case")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from training.r2_client import load_config, make_client

    cfg = load_config()
    client = make_client(cfg)
    end = args.end_date or datetime.now(UTC).date()
    start = args.start_date or (end - timedelta(days=args.days - 1))

    zf = load_gtfs_zip(args.gtfs_zip)
    reference_stops = select_reference_stops(zf)
    swt = scheduled_swt(zf, reference_stops)
    print(f"reconstructing trace {start}..{end}", file=sys.stderr)
    waits = _reconstruct_waits(cfg, client, reference_stops, start, end)
    print("building movement/supply/alert truths", file=sys.stderr)
    movement, supply, has_delays, fresh = _truths(
        cfg, client, start, end, args.fit_days
    )

    normal = confirmed_normal(movement, supply)
    baseline = typical_actual_baseline(
        waits, normal_ticks=dict.fromkeys(normal, True), min_nights=HEADWAY_MIN_NIGHTS
    )

    report = {
        "window": {"start": start.isoformat(), "end": end.isoformat()},
        "reference_stops": {
            f"{k[0]}|{k[1]}": v.stop_id
            for k, v in reference_stops.items()  # type: ignore[attr-defined]
        },
        "n_headways": sum(len(v) for v in waits.values()),
        "n_baseline_cells": len(baseline),
        "confirmed_normal_ticks": len(normal),
        "gate_a_false_alarm": {
            k: _rate_json(v)
            for k, v in false_alarm_gate(
                waits, baseline, normal, n_boot=args.bootstrap, seed=args.seed
            ).items()
        },
        "gate_b_schedule_vs_typical": {
            k: _rate_json(v)
            for k, v in schedule_vs_typical(
                waits, baseline, swt, normal, n_boot=args.bootstrap, seed=args.seed
            ).items()
        },
        "gate_b_saturday": saturday_head_to_head(
            waits, baseline, swt, args.saturday or start.isoformat(), ["1", "2"]
        ),
        "gate_c_severity": severity_report(
            waits, baseline, movement, has_delays, fresh
        ),
    }
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
