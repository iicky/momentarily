"""Offline calibration harness for the movement disrupted-arm cut and debounce.

The movement classifier (worker/src/movement_state.ts, mirrored in
training.load_r2.classify_direction) turns a route's cross-tick advance fraction
into normal / disrupted against its own baseline. This harness reconstructs that
classifier over a window from the vehicle archive, sweeps its three cut
constants and the regime-clock debounce, and reports the diagnostics that decide
an operating point.

Truth, and what each number can carry
-------------------------------------
There is no dense external truth for *advance-quality* disruption. Two kinds of
reference are used, and they are NOT interchangeable:

* Structural-consistency anchors — disrupted base rate, per-tick stickiness
  P(disrupted next | disrupted now), and the implied/measured dwell. These are
  computed from the classifier's OWN calls, so they establish that an operating
  point reproduces the population structure the state-space study described
  (rare, ~14-min-sticky regimes). They are self-referential: they show
  consistency, never independent validation.

* Trip-updates corroboration — overlap of movement episodes with the
  supply-side disruptions derive_actual_recovery finds in assigned_n. This is
  independent in derivation from vehicle positions, so it is the only
  independent reference here. It is also weak: the two feeds measure different
  things (supply level vs advance quality), so overlap is low by nature and
  precision/recall against it are corroboration, not a target to maximise.

The baseline is fitted on a CLEAN earlier sub-window (causal): a sustained
outage must not lower the baseline it is later judged against. The advance
counts are through-stop filtered to match the live worker
(vehicles.ts deriveRouteMovementMetric) and the fitted baseline; scoring
unfiltered counts against a through-stop baseline reads spuriously disrupted.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

from momentarily.hmm import tod_bin
from training.load_r2 import (
    AdvanceBaseline,
    Disruption,
    build_movement_series,
    build_movement_series_by_direction,
    build_service_series,
    classify_direction,
    compute_advance_baseline,
    compute_baseline,
    derive_actual_recovery,
    fetch_objects,
    list_keys,
)
from training.movement_backfill import resolve_stop_filter
from training.r2_client import load_config, make_client
from training.regime import replay_regimes

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

TICK_SECONDS = 300
_DIRECTIONS = ("north", "south")
_ROUTE_TICK = tuple[str, int]


# --- data loading ----------------------------------------------------------


def _fetch_bodies(client: S3Client, bucket: str, prefix: str, d0: date, d1: date):
    """Fetch archived per-tick bodies over the HALF-OPEN date range [d0, d1),
    so a baseline [start, eval_start) and an eval [eval_start, eval_end) never
    share a day. (load_r2.date_range is inclusive and would overlap them.)"""
    keys: list[str] = []
    d = d0
    while d < d1:
        keys.extend(list_keys(client, bucket, f"{prefix}{d.isoformat()}/"))
        d += timedelta(days=1)
    return fetch_objects(client, bucket, keys)


@dataclass(frozen=True)
class CalibrationWindow:
    """Everything a sweep needs, decoded once. The movement series are
    through-stop filtered; the advance baseline is causal (baseline sub-window),
    and the trip-updates disruptions are the independent corroborator over the
    eval sub-window."""

    route_series: dict[_ROUTE_TICK, dict[str, int]]
    dir_series: dict[tuple[str, str, int], dict[str, int]]
    advance_baseline: dict[tuple[str, str, int], AdvanceBaseline]
    tu_disruptions: list[Disruption]
    eval_ticks: list[int]
    n_routes: int


def load_window(
    client: S3Client,
    bucket: str,
    *,
    baseline_start: date,
    eval_start: date,
    eval_end: date,
    stop_scope: str = "through",
) -> CalibrationWindow:
    """Fetch the baseline and eval sub-windows and decode the pieces a sweep
    reuses. `baseline_start`..`eval_start` (exclusive) fits the causal advance
    and assigned_n baselines; `eval_start`..`eval_end` is the window swept."""
    stop = resolve_stop_filter(stop_scope)
    if stop is None and stop_scope == "through":
        print(
            "WARNING: through-stop filter unavailable; counts are unfiltered and "
            "will not match the live worker or baseline",
            file=sys.stderr,
        )

    veh_base = _fetch_bodies(
        client, bucket, "archive/vehicles/", baseline_start, eval_start
    )
    veh_eval = _fetch_bodies(client, bucket, "archive/vehicles/", eval_start, eval_end)
    tu_base = _fetch_bodies(
        client, bucket, "archive/trip_updates/", baseline_start, eval_start
    )
    tu_eval = _fetch_bodies(
        client, bucket, "archive/trip_updates/", eval_start, eval_end
    )

    advance_baseline = compute_advance_baseline(
        build_movement_series_by_direction(veh_base, counts_from_stop=stop)
    )
    tu_baseline = compute_baseline(build_service_series(tu_base))
    tu_disruptions = derive_actual_recovery(build_service_series(tu_eval), tu_baseline)

    route_series = build_movement_series(veh_eval, counts_from_stop=stop)
    dir_series = build_movement_series_by_direction(veh_eval, counts_from_stop=stop)
    eval_ticks = sorted({tk for (_r, tk) in route_series})
    n_routes = len({r for (r, _tk) in route_series})
    return CalibrationWindow(
        route_series=route_series,
        dir_series=dir_series,
        advance_baseline=advance_baseline,
        tu_disruptions=tu_disruptions,
        eval_ticks=eval_ticks,
        n_routes=n_routes,
    )


# --- reconstruction --------------------------------------------------------


def reconstruct_calls(
    win: CalibrationWindow,
    *,
    prior_strength: float,
    disrupted_ratio: float,
    alpha: float,
    min_matched: int = 3,
) -> dict[_ROUTE_TICK, str]:
    """(route, tick) -> normal/disrupted/suspended for judgeable route-ticks,
    for one setting of the three cut constants. Mirrors deriveMovementState: a
    route with no trains is suspended; otherwise each direction is scored
    against its own (route, direction, tod_bin) baseline and the route takes the
    worse of the two; abstains where no direction is judgeable."""
    calls: dict[_ROUTE_TICK, str] = {}
    for (route, tick), rrow in win.route_series.items():
        if rrow.get("vehicles_n", 0) <= 0:
            calls[(route, tick)] = "suspended"
            continue
        tb = tod_bin(tick)
        dir_calls: list[str] = []
        for d in _DIRECTIONS:
            drow = win.dir_series.get((route, d, tick))
            if drow is None:
                continue
            call = classify_direction(
                drow.get("advanced_n", 0),
                drow.get("stalled_n", 0),
                win.advance_baseline.get((route, d, tb)),
                prior_strength=prior_strength,
                disrupted_ratio=disrupted_ratio,
                min_matched=min_matched,
                alpha=alpha,
            )
            if call is not None:
                dir_calls.append(call)
        if not dir_calls:
            continue
        calls[(route, tick)] = "disrupted" if "disrupted" in dir_calls else "normal"
    return calls


def _calls_to_ticks(calls: dict[_ROUTE_TICK, str]):
    by_tick: dict[int, dict[str, str]] = defaultdict(dict)
    for (route, tick), state in calls.items():
        by_tick[tick][route] = state
    return sorted(by_tick.items())


@dataclass(frozen=True)
class Episode:
    route: str
    start: int
    end: int
    state: str
    completed: bool  # False = still open at window end (censored)

    @property
    def dwell_min(self) -> float:
        return (self.end - self.start) / 60.0


def episodes(
    calls: dict[_ROUTE_TICK, str],
    *,
    debounce_ticks: int = 1,
    states: tuple[str, ...] = ("disrupted",),
) -> list[Episode]:
    """NOT-NORMAL episodes through the real regime clock (training.regime), so
    the reconstruction debounces exactly as the Worker would. Completed episodes
    come from committed changes; regimes still open at the window end are carried
    as censored (their dwell is a lower bound)."""
    ticks = _calls_to_ticks(calls)
    entries, changes = replay_regimes(ticks, debounce_ticks=debounce_ticks)
    out: list[Episode] = []
    for ch in changes:
        if ch.prev_state in states:
            out.append(
                Episode(ch.key, ch.entered_at, ch.exited_at, ch.prev_state, True)
            )
    last_tick = ticks[-1][0] if ticks else 0
    for key, ent in entries.items():
        if ent.state in states:
            out.append(
                Episode(key, ent.entered_at, last_tick + TICK_SECONDS, ent.state, False)
            )
    return out


# --- diagnostics -----------------------------------------------------------


def persistence(calls: dict[_ROUTE_TICK, str]) -> dict[str, float | int | None]:
    """Structural-consistency anchors from the classifier's own calls (NOT an
    independent check): disrupted base rate and the one-step transition
    probabilities over genuinely adjacent judged ticks."""
    by_route: dict[str, dict[int, str]] = defaultdict(dict)
    for (route, tick), state in calls.items():
        by_route[route][tick] = state
    n = n_dis = dd = dn = nd = nn = 0
    for timeline in by_route.values():
        for tick, state in timeline.items():
            n += 1
            n_dis += state == "disrupted"
            nxt = timeline.get(tick + TICK_SECONDS)
            if nxt is None:
                continue
            if state == "disrupted":
                dd += nxt == "disrupted"
                dn += nxt != "disrupted"
            elif state == "normal":
                nd += nxt == "disrupted"
                nn += nxt != "disrupted"
    p_stay = dd / (dd + dn) if (dd + dn) else None
    return {
        "base_rate": n_dis / n if n else None,
        "p_dis_given_dis": p_stay,
        "p_dis_given_normal": nd / (nd + nn) if (nd + nn) else None,
        "implied_mean_dwell_min": (TICK_SECONDS / 60.0) / (1 - p_stay)
        if p_stay not in (None, 1.0)
        else None,
        "adjacent_disrupted_pairs": dd + dn,
    }


def churn(calls: dict[_ROUTE_TICK, str], within_ticks: int = 3) -> int:
    """User-facing flicker: disrupted -> not -> disrupted oscillations inside a
    short window on one route. The count a debounce would exist to reduce."""
    by_route: dict[str, dict[int, str]] = defaultdict(dict)
    for (route, tick), state in calls.items():
        by_route[route][tick] = state
    osc = 0
    for timeline in by_route.values():
        ticks = sorted(timeline)
        for i in range(len(ticks) - 2):
            a, b, c = timeline[ticks[i]], timeline[ticks[i + 1]], timeline[ticks[i + 2]]
            if (
                ticks[i + 2] - ticks[i] <= within_ticks * TICK_SECONDS
                and a == "disrupted"
                and b != "disrupted"
                and c == "disrupted"
            ):
                osc += 1
    return osc


def dwell_summary(eps: list[Episode]) -> dict[str, float | int | None]:
    completed = sorted(e.dwell_min for e in eps if e.completed)
    if not completed:
        return {"n": len(eps), "completed": 0, "mean_min": None, "median_min": None}
    return {
        "n": len(eps),
        "completed": len(completed),
        "mean_min": statistics.mean(completed),
        "median_min": statistics.median(completed),
        "single_tick_frac": sum(1 for d in completed if d <= TICK_SECONDS / 60.0)
        / len(completed),
        "p90_min": completed[int(len(completed) * 0.9)],
        "max_min": completed[-1],
    }


def corroborate_tu(
    eps: list[Episode],
    tu_disruptions: list[Disruption],
    *,
    grace_ticks: int = 1,
) -> dict[str, float | None]:
    """Overlap of movement episodes with the independent trip-updates
    disruptions. Weak by nature (the feeds measure different things) — read as
    corroboration of not being spurious, not as a fitness target."""
    tu_by: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for d in tu_disruptions:
        tu_by[d.route].append((d.start_tick, d.recovered_tick))
    grace = grace_ticks * TICK_SECONDS

    matched_ep = 0
    for e in eps:
        for ts, te in tu_by.get(e.route, []):
            if e.start < te + grace and e.end > ts - grace:
                matched_ep += 1
                break

    ep_by: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for e in eps:
        ep_by[e.route].append((e.start, e.end))
    matched_tu = 0
    for d in tu_disruptions:
        for s, en in ep_by.get(d.route, []):
            if s < d.recovered_tick + grace and en > d.start_tick - grace:
                matched_tu += 1
                break

    n_ep, n_tu = len(eps), len(tu_disruptions)
    return {
        "precision": matched_ep / n_ep if n_ep else None,
        "recall": matched_tu / n_tu if n_tu else None,
    }


@dataclass(frozen=True)
class SweepRow:
    prior_strength: float
    disrupted_ratio: float
    alpha: float
    debounce_ticks: int
    base_rate: float | None
    stickiness: float | None
    n_episodes: int
    mean_dwell_min: float | None
    median_dwell_min: float | None
    single_tick_frac: float | None
    tu_precision: float | None
    tu_recall: float | None
    churn: int


def sweep(
    win: CalibrationWindow,
    *,
    prior_strengths: tuple[float, ...] = (5, 8, 12),
    disrupted_ratios: tuple[float, ...] = (0.4, 0.5, 0.6),
    alphas: tuple[float, ...] = (0.01, 0.05, 0.10),
    debounce_ticks: tuple[int, ...] = (1,),
) -> list[SweepRow]:
    rows: list[SweepRow] = []
    for ps in prior_strengths:
        for dr in disrupted_ratios:
            for al in alphas:
                calls = reconstruct_calls(
                    win, prior_strength=ps, disrupted_ratio=dr, alpha=al
                )
                pers = persistence(calls)
                ch = churn(calls)
                for db in debounce_ticks:
                    eps = episodes(calls, debounce_ticks=db)
                    dw = dwell_summary(eps)
                    co = corroborate_tu(eps, win.tu_disruptions)
                    rows.append(
                        SweepRow(
                            prior_strength=ps,
                            disrupted_ratio=dr,
                            alpha=al,
                            debounce_ticks=db,
                            base_rate=pers["base_rate"],
                            stickiness=pers["p_dis_given_dis"],
                            n_episodes=len(eps),
                            mean_dwell_min=dw["mean_min"],
                            median_dwell_min=dw["median_min"],
                            single_tick_frac=dw.get("single_tick_frac"),
                            tu_precision=co["precision"],
                            tu_recall=co["recall"],
                            churn=ch,
                        )
                    )
    return rows


# --- CLI -------------------------------------------------------------------


def _fmt(x: Any, nd: int = 3) -> str:
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Movement cut/debounce calibration sweep")
    p.add_argument("--baseline-start", type=date.fromisoformat, required=True)
    p.add_argument(
        "--eval-start",
        type=date.fromisoformat,
        required=True,
        help="baseline is [baseline-start, eval-start); eval is [eval-start, eval-end)",
    )
    p.add_argument("--eval-end", type=date.fromisoformat, required=True)
    p.add_argument("--stop-scope", default="through", choices=("through", "all"))
    p.add_argument("--debounce", type=int, nargs="+", default=[1, 2])
    args = p.parse_args(argv)

    cfg = load_config()
    client = make_client(cfg)
    win = load_window(
        client,
        cfg.bucket,
        baseline_start=args.baseline_start,
        eval_start=args.eval_start,
        eval_end=args.eval_end,
        stop_scope=args.stop_scope,
    )
    days = (win.eval_ticks[-1] - win.eval_ticks[0]) / 86400 if win.eval_ticks else 0
    print(
        f"window: baseline {args.baseline_start}..{args.eval_start} "
        f"eval {args.eval_start}..{args.eval_end} ({days:.1f}d, {win.n_routes} routes), "
        f"advance-baseline cells={len(win.advance_baseline)}, "
        f"trip-updates disruptions={len(win.tu_disruptions)}"
    )
    rows = sweep(win, debounce_ticks=tuple(args.debounce))
    header = (
        f"{'ps':>4} {'ratio':>5} {'alpha':>5} {'db':>3} | {'base%':>6} {'stick':>6} "
        f"{'eps':>4} {'meanDw':>6} {'medDw':>5} {'1tick':>5} | {'tuPrec':>6} {'tuRec':>6} {'churn':>5}"
    )
    print(header)
    for r in sorted(
        rows,
        key=lambda r: (r.prior_strength, r.disrupted_ratio, r.alpha, r.debounce_ticks),
    ):
        print(
            f"{_fmt(r.prior_strength):>4} {_fmt(r.disrupted_ratio, 2):>5} {_fmt(r.alpha, 2):>5} "
            f"{r.debounce_ticks:>3} | {(r.base_rate or 0) * 100:>6.3f} {_fmt(r.stickiness):>6} "
            f"{r.n_episodes:>4} {_fmt(r.mean_dwell_min, 1):>6} {_fmt(r.median_dwell_min, 1):>5} "
            f"{_fmt(r.single_tick_frac):>5} | {_fmt(r.tu_precision):>6} {_fmt(r.tu_recall):>6} {r.churn:>5}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
