"""Grade the segment throughput branch before it ships: coverage, and whether
the new coverage is worth having.

WHAT THIS MEASURES

Two arms of the SAME classifier over the SAME archive, differing in exactly one
thing — whether `state/segment_params.json` carries the per-bin traversal-rate
fit (`lam`). `segment_replay.without_throughput` strips it; nothing else changes,
so every difference below is attributable to the branch.

  1. COVERAGE. Judged cells per tick, and how much of the baselined cell set ever
     gets an opinion at all. This is the number the epic exists to move.
  2. WORTH. Detection of independently-labelled disruption episodes, and false
     alarms on independently-labelled normal stretches. Coverage that only adds
     noise is not progress, and this is what says which it is.

TRUTH, AND WHY IT IS THIS TRUTH

`training.degradation_label`'s assigned_n label, unmodified: a route running
under half its own (route, schedule_bin) median dispatched-train count, debounced
with hysteresis. That module's own docstring records why the severe-alert truth
cannot grade a movement model (flat 11-13% prevalence in every hour of the day,
4 onsets in a week) and why assigned_n can: it is orthogonal IN DERIVATION to
both the alerts feed and the vehicle positions this classifier reads. Same feed
family, so independent-in-derivation rather than independent-in-source — stated
here because it bounds what the numbers below can claim.

EPISODES, NOT TICKS

Both rates bootstrap over EPISODES. journal.md 2026-08-22 measured what tick
counts hide: six consecutive 7-day windows with indistinguishable tick counts
(~58k) held between 5 and 89 independent episodes, a 17.8x swing in real
evidence behind a flat advertised n. A tick-level CI here would be a ~2000x
overstatement of the sample.

CAUSALITY

`--fit-days` of the window fit the baseline and the rates; the remainder is
scored. Nothing in the scored span contributes to the model scoring it.

Run:
  murk exec -- uv run python -m training.segment_coverage --days 21 --fit-days 14
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from training.degradation_label import (
    AT_BASELINE_LOOKBACK_TICKS,
    BIN_FN,
    build_labels,
)
from training.gtfs_static import load_topology, through_stops
from training.load import TICK_SECONDS
from training.load_r2 import (
    Disruption,
    StopFilter,
    build_segment_baseline,
    build_segment_throughput,
    build_service_series,
    compute_baseline,
    fetch_trip_update_metrics,
    fetch_vehicle_metrics,
    throughput_to_json,
)
from training.r2_client import load_config, make_client
from training.segment_replay import (
    SHIPPED,
    Policy,
    replay,
    tick_inputs,
    without_throughput,
)

# Bootstrap resamples for the episode-level CIs. 2000 is enough for a stable
# 95% percentile interval at the episode counts this label produces (tens),
# and cheap enough to run inside the report.
BOOTSTRAP_N = 2000

# A normal stretch has to be at least this long to serve as a negative unit —
# the same 30-minute scale degradation_label uses for its acute onset window and
# its at-baseline lookback, so a "quiet period" here is the same size of thing as
# an event.
MIN_NORMAL_RUN_TICKS = 6

# The competing way to buy coverage: widen the accumulator's window instead of
# giving it an expectation to test against. These are the values shipped on the
# feat/segment-status-map branch, measured there at 42.28% of cells judged per
# tick against a 1.26% status quo, selected against a route-level severity recall
# proxy whose absolute values are all sub-1%, a caveat that branch's own
# docstring records. They are
# replayed here so the two axes can be compared on one statistic, and crossed so
# the report can say whether they compose.
#
# The floor moves with the decay because that branch measured 1/2/3 tying exactly
# at every decay: movement_state.classifyAdvance's own MIN_MATCHED_TRIPS=3
# already rejects anything thinner, so a lower floor is cosmetic.
WIDE_DECAY = 0.94
WIDE_FLOOR = 3


@dataclass(frozen=True)
class Unit:
    """One graded episode or normal stretch, reduced to what the rates need.

    `share_sum` sums the route's disrupted-cell SHARE over its scored ticks —
    disrupted cells over judged cells, per tick. It is the statistic with power
    here: at ~1% of cell-ticks reading disrupted and ~60 judged cells per route,
    "does any cell on the route read disrupted" saturates near 40% by base rate
    alone and can barely discriminate anything. The share can.
    """

    alarmed_ticks: int
    ticks: int
    share_sum: float


def _boot_rates(
    units: Sequence[Unit],
    *,
    n: int = BOOTSTRAP_N,
    seed: int = 0,
) -> dict[str, float | int]:
    """Three statistics over `units`, with a CLUSTER bootstrap resampling whole
    units.

      mean_share — pooled disrupted-cell share per scored tick. The effect size:
                   compare the episode arm against the normal arm and the ratio
                   is how much more of a route reads disrupted while it is
                   genuinely degraded.
      tick_rate  — pooled ticks with at least one disrupted cell, over scored
                   ticks. Exposure-matched, unlike unit_rate, which matters
                   because episodes are mostly short and quiet stretches mostly
                   long; an any-tick-in-the-unit rate hands the longer population
                   far more chances to fire and can invert the comparison on
                   exposure alone.
      unit_rate  — share of units with at least one such tick. Not
                   exposure-matched, reported because it is the one an operator
                   experiences: did an alarm appear during this event.

    The bootstrap resamples UNITS, not ticks — ticks inside an episode are the
    same event observed repeatedly, and a tick-level interval would overstate the
    evidence by orders of magnitude (journal.md 2026-08-22: a flat 58k tick count
    hid a 17.8x swing in independent episodes).

    Empty input reports the counts and no rates, rather than zeros that read like
    measurements.
    """
    if not units:
        return {"n_units": 0, "n_ticks": 0}
    hit_ticks = sum(u.alarmed_ticks for u in units)
    total_ticks = sum(u.ticks for u in units)
    hit_units = sum(1 for u in units if u.alarmed_ticks > 0)
    share_sum = sum(u.share_sum for u in units)
    rng = random.Random(seed)
    share_draws: list[float] = []
    tick_draws: list[float] = []
    unit_draws: list[float] = []
    for _ in range(n):
        sample = [rng.choice(units) for _ in range(len(units))]
        total = sum(u.ticks for u in sample)
        share_draws.append(sum(u.share_sum for u in sample) / total if total else 0.0)
        tick_draws.append(
            sum(u.alarmed_ticks for u in sample) / total if total else 0.0
        )
        unit_draws.append(sum(1 for u in sample if u.alarmed_ticks > 0) / len(sample))
    share_draws.sort()
    tick_draws.sort()
    unit_draws.sort()
    lo, hi = int(0.025 * n), min(n - 1, int(0.975 * n))
    return {
        "n_units": len(units),
        "n_ticks": total_ticks,
        "alarmed_units": hit_units,
        "alarmed_ticks": hit_ticks,
        "mean_share": share_sum / total_ticks if total_ticks else 0.0,
        "mean_share_ci_low": share_draws[lo],
        "mean_share_ci_high": share_draws[hi],
        "tick_rate": hit_ticks / total_ticks if total_ticks else 0.0,
        "tick_rate_ci_low": tick_draws[lo],
        "tick_rate_ci_high": tick_draws[hi],
        "unit_rate": hit_units / len(units),
        "unit_rate_ci_low": unit_draws[lo],
        "unit_rate_ci_high": unit_draws[hi],
    }


def normal_runs(
    labels: Mapping[tuple[str, int], str],
    *,
    min_ticks: int = MIN_NORMAL_RUN_TICKS,
) -> list[tuple[str, int, int]]:
    """Maximal per-route runs of the 'normal' label, as (route, first, last+1)
    tick bounds — the negative units the false-alarm rate bootstraps over.

    A run breaks on a not-normal label AND on a missing tick: an unjudgeable gap
    is not evidence that normal service continued across it. Runs shorter than
    `min_ticks` are dropped so a two-tick sliver between two events cannot count
    as a quiet period.
    """
    by_route: dict[str, list[tuple[int, str]]] = {}
    for (route, tick), label in labels.items():
        by_route.setdefault(route, []).append((tick, label))

    out: list[tuple[str, int, int]] = []
    for route, points in by_route.items():
        points.sort()
        start: int | None = None
        prev: int | None = None
        for tick, label in points:
            contiguous = prev is not None and tick - prev == TICK_SECONDS
            if label == "normal" and (start is None or contiguous):
                if start is None:
                    start = tick
            else:
                if start is not None and prev is not None:
                    out.append((route, start, prev + TICK_SECONDS))
                start = tick if label == "normal" else None
            prev = tick
        if start is not None and prev is not None:
            out.append((route, start, prev + TICK_SECONDS))
    span = min_ticks * TICK_SECONDS
    return [r for r in out if r[2] - r[1] >= span]


def _route_shares_by_tick(
    calls: Sequence[tuple[int, Mapping[str, str]]],
) -> dict[int, dict[str, float]]:
    """tick -> route -> the share of that route's TESTABLE cells reading
    disrupted.

    The assigned_n label speaks in routes, so the per-cell calls have to be lifted
    to the route before they can be compared with it. Two choices in that lift,
    both load-bearing:

    Share, not a boolean. "Any cell disrupted" saturates by base rate: at ~1% of
    cell-ticks disrupted and ~60 cells on a route, an any-cell alarm fires ~40% of
    the time on nothing at all, which leaves it almost no power to discriminate.

    Testable, not judged. A 'quiet' call is the classifier saying it HAS no power
    on that cell right now — the window expects too little for absence to mean
    anything. Counting those in the denominator would dilute the share by however
    much of the network happens to be asleep (overnight, ~90% of cells), turning
    a comparison between arms into a comparison between denominators. So the
    denominator is the cells that could have gone either way: normal + disrupted.
    A route with no testable cell this tick contributes no observation, which is
    an absence of evidence rather than a normal reading.
    """
    out: dict[int, dict[str, float]] = {}
    for tick, per_cell in calls:
        testable: dict[str, int] = {}
        hit: dict[str, int] = {}
        for key, call in per_cell.items():
            if call == "quiet":
                continue
            route = key.split("|", 1)[0]
            testable[route] = testable.get(route, 0) + 1
            if call == "disrupted":
                hit[route] = hit.get(route, 0) + 1
        out[tick] = {r: hit.get(r, 0) / n for r, n in testable.items()}
    return out


def _onset_latency(
    shares_at: Mapping[int, Mapping[str, float]],
    disruptions: Sequence[Disruption],
    *,
    lead_sec: int = 0,
    clean_lookback_ticks: int = AT_BASELINE_LOOKBACK_TICKS,
) -> dict[str, Any]:
    """Time from a real onset to the surface's first disrupted call on that route,
    in minutes — the metric that prices the accumulator's window.

    `lead_sec` DEFAULTS TO ZERO, unlike scorecard.onset_latency's 30-minute
    tolerance, and that default is the whole point. This surface's alarm rate is
    high enough that on most episodes it is already firing before the onset; run
    with a lead window and the search starts inside that firing, every latency
    pins to exactly `-lead_sec`, and the median reports the boundary rather than
    any property of the classifier. Measured that way at 30 minutes, all four
    arms returned medians of -25 to -30 against a -30 floor — a censored
    distribution, not a comparison. From the onset itself the number is bounded
    below by zero and means what it says.

    `n_alarming_at_onset` is the contamination made explicit: episodes where the
    route was ALREADY disrupted at the onset tick. Their latency is zero for a
    reason that has nothing to do with detection, so they are counted here and
    excluded from the distribution.

    `fresh` is the stricter cut: episodes where the route had no disrupted call on
    any scored tick in the `clean_lookback_ticks` before the onset, so the alarm
    had to arrive. An arm with a good headline and a tiny `fresh` cohort has not
    detected anything, it has been shouting. Cohort SIZE is itself a function of
    the arm's base rate, so a quiet arm gets a big one for free — the cohorts are
    not comparable populations and the rates within them are not a head-to-head.
    """
    lookback = clean_lookback_ticks * TICK_SECONDS

    def first_alarm(route: str, onset: int, recovered: int) -> int | None:
        for tick in range(onset - lead_sec, recovered, TICK_SECONDS):
            if shares_at.get(tick, {}).get(route, 0.0) > 0:
                return tick
        return None

    def was_clean(route: str, onset: int) -> bool:
        seen = False
        for tick in range(onset - lookback, onset, TICK_SECONDS):
            share = shares_at.get(tick, {}).get(route)
            if share is None:
                continue
            seen = True
            if share > 0:
                return False
        return seen  # no scored tick at all cannot confirm cleanliness

    def summarise(rows: Sequence[float | None]) -> dict[str, Any]:
        lat = sorted(x for x in rows if x is not None)
        return {
            "n_episodes": len(rows),
            "n_detected": len(lat),
            "n_missed": len(rows) - len(lat),
            "detection_rate": len(lat) / len(rows) if rows else None,
            "median_latency_min": statistics.median(lat) if lat else None,
            "p90_latency_min": lat[min(len(lat) - 1, int(0.9 * len(lat)))]
            if lat
            else None,
        }

    clean_rows: list[float | None] = []
    fresh_rows: list[float | None] = []
    n_alarming_at_onset = 0
    for d in disruptions:
        already = shares_at.get(d.start_tick, {}).get(d.route, 0.0) > 0
        if already:
            n_alarming_at_onset += 1
        hit = first_alarm(d.route, d.start_tick, d.recovered_tick)
        latency = (hit - d.start_tick) / 60.0 if hit is not None else None
        if not already:
            clean_rows.append(latency)
        if was_clean(d.route, d.start_tick):
            fresh_rows.append(latency)
    return {
        "lead_tolerance_min": lead_sec // 60,
        "clean_lookback_min": lookback // 60,
        "n_offered": len(disruptions),
        "n_alarming_at_onset": n_alarming_at_onset,
        **summarise(clean_rows),
        "fresh": summarise(fresh_rows),
    }


def _coverage(
    calls: Sequence[tuple[int, Mapping[str, str]]],
    n_baselined: int,
) -> dict[str, Any]:
    """How much of the baselined cell set actually gets an opinion, and what the
    opinions are. `judged_per_tick` is the headline: the epic's premise is that
    it sits at ~15 against ~210 baselined cells."""
    per_tick = [len(per_cell) for _tick, per_cell in calls]
    mix: dict[str, int] = {}
    seen: set[str] = set()
    for _tick, per_cell in calls:
        for key, call in per_cell.items():
            mix[call] = mix.get(call, 0) + 1
            seen.add(key)
    return {
        "n_ticks": len(calls),
        "n_baselined_cells": n_baselined,
        "judged_per_tick_mean": statistics.fmean(per_tick) if per_tick else 0.0,
        "judged_per_tick_median": statistics.median(per_tick) if per_tick else 0.0,
        "cells_judged_at_least_once": len(seen),
        "share_of_baselined_cells_judged": (
            len(seen) / n_baselined if n_baselined else 0.0
        ),
        "call_mix_cell_ticks": dict(sorted(mix.items())),
    }


def grade(
    calls: Sequence[tuple[int, Mapping[str, str]]],
    disruptions: Sequence[Disruption],
    runs: Sequence[tuple[str, int, int]],
    n_baselined: int,
    *,
    bootstrap: int = BOOTSTRAP_N,
    seed: int = 0,
) -> dict[str, Any]:
    """One arm's report: coverage, episode detection, and false alarms on normal
    stretches. Pure — every R2 read lives in `main`, so this is unit-testable on
    synthetic calls.

    A unit's alarmed-tick count only counts ticks the replay actually classified:
    the label's clock and the vehicle archive's clock can disagree about which
    ticks exist, and charging a unit for a tick that was never scored would
    depress the tick rate by however much the two archives differ.
    """
    shares_at = _route_shares_by_tick(calls)

    def unit(route: str, start: int, end: int) -> Unit:
        shares = [
            shares_at[t][route]
            for t in range(start, end, TICK_SECONDS)
            if t in shares_at and route in shares_at[t]
        ]
        return Unit(
            alarmed_ticks=sum(1 for s in shares if s > 0),
            ticks=len(shares),
            share_sum=sum(shares),
        )

    detected = [unit(d.route, d.start_tick, d.recovered_tick) for d in disruptions]
    false_alarms = [unit(route, start, end) for route, start, end in runs]
    gradeable_episodes = [u for u in detected if u.ticks > 0]
    gradeable_runs = [u for u in false_alarms if u.ticks > 0]
    return {
        "coverage": _coverage(calls, n_baselined)
        | {
            # How much of the TRUTH the arm can even be scored on. A cell set
            # nothing testable ever lands in makes an episode unscoreable, and
            # that is its own kind of coverage failure.
            "gradeable_episodes": len(gradeable_episodes),
            "episodes_offered": len(detected),
            "gradeable_normal_runs": len(gradeable_runs),
            "normal_runs_offered": len(false_alarms),
        },
        "episode_detection": _boot_rates(gradeable_episodes, n=bootstrap, seed=seed),
        "normal_run_false_alarms": _boot_rates(
            gradeable_runs, n=bootstrap, seed=seed + 1
        ),
        "onset_latency": _onset_latency(shares_at, disruptions),
    }


def _stop_filter(through: frozenset[tuple[str, str, str]] | None) -> StopFilter | None:
    """The same through-stop restriction write_segment_params applies, so the
    graded fit describes the stop set the Worker would actually be handed."""
    if through is None:
        return None
    return lambda route, direction, frm: (route, direction, frm) in through


def fit_params(
    bodies: list[dict[str, Any]],
    through: frozenset[tuple[str, str, str]] | None,
) -> dict[str, Any]:
    """A segment_params-shaped doc fitted on `bodies` — the same p0 baseline,
    adjacency and `lam` rates write_segment_params publishes, minus the parts the
    classifier never reads (route_stops, provenance).

    Adjacency is synthesised from the baselined cells rather than fetched: the
    classifier only checks that an entry EXISTS and reads `.to` for the station
    roll-up, which this grade does not score. Using the fitted cell set keeps the
    two arms scoring the same cells.
    """
    stop_filter = _stop_filter(through)
    baseline = build_segment_baseline(bodies, counts_from_stop=stop_filter)
    rates, exposure = build_segment_throughput(bodies, counts_from_stop=stop_filter)
    lam = throughput_to_json(rates)
    cells: dict[str, Any] = {}
    for key, cell in baseline.items():
        joined = "|".join(key)
        entry: dict[str, Any] = {"p0": round(cell.p0, 6), "n": cell.n}
        cell_lam = lam.get(joined)
        if cell_lam is not None:
            entry["lam"] = cell_lam
        cells[joined] = entry
    return {
        "schema_version": "1",
        "cells": cells,
        "adjacency": {key: {"to": "", "source": "gtfs_static"} for key in cells},
        "throughput": {
            "bin": "schedule_bin",
            "min_ticks": 20,
            "ticks": dict(sorted(exposure.items())),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Grade the segment throughput branch: judged-cell coverage "
        "before/after, plus episode-bootstrapped detection and false alarms "
        "against the independent assigned_n degradation label."
    )
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument("--days", type=int, default=21, help="total trailing window")
    parser.add_argument(
        "--fit-days",
        type=int,
        default=14,
        help="leading days used to fit p0 and the rates; the rest is scored",
    )
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_N)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    today = datetime.now(UTC).date()
    end = args.end_date or today
    start = args.start_date or (end - timedelta(days=args.days - 1))
    fit_end = start + timedelta(days=args.fit_days - 1)
    if fit_end >= end:
        print("fit window leaves nothing to score", file=sys.stderr)
        return 1
    score_start = fit_end + timedelta(days=1)

    cfg = load_config()
    client = make_client(cfg)

    try:
        successors, _patterns = load_topology()
        through = through_stops(successors)
        topology_source = "gtfs_static"
    except Exception as exc:
        print(
            f"gtfs static topology unavailable ({exc}); scoring every stop",
            file=sys.stderr,
        )
        through, topology_source = None, "observed"

    print(f"fitting on vehicle archive {start}..{fit_end}", file=sys.stderr)
    fit_bodies = fetch_vehicle_metrics(
        cfg, start_date=start, end_date=fit_end, client=client
    )
    params = fit_params(fit_bodies, through)
    print(
        f"{len(fit_bodies)} fit ticks -> {len(params['cells'])} cells, "
        f"{len(params['throughput']['ticks'])} fitted bins",
        file=sys.stderr,
    )

    print(f"scoring on vehicle archive {score_start}..{end}", file=sys.stderr)
    score_bodies = fetch_vehicle_metrics(
        cfg, start_date=score_start, end_date=end, client=client
    )
    ticks = tick_inputs(score_bodies, counts_from_stop=_stop_filter(through))
    print(f"{len(ticks)} scored ticks", file=sys.stderr)

    # The truth baseline is fitted on the SAME leading window as the model, and
    # the label is drawn on the held-out span only. Scoring against a baseline
    # fitted on the scored span would let an episode lower the median it is
    # measured against and vanish — the self-training confound that left the
    # first 8-day segment measurement inconclusive (journal.md 2026-08-11).
    print(f"fetching trip-update metrics {start}..{end}", file=sys.stderr)
    svc_bodies = fetch_trip_update_metrics(
        cfg, start_date=start, end_date=end, client=client
    )
    all_series = build_service_series(svc_bodies)
    boundary = int(
        datetime.combine(score_start, datetime.min.time(), tzinfo=UTC).timestamp()
    )
    baseline = compute_baseline(
        {k: v for k, v in all_series.items() if k[1] < boundary}, bin_fn=BIN_FN
    )
    series = {k: v for k, v in all_series.items() if k[1] >= boundary}
    disruptions, labels = build_labels(series, baseline)
    runs = normal_runs(labels)
    print(
        f"{len(disruptions)} assigned_n episodes, {len(runs)} normal runs, "
        f"{len(baseline)} (route, schedule_bin) baseline cells fitted causally "
        f"on {start}..{fit_end}",
        file=sys.stderr,
    )

    n_baselined = len(params["cells"])
    bare = without_throughput(params)
    # The two axes the epic can buy coverage on, crossed. `window` spends onset
    # latency for coverage (a longer accumulator window means a verdict can be
    # stale by up to that window); `throughput` spends nothing but needs the
    # trainer's per-bin rates. The crossed arm is here because nothing about
    # either says they compose, and the report should say whether they do.
    arms: dict[str, tuple[dict[str, Any], Policy]] = {
        "status_quo": (bare, SHIPPED),
        "window": (bare, Policy(decay=WIDE_DECAY, min_eff_matched=WIDE_FLOOR)),
        "throughput": (params, SHIPPED),
        "both": (params, Policy(decay=WIDE_DECAY, min_eff_matched=WIDE_FLOOR)),
    }
    report: dict[str, Any] = {
        "window": {
            "fit_start": start.isoformat(),
            "fit_end": fit_end.isoformat(),
            "score_start": score_start.isoformat(),
            "score_end": end.isoformat(),
            "topology_source": topology_source,
            "n_fit_ticks": len(fit_bodies),
            "n_scored_ticks": len(ticks),
        },
        "truth": {
            "label": "assigned_n degradation (training.degradation_label)",
            "bin": "schedule_bin",
            "n_episodes": len(disruptions),
            "n_normal_runs": len(runs),
            "n_routes_with_episodes": len({d.route for d in disruptions}),
            # The two populations are not the same length, which is why the
            # exposure-matched tick rate is the comparable number and the
            # unit rate is not.
            "episode_ticks_median": statistics.median(
                (d.recovered_tick - d.start_tick) // TICK_SECONDS for d in disruptions
            )
            if disruptions
            else 0,
            "normal_run_ticks_median": statistics.median(
                (end_t - start_t) // TICK_SECONDS for _r, start_t, end_t in runs
            )
            if runs
            else 0,
        },
        "arms": {},
    }
    for name, (arm_params, policy) in arms.items():
        print(f"replaying arm {name} (decay={policy.decay})", file=sys.stderr)
        report["arms"][name] = {
            "policy": {
                "decay": policy.decay,
                "min_eff_matched": policy.min_eff_matched,
                "window_ticks": policy.window_ticks,
                "window_minutes": policy.window_ticks * TICK_SECONDS / 60,
                "throughput": "throughput" in arm_params,
            },
        } | grade(
            replay(ticks, arm_params, policy),
            disruptions,
            runs,
            n_baselined,
            bootstrap=args.bootstrap,
            seed=args.seed,
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
