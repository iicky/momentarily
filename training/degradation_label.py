"""Independent "route is degraded now" label from trip-updates assigned_n.

WHY: the severe-alert truth cannot grade a movement model (measured
2026-08-12). It has 11.2-13.0% prevalence in EVERY ONE of the 24 UTC
hours over a full week -- no real disruption pattern is flat across the day,
so that flatness is direct evidence the truth is chronic-standing-advisory
dominated, not incident dominated. Its acute cut (normal -> not-normal
transitions) is nearly empty: 4 onsets in a week across 4 routes. Three
movement scores all came out AUC-inverted against it. assigned_n is
orthogonal to both the alerts feed and the vehicle-position feed the movement
model is built from, so it does not inherit that coupling BY CONSTRUCTION;
`measure_degradation` / `main` below measure whether it also avoids the
FLATNESS and SPARSITY problems IN PRACTICE.

DECISION: this label is already built, not a new estimator. `load_r2.
derive_actual_recovery` already implements everything a degrade/recover call
needs -- baseline is a per-(route, tod_bin) MEDIAN (so an outage can't lower
the baseline it is later measured against), and the degrade/recover state
machine is debounced with hysteresis (recover_ratio > degrade_ratio) so a
route riding the threshold doesn't flap. What it does NOT give a caller is the
acute/chronic split the alerts truth is missing: a `Disruption` carries only
an interval's start/end, not where in that interval a movement model's "this
is a FRESH event" signal should be graded versus its "already known bad,
still bad" tail. `label_ticks` adds exactly that split on top of `Disruption`,
unmodified. Nothing here re-derives degrade/recover detection.

Three pieces:
  1. `label_ticks` -- per-tick "normal" | "acute" | "chronic" from a
     Disruption list. Acute = the first `ACUTE_ONSET_TICKS` of an interval
     (fresh collapse); chronic = the remainder (standing degradation).
     "degraded" is just "label != normal" -- acute or chronic together.
  2. `was_at_baseline` -- was a route genuinely running normally for a full
     lookback window right before an onset, a stronger claim than what
     derive_actual_recovery's own state machine guarantees about the single
     tick before start_tick (only that it wasn't already PAST the degrade
     floor).
  3. `measure_degradation` / `main` -- the report this module exists to
     produce: acute events/week and distinct routes, degraded-tick
     prevalence per UTC hour (the same check that caught the alerts truth's
     flatness), the chronic/acute split, and how many events survive the
     at-baseline-beforehand filter. Fetching from R2 lives behind `main` and
     is not exercised by the test suite; everything else here is pure and
     unit-tested on synthetic series.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from momentarily.hmm import schedule_bin
from training.load import TICK_SECONDS
from training.load_r2 import (
    Disruption,
    build_service_series,
    compute_baseline,
    derive_actual_recovery,
    fetch_trip_update_metrics,
)
from training.r2_client import load_config, make_client

# schedule_bin's own zone — the onset diagnostic has to read the same clock the
# bins are cut on.
_ET = ZoneInfo("America/New_York")

# --- thresholds passed to derive_actual_recovery ---------------------------
#
# Pinned as explicit constants here rather than inherited from
# derive_actual_recovery's own defaults, so a future retune of that callee
# can't silently retune this label without a visible diff in this file too.

# A route running under half its own (route, tod_bin) median is not a
# rounding blip -- derive_actual_recovery's own default, already validated
# there (see load_r2.py's Disruption docstring).
DEGRADE_RATIO = 0.5

# Recovery requires clearing a HIGHER bar than the degrade floor (hysteresis)
# so a route sitting near the boundary doesn't flap acute/chronic tick to
# tick. derive_actual_recovery's own default.
RECOVER_RATIO = 0.8

# Consecutive ticks required to confirm a dip or a recovery before it commits
# -- 10 minutes at the 5-min cron cadence. A single dropped tick (one train
# briefly unassigned, a feed hiccup) must not read as a collapse.
DEGRADE_DEBOUNCE_TICKS = 2

# --- acute/chronic split -----------------------------------------------

# Ticks from a Disruption's start_tick (inclusive) still counted "acute" --
# the fresh-collapse onset window a movement model's leading signal should be
# graded against. 30 minutes: a bright line at the same operational scale as
# the at-baseline lookback below, not a quantile of measured disruption
# durations, so it can't be read as tuned to the yield it produces.
ACUTE_ONSET_TICKS = 6

# --- "genuinely at baseline before the collapse" filter ---------------------

# Ticks of unbroken lookback required immediately before start_tick, all at
# or above AT_BASELINE_RATIO, before calling a route "was at baseline" --
# stronger than what derive_actual_recovery's state machine itself guarantees
# about the single tick before start_tick (only that it hadn't already
# crossed DEGRADE_RATIO, which leaves room for a route limping along just
# above that floor). Same 30-minute scale as ACUTE_ONSET_TICKS.
AT_BASELINE_LOOKBACK_TICKS = 6

# Same bar RECOVER_RATIO uses to call a route recovered -- "at baseline
# beforehand" and "recovered" are the same claim about the ratio, so reuse
# the number rather than inventing a second one.
AT_BASELINE_RATIO = RECOVER_RATIO


# --- baseline granularity ---------------------------------------------------

# This label bins by ET (weekday|weekend, hour) -- momentarily.hmm.schedule_bin,
# already mirrored in the Worker -- NOT by the HMM's five tod_bins.
#
# Why: a tod_bin spans 4-6 hours and its median is set by its busiest core, so
# the quiet edge of a block reads as a collapse against it. Measured over
# 2026-08-03..08-13 with tod_bin, degraded-tick prevalence peaked at 26.8% in
# UTC hour 10 (= ET 06:00, the first hour of the 06:00-09:59 rush block) and
# 18.9% in UTC hour 03 (= ET 23:00, the last hour of the 20:00-23:59 block),
# against 6-11% everywhere else. Neither is a disruption pattern; both are the
# block edge.
#
# The weekday/weekend split rides along for free and fixes a second confound:
# a Sunday's real service level judged against a median dominated by weekdays.
#
# tod_bin stays exactly as it is -- the HMM emission channel scores the live
# service ratio against the (route, tod_bin) baseline shipped in params.json,
# and that pairing has to keep agreeing with the Worker.
BIN_FN = schedule_bin


def label_ticks(
    series: dict[tuple[str, int], int],
    baseline: dict[tuple[str, str], float],
    disruptions: Sequence[Disruption],
    *,
    onset_ticks: int = ACUTE_ONSET_TICKS,
) -> dict[tuple[str, int], str]:
    """(route, tick) -> "normal" | "acute" | "chronic" for every tick in
    `series` judgeable against a baseline. A (route, BIN_FN) cell with no
    baseline is omitted entirely -- "can't judge", the same contract
    compute_baseline and derive_movement_state already use, never silently
    defaulted to "normal".

    `disruptions` is derive_actual_recovery's own output, unmodified: this
    function adds nothing to WHEN a route is degraded, only splits its
    already-debounced [start_tick, recovered_tick) interval into "acute"
    (the first `onset_ticks` ticks from start_tick -- a fresh collapse) and
    "chronic" (everything after -- a route already known to be down, not a
    new event)."""
    labels: dict[tuple[str, int], str] = {
        (route, tick): "normal"
        for route, tick in series
        if baseline.get((route, BIN_FN(tick))) is not None
    }
    onset_span = onset_ticks * TICK_SECONDS
    for d in disruptions:
        onset_end = d.start_tick + onset_span
        for tick in range(d.start_tick, d.recovered_tick, TICK_SECONDS):
            if (d.route, tick) in labels:
                labels[(d.route, tick)] = "acute" if tick < onset_end else "chronic"
    return labels


def was_at_baseline(
    series: dict[tuple[str, int], int],
    baseline: dict[tuple[str, str], float],
    route: str,
    start_tick: int,
    *,
    lookback_ticks: int = AT_BASELINE_LOOKBACK_TICKS,
    at_baseline_ratio: float = AT_BASELINE_RATIO,
) -> bool:
    """True when every one of the `lookback_ticks` ticks immediately before
    `start_tick` had assigned_n at or above `at_baseline_ratio` x baseline --
    genuinely normal service right up to the collapse. A missing tick or
    missing baseline anywhere in the lookback fails closed: can't confirm "at
    baseline" without data to confirm it with."""
    for i in range(1, lookback_ticks + 1):
        tick = start_tick - i * TICK_SECONDS
        base = baseline.get((route, BIN_FN(tick)))
        assigned = series.get((route, tick))
        if base is None or base <= 0 or assigned is None:
            return False
        if assigned / base < at_baseline_ratio:
            return False
    return True


def _week_windows(start: date, end: date, *, days: int = 7) -> list[tuple[date, date]]:
    """[start, end] split into `days`-day bins, mirroring incidents.
    _week_windows so weekly rates read the same way across both reports. The
    last bin is whatever is left over, never padded to a full window."""
    windows: list[tuple[date, date]] = []
    cursor = start
    while cursor <= end:
        bin_end = min(cursor + timedelta(days=days - 1), end)
        windows.append((cursor, bin_end))
        cursor = bin_end + timedelta(days=1)
    return windows


def _hourly_prevalence(
    labels: dict[tuple[str, int], str],
) -> dict[int, dict[str, float]]:
    """Degraded-tick share per UTC hour (0-23), pooled across every route and
    day the label covers -- the same per-hour check that showed the
    severe-alert truth flat at 11.2-13.0% prevalence in EVERY hour, the
    signature of a chronic-standing-advisory-dominated truth rather than a
    real time-of-day incident pattern. UTC (not tod_bin's ET/5-bin split) to
    stay directly comparable to that measurement."""
    total: dict[int, int] = defaultdict(int)
    degraded: dict[int, int] = defaultdict(int)
    for (_route, tick), label in labels.items():
        hour = datetime.fromtimestamp(tick, tz=UTC).hour
        total[hour] += 1
        if label != "normal":
            degraded[hour] += 1
    return {
        hour: {
            "prevalence": degraded[hour] / total[hour],
            "degraded_ticks": degraded[hour],
            "total_ticks": total[hour],
        }
        for hour in sorted(total)
    }


def _onsets_by_et_hour(disruptions: Sequence[Disruption]) -> dict[int, int]:
    """Disruption onsets per ET clock hour (0-23).

    The bin-edge diagnostic: a baseline whose buckets are wider than an hour
    piles onsets into the first hour of each bucket, because that is where the
    real service level sits furthest below the bucket's median. Keyed on ET,
    not UTC, because the buckets are ET-local and the artifact is an artifact
    of THEIR edges."""
    counts: dict[int, int] = defaultdict(int)
    for d in disruptions:
        counts[datetime.fromtimestamp(d.start_tick, tz=_ET).hour] += 1
    return dict(sorted(counts.items()))


def build_labels(
    series: dict[tuple[str, int], int],
    baseline: dict[tuple[str, str], float],
) -> tuple[list[Disruption], dict[tuple[str, int], str]]:
    """The label itself: debounced disruption intervals and the per-tick
    acute/chronic/normal call derived from them.

    Shared by the report below and by every consumer that grades a model
    against this label, so a grade can never be scored against a differently
    tuned label than the one this module measures and publishes. The
    thresholds are this module's pinned constants, not the callee's defaults.
    """
    disruptions = derive_actual_recovery(
        series,
        baseline,
        bin_fn=BIN_FN,
        degrade_ratio=DEGRADE_RATIO,
        recover_ratio=RECOVER_RATIO,
        debounce=DEGRADE_DEBOUNCE_TICKS,
    )
    return list(disruptions), label_ticks(series, baseline, disruptions)


def degraded_now_truth(
    series: dict[tuple[str, int], int],
    baseline: dict[tuple[str, str], float],
) -> dict[tuple[str, int], str]:
    """The assigned_n label as a current-state truth map: (route, tick) ->
    "normal" | "disrupted", over judgeable ticks only.

    acute and chronic collapse to "disrupted" -- the acute/chronic split grades
    events, but a confusion matrix asks only whether the route was degraded at a
    tick. This is the independent "is this route degraded now" reference the
    severe-alert truth could not be (0pb): assigned_n is orthogonal to both the
    alerts feed and the vehicle-position feed the movement model is built from,
    so it does not inherit the chronic-standing-advisory domination that left the
    alert truth flat across every hour and unable to grade a movement signal.

    "disrupted" (not "degraded") so the values live in the same space as the
    alert truth's, and the same confusion() consumer scores both."""
    _disruptions, labels = build_labels(series, baseline)
    return {
        key: "normal" if label == "normal" else "disrupted"
        for key, label in labels.items()
    }


def measure_degradation(
    series: dict[tuple[str, int], int],
    baseline: dict[tuple[str, str], float],
    start: date,
    end: date,
) -> dict[str, Any]:
    """Pure: everything `main` reports, computed once over one fetched
    window. Unit-tested on synthetic series; the R2 fetch behind `main` is
    not."""
    disruptions, labels = build_labels(series, baseline)
    onset_span = ACUTE_ONSET_TICKS * TICK_SECONDS

    weekly: list[dict[str, Any]] = []
    for week_start, week_end in _week_windows(start, end):
        lo = int(
            datetime.combine(week_start, datetime.min.time(), tzinfo=UTC).timestamp()
        )
        hi = int(
            datetime.combine(
                week_end + timedelta(days=1), datetime.min.time(), tzinfo=UTC
            ).timestamp()
        )
        in_week = [d for d in disruptions if lo <= d.start_tick < hi]
        weekly.append(
            {
                "week_start": week_start.isoformat(),
                "week_end": week_end.isoformat(),
                "acute_events": len(in_week),
                "distinct_routes": len({d.route for d in in_week}),
            }
        )

    surviving_30min = [
        d
        for d in disruptions
        if was_at_baseline(series, baseline, d.route, d.start_tick)
    ]

    return {
        "total_judgeable_ticks": len(labels),
        "normal_ticks": sum(1 for v in labels.values() if v == "normal"),
        "acute_ticks": sum(1 for v in labels.values() if v == "acute"),
        "chronic_ticks": sum(1 for v in labels.values() if v == "chronic"),
        "total_events": len(disruptions),
        "distinct_routes": len({d.route for d in disruptions}),
        "events_reaching_chronic": sum(
            1 for d in disruptions if d.recovered_tick - d.start_tick > onset_span
        ),
        "weekly_acute_events": weekly,
        "hourly_prevalence": _hourly_prevalence(labels),
        "onsets_by_et_hour": _onsets_by_et_hour(disruptions),
        "events_surviving_30min_at_baseline": len(surviving_30min),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure the assigned_n degradation label against the real "
        "trip-updates archive: acute events/week, hourly prevalence, the "
        "chronic/acute split, and the at-baseline-30min survival filter."
    )
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument("--days", type=int, default=11, help="trailing window")
    args = parser.parse_args(argv)

    today = datetime.now(UTC).date()
    end = args.end_date or today
    start = args.start_date or (end - timedelta(days=args.days - 1))

    cfg = load_config()
    client = make_client(cfg)
    print(f"fetching archived trip-update metrics {start}..{end}", file=sys.stderr)
    bodies = fetch_trip_update_metrics(
        cfg, start_date=start, end_date=end, client=client
    )
    print(f"{len(bodies)} archived trip-update ticks", file=sys.stderr)

    series = build_service_series(bodies)
    baseline = compute_baseline(series, bin_fn=BIN_FN)
    print(f"{len(baseline)} (route, schedule_bin) baseline cells", file=sys.stderr)

    report = measure_degradation(series, baseline, start, end)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
