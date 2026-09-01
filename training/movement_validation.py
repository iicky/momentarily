"""Validate the published movement-primary condition against an independent
truth, and be honest about exactly what that truth can and cannot say.

WHAT IS UNDER TEST
------------------
worker/src/movement_state.ts deriveMovementState, mirrored offline in
load_r2.derive_movement_state / build_movement_truth: a route's published
current condition (normal / disrupted / suspended) read from where its trains
physically are — the cross-tick advance fraction of matched trips at scheduled
through stops, scored against that cell's own (route, direction, tod_bin)
baseline, taken through the regime clock (training.regime, DEBOUNCE_TICKS=1,
the live movement operating point per worker/src/index.ts). The raw per-tick
call and the debounced published surface a rider sees are both graded, because
an abstention holds the prior published regime and can show a stale disrupted
long after the evidence stopped.

THE TRUTH, AND ITS INDEPENDENCE
-------------------------------
Vehicle positions CANNOT adjudicate this condition: the condition IS derived
from them, so scoring it against a vehicle-position truth is scoring a signal
against itself. The only reference here that is independent in derivation is
training.degradation_label's assigned_n label — the count of dispatched trains
per route vs its own (route, schedule_bin) median, from the trip-updates feed,
debounced with hysteresis. Same GTFS-RT upstream family, DIFFERENT feed product
(trip-updates vs vehicle positions): independent-in-derivation, not
independent-in-source (load_r2.py's fetch_vehicle_metrics comment;
segment_coverage.py's TRUTH section).

WHY DETECTION IS CORROBORATION, NOT A PASS BAR
----------------------------------------------
assigned_n measures SUPPLY (how many trains are dispatched); movement measures
FLOW (whether the running trains advance). They are near-orthogonal by
construction: journal.md 2026-08-20 cross-tabbed 785 assigned_n-disrupted
route-ticks and found 0% read movement-disrupted (70% unjudgeable, 27%
movement-normal). A route with trains pulled but the rest moving fine is
supply-degraded and correctly movement-normal. So a low detection rate of the
BROAD assigned_n label is expected and is NOT evidence the movement condition
is wrong. Detection is reported, but the number that carries independent weight
is measured on the coincident subset below.

THE COINCIDENT SUBSET (where the axes must physically agree)
------------------------------------------------------------
When supply collapses to near zero — assigned_n falling to a small fraction of
its median, i.e. a suspension or near-suspension — there are (almost) no trains
left to advance, so movement MUST also read not-normal if it is not blind. That
severe tail is the one place the two independent feeds describe the same
physical event, and detection + onset latency measured there is a real
validation, not a category error. Episodes are split into `severe` (min
assigned_n ratio over the span at/under SEVERE_RATIO) and `partial`.

FALSE ALARMS ARE AN UPPER BOUND
-------------------------------
On stretches assigned_n confirms as fully normal supply, how often does the
movement condition fire? Such a firing is either a genuine flow disruption the
supply feed cannot see (trains present but frozen) or a spurious call, and this
truth cannot separate them. So the false-alarm rate is an UPPER bound on the
movement false-positive rate — a small number is a strong positive result (the
condition is not crying wolf during demonstrably normal service); a large one
is ambiguous, not a refutation.

EPISODES, NOT TICKS
-------------------
Every rate bootstraps over whole episodes / normal runs, never ticks: the ticks
inside one episode are the same event observed repeatedly, and a tick-level
interval overstates the evidence by orders of magnitude (journal.md 2026-08-22:
a flat 58k tick count hid a 17.8x swing in independent episodes). n_episodes is
reported beside n_ticks everywhere.

CAUSALITY
---------
The advance baseline and the assigned_n baseline are both fitted on a leading
sub-window; only the held-out remainder is scored. A sustained outage can never
lower the baseline it is later judged against. Counts are through-stop filtered
to match the live worker and the fitted baseline.

Run:
  murk exec -- uv run python -m training.movement_validation --days 21 --fit-days 14
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from training.degradation_label import AT_BASELINE_LOOKBACK_TICKS, BIN_FN, build_labels
from training.gtfs_static import load_topology, through_stops
from training.load import TICK_SECONDS
from training.load_r2 import (
    Disruption,
    StopFilter,
    build_movement_series_by_direction,
    build_movement_truth,
    build_service_series,
    compute_advance_baseline,
    compute_baseline,
    fetch_trip_update_metrics,
    fetch_vehicle_metrics,
)
from training.r2_client import load_config, make_client
from training.regime import MAX_IDLE_SEC, advance_regimes
from training.segment_coverage import normal_runs

# Bootstrap resamples for the episode-level CIs — 2000 gives a stable 95%
# percentile interval at the episode counts this label produces (tens), matching
# segment_coverage.BOOTSTRAP_N.
BOOTSTRAP_N = 2000

# The states that count as the movement condition "firing". The disrupted arm is
# the flow signal genuinely independent of assigned_n; suspended is the no-trains
# arm, which is vehicle-only here (build_movement_truth) but coincides with the
# supply feed's own collapse, so it is reported both alone and folded in.
DISRUPTED: frozenset[str] = frozenset({"disrupted"})
NOT_NORMAL: frozenset[str] = frozenset({"disrupted", "suspended"})

# An assigned_n episode is "severe" when supply falls to at/under this fraction
# of the route's own baseline at its worst tick — a suspension or near-suspension
# where movement must physically also see the collapse. 0.15 is a bright line
# below the label's own 0.5 degrade floor, not a quantile of measured depths, so
# it can't be read as tuned to the yield it produces.
SEVERE_RATIO = 0.15


@dataclass(frozen=True)
class Unit:
    """One graded episode or normal run, reduced to what the rates need. The
    movement condition is per-route, so a route either fires this tick or does
    not — a boolean, not a cell share (contrast segment_coverage.Unit, which
    lifts per-cell calls to a route share)."""

    alarmed_ticks: int
    ticks: int


def _boot_rates(
    units: Sequence[Unit],
    *,
    n: int = BOOTSTRAP_N,
    seed: int = 0,
) -> dict[str, float | int]:
    """Two rates over `units` with a CLUSTER bootstrap resampling whole units.

      tick_rate — pooled ticks the condition fired over scored ticks. Exposure
                  matched, so it is comparable between the (short) episode
                  population and the (long) normal-run population.
      unit_rate — share of units with at least one firing tick. Not
                  exposure-matched; reported because it is what an operator
                  experiences (did an alarm appear during this event at all).

    Empty input reports the counts and NO rates — a zero rate there would read as
    a measurement and a CI on it would be fabricated (segment_coverage._boot_rates).
    """
    if not units:
        return {"n_units": 0, "n_ticks": 0}
    hit_ticks = sum(u.alarmed_ticks for u in units)
    total_ticks = sum(u.ticks for u in units)
    hit_units = sum(1 for u in units if u.alarmed_ticks > 0)
    rng = random.Random(seed)
    tick_draws: list[float] = []
    unit_draws: list[float] = []
    for _ in range(n):
        sample = [rng.choice(units) for _ in range(len(units))]
        total = sum(u.ticks for u in sample)
        tick_draws.append(
            sum(u.alarmed_ticks for u in sample) / total if total else 0.0
        )
        unit_draws.append(sum(1 for u in sample if u.alarmed_ticks > 0) / len(sample))
    tick_draws.sort()
    unit_draws.sort()
    lo, hi = int(0.025 * n), min(n - 1, int(0.975 * n))
    return {
        "n_units": len(units),
        "n_ticks": total_ticks,
        "alarmed_units": hit_units,
        "alarmed_ticks": hit_ticks,
        "tick_rate": hit_ticks / total_ticks if total_ticks else 0.0,
        "tick_rate_ci_low": tick_draws[lo],
        "tick_rate_ci_high": tick_draws[hi],
        "unit_rate": hit_units / len(units),
        "unit_rate_ci_low": unit_draws[lo],
        "unit_rate_ci_high": unit_draws[hi],
    }


def published_states(
    calls: Sequence[tuple[int, Mapping[str, str]]],
    *,
    debounce_ticks: int = 1,
    max_idle_sec: int = MAX_IDLE_SEC,
) -> list[tuple[int, dict[str, str]]]:
    """The raw per-tick movement calls taken through the regime clock, emitting
    each route's COMMITTED published state at every tick — what the snapshot
    would actually show.

    A route absent from a tick's calls is an abstention: the clock holds its last
    committed state (up to max_idle_sec), so a stale disrupted can persist through
    blind ticks. That is exactly the published-surface false alarm the raw calls
    never show, so both are graded (see grade())."""
    entries: dict[str, Any] = {}
    out: list[tuple[int, dict[str, str]]] = []
    for observed_at, tick_calls in sorted(calls, key=lambda t: t[0]):
        entries, _ = advance_regimes(
            entries,
            tick_calls,
            observed_at,
            debounce_ticks=debounce_ticks,
            max_idle_sec=max_idle_sec,
        )
        out.append((observed_at, {k: e.state for k, e in entries.items()}))
    return out


def _state_at(
    stream: Sequence[tuple[int, Mapping[str, str]]],
) -> dict[int, dict[str, str]]:
    """tick -> route -> committed/raw state, for O(1) span lookup."""
    return {tick: dict(per_route) for tick, per_route in stream}


def _grade_spans(
    state_at: Mapping[int, Mapping[str, str]],
    spans: Sequence[tuple[str, int, int]],
    alarm_states: frozenset[str],
    *,
    bootstrap: int,
    seed: int,
) -> dict[str, Any]:
    """Bootstrap the firing rate over `spans` = (route, start, end) half-open
    intervals — assigned_n episodes for detection, normal runs for false alarms.

    A span's tick count counts only ticks the movement stream actually scored on
    that route: charging a unit for a tick the route was never judged on would
    depress every rate by however much the two archives' clocks differ. A span
    whose route is never judged in its window contributes no unit — an absence of
    evidence, not a miss (segment_coverage's same rule)."""
    units: list[Unit] = []
    for route, start, end in spans:
        alarmed = ticks = 0
        for tick in range(start, end, TICK_SECONDS):
            state = state_at.get(tick, {}).get(route)
            if state is None:
                continue
            ticks += 1
            if state in alarm_states:
                alarmed += 1
        if ticks > 0:
            units.append(Unit(alarmed_ticks=alarmed, ticks=ticks))
    boot = _boot_rates(units, n=bootstrap, seed=seed)
    return {"gradeable": len(units), "offered": len(spans), **boot}


def _onset_latency(
    state_at: Mapping[int, Mapping[str, str]],
    disruptions: Sequence[Disruption],
    alarm_states: frozenset[str],
    *,
    clean_lookback_ticks: int = AT_BASELINE_LOOKBACK_TICKS,
) -> dict[str, Any]:
    """Minutes from a real assigned_n onset to the movement condition's first
    firing on that route inside the episode — measured from the onset itself, no
    lead window (segment_coverage._onset_latency: a lead window on a
    frequently-firing surface just pins every latency to the boundary).

    `n_alarming_at_onset` is the contamination made explicit: episodes already
    firing at the onset tick, whose zero latency says nothing about detection, so
    they are counted and excluded. `fresh` is the stricter cohort: no firing on
    any scored tick in the lookback before onset, so the alarm had to arrive.

    An episode whose route has NO scored movement tick anywhere in its span is
    absence of evidence, not a miss (`n_ungradeable`): charging it to n_missed
    would credit an archive-clock gap — or a near-suspension too sparse to judge
    — to the classifier's detection rate, the same rule `_grade_spans` uses. Such
    episodes are excluded from clean_rows and fresh_rows both."""
    lookback = clean_lookback_ticks * TICK_SECONDS

    def scored_in_span(route: str, onset: int, recovered: int) -> bool:
        return any(
            state_at.get(tick, {}).get(route) is not None
            for tick in range(onset, recovered, TICK_SECONDS)
        )

    def first_fire(route: str, onset: int, recovered: int) -> int | None:
        for tick in range(onset, recovered, TICK_SECONDS):
            if state_at.get(tick, {}).get(route) in alarm_states:
                return tick
        return None

    def was_clean(route: str, onset: int) -> bool:
        seen = False
        for tick in range(onset - lookback, onset, TICK_SECONDS):
            state = state_at.get(tick, {}).get(route)
            if state is None:
                continue
            seen = True
            if state in alarm_states:
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
    n_ungradeable = 0
    for d in disruptions:
        if not scored_in_span(d.route, d.start_tick, d.recovered_tick):
            n_ungradeable += 1
            continue
        already = state_at.get(d.start_tick, {}).get(d.route) in alarm_states
        if already:
            n_alarming_at_onset += 1
        hit = first_fire(d.route, d.start_tick, d.recovered_tick)
        latency = (hit - d.start_tick) / 60.0 if hit is not None else None
        if not already:
            clean_rows.append(latency)
        if was_clean(d.route, d.start_tick):
            fresh_rows.append(latency)
    return {
        "n_offered": len(disruptions),
        "n_ungradeable": n_ungradeable,
        "n_alarming_at_onset": n_alarming_at_onset,
        "clean_lookback_min": lookback // 60,
        **summarise(clean_rows),
        "fresh": summarise(fresh_rows),
    }


def episode_min_ratio(
    d: Disruption,
    series: Mapping[tuple[str, int], int],
    baseline: Mapping[tuple[str, str], float],
) -> float | None:
    """The worst (smallest) assigned_n / baseline ratio over an episode's span —
    how deep the supply collapse got. None when no tick in the span has both a
    reading and a baseline to judge it against."""
    ratios: list[float] = []
    for tick in range(d.start_tick, d.recovered_tick, TICK_SECONDS):
        base = baseline.get((d.route, BIN_FN(tick)))
        assigned = series.get((d.route, tick))
        if base is None or base <= 0 or assigned is None:
            continue
        ratios.append(assigned / base)
    return min(ratios) if ratios else None


def split_severity(
    disruptions: Sequence[Disruption],
    series: Mapping[tuple[str, int], int],
    baseline: Mapping[tuple[str, str], float],
    *,
    severe_ratio: float = SEVERE_RATIO,
) -> tuple[list[Disruption], list[Disruption], int]:
    """(severe, partial, n_unrateable). Severe = worst-tick supply at/under
    `severe_ratio` of baseline (the coincident near-suspension subset); partial =
    the rest; unrateable episodes (no judgeable tick) are counted, never bucketed
    — a silent drop would let the split's denominators lie."""
    severe: list[Disruption] = []
    partial: list[Disruption] = []
    n_unrateable = 0
    for d in disruptions:
        mr = episode_min_ratio(d, series, baseline)
        if mr is None:
            n_unrateable += 1
        elif mr <= severe_ratio:
            severe.append(d)
        else:
            partial.append(d)
    return severe, partial, n_unrateable


def grade(
    calls: Sequence[tuple[int, Mapping[str, str]]],
    disruptions: Sequence[Disruption],
    severe: Sequence[Disruption],
    partial: Sequence[Disruption],
    runs: Sequence[tuple[str, int, int]],
    *,
    alarm_states: frozenset[str],
    bootstrap: int = BOOTSTRAP_N,
    seed: int = 0,
) -> dict[str, Any]:
    """One arm's report for one set of `alarm_states`, scored on two surfaces:

      calls     — the raw per-tick movement decision. Whether the rule discriminates.
      published — those calls through the regime clock: what a rider is shown, where
                  a stale held-open regime becomes a false alarm the raw calls hide.

    Detection is split all / severe / partial: the severe (near-suspension) bucket
    is the coincident subset the two feeds physically share and the only one whose
    detection is an independent validation rather than a cross-axis category error.
    False alarms on the assigned_n-normal runs are an UPPER bound on the movement
    false-positive rate (module docstring)."""
    raw_at = _state_at(calls)
    pub_at = _state_at(published_states(calls))

    def surface(state_at: Mapping[int, Mapping[str, str]], s: int) -> dict[str, Any]:
        return {
            "detection": {
                "all": _grade_spans(
                    state_at,
                    [(d.route, d.start_tick, d.recovered_tick) for d in disruptions],
                    alarm_states,
                    bootstrap=bootstrap,
                    seed=s,
                ),
                "severe": _grade_spans(
                    state_at,
                    [(d.route, d.start_tick, d.recovered_tick) for d in severe],
                    alarm_states,
                    bootstrap=bootstrap,
                    seed=s + 1,
                ),
                "partial": _grade_spans(
                    state_at,
                    [(d.route, d.start_tick, d.recovered_tick) for d in partial],
                    alarm_states,
                    bootstrap=bootstrap,
                    seed=s + 2,
                ),
            },
            "false_alarms": _grade_spans(
                state_at, runs, alarm_states, bootstrap=bootstrap, seed=s + 3
            ),
            "onset_latency": {
                "all": _onset_latency(state_at, disruptions, alarm_states),
                "severe": _onset_latency(state_at, severe, alarm_states),
            },
        }

    return {
        "calls": surface(raw_at, seed),
        "published": surface(pub_at, seed + 100),
    }


def _calls_by_tick(
    truth: Mapping[tuple[str, int], str],
) -> list[tuple[int, dict[str, str]]]:
    """build_movement_truth's (route, tick) -> state map, grouped into the
    (tick, {route: state}) stream the regime clock and the graders consume."""
    by_tick: dict[int, dict[str, str]] = defaultdict(dict)
    for (route, tick), state in truth.items():
        by_tick[tick][route] = state
    return sorted(by_tick.items())


def _stop_filter(
    through: frozenset[tuple[str, str, str]] | None,
) -> StopFilter | None:
    if through is None:
        return None
    return lambda route, direction, frm: (route, direction, frm) in through


def build_validation_report(
    movement_truth: Mapping[tuple[str, int], str],
    service_series: Mapping[tuple[str, int], int],
    service_baseline: Mapping[tuple[str, str], float],
    *,
    window: Mapping[str, Any] | None = None,
    bootstrap: int = BOOTSTRAP_N,
    seed: int = 0,
    severe_ratio: float = SEVERE_RATIO,
) -> dict[str, Any]:
    """The full validation report — detection (split by supply-collapse
    severity), the false-alarm upper bound on confirmed-normal runs, and onset
    latency — from a reconstructed movement condition and the independent
    assigned_n supply series plus its causal baseline.

    Pure over its inputs: both movement_validation.main and training.review call
    this, so the false-alarm bound is derived exactly one way wherever it is
    reported. `movement_truth` is build_movement_truth's (route, tick) -> state
    map (its advance baseline must already be causal — fit on a window ending
    before the scored one — since this function does not refit it). The
    assigned_n episodes, normal runs, and severity split are derived here from
    `service_series` against `service_baseline` via the pinned degradation_label
    thresholds, so a grade is never scored against a differently tuned supply
    label than the one that module publishes.

    `window` carries the fetch-provenance fields the caller knows (dates, tick
    counts, topology source); the fields derivable from the calls themselves
    (call mix, disrupted base rate, scored-tick count, debounce) are filled in
    here and win on key collision."""
    calls = _calls_by_tick(movement_truth)
    call_mix: dict[str, int] = defaultdict(int)
    for state in movement_truth.values():
        call_mix[state] += 1
    n_calls = sum(call_mix.values())

    disruptions, labels = build_labels(dict(service_series), dict(service_baseline))
    runs = normal_runs(labels)
    severe, partial, n_unrateable = split_severity(
        disruptions, service_series, service_baseline, severe_ratio=severe_ratio
    )

    derived_window: dict[str, Any] = {
        "n_scored_movement_ticks": len(calls),
        "debounce_ticks": 1,
        # Sanity: the raw call mix over judged route-ticks. A disrupted base rate
        # near the ~0.3% the calibration reported (journal 2026-08-26) confirms
        # the reconstruction fires at all, so a low false-alarm rate is a
        # silent-on-normal result, not a dead classifier.
        "call_mix": dict(sorted(call_mix.items())),
        "disrupted_base_rate": call_mix.get("disrupted", 0) / n_calls
        if n_calls
        else None,
    }
    return {
        "window": {**dict(window or {}), **derived_window},
        "truth": {
            "label": "assigned_n degradation (training.degradation_label)",
            "independence": "independent-in-derivation (trip-updates supply) from "
            "the vehicle-position flow signal the condition reads; NOT "
            "independent-in-source (same GTFS-RT upstream family)",
            "n_episodes": len(disruptions),
            "n_severe_episodes": len(severe),
            "n_partial_episodes": len(partial),
            "n_unrateable_episodes": n_unrateable,
            "severe_ratio": severe_ratio,
            "n_normal_runs": len(runs),
            "n_routes_with_episodes": len({d.route for d in disruptions}),
            "episode_ticks_median": statistics.median(
                (d.recovered_tick - d.start_tick) // TICK_SECONDS for d in disruptions
            )
            if disruptions
            else 0,
        },
        "arms": {
            "disrupted": grade(
                calls,
                disruptions,
                severe,
                partial,
                runs,
                alarm_states=DISRUPTED,
                bootstrap=bootstrap,
                seed=seed,
            ),
            "not_normal": grade(
                calls,
                disruptions,
                severe,
                partial,
                runs,
                alarm_states=NOT_NORMAL,
                bootstrap=bootstrap,
                seed=seed,
            ),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the published movement condition against the "
        "independent assigned_n degradation label: episode-bootstrapped detection "
        "(split by supply-collapse severity), false alarms on confirmed-normal "
        "runs, and onset latency."
    )
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument("--days", type=int, default=21, help="total trailing window")
    parser.add_argument(
        "--fit-days",
        type=int,
        default=14,
        help="leading days used to fit both baselines; the rest is scored",
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
    stop = _stop_filter(through)

    # --- prediction: reconstruct the published movement condition -------------
    print(
        f"fitting advance baseline on vehicle archive {start}..{fit_end}",
        file=sys.stderr,
    )
    fit_veh = fetch_vehicle_metrics(
        cfg, start_date=start, end_date=fit_end, client=client
    )
    advance_baseline = compute_advance_baseline(
        build_movement_series_by_direction(fit_veh, counts_from_stop=stop)
    )
    print(f"scoring vehicle archive {score_start}..{end}", file=sys.stderr)
    score_veh = fetch_vehicle_metrics(
        cfg, start_date=score_start, end_date=end, client=client
    )
    truth = build_movement_truth(
        score_veh, movement_baseline=advance_baseline, counts_from_stop=stop
    )
    print(
        f"{len(fit_veh)} fit ticks -> {len(advance_baseline)} advance cells; "
        f"{len(truth)} scored movement ticks",
        file=sys.stderr,
    )

    # --- truth: the independent assigned_n degradation label ------------------
    print(f"fetching trip-update metrics {start}..{end}", file=sys.stderr)
    svc_bodies = fetch_trip_update_metrics(
        cfg, start_date=start, end_date=end, client=client
    )
    all_series = build_service_series(svc_bodies)
    boundary = int(
        datetime.combine(score_start, datetime.min.time(), tzinfo=UTC).timestamp()
    )
    svc_baseline = compute_baseline(
        {k: v for k, v in all_series.items() if k[1] < boundary}, bin_fn=BIN_FN
    )
    series = {k: v for k, v in all_series.items() if k[1] >= boundary}

    report = build_validation_report(
        truth,
        series,
        svc_baseline,
        window={
            "fit_start": start.isoformat(),
            "fit_end": fit_end.isoformat(),
            "score_start": score_start.isoformat(),
            "score_end": end.isoformat(),
            "topology_source": topology_source,
            "n_fit_ticks": len(fit_veh),
        },
        bootstrap=args.bootstrap,
        seed=args.seed,
    )
    t = report["truth"]
    print(
        f"{t['n_episodes']} assigned_n episodes ({t['n_severe_episodes']} severe, "
        f"{t['n_partial_episodes']} partial, {t['n_unrateable_episodes']} unrateable), "
        f"{t['n_normal_runs']} normal runs, {len(svc_baseline)} (route, schedule_bin) "
        f"baseline cells fitted on {start}..{fit_end}",
        file=sys.stderr,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
