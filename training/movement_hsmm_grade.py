"""Causal grading harness for the explicit-duration movement debounce.

WHAT THIS ANSWERS
-----------------
Should the memoryless self-loop dwell be replaced by an explicit-duration
(negative-binomial) sojourn in the movement debounce? Three gates decide it,
all three measured here, and a measured NO is a settled answer rather than a
failure of the exercise:

  1. FALSE ALARMS DO NOT WIDEN. The per-route per-tick false-alarm rate on
     assigned_n-confirmed-normal runs, episode-bootstrapped, must not put its
     upper confidence bound past the 0.00101/tick reference this arm is already
     certified at (`online_fdr`, `movement_validation`).
  2. DWELL LIKELIHOOD STRICTLY IMPROVES. The negative binomial must beat the
     geometric it nests at r = 1, both in-sample by likelihood-ratio test and
     out-of-sample on held-out episode durations.
  3. DETECTION SUPPORT DOES NOT FALL. The candidate must detect at least as
     many independent corroborated episodes as the incumbent form does.

WHY THE INCUMBENT IS THE r = 1 ARM AND NOT `regime.py`
------------------------------------------------------
The comparison runs the SAME filter twice, once with the fitted negative
binomial and once with its own r = 1 restriction, emissions refitted on each
arm by the same criterion. Grading against `regime.py`'s k-of-k clock instead
would confound the duration family with every other difference between the two
rules (abstention holding, back-dating, commit lag), and the question on the
table is specifically the dwell form. The shipped published surface is reported
alongside as a third column for continuity with the record, not as the gate's
baseline.

WHY THE DWELL POPULATION IS INJECTED AND NOT DERIVED
----------------------------------------------------
Gate 2's durations come from the archived EPISODE population, whose onsets and
recoveries the independent supply feed adjudicates. They are never read off the
movement calls and never off a model's own decode. Both alternatives are
circular in ways that reliably favour the candidate:

  - a Viterbi/EM decode's durations are partly an artefact of the negative
    binomial that produced them, so a geometric null fitted to them is being
    asked to account for a segmentation it never proposed;
  - the raw calls' own run lengths are contaminated by exactly the false calls
    the debounce exists to suppress, which manufactures over-dispersion
    (measured: r_hat = 0.315 on a synthetic stream whose true episodes were
    all 4-10 ticks).

So `grade` takes its episode population as an argument. That also means gate 2
cannot be measured without the episode archive, and this harness says so by
name rather than substituting a population it can reach.

CAUSALITY
---------
Both duration arms and both emission channels are fitted on
[train_start, train_end] and every graded number comes from episodes and normal
runs in the held-out remainder. The published surface is produced by
`movement_hsmm.hsmm_states`, a forward-only filter: no graded tick's state
depends on a later tick. Gate 2's held-out delta is scored with train-fitted
parameters against eval-window durations, so it is a forecast comparison and
not a fit comparison.

EPISODE BOOTSTRAP, EVERYWHERE
-----------------------------
Every interval resamples whole episodes or whole normal runs, never ticks. The
ticks inside one episode are the same event observed repeatedly and a
tick-level interval overstates the evidence by orders of magnitude (a flat 58k
tick count once hid a 17.8x swing in independent episodes). Intervals are
withheld below two units rather than fabricated from one.

Run:
  murk exec -- uv run python -m training.movement_hsmm_grade --train-days 14 --eval-days 7
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from training.episodes import extract_episodes
from training.eval import NOT_NORMAL, TICK_SECONDS
from training.eval_common import snap_tick
from training.movement_backfill import ticks_for
from training.movement_hsmm import (
    DWELL_MAX_TICKS,
    MIN_ROUTE_DWELL_EVENTS,
    Debounce,
    DwellModel,
    DwellObs,
    DwellPopulation,
    NbDwellFit,
    dwell_loglik,
    episode_populations,
    fit_debounce,
    fit_dwell,
    fit_dwell_model,
)
from training.movement_validation import (
    BOOTSTRAP_N,
    DISRUPTED,
    Unit,
    _boot_rates,  # pyright: ignore[reportPrivateUsage]
    published_states,
)
from training.r2_client import R2Config, load_config, make_client

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

# The certified per-route per-tick false-alarm bound this arm already holds.
# Gate 1 is stated against it rather than against the candidate's own point
# estimate: the question is whether the replacement stays inside the envelope
# the shipped rule was certified in, not whether it beats itself.
FA_REFERENCE = 0.00101

# Alarm vocabularies graded. `disrupted` is the flow arm genuinely independent
# of assigned_n; not-normal folds in the suspended arm. Both are reported
# because the shipped validation reports both and a gate stated on one of them
# would be quotable out of context.
ARMS: dict[str, frozenset[str]] = {
    "disrupted": DISRUPTED,
    "not_normal": frozenset(NOT_NORMAL),
}


# --- refusing a batch that cannot produce comparable numbers ----------------
#
# A grade that measures nothing must SAY so rather than print a table of NaN.
# Every rate here has a denominator that can legitimately be zero on a thin
# window, and a zero-denominator rate formats as a perfectly normal-looking
# 0.0000 — so the failure this guards is not a crash, it is a plausible table
# that means nothing. That is not hypothetical: a just-recovering transition
# stream, or a window before an arm shipped, produces exactly it.


@dataclass(frozen=True)
class GateBlocked(Exception):
    """The batch cannot produce comparable numbers, and precisely why.

    `reason` is one of a closed set of names so a blocked run is greppable and
    a caller can tell a thin window from a broken one:

      no_dwell_events        no completed episode durations to fit either arm
      dwell_all_censored     durations exist but none completed, so the MLE is
                             unbounded (survival alone has no upper bound)
      no_eval_dwell          nothing held out to score gate 2 out of sample
      no_normal_dwell        no normal sojourn observations, so the entry
                             hazard has nothing to be fitted from
      normal_dwell_all_censored
                             normal sojourns exist but none completed, the
                             same unbounded MLE as `dwell_all_censored`
      no_normal_runs         no assigned_n-confirmed-normal exposure, so the
                             false-alarm denominator is empty
      no_detection_episodes  no corroborated episodes, so gate 3 has no
                             population
      no_calls               the movement stream scored nothing in the window
      single_route           one route cannot support a fleet claim
    """

    reason: str
    detail: dict[str, Any] = field(default_factory=dict[str, Any])

    def __str__(self) -> str:
        return f"{self.reason}: {json.dumps(self.detail, sort_keys=True, default=str)}"


def _refuse_degenerate(
    train_dwell: DwellPopulation,
    eval_dwell: DwellPopulation,
    train_normal: DwellPopulation,
    calls: Sequence[tuple[int, Mapping[str, str]]],
    runs: Sequence[tuple[str, int, int]],
    corroborated: Sequence[object],
) -> None:
    """Raise `GateBlocked` by name for every population a gate needs and does
    not have. Checked before any fitting, so a blocked run is cheap and its
    reason is not buried under a traceback from a downstream NaN."""
    if not calls:
        raise GateBlocked("no_calls", {"n_ticks": 0})
    routes = {r for _, per in calls for r in per}
    if len(routes) < 2:
        raise GateBlocked("single_route", {"routes": sorted(routes)})
    if not train_dwell.obs:
        raise GateBlocked(
            "no_dwell_events",
            {"n_left_censored": train_dwell.n_left_censored},
        )
    if train_dwell.n_events == 0:
        raise GateBlocked(
            "dwell_all_censored",
            {"n_censored": train_dwell.n_censored},
        )
    if eval_dwell.n_events == 0:
        raise GateBlocked(
            "no_eval_dwell",
            {"n_obs": len(eval_dwell.obs), "n_censored": eval_dwell.n_censored},
        )
    # THE NORMAL POPULATION NEEDS ITS OWN GUARD, AND THIS PATH IS REACHABLE.
    #
    # `grade` fits the normal sojourn immediately and hands it to
    # `params_from_fits` as the ENTRY HAZARD — the term deciding whether one
    # call can flip the state. `fit_dwell(())` returns NaN r and p, which
    # propagate silently through `dwell_hazards` into NaN likelihoods and NaN
    # gate numbers rather than into an error.
    #
    # It is reachable precisely because left-truncated normal runs are dropped
    # on purpose (`episode_populations`): a thin train window whose routes are
    # either never disrupted or disrupted only after a leading gap has episode
    # dwells and NO usable normal dwells. Having deliberately thrown that
    # evidence away, this has to refuse rather than fit what is left of it.
    if not train_normal.obs:
        raise GateBlocked(
            "no_normal_dwell",
            {"n_left_truncated_dropped": train_normal.n_left_censored},
        )
    if train_normal.n_events == 0:
        raise GateBlocked(
            "normal_dwell_all_censored",
            {
                "n_censored": train_normal.n_censored,
                "n_left_truncated_dropped": train_normal.n_left_censored,
            },
        )
    if not runs:
        raise GateBlocked("no_normal_runs", {"offered": 0})
    if not corroborated:
        raise GateBlocked("no_detection_episodes", {"offered": 0})


# --- episode bootstrap ------------------------------------------------------


def _percentile_ci(draws: list[float], n: int) -> tuple[float, float]:
    draws.sort()
    lo, hi = int(0.025 * n), min(n - 1, int(0.975 * n))
    return draws[lo], draws[hi]


@dataclass(frozen=True)
class Interval:
    """A point estimate with a percentile episode-bootstrap interval.

    `ci_low`/`ci_high` are None below two independent units, where every
    resample is the same unit and an interval would be fabricated rather than
    wide (`online_fdr._boot_share`'s rule)."""

    value: float
    ci_low: float | None
    ci_high: float | None
    n_units: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_units": self.n_units,
        }


def bootstrap_dwell_delta(
    obs: Sequence[DwellObs],
    negbin: NbDwellFit,
    geometric: NbDwellFit,
    *,
    n: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Interval:
    """Per-episode mean log-likelihood advantage of the NB arm over the
    geometric on HELD-OUT durations, with an episode bootstrap.

    The unit is one episode. Parameters are the TRAIN fits and are not refitted
    inside the resample: the question is whether the causally fitted NB
    describes held-out durations better, and refitting each draw would answer a
    different question (how much the parameters wobble) while also letting the
    extra parameter pay for itself on the very data it is being scored on.
    """
    per_episode = [
        dwell_loglik([o], negbin.r, negbin.p)
        - dwell_loglik([o], geometric.r, geometric.p)
        for o in obs
    ]
    k = len(per_episode)
    if k == 0:
        return Interval(math.nan, None, None, 0)
    value = sum(per_episode) / k
    if k < 2:
        return Interval(value, None, None, k)
    rng = random.Random(seed)
    draws = [sum(rng.choice(per_episode) for _ in range(k)) / k for _ in range(n)]
    lo, hi = _percentile_ci(draws, n)
    return Interval(value, lo, hi, k)


def bootstrap_detection_delta(
    candidate: Sequence[bool],
    incumbent: Sequence[bool],
    *,
    n: int = BOOTSTRAP_N,
    seed: int = 0,
) -> Interval:
    """Paired episode bootstrap of (candidate - incumbent) detection share.

    PAIRED, resampling the episode and reading both arms' verdicts on it, not
    two independent bootstraps differenced. The two arms see the same episodes,
    so the between-arm difference is far better determined than either rate,
    and independent intervals would hide that behind two overlapping ones.
    """
    k = len(candidate)
    if k != len(incumbent):
        raise ValueError("paired bootstrap needs equal-length arms")
    if k == 0:
        return Interval(math.nan, None, None, 0)
    pairs = list(zip(candidate, incumbent, strict=True))
    value = sum(float(c) - float(i) for c, i in pairs) / k
    if k < 2:
        return Interval(value, None, None, k)
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n):
        acc = 0.0
        for _ in range(k):
            c, i = rng.choice(pairs)
            acc += float(c) - float(i)
        draws.append(acc / k)
    lo, hi = _percentile_ci(draws, n)
    return Interval(value, lo, hi, k)


# --- the graded surfaces ----------------------------------------------------


def false_alarm_rate(
    state_at: Mapping[int, Mapping[str, str]],
    runs: Sequence[tuple[str, int, int]],
    alarm_states: frozenset[str],
    *,
    bootstrap: int = BOOTSTRAP_N,
    seed: int = 0,
) -> dict[str, Any]:
    """Per-route per-tick false-alarm rate on assigned_n-confirmed-normal runs.

    One `Unit` per normal run and the interval bootstrapped over whole runs,
    reusing `movement_validation._boot_rates` so this number is derived exactly
    the same way as the 0.00101 reference it is compared against. Ticks are
    counted only where the surface actually scored the route: charging a run for
    a tick the route was never judged on would depress the rate by however much
    the two archives' clocks differ.
    """
    units: list[Unit] = []
    for route, start, end in runs:
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
    return {"gradeable": len(units), "offered": len(runs), **boot}


def _episode_span(ep: object) -> tuple[str, int, int] | None:
    route = getattr(ep, "route", None)
    onset = getattr(ep, "onset", None) or getattr(ep, "start_tick", None)
    recovery = getattr(ep, "recovery", None) or getattr(ep, "recovered_tick", None)
    if not isinstance(route, str) or not isinstance(onset, int):
        return None
    if not isinstance(recovery, int):
        return None
    return route, onset, recovery


def detection_flags(
    state_at: Mapping[int, Mapping[str, str]],
    episodes: Sequence[object],
    alarm_states: frozenset[str],
) -> tuple[list[bool], list[object], int]:
    """(hit per scored episode, the scored episodes, n unscored).

    An episode whose route the surface never judged inside its span is UNSCORED
    and excluded, not a miss — absence of evidence, the same rule
    `movement_validation` and `online_fdr` apply. Returning the scored episodes
    alongside the flags is what lets two arms be paired on an identical
    population, which gate 3's paired bootstrap requires.
    """
    flags: list[bool] = []
    scored_eps: list[object] = []
    unscored = 0
    for ep in episodes:
        span = _episode_span(ep)
        if span is None:
            unscored += 1
            continue
        route, onset, recovery = span
        scored = hit = False
        for tick in range(onset, recovery, TICK_SECONDS):
            state = state_at.get(tick, {}).get(route)
            if state is None:
                continue
            scored = True
            if state in alarm_states:
                hit = True
                break
        if not scored:
            unscored += 1
            continue
        flags.append(hit)
        scored_eps.append(ep)
    return flags, scored_eps, unscored


def _matched_detection(
    candidate_at: Mapping[int, Mapping[str, str]],
    incumbent_at: Mapping[int, Mapping[str, str]],
    episodes: Sequence[object],
    alarm_states: frozenset[str],
) -> tuple[list[bool], list[bool], int]:
    """Both arms' detection verdicts on the episodes BOTH can score.

    Coverage is not symmetric between arms — each emits a route only at ticks it
    filtered, and which ticks those are depends on the arm — so an unmatched
    comparison would difference two rates measured on different populations.
    The intersection is the only population on which "support did not fall" is
    a statement about the dwell form.
    """
    cand_flags, cand_eps, _ = detection_flags(candidate_at, episodes, alarm_states)
    inc_flags, inc_eps, _ = detection_flags(incumbent_at, episodes, alarm_states)
    cand_by_id = dict(zip((id(e) for e in cand_eps), cand_flags, strict=True))
    inc_by_id = dict(zip((id(e) for e in inc_eps), inc_flags, strict=True))
    shared = [e for e in episodes if id(e) in cand_by_id and id(e) in inc_by_id]
    return (
        [cand_by_id[id(e)] for e in shared],
        [inc_by_id[id(e)] for e in shared],
        len(episodes) - len(shared),
    )


# --- the three gates --------------------------------------------------------


@dataclass(frozen=True)
class Gate:
    """One gate's verdict and the numbers behind it.

    `passed` is None when the gate could not be evaluated — never False. A
    missing measurement is not a failure and must not be reported as one; the
    caller refuses the batch by name instead.
    """

    name: str
    passed: bool | None
    statement: str
    numbers: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "statement": self.statement,
            "numbers": self.numbers,
        }


def gate_false_alarms(
    candidate: Mapping[str, Any],
    incumbent: Mapping[str, Any],
    *,
    reference: float = FA_REFERENCE,
) -> Gate:
    """GATE 1: the candidate's false-alarm interval must not widen past the
    certified reference.

    Stated on the interval's UPPER bound, not the point estimate. "Must not
    widen past 0.00101" is a claim about the envelope the arm is certified in,
    and a point estimate under the reference with a bound above it is exactly
    the case the bound exists to catch.
    """
    hi = candidate.get("tick_rate_ci_high")
    rate = candidate.get("tick_rate")
    if hi is None or rate is None:
        return Gate(
            "false_alarm_ci",
            None,
            "no gradeable normal-run exposure, so no false-alarm interval",
            {"candidate": dict(candidate)},
        )
    passed = bool(hi <= reference)
    return Gate(
        "false_alarm_ci",
        passed,
        (
            f"candidate FA {rate:.5f}/tick, 95% CI high {hi:.5f} "
            f"{'<=' if passed else '>'} reference {reference:.5f}"
        ),
        {
            "reference": reference,
            "candidate_tick_rate": rate,
            "candidate_ci_low": candidate.get("tick_rate_ci_low"),
            "candidate_ci_high": hi,
            "incumbent_tick_rate": incumbent.get("tick_rate"),
            "incumbent_ci_high": incumbent.get("tick_rate_ci_high"),
            "n_units": candidate.get("n_units"),
            "n_ticks": candidate.get("n_ticks"),
        },
    )


def gate_dwell_loglik(nest: Any, held_out: Interval) -> Gate:
    """GATE 2: the negative binomial must strictly beat the geometric it nests.

    Both halves are required. The in-sample likelihood-ratio test says the
    extra parameter is buying more than noise (r = 1 is interior to r > 0, so
    the reference is a plain chi-squared with one degree of freedom and needs no
    boundary correction). The held-out per-episode delta says the improvement
    survives being a forecast, and its interval must sit strictly above zero —
    an interval straddling zero is not a strict improvement, which is what this
    gate demands.
    """
    lo = held_out.ci_low
    if math.isnan(nest.lr_stat) or math.isnan(held_out.value):
        return Gate(
            "dwell_loglik",
            None,
            "no dwell population to test either arm on",
            {"n_held_out": held_out.n_units},
        )
    in_sample = bool(nest.improves)
    out_of_sample = bool(lo is not None and lo > 0.0)
    passed = in_sample and out_of_sample
    return Gate(
        "dwell_loglik",
        passed,
        (
            f"LR {nest.lr_stat:.2f} (chi2_1 p={nest.p_value:.2e}, r_hat="
            f"{nest.negbin.r:.3f}); held-out delta {held_out.value:+.4f} "
            f"nats/episode, CI low "
            f"{'None' if lo is None else format(lo, '+.4f')}"
        ),
        {
            "lr_stat": nest.lr_stat,
            "p_value": nest.p_value,
            "df": nest.df,
            "r_hat": nest.negbin.r,
            "r_at_bound": nest.negbin.r_at_bound,
            "geometric_self_loop": nest.geometric.self_loop,
            "in_sample_improves": in_sample,
            "held_out": held_out.as_dict(),
            "held_out_strictly_positive": out_of_sample,
        },
    )


def gate_detection_support(
    candidate: Sequence[bool],
    incumbent: Sequence[bool],
    delta: Interval,
    *,
    n_unmatched: int,
) -> Gate:
    """GATE 3: detection support must not fall.

    ENFORCED ON THE POINT ESTIMATE. The candidate must detect at least as many
    of the matched corroborated episodes as the incumbent does; a negative
    delta fails this gate even when its interval includes zero. "Must not fall"
    is a requirement about support, and treating "the fall is not statistically
    resolved" as a pass would let a thin window license an arbitrary loss. The
    interval is recorded as evidence about how well determined the difference
    is, never as a route to passing.
    """
    if not candidate:
        return Gate(
            "detection_support",
            None,
            "no corroborated episode both arms can score",
            {"n_unmatched": n_unmatched},
        )
    c_rate = sum(candidate) / len(candidate)
    i_rate = sum(incumbent) / len(incumbent)
    passed = bool(sum(candidate) >= sum(incumbent))
    return Gate(
        "detection_support",
        passed,
        (
            f"candidate detected {sum(candidate)}/{len(candidate)} "
            f"({c_rate:.3f}) vs incumbent {sum(incumbent)}/{len(incumbent)} "
            f"({i_rate:.3f}); paired delta {delta.value:+.4f}"
        ),
        {
            "candidate_detected": sum(candidate),
            "incumbent_detected": sum(incumbent),
            "n_matched": len(candidate),
            "n_unmatched": n_unmatched,
            "candidate_rate": c_rate,
            "incumbent_rate": i_rate,
            "paired_delta": delta.as_dict(),
        },
    )


# --- the grade --------------------------------------------------------------


@dataclass(frozen=True)
class GradeInputs:
    """Everything the grade needs, already split. Kept explicit so the pure
    grading path can be exercised without any archive access."""

    train_calls: list[tuple[int, Mapping[str, str]]]
    eval_calls: list[tuple[int, Mapping[str, str]]]
    train_episodes: list[object]
    eval_episodes: list[object]
    normal_runs: list[tuple[str, int, int]]
    corroborated: list[object]
    routes: list[str]
    train_start: int
    train_end: int
    eval_end: int


def grade(
    inputs: GradeInputs,
    *,
    max_ticks: int = DWELL_MAX_TICKS,
    min_route_events: int = MIN_ROUTE_DWELL_EVENTS,
    bootstrap: int = BOOTSTRAP_N,
    seed: int = 0,
    arm: str = "disrupted",
) -> dict[str, Any]:
    """Fit both duration arms on the train window and grade both on the eval
    window, then state the three gates.

    Raises `GateBlocked` before fitting anything if a gate's population is
    missing.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}")
    alarm_states = ARMS[arm]

    train_pops = episode_populations(
        inputs.train_episodes,
        window_start=inputs.train_start,
        window_end=inputs.train_end,
        routes=inputs.routes,
    )
    eval_pops = episode_populations(
        inputs.eval_episodes,
        window_start=inputs.train_end,
        window_end=inputs.eval_end,
        routes=inputs.routes,
    )
    _refuse_degenerate(
        train_pops.not_moving,
        eval_pops.not_moving,
        train_pops.normal,
        inputs.eval_calls,
        inputs.normal_runs,
        inputs.corroborated,
    )

    dwell = fit_dwell_model(train_pops.not_moving, min_route_events=min_route_events)
    normal_nb = fit_dwell(train_pops.normal.obs)
    normal_geom = fit_dwell(train_pops.normal.obs, fix_r=1.0)

    arms: dict[str, Debounce] = {
        "negbin": fit_debounce(
            inputs.train_calls,
            dwell,
            normal_nb,
            max_ticks=max_ticks,
            geometric=False,
        ),
        "geometric": fit_debounce(
            inputs.train_calls,
            dwell,
            normal_geom,
            max_ticks=max_ticks,
            geometric=True,
        ),
    }
    # WARM EVERY SURFACE THROUGH THE BOUNDARY, THEN SLICE.
    #
    # Filtering the eval calls alone would cold-start every route at
    # `train_end`: each one opens a fresh regime with no age and no history, so
    # the first ticks of the eval window are graded on a belief no deployed
    # filter would ever hold, and an episode straddling the split is cut in
    # half. Both the candidate and the shipped clock are therefore run over the
    # concatenated stream and sliced afterwards. PARAMETERS are still fitted on
    # the train half only — warming carries STATE across the boundary, never
    # fitted values, so nothing from the eval window reaches a parameter.
    warm_calls = [*inputs.train_calls, *inputs.eval_calls]

    def graded(
        stream: Sequence[tuple[int, Mapping[str, str]]],
    ) -> dict[int, dict[str, str]]:
        return {tick: dict(per) for tick, per in stream if tick >= inputs.train_end}

    surfaces = {name: graded(db.states(warm_calls)) for name, db in arms.items()}
    # THE SHIPPED SURFACE IS THE INCUMBENT THE GATES ARE STATED AGAINST.
    #
    # `regime.py`'s clock through `movement_validation.published_states` is the
    # deployed rule, and it is the rule the 0.00101 reference was measured on.
    # A non-regression gate has to be against what is actually running: the
    # r = 1 arm is the same-filter control that isolates the duration family,
    # which is the more informative comparison scientifically and the wrong one
    # for ship/no-ship, because passing it would only say the candidate beats a
    # form nobody deployed.
    surfaces["shipped"] = graded(published_states(warm_calls))
    # The raw calls are reported too: they are what every filter was handed, so
    # a surface matching them exactly has debounced nothing, and that is worth
    # seeing rather than inferring.
    surfaces["raw_calls"] = graded(inputs.eval_calls)

    fa = {
        name: false_alarm_rate(
            at, inputs.normal_runs, alarm_states, bootstrap=bootstrap, seed=seed
        )
        for name, at in surfaces.items()
    }

    # Gate 3: candidate against SHIPPED.
    cand_flags, ship_flags, n_unmatched = _matched_detection(
        surfaces["negbin"], surfaces["shipped"], inputs.corroborated, alarm_states
    )
    detection_delta = bootstrap_detection_delta(
        cand_flags, ship_flags, n=bootstrap, seed=seed + 1
    )
    # Secondary: candidate against its own r = 1 restriction, which isolates
    # the duration form from every other difference between the two rules.
    nest_cand, nest_geom, nest_unmatched = _matched_detection(
        surfaces["negbin"], surfaces["geometric"], inputs.corroborated, alarm_states
    )
    nested_delta = bootstrap_detection_delta(
        nest_cand, nest_geom, n=bootstrap, seed=seed + 3
    )
    held_out = bootstrap_dwell_delta(
        eval_pops.not_moving.obs,
        dwell.pooled,
        dwell.pooled_geometric,
        n=bootstrap,
        seed=seed + 2,
    )

    gates = [
        gate_false_alarms(fa["negbin"], fa["shipped"]),
        gate_dwell_loglik(dwell.nest, held_out),
        gate_detection_support(
            cand_flags, ship_flags, detection_delta, n_unmatched=n_unmatched
        ),
    ]
    verdicts = [g.passed for g in gates]
    return {
        "window": {
            "train_start": inputs.train_start,
            "train_end": inputs.train_end,
            "eval_end": inputs.eval_end,
            "n_train_call_ticks": len(inputs.train_calls),
            "n_eval_call_ticks": len(inputs.eval_calls),
            "n_routes": len(inputs.routes),
            "arm": arm,
            "max_ticks": max_ticks,
        },
        "dwell": {
            "train": _dwell_report(train_pops.not_moving, dwell),
            "eval_n_events": eval_pops.not_moving.n_events,
            "eval_n_censored": eval_pops.not_moving.n_censored,
            "normal_train": {
                "n_events": train_pops.normal.n_events,
                "n_censored": train_pops.normal.n_censored,
                "n_left_truncated_dropped": train_pops.normal.n_left_censored,
                "negbin_mean_minutes": normal_nb.mean_minutes,
                "geometric_mean_minutes": normal_geom.mean_minutes,
            },
        },
        "emissions": {
            name: {
                "false_call": db.emissions.false_call,
                "miss": db.emissions.miss,
                "train_loglik": db.emissions.loglik,
                "n_scored": db.emissions.n_scored,
                "n_evaluations": db.emissions.n_evaluations,
                "false_call_at_bound": db.emissions.at_bound[0],
                "miss_at_bound": db.emissions.at_bound[1],
            }
            for name, db in arms.items()
        },
        "false_alarms": fa,
        "detection": {
            "vs_shipped": {
                "n_matched": len(cand_flags),
                "n_unmatched": n_unmatched,
                "negbin_detected": sum(cand_flags),
                "shipped_detected": sum(ship_flags),
                "paired_delta": detection_delta.as_dict(),
            },
            "vs_nested_geometric": {
                "n_matched": len(nest_cand),
                "n_unmatched": nest_unmatched,
                "negbin_detected": sum(nest_cand),
                "geometric_detected": sum(nest_geom),
                "paired_delta": nested_delta.as_dict(),
            },
        },
        "held_out_dwell_delta": held_out.as_dict(),
        "gates": [g.as_dict() for g in gates],
        "verdict": (
            "pass"
            if all(v is True for v in verdicts)
            else "blocked"
            if any(v is None for v in verdicts)
            else "fail"
        ),
    }


def _dwell_report(pop: DwellPopulation, dwell: DwellModel) -> dict[str, Any]:
    return {
        "n_events": pop.n_events,
        "n_censored": pop.n_censored,
        "n_left_censored_dropped": pop.n_left_censored,
        "negbin": {
            "r": dwell.pooled.r,
            "p": dwell.pooled.p,
            "r_at_bound": dwell.pooled.r_at_bound,
            "mean_minutes": dwell.pooled.mean_minutes,
            "mean_loglik": dwell.pooled.mean_loglik,
        },
        "geometric": {
            "self_loop": dwell.pooled_geometric.self_loop,
            "mean_minutes": dwell.pooled_geometric.mean_minutes,
            "mean_loglik": dwell.pooled_geometric.mean_loglik,
        },
        "nest": {
            "lr_stat": dwell.nest.lr_stat,
            "p_value": dwell.nest.p_value,
            "improves": dwell.nest.improves,
        },
        "n_own_route_fits": dwell.n_own_fits,
        "per_route_nest": {
            r: {
                "lr_stat": t.lr_stat,
                "p_value": t.p_value,
                "r_hat": t.negbin.r,
                "n_events": t.negbin.n_events,
            }
            for r, t in sorted(dwell.nest_by_route.items())
        },
    }


# --- archive access ---------------------------------------------------------


def aligned_window(start: date, end: date) -> tuple[int, int]:
    """Tick-aligned UTC epochs covering [start, end+1day) — the same convention
    `movement_dwell_grade` uses, so two harnesses pointed at the same dates
    grade the same ticks."""
    return (
        int(datetime.combine(start, datetime.min.time(), tzinfo=UTC).timestamp()),
        int(
            datetime.combine(
                end + timedelta(days=1), datetime.min.time(), tzinfo=UTC
            ).timestamp()
        ),
    )


def split_window(
    train_start: date, train_end: date, eval_end: date
) -> tuple[int, int, int]:
    """(train_start, split, eval_end) epochs for an INCLUSIVE date split.

    `train_end` and `eval_end` name days that are IN their halves, so the split
    is midnight AFTER `train_end` and the end is midnight after `eval_end`.
    Both come out of `aligned_window`'s second element, which is what makes
    them exclusive upper bounds.

    Pulled out of `load_inputs` because it is the arithmetic that got this
    wrong once and it needs a test that does not touch the archive: reading the
    split off `aligned_window(train_end, train_end)[0]` puts midnight ON
    train_end, dropping that whole day from the train half and giving it to
    eval, so a 14/7 request silently measured 13/8.
    """
    t0, split = aligned_window(train_start, train_end)
    _, t1 = aligned_window(train_end, eval_end)
    return t0, split, t1


def load_inputs(
    cfg: R2Config,
    client: S3Client,
    *,
    train_start: date,
    train_end: date,
    eval_end: date,
    source: str,
) -> GradeInputs:
    """One archive pass over the whole span, split by tick afterwards.

    The episode population and the assigned_n-confirmed-normal runs are derived
    through the SAME pinned `degradation_label` thresholds
    `movement_validation` uses, so this harness's false-alarm number is scored
    against the same supply label as the 0.00101 reference rather than against
    a differently tuned one.

    The supply baseline is fitted on the train half ONLY, which is
    `movement_validation.main`'s own convention: a sustained outage must never
    lower the baseline it is later judged against.
    """
    from training.degradation_label import BIN_FN, build_labels
    from training.load_r2 import (
        build_service_series,
        compute_baseline,
        fetch_trip_update_metrics,
    )
    from training.segment_coverage import normal_runs as _normal_runs

    t0, split, t1 = split_window(train_start, train_end, eval_end)

    ticks = ticks_for(
        client,
        cfg.bucket,
        train_start,
        eval_end,
        scope="route",
        source=source,
    )
    # SNAP TO THE SHARED 5-MINUTE GRID.
    #
    # `ticks_for` keys its calls on each body's raw `observed_at`, which is
    # whenever the collector actually ran — measured mod 300 over one day:
    # {10, 22, 52, 53, 54, 55, 56, 59, 67, ...}. Every span grader here walks
    # `range(start, end, TICK_SECONDS)` from a supply label that IS on the
    # grid, so an unsnapped call stream misses every single lookup: 162 normal
    # runs and 115 episodes offered, 0 gradeable, on the first real run.
    #
    # This is the same `snap_tick` the dwell harness applies for the same
    # reason. It has to happen before the split so both halves are on one grid.
    snapped: dict[int, dict[str, str]] = {}
    for t, per in ticks:
        snapped.setdefault(snap_tick(t), {}).update(per)
    grid: list[tuple[int, Mapping[str, str]]] = sorted(
        (t, dict(per)) for t, per in snapped.items()
    )
    train_calls = [(t, c) for t, c in grid if t < split]
    eval_calls = [(t, c) for t, c in grid if split <= t < t1]

    svc_bodies = fetch_trip_update_metrics(
        cfg, start_date=train_start, end_date=eval_end, client=client
    )
    all_series = build_service_series(svc_bodies)
    baseline = compute_baseline(
        {k: v for k, v in all_series.items() if k[1] < split}, bin_fn=BIN_FN
    )
    disruptions, labels = build_labels(dict(all_series), dict(baseline))
    # Normal runs are the false-alarm exposure and must be held out: a run
    # starting before the split describes ticks the emissions were fitted on.
    runs = [(r, s, e) for r, s, e in _normal_runs(labels) if s >= split]

    # THE DWELL POPULATION IS THE MOVEMENT EPISODES, NOT THE SUPPLY EPISODES.
    #
    # Gate 2 asks whether an explicit duration beats a memoryless one for the
    # MOVEMENT signal's dwell, so the durations have to be movement episodes.
    # Fitting the assigned_n supply episodes instead answers a different
    # question, and answers it confidently: that wiring produced a not-moving
    # mean of 178 minutes against the ~14 minutes the movement dwell is known
    # to run at, and measured only 1.21% movement-not-normal co-occurrence
    # inside supply-episode ticks (26 of 2146 judged) against a 0.13% base
    # rate. A 9.3x enrichment, so the axes are not independent — but nowhere
    # near the same event, and a duration fitted on one is not a duration model
    # of the other.
    #
    # Segmented through the SAME seam the shipped dwell harness uses
    # (`movement_dwell_grade.eval_episodes`): NOT_NORMAL runs only, standing
    # advisories held out. `not_scheduled` is excluded and that is
    # load-bearing, not tidiness — it is the absence of scheduled service, it
    # is 4825 of this window's 49167 eval calls, and admitting it would grade
    # overnight service gaps as disruptions.
    #
    # The supply feed keeps the two jobs independence actually requires of it:
    # the false-alarm exposure (`runs`) and the detection corroboration
    # (`corroborated`). Neither is a duration.
    def movement_episodes(lo: int, hi: int) -> list[object]:
        truth = {
            (route, tick): state
            for tick, calls in grid
            for route, state in calls.items()
            if lo <= tick < hi and state in NOT_NORMAL
        }
        eps = extract_episodes(truth, {}, window_start=lo, window_end=hi - TICK_SECONDS)
        return [e for e in eps if not e.standing]

    train_eps = movement_episodes(t0, split)
    eval_eps = movement_episodes(split, t1)
    corroborated = [d for d in disruptions if split <= d.start_tick < t1]
    # Routes the MOVEMENT stream actually calls. The supply feed also labels
    # shuttles (FS, SS) the movement arm never scores; handing those to
    # `episode_populations` as normal-sojourn contributors would invent quiet
    # time on routes this surface cannot see.
    routes = sorted({r for _, per in grid for r in per})
    return GradeInputs(
        train_calls=train_calls,
        eval_calls=eval_calls,
        train_episodes=train_eps,
        eval_episodes=eval_eps,
        normal_runs=runs,
        corroborated=list(corroborated),
        routes=routes,
        train_start=t0,
        train_end=split,
        eval_end=t1,
    )


# --- CLI --------------------------------------------------------------------


def _print_report(report: dict[str, Any]) -> None:
    w = report["window"]
    print("=== explicit-duration movement debounce ===")
    print(
        f"arm={w['arm']} age_cap={w['max_ticks']} "
        f"routes={w['n_routes']} train_ticks={w['n_train_call_ticks']} "
        f"eval_ticks={w['n_eval_call_ticks']}"
    )
    d = report["dwell"]["train"]
    print(
        f"\n--- train dwell: {d['n_events']} events, {d['n_censored']} censored, "
        f"{d['n_left_censored_dropped']} left-censored dropped ---"
    )
    print(
        f"negbin   r={d['negbin']['r']:.3f} p={d['negbin']['p']:.4f} "
        f"mean={d['negbin']['mean_minutes']:.1f}min "
        f"mean_ll={d['negbin']['mean_loglik']:.4f}"
        f"{'  (r AT SEARCH BOUND)' if d['negbin']['r_at_bound'] else ''}"
    )
    sl = d["geometric"]["self_loop"]
    print(
        f"geometric self_loop={sl:.4f} mean={d['geometric']['mean_minutes']:.1f}min "
        f"mean_ll={d['geometric']['mean_loglik']:.4f}"
    )
    print(f"\n{'surface':<12} {'FA/tick':>9} {'ci_low':>9} {'ci_high':>9} {'runs':>6}")
    for name, f in report["false_alarms"].items():
        rate = f.get("tick_rate")
        if rate is None:
            print(f"{name:<12} {'-':>9} {'-':>9} {'-':>9} {f['gradeable']:>6}")
            continue
        print(
            f"{name:<12} {rate:>9.5f} {f['tick_rate_ci_low']:>9.5f} "
            f"{f['tick_rate_ci_high']:>9.5f} {f['gradeable']:>6}"
        )
    print("\n--- gates ---")
    for g in report["gates"]:
        mark = {True: "PASS", False: "FAIL", None: "BLOCKED"}[g["passed"]]
        print(f"[{mark:<7}] {g['name']}: {g['statement']}")
    print(f"\nVERDICT: {report['verdict'].upper()}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=(
            "Grade the explicit-duration (negative-binomial) movement debounce "
            "against the memoryless geometric form it nests, on the three "
            "ship/no-ship gates."
        )
    )
    p.add_argument("--train-days", type=int, default=14)
    p.add_argument("--eval-days", type=int, default=7)
    p.add_argument("--train-start", type=str, default=None)
    p.add_argument(
        "--source", default="auto", choices=("auto", "predictions", "vehicles")
    )
    p.add_argument("--arm", default="disrupted", choices=sorted(ARMS))
    p.add_argument("--max-ticks", type=int, default=DWELL_MAX_TICKS)
    p.add_argument("--min-route-events", type=int, default=MIN_ROUTE_DWELL_EVENTS)
    p.add_argument("--bootstrap", type=int, default=BOOTSTRAP_N)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    today = datetime.now(UTC).date()
    if args.train_start:
        train_start = date.fromisoformat(args.train_start)
    else:
        train_start = today - timedelta(days=args.train_days + args.eval_days)
    train_end = train_start + timedelta(days=args.train_days - 1)
    eval_end = train_end + timedelta(days=args.eval_days)

    cfg = load_config()
    client = make_client(cfg)
    inputs = load_inputs(
        cfg,
        client,
        train_start=train_start,
        train_end=train_end,
        eval_end=eval_end,
        source=args.source,
    )
    try:
        report = grade(
            inputs,
            max_ticks=args.max_ticks,
            min_route_events=args.min_route_events,
            bootstrap=args.bootstrap,
            seed=args.seed,
            arm=args.arm,
        )
    except GateBlocked as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
