"""Replay online-FDR control (LORD++ and ADDIS) over the fleet's archived
movement-detector p-values, offline. Nothing here is wired into the worker or
the live surface — it answers one question against the archive: if the existing
per-tick binomial significance gate were wrapped in an online false-discovery
procedure instead of a fixed alpha, what would the fleet-wide alert stream look
like?

WHY
---
The movement detector's disrupted call is a per-(route, direction, tick)
Bayesian screen (posterior advance rate <= DISRUPTED_RATIO x baseline p0)
followed by a binomial significance gate: reject when the lower tail
P(X <= advanced_n | Binom(matched, p0)) <= CLASSIFY_ALPHA
(training.load_r2._binom_lower_tail / classify_direction). That gate bounds the
per-route per-tick false-alarm rate — measured at 0.00101/tick on
assigned_n-confirmed-normal runs (journal 2026-08-31, episode-bootstrap CI
[0.00058, 0.00163]). But the bound is per route: across ~115k route-ticks a day,
even a wholly in-control fleet is expected to emit ~116 false alarms/day. The
per-tick binomial p-value controls the RATE, not the fleet-wide false-discovery
PROPORTION. Online-FDR procedures do exactly that: they adapt the threshold
online so the expected proportion of false discoveries among ALL alerts stays
<= a target level, testing one interleaved fleet-wide stream in causal order.

THE STREAM
----------
The hypotheses tested are the detector's own significance questions: one per
(route, tick) the detector actually screens as a disruption candidate — a
direction that is judgeable (matched >= MIN_MATCHED_TRIPS, a baseline cell
exists) AND whose posterior sits at/under DISRUPTED_RATIO x p0. The p-value is
the binomial lower tail, and a route takes the worst (smallest p) of its
candidate directions, exactly as derive_movement_state takes the worse of the
two directions. Non-candidate ticks are the ones the detector reads normal
without running a significance test; they are not discoveries and are not in the
stream. So wrapping this stream is wrapping the existing significance gate: all
three replayed streams (fixed gate, LORD++, ADDIS) see the identical candidate
p-values in the identical causal order and differ ONLY in the threshold rule.

DEPENDENCE
----------
Adjacent ticks on a route and co-located routes at a tick are positively
dependent (a real freeze persists and spreads). LORD++ and the SAFFRON/ADDIS
family were shown to control FDR not only under independence but under a local
form of positive dependence (an online PRDS condition) by Fisher (2024),
"Online false discovery rate control for LORD++ and SAFFRON under positive,
local dependence" (Biometrical Journal; arXiv:2110.08161), building on Zrnic,
Ramdas & Jordan (2021). No modified recursion is needed for the dependence — the
standard procedures already carry the guarantee — which is why these two are the
dependence-robust choices reported here rather than the independence-only
alpha-investing variants.

ALGORITHMS
----------
LORD++  — Ramdas, Yang, Wainwright & Jordan (2017), "Online control of the false
          discovery rate with decaying memory" (NeurIPS). The `++` improvement
          hands back the full wealth (alpha - w0) at the first rejection.
ADDIS   — Tian & Ramdas (2019), "ADDIS: an adaptive discarding algorithm for
          online FDR control with conservative nulls" (NeurIPS). Adaptive in the
          null fraction (candidates, p <= tau*lambda) and discards conservative
          nulls (p > tau). Ported from the reference onlineFDR implementation.

Run (offline, reads the archive):
  murk exec -- uv run python -m training.online_fdr --days 21 --fit-days 14
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from training.degradation_label import BIN_FN, build_labels
from training.episodes import Episode, extract_episodes
from training.escalation import corroborate_episodes
from training.gtfs_static import load_topology, through_stops
from training.load import TICK_SECONDS
from training.load_r2 import (
    CLASSIFY_ALPHA,
    CLASSIFY_PRIOR_STRENGTH,
    DISRUPTED_RATIO,
    MIN_MATCHED_TRIPS,
    AdvanceBaseline,
    _binom_lower_tail,  # pyright: ignore[reportPrivateUsage]
    build_movement_series_by_direction,
    build_movement_truth,
    compute_advance_baseline,
    compute_baseline,
    fetch_vehicle_metrics,
    tod_bin,
)
from training.movement_validation import (
    BOOTSTRAP_N,
    Unit,
    _boot_rates,  # pyright: ignore[reportPrivateUsage]
    _stop_filter,  # pyright: ignore[reportPrivateUsage]
)
from training.segment_coverage import normal_runs

# Target fleet-wide FDR the online procedures are tuned to hold. 0.05 is the
# gate's newly-bounded fleet false-discovery proportion (vpra), deliberately the
# same 5% the per-tick binomial gate uses as its per-route alpha so the two
# levels are read on one scale.
TARGET_FDR = 0.05


# --------------------------------------------------------------------------
# The candidate p-value stream
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Candidate:
    """One disruption-candidate hypothesis: (route, tick) the detector screened
    and its binomial lower-tail p-value (the worst candidate direction). `p` is
    exactly what the fixed gate compares against CLASSIFY_ALPHA."""

    tick: int
    route: str
    p: float


def _direction_pvalue(
    row: Mapping[str, int] | None,
    baseline: AdvanceBaseline | None,
    *,
    prior_strength: float,
    disrupted_ratio: float,
    min_matched: int,
    posterior_screen: bool,
) -> float | None:
    """The binomial lower-tail p-value for one (route, direction, tick), or None
    when the direction is not judgeable (fewer than min_matched matches, or no
    baseline cell).

    With `posterior_screen` (the deployed operating point), it additionally
    returns None unless the direction is a disruption candidate — posterior
    advance rate at/under disrupted_ratio x p0 — mirroring
    load_r2.classify_direction, which only computes the binomial tail AFTER that
    Bayesian screen. Without it, every judgeable direction contributes its
    p-value: the counterfactual where the binomial test alone is the fleet-wide
    surface (the ~116/day premise), so an online-FDR layer has real multiplicity
    to control."""
    if row is None or baseline is None:
        return None
    advanced = row.get("advanced_n", 0)
    matched = advanced + row.get("stalled_n", 0)
    if matched < min_matched:
        return None
    if posterior_screen:
        post = (prior_strength * baseline.p0 + advanced) / (prior_strength + matched)
        if post > disrupted_ratio * baseline.p0:
            return None
    return _binom_lower_tail(advanced, matched, baseline.p0)


def candidate_stream(
    dir_series: Mapping[tuple[str, str, int], dict[str, int]],
    baseline: Mapping[tuple[str, str, int], AdvanceBaseline],
    *,
    prior_strength: float = CLASSIFY_PRIOR_STRENGTH,
    disrupted_ratio: float = DISRUPTED_RATIO,
    min_matched: int = MIN_MATCHED_TRIPS,
    posterior_screen: bool = True,
) -> list[Candidate]:
    """The causal fleet-wide p-value stream from the per-direction movement
    series and its causal advance baseline.

    A route enters at a tick when at least one direction contributes a p-value;
    the route takes the smallest (worst direction), the same reduction
    derive_movement_state uses. Ordered by (tick, route) so the fleet-wide
    sequence is deterministic and strictly causal: every tick's hypotheses
    precede the next tick's, and within a tick routes are ordered by name (a
    within-tick tie-break the theory is indifferent to). `posterior_screen`
    selects the deployed operating-point stream (screened candidates) vs the full
    binomial surface — see _direction_pvalue."""
    by_route_tick: dict[tuple[int, str], float] = {}
    for (route, direction, tick), row in dir_series.items():
        p = _direction_pvalue(
            row,
            baseline.get((route, direction, tod_bin(tick))),
            prior_strength=prior_strength,
            disrupted_ratio=disrupted_ratio,
            min_matched=min_matched,
            posterior_screen=posterior_screen,
        )
        if p is None:
            continue
        key = (tick, route)
        prev = by_route_tick.get(key)
        if prev is None or p < prev:
            by_route_tick[key] = p
    return [
        Candidate(tick=tick, route=route, p=p)
        for (tick, route), p in sorted(by_route_tick.items())
    ]


# --------------------------------------------------------------------------
# Online-FDR procedures. Each is an online tester: feed it the stream's
# p-values in order, it returns a reject/keep decision per hypothesis, its
# threshold adapting to the rejections so far. Nothing looks ahead.
# --------------------------------------------------------------------------


def lord_gamma(j: int) -> float:
    """LORD++ default spending sequence (Ramdas et al. 2017, matching onlineFDR):
    gamma_j = C * log(max(j, 2)) / (j * exp(sqrt(log j))), C = 0.07720838,
    a nonincreasing sequence summing to ~1. gamma_j = 0 for j <= 0 (only past
    rejections contribute wealth)."""
    if j <= 0:
        return 0.0
    return 0.07720838 * math.log(max(j, 2)) / (j * math.exp(math.sqrt(math.log(j))))


# SAFFRON/ADDIS default spending sequence: gamma_j proportional to j^-1.6,
# normalized to sum to 1 (constant from the reference onlineFDR implementation).
_ADDIS_GAMMA_C = 0.4374901658


def addis_gamma(j: int) -> float:
    if j <= 0:
        return 0.0
    return _ADDIS_GAMMA_C / (j**1.6)


class LordPlusPlus:
    """LORD++ (Ramdas, Yang, Wainwright & Jordan 2017). Test level for the i-th
    hypothesis (1-indexed):

        alpha_i = gamma_i * w0
                + (alpha - w0) * gamma_{i - tau_1}
                + alpha * sum_{j>=2} gamma_{i - tau_j}

    where tau_1 < tau_2 < ... are the rejection times. Reject when p_i <= alpha_i.
    The `++` refinement is the (alpha - w0) coefficient on the first rejection:
    the full remaining wealth is returned at the first discovery."""

    def __init__(
        self,
        *,
        alpha: float = TARGET_FDR,
        w0: float | None = None,
        gamma: Any = lord_gamma,
    ) -> None:
        self.alpha = alpha
        self.w0 = alpha / 2 if w0 is None else w0
        self.gamma = gamma
        self.t = 0  # hypotheses tested so far
        self.rejections: list[int] = []  # 1-indexed rejection times

    def test(self, p: float) -> bool:
        self.t += 1
        i = self.t
        level = self.gamma(i) * self.w0
        if self.rejections:
            tau1 = self.rejections[0]
            level += (self.alpha - self.w0) * self.gamma(i - tau1)
            for tj in self.rejections[1:]:
                level += self.alpha * self.gamma(i - tj)
        reject = p <= level
        if reject:
            self.rejections.append(i)
        return reject


@dataclass
class _AddisRejection:
    kappa_index: int  # 1-indexed hypothesis position of the rejection
    kappa_star: int  # selected count through kappa_index (inclusive)
    cand_after: int = 0  # candidates strictly after kappa_index seen so far


class Addis:
    """ADDIS (Tian & Ramdas 2019), ported from the reference onlineFDR
    implementation. Discards conservative nulls (p > tau) and is adaptive in the
    null fraction via candidates (p <= tau*lambda). Test level:

        alpha_t = min(tau*lambda, w0 * gamma[S - C0]
                  + (tau(1-lambda)alpha - w0) * gamma[S - kappa*_1 - C_1]
                  + tau(1-lambda)alpha * sum_{j>=2} gamma[S - kappa*_j - C_j])

    with S the count of selected (p <= tau) hypotheses before t, kappa*_j the
    selected count through the j-th rejection, C_j the candidate count after it,
    and C0 the candidate count before t. Reject when p_t <= alpha_t. Defaults
    lambda = tau = 0.5, w0 = tau*lambda*alpha/2.

    TWO CONVENTIONS, one algorithm (do not re-flag as a bug). The paper's
    Algorithm 1 (ADDIS*) writes the candidate threshold as an UNSCALED lambda_p
    with lambda_p < tau, an outer (tau - lambda_p) factor on the whole wealth
    bracket, and initial wealth W0 <= alpha. The onlineFDR reference
    implementation ported here re-parameterizes lambda as a RATIO: its candidate
    threshold is tau*lambda (= lambda_p), its cap is tau*lambda, and it folds the
    outer (tau - lambda_p) = tau*(1 - lambda) factor into each wealth term, so the
    coefficient is tau*(1 - lambda)*alpha and the default w0 = tau*lambda*alpha/2
    equals (tau - lambda_p) * (alpha/2), i.e. the paper's W0 = alpha/2 scaled the
    same way. The two are algebraically identical: with the default lambda = 0.5,
    tau = 0.5 the candidate threshold is 0.25 = lambda_p < tau, satisfying the
    paper's lambda_p < tau, and test_addis_matches_paper_algorithm1 checks the two
    formulations agree rejection-for-rejection on the reference example."""

    def __init__(
        self,
        *,
        alpha: float = TARGET_FDR,
        lam: float = 0.5,
        tau: float = 0.5,
        w0: float | None = None,
        gamma: Any = addis_gamma,
    ) -> None:
        # The candidate threshold is tau*lambda and the discard threshold is tau,
        # so both must be proper probabilities; the default 0.5/0.5 puts the
        # candidate cut at 0.25 and discards p > 0.5.
        if not (0.0 < lam <= 1.0 and 0.0 < tau <= 1.0):
            raise ValueError(f"require lambda, tau in (0, 1], got {lam}, {tau}")
        self.alpha = alpha
        self.lam = lam
        self.tau = tau
        self.w0 = tau * lam * alpha / 2 if w0 is None else w0
        self.gamma = gamma
        self.t = 0
        self.s_prev = 0  # selected count through t-1
        self.cand_sum = 0  # candidate count through t-1
        self.rejections: list[_AddisRejection] = []
        self._pending_selected = 0  # selected/cand indicators of hypothesis t-1,
        self._pending_cand = 0  # folded in at the next test() (the ADDIS lag)
        self._pending_index = 0

    def _g(self, k: int) -> float:
        return self.gamma(k)

    def test(self, p: float) -> bool:
        self.t += 1
        i = self.t
        # Fold in the previous hypothesis (index i-1): S and the candidate counts
        # used at step i are cumulative THROUGH i-1 (the reference's one-step lag).
        if self._pending_index:
            self.s_prev += self._pending_selected
            self.cand_sum += self._pending_cand
            if self._pending_cand:
                for rej in self.rejections:
                    if rej.kappa_index < self._pending_index:
                        rej.cand_after += 1

        cand_cap = self.tau * self.lam
        coef = self.tau * (1.0 - self.lam) * self.alpha
        base = self.w0 * self._g(self.s_prev - self.cand_sum + 1)
        k = len(self.rejections)
        if k == 0:
            level_hat = base
        else:
            first = self.rejections[0]
            g_first = self._g(self.s_prev - first.kappa_star - first.cand_after + 1)
            level_hat = base + (coef - self.w0) * g_first
            if k > 1:
                total = sum(
                    self._g(self.s_prev - r.kappa_star - r.cand_after + 1)
                    for r in self.rejections
                )
                level_hat += coef * (total - g_first)
        level = min(cand_cap, level_hat)

        selected_i = 1 if p <= self.tau else 0
        cand_i = 1 if p <= cand_cap else 0
        reject = p <= level
        if reject:
            # kappa_star is the selected count through i INCLUSIVE.
            self.rejections.append(
                _AddisRejection(kappa_index=i, kappa_star=self.s_prev + selected_i)
            )
        self._pending_selected = selected_i
        self._pending_cand = cand_i
        self._pending_index = i
        return reject


def replay(stream: Sequence[Candidate], tester: Any) -> set[tuple[str, int]]:
    """Run an online tester over the causal stream, returning the set of
    (route, tick) it rejects. `tester` is any object with a `.test(p) -> bool`
    method (LordPlusPlus / Addis); the fixed gate is wrapped to match."""
    out: set[tuple[str, int]] = set()
    for c in stream:
        if tester.test(c.p):
            out.add((c.route, c.tick))
    return out


class _FixedGate:
    def __init__(self, alpha: float = CLASSIFY_ALPHA) -> None:
        self.alpha = alpha

    def test(self, p: float) -> bool:
        return p <= self.alpha


# --------------------------------------------------------------------------
# Grading one replayed alert stream against the three gate dimensions.
# --------------------------------------------------------------------------


def _span_ticks(start: int, end: int) -> range:
    return range(start, end, TICK_SECONDS)


def false_alarm_bound(
    rejections: set[tuple[str, int]],
    judged: set[tuple[str, int]],
    runs: Sequence[tuple[str, int, int]],
    *,
    bootstrap: int = BOOTSTRAP_N,
    seed: int = 0,
) -> dict[str, Any]:
    """The per-route false-alarm tick-rate on assigned_n-confirmed-normal runs —
    the same 0.00101/tick upper bound movement_validation measures, recomputed
    for this stream's rejections. One Unit per normal run, ticks counted only
    where the route was actually judged (in `judged`), alarmed where the stream
    rejected. Episode/run-level cluster bootstrap (never ticks), reusing
    movement_validation._boot_rates."""
    units: list[Unit] = []
    for route, start, end in runs:
        ticks = alarmed = 0
        for tick in _span_ticks(start, end):
            if (route, tick) not in judged:
                continue
            ticks += 1
            if (route, tick) in rejections:
                alarmed += 1
        if ticks > 0:
            units.append(Unit(alarmed_ticks=alarmed, ticks=ticks))
    boot = _boot_rates(units, n=bootstrap, seed=seed)
    return {"gradeable": len(units), "offered": len(runs), **boot}


def _episode_hit(
    rejections: set[tuple[str, int]], ep: Episode, judged: set[tuple[str, int]]
) -> tuple[bool, bool]:
    """(scored, hit) for one episode: scored when the route was judged on at least
    one tick in its span (else absence of evidence, not a miss — the same rule
    movement_validation uses); hit when the stream rejected on any scored tick."""
    scored = hit = False
    for tick in _span_ticks(ep.onset, ep.recovery):
        if (ep.route, tick) not in judged:
            continue
        scored = True
        if (ep.route, tick) in rejections:
            hit = True
            break
    return scored, hit


def _boot_share(flags: Sequence[bool], *, n: int, seed: int) -> dict[str, Any]:
    """Cluster bootstrap of a share over independent units (episodes), withholding
    the interval below 2 units where every resample is degenerate."""
    k = len(flags)
    out: dict[str, Any] = {"n": k, "n_true": sum(flags)}
    if k == 0:
        return out
    out["rate"] = sum(flags) / k
    if k < 2:
        return out  # a CI on one cluster is fabricated
    rng = random.Random(seed)
    draws = sorted(sum(rng.choice(flags) for _ in range(k)) / k for _ in range(n))
    lo, hi = int(0.025 * n), min(n - 1, int(0.975 * n))
    out["ci_low"], out["ci_high"] = draws[lo], draws[hi]
    return out


def corroborated_retention(
    rejections: set[tuple[str, int]],
    corroborated: Sequence[Episode],
    judged: set[tuple[str, int]],
    *,
    bootstrap: int = BOOTSTRAP_N,
    seed: int = 0,
) -> dict[str, Any]:
    """Of the escalation-corroborated movement episodes (the ones an MTA alert
    independently confirms), how many does this stream still fire inside?
    Episodes whose route is never judged in-span are excluded (absence of
    evidence), counted separately. Episode-level bootstrap."""
    flags: list[bool] = []
    n_unscored = 0
    for ep in corroborated:
        scored, hit = _episode_hit(rejections, ep, judged)
        if not scored:
            n_unscored += 1
            continue
        flags.append(hit)
    return {"n_unscored": n_unscored, **_boot_share(flags, n=bootstrap, seed=seed)}


def onset_backdating(
    rejections: set[tuple[str, int]],
    corroborated: Sequence[Episode],
    judged: set[tuple[str, int]],
) -> dict[str, Any]:
    """Latency from each retained corroborated episode's onset to the stream's
    first rejection inside it (minutes). "Back-dating" because a smaller number is
    an earlier alarm. Reported over episodes the stream actually fires in, so it
    is a latency conditional on detection, read beside the retention rate."""
    lat: list[float] = []
    for ep in corroborated:
        first: int | None = None
        for tick in _span_ticks(ep.onset, ep.recovery):
            if (ep.route, tick) not in judged:
                continue
            if (ep.route, tick) in rejections:
                first = tick
                break
        if first is not None:
            lat.append((first - ep.onset) / 60.0)
    lat.sort()
    return {
        "n_detected": len(lat),
        "median_latency_min": statistics.median(lat) if lat else None,
        "p90_latency_min": lat[min(len(lat) - 1, int(0.9 * len(lat)))] if lat else None,
    }


def realized_fdp(
    rejections: set[tuple[str, int]],
    runs: Sequence[tuple[str, int, int]],
    corroborated: Sequence[Episode],
    *,
    bootstrap: int = BOOTSTRAP_N,
    seed: int = 0,
) -> dict[str, Any]:
    """The realized fleet false-discovery proportion over the ADJUDICABLE
    discoveries: a rejection on an assigned_n-confirmed-normal run is a false
    discovery; a rejection inside an escalation-corroborated episode is a true
    discovery; every other rejection (unconfirmed episode, unjudged supply) is
    set aside and counted, never guessed — the same discipline escalation.py
    holds, because alert absence cannot refute an episode.

    FDP = false / (false + true). The bootstrap resamples the adjudicable
    CLUSTERS (each normal run and each corroborated episode contributes its
    rejection tally as one cluster), never ticks; the interval is withheld unless
    at least 2 clusters carry a discovery on each side."""
    # Corroboration takes precedence over the supply-normal false label: a
    # confirmed-normal SUPPLY run and a corroborated MOVEMENT episode live on
    # different axes and can overlap (supply normal while trains are frozen — the
    # exact case the false-alarm "upper bound" cannot separate). A rejection an
    # alert independently confirms is a true discovery even if supply read normal,
    # so it is never also charged as a false one.
    true_keys = {
        (ep.route, tick)
        for ep in corroborated
        for tick in _span_ticks(ep.onset, ep.recovery)
        if (ep.route, tick) in rejections
    }
    run_clusters: list[tuple[int, int]] = []  # (false, true=0)
    for route, start, end in runs:
        f = sum(
            1
            for tick in _span_ticks(start, end)
            if (route, tick) in rejections and (route, tick) not in true_keys
        )
        run_clusters.append((f, 0))
    ep_clusters: list[tuple[int, int]] = []  # (false=0, true)
    for ep in corroborated:
        t = sum(
            1
            for tick in _span_ticks(ep.onset, ep.recovery)
            if (ep.route, tick) in rejections
        )
        ep_clusters.append((0, t))

    false_n = sum(f for f, _ in run_clusters)
    true_n = sum(t for _, t in ep_clusters)
    total = false_n + true_n
    n_false_clusters = sum(1 for f, _ in run_clusters if f > 0)
    n_true_clusters = sum(1 for _, t in ep_clusters if t > 0)
    out: dict[str, Any] = {
        "false_discoveries": false_n,
        "true_discoveries": true_n,
        "total_rejections": len(rejections),
        "adjudicable": total,
        "n_false_clusters": n_false_clusters,
        "n_true_clusters": n_true_clusters,
    }
    if total == 0:
        return out
    out["fdp"] = false_n / total
    if n_false_clusters < 2 or n_true_clusters < 2:
        return out  # a ratio bootstrap needs >=2 clusters bearing a discovery each side
    clusters = run_clusters + ep_clusters
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(bootstrap):
        sample = [rng.choice(clusters) for _ in range(len(clusters))]
        f = sum(x for x, _ in sample)
        t = sum(y for _, y in sample)
        draws.append(f / (f + t) if (f + t) else 0.0)
    draws.sort()
    lo, hi = int(0.025 * bootstrap), min(bootstrap - 1, int(0.975 * bootstrap))
    out["fdp_ci_low"], out["fdp_ci_high"] = draws[lo], draws[hi]
    return out


def grade_stream(
    rejections: set[tuple[str, int]],
    *,
    judged: set[tuple[str, int]],
    runs: Sequence[tuple[str, int, int]],
    corroborated: Sequence[Episode],
    n_days: float,
    bootstrap: int = BOOTSTRAP_N,
    seed: int = 0,
) -> dict[str, Any]:
    """One replayed alert stream scored on all three gate dimensions plus fleet
    volume: (a) the per-route false-alarm bound and the realized fleet FDP, (b)
    escalation-corroborated episodes retained, (c) onset back-dating latency."""
    fa = false_alarm_bound(rejections, judged, runs, bootstrap=bootstrap, seed=seed)
    fdp = realized_fdp(rejections, runs, corroborated, bootstrap=bootstrap, seed=seed)
    fa_ticks = fa.get("alarmed_ticks", 0)
    return {
        "n_rejections": len(rejections),
        "false_alarms_per_day": (fa_ticks / n_days) if n_days else None,
        "per_route_false_alarm": fa,
        "realized_fdp": fdp,
        "corroborated_retention": corroborated_retention(
            rejections, corroborated, judged, bootstrap=bootstrap, seed=seed
        ),
        "onset_backdating": onset_backdating(rejections, corroborated, judged),
    }


def gate_verdict(stream: dict[str, Any]) -> dict[str, Any]:
    """Where one stream lands against the two-part gate: per-route FA tick-rate
    at/under 0.00101, AND realized fleet FDP at/under TARGET_FDR. `None` where the
    support is too thin to have measured the number (never read as a pass)."""
    fa = stream["per_route_false_alarm"].get("tick_rate")
    fdp = stream["realized_fdp"].get("fdp")
    return {
        "per_route_fa_tick_rate": fa,
        "per_route_bound_met": (fa <= 0.00101) if fa is not None else None,
        "fleet_fdp": fdp,
        "fleet_fdp_met": (fdp <= TARGET_FDR) if fdp is not None else None,
    }


# --------------------------------------------------------------------------
# The three-stream comparison, pure over its inputs.
# --------------------------------------------------------------------------


def build_comparison(
    stream: Sequence[Candidate],
    *,
    movement_truth: Mapping[tuple[str, int], str],
    runs: Sequence[tuple[str, int, int]],
    alert_disrupted: set[tuple[str, int]],
    window_start: int,
    window_end: int,
    n_days: float,
    target_fdr: float = TARGET_FDR,
    bootstrap: int = BOOTSTRAP_N,
    seed: int = 0,
) -> dict[str, Any]:
    """Replay the fixed gate, LORD++ and ADDIS over one candidate stream and
    grade all three. Movement episodes (the retention/latency unit) are cut from
    the fixed-gate movement truth — the deployed surface — and corroborated
    against the independent MTA alert feed, so every stream is judged on the SAME
    reference episodes rather than on episodes it defined for itself.

    `judged` is every (route, tick) the movement channel could form a call on
    (movement_truth's keys): the honest denominator for a false-alarm rate and
    the scored set for episode detection."""
    judged = set(movement_truth.keys())

    # The reference movement episodes and which ones an alert independently
    # corroborates. extract_episodes reads a (route, tick) -> not-normal state
    # map; movement_truth already is one (normal ticks are graded as normal).
    movement_eps = extract_episodes(
        dict(movement_truth), {}, window_start=window_start, window_end=window_end
    )
    corroborations = corroborate_episodes(movement_eps, alert_disrupted)
    corroborated = [c.episode for c in corroborations if c.confirmed]

    testers: dict[str, Any] = {
        "fixed_gate": _FixedGate(CLASSIFY_ALPHA),
        "lord_pp": LordPlusPlus(alpha=target_fdr),
        "addis": Addis(alpha=target_fdr),
    }
    streams: dict[str, Any] = {}
    for i, (name, tester) in enumerate(testers.items()):
        rejections = replay(stream, tester)
        graded = grade_stream(
            rejections,
            judged=judged,
            runs=runs,
            corroborated=corroborated,
            n_days=n_days,
            bootstrap=bootstrap,
            seed=seed + i * 10,
        )
        graded["gate_verdict"] = gate_verdict(graded)
        streams[name] = graded

    return {
        "stream": {
            "n_candidates": len(stream),
            "n_candidate_routes": len({c.route for c in stream}),
            "n_candidate_ticks": len({c.tick for c in stream}),
            "p_min": min((c.p for c in stream), default=None),
            "p_median": statistics.median([c.p for c in stream]) if stream else None,
        },
        "reference": {
            "n_movement_episodes": len(movement_eps),
            "n_corroborated": len(corroborated),
            "n_normal_runs": len(runs),
            "n_judged_route_ticks": len(judged),
            "target_fdr": target_fdr,
            "note": (
                "false discovery = rejection on an assigned_n-confirmed-normal "
                "run (per-route bound holds it <= 0.00101/tick); fleet FDP = "
                "false / (false + corroborated-episode) discoveries <= target_fdr"
            ),
        },
        "streams": streams,
    }


# --------------------------------------------------------------------------
# main: fetch the archive, build the causal stream, replay, print the table.
# --------------------------------------------------------------------------


def _print_table(report: dict[str, Any], label: str) -> None:
    st = report["stream"]
    ref = report["reference"]
    print(f"\n===== {label} =====", file=sys.stderr)
    pmin = f"{st['p_min']:.2e}" if st["p_min"] is not None else "—"
    pmed = f"{st['p_median']:.3f}" if st["p_median"] is not None else "—"
    print(
        f"candidate p-value stream: {st['n_candidates']} hypotheses "
        f"({st['n_candidate_ticks']} ticks x {st['n_candidate_routes']} routes), "
        f"p_min={pmin} p_median={pmed}",
        file=sys.stderr,
    )
    print(
        f"reference: {ref['n_movement_episodes']} movement episodes, "
        f"{ref['n_corroborated']} escalation-corroborated, "
        f"{ref['n_normal_runs']} confirmed-normal runs, "
        f"{ref['n_judged_route_ticks']} judged route-ticks, target FDR "
        f"{ref['target_fdr']}",
        file=sys.stderr,
    )
    hdr = (
        f"\n{'stream':<12} {'alerts':>8} {'FA/day':>8} {'FA tick-rate':>22} "
        f"{'fleet FDP':>22} {'corrob. kept':>18} {'onset med':>10}  gate"
    )
    print(hdr, file=sys.stderr)
    for name, s in report["streams"].items():
        fa = s["per_route_false_alarm"]
        fr = fa.get("tick_rate")
        fr_lo, fr_hi = fa.get("tick_rate_ci_low"), fa.get("tick_rate_ci_high")
        fdp = s["realized_fdp"]
        ret = s["corroborated_retention"]
        onset = s["onset_backdating"]["median_latency_min"]
        gv = s["gate_verdict"]
        fr_s = (
            f"{fr:.5f}[{fr_lo:.5f},{fr_hi:.5f}]"
            if fr is not None and fr_lo is not None
            else (f"{fr:.5f}" if fr is not None else "—")
        )
        fdp_v = fdp.get("fdp")
        fdp_lo, fdp_hi = fdp.get("fdp_ci_low"), fdp.get("fdp_ci_high")
        fdp_s = (
            f"{fdp_v:.3f}[{fdp_lo:.3f},{fdp_hi:.3f}]"
            if fdp_v is not None and fdp_lo is not None
            else (f"{fdp_v:.3f}" if fdp_v is not None else "—")
        )
        ret_v = ret.get("rate")
        ret_s = (
            f"{ret_v:.3f} ({ret.get('n_true')}/{ret.get('n')})"
            if ret_v is not None
            else f"— ({ret.get('n', 0)})"
        )
        onset_s = f"{onset:.1f}" if onset is not None else "—"
        verdict = f"route:{gv['per_route_bound_met']} fdp:{gv['fleet_fdp_met']}"
        print(
            f"{name:<12} {s['n_rejections']:>8} "
            f"{s['false_alarms_per_day']:>8.2f} {fr_s:>22} {fdp_s:>22} "
            f"{ret_s:>18} {onset_s:>10}  {verdict}",
            file=sys.stderr,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay LORD++ and ADDIS online-FDR control over the archived "
        "movement-detector candidate p-value stream, comparing the fixed gate, "
        "LORD++ and ADDIS on the per-route false-alarm bound, realized fleet FDP, "
        "escalation-corroborated episodes retained, and onset latency."
    )
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument("--days", type=int, default=21, help="total trailing window")
    parser.add_argument(
        "--fit-days", type=int, default=14, help="leading days to fit baselines"
    )
    parser.add_argument("--target-fdr", type=float, default=TARGET_FDR)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP_N)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    # Imported lazily so the pure half above needs no R2 / alert-truth deps.
    from training.r2_client import load_config, make_client
    from training.review import build_mta_truth

    today = datetime.now(UTC).date()
    end = args.end_date or today
    start = args.start_date or (end - timedelta(days=args.days - 1))
    fit_end = start + timedelta(days=args.fit_days - 1)
    if fit_end >= end:
        print("fit window leaves nothing to score", file=sys.stderr)
        return 1
    score_start = fit_end + timedelta(days=1)
    n_days = (end - score_start).days + 1

    cfg = load_config()
    client = make_client(cfg)

    try:
        successors, _patterns = load_topology()
        through = through_stops(successors)
        topology_source = "gtfs_static"
    except Exception as exc:  # pragma: no cover - operational fallback
        print(f"gtfs static unavailable ({exc}); scoring every stop", file=sys.stderr)
        through, topology_source = None, "observed"
    stop = _stop_filter(through)

    print(f"fitting advance baseline {start}..{fit_end}", file=sys.stderr)
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
    dir_series = build_movement_series_by_direction(score_veh, counts_from_stop=stop)
    # Two surfaces from the same archive: the deployed operating point (the
    # binomial test only where the Bayesian posterior already flagged a candidate)
    # and the full binomial surface (every judgeable route-tick), the ~116/day
    # premise where the significance test alone carries the fleet multiplicity.
    stream_op = candidate_stream(dir_series, advance_baseline, posterior_screen=True)
    stream_full = candidate_stream(dir_series, advance_baseline, posterior_screen=False)
    movement_truth = build_movement_truth(
        score_veh, movement_baseline=advance_baseline, counts_from_stop=stop
    )
    print(
        f"{len(fit_veh)} fit ticks -> {len(advance_baseline)} advance cells; "
        f"operating-point stream {len(stream_op)} / binomial surface "
        f"{len(stream_full)} hypotheses over {len(movement_truth)} judged "
        f"route-ticks",
        file=sys.stderr,
    )

    # Independent supply truth (assigned_n) for the confirmed-normal runs, fit
    # causally on the leading window (movement_validation's construction).
    from training.load_r2 import build_service_series, fetch_trip_update_metrics

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
    score_series = {k: v for k, v in all_series.items() if k[1] >= boundary}
    _disruptions, labels = build_labels(dict(score_series), dict(svc_baseline))
    runs = normal_runs(labels)

    # The independent MTA alert feed for escalation corroboration.
    print(f"fetching alert truth {score_start}..{end}", file=sys.stderr)
    alert_truth = build_mta_truth(client, cfg.bucket, score_start, end)
    alert_disrupted = {k for k, v in alert_truth.items() if v != "normal"}

    window_start = int(
        datetime.combine(score_start, datetime.min.time(), tzinfo=UTC).timestamp()
    )
    window_end = int(
        datetime.combine(
            end + timedelta(days=1), datetime.min.time(), tzinfo=UTC
        ).timestamp()
    )
    window = {
        "fit_start": start.isoformat(),
        "fit_end": fit_end.isoformat(),
        "score_start": score_start.isoformat(),
        "score_end": end.isoformat(),
        "score_days": n_days,
        "topology_source": topology_source,
    }
    surfaces: dict[str, Any] = {}
    for i, (label, stream) in enumerate(
        (("operating_point", stream_op), ("binomial_surface", stream_full))
    ):
        surfaces[label] = build_comparison(
            stream,
            movement_truth=movement_truth,
            runs=runs,
            alert_disrupted=alert_disrupted,
            window_start=window_start,
            window_end=window_end,
            n_days=n_days,
            target_fdr=args.target_fdr,
            bootstrap=args.bootstrap,
            seed=args.seed + i * 1000,
        )
        _print_table(surfaces[label], label)
    print(json.dumps({"window": window, "surfaces": surfaces}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
