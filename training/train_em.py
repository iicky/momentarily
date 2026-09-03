"""Per-route EM trainer with empirical-Bayes prior anchoring.

Each run:
  1. Pull the alerts archive over a window (default 14 days) from R2.
  2. Pool all routes into one corpus and fit a global HMM — the prior.
  3. For each route with enough data, fit again with `prior_params=global`
     and Dirichlet/Gamma/Beta pseudo-counts (`prior_strength`).
  4. Routes with thin data inherit the global prior as-is.
  5. Write state/params.json (live pointer) + state/params/v<epoch>.json
     (immutable per-run snapshot) — the Worker picks up params.json on its
     next cron tick; the versioned copies are the rollback trail.

Run with:
    murk exec -- python -m training.train_em [--days 14] [--start/--end DATE]
        [--routes A,C,E] [--min-ticks N] [--prior-strength 100] [--dry-run]

params.json records what it took to produce it — provenance.code_sha, the
hyperparams block (resolved window + prior_strength + min_ticks + routes), and
training_corpus.input_blake3. Against the immutable archive, re-running this
tool at that code_sha with that hyperparams block reproduces the version.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from momentarily.hmm import (
    STATES,
    EmissionParams,
    HMMParams,
    Observation,
    advance_responsibility,
    fit_em,
    schedule_bin,
    service_responsibility,
)
from momentarily.mapping import CANONICAL_SEVERITY_FLOOR, LEGACY_SEVERITY_FLOOR
from training.drift import build_input_profile
from training.dwell import (
    DwellQuantiles,
    compute_dwell_quantiles,
    compute_dwell_quantiles_by_alert,
    compute_dwell_quantiles_by_cause,
)
from training.gtfs_static import (
    RoutePattern,
    SegmentKey,
    dominant_successor,
    load_topology,
    patterns_to_json,
    stops_to_json,
    through_stops,
)
from training.load import TICK_SECONDS, TickObservation, fill_quiet_ticks
from training.load_r2 import (
    MIN_THROUGHPUT_TICKS,
    SERVICE_MIN_NIGHTS,
    StopFilter,
    advance_baseline_to_json,
    build_movement_series,
    build_movement_series_by_direction,
    build_segment_baseline,
    build_segment_throughput,
    build_service_series,
    build_tick_observations,
    compute_advance_baseline,
    compute_advance_baseline_by_route,
    compute_baseline,
    compute_schedule_rate,
    compute_service_quantiles,
    fetch_objects,
    fetch_trip_update_metrics,
    fetch_vehicle_metrics,
    input_manifest_hash,
    list_alert_keys,
    movement_observation_fields,
    presence_mask_from_predictions,
    schedule_rate_to_json,
    service_baseline_to_json,
    service_observation_fields,
    service_quantiles_to_json,
    throughput_to_json,
)
from training.pooled_dwell import MIN_VOTER_EVENTS, pooled_dwell_cells
from training.provenance import code_provenance
from training.r2_client import R2Config, load_config, make_client
from training.recovery_recalibration import (
    fit_published_recovery_gamma,
    recalibrate_dwell_cells,
    recalibrate_dwell_cells_by_key,
)
from training.reliability import MIN_SHARE
from training.run_filter import BOOTSTRAP_PARAMS
from training.segment_dwell import SegmentDwellStats, build_segment_dwell
from training.segments import canonical_adjacency
from training.survival import loglogistic_tail

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


PARAMS_KEY = "state/params.json"
# Immutable per-run snapshots live under this prefix as v<trained_at>.json.
VERSIONED_PARAMS_PREFIX = "state/params/"
SCHEMA_VERSION = "1"

# A route needs at least this many ticks of data to fit per-route — under that,
# we fall back to the global prior.
MIN_TICKS_PER_ROUTE = 288  # one day at the 5-min grid


@dataclass(frozen=True)
class CorpusStats:
    """Audit metadata about the archive window a run actually trained on."""

    start_tick: int
    end_tick: int
    n_observations: int  # real (alert-bearing) tick-observations, pre-quiet-fill
    n_input_versions: int = 0  # archived alert-version objects that fed the fit
    input_blake3: str = ""  # BLAKE3 over those object keys — lineage fingerprint

    @property
    def span_seconds(self) -> int:
        return self.end_tick - self.start_tick


# EM on a thin or mostly-quiet corpus drives transition self-loops toward 1.0, which
# pins the forward filter so a route can never leave a regime. Cap the diagonal, and
# refuse to publish at all under a week of archive. The original bound was two weeks;
# once the EM variance/Bernoulli floors landed the dominant risk of thin data —
# degenerate emissions — was no longer in play, so we relaxed the gate. _cap_self_loops
# still bounds the transition self-loops independently.
#
# Per-state ceilings, set from the actual median regime dwell in the
# v1/regime_transitions stream (14d): normal ~135min, disrupted ~45min,
# suspended ~50min. A single 0.97 cap modeled every regime as ~114min, making
# the filter 2.5x too pessimistic about recovery from disruption (it predicted
# 17% recovered-in-30min against 35% actual). self_loop = exp(ln(0.5) / (median
# dwell minutes / 5)) reproduces each regime's real persistence. Indexed
# (normal, disrupted, suspended).
MAX_SELF_LOOP: tuple[float, float, float] = (0.975, 0.93, 0.93)
MIN_DATA_DAYS = 5


def _cap_self_loops(
    params: HMMParams, max_self: tuple[float, float, float] = MAX_SELF_LOOP
) -> HMMParams:
    """Clamp each transition row's diagonal to its per-state ceiling `max_self[s]`,
    redistributing the freed mass across that row's off-diagonal entries
    (proportionally, or evenly when they're all zero)."""
    rows: list[tuple[float, float, float]] = []
    for s in range(3):
        row = list(params.transition[s])
        cap = max_self[s]
        if row[s] <= cap:
            rows.append((row[0], row[1], row[2]))
            continue
        freed = row[s] - cap
        row[s] = cap
        off = [j for j in range(3) if j != s]
        off_sum = sum(row[j] for j in off)
        for j in off:
            share = row[j] / off_sum if off_sum > 0 else 1.0 / len(off)
            row[j] += freed * share
        rows.append((row[0], row[1], row[2]))
    return HMMParams(
        transition=tuple(rows),
        initial=params.initial,
        emissions=params.emissions,
        emissions_by_bin=params.emissions_by_bin,
    )


def self_loop_diagonal(params: HMMParams) -> tuple[float, float, float]:
    """The transition matrix's self-loop diagonal, per state. Read BEFORE
    _cap_self_loops to see what EM actually wanted: whether the clamp is a rare
    guard or is doing the modelling, and by how much it has to move each state,
    is only visible pre-clamp. This has been hand-patched in twice to answer
    that; it lives here so the answer is reproducible from a plain run."""
    return (
        params.transition[0][0],
        params.transition[1][1],
        params.transition[2][2],
    )


def self_loop_excess(
    params: HMMParams, max_self: tuple[float, float, float] = MAX_SELF_LOOP
) -> tuple[float, float, float]:
    """Per-state amount by which the pre-clamp diagonal exceeds its ceiling.
    Zero when the clamp would not fire; never negative, so a state under its cap
    does not net out against one over it when these are averaged."""
    diag = self_loop_diagonal(params)
    return (
        max(0.0, diag[0] - max_self[0]),
        max(0.0, diag[1] - max_self[1]),
        max(0.0, diag[2] - max_self[2]),
    )


def implied_median_dwell_minutes(self_loop: float) -> float:
    """Median dwell in minutes implied by a geometric self-loop on the 5-min
    grid: the p such that P(dwell > p) = 0.5 under repeated Bernoulli(1 - a_ss)
    exit trials. inf when the self-loop is degenerate (>= 1)."""
    if self_loop >= 1.0:
        return math.inf
    if self_loop <= 0.0:
        return TICK_SECONDS / 60.0
    return (TICK_SECONDS / 60.0) * math.log(0.5) / math.log(self_loop)


def _aligned_window(start: date, end: date) -> tuple[int, int]:
    """Tick-aligned UTC window covering [start, end+1day)."""
    start_dt = datetime(start.year, start.month, start.day, tzinfo=UTC)
    end_dt = datetime(end.year, end.month, end.day, tzinfo=UTC) + timedelta(days=1)
    start_epoch = (int(start_dt.timestamp()) // TICK_SECONDS) * TICK_SECONDS
    end_epoch = (int(end_dt.timestamp()) // TICK_SECONDS) * TICK_SECONDS
    return start_epoch, end_epoch


def load_series_by_route(
    cfg: R2Config,
    start: date,
    end: date,
    *,
    movement_fields: Callable[[str, int], dict[str, Any] | None] | None = None,
    service_fields: Callable[[str, int], dict[str, Any] | None] | None = None,
    severity_floor: int = LEGACY_SEVERITY_FLOOR,
) -> tuple[dict[str, list[Observation]], CorpusStats, dict[str, Any]]:
    """Single R2 pass: fetch alerts, build per-route quiet-filled series.

    Returns (by_route, corpus, input_profile). `corpus` describes the *actual*
    observed ticks before quiet-filling — fill_quiet_ticks pads every series to
    the requested window, so series length can't tell us how much real archive
    we have. The publish gate and the params.json audit block both need the
    unpadded view. `input_profile` is the emission-channel reference profile (over
    the real ticks) that the eval job's drift check compares against.

    `movement_fields(route, tick)` folds the movement channel into each
    observation before the tick tag is dropped: it returns the
    advanced_n/matched_n/has_movement for that cell (or None to leave the channel
    off), reconstructing exactly what the live filter carries in.
    `service_fields(route, tick)` does the same for the service channel
    (service_ratio/has_service). Both are merged onto the same observation, the
    way the live filter folds both the previous tick's movement and service
    metrics into one Observation. Without either the series is alerts-only.

    `severity_floor` decides which alerts count as disruption evidence in the
    built observation (see load.alert_observation). LEGACY_SEVERITY_FLOOR is the
    no-op serving build; a higher floor is a DIAGNOSTIC build whose emission
    distribution the Worker does not reproduce, so it must not be published —
    main() enforces that.
    """
    client = make_client(cfg)
    # Hash the exact key set we fetch — the manifest fingerprint and the training
    # input are then guaranteed to describe the same objects.
    keys = list_alert_keys(client, cfg.bucket, start, end)
    bodies = fetch_objects(client, cfg.bucket, keys)
    input_blake3 = input_manifest_hash(keys)
    # Mask the reconstruction against what the live Worker actually saw active, so an
    # alert that left the feed without a superseding version doesn't train as
    # still-active to its active_period end. Degrades to the raw reconstruction if
    # predictions are unavailable (e.g. pre-stream).
    mask = None
    try:
        from training.eval import load_predictions

        predictions = load_predictions(client, cfg.bucket, start, end)
        mask = presence_mask_from_predictions(predictions)
    except Exception as exc:
        print(f"presence-mask: prediction load failed ({exc}); raw reconstruction")
    all_ticks = build_tick_observations(
        bodies, active_mask=mask, severity_floor=severity_floor
    )
    if not all_ticks:
        return (
            {},
            CorpusStats(
                start_tick=0,
                end_tick=0,
                n_observations=0,
                n_input_versions=len(keys),
                input_blake3=input_blake3,
            ),
            {},
        )

    input_profile = build_input_profile(all_ticks)
    ticks = [t.tick for t in all_ticks]
    corpus = CorpusStats(
        start_tick=min(ticks),
        end_tick=max(ticks),
        n_observations=len(all_ticks),
        n_input_versions=len(keys),
        input_blake3=input_blake3,
    )

    start_tick, end_tick_excl = _aligned_window(start, end)
    last_tick = end_tick_excl - TICK_SECONDS

    by_route: dict[str, list[Observation]] = {}
    seen_routes = {t.route_id for t in all_ticks}
    for route in sorted(seen_routes):
        filled: list[TickObservation] = fill_quiet_ticks(
            all_ticks,
            route,
            start_tick=start_tick,
            end_tick=last_tick,
            severity_floor=severity_floor,
        )
        if movement_fields is None and service_fields is None:
            by_route[route] = [t.observation for t in filled]
        else:
            route_obs: list[Observation] = []
            for t in filled:
                extra: dict[str, Any] = {}
                if movement_fields is not None and (
                    mv := movement_fields(route, t.tick)
                ):
                    extra.update(mv)
                if service_fields is not None and (sv := service_fields(route, t.tick)):
                    extra.update(sv)
                route_obs.append(
                    replace(t.observation, **extra) if extra else t.observation
                )
            by_route[route] = route_obs
    return by_route, corpus, input_profile


def train(
    series_by_route: dict[str, list[Observation]],
    *,
    prior_strength: float = 100.0,
    min_ticks: int = MIN_TICKS_PER_ROUTE,
    advance_priors: dict[str, float] | None = None,
    pre_clamp_diagonals: dict[str | None, tuple[float, float, float]] | None = None,
) -> tuple[HMMParams, dict[str, HMMParams]]:
    """Returns (global_prior, per_route_params). Doesn't touch R2.

    `advance_priors` maps a route to its measured normal-state advance rate; when
    present, that route's prior carries the measured rate instead of the
    hardcoded default, so the movement emission's normal state anchors on the
    line's real cross-tick advance fraction. Routes without a measured baseline
    keep the global prior's default.

    `pre_clamp_diagonals`, when given, is filled with the self-loop diagonal EM
    converged to BEFORE _cap_self_loops touched it — keyed by route, with None
    for the pooled global-prior fit. It is an out-parameter rather than a return
    value so every existing caller is unaffected; the only consumer is the
    clamp-pressure diagnostic, and a route that inherited the prior (too few
    ticks) is absent because it had no fit of its own.
    """
    if not series_by_route:
        raise ValueError("no observations to train on")
    advance_priors = advance_priors or {}

    pooled: list[Observation] = []
    for series in series_by_route.values():
        pooled.extend(series)
    global_prior, _ = fit_em(pooled, BOOTSTRAP_PARAMS, max_iterations=50)
    # fit_em returns canonical state order (normal/disrupted/suspended), so the
    # per-state self-loop caps land on the regimes they were tuned for. Capping
    # before canonicalization applied them to arbitrary EM indices.
    if pre_clamp_diagonals is not None:
        pre_clamp_diagonals[None] = self_loop_diagonal(global_prior)
    global_prior = _cap_self_loops(global_prior)

    out: dict[str, HMMParams] = {}
    for route, series in series_by_route.items():
        rate = advance_priors.get(route)
        prior = (
            _apply_advance_prior(global_prior, rate)
            if rate is not None
            else global_prior
        )
        if len(series) < min_ticks:
            out[route] = prior
            continue
        fitted, _ = fit_em(
            series,
            prior,
            max_iterations=30,
            prior_params=prior,
            prior_strength=prior_strength,
        )
        if pre_clamp_diagonals is not None:
            pre_clamp_diagonals[route] = self_loop_diagonal(fitted)
        out[route] = _cap_self_loops(fitted)
    return global_prior, out


def _apply_advance_prior(params: HMMParams, normal_rate: float) -> HMMParams:
    """Return `params` with the normal state's advance-rate prior set to the
    measured route baseline `normal_rate`, leaving disrupted/suspended alone.
    The flat emissions carry it; per-bin emissions, if ever present, get the
    same override so the prior survives a TOD-conditioned model."""

    def override(em: EmissionParams) -> EmissionParams:
        a = em.advance_rate
        return replace(em, advance_rate=(normal_rate, a[1], a[2]))

    by_bin = (
        tuple(override(e) for e in params.emissions_by_bin)
        if params.emissions_by_bin is not None
        else None
    )
    return replace(
        params, emissions=override(params.emissions), emissions_by_bin=by_bin
    )


def _report_advance_diagnostics(
    series_by_route: dict[str, list[Observation]],
    global_prior: HMMParams,
    per_route: dict[str, HMMParams],
    advance_priors: dict[str, float],
    *,
    min_ticks: int,
) -> None:
    """Print, per fitted route, each canonical state's advance-rate movement
    responsibility (mov_n/mov_k), the rate the data alone implies (k/n), and the
    fitted rate — all read off one E-step at the canonical fitted params, so the
    three agree by construction.

    Reading: a state with large mov_n whose fitted rate tracks k/n is genuinely
    fitted on movement; a state with mov_n≈0 simply carries whatever prior it
    inherited (fitted==prior), and its index<->label slot is then set by
    canonicalize_states on the alert channels, not by movement.

    The pre-fit anchor triple printed per route is the RAW prior EM started from
    (normal seeded by the route's movement baseline, disrupted/suspended by the
    global prior). It is deliberately NOT lined up against the fitted rows: final
    canonicalization may permute a zero-mass state, so a per-canonical-state prior
    column would compare across frames. mov_n is the honest per-state signal.
    """
    gp = global_prior.emissions.advance_rate
    print("=== advance-rate diagnostics ===")
    print(
        "global prior advance_rate "
        f"(normal, disrupted, suspended) = ({gp[0]:.6f}, {gp[1]:.6f}, {gp[2]:.6f})"
    )
    print(
        f"{'route':<5} {'state':<10} {'mov_n':>10} {'mov_k':>10} {'k/n':>7} {'fitted':>8}"
    )
    for route in sorted(per_route):
        series = series_by_route.get(route, [])
        if len(series) < min_ticks:
            continue  # inherited the prior unfitted; no per-route responsibility
        params = per_route[route]
        resp = advance_responsibility(series, params)
        fitted = params.emissions.advance_rate
        route_rate = advance_priors.get(route)
        anchor_normal = route_rate if route_rate is not None else gp[0]
        print(
            f"{route:<5} pre-fit anchor (raw): "
            f"normal={anchor_normal:.4f} disrupted={gp[1]:.4f} suspended={gp[2]:.4f}"
        )
        for s in range(len(STATES)):
            mov_n, mov_k = resp[s]
            kn = f"{mov_k / mov_n:.3f}" if mov_n > 0 else "--"
            print(
                f"{route:<5} {STATES[s]:<10} {mov_n:>10.2f} {mov_k:>10.2f} "
                f"{kn:>7} {fitted[s]:>8.4f}"
            )


def _report_service_diagnostics(
    series_by_route: dict[str, list[Observation]],
    global_prior: HMMParams,
    per_route: dict[str, HMMParams],
    *,
    min_ticks: int,
) -> None:
    """Print, per fitted route, each canonical state's service-ratio Gaussian
    responsibility (svc_w), the mean ratio the data alone implies, and the fitted
    mu/sigma — read off one E-step at the canonical fitted params, so they agree
    by construction. The service-channel analog of _report_advance_diagnostics.

    Reading: a state with large svc_w whose fitted mu tracks the data mean is
    genuinely fitted on service; a state with svc_w≈0 carries the prior
    (fitted==prior). This is the fit-or-drop evidence: if svc_w is ~0 everywhere,
    or the fitted per-state mu/sigma barely separate normal from suspended, the
    Gaussian adds no discrimination the alert + movement channels don't already
    carry.
    """
    gm = global_prior.emissions.service_mu
    gs = global_prior.emissions.service_sigma
    print("=== service-ratio diagnostics ===")
    print(
        "global prior service_mu (normal, disrupted, suspended) = "
        f"({gm[0]:.4f}, {gm[1]:.4f}, {gm[2]:.4f}); sigma = "
        f"({gs[0]:.4f}, {gs[1]:.4f}, {gs[2]:.4f})"
    )
    print(
        f"{'route':<5} {'state':<10} {'svc_w':>10} {'data_mu':>9} "
        f"{'fit_mu':>8} {'fit_sig':>8}"
    )
    for route in sorted(per_route):
        series = series_by_route.get(route, [])
        if len(series) < min_ticks:
            continue
        params = per_route[route]
        resp = service_responsibility(series, params)
        mu = params.emissions.service_mu
        sigma = params.emissions.service_sigma
        for s in range(len(STATES)):
            svc_w, data_mu = resp[s]
            dm = f"{data_mu:.3f}" if svc_w > 0 else "--"
            print(
                f"{route:<5} {STATES[s]:<10} {svc_w:>10.2f} {dm:>9} "
                f"{mu[s]:>8.4f} {sigma[s]:>8.4f}"
            )


# Severity tier a route-tick must reach to count as a severe episode in the
# dwell-fit diagnostic. Deliberately the canonical grading floor, so the fit is
# scored against the same population the grader treats as a real incident.
SEVERE_EPISODE_TIER = CANONICAL_SEVERITY_FLOOR


@dataclass(frozen=True)
class DwellFit:
    """Quality of one fitted geometric self-loop against observed episode
    durations, in ticks. `n` counts only uncensored episodes — a run touching
    either end of the window has an unobserved duration and would bias every
    statistic downward, so it is excluded and counted separately."""

    n: int
    n_censored: int
    empirical_median_ticks: float
    implied_median_minutes: float
    mean_loglik: float
    ks: float


def severe_episode_ticks(
    series: list[Observation], *, tier: int = SEVERE_EPISODE_TIER
) -> tuple[list[int], int]:
    """Durations, in ticks, of the maximal runs where the tick's peak alert
    severity reaches `tier`. Returns (uncensored_durations, n_censored).

    Read off max_severity_tier, which every builder populates from the
    unfiltered alert list, so this segments the SAME severe population whatever
    floor the series' likelihood channels were built under — that is what makes
    the floored and unfloored fits comparable on one yardstick. `series` must be
    the contiguous quiet-filled tick sequence the fit consumed.
    """
    runs: list[int] = []
    censored = 0
    run = 0
    for i, obs in enumerate(series):
        if obs.max_severity_tier >= tier:
            run += 1
            continue
        if run:
            # A run starting at index 0 was already active when the window
            # opened; its onset is unobserved.
            if run == i:
                censored += 1
            else:
                runs.append(run)
            run = 0
    if run:
        censored += 1  # still active at the last tick: no observed recovery
    return runs, censored


def geometric_dwell_fit(
    durations: list[int], self_loop: float, n_censored: int
) -> DwellFit:
    """Score a geometric dwell with parameter `self_loop` against observed
    durations in ticks.

    The self-loop IS a geometric dwell model: P(k ticks) = a^(k-1)(1-a). So the
    honest question is not whether the implied median looks plausible but how
    well that geometric describes the durations — mean per-episode log-likelihood
    (higher is better) and the KS distance between the empirical ECDF and
    F(k) = 1 - a^k (lower is better).

    The dwell is a DISCRETE distribution on the tick grid, so the KS statistic
    is the sup over integer k of |ECDF(k) - F(k)| with both step functions taken
    right-continuous. Taking the sup over the closure instead — reading the ECDF
    on both sides of each jump — would floor the statistic at the largest atom's
    probability (0.2 for a=0.8) even on a perfectly matching sample, which makes
    it useless for telling the two arms apart. Every integer up to the longest
    observed episode is evaluated, not only the observed ones, because the model
    has mass on unobserved values and the sup can sit there.
    """
    n = len(durations)
    implied = implied_median_dwell_minutes(self_loop)
    if n == 0:
        return DwellFit(0, n_censored, math.nan, implied, math.nan, math.nan)
    ordered = sorted(durations)
    mid = n // 2
    median = float(ordered[mid]) if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    a = min(max(self_loop, 1e-12), 1.0 - 1e-12)
    log_a = math.log(a)
    mean_loglik = sum((k - 1) * log_a + math.log1p(-a) for k in ordered) / n
    ks = 0.0
    seen = 0
    idx = 0
    for k in range(1, ordered[-1] + 1):
        while idx < n and ordered[idx] == k:
            seen += 1
            idx += 1
        ks = max(ks, abs(seen / n - (1.0 - a**k)))
    return DwellFit(n, n_censored, median, implied, mean_loglik, ks)


def _report_severity_diagnostics(
    series_by_route: dict[str, list[Observation]],
    global_prior: HMMParams,
    per_route: dict[str, HMMParams],
    pre_clamp: dict[str | None, tuple[float, float, float]],
    *,
    min_ticks: int,
    severity_floor: int,
) -> None:
    """Print the two things a severity-floored refit can actually be judged on:
    clamp pressure on the pre-clamp diagonals, and how well each fitted
    disrupted self-loop describes the tier>=2 episode durations.

    Reading: if the disrupted regime was being defined by chronic ordinary
    Delays, its pre-clamp diagonal is pinned against the ceiling (EM wants the
    long tier-1 tail) and its geometric fits the severe durations badly. Under a
    floor that removes tier-1 from the state definition, clamp pressure on the
    disrupted row should fall and the severe dwell fit should improve. Both
    numbers are printed for every state and route so a null result is as legible
    as a positive one.
    """
    minutes_per_tick = TICK_SECONDS / 60.0
    print("=== severity diagnostics ===")
    print(
        f"severity_floor={severity_floor} "
        f"(serving floor {LEGACY_SEVERITY_FLOOR}; canonical truth floor "
        f"{CANONICAL_SEVERITY_FLOOR})"
    )
    print(f"self-loop caps (normal, disrupted, suspended) = {MAX_SELF_LOOP}")

    fitted_routes = [r for r in sorted(per_route) if r in pre_clamp]
    print(
        f"\n--- clamp pressure: {len(fitted_routes)} routes with their own fit "
        f"({len(per_route) - len(fitted_routes)} inherited the prior at "
        f"min_ticks={min_ticks}) ---"
    )
    gdiag = pre_clamp.get(None)
    if gdiag is not None:
        gexc = tuple(max(0.0, gdiag[s] - MAX_SELF_LOOP[s]) for s in range(len(STATES)))
        print(
            "global prior pre-clamp diag = "
            f"({gdiag[0]:.4f}, {gdiag[1]:.4f}, {gdiag[2]:.4f}) "
            f"excess = ({gexc[0]:.4f}, {gexc[1]:.4f}, {gexc[2]:.4f})"
        )
    print(f"{'state':<10} {'over_cap':>10} {'mean_excess':>12} {'max_excess':>11}")
    for s in range(len(STATES)):
        excesses = [max(0.0, pre_clamp[r][s] - MAX_SELF_LOOP[s]) for r in fitted_routes]
        over = sum(1 for e in excesses if e > 0)
        mean_e = sum(excesses) / len(excesses) if excesses else math.nan
        print(
            f"{STATES[s]:<10} {f'{over}/{len(fitted_routes)}':>10} "
            f"{mean_e:>12.4f} {max(excesses, default=math.nan):>11.4f}"
        )

    print(
        f"\n--- disrupted dwell fit vs tier>={SEVERE_EPISODE_TIER} episodes "
        "(post-clamp self-loop, the one that ships) ---"
    )
    print(
        f"{'route':<6} {'n':>4} {'cens':>5} {'emp_med_m':>10} {'impl_med_m':>11} "
        f"{'mean_ll':>9} {'ks':>7} {'a11':>7} {'a11_pre':>8}"
    )
    pooled: list[int] = []
    pooled_censored = 0
    for route in fitted_routes:
        series = series_by_route.get(route, [])
        durations, censored = severe_episode_ticks(series)
        pooled.extend(durations)
        pooled_censored += censored
        a11 = per_route[route].transition[1][1]
        fit = geometric_dwell_fit(durations, a11, censored)
        print(
            f"{route:<6} {fit.n:>4} {fit.n_censored:>5} "
            f"{fit.empirical_median_ticks * minutes_per_tick:>10.1f} "
            f"{fit.implied_median_minutes:>11.1f} {fit.mean_loglik:>9.4f} "
            f"{fit.ks:>7.4f} {a11:>7.4f} {pre_clamp[route][1]:>8.4f}"
        )
    pooled_fit = geometric_dwell_fit(
        pooled, global_prior.transition[1][1], pooled_censored
    )
    print(
        f"{'POOLED':<6} {pooled_fit.n:>4} {pooled_fit.n_censored:>5} "
        f"{pooled_fit.empirical_median_ticks * minutes_per_tick:>10.1f} "
        f"{pooled_fit.implied_median_minutes:>11.1f} {pooled_fit.mean_loglik:>9.4f} "
        f"{pooled_fit.ks:>7.4f} {global_prior.transition[1][1]:>7.4f} "
        f"{(gdiag[1] if gdiag else math.nan):>8.4f}"
    )
    print(
        "\nPOOLED scores every route's severe episodes against the global "
        "prior's disrupted self-loop — the params a thin route inherits."
    )


# Emission channels dropped from the published params. The Worker reads
# service_mu/service_sigma as OPTIONAL (worker/src/hmm.ts: the service term only
# scores when `em.service_mu !== undefined`), so omitting them here turns the
# service channel off live via the exact back-compat gate pre-service params used
# — no Worker deploy required. This is the 2026-08-31 fit-or-drop verdict: fitting
# the service Gaussian showed the per-state means barely separate (median spread
# 0.15 on a ~1.0 scale, sigma ~0.25) and the fit is severity-INVERTED on 15/28
# routes, because assigned_n supply is statistically independent of the
# alert-defined disruption axis the states are anchored on (journal 2026-08-31
# assigned_n-independence result). A sub-nat channel that points the wrong way
# half the time cannot help a posterior already one-hot at log-odds in the
# hundreds, and the live suspended arm already reads assigned_n on its own axis.
# The channel is still FITTED (see --diagnose-service) so the decision stays
# reproducible; it is simply not shipped for scoring.
_DROPPED_EMISSION_KEYS = ("service_mu", "service_sigma")


def _params_to_json(params: HMMParams) -> dict[str, Any]:
    """Serialize HMMParams to the loose schema the Worker reads.

    Drops the service Gaussian (see _DROPPED_EMISSION_KEYS): the trained value is
    not shipped, so the Worker's optional-param gate leaves the channel unscored.
    """

    def emit(em: EmissionParams) -> dict[str, Any]:
        d = asdict(em)
        for k in _DROPPED_EMISSION_KEYS:
            d.pop(k, None)
        return d

    body: dict[str, Any] = {
        "transition": [list(row) for row in params.transition],
        "initial": list(params.initial),
        "emissions": emit(params.emissions),
    }
    if params.emissions_by_bin is not None:
        body["emissions_by_bin"] = [emit(e) for e in params.emissions_by_bin]
    return body


def write_params(
    client: S3Client,
    bucket: str,
    per_route: dict[str, HMMParams],
    *,
    corpus: CorpusStats,
    n_routes_trained: int,
    dwell_quantiles: dict[str, dict[str, DwellQuantiles]] | None = None,
    dwell_quantiles_by_alert: (
        dict[str, dict[str, dict[str, DwellQuantiles]]] | None
    ) = None,
    dwell_quantiles_by_cause: (
        dict[str, dict[str, dict[str, DwellQuantiles]]] | None
    ) = None,
    dwell_movement: dict[str, dict[str, DwellQuantiles]] | None = None,
    hyperparams: dict[str, Any] | None = None,
    input_profile: dict[str, Any] | None = None,
    movement_baseline: dict[str, Any] | None = None,
    movement_through_stops: dict[str, dict[str, list[str]]] | None = None,
    service_baseline: dict[str, Any] | None = None,
    schedule_rate: dict[str, Any] | None = None,
    trained_at: int | None = None,
) -> str:
    """Write the live params pointer plus an immutable versioned snapshot.

    The Worker reads state/params.json; the state/params/v<epoch>.json copies
    give us a per-run rollback trail. Returns the versioned key.
    """
    trained_at = trained_at or int(datetime.now(UTC).timestamp())
    routes_doc = {r: _params_to_json(p) for r, p in per_route.items()}
    if dwell_quantiles:
        # Merge per-route empirical dwell into the same per-route subdoc — the
        # Worker reads it as an optional sibling of `emissions`/`transition`.
        for r, by_state in dwell_quantiles.items():
            if r in routes_doc:
                routes_doc[r]["dwell_quantiles"] = by_state
    if dwell_quantiles_by_alert:
        # Cause-segmented dwell, layered on top of the (route, state) aggregate.
        # The Worker prefers (route, state, alert_type) and falls back to the
        # aggregate above when a cause cell is absent.
        for r, by_state_alert in dwell_quantiles_by_alert.items():
            if r in routes_doc:
                routes_doc[r]["dwell_quantiles_by_alert"] = by_state_alert
    if dwell_quantiles_by_cause:
        # Cause-CATEGORY dwell for the episode-recovery grader (Episode.cause is a
        # coarse category, not a raw alert_type). The Worker ignores this key
        # (zod strips it); scorecard.dwell_lookup_from_params reads it so the
        # grade stops silently falling back to the (route, state) aggregate.
        for r, by_state_cause in dwell_quantiles_by_cause.items():
            if r in routes_doc:
                routes_doc[r]["dwell_quantiles_by_cause"] = by_state_cause
    doc = {
        "schema_version": SCHEMA_VERSION,
        "trained_at": trained_at,
        "provenance": code_provenance(),
        "hyperparams": hyperparams or {},
        "input_profile": input_profile or {},
        "training_corpus": {
            "start_tick": corpus.start_tick,
            "end_tick": corpus.end_tick,
            "n_routes_trained": n_routes_trained,
            "n_observations": corpus.n_observations,
            "n_input_versions": corpus.n_input_versions,
            "input_blake3": corpus.input_blake3,
        },
        "routes": routes_doc,
    }
    # Per-(route, direction, tod_bin) advance-rate baseline the Worker needs live to
    # gate and score the movement channel. Top-level (not per-route) so the assigned_n
    # service baseline can sit beside it under the same delivery.
    if movement_baseline:
        doc["movement_baseline"] = movement_baseline
    # The stops that baseline was fitted on: from_stops with a scheduled
    # predecessor and successor. The Worker counts a cross-tick advance or stall
    # only at these, so a terminal layover is not evidence of a stall. Travels in
    # the same object as the baseline deliberately — scoring against a stop set
    # the baseline was not fitted with judges layovers against a through-stop
    # normal. Absent means unfiltered on both sides.
    if movement_through_stops:
        doc["movement_through_stops"] = movement_through_stops
    # Per-(route, tod_bin) assigned_n baseline the Worker divides live assigned_n
    # by to form the service ratio the emission scores. Top-level beside
    # movement_baseline.
    if service_baseline:
        doc["service_baseline"] = service_baseline
    # Per-(route, schedule_bin) scheduled-presence rate the Worker uses to split a
    # no-service reading into suspended vs not_scheduled. Top-level beside the
    # baselines.
    if schedule_rate:
        doc["schedule_rate"] = schedule_rate
    # Movement-primary dwell (C2). Route scope only -- segment scope is
    # training.segment_dwell's own object at state/segment_dwell.json. Top-
    # level like the baselines above, not nested per-route: the Worker's
    # movementDwellFor and the scorecard's movement_dwell_lookup_from_params
    # both read it that way.
    if dwell_movement:
        doc["dwell_movement"] = dwell_movement
    body = json.dumps(doc).encode()
    versioned_key = f"{VERSIONED_PARAMS_PREFIX}v{trained_at}.json"
    for key in (PARAMS_KEY, versioned_key):
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            CacheControl="public, max-age=300, s-maxage=900",
        )
    return versioned_key


SEGMENT_PARAMS_KEY = "state/segment_params.json"
VERSIONED_SEGMENT_PREFIX = "state/segment_params/"


def _static_topology() -> tuple[
    dict[SegmentKey, list[tuple[str, int]]] | None,
    dict[tuple[str, str], list[RoutePattern]] | None,
    str,
]:
    """The static successor skeleton and stopping patterns for this run, or
    (None, None, "observed") with the reason printed.

    Fetched once and shared: the advance baseline, the through-stop set it is
    fitted against and the published segment topology all have to describe the
    same timetable. A Worker scoring against a stop set the baseline was not
    fitted with would judge layovers against a through-stop normal.
    """
    try:
        successors, patterns = load_topology()
        return successors, patterns, "gtfs_static"
    except Exception as exc:
        print(
            f"gtfs static topology unavailable, using observed adjacency ({exc})",
            file=sys.stderr,
        )
        return None, None, "observed"


def _stop_filter(
    through: frozenset[tuple[str, str, str]] | None,
) -> StopFilter | None:
    """Admit only from_stops the timetable puts mid-chain. None passes
    everything, which is what an unavailable static feed leaves us with.

    One definition for every fit in this run: a rate fitted over a different
    stop set than the one published in movement_through_stops would have the
    Worker judging layovers against a through-stop normal.
    """
    if through is None:
        return None
    return lambda route, direction, frm: (route, direction, frm) in through


def write_segment_params(
    cfg: R2Config,
    client: S3Client,
    bucket: str,
    start_date: date,
    end_date: date,
    trained_at: int,
    static_successors: dict[SegmentKey, list[tuple[str, int]]] | None,
    static_patterns: dict[tuple[str, str], list[RoutePattern]] | None,
    topology_source: str,
    through: frozenset[tuple[str, str, str]] | None,
) -> int:
    """Write the segment baseline + adjacency as their OWN R2 object (not
    folded into params.json, which the Worker parses on the hot per-tick
    path). The Worker reads this at step 8b, off the publish path, to score
    per-segment movement and roll it up to station service flow.

    Topology (adjacency) comes from the static GTFS timetable the caller fetched
    for the whole run (`_static_topology`), when that fetch succeeded: a segment
    exists because the schedule says so, keyed
    'route|dir|from'. A from_stop with more than one static successor
    (branch/express) keeps its full successor list, not just the modal
    winner. canonical_adjacency (observed cross-tick transitions) is now only
    the fallback for when the GTFS fetch itself fails, plus a `share`/`n`
    reliability annotation riding along on whichever entries the vehicle
    archive also observed that window — annotation only, it no longer decides
    whether an entry is published.

    cells (the pooled advance-rate baseline) still needs actual cross-tick
    vehicle data, so it's scoped to baseline.items() regardless of topology
    source; the whole object is skipped when that's empty (an archive hiccup
    leaves nothing to pair the topology with). `through` restricts those leaves
    to mid-chain from_stops, the same set params.json publishes — a leaf fitted
    over layovers would hand the Worker a normal it never scores against.

    Each cell also carries `lam`: its expected matched traversals per tick by
    time bin (load_r2.build_segment_throughput), the denominator that lets the
    Worker read an empty window as evidence instead of abstaining. Fitted over
    the same bodies and the same `through` filter as p0, so the rate and the
    normal it complements describe one stop set. Bins where the cell runs
    nothing are dropped from `lam` and read as zero against the bin set in
    `throughput.ticks` — see load_r2.throughput_to_json.

    Fail-soft: a vehicle-archive hiccup skips the object, leaving the last good
    one; the station-flow surface just goes stale, never blocks the params run.
    """
    try:
        bodies = fetch_vehicle_metrics(
            cfg, start_date=start_date, end_date=end_date, client=client
        )
        stop_filter = _stop_filter(through)
        baseline = build_segment_baseline(bodies, counts_from_stop=stop_filter)
        observed_adjacency = canonical_adjacency(bodies)
        rates, exposure = build_segment_throughput(bodies, counts_from_stop=stop_filter)
        lam = throughput_to_json(rates)

        cells: dict[str, dict[str, Any]] = {}
        for key, cell in baseline.items():
            entry: dict[str, Any] = {"p0": round(cell.p0, 6), "n": cell.n}
            cell_lam = lam.get("|".join(key))
            if cell_lam is not None:
                entry["lam"] = cell_lam
            cells["|".join(key)] = entry
        if not cells:
            print("segment params skipped (no through-segments)", file=sys.stderr)
            return 0

        adj_doc: dict[str, dict[str, Any]] = {}
        if static_successors is not None:
            for key, succs in static_successors.items():
                if not succs:
                    continue
                to_stop, _n_trips = dominant_successor(succs)
                adj_entry: dict[str, Any] = {
                    "to": to_stop,
                    "source": "gtfs_static",
                    "successors": [{"to": t, "n_trips": n} for t, n in succs],
                }
                obs = observed_adjacency.get(key)
                if obs is not None:
                    adj_entry["share"] = round(obs.share, 4)
                    adj_entry["n"] = obs.n
                adj_doc["|".join(key)] = adj_entry
        else:
            for key, adj in observed_adjacency.items():
                adj_doc["|".join(key)] = {
                    "to": adj.to_stop,
                    "source": "observed",
                    "share": round(adj.share, 4),
                    "n": adj.n,
                }

        doc = {
            "schema_version": SCHEMA_VERSION,
            "trained_at": trained_at,
            # Which code produced this doc, matching params.json/eval.json — the
            # off-Worker consumers (viz) read the topology and its ordering, so
            # they can name the tree that built it. See training/provenance.py.
            "provenance": code_provenance(),
            "min_share": MIN_SHARE,
            "topology_source": topology_source,
            "cells": cells,
            "adjacency": adj_doc,
            # How the per-cell `lam` rates were fitted: the bin function, the
            # exposure floor, and the observed ticks per published bin. The bin
            # set is exactly these keys — a bin missing here was never fitted
            # (the Worker abstains), a bin here but missing from a cell's `lam`
            # was fitted at zero (nothing scheduled, silence is normal).
            "throughput": {
                "bin": "schedule_bin",
                "min_ticks": MIN_THROUGHPUT_TICKS,
                "ticks": dict(sorted(exposure.items())),
            },
            # Canonical per-(route, direction) stop order: the actual scheduled
            # trip patterns, most-run first. A consumer reads line order off
            # these instead of relinearizing the single-successor adjacency
            # graph, which mangles express/local and branch splits. Empty on the
            # observed-adjacency fallback (no static feed to read patterns from).
            "route_stops": (
                patterns_to_json(static_patterns) if static_patterns is not None else {}
            ),
        }
        body = json.dumps(doc).encode()
        versioned = f"{VERSIONED_SEGMENT_PREFIX}v{trained_at}.json"
        for key in (SEGMENT_PARAMS_KEY, versioned):
            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
                CacheControl="no-store",
            )
        return len(cells)
    except Exception as exc:
        print(f"segment params skipped ({exc})", file=sys.stderr)
        return 0


SERVICE_BASELINE_KEY = "state/service_baseline.json"
VERSIONED_SERVICE_PREFIX = "state/service_baseline/"

# Trailing window for the published service-baseline sidecar, deliberately wider
# than a typical HMM retrain window. Sized so a weekend-hourly (route,
# schedule_bin) cell clears the sidecar's SERVICE_MIN_NIGHTS gate with margin: 35
# days is 5 weekends = 10 Sat/Sun nights per cell against a floor of 8, so a
# normally-thin weekend late-night cell publishes a trusted median instead of
# abstaining. Still inside archive-retention headroom. Shared with
# backfill_service_baseline so a retrain publish and a standalone backfill write
# the same sidecar.
SERVICE_SIDECAR_WINDOW_DAYS = 35


def write_service_baseline(
    client: S3Client,
    bucket: str,
    hourly: dict[str, Any],
    generated_at: int,
    params_trained_at: int | None = None,
    quantiles: dict[str, Any] | None = None,
) -> int:
    """Write the per-(route, schedule_bin) assigned_n baseline -- the supply
    axis's denominator -- as its OWN versioned R2 object, decoupled from
    params.json. Refreshing it never moves the HMM artifact's trained_at, so it
    cannot reseed the Worker's filter or split the grader's params-version
    window. Versioned by its OWN `generated_at` (not the model's trained_at) so a
    later refresh can't overwrite a prior immutable snapshot; `params_trained_at`
    records which frozen model it was computed to accompany. `quantiles` is the
    sibling per-(route, schedule_bin) p10/p90 spread (training.load_r2.
    compute_service_quantiles / service_quantiles_to_json), same keying as
    `hourly`; omitted from the doc when absent (None or empty), so a caller with
    no quantile data round-trips a sidecar exactly like today's. Mirrors
    write_segment_params: live pointer + immutable versioned snapshot, skipped
    when empty. Returns the route count."""
    if not hourly:
        return 0
    doc: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "baseline": hourly,
    }
    if params_trained_at is not None:
        doc["params_trained_at"] = params_trained_at
    if quantiles:
        doc["quantiles"] = quantiles
    body = json.dumps(doc).encode()
    versioned = f"{VERSIONED_SERVICE_PREFIX}v{generated_at}.json"
    for key in (SERVICE_BASELINE_KEY, versioned):
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            CacheControl="no-store",
        )
    return len(hourly)


SEGMENT_DWELL_KEY = "state/segment_dwell.json"
VERSIONED_SEGMENT_DWELL_PREFIX = "state/segment_dwell/"


def write_segment_dwell(
    client: S3Client,
    bucket: str,
    start_date: date,
    end_date: date,
    trained_at: int,
    through: frozenset[tuple[str, str, str]] | None,
) -> tuple[int, SegmentDwellStats]:
    """Write the per-segment dwell curves as their OWN R2 object (not folded
    into segment_params.json), hierarchically pooled leaf -> route -> system
    (training.segment_dwell) from the segment-scope movement regimes over this
    run's training window.

    Episodes are reconstructed from archive/vehicles through the identical
    regime clock (training.regime) the Worker runs online, counting only
    `through` from_stops so the curves and the published baseline describe the
    same segments. The Worker's own committed v1/movement_transitions stream is
    deliberately not read here — see _movement_dwell.

    Fail-soft like write_segment_params: an archive hiccup or an empty
    stream just skips the object, leaving the last good one, and never
    blocks the params publish. Returns (n_cells, stats) — stats is all-zero
    on skip.
    """
    empty_stats = SegmentDwellStats(
        n_cells_own=0, n_cells_route=0, n_cells_system=0, n_cells_skipped=0
    )
    try:
        from training.movement_backfill import reconstruct_movement_transitions

        transitions = reconstruct_movement_transitions(
            client=client,
            bucket=bucket,
            start_date=start_date,
            end_date=end_date,
            scope="segment",
            counts_from_stop=_stop_filter(through),
        )
        # Same censoring boundary as the route-level dwell fit: "now", clamped
        # to the requested window.
        _, end_epoch = _aligned_window(start_date, end_date)
        window_end = min(int(datetime.now(UTC).timestamp()), end_epoch)
        cells, stats = build_segment_dwell(transitions, window_end=window_end)
        if not cells:
            print(
                "segment dwell skipped (no segment-scope transitions)",
                file=sys.stderr,
            )
            return 0, stats
        doc = {
            "schema_version": SCHEMA_VERSION,
            "trained_at": trained_at,
            "cells": cells,
        }
        body = json.dumps(doc).encode()
        versioned = f"{VERSIONED_SEGMENT_DWELL_PREFIX}v{trained_at}.json"
        for key in (SEGMENT_DWELL_KEY, versioned):
            client.put_object(
                Bucket=bucket,
                Key=key,
                Body=body,
                ContentType="application/json",
                CacheControl="no-store",
            )
        return len(cells), stats
    except Exception as exc:
        print(f"segment dwell skipped ({exc})", file=sys.stderr)
        return 0, empty_stats


@dataclass(frozen=True)
class MovementInputs:
    """Everything the movement channel needs from one vehicle-archive fetch.

    `baseline_json`/`n_cells`/`route_rates` are the published baseline and the
    per-route normal-state prior seeds (as before). `baseline_cells` is the raw
    (route, direction, tod_bin) key set of the fitted baseline — the presence
    gate the live filter keys `has_movement` on. `movement_by_tick` is the
    through-filtered per-(route, tick) counts the live filter folds into an
    observation; the emission is fitted on the same filtered population as the
    baseline and the live classifier, so training scores what inference scores.
    """

    baseline_json: dict[str, Any]
    n_cells: int
    route_rates: dict[str, float]
    baseline_cells: set[tuple[str, str, int]]
    movement_by_tick: dict[tuple[str, int], dict[str, int]]


def _movement_baseline(
    cfg: R2Config,
    client: S3Client,
    start_date: date,
    end_date: date,
    through: frozenset[tuple[str, str, str]] | None,
) -> MovementInputs:
    """Advance-rate baseline + raw movement counts over the training window, from
    one vehicle-archive fetch:

    - the per-(route, direction, tod) baseline, serialized for params.json
      delivery to the Worker's movement posterior, plus its raw cell key set;
    - the per-route normal advance rate that seeds each route's EM prior;
    - the through-filtered per-(route, tick) counts, both directions summed, that
      the HMM movement emission is fitted on (same filter as the baseline).

    `through` restricts the movement counts — the baseline AND the per-tick counts
    the emission is fitted on — to trips whose from_stop has a scheduled
    predecessor and successor. This is the SAME filter the live Worker applies
    before it classifies (worker/src/vehicles.ts deriveRouteMovementMetric, since
    2026-08-12), so the baseline, the training emission, and the live classifier
    all score the identical population. Terminal/chain-endpoint layovers stall
    ~89% of the time (measured 2026-08-05..08-11: 83% of all stall mass vs 11.6%
    mid-line); counting them blends two physically different populations and makes
    every route read chronically sub-normal. None means the static feed was
    unavailable — then nothing is filtered, on any side.

    Uses the explicit training window — fetch_vehicle_metrics defaults to
    yesterday..today, too narrow for a stable prior. Fail-soft: any
    vehicle-archive error returns empty inputs so a movement hiccup never blocks
    the params publish and the emission channel simply stays off."""
    counts_from_stop: StopFilter | None = (
        None
        if through is None
        else lambda route, direction, frm: (route, direction, frm) in through
    )
    try:
        bodies = fetch_vehicle_metrics(
            cfg, start_date=start_date, end_date=end_date, client=client
        )
        series = build_movement_series_by_direction(
            bodies, counts_from_stop=counts_from_stop
        )
        baseline = compute_advance_baseline(series)
        route_rates = compute_advance_baseline_by_route(series)
        return MovementInputs(
            baseline_json=advance_baseline_to_json(baseline),
            n_cells=len(baseline),
            route_rates=route_rates,
            baseline_cells=set(baseline.keys()),
            movement_by_tick=build_movement_series(
                bodies, counts_from_stop=counts_from_stop
            ),
        )
    except Exception as exc:
        print(f"movement baseline skipped ({exc})", file=sys.stderr)
        return MovementInputs({}, 0, {}, set(), {})


@dataclass(frozen=True)
class ServiceInputs:
    """Everything the service channel needs from one trip-updates fetch.

    The `*_json` / `n_*` fields are the published baselines/quantiles/schedule
    rate (as before). `baseline_by_cell` is the raw (route, tod_bin) -> median
    assigned_n — the live-ratio denominator, kept in its dict form so the HMM
    service emission is fitted against the exact denominator inference divides
    by. `service_by_tick` is the per-(route, tick) assigned_n the live filter
    folds into an observation (previous tick, option B lag). Parity with
    MovementInputs so the two channels wire the same way.
    """

    baseline_json: dict[str, Any]
    n_cells: int
    schedule_json: dict[str, Any]
    n_schedule: int
    hourly_json: dict[str, Any]
    n_hourly: int
    hourly_quantiles_json: dict[str, Any]
    n_hourly_quantiles: int
    baseline_by_cell: dict[tuple[str, int], float]
    service_by_tick: dict[tuple[str, int], int]


def _service_baseline(
    cfg: R2Config,
    client: S3Client,
    start_date: date,
    end_date: date,
) -> ServiceInputs:
    """Per-(route, tod) assigned_n baseline, per-(route, schedule_bin) assigned_n
    baseline, per-(route, schedule_bin) assigned_n p10/p90 quantiles, AND
    per-(route, schedule_bin) scheduled-presence rate over the training window,
    from one trip-updates fetch, serialized for params.json / the sidecar.

    The tod_bin baseline is the service emission's live-ratio denominator; the
    schedule_bin baseline is the published service-degradation axis's denominator
    (finer, so the quiet edge of a wide tod block doesn't read as a supply cut —
    see degradation_label's BIN_FN note). The quantiles are computed off the
    SAME schedule_bin series as the schedule_bin baseline (same fetch, same
    bucketing, same min_samples AND min_nights gate), so a cell has a quantile
    iff it has a baseline. The published schedule_bin axis additionally gates on
    SERVICE_MIN_NIGHTS distinct nights (thin weekend-hourly cells abstain rather
    than publish an off-distribution median); the tod_bin emission denominator
    keeps the default night gate so its frozen operating point is untouched. The
    schedule rate splits a no-service reading into suspended vs
    not_scheduled. Fail-soft: a trip-updates archive error returns empty
    sidecars (all optional and back-compat)."""
    try:
        bodies = fetch_trip_update_metrics(
            cfg, start_date=start_date, end_date=end_date, client=client
        )
        series = build_service_series(bodies)
        baseline = compute_baseline(series)
        hourly = compute_baseline(
            series, bin_fn=schedule_bin, min_nights=SERVICE_MIN_NIGHTS
        )
        hourly_quantiles = compute_service_quantiles(
            series, bin_fn=schedule_bin, min_nights=SERVICE_MIN_NIGHTS
        )
        rate = compute_schedule_rate(bodies)
        return ServiceInputs(
            baseline_json=service_baseline_to_json(baseline),
            n_cells=len(baseline),
            schedule_json=schedule_rate_to_json(rate),
            n_schedule=len(rate),
            hourly_json=service_baseline_to_json(hourly),
            n_hourly=len(hourly),
            hourly_quantiles_json=service_quantiles_to_json(hourly_quantiles),
            n_hourly_quantiles=len(hourly_quantiles),
            baseline_by_cell=baseline,
            service_by_tick=series,
        )
    except Exception as exc:
        print(f"service baseline skipped ({exc})", file=sys.stderr)
        return ServiceInputs({}, 0, {}, 0, {}, 0, {}, 0, {}, {})


def _movement_dwell(
    cfg: R2Config,
    client: S3Client,
    start_date: date,
    end_date: date,
    window_end: int,
) -> tuple[dict[str, dict[str, DwellQuantiles]], dict[str, Any]]:
    """Partially-pooled {route: {state: DwellQuantiles}} off the movement
    regime-transition stream -- the C2 `dwell_movement` params block, route
    scope only (segment scope is training.segment_dwell's own object).

    Every state runs through pooled_dwell_cells, not compute_dwell_quantiles:
    the movement classifier is conservative enough (17 route episodes across 6
    routes over its first usable week) that the MIN_SAMPLES_FOR_EMPIRICAL floor
    would leave almost every cell empty. Partial pooling gives every route with
    ANY observation a fitted cell -- including a route that has simply held one
    state for the whole window and so has zero completed episodes there --
    shrunk toward the population centre until its own evidence outweighs the
    prior.

    Episodes are reconstructed from the published_condition ticks the prediction
    stream carries, through the identical regime clock (training.regime) the
    Worker runs online. The Worker now also commits its own
    v1/movement_transitions stream, and this deliberately does not read it: that
    stream only starts at the deploy that introduced it, so over a two-week
    window it holds 13 route transitions against the replay's 124, and letting
    it win on presence alone would shrink the fit to whatever tail of the window
    it happens to cover. Over the overlap the two agree on 10 of those 13 (same
    route, same target state, onset within a tick), so the switch is a change of
    source rather than of signal -- but it is a model change, and it gets made
    once the stream spans a window and the two fits have been graded against
    each other.

    Open regimes come from the same tick replay: a route sitting in one state
    for the whole window contributes only a censored observation, and dropping
    it would bias every cell short.

    Fail-soft like every other optional params sidecar in this file: an
    archive error yields an empty block rather than blocking the publish.
    """
    empty_stats: dict[str, Any] = {
        "source": "unavailable",
        "n_transitions": 0,
        "n_own": 0,
        "n_pooled": 0,
        "n_atom": 0,
    }
    try:
        from training.eval import STATES
        from training.movement_backfill import (
            movement_open_regimes,
            reconstruct_movement_transitions,
        )

        transitions = reconstruct_movement_transitions(
            client=client,
            bucket=cfg.bucket,
            start_date=start_date,
            end_date=end_date,
            scope="route",
        )
        open_regimes = (
            movement_open_regimes(
                client=client,
                bucket=cfg.bucket,
                start_date=start_date,
                end_date=end_date,
                scope="route",
            )
            or None
        )
        out: dict[str, dict[str, DwellQuantiles]] = {}
        n_own = 0
        n_pooled = 0
        n_atom = 0
        for state in STATES:
            # Every movement state is a mixture candidate; pooled_dwell_cells
            # only publishes an atom where the population actually has one, so
            # `normal` (hours-long dwells) falls through to the continuous fit
            # untouched while the disrupted states pick up their one-tick spike.
            cells = pooled_dwell_cells(
                transitions,
                state=state,
                window_end=window_end,
                open_regimes=open_regimes,
                atom_sec=TICK_SECONDS,
            )
            for route, cell in cells.items():
                out.setdefault(route, {})[state] = cell
                if cell["n"] >= MIN_VOTER_EVENTS:
                    n_own += 1
                else:
                    n_pooled += 1
                if "atom_p" in cell:
                    n_atom += 1
        return out, {
            "source": "tick_replay",
            "n_transitions": len(transitions),
            "n_own": n_own,
            "n_pooled": n_pooled,
            "n_atom": n_atom,
        }
    except Exception as exc:
        print(f"movement dwell skipped ({exc})", file=sys.stderr)
        return {}, empty_stats


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-route EM trainer")
    parser.add_argument(
        "--days",
        type=int,
        default=14,
        help="trailing-window size when --start is unset",
    )
    parser.add_argument(
        "--start", help="window start date YYYY-MM-DD (overrides --days)"
    )
    parser.add_argument("--end", help="window end date YYYY-MM-DD (default: today UTC)")
    parser.add_argument(
        "--routes", help="comma-separated route whitelist (default: all observed)"
    )
    parser.add_argument(
        "--min-ticks",
        type=int,
        default=MIN_TICKS_PER_ROUTE,
        help="routes with fewer observations inherit the global prior",
    )
    parser.add_argument(
        "--prior-strength",
        type=float,
        default=100.0,
        help="pseudo-counts strength for per-route prior anchor (in tick units)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print learned params instead of writing to R2",
    )
    parser.add_argument(
        "--allow-empty-baseline",
        action="store_true",
        help="publish even if the movement advance-baseline is empty (0 cells); "
        "the movement-primary condition stays off. Default: refuse to publish.",
    )
    parser.add_argument(
        "--diagnose-advance",
        action="store_true",
        help="after fitting, report per-route per-state advance-rate movement "
        "responsibility (mov_n/mov_k), the fitted rate, and the prior it blends "
        "against, then exit without writing. Answers whether a state's serialized "
        "advance_rate reflects observed movement mass or just its prior.",
    )
    parser.add_argument(
        "--diagnose-service",
        action="store_true",
        help="after fitting, report per-route per-state service-ratio "
        "responsibility (svc_w), the data-implied mean, and the fitted mu/sigma, "
        "then exit without writing. Answers whether the service Gaussian is fitted "
        "on observed supply or just carries its prior — the fit-or-drop evidence.",
    )
    parser.add_argument(
        "--severity-floor",
        type=int,
        default=LEGACY_SEVERITY_FLOOR,
        help="severity tier a disruptive alert must reach to count as disruption "
        f"evidence in the training observation (default {LEGACY_SEVERITY_FLOOR}, "
        "the no-op serving build). Above the serving floor the emission "
        "distribution no longer matches what the Worker produces, so the run is "
        "diagnostic only and refuses to write params.",
    )
    parser.add_argument(
        "--diagnose-severity",
        action="store_true",
        help="after fitting, report pre-clamp self-loop diagonals per state and "
        f"how well each fitted disrupted self-loop describes the tier>="
        f"{CANONICAL_SEVERITY_FLOOR} episode durations, then exit without "
        "writing. Pair with --severity-floor to compare the severe-only build "
        "against the serving build on one window.",
    )
    args = parser.parse_args(argv)
    if args.severity_floor != LEGACY_SEVERITY_FLOOR and not (
        args.dry_run
        or args.diagnose_severity
        or args.diagnose_advance
        or args.diagnose_service
    ):
        print(
            f"ERROR: --severity-floor {args.severity_floor} differs from the "
            f"serving floor {LEGACY_SEVERITY_FLOOR}, so these params would be "
            "fitted on an alert-count distribution the Worker never produces "
            "(worker/src/derive.ts counts every non-planned alert). Publishing "
            "them would ship a train/serve emission mismatch. Re-run with "
            "--diagnose-severity or --dry-run, or wire the identical floor into "
            "the Worker first.",
            file=sys.stderr,
        )
        return 1

    cfg = load_config()
    end_date = date.fromisoformat(args.end) if args.end else datetime.now(UTC).date()
    start_date = (
        date.fromisoformat(args.start)
        if args.start
        else end_date - timedelta(days=args.days - 1)
    )
    client = make_client(cfg)
    # One static-timetable fetch for the whole run: it decides which stops the
    # advance baseline is fitted on, which set ships to the Worker, and the
    # published segment topology.
    static_successors, static_patterns, topology_source = _static_topology()
    through = None if static_successors is None else through_stops(static_successors)
    # Movement advance-rate baseline + raw per-tick counts over the window. The
    # baseline ships to the Worker and seeds each route's normal-state prior; the
    # raw counts feed the HMM movement emission below (fail-soft — see helper).
    mv = _movement_baseline(cfg, client, start_date, end_date, through)
    movement_baseline = mv.baseline_json
    n_baseline_cells = mv.n_cells
    route_advance_rates = mv.route_rates
    # Fold the movement channel into each observation the way the live filter
    # does — previous tick's counts, same match/baseline gates — so the emission
    # is fitted on the samples inference will score it against, not left inert.
    # None when no counts were fetched: the series stays alerts-only.
    movement_fields: Callable[[str, int], dict[str, Any] | None] | None = None
    if mv.movement_by_tick:

        def _movement_fields(route: str, tick: int) -> dict[str, Any] | None:
            return movement_observation_fields(
                mv.movement_by_tick, mv.baseline_cells, route, tick
            )

        movement_fields = _movement_fields

    # Assigned_n service baseline + raw per-tick counts over the window. The
    # tod_bin baseline ships to the Worker and is the service emission's live-ratio
    # denominator; the raw counts feed the HMM service emission below the same way
    # movement does (fail-soft — see helper).
    svc = _service_baseline(cfg, client, start_date, end_date)
    service_baseline = svc.baseline_json
    n_service_cells = svc.n_cells
    schedule_rate = svc.schedule_json
    n_schedule_cells = svc.n_schedule
    # The published schedule_bin sidecar is computed over a DEDICATED window,
    # decoupled from (and never narrower than) the HMM training window. A short
    # retrain (--days 14) offers only ~4 weekend nights per thin late-night cell,
    # below the sidecar's SERVICE_MIN_NIGHTS gate, so publishing it over the
    # training window would regress every weekend cell to abstention and stomp
    # the wide-window sidecar backfill_service_baseline maintains. Over
    # >= SERVICE_SIDECAR_WINDOW_DAYS a thin cell clears the gate, so the DEFAULT
    # publish refreshes the sidecar correctly rather than overwriting it. The
    # params-embedded tod_bin emission baseline stays on the training window
    # (svc, above) so the frozen emission operating point is untouched.
    sidecar_start = min(
        start_date, end_date - timedelta(days=SERVICE_SIDECAR_WINDOW_DAYS - 1)
    )
    sidecar = (
        _service_baseline(cfg, client, sidecar_start, end_date)
        if sidecar_start < start_date
        else svc
    )
    service_baseline_hourly = sidecar.hourly_json
    n_service_hourly_cells = sidecar.n_hourly
    service_baseline_hourly_quantiles = sidecar.hourly_quantiles_json
    n_service_hourly_quantile_cells = sidecar.n_hourly_quantiles
    # Fold the service channel into each observation the way the live filter WOULD
    # — previous tick's assigned_n over the (route, tod_bin) baseline, same lag and
    # gate. This runs ONLY under --diagnose-service: the 2026-08-31 fit-or-drop
    # verdict is DROP (see _DROPPED_EMISSION_KEYS), so the production fit stays
    # service-free to match the service-free live Worker — folding it into the
    # normal fit would refit the alert/movement params under a service-scored
    # E-step the Worker never runs (the very train/serve skew the movement wiring
    # exists to avoid). The diagnostic path still reconstructs it so the decision
    # stays reproducible via service_responsibility.
    service_fields: Callable[[str, int], dict[str, Any] | None] | None = None
    if args.diagnose_service and svc.service_by_tick and svc.baseline_by_cell:
        # The set of ticks that produced a service snapshot — used to resolve the
        # single carried doc the live filter would have read, so a route absent
        # from the most-recent doc abstains rather than reaching to an older one.
        snapshot_ticks = frozenset(t for _r, t in svc.service_by_tick)

        def _service_fields(route: str, tick: int) -> dict[str, Any] | None:
            return service_observation_fields(
                svc.service_by_tick,
                svc.baseline_by_cell,
                snapshot_ticks,
                route,
                tick,
            )

        service_fields = _service_fields

    series, corpus, input_profile = load_series_by_route(
        cfg,
        start_date,
        end_date,
        movement_fields=movement_fields,
        service_fields=service_fields,
        severity_floor=args.severity_floor,
    )
    if not series:
        print("no observations in archive — skipping training", file=sys.stderr)
        return 1

    if args.routes:
        whitelist = {r.strip() for r in args.routes.split(",") if r.strip()}
        series = {r: s for r, s in series.items() if r in whitelist}
        if not series:
            print(
                f"none of --routes {sorted(whitelist)} present in archive",
                file=sys.stderr,
            )
            return 1
    if n_baseline_cells == 0 and not (
        args.dry_run
        or args.allow_empty_baseline
        or args.diagnose_advance
        or args.diagnose_severity
    ):
        print(
            "ERROR: movement advance-baseline is EMPTY (0 cells) -- refusing to "
            "publish params that would silently disable the movement-primary "
            "condition (every route reads 'unknown'). Check vehicle-archive "
            "by_direction coverage over the training window, or pass "
            "--allow-empty-baseline to publish an alerts-only params set.",
            file=sys.stderr,
        )
        return 1
    if n_baseline_cells == 0:
        print(
            "WARNING: movement advance-baseline is EMPTY (0 cells) -- the "
            "movement-primary condition will publish 'unknown' for every route.",
            file=sys.stderr,
        )
    # The sink is only requested by the clamp-pressure diagnostic, so a normal
    # publish run calls train() exactly as it always did.
    pre_clamp: dict[str | None, tuple[float, float, float]] = {}
    train_kwargs: dict[str, Any] = {}
    if args.diagnose_severity:
        train_kwargs["pre_clamp_diagonals"] = pre_clamp
    global_prior, per_route = train(
        series,
        prior_strength=args.prior_strength,
        min_ticks=args.min_ticks,
        advance_priors=route_advance_rates,
        **train_kwargs,
    )

    if args.diagnose_severity:
        _report_severity_diagnostics(
            series,
            global_prior,
            per_route,
            pre_clamp,
            min_ticks=args.min_ticks,
            severity_floor=args.severity_floor,
        )
        return 0

    if args.diagnose_advance:
        _report_advance_diagnostics(
            series,
            global_prior,
            per_route,
            route_advance_rates,
            min_ticks=args.min_ticks,
        )
        return 0

    if args.diagnose_service:
        _report_service_diagnostics(
            series,
            global_prior,
            per_route,
            min_ticks=args.min_ticks,
        )
        return 0

    # Empirical dwell quantiles from the regime_transitions stream over the
    # same window. Cells below MIN_SAMPLES_FOR_EMPIRICAL fall back to the
    # geometric dwell in the Worker — no-op if the stream is empty.
    #
    # `normal` is the exception: its cells come from the partially-pooled
    # estimator instead, for every route and with no min-samples gate. A route
    # only completes a normal regime by leaving normal, so that gate admits the
    # flappiest routes and drops the steadiest ones onto a memoryless geometric
    # projection. See training/pooled_dwell.py.
    from training.eval import (
        load_predictions,
        load_transitions,
        open_regimes_from_predictions,
    )

    transitions = load_transitions(client, cfg.bucket, start_date, end_date)
    # Censoring boundary for still-open regimes: "now", clamped to the
    # requested window so a backdated --end doesn't fabricate giant censored
    # durations from regimes that actually ended after the window.
    _, end_epoch = _aligned_window(start_date, end_date)
    window_end = min(int(datetime.now(UTC).timestamp()), end_epoch)
    # Still-open regimes come from the prediction stream, not from the last
    # transition record. A route with no transitions in the window has no
    # transition to read a regime off, so inferring from transitions alone drops
    # it entirely — and for `normal` those are the steadiest routes, the ones
    # the pooled estimator below exists to serve.
    # None when the prediction stream is unavailable, which falls the censoring
    # back to transition inference: degraded and blind to the quiet routes, but
    # better than dropping every censored observation on the floor.
    open_regimes = (
        open_regimes_from_predictions(
            load_predictions(client, cfg.bucket, start_date, end_date),
            window_end=window_end,
        )
        or None
    )
    dwell_q = compute_dwell_quantiles(
        transitions,
        window_end=window_end,
        tail_fn=loglogistic_tail,
        open_regimes=open_regimes,
    )
    dwell_q_by_alert = compute_dwell_quantiles_by_alert(
        transitions, tail_fn=loglogistic_tail
    )
    dwell_q_by_cause = compute_dwell_quantiles_by_cause(
        transitions, tail_fn=loglogistic_tail
    )
    dwell_q_normal = pooled_dwell_cells(
        transitions,
        state="normal",
        window_end=window_end,
        open_regimes=open_regimes,
    )
    for route, cell in dwell_q_normal.items():
        dwell_q.setdefault(route, {})["normal"] = cell
    n_dwell_cells = sum(len(by_state) for by_state in dwell_q.values())
    n_dwell_alert_cells = sum(
        len(by_alert)
        for by_state in dwell_q_by_alert.values()
        for by_alert in by_state.values()
    )
    n_dwell_cause_cells = sum(
        len(by_cause)
        for by_state in dwell_q_by_cause.values()
        for by_cause in by_state.values()
    )
    dwell_movement, movement_dwell_stats = _movement_dwell(
        cfg, client, start_date, end_date, window_end
    )
    n_movement_dwell_cells = sum(len(by_state) for by_state in dwell_movement.values())

    # Grade-driven recovery recalibration: the alert-shadow dwell quantiles run
    # systematically optimistic (predicted recover-by-H above the observed
    # clearance rate, worst at 60-120min). Fit a monotone CDF warp F' = F**gamma
    # on a HELD-OUT tail of this window's incidents and apply it to the published
    # curves, so recover-by drops toward reality. gamma 1.0 (identity) when the
    # tail is too short/thin or already calibrated — the Worker reads the
    # reshaped quantiles unchanged. The movement dwell block is a separate arm
    # with its own point-mass calibration and is left untouched here.
    recovery_gamma, recovery_recalib = fit_published_recovery_gamma(
        cfg, client, start_date, end_date
    )
    if recovery_gamma != 1.0:
        dwell_q = recalibrate_dwell_cells(dwell_q, recovery_gamma)
        dwell_q_by_alert = recalibrate_dwell_cells_by_key(
            dwell_q_by_alert, recovery_gamma
        )
        dwell_q_by_cause = recalibrate_dwell_cells_by_key(
            dwell_q_by_cause, recovery_gamma
        )

    if args.dry_run:
        dry_routes = {r: _params_to_json(p) for r, p in per_route.items()}
        for r, by_state in dwell_q.items():
            if r in dry_routes:
                dry_routes[r]["dwell_quantiles"] = by_state
        for r, by_state_alert in dwell_q_by_alert.items():
            if r in dry_routes:
                dry_routes[r]["dwell_quantiles_by_alert"] = by_state_alert
        for r, by_state_cause in dwell_q_by_cause.items():
            if r in dry_routes:
                dry_routes[r]["dwell_quantiles_by_cause"] = by_state_cause
        print(
            json.dumps(
                {
                    "global_prior": _params_to_json(global_prior),
                    "routes": dry_routes,
                    "dwell_cells": n_dwell_cells,
                    "dwell_alert_cells": n_dwell_alert_cells,
                    "dwell_cause_cells": n_dwell_cause_cells,
                    "dwell_movement_cells": n_movement_dwell_cells,
                    "baseline_cells": n_baseline_cells,
                    "service_cells": n_service_cells,
                },
                indent=2,
            )
        )
        return 0

    if corpus.span_seconds < MIN_DATA_DAYS * 86_400:
        print(
            f"archive spans {corpus.span_seconds / 86_400:.1f}d "
            f"(< {MIN_DATA_DAYS}d minimum) — refusing to publish; thin data "
            "overfits transition self-loops",
            file=sys.stderr,
        )
        return 1

    n_routes_trained = sum(1 for s in series.values() if len(s) >= args.min_ticks)
    # The knobs that determine the fit — with code_sha + the immutable archive
    # these make a params_version re-derivable. Window is recorded as resolved
    # dates so it reproduces regardless of when --days was relative to.
    hyperparams = {
        "window_start": start_date.isoformat(),
        "window_end": end_date.isoformat(),
        "prior_strength": args.prior_strength,
        "min_ticks": args.min_ticks,
        "routes": sorted(args.routes.split(",")) if args.routes else None,
        "recovery_recalibration": recovery_recalib,
    }
    trained_at = int(datetime.now(UTC).timestamp())
    versioned_key = write_params(
        client,
        cfg.bucket,
        per_route,
        corpus=corpus,
        n_routes_trained=n_routes_trained,
        dwell_quantiles=dwell_q,
        dwell_quantiles_by_alert=dwell_q_by_alert,
        dwell_quantiles_by_cause=dwell_q_by_cause,
        dwell_movement=dwell_movement,
        hyperparams=hyperparams,
        input_profile=input_profile,
        movement_baseline=movement_baseline,
        movement_through_stops=(None if through is None else stops_to_json(through)),
        service_baseline=service_baseline,
        schedule_rate=schedule_rate,
        trained_at=trained_at,
    )
    write_service_baseline(
        client,
        cfg.bucket,
        service_baseline_hourly,
        trained_at,
        params_trained_at=trained_at,
        quantiles=service_baseline_hourly_quantiles,
    )
    n_segment_cells = write_segment_params(
        cfg,
        client,
        cfg.bucket,
        start_date,
        end_date,
        trained_at,
        static_successors,
        static_patterns,
        topology_source,
        through,
    )
    n_segment_dwell_cells, segment_dwell_stats = write_segment_dwell(
        client, cfg.bucket, start_date, end_date, trained_at, through
    )
    print(
        f"published {PARAMS_KEY} + {versioned_key}: "
        f"{n_routes_trained}/{len(per_route)} routes fitted "
        f"(prior_strength={args.prior_strength}, dwell_cells={n_dwell_cells}, "
        f"dwell_alert_cells={n_dwell_alert_cells}, "
        f"dwell_cause_cells={n_dwell_cause_cells}, "
        f"recovery_gamma={recovery_gamma}, "
        f"dwell_movement_cells={n_movement_dwell_cells} "
        f"[own={movement_dwell_stats['n_own']}, pooled={movement_dwell_stats['n_pooled']}, "
        f"atom={movement_dwell_stats['n_atom']}, "
        f"source={movement_dwell_stats['source']}], "
        f"baseline_cells={n_baseline_cells}, "
        f"service_cells={n_service_cells} (hourly sidecar {n_service_hourly_cells}, "
        f"quantiles {n_service_hourly_quantile_cells}), "
        f"schedule_cells={n_schedule_cells}, "
        f"segment_cells={n_segment_cells}, segment_dwell_cells={n_segment_dwell_cells} "
        f"[own={segment_dwell_stats.n_cells_own}, "
        f"route={segment_dwell_stats.n_cells_route}, "
        f"system={segment_dwell_stats.n_cells_system}])"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
