"""The causal recovery-duration climatology: the empirical distribution of how
long severe-truth incidents actually last, keyed (route, alert_type) with
pooling, conditioned at serve time on how long THIS disruption has already run.

Recovery is the product's differentiator and every fitted dwell curve has lost
to this population (the 2026-09-04 review graded causal_skill -1.70, later -0.17,
still negative). So rather than publish another curve, the trainer publishes the
climatology itself and makes it the reference every future conditioner must beat.

This module owns the ONE population builder both the review and the trainer read,
so the number the review grades against and the number the Worker serves cannot
drift apart. It deliberately does NOT import training.review (matplotlib-heavy):
the publish path must stay light, the same reason recovery_recalibration.py
inlines its truth derivation.

The population is exactly the one review.py graded recovery skill against:
load predictions -> presence_mask_from_predictions -> load_truth_observations
(masked so over-extended severe tails close) -> mta_truth(floor=2) ->
disruptive_types_by_key -> extract_episodes -> keep the uncensored, non-standing
episodes. Each episode's alert_type is the Worker's own primary_alert_type at
onset, read from the SAME prediction stream field the Worker publishes live, so
the trainer's cell key and the Worker's serve-time key are the identical string.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

from momentarily.hmm import Observation
from momentarily.mapping import CANONICAL_SEVERITY_FLOOR, severity_tier
from training.episodes import Episode, disruptive_types_by_key, extract_episodes
from training.eval import load_predictions, snap_tick
from training.load_r2 import (
    PresenceMask,
    build_tick_observations,
    fetch_objects,
    list_alert_keys,
    presence_mask_from_predictions,
)

if TYPE_CHECKING:
    from datetime import date

    from mypy_boto3_s3 import S3Client

    from training.load import TickObservation

# A (route, alert_type) cell needs at least this many uncensored severe-truth
# durations to stand on its own; below it the cell pools up to (alert_type) then
# system-wide. Also the floor for the serve-time conditional: fewer than this
# many samples still running past the elapsed time and the remaining-duration
# distribution is published null (recovery_indeterminate) rather than
# extrapolated. Chosen from the fit: over the trailing 35d the raw (route,
# alert_type) severe-incident counts are thin (most routes see a handful of
# severe episodes of any one alert_type), so a floor of 8 — the same order as
# the service sidecar's SERVICE_MIN_NIGHTS gate — keeps a published route cell
# from resting on two or three incidents while still letting the busiest
# route/alert pairs stand alone. One constant governs both the pooling gate and
# the conditional floor so a consumer cannot see a cell the trainer would have
# pooled away.
MIN_CELL_SAMPLES = 8


# --- MTA-derived ground truth (moved here from training.review so the publish
# path and the Worker-parity path never import matplotlib) --------------------


def derive_mta_state(obs: Observation) -> str:
    """MTA-derived ground-truth state for a tick — mirrors the HMM training
    flags (planned-work explicitly excluded upstream in build_tick_observations)."""
    if obs.has_suspended_alert:
        return "suspended"
    if obs.has_delays or obs.has_service_change:
        return "disrupted"
    return "normal"


def derive_graded_mta_state(alert_types: tuple[str, ...], *, floor: int) -> str:
    """Severity-graded ground-truth state from a route-tick's active alert_types.

    suspended if any suspension alert (tier 3); disrupted if any alert reaches
    `floor`; otherwise normal — so sub-floor alerts (minor delays, routine
    reroutes) read normal and the HMM filtering them is scored as correct, not a
    miss. floor=1 reproduces the breadth-dominated truth; floor=2 is severe-only.
    """
    tiers = [severity_tier(at) for at in alert_types]
    if any(t == 3 for t in tiers):
        return "suspended"
    if any(t >= floor for t in tiers):
        return "disrupted"
    return "normal"


def load_truth_observations(
    client: S3Client,
    bucket: str,
    start_date: Any,
    end_date: Any,
    *,
    mask: PresenceMask | None = None,
) -> list[TickObservation]:
    """Per-(route, tick) observations from the alerts archive, with the active
    alert_types retained for severity grading. Fetched once; both the broad and
    graded truths derive from it. With `mask` (built from the predictions
    stream), over-extended open-ended alert tails are dropped so severe episodes
    actually close -- otherwise every open alert runs to corpus end. See 06j."""
    # The caller's client and bucket are the whole R2 surface here — never
    # reload config (which reads the vault) to rediscover a bucket we were given.
    bodies = fetch_objects(
        client, bucket, list_alert_keys(client, bucket, start_date, end_date)
    )
    return build_tick_observations(bodies, active_mask=mask)


def mta_truth(
    obs_list: list[TickObservation], *, severity_floor: int = CANONICAL_SEVERITY_FLOOR
) -> dict[tuple[str, int], str]:
    """(route, tick) → MTA-derived state. Defaults to the canonical severe-only
    truth (severity_floor >= 2: only Severe Delays / suspension count as
    disrupted). severity_floor <= 1 is the legacy breadth truth (any
    delays/service-change = disrupted), kept as a sensitivity. Ticks not in the
    dict had no active alerts → 'normal'."""
    if severity_floor <= 1:
        return {(o.route_id, o.tick): derive_mta_state(o.observation) for o in obs_list}
    return {
        (o.route_id, o.tick): derive_graded_mta_state(
            o.disruptive_types, floor=severity_floor
        )
        for o in obs_list
    }


def build_mta_truth(
    client: S3Client,
    bucket: str,
    start_date: Any,
    end_date: Any,
    *,
    severity_floor: int = CANONICAL_SEVERITY_FLOOR,
    mask: PresenceMask | None = None,
) -> dict[tuple[str, int], str]:
    """Convenience: fetch + derive the truth in one call."""
    obs_list = load_truth_observations(client, bucket, start_date, end_date, mask=mask)
    return mta_truth(obs_list, severity_floor=severity_floor)


# --- The shared population ----------------------------------------------------


@dataclass(frozen=True)
class PopulationEpisode:
    """One kept incident plus the alert_type its cell keys on.

    `alert_type` is the Worker's primary_alert_type at the onset tick, read from
    the prediction stream (the archived Worker output), so the trainer keys on
    the exact string the Worker serves on. None when the onset tick carried no
    prediction primary_alert_type (rare — the presence mask makes a severe-truth
    onset an active route-tick); such an episode still counts into the
    system-wide pool but keys no (route, alert_type) or (alert_type) cell.
    """

    episode: Episode
    alert_type: str | None

    @property
    def duration_min(self) -> float:
        return self.episode.duration_sec / 60.0


@dataclass(frozen=True)
class RecoveryPopulation:
    """The uncensored, non-standing severe-truth episodes over one window."""

    episodes: list[PopulationEpisode]
    window_start: int
    window_end: int
    severity_floor: int

    def durations_min(self) -> list[float]:
        """Every kept episode's duration in minutes — the review's
        recovery_causal_baseline population, unkeyed."""
        return [pe.duration_min for pe in self.episodes]


def window_bounds(start_date: date, end_date: date) -> tuple[int, int]:
    """[midnight(start_date), midnight(end_date)+1d) as epoch seconds — the
    same convention review.py's pre-window causal baseline uses."""
    window_start = int(
        datetime(
            start_date.year, start_date.month, start_date.day, tzinfo=UTC
        ).timestamp()
    )
    window_end = int(
        (
            datetime(end_date.year, end_date.month, end_date.day, tzinfo=UTC)
            + timedelta(days=1)
        ).timestamp()
    )
    return window_start, window_end


def build_recovery_population(
    client: S3Client,
    bucket: str,
    start_date: date,
    end_date: date,
    *,
    severity_floor: int = CANONICAL_SEVERITY_FLOOR,
) -> RecoveryPopulation:
    """The one severe-truth incident-duration population the review grades
    recovery skill against AND the trainer fits the climatology on.

    Exactly review.py's recovery_causal_baseline build: load the prediction
    stream (for the presence mask AND each route-tick's primary_alert_type),
    mask the alert archive so over-extended severe tails close, grade the
    severe-only truth, segment into episodes, and keep only the uncensored,
    non-standing ones (a left/right-censored run has no observed duration; a
    standing advisory is the alert feed, not an incident)."""
    window_start, window_end = window_bounds(start_date, end_date)
    preds = load_predictions(client, bucket, start_date, end_date)
    mask = presence_mask_from_predictions(preds)
    truth_obs = load_truth_observations(client, bucket, start_date, end_date, mask=mask)
    truth = mta_truth(truth_obs, severity_floor=severity_floor)
    types = disruptive_types_by_key(truth_obs)
    # The Worker's own primary_alert_type per route-tick, from the archived
    # prediction rows — the identical field and computation the Worker serves.
    primary_by_key: dict[tuple[str, int], str] = {
        (p.route, snap_tick(p.ts)): p.primary_alert_type
        for p in preds
        if p.primary_alert_type is not None
    }
    episodes = extract_episodes(
        truth, types, window_start=window_start, window_end=window_end
    )
    kept: list[PopulationEpisode] = [
        PopulationEpisode(
            episode=ep,
            alert_type=primary_by_key.get((ep.route, ep.onset)),
        )
        for ep in episodes
        if not (ep.left_censored or ep.right_censored or ep.standing)
    ]
    return RecoveryPopulation(
        episodes=kept,
        window_start=window_start,
        window_end=window_end,
        severity_floor=severity_floor,
    )


# --- Conditional remaining-duration math (mirrored in worker/src/dwell.ts's
# climatology path; keep the two in sync via the parity fixture) --------------


def empirical_quantile(sorted_values: Sequence[float], q: float) -> float:
    """The q-th quantile of an ascending sample list by linear interpolation
    between order statistics (numpy's default 'linear'/type-7). Small enough to
    reproduce byte-for-byte in TypeScript, which is the whole point."""
    n = len(sorted_values)
    if n == 0:
        raise ValueError("empirical_quantile of an empty sample")
    if n == 1:
        return float(sorted_values[0])
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo]))


@dataclass(frozen=True)
class RemainingQuantiles:
    """The remaining-duration distribution D - t | D > t, in minutes."""

    p25: float
    p50: float
    p75: float
    n: int  # samples still running past the elapsed time


def conditional_remaining_quantiles(
    samples_min: Sequence[float],
    elapsed_min: float,
    *,
    min_samples: int = MIN_CELL_SAMPLES,
) -> RemainingQuantiles | None:
    """Median and IQR of how much LONGER a disruption runs given it has already
    lasted `elapsed_min`, from the cell's realized durations.

    Conditions the population on survival: keeps only durations that exceed the
    elapsed time, subtracts it, and reads the empirical p25/p50/p75 of what
    remains. None when fewer than `min_samples` durations run past the elapsed
    time — the disruption has outlived its population and the honest answer is
    'indeterminate', never an extrapolated tail."""
    remaining = sorted(s - elapsed_min for s in samples_min if s > elapsed_min)
    if len(remaining) < min_samples:
        return None
    return RemainingQuantiles(
        p25=empirical_quantile(remaining, 0.25),
        p50=empirical_quantile(remaining, 0.5),
        p75=empirical_quantile(remaining, 0.75),
        n=len(remaining),
    )


# --- Cell construction, resolution, and serve-time conditional ----------------
#
# A cell is a self-contained record: its n, its pooling `level`
# ('route'|'alert_type'|'system'), and its sorted duration samples (minutes).
# The single fit window is recorded once at the top of the doc — every cell
# shares it. The trainer resolves the pooling authoritatively (a raw (route,
# alert_type) that clears MIN stands alone; else it pools to the alert_type
# pool if THAT clears MIN; else to the system pool) and writes one resolved,
# sample-bearing cell per OBSERVED (route, alert_type). No cell is a reference:
# the Worker reads the samples it needs straight off the resolved cell. The
# by_alert_type and system pools are published too, so a live (route,
# alert_type) the window never saw still resolves (alert_type pool, else
# system). Inlining the resolved samples costs a copy over a reference index,
# measured at ~97KiB for the trailing 35d (vs ~14KiB pooled-by-reference) — a
# no-store state sidecar off the hot path, and the copy keeps every cell
# self-describing, so the Worker never rebuilds the pooling rule.


def _sorted_durations(episodes: Sequence[PopulationEpisode]) -> list[float]:
    return sorted(pe.duration_min for pe in episodes)


def _cell(
    samples: list[float], level: str, window_start: int, window_end: int
) -> dict[str, Any]:
    # Each cell is self-describing: it carries the fit window it was measured
    # over so a consumer reading one cell needs nothing else. All cells in a doc
    # share the one window (also recorded top-level), so this is a cheap echo,
    # not independent metadata.
    return {
        "n": len(samples),
        "level": level,
        "window_start": window_start,
        "window_end": window_end,
        "samples_min": samples,
    }


def build_climatology_cells(
    population: RecoveryPopulation,
    *,
    min_samples: int = MIN_CELL_SAMPLES,
) -> dict[str, Any]:
    """The published climatology: one resolved, self-contained cell per observed
    (route, alert_type), plus the alert_type and system pools that back the
    serve-time fallback for a pair the window never saw."""
    ws, we = population.window_start, population.window_end
    system = _sorted_durations(population.episodes)

    by_alert: dict[str, list[PopulationEpisode]] = {}
    by_route_alert: dict[tuple[str, str], list[PopulationEpisode]] = {}
    for pe in population.episodes:
        if pe.alert_type is None:
            continue
        by_alert.setdefault(pe.alert_type, []).append(pe)
        by_route_alert.setdefault((pe.episode.route, pe.alert_type), []).append(pe)

    alert_samples: dict[str, list[float]] = {
        at: _sorted_durations(eps) for at, eps in by_alert.items()
    }
    by_alert_type_doc: dict[str, dict[str, Any]] = {
        at: _cell(samples, "alert_type", ws, we)
        for at, samples in alert_samples.items()
        if len(samples) >= min_samples
    }

    cells: dict[str, dict[str, dict[str, Any]]] = {}
    for (route, at), eps in by_route_alert.items():
        route_samples = _sorted_durations(eps)
        if len(route_samples) >= min_samples:
            cells.setdefault(route, {})[at] = _cell(route_samples, "route", ws, we)
        elif len(alert_samples.get(at, ())) >= min_samples:
            cells.setdefault(route, {})[at] = _cell(
                alert_samples[at], "alert_type", ws, we
            )
        elif system:
            cells.setdefault(route, {})[at] = _cell(system, "system", ws, we)

    doc: dict[str, Any] = {
        "min_samples": min_samples,
        "window_start": population.window_start,
        "window_end": population.window_end,
        "severity_floor": population.severity_floor,
        "n_episodes": len(population.episodes),
        "cells": cells,
        "by_alert_type": by_alert_type_doc,
    }
    if system:
        doc["system"] = _cell(system, "system", ws, we)
    return doc


def resolve_climatology_cell(
    doc: Mapping[str, Any], route: str, alert_type: str | None
) -> dict[str, Any] | None:
    """The cell a Worker serves for (route, alert_type): the resolved route cell,
    else the alert_type pool, else the system pool. None when alert_type is None
    — the climatology requires a primary alert type (the published condition
    alone does not key a cell), so a disrupted route with no active alert gets no
    climatology. Mirrors resolveRecoveryCell in worker/src/params.ts so the two
    cannot disagree."""
    if alert_type is None:
        return None
    route_cells = doc.get("cells", {}).get(route)
    if route_cells is not None and alert_type in route_cells:
        return route_cells[alert_type]
    alert_cell = doc.get("by_alert_type", {}).get(alert_type)
    if alert_cell is not None:
        return alert_cell
    system = doc.get("system")
    return system if system else None


@dataclass(frozen=True)
class ClimatologyRecovery:
    """The served remaining-duration estimate, in whole minutes.

    `minutes`/`low`/`high` are the median and p25/p75 of D - t | D > t, or None
    when the disruption has outlived its population (`indeterminate`)."""

    minutes: int | None
    low: int | None
    high: int | None
    n: int
    level: str
    indeterminate: bool


def climatology_recovery(
    doc: Mapping[str, Any],
    route: str,
    alert_type: str | None,
    elapsed_min: float,
) -> ClimatologyRecovery | None:
    """Serve recovery for a disrupted route from the published climatology.

    None when no cell describes (route, alert_type) at all (absent climatology).
    Otherwise the median/IQR of the remaining duration, or an indeterminate
    record (minutes None) when fewer than `min_samples` durations outlast the
    elapsed time. The Worker mirrors this exactly."""
    cell = resolve_climatology_cell(doc, route, alert_type)
    if cell is None:
        return None
    min_samples = int(doc.get("min_samples", MIN_CELL_SAMPLES))
    rq = conditional_remaining_quantiles(
        cell["samples_min"], elapsed_min, min_samples=min_samples
    )
    if rq is None:
        return ClimatologyRecovery(
            minutes=None,
            low=None,
            high=None,
            n=int(cell["n"]),
            level=str(cell["level"]),
            indeterminate=True,
        )
    # Half-up rounding (floor(x + 0.5)) on the non-negative remaining durations,
    # matching JavaScript's Math.round in worker/src/recovery.ts so the published
    # integer minutes are identical cross-language (Python's round() is banker's).
    return ClimatologyRecovery(
        minutes=math.floor(rq.p50 + 0.5),
        low=math.floor(rq.p25 + 0.5),
        high=math.floor(rq.p75 + 0.5),
        n=int(cell["n"]),
        level=str(cell["level"]),
        indeterminate=False,
    )
