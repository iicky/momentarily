"""Event-based eval scorecard — grade per incident episode, not per tick.

Three metric families over the severe-only truth episodes (episodes.py), each
reported with its event count so a headline never hides a tiny n:

  - onset latency: signed minutes from a truth episode's onset to the model's
    detection of it, matching model episodes to truth episodes by time overlap.
    Model episodes are segmented from the published-condition stream on the same
    grid as the truth, so detection is symmetric with the truth definition.
  - per-episode recovery CRPS / PIT: score the model's predicted recovery-time
    distribution against realized duration, on UNCENSORED episodes only (a
    right/left-censored episode has no observed duration, so scoring it would
    bias the metric). Censored and curve-less episodes are counted, not scored.
  - false-alarm episodes: model episodes with no overlapping truth episode,
    cross-checked against the independent movement truth (a false alarm the
    movement also disputes is a genuine over-call; one the movement confirms is
    an alert-truth gap, not a model error).

Tick-level Brier stays available upstream but as an appendix, not the headline.
Pure over its inputs so it grades without R2 and unit-tests on fixtures.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping
from statistics import median
from typing import Any

from training.episodes import Episode, extract_episodes
from training.eval import TICK_SECONDS, PredictionRecord, snap_tick
from training.recovery_dist import (
    RecoveryDistSample,
    predicted_recovery_curve,
    recovery_dist_report,
    recovery_verdict,
    report_as_dict,
    verdict_as_dict,
)

NOT_NORMAL = ("disrupted", "suspended")

# Curve + optional log-logistic tail for a (route, state, cause) dwell cell. A
# cause-aware lookup falls back cause -> state -> pooled, so an unknown cause
# degrades to the state-level curve rather than missing.
DwellLookup = Callable[[str, str, str], "tuple[list[int], list[float] | None] | None"]


def model_episodes(
    predictions: list[PredictionRecord], *, window_start: int, window_end: int
) -> list[Episode]:
    """Segment the model's published-condition stream into episodes, the same way
    the truth is segmented (absent/normal tick ends a run)."""
    state: dict[tuple[str, int], str] = {}
    for p in predictions:
        if p.condition in NOT_NORMAL:
            state[(p.route, snap_tick(p.ts))] = p.condition
    return extract_episodes(state, {}, window_start=window_start, window_end=window_end)


def _overlaps(a: Episode, b: Episode) -> bool:
    return a.route == b.route and a.onset < b.recovery and b.onset < a.recovery


def onset_latency(
    truth_eps: list[Episode],
    model_eps: list[Episode],
) -> dict[str, Any]:
    """Signed onset latency (model minus truth, minutes) per truth episode, with
    detection rate. A truth episode is detected iff a model episode overlaps it —
    the same overlap predicate false_alarms uses, so a model episode is either a
    detection or a false alarm, never both; latency uses the overlapping model
    episode whose onset is nearest."""
    by_route: dict[str, list[Episode]] = defaultdict(list)
    for m in model_eps:
        by_route[m.route].append(m)

    latencies: list[float] = []
    detected = 0
    for t in truth_eps:
        covering = [m for m in by_route.get(t.route, []) if _overlaps(m, t)]
        if covering:
            nearest = min(covering, key=lambda m: abs(m.onset - t.onset))
            latencies.append((nearest.onset - t.onset) / 60.0)
            detected += 1

    n = len(truth_eps)
    return {
        "n_episodes": n,
        "n_detected": detected,
        "n_missed": n - detected,
        "detection_rate": detected / n if n else None,
        "median_latency_min": median(latencies) if latencies else None,
        "mean_latency_min": sum(latencies) / len(latencies) if latencies else None,
    }


def _movement_verdict(
    ep: Episode,
    movement_truth: dict[tuple[str, int], str],
    *,
    min_frac: float,
) -> str:
    """Classify a model episode against the movement truth over its ticks."""
    judged = [
        movement_truth[(ep.route, tick)]
        for tick in range(ep.onset, ep.recovery, TICK_SECONDS)
        if (ep.route, tick) in movement_truth
    ]
    if not judged:
        return "unjudgeable"
    not_normal = sum(s != "normal" for s in judged)
    return "confirmed" if not_normal / len(judged) >= min_frac else "contradicted"


def false_alarms(
    model_eps: list[Episode],
    truth_eps: list[Episode],
    movement_truth: dict[tuple[str, int], str],
    *,
    min_frac: float = 0.5,
) -> dict[str, Any]:
    """Model episodes with no overlapping truth episode, split by whether the
    independent movement truth confirms (real incident the alert-truth missed) or
    contradicts (a genuine over-call) them."""
    by_route: dict[str, list[Episode]] = defaultdict(list)
    for t in truth_eps:
        by_route[t.route].append(t)

    fa = [
        m
        for m in model_eps
        if not any(_overlaps(m, t) for t in by_route.get(m.route, []))
    ]
    verdicts = Counter(
        _movement_verdict(m, movement_truth, min_frac=min_frac) for m in fa
    )
    n_model = len(model_eps)
    return {
        "n_model_episodes": n_model,
        "n_false_alarm": len(fa),
        "false_alarm_rate": len(fa) / n_model if n_model else None,
        "movement_contradicted": verdicts.get("contradicted", 0),
        "movement_confirmed": verdicts.get("confirmed", 0),
        "movement_unjudgeable": verdicts.get("unjudgeable", 0),
    }


def episode_recovery(
    truth_eps: list[Episode], dwell_lookup: DwellLookup
) -> dict[str, Any]:
    """Per-episode recovery CRPS/PIT over uncensored episodes with a dwell curve.
    The predicted curve is the model's recovery forecast for the episode's peak
    state and cause at onset (elapsed 0); the outcome is the realized duration."""
    samples: list[RecoveryDistSample] = []
    n_censored = 0
    n_no_curve = 0
    for e in truth_eps:
        if e.left_censored or e.right_censored:
            n_censored += 1
            continue
        cell = dwell_lookup(e.route, e.peak_state, e.cause)
        if cell is None or len(cell[0]) < 2:
            n_no_curve += 1
            continue
        curve_sec, tail_ll = cell
        samples.append(
            RecoveryDistSample(
                pred_curve=predicted_recovery_curve(0.0, curve_sec, tail_ll),
                actual_min=e.duration_sec / 60.0,
                regime_key=f"{e.route}:{e.onset}",
            )
        )
    report = recovery_dist_report(samples)
    return {
        "n_scored": len(samples),
        "n_censored_excluded": n_censored,
        "n_no_curve": n_no_curve,
        "report": report_as_dict(report),
        "verdict": verdict_as_dict(recovery_verdict(report)),
    }


def _cell_curve(cell: Any) -> tuple[list[int], list[float] | None] | None:
    """Extract (curve_sec, tail_ll) from a dwell-cell dict, or None if unusable."""
    if not cell:
        return None
    curve: list[int] = cell.get("curve_sec") or []
    if len(curve) < 2:
        return None
    tail: list[float] | None = cell.get("tail_ll")
    return curve, tail


def dwell_lookup_from_params(params: dict[str, Any]) -> DwellLookup:
    """Cause-aware (route, state, cause) -> (curve_sec, tail_ll) lookup over a
    params.json doc. Fallback chain: the cause-conditioned cell
    (routes[route]['dwell_quantiles_by_cause'][state][cause]) if present, else the
    (route, state) aggregate (routes[route]['dwell_quantiles'][state]). params is
    prequential (trained strictly before the graded window), so scoring against it
    does not leak outcomes."""
    routes: dict[str, Any] = params.get("routes") or {}

    def lookup(
        route: str, state: str, cause: str
    ) -> tuple[list[int], list[float] | None] | None:
        route_doc: dict[str, Any] = routes.get(route) or {}
        by_cause: dict[str, Any] = route_doc.get("dwell_quantiles_by_cause") or {}
        state_causes: dict[str, Any] = by_cause.get(state) or {}
        cell = state_causes.get(cause)
        if not cell:
            quantiles: dict[str, Any] = route_doc.get("dwell_quantiles") or {}
            cell = quantiles.get(state)
        return _cell_curve(cell)

    return lookup


def cause_dwell_lookup(
    by_cause: Mapping[str, Any],
    by_state: Mapping[str, Any],
    pooled: Mapping[str, Any],
) -> DwellLookup:
    """Cause-aware lookup over TRAIN-DERIVED cells (compute_dwell_quantiles* on the
    training window — never the scored window, which would leak outcomes). Fallback
    chain: (route, state, cause) -> (route, state) -> pooled(state)."""

    def lookup(
        route: str, state: str, cause: str
    ) -> tuple[list[int], list[float] | None] | None:
        route_causes: dict[str, Any] = by_cause.get(route) or {}
        state_causes: dict[str, Any] = route_causes.get(state) or {}
        cell = state_causes.get(cause)
        if not cell:
            route_states: dict[str, Any] = by_state.get(route) or {}
            cell = route_states.get(state)
        if not cell:
            cell = pooled.get(state)
        return _cell_curve(cell)

    return lookup


def episode_scorecard(
    truth_eps: list[Episode],
    predictions: list[PredictionRecord],
    movement_truth: dict[tuple[str, int], str],
    dwell_lookup: DwellLookup,
    *,
    window_start: int,
    window_end: int,
) -> dict[str, Any]:
    """Assemble the event-based scorecard: onset latency, per-episode recovery,
    and false-alarm episodes, each with its event count."""
    model_eps = model_episodes(
        predictions, window_start=window_start, window_end=window_end
    )
    return {
        "n_truth_episodes": len(truth_eps),
        "n_model_episodes": len(model_eps),
        "onset_latency": onset_latency(truth_eps, model_eps),
        "recovery": episode_recovery(truth_eps, dwell_lookup),
        "false_alarms": false_alarms(model_eps, truth_eps, movement_truth),
    }
