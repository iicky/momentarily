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
from collections.abc import Callable, Mapping, Sequence
from statistics import median
from typing import Any

from training.episodes import Episode, extract_episodes
from training.eval import (
    MOVEMENT_ARM_LABEL,
    NOT_NORMAL,
    SHADOW_ARM_LABEL,
    TICK_SECONDS,
    PredictionRecord,
    published_arm,
    published_condition_coverage,
    snap_tick,
)
from training.recovery_dist import (
    RecoveryDistSample,
    predicted_recovery_curve,
    recovery_dist_report,
    recovery_verdict,
    report_as_dict,
    verdict_as_dict,
)

# How far ahead of a truth onset a model episode may fire and still count as a
# detection of it. Requiring bare overlap penalises the exact behaviour the
# movement signal exists for: the model calls a stall, the MTA posts the alert
# 20 minutes later, and if the model episode closed in between it scored as a
# miss AND a false alarm. Matches the changepoint alignment's +/-30min window.
ONSET_LEAD_TOLERANCE_SEC = 30 * 60

# Curve + optional log-logistic tail for a (route, state, cause) dwell cell. A
# cause-aware lookup falls back cause -> state -> pooled, so an unknown cause
# degrades to the state-level curve rather than missing.
# A lookup yields (curve_sec, tail_ll, atom) where atom is (atom_p, atom_sec) for
# a cell published as a point-mass mixture and None for a plain continuous cell.
# The atom rides along with the curve rather than being fetched separately so a
# cell can never be graded with half its distribution.
DwellLookup = Callable[
    [str, str, str],
    "tuple[list[int], list[float] | None, tuple[float, float] | None] | None",
]


def model_episodes(
    predictions: list[PredictionRecord], *, window_start: int, window_end: int
) -> list[Episode]:
    """Segment the model's published-condition stream into episodes, the same way
    the truth is segmented (absent/normal tick ends a run).

    Grades `published_condition` — the movement-primary state consumers actually
    read — not the `condition` alert-shadow. The two are different arms and
    disagree: over 08-04..08-07 the shadow called disrupted on 58 route-ticks and
    the published arm on 12, so grading the shadow scored something no consumer
    sees. Falls back to `condition` only for rows written before the published
    field existed.

    `unknown` (no movement reading) is not not-normal, so it closes a run the
    same as normal. That is a real limitation, not a neutral gap — see
    published_condition_coverage for how much of the window it covers.
    """
    state: dict[tuple[str, int], str] = {}
    for p in predictions:
        published = published_arm(p)
        if published in NOT_NORMAL:
            state[(p.route, snap_tick(p.ts))] = published
    return extract_episodes(state, {}, window_start=window_start, window_end=window_end)


def _matches(
    model: Episode, truth: Episode, *, lead_sec: int = ONSET_LEAD_TOLERANCE_SEC
) -> bool:
    """Whether a model episode is a detection of a truth episode.

    Overlap, with the model episode's window extended forward by `lead_sec` so an
    early call that closed before the alert landed still counts. lead_sec=0 is
    bare overlap.
    """
    return (
        model.route == truth.route
        and model.onset < truth.recovery
        and truth.onset < model.recovery + lead_sec
    )


def onset_latency(
    truth_eps: list[Episode],
    model_eps: list[Episode],
    *,
    lead_sec: int = ONSET_LEAD_TOLERANCE_SEC,
) -> dict[str, Any]:
    """Signed onset latency (model minus truth, minutes) per truth episode, with
    detection rate. A truth episode is detected iff a model episode matches it —
    the same predicate false_alarms uses, so a model episode is either a
    detection or a false alarm, never both; latency uses the matching model
    episode whose onset is nearest.

    Negative latency is the model leading the alert feed. `n_detected_leading`
    counts detections whose model onset preceded the truth onset — the headline
    number for the early-warning claim.
    """
    by_route: dict[str, list[Episode]] = defaultdict(list)
    for m in model_eps:
        by_route[m.route].append(m)

    latencies: list[float] = []
    detected = 0
    for t in truth_eps:
        covering = [
            m for m in by_route.get(t.route, []) if _matches(m, t, lead_sec=lead_sec)
        ]
        if covering:
            nearest = min(covering, key=lambda m: abs(m.onset - t.onset))
            latencies.append((nearest.onset - t.onset) / 60.0)
            detected += 1

    n = len(truth_eps)
    leading = [x for x in latencies if x < 0]
    return {
        "n_episodes": n,
        "n_detected": detected,
        "n_missed": n - detected,
        "detection_rate": detected / n if n else None,
        "median_latency_min": median(latencies) if latencies else None,
        "mean_latency_min": sum(latencies) / len(latencies) if latencies else None,
        "lead_tolerance_min": lead_sec // 60,
        "n_detected_leading": len(leading),
        "median_lead_min": -median(leading) if leading else None,
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
    lead_sec: int = ONSET_LEAD_TOLERANCE_SEC,
) -> dict[str, Any]:
    """Model episodes matching no truth episode, split by whether the movement
    state confirms (real incident the alert-truth missed) or contradicts (a
    genuine over-call) them. Movement now feeds the HMM, so this split is a
    self-consistency diagnostic, not an independent adjudication.

    Uses the same predicate as onset_latency, so an early call credited there as
    a lead is not also counted here as a false alarm."""
    by_route: dict[str, list[Episode]] = defaultdict(list)
    for t in truth_eps:
        by_route[t.route].append(t)

    fa = [
        m
        for m in model_eps
        if not any(_matches(m, t, lead_sec=lead_sec) for t in by_route.get(m.route, []))
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
        "movement_cross_check": "self_consistency (movement is an HMM input)",
    }


def episode_recovery(
    truth_eps: list[Episode],
    dwell_lookup: DwellLookup,
    *,
    graded_arm: str = SHADOW_ARM_LABEL,
    baseline_durations_min: Sequence[float] | None,
) -> dict[str, Any]:
    """Per-episode recovery CRPS/PIT over uncensored episodes with a dwell curve.
    The predicted curve is the model's recovery forecast for the episode's peak
    state and cause at onset (elapsed 0); the outcome is the realized duration.

    `graded_arm` tags which dwell-curve population fed `dwell_lookup` — the
    alert-shadow HMM regime by default (dwell_lookup_from_params /
    cause_dwell_lookup, existing callers), or MOVEMENT_ARM_LABEL when the
    caller passes movement_dwell_lookup_from_params.

    TWO BASELINES, ONE OF WHICH IS HINDSIGHT. The report's
    `oracle_baseline_crps` is the empirical CDF of THIS call's own `truth_eps`
    durations — the graded window grading itself. Every published recovery
    skill number predates the distinction and is measured against it, so it is
    kept under a name that says so. Grading the two arms via separate calls, as
    episode_scorecard does, still matters for it: movement's minutes-long
    episodes never dilute the alert shadow's hours-long baseline or vice versa.

    `baseline_durations_min` supplies the CAUSAL baseline — realized durations,
    in minutes, from a window that closes before this one. Only then does the
    report carry a `causal_skill`; passing None leaves that field None rather
    than letting the oracle number pass for "vs climatology". It has no default:
    every caller states which of the two it is, because the whole class of bug
    here is a caller that never noticed there was a choice.
    """
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
        curve_sec, tail_ll, atom = cell
        # A mixture puts no mass strictly below its atom, so an episode that
        # lands exactly on the atom sits on a jump from 0 up to atom_p. Handing
        # the grader that lower edge is what lets it spread these episodes across
        # the jump instead of stacking every one of them in a single PIT bin.
        pred_left = (
            0.0 if atom is not None and abs(e.duration_sec - atom[1]) < 1.0 else None
        )
        samples.append(
            RecoveryDistSample(
                pred_curve=predicted_recovery_curve(0.0, curve_sec, tail_ll, atom),
                actual_min=e.duration_sec / 60.0,
                regime_key=f"{e.route}:{e.onset}",
                pred_left=pred_left,
            )
        )
    report = recovery_dist_report(
        samples, baseline_durations_min=baseline_durations_min
    )
    return {
        "graded_arm": graded_arm,
        "n_scored": len(samples),
        "n_censored_excluded": n_censored,
        "n_no_curve": n_no_curve,
        "report": report_as_dict(report),
        "verdict": verdict_as_dict(recovery_verdict(report)),
    }


def _cell_curve(
    cell: Any,
) -> tuple[list[int], list[float] | None, tuple[float, float] | None] | None:
    """Extract (curve_sec, tail_ll, atom) from a dwell-cell dict, or None if
    unusable. `atom` is (atom_p, atom_sec) only when the cell carries both, so a
    params doc written before the mixture existed reads exactly as it did."""
    if not cell:
        return None
    curve: list[int] = cell.get("curve_sec") or []
    if len(curve) < 2:
        return None
    tail: list[float] | None = cell.get("tail_ll")
    atom_p = cell.get("atom_p")
    atom_sec = cell.get("atom_sec")
    atom = (
        (float(atom_p), float(atom_sec))
        if atom_p is not None and atom_sec is not None
        else None
    )
    return curve, tail, atom


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
    ) -> tuple[list[int], list[float] | None, tuple[float, float] | None] | None:
        route_doc: dict[str, Any] = routes.get(route) or {}
        by_cause: dict[str, Any] = route_doc.get("dwell_quantiles_by_cause") or {}
        state_causes: dict[str, Any] = by_cause.get(state) or {}
        cell = state_causes.get(cause)
        if not cell:
            quantiles: dict[str, Any] = route_doc.get("dwell_quantiles") or {}
            cell = quantiles.get(state)
        return _cell_curve(cell)

    return lookup


def movement_dwell_lookup_from_params(params: dict[str, Any]) -> DwellLookup:
    """(route, state) -> (curve_sec, tail_ll) lookup over the movement-regime
    dwell block (params['dwell_movement'][route][state], contract C2). Route
    scope only: the movement clock carries no cause dimension, so `cause` is
    accepted only for DwellLookup shape compatibility and ignored. Absent
    block, absent route, absent state, or a too-short curve all read as None —
    same n_no_curve convention as dwell_lookup_from_params, not a crash."""
    routes: dict[str, Any] = params.get("dwell_movement") or {}

    def lookup(
        route: str, state: str, _cause: str
    ) -> tuple[list[int], list[float] | None, tuple[float, float] | None] | None:
        route_doc: dict[str, Any] = routes.get(route) or {}
        return _cell_curve(route_doc.get(state))

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
    ) -> tuple[list[int], list[float] | None, tuple[float, float] | None] | None:
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
    movement_dwell_lookup: DwellLookup | None = None,
) -> dict[str, Any]:
    """Assemble the event-based scorecard: onset latency, per-episode recovery,
    and false-alarm episodes, each with its event count.

    Standing advisories (Episode.standing) are held out of every metric: they are
    a property of how long the MTA leaves an alert posted, not of the incident,
    and their tick mass otherwise swamps the acute events the model exists to
    call. They stay in `n_truth_episodes` and get their own count so the holdout
    is visible rather than silent. A model episode overlapping only a standing
    advisory is likewise not scored as a false alarm — the alert feed says
    something is wrong there, so it is not evidence of an over-call.

    `movement_dwell_lookup`, when given, grades a second recovery arm under
    `recovery_movement`: movement itself is re-segmented into its own episodes
    from `movement_truth` (movement is the truth here, not the cross-check
    false_alarms uses it for) and scored against movement dwell curves, tagged
    MOVEMENT_ARM_LABEL. This runs alongside, never instead of, the alert-shadow
    `recovery` block — each arm keeps its own CRPS baseline (episode_recovery),
    so a movement episode's minutes are never averaged against the alert
    shadow's hours. Omitted (the default) reproduces the pre-movement-arm
    payload shape exactly — no `recovery_movement` key at all.

    Neither arm gets a CAUSAL baseline here: this scorecard is handed exactly
    one window of truth episodes, and a climatology fitted before it needs a
    pre-window episode population the caller does not load. Both arms therefore
    publish `causal_skill: null` and an `oracle_skill` that is explicitly
    hindsight-relative. Callers that DO hold a train/eval split (backtest's
    grade_recovery_timing) pass `baseline_durations_min` to episode_recovery
    directly and get the honest column.
    """
    model_eps = model_episodes(
        predictions, window_start=window_start, window_end=window_end
    )
    graded = [t for t in truth_eps if not t.standing]
    standing = [t for t in truth_eps if t.standing]
    # Bare overlap, not the detection predicate: the lead tolerance exists to
    # credit an early call against a truth onset, and reusing it here would
    # excuse a model episode that closed before the advisory ever went up.
    gradeable_model_eps = [
        m for m in model_eps if not any(_matches(m, s, lead_sec=0) for s in standing)
    ]
    card: dict[str, Any] = {
        "n_truth_episodes": len(truth_eps),
        "n_model_episodes": len(model_eps),
        "n_standing_excluded": len(standing),
        "n_model_episodes_in_standing": len(model_eps) - len(gradeable_model_eps),
        "graded_arm": MOVEMENT_ARM_LABEL,
        "published_coverage": published_condition_coverage(predictions),
        "onset_latency": onset_latency(graded, model_eps),
        # None: this scorecard holds one window of episodes and no pre-window
        # population, so neither arm can be given a causal climatology. See the
        # note above — the omission is published, not papered over.
        "recovery": episode_recovery(
            graded,
            dwell_lookup,
            graded_arm=SHADOW_ARM_LABEL,
            baseline_durations_min=None,
        ),
        "false_alarms": false_alarms(gradeable_model_eps, graded, movement_truth),
    }
    if movement_dwell_lookup is not None:
        movement_eps = extract_episodes(
            movement_truth, {}, window_start=window_start, window_end=window_end
        )
        movement_graded = [e for e in movement_eps if not e.standing]
        card["recovery_movement"] = episode_recovery(
            movement_graded,
            movement_dwell_lookup,
            graded_arm=MOVEMENT_ARM_LABEL,
            baseline_durations_min=None,
        )
    return card
