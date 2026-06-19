"""Empirical dwell-time quantiles per (route, state) from the regime_transitions stream.

The Worker's `recovery_minutes` was geometric — derived from the trained
transition self-loop, which can't represent bimodal dwell distributions and
saturates for any high self-loop (a route with sustained planned-work alerts
spends hours in one regime). Replacing it with the empirical distribution of
how long each route actually stays in each non-normal state typically slashes
MAE by an order of magnitude, since regime durations are heavy-tailed and the
geometric model under-represents the body.

Returns one quantile triple per (route, state) — Worker uses these as the
recovery_minutes_low/median/high bounds whenever sample size crosses the
floor; otherwise it falls back to the geometric estimate.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TypedDict

from training.eval import TransitionRecord

# A (route, state) cell needs at least this many transitions to back an
# empirical quantile estimate. Below the floor the Worker falls back to the
# geometric dwell from the trained transition self-loop.
MIN_SAMPLES_FOR_EMPIRICAL = 5

# Resolution of curve_sec: dwell quantiles at probabilities 0, 1/(K-1), ..., 1.
# 21 points = 5% steps; fine enough for interpolation, small enough that the
# params.json sidecar stays compact.
CURVE_POINTS = 21


class DwellQuantiles(TypedDict):
    """Empirical dwell-duration summary for a (route, state[, alert_type]) cell.

    Quantiles back recovery_minutes_low/median/high. The recover_by_* fractions
    are the empirical P(dwell <= horizon) — the Worker uses them for the
    p_normal_in_30/60/120min projection, which is otherwise a geometric estimate
    from the transition self-loop that can't represent the heavy-tailed,
    cause-dependent recovery curve (delays clear fast, planned work lingers).

    `curve_sec` is the full dwell distribution as quantiles at CURVE_POINTS
    evenly spaced probabilities. The Worker uses it to condition every recovery
    output on how long the regime has *already* lasted — the unconditional
    quantiles/fractions above are only correct at elapsed=0, and for a
    heavy-tailed dwell distribution P(recover in 30min | disrupted 3h already)
    is far below P(dwell <= 30min). See momentarily-vk0.1.
    """

    n: int  # completed (event) observations — the min-samples floor keys on this
    n_censored: int  # right-censored (still-running at window end) observations
    q25_sec: int
    median_sec: int
    q75_sec: int
    recover_by_30: float
    recover_by_60: float
    recover_by_120: float
    curve_sec: list[int]


# One observation for the estimator: (duration_sec, completed). completed=False
# means the regime was still running at the observation boundary (right-
# censored) — we know dwell > duration, not its value.
DwellSample = tuple[int, bool]


def _km_cdf_points(samples: list[DwellSample]) -> list[tuple[int, float]]:
    """Kaplan-Meier product-limit CDF: [(event_time, F(event_time))], ascending.

    Censored observations reduce the at-risk count without registering an
    event; ties between a censored mark and an event at the same time follow
    the standard convention (censored stays at risk through the event). With
    no censoring this reduces exactly to the empirical CDF k/n.
    """
    ordered = sorted(samples)
    n_total = len(ordered)
    points: list[tuple[int, float]] = []
    survival = 1.0
    at_risk = n_total
    i = 0
    while i < n_total:
        t = ordered[i][0]
        deaths = 0
        ties = 0
        while i < n_total and ordered[i][0] == t:
            if ordered[i][1]:
                deaths += 1
            ties += 1
            i += 1
        if deaths > 0:
            survival *= 1.0 - deaths / at_risk
            points.append((t, 1.0 - survival))
        at_risk -= ties
    return points


def _km_quantile(points: list[tuple[int, float]], q: float, max_duration: int) -> int:
    """Smallest event time t with F(t) >= q. Under heavy censoring the KM CDF
    may never reach q; clamp to the largest observed duration (censored or
    not) — biased low, but bounded and still >= every completed dwell."""
    for t, f in points:
        if f >= q - 1e-12:
            return t
    return max_duration


def _km_cdf_at(points: list[tuple[int, float]], horizon: int) -> float:
    """F(horizon): the last KM step at or below the horizon."""
    out = 0.0
    for t, f in points:
        if t > horizon:
            break
        out = f
    return out


def _make_cell(samples: list[DwellSample]) -> DwellQuantiles:
    """Build a DwellQuantiles from (duration, completed) samples via
    Kaplan-Meier, so right-censored (still-running) regimes push the tail up
    instead of silently vanishing. See momentarily-vk0.6."""
    points = _km_cdf_points(samples)
    n_events = sum(1 for _d, completed in samples if completed)
    n_censored = len(samples) - n_events
    max_duration = max(d for d, _completed in samples)
    return DwellQuantiles(
        n=n_events,
        n_censored=n_censored,
        q25_sec=_km_quantile(points, 0.25, max_duration),
        median_sec=_km_quantile(points, 0.50, max_duration),
        q75_sec=_km_quantile(points, 0.75, max_duration),
        recover_by_30=_km_cdf_at(points, 1800),
        recover_by_60=_km_cdf_at(points, 3600),
        recover_by_120=_km_cdf_at(points, 7200),
        curve_sec=[
            _km_quantile(points, i / (CURVE_POINTS - 1), max_duration)
            for i in range(CURVE_POINTS)
        ],
    )


def _open_regimes(
    transitions: list[TransitionRecord], window_end: int
) -> dict[tuple[str, str], int]:
    """Right-censored observations: each route's final regime (the new_state of
    its last transition) is still running at window_end — we know its dwell
    exceeds window_end − exited_at. Returns {(route, state): censored_duration}.

    Only the final regime per route is open; every earlier regime is fully
    described by the next transition's prev_state record.
    """
    last_by_route: dict[str, TransitionRecord] = {}
    for t in transitions:
        prev = last_by_route.get(t.route)
        if prev is None or t.ts > prev.ts:
            last_by_route[t.route] = t
    out: dict[tuple[str, str], int] = {}
    for route, t in last_by_route.items():
        duration = window_end - t.exited_at
        if duration > 0:
            out[(route, t.new_state)] = duration
    return out


# --- Conditional survival math (reference implementation) ---
#
# The Worker mirrors these in worker/src/dwell.ts; keep the two in sync. All
# functions treat `curve_sec` as the dwell CDF sampled at evenly spaced
# probabilities, linearly interpolated between points.


def dwell_cdf(curve_sec: list[int], x: float) -> float:
    """Empirical P(dwell <= x) from the quantile curve, interpolated."""
    k = len(curve_sec)
    # Upper bound first so a degenerate flat curve (all samples equal) reads
    # as "outlived" at x == that value, not as P=0.
    if x >= curve_sec[-1]:
        return 1.0
    if x <= curve_sec[0]:
        return 0.0
    for i in range(k - 1):
        lo, hi = curve_sec[i], curve_sec[i + 1]
        if lo <= x <= hi:
            frac = 0.0 if hi == lo else (x - lo) / (hi - lo)
            return (i + frac) / (k - 1)
    return 1.0  # unreachable for a monotone curve


def _dwell_quantile(curve_sec: list[int], p: float) -> float:
    """Inverse of dwell_cdf: dwell duration at cumulative probability p."""
    k = len(curve_sec)
    pos = min(max(p, 0.0), 1.0) * (k - 1)
    i = min(int(pos), k - 2)
    frac = pos - i
    return curve_sec[i] + frac * (curve_sec[i + 1] - curve_sec[i])


def conditional_recover_by(
    curve_sec: list[int], elapsed_sec: float, horizon_sec: float
) -> float | None:
    """P(dwell <= elapsed + horizon | dwell > elapsed).

    None when the regime has outlived every observed dwell — the empirical
    distribution says nothing about it and the caller should mark the
    prediction indeterminate rather than fabricate a number.
    """
    p_elapsed = dwell_cdf(curve_sec, elapsed_sec)
    if p_elapsed >= 1.0:
        return None
    p_horizon = dwell_cdf(curve_sec, elapsed_sec + horizon_sec)
    return (p_horizon - p_elapsed) / (1.0 - p_elapsed)


def p_leave_by(curve_sec: list[int], elapsed_sec: float, horizon_sec: float) -> float:
    """P(dwell <= elapsed + horizon | dwell > elapsed), extrapolating an
    exponential tail once the regime has outlived every observed dwell instead of
    saturating at the curve max. Unlike conditional_recover_by (which returns None
    past the curve, for a recovery *time* we won't fabricate), this keeps the
    conditional exit *probability* meaningful in the long-lived tail. Mirrored in
    worker/src/dwell.ts; keep in sync."""
    k = len(curve_sec)
    if k < 2:
        return 0.0
    p_elapsed = dwell_cdf(curve_sec, elapsed_sec)
    if p_elapsed < 1.0:
        return (dwell_cdf(curve_sec, elapsed_sec + horizon_sec) - p_elapsed) / (
            1.0 - p_elapsed
        )
    # Outlived the curve: constant tail hazard from the top segment (the top
    # 1/(k-1) of mass is lost over its width), projected across the horizon.
    seg = curve_sec[-1] - curve_sec[-2]
    lam = (1.0 / (k - 1)) / seg if seg > 0 else 1.0 / max(1.0, float(curve_sec[-1]))
    return 1.0 - math.exp(-max(lam, 1e-12) * horizon_sec)


def conditional_remaining_quantile(
    curve_sec: list[int], elapsed_sec: float, q: float
) -> float | None:
    """q-th quantile of remaining dwell given the regime survived elapsed_sec.

    Solves P(dwell <= t | dwell > elapsed) = q for t, returns t − elapsed.
    None when elapsed exceeds every observed dwell (see conditional_recover_by).
    """
    p_elapsed = dwell_cdf(curve_sec, elapsed_sec)
    if p_elapsed >= 1.0:
        return None
    total = _dwell_quantile(curve_sec, p_elapsed + q * (1.0 - p_elapsed))
    return max(0.0, total - elapsed_sec)


def compute_dwell_quantiles(
    transitions: list[TransitionRecord],
    *,
    min_samples: int = MIN_SAMPLES_FOR_EMPIRICAL,
    window_end: int | None = None,
) -> dict[str, dict[str, DwellQuantiles]]:
    """Return {route: {state: DwellQuantiles}} for each (route, prev_state)
    with at least `min_samples` completed transitions. Sparser cells are
    omitted — the consumer should fall back to its analytic estimate.

    With `window_end`, each route's still-open final regime joins its cell as
    a right-censored observation (Kaplan-Meier), so a marathon regime in
    progress pushes the tail up instead of being invisible until it ends.
    See momentarily-vk0.6.
    """
    by_cell: dict[tuple[str, str], list[DwellSample]] = defaultdict(list)
    for t in transitions:
        by_cell[(t.route, t.prev_state)].append((int(t.dwell_sec), True))
    if window_end is not None:
        for (route, state), duration in _open_regimes(transitions, window_end).items():
            by_cell[(route, state)].append((duration, False))

    out: dict[str, dict[str, DwellQuantiles]] = defaultdict(dict)
    for (route, state), samples in by_cell.items():
        if sum(1 for _d, completed in samples if completed) < min_samples:
            continue
        out[route][state] = _make_cell(samples)
    return dict(out)


def compute_dwell_quantiles_by_alert(
    transitions: list[TransitionRecord],
    *,
    min_samples: int = MIN_SAMPLES_FOR_EMPIRICAL,
) -> dict[str, dict[str, dict[str, DwellQuantiles]]]:
    """Return {route: {state: {alert_type: DwellQuantiles}}} for each
    (route, prev_state, alert_type_at_entry) cell with at least `min_samples`
    transitions.

    Transitions with no alert_type_at_entry (older records, or regimes that
    began with no active alert) are skipped — they're already represented in
    the (route, state) aggregate from `compute_dwell_quantiles`, which the
    consumer falls back to when a (route, state, alert_type) cell is absent.
    This is the recovery-by-cause segmentation (momentarily-alu): a route's
    dwell under "Planned - Stops Skipped" is structurally different from the
    same route under "Delays", so conditioning on the cause tightens the
    recovery interval.

    No censored observations here: transition records only carry the
    alert_type for the *completed* (prev_state) regime, so a route's open
    final regime has no known cause. It is censored into the (route, state)
    aggregate instead — the consumer's fallback when a cause cell is absent.
    """
    by_cell: dict[tuple[str, str, str], list[DwellSample]] = defaultdict(list)
    for t in transitions:
        if t.alert_type_at_entry is None:
            continue
        by_cell[(t.route, t.prev_state, t.alert_type_at_entry)].append(
            (int(t.dwell_sec), True)
        )

    out: dict[str, dict[str, dict[str, DwellQuantiles]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (route, state, alert_type), samples in by_cell.items():
        if len(samples) < min_samples:
            continue
        out[route][state][alert_type] = _make_cell(samples)
    return {
        r: {s: dict(by_at) for s, by_at in by_state.items()}
        for r, by_state in out.items()
    }
