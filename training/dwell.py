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
from collections.abc import Callable, Mapping, Sequence
from typing import NotRequired, Protocol, TypedDict

from momentarily.mapping import category_for_label, coarse_status
from training.eval import TransitionRecord

# A (route, state) cell needs at least this many transitions to back an
# empirical quantile estimate. Below the floor the Worker falls back to the
# geometric dwell from the trained transition self-loop.
MIN_SAMPLES_FOR_EMPIRICAL = 5

# Resolution of curve_sec: dwell quantiles at probabilities 0, 1/(K-1), ..., 1.
# 21 points = 5% steps; fine enough for interpolation, small enough that the
# params.json sidecar stays compact.
CURVE_POINTS = 21

# Route -> (state, regime_entered_at) for regimes still open at the censoring
# boundary. The prediction stream carries this for every live route each tick,
# including routes that never transitioned and so have no transition record.
OpenRegimes = Mapping[str, tuple[str, int]]


class RegimeTransition(Protocol):
    """Structural shape `dwell_samples_by_cell` needs off a transition record.

    `TransitionRecord` (alert regime) and `MovementTransitionRecord` (movement
    regime, route or segment scope) both satisfy this without a cast — one
    grouping function feeds both streams instead of a parallel movement-only
    copy. Alert-only fields (e.g. `alert_type_at_entry`) stay off this
    protocol; callers that need them keep the concrete `TransitionRecord` type.
    """

    @property
    def route(self) -> str: ...
    @property
    def prev_state(self) -> str: ...
    @property
    def new_state(self) -> str: ...
    @property
    def ts(self) -> int: ...
    @property
    def exited_at(self) -> int: ...
    @property
    def dwell_sec(self) -> int: ...


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
    is far below P(dwell <= 30min).
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
    # [shape, scale] of a log-logistic fit to this cell's censored dwells, used by
    # p_leave_by to extrapolate the tail past the last observed quantile instead
    # of the coarse constant-hazard exponential patch. Absent when no fit
    # converged (no completed events).
    tail_ll: NotRequired[list[float]]
    # Point mass ("atom") at exactly one tick, mixed with the tail_ll body. Both
    # fields are present together or not at all: a cell either carries the whole
    # mixture or none of it, so no half-configured state is representable.
    #
    # 70.4% of disrupted movement episodes last exactly one publisher tick. A
    # single continuous curve cannot front-load that much mass and still reserve
    # a tail for the multi-tick minority — it ends up too pessimistic on the
    # one-tick majority and too optimistic on everything else, which is what the
    # bimodal PIT histogram was reporting. Splitting the two populations apart
    # costs one parameter and fixes both lobes.
    #
    # When these are present, tail_ll is the log-logistic LEFT-TRUNCATED at
    # atom_sec (fitted on T > atom_sec alone, so the spike and the tail do not
    # double-count the same episodes) and the closed form in mixture_survival
    # fully replaces curve_sec. A reader that honours tail_ll while ignoring
    # atom_p computes the wrong distribution, so every consumer moves together.
    atom_p: NotRequired[float]  # P(dwell == atom_sec), strictly in (0, 1)
    atom_sec: NotRequired[int]  # location of the point mass, seconds


# One observation for the estimator: (duration_sec, completed). completed=False
# means the regime was still running at the observation boundary (right-
# censored) — we know dwell > duration, not its value.
DwellSample = tuple[int, bool]

# Fits a [shape, scale] log-logistic tail to a cell's samples, or None if it
# can't. Injected (rather than imported) so dwell.py stays free of survival.py,
# which depends on this module.
TailFn = Callable[[list["DwellSample"]], "list[float] | None"]


def km_cdf_points(samples: list[DwellSample]) -> list[tuple[int, float]]:
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


def _make_cell(
    samples: list[DwellSample], tail_fn: TailFn | None = None
) -> DwellQuantiles:
    """Build a DwellQuantiles from (duration, completed) samples via
    Kaplan-Meier, so right-censored (still-running) regimes push the tail up
    instead of silently vanishing.

    With `tail_fn`, a log-logistic [shape, scale] tail is fit to the same
    samples and stored on the cell for the Worker's past-the-curve splice.
    """
    points = km_cdf_points(samples)
    n_events = sum(1 for _d, completed in samples if completed)
    n_censored = len(samples) - n_events
    max_duration = max(d for d, _completed in samples)
    cell = DwellQuantiles(
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
    if tail_fn is not None:
        tail = tail_fn(samples)
        if tail is not None:
            cell["tail_ll"] = tail
    return cell


def _open_regimes(
    transitions: Sequence[RegimeTransition], window_end: int
) -> dict[tuple[str, str], int]:
    """Right-censored observations inferred from transition records: each route's
    final regime (the new_state of its last transition) is still running at
    window_end — we know its dwell exceeds window_end − exited_at. Returns
    {(route, state): censored_duration}.

    Only the final regime per route is open; every earlier regime is fully
    described by the next transition's prev_state record.

    Blind to routes that never transitioned inside the window: with no record to
    read a regime off, they contribute neither an event nor a censored
    observation. Since a route completes a `normal` regime only by leaving
    normal, that blind spot covers exactly the steadiest routes. Pass
    `open_regimes` to dwell_samples_by_cell to source them from the prediction
    stream instead, which carries every live route every tick.
    """
    last_by_route: dict[str, RegimeTransition] = {}
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


def _censored_from_open(
    open_regimes: OpenRegimes, window_end: int
) -> dict[tuple[str, str], int]:
    """Right-censored observations from explicit open-regime facts, for callers
    that can observe a route's current regime directly rather than inferring it
    from the last transition."""
    out: dict[tuple[str, str], int] = {}
    for route, (state, entered_at) in open_regimes.items():
        duration = window_end - entered_at
        if duration > 0:
            out[(route, state)] = duration
    return out


# --- Conditional survival math (reference implementation) ---
#
# The Worker mirrors these in worker/src/dwell.ts; keep the two in sync. All
# functions treat `curve_sec` as the dwell CDF sampled at evenly spaced
# probabilities, linearly interpolated between points.


def dwell_cdf(curve_sec: list[int], x: float) -> float:
    """Empirical P(dwell <= x) from the quantile curve, interpolated.

    At a repeated knot the answer is the TOP of the flat run, not the bottom.
    `curve_sec` is a quantile function, so a run of equal knots is a point mass,
    and P(dwell <= x) at that value has to include the whole mass. Taking the
    bottom instead is how a one-tick episode used to read P=0: the lower guard
    was inclusive and curve_sec[0] is exactly one tick for any cell whose
    shortest dwell is one tick, so the CDF returned 0 evaluated at its own grid
    point and every one-tick episode graded with PIT=0.

    For a strictly increasing curve this is identical to plain interpolation —
    the flat-run case is the only behaviour that changes.
    """
    k = len(curve_sec)
    # Upper bound first so a degenerate flat curve (all samples equal) reads
    # as "outlived" at x == that value, not as P=0.
    if x >= curve_sec[-1]:
        return 1.0
    if x < curve_sec[0]:
        return 0.0
    # Largest index at or below x. Scanning forward past equal knots is what
    # lands on the top of a flat run.
    j = 0
    for i in range(k):
        if curve_sec[i] <= x:
            j = i
        else:
            break
    if j >= k - 1:
        return 1.0
    if curve_sec[j] == x:
        return j / (k - 1)
    span = curve_sec[j + 1] - curve_sec[j]
    frac = 0.0 if span == 0 else (x - curve_sec[j]) / span
    return (j + frac) / (k - 1)


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


def p_leave_by(
    curve_sec: list[int],
    elapsed_sec: float,
    horizon_sec: float,
    tail_ll: list[float] | None = None,
    atom: tuple[float, float] | None = None,
) -> float:
    """P(dwell <= elapsed + horizon | dwell > elapsed), extrapolating a tail once
    the regime has outlived every observed dwell instead of saturating at the
    curve max. Unlike conditional_recover_by (which returns None past the curve,
    for a recovery *time* we won't fabricate), this keeps the conditional exit
    *probability* meaningful in the long-lived tail.

    With `atom` ((atom_p, atom_sec)) the cell publishes a mixture and the whole
    answer is closed-form: curve_sec is not consulted at all, and neither is the
    past-the-curve splice below, because a parametric mixture has no "past the
    curve". The curve path is what runs for every cell without an atom.

    Past the curve the tail is the fitted log-logistic conditional survival when
    `tail_ll` ([shape, scale]) is supplied, else a constant-hazard exponential
    patch read off the top segment. The log-logistic's decreasing hazard models
    the heavy dwell tail better — a long-calm regime stays confident rather than
    being told it's about to leave (per the Brier backtest). The body stays
    empirical either way. Mirrored in worker/src/dwell.ts; keep in sync."""
    mix = _atom_params(tail_ll, atom)
    if mix is not None:
        shape, scale, atom_p, atom_sec = mix
        s_now = mixture_survival(elapsed_sec, shape, scale, atom_p, atom_sec)
        if s_now <= 0.0:
            return 1.0
        s_fut = mixture_survival(
            elapsed_sec + horizon_sec, shape, scale, atom_p, atom_sec
        )
        return max(0.0, min(1.0, 1.0 - s_fut / s_now))
    k = len(curve_sec)
    if k < 2:
        return 0.0
    p_elapsed = dwell_cdf(curve_sec, elapsed_sec)
    if p_elapsed < 1.0:
        return (dwell_cdf(curve_sec, elapsed_sec + horizon_sec) - p_elapsed) / (
            1.0 - p_elapsed
        )
    if tail_ll is not None:
        shape, scale = tail_ll
        s_now = _loglogistic_survival(elapsed_sec, shape, scale)
        if s_now <= 0.0:
            return 1.0
        s_fut = _loglogistic_survival(elapsed_sec + horizon_sec, shape, scale)
        return max(0.0, min(1.0, 1.0 - s_fut / s_now))
    # Outlived the curve: constant tail hazard from the top segment (the top
    # 1/(k-1) of mass is lost over its width), projected across the horizon.
    seg = curve_sec[-1] - curve_sec[-2]
    lam = (1.0 / (k - 1)) / seg if seg > 0 else 1.0 / max(1.0, float(curve_sec[-1]))
    return 1.0 - math.exp(-max(lam, 1e-12) * horizon_sec)


def _loglogistic_survival(t: float, shape: float, scale: float) -> float:
    """S(t) = 1 / (1 + (t/scale)^shape). Inlined (not imported from survival.py)
    to keep the conditional-survival math free of that module's import cycle."""
    if t <= 0.0 or scale <= 0.0:
        return 1.0
    return 1.0 / (1.0 + (t / scale) ** shape)


# --- Atom + truncated log-logistic mixture ---
#
# A dwell cell carrying (atom_p, atom_sec) publishes a point mass at atom_sec
# mixed with a log-logistic left-truncated there:
#
#   S(t) = 1                                       t <  atom_sec
#   S(t) = (1 - atom_p) * S_ll(t) / S_ll(atom_sec) t >= atom_sec
#
# so F(atom_sec) == atom_p exactly. That exactness is the whole fix: the
# quantile-curve representation can only reach the mass as F(atom_sec + eps),
# never at the grid point itself, which is what graded the one-tick majority at
# PIT=0. The closed form has no such boundary.
#
# One useful property, worth keeping in mind when reading the call sites: for
# elapsed >= atom_sec the atom cancels out of the conditional and the answer
# reduces to the plain log-logistic 1 - S_ll(e+h)/S_ll(e). The mixture only
# moves anything inside the first tick — which is exactly where a forecast made
# at regime onset lives, and where the old fit was worst.


def mixture_survival(
    t: float, shape: float, scale: float, atom_p: float, atom_sec: float
) -> float:
    """S(t) for the atom + left-truncated log-logistic mixture."""
    if t < atom_sec:
        return 1.0
    s_tau = _loglogistic_survival(atom_sec, shape, scale)
    if s_tau <= 0.0:
        return 0.0
    return max(
        0.0, min(1.0, (1.0 - atom_p) * _loglogistic_survival(t, shape, scale) / s_tau)
    )


def mixture_quantile(
    u: float, shape: float, scale: float, atom_p: float, atom_sec: float
) -> float:
    """Inverse CDF of the mixture: smallest t with F(t) >= u.

    Flat at atom_sec for every u up to atom_p — the atom is an interval of the
    quantile function, not a point, which is why curve_sec renders it as a run
    of equal knots.
    """
    u = min(max(u, 0.0), 1.0 - 1e-12)
    if u <= atom_p:
        return atom_sec
    if shape <= 0.0 or scale <= 0.0:
        return atom_sec
    s_tau = _loglogistic_survival(atom_sec, shape, scale)
    s_target = (1.0 - u) * s_tau / (1.0 - atom_p)
    if s_target <= 0.0:
        return math.inf
    if s_target >= 1.0:
        return atom_sec
    return scale * ((1.0 - s_target) / s_target) ** (1.0 / shape)


def _atom_params(
    tail_ll: list[float] | None, atom: tuple[float, float] | None
) -> tuple[float, float, float, float] | None:
    """(shape, scale, atom_p, atom_sec) when a cell carries a usable mixture.

    The mixture needs BOTH a tail and an atom: tail_ll alone is the legacy
    unconditional fit, and an atom without a tail has nothing to spend its
    remaining mass on. Anything partial falls back to the curve path rather than
    inventing a component.
    """
    if atom is None or tail_ll is None or len(tail_ll) < 2:
        return None
    atom_p, atom_sec = atom
    if not (0.0 < atom_p < 1.0) or atom_sec <= 0.0:
        return None
    shape, scale = tail_ll[0], tail_ll[1]
    if shape <= 0.0 or scale <= 0.0:
        return None
    return shape, scale, atom_p, atom_sec


def conditional_remaining_quantile(
    curve_sec: list[int],
    elapsed_sec: float,
    q: float,
    tail_ll: list[float] | None = None,
    atom: tuple[float, float] | None = None,
) -> float | None:
    """q-th quantile of remaining dwell given the regime survived elapsed_sec.

    Solves P(dwell <= t | dwell > elapsed) = q for t, returns t − elapsed.
    None when elapsed exceeds every observed dwell (see conditional_recover_by).

    With an atom the median remaining time at onset collapses to a single tick
    whenever atom_p > 0.5 — which is the honest answer for a population where
    most disruptions clear on the next poll, and one a continuous fit can only
    approximate by pulling its whole body down.
    """
    mix = _atom_params(tail_ll, atom)
    if mix is not None:
        shape, scale, atom_p, atom_sec = mix
        f_elapsed = 1.0 - mixture_survival(elapsed_sec, shape, scale, atom_p, atom_sec)
        if f_elapsed >= 1.0:
            return None
        total = mixture_quantile(
            f_elapsed + q * (1.0 - f_elapsed), shape, scale, atom_p, atom_sec
        )
        if not math.isfinite(total):
            return None
        return max(0.0, total - elapsed_sec)
    p_elapsed = dwell_cdf(curve_sec, elapsed_sec)
    if p_elapsed >= 1.0:
        return None
    total = _dwell_quantile(curve_sec, p_elapsed + q * (1.0 - p_elapsed))
    return max(0.0, total - elapsed_sec)


def dwell_samples_by_cell(
    transitions: Sequence[RegimeTransition],
    *,
    window_end: int | None = None,
    open_regimes: OpenRegimes | None = None,
) -> dict[tuple[str, str], list[DwellSample]]:
    """Group transitions into per-(route, state) dwell samples. Each completed
    transition contributes a (dwell_sec, True) event; with `window_end`, each
    route's still-open regime joins its cell as a right-censored (duration,
    False) observation (Kaplan-Meier). The raw samples backing every cell, so a
    parametric fit and the empirical curve see identical data.

    `open_regimes` supersedes the transition-derived guess at what is still
    running. It is strictly more complete — the prediction stream observes every
    live route, while a transition record only exists for routes that moved — so
    routes that held one regime for the whole window contribute a censored
    observation instead of nothing at all.
    """
    by_cell: dict[tuple[str, str], list[DwellSample]] = defaultdict(list)
    for t in transitions:
        by_cell[(t.route, t.prev_state)].append((int(t.dwell_sec), True))
    if window_end is not None:
        censored = (
            _censored_from_open(open_regimes, window_end)
            if open_regimes is not None
            else _open_regimes(transitions, window_end)
        )
        for (route, state), duration in censored.items():
            by_cell[(route, state)].append((duration, False))
    return by_cell


def compute_dwell_quantiles(
    transitions: list[TransitionRecord],
    *,
    min_samples: int = MIN_SAMPLES_FOR_EMPIRICAL,
    window_end: int | None = None,
    tail_fn: TailFn | None = None,
    open_regimes: OpenRegimes | None = None,
) -> dict[str, dict[str, DwellQuantiles]]:
    """Return {route: {state: DwellQuantiles}} for each (route, prev_state)
    with at least `min_samples` completed transitions. Sparser cells are
    omitted — the consumer should fall back to its analytic estimate.

    With `window_end`, each route's still-open regime joins its cell as a
    right-censored observation (Kaplan-Meier), so a marathon regime in progress
    pushes the tail up instead of being invisible until it ends. `open_regimes`
    sources those still-open regimes from the prediction stream rather than
    inferring them from the last transition. With `tail_fn`, each cell carries a
    log-logistic tail for the Worker's past-the-curve splice.
    """
    by_cell = dwell_samples_by_cell(
        transitions, window_end=window_end, open_regimes=open_regimes
    )
    out: dict[str, dict[str, DwellQuantiles]] = defaultdict(dict)
    for (route, state), samples in by_cell.items():
        if sum(1 for _d, completed in samples if completed) < min_samples:
            continue
        out[route][state] = _make_cell(samples, tail_fn)
    return dict(out)


def compute_dwell_quantiles_by_alert(
    transitions: list[TransitionRecord],
    *,
    min_samples: int = MIN_SAMPLES_FOR_EMPIRICAL,
    tail_fn: TailFn | None = None,
) -> dict[str, dict[str, dict[str, DwellQuantiles]]]:
    """Return {route: {state: {alert_type: DwellQuantiles}}} for each
    (route, prev_state, alert_type_at_entry) cell with at least `min_samples`
    transitions.

    Transitions with no alert_type_at_entry (older records, or regimes that
    began with no active alert) are skipped — they're already represented in
    the (route, state) aggregate from `compute_dwell_quantiles`, which the
    consumer falls back to when a (route, state, alert_type) cell is absent.
    This is the recovery-by-cause segmentation: a route's
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
        out[route][state][alert_type] = _make_cell(samples, tail_fn)
    return {
        r: {s: dict(by_at) for s, by_at in by_state.items()}
        for r, by_state in out.items()
    }


def cause_of(alert_type: str | None) -> str | None:
    """Cause-category key for a dwell cell — the same coarse cause the episode
    grader attributes (category_for_label(coarse_status(alert_type))). None when
    there is no alert_type to key on."""
    if alert_type is None:
        return None
    return category_for_label(coarse_status(alert_type))


def compute_dwell_quantiles_by_cause(
    transitions: list[TransitionRecord],
    *,
    min_samples: int = MIN_SAMPLES_FOR_EMPIRICAL,
    tail_fn: TailFn | None = None,
) -> dict[str, dict[str, dict[str, DwellQuantiles]]]:
    """Return {route: {state: {cause: DwellQuantiles}}} keyed on the cause
    CATEGORY of alert_type_at_entry, matching the episode grader's cause so
    per-episode recovery can look a curve up by the same key. Like
    compute_dwell_quantiles_by_alert but grouped by coarse cause rather than raw
    alert_type; transitions with no alert_type_at_entry are skipped (they live in
    the (route, state) aggregate the consumer falls back to). Completed
    transitions only — an open final regime has no known cause and is censored
    into the (route, state) aggregate instead."""
    by_cell: dict[tuple[str, str, str], list[DwellSample]] = defaultdict(list)
    for t in transitions:
        cause = cause_of(t.alert_type_at_entry)
        if cause is None:
            continue
        by_cell[(t.route, t.prev_state, cause)].append((int(t.dwell_sec), True))
    out: dict[str, dict[str, dict[str, DwellQuantiles]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (route, state, cause), samples in by_cell.items():
        if len(samples) < min_samples:
            continue
        out[route][state][cause] = _make_cell(samples, tail_fn)
    return {
        r: {s: dict(by_cause) for s, by_cause in by_state.items()}
        for r, by_state in out.items()
    }
