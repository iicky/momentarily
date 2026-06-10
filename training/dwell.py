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

    n: int
    q25_sec: int
    median_sec: int
    q75_sec: int
    recover_by_30: float
    recover_by_60: float
    recover_by_120: float
    curve_sec: list[int]


def _quantile(sorted_sec: list[int], q: float) -> int:
    """Sample quantile via the nearest-rank rule. sorted_sec must be non-empty."""
    idx = max(0, min(len(sorted_sec) - 1, int(q * len(sorted_sec))))
    return sorted_sec[idx]


def _make_cell(dwells: list[int]) -> DwellQuantiles:
    """Build a DwellQuantiles (quantiles + recovery fractions) from dwell secs."""
    dwells.sort()
    n = len(dwells)
    return DwellQuantiles(
        n=n,
        q25_sec=_quantile(dwells, 0.25),
        median_sec=_quantile(dwells, 0.50),
        q75_sec=_quantile(dwells, 0.75),
        recover_by_30=sum(1 for s in dwells if s <= 1800) / n,
        recover_by_60=sum(1 for s in dwells if s <= 3600) / n,
        recover_by_120=sum(1 for s in dwells if s <= 7200) / n,
        curve_sec=[
            _quantile(dwells, i / (CURVE_POINTS - 1)) for i in range(CURVE_POINTS)
        ],
    )


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
) -> dict[str, dict[str, DwellQuantiles]]:
    """Return {route: {state: DwellQuantiles}} for each (route, prev_state)
    with at least `min_samples` transitions. Sparser cells are omitted — the
    consumer should fall back to its analytic estimate."""
    by_cell: dict[tuple[str, str], list[int]] = defaultdict(list)
    for t in transitions:
        by_cell[(t.route, t.prev_state)].append(int(t.dwell_sec))

    out: dict[str, dict[str, DwellQuantiles]] = defaultdict(dict)
    for (route, state), dwells in by_cell.items():
        if len(dwells) < min_samples:
            continue
        out[route][state] = _make_cell(dwells)
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
    """
    by_cell: dict[tuple[str, str, str], list[int]] = defaultdict(list)
    for t in transitions:
        if t.alert_type_at_entry is None:
            continue
        by_cell[(t.route, t.prev_state, t.alert_type_at_entry)].append(int(t.dwell_sec))

    out: dict[str, dict[str, dict[str, DwellQuantiles]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for (route, state, alert_type), dwells in by_cell.items():
        if len(dwells) < min_samples:
            continue
        out[route][state][alert_type] = _make_cell(dwells)
    return {
        r: {s: dict(by_at) for s, by_at in by_state.items()}
        for r, by_state in out.items()
    }
