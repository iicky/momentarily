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


class DwellQuantiles(TypedDict):
    """Empirical dwell-duration summary for a (route, state[, alert_type]) cell.

    Quantiles back recovery_minutes_low/median/high. The recover_by_* fractions
    are the empirical P(dwell <= horizon) — the Worker uses them for the
    p_normal_in_30/60/120min projection, which is otherwise a geometric estimate
    from the transition self-loop that can't represent the heavy-tailed,
    cause-dependent recovery curve (delays clear fast, planned work lingers).
    See momentarily-<recovery-prob>.
    """

    n: int
    q25_sec: int
    median_sec: int
    q75_sec: int
    recover_by_30: float
    recover_by_60: float
    recover_by_120: float


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
    )


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
