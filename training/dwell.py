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
    """Quantiles of dwell duration (seconds) for a (route, state) cell."""

    n: int
    q25_sec: int
    median_sec: int
    q75_sec: int


def _quantile(sorted_sec: list[int], q: float) -> int:
    """Sample quantile via the nearest-rank rule. sorted_sec must be non-empty."""
    idx = max(0, min(len(sorted_sec) - 1, int(q * len(sorted_sec))))
    return sorted_sec[idx]


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
        dwells.sort()
        out[route][state] = DwellQuantiles(
            n=len(dwells),
            q25_sec=_quantile(dwells, 0.25),
            median_sec=_quantile(dwells, 0.50),
            q75_sec=_quantile(dwells, 0.75),
        )
    return dict(out)
