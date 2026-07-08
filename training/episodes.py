"""Incident episodes as the grading unit.

Segments each route's canonical-truth series into incident episodes: a maximal
contiguous run of not-normal (disrupted / suspended) ticks on the 5-min grid,
walked the same way the changepoint alignment does (absent tick = normal). Each
episode carries onset / recovery ticks, the dominant cause bucket, the peak
state reached, and left/right censoring flags. These are the unit of account for
onset latency, per-episode recovery scoring, and false-alarm counting.

Pure over its inputs (a truth-state map + a per-tick alert-type map + window
bounds) so it grades without R2 and unit-tests on synthetic fixtures. The truth
map is whatever the caller passes; under the canonical severe-only truth an
episode is a genuine severe incident, and planned work (tier 0) never opens one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from momentarily.mapping import category_for_label, coarse_status, severity_tier
from training.eval import TICK_SECONDS, snap_tick


@dataclass(frozen=True)
class Episode:
    """One incident: a contiguous not-normal run on one route.

    onset    — epoch of the first not-normal grid tick.
    recovery — epoch of the first normal grid tick after the run; for a
               right-censored run, one tick past the window end (so duration_sec
               is a lower bound).
    """

    route: str
    onset: int
    recovery: int
    peak_state: str  # "suspended" if any tick reached it, else "disrupted"
    cause: str  # dominant alert category over the run (see _dominant_cause)
    n_ticks: int  # not-normal grid ticks in the run
    left_censored: (
        bool  # active at the first grid tick — true onset precedes the window
    )
    right_censored: bool  # active at the last grid tick — no observed recovery

    @property
    def duration_sec(self) -> int:
        return self.recovery - self.onset


def _dominant_cause(types_per_tick: list[tuple[str, ...]]) -> str | None:
    """Most-voted disruptive category across the run's ticks. Each tick casts one
    vote for the category of its highest-severity alert, so a lone suspension tick
    cannot relabel a delays-dominated incident. Ties break by higher peak severity
    then category name (deterministic)."""
    votes: Counter[str] = Counter()
    peak_tier: dict[str, int] = {}
    for types in types_per_tick:
        disruptive = [(severity_tier(at), at) for at in types if severity_tier(at) >= 1]
        if not disruptive:
            continue
        tier, alert_type = max(disruptive)
        cause = category_for_label(coarse_status(alert_type))
        votes[cause] += 1
        peak_tier[cause] = max(peak_tier.get(cause, 0), tier)
    if not votes:
        return None
    return max(votes, key=lambda c: (votes[c], peak_tier[c], c))


def extract_episodes(
    truth: dict[tuple[str, int], str],
    types: dict[tuple[str, int], tuple[str, ...]],
    *,
    window_start: int,
    window_end: int,
) -> list[Episode]:
    """Segment the canonical-truth series into incident episodes.

    `truth[(route, tick)]` is the state at an alert-active tick; ticks not present
    read 'normal'. `types[(route, tick)]` is the disruptive alert_types active
    there (for cause attribution). Both are keyed on the 5-min grid. Walks the
    full grid per route so recoveries (the change back to normal when alerts
    clear) are seen, mirroring changepoint_alignment.
    """
    first = snap_tick(window_start)
    last = snap_tick(window_end)
    if first > last:
        return []
    routes = {route for route, _ in truth} | {route for route, _ in types}

    episodes: list[Episode] = []
    for route in sorted(routes):
        in_ep = False
        onset = first
        run_types: list[tuple[str, ...]] = []
        peak_suspended = False
        n = 0
        left_censored = False
        tick = first
        while tick <= last:
            state = truth.get((route, tick), "normal")
            if state != "normal":
                if not in_ep:
                    in_ep = True
                    onset = tick
                    run_types = []
                    peak_suspended = False
                    n = 0
                    left_censored = tick == first
                n += 1
                peak_suspended = peak_suspended or state == "suspended"
                run_types.append(types.get((route, tick), ()))
            elif in_ep:
                episodes.append(
                    _build(
                        route,
                        onset,
                        tick,
                        peak_suspended,
                        run_types,
                        n,
                        left_censored,
                        False,
                    )
                )
                in_ep = False
            tick += TICK_SECONDS
        if in_ep:
            episodes.append(
                _build(
                    route,
                    onset,
                    last + TICK_SECONDS,
                    peak_suspended,
                    run_types,
                    n,
                    left_censored,
                    True,
                )
            )
    return episodes


def _build(
    route: str,
    onset: int,
    recovery: int,
    peak_suspended: bool,
    run_types: list[tuple[str, ...]],
    n: int,
    left_censored: bool,
    right_censored: bool,
) -> Episode:
    peak_state = "suspended" if peak_suspended else "disrupted"
    cause = _dominant_cause(run_types) or (
        "service_suspension" if peak_suspended else "other"
    )
    return Episode(
        route=route,
        onset=onset,
        recovery=recovery,
        peak_state=peak_state,
        cause=cause,
        n_ticks=n,
        left_censored=left_censored,
        right_censored=right_censored,
    )


def disruptive_types_by_key(
    obs_list: list[Any],
) -> dict[tuple[str, int], tuple[str, ...]]:
    """Build the per-(route, tick) alert-type map from a TickObservation list."""
    return {(o.route_id, o.tick): o.disruptive_types for o in obs_list}


def episode_as_dict(ep: Episode) -> dict[str, Any]:
    return {
        "route": ep.route,
        "onset": ep.onset,
        "recovery": ep.recovery,
        "duration_sec": ep.duration_sec,
        "peak_state": ep.peak_state,
        "cause": ep.cause,
        "n_ticks": ep.n_ticks,
        "left_censored": ep.left_censored,
        "right_censored": ep.right_censored,
    }


def episodes_summary(episodes: list[Episode]) -> dict[str, Any]:
    """Per-window episode counts plus the full episode table."""
    by_cause: Counter[str] = Counter(ep.cause for ep in episodes)
    by_peak: Counter[str] = Counter(ep.peak_state for ep in episodes)
    return {
        "n": len(episodes),
        "n_left_censored": sum(ep.left_censored for ep in episodes),
        "n_right_censored": sum(ep.right_censored for ep in episodes),
        "by_cause": dict(by_cause),
        "by_peak_state": dict(by_peak),
        "table": [episode_as_dict(ep) for ep in episodes],
    }
