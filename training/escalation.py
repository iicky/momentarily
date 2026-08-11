"""Alert corroboration for movement-disrupted episodes: does the alert feed
independently confirm what movement already saw, and how far ahead or behind?

The movement arm's disrupted state comes from vehicle positions; the alert
feed comes from MTA dispatch. An alert appearing near a movement episode's
onset IS positive evidence the disruption was real. Alert ABSENCE is NOT
positive evidence it wasn't — the MTA may simply never post one, or post it
late (measured: over one window, all 29 movement-disrupted ticks the alert
feed read "normal" on later resolved with no counter-evidence ever surfacing).
There is no true-negative class here, so no false-positive rate, precision,
specificity, or "accuracy" against alerts is a valid number. Exactly two are
computed:

  confirmation_rate — of movement-disrupted episodes (the training/episodes.py
                       grading unit: one long stall is one observation, not one
                       per tick), the share a corroborating alert covers within
                       a window around onset.
  lead_time          — for confirmed episodes only, signed minutes from
                        movement onset to the nearest corroborating alert tick.
                        Negative = the alert led (posted before movement saw
                        the effect).

Unconfirmed episodes are counted (`n_unconfirmed`) and never divided into a
rate implying they were false — that would just resurrect the asymmetry error
this module exists to avoid under a different name.
"""

from __future__ import annotations

import statistics
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Any

from training.episodes import Episode

# Forward corroboration window: how far past a movement episode's onset to
# look for an alert catching up. 60 min — long enough for a lagging MTA alert.
DEFAULT_FORWARD_MINUTES = 60

# Backward corroboration window: how far before onset an alert still counts as
# corroborating — the less common case where dispatch posts before movement
# shows the effect.
DEFAULT_BACKWARD_MINUTES = 30


@dataclass(frozen=True)
class EpisodeCorroboration:
    """One movement-disrupted episode scored against the alert feed within the
    corroboration window around its onset.

    `lead_minutes` is the signed offset (alert tick minus movement onset) of
    the alert-disrupted tick nearest to onset, or None if none fell inside the
    window. None is NOT evidence against the episode — see module docstring.
    """

    episode: Episode
    lead_minutes: float | None

    @property
    def confirmed(self) -> bool:
        return self.lead_minutes is not None


def corroborate_episodes(
    episodes: list[Episode],
    alert_disrupted: AbstractSet[tuple[str, int]],
    *,
    forward_minutes: int = DEFAULT_FORWARD_MINUTES,
    backward_minutes: int = DEFAULT_BACKWARD_MINUTES,
    tick_seconds: int = 300,
) -> list[EpisodeCorroboration]:
    """Score each movement episode against the alert feed's per-(route, tick)
    disrupted set, searching [onset - backward, onset + forward].

    Only the episode's onset is scored — its later ticks are the same
    movement signal continuing, not fresh corroboration. When more than one
    alert-disrupted tick falls in the window, the one nearest onset wins (ties
    — equally near on both sides — keep the earlier, negative-lead one, since
    the scan runs backward-to-forward).
    """
    forward_ticks = forward_minutes * 60 // tick_seconds
    backward_ticks = backward_minutes * 60 // tick_seconds
    out: list[EpisodeCorroboration] = []
    for ep in episodes:
        best: float | None = None
        for k in range(-backward_ticks, forward_ticks + 1):
            if (ep.route, ep.onset + k * tick_seconds) not in alert_disrupted:
                continue
            minutes = k * tick_seconds / 60.0
            if best is None or abs(minutes) < abs(best):
                best = minutes
        out.append(EpisodeCorroboration(episode=ep, lead_minutes=best))
    return out


def confirmation_summary(
    corroborations: list[EpisodeCorroboration],
    *,
    forward_minutes: int = DEFAULT_FORWARD_MINUTES,
    backward_minutes: int = DEFAULT_BACKWARD_MINUTES,
) -> dict[str, Any]:
    """Aggregate corroborate_episodes() into the two valid metric families:
    how often a corroborating alert covers a movement-disrupted episode
    (confirmation_rate), and how far ahead or behind it landed (lead_time).

    No precision, false-positive, specificity, or "accuracy" number is
    computed here, on purpose: alert absence cannot refute a movement-disrupted
    episode, so there is no valid denominator for one. n_unconfirmed is a
    count, not a rate — folding it into one would launder that same
    unfalsifiable comparison back in.
    """
    n = len(corroborations)
    confirmed = [c for c in corroborations if c.confirmed]
    n_confirmed = len(confirmed)
    leads = sorted(c.lead_minutes for c in confirmed if c.lead_minutes is not None)
    if len(leads) >= 2:
        q1, _, q3 = statistics.quantiles(leads, n=4, method="inclusive")
        iqr: list[float] | None = [q1, q3]
    elif leads:
        iqr = [leads[0], leads[0]]
    else:
        iqr = None
    return {
        "confirmation_rate": (n_confirmed / n) if n else None,
        "n_episodes": n,
        "n_confirmed": n_confirmed,
        # NOT a false-alarm count. An unconfirmed episode has no alert in the
        # corroboration window; a lagging/incomplete MTA feed explains that as
        # readily as a spurious movement read. Never divide this into a rate.
        "n_unconfirmed": n - n_confirmed,
        "corroboration_window_minutes": {
            "forward": forward_minutes,
            "backward": backward_minutes,
        },
        "lead_time_minutes": {
            "n": len(leads),
            "median": statistics.median(leads) if leads else None,
            "iqr": iqr,
        },
        "note": (
            "confirmation_rate is a LOWER bound (MTA alerts lag/are incomplete); "
            "n_unconfirmed carries no information against the episode it counts"
        ),
    }
