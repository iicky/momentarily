"""Escalation-arm validation: are movement disruptions that go beyond the alert
feed genuine leading indicators, or false alarms?

The movement-primary nowcast's disrupted arm is derived from vehicle positions,
so no contemporaneous machine signal can independently adjudicate it: the alert
feed is what an escalation goes *beyond*, and the vehicle feed IS the escalation
signal. The one honest test left is temporal — treat the *future* as truth.

An escalation is a route reading disrupted-by-movement while the alert feed reads
normal (no alert of any kind) and had no alert in the recent past (else it is a
post-alert tail — movement lagging a cleared alert, not leading one). It is
"confirmed" when an alert later appears in a forward window (the alert catching
up = the disruption was real and movement led it), and "evaporated" when movement
returns to normal at the very next tick with no alert following (the false-alarm
candidate).

Caveats baked into every reading:
  - Alert confirmation is a LOWER bound: MTA alerts lag and are incomplete, so a
    sustained-but-unconfirmed escalation is not necessarily a false alarm — the
    MTA may simply never post one.
  - Persistence (movement staying disrupted) is the SAME signal, so it is a
    self-consistency check, not independent corroboration.
  - Scored on the offline movement recompute (before an archived published
    condition exists), the classifier baseline differs from the live params, so
    the cohort is "offline movement-rule escalations" — a proxy for the published
    arm, not the arm itself.
"""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Any

# Confirmation horizon: how far forward to look for an alert catching up to a
# movement escalation. 12 ticks * 5 min = 60 min — long enough for a lagging MTA
# alert, with the lead-time distribution reported so the cutoff stays visible.
DEFAULT_HORIZON_TICKS = 12

# How far back a cleared alert still taints a fresh movement disruption. Within
# this window, a new disrupted run is a post-alert tail (movement lagging the
# alert's clearance), not movement leading alerts. 6 ticks = 30 min.
DEFAULT_PRIOR_ALERT_LOOKBACK_TICKS = 6

# Consecutive unobserved (feed-gap) ticks to bridge before a resuming disrupted
# run counts as a new onset rather than a continuation — keeps a brief archive
# gap mid-disruption from splitting one escalation into several. 2 ticks = 10 min.
DEFAULT_GAP_TOLERANCE_TICKS = 2

# A movement escalation that holds this long without any alert is the notable
# "movement sees what alerts never confirm" case — a real gap in the alert feed
# or a persistent false signal, worth surfacing either way. 3 ticks = 15 min.
SUSTAINED_TICKS = 3


@dataclass(frozen=True)
class EscalationEvent:
    """One escalation onset: a route reading movement-disrupted while the alert
    feed read normal, at the tick the escalation began."""

    route: str
    tick: int
    # First forward tick (1..horizon) at which an alert appeared, or None if none
    # did within the horizon. lead_minutes derives from this.
    lead_ticks: int | None
    # Consecutive movement-disrupted ticks from onset (inclusive) — how long the
    # movement signal itself sustained the call. Same-signal persistence; a feed
    # gap truncates it.
    persisted_ticks: int
    # Forward ticks (1..horizon) actually present in the movement series — the
    # coverage the forward labels are conditioned on.
    observed_forward: int
    # This onset followed an alert that cleared within the lookback window: a tail
    # of a prior alert, not movement leading one. Excluded from the leading-
    # indicator cohort but reported separately.
    post_alert_tail: bool

    @property
    def alert_confirmed(self) -> bool:
        return self.lead_ticks is not None

    @property
    def evaporated(self) -> bool:
        # Movement disrupted for a single tick and no alert ever followed: the
        # false-alarm candidate. Sustained-but-unconfirmed is NOT evaporated —
        # alerts may simply never post for a real disruption.
        return self.persisted_ticks <= 1 and self.lead_ticks is None


def _disrupted(state: str | None) -> bool:
    return state == "disrupted"


def escalation_events(
    movement_state: Mapping[tuple[str, int], str],
    alert_disrupted: AbstractSet[tuple[str, int]],
    *,
    horizon_ticks: int = DEFAULT_HORIZON_TICKS,
    prior_alert_lookback_ticks: int = DEFAULT_PRIOR_ALERT_LOOKBACK_TICKS,
    gap_tolerance_ticks: int = DEFAULT_GAP_TOLERANCE_TICKS,
    tick_seconds: int = 300,
) -> list[EscalationEvent]:
    """Escalation onsets across the (route, tick) movement series.

    An escalation is *active* at (route, t) when movement reads disrupted there
    and the alert feed does not flag the route ((route, t) not in
    alert_disrupted). An onset is a tick where the escalation is active and the
    most recent OBSERVED prior tick (bridging up to gap_tolerance_ticks of feed
    gap) is not active — so a resuming disrupted run after a real return-to-normal
    or a cleared alert is an onset, but a brief archive gap mid-disruption is not.

    Each onset is scanned forward over t+1..t+horizon for the first tick the
    alert feed flags the route (lead_ticks) and for observed coverage; scanned
    backward over the lookback for a recent alert (post_alert_tail); and its
    movement-disrupted run length from onset gives persistence.
    """

    def active(route: str, tick: int) -> bool:
        return _disrupted(movement_state.get((route, tick))) and (
            (route, tick) not in alert_disrupted
        )

    def is_onset(route: str, tick: int) -> bool:
        # Walk back over unobserved ticks (feed gaps) up to the tolerance; the
        # first OBSERVED prior tick decides: active there → continuation, inactive
        # → a real return-to-normal so this is a new onset. All-unobserved past
        # the tolerance is a new episode too.
        for j in range(1, gap_tolerance_ticks + 2):
            prev = tick - j * tick_seconds
            if (route, prev) in movement_state:
                return not active(route, prev)
        return True

    events: list[EscalationEvent] = []
    for route, tick in movement_state:
        if not active(route, tick):
            continue
        if not is_onset(route, tick):
            continue  # continuation of an escalation already under way
        lead_ticks: int | None = None
        observed_forward = 0
        for k in range(1, horizon_ticks + 1):
            forward_tick = tick + k * tick_seconds
            if (route, forward_tick) in movement_state:
                observed_forward += 1
            if lead_ticks is None and (route, forward_tick) in alert_disrupted:
                lead_ticks = k

        post_alert_tail = any(
            (route, tick - j * tick_seconds) in alert_disrupted
            for j in range(1, prior_alert_lookback_ticks + 1)
        )

        persisted = 1
        while _disrupted(movement_state.get((route, tick + persisted * tick_seconds))):
            persisted += 1

        events.append(
            EscalationEvent(
                route=route,
                tick=tick,
                lead_ticks=lead_ticks,
                persisted_ticks=persisted,
                observed_forward=observed_forward,
                post_alert_tail=post_alert_tail,
            )
        )
    events.sort(key=lambda e: (e.route, e.tick))
    return events


def _median(xs: list[float]) -> float | None:
    if not xs:
        return None
    mid = len(xs) // 2
    return xs[mid] if len(xs) % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def escalation_summary(
    events: list[EscalationEvent],
    *,
    horizon_ticks: int = DEFAULT_HORIZON_TICKS,
    tick_seconds: int = 300,
    source: str = "offline_movement_rule",
) -> dict[str, Any]:
    """Aggregate escalation events into leading-indicator metrics.

    Headline metrics cover the genuine leading-indicator cohort (post-alert tails
    excluded): alert-confirmation rate + lead-time distribution (is movement a
    genuine leading indicator?) and the evaporation rate (the false-alarm proxy).
    `source` records whether the cohort came from the archived published
    condition or the offline movement recompute; `n_post_alert_tail` is reported
    beside the cohort for transparency.
    """
    genuine = [e for e in events if not e.post_alert_tail]
    n = len(genuine)
    confirmed = [e for e in genuine if e.alert_confirmed]
    leads_min = sorted(
        e.lead_ticks * tick_seconds / 60.0
        for e in confirmed
        if e.lead_ticks is not None
    )

    def within(minutes: float) -> int:
        return sum(1 for m in leads_min if m <= minutes)

    n_evaporated = sum(1 for e in genuine if e.evaporated)
    n_sustained_unconfirmed = sum(
        1
        for e in genuine
        if not e.alert_confirmed and e.persisted_ticks >= SUSTAINED_TICKS
    )

    per_route: dict[str, dict[str, int]] = {}
    for e in genuine:
        row = per_route.setdefault(
            e.route, {"n": 0, "alert_confirmed": 0, "evaporated": 0}
        )
        row["n"] += 1
        if e.alert_confirmed:
            row["alert_confirmed"] += 1
        if e.evaporated:
            row["evaporated"] += 1

    return {
        "source": source,
        "horizon_minutes": horizon_ticks * tick_seconds // 60,
        "n_escalations": n,
        "n_post_alert_tail": sum(1 for e in events if e.post_alert_tail),
        "n_alert_confirmed": len(confirmed),
        "alert_confirmed_rate": (len(confirmed) / n) if n else None,
        "confirmed_within_15min": within(15),
        "confirmed_within_30min": within(30),
        "confirmed_within_60min": within(60),
        "lead_minutes_median": _median(leads_min),
        "lead_minutes_mean": (sum(leads_min) / len(leads_min)) if leads_min else None,
        "n_evaporated": n_evaporated,
        "evaporated_rate": (n_evaporated / n) if n else None,
        "n_sustained_unconfirmed": n_sustained_unconfirmed,
        "per_route": dict(sorted(per_route.items())),
        "note": (
            "escalation = movement-disrupted while the alert feed read normal "
            "(post-alert tails excluded), scored temporally against later alerts; "
            "alert confirmation is a LOWER bound (MTA alerts lag/incomplete), "
            "persistence is same-signal"
        ),
    }
