"""Generic regime clock over a keyed stream of per-tick calls.

A classifier answers "what is this cell doing right now", one tick at a time.
A regime is the debounced run of that answer plus the timestamp it began. The
same code clocks a route (key = route id) and a segment (key =
``route|direction|from_stop``); the abstraction is the clock, not the thing
being clocked.

Debounce mirrors :func:`load_r2.derive_actual_recovery`: a change commits only
after ``debounce_ticks`` consecutive calls agree, and the new regime is
back-dated to the first tick of that run rather than the tick the run
completed. An abstention — the key absent from this tick's calls — resets the
candidate run but never ends an open regime: no reading is not a reading of
change.

A cell whose first-ever call arrives mid-window opens its regime at that tick.
There is no prior regime to protect, so there is nothing to debounce against.

Online (``worker/src/regime.ts``) and offline (here) must segment identically,
or curves fitted offline describe regimes the Worker never enters. The pin is
``tests/fixtures/parity_regime.json``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

# Consecutive agreeing calls required to commit a regime change.
#
# One, not two. The movement classifier is already conservative: over
# 2026-08-04..08-11 it opened 17 route episodes in 65,685 route-ticks, ~193x
# rarer than its nominal alpha=0.05 tail would produce, because the
# DISRUPTED_RATIO posterior gate binds long before the significance test does.
# A second confirming tick erased 13 of those 17 episodes and every episode
# under 10 minutes, against a population whose own median is 5.0 min — one
# tick. Debouncing that is deleting signal, not filtering noise. The mechanism
# stays (callers may raise it, and the parity fixture pins it at 2); the
# default does not use it.
DEBOUNCE_TICKS = 1

# A cell unobserved for longer than this drops out. A brief abstention holds
# the regime open; an hour of blindness means the regime that resumes is not
# knowably the one that stopped.
MAX_IDLE_SEC = 3600


@dataclass(frozen=True)
class RegimeEntry:
    """One cell's debounced regime and the clock for it."""

    state: str
    entered_at: int
    last_seen_at: int
    # Candidate state seen but not yet debounced, and the tick its run started.
    # A committed change back-dates entered_at to pending_since.
    pending: str | None = None
    pending_since: int = 0
    pending_run: int = 0


@dataclass(frozen=True)
class RegimeChange:
    """A committed regime change. Field names match the Worker's
    TransitionRecord so the two streams grade through one code path."""

    key: str
    prev_state: str
    new_state: str
    entered_at: int
    exited_at: int
    dwell_sec: int


def advance_regimes(
    prev: Mapping[str, RegimeEntry] | None,
    observed: Mapping[str, str],
    observed_at: int,
    *,
    debounce_ticks: int = DEBOUNCE_TICKS,
    max_idle_sec: int = MAX_IDLE_SEC,
) -> tuple[dict[str, RegimeEntry], list[RegimeChange]]:
    """Advance every cell's regime by one tick.

    `observed` holds this tick's raw classifier calls; a key absent from it is
    an abstention, not a state. Returns the new entry map and the changes that
    committed this tick, ordered by key so both languages emit the same list.
    """
    entries: dict[str, RegimeEntry] = {}
    for key, entry in (prev or {}).items():
        if observed_at - entry.last_seen_at > max_idle_sec:
            continue
        if key in observed:
            entries[key] = entry
        else:
            entries[key] = replace(entry, pending=None, pending_since=0, pending_run=0)

    changes: list[RegimeChange] = []
    for key in sorted(observed):
        call = observed[key]
        entry = entries.get(key)
        if entry is None:
            entries[key] = RegimeEntry(
                state=call, entered_at=observed_at, last_seen_at=observed_at
            )
            continue
        if call == entry.state:
            entries[key] = replace(
                entry,
                last_seen_at=observed_at,
                pending=None,
                pending_since=0,
                pending_run=0,
            )
            continue
        same_candidate = entry.pending == call
        run = entry.pending_run + 1 if same_candidate else 1
        since = entry.pending_since if same_candidate else observed_at
        if run >= debounce_ticks:
            changes.append(
                RegimeChange(
                    key=key,
                    prev_state=entry.state,
                    new_state=call,
                    entered_at=entry.entered_at,
                    exited_at=since,
                    dwell_sec=since - entry.entered_at,
                )
            )
            entries[key] = RegimeEntry(
                state=call, entered_at=since, last_seen_at=observed_at
            )
        else:
            entries[key] = replace(
                entry,
                last_seen_at=observed_at,
                pending=call,
                pending_since=since,
                pending_run=run,
            )
    return entries, changes


def replay_regimes(
    ticks: Sequence[tuple[int, Mapping[str, str]]],
    *,
    debounce_ticks: int = DEBOUNCE_TICKS,
    max_idle_sec: int = MAX_IDLE_SEC,
) -> tuple[dict[str, RegimeEntry], list[RegimeChange]]:
    """Fold `advance_regimes` over a whole (observed_at, calls) history.

    This is how the offline backfill reconstructs the transition stream the
    Worker would have emitted, so it must go through the same function rather
    than reimplementing the debounce.
    """
    entries: dict[str, RegimeEntry] = {}
    changes: list[RegimeChange] = []
    for observed_at, calls in sorted(ticks, key=lambda t: t[0]):
        entries, tick_changes = advance_regimes(
            entries,
            calls,
            observed_at,
            debounce_ticks=debounce_ticks,
            max_idle_sec=max_idle_sec,
        )
        changes.extend(tick_changes)
    return entries, changes
