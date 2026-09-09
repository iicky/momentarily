"""Per-stop next-N arrivals — the Python mirror of worker/src/arrivals.ts.

The Worker derives the published `arrivals` surface; this reproduces the same
derivation so the two stay in lockstep (pinned by tests/fixtures/parity_arrivals
.json). Same rules, not a second set: fold the express variant to the base
route, key by the stop id incl. its direction suffix (so N/S key separately),
drop SKIPPED / timeless / past / beyond-horizon rows, one entry per trip per
stop (first occurrence wins), soonest first, capped at N.

Times are the feed's own absolute POSIX seconds; seconds_away is time - now.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict

# Don't publish a countdown further out than this — past an hour the times are
# speculative and just pad the list. Matches worker/src/arrivals.ts.
HORIZON_SECONDS = 60 * 60
# The next few trains is all a countdown board shows.
MAX_ARRIVALS = 6

# schedule_relationship enum: 0=SCHEDULED, 1=SKIPPED, 2=NO_DATA, 3=UNSCHEDULED.
SKIPPED = 1


@dataclass(frozen=True)
class StopTimeLite:
    """One decoded StopTimeUpdate row — mirror of the TS StopTimeLite."""

    stop_id: str  # GTFS stop id incl. direction suffix; '' when absent
    arrival: int | None
    departure: int | None
    schedule_relationship: int = 0


@dataclass(frozen=True)
class TripLite:
    """A decoded trip carrying its stop times — the arrivals-relevant slice of
    the TS TripLite. `direction` is the NYCT extension enum (1=N, 3=S) or None."""

    route_id: str
    trip_id: str
    direction: int | None = None
    stop_times: list[StopTimeLite] = field(default_factory=list[StopTimeLite])


class ArrivalDict(TypedDict):
    """One entry in the per-stop arrivals list — the JSON shape schema.Arrival
    validates and worker/src/arrivals.ts Arrival mirrors."""

    route: str
    eta_epoch: int
    seconds_away: int
    trip_id: str | None


def base_route(route_id: str) -> str:
    """Fold the express variant to its base route (6X -> 6). Mirrors
    worker/src/trip_updates.ts baseRoute."""
    return route_id[:-1] if route_id.endswith("X") else route_id


def derive_arrivals(trips: list[TripLite], now: int) -> dict[str, list[ArrivalDict]]:
    """Fold decoded trips into per-stop arrivals, soonest first. One pass over
    every row, then a sort + truncate per stop. See the module docstring for the
    drop rules; each returned entry is {route, eta_epoch, seconds_away, trip_id}.
    """
    out: dict[str, list[ArrivalDict]] = {}
    seen: dict[str, set[str]] = {}
    horizon = now + HORIZON_SECONDS

    for trip in trips:
        route = base_route(trip.route_id)
        for st in trip.stop_times:
            if st.schedule_relationship == SKIPPED:
                continue
            time = st.arrival if st.arrival is not None else st.departure
            if time is None:
                continue
            stop_id = st.stop_id
            if stop_id == "":
                continue
            if time < now or time > horizon:
                continue

            stop_seen = seen.get(stop_id)
            if stop_seen is None:
                stop_seen = set[str]()
                seen[stop_id] = stop_seen
                out[stop_id] = []
            # An anonymous trip cannot be told from another anonymous trip, so
            # it is never deduped: two trains the feed left unnamed are two
            # arrivals, not one.
            if trip.trip_id != "":
                if trip.trip_id in stop_seen:
                    continue
                stop_seen.add(trip.trip_id)

            out[stop_id].append(
                ArrivalDict(
                    route=route,
                    eta_epoch=time,
                    seconds_away=time - now,
                    trip_id=trip.trip_id if trip.trip_id != "" else None,
                )
            )

    for entries in out.values():
        entries.sort(key=lambda a: a["eta_epoch"])
        del entries[MAX_ARRIVALS:]
    return out
