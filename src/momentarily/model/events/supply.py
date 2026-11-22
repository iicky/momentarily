"""Supply: observed trains against the schedule at each good fetch (§5.2).

Per good fetch of a feed and per route, a ``SupplyObservation`` counts:

- ``vehicles``: real rows (placeholders removed, §2.2) with a fresh
  ``vehicle_timestamp``;
- ``vehicles_stale_tail``: real rows with an old one (a held train or a
  ghost; never turned into an absence). The test reads the row alone, so the
  count at a fetch never depends on later fetches;
- ``scheduled_in_service``: layer-1 trips of the route in service in the
  fetch's minute, summed over the two service days that overlap it (the
  match indexes' ``supply``);
- ``listed_trips`` (raw): trip updates of the fetch naming the route.

A vehicle counts toward the route of the scheduled trip its run matched (Z
trains carry ``J`` in their trip id), else its trip-id route field. Routes
fold 6X, 7X and FX into 6, 7 and F, as the scheduled supply does. Every
route with a vehicle, a scheduled trip or a listed trip at the fetch gets a
record, so a route running no trains against its schedule shows 0. Two good
fetches of one feed in one second share a key; the later fetch id is kept.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence

from momentarily.gtfs.static import base_route
from momentarily.model.conventions import Unknown
from momentarily.model.events.contracts import (
    LINEAGE_SOURCE_BASIS,
    Lineage,
    SupplyObservation,
    TripMatch,
)
from momentarily.model.events.match_index import MatchIndex, candidate_service_dates
from momentarily.model.events.runs import Run
from momentarily.model.observations.contracts import GtfsRtSource

LISTED_UNKNOWN = Unknown(unknown=True, reason="archive/trace keeps no trip updates")


def supply_observations(
    lineage: Lineage,
    polls: Mapping[GtfsRtSource, Sequence[tuple[int, str]]],
    runs: Sequence[Run],
    matches: Sequence[TripMatch],
    listed: Mapping[str, Counter[str]] | None,
    index_for: Callable[[str], MatchIndex | None],
    tz: str,
    span: tuple[int, int],
) -> list[SupplyObservation]:
    """Records for the good polls ``(observed_at, poll id)`` of each feed
    inside ``span``. ``listed``: per raw poll id, trip updates per folded
    route (None on legacy)."""
    by_run = {m.run_id: m for m in matches}
    # Keyed by (feed, poll): a legacy poll is one trace object holding every
    # feed's rows.
    counts: dict[tuple[str, str], dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    basis: dict[tuple[str, str, str], bool] = {}
    for run in runs:
        m = by_run[run.run_id]
        if m.outcome == "matched" and m.route_id is not None:
            route, matched = base_route(m.route_id), True
        else:
            parsed = run.parsed
            if parsed is None:
                continue
            route, matched = base_route(parsed.route_field), False
        for r in run.rows:
            counts[(r.feed, r.poll)][route]["stale" if r.old else "fresh"] += 1
            k = (r.feed, r.poll, route)
            basis[k] = basis.get(k, False) or matched
    out: list[SupplyObservation] = []
    for feed in sorted(polls):
        latest: dict[int, str] = {}
        for at, poll in polls[feed]:
            if span[0] <= at < span[1] and (at not in latest or poll > latest[at]):
                latest[at] = poll
        for at, poll in sorted(latest.items()):
            scheduled: Counter[str] = Counter()
            for day in candidate_service_dates(at, tz)[:2]:
                ix = index_for(day)
                if ix is None:
                    continue
                for route in ix.document.supply:
                    if ix.feed_of(route) == feed:
                        scheduled[route] += ix.supply(route, at)
            seen = counts.get((feed, poll), {})
            lists = None if listed is None else listed.get(poll, Counter())
            routes = {r for r, n in scheduled.items() if n} | set(seen)
            if lists is not None:
                routes |= {r for r, n in lists.items() if n}
            for route in sorted(routes):
                c = seen.get(route, Counter())
                out.append(
                    SupplyObservation(
                        observed_at=at,
                        lineage=lineage,
                        source=feed,
                        source_basis=LINEAGE_SOURCE_BASIS[lineage],
                        route_id=route,
                        route_basis="scheduled_trip"
                        if basis.get((feed, poll, route))
                        else "trip_id_field",
                        vehicles=c["fresh"],
                        vehicles_stale_tail=c["stale"],
                        scheduled_in_service=scheduled[route],
                        listed_trips=LISTED_UNKNOWN if lists is None else lists[route],
                    )
                )
    return out
