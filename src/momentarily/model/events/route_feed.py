"""``RouteFeed``: which GTFS-RT feed publishes each route.

Legacy rows carry no ``source``, so layer 3 attributes them to a feed through
this table. It is measured, not assumed. ``scripts.gen_route_feed`` counts the
distinct trip ids each feed published per **route key** in layer-2 raw-lineage
VehicleObservation and TripUpdateObservation records, and writes
``route_feed.json`` beside this module, with the observation window it read.

A route key is a published ``route_id`` or the route field of a trip id. Both
are needed: ``W`` trains carry ``N`` in the trip id, ``Z`` carry ``J``, ``6X``
carry ``6`` and ``SS`` carry ``SI``. A route's ``feed`` is the feed that
published most of its trips. Every feed that published it is in ``trips``:
a feed can publish another feed's route (in the 2026-10-21 sample, ``ace``
published two F trips and one R trip), so a legacy row's feed is the route's
usual feed, not a proven one.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from importlib.resources import files
from typing import Literal

import blake3
from pydantic import Field, model_validator

from momentarily.model.conventions import Instant, Model
from momentarily.model.events.trip_ids import parse_realtime
from momentarily.model.observations.contracts import GtfsRtSource

TABLE_FILE = "route_feed.json"

ObservedRecord = Literal["vehicle", "trip_update"]


class RouteFeedSource(Model):
    """Where the table was measured."""

    lineage: Literal["raw"]
    records: list[ObservedRecord]
    observed_from: Instant = Field(description="First observed_at read.")
    observed_to: Instant = Field(description="Last observed_at read.")
    rows: int = Field(ge=1, description="Records read.")


class RouteFeedEntry(Model):
    feed: GtfsRtSource = Field(description="The feed that published most trips.")
    trips: dict[GtfsRtSource, int] = Field(
        description="Distinct trip ids each feed published under this route key."
    )

    @model_validator(mode="after")
    def _feed_is_the_majority(self) -> RouteFeedEntry:
        top = max(self.trips.values(), default=0)
        leaders = sorted(f for f, n in self.trips.items() if n == top)
        if leaders != [self.feed]:
            msg = f"feed {self.feed!r} is not the single largest of {self.trips}"
            raise ValueError(msg)
        return self

    @property
    def also_in(self) -> list[GtfsRtSource]:
        return sorted(f for f in self.trips if f != self.feed)


class RouteFeedTable(Model):
    source: RouteFeedSource
    routes: dict[str, RouteFeedEntry] = Field(
        description="Route key (published route_id or trip-id route field)."
    )

    def feed_of(self, route_key: str) -> GtfsRtSource | None:
        entry = self.routes.get(route_key)
        return None if entry is None else entry.feed


class RouteFeedError(ValueError):
    pass


def derive_route_feed(
    rows: Iterable[tuple[GtfsRtSource, str | None, str | None, int]],
    records: list[ObservedRecord],
) -> RouteFeedTable:
    """The table from ``(source, route_id, trip_id, observed_at)`` rows.

    A row adds its trip id (or, with no trip id, the row itself) to its
    feed's count for its ``route_id`` and for its trip id's route field.
    Fails when a route key is published equally often by two feeds, because
    then no feed is the route's own.
    """
    trips: dict[str, dict[GtfsRtSource, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    first: int | None = None
    last: int | None = None
    n = 0
    for source, route_id, trip_id, observed_at in rows:
        n += 1
        first = observed_at if first is None else min(first, observed_at)
        last = observed_at if last is None else max(last, observed_at)
        ident = trip_id if trip_id is not None else f"#{n}"
        keys = set[str]()
        if route_id:
            keys.add(route_id)
        parsed = parse_realtime(trip_id) if trip_id is not None else None
        if parsed is not None:
            keys.add(parsed.route_field)
        for key in keys:
            trips[key][source].add(ident)
    if first is None or last is None:
        msg = "no rows"
        raise RouteFeedError(msg)
    routes: dict[str, RouteFeedEntry] = {}
    ties: list[str] = []
    for key in sorted(trips):
        counts: dict[GtfsRtSource, int] = {
            f: len(ids) for f, ids in sorted(trips[key].items())
        }
        top = max(counts.values())
        leaders: list[GtfsRtSource] = [f for f, c in counts.items() if c == top]
        if len(leaders) > 1:
            ties.append(f"{key}: {counts}")
            continue
        routes[key] = RouteFeedEntry(feed=leaders[0], trips=counts)
    if ties:
        msg = "route keys with no majority feed: " + "; ".join(ties)
        raise RouteFeedError(msg)
    return RouteFeedTable(
        source=RouteFeedSource(
            lineage="raw",
            records=sorted(records),
            observed_from=first,
            observed_to=last,
            rows=n,
        ),
        routes=routes,
    )


def serialize_route_feed(table: RouteFeedTable) -> bytes:
    """Indented, sorted JSON with a trailing newline, for review in diffs."""
    doc = table.model_dump(mode="json", by_alias=True)
    return (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode()


def route_feed_digest(data: bytes) -> str:
    return blake3.blake3(data).hexdigest()


def load_route_feed() -> tuple[RouteFeedTable, str]:
    """The packaged table and the blake3 digest of its file."""
    data = files("momentarily.model.events").joinpath(TABLE_FILE).read_bytes()
    return RouteFeedTable.model_validate_json(data), route_feed_digest(data)
