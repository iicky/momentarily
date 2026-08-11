"""Transfer viability from station_complex_id + routes_served (training/transfers.py).

Pure join over the static station metadata cached at state/stations.json:
{fetched_at, stations: Record<gtfs_stop_id, StationOut>} (worker/src/
stations_static.ts, mirrored by src/momentarily/schema.py:Station). No
network access here — callers load the JSON (e.g. via training.r2_client)
and pass the `stations` dict straight through; StationRecord is JSON-shaped
so no conversion step is needed.

The question this answers: given a rider standing on route A, can they reach
route B without leaving the fare-controlled area, and where? Two routes are
transferable at a station complex when both appear somewhere in the
routes_served of stops sharing that station_complex_id (build_complexes,
complex_key). This is independent of the movement/regime machinery — it is
static topology, not a live state — but it shares the segment world's
station-id convention (station_id mirrors worker/src/segment_flow.ts
stationId) so a path query can move between "which segment am I on" and
"can I transfer here" using the same key.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypedDict


class StationRecord(TypedDict):
    """One physical stop's static metadata. Mirrors worker/src/stations_static.ts
    StationOut and its pydantic twin src/momentarily/schema.py:Station field for
    field — JSON-shaped so a station loaded straight from state/stations.json
    needs no conversion."""

    gtfs_stop_id: str
    station_complex_id: str | None
    name: str
    borough: str | None
    routes_served: list[str]
    ada: Literal[0, 1, 2]  # 0 not accessible | 1 fully accessible | 2 partial
    ada_northbound: bool
    ada_southbound: bool


# state/stations.json's `stations` map: gtfs_stop_id -> StationRecord.
StationsById = Mapping[str, StationRecord]


def station_id(stop_id: str) -> str:
    """Collapse a directional stop id to its station: strip a trailing N/S
    suffix ('A09N' -> 'A09'). Mirrors worker/src/segment_flow.ts stationId
    exactly — the segment world's from_stop/to_stop and this module's stop
    lookups must agree on one rule, not two."""
    return stop_id[:-1] if stop_id and stop_id[-1] in ("N", "S") else stop_id


# ada codes counted as "has some documented accessibility". 0 means the feed
# found no accessible path; 1 (fully accessible) and 2 (partial, e.g. only one
# direction or one entrance) both count — routes_served has no per-platform
# granularity, so "partial" is the best positive signal available and still
# beats "none".
_ADA_ACCESSIBLE: frozenset[int] = frozenset({1, 2})


def complex_key(station: StationRecord) -> str:
    """Join key grouping stops into one transfer complex.

    A non-null station_complex_id is 39hk-dx4f's own notion of a complex: the
    stop rows that share a fare-controlled area (e.g. Times Sq-42 St spans
    several rows, one per line/direction). Every stop with that id joins the
    same complex.

    A null station_complex_id means the feed does NOT document this stop as
    physically joined to any other, so it is a singleton complex keyed on its
    own gtfs_stop_id — never merged with other null-complex stops. Merging
    every null-complex stop into one shared bucket would call routes at two
    unrelated, unconnected single-platform stations "transferable" purely
    because neither happens to carry a documented complex id. Singleton is
    the conservative read: a transfer is only viable where the data actually
    says the stops are joined (or it's the same stop serving both routes).
    """
    return station["station_complex_id"] or f"stop:{station['gtfs_stop_id']}"


def routes_at(stations: StationsById, stop_id: str) -> frozenset[str]:
    """Routes served at one physical stop. `stop_id` may be directional
    (segment-world from_stop/to_stop) or bare — station_id() is applied
    first, the same collapse the segment world already uses, so a path query
    can pass either form."""
    station = stations.get(station_id(stop_id))
    return frozenset(station["routes_served"]) if station is not None else frozenset()


@dataclass(frozen=True)
class TransferComplex:
    """One transfer complex: the stops joined at it and the routes reachable
    there. `ada_routes` is the subset of `routes` served by at least one stop
    in the complex with documented ADA access (ada in {1, 2}) — a route can
    be in `routes` without being in `ada_routes` when every stop serving it
    at this complex is not accessible."""

    complex_id: str
    stops: tuple[str, ...]
    routes: frozenset[str]
    ada_routes: frozenset[str]


def build_complexes(stations: StationsById) -> dict[str, TransferComplex]:
    """Group every stop into its transfer complex (complex_key) and roll up
    the routes reachable there, plain and ADA-accessible."""
    grouped: dict[str, list[StationRecord]] = defaultdict(list)
    for station in stations.values():
        grouped[complex_key(station)].append(station)

    out: dict[str, TransferComplex] = {}
    for complex_id, stops in grouped.items():
        routes = frozenset(r for s in stops for r in s["routes_served"])
        ada_routes = frozenset(
            r for s in stops if s["ada"] in _ADA_ACCESSIBLE for r in s["routes_served"]
        )
        out[complex_id] = TransferComplex(
            complex_id=complex_id,
            stops=tuple(sorted(s["gtfs_stop_id"] for s in stops)),
            routes=routes,
            ada_routes=ada_routes,
        )
    return out


@dataclass(frozen=True)
class TransferResult:
    """Whether route_a <-> route_b is a viable transfer, and where.
    `viable`/`complexes` answer "can you"; `ada_viable`/`ada_complexes` are a
    SEPARATE axis layered on top — a pair can be viable with zero
    ADA-accessible complexes, and that never changes `viable`."""

    route_a: str
    route_b: str
    complexes: tuple[str, ...]
    ada_complexes: tuple[str, ...]

    @property
    def viable(self) -> bool:
        return len(self.complexes) > 0

    @property
    def ada_viable(self) -> bool:
        return len(self.ada_complexes) > 0


def complexes_joining(
    complexes: Mapping[str, TransferComplex], route_a: str, route_b: str
) -> TransferResult:
    """Every complex where both routes are reachable (plain), and the subset
    where both are reachable via an ADA-accessible stop. Sorted complex ids."""
    plain: list[str] = []
    ada: list[str] = []
    for c in complexes.values():
        if route_a in c.routes and route_b in c.routes:
            plain.append(c.complex_id)
        if route_a in c.ada_routes and route_b in c.ada_routes:
            ada.append(c.complex_id)
    return TransferResult(
        route_a=route_a,
        route_b=route_b,
        complexes=tuple(sorted(plain)),
        ada_complexes=tuple(sorted(ada)),
    )
