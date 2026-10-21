"""The per-service-day match index: what the trip matcher reads from layer 1.

One index holds one service day of one layer-1 build:

- every scheduled trip active that day, with its stripped id key (origin,
  route field, direction, path) and block, and the 24:00 wrap: a realtime
  origin ``o`` names this day's static origin ``o`` or ``o + WRAP``; a hit on
  the second is ``wrapped`` and still belongs to this service day, which is
  the calendar day before the observation;
- per pattern, the arrival instant of each active trip at each stop, from
  which ``stop_times`` gives the sorted times per (pattern, stop);
- every pattern's stop list and the platform set, for fitting a run's
  observed stops to patterns without a trip match;
- scheduled supply: trips of each route in service per minute, with 6X, 7X
  and FX folded into 6, 7 and F (riders see the 6; patterns keep them
  apart), a trip counting from the minute of its first departure through
  the minute of its last arrival;
- the active services, and the ``RouteFeed`` table.

Instants are ``instant_of(service_date, gtfs_seconds, tz)``, which counts
from noon minus 12 h, so DST change days are placed right. The matcher reads
the indexes of the candidate days ``candidate_service_dates(at)``.

The stored form is ``MatchIndexDocument`` as canonical JSON (sorted keys,
compact), gzipped with mtime 0. The same inputs and code give the same bytes.
``MatchIndex`` is the in-memory lookup built from a document.
"""

from __future__ import annotations

import gzip
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import blake3
from pydantic import Field

from momentarily.gtfs.static import base_route
from momentarily.model.conventions import (
    Instant,
    Model,
    Provenance,
    ServiceDate,
    instant_of,
)
from momentarily.model.events.keys import MatchIndexKey
from momentarily.model.events.route_feed import RouteFeedEntry, RouteFeedTable
from momentarily.model.events.trip_ids import WRAP, TripIdKey, parse_static
from momentarily.model.network.contracts import (
    DayFeedMap,
    DayType,
    DirectionCode,
    NetworkIndex,
    NetworkSnapshot,
)
from momentarily.model.observations.contracts import GtfsRtSource

SCHEMA_ID = "events/s1"

# Bump whenever a change alters the index built from the same inputs.
INDEX_VERSION = 1

BuildBasis = Literal["verified", "inferred", "explicit"]


class MatchIndexBuildError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__(
            f"match index build failed ({len(problems)} problem(s)):\n  "
            + "\n  ".join(problems)
        )


# --- stored document ------------------------------------------------------------


class IndexTrip(Model):
    trip_id: str
    route_id: str
    pattern_id: str
    service_id: str
    block: str = Field(description="The static id's prefix before the origin.")
    origin: int = Field(
        ge=0, description="Static origin, hundredths of a minute; >= WRAP past 24:00."
    )
    route_field: str
    direction: DirectionCode
    path: str
    first_departure: Instant
    last_arrival: Instant


class IndexPattern(Model):
    pattern_id: str
    route_id: str
    direction: DirectionCode | None
    stops: list[str]
    trips: list[int] = Field(
        description=(
            "Indexes into MatchIndexDocument.trips of the active trips of this "
            "pattern, by (first arrival, trip_id). Empty when none runs."
        )
    )
    arrivals: list[list[Instant]] = Field(
        description="Per stop, the arrival of each trip in ``trips`` order."
    )


class RouteSupply(Model):
    start_minute: int = Field(description="Epoch minute (instant // 60) of counts[0].")
    counts: list[int]


class MatchIndexMetadata(Model):
    schema_: Literal["events/s1"] = Field(alias="schema")
    file_key: str
    index_version: int
    route_feed_digest: str = Field(description="blake3 of route_feed.json.")
    service_date: ServiceDate
    timezone: str
    day_origin: Instant = Field(description="instant_of(service_date, 0, timezone).")
    build_id: str
    feed_id: str
    build_basis: BuildBasis = Field(
        description=(
            "How the build was chosen: the day map's verified feed, its "
            "inferred feed, or named by the caller."
        )
    )
    day_type: DayType
    is_exception: bool
    entries: dict[str, int]
    provenance: Provenance


class MatchIndexDocument(Model):
    metadata: MatchIndexMetadata
    active_service_ids: list[str]
    platforms: list[str]
    patterns: list[IndexPattern]
    trips: list[IndexTrip]
    supply: dict[str, RouteSupply]
    route_feed: dict[str, RouteFeedEntry]


# --- choosing the build ---------------------------------------------------------


@dataclass(frozen=True)
class BuildChoice:
    feed_id: str
    build_id: str
    basis: BuildBasis


def select_build(
    service_date: str,
    index: NetworkIndex,
    day_map: DayFeedMap | None,
    build_id: str | None = None,
) -> BuildChoice:
    """The layer-1 build for a service day.

    With ``build_id``, that build. Otherwise the feed the day map gives the
    UTC day of the same date: its verified ``feed_id``, else its
    ``inferred_feed_id``. A service day runs from about 04:00 UTC on its date
    to about 08:00 UTC the next day, so that UTC day holds most of it.
    """
    if build_id is not None:
        found = [(fid, e) for fid, e in index.feeds.items() if e.build_id == build_id]
        if not found:
            raise MatchIndexBuildError([f"build {build_id} is not in the index"])
        return BuildChoice(found[0][0], build_id, "explicit")
    if day_map is None:
        raise MatchIndexBuildError(["no day map and no build named"])
    entry = day_map.days.get(service_date)
    if entry is None:
        raise MatchIndexBuildError([f"day map has no entry for {service_date}"])
    if entry.feed_id is not None:
        feed_id, basis = entry.feed_id, "verified"
    elif entry.inferred_feed_id is not None:
        feed_id, basis = entry.inferred_feed_id, "inferred"
    else:
        raise MatchIndexBuildError(
            [f"day map names no feed for {service_date}: {entry.reason}"]
        )
    built = index.feeds.get(feed_id)
    if built is None:
        raise MatchIndexBuildError([f"feed {feed_id} of {service_date} is not built"])
    return BuildChoice(feed_id, built.build_id, basis)


def read_network_index(root: Path) -> NetworkIndex:
    """``index.json`` of a layer-1 output directory."""
    path = root / "index.json"
    if not path.is_file():
        raise MatchIndexBuildError([f"no layer-1 index {path}"])
    return NetworkIndex.model_validate_json(path.read_bytes())


def read_day_map(path: Path) -> tuple[DayFeedMap, str]:
    """A layer-1 day map and the blake3 digest of its file."""
    if not path.is_file():
        raise MatchIndexBuildError([f"no day map {path}; name a build with --build-id"])
    raw = path.read_bytes()
    return DayFeedMap.model_validate_json(raw), blake3.blake3(raw).hexdigest()


def read_network(
    root: Path, choice: BuildChoice, index: NetworkIndex
) -> NetworkSnapshot:
    """The build's snapshot from a layer-1 output directory, checked against
    the index's ``content_digest``."""
    path = root / f"{choice.build_id}.json.gz"
    if not path.is_file():
        raise MatchIndexBuildError([f"no network build file {path}"])
    data = gzip.decompress(path.read_bytes())
    expected = index.feeds[choice.feed_id].content_digest
    if blake3.blake3(data).hexdigest() != expected:
        raise MatchIndexBuildError(
            [f"{choice.build_id}.json.gz does not match its index content_digest"]
        )
    return NetworkSnapshot.model_validate_json(data)


# --- building -------------------------------------------------------------------


@dataclass(frozen=True)
class IndexInputs:
    network: NetworkSnapshot
    network_digest: str
    choice: BuildChoice
    day_map_digest: str | None
    route_feed: RouteFeedTable
    route_feed_digest: str
    code_sha: str
    code_dirty: bool | None


def build_match_index(service_date: str, inputs: IndexInputs) -> MatchIndexDocument:
    """The index of one service day. Fails, listing every problem, when the
    day is outside the build, a trip id does not parse as an NYCT id, a trip's
    stops differ from its pattern's, or a route has no ``RouteFeed`` entry."""
    net = inputs.network
    tz = net.system.timezone
    day = next((d for d in net.service_days if d.date == service_date), None)
    if day is None:
        raise MatchIndexBuildError(
            [f"{service_date} is not a service day of build {net.build_id}"]
        )
    active = frozenset(day.active_service_ids)
    origin = instant_of(service_date, 0, tz)
    patterns = {p.pattern_id: p for p in net.patterns}
    problems: list[str] = []

    trips: list[IndexTrip] = []
    arrivals: dict[str, list[int]] = {}
    for t in net.trips:
        if t.service_id not in active:
            continue
        parsed = parse_static(t.trip_id)
        if parsed is None:
            problems.append(f"trip {t.trip_id}: not an NYCT trip id")
            continue
        stops = [s for s, _a, _d in t.stop_times]
        if stops != patterns[t.pattern_id].stops:
            problems.append(f"trip {t.trip_id}: stops differ from {t.pattern_id}")
            continue
        for key in (t.route_id, parsed.route_field):
            if key not in inputs.route_feed.routes:
                problems.append(f"trip {t.trip_id}: no RouteFeed entry for {key!r}")
        arrivals[t.trip_id] = [origin + a for _s, a, _d in t.stop_times]
        trips.append(
            IndexTrip(
                trip_id=t.trip_id,
                route_id=t.route_id,
                pattern_id=t.pattern_id,
                service_id=t.service_id,
                block=parsed.block,
                origin=parsed.origin,
                route_field=parsed.route_field,
                direction=parsed.direction,
                path=parsed.path,
                first_departure=origin + t.stop_times[0][2],
                last_arrival=origin + t.stop_times[-1][1],
            )
        )
    if problems:
        raise MatchIndexBuildError(sorted(set(problems)))
    trips.sort(key=lambda t: t.trip_id)
    position = {t.trip_id: i for i, t in enumerate(trips)}

    by_pattern: dict[str, list[IndexTrip]] = defaultdict(list)
    for t in trips:
        by_pattern[t.pattern_id].append(t)
    out_patterns: list[IndexPattern] = []
    for pid in sorted(patterns):
        p = patterns[pid]
        members = sorted(
            by_pattern.get(pid, []), key=lambda t: (arrivals[t.trip_id][0], t.trip_id)
        )
        out_patterns.append(
            IndexPattern(
                pattern_id=pid,
                route_id=p.route_id,
                direction=p.direction.code,
                stops=list(p.stops),
                trips=[position[t.trip_id] for t in members],
                arrivals=[
                    [arrivals[t.trip_id][i] for t in members]
                    for i in range(len(p.stops))
                ],
            )
        )

    supply = _supply(trips)
    key = MatchIndexKey(
        INDEX_VERSION, service_date, net.build_id, inputs.route_feed_digest[:16]
    )
    inputs_digests = {
        "network": inputs.network_digest,
        "route_feed": inputs.route_feed_digest,
    }
    if inputs.day_map_digest is not None:
        inputs_digests["day_feed_map"] = inputs.day_map_digest
    platforms = sorted(p.stop_id for p in net.platforms)
    metadata = MatchIndexMetadata(
        schema=SCHEMA_ID,
        file_key=key.key,
        index_version=INDEX_VERSION,
        route_feed_digest=inputs.route_feed_digest,
        service_date=service_date,
        timezone=tz,
        day_origin=origin,
        build_id=net.build_id,
        feed_id=inputs.choice.feed_id,
        build_basis=inputs.choice.basis,
        day_type=day.day_type,
        is_exception=day.is_exception,
        entries={
            "trips": len(trips),
            "patterns": len(out_patterns),
            "patterns_running": sum(1 for p in out_patterns if p.trips),
            "stop_times": sum(len(a) for a in arrivals.values()),
            "platforms": len(platforms),
            "supply_minutes": sum(len(s.counts) for s in supply.values()),
            "route_feed": len(inputs.route_feed.routes),
        },
        provenance=Provenance(
            code_sha=inputs.code_sha,
            code_dirty=inputs.code_dirty,
            schema=SCHEMA_ID,
            inputs=inputs_digests,
            captured_at={},
            period_start=service_date,
            observed_from_utc_day=None,
            applied_retroactively=[],
            built_at=net.provenance.built_at,
        ),
    )
    return MatchIndexDocument(
        metadata=metadata,
        active_service_ids=sorted(active),
        platforms=platforms,
        patterns=out_patterns,
        trips=trips,
        supply=supply,
        route_feed=dict(sorted(inputs.route_feed.routes.items())),
    )


def _supply(trips: list[IndexTrip]) -> dict[str, RouteSupply]:
    spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for t in trips:
        spans[base_route(t.route_id)].append(
            (t.first_departure // 60, t.last_arrival // 60)
        )
    out: dict[str, RouteSupply] = {}
    for route in sorted(spans):
        lo = min(a for a, _b in spans[route])
        hi = max(b for _a, b in spans[route])
        delta = [0] * (hi - lo + 2)
        for a, b in spans[route]:
            delta[a - lo] += 1
            delta[b - lo + 1] -= 1
        counts: list[int] = []
        running = 0
        for d in delta[:-1]:
            running += d
            counts.append(running)
        out[route] = RouteSupply(start_minute=lo, counts=counts)
    return out


# --- storage --------------------------------------------------------------------


def canonical_json(doc: MatchIndexDocument) -> bytes:
    return json.dumps(
        doc.model_dump(mode="json", by_alias=True),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()


def serialize(doc: MatchIndexDocument) -> bytes:
    """Gzip of ``canonical_json`` with mtime 0 and no filename."""
    buf = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9
    ) as gz:
        gz.write(canonical_json(doc))
    return buf.getvalue()


def deserialize(data: bytes) -> MatchIndexDocument:
    return MatchIndexDocument.model_validate_json(gzip.decompress(data))


WriteOutcome = Literal["written", "unchanged", "replaced", "conflict"]


def resolve_write(
    stored: bytes | None, doc: MatchIndexDocument, overwrite: bool
) -> tuple[WriteOutcome, bytes | None]:
    """What to do with the bytes already stored under ``doc``'s key, by the
    layer-2 derive job's rule: no file, write; identical bytes, or bytes that
    differ only in the stored file's ``code_sha`` and ``code_dirty``, keep it;
    any other difference, or an unreadable file, is a conflict unless
    ``overwrite``. Returns the outcome and the bytes to write, if any."""
    data = serialize(doc)
    if stored is None:
        return "written", data
    if stored == data:
        return "unchanged", None
    try:
        old = deserialize(stored).metadata.provenance
    except (OSError, EOFError, ValueError):
        old = None
    if old is not None:
        meta = doc.metadata
        prov = meta.provenance.model_copy(
            update={"code_sha": old.code_sha, "code_dirty": old.code_dirty}
        )
        restamped = doc.model_copy(
            update={"metadata": meta.model_copy(update={"provenance": prov})}
        )
        if serialize(restamped) == stored:
            return "unchanged", None
    if overwrite:
        return "replaced", data
    return "conflict", None


# --- lookups --------------------------------------------------------------------


def candidate_service_dates(at: int, tz: str) -> tuple[str, str, str]:
    """The service days an observation at ``at`` can belong to: the local
    date before, of, and after it."""
    local = datetime.fromtimestamp(at, ZoneInfo(tz)).date()
    d0, d1, d2 = (local + timedelta(days=k) for k in (-1, 0, 1))
    return (d0.isoformat(), d1.isoformat(), d2.isoformat())


@dataclass(frozen=True, slots=True)
class KeyHit:
    """A trip a realtime id names; ``wrapped`` when it matched on the
    realtime origin + 24 h (``WRAP``)."""

    trip: IndexTrip
    wrapped: bool


@dataclass(frozen=True, slots=True)
class StopTimes:
    """Scheduled arrivals of one pattern's active trips at one stop, sorted by
    (instant, trip_id)."""

    instants: tuple[int, ...]
    trip_ids: tuple[str, ...]


StaticKey = tuple[int, str, str]


class MatchIndex:
    """Lookups over one service day's ``MatchIndexDocument``."""

    def __init__(self, doc: MatchIndexDocument) -> None:
        self.document = doc
        self.metadata = doc.metadata
        self.service_date = doc.metadata.service_date
        self.timezone = doc.metadata.timezone
        self.build_id = doc.metadata.build_id
        self.active_service_ids = frozenset(doc.active_service_ids)
        self.platforms = frozenset(doc.platforms)
        self.trips: tuple[IndexTrip, ...] = tuple(doc.trips)
        self._trip = {t.trip_id: t for t in doc.trips}
        self._by_key: dict[StaticKey, list[IndexTrip]] = defaultdict(list)
        self._by_origin_dir: dict[tuple[int, str], list[IndexTrip]] = defaultdict(list)
        for t in doc.trips:
            self._by_key[(t.origin, t.route_field, t.direction)].append(t)
            self._by_origin_dir[(t.origin, t.direction)].append(t)
        self._pattern = {p.pattern_id: p for p in doc.patterns}
        self._position = {
            p.pattern_id: {s: i for i, s in enumerate(p.stops)} for p in doc.patterns
        }
        self._column = {
            p.pattern_id: {k: j for j, k in enumerate(p.trips)} for p in doc.patterns
        }
        self._by_route_dir: dict[tuple[str, str | None], list[str]] = defaultdict(list)
        for p in doc.patterns:
            self._by_route_dir[(base_route(p.route_id), p.direction)].append(
                p.pattern_id
            )
        self._row = {t.trip_id: i for i, t in enumerate(doc.trips)}
        self._stop_times: dict[tuple[str, str], StopTimes] = {}

    # trips and ids
    def trip(self, trip_id: str) -> IndexTrip | None:
        return self._trip.get(trip_id)

    def static_key(
        self, origin: int, route_field: str, direction: str
    ) -> tuple[IndexTrip, ...]:
        """Active trips whose stripped static id has this origin, route field
        and direction."""
        return tuple(self._by_key.get((origin, route_field, direction), ()))

    def realtime_key(self, key: TripIdKey) -> tuple[KeyHit, ...]:
        """Active trips a realtime id's (origin, route field, direction) can
        name: the same static origin, then the origin + 24 h (wrapped)."""
        same = self.static_key(key.origin, key.route_field, key.direction)
        wrapped = self.static_key(key.origin + WRAP, key.route_field, key.direction)
        return tuple(KeyHit(t, False) for t in same) + tuple(
            KeyHit(t, True) for t in wrapped
        )

    def origin_direction(self, origin: int, direction: str) -> tuple[IndexTrip, ...]:
        return tuple(self._by_origin_dir.get((origin, direction), ()))

    def arrival(self, trip_id: str, stop: str) -> int | None:
        """Scheduled arrival instant of a trip at a stop, or None when the trip
        is not active today or does not serve the stop."""
        t = self._trip.get(trip_id)
        if t is None:
            return None
        i = self._position[t.pattern_id].get(stop)
        if i is None:
            return None
        j = self._column[t.pattern_id][self._row[trip_id]]
        return self._pattern[t.pattern_id].arrivals[i][j]

    # patterns
    def patterns_of(self, route_id: str, direction: str | None) -> tuple[str, ...]:
        """Patterns of a route, 6X/7X/FX folded into 6/7/F, in a direction,
        whether or not they run today."""
        return tuple(self._by_route_dir.get((base_route(route_id), direction), ()))

    def pattern_stops(self, pattern_id: str) -> tuple[str, ...]:
        return tuple(self._pattern[pattern_id].stops)

    def stop_position(self, pattern_id: str, stop: str) -> int | None:
        return self._position[pattern_id].get(stop)

    def stop_times(self, pattern_id: str, stop: str) -> StopTimes:
        """Sorted scheduled arrivals at a stop of a pattern's active trips."""
        cached = self._stop_times.get((pattern_id, stop))
        if cached is not None:
            return cached
        p = self._pattern[pattern_id]
        i = self._position[pattern_id].get(stop)
        pairs = (
            []
            if i is None
            else sorted(
                zip(
                    p.arrivals[i], (self.trips[k].trip_id for k in p.trips), strict=True
                )
            )
        )
        out = StopTimes(tuple(a for a, _ in pairs), tuple(t for _, t in pairs))
        self._stop_times[(pattern_id, stop)] = out
        return out

    # supply and feeds
    def supply(self, route_id: str, at: int) -> int:
        """Trips of a route in service in the minute holding ``at``; 6X, 7X
        and FX count toward 6, 7 and F, and are folded here too."""
        s = self.document.supply.get(base_route(route_id))
        if s is None:
            return 0
        k = at // 60 - s.start_minute
        return s.counts[k] if 0 <= k < len(s.counts) else 0

    def supply_by_feed(self, at: int) -> dict[GtfsRtSource, int]:
        out: dict[GtfsRtSource, int] = {}
        for route_id in self.document.supply:
            n = self.supply(route_id, at)
            feed = self.feed_of(route_id)
            if n and feed is not None:
                out[feed] = out.get(feed, 0) + n
        return out

    def feed_of(self, route_key: str) -> GtfsRtSource | None:
        entry = self.document.route_feed.get(route_key)
        return None if entry is None else entry.feed
