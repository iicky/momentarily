"""Storage key layout of the events layer.

Every key sits under ``model/events/s1/``:

    match_index/v<index_version>/<service_date>/<build_id>-<route_feed16>.json.gz
    <lineage>/<record>/<YYYY-MM-DD>.parquet

The index of a service day depends on the index code version, the layer-1
build it was made from, and the ``RouteFeed`` table (first 16 hex of its
blake3), so all three are in the key: a day whose build choice changes (an
unverified day verified later), a new table or a new ``INDEX_VERSION`` gets a
new key rather than different bytes under the old one.

Event records are filed per lineage (``raw``, ``legacy``; a run is never built
from both) and partitioned by ServiceDate (decision 2): ``trip_match``,
``stop_event``, ``segment_traversal``, ``headway`` and ``open_state`` under the
service day whose build wrote them (unmatched runs by their ``id_origin``
date; a build's window-end tail and the next day's checkpoint ``OpenState``
under the building day). ``coverage`` and ``supply`` are keyed to fetch
instants, not to runs, so they are filed by UTC day: a supply record under
the day of its ``observed_at``, and a coverage interval under the day of its
start, with every interval split at 00:00Z so that a day's file covers
exactly that day. ``parse_key(k.key) == k`` for every valid key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Literal, get_args

from momentarily.model.events.contracts import Lineage as EventLineage
from momentarily.model.observations.keys import KeyLayoutError

PREFIX = "model/events/s1/"

ServiceDayRecord = Literal[
    "trip_match", "stop_event", "segment_traversal", "headway", "open_state"
]
UtcDayRecord = Literal["coverage", "supply"]

LINEAGES: tuple[EventLineage, ...] = get_args(EventLineage)
SERVICE_DAY_RECORDS: tuple[ServiceDayRecord, ...] = get_args(ServiceDayRecord)
UTC_DAY_RECORDS: tuple[UtcDayRecord, ...] = get_args(UtcDayRecord)

_HEX16 = re.compile(r"[0-9a-f]{16}")
_MATCH_INDEX = re.compile(
    rf"^{re.escape(PREFIX)}match_index/v(?P<version>[1-9][0-9]*)/"
    r"(?P<day>\d{4}-\d{2}-\d{2})/(?P<build>[0-9a-f]{16})-(?P<rf>[0-9a-f]{16})\.json\.gz$"
)
_EVENT_FILE = re.compile(
    rf"^{re.escape(PREFIX)}(?P<lineage>[a-z]+)/(?P<record>[a-z_]+)/"
    r"(?P<day>\d{4}-\d{2}-\d{2})\.parquet$"
)


def _check_date(value: str, what: str) -> None:
    try:
        ok = date.fromisoformat(value).isoformat() == value
    except ValueError:
        ok = False
    if not ok:
        raise KeyLayoutError(f"not an ISO {what}: {value!r}")


@dataclass(frozen=True)
class MatchIndexKey:
    index_version: int
    service_date: str
    build_id: str
    route_feed: str

    def __post_init__(self) -> None:
        if self.index_version < 1:
            raise KeyLayoutError(f"index version must be >= 1: {self.index_version}")
        _check_date(self.service_date, "service date")
        if not _HEX16.fullmatch(self.build_id):
            raise KeyLayoutError(f"not a 16-hex build id: {self.build_id!r}")
        if not _HEX16.fullmatch(self.route_feed):
            raise KeyLayoutError(f"not a 16-hex RouteFeed digest: {self.route_feed!r}")

    @property
    def key(self) -> str:
        return (
            f"{PREFIX}match_index/v{self.index_version}/{self.service_date}/"
            f"{self.build_id}-{self.route_feed}.json.gz"
        )


@dataclass(frozen=True)
class ServiceDayKey:
    """An event file of one lineage and one service day."""

    lineage: EventLineage
    record: ServiceDayRecord
    service_date: str

    def __post_init__(self) -> None:
        if self.lineage not in LINEAGES:
            raise KeyLayoutError(f"not an events lineage: {self.lineage!r}")
        if self.record not in SERVICE_DAY_RECORDS:
            raise KeyLayoutError(f"not a service-day record: {self.record!r}")
        _check_date(self.service_date, "service date")

    @property
    def key(self) -> str:
        return f"{PREFIX}{self.lineage}/{self.record}/{self.service_date}.parquet"


@dataclass(frozen=True)
class UtcDayKey:
    """A coverage or supply file of one lineage and one UTC day."""

    lineage: EventLineage
    record: UtcDayRecord
    day: str

    def __post_init__(self) -> None:
        if self.lineage not in LINEAGES:
            raise KeyLayoutError(f"not an events lineage: {self.lineage!r}")
        if self.record not in UTC_DAY_RECORDS:
            raise KeyLayoutError(f"not a UTC-day record: {self.record!r}")
        _check_date(self.day, "UTC day")

    @property
    def key(self) -> str:
        return f"{PREFIX}{self.lineage}/{self.record}/{self.day}.parquet"


EventsKey = MatchIndexKey | ServiceDayKey | UtcDayKey

_LINEAGE_BY_NAME: dict[str, EventLineage] = {n: n for n in LINEAGES}
_SERVICE_DAY_BY_NAME: dict[str, ServiceDayRecord] = {r: r for r in SERVICE_DAY_RECORDS}
_UTC_DAY_BY_NAME: dict[str, UtcDayRecord] = {r: r for r in UTC_DAY_RECORDS}


def parse_key(key: str) -> EventsKey:
    m = _MATCH_INDEX.match(key)
    if m is not None:
        return MatchIndexKey(
            index_version=int(m["version"]),
            service_date=m["day"],
            build_id=m["build"],
            route_feed=m["rf"],
        )
    m = _EVENT_FILE.match(key)
    lineage = _LINEAGE_BY_NAME.get(m["lineage"]) if m is not None else None
    if m is None or lineage is None:
        raise KeyLayoutError(f"not an events/s1 key: {key!r}")
    record = m["record"]
    if record in _SERVICE_DAY_BY_NAME:
        return ServiceDayKey(lineage, _SERVICE_DAY_BY_NAME[record], m["day"])
    if record in _UTC_DAY_BY_NAME:
        return UtcDayKey(lineage, _UTC_DAY_BY_NAME[record], m["day"])
    raise KeyLayoutError(f"not an events record: {record!r}")
