"""Network layer contracts (schema ``network/s1``).

The static network every later layer keys off: what exists (stations,
platforms, routes), how trains move through it (stopping patterns and the
directed stop-to-stop segments they are built from), and when it runs (services
resolved onto every service day of the feed's validity window).

Identity rules:
  * ``Feed.feed_id`` is the sha256 hex of the GTFS zip bytes, the same key the
    raw GTFS archive uses.
  * A ``Pattern`` is one (route, direction, ordered stop list). Trips with the
    same three share it; shapes are an attribute, not part of the identity.
  * A ``Segment`` is one directed platform-to-platform hop between consecutive
    stops of some pattern: ``f"{from_stop}>{to_stop}"``. An express hop that
    skips stations is its own segment.
  * Direction belongs to trips and patterns. ``direction_id`` is nullable
    because GTFS makes it optional; platforms carry only the NYC N/S code and
    the rider-facing label for that side of the station.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from momentarily.model.conventions import (
    Digest,
    GtfsSeconds,
    Model,
    Provenance,
    ServiceDate,
)

SCHEMA_ID = "network/s1"

DirectionCode = Literal["N", "S"]
Weekday = Literal[
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
]
DayType = Literal["weekday", "saturday", "sunday", "mixed", "none"]
Ada = Literal["none", "full", "partial"]


class System(Model):
    system_id: str = Field(description="Stable system key, e.g. 'nyct-subway'.")
    timezone: str = Field(description="IANA timezone of the service days.")
    agency_id: str
    agency_name: str


class Feed(Model):
    feed_id: Digest = Field(description="sha256 hex of the GTFS zip bytes.")
    feed_version: str | None
    valid_from: ServiceDate | None = Field(
        description="First service day the feed declares (feed_info)."
    )
    valid_to: ServiceDate | None = Field(
        description="Exclusive end: the day after feed_info's feed_end_date."
    )


class Route(Model):
    route_id: str
    short_name: str
    long_name: str
    route_type: int
    color: str | None
    text_color: str | None
    sort_order: int | None


class Station(Model):
    """A GTFS parent stop (location_type 1)."""

    station_id: str
    name: str
    lat: float
    lon: float
    complex_id: str | None
    borough: str | None
    ada: Ada | None


class Complex(Model):
    """Stations a rider can transfer between without leaving the system."""

    complex_id: str
    name: str
    station_ids: list[str]


class Platform(Model):
    """A GTFS child stop: where trains actually stop."""

    stop_id: str
    station_id: str
    name: str
    direction_code: DirectionCode | None = Field(
        description="NYC N/S suffix; null for a platform serving both directions."
    )
    direction_label: str | None = Field(
        description="Rider-facing label for this side of the station, e.g. 'Manhattan'."
    )


class Direction(Model):
    direction_id: Literal[0, 1] | None = Field(
        description="GTFS direction_id; null when the feed omits it."
    )
    code: DirectionCode | None = Field(
        description="The N/S suffix shared by every stop, or null if they differ."
    )
    label: str | None = Field(
        description=(
            "Pattern-wide label. Null in s1: labels vary along a line, so they "
            "live on Platform."
        )
    )


class Pattern(Model):
    pattern_id: str
    route_id: str
    direction: Direction
    headsign: str | None = Field(
        description="Most common trip headsign (ties broken lexicographically)."
    )
    stops: list[str] = Field(min_length=2, description="Platform stop_ids in order.")
    shape_ids: list[str]
    n_trips: int = Field(ge=1)


class Segment(Model):
    segment_id: str = Field(description="f'{from_stop}>{to_stop}'.")
    from_stop: str
    to_stop: str
    route_ids: list[str]
    pattern_ids: list[str]


class PatternSegment(Model):
    pattern_id: str
    seq: int = Field(ge=0, description="0-based hop index within the pattern.")
    segment_id: str


class Service(Model):
    """One GTFS service_id, exactly as declared (never collapsed to a day type)."""

    service_id: str
    weekdays: list[Weekday]
    start: ServiceDate | None = Field(description="First day of the weekly rule.")
    end: ServiceDate | None = Field(
        description="Exclusive end: the day after calendar.txt's end_date."
    )
    added_dates: list[ServiceDate]
    removed_dates: list[ServiceDate]


class ServiceDay(Model):
    date: ServiceDate
    active_service_ids: list[str]
    day_type: DayType = Field(
        description=(
            "weekday/saturday/sunday when every active service's weekly rule is "
            "exactly that; none when nothing runs; otherwise mixed."
        )
    )
    is_exception: bool = Field(description="calendar_dates.txt touches this date.")


StopTime = tuple[str, GtfsSeconds, GtfsSeconds]


class ScheduledTrip(Model):
    trip_id: str
    route_id: str
    pattern_id: str
    service_id: str
    headsign: str | None
    shape_id: str | None
    direction: Direction
    stop_times: list[StopTime] = Field(
        description="[stop_id, arrival, departure] in stop_sequence order."
    )


class NetworkSnapshot(Model):
    build_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    provenance: Provenance
    system: System
    feed: Feed
    routes: list[Route]
    stations: list[Station]
    complexes: list[Complex]
    platforms: list[Platform]
    patterns: list[Pattern]
    segments: list[Segment]
    pattern_segments: list[PatternSegment]
    services: list[Service]
    service_days: list[ServiceDay]
    trips: list[ScheduledTrip]


class IndexEntry(Model):
    build_id: str
    content_digest: Digest = Field(
        description="blake3 of the build's canonical uncompressed JSON."
    )
    feed_version: str | None
    valid_from: ServiceDate | None = Field(
        description="First service day the feed declares (inclusive)."
    )
    valid_to: ServiceDate | None = Field(
        description="Exclusive end: the day after feed_info's feed_end_date."
    )
    inputs: dict[str, Digest]
    applied_retroactively: list[str]


class NetworkIndex(Model):
    """Every network build in an output directory, keyed by feed_id."""

    schema_: str = Field(alias="schema")
    feeds: dict[str, IndexEntry]


class DayFeed(Model):
    """Which feed served one UTC collection day of the traversal archive."""

    feed_id: Digest | None = Field(
        description="The single stamped digest; null unless verified."
    )
    verified: bool
    reason: str | None = Field(description="Why the day is unverified.")
    feed_version: str | None
    inferred_feed_id: Digest | None = Field(
        description="Unverified days: the archived feed whose version matches."
    )


class DayFeedMap(Model):
    days: dict[str, DayFeed] = Field(description="Keyed by UTC date YYYY-MM-DD.")
