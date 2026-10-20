"""Raw observation contracts (schema ``observations/s1``).

Three groups:

  * Collector metadata, written by the Worker and live only: ``CollectorTick``
    (one per invocation, holding a ``FeedFetch`` per attempt) and
    ``TickBundleIndex`` (the index member of a GTFS-RT tick bundle). Every
    field is required and known; the raw payload bytes they point at are the
    source of truth.
  * Observation records, derived offline: ``VehicleObservation``,
    ``TripUpdateObservation``, ``FeedHeaderObservation``, ``AlertPresence``,
    ``AlertVersion`` and ``EquipmentObservation``, stored as parquet (see
    ``columns``). Each carries a ``lineage``: ``RawLineage`` (decoded from a
    stored payload, named by its bundle or blob key) or, where a legacy
    archive can be backfilled, ``LegacyLineage`` (which archive, which object,
    and the legacy period). A consumer filters legacy rows on
    ``lineage.kind`` alone.
  * Legacy-only records, each with a ``LegacyLineage``:
    ``LegacyRouteAlertState`` (``v1/predictions``), ``LegacyFetchOutcome``
    (per-source fetch results the old archives kept) and
    ``LegacyCollectorHealth`` (``archive/health``).

Every upstream field has three states:

  * a value: the payload carried it;
  * ``None``: the payload omitted it (GTFS-RT is proto2, so presence is
    knowable). ``None`` is never zero, false, or an empty string;
  * ``Unknown(reason)``: a legacy archive did not keep it, or (only where a
    record lists it in ``DECODE_UNKNOWN``) the decoder could not read it. A
    raw-lineage record holding any other ``Unknown`` is rejected.

Identities stay as published: route ids are never folded (``6X``, ``SS``),
realtime-only stop ids (``R60N``) are kept, ``start_date`` stays a
``YYYYMMDD`` string. This layer assigns no service day and matches nothing to
the network.

Time fields: ``scheduled_at`` is the cron minute an attempt served;
``started_at_ms``/``finished_at_ms`` the Worker clock around the request;
``observed_at`` = ``finished_at_ms // 1000``, the instant we learned the
content. Upstream timestamps (header, vehicle, trip update, alert) are kept
verbatim and may lie in the future.

Validation failures raise ``PydanticCustomError`` with a stable error type
(``fetch_storage``, ``alert_bounds``, ...), so callers and tests match on the
type, not the wording.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Annotated, Any, ClassVar, Literal, LiteralString, get_args

from pydantic import BaseModel, Field, JsonValue, model_validator
from pydantic_core import PydanticCustomError

from momentarily.model.conventions import (
    Instant,
    Interval,
    Model,
    Provenance,
    Unknown,
    UtcDate,
)
from momentarily.model.observations.keys import (
    BlobKey,
    BundleKey,
    DerivedDayKey,
    DerivedHourKey,
    GtfsArchiveKey,
    KeyLayoutError,
    LegacyPredictionsKey,
    ObservationKey,
    parse_key,
    utc_day,
)

SCHEMA_ID = "observations/s1"

InstantMs = Annotated[int, Field(ge=0, description="UTC epoch milliseconds.")]
Sha256 = Annotated[
    str, Field(pattern=r"^[0-9a-f]{64}$", description="Lowercase sha256 hex.")
]

GtfsRtSource = Literal["ace", "bdfm", "g", "jz", "nqrw", "l", "numbered", "si"]
SourceId = Literal[
    GtfsRtSource,
    "alerts",
    "ene_current",
    "ene_upcoming",
    "ene_equipments",
    "gtfs_static",
    "stations",
    "entrances",
]
AlertSource = Literal["alerts", "numbered"]
EquipmentSource = Literal["ene_current", "ene_upcoming"]
GTFSRT_SOURCES: tuple[str, ...] = get_args(GtfsRtSource)

LegacyArchive = Literal[
    "archive/trace",
    "archive/trip_updates",
    "archive/vehicles",
    "archive/alerts",
    "archive/alerts_liveness",
    "archive/ene",
    "archive/health",
    "v1/predictions",
]

FetchOutcome = Literal[
    "ok",  # HTTP 2xx, body read, parsed as the expected format
    "empty",  # HTTP 2xx, zero entities / records (payload still stored)
    "unparseable",  # HTTP 2xx, body is not valid protobuf / JSON (payload stored)
    "http_error",  # non-2xx; the body is not read
    "timeout",  # the fetch budget expired (headers may have arrived)
    "network_error",  # DNS / connect / reset (headers may have arrived)
]
_BODY_OUTCOMES = ("ok", "empty", "unparseable")
_PARSED_OUTCOMES = ("ok", "empty")


def _fail(error_type: LiteralString, message: str) -> PydanticCustomError:
    """A validation error with a stable type; ``message`` is context, never
    a template."""
    return PydanticCustomError(error_type, "{detail}", {"detail": message})


def _is_2xx(status: int | None) -> bool:
    return status is not None and 200 <= status < 300


def _parse_key(key: str, error_type: LiteralString) -> ObservationKey:
    try:
        return parse_key(key)
    except KeyLayoutError as err:
        raise _fail(error_type, str(err)) from err


# --- collector metadata (Worker, live only) ---------------------------------


class FeedFetch(Model):
    """One HTTP attempt against one source, whatever its outcome."""

    fetch_id: str = Field(
        description="f'{source}@{started_at_ms}'; unique per attempt."
    )
    source: SourceId
    url: str
    method: Literal["GET", "HEAD"] = Field(description="HEAD only for gtfs_static.")
    scheduled_at: Instant = Field(description="The cron minute this attempt served.")
    started_at_ms: InstantMs
    finished_at_ms: InstantMs | None = Field(description="null exactly on timeout.")
    outcome: FetchOutcome
    http_status: int | None = Field(
        description=(
            "null when no response headers arrived. A timeout or network error "
            "after 2xx headers keeps the status that arrived."
        )
    )
    content_type: str | None = Field(description="null when no headers arrived.")
    http_date: Instant | None = Field(description="The response Date header.")
    http_last_modified: Instant | None
    http_etag: str | None
    body_bytes: int | None = Field(
        ge=0, description="Decoded body length; null when no body was read."
    )
    payload_sha256: Sha256 | None = Field(
        description="sha256 of the decoded body bytes; null when no body was read."
    )
    payload_key: str | None = Field(
        description="R2 key holding the bytes; null if not stored."
    )
    payload_member: int | None = Field(
        ge=1,
        description="Member index inside a tick bundle (0 is the index); null for blobs.",
    )
    header_timestamp: Instant | None = Field(
        description="The payload's own build time (GTFS-RT or alerts header)."
    )
    entity_count: int | None = Field(
        ge=0,
        description="Top-level entities (GTFS-RT, alerts) or records (ENE, Socrata).",
    )
    store_error: str | None = Field(
        description="Why a body that was read was not stored; set exactly when payload_key is null."
    )

    @model_validator(mode="after")
    def _consistent(self) -> FeedFetch:
        if self.fetch_id != f"{self.source}@{self.started_at_ms}":
            raise _fail("fetch_identity", "fetch_id is not source@started_at_ms")
        if self.method == "HEAD" and self.source != "gtfs_static":
            raise _fail("fetch_identity", "only gtfs_static is fetched with HEAD")
        if (self.finished_at_ms is None) != (self.outcome == "timeout"):
            raise _fail("fetch_timing", "finished_at_ms is null exactly on timeout")
        if self.finished_at_ms is not None and self.finished_at_ms < self.started_at_ms:
            raise _fail("fetch_timing", "finished_at_ms precedes started_at_ms")
        self._check_status()
        self._check_body()
        if self.outcome not in _PARSED_OUTCOMES and (
            self.entity_count is not None or self.header_timestamp is not None
        ):
            raise _fail(
                "fetch_parse",
                "only a parsed payload has entity_count or header_timestamp",
            )
        if self.outcome == "empty" and self.entity_count not in (None, 0):
            raise _fail("fetch_parse", "outcome empty means entity_count 0")
        if self.outcome == "ok" and self.entity_count == 0:
            raise _fail("fetch_parse", "entity_count 0 is outcome empty, not ok")
        return self

    def _check_status(self) -> None:
        status = self.http_status
        if self.outcome in _BODY_OUTCOMES and not _is_2xx(status):
            raise _fail(
                "fetch_status", f"outcome {self.outcome} needs a 2xx http_status"
            )
        if self.outcome == "http_error" and (status is None or _is_2xx(status)):
            raise _fail(
                "fetch_status", "outcome http_error needs a non-2xx http_status"
            )
        if self.outcome in ("timeout", "network_error") and not (
            status is None or _is_2xx(status)
        ):
            raise _fail(
                "fetch_status",
                f"a non-2xx status is outcome http_error, not {self.outcome}",
            )
        headers = (
            self.content_type,
            self.http_date,
            self.http_last_modified,
            self.http_etag,
        )
        if status is None and any(h is not None for h in headers):
            raise _fail("fetch_headers", "response headers without an http_status")

    def _check_body(self) -> None:
        body = (self.body_bytes, self.payload_sha256)
        storage = (self.payload_key, self.payload_member, self.store_error)
        read_body = self.outcome in _BODY_OUTCOMES and self.method == "GET"
        if not read_body:
            if any(v is not None for v in (*body, *storage)):
                raise _fail(
                    "fetch_body", f"a {self.method} {self.outcome} read no body"
                )
            if self.method == "HEAD" and self.outcome not in (
                "ok",
                "http_error",
                "timeout",
                "network_error",
            ):
                raise _fail("fetch_body", f"a HEAD cannot be {self.outcome}")
            return
        if any(v is None for v in body):
            raise _fail(
                "fetch_body", "a body that was read needs body_bytes and payload_sha256"
            )
        if (self.payload_key is None) == (self.store_error is None):
            raise _fail(
                "fetch_storage",
                "a read body has exactly one of payload_key, store_error",
            )
        if self.payload_key is None:
            if self.payload_member is not None:
                raise _fail("fetch_storage", "payload_member without payload_key")
            return
        key = _parse_key(self.payload_key, "fetch_storage")
        if isinstance(key, BundleKey):
            if self.source not in GTFSRT_SOURCES or self.payload_member is None:
                raise _fail(
                    "fetch_storage",
                    "a bundle holds GTFS-RT payloads, addressed by member",
                )
            if key.scheduled_at != self.scheduled_at:
                raise _fail("fetch_storage", "payload bundle belongs to another tick")
        elif isinstance(key, BlobKey | GtfsArchiveKey):
            if self.source in GTFSRT_SOURCES or self.payload_member is not None:
                raise _fail(
                    "fetch_storage", "GTFS-RT payloads live in bundles, not blobs"
                )
            if key.sha256 != self.payload_sha256:
                raise _fail(
                    "fetch_storage", "payload_key digest differs from payload_sha256"
                )
            if isinstance(key, GtfsArchiveKey) != (self.source == "gtfs_static"):
                raise _fail("fetch_storage", "only gtfs_static lives in archive/gtfs/")
            finished = self.finished_at_ms
            if (
                isinstance(key, BlobKey)
                and finished is not None
                and key.day != utc_day(finished // 1000)
            ):
                raise _fail(
                    "fetch_storage", "a blob is filed under its fetch's UTC day"
                )
        else:
            raise _fail("fetch_storage", "payload_key names no payload store")


class CollectorBuild(Model):
    code_sha: str
    code_dirty: bool | None = Field(description="null when the build did not say.")


def _check_fetches(
    fetches: list[FeedFetch],
    scheduled_at: int,
    started_at_ms: int,
    error_type: LiteralString,
) -> None:
    ids = [f.fetch_id for f in fetches]
    if len(set(ids)) != len(ids):
        raise _fail(error_type, "fetch_id repeats within a tick")
    own = BundleKey(scheduled_at, started_at_ms).key
    for f in fetches:
        if f.scheduled_at != scheduled_at:
            raise _fail(error_type, f"{f.fetch_id} serves another scheduled_at")
        if f.started_at_ms < started_at_ms:
            raise _fail(error_type, f"{f.fetch_id} started before its tick")
        if f.payload_member is not None and f.payload_key != own:
            raise _fail(
                error_type, f"{f.fetch_id} points into another invocation's bundle"
            )
    members = [f.payload_member for f in fetches if f.payload_member is not None]
    if len(set(members)) != len(members):
        raise _fail(error_type, "payload_member repeats within a bundle")


class CollectorTick(Model):
    """One Worker invocation. No tick for a minute means the collector did
    not run; a failed source shows as a fetch with its outcome."""

    schema_: Literal["observations/s1"] = Field(alias="schema")
    tick_id: str = Field(
        description="f'{scheduled_at}-{started_at_ms}'; retries keep both."
    )
    scheduled_at: Instant
    started_at_ms: InstantMs
    finished_at_ms: InstantMs
    collector: CollectorBuild
    planned_sources: list[SourceId] = Field(
        description="What this tick was due to fetch."
    )
    fetches: list[FeedFetch]

    @model_validator(mode="after")
    def _consistent(self) -> CollectorTick:
        if self.tick_id != f"{self.scheduled_at}-{self.started_at_ms}":
            raise _fail("tick_identity", "tick_id is not scheduled_at-started_at_ms")
        if self.finished_at_ms < self.started_at_ms:
            raise _fail("tick_timing", "finished_at_ms precedes started_at_ms")
        if len(set(self.planned_sources)) != len(self.planned_sources):
            raise _fail("tick_fetches", "planned_sources repeats a source")
        _check_fetches(
            self.fetches, self.scheduled_at, self.started_at_ms, "tick_fetches"
        )
        for f in self.fetches:
            if f.finished_at_ms is not None and f.finished_at_ms > self.finished_at_ms:
                raise _fail("tick_timing", f"{f.fetch_id} finished after its tick")
        unplanned = {f.source for f in self.fetches} - set(self.planned_sources)
        if unplanned:
            raise _fail(
                "tick_fetches", f"fetched sources not planned: {sorted(unplanned)}"
            )
        return self


class TickBundleIndex(Model):
    """Member 0 of a GTFS-RT tick bundle: the tick's GTFS-RT fetches, so the
    bundle describes itself when the tick record is lost."""

    schema_: Literal["observations/s1"] = Field(alias="schema")
    tick_id: str
    scheduled_at: Instant
    started_at_ms: InstantMs
    fetches: list[FeedFetch] = Field(
        description="Every GTFS-RT attempt of the tick, stored or not."
    )

    @model_validator(mode="after")
    def _consistent(self) -> TickBundleIndex:
        if self.tick_id != f"{self.scheduled_at}-{self.started_at_ms}":
            raise _fail("tick_identity", "tick_id is not scheduled_at-started_at_ms")
        if any(f.source not in GTFSRT_SOURCES for f in self.fetches):
            raise _fail("tick_fetches", "a bundle index lists GTFS-RT fetches only")
        _check_fetches(
            self.fetches, self.scheduled_at, self.started_at_ms, "tick_fetches"
        )
        return self


# --- lineage ------------------------------------------------------------------


def _real_date(value: str) -> None:
    try:
        ok = date.fromisoformat(value).isoformat() == value
    except ValueError:
        ok = False
    if not ok:
        raise _fail("legacy_period", f"{value} is not a calendar date")


class LegacyPeriod(Model):
    """The UTC days ``[start, end)`` for which a legacy archive stands in for
    the recorder's records."""

    start: UtcDate = Field(description="First UTC day the legacy archive covers.")
    end: UtcDate | None = Field(
        description="Exclusive: the first UTC day the recorder wrote the replacing record; null while it has not."
    )

    @model_validator(mode="after")
    def _ordered(self) -> LegacyPeriod:
        _real_date(self.start)
        if self.end is not None:
            _real_date(self.end)
            if self.end <= self.start:
                raise _fail("legacy_period", "legacy period end is not after its start")
        return self

    def contains(self, day: str) -> bool:
        return self.start <= day and (self.end is None or day < self.end)


class RawLineage(Model):
    """Decoded from a stored raw payload."""

    kind: Literal["raw"]
    payload_key: str = Field(description="The tick bundle or blob holding the bytes.")
    payload_member: int | None = Field(
        ge=1, description="Bundle member; null for a blob."
    )
    payload_sha256: Sha256

    @model_validator(mode="after")
    def _consistent(self) -> RawLineage:
        key = _parse_key(self.payload_key, "lineage")
        if isinstance(key, BundleKey):
            if self.payload_member is None:
                raise _fail("lineage", "a bundle payload needs its member index")
        elif isinstance(key, BlobKey):
            if self.payload_member is not None:
                raise _fail("lineage", "a blob has no member index")
            if key.sha256 != self.payload_sha256:
                raise _fail("lineage", "payload_key digest differs from payload_sha256")
        else:
            raise _fail("lineage", "raw lineage names a tick bundle or a blob")
        return self

    def blob_day(self) -> str | None:
        key = parse_key(self.payload_key)
        return key.day if isinstance(key, BlobKey) else None


class LegacyLineage(Model):
    """Read from a legacy archive object. What it lacks is ``Unknown`` on the
    record; layer 3 labels anything it derives from it as legacy-derived."""

    kind: Literal["legacy"]
    archive: LegacyArchive
    source_key: str = Field(description="The legacy object the record came from.")
    row_index: int | None = Field(
        ge=0, description="Position of the row in that object; null when it holds one."
    )
    period: LegacyPeriod

    @model_validator(mode="after")
    def _consistent(self) -> LegacyLineage:
        if not self.source_key.startswith(f"{self.archive}/"):
            raise _fail("lineage", f"{self.source_key} is not under {self.archive}/")
        return self


Lineage = Annotated[RawLineage | LegacyLineage, Field(discriminator="kind")]


def _unknown_paths(record: BaseModel, prefix: str = "") -> list[str]:
    """Dotted paths of every ``Unknown`` value in ``record``, nested models
    included."""
    found: list[str] = []
    for name in type(record).model_fields:
        value = getattr(record, name)
        if isinstance(value, Unknown):
            found.append(prefix + name)
        elif isinstance(value, BaseModel) and not isinstance(
            value, RawLineage | LegacyLineage
        ):
            found += _unknown_paths(value, f"{prefix}{name}.")
    return found


def _check_lineage(
    record: BaseModel,
    lineage: RawLineage | LegacyLineage,
    at: int | None,
    legacy_archives: tuple[str, ...],
    decode_unknown: frozenset[str],
) -> None:
    """Raw records hold Unknown only where decoding can fail, and their blob
    is filed under the day of ``at``; legacy records come from an allowed
    archive and fall inside its period."""
    if isinstance(lineage, RawLineage):
        legacy_only = sorted(set(_unknown_paths(record)) - decode_unknown)
        if legacy_only:
            raise _fail(
                "legacy_only_unknown",
                f"raw records cannot hold Unknown in: {', '.join(legacy_only)}",
            )
        day = lineage.blob_day()
        if at is not None and day is not None and day != utc_day(at):
            raise _fail("lineage", "the payload blob is filed under another day")
        return
    if lineage.archive not in legacy_archives:
        raise _fail("lineage", f"{lineage.archive} cannot back this record")
    if at is not None and not lineage.period.contains(utc_day(at)):
        raise _fail("legacy_period", f"{utc_day(at)} is outside the legacy period")


def _check_source(fetch_id: str | Unknown, source: str | Unknown, lineage: Any) -> None:
    if isinstance(source, str):
        if isinstance(fetch_id, str) and not fetch_id.startswith(f"{source}@"):
            raise _fail(
                "fetch_identity", "fetch_id does not belong to the record's source"
            )
        if isinstance(lineage, RawLineage):
            bundled = parse_key(lineage.payload_key)
            if isinstance(bundled, BundleKey) != (source in GTFSRT_SOURCES):
                raise _fail(
                    "lineage", "GTFS-RT payloads live in bundles, others in blobs"
                )


# --- observation records ---------------------------------------------------


class TripDescriptorObs(Model):
    """A GTFS-RT TripDescriptor with the NYCT extension, as published.
    ``Unknown`` only on legacy rows (the trace kept trip_id alone)."""

    trip_id: str | None
    route_id: str | None | Unknown = Field(
        description="As published: '6X', 'SS', never folded."
    )
    start_time: str | None | Unknown = Field(
        description="Published 'HH:MM:SS'; may exceed 24h."
    )
    start_date: str | None | Unknown = Field(
        description="Published 'YYYYMMDD'; not a ServiceDate."
    )
    schedule_relationship: int | None | Unknown
    direction_id: Literal[0, 1] | None | Unknown
    nyct_train_id: str | None | Unknown
    nyct_is_assigned: bool | None | Unknown = Field(
        description="null when the extension omits it; never read as false."
    )
    nyct_direction: Literal[1, 2, 3, 4] | None | Unknown


class VehicleObservation(Model):
    """One VehiclePosition entity. A placeholder for an undispatched trip is
    recorded as published: no status, future timestamp. Legacy rows come
    from ``archive/trace``."""

    LEGACY_ARCHIVES: ClassVar[tuple[str, ...]] = ("archive/trace",)
    DECODE_UNKNOWN: ClassVar[frozenset[str]] = frozenset()

    lineage: Lineage
    fetch_id: str | Unknown = Field(
        description="The FeedFetch decoded; Unknown on legacy rows."
    )
    source: GtfsRtSource | Unknown
    observed_at: Instant
    entity_index: Annotated[int, Field(ge=0)] | Unknown = Field(
        description="Position of the entity in the feed."
    )
    entity_id: str | Unknown
    trip: TripDescriptorObs | None = Field(
        description="null when the vehicle has no trip."
    )
    stop_id: str | None = Field(description="As published, realtime-only ids kept.")
    current_stop_sequence: int | None | Unknown
    current_status: Literal[0, 1, 2] | None | Unknown = Field(
        description="0 INCOMING_AT, 1 STOPPED_AT, 2 IN_TRANSIT_TO; null when omitted."
    )
    vehicle_timestamp: Instant | None = Field(
        description="Verbatim; may be in the future."
    )
    congestion_level: int | None | Unknown
    unmodeled_fields: list[str] | Unknown = Field(
        description="Field paths present but not decoded, e.g. '2' (position)."
    )

    @model_validator(mode="after")
    def _consistent(self) -> VehicleObservation:
        _check_lineage(
            self,
            self.lineage,
            self.observed_at,
            self.LEGACY_ARCHIVES,
            self.DECODE_UNKNOWN,
        )
        _check_source(self.fetch_id, self.source, self.lineage)
        return self


class StopTimePrediction(Model):
    index: int = Field(
        ge=0, description="Position in the trip's stop_time_update list."
    )
    stop_id: str | None
    stop_sequence: int | None
    arrival_time: Instant | None
    arrival_delay: int | None = Field(description="Seconds, signed: negative is early.")
    arrival_uncertainty: int | None
    departure_time: Instant | None
    departure_delay: int | None = Field(
        description="Seconds, signed: negative is early."
    )
    departure_uncertainty: int | None
    schedule_relationship: int | None
    departure_occupancy_status: int | None
    nyct_scheduled_track: str | None
    nyct_actual_track: str | None


class TripUpdateObservation(Model):
    """One TripUpdate entity. Arrival and departure stay separate; nothing is
    truncated or filtered. Raw only: no legacy archive keeps trip updates."""

    DECODE_UNKNOWN: ClassVar[frozenset[str]] = frozenset()

    lineage: RawLineage
    fetch_id: str
    source: GtfsRtSource
    observed_at: Instant
    entity_index: int = Field(ge=0, description="Position of the entity in the feed.")
    entity_id: str
    trip: TripDescriptorObs
    trip_update_timestamp: Instant | None
    delay: int | None = Field(description="Seconds, signed: negative is early.")
    stop_times: list[StopTimePrediction]
    unmodeled_fields: list[str]

    @model_validator(mode="after")
    def _consistent(self) -> TripUpdateObservation:
        _check_lineage(self, self.lineage, self.observed_at, (), self.DECODE_UNKNOWN)
        _check_source(self.fetch_id, self.source, self.lineage)
        if [s.index for s in self.stop_times] != list(range(len(self.stop_times))):
            raise _fail(
                "stop_time_order", "stop_times must be in feed order, indexed from 0"
            )
        return self


class FeedHeaderObservation(Model):
    """The FeedHeader of one GTFS-RT payload. Raw only."""

    lineage: RawLineage
    fetch_id: str
    source: GtfsRtSource
    observed_at: Instant
    gtfs_realtime_version: str | None
    incrementality: int | None
    header_timestamp: Instant | None
    nyct_version: str | None
    trip_replacement_periods: dict[str, Instant | None] = Field(
        description="route_id -> end of the replacement window, as published."
    )

    @model_validator(mode="after")
    def _consistent(self) -> FeedHeaderObservation:
        _check_lineage(self, self.lineage, self.observed_at, (), frozenset())
        _check_source(self.fetch_id, self.source, self.lineage)
        return self


def _js_numbers(value: JsonValue) -> JsonValue:
    """Integral floats as ints, as JavaScript's JSON.stringify writes them."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, dict):
        return {k: _js_numbers(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_js_numbers(v) for v in value]
    return value


def canonical_entity_json(entity: dict[str, JsonValue]) -> bytes:
    """The canonical bytes an alert's ``content_digest`` hashes: sorted keys,
    no whitespace, non-ASCII kept, UTF-8, integral floats written as ints
    (``1.0`` -> ``1``, as JSON.stringify does, so the live feed and the
    re-serialized legacy archive agree). NaN and infinities are rejected.
    Non-integral floats below 1e-4 or from 1e16 up are formatted differently
    by Python and JavaScript; alert entities carry none [inference]."""
    return json.dumps(
        _js_numbers(entity),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def alert_content_digest(entity: dict[str, JsonValue]) -> str:
    return hashlib.sha256(canonical_entity_json(entity)).hexdigest()


class AlertEntry(Model):
    alert_id: str
    content_digest: Sha256 = Field(description="sha256 of the entity's canonical JSON.")
    updated_at: Instant | None


class AlertPresence(Model):
    """The full alert set at one fetch with outcome ok or empty. Comparing
    consecutive presences bounds when an alert left the feed. Raw only."""

    lineage: RawLineage
    fetch_id: str
    source: AlertSource
    observed_at: Instant
    header_timestamp: Instant | None
    entries: list[AlertEntry] = Field(
        description="Sorted by alert_id then digest; [] = the feed said none (outcome empty)."
    )

    @model_validator(mode="after")
    def _consistent(self) -> AlertPresence:
        _check_lineage(self, self.lineage, self.observed_at, (), frozenset())
        _check_source(self.fetch_id, self.source, self.lineage)
        keys = [(e.alert_id, e.content_digest) for e in self.entries]
        if keys != sorted(keys) or len(set(keys)) != len(keys):
            raise _fail("presence_order", "entries must be sorted and distinct")
        return self


def _known(value: int | Unknown | None) -> int | None:
    return value if isinstance(value, int) else None


GoneKind = Literal["removed", "superseded"]


class AlertVersion(Model):
    """One distinct alert content, and the window in which it was seen.

    "Fetch" below means a fetch with outcome ok or empty: an empty feed said
    there were no alerts, which is evidence, not a gap. The version left the
    feed in ``(last_seen, gone_by]``. Where ``unobserved`` overlaps that span,
    the end is only known to lie in the wider span. ``gone_kind``
    ``superseded`` means the same alert id continued with other content.
    ``window`` is the presence window the index was built over (a UTC day
    plus carry). Legacy versions come from ``archive/alerts``.
    """

    LEGACY_ARCHIVES: ClassVar[tuple[str, ...]] = ("archive/alerts",)
    DECODE_UNKNOWN: ClassVar[frozenset[str]] = frozenset()

    lineage: Lineage = Field(description="Raw: the payload of the first sighting.")
    source: AlertSource
    alert_id: str
    content_digest: Sha256 = Field(description="alert_content_digest(entity).")
    entity: dict[str, JsonValue] = Field(description="Verbatim feed entity.")
    alert_type: str | None = Field(description="The entity's mercury alert_type.")
    created_at: Instant | None = Field(description="The entity's mercury created_at.")
    updated_at: Instant | None = Field(description="The entity's mercury updated_at.")
    absent_before: Instant | None | Unknown = Field(
        description="Last fetch before first_seen without this version; null = first fetch of the window."
    )
    first_seen: Instant | Unknown
    last_seen: Instant | Unknown
    gone_by: Instant | None | Unknown = Field(
        description="First fetch after last_seen without this version; null = still present at window end."
    )
    gone_kind: GoneKind | None | Unknown = Field(
        description="null exactly when gone_by is null. With gone_by Unknown: Unknown, or superseded when a later updated_at of the same id proves it."
    )
    unobserved: list[Interval] | Unknown = Field(
        description="Sorted, disjoint gaps inside [first_seen, gone_by) with no fetch."
    )
    window: Interval | Unknown

    @model_validator(mode="after")
    def _consistent(self) -> AlertVersion:
        at = _known(self.first_seen)
        _check_lineage(
            self, self.lineage, at, self.LEGACY_ARCHIVES, self.DECODE_UNKNOWN
        )
        if isinstance(self.lineage, RawLineage):
            bundled = isinstance(parse_key(self.lineage.payload_key), BundleKey)
            if bundled != (self.source == "numbered"):
                raise _fail(
                    "lineage", "numbered alerts live in bundles, camsys alerts in blobs"
                )
        self._check_entity()
        self._check_bounds()
        self._check_gaps()
        return self

    def _check_entity(self) -> None:
        if self.content_digest != alert_content_digest(self.entity):
            raise _fail("alert_digest", "content_digest is not the digest of entity")
        entity_id = self.entity.get("id")
        if entity_id is not None and entity_id != self.alert_id:
            raise _fail("alert_entity", "entity id differs from alert_id")
        alert = self.entity.get("alert")
        mercury = (
            alert.get("transit_realtime.mercury_alert")
            if isinstance(alert, dict)
            else None
        )
        published = mercury if isinstance(mercury, dict) else {}
        for name in ("alert_type", "created_at", "updated_at"):
            if getattr(self, name) != published.get(name):
                raise _fail(
                    "alert_entity", f"{name} differs from the entity's mercury_alert"
                )

    def _check_bounds(self) -> None:
        before, first = _known(self.absent_before), _known(self.first_seen)
        last, gone = _known(self.last_seen), _known(self.gone_by)
        if before is not None and first is not None and not before < first:
            raise _fail("alert_bounds", "absent_before must precede first_seen")
        if first is not None and last is not None and not first <= last:
            raise _fail("alert_bounds", "last_seen precedes first_seen")
        if last is not None and gone is not None and not last < gone:
            raise _fail("alert_bounds", "gone_by must follow last_seen")
        if first is not None and gone is not None and not first < gone:
            raise _fail("alert_bounds", "gone_by must follow first_seen")
        if isinstance(self.window, Interval):
            for name, value in (("first_seen", first), ("last_seen", last)):
                if (
                    value is not None
                    and not self.window.start <= value < self.window.end
                ):
                    raise _fail("alert_bounds", f"{name} lies outside the window")
        kind = self.gone_kind
        if (self.gone_by is None) != (kind is None):
            raise _fail("alert_gone_kind", "gone_kind is null exactly when gone_by is")
        if isinstance(self.gone_by, int) and isinstance(kind, Unknown):
            raise _fail("alert_gone_kind", "an observed gone_by has a known gone_kind")
        if isinstance(self.gone_by, Unknown) and kind == "removed":
            raise _fail(
                "alert_gone_kind", "with gone_by Unknown only superseded can be proven"
            )

    def _check_gaps(self) -> None:
        if isinstance(self.unobserved, Unknown):
            return
        first, gone = _known(self.first_seen), _known(self.gone_by)
        window = self.window if isinstance(self.window, Interval) else None
        lo = first if first is not None else (window.start if window else None)
        hi = gone if gone is not None else (window.end if window else None)
        previous_end: int | None = None
        for gap in self.unobserved:
            if gap.end <= gap.start:
                raise _fail("alert_gaps", "an unobserved gap is empty")
            if previous_end is not None and gap.start <= previous_end:
                raise _fail("alert_gaps", "unobserved gaps must be sorted and disjoint")
            if (lo is not None and gap.start < lo) or (hi is not None and gap.end > hi):
                raise _fail("alert_gaps", "an unobserved gap lies outside the sighting")
            previous_end = gap.end


class EquipmentObservation(Model):
    """One record of one ENE outage payload, verbatim, with its two published
    local times parsed. Legacy rows come from ``archive/ene``."""

    LEGACY_ARCHIVES: ClassVar[tuple[str, ...]] = ("archive/ene",)
    DECODE_UNKNOWN: ClassVar[frozenset[str]] = frozenset(
        {"outage_start", "estimated_return"}
    )

    lineage: Lineage
    fetch_id: str | Unknown = Field(description="Unknown on legacy rows.")
    source: EquipmentSource
    observed_at: Instant
    record_index: int = Field(
        ge=0, description="Position of the record in the payload."
    )
    equipment_id: str
    record: dict[str, str] = Field(description="The record verbatim.")
    outage_start_local: str | None = Field(
        description="'outagedate' as published (ET wall time)."
    )
    estimated_return_local: str | None = Field(
        description="'estimatedreturntoservice' as published (ET wall time)."
    )
    outage_start: Instant | None | Unknown = Field(
        description="Parsed in America/New_York; Unknown if ambiguous or unparseable."
    )
    estimated_return: Instant | None | Unknown

    @model_validator(mode="after")
    def _consistent(self) -> EquipmentObservation:
        _check_lineage(
            self,
            self.lineage,
            self.observed_at,
            self.LEGACY_ARCHIVES,
            self.DECODE_UNKNOWN,
        )
        _check_source(self.fetch_id, self.source, self.lineage)
        published = self.record.get("equipment")
        if published is not None and published != self.equipment_id:
            raise _fail("equipment_id", "record equipment differs from equipment_id")
        return self


# --- legacy-only records ---------------------------------------------------


def _check_legacy(
    lineage: LegacyLineage, archives: tuple[str, ...], at: int, source_key: str | None
) -> None:
    if lineage.archive not in archives:
        raise _fail("lineage", f"{lineage.archive} cannot back this record")
    if not lineage.period.contains(utc_day(at)):
        raise _fail("legacy_period", f"{utc_day(at)} is outside the legacy period")
    if source_key is not None and lineage.source_key != source_key:
        raise _fail("legacy_key", f"the record at {at} lives in {source_key}")


class RouteAlertGranularity(Model):
    """What a ``v1/predictions`` row holds: per route, per 5-minute tick, the
    single highest-priority alert type, with no alert identity."""

    scope: Literal["route"]
    tick_seconds: Literal[300]
    content: Literal["primary_alert_type_presence"]


class LegacyRouteAlertState(Model):
    """Lossy legacy alert state read from ``v1/predictions``. Use only for
    days without ``AlertPresence``; a bound derived from it is legacy-derived.
    It can bound a route-level alert end, never an alert version end."""

    lineage: LegacyLineage
    granularity: RouteAlertGranularity
    observed_at: Instant = Field(description="The prediction row's ts.")
    route_id: str = Field(description="The prediction row's route, as written.")
    primary_alert_type: str | None = Field(
        description="null = the route had an inference row and no active alert."
    )

    @model_validator(mode="after")
    def _consistent(self) -> LegacyRouteAlertState:
        key = LegacyPredictionsKey(self.observed_at).key
        _check_legacy(self.lineage, ("v1/predictions",), self.observed_at, key)
        if self.lineage.row_index is None:
            raise _fail("lineage", "a predictions row needs its line index")
        return self


LEGACY_FETCH_ARCHIVES: tuple[str, ...] = (
    "archive/trace",
    "archive/trip_updates",
    "archive/vehicles",
    "archive/alerts_liveness",
    "archive/ene",
)


class LegacyFetchOutcome(Model):
    """What a legacy archive kept about one source's fetch: whether it
    succeeded, nothing more. The trace, trip-update and vehicle archives list
    the GTFS-RT feeds that were fresh; ``archive/alerts_liveness`` records
    success or failure of the alerts fetch; an ``archive/ene`` object exists
    only for a successful fetch."""

    legacy_fetch_id: str = Field(description="f'{lineage.source_key}#{source}'.")
    lineage: LegacyLineage
    source: SourceId
    observed_at: Instant = Field(description="The legacy record's observed_at.")
    scheduled_at: Instant | Unknown = Field(description="Kept by archive/trace only.")
    succeeded: bool = Field(
        description="false: failed or stale; the archive kept no cause."
    )
    last_success_at: Instant | Unknown = Field(
        description="alerts_liveness fetched_at: the last successful fetch's time."
    )

    @model_validator(mode="after")
    def _consistent(self) -> LegacyFetchOutcome:
        _check_legacy(self.lineage, LEGACY_FETCH_ARCHIVES, self.observed_at, None)
        if self.legacy_fetch_id != f"{self.lineage.source_key}#{self.source}":
            raise _fail("legacy_key", "legacy_fetch_id is not source_key#source")
        if self.lineage.archive == "archive/ene" and not self.succeeded:
            raise _fail("legacy_key", "archive/ene kept successful fetches only")
        return self


class LegacyCollectorHealth(Model):
    """One ``archive/health`` object: the write failures of one Worker
    invocation. Planned sources and fetch timing were not kept."""

    legacy_tick_id: str = Field(description="f'archive/health@{observed_at}'.")
    lineage: LegacyLineage
    observed_at: Instant = Field(
        description="The invocation's wall clock (not the cron minute)."
    )
    write_failures: dict[str, int] = Field(
        description="Failed writes per kind; {} = a clean tick."
    )

    @model_validator(mode="after")
    def _consistent(self) -> LegacyCollectorHealth:
        key = f"archive/health/{utc_day(self.observed_at)}/{self.observed_at}.json"
        _check_legacy(self.lineage, ("archive/health",), self.observed_at, key)
        if self.legacy_tick_id != f"archive/health@{self.observed_at}":
            raise _fail(
                "legacy_key", "legacy_tick_id is not archive/health@observed_at"
            )
        return self


# --- derived files -------------------------------------------------------------

PARQUET_METADATA_KEY = "momentarily.observations"


class DerivedFileMetadata(Model):
    """The parquet key-value metadata of every derived file, stored as
    canonical JSON under ``PARQUET_METADATA_KEY``.

    ``provenance`` reuses layer 1's: ``inputs`` maps each raw or legacy key
    read to its sha256, ``captured_at`` maps the same keys to their
    ``observed_at``, and ``built_at`` is the latest of those (never the wall
    clock), so a rebuild gives the same bytes. Layer 2 assigns no service
    day (``period_start`` null) and is never retroactive.
    """

    schema_: Literal["observations/s1"] = Field(alias="schema")
    file_key: str = Field(description="The derived/ key of this file.")
    layout_version: int = Field(
        ge=1, description="columns.LAYOUT_VERSION it was written with."
    )
    decoder_version: str = Field(
        description="Version of the decoder that produced the rows."
    )
    row_count: int = Field(ge=0)
    provenance: Provenance

    @model_validator(mode="after")
    def _consistent(self) -> DerivedFileMetadata:
        key = _parse_key(self.file_key, "metadata")
        if not isinstance(key, DerivedHourKey | DerivedDayKey):
            raise _fail("metadata", "file_key is not a derived/ key")
        p = self.provenance
        if p.schema_ != SCHEMA_ID:
            raise _fail("metadata", f"provenance schema is not {SCHEMA_ID}")
        if set(p.captured_at) != set(p.inputs):
            raise _fail("metadata", "captured_at and inputs name different keys")
        if p.captured_at and p.built_at != max(p.captured_at.values()):
            raise _fail("metadata", "built_at is the latest captured_at")
        if p.period_start is not None or p.applied_retroactively:
            raise _fail(
                "metadata", "layer 2 assigns no service day and is never retroactive"
            )
        day = key.day if isinstance(key, DerivedDayKey) else utc_day(key.hour_start)
        if p.observed_from_utc_day != day:
            raise _fail("metadata", "observed_from_utc_day is the file's UTC day")
        return self
