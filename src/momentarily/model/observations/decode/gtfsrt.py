"""GTFS-RT payloads to ``FeedHeaderObservation``, ``VehicleObservation``,
``TripUpdateObservation`` and, for ``numbered``, ``AlertPresence``.

Decoding uses the official bindings (``google.transit.gtfs_realtime_pb2``)
with the vendored NYCT extension registered. GTFS-RT is proto2, so every
optional field is read through ``HasField``/``HasExtension``: an omitted
field is ``None``, never the proto default. A payload that does not parse,
or lacks a required field, raises ``DecodeError``.

``unmodeled_fields`` lists what a record's entity carried that its contract
does not hold, as dotted field-number paths:

* relative to the record's message (VehiclePosition or TripUpdate), so
  ``2`` is a vehicle's ``position`` and ``2.6`` a stop-time update's
  ``stop_time_properties``;
* a leading ``..`` names a field of the enclosing FeedEntity, other than
  ``id`` and the trip update, vehicle and alert it holds (``..2`` is
  ``is_deleted``);
* a path stops at the first unmodelled field (``2``, not ``2.1``), names
  extensions and unknown fields by number like any other field, and holds no
  list index: a field seen in any element of a repeated message is listed
  once;
* the list is sorted, ``..`` paths first, then by field numbers.

Unmodelled fields of the FeedMessage and FeedHeader (an unknown tag appended
to a payload's bytes lands on the FeedMessage) are not recorded, and an
undefined ``incrementality`` value is None with no path: no contract has a
place for them. Entities other than trip updates, vehicles and
(``numbered``) alerts produce no record; Alert entities in other feeds are
dropped.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

from google.protobuf import message as protobuf_message
from google.protobuf.unknown_fields import UnknownFieldSet
from google.transit import gtfs_realtime_pb2 as rt
from pydantic import JsonValue

from momentarily.model.observations.contracts import (
    GTFSRT_SOURCES,
    AlertPresence,
    FeedHeaderObservation,
    GtfsRtSource,
    NyctFeedHeaderObs,
    StopTimePrediction,
    TimeRangeObs,
    TripDescriptorObs,
    TripReplacementPeriodObs,
    TripUpdateObservation,
    VehicleObservation,
)
from momentarily.model.observations.decode import nyct_subway_pb2 as nyct
from momentarily.model.observations.decode.alerts import (
    alert_presence,
    numbered_alert_entity,
)
from momentarily.model.observations.decode.common import (
    DecodeError,
    RawFetch,
    check_payload,
    decode_errors,
)

if TYPE_CHECKING:
    from google.protobuf.internal.extension_dict import (
        _ExtensionFieldDescriptor,  # pyright: ignore[reportPrivateUsage]
    )


@dataclass(frozen=True, slots=True)
class GtfsRtRecords:
    """Everything one GTFS-RT payload decodes to, in entity order."""

    header: FeedHeaderObservation
    vehicles: list[VehicleObservation]
    trip_updates: list[TripUpdateObservation]
    alerts: AlertPresence | None  # numbered only; [] entries when it has none


# --- what the contracts model, by field number --------------------------------

type _Spec = Mapping[int, _Spec | None]
"""Modelled fields of a message: a leaf (None) or the spec of a submessage."""

_T, _V, _U = rt.TripDescriptor, rt.VehiclePosition, rt.TripUpdate
_S, _E = rt.TripUpdate.StopTimeUpdate, rt.TripUpdate.StopTimeEvent
_NT, _NS = nyct.NyctTripDescriptor, nyct.NyctStopTimeUpdate

_TRIP: _Spec = {
    _T.TRIP_ID_FIELD_NUMBER: None,
    _T.ROUTE_ID_FIELD_NUMBER: None,
    _T.START_TIME_FIELD_NUMBER: None,
    _T.START_DATE_FIELD_NUMBER: None,
    _T.SCHEDULE_RELATIONSHIP_FIELD_NUMBER: None,
    _T.DIRECTION_ID_FIELD_NUMBER: None,
    nyct.NYCT_TRIP_DESCRIPTOR_FIELD_NUMBER: {
        _NT.TRAIN_ID_FIELD_NUMBER: None,
        _NT.IS_ASSIGNED_FIELD_NUMBER: None,
        _NT.DIRECTION_FIELD_NUMBER: None,
    },
}
_VEHICLE: _Spec = {
    _V.TRIP_FIELD_NUMBER: _TRIP,
    _V.STOP_ID_FIELD_NUMBER: None,
    _V.CURRENT_STOP_SEQUENCE_FIELD_NUMBER: None,
    _V.CURRENT_STATUS_FIELD_NUMBER: None,
    _V.TIMESTAMP_FIELD_NUMBER: None,
    _V.CONGESTION_LEVEL_FIELD_NUMBER: None,
}
_EVENT: _Spec = {
    _E.TIME_FIELD_NUMBER: None,
    _E.DELAY_FIELD_NUMBER: None,
    _E.UNCERTAINTY_FIELD_NUMBER: None,
}
_TRIP_UPDATE: _Spec = {
    _U.TRIP_FIELD_NUMBER: _TRIP,
    _U.TIMESTAMP_FIELD_NUMBER: None,
    _U.DELAY_FIELD_NUMBER: None,
    _U.STOP_TIME_UPDATE_FIELD_NUMBER: {
        _S.STOP_SEQUENCE_FIELD_NUMBER: None,
        _S.STOP_ID_FIELD_NUMBER: None,
        _S.ARRIVAL_FIELD_NUMBER: _EVENT,
        _S.DEPARTURE_FIELD_NUMBER: _EVENT,
        _S.SCHEDULE_RELATIONSHIP_FIELD_NUMBER: None,
        _S.DEPARTURE_OCCUPANCY_STATUS_FIELD_NUMBER: None,
        nyct.NYCT_STOP_TIME_UPDATE_FIELD_NUMBER: {
            _NS.SCHEDULED_TRACK_FIELD_NUMBER: None,
            _NS.ACTUAL_TRACK_FIELD_NUMBER: None,
        },
    },
}
_ENTITY_HELD = frozenset(
    {
        rt.FeedEntity.ID_FIELD_NUMBER,
        rt.FeedEntity.TRIP_UPDATE_FIELD_NUMBER,
        rt.FeedEntity.VEHICLE_FIELD_NUMBER,
        rt.FeedEntity.ALERT_FIELD_NUMBER,
    }
)


def _collect(
    msg: protobuf_message.Message,
    spec: _Spec,
    prefix: tuple[int, ...],
    found: set[tuple[int, ...]],
) -> None:
    for field, value in msg.ListFields():
        path = (*prefix, field.number)
        if field.number not in spec:
            found.add(path)
            continue
        child = spec[field.number]
        if child is None:
            continue
        values = value if field.is_repeated else (value,)
        for item in cast("list[protobuf_message.Message]", values):
            _collect(item, child, path, found)
    for unknown in UnknownFieldSet(msg):
        found.add((*prefix, unknown.field_number))


def _entity_paths(entity: rt.FeedEntity) -> set[tuple[int, ...]]:
    paths = {
        (f.number,) for f, _ in entity.ListFields() if f.number not in _ENTITY_HELD
    }
    return paths | {(u.field_number,) for u in UnknownFieldSet(entity)}


def unmodeled_fields(
    record_message: protobuf_message.Message,
    spec: _Spec,
    entity_paths: set[tuple[int, ...]],
) -> list[str]:
    found: set[tuple[int, ...]] = set()
    _collect(record_message, spec, (), found)
    return [".." + ".".join(map(str, p)) for p in sorted(entity_paths)] + [
        ".".join(map(str, p)) for p in sorted(found)
    ]


# --- field readers -------------------------------------------------------------


def _time_range(msg: rt.TimeRange) -> TimeRangeObs:
    return TimeRangeObs(
        start=msg.start if msg.HasField("start") else None,
        end=msg.end if msg.HasField("end") else None,
    )


def _extension[C: protobuf_message.Message, X: protobuf_message.Message](
    msg: C, field: _ExtensionFieldDescriptor[C, X]
) -> X | None:
    """The extension ``field`` of ``msg``; None when the payload omits it."""
    if not msg.HasExtension(field):
        return None
    return msg.Extensions[field]


def _nyct_header(header: rt.FeedHeader) -> NyctFeedHeaderObs | None:
    ext = _extension(header, nyct.nyct_feed_header)
    if ext is None:
        return None
    return NyctFeedHeaderObs(
        nyct_subway_version=ext.nyct_subway_version,
        trip_replacement_periods=[
            TripReplacementPeriodObs(
                route_id=p.route_id if p.HasField("route_id") else None,
                replacement_period=_time_range(p.replacement_period)
                if p.HasField("replacement_period")
                else None,
            )
            for p in ext.trip_replacement_period
        ],
    )


def _header(
    msg: rt.FeedMessage, fetch: RawFetch, source: GtfsRtSource
) -> FeedHeaderObservation:
    header = msg.header
    return FeedHeaderObservation(
        lineage=fetch.lineage,
        fetch_id=fetch.fetch_id,
        source=source,
        observed_at=fetch.observed_at,
        gtfs_realtime_version=header.gtfs_realtime_version,
        incrementality=header.incrementality
        if header.HasField("incrementality")
        else None,
        header_timestamp=header.timestamp if header.HasField("timestamp") else None,
        feed_version=header.feed_version if header.HasField("feed_version") else None,
        nyct_feed_header=_nyct_header(header),
    )


def _trip(td: rt.TripDescriptor) -> TripDescriptorObs:
    ext = _extension(td, nyct.nyct_trip_descriptor)
    return TripDescriptorObs(
        trip_id=td.trip_id if td.HasField("trip_id") else None,
        route_id=td.route_id if td.HasField("route_id") else None,
        start_time=td.start_time if td.HasField("start_time") else None,
        start_date=td.start_date if td.HasField("start_date") else None,
        schedule_relationship=td.schedule_relationship
        if td.HasField("schedule_relationship")
        else None,
        direction_id=cast(Literal[0, 1], td.direction_id)
        if td.HasField("direction_id")
        else None,
        nyct_train_id=ext.train_id if ext and ext.HasField("train_id") else None,
        nyct_is_assigned=ext.is_assigned
        if ext and ext.HasField("is_assigned")
        else None,
        nyct_direction=cast(Literal[1, 2, 3, 4], ext.direction)
        if ext and ext.HasField("direction")
        else None,
    )


def _vehicle(
    v: rt.VehiclePosition,
    index: int,
    entity: rt.FeedEntity,
    fetch: RawFetch,
    source: GtfsRtSource,
    entity_paths: set[tuple[int, ...]],
) -> VehicleObservation:
    return VehicleObservation(
        lineage=fetch.lineage,
        fetch_id=fetch.fetch_id,
        source=source,
        observed_at=fetch.observed_at,
        entity_index=index,
        entity_id=entity.id,
        trip=_trip(v.trip) if v.HasField("trip") else None,
        stop_id=v.stop_id if v.HasField("stop_id") else None,
        current_stop_sequence=v.current_stop_sequence
        if v.HasField("current_stop_sequence")
        else None,
        current_status=cast(Literal[0, 1, 2], v.current_status)
        if v.HasField("current_status")
        else None,
        vehicle_timestamp=v.timestamp if v.HasField("timestamp") else None,
        congestion_level=v.congestion_level if v.HasField("congestion_level") else None,
        unmodeled_fields=unmodeled_fields(v, _VEHICLE, entity_paths),
    )


def _stop_time(s: rt.TripUpdate.StopTimeUpdate, index: int) -> StopTimePrediction:
    arrival = s.arrival if s.HasField("arrival") else None
    departure = s.departure if s.HasField("departure") else None
    ext = _extension(s, nyct.nyct_stop_time_update)
    return StopTimePrediction(
        index=index,
        stop_id=s.stop_id if s.HasField("stop_id") else None,
        stop_sequence=s.stop_sequence if s.HasField("stop_sequence") else None,
        arrival_time=_event(arrival, "time"),
        arrival_delay=_event(arrival, "delay"),
        arrival_uncertainty=_event(arrival, "uncertainty"),
        departure_time=_event(departure, "time"),
        departure_delay=_event(departure, "delay"),
        departure_uncertainty=_event(departure, "uncertainty"),
        schedule_relationship=s.schedule_relationship
        if s.HasField("schedule_relationship")
        else None,
        departure_occupancy_status=s.departure_occupancy_status
        if s.HasField("departure_occupancy_status")
        else None,
        nyct_scheduled_track=ext.scheduled_track
        if ext and ext.HasField("scheduled_track")
        else None,
        nyct_actual_track=ext.actual_track
        if ext and ext.HasField("actual_track")
        else None,
    )


def _event(
    event: rt.TripUpdate.StopTimeEvent | None,
    name: Literal["time", "delay", "uncertainty"],
) -> int | None:
    if event is None or not event.HasField(name):
        return None
    return cast(int, getattr(event, name))


def _trip_update(
    u: rt.TripUpdate,
    index: int,
    entity: rt.FeedEntity,
    fetch: RawFetch,
    source: GtfsRtSource,
    entity_paths: set[tuple[int, ...]],
) -> TripUpdateObservation:
    return TripUpdateObservation(
        lineage=fetch.lineage,
        fetch_id=fetch.fetch_id,
        source=source,
        observed_at=fetch.observed_at,
        entity_index=index,
        entity_id=entity.id,
        trip=_trip(u.trip),
        trip_update_timestamp=u.timestamp if u.HasField("timestamp") else None,
        delay=u.delay if u.HasField("delay") else None,
        stop_times=[_stop_time(s, i) for i, s in enumerate(u.stop_time_update)],
        unmodeled_fields=unmodeled_fields(u, _TRIP_UPDATE, entity_paths),
    )


def parse_feed(payload: bytes) -> rt.FeedMessage:
    """The FeedMessage in ``payload``, NYCT extensions included."""
    msg = rt.FeedMessage()
    try:
        msg.ParseFromString(payload)
    except protobuf_message.DecodeError as err:
        raise DecodeError(f"not a GTFS-RT FeedMessage: {err}") from err
    if not msg.IsInitialized():
        raise DecodeError("GTFS-RT payload lacks a required field")
    return msg


def decode_gtfsrt(payload: bytes, fetch: RawFetch) -> GtfsRtRecords:
    """Every record one GTFS-RT payload holds. An entity holding both a trip
    update and a vehicle gives one of each, with the same ``entity_index``.
    A value the contracts reject (a ``direction_id`` of 2, a negative time)
    raises ``DecodeError`` like a payload that does not parse."""
    check_payload(payload, fetch, GTFSRT_SOURCES)
    source = cast(GtfsRtSource, fetch.source)
    msg = parse_feed(payload)
    with decode_errors(fetch):
        return _records(msg, fetch, source)


def _records(
    msg: rt.FeedMessage, fetch: RawFetch, source: GtfsRtSource
) -> GtfsRtRecords:
    vehicles: list[VehicleObservation] = []
    trip_updates: list[TripUpdateObservation] = []
    alerts: list[tuple[str, dict[str, JsonValue], int | None]] = []
    for index, entity in enumerate(msg.entity):
        paths = _entity_paths(entity)
        if entity.HasField("trip_update"):
            trip_updates.append(
                _trip_update(entity.trip_update, index, entity, fetch, source, paths)
            )
        if entity.HasField("vehicle"):
            vehicles.append(
                _vehicle(entity.vehicle, index, entity, fetch, source, paths)
            )
        if entity.HasField("alert"):
            alerts.append((entity.id, numbered_alert_entity(entity), None))
    header = _header(msg, fetch, source)
    presence = (
        alert_presence(fetch, "numbered", header.header_timestamp, alerts)
        if source == "numbered"
        else None
    )
    return GtfsRtRecords(header, vehicles, trip_updates, presence)
