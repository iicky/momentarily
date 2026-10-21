# ruff: noqa
# fmt: off
# pyright: basic
#
# Vendored NYCT Subway GTFS-realtime extension, compiled from nyct-subway.proto
# beside this file (verbatim upstream copy).
#   Source:   https://api.mta.info/nyct-subway.proto.txt (MTA developer resources)
#             fetched 2026-10-20; Last-Modified 2024-10-29 15:32:18 GMT,
#             ETag "c20af658e6b453adee9e590a56f645f2",
#             sha256 105db0e484601310ebef9ebe0cedbbb6bdf574aaa8f52b5caa52f59d44d113b7
#   Compiler: protoc (libprotoc 36.2), against the descriptor of
#             gtfs-realtime-bindings 3.0.0 (google.transit.gtfs_realtime_pb2),
#             for the protobuf 7.36.2 Python runtime:
#               protoc --descriptor_set_in=gtfs-realtime.pb -I. \
#                 --python_out=. --pyi_out=. nyct-subway.proto
#             where gtfs-realtime.pb is a FileDescriptorSet holding
#             gtfs_realtime_pb2.DESCRIPTOR.
#   Edited:   the import of gtfs_realtime_pb2, rewritten to
#             `from google.transit import gtfs_realtime_pb2` so the extension
#             registers against the bindings' FeedHeader, TripDescriptor and
#             StopTimeUpdate; and the three module-level extension
#             declarations, typed as extension descriptors of their container
#             and extension messages, with the import of
#             _ExtensionFieldDescriptor that needs (protoc writes ClassVar and a bare
#             FieldDescriptor there, which no checker accepts at module level);
#             and this comment block with the pragma lines above it prepended.
from google.transit import gtfs_realtime_pb2 as _gtfs_realtime_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf.internal.extension_dict import _ExtensionFieldDescriptor
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor
NYCT_FEED_HEADER_FIELD_NUMBER: int
nyct_feed_header: _ExtensionFieldDescriptor[_gtfs_realtime_pb2.FeedHeader, NyctFeedHeader]
NYCT_TRIP_DESCRIPTOR_FIELD_NUMBER: int
nyct_trip_descriptor: _ExtensionFieldDescriptor[_gtfs_realtime_pb2.TripDescriptor, NyctTripDescriptor]
NYCT_STOP_TIME_UPDATE_FIELD_NUMBER: int
nyct_stop_time_update: _ExtensionFieldDescriptor[_gtfs_realtime_pb2.TripUpdate.StopTimeUpdate, NyctStopTimeUpdate]

class TripReplacementPeriod(_message.Message):
    __slots__ = ("route_id", "replacement_period")
    ROUTE_ID_FIELD_NUMBER: _ClassVar[int]
    REPLACEMENT_PERIOD_FIELD_NUMBER: _ClassVar[int]
    route_id: str
    replacement_period: _gtfs_realtime_pb2.TimeRange
    def __init__(self, route_id: _Optional[str] = ..., replacement_period: _Optional[_Union[_gtfs_realtime_pb2.TimeRange, _Mapping]] = ...) -> None: ...

class NyctFeedHeader(_message.Message):
    __slots__ = ("nyct_subway_version", "trip_replacement_period")
    NYCT_SUBWAY_VERSION_FIELD_NUMBER: _ClassVar[int]
    TRIP_REPLACEMENT_PERIOD_FIELD_NUMBER: _ClassVar[int]
    nyct_subway_version: str
    trip_replacement_period: _containers.RepeatedCompositeFieldContainer[TripReplacementPeriod]
    def __init__(self, nyct_subway_version: _Optional[str] = ..., trip_replacement_period: _Optional[_Iterable[_Union[TripReplacementPeriod, _Mapping]]] = ...) -> None: ...

class NyctTripDescriptor(_message.Message):
    __slots__ = ("train_id", "is_assigned", "direction")
    class Direction(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        NORTH: _ClassVar[NyctTripDescriptor.Direction]
        EAST: _ClassVar[NyctTripDescriptor.Direction]
        SOUTH: _ClassVar[NyctTripDescriptor.Direction]
        WEST: _ClassVar[NyctTripDescriptor.Direction]
    NORTH: NyctTripDescriptor.Direction
    EAST: NyctTripDescriptor.Direction
    SOUTH: NyctTripDescriptor.Direction
    WEST: NyctTripDescriptor.Direction
    TRAIN_ID_FIELD_NUMBER: _ClassVar[int]
    IS_ASSIGNED_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    train_id: str
    is_assigned: bool
    direction: NyctTripDescriptor.Direction
    def __init__(self, train_id: _Optional[str] = ..., is_assigned: _Optional[bool] = ..., direction: _Optional[_Union[NyctTripDescriptor.Direction, str]] = ...) -> None: ...

class NyctStopTimeUpdate(_message.Message):
    __slots__ = ("scheduled_track", "actual_track")
    SCHEDULED_TRACK_FIELD_NUMBER: _ClassVar[int]
    ACTUAL_TRACK_FIELD_NUMBER: _ClassVar[int]
    scheduled_track: str
    actual_track: str
    def __init__(self, scheduled_track: _Optional[str] = ..., actual_track: _Optional[str] = ...) -> None: ...
