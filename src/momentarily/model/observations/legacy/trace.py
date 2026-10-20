"""``archive/trace`` -> partial ``VehicleObservation`` per row, plus a
``LegacyFetchOutcome`` per GTFS-RT feed and trace minute.

Object: ``archive/trace/<YYYY-MM-DD>/<scheduled_at>.json`` holding
``{observed_at, scheduled_at, fresh_feeds, [feed_digest, feed_etag], rows}``,
each row ``{trip_id, route_id, direction, stop_id, stop_seq, stopped,
vehicle_ts}``. The Worker merged all eight feeds, folded the route
(``6X`` -> ``6``), dropped vehicles without a route, trip or stop id, and
kept ``stop_seq`` only for stopped vehicles. So a row keeps ``trip_id``,
``stop_id``, ``vehicle_ts`` and, when stopped, ``current_status`` 1 and the
stop sequence; everything else is ``Unknown``. The folded route and the
direction are not kept: neither is a published value.
"""

from __future__ import annotations

import re

from momentarily.model.observations.contracts import (
    LegacyArchive,
    TripDescriptorObs,
    VehicleObservation,
)
from momentarily.model.observations.legacy.common import (
    Emitted,
    LegacyObject,
    MalformedObjectError,
    as_dict,
    as_list,
    check_day,
    get_bool,
    get_int,
    get_opt_int,
    get_str,
    lineage,
    load_json,
    match_key,
    unknown,
)
from momentarily.model.observations.legacy.fresh_feeds import (
    feed_outcomes,
    fresh_feeds,
)

ARCHIVE: LegacyArchive = "archive/trace"

_KEY = re.compile(r"archive/trace/(?P<day>\d{4}-\d{2}-\d{2})/(?P<at>\d+)\.json")

_NO_FETCH = unknown("archive/trace keeps no fetch identity")
_NO_FEED = unknown("archive/trace merges all feeds without the feed of a row")
_NO_ENTITY = unknown("archive/trace keeps no entity id or position")
_FOLDED = unknown("archive/trace folds the route id (6X, 7X, FX become 6, 7, F)")
_TRIP_ONLY = unknown("archive/trace keeps only trip_id of the trip descriptor")
_MOVING = unknown("archive/trace keeps only whether a vehicle is stopped")
_SEQ_MOVING = unknown("archive/trace keeps stop_seq of stopped vehicles only")
_NO_CONGESTION = unknown("archive/trace does not keep congestion_level")
_NO_FIELDS = unknown("archive/trace keeps no record of undecoded fields")
_STOPPED_AT = 1


def watermark(key: str) -> int:
    """No object at or after ``key`` holds a row observed before this: an
    attempt is observed after the minute it is scheduled for."""
    return int(match_key(_KEY, key)["at"])


def _vehicle(
    key: str, row_index: int, observed_at: int, row: object
) -> VehicleObservation:
    where = f"{key} row {row_index}"
    doc = as_dict(row, where)
    stopped = get_bool(doc, "stopped", where)
    seq = get_opt_int(doc, "stop_seq", where)
    if not stopped and seq is not None:
        raise MalformedObjectError(f"{where} has a stop_seq while moving")
    trip = TripDescriptorObs(
        trip_id=get_str(doc, "trip_id", where),
        route_id=_FOLDED,
        start_time=_TRIP_ONLY,
        start_date=_TRIP_ONLY,
        schedule_relationship=_TRIP_ONLY,
        direction_id=_TRIP_ONLY,
        nyct_train_id=_TRIP_ONLY,
        nyct_is_assigned=_TRIP_ONLY,
        nyct_direction=_TRIP_ONLY,
    )
    return VehicleObservation(
        lineage=lineage(ARCHIVE, key, row_index),
        fetch_id=_NO_FETCH,
        source=_NO_FEED,
        observed_at=observed_at,
        entity_index=_NO_ENTITY,
        entity_id=_NO_ENTITY,
        trip=trip,
        stop_id=get_str(doc, "stop_id", where),
        current_stop_sequence=seq if stopped else _SEQ_MOVING,
        current_status=_STOPPED_AT if stopped else _MOVING,
        vehicle_timestamp=get_opt_int(doc, "vehicle_ts", where),
        congestion_level=_NO_CONGESTION,
        unmodeled_fields=_NO_FIELDS,
    )


def adapt(obj: LegacyObject) -> list[Emitted]:
    m = match_key(_KEY, obj.key)
    doc = as_dict(load_json(obj), obj.key)
    observed_at = get_int(doc, "observed_at", obj.key)
    scheduled_at = get_int(doc, "scheduled_at", obj.key)
    if int(m["at"]) != scheduled_at:
        raise MalformedObjectError(f"{obj.key} names another scheduled_at")
    check_day(obj.key, m["day"], scheduled_at)
    source = obj.source(observed_at)
    rows = as_list(doc.get("rows"), f"{obj.key}.rows")
    out = [
        Emitted(_vehicle(obj.key, i, observed_at, row), (source,))
        for i, row in enumerate(rows)
    ]
    out += feed_outcomes(ARCHIVE, source, scheduled_at, fresh_feeds(doc, obj.key))
    return out
