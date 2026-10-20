"""ENE outage payloads (``ene_current``, ``ene_upcoming``) to
``EquipmentObservation`` records, one per record in published order.

``outagedate`` and ``estimatedreturntoservice`` are New York wall times
published as ``MM/DD/YYYY hh:mm:ss AM``. A field the record omits, or
publishes blank or whitespace-only, is ``None`` (the ``_local`` field keeps
the published text); one that does not name exactly one instant is
``Unknown``: unparseable (any other text),
ambiguous (a wall time the fall-back hour repeats), nonexistent (one the
spring-forward hour skips) or before 1970 (no ``Instant`` holds it).
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from typing import cast
from zoneinfo import ZoneInfo

from pydantic import JsonValue

from momentarily.model.conventions import Unknown
from momentarily.model.observations.contracts import (
    EquipmentObservation,
    EquipmentSource,
)
from momentarily.model.observations.decode.common import (
    DecodeError,
    RawFetch,
    as_object,
    check_payload,
    decode_errors,
    load_json,
)

NEW_YORK = ZoneInfo("America/New_York")
SOURCES = ("ene_current", "ene_upcoming")

_WALL_TIME = re.compile(
    r"(?P<month>\d{2})/(?P<day>\d{2})/(?P<year>\d{4}) "
    r"(?P<hour>\d{2}):(?P<minute>\d{2}):(?P<second>\d{2}) (?P<half>AM|PM)",
    re.ASCII,
)


def new_york_instant(published: str | None) -> int | Unknown | None:
    """The instant a published ENE wall time names, in America/New_York."""
    if published is None or not published.strip():
        return None
    match = _WALL_TIME.fullmatch(published)
    if match is None:
        return Unknown(unknown=True, reason=f"unparseable: {published!r}")
    hour = int(match["hour"])
    if not 1 <= hour <= 12:
        return Unknown(unknown=True, reason=f"unparseable: {published!r}")
    try:
        wall = datetime(
            int(match["year"]),
            int(match["month"]),
            int(match["day"]),
            hour % 12 + (12 if match["half"] == "PM" else 0),
            int(match["minute"]),
            int(match["second"]),
        )
    except ValueError:
        return Unknown(unknown=True, reason=f"unparseable: {published!r}")
    try:
        first = wall.replace(tzinfo=NEW_YORK, fold=0)
        if first.utcoffset() != wall.replace(tzinfo=NEW_YORK, fold=1).utcoffset():
            kept = first.astimezone(UTC).astimezone(NEW_YORK).replace(tzinfo=None)
            clock = f"{wall:%H:%M:%S} on {wall:%Y-%m-%d} in America/New_York"
            if kept == wall:
                return Unknown(unknown=True, reason=f"ambiguous: {clock} occurs twice")
            return Unknown(unknown=True, reason=f"nonexistent: {clock} is skipped")
        instant = int(first.timestamp())
    except (ValueError, OverflowError):
        return Unknown(unknown=True, reason=f"unparseable: {published!r}")
    if instant < 0:
        return Unknown(unknown=True, reason=f"before 1970: {published!r}")
    return instant


def _record(value: JsonValue, index: int) -> dict[str, str]:
    record = as_object(value, f"ENE record {index}")
    strings = {k: v for k, v in record.items() if isinstance(v, str)}
    if len(strings) != len(record):
        raise DecodeError(f"ENE record {index} has a value that is not a string")
    return strings


def decode_ene(payload: bytes, fetch: RawFetch) -> list[EquipmentObservation]:
    """An ENE outage payload (a JSON array of records) as observations."""
    check_payload(payload, fetch, SOURCES)
    source = cast(EquipmentSource, fetch.source)
    with decode_errors(fetch):
        records = load_json(payload)
        if not isinstance(records, list):
            raise DecodeError("an ENE payload is not a JSON array")
        observations: list[EquipmentObservation] = []
        for index, value in enumerate(records):
            record = _record(value, index)
            equipment_id = record.get("equipment")
            if equipment_id is None:
                raise DecodeError(f"ENE record {index} names no equipment")
            start = record.get("outagedate")
            back = record.get("estimatedreturntoservice")
            observations.append(
                EquipmentObservation(
                    lineage=fetch.lineage,
                    fetch_id=fetch.fetch_id,
                    source=source,
                    observed_at=fetch.observed_at,
                    record_index=index,
                    equipment_id=equipment_id,
                    record=record,
                    outage_start_local=start,
                    estimated_return_local=back,
                    outage_start=new_york_instant(start),
                    estimated_return=new_york_instant(back),
                )
            )
        return observations
