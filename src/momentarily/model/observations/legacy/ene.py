"""``archive/ene`` -> ``EquipmentObservation`` per outage record, plus a
``LegacyFetchOutcome`` for each archived (so successful) fetch.

Object: ``archive/ene/<YYYY-MM-DD>/<HH>0000-<source>.json`` holding
``{observed_at, source, payload}``: the last successful fetch of that UTC
hour, re-serialized. ``ene_current`` and ``ene_upcoming`` hold outage
records; ``ene_equipments`` is the catalog, which layer 2 keeps raw only, so
it yields the fetch outcome alone. Failed fetches left no object.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from momentarily.model.conventions import Unknown
from momentarily.model.observations.contracts import (
    EquipmentObservation,
    EquipmentSource,
    LegacyArchive,
    LegacyFetchOutcome,
)
from momentarily.model.observations.legacy.common import (
    Emitted,
    LegacyObject,
    MalformedObjectError,
    as_dict,
    as_list,
    check_day,
    get_int,
    get_str,
    lineage,
    load_json,
    match_key,
    unknown,
)

ARCHIVE: LegacyArchive = "archive/ene"

_KEY = re.compile(
    r"archive/ene/(?P<day>\d{4}-\d{2}-\d{2})/(?P<hour>\d{2})0000-(?P<source>ene_current|ene_upcoming|ene_equipments)\.json"
)
_EQUIPMENT: dict[str, EquipmentSource] = {
    "ene_current": "ene_current",
    "ene_upcoming": "ene_upcoming",
}
_ET = ZoneInfo("America/New_York")
_FORMAT = "%m/%d/%Y %I:%M:%S %p"
_NO_FETCH = unknown("archive/ene keeps no fetch identity")
_WALL_CLOCK_ONLY = unknown("archive/ene keeps the wall clock only")
_NO_LAST_SUCCESS = unknown("archive/ene keeps no fetch time")
_UNPARSEABLE = unknown("not an ENE wall time (MM/DD/YYYY hh:mm:ss AM)")
_AMBIGUOUS = unknown("ambiguous in America/New_York (clocks went back)")
_NONEXISTENT = unknown("does not exist in America/New_York (clocks went forward)")


def eastern_instant(text: str | None) -> int | None | Unknown:
    """A published ENE wall time as an instant: ``None`` when omitted, blank
    or whitespace only (the feed's way of saying absent); ``Unknown`` when it
    is unparseable, repeated or skipped by DST."""
    if text is None or not text.strip():
        return None
    try:
        naive = datetime.strptime(text, _FORMAT)
    except ValueError:
        return _UNPARSEABLE
    early = naive.replace(tzinfo=_ET, fold=0)
    late = naive.replace(tzinfo=_ET, fold=1)
    if early.utcoffset() != late.utcoffset():
        # fold differs only in a repeated or a skipped hour; the round trip
        # through UTC tells them apart.
        round_trip = early.astimezone(UTC).astimezone(_ET).replace(tzinfo=None)
        return _AMBIGUOUS if round_trip == naive else _NONEXISTENT
    return int(early.timestamp())


def watermark(key: str) -> int:
    """No object at or after ``key`` holds a record observed before the UTC
    hour it is filed under."""
    m = match_key(_KEY, key)
    start = datetime.fromisoformat(f"{m['day']}T{m['hour']}:00:00+00:00")
    return int(start.timestamp())


def _record(
    key: str, source: EquipmentSource, index: int, observed_at: int, value: object
) -> EquipmentObservation:
    where = f"{key} record {index}"
    doc = as_dict(value, where)
    record = {name: get_str(doc, name, where) for name in doc}
    outage_local = record.get("outagedate")
    return_local = record.get("estimatedreturntoservice")
    return EquipmentObservation(
        lineage=lineage(ARCHIVE, key, index),
        fetch_id=_NO_FETCH,
        source=source,
        observed_at=observed_at,
        record_index=index,
        equipment_id=get_str(record, "equipment", where),
        record=record,
        outage_start_local=outage_local,
        estimated_return_local=return_local,
        outage_start=eastern_instant(outage_local),
        estimated_return=eastern_instant(return_local),
    )


def adapt(obj: LegacyObject) -> list[Emitted]:
    m = match_key(_KEY, obj.key)
    doc = as_dict(load_json(obj), obj.key)
    observed_at = get_int(doc, "observed_at", obj.key)
    check_day(obj.key, m["day"], observed_at)
    if watermark(obj.key) != observed_at // 3600 * 3600:
        raise MalformedObjectError(f"{obj.key} is filed under another hour")
    name = get_str(doc, "source", obj.key)
    if name != m["source"]:
        raise MalformedObjectError(f"{obj.key} holds source {name!r}")
    payload = as_list(doc.get("payload"), f"{obj.key}.payload")
    source = obj.source(observed_at)
    out: list[Emitted] = []
    equipment = _EQUIPMENT.get(name)
    if equipment is not None:
        out = [
            Emitted(_record(obj.key, equipment, i, observed_at, value), (source,))
            for i, value in enumerate(payload)
        ]
    outcome = LegacyFetchOutcome(
        legacy_fetch_id=f"{obj.key}#{name}",
        lineage=lineage(ARCHIVE, obj.key, None),
        source="ene_equipments" if equipment is None else equipment,
        observed_at=observed_at,
        scheduled_at=_WALL_CLOCK_ONLY,
        succeeded=True,
        last_success_at=_NO_LAST_SUCCESS,
    )
    out.append(Emitted(outcome, (source,)))
    return out
