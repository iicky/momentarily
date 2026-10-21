"""Alert presence: the camsys alerts JSON, and the Alert entities of the
``numbered`` GTFS-RT feed.

Each entry's ``content_digest`` is ``alert_content_digest`` of the entity as a
JSON object. A camsys entity is that object already. A ``numbered`` entity is
protobuf, and its JSON object is ``numbered_alert_entity``: protobuf's
``MessageToDict`` of the whole FeedEntity with ``preserving_proto_field_name``
and every other option at its default, so field names are the proto's
(``informed_entity``), enums are names, 64-bit integers are strings (the
proto3 JSON mapping), absent fields are left out, extensions are keyed
``[<full name>]`` and unknown fields are dropped.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import cast

from google.protobuf.json_format import MessageToDict
from google.transit import gtfs_realtime_pb2
from pydantic import JsonValue

from momentarily.model.observations.contracts import (
    AlertEntry,
    AlertPresence,
    AlertSource,
    alert_content_digest,
)
from momentarily.model.observations.decode.common import (
    DecodeError,
    RawFetch,
    as_instant,
    as_object,
    check_payload,
    decode_errors,
    header_epoch_seconds,
    load_json,
)

MERCURY_ALERT = "transit_realtime.mercury_alert"


def numbered_alert_entity(entity: gtfs_realtime_pb2.FeedEntity) -> dict[str, JsonValue]:
    """The JSON object a ``numbered`` Alert entity is digested and stored as."""
    return cast(
        dict[str, JsonValue], MessageToDict(entity, preserving_proto_field_name=True)
    )


def alert_presence(
    fetch: RawFetch,
    source: AlertSource,
    header_timestamp: int | None,
    alerts: Iterable[tuple[str, dict[str, JsonValue], int | None]],
) -> AlertPresence:
    """The presence of ``(alert_id, entity, updated_at)`` alerts at one fetch.
    An entity published twice with identical content is one entry."""
    entries: dict[tuple[str, str], AlertEntry] = {}
    for alert_id, entity, updated_at in alerts:
        digest = alert_content_digest(entity)
        entries[alert_id, digest] = AlertEntry(
            alert_id=alert_id, content_digest=digest, updated_at=updated_at
        )
    return AlertPresence(
        lineage=fetch.lineage,
        fetch_id=fetch.fetch_id,
        source=source,
        observed_at=fetch.observed_at,
        header_timestamp=header_timestamp,
        entries=[entries[key] for key in sorted(entries)],
    )


def _camsys_alert(
    entity: dict[str, JsonValue],
) -> tuple[str, dict[str, JsonValue], int | None]:
    alert_id = entity.get("id")
    if not isinstance(alert_id, str):
        raise DecodeError("an alert entity has no string id")
    alert = entity.get("alert")
    mercury = (
        as_object(alert, alert_id).get(MERCURY_ALERT) if alert is not None else None
    )
    updated_at = (
        as_instant(
            as_object(mercury, alert_id).get("updated_at"), f"{alert_id} updated_at"
        )
        if mercury is not None
        else None
    )
    return alert_id, entity, updated_at


def decode_alerts_json(payload: bytes, fetch: RawFetch) -> AlertPresence:
    """The camsys subway alerts JSON (source ``alerts``) as one presence.
    ``entity`` absent means no alerts, and the header timestamp is read as
    the recorder reads it, so a fetch it calls ok or empty decodes unless an
    entity lacks a string id or carries a malformed ``updated_at``."""
    check_payload(payload, fetch, ("alerts",))
    with decode_errors(fetch):
        doc = as_object(load_json(payload), "the alerts feed")
        header = doc.get("header")
        header_timestamp = (
            header_epoch_seconds(header.get("timestamp"))
            if isinstance(header, dict)
            else None
        )
        entities = doc.get("entity", [])
        if not isinstance(entities, list):
            raise DecodeError("the alerts feed entity is not a list")
        return alert_presence(
            fetch,
            "alerts",
            header_timestamp,
            [_camsys_alert(as_object(e, "an alert entity")) for e in entities],
        )
