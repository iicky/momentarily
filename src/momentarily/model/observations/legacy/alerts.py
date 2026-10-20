"""``archive/alerts`` -> legacy ``AlertVersion``.

Object: ``archive/alerts/<YYYY-MM-DD>/<updated_at>-<alert id>.json`` holding
``{observed_at, alert}``: written when the Worker first saw an
``(alert_id, updated_at)`` pair, at 5-minute resolution, the entity
re-serialized. A version is a distinct ``(alert_id, content_digest)``; a
re-sighting of an identical digest (the Worker lost its dedupe state) adds
an input, not a version. ``first_seen`` is the earliest sighting. The
presence window was not kept, so the bounds are ``Unknown``, except that a
later ``updated_at`` of the same id proves the version was ``superseded``.

Unlike the other adapters this one needs every object at once: whether a
version was superseded depends on later objects.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, cast

from pydantic import JsonValue

from momentarily.model.observations.contracts import (
    AlertVersion,
    LegacyArchive,
    alert_content_digest,
)
from momentarily.model.observations.legacy.common import (
    Emitted,
    LegacyObject,
    MalformedObjectError,
    SourceObject,
    as_dict,
    check_day,
    get,
    get_int,
    get_str,
    lineage,
    load_json,
    match_key,
    unknown,
)

ARCHIVE: LegacyArchive = "archive/alerts"

_KEY = re.compile(
    r"archive/alerts/(?P<day>\d{4}-\d{2}-\d{2})/(?P<updated>\d+)-(?P<id>[A-Za-z0-9._-]+)\.json"
)
_NO_WINDOW = unknown("archive/alerts keeps first sightings only, no presence window")
_NOT_PROVEN = unknown(
    "archive/alerts keeps no presence window; no later updated_at proves supersession"
)


@dataclass(frozen=True)
class Sighting:
    """One parsed ``archive/alerts`` object."""

    source: SourceObject
    alert_id: str
    digest: str
    entity: dict[str, JsonValue]
    mercury: dict[str, Any]


def _mercury(entity: dict[str, Any]) -> dict[str, Any]:
    """The entity's ``mercury_alert`` fields, ``{}`` when it has none."""
    alert = entity.get("alert")
    if not isinstance(alert, dict):
        return {}
    mercury = cast(dict[str, Any], alert).get("transit_realtime.mercury_alert")
    return cast(dict[str, Any], mercury) if isinstance(mercury, dict) else {}


def parse(obj: LegacyObject) -> Sighting:
    m = match_key(_KEY, obj.key)
    doc = as_dict(load_json(obj), obj.key)
    observed_at = get_int(doc, "observed_at", obj.key)
    check_day(obj.key, m["day"], observed_at)
    entity = as_dict(get(doc, "alert", obj.key), f"{obj.key}.alert")
    alert_id = get_str(entity, "id", f"{obj.key}.alert")
    published = _mercury(entity)
    try:
        digest = alert_content_digest(entity)
    except ValueError as err:
        raise MalformedObjectError(f"{obj.key}: {err}") from err
    return Sighting(obj.source(observed_at), alert_id, digest, entity, published)


def _updated_at(s: Sighting) -> int | None:
    value = s.mercury.get("updated_at")
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def versions(sightings: Iterable[Sighting]) -> list[Emitted]:
    """One AlertVersion per distinct ``(alert_id, content_digest)``, in
    first-sighting order."""
    groups: dict[tuple[str, str], list[Sighting]] = {}
    for s in sorted(sightings, key=lambda s: (s.source.observed_at, s.source.key)):
        groups.setdefault((s.alert_id, s.digest), []).append(s)
    by_id: dict[str, list[Sighting]] = {}
    for (alert_id, _), seen in groups.items():
        by_id.setdefault(alert_id, []).append(seen[0])
    out: list[Emitted] = []
    for seen in groups.values():
        first = seen[0]
        inputs = [s.source for s in seen]
        updated = _updated_at(first)
        later = [
            s
            for s in by_id[first.alert_id]
            if updated is not None and (u := _updated_at(s)) is not None and u > updated
        ]
        if later:
            proof = min(later, key=lambda s: (_updated_at(s), s.source.observed_at))
            inputs.append(proof.source)
        record = AlertVersion(
            lineage=lineage(ARCHIVE, first.source.key, None),
            source="alerts",
            alert_id=first.alert_id,
            content_digest=first.digest,
            entity=first.entity,
            alert_type=first.mercury.get("alert_type"),
            created_at=first.mercury.get("created_at"),
            updated_at=first.mercury.get("updated_at"),
            absent_before=_NO_WINDOW,
            first_seen=first.source.observed_at,
            last_seen=_NO_WINDOW,
            gone_by=_NO_WINDOW,
            gone_kind="superseded" if later else _NOT_PROVEN,
            unobserved=_NO_WINDOW,
            window=_NO_WINDOW,
        )
        out.append(Emitted(record, tuple(dict.fromkeys(inputs))))
    return out
