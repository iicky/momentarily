"""What every legacy adapter shares: the objects it reads, the legacy
periods, lineage, and strict reading of the archived JSON.

An adapter reads one archived object and returns the records it backs, each
with the legacy objects it was built from. An object that is not the JSON its
archive writes raises ``MalformedObjectError``; the caller counts and lists
it. A well-formed object whose records a contract rejects is not malformed:
that error propagates.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import cached_property
from typing import Any, cast

from pydantic import BaseModel

from momentarily.model.conventions import Unknown
from momentarily.model.observations.contracts import (
    LegacyArchive,
    LegacyLineage,
    LegacyPeriod,
)
from momentarily.model.observations.keys import utc_day

DECODER_VERSION = "legacy-1"


def _period(start: str) -> LegacyPeriod:
    return LegacyPeriod.model_validate({"start": start, "end": None})


# The first UTC day each archive holds, read from the mirror on 2026-10-20.
# ``end`` stays null until the recorder writes the replacing record.
PERIODS: dict[LegacyArchive, LegacyPeriod] = {
    "archive/trace": _period("2026-08-12"),
    "archive/trip_updates": _period("2026-06-15"),
    "archive/vehicles": _period("2026-06-21"),
    "archive/alerts": _period("2026-07-21"),
    "archive/alerts_liveness": _period("2026-09-02"),
    "archive/ene": _period("2026-07-21"),
    "archive/health": _period("2026-09-11"),
    "v1/predictions": _period("2026-07-21"),
}


class MalformedObjectError(ValueError):
    """An archived object that is not what its archive writes."""


@dataclass(frozen=True)
class LegacyObject:
    """One archived object: its R2 key and its bytes."""

    key: str
    body: bytes

    @cached_property
    def sha256(self) -> str:
        return hashlib.sha256(self.body).hexdigest()

    def source(self, observed_at: int) -> SourceObject:
        return SourceObject(self.key, self.sha256, observed_at)


@dataclass(frozen=True)
class SourceObject:
    """A legacy object as file provenance names it."""

    key: str
    sha256: str
    observed_at: int


@dataclass(frozen=True)
class Emitted:
    """A record and the legacy objects it was built from."""

    record: BaseModel
    inputs: tuple[SourceObject, ...]


_UNKNOWN: dict[str, Unknown] = {}


def unknown(reason: str) -> Unknown:
    """The shared ``Unknown`` for ``reason`` (records are immutable)."""
    found = _UNKNOWN.get(reason)
    if found is None:
        found = _UNKNOWN[reason] = Unknown(unknown=True, reason=reason)
    return found


def lineage(archive: LegacyArchive, key: str, row_index: int | None) -> LegacyLineage:
    return LegacyLineage(
        kind="legacy",
        archive=archive,
        source_key=key,
        row_index=row_index,
        period=PERIODS[archive],
    )


def match_key(pattern: re.Pattern[str], key: str) -> re.Match[str]:
    m = pattern.fullmatch(key)
    if m is None:
        raise MalformedObjectError(f"{key} is not a key this archive writes")
    return m


def check_day(key: str, folder: str, at: int) -> None:
    """The archive files every object under the UTC day of ``at``."""
    if folder != utc_day(at):
        raise MalformedObjectError(f"{key} is filed under another day than {at}")


def load_json(obj: LegacyObject) -> Any:
    try:
        return json.loads(obj.body)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise MalformedObjectError(f"{obj.key}: not JSON ({err})") from err


def as_dict(value: Any, where: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MalformedObjectError(f"{where} is not an object")
    return cast(dict[str, Any], value)


def as_list(value: Any, where: str) -> list[Any]:
    if not isinstance(value, list):
        raise MalformedObjectError(f"{where} is not a list")
    return cast(list[Any], value)


def get(doc: dict[str, Any], name: str, where: str) -> Any:
    if name not in doc:
        raise MalformedObjectError(f"{where} has no {name}")
    return doc[name]


def get_int(doc: dict[str, Any], name: str, where: str) -> int:
    value = get(doc, name, where)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise MalformedObjectError(f"{where}.{name} is not a non-negative integer")
    return value


def get_opt_int(doc: dict[str, Any], name: str, where: str) -> int | None:
    if get(doc, name, where) is None:
        return None
    return get_int(doc, name, where)


def get_str(doc: dict[str, Any], name: str, where: str) -> str:
    value = get(doc, name, where)
    if not isinstance(value, str):
        raise MalformedObjectError(f"{where}.{name} is not a string")
    return value


def get_opt_str(doc: dict[str, Any], name: str, where: str) -> str | None:
    if get(doc, name, where) is None:
        return None
    return get_str(doc, name, where)


def get_bool(doc: dict[str, Any], name: str, where: str) -> bool:
    value = get(doc, name, where)
    if not isinstance(value, bool):
        raise MalformedObjectError(f"{where}.{name} is not a boolean")
    return value
