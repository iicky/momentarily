"""The reviewed list of legacy archive objects the backfill treats as absent
(docs/model-observations.md §5.1, **Quarantine**).

``quarantine.json`` beside this module lists each object by its key, the
sha256 of the bytes that were reviewed, the reason and the day it was added.
Only a reviewed commit changes it. The backfill skips a listed object whose
bytes still hash to the listed sha256, writes its days without it and
records the exclusion in every file it could have filled. A listed object
whose bytes hash to anything else has changed since the review, and is an
error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from importlib.resources import files
from typing import Any, cast, get_args

from momentarily.model.observations.contracts import LegacyArchive

QUARANTINE_FILE = "quarantine.json"
_PREFIXES = tuple(f"{a}/" for a in get_args(LegacyArchive))
_FIELDS = {"key", "sha256", "reason", "added"}


class QuarantineError(ValueError):
    """A quarantine file that is not a list of valid, distinct entries."""


@dataclass(frozen=True)
class Entry:
    key: str
    sha256: str
    reason: str
    added: str


def _entry(doc: Any, where: str) -> Entry:
    if not isinstance(doc, dict) or set(cast(dict[str, Any], doc)) != _FIELDS:
        raise QuarantineError(f"{where} needs exactly {sorted(_FIELDS)}")
    entry = cast(dict[str, Any], doc)
    if not all(isinstance(entry[f], str) for f in _FIELDS):
        raise QuarantineError(f"{where}: every field is a string")
    key, sha, reason, added = (entry[f] for f in ("key", "sha256", "reason", "added"))
    if not key.startswith(_PREFIXES) or key.endswith("/"):
        raise QuarantineError(f"{where}: {key!r} is not a legacy archive object")
    if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
        raise QuarantineError(f"{where}: sha256 is not 64 lowercase hex digits")
    if not reason.strip():
        raise QuarantineError(f"{where}: the reason is empty")
    try:
        valid = date.fromisoformat(added).isoformat() == added
    except ValueError:
        valid = False
    if not valid:
        raise QuarantineError(f"{where}: added is not a YYYY-MM-DD day")
    return Entry(key, sha, reason, added)


def parse(data: bytes) -> dict[str, Entry]:
    """The entries of a quarantine file, by key."""
    try:
        doc = json.loads(data)
    except ValueError as err:
        raise QuarantineError(f"not JSON: {err}") from err
    if not isinstance(doc, dict) or set(cast(dict[str, Any], doc)) != {"objects"}:
        raise QuarantineError('the file is {"objects": [...]}')
    objects = cast(dict[str, Any], doc)["objects"]
    if not isinstance(objects, list):
        raise QuarantineError("objects is not a list")
    entries: dict[str, Entry] = {}
    for i, item in enumerate(cast(list[Any], objects)):
        entry = _entry(item, f"objects[{i}]")
        if entry.key in entries:
            raise QuarantineError(f"{entry.key} is listed twice")
        entries[entry.key] = entry
    return entries


def load() -> dict[str, Entry]:
    """The packaged quarantine file's entries."""
    data = files("momentarily.model.observations.legacy").joinpath(QUARANTINE_FILE)
    return parse(data.read_bytes())
