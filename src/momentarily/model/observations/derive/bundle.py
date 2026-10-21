"""Reading the recorder's raw objects: GTFS-RT tick bundles and gzip blobs.

A tick bundle (``raw/gtfsrt/...bin``) is a series of independent gzip
members: member 0 is the ``TickBundleIndex`` JSON, then one member per
GTFS-RT body the tick read, numbered from 1 in the order the index lists the
stored fetches (``worker/src/recorder.ts``). A blob (``raw/blob/...gz``) is
one gzip member holding one body.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

from pydantic import ValidationError

from momentarily.model.observations.contracts import TickBundleIndex
from momentarily.model.observations.keys import BundleKey, KeyLayoutError, parse_key


class RawObjectError(ValueError):
    """A stored object that is not what the recorder writes under its key."""


def gzip_members(data: bytes) -> list[bytes]:
    """The decompressed members of a concatenation of gzip members."""
    members: list[bytes] = []
    rest = data
    while rest:
        d = zlib.decompressobj(31)
        try:
            body = d.decompress(rest) + d.flush()
        except zlib.error as err:
            raise RawObjectError(f"not gzip: {err}") from err
        if not d.eof:
            raise RawObjectError("a gzip member is truncated")
        members.append(body)
        rest = d.unused_data
    return members


def read_blob(key: str, data: bytes) -> bytes:
    """The body a blob holds."""
    members = gzip_members(data)
    if len(members) != 1:
        raise RawObjectError(f"{key} holds {len(members)} gzip members, not 1")
    return members[0]


@dataclass(frozen=True)
class Bundle:
    index: TickBundleIndex
    payloads: dict[int, bytes]
    """Body by member number (1-based)."""


def read_bundle(key: str, data: bytes) -> Bundle:
    """The index and bodies of the tick bundle stored under ``key``. The
    index must name this bundle's invocation and number its stored fetches
    1..n, one per body member."""
    try:
        parsed = parse_key(key)
    except KeyLayoutError as err:
        raise RawObjectError(str(err)) from err
    if not isinstance(parsed, BundleKey):
        raise RawObjectError(f"{key} is not a tick bundle key")
    members = gzip_members(data)
    if not members:
        raise RawObjectError(f"{key} is empty")
    try:
        index = TickBundleIndex.model_validate_json(members[0])
    except ValidationError as err:
        raise RawObjectError(f"{key}: index member: {err}") from err
    if (index.scheduled_at, index.started_at_ms) != (
        parsed.scheduled_at,
        parsed.started_at_ms,
    ):
        raise RawObjectError(f"{key}: index names tick {index.tick_id}")
    stored = [f for f in index.fetches if f.payload_member is not None]
    if [f.payload_member for f in stored] != list(range(1, len(members))):
        raise RawObjectError(
            f"{key}: {len(members) - 1} body members, index numbers "
            f"{[f.payload_member for f in stored]}"
        )
    if any(f.payload_key != key for f in stored):
        raise RawObjectError(f"{key}: index points a member at another key")
    return Bundle(index, {i: members[i] for i in range(1, len(members))})
