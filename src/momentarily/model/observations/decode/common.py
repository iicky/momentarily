"""What every decoder shares: its version, its error, and the fetch it reads."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import cast

from pydantic import JsonValue, ValidationError

from momentarily.model.observations.contracts import RawLineage, SourceId

# Pinned in every derived file's metadata. Bump it whenever a decoder's output
# for the same bytes changes; the golden digests in the tests pin each version.
DECODER_VERSION = "1"


class DecodeError(ValueError):
    """The payload is not what its source publishes: not parseable, missing
    a required field, or not the bytes its lineage names. Nothing is
    returned for such a payload, never a partial result."""


@dataclass(frozen=True, slots=True)
class RawFetch:
    """The fetch a payload came from, as every decoded record repeats it."""

    fetch_id: str
    source: SourceId
    observed_at: int
    lineage: RawLineage


def check_payload(payload: bytes, fetch: RawFetch, sources: tuple[str, ...]) -> None:
    """Reject bytes a decoder cannot attribute: another source's payload, or
    bytes whose sha256 differs from the lineage's."""
    if fetch.source not in sources:
        raise DecodeError(f"{fetch.source} is not decoded here ({', '.join(sources)})")
    if hashlib.sha256(payload).hexdigest() != fetch.lineage.payload_sha256:
        raise DecodeError(f"{fetch.fetch_id}: payload bytes differ from payload_sha256")


def _reject_constant(name: str) -> JsonValue:
    raise DecodeError(f"{name} is not JSON")


def load_json(payload: bytes) -> JsonValue:
    """Strict JSON: UTF-8, no NaN or infinities (the alert digest cannot
    hold them, and no upstream sends them)."""
    try:
        return cast(JsonValue, json.loads(payload, parse_constant=_reject_constant))
    except ValueError as err:  # JSONDecodeError and UnicodeDecodeError
        if isinstance(err, DecodeError):
            raise
        raise DecodeError(f"payload is not JSON: {err}") from err


def as_object(value: JsonValue, what: str) -> dict[str, JsonValue]:
    if not isinstance(value, dict):
        raise DecodeError(f"{what} is not a JSON object")
    return value


def as_instant(value: JsonValue, what: str) -> int | None:
    """A published epoch-seconds field: absent or null is None; anything but
    a non-negative integral number is a payload error."""
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DecodeError(f"{what} is not epoch seconds: {value!r}")
    return value


_MAX_SAFE_INTEGER = 2**53 - 1


def header_epoch_seconds(value: JsonValue) -> int | None:
    """A JSON feed's header timestamp, read exactly as the recorder reads it
    for ``FeedFetch.header_timestamp``: a non-negative safe integer, or a
    string of 1 to 15 digits. Anything else is None, never an error."""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return value if 0 <= value <= _MAX_SAFE_INTEGER else None
    if isinstance(value, str) and re.fullmatch(r"[0-9]{1,15}", value):
        return int(value)
    return None


@contextmanager
def decode_errors(fetch: RawFetch) -> Generator[None]:
    """Turn what the contracts reject, or a payload nested too deep to read,
    into ``DecodeError`` so callers see one error type."""
    try:
        yield
    except (ValidationError, RecursionError) as err:
        raise DecodeError(f"{fetch.fetch_id}: {type(err).__name__}: {err}") from err
