"""Event files: their metadata and the deterministic parquet writer.

An event file (``keys``: ``model/events/s1/<lineage>/<record>/<day>.parquet``)
holds records of one contract in the layout ``columns`` pins, sorted by
natural key, with one ``EventFileMetadata`` in its key-value metadata under
``PARQUET_METADATA_KEY``. The writer settings are layer 2's
(``observations.parquet.write_parquet``), and ``built_at`` comes from the
inputs, so the same records and inputs give the same bytes.

Overwrite rule (``resolve_write``), layer 2's:

* no file under the key: written;
* identical bytes: unchanged;
* bytes that equal this build's under the stored file's ``code_sha`` and
  ``code_dirty`` (only the code provenance differs): unchanged, kept;
* a stored file whose metadata names another key, such as another lineage's
  file: a conflict, kept even with ``overwrite``;
* a file written by a lower ``builder_version`` (and no newer layout):
  replaced, as layer 2 replaces a file of another ``DECODER_VERSION``;
* a file written by a higher ``builder_version`` or ``layout_version``: a
  conflict, kept even with ``overwrite``, so an older checkout never
  downgrades a newer build;
* any other difference, or a file that cannot be read: a conflict, kept
  unless ``overwrite`` replaces it.

The lineage is part of the key, so a build of one lineage never writes under
the other's keys.
"""

# pyarrow-stubs leave some generics unknown: silence that noise in this file.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, Field, ValidationError, model_validator
from pydantic_core import PydanticCustomError

from momentarily.model.conventions import Model, Provenance
from momentarily.model.events.columns import LAYOUT_VERSION, RECORDS
from momentarily.model.events.contracts import SCHEMA_ID, CoverageInterval
from momentarily.model.events.keys import (
    ServiceDayKey,
    UtcDayKey,
    parse_key,
)
from momentarily.model.observations.columns import from_row, to_row
from momentarily.model.observations.contracts import canonical_entity_json
from momentarily.model.observations.keys import KeyLayoutError, utc_day
from momentarily.model.observations.parquet import arrow_schema, write_parquet

PARQUET_METADATA_KEY = "momentarily.events"


def _fail(message: str) -> PydanticCustomError:
    return PydanticCustomError("metadata", "{detail}", {"detail": message})


class EventFileMetadata(Model):
    """The parquet key-value metadata of every event file, stored as
    canonical JSON under ``PARQUET_METADATA_KEY``.

    ``provenance`` reuses layer 1's: ``inputs`` maps each object read (a
    recorder tick or bundle, a layer-2 derived file, a packaged table) to its
    digest, ``captured_at`` maps the same keys to the latest instant of data
    each holds, and ``built_at`` is the latest of those (never the wall
    clock), so a rebuild gives the same bytes. ``applied_retroactively``
    names reference inputs measured after the period starts (the
    ``RouteFeed`` table on legacy days). A UTC-day file (coverage, supply)
    has no service day and observes its own day; a service-day file names
    its service date as ``period_start``.
    """

    schema_: Literal["events/s1"] = Field(alias="schema")
    file_key: str = Field(description="The events key of this file.")
    layout_version: int = Field(
        ge=1, description="columns.LAYOUT_VERSION it was written with."
    )
    builder_version: int = Field(
        ge=1,
        description="Version of the rules that built this record kind "
        "(coverage: COVERAGE_VERSION).",
    )
    row_count: int = Field(ge=0)
    provenance: Provenance

    @model_validator(mode="after")
    def _consistent(self) -> EventFileMetadata:
        try:
            key = parse_key(self.file_key)
        except KeyLayoutError as err:
            raise _fail(str(err)) from err
        p = self.provenance
        if p.schema_ != SCHEMA_ID:
            raise _fail(f"provenance schema is not {SCHEMA_ID}")
        if set(p.captured_at) != set(p.inputs):
            raise _fail("captured_at and inputs name different keys")
        if p.captured_at and p.built_at != max(p.captured_at.values()):
            raise _fail("built_at is the latest captured_at")
        if not set(p.applied_retroactively) <= set(p.inputs):
            raise _fail("applied_retroactively names inputs only")
        if isinstance(key, UtcDayKey):
            if p.period_start is not None or p.observed_from_utc_day != key.day:
                raise _fail("a UTC-day file has no service day and observes its day")
        elif isinstance(key, ServiceDayKey):
            if p.period_start != key.service_date:
                raise _fail("a service-day file starts on its service date")
        else:
            raise _fail("file_key is not an event file key")
        return self


class EventFileError(ValueError):
    """Records that cannot form one event file."""


NaturalKey = tuple[str | int, ...]


def _coverage_key(record: BaseModel) -> NaturalKey:
    assert isinstance(record, CoverageInterval)
    return (record.source, record.interval.start)


def _coverage_belongs(record: BaseModel, key: UtcDayKey | ServiceDayKey) -> bool:
    assert isinstance(record, CoverageInterval)
    assert isinstance(key, UtcDayKey)
    return record.lineage == key.lineage and utc_day(record.interval.start) == key.day


# Record kinds the writer can file so far: natural key and partition rule.
# Later slices add theirs as they start writing files.
_FILING: dict[
    str,
    tuple[
        Callable[[BaseModel], NaturalKey],
        Callable[[BaseModel, UtcDayKey | ServiceDayKey], bool],
    ],
] = {
    "coverage": (_coverage_key, _coverage_belongs),
}


def event_file_metadata(
    file_key: str,
    builder_version: int,
    row_count: int,
    inputs: Mapping[str, tuple[str, int]],
    code_sha: str,
    code_dirty: bool | None,
    applied_retroactively: Sequence[str] = (),
) -> EventFileMetadata:
    """Metadata of an event file built from ``inputs`` (key -> (digest,
    latest instant of data)). ``built_at`` is the latest of those."""
    if not inputs:
        raise EventFileError("an event file needs at least one input")
    key = parse_key(file_key)
    ordered = sorted(inputs)
    captured = {k: inputs[k][1] for k in ordered}
    return EventFileMetadata.model_validate(
        {
            "schema": SCHEMA_ID,
            "file_key": file_key,
            "layout_version": LAYOUT_VERSION,
            "builder_version": builder_version,
            "row_count": row_count,
            "provenance": {
                "code_sha": code_sha,
                "code_dirty": code_dirty,
                "schema": SCHEMA_ID,
                "inputs": {k: inputs[k][0] for k in ordered},
                "captured_at": captured,
                "period_start": key.service_date
                if isinstance(key, ServiceDayKey)
                else None,
                "observed_from_utc_day": key.day
                if isinstance(key, UtcDayKey)
                else None,
                "applied_retroactively": sorted(applied_retroactively),
                "built_at": max(captured.values()),
            },
        }
    )


def render(records: Sequence[BaseModel], metadata: EventFileMetadata) -> bytes:
    """The bytes of the event file holding ``records`` (one contract, the
    one its key names), in natural-key order. Raises ``EventFileError`` on
    an empty file, a record outside the file's lineage and day, or two
    records sharing a natural key."""
    if not records:
        raise EventFileError("an event file holds at least one record")
    key = parse_key(metadata.file_key)
    if not isinstance(key, UtcDayKey | ServiceDayKey):
        raise EventFileError(f"{metadata.file_key} is not an event file key")
    model = RECORDS[key.record]
    filing = _FILING.get(key.record)
    if filing is None:
        raise EventFileError(f"no writer files {key.record} records yet")
    natural, belongs = filing
    if any(type(r) is not model for r in records):
        raise EventFileError(f"{metadata.file_key} holds {model.__name__} records")
    if any(not belongs(r, key) for r in records):
        raise EventFileError(f"a record does not belong in {metadata.file_key}")
    if metadata.row_count != len(records):
        raise EventFileError("metadata row_count differs from the records")
    if metadata.layout_version != LAYOUT_VERSION:
        raise EventFileError("metadata names another layout version")
    ordered = sorted(records, key=natural)
    for a, b in pairwise(ordered):
        if natural(a) == natural(b):
            raise EventFileError(f"two records share the natural key {natural(a)!r}")
    doc = metadata.model_dump(mode="json", by_alias=True)
    table = pa.Table.from_pylist(
        [to_row(r) for r in ordered], schema=arrow_schema(model)
    ).replace_schema_metadata({PARQUET_METADATA_KEY: canonical_entity_json(doc)})
    sink = pa.BufferOutputStream()
    write_parquet(table, sink)
    return sink.getvalue().to_pybytes()


def read_metadata(data: bytes) -> EventFileMetadata:
    """The metadata of an event file, read from its footer alone. Raises
    ``ValueError`` (or an arrow error) for anything else."""
    kv = pq.read_metadata(pa.BufferReader(data)).metadata or {}
    value = kv.get(PARQUET_METADATA_KEY.encode())
    if value is None:
        raise ValueError(f"no {PARQUET_METADATA_KEY} in the file metadata")
    return EventFileMetadata.model_validate_json(value)


def read_records[M: BaseModel](
    data: bytes, model: type[M]
) -> tuple[list[M], EventFileMetadata]:
    """The records and metadata of an event file."""
    rows: list[dict[str, object]] = pq.read_table(pa.BufferReader(data)).to_pylist()
    return [from_row(model, r) for r in rows], read_metadata(data)


WriteOutcome = Literal["written", "unchanged", "replaced", "conflict"]


@dataclass(frozen=True)
class Resolution:
    outcome: WriteOutcome
    data: bytes | None
    """The bytes to store; None when the stored file stays."""
    problem: str | None
    """Why a conflict is one."""


def resolve_write(
    stored: bytes | None,
    records: Sequence[BaseModel],
    metadata: EventFileMetadata,
    overwrite: bool,
) -> Resolution:
    """What to do with the bytes already stored under ``metadata.file_key``
    (the module's overwrite rule)."""
    data = render(records, metadata)
    if stored is None:
        return Resolution("written", data, None)
    if stored == data:
        return Resolution("unchanged", None, None)
    try:
        old: EventFileMetadata | None = read_metadata(stored)
    except (pa.ArrowException, ValueError, ValidationError):
        old = None
    if old is not None:
        if old.file_key != metadata.file_key:
            return Resolution(
                "conflict", None, f"the stored file is {old.file_key}; never replaced"
            )
        if (
            old.builder_version > metadata.builder_version
            or old.layout_version > metadata.layout_version
        ):
            return Resolution(
                "conflict",
                None,
                f"the stored file is from builder {old.builder_version} (layout "
                f"{old.layout_version}), newer than this build; never replaced",
            )
        if old.builder_version < metadata.builder_version:
            return Resolution("replaced", data, None)
        code = {
            "code_sha": old.provenance.code_sha,
            "code_dirty": old.provenance.code_dirty,
        }
        restamped = metadata.model_copy(
            update={"provenance": metadata.provenance.model_copy(update=code)}
        )
        if render(records, restamped) == stored:
            return Resolution("unchanged", None, None)
    if overwrite:
        return Resolution("replaced", data, None)
    problem = "content differs" if old is not None else "not a readable event file"
    return Resolution("conflict", None, problem)
