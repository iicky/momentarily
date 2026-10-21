"""Deterministic parquet files of observation records (needs the ``derive``
extra, ``pyarrow``).

A file holds records of one contract, in the layout ``columns`` derives and
``schema/model/s1/observations/parquet/`` pins, sorted by natural key, with
one ``DerivedFileMetadata`` in its key-value metadata. Compression, row
groups and every other writer setting are fixed here, and ``built_at`` comes
from the inputs, so the same records and inputs give the same bytes.
"""

# pyarrow-stubs leave some generics unknown (Field item types, Schema
# metadata, the parquet reader and writer): silence that noise in this file.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Literal, get_args

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel

from momentarily.model.conventions import Interval, Provenance, Unknown
from momentarily.model.observations.columns import (
    LAYOUT_VERSION,
    Column,
    ColumnType,
    ListOf,
    MapOf,
    Scalar,
    Struct,
    Variants,
    from_row,
    layout,
    parquet_metadata,
    read_parquet_metadata,
    to_row,
)
from momentarily.model.observations.contracts import (
    SCHEMA_ID,
    AlertPresence,
    AlertVersion,
    DerivedFileMetadata,
    EquipmentObservation,
    FeedHeaderObservation,
    LegacyArchive,
    LegacyCollectorHealth,
    LegacyFetchOutcome,
    LegacyLineage,
    LegacyRouteAlertState,
    RawLineage,
    TripUpdateObservation,
    VehicleObservation,
)
from momentarily.model.observations.keys import (
    BlobKey,
    BundleKey,
    DerivedDayKey,
    DerivedHourKey,
    KeyLayoutError,
    TickKey,
    parse_key,
    utc_day,
)

COMPRESSION = "zstd"
COMPRESSION_LEVEL = 9
ROW_GROUP_SIZE = 1 << 20
DATA_PAGE_SIZE = 1 << 20

NaturalKey = tuple[str | int, ...]
FileLineage = Literal["raw", "legacy"]
_LEGACY_PREFIXES = tuple(f"{a}/" for a in get_args(LegacyArchive))


class DerivedFileError(ValueError):
    """Records that cannot form one derived file."""


def _arrow_type(tp: ColumnType) -> pa.DataType:
    match tp:
        case Scalar(kind="int64"):
            return pa.int64()
        case Scalar(kind="bool"):
            return pa.bool_()
        case Scalar(kind="float64"):
            return pa.float64()
        case Scalar():
            return pa.string()
        case Struct(fields=fields):
            return pa.struct([_arrow_field(c) for c in fields])
        case Variants(options=options):
            return pa.struct(
                [pa.field("kind", pa.string(), nullable=False)]
                + [pa.field(name, _arrow_type(s), nullable=True) for name, s in options]
            )
        case ListOf(item=item, item_nullable=nullable):
            return pa.list_(pa.field("element", _arrow_type(item), nullable=nullable))
        case MapOf(value=value, value_nullable=nullable):
            item = pa.field("value", _arrow_type(value), nullable=nullable)
            # pyarrow takes a Field as the item (nullability); the stubs lag.
            return pa.map_(pa.string(), item)  # pyright: ignore[reportArgumentType]


def _arrow_field(column: Column) -> pa.Field[pa.DataType]:
    return pa.field(column.name, _arrow_type(column.type), nullable=column.nullable)


def arrow_schema(model: type[BaseModel]) -> pa.Schema:
    """The pyarrow schema of ``columns.layout(model)``."""
    return pa.schema([_arrow_field(c) for c in layout(model)])


def _not_null(field: pa.Field[pa.DataType]) -> str:
    return "" if field.nullable else " not null"


_SCALAR_NAMES: tuple[tuple[pa.DataType, str], ...] = (
    (pa.int64(), "int64"),
    (pa.bool_(), "bool"),
    (pa.float64(), "float64"),
    (pa.string(), "string"),
)


def _describe_type(tp: pa.DataType) -> str:
    for scalar, name in _SCALAR_NAMES:
        if tp.equals(scalar):
            return name
    if isinstance(tp, pa.MapType):
        key: pa.Field[pa.DataType] = tp.key_field
        item: pa.Field[pa.DataType] = tp.item_field
        return f"map<{_describe_type(key.type)}, {_describe_type(item.type)}{_not_null(item)}>"
    if isinstance(tp, pa.ListType):
        element: pa.Field[pa.DataType] = tp.value_field
        return f"list<{_describe_type(element.type)}{_not_null(element)}>"
    if isinstance(tp, pa.StructType):
        fields: list[pa.Field[pa.DataType]] = [
            tp.field(i) for i in range(tp.num_fields)
        ]
        return f"struct<{', '.join(_describe_field(f) for f in fields)}>"
    raise DerivedFileError(f"no layout name for arrow type {tp}")


def _describe_field(field: pa.Field[pa.DataType]) -> str:
    return f"{field.name}: {_describe_type(field.type)}{_not_null(field)}"


def file_layout(path: Path) -> str:
    """The schema of the parquet file at ``path`` in the text form of
    ``columns.describe``, to compare with the committed layout."""
    fields: list[pa.Field[pa.DataType]] = list(pq.read_schema(path))
    return "".join(f"{_describe_field(f)}\n" for f in fields)


def _lineage_key(lineage: RawLineage | LegacyLineage) -> NaturalKey:
    if isinstance(lineage, LegacyLineage):
        row = -1 if lineage.row_index is None else lineage.row_index
        return ("legacy", lineage.source_key, row)
    raise DerivedFileError("a raw-lineage record keys on its fetch, not its lineage")


def _index(value: int | Unknown) -> int:
    if isinstance(value, Unknown):
        raise DerivedFileError("a raw record needs its index")
    return value


def natural_key(record: BaseModel) -> NaturalKey:
    """The key that identifies ``record`` within its contract (doc §3.2 and
    §3.4): raw entity records on their fetch and position, legacy rows on
    their source object and row."""
    match record:
        case VehicleObservation() | TripUpdateObservation():
            if isinstance(record.lineage, LegacyLineage):
                return _lineage_key(record.lineage)
            return ("raw", str(record.fetch_id), _index(record.entity_index))
        case EquipmentObservation():
            if isinstance(record.lineage, LegacyLineage):
                return _lineage_key(record.lineage)
            return (
                "raw",
                str(record.fetch_id),
                record.equipment_id,
                record.record_index,
            )
        case FeedHeaderObservation() | AlertPresence():
            return (record.fetch_id,)
        case AlertVersion():
            # One row per run of sightings: a version that leaves and comes
            # back is two rows, told apart by first_seen (-1 when Unknown).
            first = record.first_seen
            return (
                record.source,
                record.alert_id,
                record.content_digest,
                first if isinstance(first, int) else -1,
            )
        case LegacyRouteAlertState():
            return _lineage_key(record.lineage)
        case LegacyFetchOutcome():
            return (record.legacy_fetch_id,)
        case LegacyCollectorHealth():
            return (record.legacy_tick_id,)
        case _:
            raise DerivedFileError(f"{type(record).__name__} is not a derived record")


def file_metadata(
    file_key: str,
    decoder_version: str,
    row_count: int,
    inputs: Mapping[str, tuple[str, int]],
    code_sha: str,
    code_dirty: bool | None,
) -> DerivedFileMetadata:
    """Metadata of a derived file built from ``inputs`` (key -> (sha256,
    observed_at)). ``built_at`` is the latest ``observed_at``."""
    if not inputs:
        raise DerivedFileError("a derived file needs at least one input")
    key = parse_key(file_key)
    if isinstance(key, DerivedDayKey):
        day = key.day
    elif isinstance(key, DerivedHourKey):
        day = utc_day(key.hour_start)
    else:
        raise DerivedFileError(f"{file_key} is not a derived/ key")
    ordered = sorted(inputs)
    captured = {k: inputs[k][1] for k in ordered}
    return DerivedFileMetadata.model_validate(
        {
            "schema": SCHEMA_ID,
            "file_key": file_key,
            "layout_version": LAYOUT_VERSION,
            "decoder_version": decoder_version,
            "row_count": row_count,
            "provenance": Provenance.model_validate(
                {
                    "code_sha": code_sha,
                    "code_dirty": code_dirty,
                    "schema": SCHEMA_ID,
                    "inputs": {k: inputs[k][0] for k in ordered},
                    "captured_at": captured,
                    "period_start": None,
                    "observed_from_utc_day": day,
                    "applied_retroactively": [],
                    "built_at": max(captured.values()),
                }
            ),
        }
    )


def _raw_input(key: str) -> bool:
    try:
        return isinstance(parse_key(key), TickKey | BundleKey | BlobKey)
    except KeyLayoutError:
        return False


def file_lineage(metadata: DerivedFileMetadata) -> FileLineage | None:
    """Which writer built a derived file, read from the inputs its provenance
    names (not from ``decoder_version``): ``raw`` when every input is a
    recorder tick, bundle or blob, ``legacy`` when every input is a legacy
    archive object, None for anything else."""
    inputs = metadata.provenance.inputs
    if not inputs:
        return None
    if all(k.startswith(_LEGACY_PREFIXES) for k in inputs):
        return "legacy"
    if all(_raw_input(k) for k in inputs):
        return "raw"
    return None


def read_file_metadata(data: bytes) -> DerivedFileMetadata:
    """The metadata of a derived file, read from its footer alone."""
    meta = pq.read_metadata(pa.BufferReader(data))
    return read_parquet_metadata(meta.metadata or {})


def _partition_day(record: BaseModel, key: DerivedDayKey) -> bool:
    """Whether ``record`` belongs in the daily file ``key``."""
    if isinstance(record, AlertVersion):
        if isinstance(record.window, Interval):
            start, end = (
                record.window.start,
                max(record.window.start, record.window.end - 1),
            )
            return utc_day(start) <= key.day <= utc_day(end)
        return (
            isinstance(record.first_seen, int) and utc_day(record.first_seen) == key.day
        )
    at = getattr(record, "observed_at", None)
    return isinstance(at, int) and utc_day(at) == key.day


def _check_partition(file_key: str, records: Sequence[BaseModel]) -> None:
    key = parse_key(file_key)
    for r in records:
        if isinstance(key, DerivedHourKey):
            at = getattr(r, "observed_at", None)
            ok = isinstance(at, int) and key.hour_start <= at < key.hour_start + 3600
        elif isinstance(key, DerivedDayKey):
            ok = _partition_day(r, key)
        else:
            ok = False
        if not ok:
            raise DerivedFileError(f"a record does not belong in {file_key}")


class Rows:
    """Records of one contract bound for the file ``file_key``, held as
    arrow rows: far smaller than the models, so a job can drop each batch
    of models as soon as it is added."""

    def __init__(self, model: type[BaseModel], file_key: str) -> None:
        self.model = model
        self.file_key = file_key
        self.keys: list[NaturalKey] = []
        self._schema = arrow_schema(model)
        self._tables: list[pa.Table] = []

    def __len__(self) -> int:
        return len(self.keys)

    def add(self, records: Sequence[BaseModel]) -> None:
        """Add ``records``; each must be a ``model`` that belongs in the file."""
        if not records:
            return
        if any(type(r) is not self.model for r in records):
            raise DerivedFileError("a derived file holds records of one contract")
        _check_partition(self.file_key, records)
        self.keys += [natural_key(r) for r in records]
        self._tables.append(
            pa.Table.from_pylist([to_row(r) for r in records], schema=self._schema)
        )

    def table(self) -> pa.Table:
        """Every row, in natural-key order. Raises on a repeated key."""
        order = sorted(range(len(self.keys)), key=self.keys.__getitem__)
        for a, b in pairwise(order):
            if self.keys[a] == self.keys[b]:
                raise DerivedFileError(
                    f"two records share the natural key {self.keys[a]!r}"
                )
        table = pa.concat_tables(self._tables).take(pa.array(order, pa.int64()))
        return table.combine_chunks()


def write_records(
    path: Path, records: Sequence[BaseModel], metadata: DerivedFileMetadata
) -> int:
    """Write ``records`` (one contract) to ``path``, which must end in
    ``metadata.file_key``, in natural-key order, and return the file size.
    Every record must belong in the file's hour or day. Replaces ``path``
    atomically. The bytes are reproducible for a given pyarrow version."""
    if not records:
        raise DerivedFileError("a derived file holds at least one record")
    rows = Rows(type(records[0]), metadata.file_key)
    rows.add(records)
    return write_rows(path, rows, metadata)


def write_rows(path: Path, rows: Rows, metadata: DerivedFileMetadata) -> int:
    """``write_records`` for rows already gathered in ``rows``."""
    if not rows:
        raise DerivedFileError("a derived file holds at least one record")
    if rows.file_key != metadata.file_key:
        raise DerivedFileError(f"rows for {rows.file_key}, not {metadata.file_key}")
    if not path.as_posix().endswith(f"/{metadata.file_key}"):
        raise DerivedFileError(f"{path} is not the path of {metadata.file_key}")
    if metadata.row_count != len(rows):
        raise DerivedFileError("metadata row_count differs from the records")
    if metadata.layout_version != LAYOUT_VERSION:
        raise DerivedFileError("metadata names another layout version")
    table = rows.table().replace_schema_metadata(parquet_metadata(metadata))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    # Every option that changes the bytes is pinned, defaults included.
    pq.write_table(
        table,
        tmp,
        row_group_size=ROW_GROUP_SIZE,
        version="2.6",
        use_dictionary=True,
        compression=COMPRESSION,
        write_statistics=True,
        data_page_size=DATA_PAGE_SIZE,
        compression_level=COMPRESSION_LEVEL,
        use_byte_stream_split=False,
        data_page_version="1.0",
        use_compliant_nested_type=True,
        store_schema=True,
        write_page_index=False,
        write_page_checksum=False,
    )
    os.replace(tmp, path)
    return path.stat().st_size


def read_records[M: BaseModel](
    path: Path, model: type[M]
) -> tuple[list[M], DerivedFileMetadata]:
    """The records and metadata of a file ``write_records`` wrote."""
    table = pq.read_table(path)
    kv = table.schema.metadata or {}
    rows: list[dict[str, object]] = table.to_pylist()
    return [from_row(model, r) for r in rows], read_parquet_metadata(kv)
