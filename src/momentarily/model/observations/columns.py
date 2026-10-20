"""Parquet column layout of the normalized observation records.

The layout is derived from the Pydantic contract, so there is one source of
truth, and pinned: ``schema/model/s1/observations/parquet/<record>.txt`` holds
``describe()`` of every record, and a test fails when the derived layout
drifts from it or from the digest pinned for ``LAYOUT_VERSION``. Parquet
files are kept indefinitely, so any change bumps ``LAYOUT_VERSION``.

Type names follow Arrow's (``int64``, ``bool``, ``string``, ``float64``,
``list``, ``struct``, ``map``); the derive job turns them into a pyarrow
schema. Rules:

  * ``int`` (and int ``Literal``) -> ``int64``; ``str`` (and str ``Literal``)
    -> ``string``; ``bool`` -> ``bool``; ``float`` -> ``float64``.
  * A nested model -> ``struct``; ``list[X]`` -> ``list<X>``;
    ``dict[str, X]`` -> ``map<string, X>`` written as key-sorted pairs.
  * ``dict[str, JsonValue]`` -> ``json``: a ``string`` column holding
    ``canonical_entity_json`` of the value.
  * A union of models tagged by ``kind`` (``Lineage``) -> ``struct<kind,
    <variant>...>``: the tag, then one nullable struct per variant, exactly
    one of them set.
  * ``X | None`` -> a nullable column. Null means "the payload omitted it".
  * ``X | Unknown`` -> the nullable column plus a nullable ``string`` sibling
    ``<name>__unknown`` holding the reason. Value null and reason null is
    ``None``; reason set is ``Unknown``. The three states stay distinct.

``to_row`` and ``from_row`` convert a record to and from this layout;
``from_row(cls, to_row(r)) == r`` for every valid record. Each file's
key-value metadata holds one ``DerivedFileMetadata`` under
``PARQUET_METADATA_KEY`` (``parquet_metadata`` / ``read_parquet_metadata``).
"""

from __future__ import annotations

import json
import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel, JsonValue

from momentarily.model.conventions import Unknown
from momentarily.model.observations.contracts import (
    PARQUET_METADATA_KEY,
    AlertPresence,
    AlertVersion,
    DerivedFileMetadata,
    EquipmentObservation,
    FeedHeaderObservation,
    LegacyCollectorHealth,
    LegacyFetchOutcome,
    LegacyRouteAlertState,
    TripUpdateObservation,
    VehicleObservation,
    canonical_entity_json,
)
from momentarily.model.observations.keys import DailyRecord, HourlyRecord

LAYOUT_VERSION = 2
UNKNOWN_SUFFIX = "__unknown"
TAG = "kind"

NORMALIZED: dict[HourlyRecord | DailyRecord, type[BaseModel]] = {
    "vehicle": VehicleObservation,
    "trip_update": TripUpdateObservation,
    "feed_header": FeedHeaderObservation,
    "alert_presence": AlertPresence,
    "equipment": EquipmentObservation,
    "alert_version": AlertVersion,
    "legacy_route_alert_state": LegacyRouteAlertState,
    "legacy_fetch_outcome": LegacyFetchOutcome,
    "legacy_collector_health": LegacyCollectorHealth,
}


def _not_null(nullable: bool) -> str:
    return "" if nullable else " not null"


ScalarKind = Literal["int64", "bool", "string", "float64", "json"]


@dataclass(frozen=True)
class Scalar:
    kind: ScalarKind

    def arrow(self) -> str:
        return "string" if self.kind == "json" else self.kind


@dataclass(frozen=True)
class Struct:
    fields: tuple[Column, ...]

    def arrow(self) -> str:
        return f"struct<{', '.join(c.arrow() for c in self.fields)}>"


@dataclass(frozen=True)
class Variants:
    """A tagged union of models: the ``kind`` tag, then one nullable struct
    per variant (without the tag), exactly one set."""

    options: tuple[tuple[str, Struct], ...]

    def arrow(self) -> str:
        inner = ", ".join(f"{name}: {s.arrow()}" for name, s in self.options)
        return f"struct<{TAG}: string not null, {inner}>"


@dataclass(frozen=True)
class ListOf:
    item: ColumnType
    item_nullable: bool

    def arrow(self) -> str:
        return f"list<{self.item.arrow()}{_not_null(self.item_nullable)}>"


@dataclass(frozen=True)
class MapOf:
    """Keys are always non-null strings."""

    value: ColumnType
    value_nullable: bool

    def arrow(self) -> str:
        return f"map<string, {self.value.arrow()}{_not_null(self.value_nullable)}>"


ColumnType = Scalar | Struct | Variants | ListOf | MapOf


@dataclass(frozen=True)
class Column:
    name: str
    type: ColumnType
    nullable: bool
    unknown_of: str | None
    """For a ``__unknown`` column: the column whose reason it holds."""

    def arrow(self) -> str:
        return f"{self.name}: {self.type.arrow()}{_not_null(self.nullable)}"


class UnsupportedFieldError(TypeError):
    """A contract field this layout has no column type for."""


def _strip(tp: Any) -> Any:
    while get_origin(tp) is Annotated:
        tp = get_args(tp)[0]
    return tp


def _split(tp: Any) -> tuple[list[Any], bool, bool]:
    """(value types, nullable, unknown-able) of a field annotation."""
    tp = _strip(tp)
    if get_origin(tp) not in (Union, types.UnionType):
        return [tp], False, False
    members = [_strip(a) for a in get_args(tp)]
    rest = [m for m in members if m not in (type(None), Unknown)]
    return rest, type(None) in members, Unknown in members


def _tag_of(model: Any) -> str | None:
    if not (isinstance(model, type) and issubclass(model, BaseModel)):
        return None
    field = model.model_fields.get(TAG)
    args = get_args(_strip(field.annotation)) if field is not None else ()
    return args[0] if len(args) == 1 and isinstance(args[0], str) else None


def _type_of(options: list[Any], where: str) -> ColumnType:
    if len(options) > 1:
        tags = [_tag_of(m) for m in options]
        if any(t is None for t in tags):
            raise UnsupportedFieldError(f"{where}: union of untagged types {options!r}")
        return Variants(
            tuple(
                (str(t), Struct(tuple(c for c in _columns(m) if c.name != TAG)))
                for t, m in zip(tags, options, strict=True)
            )
        )
    tp = options[0]
    if tp is JsonValue:
        raise UnsupportedFieldError(
            f"{where}: bare JsonValue; use dict[str, JsonValue]"
        )
    origin = get_origin(tp)
    if origin is Literal:
        kinds: set[type[Any]] = {type(a) for a in get_args(tp)}
        if kinds == {str}:
            return Scalar("string")
        if kinds == {bool}:
            return Scalar("bool")
        if kinds == {int}:
            return Scalar("int64")
        raise UnsupportedFieldError(f"{where}: mixed Literal {tp!r}")
    if origin is list:
        item, nullable, unknown = _split(get_args(tp)[0])
        if unknown:
            raise UnsupportedFieldError(f"{where}: Unknown inside a list")
        return ListOf(_type_of(item, where), nullable)
    if origin is dict:
        key, value = get_args(tp)
        if _strip(key) is not str:
            raise UnsupportedFieldError(f"{where}: map keys must be str")
        if value is JsonValue:
            return Scalar("json")
        value_types, nullable, unknown = _split(value)
        if unknown:
            raise UnsupportedFieldError(f"{where}: Unknown inside a map")
        return MapOf(_type_of(value_types, where), nullable)
    if isinstance(tp, type) and issubclass(tp, BaseModel):
        return Struct(_columns(tp))
    simple: dict[Any, ScalarKind] = {
        bool: "bool",
        int: "int64",
        str: "string",
        float: "float64",
    }
    if tp in simple:
        return Scalar(simple[tp])
    raise UnsupportedFieldError(f"{where}: no column type for {tp!r}")


def _columns(model: type[BaseModel]) -> tuple[Column, ...]:
    out: list[Column] = []
    for name, info in model.model_fields.items():
        col = info.alias or name
        options, nullable, unknown = _split(info.annotation)
        where = f"{model.__name__}.{name}"
        out.append(Column(col, _type_of(options, where), nullable or unknown, None))
        if unknown:
            out.append(Column(col + UNKNOWN_SUFFIX, Scalar("string"), True, col))
    return tuple(out)


def layout(model: type[BaseModel]) -> tuple[Column, ...]:
    """The parquet columns of ``model``, in field order."""
    return _columns(model)


def describe(model: type[BaseModel]) -> str:
    """One ``name: arrow_type [not null]`` line per column, newline-ended."""
    return "".join(f"{c.arrow()}\n" for c in layout(model))


def _encode(value: Any, tp: ColumnType) -> Any:
    if value is None:
        return None
    match tp:
        case Scalar(kind="json"):
            return canonical_entity_json(value).decode()
        case Scalar():
            return value
        case Struct(fields=fields):
            return _encode_struct(value, fields)
        case Variants(options=options):
            tag = value[TAG]
            return {
                TAG: tag,
                **{
                    name: _encode_struct(value, s.fields) if name == tag else None
                    for name, s in options
                },
            }
        case ListOf(item=item):
            return [_encode(v, item) for v in value]
        case MapOf(value=inner):
            return [(k, _encode(v, inner)) for k, v in sorted(value.items())]


def _is_unknown(value: Any) -> bool:
    return isinstance(value, dict) and value.get("unknown") is True  # pyright: ignore[reportUnknownMemberType]


def _encode_struct(doc: dict[str, Any], columns: tuple[Column, ...]) -> dict[str, Any]:
    """Only a column with an ``__unknown`` sibling can hold an Unknown, so an
    opaque payload that happens to contain ``"unknown": true`` stays data."""
    unknowable = {c.unknown_of for c in columns if c.unknown_of is not None}
    row: dict[str, Any] = {}
    for c in columns:
        if c.unknown_of is not None:
            raw = doc[c.unknown_of]
            row[c.name] = raw["reason"] if _is_unknown(raw) else None
            continue
        raw = doc[c.name]
        unknown = c.name in unknowable and _is_unknown(raw)
        row[c.name] = None if unknown else _encode(raw, c.type)
    return row


def _decode(value: Any, tp: ColumnType) -> Any:
    if value is None:
        return None
    match tp:
        case Scalar(kind="json"):
            return json.loads(value)
        case Scalar():
            return value
        case Struct(fields=fields):
            return _decode_struct(value, fields)
        case Variants(options=options):
            tag = value[TAG]
            set_variants = [name for name, _ in options if value[name] is not None]
            if set_variants != [tag]:
                raise ValueError(f"{TAG} {tag!r} but variants set: {set_variants}")
            fields = dict(options)[tag].fields
            return {TAG: tag, **_decode_struct(value[tag], fields)}
        case ListOf(item=item):
            return [_decode(v, item) for v in value]
        case MapOf(value=inner):
            return {k: _decode(v, inner) for k, v in value}


def _decode_struct(row: dict[str, Any], columns: tuple[Column, ...]) -> dict[str, Any]:
    doc: dict[str, Any] = {}
    reasons = {c.unknown_of: row[c.name] for c in columns if c.unknown_of is not None}
    for c in columns:
        if c.unknown_of is not None:
            continue
        reason = reasons.get(c.name)
        if reason is not None:
            if row[c.name] is not None:
                raise ValueError(f"{c.name} has both a value and an unknown reason")
            doc[c.name] = {"unknown": True, "reason": reason}
        else:
            doc[c.name] = _decode(row[c.name], c.type)
    return doc


def to_row(record: BaseModel) -> dict[str, Any]:
    """``record`` as one parquet row of ``layout(type(record))``."""
    doc = record.model_dump(mode="json", by_alias=True)
    return _encode_struct(doc, layout(type(record)))


def from_row[M: BaseModel](model: type[M], row: dict[str, Any]) -> M:
    """The record a ``to_row`` row holds, validated by ``model``."""
    return model.model_validate(_decode_struct(row, layout(model)))


def parquet_metadata(meta: DerivedFileMetadata) -> dict[str, str]:
    """The key-value metadata a derived parquet file is written with."""
    doc = meta.model_dump(mode="json", by_alias=True)
    return {PARQUET_METADATA_KEY: canonical_entity_json(doc).decode()}


def read_parquet_metadata(
    kv: Mapping[bytes, bytes] | Mapping[str, str],
) -> DerivedFileMetadata:
    """The ``DerivedFileMetadata`` of a file's key-value metadata (pyarrow
    returns bytes keys and values)."""
    for key, value in kv.items():
        k = key.decode() if isinstance(key, bytes) else key
        if k == PARQUET_METADATA_KEY:
            return DerivedFileMetadata.model_validate_json(value)
    raise KeyError(f"no {PARQUET_METADATA_KEY} in the file metadata")
