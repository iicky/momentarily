"""Conventions every data-model layer shares.

Time
  * ``Instant`` — a moment: integer UTC epoch seconds. Never a local wall time.
  * ``ServiceDate`` — the ISO date (YYYY-MM-DD) of a GTFS *service day*: the
    operating day a trip belongs to, which runs past midnight. It is never the
    UTC date of an instant; a train at 00:30 on the 2nd usually belongs to the
    1st's service day.
  * ``GtfsSeconds`` — seconds after "noon minus 12 h" on the service day (the
    GTFS rule, which keeps the arithmetic right across DST changes). Values at
    or past 86400 are legal: 24:04:00 is 86640.

Intervals
  Every range is half-open, ``[start, end)``: ``start`` is included and
  ``end`` is not, for instants and service dates alike. GTFS's own inclusive
  ``end_date`` is converted at the boundary (end + 1 day), never carried
  through.

Missing data
  No contract field has a default. A value the source does not provide is
  written as an explicit ``null`` (or as ``Unknown`` with a reason where a
  measurement is expected), so a missing measurement can never silently read
  as a normal one. ``Model`` enforces this when a subclass is defined.
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_core import PydanticUndefined

Instant = Annotated[int, Field(ge=0, description="UTC epoch seconds.")]
ServiceDate = Annotated[
    str,
    Field(
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="ISO date of a GTFS service day (never a UTC date).",
    ),
]
GtfsSeconds = Annotated[
    int,
    Field(
        ge=0,
        description=(
            "Seconds after noon-minus-12h of the service day; values >= 86400 "
            "are past midnight (24:04:00 = 86640)."
        ),
    ),
]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{16,64}$")]


class Model(BaseModel):
    """Base for every contract: immutable, closed, and default-free."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        frozen=True, extra="forbid", populate_by_name=True, serialize_by_alias=True
    )

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        defaulted = [
            name
            for name, info in cls.model_fields.items()
            if info.default is not PydanticUndefined or info.default_factory is not None
        ]
        if defaulted:
            msg = f"{cls.__name__}: contract fields may not have defaults: {defaulted}"
            raise TypeError(msg)


class Unknown(Model):
    """A measurement that was expected but could not be taken, and why."""

    unknown: Literal[True]
    reason: str


class Interval(Model):
    """A half-open span of instants, ``[start, end)``."""

    start: Instant
    end: Instant

    @model_validator(mode="after")
    def _ordered(self) -> Interval:
        if self.end < self.start:
            msg = f"interval end {self.end} precedes start {self.start}"
            raise ValueError(msg)
        return self


class Provenance(Model):
    """Which code and which exact inputs produced an artifact.

    ``built_at`` is a *logical* build time derived from the inputs (the capture
    instant of the newest input), not the wall clock, so rebuilding from the same
    inputs yields identical bytes. ``applied_retroactively`` names inputs that
    were captured after part of the period the artifact describes, i.e. today's
    reference data stamped onto an older timetable.
    """

    code_sha: str
    code_dirty: bool | None
    schema_: str = Field(alias="schema")
    inputs: dict[str, Digest]
    applied_retroactively: list[str]
    built_at: Instant
