"""Parquet column layout of the event records.

The rules are layer 2's (``momentarily.model.observations.columns``): the
layout is derived from the Pydantic contract and pinned.
``schema/model/s1/events/parquet/<record>.txt`` holds ``describe()`` of every
record, and a test fails when the derived layout drifts from it or from the
digest pinned for ``LAYOUT_VERSION``. Event files are kept as layer 2's are,
so any change bumps ``LAYOUT_VERSION``. ``to_row`` and ``from_row`` from that
module convert records to and from this layout.
"""

from __future__ import annotations

from pydantic import BaseModel

from momentarily.model.events.contracts import (
    CoverageInterval,
    Headway,
    OpenState,
    SegmentTraversal,
    StopEvent,
    SupplyObservation,
    TripMatch,
)
from momentarily.model.events.keys import ServiceDayRecord, UtcDayRecord

LAYOUT_VERSION = 2

RECORDS: dict[ServiceDayRecord | UtcDayRecord, type[BaseModel]] = {
    "trip_match": TripMatch,
    "stop_event": StopEvent,
    "segment_traversal": SegmentTraversal,
    "headway": Headway,
    "open_state": OpenState,
    "coverage": CoverageInterval,
    "supply": SupplyObservation,
}
