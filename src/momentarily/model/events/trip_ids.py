"""NYCT trip ids, stripped to the fields realtime and static ids share.

A static id is ``<block>_<origin>_<route field>.<direction><path>`` (one or
two dots), for example ``ASP26GEN-1038-Weekday-00_063400_A..S55R``. A realtime
id has no block and often a shorter path: ``063400_A..S``. ``origin`` is the
scheduled origin in hundredths of a minute after the service day's start
(``063400`` is 634.00 minutes, 10:34). Static ids count past midnight
(``144100`` is 24:01); realtime ids wrap it (``000100``), so a realtime origin
``o`` may also name the static origin ``o + WRAP`` of the previous service day.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# 24 hours in the origin's unit (hundredths of a minute).
WRAP = 144000

_STATIC = re.compile(
    r"^(?:(?P<block>.*?)_)?(?P<origin>\d{6})_(?P<field>[A-Z0-9]+)\.\.?(?P<dir>[NS])(?P<path>.*)$"
)
_REALTIME = re.compile(
    r"^(?P<origin>\d{6})_(?P<field>[A-Z0-9]+)\.\.?(?P<dir>[NS])(?P<path>.*)$"
)

Direction = Literal["N", "S"]


@dataclass(frozen=True, slots=True)
class TripIdKey:
    """The fields a realtime id shares with the static id it names."""

    origin: int
    route_field: str
    direction: Direction
    path: str

    @property
    def origin_seconds(self) -> int:
        return self.origin * 60 // 100


@dataclass(frozen=True, slots=True)
class StaticTripId(TripIdKey):
    block: str


def _direction(value: str) -> Direction:
    return "N" if value == "N" else "S"


def parse_static(trip_id: str) -> StaticTripId | None:
    m = _STATIC.match(trip_id)
    if m is None:
        return None
    return StaticTripId(
        origin=int(m["origin"]),
        route_field=m["field"],
        direction=_direction(m["dir"]),
        path=m["path"],
        block=m["block"] or "",
    )


def parse_realtime(trip_id: str) -> TripIdKey | None:
    m = _REALTIME.match(trip_id)
    if m is None:
        return None
    return TripIdKey(
        origin=int(m["origin"]),
        route_field=m["field"],
        direction=_direction(m["dir"]),
        path=m["path"],
    )
