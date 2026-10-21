"""Storage key layout of the events layer.

Every key sits under ``model/events/s1/``. So far there is one kind:

    match_index/v<index_version>/<service_date>/<build_id>-<route_feed16>.json.gz

The index of a service day depends on the index code version, the layer-1
build it was made from, and the ``RouteFeed`` table (first 16 hex of its
blake3), so all three are in the key: a day whose build choice changes (an
unverified day verified later), a new table or a new ``INDEX_VERSION`` gets a
new key rather than different bytes under the old one.
``parse_key(k.key) == k`` for every valid key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from momentarily.model.observations.keys import KeyLayoutError

PREFIX = "model/events/s1/"

_HEX16 = re.compile(r"[0-9a-f]{16}")
_MATCH_INDEX = re.compile(
    rf"^{re.escape(PREFIX)}match_index/v(?P<version>[1-9][0-9]*)/"
    r"(?P<day>\d{4}-\d{2}-\d{2})/(?P<build>[0-9a-f]{16})-(?P<rf>[0-9a-f]{16})\.json\.gz$"
)


@dataclass(frozen=True)
class MatchIndexKey:
    index_version: int
    service_date: str
    build_id: str
    route_feed: str

    def __post_init__(self) -> None:
        if self.index_version < 1:
            raise KeyLayoutError(f"index version must be >= 1: {self.index_version}")
        try:
            ok = date.fromisoformat(self.service_date).isoformat() == self.service_date
        except ValueError:
            ok = False
        if not ok:
            raise KeyLayoutError(f"not an ISO service date: {self.service_date!r}")
        if not _HEX16.fullmatch(self.build_id):
            raise KeyLayoutError(f"not a 16-hex build id: {self.build_id!r}")
        if not _HEX16.fullmatch(self.route_feed):
            raise KeyLayoutError(f"not a 16-hex RouteFeed digest: {self.route_feed!r}")

    @property
    def key(self) -> str:
        return (
            f"{PREFIX}match_index/v{self.index_version}/{self.service_date}/"
            f"{self.build_id}-{self.route_feed}.json.gz"
        )


def parse_key(key: str) -> MatchIndexKey:
    m = _MATCH_INDEX.match(key)
    if m is None:
        raise KeyLayoutError(f"not an events/s1 key: {key!r}")
    return MatchIndexKey(
        index_version=int(m["version"]),
        service_date=m["day"],
        build_id=m["build"],
        route_feed=m["rf"],
    )
