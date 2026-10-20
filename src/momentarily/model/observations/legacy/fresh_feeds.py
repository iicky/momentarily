"""``archive/trip_updates`` and ``archive/vehicles`` -> ``LegacyFetchOutcome``.

Both archives keep, every 5 minutes, per-route aggregates and the list of
GTFS-RT feeds that decoded that tick (``fresh_feeds``). The aggregates stay
what they are; only the per-feed success becomes a record. The trace archive
keeps the same list and shares ``feed_outcomes``.

Object: ``<archive>/<YYYY-MM-DD>/<observed_at>.json`` holding
``{observed_at, fresh_feeds, rows}``.
"""

from __future__ import annotations

import re
from typing import Any, cast

from momentarily.model.conventions import Unknown
from momentarily.model.observations.contracts import (
    GTFSRT_SOURCES,
    GtfsRtSource,
    LegacyArchive,
    LegacyFetchOutcome,
)
from momentarily.model.observations.legacy.common import (
    Emitted,
    LegacyObject,
    MalformedObjectError,
    SourceObject,
    as_dict,
    as_list,
    check_day,
    get_int,
    lineage,
    load_json,
    match_key,
    unknown,
)

_FEEDS = cast(tuple[GtfsRtSource, ...], GTFSRT_SOURCES)


def fresh_feeds(doc: dict[str, Any], where: str) -> frozenset[str]:
    names = as_list(doc.get("fresh_feeds"), f"{where}.fresh_feeds")
    unknown_feeds = [n for n in names if n not in GTFSRT_SOURCES]
    if unknown_feeds:
        raise MalformedObjectError(f"{where} lists unknown feeds {unknown_feeds!r}")
    return frozenset(names)


def feed_outcomes(
    archive: LegacyArchive,
    source: SourceObject,
    scheduled_at: int | Unknown,
    fresh: frozenset[str],
) -> list[Emitted]:
    """One outcome per GTFS-RT feed: fresh or not. The archive kept no
    cause and no fetch time."""
    line = lineage(archive, source.key, None)
    last_success = unknown(f"{archive} keeps no fetch time")
    return [
        Emitted(
            LegacyFetchOutcome(
                legacy_fetch_id=f"{source.key}#{feed}",
                lineage=line,
                source=feed,
                observed_at=source.observed_at,
                scheduled_at=scheduled_at,
                succeeded=feed in fresh,
                last_success_at=last_success,
            ),
            (source,),
        )
        for feed in _FEEDS
    ]


_KEY = re.compile(
    r"(?P<archive>archive/(?:trip_updates|vehicles))/(?P<day>\d{4}-\d{2}-\d{2})/(?P<at>\d+)\.json"
)


def adapt(obj: LegacyObject) -> list[Emitted]:
    m = match_key(_KEY, obj.key)
    archive: LegacyArchive = (
        "archive/trip_updates"
        if m["archive"] == "archive/trip_updates"
        else "archive/vehicles"
    )
    doc = as_dict(load_json(obj), obj.key)
    observed_at = get_int(doc, "observed_at", obj.key)
    if int(m["at"]) != observed_at:
        raise MalformedObjectError(f"{obj.key} names another observed_at")
    check_day(obj.key, m["day"], observed_at)
    scheduled = unknown(f"{archive} keeps the wall clock only")
    return feed_outcomes(
        archive, obj.source(observed_at), scheduled, fresh_feeds(doc, obj.key)
    )
