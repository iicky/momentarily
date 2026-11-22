"""``archive/trip_updates`` and ``archive/vehicles`` -> ``LegacyFetchOutcome``.

Both archives keep, every 5 minutes, per-route aggregates and the list of
GTFS-RT feeds that decoded that tick (``fresh_feeds``). The aggregates stay
what they are; only the per-feed success becomes a record. The trace archive
keeps the same list and shares ``feed_outcomes``.

Object: ``<archive>/<YYYY-MM-DD>/<name>.json`` holding
``{observed_at, fresh_feeds, rows}``. ``name`` is the run's
``<scheduled_at>-<started_at_ms>``; objects written before per-run names are
named by ``<observed_at>``, the wall clock, and keep no scheduled minute.
"""

from __future__ import annotations

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
    run_key,
    run_scheduled_at,
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
    line = lineage(archive, source.key, None, replaceable=False)
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


_KEY = run_key(r"(?P<archive>archive/(?:trip_updates|vehicles))", ".json")


def adapt(obj: LegacyObject) -> list[Emitted]:
    m = match_key(_KEY, obj.key)
    archive: LegacyArchive = (
        "archive/trip_updates"
        if m["archive"] == "archive/trip_updates"
        else "archive/vehicles"
    )
    doc = as_dict(load_json(obj), obj.key)
    observed_at = get_int(doc, "observed_at", obj.key)
    scheduled_at = run_scheduled_at(m, obj.key, observed_at)
    check_day(obj.key, m["day"], observed_at)
    scheduled = (
        unknown(f"{archive} keeps the wall clock only")
        if scheduled_at is None
        else scheduled_at
    )
    return feed_outcomes(
        archive, obj.source(observed_at), scheduled, fresh_feeds(doc, obj.key)
    )
