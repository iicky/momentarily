"""The on-disk format of one day in the derived traversal archive.

A day is one gzipped JSON object (`traversals.json.gz`): its provenance, the
field order, and the traversal rows. This module is the codec only. Deriving
the rows from the raw trace and storing days in R2 stay with the tooling
(`training.trace`, `training.traversal_archive`), which imports from here.

Why provenance travels with the rows, and why the decoder refuses rather than
guesses, is set out in `training.traversal_archive`.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from momentarily.provenance import code_provenance, input_manifest_hash

# Censoring kinds on a Traversal. Named rather than boolean because the three
# are statistically different and the survival machinery only handles two of
# them (see training.trace.to_dwell_samples).
EXACT = "exact"  # both ends observed, consecutive stop_seq
INTERVAL = "interval"  # an arrival was missed: n_hops covered in one span
RIGHT = "right"  # in progress when the trip was last seen

# Row layout. Bump on ANY change to the tuple below.
SCHEMA_VERSION = 1

# Semantics of training.trace.traversals_from_trace. Bump when what counts as an
# arrival, a hop, or a censoring kind changes, even if the row layout does not.
# Guarded by a golden test so the bump cannot be forgotten silently.
EXTRACTOR_VERSION = 1

# Fields, in order, of a serialized row. Kept as a constant so the writer, the
# reader and the schema test cannot disagree about the layout.
FIELDS: tuple[str, ...] = (
    "trip_id",
    "route_id",
    "direction",
    "from_stop",
    "to_stop",
    "at",
    "seconds",
    "moving_seconds",
    "n_hops",
    "censoring",
)


@dataclass(frozen=True)
class Traversal:
    """One trip's movement from one arrival to the next.

    `at` is the arrival at from_stop — when the span started. It is what lets a
    traversal be matched to a service day, and so to the timetable that was
    actually running.

    Two durations, because a 1-minute poll brackets rather than pins the hop:
    `seconds` is arrival to arrival, which includes the dwell at from_stop and
    is the quantity the baselines fit; `moving_seconds` runs from the first
    sighting that had left from_stop, which starts the clock late and is
    therefore a LOWER bound on travel. `moving_seconds` is None when the
    departure was never observed, which is three quarters of the time.

    `n_hops` is how many stops the span covered in the REALTIME feed's own stop
    sequence: 1 is a station-to-station traversal, more means an arrival went
    unobserved between polls and the individual hop times are only known to sum
    to `seconds`. It is not the timetable's opinion — a train that bypasses a
    station reports consecutive stops the schedule puts a station between, which
    is what momentarily.gtfs.static.Span.n_hops is for. `to_stop`/`n_hops` are
    None when the traversal was still in progress the last time the trip was
    seen.
    """

    trip_id: str
    route_id: str
    direction: str | None
    from_stop: str
    to_stop: str | None
    at: int
    seconds: int
    moving_seconds: int | None
    n_hops: int | None
    censoring: str


def _row(t: Traversal) -> list[Any]:
    return [getattr(t, f) for f in FIELDS]


def _traversal(row: Sequence[Any]) -> Traversal:
    return Traversal(**dict(zip(FIELDS, row, strict=True)))


@dataclass(frozen=True)
class FeedDigestRange:
    """One contiguous span of trace ticks that agreed on a single feed digest.

    Exists because a day is not always served by one static feed end to end: a
    republish can land mid-day, and the Worker stamps whatever digest was
    actually in force at each tick (see gtfs_feed.ts). When a day's ticks do
    not all agree, `write_day` records the ranges instead of picking a winner
    or discarding the information, so a later reader can still tell exactly
    which artifact priced which part of the day.
    """

    digest: str
    first_ts: int
    last_ts: int


# Named type for the hashable identity DayProvenance.comparable_key returns.
# Used by archive_read.py and window_archive.py so the return annotation is
# defined once and stays in sync when the provenance shape changes.
ComparableKey = tuple[
    int, int, str | None, str | None, tuple[tuple[str, int, int], ...] | None
]


@dataclass(frozen=True)
class DayProvenance:
    """Everything needed to decide whether two days may be compared."""

    schema: int
    extractor: int
    feed_version: str | None
    # The sha256 of the feed artifact that was ACTUALLY IN FORCE while this day
    # was being collected (see training.gtfs_archive for the store).
    #
    # None for every backfilled day, and that is not a gap to be filled in
    # later. A backfill can only fetch the CURRENT feed, and `covers()` spans
    # months, so stamping today's digest on a three-week-old day would assert
    # bytes that may have been published after the trains ran — provenance that
    # is WRONG rather than missing, which is the worse failure because
    # `homogeneous` would then call the span replayable.
    #
    # The Worker now reads archive/gtfs/latest.json (the pointer
    # gtfs_archive.store() writes) on every trace tick and stamps the digest on
    # the trace body. write_day extracts the consensus digest from those bodies.
    # The digest names bytes that exist in archive/gtfs/<digest>.zip, so a day
    # carrying it is genuinely replayable. Days collected before that change, or
    # before the first archive run, carry None.
    #
    # Set only when every trace body that carried a digest agreed on ONE value
    # (see `feed_digests` below for the mid-day-republish case).
    feed_digest: str | None
    # Set instead of `feed_digest` when a day's trace bodies did not all agree
    # on one digest — a republish landed mid-day and the Worker's HEAD/ETag
    # check caught it partway through. None in every other case, including the
    # unanimous one, so old readers that only know `feed_digest` keep working:
    # a day either has ONE digest for its whole span, or its span here.
    feed_digests: list[FeedDigestRange] | None = field(default=None, kw_only=True)
    n_rows: int
    n_source_objects: int
    source_manifest: str
    code_sha: str | None
    written_at: int

    @property
    def comparable_key(self) -> ComparableKey:
        """The identity two days must share to be pooled without comment.

        Carries the feed DIGEST as well as its version: two snapshots can share
        a version label, and only the bytes decide what a scheduled time was.
        A mixed day's ranges are folded in too — two days that both mix the
        same two digests over different spans of the day are not the same
        claim and must not pool silently either.
        """
        ranges = (
            tuple((r.digest, r.first_ts, r.last_ts) for r in self.feed_digests)
            if self.feed_digests is not None
            else None
        )
        return (
            self.schema,
            self.extractor,
            self.feed_version,
            self.feed_digest,
            ranges,
        )


def encode_day(
    traversals: Sequence[Traversal],
    *,
    feed_version: str | None,
    feed_digest: str | None = None,
    feed_digests: Sequence[FeedDigestRange] | None = None,
    source_keys: Sequence[str],
) -> bytes:
    """One day's traversals plus its provenance, gzipped.

    Rows are positional tuples rather than objects: measured over 2026-08-15 the
    dict form is 26.4 MB against 11.0 MB, and both compress to about 1.5 MB, so
    the tuple form is chosen for the uncompressed footprint a reader pays to
    parse rather than for the stored bytes.
    """
    prov: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "extractor": EXTRACTOR_VERSION,
        "feed_version": feed_version,
        "feed_digest": feed_digest,
        "n_rows": len(traversals),
        "n_source_objects": len(source_keys),
        "source_manifest": input_manifest_hash(list(source_keys)),
        "code_sha": code_provenance()["code_sha"],
        "written_at": int(datetime.now(UTC).timestamp()),
    }
    if feed_digests is not None:
        prov["feed_digests"] = [
            {"digest": r.digest, "first_ts": r.first_ts, "last_ts": r.last_ts}
            for r in feed_digests
        ]
    doc = {
        "provenance": prov,
        "fields": list(FIELDS),
        "rows": [_row(t) for t in traversals],
    }
    return gzip.compress(json.dumps(doc, separators=(",", ":")).encode(), 9)


def decode_day(blob: bytes) -> tuple[list[Traversal], DayProvenance]:
    """Inverse of `encode_day`, refusing anything it cannot read faithfully.

    An unknown schema raises rather than best-efforts a partial parse: a
    silently mis-parsed historical day is worse than a loud failure, because the
    raw input it came from no longer exists to check against.
    """
    doc = cast(dict[str, Any], json.loads(gzip.decompress(blob)))
    prov_raw = cast(dict[str, Any], doc.get("provenance") or {})
    schema = int(prov_raw.get("schema") or 0)
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"traversal archive schema {schema} is not readable by this build "
            f"(expects {SCHEMA_VERSION})"
        )
    missing = {"extractor", "n_rows", "n_source_objects", "source_manifest"} - set(
        prov_raw
    )
    if missing:
        # Silently defaulting a missing count to 0 is how a format rename turns
        # historical rows into plausible nonsense. Absent provenance is a hard
        # failure, because the inputs that could have rebuilt it are gone.
        raise ValueError(f"provenance is missing {sorted(missing)}")
    fields = tuple(cast(list[str], doc.get("fields") or ()))
    if fields != FIELDS:
        raise ValueError(f"traversal archive field order {fields} != {FIELDS}")
    feed_digests_raw = prov_raw.get("feed_digests")
    feed_digests = (
        [
            FeedDigestRange(
                digest=str(r["digest"]),
                first_ts=int(r["first_ts"]),
                last_ts=int(r["last_ts"]),
            )
            for r in cast(list[Any], feed_digests_raw)
        ]
        if feed_digests_raw is not None
        else None
    )
    prov = DayProvenance(
        schema=schema,
        extractor=int(prov_raw.get("extractor") or 0),
        feed_version=cast(str | None, prov_raw.get("feed_version")),
        feed_digest=cast(str | None, prov_raw.get("feed_digest")),
        feed_digests=feed_digests,
        n_rows=int(prov_raw.get("n_rows") or 0),
        n_source_objects=int(prov_raw.get("n_source_objects") or 0),
        source_manifest=str(prov_raw.get("source_manifest") or ""),
        code_sha=cast(str | None, prov_raw.get("code_sha")),
        written_at=int(prov_raw.get("written_at") or 0),
    )
    rows = [_traversal(cast(Sequence[Any], r)) for r in cast(list[Any], doc["rows"])]
    return rows, prov
