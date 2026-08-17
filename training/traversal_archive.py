"""A long-lived archive of DERIVED traversals, kept when the raw trace is not.

WHY THIS EXISTS. `archive/trace/` is pruned at 30 days because of its size (81
MB/day measured 2026-08-15), and that prune is the reason every model in this
repo has been evaluated on at most a month of history. Waiting three weeks,
grading a month, and finding the older data already deleted is not a data
problem, it is a retention policy — the evaluation window could never grow.

The raw trace is an INTERMEDIATE. Every measure built on it — traversal
baselines, the planned-work grade, the progress ratio, the segment grade —
starts by calling `trace.traversals_from_trace` and never looks at a raw row
again. Storing the derived traversals instead costs 1.5 MB/day gzipped against
81 MB/day raw, measured over the same day: 56x smaller, so the storage
currently spent on a rolling 30-day raw window holds about four and a half years
of this. That converts "wait for new collection" into "query what we already
have".

WHAT THIS DOES NOT DO. It cannot invent history. The 5-minute
`archive/vehicles/` stream that reaches further back records per-(route,
direction) COUNTS with no trip identity, so traversal times are not recoverable
from it at any effort. This archive begins when the trace began.

PROVENANCE IS PART OF THE RECORD, NOT A NICETY. A derived store outlives its
inputs: once the raw trace for a day is pruned, a row in here cannot be
regenerated or checked. So any later change to how traversals are extracted, or
to which static feed named the service, would silently make old rows
incomparable with new ones — and the comparison would still run and still
produce a number. Every day written therefore carries:

  * `schema` — the row layout. Bump on any field change.
  * `extractor` — the SEMANTICS of trace.traversals_from_trace: what counts as
    an arrival, a hop, a censoring kind. Bump whenever that changes, even when
    the row layout does not. `tests/test_traversal_archive.py` pins the
    extractor's output against a golden fixture so a semantic change fails a
    test rather than quietly writing incomparable rows.
  * `feed_version` — the static GTFS snapshot in force, which decides scheduled
    times and bypasses.
  * `source_manifest` — a digest of the exact trace object keys that produced
    the day, and `code_sha` for the tree that ran it.

A reader that spans a version boundary is told so (`ReadResult.versions`) rather
than left to average across it.

HOLDOUTS. Days are addressable, immutable, and independent, so a holdout is a
date filter and needs nothing added here. Deliberately not implemented until
there is enough history to spend on one.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from training.gtfs_static import FeedVersion
from training.load_r2 import date_range, input_manifest_hash, list_keys
from training.provenance import code_provenance
from training.r2_client import R2Config, get_object_bytes, load_config, make_client
from training.trace import Traversal, traversals_from_trace

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

PREFIX = "archive/traversals/"

# Row layout. Bump on ANY change to the tuple below.
SCHEMA_VERSION = 1

# Semantics of trace.traversals_from_trace. Bump when what counts as an arrival,
# a hop, or a censoring kind changes, even if the row layout does not. Guarded by
# a golden test so the bump cannot be forgotten silently.
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


def _row(t: Traversal) -> list[Any]:
    return [getattr(t, f) for f in FIELDS]


def _traversal(row: Sequence[Any]) -> Traversal:
    return Traversal(**dict(zip(FIELDS, row, strict=True)))


@dataclass(frozen=True)
class DayProvenance:
    """Everything needed to decide whether two days may be compared."""

    schema: int
    extractor: int
    feed_version: str | None
    n_rows: int
    n_source_objects: int
    source_manifest: str
    code_sha: str | None
    written_at: int

    @property
    def comparable_key(self) -> tuple[int, int, str | None]:
        """The identity two days must share to be pooled without comment."""
        return (self.schema, self.extractor, self.feed_version)


def encode_day(
    traversals: Sequence[Traversal],
    *,
    feed_version: str | None,
    source_keys: Sequence[str],
) -> bytes:
    """One day's traversals plus its provenance, gzipped.

    Rows are positional tuples rather than objects: measured over 2026-08-15 the
    dict form is 26.4 MB against 11.0 MB, and both compress to about 1.5 MB, so
    the tuple form is chosen for the uncompressed footprint a reader pays to
    parse rather than for the stored bytes.
    """
    doc = {
        "provenance": {
            "schema": SCHEMA_VERSION,
            "extractor": EXTRACTOR_VERSION,
            "feed_version": feed_version,
            "n_rows": len(traversals),
            "n_source_objects": len(source_keys),
            "source_manifest": input_manifest_hash(list(source_keys)),
            "code_sha": code_provenance()["code_sha"],
            "written_at": int(datetime.now(UTC).timestamp()),
        },
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
    prov = DayProvenance(
        schema=schema,
        extractor=int(prov_raw.get("extractor") or 0),
        feed_version=cast(str | None, prov_raw.get("feed_version")),
        n_rows=int(prov_raw.get("n_rows") or 0),
        n_source_objects=int(prov_raw.get("n_source_objects") or 0),
        source_manifest=str(prov_raw.get("source_manifest") or ""),
        code_sha=cast(str | None, prov_raw.get("code_sha")),
        written_at=int(prov_raw.get("written_at") or 0),
    )
    rows = [_traversal(cast(Sequence[Any], r)) for r in cast(list[Any], doc["rows"])]
    return rows, prov


def key_for(day: date) -> str:
    """One immutable object per day, under a DATE PATH SEGMENT.

    The segment form (not `<date>.json.gz`) matches every other stream in the
    bucket and is what `training.prune`'s date matcher recognises, so this
    prefix is policed by the same retention machinery as the rest instead of
    needing its own.
    """
    return f"{PREFIX}{day.isoformat()}/traversals.json.gz"


@dataclass(frozen=True)
class ReadResult:
    """Traversals over a span, and whether they may be pooled.

    `versions` holding more than one entry means the span crosses a change in
    extraction or in the static feed. That is not automatically fatal — a feed
    version changes every few weeks — but it must be visible at the point of
    use, because the alternative is averaging across a definitional change with
    no way left to detect it.
    """

    traversals: list[Traversal]
    provenance: dict[date, DayProvenance]

    @property
    def versions(self) -> set[tuple[int, int, str | None]]:
        return {p.comparable_key for p in self.provenance.values()}

    @property
    def homogeneous(self) -> bool:
        return len(self.versions) <= 1


def read_days(
    start: date,
    end: date,
    *,
    config: R2Config | None = None,
    client: S3Client | None = None,
) -> ReadResult:
    """Load the derived archive over [start, end]. Missing days are skipped."""
    cfg = config or load_config()
    client = client or make_client(cfg)
    present = set(list_keys(client, cfg.bucket, PREFIX))
    out: list[Traversal] = []
    prov: dict[date, DayProvenance] = {}
    for day in date_range(start, end):
        key = key_for(day)
        if key not in present:
            continue
        rows, p = decode_day(get_object_bytes(client, cfg.bucket, key))
        out.extend(rows)
        prov[day] = p
    return ReadResult(traversals=out, provenance=prov)


def is_closed(day: date, *, now: date | None = None) -> bool:
    """Whether a day can no longer gain trace objects.

    Trace keys are partitioned on the UTC date (`worker/src/archive.ts` uses
    utcDate), so closure is a UTC question and not a local one.
    """
    return day < (now or datetime.now(UTC).date())


def resolve_feed_version(feed: FeedVersion | None, day: date) -> str | None:
    """The feed version to stamp on `day`, or None when it cannot be known.

    A backfill fetches ONE static feed — today's — but spans up to 28 days, and
    MTA republishes every few weeks. Stamping today's version on a day it does
    not describe records provenance that is WRONG rather than missing, which is
    strictly worse: `ReadResult.homogeneous` would then report a span as poolable
    across a real feed boundary, defeating the check this archive exists for.

    The feed declares the window it applies to, so the honest answer is available
    without archiving historical feeds: stamp it only when it claims to cover the
    day, and record None otherwise. None is its own comparability bucket, so a
    span mixing known and unknown days still reports itself as mixed.
    """
    if feed is None or not feed.covers(day):
        return None
    return feed.version


def write_day(
    day: date,
    *,
    feed: FeedVersion | None,
    config: R2Config | None = None,
    client: S3Client | None = None,
    overwrite: bool = False,
    allow_partial: bool = False,
    now: date | None = None,
    present: set[str] | None = None,
) -> DayProvenance | None:
    """Derive one day from the raw trace and store it.

    Returns None when nothing was written: the day is still open, or is already
    present and `overwrite` is not set, or the trace holds nothing for it.

    ONLY CLOSED DAYS ARE FINALIZED. Writing the day still in progress would
    store a truncated one — at 10:00 UTC roughly 600 of its 1,440 minutes exist
    — and because a present day is skipped by default, every later run would
    skip it forever. The archive would keep a permanently short day, and once
    the raw trace is pruned at 30 days there would be nothing left to repair it
    from. That failure is silent, so it is refused here rather than documented.
    `allow_partial` exists for deliberate inspection and pairs with `overwrite`,
    because a partial day is only safe if it will be replaced.
    """
    cfg = config or load_config()
    client = client or make_client(cfg)
    if not allow_partial and not is_closed(day, now=now):
        return None
    key = key_for(day)
    if present is None:
        present = set(list_keys(client, cfg.bucket, PREFIX))
    if not overwrite and key in present:
        return None

    feed_version = resolve_feed_version(feed, day)
    trace_keys = list_keys(client, cfg.bucket, f"archive/trace/{day.isoformat()}/")
    if not trace_keys:
        return None
    bodies = [
        cast(dict[str, Any], json.loads(get_object_bytes(client, cfg.bucket, k)))
        for k in trace_keys
    ]
    traversals, _stats = traversals_from_trace(bodies)
    blob = encode_day(traversals, feed_version=feed_version, source_keys=trace_keys)
    client.put_object(
        Bucket=cfg.bucket,
        Key=key,
        Body=blob,
        ContentType="application/json",
        ContentEncoding="gzip",
    )
    _rows, prov = decode_day(blob)
    return prov


def backfill(
    days: Iterable[date],
    *,
    feed: FeedVersion | None,
    overwrite: bool = False,
    allow_partial: bool = False,
) -> list[tuple[date, DayProvenance | None]]:
    cfg = load_config()
    client = make_client(cfg)
    # Listed once, not per day: a trailing self-healing window re-checks many
    # days each run and only the missing ones cost a fetch.
    present = set(list_keys(client, cfg.bucket, PREFIX))
    out: list[tuple[date, DayProvenance | None]] = []
    for day in days:
        prov = write_day(
            day,
            feed=feed,
            config=cfg,
            client=client,
            overwrite=overwrite,
            allow_partial=allow_partial,
            present=present,
        )
        out.append((day, prov))
        if prov is not None:
            reason = f"{prov.n_rows} traversals"
        elif not allow_partial and not is_closed(day):
            reason = "skipped: day still open"
        else:
            reason = "skipped: already present or no trace"
        print(f"{day}: {reason}", file=sys.stderr)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write the long-lived derived traversal archive from raw trace"
    )
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rewrite days already present (needed after an extractor bump)",
    )
    parser.add_argument(
        "--no-feed-version",
        action="store_true",
        help="skip the static feed fetch; records carry a null feed_version",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="also write the day still in progress; pair with --overwrite, since "
        "a partial day is only safe if a later run replaces it",
    )
    args = parser.parse_args(argv)

    # Structural, not documented: a partial day written once would be skipped by
    # every later run and left permanently truncated, which is the exact loss
    # this module exists to prevent. --allow-partial is only safe when a later
    # run will replace the object, so it REQUIRES --overwrite. Checked before any
    # fetch so a bad invocation costs nothing.
    if args.allow_partial and not args.overwrite:
        parser.error("--allow-partial requires --overwrite")

    # Default to yesterday, not today: today is still gaining trace objects and
    # writing it would finalize a truncated day. See write_day.
    today = datetime.now(UTC).date()
    end = args.end_date or (today - timedelta(days=1))
    start = args.start_date or end

    feed: FeedVersion | None = None
    if not args.no_feed_version:
        import io
        import zipfile

        from training.gtfs_static import fetch_gtfs_zip
        from training.gtfs_static import timetable as parse_timetable

        with zipfile.ZipFile(io.BytesIO(fetch_gtfs_zip())) as zf:
            feed = parse_timetable(zf).version
        print(
            f"static feed {feed.version} valid {feed.start}..{feed.end}",
            file=sys.stderr,
        )

    written = backfill(
        date_range(start, end),
        feed=feed,
        overwrite=args.overwrite,
        allow_partial=args.allow_partial,
    )
    total = sum(p.n_rows for _d, p in written if p is not None)
    print(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "extractor": EXTRACTOR_VERSION,
                "feed_version": None if feed is None else feed.version,
                "feed_valid": (
                    None if feed is None else [str(feed.start), str(feed.end)]
                ),
                "days_written": sum(1 for _d, p in written if p is not None),
                "days_skipped": sum(1 for _d, p in written if p is None),
                "traversals": total,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
