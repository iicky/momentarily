"""A long-lived archive of DERIVED traversals, cheap to read where raw is not.

WHY THIS EXISTS. `archive/trace/` used to be pruned at 30 days because of its
size (81 MB/day measured 2026-08-15), and that prune is the reason every model
in this repo was evaluated on at most a month of history. The raw trace is now
kept in full by owner decision, but it is still too large to re-read for every
grade, so the evaluation window grows through this archive.

The raw trace is an INTERMEDIATE. Every measure built on it — traversal
baselines, the planned-work grade, the progress ratio, the segment grade —
starts by calling `trace.traversals_from_trace` and never looks at a raw row
again. Storing the derived traversals instead costs 1.5 MB/day gzipped against
81 MB/day raw, measured over the same day: 56x smaller to read. That converts
"re-derive the whole history" into "query what we already have".

WHAT THIS DOES NOT DO. It cannot invent history. The 5-minute
`archive/vehicles/` stream that reaches further back records per-(route,
direction) COUNTS with no trip identity, so traversal times are not recoverable
from it at any effort. This archive begins when the trace began.

PROVENANCE IS PART OF THE RECORD, NOT A NICETY. A derived store is read in
place of its inputs, so a row in here is not re-checked against the raw trace.
So any later change to how traversals are extracted, or
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
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from momentarily.archive.traversals import (
    EXTRACTOR_VERSION,
    SCHEMA_VERSION,
    DayProvenance,
    FeedDigestRange,
    Traversal,
    decode_day,
    encode_day,
)
from momentarily.gtfs.static import FeedVersion
from training.load_r2 import date_range, list_keys
from training.r2_client import R2Config, get_object_bytes, load_config, make_client
from training.trace import traversals_from_trace

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

PREFIX = "archive/traversals/"


def key_for(day: date) -> str:
    """One immutable object per day, under a DATE PATH SEGMENT.

    The segment form (not `<date>.json.gz`) matches every other stream in the
    bucket, so date-scoped listing works the same way here as elsewhere.
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
    def versions(
        self,
    ) -> set[
        tuple[int, int, str | None, str | None, tuple[tuple[str, int, int], ...] | None]
    ]:
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


def _resolve_etag(etag: str, client: S3Client, bucket: str) -> str | None:
    """Look up a feed_etag in archive/gtfs/by_etag/ to get the digest.

    Returns None if the etag object doesn't exist (capture failed or hasn't
    run yet), never raises on a missing key.
    """
    from training.gtfs_archive import etag_key

    key = etag_key(etag)
    try:
        blob = get_object_bytes(client, bucket, key)
        doc = json.loads(blob)
        d = doc.get("digest")
        return d if isinstance(d, str) else None
    except Exception:
        return None


def _consensus_digest(
    bodies: Sequence[dict[str, Any]],
    client: S3Client,
    bucket: str,
) -> tuple[str | None, list[FeedDigestRange] | None]:
    """Derive the day's feed-digest provenance from its trace bodies.

    A body carries ``feed_digest: <sha256>`` when the Worker's HEAD matched the
    pointer, ``feed_digest: null`` + ``feed_etag: <etag>`` on a mismatch (the
    capture ran detached; by_etag/ maps the etag to the digest after the fact),
    and omits both fields entirely for pre-feature archive objects.

    Resolution order per body:
    1. ``feed_digest`` if it's a non-null string — the happy path.
    2. ``feed_etag`` resolved through ``archive/gtfs/by_etag/<etag>.json`` —
       the mismatch/transition tick whose capture has since completed.
    3. Neither resolves — body contributes no digest (HEAD failure, pre-feature,
       or capture that never finished).

    Rules after resolution:
    * EVERY body resolves to the SAME digest -> ``(that_digest, None)``
      (the legacy unanimous path; backward-compatible).
    * Some bodies can't resolve, or bodies disagree -> ``(None, runs)`` where
      each run is a maximal CONTIGUOUS stretch of resolved ticks, in
      ``scheduled_at`` order, carrying one digest. A day that goes A -> B -> A
      yields three runs, never one A range spanning B's ticks, and an
      unresolved tick breaks a run the same way a digest change does.
    * No body resolves -> ``(None, None)``.
    """
    if not bodies:
        return None, None

    # Every body in tick order; an unresolved tick keeps its slot as None so a
    # run can never be stretched across a stretch whose feed is unknown.
    ticks: list[tuple[int, str | None]] = []
    for body in bodies:
        digest = body.get("feed_digest")
        if not isinstance(digest, str):
            # Try resolving via the etag mapping (mismatch tick).
            etag = body.get("feed_etag")
            if isinstance(etag, str):
                digest = _resolve_etag(etag, client, bucket)
        ts = body.get("scheduled_at")
        ticks.append(
            (
                int(ts) if isinstance(ts, int | float) else 0,
                digest if isinstance(digest, str) else None,
            )
        )

    digests = {d for _, d in ticks if d is not None}
    if not digests:
        return None, None

    # Unanimous: every body (not just those that resolved) must carry the same
    # value.  A day where some ticks couldn't resolve is NOT unanimous — the
    # legacy field must not claim whole-day coverage.
    if len(digests) == 1 and all(d is not None for _, d in ticks):
        (only_digest,) = digests
        return only_digest, None

    # Contiguous runs in tick order: a run closes when the digest changes or a
    # tick is unresolved, so A -> B -> A yields three runs and an A range never
    # claims the ticks under B or under a failed HEAD.
    ticks.sort(key=lambda t: t[0])
    runs: list[FeedDigestRange] = []
    open_run = False
    for ts, d in ticks:
        if d is None:
            open_run = False
            continue
        if open_run and runs[-1].digest == d:
            runs[-1] = FeedDigestRange(digest=d, first_ts=runs[-1].first_ts, last_ts=ts)
        else:
            runs.append(FeedDigestRange(digest=d, first_ts=ts, last_ts=ts))
            open_run = True
    return None, runs


def write_day(
    day: date,
    *,
    feed: FeedVersion | None,
    feed_digest: str | None = None,
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
    skip it forever. The archive would keep a permanently short day that no
    later run repairs, and readers prefer the archive over the raw trace. That
    failure is silent, so it is refused here rather than documented.
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
    # Prefer the caller's explicit digest (a proven, in-hand value); it wins
    # outright and no per-tick ranges are derived from the bodies. When none
    # was passed, derive it from the collection-time `feed_digest` each trace
    # body recorded: unanimous agreement stamps one digest for the whole day,
    # a republish that landed mid-day stamps per-tick ranges instead (see
    # `_consensus_digest`).
    feed_digests: list[FeedDigestRange] | None = None
    if feed_digest is None and bodies:
        feed_digest, feed_digests = _consensus_digest(bodies, client, cfg.bucket)
    blob = encode_day(
        traversals,
        feed_version=feed_version,
        feed_digest=feed_digest,
        feed_digests=feed_digests,
        source_keys=trace_keys,
    )
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
    feed_digest: str | None = None,
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
            feed_digest=feed_digest,
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

        from momentarily.gtfs.static import fetch_gtfs_zip
        from momentarily.gtfs.static import timetable as parse_timetable

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
