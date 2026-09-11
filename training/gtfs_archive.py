"""The static GTFS feed, kept content-addressed so a grade stays replayable.

WHY A VERSION STRING IS NOT ENOUGH. The derived archives keep the measurements
(`traversal_archive`) and the answer key (`window_archive`), and both record
which feed version was in force. That label cannot rebuild anything. Every
baseline in this repo is built through a `Timetable`:
`traversal.traversal_baseline` -> `hop_samples` needs `timetable.covers()` to
reject observations outside the feed's validity and `scheduled_for()` to price a
hop and catch bypasses. So replaying a historical grade needs the FEED ARTIFACT,
not its name.

And the artifact is not retrievable later. `GTFS_STATIC_URL` serves whatever is
current; MTA publishes no archive of superseded snapshots. Once it republishes,
the feed that priced last month's hops is gone from the internet. A grade of
retained traversals would then either fail or silently use today's timetable to
judge movement that ran under a different one — which is exactly how the modal
chain misprices a bypassing train, applied to a whole month at once.

CONTENT-ADDRESSED, NOT DATE-KEYED. The key is the sha256 of the zip, so a
republish stores a new object and an unchanged fetch stores nothing. Measured
2026-08-17 the zip is 5.6 MB; at roughly 26 republishes a year that is about
0.15 GB annually before dedup, which is cheaper than the traversals it makes
replayable.

The digest, not the version string, is what a stored day should point at: two
snapshots can share a `feed_version` label, and only the digest identifies the
bytes that produced a number.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx

from training.gtfs_static import (
    FETCH_TIMEOUT,
    GTFS_STATIC_URL,
    Timetable,
)
from training.gtfs_static import timetable as parse_timetable
from training.load_r2 import list_keys
from training.r2_client import R2Config, get_object_bytes, load_config, make_client

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

PREFIX = "archive/gtfs/"
# A tiny pointer to the most recently archived feed: the Worker reads this at
# trace time so each trace object names the feed artifact that was in force when
# it was collected.  Written by store() on every archive run, so it tracks the
# feed MTA is currently serving.  The digest it names is guaranteed to exist
# under PREFIX as a content-addressed .zip — the pointer is written AFTER the
# artifact, never before.
LATEST_KEY = f"{PREFIX}latest.json"
# One tiny object per HTTP ETag the feed has ever been seen under, so the
# Worker's intra-day HEAD-vs-ETag comparison can resolve "have we already
# captured this exact republish?" with a single R2 read keyed on the ETag
# alone, without downloading or re-hashing the body. Written by store()
# AFTER the zip lands but BEFORE latest.json, mirroring the zip-before-pointer
# rule below: neither pointer may name something not yet in place.
BY_ETAG_PREFIX = f"{PREFIX}by_etag/"


def etag_key(etag: str) -> str:
    """archive/gtfs/by_etag/<etag>.json, with the quoting an HTTP ETag header
    carries (and the weak-validator `W/` prefix, if present) stripped so the
    R2 key is a clean identifier rather than embedding a literal quote."""
    stripped = etag.strip()
    if stripped[:2] in ("W/", "w/"):
        stripped = stripped[2:]
    stripped = stripped.strip('"')
    return f"{BY_ETAG_PREFIX}{stripped}.json"


def digest_of(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def key_for(digest: str) -> str:
    return f"{PREFIX}{digest}.zip"


@dataclass(frozen=True)
class FeedSnapshot:
    """One stored feed artifact, identified by its bytes."""

    digest: str
    version: str
    key: str
    n_bytes: int
    etag: str | None  # the HTTP ETag this snapshot was fetched under, if any
    stored: bool  # False when this digest was already present


def store(
    blob: bytes,
    *,
    etag: str | None = None,
    last_modified: str | None = None,
    config: R2Config | None = None,
    client: S3Client | None = None,
) -> FeedSnapshot:
    """Store a feed zip under its own digest, skipping bytes already held.

    Idempotent by construction: the key IS the content hash, so a nightly job
    that fetches an unchanged feed writes nothing and cannot create a second
    copy under a different name.

    Also writes (or refreshes) a ``latest.json`` pointer so the Worker can
    learn the current feed digest with a single tiny R2 read, and — when the
    caller has an HTTP ``etag`` in hand — a ``by_etag/<etag>.json`` mapping so
    the Worker's intra-day comparison can resolve a repeated ETag without
    re-downloading or re-hashing the feed.

    WRITE ORDER is load-bearing: the zip lands first, the etag mapping second,
    the ``latest.json`` pointer last, so neither pointer can ever be read
    naming a digest or an etag mapping that has not actually landed yet.
    """
    cfg = config or load_config()
    client = client or make_client(cfg)
    sha = digest_of(blob)
    key = key_for(sha)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        version = parse_timetable(zf).version.version

    already_held = key in set(list_keys(client, cfg.bucket, PREFIX))
    if not already_held:
        client.put_object(
            Bucket=cfg.bucket, Key=key, Body=blob, ContentType="application/zip"
        )

    if etag is not None:
        client.put_object(
            Bucket=cfg.bucket,
            Key=etag_key(etag),
            Body=json.dumps({"digest": sha}, separators=(",", ":")),
            ContentType="application/json",
        )

    # Always refresh the pointer, even when the artifact was already held: a
    # prior run may have stored the artifact but crashed before writing the
    # pointer, and the pointer's own timestamp lets a reader see how recent the
    # last archive run was.
    _write_latest(
        client, cfg.bucket, sha, version, etag=etag, last_modified=last_modified
    )

    return FeedSnapshot(
        digest=sha,
        version=version,
        key=key,
        n_bytes=len(blob),
        etag=etag,
        stored=not already_held,
    )


def _write_latest(
    client: S3Client,
    bucket: str,
    digest: str,
    version: str,
    *,
    etag: str | None = None,
    last_modified: str | None = None,
) -> None:
    """Overwrite the latest-digest pointer."""
    from datetime import UTC, datetime

    doc = {
        "digest": digest,
        "version": version,
        "etag": etag,
        "last_modified": last_modified,
        "stored_at": int(datetime.now(UTC).timestamp()),
    }
    client.put_object(
        Bucket=bucket,
        Key=LATEST_KEY,
        Body=json.dumps(doc, separators=(",", ":")),
        ContentType="application/json",
    )


def store_current(
    url: str = GTFS_STATIC_URL,
    *,
    config: R2Config | None = None,
    client: S3Client | None = None,
) -> FeedSnapshot:
    """Fetch the feed unconditionally and store it, carrying the ETag and
    Last-Modified validators the response was served with.

    Unlike `fetch_gtfs_zip` (which trades the body away and keeps only bytes
    for its local 304 cache), this needs the response object itself: the
    validators it carries are what `store()` stamps on `latest.json` and
    `by_etag/`, so the Worker can later tell "same republish" from "new one"
    without re-hashing.
    """
    with httpx.Client(timeout=FETCH_TIMEOUT, follow_redirects=True) as http_client:
        response = http_client.get(url)
        response.raise_for_status()
        blob = response.content
    etag = response.headers.get("etag")
    last_modified = response.headers.get("last-modified")
    return store(
        blob, etag=etag, last_modified=last_modified, config=config, client=client
    )


def load(
    digest: str,
    *,
    config: R2Config | None = None,
    client: S3Client | None = None,
) -> bytes:
    """The exact bytes of a stored feed.

    Verifies the digest on read: a content-addressed store whose contents do not
    hash to their own key has been corrupted, and every historical number derived
    from it would be quietly wrong.
    """
    cfg = config or load_config()
    client = client or make_client(cfg)
    blob = get_object_bytes(client, cfg.bucket, key_for(digest))
    got = digest_of(blob)
    if got != digest:
        raise ValueError(f"stored feed {digest} hashes to {got}")
    return blob


def timetable_for(
    digest: str,
    *,
    config: R2Config | None = None,
    client: S3Client | None = None,
) -> Timetable:
    """The timetable that was in force, rebuilt from the stored artifact — the
    whole reason this archive exists."""
    with zipfile.ZipFile(io.BytesIO(load(digest, config=config, client=client))) as zf:
        return parse_timetable(zf)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Store the current static GTFS feed content-addressed in R2"
    )
    parser.add_argument("--url", default=GTFS_STATIC_URL)
    args = parser.parse_args(argv)

    snap = store_current(args.url)
    print(
        f"{'stored' if snap.stored else 'already held'} {snap.key} "
        f"({snap.n_bytes / 1e6:.1f} MB, {snap.version})",
        file=sys.stderr,
    )
    print(
        json.dumps(
            {
                "digest": snap.digest,
                "version": snap.version,
                "key": snap.key,
                "n_bytes": snap.n_bytes,
                "etag": snap.etag,
                "stored": snap.stored,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
