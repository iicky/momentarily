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

from training.gtfs_static import (
    GTFS_STATIC_URL,
    Timetable,
    fetch_gtfs_zip,
)
from training.gtfs_static import timetable as parse_timetable
from training.load_r2 import list_keys
from training.r2_client import R2Config, get_object_bytes, load_config, make_client

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

PREFIX = "archive/gtfs/"


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
    stored: bool  # False when this digest was already present


def store(
    blob: bytes,
    *,
    config: R2Config | None = None,
    client: S3Client | None = None,
) -> FeedSnapshot:
    """Store a feed zip under its own digest, skipping bytes already held.

    Idempotent by construction: the key IS the content hash, so a nightly job
    that fetches an unchanged feed writes nothing and cannot create a second
    copy under a different name.
    """
    cfg = config or load_config()
    client = client or make_client(cfg)
    sha = digest_of(blob)
    key = key_for(sha)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        version = parse_timetable(zf).version.version

    if key in set(list_keys(client, cfg.bucket, PREFIX)):
        return FeedSnapshot(
            digest=sha, version=version, key=key, n_bytes=len(blob), stored=False
        )
    client.put_object(
        Bucket=cfg.bucket, Key=key, Body=blob, ContentType="application/zip"
    )
    return FeedSnapshot(
        digest=sha, version=version, key=key, n_bytes=len(blob), stored=True
    )


def store_current(
    url: str = GTFS_STATIC_URL,
    *,
    config: R2Config | None = None,
    client: S3Client | None = None,
) -> FeedSnapshot:
    return store(fetch_gtfs_zip(url), config=config, client=client)


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
                "stored": snap.stored,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
