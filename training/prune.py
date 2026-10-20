"""Prune aged params rollback snapshots from R2.

The bucket credentials are object-scoped, so R2 lifecycle rules aren't
available to us — this job is the retention policy instead, run as the last
step of the nightly archive workflow (.github/workflows/prune-nightly.yml).

The ONLY thing it deletes is versioned params rollback snapshots,
state/params/v<epoch>.json, by the epoch in their filename once they are older
than PARAMS_RETENTION_DAYS. The live state/params.json pointer is never
touched.

Every date-partitioned archive/ and v1/ stream is kept in full, indefinitely,
by owner decision: the raw captures (archive/trace/, archive/alerts/,
archive/vehicles/, ...) record live feeds that cannot be fetched again, so a
deleted day is gone for good, and the full history is worth more than its
storage. The derived archives (archive/traversals/, archive/windows/) and the
grading streams (v1/predictions/, v1/regime_transitions/,
v1/movement_transitions/) are kept for the same reason. Nothing here lists those
prefixes, so the job's cost does not grow with the archive.

Run with:
    murk exec -- python -m training.prune [--dry-run]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from training.r2_client import load_config, make_client

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

# state/params/v<epoch>.json rollback snapshots.
PARAMS_PREFIX = "state/params/"
PARAMS_RETENTION_DAYS = 180

_PARAMS_RE = re.compile(r"v(\d+)\.json$")


def _list_keys(client: S3Client, bucket: str, prefix: str) -> Iterable[str]:
    for page in client.get_paginator("list_objects_v2").paginate(
        Bucket=bucket, Prefix=prefix
    ):
        for obj in page.get("Contents") or []:
            key = obj.get("Key")
            if key:
                yield key


def _delete_batch(client: S3Client, bucket: str, keys: list[str]) -> None:
    for i in range(0, len(keys), 1000):
        chunk = keys[i : i + 1000]
        client.delete_objects(
            Bucket=bucket,
            Delete={"Objects": [{"Key": k} for k in chunk], "Quiet": True},
        )


def collect_expired(client: S3Client, bucket: str, now: datetime) -> list[str]:
    """Return the params rollback snapshots older than PARAMS_RETENTION_DAYS."""
    cutoff_epoch = int((now - timedelta(days=PARAMS_RETENTION_DAYS)).timestamp())
    return [
        key
        for key in _list_keys(client, bucket, PARAMS_PREFIX)
        if (m := _PARAMS_RE.search(key)) and int(m.group(1)) < cutoff_epoch
    ]


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prune aged state/params/ rollback snapshots from R2"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="list what would be deleted"
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    client = make_client(cfg)
    now = datetime.now(UTC)

    expired = collect_expired(client, cfg.bucket, now)
    print(f"{PARAMS_PREFIX}: {len(expired)} expired", file=sys.stderr)
    if not expired:
        print("nothing to prune")
        return 0
    if args.dry_run:
        print(f"dry-run: would delete {len(expired)} objects")
        return 0

    _delete_batch(client, cfg.bucket, expired)
    print(f"pruned {len(expired)} objects")
    return 0


if __name__ == "__main__":
    sys.exit(main())
