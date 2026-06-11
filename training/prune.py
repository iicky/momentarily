"""Prune aged R2 objects from the date-partitioned streams.

The bucket credentials are object-scoped, so R2 lifecycle rules aren't
available to us — this job is the retention policy instead, run at the end of
the nightly training workflow. Both date-keyed prefixes (archive/*,
v1/predictions, v1/regime_transitions) are pruned by their YYYY-MM-DD path
segment; versioned params snapshots by their v<epoch> filename. Retention
windows leave plenty of headroom over what training (14d) and eval (7d) read.

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

# (prefix, retention_days) — objects under date-partitioned paths older than
# the window are deleted.
DATED_PREFIXES: tuple[tuple[str, int], ...] = (
    ("archive/alerts/", 90),
    ("archive/ene/", 90),
    ("v1/predictions/", 90),
    ("v1/regime_transitions/", 90),
)

# state/params/v<epoch>.json rollback snapshots.
PARAMS_PREFIX = "state/params/"
PARAMS_RETENTION_DAYS = 180

_DATE_RE = re.compile(r"/(\d{4}-\d{2}-\d{2})/")
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


def collect_expired(
    client: S3Client, bucket: str, now: datetime
) -> dict[str, list[str]]:
    """Return {prefix: [expired keys]} across all retention rules."""
    out: dict[str, list[str]] = {}

    for prefix, days in DATED_PREFIXES:
        cutoff = (now - timedelta(days=days)).date()
        expired: list[str] = []
        for key in _list_keys(client, bucket, prefix):
            m = _DATE_RE.search(key)
            if not m:
                continue
            try:
                day = datetime.strptime(m.group(1), "%Y-%m-%d").date()
            except ValueError:
                continue
            if day < cutoff:
                expired.append(key)
        out[prefix] = expired

    cutoff_epoch = int((now - timedelta(days=PARAMS_RETENTION_DAYS)).timestamp())
    out[PARAMS_PREFIX] = [
        key
        for key in _list_keys(client, bucket, PARAMS_PREFIX)
        if (m := _PARAMS_RE.search(key)) and int(m.group(1)) < cutoff_epoch
    ]
    return out


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prune aged R2 stream objects")
    parser.add_argument(
        "--dry-run", action="store_true", help="list what would be deleted"
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    client = make_client(cfg)
    now = datetime.now(UTC)

    expired = collect_expired(client, cfg.bucket, now)
    total = sum(len(keys) for keys in expired.values())
    for prefix, keys in expired.items():
        print(f"{prefix}: {len(keys)} expired", file=sys.stderr)
    if total == 0:
        print("nothing to prune")
        return 0
    if args.dry_run:
        print(f"dry-run: would delete {total} objects")
        return 0

    for keys in expired.values():
        if keys:
            _delete_batch(client, cfg.bucket, keys)
    print(f"pruned {total} objects")
    return 0


if __name__ == "__main__":
    sys.exit(main())
