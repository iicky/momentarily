"""Prune aged R2 objects from the date-partitioned streams.

The bucket credentials are object-scoped, so R2 lifecycle rules aren't
available to us — this job is the retention policy instead, run at the end of
the nightly training workflow. Both date-keyed prefixes (archive/*,
v1/predictions, v1/regime_transitions) are pruned by their YYYY-MM-DD path
segment; versioned params snapshots by their v<epoch> filename. Retention
windows leave plenty of headroom over what training (14d) and eval (7d) read
— except archive/trace/, which is capped tighter than that headroom because
of its size, not its readers; see the comment on its entry below.

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
    # One object/minute, ~85 KB each: ~125 MB/day, ~45.6 GB/year unbounded —
    # two orders of magnitude more per day than the other prefixes here, so
    # it does not get their 90-day window. training/traversal.py's baseline
    # is the only reader and wants a few weeks of history, not a year; 30d
    # covers that with headroom while keeping steady-state size in the
    # single-digit GB.
    ("archive/trace/", 30),
    # The DERIVED traversals, kept far longer than the raw trace they come from
    # because they are ~56x smaller (1.5 MB/day gzipped against 81 MB/day,
    # measured 2026-08-15) and are what every downstream measure actually reads.
    # This window is the whole point: a 30-day cap on the raw trace meant no
    # model could ever be evaluated on more than a month, and 10 years of this
    # costs about 5 GB. Long enough to be effectively "keep", explicit so the
    # prefix is still policed rather than silently unbounded.
    ("archive/traversals/", 3650),
    # The parsed ANSWER KEY. Same reasoning and same window as the traversals it
    # grades: keeping measurements without the announced work they are graded
    # against would leave a decade of traversals and nothing to compare them to.
    # Tiny — 541 windows over five days — so the window is set by usefulness
    # rather than size.
    ("archive/windows/", 3650),
    # NOT date-partitioned: archive/gtfs/ is content-addressed by sha256, so the
    # date matcher below never matches it and it is intentionally absent from
    # this table. Deleting a feed artifact by age would break replay of exactly
    # the oldest days the traversal archive still holds, and at 5.6 MB per
    # republish the whole history is a rounding error. See training.gtfs_archive.
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
