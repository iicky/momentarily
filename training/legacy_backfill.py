"""Backfill the legacy archives in R2 into derived parquet files in R2.

    murk exec -- uv run python -m training.legacy_backfill \\
        --from YYYY-MM-DD --through YYYY-MM-DD [--overwrite] [--report r.json]

Reads ``archive/trace``, ``archive/trip_updates``, ``archive/vehicles``,
``archive/alerts``, ``archive/alerts_liveness``, ``archive/ene``,
``archive/health`` and ``v1/predictions`` from the bucket, one UTC day at a
time, and writes the legacy-lineage files under the same bucket's
``model/observations/s1/derived/`` keys. The arguments, the printed and JSON
reports, the write rules and the exit status are those of
``python -m momentarily.model.observations.legacy`` (docs/model-observations.md
§5.1); only the store differs, and the range is required. A new key is
written with a conditional put, so a file another run wrote meanwhile is
never replaced. ``--through`` must have ended at least 30 minutes ago
(``SETTLE``): the Worker writes a day's last objects after 00:00 UTC.
Credentials come from ``training.r2_client``.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime, time, timedelta

from momentarily.model.observations.legacy.cli import (
    EXIT_USAGE,
    add_arguments,
    execute,
)
from training.derive_observations import R2Storage
from training.r2_client import load_config, make_client

# Concurrent GETs; below botocore's default pool of 10 connections.
READERS = 8
# How long after a UTC day ends its last archive objects may still land.
SETTLE = timedelta(minutes=30)


def unsettled(day: str, now: datetime) -> str | None:
    """Why ``day`` cannot be the last day of a run at ``now``; None when
    it can."""
    ready = datetime.combine(date.fromisoformat(day), time(), UTC)
    ready += timedelta(days=1) + SETTLE
    if now >= ready:
        return None
    return (
        f"--through {day} is not settled: its last objects can land until "
        f"{ready:%Y-%m-%dT%H:%MZ} (30 minutes after the day ends); "
        f"it is {now:%Y-%m-%dT%H:%MZ} now"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m training.legacy_backfill")
    add_arguments(parser, range_required=True)
    args = parser.parse_args(argv)
    problem = unsettled(args.last, datetime.now(UTC))
    if problem is not None:
        print(problem, file=sys.stderr)
        return EXIT_USAGE
    config = load_config()
    store = R2Storage(make_client(config), config.bucket)
    return execute(args, store, store, readers=READERS)


if __name__ == "__main__":
    raise SystemExit(main())
