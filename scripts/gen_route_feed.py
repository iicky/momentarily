"""Measure the RouteFeed table from layer-2 raw-lineage records.

Reads every derived ``vehicle`` and ``trip_update`` parquet file of the given
UTC days under a local layer-2 root (the derive job's ``--out``), and writes
``src/momentarily/model/events/route_feed.json``: per route key, the distinct
trip ids each GTFS-RT feed published, and the majority feed. See
``momentarily.model.events.route_feed`` for the rule. Fails when a route key
has no majority feed.

The committed table was measured on 2026-10-21 03:57–13:30 UTC (the first
recorder hours: Tuesday 2026-10-20 23:57 EDT to Wednesday 09:30 EDT, a
weeknight and the AM peak):

    uv run python -m momentarily.model.observations.derive --root <mirror> \\
        --day 2026-10-21 --out <derived>
    uv run python -m scripts.gen_route_feed --root <derived> --day 2026-10-21

Rerun on a longer window (weekends, nights with planned work) as the recorder
accumulates days; every network route a service day runs must have a key, or
the match index build fails.
"""

# pyarrow's stubs leave the parquet reader partly untyped.
# pyright: reportUnknownMemberType=false

from __future__ import annotations

import argparse
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pyarrow.parquet as pq

from momentarily.model.events.route_feed import (
    TABLE_FILE,
    ObservedRecord,
    derive_route_feed,
    serialize_route_feed,
)
from momentarily.model.observations.contracts import GtfsRtSource
from momentarily.model.observations.keys import PREFIX
from momentarily.model.observations.parquet import file_lineage, read_file_metadata

RECORDS: list[ObservedRecord] = ["trip_update", "vehicle"]
TABLE_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "momentarily"
    / "model"
    / "events"
    / TABLE_FILE
)


def _files(root: Path, days: list[str]) -> list[Path]:
    """Every derived vehicle and trip-update file of ``days``. Exits when there
    is none, or when one was not built from raw recorder payloads (a legacy
    backfill file has no ``source``)."""
    paths = [
        path
        for record in RECORDS
        for day in days
        for path in sorted((root / PREFIX / "derived" / record / day).glob("*.parquet"))
    ]
    if not paths:
        raise SystemExit(
            f"no derived vehicle or trip_update files for {days} in {root}"
        )
    for path in paths:
        lineage = file_lineage(read_file_metadata(path.read_bytes()))
        if lineage != "raw":
            raise SystemExit(
                f"{path} has {lineage or 'mixed'} lineage; RouteFeed is measured "
                "on raw-lineage files only (recorder days)"
            )
    return paths


def _rows(
    paths: list[Path],
) -> Iterator[tuple[GtfsRtSource, str | None, str | None, int]]:
    for path in paths:
        table = pq.read_table(path).select(["source", "observed_at", "trip"])
        rows: list[dict[str, object]] = table.to_pylist()
        for row in rows:
            trip = cast("dict[str, object] | None", row["trip"])
            route_id = trip.get("route_id") if trip is not None else None
            trip_id = trip.get("trip_id") if trip is not None else None
            yield (
                cast("GtfsRtSource", row["source"]),
                route_id if isinstance(route_id, str) else None,
                trip_id if isinstance(trip_id, str) else None,
                cast("int", row["observed_at"]),
            )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m scripts.gen_route_feed")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--day", action="append", required=True, dest="days")
    parser.add_argument("--out", type=Path, default=TABLE_PATH)
    args = parser.parse_args(argv)
    table = derive_route_feed(_rows(_files(args.root, sorted(args.days))), RECORDS)
    args.out.write_bytes(serialize_route_feed(table))
    print(args.out)


if __name__ == "__main__":
    main()
