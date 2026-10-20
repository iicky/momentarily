"""Command line for the network layer. Local only: nothing here uploads.

snapshot-stations --out DIR
    Fetch the MTA Subway Stations dataset once and write a canonical
    ``stations-<digest16>.json`` snapshot into DIR.

build (--gtfs ZIP | --live) --stations SNAPSHOT [--overrides JSON]
      [--day-feed-map JSON] --out DIR
    Build ``<build_id>.json.gz`` into DIR and record it in DIR/index.json.
    With --day-feed-map, the stations input counts as retroactive from the
    first UTC archive day observed on the feed rather than from valid_from.
    A build_id names one network's data: an existing file with the same data
    has its provenance refreshed; one whose data differs is refused.

day-feed-map --traversals DIR --out DIR
    Write DIR/day_feed_map.json: the feed each archived traversal day was
    collected against, read from that day's own provenance.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import zlib
from pathlib import Path

from momentarily.model.network.build import (
    DEFAULT_SYSTEM_ID,
    Inputs,
    NetworkBuildError,
    build_network,
    current_code,
    first_observed,
    index_entry,
    load_overrides,
    network_digest,
    serialize,
)
from momentarily.model.network.contracts import (
    SCHEMA_ID,
    IndexEntry,
    NetworkIndex,
    NetworkSnapshot,
)
from momentarily.model.network.stations import parse_snapshot, snapshot_now

INDEX_NAME = "index.json"
DAY_FEED_MAP_NAME = "day_feed_map.json"


class BuildConflictError(RuntimeError):
    """The build file already at ``path`` cannot be replaced: it holds
    different network data, or it is not a readable network build."""

    def __init__(self, path: Path, reason: str) -> None:
        self.path = path
        self.reason = reason
        super().__init__(f"{path}: {reason}")


def dump_json(doc: object) -> str:
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def read_index(out_dir: Path) -> NetworkIndex:
    path = out_dir / INDEX_NAME
    if not path.is_file():
        return NetworkIndex(schema=SCHEMA_ID, feeds={})
    return NetworkIndex.model_validate_json(path.read_bytes())


def record_build(out_dir: Path, feed_id: str, entry: IndexEntry) -> None:
    index = read_index(out_dir)
    feeds = {**index.feeds, feed_id: entry}
    updated = NetworkIndex(schema=SCHEMA_ID, feeds=feeds)
    _write_atomic(
        out_dir / INDEX_NAME, dump_json(updated.model_dump(mode="json")).encode()
    )


def write_build(
    out_dir: Path, snapshot: NetworkSnapshot
) -> tuple[Path, IndexEntry, bool]:
    """Write ``<build_id>.json.gz``; a build_id names one network's data.

    Identical bytes: nothing to do. Same network data but different
    provenance (another commit, a newer day map): the file is replaced, since
    provenance is metadata about the build and may be refreshed. Different
    network data, or an existing file that cannot be read as a network:
    refused. Returns the path, the index entry for the file now on disk, and
    whether it was written.
    """
    path = out_dir / f"{snapshot.build_id}.json.gz"
    data = serialize(snapshot)
    if path.is_file():
        existing_bytes = path.read_bytes()
        if existing_bytes == data:
            return path, index_entry(snapshot), False
        try:
            existing = NetworkSnapshot.model_validate_json(
                gzip.decompress(existing_bytes)
            )
        except (OSError, EOFError, ValueError, zlib.error) as exc:
            reason = f"exists but is not a readable network/s1 build ({exc})"
            raise BuildConflictError(path, reason) from exc
        if network_digest(existing) != network_digest(snapshot):
            reason = (
                f"already holds a different network under build_id "
                f"{snapshot.build_id}; bump BUILDER_VERSION if the builder changed"
            )
            raise BuildConflictError(path, reason)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, data)
    return path, index_entry(snapshot), True


def _cmd_snapshot_stations(args: argparse.Namespace) -> int:
    path = snapshot_now(Path(args.out))
    print(path)
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    if args.live:
        from training.gtfs_static import fetch_gtfs_zip

        gtfs = fetch_gtfs_zip()
    else:
        gtfs = Path(args.gtfs).read_bytes()
    overrides, overrides_digest = load_overrides(
        Path(args.overrides).read_bytes() if args.overrides else None
    )
    inputs = Inputs(
        gtfs=gtfs,
        stations=parse_snapshot(Path(args.stations).read_bytes()),
        direction_overrides=overrides,
        overrides_digest=overrides_digest,
        first_observed=first_observed(Path(args.day_feed_map).read_bytes(), gtfs)
        if args.day_feed_map
        else None,
    )
    try:
        snapshot = build_network(inputs, system_id=args.system_id, stamp=current_code())
    except NetworkBuildError as exc:
        print(exc, file=sys.stderr)
        return 1
    out_dir = Path(args.out)
    try:
        path, entry, written = write_build(out_dir, snapshot)
    except BuildConflictError as exc:
        print(exc, file=sys.stderr)
        return 1
    record_build(out_dir, snapshot.feed.feed_id, entry)
    state = "wrote" if written else "unchanged"
    print(
        f"{state} {path} {path.stat().st_size} bytes "
        f"feed={snapshot.feed.feed_id} content={entry.content_digest}"
    )
    return 0


def _cmd_day_feed_map(args: argparse.Namespace) -> int:
    # Lazy: traversal decoding pulls in the archive reader, which only this
    # command needs.
    from momentarily.model.network.day_feed import day_feed_map

    out_dir = Path(args.out)
    index = read_index(out_dir)
    result = day_feed_map(Path(args.traversals), index)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_atomic(
        out_dir / DAY_FEED_MAP_NAME, dump_json(result.model_dump(mode="json")).encode()
    )
    verified = sum(1 for d in result.days.values() if d.verified)
    print(
        f"{out_dir / DAY_FEED_MAP_NAME}: {len(result.days)} days, "
        f"{verified} verified, {len(result.days) - verified} unverified"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m momentarily.model.network")
    sub = parser.add_subparsers(dest="command", required=True)

    snap = sub.add_parser("snapshot-stations", help="snapshot the stations dataset")
    snap.add_argument("--out", required=True)
    snap.set_defaults(func=_cmd_snapshot_stations)

    build = sub.add_parser("build", help="build a network snapshot")
    source = build.add_mutually_exclusive_group(required=True)
    source.add_argument("--gtfs", help="path to a GTFS zip")
    source.add_argument("--live", action="store_true", help="fetch the live feed")
    build.add_argument("--stations", required=True, help="stations snapshot file")
    build.add_argument("--overrides", help="JSON stop_id -> direction label fills")
    build.add_argument(
        "--day-feed-map", help="day_feed_map.json giving the feed's first day"
    )
    build.add_argument("--system-id", default=DEFAULT_SYSTEM_ID)
    build.add_argument("--out", required=True)
    build.set_defaults(func=_cmd_build)

    dfm = sub.add_parser("day-feed-map", help="map archived days to feeds")
    dfm.add_argument("--traversals", required=True)
    dfm.add_argument("--out", required=True)
    dfm.set_defaults(func=_cmd_day_feed_map)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
