"""Command line for the network layer. Local only: nothing here uploads.

snapshot-stations --out DIR
    Fetch the MTA Subway Stations dataset once and write a canonical
    ``stations-<digest16>.json`` snapshot into DIR.

build (--gtfs ZIP | --live) --stations SNAPSHOT [--overrides JSON] --out DIR
    Build ``<build_id>.json.gz`` into DIR and record it in DIR/index.json.

day-feed-map --traversals DIR --out DIR
    Write DIR/day_feed_map.json: the feed each archived traversal day was
    collected against, read from that day's own provenance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from momentarily.model.network.build import (
    DEFAULT_SYSTEM_ID,
    Inputs,
    NetworkBuildError,
    build_network,
    current_code,
    load_overrides,
    serialize,
)
from momentarily.model.network.contracts import (
    SCHEMA_ID,
    IndexEntry,
    NetworkIndex,
    NetworkSnapshot,
)
from momentarily.model.network.day_feed import day_feed_map
from momentarily.model.network.stations import parse_snapshot, snapshot_now

INDEX_NAME = "index.json"
DAY_FEED_MAP_NAME = "day_feed_map.json"


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


def record_build(out_dir: Path, snapshot: NetworkSnapshot) -> None:
    index = read_index(out_dir)
    feeds = dict(index.feeds)
    feeds[snapshot.feed.feed_id] = IndexEntry(
        build_id=snapshot.build_id,
        feed_version=snapshot.feed.feed_version,
        valid_from=snapshot.feed.valid_from,
        valid_to=snapshot.feed.valid_to,
        inputs=snapshot.provenance.inputs,
        applied_retroactively=snapshot.provenance.applied_retroactively,
    )
    updated = NetworkIndex(schema=SCHEMA_ID, feeds=feeds)
    _write_atomic(
        out_dir / INDEX_NAME, dump_json(updated.model_dump(mode="json")).encode()
    )


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
    )
    try:
        snapshot = build_network(inputs, system_id=args.system_id, stamp=current_code())
    except NetworkBuildError as exc:
        print(exc, file=sys.stderr)
        return 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = serialize(snapshot)
    path = out_dir / f"{snapshot.build_id}.json.gz"
    _write_atomic(path, data)
    record_build(out_dir, snapshot)
    print(f"{path} {len(data)} bytes feed={snapshot.feed.feed_id}")
    return 0


def _cmd_day_feed_map(args: argparse.Namespace) -> int:
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
