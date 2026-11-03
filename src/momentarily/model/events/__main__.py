"""Command line for the events layer. Local only: nothing here uploads.

    python -m momentarily.model.events match-index --network <layer-1 dir> \\
        --date YYYY-MM-DD [--date ...] [--build-id ID] [--day-feed-map PATH] \\
        --out <dir> [--overwrite]
    python -m momentarily.model.events coverage --root <layer-2 store> \\
        --out <dir> --day YYYY-MM-DD [--lineage raw|legacy] [--overwrite] \\
        [--report PATH]

``match-index``: ``--network`` is a layer-1 output directory (``index.json``,
``<build_id>.json.gz``, and by default ``day_feed_map.json``). The build of
each date is the live feed's per the day map, or an earlier one when the live
feed schedules nothing that day (``match_index.select_build``), unless
``--build-id`` names one. Each index
is written to ``<out>/<key>`` (see ``keys``), and one JSON line per date is
printed: key, the built index's bytes and uncompressed bytes, entry counts, and the outcome
(``written``, ``unchanged``, ``replaced`` or ``conflict``; see
``match_index.resolve_write``). Exit status 1 means a build failed or a
stored file conflicts; problems are printed to stderr.

``coverage``: ``--root`` is a store root holding layer-2 keys
(``model/observations/s1/``: recorder ticks and bundles, derived files). The
day's coverage file of each lineage (``--lineage``; by default every
lineage that holds the day, both on the cutover day) is written to
``<out>/<key>`` under ``files.resolve_write``, and a JSON report is printed
(and written to ``--report``). Exit status 1 means an input was missing or
inconsistent (nothing is written then) or a stored file conflicts; 2 a bad
day or a lineage that does not hold it.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from momentarily.model.events.contracts import Lineage
from momentarily.model.events.coverage import (
    CoverageReport,
    CoverageWindowError,
    build_coverage,
    coverage_window,
)
from momentarily.model.events.match_index import (
    IndexInputs,
    MatchIndexBuildError,
    build_match_index,
    read_day_map,
    read_network,
    read_network_index,
    resolve_write,
    select_build,
    serialize,
)
from momentarily.model.events.route_feed import load_route_feed
from momentarily.model.network.contracts import DayFeedMap, NetworkSnapshot
from momentarily.model.observations.derive.run import check_day
from momentarily.model.observations.derive.storage import LocalStorage
from momentarily.provenance import code_provenance


def _cmd_match_index(args: argparse.Namespace) -> int:
    root: Path = args.network
    day_map: DayFeedMap | None = None
    map_digest: str | None = None
    try:
        index = read_network_index(root)
        if args.build_id is None:
            day_map, map_digest = read_day_map(
                args.day_feed_map or root / "day_feed_map.json"
            )
    except MatchIndexBuildError as err:
        print(err, file=sys.stderr)
        return 1
    route_feed, route_feed_digest = load_route_feed()
    code = code_provenance()
    networks: dict[str, NetworkSnapshot] = {}

    def snapshot(feed_id: str) -> NetworkSnapshot:
        net = networks.get(feed_id)
        if net is None:
            net = networks[feed_id] = read_network(root, feed_id, index)
        return net

    status = 0
    for service_date in args.dates:
        try:
            choice = select_build(
                service_date, index, day_map, args.build_id, snapshot=snapshot
            )
            net = snapshot(choice.feed_id)
            doc = build_match_index(
                service_date,
                IndexInputs(
                    network=net,
                    network_digest=index.feeds[choice.feed_id].content_digest,
                    choice=choice,
                    day_map_digest=map_digest,
                    route_feed=route_feed,
                    route_feed_digest=route_feed_digest,
                    code_sha=code["code_sha"],
                    code_dirty=code["dirty"],
                ),
            )
        except MatchIndexBuildError as err:
            print(f"{service_date}: {err}", file=sys.stderr)
            status = 1
            continue
        path: Path = args.out / doc.metadata.file_key
        stored = path.read_bytes() if path.exists() else None
        outcome, data = resolve_write(stored, doc, args.overwrite)
        if data is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(path.name + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(path)
        if outcome == "conflict":
            print(
                f"{service_date}: {path} holds other bytes; kept "
                "(rerun with --overwrite to replace it)",
                file=sys.stderr,
            )
            status = 1
        built = serialize(doc)
        print(
            json.dumps(
                {
                    "key": doc.metadata.file_key,
                    "build_basis": doc.metadata.build_basis,
                    "bytes": len(built),
                    "uncompressed_bytes": len(gzip.decompress(built)),
                    "entries": doc.metadata.entries,
                    "outcome": outcome,
                },
                sort_keys=True,
            )
        )
    return status


def _coverage_json(report: CoverageReport) -> dict[str, Any]:
    doc = asdict(report)
    for feed, stats in report.feeds.items():
        doc["feeds"][feed]["uncovered_s"] = stats.uncovered_s
    doc["failed"] = report.failed
    return doc


def _cmd_coverage(args: argparse.Namespace) -> int:
    try:
        check_day(args.day)
    except ValueError as err:
        print(err, file=sys.stderr)
        return 2
    lineages: list[Lineage] = (
        [args.lineage] if args.lineage is not None else ["legacy", "raw"]
    )
    held: list[Lineage] = []
    for lineage in lineages:
        try:
            coverage_window(args.day, lineage)
        except CoverageWindowError as err:
            if args.lineage is not None:
                print(err, file=sys.stderr)
                return 2
            continue
        held.append(lineage)
    code = code_provenance()
    reports = [
        build_coverage(
            LocalStorage(args.root),
            LocalStorage(args.out),
            args.day,
            lineage,
            code["code_sha"],
            code["dirty"],
            overwrite=args.overwrite,
        )
        for lineage in held
    ]
    text = (
        json.dumps(
            {"day": args.day, "reports": [_coverage_json(r) for r in reports]},
            indent=2,
        )
        + "\n"
    )
    sys.stdout.write(text)
    if args.report is not None:
        args.report.write_text(text)
    status = 0
    for r in reports:
        if r.failed:
            print(
                f"{r.key}: {len(r.missing_inputs)} missing and "
                f"{len(r.inconsistent_inputs)} inconsistent inputs, "
                f"{len(r.conflicts)} conflicting files",
                file=sys.stderr,
            )
            status = 1
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m momentarily.model.events")
    sub = parser.add_subparsers(dest="command", required=True)
    mi = sub.add_parser("match-index", help="build per-service-day match indexes")
    mi.add_argument("--network", type=Path, required=True)
    mi.add_argument("--date", action="append", required=True, dest="dates")
    mi.add_argument("--build-id")
    mi.add_argument("--day-feed-map", type=Path)
    mi.add_argument("--out", type=Path, required=True)
    mi.add_argument("--overwrite", action="store_true")
    mi.set_defaults(func=_cmd_match_index)
    cov = sub.add_parser("coverage", help="build a UTC day's coverage files")
    cov.add_argument("--root", type=Path, required=True)
    cov.add_argument("--out", type=Path, required=True)
    cov.add_argument("--day", required=True, help="UTC day, YYYY-MM-DD")
    cov.add_argument("--lineage", choices=("raw", "legacy"))
    cov.add_argument("--overwrite", action="store_true")
    cov.add_argument("--report", type=Path, help="also write the JSON report here")
    cov.set_defaults(func=_cmd_coverage)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
