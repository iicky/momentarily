"""What the local and the R2 backfill commands share: arguments, the printed
and JSON reports and the exit status."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

from momentarily.model.observations.derive.storage import (
    ConditionalStorage,
    Storage,
)
from momentarily.model.observations.legacy.run import DayReport, Report, run
from momentarily.provenance import code_provenance

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def _day(text: str) -> str:
    try:
        if date.fromisoformat(text).isoformat() == text:
            return text
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(f"{text!r} is not a YYYY-MM-DD day")


def add_arguments(parser: argparse.ArgumentParser, *, range_required: bool) -> None:
    default = "" if range_required else " (default: the first day the source holds)"
    parser.add_argument(
        "--from",
        dest="first",
        type=_day,
        required=range_required,
        help=f"first UTC day, YYYY-MM-DD{default}",
    )
    default = "" if range_required else " (default: the last day the source holds)"
    parser.add_argument(
        "--through",
        dest="last",
        type=_day,
        required=range_required,
        help=f"last UTC day, YYYY-MM-DD, inclusive{default}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace stored legacy files whose content differs (default: keep "
        "them and fail); a raw-lineage or unreadable file is never replaced",
    )
    parser.add_argument("--report", type=Path, help="also write the report as JSON")


def _print_day(d: DayReport) -> None:
    print(
        f"{d.day}  objects {d.objects:>6}  malformed {d.malformed:>3}  "
        f"written {d.written:>3}  unchanged {d.unchanged:>3}  "
        f"replaced {d.replaced:>3}  conflicts {d.conflicts:>3}  "
        f"obsolete {d.obsolete:>3}  {d.seconds:>7.1f} s"
        + (f"  ABORTED, nothing written: {d.aborted} unreadable" if d.aborted else ""),
        flush=True,
    )


def _print(report: Report) -> None:
    print(f"\n{'archive':<26}{'objects':>9}{'rows':>10}{'malformed':>11}{'seconds':>9}")
    for name, a in report.archives.items():
        rows = sum(a.rows.values())
        print(
            f"{name:<26}{a.inputs:>9}{rows:>10}{len(a.failures):>11}{a.seconds:>9.1f}"
        )
        for record, n in sorted(a.rows.items()):
            print(f"  -> {record:<22}{'':>9}{n:>10}")
    print(f"\n{'record':<26}{'files':>9}{'rows':>10}{'bytes':>14}{'seconds':>9}")
    for name, r in sorted(report.records.items()):
        print(f"{name:<26}{r.files:>9}{r.rows:>10}{r.bytes:>14}{r.seconds:>9.1f}")
    print(
        f"\nfiles written {len(report.files) - len(report.replaced)}, "
        f"unchanged {len(report.unchanged)}, replaced {len(report.replaced)}, "
        f"conflicts {len(report.conflicts)}, obsolete {len(report.obsolete)}, "
        f"deferred {len(report.deferred)}, aborted days {len(report.aborted)}"
    )
    for day, key in report.aborted:
        print(
            f"aborted {day}: {key} is listed but could not be read; "
            "nothing of it written, rerun it",
            file=sys.stderr,
        )
    for name, a in report.archives.items():
        for key, reason in a.failures:
            print(f"malformed {name}: {key}: {reason}", file=sys.stderr)
    for key, reason in report.conflicts:
        print(f"conflict: {key}: {reason}; kept", file=sys.stderr)
    for key in report.obsolete:
        print(
            f"obsolete: {key} is a raw file before the cutover; kept", file=sys.stderr
        )


def execute(
    args: argparse.Namespace,
    source: Storage,
    out: ConditionalStorage,
    *,
    readers: int = 1,
) -> int:
    """Run the backfill the parsed ``args`` ask for and print its report."""
    if args.first is not None and args.last is not None and args.first > args.last:
        print(f"--from {args.first} is after --through {args.last}", file=sys.stderr)
        return EXIT_USAGE
    code = code_provenance()
    report = run(
        source,
        out,
        code["code_sha"],
        code["dirty"],
        first=args.first,
        last=args.last,
        overwrite=args.overwrite,
        readers=readers,
        on_day=_print_day,
    )
    _print(report)
    if args.report is not None:
        args.report.write_text(json.dumps(asdict(report), indent=2) + "\n")
    return EXIT_FAILED if report.failed else EXIT_OK
