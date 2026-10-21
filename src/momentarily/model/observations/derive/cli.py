"""What the local and the R2 derive commands share: arguments, the JSON
report and the exit status."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from momentarily.model.observations.derive.run import Report, check_day, derive
from momentarily.model.observations.derive.storage import Storage
from momentarily.provenance import code_provenance

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--day", required=True, help="UTC day, YYYY-MM-DD")
    parser.add_argument(
        "--hours",
        help="only these UTC hours of the day, e.g. 14-15 (inclusive); "
        "the daily alert_version file is built only for a whole day",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing files whose content differs (default: keep "
        "them and fail, unless DECODER_VERSION changed)",
    )
    parser.add_argument("--report", type=Path, help="also write the JSON report here")


def _hours(text: str | None) -> tuple[int, int]:
    if text is None:
        return (0, 23)
    m = re.fullmatch(r"(\d{1,2})-(\d{1,2})", text)
    if m is None or not 0 <= int(m[1]) <= int(m[2]) <= 23:
        raise ValueError(f"--hours {text!r} is not H-H within 0-23")
    return int(m[1]), int(m[2])


def report_json(report: Report) -> dict[str, Any]:
    return {**asdict(report), "failed": report.failed}


def execute(args: argparse.Namespace, source: Storage, out: Storage) -> int:
    """Run the derive the parsed ``args`` ask for and print its report."""
    try:
        check_day(args.day)
        hours = _hours(args.hours)
    except ValueError as err:
        print(err, file=sys.stderr)
        return EXIT_USAGE
    code = code_provenance()
    report = derive(
        source,
        out,
        args.day,
        code["code_sha"],
        code["dirty"],
        hours=hours,
        overwrite=args.overwrite,
    )
    text = json.dumps(report_json(report), indent=2) + "\n"
    sys.stdout.write(text)
    if args.report is not None:
        args.report.write_text(text)
    if report.status == "no_ticks":
        print(
            f"no recorded ticks observed in {args.day}; nothing derived",
            file=sys.stderr,
        )
    elif report.failed:
        print(
            f"{args.day}: {len(report.decode_failures)} decode failures, "
            f"{len(report.missing_inputs)} missing and "
            f"{len(report.inconsistent_inputs)} inconsistent inputs, "
            f"{len(report.conflicts)} conflicting files",
            file=sys.stderr,
        )
    return EXIT_FAILED if report.failed else EXIT_OK
