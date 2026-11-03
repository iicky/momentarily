"""What the local and the R2 backfill commands share: arguments, the printed
and JSON reports, the failure report and the exit status."""

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
from momentarily.model.observations.legacy.quarantine import (
    Entry,
    QuarantineError,
    load,
)
from momentarily.model.observations.legacy.run import DayReport, Report, run
from momentarily.provenance import code_provenance

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2

# Lines per section of the failure report: a GitHub issue body holds at most
# 65,536 characters.
_LIST_LIMIT = 50
_REASON_LIMIT = 300


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
    parser.add_argument(
        "--failure-report",
        type=Path,
        help="also write the aborted days, the objects that aborted them (key "
        "and reason), conflicts, obsolete files and quarantined objects as "
        "Markdown (the workflow's failure issue reads it)",
    )


def _print_day(d: DayReport) -> None:
    print(
        f"{d.day}  objects {d.objects:>6}  malformed {d.malformed:>3}  "
        f"written {d.written:>3}  unchanged {d.unchanged:>3}  "
        f"replaced {d.replaced:>3}  conflicts {d.conflicts:>3}  "
        f"obsolete {d.obsolete:>3}  {d.seconds:>7.1f} s"
        + (
            f"  ABORTED, nothing written: {d.aborted} unreadable or malformed"
            if d.aborted
            else ""
        ),
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
            f"aborted {day}: {key} is unreadable or malformed; "
            "nothing of it written, rerun it",
            file=sys.stderr,
        )
    for name, a in report.archives.items():
        for key, reason in a.failures:
            print(f"malformed {name}: {key}: {reason}", file=sys.stderr)
    for key, reason in report.excluded:
        print(f"quarantined: {key} read as absent ({reason})")
    for key, reason in report.conflicts:
        print(f"conflict: {key}: {reason}; kept", file=sys.stderr)
    for key in report.obsolete:
        print(
            f"obsolete: {key} is a raw file before the cutover; kept", file=sys.stderr
        )


def _code(text: str) -> str:
    """``text`` as one Markdown code span on one line, shortened."""
    flat = " ".join(text.split()).replace("`", "'")
    if len(flat) > _REASON_LIMIT:
        flat = flat[: _REASON_LIMIT - 1] + "…"
    return f"`{flat}`"


def _section(title: str, lines: list[str]) -> list[str]:
    if not lines:
        return []
    more = len(lines) - _LIST_LIMIT
    shown = lines[:_LIST_LIMIT] + ([f"- … and {more} more"] if more > 0 else [])
    return [f"### {title}", "", *shown, ""]


def failure_report(report: Report) -> str:
    """The report's failures as Markdown: each aborted day with the object
    that aborted it, each unreadable or malformed object with its reason,
    conflicts, obsolete files, and the quarantined objects read as absent.
    Keys, counts and reasons only."""
    reasons = {
        key: reason for a in report.archives.values() for key, reason in a.failures
    }

    def why(key: str) -> str:
        return reasons.get(key, "listed but could not be read (after retries)")

    stopped = [(d, k) for d, k in report.aborted if not _is_day(d)]
    days = [(d, k) for d, k in report.aborted if _is_day(d)]
    objects = sorted({k for _, k in report.aborted} | reasons.keys())
    out = [
        f"Aborted days: {len(days)}; unreadable or malformed objects: "
        f"{len(objects)}; conflicts: {len(report.conflicts)}; obsolete: "
        f"{len(report.obsolete)}; quarantined objects read as absent: "
        f"{len(report.excluded)}.",
        "",
    ]
    out += _section(
        "Run stopped before any day",
        [f"- {_code(k)} ({_code(d)} is read in full first)" for d, k in stopped],
    )
    out += _section(
        "Aborted days (nothing written)",
        [f"- {_code(d)}: {_code(k)}" for d, k in days],
    )
    out += _section(
        "Unreadable or malformed objects",
        [f"- {_code(k)}: {_code(why(k))}" for k in objects],
    )
    out += _section(
        "Conflicts (stored file kept)",
        [f"- {_code(k)}: {_code(r)}" for k, r in report.conflicts],
    )
    out += _section(
        "Obsolete raw files (kept)", [f"- {_code(k)}" for k in report.obsolete]
    )
    out += _section(
        "Quarantined objects read as absent",
        [f"- {_code(k)}: {_code(r)}" for k, r in report.excluded],
    )
    return "\n".join(out)


def _is_day(text: str) -> bool:
    try:
        return _day(text) == text
    except argparse.ArgumentTypeError:
        return False


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
    failures: Path | None = args.failure_report
    if failures is not None and not failures.parent.is_dir():
        print(f"--failure-report: no directory {failures.parent}", file=sys.stderr)
        return EXIT_USAGE
    try:
        quarantined: dict[str, Entry] = load()
    except QuarantineError as err:
        problem = f"the quarantine file is invalid: {err}"
        print(problem, file=sys.stderr)
        if args.failure_report is not None:
            args.failure_report.write_text(f"{_code(problem)}\n")
        return EXIT_FAILED
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
        quarantined=quarantined,
    )
    _print(report)
    if args.report is not None:
        args.report.write_text(json.dumps(asdict(report), indent=2) + "\n")
    if args.failure_report is not None:
        args.failure_report.write_text(failure_report(report))
    return EXIT_FAILED if report.failed else EXIT_OK
