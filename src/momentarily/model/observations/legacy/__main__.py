"""Backfill a local R2 mirror's legacy archives into derived parquet files.

    python -m momentarily.model.observations.legacy --mirror <mirror-root> \\
        --out <out-root> [--report report.json]

Writes ``<out>/model/observations/s1/derived/...`` (local only, never R2) and
prints, per archive, objects read, rows, malformed objects and time, and per
record, files and bytes. ``--report`` also writes that report as JSON, with
every malformed object's key and reason and every conflicting file. A file
the derive job wrote (raw lineage), or an unreadable one, is never
overwritten: it is kept and listed as a conflict. Exit status: 0 clean, 1
when any object was skipped as malformed or any file conflicted (the rest is
still written), 2 when the mirror does not exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from momentarily.model.observations.legacy.run import Report, run
from momentarily.provenance import code_provenance


def _print(report: Report) -> None:
    print(f"{'archive':<26}{'objects':>9}{'rows':>10}{'malformed':>11}{'seconds':>9}")
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
    for name, a in report.archives.items():
        for key, reason in a.failures:
            print(f"malformed {name}: {key}: {reason}", file=sys.stderr)
    for key in report.conflicts:
        print(f"conflict: {key} is not a readable legacy file; kept", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m momentarily.model.observations.legacy"
    )
    parser.add_argument(
        "--mirror", type=Path, required=True, help="local R2 mirror root"
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="output root for derived/ keys"
    )
    parser.add_argument("--report", type=Path, help="also write the report as JSON")
    args = parser.parse_args(argv)
    mirror: Path = args.mirror.expanduser()
    if not mirror.is_dir():
        print(f"no mirror at {mirror}", file=sys.stderr)
        return 2
    code = code_provenance()
    report = run(mirror, args.out.expanduser(), code["code_sha"], code["dirty"])
    _print(report)
    if args.report is not None:
        args.report.write_text(json.dumps(asdict(report), indent=2) + "\n")
    malformed = sum(len(a.failures) for a in report.archives.values())
    return 1 if malformed or report.conflicts else 0


if __name__ == "__main__":
    raise SystemExit(main())
