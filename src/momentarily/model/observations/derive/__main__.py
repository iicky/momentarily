"""Derive one UTC day of a local store of recorded raw objects.

    python -m momentarily.model.observations.derive --root <dir> \\
        --day YYYY-MM-DD [--out <dir>] [--hours H-H] [--overwrite] \\
        [--report report.json]

``--root`` holds the bucket's keys as paths (``model/observations/s1/...``);
derived files go under ``--out`` (default: ``--root``) with the same keys.
Prints a JSON report: fetches by outcome, what was not decoded and why, per
record type the inputs, rows and files (and what happened to each file),
every decode failure, and every missing or inconsistent input. Exit status:
0 clean (a day with no recorded ticks is a clean no-op), 1 when a fetch
failed to decode, an input is missing or inconsistent, or an existing file
conflicts, 2 on bad arguments or a missing root.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from momentarily.model.observations.derive.cli import EXIT_USAGE, add_arguments, execute
from momentarily.model.observations.derive.storage import LocalStorage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m momentarily.model.observations.derive"
    )
    parser.add_argument("--root", type=Path, required=True, help="local store root")
    parser.add_argument("--out", type=Path, help="output root (default: --root)")
    add_arguments(parser)
    args = parser.parse_args(argv)
    root: Path = args.root.expanduser()
    if not root.is_dir():
        print(f"no store at {root}", file=sys.stderr)
        return EXIT_USAGE
    out = root if args.out is None else args.out.expanduser()
    return execute(args, LocalStorage(root), LocalStorage(out))


if __name__ == "__main__":
    raise SystemExit(main())
