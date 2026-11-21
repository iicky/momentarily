"""Backfill a local R2 mirror's legacy archives into derived parquet files.

    python -m momentarily.model.observations.legacy --mirror <mirror-root> \\
        --out <out-root> [--from YYYY-MM-DD] [--through YYYY-MM-DD] \\
        [--overwrite] [--report report.json] [--failure-report failures.md]

Writes ``<out>/model/observations/s1/derived/...`` (local only; R2 is
``python -m training.legacy_backfill``) for the UTC days ``--from`` through
``--through``, by default every day the mirror holds. Prints a line per day
as it ends (objects read, malformed objects, files written, unchanged,
replaced, conflicting and obsolete), then, per archive, objects read, rows,
malformed objects and time, and per record, files and bytes. ``--report``
also writes that report as JSON, with every malformed object's key and
reason, every conflicting file and its reason and every obsolete file.
``--failure-report`` writes the aborted days, the objects that aborted them
(key and reason), conflicts, obsolete files and quarantined objects as
Markdown. An object the reviewed quarantine file (``quarantine.py``) lists
with the sha256 of its bytes is read as absent and named in the files it
could have filled.

A stored file is never deleted. A file the derive job wrote (raw lineage),
or an unreadable one, is never overwritten: it is kept and listed as a
conflict, or as obsolete when it is a raw file wholly before the recorder
cutover (the derive job's ``--prune-obsolete`` removes it). A legacy file
with the same content is kept as unchanged; one with other content is a
conflict unless ``--overwrite`` replaces it. Exit status: 0 clean, 1 when
an object was malformed or unreadable (every day that reads it is aborted
and left unwritten) or any file conflicted or was obsolete (the rest is still
written), 2 when the mirror does not exist, the range is empty or the
``--failure-report`` directory does not exist.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from momentarily.model.observations.derive.storage import LocalStorage
from momentarily.model.observations.legacy.cli import (
    EXIT_USAGE,
    add_arguments,
    execute,
)


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
    add_arguments(parser, range_required=False)
    args = parser.parse_args(argv)
    mirror: Path = args.mirror.expanduser()
    if not mirror.is_dir():
        print(f"no mirror at {mirror}", file=sys.stderr)
        return EXIT_USAGE
    out = LocalStorage(args.out.expanduser())
    return execute(args, LocalStorage(mirror), out)


if __name__ == "__main__":
    raise SystemExit(main())
