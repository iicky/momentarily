"""Generate the committed diagram asset the map view draws.

The geometry is derived from the static GTFS feed (training/diagram.py), which
is a ~40 MB fetch and a full stop_times pass — far too slow for a page load and
static between service changes. So it's built here and committed as an asset,
and the map view is a plain fetch of it.

Regenerate after an MTA service change (a new station, a new branch, a route
withdrawn). Output is deterministic for a given feed: no timestamp, sorted
keys, so an empty diff means the timetable didn't move.

Run:  uv run python -m scripts.gen_diagram
      uv run python -m scripts.gen_diagram --zip /tmp/gtfs_subway.zip
"""

from __future__ import annotations

import argparse
import io
import json
import zipfile
from pathlib import Path

from training.diagram import build, to_json
from training.gtfs_static import GTFS_STATIC_URL, fetch_gtfs_zip

DIAGRAM_PATH = (
    Path(__file__).resolve().parent.parent / "viz" / "public" / "diagram.json"
)


def render(zf: zipfile.ZipFile) -> str:
    """The committed asset text: sorted keys + trailing newline, so the diff is
    stable and reviewable."""
    return json.dumps(to_json(build(zf)), indent=1, sort_keys=True) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--zip",
        type=Path,
        default=None,
        help=f"local GTFS zip to build from instead of fetching {GTFS_STATIC_URL}",
    )
    parser.add_argument("--out", type=Path, default=DIAGRAM_PATH)
    args = parser.parse_args()

    raw = args.zip.read_bytes() if args.zip else fetch_gtfs_zip()
    with zipfile.ZipFile(io.BytesIO(raw)) as zf:
        text = render(zf)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    payload = json.loads(text)
    edges = payload["edges"]
    # (class, direction) cells: one per edge per service class per direction
    # scheduled. Coverage of this against the drawn edge count is what
    # decides whether the scheduled-time overlay is worth rendering at all —
    # see gen_diagram's own docstring for how to regenerate after a service
    # change.
    n_cells = sum(len(dirs) for e in edges for dirs in e["seconds"].values())
    n_untimed = sum(1 for e in edges if not e["seconds"])
    n_stops = sum(len(pats) for pats in payload["route_stops"].values())
    print(
        f"wrote {args.out} — feed {payload['feed_version']['version']}, "
        f"{len(payload['stations'])} stations, {len(edges)} edges, "
        f"{n_cells} (class, direction) cells with a scheduled time, "
        f"{n_untimed} edges with no timing in any class or direction, "
        f"{len(payload['adjacency'])} adjacency keys, "
        f"{len(payload['route_stops'])} route|direction patterns "
        f"({n_stops} total, incl. minor variants)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
