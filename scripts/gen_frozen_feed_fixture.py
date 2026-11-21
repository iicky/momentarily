"""Cut the 2026-10-21 frozen-feed excerpt from a recorder store.

    uv run python -m scripts.gen_frozen_feed_fixture --store <bucket mirror>

``<bucket mirror>`` is a local copy of the bucket holding the recorder's keys
(``model/observations/s1/ticks``, ``raw/gtfsrt``, ``raw/blob``). The output,
``tests/fixtures/model/events/frozen_feed``, is a store root with the objects
of four invocations, copied byte for byte: 03:59:19, the last one before the
recorder cutover (04:00:00 UTC), and 04:00:19, 04:01:19 and 04:02:19. All
four fall inside the upstream freeze of docs/model-events.md §5: ``ace``,
``bdfm``, ``g``, ``jz``, ``nqrw`` and ``si`` serve a header stuck at
03:22:05-03:22:10, while ``l`` and ``numbered`` stay live. Each invocation
keeps its tick record, its GTFS-RT bundle and the JSON blobs its tick names,
so the layer-2 derive job runs on it unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from momentarily.model.observations.keys import BundleKey, TickKey

OUT = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "model"
    / "events"
    / "frozen_feed"
)

# (scheduled_at, started_at_ms) of the invocations kept.
INVOCATIONS = (
    (1792555159, 1792555161107),
    (1792555219, 1792555220662),
    (1792555279, 1792555302015),
    (1792555339, 1792555342631),
)


def keys(store: Path) -> list[str]:
    """Every key the excerpt holds: ticks, bundles and the blobs named."""
    found: set[str] = set()
    for ident in INVOCATIONS:
        tick = TickKey(*ident).key
        found |= {tick, BundleKey(*ident).key}
        doc = json.loads((store / tick).read_text())
        for fetch in doc["fetches"]:
            key = fetch["payload_key"]
            if key is not None and fetch["payload_member"] is None:
                found.add(key)
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    args = parser.parse_args()
    if OUT.exists():
        shutil.rmtree(OUT)
    for key in keys(args.store):
        target = OUT / key
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(args.store / key, target)
        print(key)


if __name__ == "__main__":
    main()
