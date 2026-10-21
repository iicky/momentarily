"""Generate the Python<->TypeScript observations key-layout parity fixture.

src/momentarily/model/observations/keys.py and worker/src/observation_keys.ts
must format the keys the Worker recorder writes (tick record, GTFS-RT tick
bundle, deduplicated blob) to the same strings, or the offline derive job
will not find what the Worker stored.

The fixture lists inputs and the keys keys.py formats for them, at the edges
that a hand-rolled UTC conversion gets wrong: the epoch, the last second of
an hour and a day, a leap day, and the last second Python can represent.
tests/test_observation_keys_parity.py checks the fixture is current and
worker/test/recorder.test.ts replays it.

Run:  uv run python -m scripts.gen_observation_keys_fixture
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from momentarily.model.observations.keys import (
    BlobExt,
    BlobKey,
    BundleKey,
    TickKey,
    utc_day,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "parity_observation_keys.json"
)

SHA = "85d9b90685e0d03818241c8d339bb711e0ba51196249900529b7ddb99e560d1f"

ATTEMPTS: list[tuple[int, int]] = [
    (0, 0),
    (1792514520, 1792514520412),  # 2026-10-20 16:42 UTC, the contract fixtures' tick
    (1792519199, 1792519199999),  # last second of an hour
    (1798761599, 1798761600003),  # last second of a day; started in the next
    (1709164800, 1709164801000),  # 2024-02-29, a leap day
    (253402300799, 253402300799999),  # 9999-12-31T23:59:59Z
]

BLOBS: list[tuple[int, BlobExt]] = [
    (1792514521, "json"),
    (1798761599, "json"),  # last second of a day
    (1798761600, "pb"),  # first second of the next
    (1709164800, "zip"),
]


def build() -> dict[str, Any]:
    return {
        "attempts": [
            {
                "scheduled_at": scheduled_at,
                "started_at_ms": started_at_ms,
                "utc_day": utc_day(scheduled_at),
                "tick": TickKey(scheduled_at, started_at_ms).key,
                "bundle": BundleKey(scheduled_at, started_at_ms).key,
            }
            for scheduled_at, started_at_ms in ATTEMPTS
        ],
        "blobs": [
            {
                "observed_at": observed_at,
                "sha256": SHA,
                "ext": ext,
                "key": BlobKey.for_fetch(observed_at, SHA, ext).key,
            }
            for observed_at, ext in BLOBS
        ],
    }


def render() -> str:
    return json.dumps(build(), indent=2) + "\n"


def main() -> None:
    FIXTURE_PATH.write_text(render())
    print(f"wrote {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
