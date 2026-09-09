"""Cross-language contract guard for the arrivals derivation.

src/momentarily/arrivals.py derive_arrivals and worker/src/arrivals.ts
deriveArrivals fold decoded trips into the same per-stop arrivals surface. They
drift the moment one side's HORIZON_SECONDS, MAX_ARRIVALS, express fold, or a
drop rule moves without the other. The committed fixture
(tests/fixtures/parity_arrivals.json) pins them together.

This guards the Python side: derive_arrivals must still produce the fixture's
`expected` map. worker/test/arrivals_parity.test.ts guards the TS side against
the same fixture.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from momentarily.arrivals import StopTimeLite, TripLite, derive_arrivals

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "parity_arrivals.json"


def _trip(raw: dict[str, Any]) -> TripLite:
    return TripLite(
        route_id=raw["route_id"],
        trip_id=raw["trip_id"],
        direction=raw["direction"],
        stop_times=[
            StopTimeLite(
                stop_id=st["stop_id"],
                arrival=st["arrival"],
                departure=st["departure"],
                schedule_relationship=st["schedule_relationship"],
            )
            for st in raw["stop_times"]
        ],
    )


def test_arrivals_parity_fixture_reproduces_derive_arrivals() -> None:
    """The committed fixture must match what derive_arrivals produces today.
    If this fails, run: uv run python -m scripts.gen_arrivals_parity_fixture"""
    fixture = cast("dict[str, Any]", json.loads(FIXTURE_PATH.read_text()))
    trips = [_trip(t) for t in fixture["trips"]]
    actual = derive_arrivals(trips, fixture["now"])
    assert actual == fixture["expected"]
