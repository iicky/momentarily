"""Generate the Python<->TypeScript arrivals-derivation parity fixture.

src/momentarily/arrivals.py derive_arrivals and worker/src/arrivals.ts
deriveArrivals must fold the same decoded trips into the same per-stop arrivals
surface. Both apply one rule set — express fold to the base route, key by the
stop id incl. its N/S suffix, drop SKIPPED / timeless / past / beyond-horizon
rows, one entry per trip per stop (first occurrence wins),
soonest first, capped at MAX_ARRIVALS — and will drift the moment one side's
HORIZON_SECONDS, MAX_ARRIVALS, or a drop rule moves.

The fixture is a single scenario: one `trips` list and a `now`, plus the
`expected` per-stop map derive_arrivals produces for them today. The trips are
built to exercise every observable behaviour — ordering, the N cap, the horizon
cut, past-arrival drop, SKIPPED drop, per-trip dedupe, arrival-then-departure
fallback, express fold, and an omitted trip id.
tests/test_arrivals_parity.py (Python) and worker/test/arrivals_parity.test.ts
replay it and must reproduce `expected` exactly.

Run:  uv run python -m scripts.gen_arrivals_parity_fixture
"""

from __future__ import annotations

import json
from pathlib import Path

from momentarily.arrivals import StopTimeLite, TripLite, derive_arrivals

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "parity_arrivals.json"
)

NOW = 1_000_000


def _st(
    stop_id: str,
    *,
    arrival: int | None = None,
    departure: int | None = None,
    sr: int = 0,
) -> StopTimeLite:
    return StopTimeLite(
        stop_id=stop_id,
        arrival=arrival,
        departure=departure,
        schedule_relationship=sr,
    )


def _trips() -> list[TripLite]:
    trips: list[TripLite] = []

    # Ordering + N cap: 8 southbound Q trains at one stop, out of time order in
    # the feed. Sorted ascending the offsets are 60,120,300,600,900,1200,1500,
    # 1800; only the first 6 survive the cap.
    offsets = [600, 120, 900, 300, 1500, 60, 1200, 1800]
    for i, off in enumerate(offsets):
        trips.append(
            TripLite(
                route_id="Q",
                trip_id=f"q{i}",
                direction=3,  # NYCT enum S, matches the Q05S suffix
                stop_times=[_st("Q05S", arrival=NOW + off)],
            )
        )

    # arrival-then-departure fallback at one northbound stop: one row has only a
    # departure (used), one has both (arrival wins over its later departure).
    trips.append(
        TripLite(
            route_id="F",
            trip_id="f-dep",
            direction=1,
            stop_times=[_st("F10N", departure=NOW + 500)],
        )
    )
    trips.append(
        TripLite(
            route_id="F",
            trip_id="f-arr",
            direction=1,
            stop_times=[_st("F10N", arrival=NOW + 300, departure=NOW + 999)],
        )
    )

    # SKIPPED drop: the skipped row is dropped, the scheduled one survives.
    trips.append(
        TripLite(
            route_id="A",
            trip_id="a-skip",
            direction=3,
            stop_times=[_st("A15S", arrival=NOW + 250, sr=1)],
        )
    )
    trips.append(
        TripLite(
            route_id="A",
            trip_id="a-ok",
            direction=3,
            stop_times=[_st("A15S", arrival=NOW + 700)],
        )
    )

    # Per-trip dedupe: one trip lists the same stop twice; the first row in feed
    # order wins even though the second is earlier.
    trips.append(
        TripLite(
            route_id="R",
            trip_id="r-dup",
            direction=1,
            stop_times=[_st("R10N", arrival=NOW + 400), _st("R10N", arrival=NOW + 200)],
        )
    )

    # Past drop: the only row at this stop is already gone, so the stop is absent.
    trips.append(
        TripLite(
            route_id="B",
            trip_id="b-past",
            direction=3,
            stop_times=[_st("B20S", arrival=NOW - 100)],
        )
    )

    # Horizon drop: one second past the hour horizon, so absent.
    trips.append(
        TripLite(
            route_id="C",
            trip_id="c-far",
            direction=3,
            stop_times=[_st("C30S", arrival=NOW + 3601)],
        )
    )

    # Express fold: 6X folds to base route 6.
    trips.append(
        TripLite(
            route_id="6X",
            trip_id="6x-exp",
            direction=1,
            stop_times=[_st("D40N", arrival=NOW + 350)],
        )
    )

    # Omitted trip id -> null, and an empty stop id row is unkeyable and dropped.
    # Two anonymous trips at one stop are two arrivals: dedupe needs an id.
    trips.append(
        TripLite(
            route_id="E",
            trip_id="",
            direction=1,
            stop_times=[_st("E50N", arrival=NOW + 150), _st("", arrival=NOW + 150)],
        )
    )
    trips.append(
        TripLite(
            route_id="E",
            trip_id="",
            direction=1,
            stop_times=[_st("E50N", arrival=NOW + 450)],
        )
    )

    return trips


def _trip_json(trip: TripLite) -> dict[str, object]:
    return {
        "route_id": trip.route_id,
        "trip_id": trip.trip_id,
        "direction": trip.direction,
        "stop_times": [
            {
                "stop_id": st.stop_id,
                "arrival": st.arrival,
                "departure": st.departure,
                "schedule_relationship": st.schedule_relationship,
            }
            for st in trip.stop_times
        ],
    }


def build_fixture() -> dict[str, object]:
    trips = _trips()
    expected = derive_arrivals(trips, NOW)
    return {
        "now": NOW,
        "trips": [_trip_json(t) for t in trips],
        "expected": expected,
    }


def main() -> int:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(build_fixture(), indent=2) + "\n")
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
