"""Origin-destination direction-demand DIAGNOSTIC (training/od_direction.py).

A parked validation instrument, NOT a pipeline stage. It publishes nothing,
touches no R2, and the Worker never reads it. Its job is to size and locate the
one bias the live platform-crowding surface knowingly does not correct: the
surface splits a complex's entry demand across its served platforms without any
sense of DIRECTION, so at a station where riders overwhelmingly head one way at
rush it overstates the counter-peak platform. This measures how big that is and
where, so "we deferred the direction axis" is a number rather than a shrug --
and so a future direction split has something to be graded against.

Source is NYS Open Data `28vm-gjqr` (MTA Subway Origin-Destination Ridership
Estimate: Beginning 2026), keyed by (origin complex, destination complex, month,
day_of_week, hour_of_day) -> estimated_average_ridership, with lat/lon on both
ends. For each origin complex it asks, per (weekday/weekend, local hour), what
share of demand goes to a destination NORTH of the origin (higher latitude)
versus south. That north/south share is the direction demand the 50/50 split
assumes away, and it is reported at each complex's own busiest weekday hour, so
the headline is a real peak, never a thin overnight cell.

Two things this measure is NOT, both of which keep it a diagnostic rather than a
split input:

  1. It does NOT establish which physical platform absorbs the demand. "North by
     latitude" only maps onto a platform's N/S label where the line runs compass
     north/south (Lex, 7th/8th/6th Ave, Broadway); on a crosstown line (the 7,
     L, shuttles) the label is nominal and the mapping is wrong. And a genuinely
     N/S line can carry a diagonal demand vector -- an Astoria rider bound for
     Manhattan travels south AND west -- so a strong measured asymmetry can be
     entirely real yet still not tell you the platform split without each line's
     own geometry, which this tool does not read.
  2. It is an estimate on an estimate (MTA infers destination from the next tap;
     ~80% of trips link, the rest are allocated; timestamp is the ENTRY hour, so
     an arrivals view is displaced earlier by the ride time).

The public endpoint cannot group by origin with a per-row latitude comparison
across the whole 72M-row table without timing out, but an origin-FILTERED query
hits the index, so this loops one query per origin. A full run over every
complex is therefore minutes-to-hours; `--complex` and `--limit` scope it.

Run with:
    uv run python -m training.od_direction [--complex ID ...] [--limit N]
        [--min-ridership R] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass

import httpx

DATASET_ID = "28vm-gjqr"
FEED_URL = f"https://data.ny.gov/resource/{DATASET_ID}.json"
STATIONS_DATASET = "39hk-dx4f"
STATIONS_URL = f"https://data.ny.gov/resource/{STATIONS_DATASET}.json"

# Origin-filtered aggregates still scan a complex's whole slice of the table
# (every month/dow/hour/destination), so the read runs tens of seconds; the
# connect budget stays short. This is a manual diagnostic, never a request path.
FETCH_TIMEOUT = httpx.Timeout(10.0, read=150.0)

# North vs south of the origin by latitude, and weekday vs weekend, both decided
# server-side so the query returns ~96 rows per origin instead of the raw slice.
_NS_CASE = "case(destination_latitude > origin_latitude,'N',true,'S')"
_CLS_CASE = "case(day_of_week in ('Saturday','Sunday'),'we',true,'wd')"


@dataclass(frozen=True)
class OriginDirection:
    """One origin complex's measured direction demand.

    `north_share[cls][hour]` is the fraction of demand heading north in that
    (weekday/weekend, local hour) cell, or None where the origin has no demand
    that cell -- never a fabricated 0.5. `demand[cls][hour]` is the total
    ridership in that cell, carried so the busiest hour (not a thin overnight
    one) sets the headline.
    """

    complex_id: str
    total_ridership: float
    north_share: dict[str, list[float | None]]
    demand: dict[str, list[float]]

    def busiest_weekday_hour(self) -> int | None:
        """The weekday hour with the most demand, or None if this complex has no
        weekday demand at all. The operationally relevant hour, and busy by
        construction, so a bias read here is never a thin-cell artifact."""
        cells = self.demand["wd"]
        if sum(cells) <= 0:
            return None
        return max(range(24), key=lambda h: cells[h])

    def rush_north_share(self) -> float | None:
        """North share at the busiest weekday hour, or None when undefined."""
        hour = self.busiest_weekday_hour()
        return None if hour is None else self.north_share["wd"][hour]

    def rush_bias(self) -> float:
        """|north_share - 0.5| at the busiest weekday hour: how far the demand at
        this complex's peak departs from the 50/50 the live split assumes. 0
        when undefined, so it sorts to the bottom."""
        share = self.rush_north_share()
        return 0.0 if share is None else abs(share - 0.5)


def reduce_origin(
    complex_id: str, rows: Iterable[Mapping[str, object]]
) -> OriginDirection:
    """Fold one origin's aggregate rows (cls, hh, dir, r) into its per-cell north
    share and per-cell demand. Pure -- no network. A cell with demand in neither
    direction stays None (not 0.0): a 0.5 default would fabricate a symmetric
    reading the data never supported."""
    north: dict[str, list[float]] = {"wd": [0.0] * 24, "we": [0.0] * 24}
    south: dict[str, list[float]] = {"wd": [0.0] * 24, "we": [0.0] * 24}
    total = 0.0
    for row in rows:
        cls = str(row["cls"])
        if cls not in north:
            raise ValueError(f"complex {complex_id}: unexpected class {cls!r}")
        hour = int(float(str(row["hh"])))
        if not 0 <= hour < 24:
            raise ValueError(f"complex {complex_id}: hour {hour!r} out of range")
        ridership = float(str(row["r"]))
        direction = str(row["dir"])
        if direction not in ("N", "S"):
            raise ValueError(
                f"complex {complex_id}: unexpected direction {direction!r}"
            )
        (north if direction == "N" else south)[cls][hour] += ridership
        total += ridership

    north_share: dict[str, list[float | None]] = {}
    demand: dict[str, list[float]] = {}
    for cls in ("wd", "we"):
        shares: list[float | None] = []
        totals: list[float] = []
        for hour in range(24):
            n = north[cls][hour]
            s = south[cls][hour]
            totals.append(round(n + s, 3))
            shares.append(n / (n + s) if (n + s) > 0 else None)
        north_share[cls] = shares
        demand[cls] = totals

    return OriginDirection(
        complex_id=complex_id,
        total_ridership=round(total, 1),
        north_share=north_share,
        demand=demand,
    )


def _get_json(url: str, params: dict[str, str | int]) -> list[dict[str, object]]:
    with httpx.Client(timeout=FETCH_TIMEOUT) as client:
        response = client.get(url, params=params)
        response.raise_for_status()
        return response.json()


def fetch_origin(complex_id: str, url: str = FEED_URL) -> list[dict[str, object]]:
    """The per-origin aggregate: north/south x weekday/weekend x hour ridership
    sums. Origin-filtered so it hits the index."""
    return _get_json(
        url,
        {
            "$select": (
                f"{_CLS_CASE} as cls, hour_of_day as hh, {_NS_CASE} as dir, "
                "sum(estimated_average_ridership) as r"
            ),
            "$where": f"origin_station_complex_id='{complex_id}'",
            "$group": f"{_CLS_CASE}, hour_of_day, {_NS_CASE}",
            "$limit": 50_000,
        },
    )


def fetch_complex_ids(url: str = STATIONS_URL) -> list[tuple[str, str, str]]:
    """(complex_id, stop_name, daytime_routes) for every complex, from the
    stations dataset -- a cheap way to enumerate origins to loop, and to label
    the report with routes so a reader can weigh the line-geometry caveat."""
    rows = _get_json(
        url,
        {"$select": "complex_id, stop_name, daytime_routes", "$limit": 2000},
    )
    seen: dict[str, tuple[str, str, str]] = {}
    for row in rows:
        cid = str(row.get("complex_id") or "")
        if cid and cid not in seen:
            seen[cid] = (
                cid,
                str(row.get("stop_name") or ""),
                str(row.get("daytime_routes") or ""),
            )
    return list(seen.values())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose per-complex north/south direction-demand asymmetry from the "
            "OD estimate (28vm-gjqr). Publishes nothing; sizes the bias the live "
            "50/50 platform split assumes away."
        )
    )
    parser.add_argument(
        "--complex",
        action="append",
        dest="complexes",
        help="origin complex id to measure (repeatable); default loops all",
    )
    parser.add_argument(
        "--limit", type=int, help="measure only the first N complexes (quick look)"
    )
    parser.add_argument(
        "--min-ridership",
        type=float,
        default=0.0,
        help="skip complexes below this total OD ridership in the report",
    )
    parser.add_argument("--out", help="write the per-complex results as JSON here")
    args = parser.parse_args(argv)

    if args.complexes:
        targets = [(c, "", "") for c in args.complexes]
    else:
        targets = fetch_complex_ids()
        if args.limit is not None:
            targets = targets[: args.limit]

    results: list[tuple[OriginDirection, str, str]] = []
    for cid, name, routes in targets:
        try:
            origin = reduce_origin(cid, fetch_origin(cid))
        except httpx.HTTPError as err:
            print(f"  {cid}: fetch failed ({err})", file=sys.stderr)
            continue
        if origin.total_ridership < args.min_ridership:
            continue
        results.append((origin, name, routes))
        share = origin.rush_north_share()
        share_str = "--" if share is None else f"{share * 100:.0f}%N"
        print(
            f"  {cid:>4} {name[:26]:26} rush {share_str:>5} "
            f"bias={origin.rush_bias():.2f} ridership={origin.total_ridership:,.0f} "
            f"[{routes}]",
            file=sys.stderr,
        )

    if not results:
        print("no complexes measured", file=sys.stderr)
        return 1

    # The headline is the measured demand asymmetry at each complex's own busy
    # hour. It is NOT filtered by any geometry test: mapping these onto physical
    # platforms needs each line's direction, which this tool does not read, so
    # every row carries the same caveat rather than a false "trustworthy" tier.
    ranked = sorted(results, key=lambda r: r[0].rush_bias(), reverse=True)
    print(
        f"\n{len(results)} complexes measured. Largest weekday rush-hour N/S "
        "demand asymmetry (share of demand heading north at the busiest hour):"
    )
    for origin, name, routes in ranked[:15]:
        share = origin.rush_north_share()
        if share is None:
            continue
        hour = origin.busiest_weekday_hour()
        print(
            f"  {origin.complex_id:>4} {name[:24]:24} h{hour:>2}: "
            f"{share * 100:.0f}% N / {(1 - share) * 100:.0f}% S  [{routes}]"
        )
    print(
        "\nCaveat: north-by-latitude maps onto a platform's N/S label only where "
        "the line runs compass N/S; a diagonal demand vector or a crosstown line "
        "breaks that mapping, which this tool does not resolve. Demand asymmetry, "
        "not a per-platform correction."
    )

    if args.out:
        payload = [
            {"name": name, "routes": routes, **asdict(o)} for o, name, routes in results
        ]
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
