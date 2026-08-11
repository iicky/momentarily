"""Segment topology from the static GTFS feed, not inferred modal adjacency.

training.segments.canonical_adjacency infers "next stop" from observed
cross-tick vehicle transitions: the modal to_stop out of a from_stop. That only
forms an unbroken (route, direction) chain where transitions are frequent,
unambiguous, and archived without gaps — never true system-wide (branch points
split the modal vote, thin off-peak windows never accumulate a modal successor
at all). The static feed carries the agency's own stop_sequence ordering
directly, so segment existence stops depending on how much cross-tick data
happened to survive.

Two entrypoints:
- fetch_gtfs_zip / load_successors: the network path a training run uses.
- successors: the pure parsing entrypoint, taking an already-open zip so tests
  run against a small synthetic fixture with no network access.
"""

from __future__ import annotations

import csv
import io
import itertools
import zipfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TextIO

import httpx

# Verified 2026-08-11 by fetching both candidates directly: the legacy
# web.mta.info URL 301-redirects to this exact object (same ETag, same
# Content-Length) — one candidate, so this is the one we hit.
GTFS_STATIC_URL = "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip"

FETCH_TIMEOUT = httpx.Timeout(30.0)

# (route, direction, from_stop) — the same cell key convention as
# training.segments.Adjacency, keyed on the directional stop_id.
SegmentKey = tuple[str, str, str]


def fetch_gtfs_zip(url: str = GTFS_STATIC_URL) -> bytes:
    """Download the static GTFS zip. Raises on a non-2xx response; the caller
    decides what "unavailable" means for its own fallback."""
    with httpx.Client(timeout=FETCH_TIMEOUT, follow_redirects=True) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.content


def base_route(route_id: str) -> str:
    """Strip the trailing express 'X' (6X/7X/FX -> 6/7/F). Mirrors
    worker/src/vehicles.ts baseRoute — same rule, not a second one."""
    return route_id[:-1] if route_id.endswith("X") else route_id


def direction_of(stop_id: str, trip_id: str = "") -> str | None:
    """North/south from the stop_id N/S suffix, falling back to the trip_id
    '..N'/'..S' char. Mirrors worker/src/vehicles.ts directionOf. NYCT
    stop_times.stop_id always carries the suffix, but the fallback keeps this
    in lockstep with the one place in the repo that already does this."""
    last = stop_id[-1:]
    if last == "N":
        return "north"
    if last == "S":
        return "south"
    i = trip_id.find("..")
    if i >= 0 and len(trip_id) > i + 2:
        c = trip_id[i + 2]
        if c == "N":
            return "north"
        if c == "S":
            return "south"
    return None


def _trip_routes(raw: TextIO) -> dict[str, str]:
    """trip_id -> base route_id from trips.txt."""
    reader = csv.reader(raw)
    header = next(reader)
    trip_col = header.index("trip_id")
    route_col = header.index("route_id")
    out: dict[str, str] = {}
    for row in reader:
        out[row[trip_col]] = base_route(row[route_col])
    return out


def successors(zf: zipfile.ZipFile) -> dict[SegmentKey, list[tuple[str, int]]]:
    """Pure parsing entrypoint: (route, direction, from_stop) -> [(to_stop,
    n_trips), ...], built from consecutive stop_sequence pairs within each
    trip. n_trips is the number of trips that traverse that exact pair — a
    from_stop with more than one successor is a real branch/express split in
    the timetable, not truncated to one.

    Streams both files off the open zip; stop_times.txt (the big one, tens of
    MB) is read trip-by-trip with a small per-trip buffer rather than loaded
    whole, relying on the NYCT export grouping every trip's rows contiguously.
    """
    with zf.open("trips.txt") as raw:
        trip_routes = _trip_routes(io.TextIOWrapper(raw, encoding="utf-8-sig"))

    counts: dict[SegmentKey, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def flush(route: str, trip_id: str, stops: list[tuple[int, str]]) -> None:
        if len(stops) < 2:
            return
        stops.sort(key=lambda s: s[0])
        for (_, frm), (_, to) in itertools.pairwise(stops):
            direction = direction_of(frm, trip_id)
            if direction is None or frm == to:
                continue
            counts[(route, direction, frm)][to] += 1

    with zf.open("stop_times.txt") as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
        header = next(reader)
        trip_col = header.index("trip_id")
        stop_col = header.index("stop_id")
        seq_col = header.index("stop_sequence")

        current_trip: str | None = None
        current_route = ""
        buf: list[tuple[int, str]] = []
        for row in reader:
            trip_id = row[trip_col]
            if trip_id != current_trip:
                if current_trip is not None:
                    flush(current_route, current_trip, buf)
                current_trip = trip_id
                current_route = trip_routes.get(trip_id, "")
                buf = []
            if current_route:
                buf.append((int(row[seq_col]), row[stop_col]))
        if current_trip is not None:
            flush(current_route, current_trip, buf)

    return {
        key: sorted(tos.items(), key=lambda t: (-t[1], t[0]))
        for key, tos in counts.items()
    }


def load_successors(
    url: str = GTFS_STATIC_URL,
) -> dict[SegmentKey, list[tuple[str, int]]]:
    """Fetch + parse in one call — the entrypoint a training run uses."""
    data = fetch_gtfs_zip(url)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return successors(zf)


def dominant_successor(succs: list[tuple[str, int]]) -> tuple[str, int]:
    """The highest-n_trips entry, ties broken on the smaller stop id — same
    tie-break rule as training.segments.canonical_adjacency, so a from_stop's
    "primary" successor is picked the same way everywhere in the repo."""
    return min(succs, key=lambda t: (-t[1], t[0]))


@dataclass(frozen=True)
class Chain:
    """One weakly-connected piece of a (route, direction)'s dominant-successor
    graph (the highest-n_trips edge per from_stop), walked in travel order
    from each of its entry stops (stops with no incoming dominant edge).
    `stops` lists every stop in the piece; a (route, direction) reduced to a
    single Chain is an unbroken run end to end under the static timetable."""

    stops: tuple[str, ...]


def chains(
    succ: Mapping[SegmentKey, list[tuple[str, int]]],
) -> dict[tuple[str, str], list[Chain]]:
    """Group each (route, direction) into weakly-connected components of its
    dominant-successor graph. One Chain in the result means static GTFS names
    an unbroken run for that (route, direction); more than one means the
    timetable itself has a gap in this feed cut (e.g. a shuttle/rare pattern
    disjoint from the main line)."""
    dominant_by_group: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for (route, direction, frm), succs in succ.items():
        if succs:
            dominant_by_group[(route, direction)][frm] = dominant_successor(succs)[0]

    out: dict[tuple[str, str], list[Chain]] = {}
    for group, dominant in dominant_by_group.items():
        nodes = set(dominant.keys()) | set(dominant.values())
        parent = {n: n for n in nodes}

        def find(x: str, parent: dict[str, str] = parent) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for frm, to in dominant.items():
            ra, rb = find(frm), find(to)
            if ra != rb:
                parent[ra] = rb

        comp_nodes: dict[str, set[str]] = defaultdict(set)
        for n in nodes:
            comp_nodes[find(n)].add(n)

        result: list[Chain] = []
        for members in comp_nodes.values():
            incoming = {to for frm, to in dominant.items() if frm in members}
            sources = sorted(n for n in members if n not in incoming) or [min(members)]
            visited: set[str] = set()
            stops: list[str] = []
            for src in sources:
                cur: str | None = src
                while cur is not None and cur not in visited:
                    visited.add(cur)
                    stops.append(cur)
                    cur = dominant.get(cur)
            for n in sorted(members - visited):  # cycle remnants, if any
                stops.append(n)
            result.append(Chain(stops=tuple(stops)))
        result.sort(key=lambda c: (-len(c.stops), c.stops[0] if c.stops else ""))
        out[group] = result
    return out
