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
import statistics
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


# (route, direction, from_stop, to_stop) -> scheduled seconds for that one hop.
HopKey = tuple[str, str, str, str]


def _gtfs_seconds(value: str) -> int | None:
    """HH:MM:SS since noon-minus-12h into seconds. Hours run past 24 for trips
    that cross midnight (25:14:00 is a real value in this feed), so this cannot
    use a time parser — it is deliberately plain arithmetic."""
    parts = value.strip().split(":")
    if len(parts) != 3:
        return None
    try:
        h, m, s = (int(p) for p in parts)
    except ValueError:
        return None
    return h * 3600 + m * 60 + s


def hop_seconds(zf: zipfile.ZipFile) -> dict[HopKey, int]:
    """(route, direction, from_stop, to_stop) -> median scheduled seconds for
    that hop, from consecutive stop_sequence pairs in stop_times.txt.

    This is the external reference the movement model has never had. Everything
    the repo currently calls a baseline is fitted on the same vehicle archive it
    then grades against, so a systematic error in the archive cannot be detected
    from inside it. The timetable is an independent statement of how long a hop
    is supposed to take, published by the agency, and it converts "the stop_id
    changed" into "the train covered N scheduled seconds of ground".

    Median, not mean, over the trips serving a hop: run times differ by time of
    day and a handful of padded late-night trips would otherwise drag a hop long.

    Measured from departure at from_stop to arrival at to_stop, so scheduled
    dwell at from_stop is excluded — the quantity is travel, not travel plus
    standing. Hops where either time is missing or non-increasing are dropped.
    """
    with zf.open("trips.txt") as raw:
        trip_routes = _trip_routes(io.TextIOWrapper(raw, encoding="utf-8-sig"))

    samples: dict[HopKey, list[int]] = defaultdict(list)

    def flush(route: str, trip_id: str, stops: list[tuple[int, str, int, int]]) -> None:
        if len(stops) < 2:
            return
        stops.sort(key=lambda s: s[0])
        for (_, frm, _, dep), (_, to, arr, _) in itertools.pairwise(stops):
            direction = direction_of(frm, trip_id)
            if direction is None or frm == to:
                continue
            if arr <= dep:
                continue
            samples[(route, direction, frm, to)].append(arr - dep)

    with zf.open("stop_times.txt") as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
        header = next(reader)
        trip_col = header.index("trip_id")
        stop_col = header.index("stop_id")
        seq_col = header.index("stop_sequence")
        arr_col = header.index("arrival_time")
        dep_col = header.index("departure_time")

        current_trip: str | None = None
        current_route = ""
        buf: list[tuple[int, str, int, int]] = []
        for row in reader:
            trip_id = row[trip_col]
            if trip_id != current_trip:
                if current_trip is not None:
                    flush(current_route, current_trip, buf)
                current_trip = trip_id
                current_route = trip_routes.get(trip_id, "")
                buf = []
            if not current_route:
                continue
            arr = _gtfs_seconds(row[arr_col])
            dep = _gtfs_seconds(row[dep_col])
            if arr is None or dep is None:
                continue
            buf.append((int(row[seq_col]), row[stop_col], arr, dep))
        if current_trip is not None:
            flush(current_route, current_trip, buf)

    return {key: int(statistics.median(v)) for key, v in samples.items()}


def load_hop_seconds(url: str = GTFS_STATIC_URL) -> dict[HopKey, int]:
    """Fetch + parse in one call, mirroring load_successors."""
    data = fetch_gtfs_zip(url)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return hop_seconds(zf)


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


def _dominant_edges(
    succ: Mapping[SegmentKey, list[tuple[str, int]]],
) -> dict[tuple[str, str], dict[str, str]]:
    """(route, direction) -> {from_stop: dominant to_stop}. The single-successor
    skeleton `chains`, `terminals` and `through_stops` are all read off."""
    out: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for (route, direction, frm), succs in succ.items():
        if succs:
            out[(route, direction)][frm] = dominant_successor(succs)[0]
    return out


# (route, direction, stop) — a flat membership set, the shape the movement
# accounting tests each observed from_stop against.
TerminalKey = tuple[str, str, str]


def terminals(
    succ: Mapping[SegmentKey, list[tuple[str, int]]],
) -> frozenset[TerminalKey]:
    """Every (route, direction, stop) that ends a scheduled run: a source (no
    incoming dominant edge, i.e. an origin terminal) or a sink (no outgoing one,
    i.e. a destination terminal).

    Read off the dominant-successor skeleton rather than `chains`, whose `stops`
    concatenates one walk per entry stop — a component with two entries has two
    origins, and only the source/sink test names both.
    """
    out: set[TerminalKey] = set()
    for (route, direction), dominant in _dominant_edges(succ).items():
        incoming = set(dominant.values())
        for stop in set(dominant) | incoming:
            if stop not in incoming or stop not in dominant:
                out.add((route, direction, stop))
    return frozenset(out)


def stops_to_json(
    stops: frozenset[TerminalKey],
) -> dict[str, dict[str, list[str]]]:
    """Serialize a (route, direction, stop) set for params.json delivery to the
    Worker: route -> direction -> sorted stops, the same nesting as
    load_r2.advance_baseline_to_json."""
    out: dict[str, dict[str, list[str]]] = {}
    for route, direction, stop in sorted(stops):
        out.setdefault(route, {}).setdefault(direction, []).append(stop)
    return out


def through_stops(
    succ: Mapping[SegmentKey, list[tuple[str, int]]],
) -> frozenset[TerminalKey]:
    """The complement of `terminals` within the skeleton: stops with both a
    scheduled predecessor and a scheduled successor.

    Stricter than excluding `terminals` from observed data, because a stop the
    skeleton never names at all — a yard lead, a rare pattern, a stop_id the
    vehicle feed reports but the timetable doesn't — is absent here too. Those
    stops have no scheduled successor, so "it should have moved by now" is not
    defined for a train sitting at one.
    """
    out: set[TerminalKey] = set()
    for (route, direction), dominant in _dominant_edges(succ).items():
        incoming = set(dominant.values())
        for stop in dominant:
            if stop in incoming:
                out.add((route, direction, stop))
    return frozenset(out)


def chains(
    succ: Mapping[SegmentKey, list[tuple[str, int]]],
) -> dict[tuple[str, str], list[Chain]]:
    """Group each (route, direction) into weakly-connected components of its
    dominant-successor graph. One Chain in the result means static GTFS names
    an unbroken run for that (route, direction); more than one means the
    timetable itself has a gap in this feed cut (e.g. a shuttle/rare pattern
    disjoint from the main line)."""
    out: dict[tuple[str, str], list[Chain]] = {}
    for group, dominant in _dominant_edges(succ).items():
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
