"""Diagram geometry for the spatial views: the drawable graph the map overlay
paints per-segment status onto.

The published measure is a DIRECTIONAL segment — one (route, direction,
from_stop) cell in `segment_flow.segments`, keyed `route|direction|from_stop`.
A map has to hand back exactly that cell, which is what rules out drawing the
system as route shapes: on the Lexington Ave trunk the 4, 5 and 6 run the same
track in both directions, so six separately-measured cells would land on one
stroke and not one of them would be readable.

So the unit here is a drawable EDGE: one per (route, unordered station pair),
carrying the segment_flow key for each direction that pair is scheduled in,
plus that direction's scheduled run time in seconds, split by NYCT service
class (weekday, Saturday, Sunday) because those really are different
timetables and blending them lies about what a viewer on any one of those
days will actually see. Routes sharing a pair are fanned out onto parallel
offsets, so every measured cell owns its own stroke and the renderer never
has to decide which cell a pixel belongs to.

Stations sit at their own GTFS coordinates in Web Mercator, and edges run on
the eight compass bearings between them — a schematic, not a geographic map: no
basemap, no tiles, no track curvature. The layout is ours, derived from the
timetable and the stop coordinates, both facts published in MTA's own feeds.
Nothing here is traced from MTA's map artwork, and the octilinear idiom itself
is a genre — Beck's 1933 London diagram — not anyone's property.

Stations are NOT moved onto a grid, which is the honest limit of this layout: a
long gentle run becomes a staircase of shallow bends, because each hop is
routed independently. Removing that means repositioning stations so consecutive
hops share a bearing, which is a global layout solve, not a change here.
Nothing in the output shape blocks it: an edge carries a `path` polyline, so
the status layer and the renderer don't move when this does.

Run: uv run python -m scripts.gen_diagram
"""

from __future__ import annotations

import csv
import io
import itertools
import math
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from training.gtfs_static import (
    FeedVersion,
    HopKey,
    base_route,
    parent_station,
    successors,
    timetable,
)

# Left-to-right order routes are fanned out in on a shared pair, grouped by the
# trunk they share: a route keeps the same side of a trunk over its whole run,
# so parallel strokes don't braid where the trunk's route set changes. Anything
# unlisted sorts last, alphabetically, rather than being dropped.
_TRUNKS: tuple[tuple[str, ...], ...] = (
    ("1", "2", "3", "4", "5", "6", "7"),
    ("A", "C", "E"),
    ("B", "D", "F", "M"),
    ("G",),
    ("J", "Z"),
    ("L",),
    ("N", "Q", "R", "W"),
    ("GS", "FS", "H", "SI"),
)

ROUTE_ORDER: tuple[str, ...] = tuple(r for trunk in _TRUNKS for r in trunk)

_ROUTE_RANK: Mapping[str, int] = {r: i for i, r in enumerate(ROUTE_ORDER)}

# Perpendicular gap between two routes sharing a station pair, in output units.
# The view box is VIEW_WIDTH across, and adjacent stations land ~10 units apart,
# so this keeps the four Brooklyn-trunk strokes separable without the fan
# reading wider than the hop it decorates.
OFFSET_SPACING = 2.6

# Output coordinates are normalized to this width; height follows from the
# projection's aspect ratio, never stretched to fit a target box.
VIEW_WIDTH = 1000.0

# Room for the station dot and its label at the extremes of the box.
PAD = 14.0

# The network isn't connected: the Staten Island Railway shares no track with
# the subway, and placed at its true position it stretches the bounding box
# southwest across open water, spending ~45% of the canvas on nothing. Every
# component but the largest is scaled down and tucked into the emptiest part of
# the main body's own box, the way MTA's sheet insets it. INSET_GRID is the
# resolution the empty region is searched at, INSET_MARGIN the fraction of the
# found block left as breathing room, and INSET_NEAR how much smaller than the
# largest empty block a candidate may be to win on being nearer the
# component's true position — an inset in the wrong corner is a map that lies
# about where Staten Island is, even when it's labelled.
INSET_GRID = 36
INSET_MARGIN = 0.12
INSET_NEAR = 0.8

# How far off a 45° axis a hop may run before it's redrawn as a dogleg. Tuned
# against the feed: every NYC hop is already within 22° of an axis and the
# median is 13°, so doglegging every hop staircases the long gentle runs while
# 10° catches the two-thirds that actually read as crooked.
OCTILINEAR_TOL = 10.0

# (route, low station id, high station id) — one drawable edge. The station
# pair is ordered so a pair scheduled in both directions is one edge, not two.
PairKey = tuple[str, str, str]

_Segment = tuple[str, str]
_Point = tuple[float, float]


def project(lat: float, lon: float) -> _Point:
    """Web Mercator in radians-scaled space, y growing southward so the result
    drops straight into SVG's top-left origin without a second flip."""
    return math.radians(lon), -math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def route_sort_key(route: str) -> tuple[int, str]:
    """ROUTE_ORDER position, with unlisted routes after every listed one."""
    rank = _ROUTE_RANK.get(route)
    return (len(ROUTE_ORDER), route) if rank is None else (rank, "")


# The eight compass bearings a diagram edge is allowed to run along.
_AXES = tuple(i * math.pi / 4 for i in range(-4, 5))


def axis_deviation(p: _Point, q: _Point) -> float:
    """Degrees between this hop's bearing and the nearest 45° axis."""
    bearing = math.atan2(q[1] - p[1], q[0] - p[0])
    return min(abs(bearing - axis) for axis in _AXES) * 180 / math.pi


def octilinear(p: _Point, q: _Point, tol: float = OCTILINEAR_TOL) -> tuple[_Point, ...]:
    """A hop routed on the eight compass bearings: straight, 45° diagonal,
    straight. Endpoints are returned unmoved, so consecutive edges still meet
    exactly and the whole path stays inside the hop's own bounding box.

    Splitting the axis-aligned run evenly across both ends is what makes it
    join cleanly: each edge LEAVES a station along an axis, so two edges
    meeting at a through-stop are collinear there instead of kinking.

    A hop already within `tol` of an axis is left straight. Every hop in the
    NYC feed sits within 22° of one (median 13°), so doglegging all of them
    turns a long gentle run into a staircase of shallow bends; the threshold
    spends the dogleg only where the hop is genuinely off-axis.
    """
    if axis_deviation(p, q) <= tol:
        return (p, q)
    (x1, y1), (x2, y2) = p, q
    dx, dy = x2 - x1, y2 - y1
    diagonal = min(abs(dx), abs(dy))
    run = (max(abs(dx), abs(dy)) - diagonal) / 2
    step_x = math.copysign(1.0, dx) if dx else 0.0
    step_y = math.copysign(1.0, dy) if dy else 0.0
    if abs(dx) >= abs(dy):
        first = (x1 + step_x * run, y1)
        second = (first[0] + step_x * diagonal, y1 + step_y * diagonal)
    else:
        first = (x1, y1 + step_y * run)
        second = (x1 + step_x * diagonal, first[1] + step_y * diagonal)
    out = [p]
    for point in (first, second, q):
        if math.dist(point, out[-1]) > 1e-9:
            out.append(point)
    return tuple(out)


@dataclass(frozen=True)
class RouteMeta:
    """Per-route display metadata straight off routes.txt. `color` is the feed's
    own `route_color` — the MTA publishes it as data, so the diagram doesn't
    carry a second hand-maintained palette."""

    id: str
    name: str
    color: str


@dataclass(frozen=True)
class Station:
    """A parent station at its projected position. Position is the station's
    own, unoffset: the fan applies to strokes, not to dots, so two routes
    through one station still share one dot."""

    id: str
    name: str
    x: float
    y: float
    routes: tuple[str, ...]


@dataclass(frozen=True)
class Edge:
    """One route's drawable run between two adjacent stations.

    `keys` maps direction -> the `segment_flow.segments` key measuring this
    pair in that direction. A pair the timetable only ever schedules one way
    (a one-way branch, a shuttle turnback) carries one entry; the renderer
    reads absence as "not measured here", never as a healthy reading.

    `seconds` maps NYCT service class ("weekday", "saturday", "sunday") ->
    direction -> the timetable's scheduled run time for that same
    (from_stop, to_stop) hop on a representative day of that class, in whole
    seconds. A class or direction missing from `seconds` means the timetable
    gave no arrival-to-arrival time there — never that the hop is instant —
    so the renderer must read absence as "no data", not 0. Classes are
    separate rather than pooled because they really are different
    timetables: `gtfs_static.Timetable`'s own docstring measures pooling all
    three at 26% of observed weekday hops wrong by more than 10%.
    """

    route: str
    a: str
    b: str
    path: tuple[_Point, ...]
    keys: Mapping[str, str]
    seconds: Mapping[str, Mapping[str, int]]


@dataclass(frozen=True)
class Inset:
    """A component drawn away from its true position, and where it went.

    Carried so the renderer can outline and label the panel: an inset that
    isn't marked as one is a map that lies about where Staten Island is.
    """

    routes: tuple[str, ...]
    # x, y, width, height in output units — the block the component occupies.
    box: tuple[float, float, float, float]
    # Output units per unit of the main body's scale. Below 1: drawn smaller.
    scale: float


@dataclass(frozen=True)
class Diagram:
    """The whole drawable graph plus the timetable it was derived from.

    `feed_version` is the provenance that matters: the static feed is a
    snapshot republished on every service change, so a diagram missing a branch
    is answered by which feed built it rather than by re-deriving it.
    """

    feed_version: FeedVersion
    view_box: tuple[float, float, float, float]
    routes: Mapping[str, RouteMeta]
    stations: Mapping[str, Station]
    edges: tuple[Edge, ...]
    insets: tuple[Inset, ...]


def _read_stops(zf: zipfile.ZipFile) -> dict[str, tuple[str, float, float]]:
    """Parent stations only, as (name, lat, lon).

    stops.txt carries both the station and its two platform rows; a platform
    declares `parent_station`, so skipping those leaves one row per station,
    which is the granularity a drawn dot has.
    """
    out: dict[str, tuple[str, float, float]] = {}
    with zf.open("stops.txt") as raw:
        for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig")):
            if row.get("parent_station"):
                continue
            try:
                lat = float(row["stop_lat"])
                lon = float(row["stop_lon"])
            except (KeyError, TypeError, ValueError):
                continue
            out[row["stop_id"]] = (row["stop_name"] or row["stop_id"], lat, lon)
    return out


def _read_routes(zf: zipfile.ZipFile) -> dict[str, RouteMeta]:
    """Route metadata folded onto base routes, so 6X/7X/FX read as the 6/7/F —
    the same collapse the segment keys are built on."""
    out: dict[str, RouteMeta] = {}
    with zf.open("routes.txt") as raw:
        for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig")):
            rid = base_route(row.get("route_id", ""))
            if not rid or rid in out:
                continue
            color = (row.get("route_color") or "").strip()
            out[rid] = RouteMeta(
                id=rid,
                name=(row.get("route_long_name") or rid).strip(),
                color=f"#{color}" if color else "#6e6e73",
            )
    return out


def _read_transfers(zf: zipfile.ZipFile) -> list[_Segment]:
    """Cross-stop transfer pairs, collapsed to parent stations.

    Not a routing input — the only thing this is used for is deciding which
    parts of the drawn graph are one place, since a shared complex is the only
    thing tying the IRT's stop ids to the IND/BMT's. transfers.txt is optional
    in GTFS, so an absent file is empty rather than fatal.
    """
    if "transfers.txt" not in zf.namelist():
        return []
    out: list[_Segment] = []
    with zf.open("transfers.txt") as raw:
        for row in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig")):
            a = parent_station(row.get("from_stop_id", ""))
            b = parent_station(row.get("to_stop_id", ""))
            if a and b and a != b:
                out.append((a, b))
    return out


def _dominant_hops(
    succ: Mapping[tuple[str, str, str], Sequence[tuple[str, int]]],
) -> dict[tuple[PairKey, str], tuple[str, str]]:
    """(pair, direction) -> (from_stop, to_stop): the busiest hop reaching this
    station pair in this direction, by scheduled trip count — the same rule as
    gtfs_static.dominant_successor. `edge_directions` and `edge_seconds` both
    read off this one resolution, so a pair's segment_flow key and its
    scheduled time always describe the same physical hop rather than two that
    happen to share a drawn edge.
    """
    best: dict[tuple[PairKey, str], tuple[int, str, str]] = {}
    for (route, direction, frm), cands in succ.items():
        origin = parent_station(frm)
        for to, n_trips in cands:
            target = parent_station(to)
            if origin == target:
                continue
            lo, hi = (origin, target) if origin <= target else (target, origin)
            slot = ((route, lo, hi), direction)
            prev = best.get(slot)
            if prev is None or n_trips > prev[0]:
                best[slot] = (n_trips, frm, to)
    return {slot: (frm, to) for slot, (_, frm, to) in best.items()}


def edge_directions(
    succ: Mapping[tuple[str, str, str], Sequence[tuple[str, int]]],
) -> dict[PairKey, dict[str, str]]:
    """Every scheduled station pair -> {direction: segment_flow key}.

    The keys are read off the timetable's own directional stop ids rather than
    rebuilt by re-suffixing a station id, so a pair scheduled in one direction
    only yields one key instead of a fabricated northbound twin. Where a
    from_stop reaches the same neighbour under more than one pattern the
    busiest wins, matching gtfs_static.dominant_successor's rule.
    """
    out: dict[PairKey, dict[str, str]] = defaultdict(dict)
    for (pair, direction), (frm, _) in _dominant_hops(succ).items():
        out[pair][direction] = f"{pair[0]}|{direction}|{frm}"
    return dict(out)


# NYCT's own service classes: Mon-Fri share one timetable, Saturday and Sunday
# each their own. date.weekday() is 0=Monday..6=Sunday.
_SERVICE_CLASS_BY_WEEKDAY: Mapping[int, str] = {
    0: "weekday",
    1: "weekday",
    2: "weekday",
    3: "weekday",
    4: "weekday",
    5: "saturday",
    6: "sunday",
}


def representative_days(version: FeedVersion) -> dict[str, date]:
    """One date per NYCT service class, the first of each found starting at
    the feed's own validity window rather than a hardcoded date — so the
    picks don't quietly go stale the next time MTA republishes with a shifted
    window. `version.start` is None for a feed with no feed_info.txt (an
    optional file), so today stands in for "the window that applies right
    now" in that case. Seven consecutive days always contain each of the
    three classes at least once.
    """
    day = version.start or date.today()
    found: dict[str, date] = {}
    for _ in range(7):
        found.setdefault(_SERVICE_CLASS_BY_WEEKDAY[day.weekday()], day)
        day += timedelta(days=1)
    return found


def edge_seconds(
    succ: Mapping[tuple[str, str, str], Sequence[tuple[str, int]]],
    hops_by_class: Mapping[str, Mapping[HopKey, int]],
) -> dict[PairKey, dict[str, dict[str, int]]]:
    """Every scheduled station pair -> {service class: {direction: scheduled
    seconds}}, read off the SAME (from_stop, to_stop) hop `edge_directions`
    keyed that direction with — `_dominant_hops` is the one resolution both
    draw from, so a pair's segment_flow key and its scheduled time never
    disagree about which hop on a branch they describe.

    `hops_by_class` is one `gtfs_static.DayTimetable.hops` per service class,
    each built from a single representative day (`representative_days`) — NOT
    pooled across classes. `Timetable`'s own docstring measures pooling all
    three at 26% of observed weekday hops wrong by more than 10%, because the
    weekend timetable really is a different schedule, not noise around the
    weekday one. A (class, direction) with no scheduled time for this hop is
    left out of the mapping entirely rather than published as 0 seconds, so a
    renderer can tell "no data" from "instant".
    """
    out: dict[PairKey, dict[str, dict[str, int]]] = defaultdict(dict)
    for (pair, direction), (frm, to) in _dominant_hops(succ).items():
        for cls, hops in hops_by_class.items():
            seconds = hops.get((pair[0], direction, frm, to))
            if seconds is not None:
                out[pair].setdefault(cls, {})[direction] = seconds
    return dict(out)


def _fan(routes: Iterable[str]) -> dict[str, float]:
    """Perpendicular offset per route sharing one station pair, centred on the
    pair's own line so a solo route draws down the middle."""
    order = sorted(routes, key=route_sort_key)
    mid = (len(order) - 1) / 2
    return {route: (i - mid) * OFFSET_SPACING for i, route in enumerate(order)}


def components(
    pairs: Iterable[PairKey], links: Iterable[_Segment] = ()
) -> list[frozenset[str]]:
    """Weakly-connected groups of stations, largest first.

    Track alone does NOT connect the network here, and this is the trap: the
    IRT, IND/BMT, the 7, the L and the shuttles each use their own stop ids at
    a shared complex (Times Sq is `127` to the 1, `R16` to the N, `A27` to the
    A), so a pure track graph splits the subway into six pieces and would inset
    five of them. `links` is transfers.txt's cross-stop rows, which is how the
    feed says two stop ids are one place. With those in, the subway is one
    component and the Staten Island Railway — reachable only by ferry, which
    this feed doesn't carry — is the one that is genuinely detached.
    """
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        root = node
        while parent[root] != root:
            root = parent[root]
        while parent[node] != root:
            parent[node], node = root, parent[node]
        return root

    edges = [(lo, hi) for _, lo, hi in pairs]
    edges.extend(links)
    for lo, hi in edges:
        a, b = find(lo), find(hi)
        if a != b:
            parent[a] = b

    groups: dict[str, set[str]] = defaultdict(set)
    for node in list(parent):
        groups[find(node)].add(node)
    return sorted(
        (frozenset(g) for g in groups.values()),
        key=lambda g: (-len(g), min(g)),
    )


def _empty_block(
    pos: Mapping[str, _Point],
    pairs: Iterable[PairKey],
    box: tuple[float, float, float, float],
    toward: _Point,
) -> tuple[float, float, float, float]:
    """A large empty square in `box`, as near `toward` as one can be found.

    The drawn edges are marked onto an INSET_GRID lattice and the biggest run
    of untouched cells wins — data-driven on purpose, because hard-coding
    "bottom left" would drop an inset on top of Coney Island the first time the
    timetable moved. Among blocks within INSET_NEAR of the largest, the one
    closest to `toward` (where the component would have been drawn at true
    position) wins, so the panel still points at the right part of the city
    instead of landing in whichever corner happened to be emptiest.
    """
    x0, y0, width, height = box
    cell_w = width / INSET_GRID
    cell_h = height / INSET_GRID
    used = [[False] * INSET_GRID for _ in range(INSET_GRID)]

    def mark(x: float, y: float) -> None:
        col = min(INSET_GRID - 1, max(0, int((x - x0) / cell_w)))
        row = min(INSET_GRID - 1, max(0, int((y - y0) / cell_h)))
        used[row][col] = True

    # The DRAWN route, not the chord between the stations: a dogleg leaves the
    # chord, so marking the chord would report cells as empty that the map
    # actually paints and drop the inset on top of them.
    step = min(cell_w, cell_h) / 2
    for _, lo, hi in pairs:
        route = octilinear(pos[lo], pos[hi])
        for (ax, ay), (bx, by) in itertools.pairwise(route):
            n = max(1, int(math.hypot(bx - ax, by - ay) / step))
            for i in range(n + 1):
                t = i / n
                mark(ax + (bx - ax) * t, ay + (by - ay) * t)

    # Largest all-empty square, standard DP: best[row][col] is the side of the
    # biggest square whose bottom-right corner is that cell.
    best = [[0] * INSET_GRID for _ in range(INSET_GRID)]
    side = 0
    for row in range(INSET_GRID):
        for col in range(INSET_GRID):
            if used[row][col]:
                continue
            best[row][col] = (
                1
                if row == 0 or col == 0
                else 1
                + min(best[row - 1][col], best[row][col - 1], best[row - 1][col - 1])
            )
            side = max(side, best[row][col])

    floor = max(1, int(side * INSET_NEAR))
    corner = min(
        (
            (row, col)
            for row in range(INSET_GRID)
            for col in range(INSET_GRID)
            if best[row][col] >= floor
        ),
        key=lambda rc: (
            math.dist(
                (
                    x0 + (rc[1] - best[rc[0]][rc[1]] / 2) * cell_w,
                    y0 + (rc[0] - best[rc[0]][rc[1]] / 2) * cell_h,
                ),
                toward,
            ),
            rc,
        ),
    )
    side = best[corner[0]][corner[1]]

    top = (corner[0] - side + 1) * cell_h + y0
    left = (corner[1] - side + 1) * cell_w + x0
    span_x, span_y = side * cell_w, side * cell_h
    keep = 1 - 2 * INSET_MARGIN
    return (
        left + span_x * INSET_MARGIN,
        top + span_y * INSET_MARGIN,
        span_x * keep,
        span_y * keep,
    )


def _place(
    stops: Mapping[str, tuple[str, float, float]],
    pairs: Iterable[PairKey],
    links: Iterable[_Segment],
) -> tuple[dict[str, _Point], tuple[float, float, float, float], tuple[Inset, ...]]:
    """Projected positions scaled to VIEW_WIDTH, the view box that holds them,
    and where any detached component was tucked.

    The main component sets the scale and the box. Aspect is preserved within
    each component — a stretched subway map lies about direction — but a
    detached component is drawn at its own smaller scale, which is exactly what
    marking it an inset declares.
    """
    pairs = list(pairs)
    raw = {
        sid: project(stops[sid][1], stops[sid][2])
        for _, lo, hi in pairs
        for sid in (lo, hi)
    }
    # Transfers can name a stop with no drawn pair (a complex whose platforms
    # this feed schedules nothing through), so intersect and re-rank: the group
    # sizes that matter are the drawn ones.
    drawn = {sid for _, lo, hi in pairs for sid in (lo, hi)}
    groups = sorted(
        (g for g in (group & drawn for group in components(pairs, links)) if g),
        key=lambda g: (-len(g), min(g)),
    )
    main = groups[0]

    main_xs = [raw[sid][0] for sid in main]
    main_ys = [raw[sid][1] for sid in main]
    span_x = max(main_xs) - min(main_xs)
    scale = VIEW_WIDTH / span_x if span_x > 0 else 1.0
    pos = {
        sid: (
            (raw[sid][0] - min(main_xs)) * scale + PAD,
            (raw[sid][1] - min(main_ys)) * scale + PAD,
        )
        for sid in main
    }
    height = (max(main_ys) - min(main_ys)) * scale + 2 * PAD
    view_box = (0.0, 0.0, VIEW_WIDTH + 2 * PAD, height)

    insets: list[Inset] = []
    for group in groups[1:]:
        xs = [raw[sid][0] for sid in group]
        ys = [raw[sid][1] for sid in group]
        own_x = max(xs) - min(xs)
        own_y = max(ys) - min(ys)
        # Where this component's centre would have landed at true position and
        # the main body's scale — outside the box when it's across water, which
        # is exactly the pull that keeps its panel on the right side of the map.
        true_centre = (
            (sum(xs) / len(xs) - min(main_xs)) * scale + PAD,
            (sum(ys) / len(ys) - min(main_ys)) * scale + PAD,
        )
        bx, by, bw, bh = _empty_block(
            pos,
            [p for p in pairs if p[1] in pos and p[2] in pos],
            (PAD, PAD, VIEW_WIDTH, height - 2 * PAD),
            true_centre,
        )
        fit = min(
            bw / own_x if own_x > 0 else bw,
            bh / own_y if own_y > 0 else bh,
        )
        # Centred in the block, so a long thin component doesn't hug one edge.
        offset_x = bx + (bw - own_x * fit) / 2
        offset_y = by + (bh - own_y * fit) / 2
        for sid in group:
            pos[sid] = (
                (raw[sid][0] - min(xs)) * fit + offset_x,
                (raw[sid][1] - min(ys)) * fit + offset_y,
            )
        insets.append(
            Inset(
                routes=tuple(
                    sorted(
                        {route for route, lo, _ in pairs if lo in group},
                        key=route_sort_key,
                    )
                ),
                box=(
                    round(offset_x, 2),
                    round(offset_y, 2),
                    round(own_x * fit, 2),
                    round(own_y * fit, 2),
                ),
                scale=round(fit / scale, 4),
            )
        )
    return pos, view_box, tuple(insets)


def build(zf: zipfile.ZipFile) -> Diagram:
    """The drawable diagram for one static feed."""
    stops = _read_stops(zf)
    succ = successors(zf)
    pairs = {
        pair: keys
        for pair, keys in edge_directions(succ).items()
        if pair[1] in stops and pair[2] in stops
    }
    if not pairs:
        raise ValueError("no drawable station pairs in feed")

    # A second read of stop_times.txt, this one for arrival_time rather than
    # stop_sequence. Kept separate from `successors` above instead of merged
    # into one parser: gen_diagram.py runs by hand after a service change, not
    # on a request path, so the extra pass costs a developer a few seconds,
    # not a page load, and the two parsers answer genuinely different
    # questions (successive-stop counts vs. arrival-to-arrival medians).
    tt = timetable(zf)
    hops_by_class = {
        cls: tt.day(day).hops for cls, day in representative_days(tt.version).items()
    }
    seconds = edge_seconds(succ, hops_by_class)

    used = sorted({sid for _, lo, hi in pairs for sid in (lo, hi)})
    pos, view_box, insets = _place(stops, pairs, _read_transfers(zf))

    # The fan is a property of the geometric pair, not of one route on it: the
    # routes sharing a pair have to agree on who sits where or their strokes
    # cross mid-hop.
    sharing: dict[_Segment, list[str]] = defaultdict(list)
    for route, lo, hi in pairs:
        sharing[(lo, hi)].append(route)

    normal: dict[_Segment, _Point] = {}
    offset: dict[PairKey, float] = {}
    for seg, routes in sharing.items():
        (x1, y1), (x2, y2) = pos[seg[0]], pos[seg[1]]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy) or 1.0
        normal[seg] = (-dy / length, dx / length)
        for route, off in _fan(routes).items():
            offset[(route, seg[0], seg[1])] = off

    # A route's displacement at a station is the mean of what its adjacent hops
    # ask for. Averaging at the shared vertex is what makes consecutive edges
    # meet exactly: offsetting each hop independently leaves a visible gap at
    # every station where the hop's bearing turns.
    shifts: dict[tuple[str, str], list[_Point]] = defaultdict(list)
    for pair in pairs:
        route, lo, hi = pair
        nx, ny = normal[(lo, hi)]
        off = offset[pair]
        for sid in (lo, hi):
            shifts[(route, sid)].append((nx * off, ny * off))
    shift = {
        key: (
            sum(p[0] for p in points) / len(points),
            sum(p[1] for p in points) / len(points),
        )
        for key, points in shifts.items()
    }

    edges: list[Edge] = []
    for pair, keys in sorted(pairs.items()):
        route, lo, hi = pair
        # Routed from the FANNED endpoints, not the station centres: two routes
        # sharing a pair get the same dogleg translated, so they stay parallel
        # over every segment of it instead of converging mid-hop.
        ends = tuple(
            (pos[sid][0] + shift[(route, sid)][0], pos[sid][1] + shift[(route, sid)][1])
            for sid in (lo, hi)
        )
        path = tuple(
            (round(x, 2), round(y, 2)) for x, y in octilinear(ends[0], ends[1])
        )
        edges.append(
            Edge(
                route=route,
                a=lo,
                b=hi,
                path=path,
                keys=dict(sorted(keys.items())),
                seconds={
                    cls: dict(sorted(dirs.items()))
                    for cls, dirs in sorted(seconds.get(pair, {}).items())
                },
            )
        )

    served: dict[str, set[str]] = defaultdict(set)
    for route, lo, hi in pairs:
        served[lo].add(route)
        served[hi].add(route)
    stations = {
        sid: Station(
            id=sid,
            name=stops[sid][0],
            x=round(pos[sid][0], 2),
            y=round(pos[sid][1], 2),
            routes=tuple(sorted(served[sid], key=route_sort_key)),
        )
        for sid in used
    }

    drawn = {route for route, _, _ in pairs}
    routes = {
        rid: meta for rid, meta in sorted(_read_routes(zf).items()) if rid in drawn
    }
    return Diagram(
        feed_version=tt.version,
        view_box=view_box,
        routes=routes,
        stations=stations,
        edges=tuple(edges),
        insets=insets,
    )


def to_json(diagram: Diagram) -> dict[str, Any]:
    """The committed asset's shape. No generation timestamp: regenerating
    against the same static feed produces a byte-identical file, so a diff
    means the timetable moved."""
    version = diagram.feed_version
    return {
        "feed_version": {
            "version": version.version,
            "start": version.start.isoformat() if version.start else None,
            "end": version.end.isoformat() if version.end else None,
        },
        "view_box": list(diagram.view_box),
        "routes": {
            rid: {"name": meta.name, "color": meta.color}
            for rid, meta in diagram.routes.items()
        },
        "stations": {
            sid: {
                "name": station.name,
                "x": station.x,
                "y": station.y,
                "routes": list(station.routes),
            }
            for sid, station in diagram.stations.items()
        },
        "edges": [
            {
                "route": edge.route,
                "a": edge.a,
                "b": edge.b,
                "path": [list(point) for point in edge.path],
                "keys": dict(edge.keys),
                "seconds": {cls: dict(dirs) for cls, dirs in edge.seconds.items()},
            }
            for edge in diagram.edges
        ],
        "insets": [
            {
                "routes": list(inset.routes),
                "box": list(inset.box),
                "scale": inset.scale,
            }
            for inset in diagram.insets
        ],
    }
