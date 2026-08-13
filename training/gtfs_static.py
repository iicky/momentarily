"""Segment topology and scheduled run times from the static GTFS feed.

The feed answers two questions the vehicle archive cannot answer about itself.

WHICH STOPS FOLLOW WHICH. training.segments.canonical_adjacency infers "next
stop" from observed cross-tick vehicle transitions: the modal to_stop out of a
from_stop. That only forms an unbroken (route, direction) chain where
transitions are frequent, unambiguous, and archived without gaps — never true
system-wide (branch points split the modal vote, thin off-peak windows never
accumulate a modal successor at all). The static feed carries the agency's own
stop_sequence ordering directly, so segment existence stops depending on how
much cross-tick data happened to survive.

HOW LONG THEY ARE SUPPOSED TO TAKE. Everything the repo calls a baseline is
otherwise fitted on the same archive it then grades against, so a systematic
error in the archive cannot be detected from inside it. `Timetable` is the
agency's own statement of how long a hop should take, sliced by service day and
resolved against the trip's own stopping pattern — see its docstring for why
each of those qualifiers is load-bearing.

Entrypoints come in pairs: `load_successors` / `load_timetable` take the network
path a training run uses, and `successors` / `timetable` take an already-open
zip so tests run against a small synthetic fixture with no network access.
"""

from __future__ import annotations

import bisect
import csv
import io
import itertools
import statistics
import zipfile
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import TextIO
from zoneinfo import ZoneInfo

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


def parent_station(stop_id: str) -> str:
    """Drop the directional platform suffix: 'J20N' -> 'J20', 'J20' unchanged.

    The two halves of the MTA's own data disagree on which id they mean. The
    realtime trace and stop_times.txt report directional platforms; the alerts
    feed's informed_entity names parent stations. Joining them without this finds
    zero matches and looks exactly like a signal that was not there.
    """
    return stop_id[:-1] if stop_id[-1:] in ("N", "S") else stop_id


def is_express_variant(trip_id: str) -> bool:
    """Whether a realtime trip id declares the express variant of its route:
    '124850_7X..N' does, '112750_4..S06R' does not.

    worker/src/vehicles.ts folds 6X/7X to 6/7 before the row is ever archived, so
    route_id cannot tell an express from a local and every downstream key pools
    the two. The trip id is the only surviving evidence, and anything that has to
    distinguish the two SERVICES rather than the two tracks needs it — measured
    2026-08-13, 870 of 5,996 traversals on route 7 were run by a 7X.

    Reads only the route field ahead of '..', so a path suffix that happens to
    contain an X ('6..S01X016') is not mistaken for an express.
    """
    return path_code(trip_id).split("..")[0].endswith("X")


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


def _trip_meta(raw: TextIO) -> dict[str, tuple[str, str]]:
    """trip_id -> (base route_id, service_id) from trips.txt."""
    reader = csv.reader(raw)
    header = next(reader)
    trip_col = header.index("trip_id")
    route_col = header.index("route_id")
    service_col = header.index("service_id")
    out: dict[str, tuple[str, str]] = {}
    for row in reader:
        out[row[trip_col]] = (base_route(row[route_col]), row[service_col])
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


# NYCT trip ids carry their own identity. The realtime feed publishes
# '063400_A..S' where the static feed has
# 'ASP26GEN-1038-Weekday-00_063400_A..S55R': the same id with the service-block
# prefix stripped, and often with the trailing path code cut short. The two
# fields that survive are the scheduled origin and the stopping pattern.
def path_code(trip_id: str) -> str:
    """The trailing field of a trip id, which names its stopping pattern."""
    return trip_id.rpartition("_")[2]


def origin_seconds(trip_id: str) -> int | None:
    """A run's scheduled origin, in seconds past its service day's midnight.

    The leading field is hundredths of a MINUTE, not HH:MM:SS: '060250' is 602.50
    minutes, 10:02:30. Measured over the whole feed — read as hundredths it
    reproduces the trip's own first scheduled arrival exactly for 20,311 of
    20,621 trips (the remaining 310 are off by a flat 90s, trips whose
    stop_times begin past the origin terminal); read as HH:MM:SS it matches 273,
    and would put most of the fleet hours early.
    """
    field = trip_id.partition("_")[0]
    return int(field) * 60 // 100 if field.isdigit() else None


# The service day is a local-time question and New York observes DST, so the
# day boundary moves twice a year.
_ET = ZoneInfo("America/New_York")


def _midnight(day: date) -> int:
    return int(datetime.combine(day, time.min, tzinfo=_ET).timestamp())


def service_dates(at: int, trip_id: str) -> tuple[date, ...]:
    """The service days an observation of this trip could belong to, nearest
    first.

    A trip id names its own scheduled origin, so a candidate day is ranked by
    how near its midnight puts the observation to that origin. That settles both
    cases the calendar date alone gets wrong, and needs no wall-clock cutoff: a
    run scheduled past midnight (origin 25:30, seen at 01:40) belongs to
    yesterday, and a train put in service before its scheduled origin (seen at
    07:55 for an 08:00 origin) belongs to today. Over 106,172 live arrivals the
    nearest candidate was right every time, none more than 50 minutes early
    against its own origin — the second candidate exists so a day whose calendar
    never ran the trip can be passed over rather than used.
    """
    local = datetime.fromtimestamp(at, _ET).date()
    origin = origin_seconds(trip_id)
    if origin is None:
        return (local,)
    return tuple(
        sorted(
            (local, local - timedelta(days=1)),
            key=lambda day: abs(at - _midnight(day) - origin),
        )
    )


_DAY_COLUMNS = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


@dataclass(frozen=True)
class _Weekly:
    """One calendar.txt row: the weekdays a service runs and its date range."""

    service_id: str
    weekdays: frozenset[int]
    start: str
    end: str


@dataclass(frozen=True)
class _Calendar:
    """calendar.txt with its calendar_dates.txt exceptions applied."""

    weekly: tuple[_Weekly, ...]
    added: Mapping[str, frozenset[str]]
    removed: Mapping[str, frozenset[str]]

    def active(self, day: date) -> frozenset[str]:
        stamp = day.strftime("%Y%m%d")
        running = {
            w.service_id
            for w in self.weekly
            if w.start <= stamp <= w.end and day.weekday() in w.weekdays
        }
        running |= self.added.get(stamp, frozenset())
        return frozenset(running - self.removed.get(stamp, frozenset()))


def _rows(zf: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    """Every row of an OPTIONAL feed file. GTFS lets a feed carry calendar.txt,
    calendar_dates.txt, or both, so an absent file is empty rather than fatal."""
    if name not in zf.namelist():
        return []
    with zf.open(name) as raw:
        return list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig")))


def _read_calendar(zf: zipfile.ZipFile) -> _Calendar:
    weekly = tuple(
        _Weekly(
            service_id=row["service_id"],
            weekdays=frozenset(
                i for i, col in enumerate(_DAY_COLUMNS) if row[col] == "1"
            ),
            start=row["start_date"],
            end=row["end_date"],
        )
        for row in _rows(zf, "calendar.txt")
    )
    added: dict[str, set[str]] = defaultdict(set)
    removed: dict[str, set[str]] = defaultdict(set)
    for row in _rows(zf, "calendar_dates.txt"):
        target = added if row["exception_type"] == "1" else removed
        target[row["date"]].add(row["service_id"])
    return _Calendar(
        weekly=weekly,
        added={k: frozenset(v) for k, v in added.items()},
        removed={k: frozenset(v) for k, v in removed.items()},
    )


@dataclass(frozen=True)
class FeedVersion:
    """Which timetable a measurement was taken against.

    The static feed is a snapshot, not a standing truth: MTA republishes on
    every service change with a self-dated, self-describing version (the copy
    read on 2026-08-13 was `20260807-H-rockaways-extension-removed`, six days
    old, and the name says it dropped a branch). Any number derived from the
    timetable is meaningless without saying which one, so this rides along with
    it.

    `covers` is the guard that matters. The feed declares the window it applies
    to, and the vehicle archive reaches back further than any one feed does —
    replaying an old window against today's timetable would compare trains
    against a schedule that did not exist when they ran.
    """

    version: str
    start: date | None
    end: date | None

    def covers(self, day: date) -> bool:
        if self.start is not None and day < self.start:
            return False
        return self.end is None or day <= self.end


def _feed_date(value: str) -> date | None:
    try:
        return datetime.strptime(value.strip(), "%Y%m%d").date()
    except ValueError:
        return None


def _read_version(zf: zipfile.ZipFile) -> FeedVersion:
    rows = _rows(zf, "feed_info.txt")
    row = rows[0] if rows else {}
    return FeedVersion(
        version=row.get("feed_version") or "unknown",
        start=_feed_date(row.get("feed_start_date") or ""),
        end=_feed_date(row.get("feed_end_date") or ""),
    )


@dataclass(frozen=True)
class Span:
    """A stretch of one trip's own scheduled pattern: how long the timetable
    allows for it, and how many of that trip's OWN stops it covers. `n_hops` is
    the part a caller cannot get anywhere else — a move the realtime feed reports
    as one hop but the timetable puts stations inside is a bypass, not a hop."""

    seconds: int
    n_hops: int


@dataclass(frozen=True)
class Pattern:
    """One scheduled stopping pattern: the ordered stops a path code names, and
    the cumulative arrival-to-arrival seconds from the first of them.

    The path code IS the trip's own stop sequence — across the whole feed there
    are 433 (service, path) pairs and not one of them carries a second stop list.
    Times are the median over the trips running the pattern, so the profile
    describes the pattern's typical day rather than any one departure.
    """

    stops: tuple[str, ...]
    offsets: tuple[int, ...]
    index: Mapping[str, int]

    def span(self, from_stop: str, to_stop: str) -> Span | None:
        i = self.index.get(from_stop)
        j = self.index.get(to_stop)
        if i is None or j is None or j <= i:
            return None
        return Span(seconds=self.offsets[j] - self.offsets[i], n_hops=j - i)


@dataclass(frozen=True)
class DayTimetable:
    """The timetable as it stands on one service day."""

    hops: Mapping[HopKey, int]
    patterns: Mapping[str, tuple[Pattern, ...]]
    codes: tuple[str, ...]

    def span(self, trip_id: str, from_stop: str, to_stop: str) -> Span | None:
        """What the timetable allows THIS trip between two stops it was seen at,
        measured along that trip's own stops.

        None unless every candidate pattern agrees. A path code can name more
        than one pattern: the realtime feed truncates some of them ('W..N' for
        'W..N30R'), and a service day can run two calendars at once. A
        disagreement between candidates is not a reading of the timetable, so the
        caller falls back to the day's per-hop median rather than picking one.
        """
        answers = {
            span
            for pattern in self._candidates(path_code(trip_id))
            if (span := pattern.span(from_stop, to_stop)) is not None
        }
        return answers.pop() if len(answers) == 1 else None

    def knows(self, trip_id: str) -> bool:
        """Whether this day's calendar runs anything with the trip's pattern."""
        return bool(self._candidates(path_code(trip_id)))

    def _candidates(self, code: str) -> tuple[Pattern, ...]:
        exact = self.patterns.get(code)
        if exact is not None:
            return exact
        out: list[Pattern] = []
        i = bisect.bisect_left(self.codes, code)
        while i < len(self.codes) and self.codes[i].startswith(code):
            out.extend(self.patterns[self.codes[i]])
            i += 1
        return tuple(out)


class Timetable:
    """The static timetable, sliced by service day.

    Two references, because neither alone covers the traffic. `DayTimetable.span`
    gives a move the scheduled time along the trip's OWN stops, which is the only
    honest way to read a train that ran express or bypassed a station: measured
    against
    the route's modal chain it is credited with the time of every stop it missed
    and reads as running fast, so the measure inverts exactly when it matters.
    `DayTimetable.hops` is the median over every trip serving a consecutive pair
    that day, and carries the fifth of realtime trip ids the static feed never
    names — dispatch origins that drift off the timetable, rerouted path codes,
    Staten Island. Measured over 92,606 live single hops: the trip's own pattern
    covers 90.1%, the day's median another 8.6%, and 1.2% get no scheduled
    time at all.

    Both references are ARRIVAL-TO-ARRIVAL, which is what training.trace can
    measure. NYCT schedules no dwell at 95.7% of stop_times rows so the two
    clocks are close, but departure-to-arrival still gets 4.9% of observed hops
    wrong by more than 10%.

    Sliced by service day because the weekend timetable is not the weekday one:
    pooling all three gets 26% of observed weekday hops wrong by more than 10%,
    the largest single error in the reference.
    """

    def __init__(
        self,
        hop_samples: Mapping[str, Mapping[HopKey, tuple[int, ...]]],
        patterns: Mapping[str, Mapping[str, Pattern]],
        calendar: _Calendar,
        version: FeedVersion,
    ) -> None:
        self._hop_samples = hop_samples
        self._patterns = patterns
        self._calendar = calendar
        self.version = version
        self._days: dict[frozenset[str], DayTimetable] = {}

    def covers(self, at: int, trip_id: str) -> bool:
        """Whether this feed claims to describe the day an observation fell on.
        False means the comparison is against a schedule that was not in force."""
        return self.version.covers(service_dates(at, trip_id)[0])

    def day_for(self, at: int, trip_id: str) -> DayTimetable:
        """The service day whose timetable describes this observation.

        The nearest candidate by scheduled origin wins unless its calendar never
        ran the trip's pattern. Holidays are why the second candidate is
        consulted at all: calendar_dates.txt can put Sunday service on a Monday,
        and a run whose origin drifted across midnight would otherwise be measured
        against a calendar it was never part of.
        """
        days = service_dates(at, trip_id)
        for day in days:
            table = self.day(day)
            if table.knows(trip_id):
                return table
        return self.day(days[0])

    def day(self, service_day: date) -> DayTimetable:
        """The timetable for one service day, built once per distinct set of
        services rather than per date — a month of weekdays is one object."""
        services = self._calendar.active(service_day)
        cached = self._days.get(services)
        if cached is None:
            cached = self._build(services)
            self._days[services] = cached
        return cached

    def _build(self, services: Iterable[str]) -> DayTimetable:
        pooled: dict[HopKey, list[int]] = defaultdict(list)
        patterns: dict[str, list[Pattern]] = defaultdict(list)
        for service in services:
            for key, samples in self._hop_samples.get(service, {}).items():
                pooled[key].extend(samples)
            for code, pattern in self._patterns.get(service, {}).items():
                patterns[code].append(pattern)
        by_code = {code: tuple(v) for code, v in patterns.items()}
        return DayTimetable(
            hops={k: int(statistics.median(v)) for k, v in pooled.items()},
            patterns=by_code,
            codes=tuple(sorted(by_code)),
        )


def _pattern(stops: list[str], hops: Mapping[int, list[int]]) -> Pattern | None:
    """Cumulative offsets along one stopping pattern, or None when any hop in it
    has no usable scheduled time — a pattern with a hole cannot be summed across,
    and half a pattern would silently give every span that crossed it the wrong
    scheduled time."""
    offsets = [0]
    for i in range(len(stops) - 1):
        samples = hops.get(i)
        if not samples:
            return None
        offsets.append(offsets[-1] + int(statistics.median(samples)))
    return Pattern(
        stops=tuple(stops),
        offsets=tuple(offsets),
        index={stop: i for i, stop in enumerate(stops)},
    )


def timetable(zf: zipfile.ZipFile) -> Timetable:
    """Parse the static feed into the scheduled reference the traversal measure
    needs: per-service hop medians and per-service stopping patterns, both
    arrival to arrival.

    Median rather than mean over the trips serving a hop: run times differ by
    time of day and a handful of padded late-night trips would drag a hop long.
    """
    with zf.open("trips.txt") as raw:
        meta = _trip_meta(io.TextIOWrapper(raw, encoding="utf-8-sig"))

    hop_samples: dict[str, dict[HopKey, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    pattern_stops: dict[tuple[str, str], list[str]] = {}
    pattern_hops: dict[tuple[str, str], dict[int, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )

    def flush(trip_id: str, stops: list[tuple[int, str, int | None]]) -> None:
        route, service = meta.get(trip_id, ("", ""))
        if not route or len(stops) < 2:
            return
        stops.sort(key=lambda s: s[0])
        code = (service, path_code(trip_id))
        pattern_stops.setdefault(code, [s[1] for s in stops])
        for i, ((_, frm, first), (_, to, second)) in enumerate(
            itertools.pairwise(stops)
        ):
            if first is None or second is None or second <= first or frm == to:
                continue
            pattern_hops[code][i].append(second - first)
            direction = direction_of(frm, trip_id)
            if direction is not None:
                hop_samples[service][(route, direction, frm, to)].append(second - first)

    with zf.open("stop_times.txt") as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
        header = next(reader)
        trip_col = header.index("trip_id")
        stop_col = header.index("stop_id")
        seq_col = header.index("stop_sequence")
        arr_col = header.index("arrival_time")

        current_trip: str | None = None
        buf: list[tuple[int, str, int | None]] = []
        for row in reader:
            trip_id = row[trip_col]
            if trip_id != current_trip:
                if current_trip is not None:
                    flush(current_trip, buf)
                current_trip = trip_id
                buf = []
            buf.append((int(row[seq_col]), row[stop_col], _gtfs_seconds(row[arr_col])))
        if current_trip is not None:
            flush(current_trip, buf)

    patterns: dict[str, dict[str, Pattern]] = defaultdict(dict)
    for (service, code), stops in pattern_stops.items():
        built = _pattern(stops, pattern_hops[(service, code)])
        if built is not None:
            patterns[service][code] = built

    return Timetable(
        hop_samples={
            service: {key: tuple(v) for key, v in keys.items()}
            for service, keys in hop_samples.items()
        },
        patterns=patterns,
        calendar=_read_calendar(zf),
        version=_read_version(zf),
    )


def load_timetable(url: str = GTFS_STATIC_URL) -> Timetable:
    """Fetch + parse in one call, mirroring load_successors."""
    data = fetch_gtfs_zip(url)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        return timetable(zf)


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
        meta = _trip_meta(io.TextIOWrapper(raw, encoding="utf-8-sig"))

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
                current_route = meta.get(trip_id, ("", ""))[0]
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
