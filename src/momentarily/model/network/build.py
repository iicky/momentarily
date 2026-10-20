"""Build a ``NetworkSnapshot`` from a GTFS zip and a stations snapshot.

Deterministic by construction: every collection is emitted in a fixed order,
``built_at`` comes from the inputs rather than the clock, and the gzip header
carries mtime 0, so the same inputs (and code) give the same output.
``content_digest`` (blake3 of the canonical uncompressed JSON) is the
identity to compare, since gzip bytes can differ between zlib builds.

A build that cannot satisfy the network's invariants fails with every problem
listed rather than emitting a partial network: each GTFS parent station maps
to exactly one stations-dataset row, every platform has a rider-facing label,
every trip resolves to exactly one pattern, and each pattern's segments chain
end to end.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal, cast
from zoneinfo import ZoneInfo

import blake3

from momentarily.gtfs.static import (
    Calendar,
    feed_rows,
    gtfs_seconds,
    read_calendar,
    read_version,
)
from momentarily.model.conventions import Provenance
from momentarily.model.network.contracts import (
    SCHEMA_ID,
    Ada,
    Complex,
    DayFeedMap,
    DayType,
    Direction,
    DirectionCode,
    Feed,
    IndexEntry,
    NetworkSnapshot,
    Pattern,
    PatternSegment,
    Platform,
    Route,
    ScheduledTrip,
    Segment,
    Service,
    ServiceDay,
    Station,
    StopTime,
    System,
    Weekday,
)
from momentarily.model.network.stations import StationsSnapshot, digest_of
from momentarily.provenance import code_provenance

DEFAULT_SYSTEM_ID = "nyct-subway"

# Part of every build_id. Bump whenever a builder change alters the output for
# the same inputs, so an old build_id never names different bytes.
BUILDER_VERSION = "1"

_WEEKDAYS: tuple[Weekday, ...] = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)
_DAY_TYPE_OF_WEEKDAYS: dict[frozenset[int], DayType] = {
    frozenset({0, 1, 2, 3, 4}): "weekday",
    frozenset({5}): "saturday",
    frozenset({6}): "sunday",
}
_ADA: dict[str, Ada] = {"0": "none", "1": "full", "2": "partial"}


PatternKey = tuple[str, int | None, DirectionCode | None, tuple[str, ...]]


class NetworkBuildError(ValueError):
    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__(
            f"network build failed ({len(problems)} problem(s)):\n  "
            + "\n  ".join(problems)
        )


@dataclass(frozen=True)
class FirstObserved:
    """The earliest UTC archive day collected against this feed (the traversal
    archive is keyed by UTC date), and the digest of the day->feed map it was
    read from."""

    utc_day: date
    source_digest: str


@dataclass(frozen=True)
class Inputs:
    gtfs: bytes
    stations: StationsSnapshot
    direction_overrides: dict[str, str]
    overrides_digest: str | None
    first_observed: FirstObserved | None


@dataclass(frozen=True)
class CodeStamp:
    sha: str
    dirty: bool | None


def current_code() -> CodeStamp:
    prov = code_provenance()
    return CodeStamp(sha=prov["code_sha"], dirty=prov["dirty"])


def build_id_for(
    system_id: str,
    feed_id: str,
    stations_digest: str,
    overrides_digest: str | None,
) -> str:
    """Data identity of a build: schema, builder version, system, and the exact
    data inputs (not the code commit)."""
    parts = [SCHEMA_ID, BUILDER_VERSION, system_id, feed_id, stations_digest]
    if overrides_digest is not None:
        parts.append(overrides_digest)
    return blake3.blake3("\x1f".join(parts).encode()).hexdigest()[:16]


def pattern_id_for(
    system_id: str,
    route_id: str,
    direction_id: int | None,
    code: str | None,
    stops: list[str],
) -> str:
    """``{route}-{code | direction_id | x}-{blake3[:12]}``; the hash covers the
    system, route, direction code, and ordered stops."""
    payload = "\x1f".join([system_id, route_id, code or "", *stops])
    digest = blake3.blake3(payload.encode()).hexdigest()[:12]
    tag = code or (str(direction_id) if direction_id is not None else "x")
    return f"{route_id}-{tag}-{digest}"


def _iso(day: date) -> str:
    return day.isoformat()


def _stamp_iso(stamp: str) -> str:
    return datetime.strptime(stamp, "%Y%m%d").date().isoformat()


def _opt(value: str | None) -> str | None:
    if value is None:
        return None
    text = value.strip()
    return text or None


def _day_type(active: frozenset[str], weekly: dict[str, frozenset[int]]) -> DayType:
    if not active:
        return "none"
    kinds: set[DayType | None] = {
        _DAY_TYPE_OF_WEEKDAYS.get(weekly.get(s, frozenset())) for s in active
    }
    kind = kinds.pop() if len(kinds) == 1 else None
    return "mixed" if kind is None else kind


def _service_window(calendar: Calendar) -> tuple[date, date] | None:
    stamps = [w.start for w in calendar.weekly] + [w.end for w in calendar.weekly]
    stamps += [*calendar.added, *calendar.removed]
    if not stamps:
        return None
    lo = datetime.strptime(min(stamps), "%Y%m%d").date()
    hi = datetime.strptime(max(stamps), "%Y%m%d").date()
    return lo, hi


def _services(calendar: Calendar) -> list[Service]:
    weekly = {w.service_id: w for w in calendar.weekly}
    added: dict[str, list[str]] = defaultdict(list)
    removed: dict[str, list[str]] = defaultdict(list)
    for stamp, ids in calendar.added.items():
        for sid in ids:
            added[sid].append(_stamp_iso(stamp))
    for stamp, ids in calendar.removed.items():
        for sid in ids:
            removed[sid].append(_stamp_iso(stamp))
    out: list[Service] = []
    for sid in sorted(set(weekly) | set(added) | set(removed)):
        w = weekly.get(sid)
        end = (
            None
            if w is None
            else _iso(datetime.strptime(w.end, "%Y%m%d").date() + timedelta(days=1))
        )
        out.append(
            Service(
                service_id=sid,
                weekdays=[]
                if w is None
                else [_WEEKDAYS[i] for i in sorted(w.weekdays)],
                start=None if w is None else _stamp_iso(w.start),
                end=end,
                added_dates=sorted(added.get(sid, [])),
                removed_dates=sorted(removed.get(sid, [])),
            )
        )
    return out


def service_days(calendar: Calendar, first: date, end: date) -> list[ServiceDay]:
    """One ServiceDay per date in ``[first, end)``."""
    weekly = {w.service_id: w.weekdays for w in calendar.weekly}
    exceptions = set(calendar.added) | set(calendar.removed)
    days: list[ServiceDay] = []
    day = first
    while day < end:
        active = calendar.active(day)
        days.append(
            ServiceDay(
                date=_iso(day),
                active_service_ids=sorted(active),
                day_type=_day_type(active, weekly),
                is_exception=day.strftime("%Y%m%d") in exceptions,
            )
        )
        day += timedelta(days=1)
    return days


@dataclass
class _TripRow:
    trip_id: str
    route_id: str
    service_id: str
    headsign: str | None
    shape_id: str | None
    direction_id: int | None


def build_network(
    inputs: Inputs,
    *,
    system_id: str = DEFAULT_SYSTEM_ID,
    stamp: CodeStamp,
) -> NetworkSnapshot:
    problems: list[str] = []
    feed_id = hashlib.sha256(inputs.gtfs).hexdigest()
    zf = zipfile.ZipFile(io.BytesIO(inputs.gtfs))

    # --- system / feed ---------------------------------------------------
    agencies = feed_rows(zf, "agency.txt")
    if not agencies:
        raise NetworkBuildError(["agency.txt is missing or empty"])
    agency = agencies[0]
    timezone = agency["agency_timezone"].strip()
    system = System(
        system_id=system_id,
        timezone=timezone,
        agency_id=(agency.get("agency_id") or "").strip(),
        agency_name=agency["agency_name"].strip(),
    )
    version = read_version(zf)
    feed = Feed(
        feed_id=feed_id,
        feed_version=None if version.version == "unknown" else version.version,
        valid_from=None if version.start is None else _iso(version.start),
        valid_to=None if version.end is None else _iso(version.end + timedelta(days=1)),
    )

    # --- routes ----------------------------------------------------------
    routes = [
        Route(
            route_id=r["route_id"],
            short_name=r.get("route_short_name", "").strip(),
            long_name=r.get("route_long_name", "").strip(),
            route_type=int(r["route_type"]),
            color=_opt(r.get("route_color")),
            text_color=_opt(r.get("route_text_color")),
            sort_order=int(r["route_sort_order"])
            if _opt(r.get("route_sort_order"))
            else None,
        )
        for r in sorted(feed_rows(zf, "routes.txt"), key=lambda r: r["route_id"])
    ]
    if not routes:
        problems.append("routes.txt is missing or empty")
    route_ids = {r.route_id for r in routes}

    # --- stations / platforms / complexes --------------------------------
    stop_rows = feed_rows(zf, "stops.txt")
    parents = {
        s["stop_id"]: s for s in stop_rows if (s.get("location_type") or "") == "1"
    }
    by_gtfs: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(inputs.stations.rows):
        by_gtfs[row.gtfs_stop_id].append(i)
    unmatched_gtfs = sorted(s for s in parents if s not in by_gtfs)
    if unmatched_gtfs:
        problems.append(
            "GTFS parent stations with no stations-dataset row: "
            + ", ".join(unmatched_gtfs)
        )
    dupes = sorted(s for s in parents if len(by_gtfs.get(s, [])) > 1)
    if dupes:
        problems.append(
            "GTFS parent stations with more than one stations-dataset row: "
            + ", ".join(dupes)
        )
    unmatched_rows = sorted(g for g in by_gtfs if g not in parents)
    if unmatched_rows:
        problems.append(
            "stations-dataset rows with no GTFS parent station: "
            + ", ".join(unmatched_rows)
        )

    stations: list[Station] = []
    members: dict[str, list[str]] = defaultdict(list)
    complex_names: dict[str, list[str]] = defaultdict(list)
    for sid in sorted(parents):
        s = parents[sid]
        hits = by_gtfs.get(sid, [])
        row = inputs.stations.rows[hits[0]] if len(hits) == 1 else None
        ada: Ada | None = None
        if row is not None and row.ada is not None:
            if row.ada not in _ADA:
                problems.append(f"station {sid}: unrecognised ada value {row.ada!r}")
            ada = _ADA.get(row.ada)
        stations.append(
            Station(
                station_id=sid,
                name=s["stop_name"].strip(),
                lat=float(s["stop_lat"]),
                lon=float(s["stop_lon"]),
                complex_id=None if row is None else row.complex_id,
                borough=None if row is None else row.borough,
                ada=ada,
            )
        )
        if row is not None and row.complex_id is not None:
            members[row.complex_id].append(sid)
            complex_names[row.complex_id].append(row.stop_name)
    complexes = [
        Complex(
            complex_id=cid,
            name=" / ".join(dict.fromkeys(complex_names[cid])),
            station_ids=members[cid],
        )
        for cid in sorted(members, key=lambda c: (len(c), c))
    ]
    station_rows = {
        sid: inputs.stations.rows[hits[0]]
        for sid, hits in by_gtfs.items()
        if len(hits) == 1
    }

    platforms: list[Platform] = []
    used_overrides: set[str] = set()
    for s in sorted(stop_rows, key=lambda s: s["stop_id"]):
        if (s.get("location_type") or "0") not in ("0", ""):
            continue
        stop_id = s["stop_id"]
        parent = _opt(s.get("parent_station"))
        if parent is None or parent not in parents:
            problems.append(f"platform {stop_id} has no parent station")
            continue
        code: DirectionCode | None = None
        if stop_id == f"{parent}N":
            code = "N"
        elif stop_id == f"{parent}S":
            code = "S"
        row = station_rows.get(parent)
        label: str | None = None
        if row is not None and code is not None:
            label = (
                row.north_direction_label if code == "N" else row.south_direction_label
            )
        if label is None and stop_id in inputs.direction_overrides:
            label = inputs.direction_overrides[stop_id]
            used_overrides.add(stop_id)
        if label is None and row is not None:
            problems.append(f"platform {stop_id}: no direction label")
        platforms.append(
            Platform(
                stop_id=stop_id,
                station_id=parent,
                name=s["stop_name"].strip(),
                direction_code=code,
                direction_label=label,
            )
        )
    stale = sorted(set(inputs.direction_overrides) - used_overrides)
    if stale:
        problems.append(
            "direction overrides that fill no gap (stale): " + ", ".join(stale)
        )
    platform_code: dict[str, DirectionCode | None] = {
        p.stop_id: p.direction_code for p in platforms
    }

    # --- calendar --------------------------------------------------------
    calendar = read_calendar(zf)
    services = _services(calendar)
    service_ids = {s.service_id for s in services}

    # --- trips + stop times ----------------------------------------------
    trips_raw: dict[str, _TripRow] = {}
    trip_rows = feed_rows(zf, "trips.txt")
    if not trip_rows:
        problems.append("trips.txt is missing or empty")
    for r in trip_rows:
        if r["trip_id"] in trips_raw:
            problems.append(f"duplicate trip_id {r['trip_id']}")
            continue
        raw_dir = (r.get("direction_id") or "").strip()
        direction_id: int | None = None
        if raw_dir in ("0", "1"):
            direction_id = int(raw_dir)
        elif raw_dir:
            problems.append(f"trip {r['trip_id']}: invalid direction_id {raw_dir!r}")
        if r["route_id"] not in route_ids:
            problems.append(f"trip {r['trip_id']}: unknown route {r['route_id']}")
        if r["service_id"] not in service_ids:
            problems.append(f"trip {r['trip_id']}: unknown service {r['service_id']}")
        trips_raw[r["trip_id"]] = _TripRow(
            trip_id=r["trip_id"],
            route_id=r["route_id"],
            service_id=r["service_id"],
            headsign=_opt(r.get("trip_headsign")),
            shape_id=_opt(r.get("shape_id")),
            direction_id=direction_id,
        )

    times: dict[str, list[tuple[int, StopTime]]] = defaultdict(list)
    time_rows = feed_rows(zf, "stop_times.txt")
    if not time_rows:
        problems.append("stop_times.txt is missing or empty")
    for r in time_rows:
        tid = r["trip_id"]
        arr = gtfs_seconds(r.get("arrival_time") or "")
        dep = gtfs_seconds(r.get("departure_time") or "")
        if arr is None or dep is None:
            problems.append(f"trip {tid}: stop {r['stop_id']} has no scheduled time")
            continue
        times[tid].append((int(r["stop_sequence"]), (r["stop_id"], arr, dep)))

    pattern_trips: dict[PatternKey, list[str]]
    pattern_trips = defaultdict(list)
    trip_key: dict[str, PatternKey] = {}
    trip_stop_times: dict[str, list[StopTime]] = {}
    for tid in sorted(trips_raw):
        t = trips_raw[tid]
        seq = sorted(times.get(tid, []), key=lambda x: x[0])
        if len({s for s, _ in seq}) != len(seq):
            problems.append(f"trip {tid}: duplicate stop_sequence")
            continue
        stop_times = [st for _, st in seq]
        stops = [st[0] for st in stop_times]
        missing = sorted({s for s in stops if s not in platform_code})
        if missing:
            problems.append(f"trip {tid}: stops not in stops.txt: {', '.join(missing)}")
            continue
        if len(stops) < 2:
            problems.append(f"trip {tid}: fewer than two stops; resolves to no pattern")
            continue
        codes: set[DirectionCode | None] = {platform_code[s] for s in stops}
        dcode = codes.pop() if len(codes) == 1 else None
        key = (t.route_id, t.direction_id, dcode, tuple(stops))
        pattern_trips[key].append(tid)
        trip_key[tid] = key
        trip_stop_times[tid] = stop_times
    for tid in sorted(set(times) - set(trips_raw)):
        problems.append(f"stop_times reference unknown trip {tid}")

    patterns: list[Pattern] = []
    pattern_of_key: dict[PatternKey, str] = {}
    seen_ids: dict[str, PatternKey] = {}
    for key in sorted(pattern_trips, key=lambda k: (k[0], str(k[1]), str(k[2]), k[3])):
        route_id, direction_id, code, stops = key
        pid = pattern_id_for(system_id, route_id, direction_id, code, list(stops))
        if pid in seen_ids:
            problems.append(
                f"pattern id collision {pid}: trips differ only by direction_id"
            )
            continue
        seen_ids[pid] = key
        pattern_of_key[key] = pid
        tids = pattern_trips[key]
        heads = Counter(trips_raw[t].headsign for t in tids)
        headsign = min(heads, key=lambda h: (-heads[h], h or ""))
        patterns.append(
            Pattern(
                pattern_id=pid,
                route_id=route_id,
                direction=Direction(
                    direction_id=_dir(direction_id), code=code, label=None
                ),
                headsign=headsign,
                stops=list(stops),
                shape_ids=sorted({s for t in tids if (s := trips_raw[t].shape_id)}),
                n_trips=len(tids),
            )
        )
    patterns.sort(key=lambda p: p.pattern_id)

    # --- segments --------------------------------------------------------
    seg_routes: dict[str, set[str]] = defaultdict(set)
    seg_patterns: dict[str, set[str]] = defaultdict(set)
    pattern_segments: list[PatternSegment] = []
    for p in patterns:
        for i, (a, b) in enumerate(zip(p.stops, p.stops[1:], strict=False)):
            sid = f"{a}>{b}"
            seg_routes[sid].add(p.route_id)
            seg_patterns[sid].add(p.pattern_id)
            pattern_segments.append(
                PatternSegment(pattern_id=p.pattern_id, seq=i, segment_id=sid)
            )
    segments = [
        Segment(
            segment_id=sid,
            from_stop=sid.split(">")[0],
            to_stop=sid.split(">")[1],
            route_ids=sorted(seg_routes[sid]),
            pattern_ids=sorted(seg_patterns[sid]),
        )
        for sid in sorted(seg_routes)
    ]
    problems += check_chains(patterns, pattern_segments, segments)

    # --- trips -----------------------------------------------------------
    trips: list[ScheduledTrip] = []
    for tid in sorted(trip_key):
        t = trips_raw[tid]
        key = trip_key[tid]
        pid = pattern_of_key.get(key)
        if pid is None:
            problems.append(f"trip {tid}: resolves to no pattern")
            continue
        trips.append(
            ScheduledTrip(
                trip_id=tid,
                route_id=t.route_id,
                pattern_id=pid,
                service_id=t.service_id,
                headsign=t.headsign,
                shape_id=t.shape_id,
                direction=Direction(
                    direction_id=_dir(t.direction_id), code=key[2], label=None
                ),
                stop_times=trip_stop_times[tid],
            )
        )

    # --- service days ----------------------------------------------------
    if version.start is not None and version.end is not None:
        window = (version.start, version.end)
    else:
        window = _service_window(calendar)
    days = (
        []
        if window is None
        else service_days(calendar, window[0], window[1] + timedelta(days=1))
    )

    if problems:
        raise NetworkBuildError(problems)

    # --- provenance ------------------------------------------------------
    stations_digest = inputs.stations.digest
    input_digests = {"gtfs": feed_id, "stations": stations_digest}
    if inputs.overrides_digest is not None:
        input_digests["direction_overrides"] = inputs.overrides_digest
    # The stations data is retroactive when it was captured on or after the
    # first day it is applied to: the first UTC archive day observed on this
    # feed when known (compared in UTC), else the feed's first service day
    # (compared in the system timezone). A same-day capture counts: part of
    # that day ran before the snapshot existed.
    period_start: date | None = None if window is None else window[0]
    captured_at = inputs.stations.retrieved_at
    observed = inputs.first_observed
    if observed is not None:
        input_digests["day_feed_map"] = observed.source_digest
        captured_utc = datetime.fromtimestamp(captured_at, UTC).date()
        retroactive = captured_utc >= observed.utc_day
    elif period_start is not None:
        captured_local = datetime.fromtimestamp(captured_at, ZoneInfo(timezone)).date()
        retroactive = captured_local >= period_start
    else:
        retroactive = True
    provenance = Provenance(
        code_sha=stamp.sha,
        code_dirty=stamp.dirty,
        schema=SCHEMA_ID,
        inputs=input_digests,
        captured_at={"stations": captured_at},
        period_start=None if period_start is None else _iso(period_start),
        observed_from_utc_day=None if observed is None else _iso(observed.utc_day),
        applied_retroactively=["stations"] if retroactive else [],
        built_at=captured_at,
    )
    return NetworkSnapshot(
        build_id=build_id_for(
            system_id, feed_id, stations_digest, inputs.overrides_digest
        ),
        provenance=provenance,
        system=system,
        feed=feed,
        routes=routes,
        stations=stations,
        complexes=complexes,
        platforms=platforms,
        patterns=patterns,
        segments=segments,
        pattern_segments=pattern_segments,
        services=services,
        service_days=days,
        trips=trips,
    )


def _dir(direction_id: int | None) -> Literal[0, 1] | None:
    if direction_id == 0:
        return 0
    if direction_id == 1:
        return 1
    return None


def check_chains(
    patterns: list[Pattern],
    pattern_segments: list[PatternSegment],
    segments: list[Segment],
) -> list[str]:
    """Each pattern's segments, in seq order, must walk its stops end to end."""
    by_id = {s.segment_id: s for s in segments}
    hops: dict[str, list[PatternSegment]] = defaultdict(list)
    for ps in pattern_segments:
        hops[ps.pattern_id].append(ps)
    problems: list[str] = []
    for p in patterns:
        chain = sorted(hops.get(p.pattern_id, []), key=lambda ps: ps.seq)
        if [ps.seq for ps in chain] != list(range(len(p.stops) - 1)):
            problems.append(f"pattern {p.pattern_id}: segment seq is not 0..n-2")
            continue
        walked = [p.stops[0]]
        for ps in chain:
            seg = by_id.get(ps.segment_id)
            if seg is None or seg.from_stop != walked[-1]:
                problems.append(
                    f"pattern {p.pattern_id}: segments do not chain at seq {ps.seq}"
                )
                break
            walked.append(seg.to_stop)
        else:
            if walked != p.stops:
                problems.append(
                    f"pattern {p.pattern_id}: segments do not cover its stops"
                )
    return problems


def canonical_json(snapshot: NetworkSnapshot) -> bytes:
    """The one uncompressed serialisation: sorted keys, compact, UTF-8."""
    return json.dumps(
        snapshot.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()


def content_digest(snapshot: NetworkSnapshot) -> str:
    """blake3 of ``canonical_json``: stable across zlib builds, unlike the
    gzip bytes."""
    return digest_of(canonical_json(snapshot))


def network_digest(snapshot: NetworkSnapshot) -> str:
    """blake3 of the canonical JSON without provenance: the network data a
    build_id names. Provenance is metadata about the build and may change
    under the same build_id; this may not without a BUILDER_VERSION bump."""
    data = json.dumps(
        snapshot.model_dump(mode="json", exclude={"provenance"}),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode()
    return digest_of(data)


def serialize(snapshot: NetworkSnapshot) -> bytes:
    """Gzip of ``canonical_json`` with mtime 0 and no filename."""
    buf = io.BytesIO()
    with gzip.GzipFile(
        filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9
    ) as gz:
        gz.write(canonical_json(snapshot))
    return buf.getvalue()


def index_entry(snapshot: NetworkSnapshot) -> IndexEntry:
    return IndexEntry(
        build_id=snapshot.build_id,
        content_digest=content_digest(snapshot),
        feed_version=snapshot.feed.feed_version,
        valid_from=snapshot.feed.valid_from,
        valid_to=snapshot.feed.valid_to,
        inputs=snapshot.provenance.inputs,
        applied_retroactively=snapshot.provenance.applied_retroactively,
    )


def first_observed(day_map: bytes, gtfs: bytes) -> FirstObserved | None:
    """The earliest UTC day in a day->feed map collected against this feed:
    the verified feed_id, or, for an unverified day, its inferred feed."""
    feed_id = hashlib.sha256(gtfs).hexdigest()
    parsed = DayFeedMap.model_validate_json(day_map)
    days = [
        day
        for day, entry in parsed.days.items()
        if (entry.feed_id or entry.inferred_feed_id) == feed_id
    ]
    if not days:
        return None
    return FirstObserved(
        utc_day=date.fromisoformat(min(days)), source_digest=digest_of(day_map)
    )


def load_overrides(data: bytes | None) -> tuple[dict[str, str], str | None]:
    """Optional stop_id -> direction label fills for platforms the stations
    dataset leaves unlabelled. Expected to be empty for the NYCT feed."""
    if data is None:
        return {}, None
    raw: Any = json.loads(data)
    if not isinstance(raw, dict):
        msg = "direction overrides must be a JSON object of stop_id -> label"
        raise ValueError(msg)
    items = cast("dict[Any, Any]", raw)
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in items.items()):
        msg = "direction overrides must map stop_id strings to label strings"
        raise ValueError(msg)
    return {str(k): str(v) for k, v in items.items()}, digest_of(data)
