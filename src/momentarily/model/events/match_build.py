"""Building TripMatch and SupplyObservation files from layer-2 files.

``read_window`` reads one lineage's layer-2 records of an input window:

- raw: hourly ``vehicle``, ``feed_header`` and ``trip_update`` files from the
  recorder cutover on. A fetch is good when its header is at most
  ``MAX_HEADER_AGE_S`` old (coverage's rule); rows of other fetches are
  dropped, because they carry no fresh position (the 2026-10-21 freeze);
- legacy: hourly ``vehicle`` files and the daily ``legacy_fetch_outcome``
  files before the cutover. A row's feed is the ``RouteFeed`` table's for
  its trip-id route field, a poll is a trace object, and a feed's poll is
  good when the object lists the feed fresh.

Every hour of the window needs its ``vehicle`` file (and, raw, its
``feed_header`` file). A missing one is a missing input unless ``partial``,
which builds from what there is and reports the hours left out.

``build_trip_match`` files the TripMatch records of one service day: the
window is that day's (§6: local midnight − 2 h to the next day's 08:00),
every run of the window is matched, and the runs whose service date is the
day are written. ``build_supply`` files the SupplyObservations of one UTC
day, from runs and matches of the day ± ``SUPPLY_MARGIN_S``.
"""

# pyarrow-stubs leave some generics unknown: silence that noise in this file.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import pyarrow as pa
import pyarrow.parquet as pq

from momentarily.gtfs.static import base_route
from momentarily.model.events.contracts import Lineage, SupplyObservation, TripMatch
from momentarily.model.events.coverage import MAX_HEADER_AGE_S, ROUTE_FEED_INPUT
from momentarily.model.events.files import (
    EventFileError,
    WriteOutcome,
    event_file_metadata,
    resolve_write,
)
from momentarily.model.events.keys import ServiceDayKey, UtcDayKey
from momentarily.model.events.match_index import (
    IndexInputs,
    MatchIndex,
    MatchIndexBuildError,
    build_match_index,
    read_day_map,
    read_network,
    read_network_index,
    select_build,
)
from momentarily.model.events.matching import (
    MATCH_VERSION,
    Matcher,
    service_day_window,
    trip_matches,
)
from momentarily.model.events.route_feed import RouteFeedTable, load_route_feed
from momentarily.model.events.runs import (
    PatternBook,
    Row,
    Run,
    build_runs,
    is_legacy_placeholder,
    is_raw_placeholder,
)
from momentarily.model.events.supply import supply_observations
from momentarily.model.events.trip_ids import parse_realtime
from momentarily.model.network.contracts import (
    DayFeedMap,
    NetworkIndex,
    NetworkSnapshot,
)
from momentarily.model.observations.contracts import GTFSRT_SOURCES, GtfsRtSource
from momentarily.model.observations.derive.run import InputProblem
from momentarily.model.observations.derive.storage import Storage
from momentarily.model.observations.keys import (
    RECORDER_CUTOVER,
    DerivedDayKey,
    DerivedHourKey,
    utc_day,
)
from momentarily.model.observations.parquet import file_lineage, read_file_metadata

TZ = "America/New_York"
SUPPLY_MARGIN_S = 6 * 3600
_HOUR = 3600
_DAY = 86400
_STOPPED_AT = 1

Lost = Callable[[str, int], None]
"""Reports a missing layer-2 file of an hour."""


# --- match indexes ----------------------------------------------------------------


class Indexes:
    """Match indexes built on demand from a layer-1 output directory, one per
    service date (the build ``select_build`` chooses, or ``build_id``)."""

    def __init__(self, network: Path, build_id: str | None = None) -> None:
        self.root = network
        self.build_id = build_id
        self.index: NetworkIndex = read_network_index(network)
        self.day_map: DayFeedMap | None = None
        self.map_digest: str | None = None
        if build_id is None:
            self.day_map, self.map_digest = read_day_map(network / "day_feed_map.json")
        self.route_feed, self.route_feed_digest = load_route_feed()
        self._nets: dict[str, NetworkSnapshot] = {}
        self._ix: dict[str, MatchIndex | None] = {}
        self.problems: dict[str, str] = {}
        self.used: dict[str, tuple[str, int]] = {}
        """Index key -> (network content digest, network built_at) of every
        index read."""

    def _snapshot(self, feed_id: str) -> NetworkSnapshot:
        net = self._nets.get(feed_id)
        if net is None:
            net = self._nets[feed_id] = read_network(self.root, feed_id, self.index)
        return net

    def __call__(self, service_date: str) -> MatchIndex | None:
        if service_date in self._ix:
            ix = self._ix[service_date]
        else:
            try:
                choice = select_build(
                    service_date,
                    self.index,
                    self.day_map,
                    self.build_id,
                    snapshot=self._snapshot,
                )
                net = self._snapshot(choice.feed_id)
                doc = build_match_index(
                    service_date,
                    IndexInputs(
                        network=net,
                        network_digest=self.index.feeds[choice.feed_id].content_digest,
                        choice=choice,
                        day_map_digest=self.map_digest,
                        route_feed=self.route_feed,
                        route_feed_digest=self.route_feed_digest,
                        code_sha="",
                        code_dirty=None,
                    ),
                )
                ix = MatchIndex(doc)
            except MatchIndexBuildError as err:
                self.problems[service_date] = str(err)
                ix = None
            self._ix[service_date] = ix
        if ix is not None:
            meta = ix.metadata
            self.used[meta.file_key] = (
                meta.provenance.inputs["network"],
                meta.provenance.built_at,
            )
        return ix


# --- layer-2 inputs ---------------------------------------------------------------


@dataclass
class WindowInputs:
    lineage: Lineage
    window: tuple[int, int]
    rows: list[Row] = field(default_factory=list[Row])
    placeholders: list[tuple[GtfsRtSource, str, int]] = field(
        default_factory=list[tuple[GtfsRtSource, str, int]]
    )
    polls: dict[GtfsRtSource, list[tuple[int, str]]] = field(
        default_factory=dict[GtfsRtSource, list[tuple[int, str]]]
    )
    """Good polls per feed: (observed_at, poll id)."""
    listed: dict[str, Counter[str]] | None = None
    read: dict[str, tuple[str, int]] = field(default_factory=dict[str, tuple[str, int]])
    missing: list[InputProblem] = field(default_factory=list[InputProblem])
    inconsistent: list[InputProblem] = field(default_factory=list[InputProblem])
    hours_left_out: list[int] = field(default_factory=list[int])
    rows_without_stop: int = 0
    unattributed_rows: int = 0
    rows_of_bad_fetches: int = 0

    @property
    def last_poll(self) -> dict[GtfsRtSource, int]:
        return {f: max(at for at, _ in ps) for f, ps in self.polls.items() if ps}


def lineage_window(lineage: Lineage, start: int, end: int) -> tuple[int, int]:
    """The part of ``[start, end)`` a lineage holds, by the recorder cutover."""
    if lineage == "raw":
        return max(start, RECORDER_CUTOVER), end
    return start, min(end, RECORDER_CUTOVER)


def _table(
    storage: Storage,
    key: str,
    lineage: Lineage,
    columns: list[str],
    inputs: WindowInputs,
) -> pa.Table | None:
    try:
        data = storage.get(key)
    except KeyError:
        return None
    try:
        meta = read_file_metadata(data)
        table = pq.read_table(pa.BufferReader(data), columns=columns)
    except (pa.ArrowException, KeyError, ValueError) as err:
        inputs.inconsistent.append(
            InputProblem(key, f"not a readable derived file: {err}")
        )
        return None
    if file_lineage(meta) != lineage:
        inputs.inconsistent.append(InputProblem(key, f"not a {lineage}-lineage file"))
        return None
    inputs.read[key] = (hashlib.sha256(data).hexdigest(), meta.provenance.built_at)
    return table


def _col(table: pa.Table, name: str) -> list[Any]:
    return table.column(name).to_pylist()


def _field(table: pa.Table, column: str, name: str) -> list[Any]:
    return (
        cast(pa.StructArray, table.column(column).combine_chunks())
        .field(name)
        .to_pylist()
    )


_VEHICLE = [
    "lineage",
    "fetch_id",
    "source",
    "observed_at",
    "entity_index",
    "trip",
    "stop_id",
    "current_status",
    "vehicle_timestamp",
]


def read_window(
    storage: Storage,
    lineage: Lineage,
    start: int,
    end: int,
    route_feed: RouteFeedTable,
    route_feed_digest: str,
    *,
    partial: bool = False,
    trip_updates: bool = False,
) -> WindowInputs:
    window = lineage_window(lineage, start, end)
    inputs = WindowInputs(lineage, window)
    if window[0] >= window[1]:
        return inputs
    hours = range(window[0] - window[0] % _HOUR, window[1], _HOUR)

    def lost(key: str, hour: int) -> None:
        if any(p.key == key for p in inputs.inconsistent):
            return  # present but unusable: already an inconsistent input
        if partial:
            inputs.hours_left_out.append(hour)
        else:
            inputs.missing.append(InputProblem(key, "no layer-2 file"))

    if lineage == "raw":
        _read_raw(storage, inputs, hours, lost, trip_updates)
    else:
        _read_legacy(storage, inputs, hours, lost, route_feed)
        inputs.read[ROUTE_FEED_INPUT] = (
            route_feed_digest,
            route_feed.source.observed_to,
        )
    for ps in inputs.polls.values():
        ps.sort()
    inputs.hours_left_out = sorted(set(inputs.hours_left_out))
    return inputs


def _read_raw(
    storage: Storage, inputs: WindowInputs, hours: range, lost: Lost, trip_updates: bool
) -> None:
    lo, hi = inputs.window
    good: dict[str, int] = {}
    usable: set[int] = set()
    report = lost
    for h in hours:
        key = DerivedHourKey("feed_header", h).key
        t = _table(
            storage,
            key,
            "raw",
            ["fetch_id", "source", "observed_at", "header_timestamp"],
            inputs,
        )
        if t is None:
            report(key, h)
            continue
        usable.add(h)
        for fid, src, at, ht in zip(
            _col(t, "fetch_id"),
            _col(t, "source"),
            _col(t, "observed_at"),
            _col(t, "header_timestamp"),
            strict=True,
        ):
            at = cast(int, at)
            if src not in GTFSRT_SOURCES or not lo <= at < hi:
                continue
            if ht is not None and at - cast(int, ht) <= MAX_HEADER_AGE_S:
                good[cast(str, fid)] = at
                inputs.polls.setdefault(cast(GtfsRtSource, src), []).append(
                    (at, cast(str, fid))
                )
    for h in hours:
        if h not in usable:
            continue
        key = DerivedHourKey("vehicle", h).key
        t = _table(storage, key, "raw", _VEHICLE, inputs)
        if t is None:
            report(key, h)
            continue
        cols: list[list[Any]] = [
            _col(t, "fetch_id"),
            _col(t, "source"),
            _col(t, "observed_at"),
            _col(t, "entity_index"),
            _field(t, "trip", "trip_id"),
            _field(t, "trip", "route_id"),
            _field(t, "trip", "start_date"),
            _field(t, "trip", "nyct_train_id"),
            _field(t, "trip", "nyct_is_assigned"),
            _col(t, "stop_id"),
            _col(t, "current_status"),
            _col(t, "vehicle_timestamp"),
        ]
        for (
            fid,
            src,
            at,
            ent,
            tid,
            route,
            sdate,
            train,
            assigned,
            stop,
            status,
            vts,
        ) in zip(*cols, strict=True):
            at = cast(int, at)
            if not lo <= at < hi or tid is None or src not in GTFSRT_SOURCES:
                continue
            feed = cast(GtfsRtSource, src)
            if fid not in good:
                inputs.rows_of_bad_fetches += 1
                continue
            if is_raw_placeholder(
                cast(bool | None, assigned),
                cast(int | None, status),
                cast(int | None, vts),
                at,
            ):
                inputs.placeholders.append((feed, cast(str, tid), at))
                continue
            if stop is None:
                inputs.rows_without_stop += 1
                continue
            inputs.rows.append(
                Row(
                    at,
                    cast(str, fid),
                    cast(int, ent),
                    feed,
                    cast(str, tid),
                    cast(str, stop),
                    cast(int | None, status),
                    cast(int | None, vts),
                    cast(str | None, route),
                    cast(str | None, sdate),
                    cast(str | None, train),
                )
            )
    if not trip_updates:
        return
    inputs.listed = defaultdict(Counter)
    for h in hours:
        if h not in usable:
            continue
        key = DerivedHourKey("trip_update", h).key
        t = _table(storage, key, "raw", ["fetch_id", "observed_at", "trip"], inputs)
        if t is None:
            report(key, h)
            continue
        for fid, tid, route in zip(
            _col(t, "fetch_id"),
            _field(t, "trip", "trip_id"),
            _field(t, "trip", "route_id"),
            strict=True,
        ):
            if fid not in good:
                continue
            name = route if isinstance(route, str) else None
            if name is None and isinstance(tid, str):
                p = parse_realtime(tid)
                name = None if p is None else p.route_field
            if name is not None:
                inputs.listed[cast(str, fid)][base_route(name)] += 1


def _read_legacy(
    storage: Storage,
    inputs: WindowInputs,
    hours: range,
    lost: Lost,
    route_feed: RouteFeedTable,
) -> None:
    lo, hi = inputs.window
    report = lost
    fresh: dict[tuple[str, GtfsRtSource], bool] = {}
    for d in sorted({utc_day(h) for h in hours}):
        key = DerivedDayKey("legacy_fetch_outcome", d).key
        t = _table(
            storage,
            key,
            "legacy",
            ["lineage", "source", "observed_at", "succeeded"],
            inputs,
        )
        if t is None:
            report(key, hours[0] if d == utc_day(hours[0]) else hours[-1])
            continue
        keys = _field(t, "lineage", "source_key")
        for sk, src, at, ok in zip(
            keys,
            _col(t, "source"),
            _col(t, "observed_at"),
            _col(t, "succeeded"),
            strict=True,
        ):
            at = cast(int, at)
            if (
                not str(sk).startswith("archive/trace/")
                or src not in GTFSRT_SOURCES
                or not lo <= at < hi
            ):
                continue
            feed = cast(GtfsRtSource, src)
            fresh[(cast(str, sk), feed)] = bool(ok)
            if ok:
                inputs.polls.setdefault(feed, []).append((at, cast(str, sk)))
    feeds: dict[str, GtfsRtSource | None] = {}
    for h in hours:
        key = DerivedHourKey("vehicle", h).key
        t = _table(
            storage,
            key,
            "legacy",
            [
                "lineage",
                "observed_at",
                "trip",
                "stop_id",
                "current_status",
                "vehicle_timestamp",
            ],
            inputs,
        )
        if t is None:
            report(key, h)
            continue
        lin = cast(
            pa.StructArray,
            cast(pa.StructArray, t.column("lineage").combine_chunks()).field("legacy"),
        )
        cols: list[list[Any]] = [
            lin.field("source_key").to_pylist(),
            lin.field("row_index").to_pylist(),
            _col(t, "observed_at"),
            _field(t, "trip", "trip_id"),
            _col(t, "stop_id"),
            _col(t, "current_status"),
            _col(t, "vehicle_timestamp"),
        ]
        for sk, ri, at, tid, stop, status, vts in zip(*cols, strict=True):
            at = cast(int, at)
            if not lo <= at < hi or tid is None:
                continue
            tid = cast(str, tid)
            if tid not in feeds:
                p = parse_realtime(tid)
                feeds[tid] = None if p is None else route_feed.feed_of(p.route_field)
            feed = feeds[tid]
            if feed is None:
                inputs.unattributed_rows += 1
                continue
            if not fresh.get((cast(str, sk), feed), False):
                inputs.rows_of_bad_fetches += 1
                continue
            stopped = status == _STOPPED_AT
            if is_legacy_placeholder(feed, stopped, cast(int | None, vts), at):
                inputs.placeholders.append((feed, tid, at))
                continue
            if stop is None:
                inputs.rows_without_stop += 1
                continue
            inputs.rows.append(
                Row(
                    at,
                    cast(str, sk),
                    cast(int, ri or 0),
                    feed,
                    tid,
                    cast(str, stop),
                    1 if stopped else None,
                    cast(int | None, vts),
                    None,
                    None,
                    None,
                )
            )


# --- builds -----------------------------------------------------------------------


@dataclass
class MatchReport:
    """What a build read and made."""

    record: Literal["trip_match", "supply"]
    lineage: Lineage
    period: str
    key: str
    window: tuple[int, int]
    builder_version: int = MATCH_VERSION
    runs: int = 0
    matched: int = 0
    tiers: dict[str, int] = field(default_factory=dict[str, int])
    miss_classes: dict[str, int] = field(default_factory=dict[str, int])
    flags: dict[str, int] = field(default_factory=dict[str, int])
    pattern_attribution: dict[str, int] = field(default_factory=dict[str, int])
    end_states: dict[str, int] = field(default_factory=dict[str, int])
    rows: int = 0
    placeholder_rows: int = 0
    rows_of_bad_fetches: int = 0
    rows_without_stop: int = 0
    unattributed_rows: int = 0
    hours_left_out: list[int] = field(default_factory=list[int])
    index_problems: dict[str, str] = field(default_factory=dict[str, str])
    records: int = 0
    outcome: WriteOutcome | None = None
    missing_inputs: list[InputProblem] = field(default_factory=list[InputProblem])
    inconsistent_inputs: list[InputProblem] = field(default_factory=list[InputProblem])
    conflicts: list[InputProblem] = field(default_factory=list[InputProblem])

    @property
    def failed(self) -> bool:
        return bool(self.missing_inputs or self.inconsistent_inputs or self.conflicts)

    @property
    def match_rate(self) -> float | None:
        return None if not self.runs else round(self.matched / self.runs, 4)


@dataclass
class WindowResult:
    inputs: WindowInputs
    runs: list[Run]
    matches: list[TripMatch]


def match_window(
    storage: Storage,
    indexes: Indexes,
    lineage: Lineage,
    start: int,
    end: int,
    book_date: str,
    *,
    partial: bool = False,
    trip_updates: bool = False,
    withheld: bool = False,
) -> WindowResult:
    """Runs and TripMatch records of every run of one input window."""
    inputs = read_window(
        storage,
        lineage,
        start,
        end,
        indexes.route_feed,
        indexes.route_feed_digest,
        partial=partial,
        trip_updates=trip_updates,
    )
    ix = indexes(book_date)
    book = None if ix is None else PatternBook(ix)
    runs = build_runs(lineage, inputs.rows, inputs.placeholders, inputs.last_poll, book)
    fallback = "" if ix is None else ix.build_id
    matches = trip_matches(runs, Matcher(indexes, TZ, withheld), fallback)
    return WindowResult(inputs, runs, matches)


def _tally(
    report: MatchReport,
    matches: list[TripMatch],
    inputs: WindowInputs,
    indexes: Indexes,
) -> None:
    report.runs = len(matches)
    report.matched = sum(m.outcome == "matched" for m in matches)
    report.tiers = dict(sorted(Counter(m.tier for m in matches if m.tier).items()))
    report.miss_classes = dict(
        sorted(Counter(m.miss_class for m in matches if m.miss_class).items())
    )
    report.flags = dict(sorted(Counter(f for m in matches for f in m.flags).items()))
    report.pattern_attribution = dict(
        sorted(Counter(m.pattern_attribution for m in matches).items())
    )
    report.end_states = dict(sorted(Counter(m.end_state for m in matches).items()))
    report.rows = len(inputs.rows)
    report.placeholder_rows = len(inputs.placeholders)
    report.rows_of_bad_fetches = inputs.rows_of_bad_fetches
    report.rows_without_stop = inputs.rows_without_stop
    report.unattributed_rows = inputs.unattributed_rows
    report.hours_left_out = inputs.hours_left_out
    report.index_problems = dict(indexes.problems)
    report.missing_inputs += inputs.missing
    report.inconsistent_inputs += inputs.inconsistent


def _write(
    report: MatchReport,
    out: Storage,
    records: list[TripMatch] | list[SupplyObservation],
    read: dict[str, tuple[str, int]],
    code_sha: str,
    code_dirty: bool | None,
    overwrite: bool,
    retroactive: list[str],
) -> None:
    report.records = len(records)
    if report.failed or not records:
        return
    metadata = event_file_metadata(
        report.key, MATCH_VERSION, len(records), read, code_sha, code_dirty, retroactive
    )
    stored = out.get(report.key) if report.key in out.list(report.key) else None
    try:
        resolution = resolve_write(stored, records, metadata, overwrite)
    except EventFileError as err:
        report.inconsistent_inputs.append(InputProblem(report.key, str(err)))
        return
    report.outcome = resolution.outcome
    if resolution.problem is not None:
        report.conflicts.append(InputProblem(report.key, resolution.problem))
    if resolution.data is not None:
        out.put(report.key, resolution.data)


def _provenance(
    inputs: WindowInputs, indexes: Indexes
) -> tuple[dict[str, tuple[str, int]], list[str]]:
    read = dict(inputs.read)
    read.update(indexes.used)
    retroactive: list[str] = []
    if (
        inputs.lineage == "legacy"
        and inputs.window[0] <= indexes.route_feed.source.observed_to
    ):
        retroactive.append(ROUTE_FEED_INPUT)
    return read, retroactive


def build_trip_match(
    storage: Storage,
    out: Storage,
    indexes: Indexes,
    service_date: str,
    lineage: Lineage,
    code_sha: str,
    code_dirty: bool | None,
    *,
    overwrite: bool = False,
    partial: bool = False,
) -> MatchReport:
    start, end = service_day_window(service_date, TZ)
    key = ServiceDayKey(lineage, "trip_match", service_date).key
    report = MatchReport(
        "trip_match", lineage, service_date, key, lineage_window(lineage, start, end)
    )
    result = match_window(
        storage, indexes, lineage, start, end, service_date, partial=partial
    )
    mine = [m for m in result.matches if m.service_date == service_date]
    _tally(report, mine, result.inputs, indexes)
    if indexes(service_date) is None:
        report.missing_inputs.append(
            InputProblem(
                f"match index {service_date}",
                indexes.problems.get(service_date, "no build"),
            )
        )
    read, retroactive = _provenance(result.inputs, indexes)
    _write(report, out, mine, read, code_sha, code_dirty, overwrite, retroactive)
    return report


def build_supply(
    storage: Storage,
    out: Storage,
    indexes: Indexes,
    day: str,
    lineage: Lineage,
    code_sha: str,
    code_dirty: bool | None,
    *,
    overwrite: bool = False,
    partial: bool = False,
) -> MatchReport:
    from datetime import UTC, datetime

    d0 = int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp())
    span = lineage_window(lineage, d0, d0 + _DAY)
    key = UtcDayKey(lineage, "supply", day).key
    report = MatchReport("supply", lineage, day, key, span)
    result = match_window(
        storage,
        indexes,
        lineage,
        d0 - SUPPLY_MARGIN_S,
        d0 + _DAY + SUPPLY_MARGIN_S,
        day,
        partial=partial,
        trip_updates=lineage == "raw",
    )
    _tally(report, result.matches, result.inputs, indexes)
    records = supply_observations(
        lineage,
        result.inputs.polls,
        result.runs,
        result.matches,
        result.inputs.listed,
        indexes,
        TZ,
        span,
    )
    read, retroactive = _provenance(result.inputs, indexes)
    _write(report, out, records, read, code_sha, code_dirty, overwrite, retroactive)
    return report
