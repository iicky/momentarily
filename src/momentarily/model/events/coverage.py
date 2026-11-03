"""Coverage: transport evidence per feed (docs/model-events.md §5.1, with the
combined rule of the 10-21 contract verdict).

Coverage is keyed to fetch instants. Per feed, every fetch is good or not:

* raw: the fetch parsed (``ok`` or ``empty``: a zero-entity feed is supply
  evidence, never a gap), its non-empty body was stored, layer 2 decoded its
  ``FeedHeaderObservation``, and the header was at most ``MAX_HEADER_AGE_S``
  old when it was observed. Otherwise it is ``fetch_failed`` (no parse, an
  empty body, or a body layer 2 could not decode), ``not_stored`` or
  ``stale_header`` (no header timestamp, or an older one);
* legacy: the ``archive/trace`` object lists the feed in ``fresh_feeds``
  (``LegacyFetchOutcome.succeeded``), else ``fetch_failed``.

The stretch between two consecutive good fetches ``[a, b)`` takes the first
state that applies:

1. ``fetch_failed``, ``not_stored``, ``stale_header``: a fetch of that kind
   lies between them, in that order;
2. ``no_invocation``: a slot of the cron grid whose own window
   ``[S, S + SLOT_S)`` holds no good fetch of the feed lies in ``(a, b)``,
   that is ``a < S`` and ``S + SLOT_S <= b``; or else ``b - a > MAX_GAP_S``.
   The grid is every ``scheduled_at`` read and the slots missing between
   them (``missing_slots``; bundle-index-only invocations count, duplicates
   do no harm). Presence is keyed to fetch instants, never to an
   invocation's ``scheduled_at``: a late invocation (a cron retry) fills
   only the slot its fetches finish in, nothing earlier;
3. ``suspect_stale`` (legacy): both fetches are in a run of
   ``SUSPECT_RUN`` or more consecutive trace objects whose non-empty row set
   for the feed (trip, stop, ``vehicle_timestamp``, placeholders excluded)
   is identical. Empty sets never count;
4. ``covered``.

A mid-day stale episode is therefore ``stale_header`` from the last good
fetch before it to the first good one after it. Before a feed's first good
fetch and after its last, the fetches that are not good make one stretch
with the state of rule 1, so an episode at the start of the evidence is
``stale_header`` from its first stale fetch. Where the inputs hold no fetch
on one side (before the lineage's first fetch, after the last one read) the
state is ``Unknown``.

A day's file covers exactly the UTC day: the neighbouring hours are read so
that a stretch crossing 00:00Z gets its state, then every stretch is clipped
to the day. Adjacent stretches with one state merge.

Lineage and the recorder cutover (``observations.keys.RECORDER_CUTOVER``):
raw evidence is every fetch observed at or after the cutover, legacy every
trace object observed before it. So raw coverage starts at 2026-10-21
04:00:00Z and legacy coverage stops there; neither lineage reads the
other's evidence.
"""

# pyarrow-stubs leave some generics unknown: silence that noise in this file.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false

from __future__ import annotations

import hashlib
from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from itertools import pairwise
from typing import Literal, cast

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ValidationError

from momentarily.model.conventions import Interval, Unknown
from momentarily.model.events.contracts import (
    FEED_FAMILY,
    CoverageInterval,
    CoverageState,
    Lineage,
)
from momentarily.model.events.files import (
    EventFileError,
    WriteOutcome,
    event_file_metadata,
    resolve_write,
)
from momentarily.model.events.keys import UtcDayKey
from momentarily.model.events.route_feed import RouteFeedTable, load_route_feed
from momentarily.model.events.trip_ids import parse_realtime
from momentarily.model.observations.columns import from_row
from momentarily.model.observations.contracts import (
    GTFSRT_SOURCES,
    CollectorTick,
    FeedFetch,
    FeedHeaderObservation,
    GtfsRtSource,
    LegacyFetchOutcome,
)
from momentarily.model.observations.derive.bundle import RawObjectError, read_bundle
from momentarily.model.observations.derive.run import InputProblem
from momentarily.model.observations.derive.storage import Storage
from momentarily.model.observations.keys import (
    PREFIX,
    RECORDER_CUTOVER,
    BundleKey,
    DerivedDayKey,
    DerivedHourKey,
    KeyLayoutError,
    TickKey,
    parse_key,
    utc_day,
)
from momentarily.model.observations.parquet import file_lineage, read_file_metadata

COVERAGE_VERSION = 2
"""Bump when the rules change what a file holds. 2: slot presence keyed to
fetch instants, and a legacy vehicle hour required for fresh polls."""

MAX_GAP_S = 120
MAX_HEADER_AGE_S = 120
SLOT_S = 60
SUSPECT_RUN = 3

ROUTE_FEED_INPUT = "momentarily/model/events/route_feed.json"
"""The provenance input name of the packaged ``RouteFeed`` table."""

_HOUR = 3600
_DAY = 86400
_FEEDS = cast(tuple[GtfsRtSource, ...], GTFSRT_SOURCES)
_PARSED = ("ok", "empty")
_STOPPED_AT = 1

FetchKind = Literal["good", "fetch_failed", "not_stored", "stale_header"]
_TRANSPORT: tuple[CoverageState, ...] = ("fetch_failed", "not_stored", "stale_header")
Cause = Literal["fetch", "missing_slot", "late", "suspect", "covered", "unknown"]

NO_EARLIER_FETCH = "no earlier fetch of this lineage in the inputs"
NO_LATER_FETCH = "no later fetch of this lineage in the inputs"
NO_FETCH = "no fetch of this feed in the inputs"

RowKey = tuple[str, str | None, int | None]
"""A legacy row as the staleness proxy compares it: trip, stop, timestamp."""


# --- the rule ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Poll:
    """One fetch of one feed: its instant, what it was, and (legacy) the
    feed's row set in that trace object."""

    at: int
    kind: FetchKind
    rows: frozenset[RowKey] = frozenset()


@dataclass(frozen=True)
class Evidence:
    """What one lineage holds for a day: the window the file covers, the
    ``scheduled_at`` of every invocation read, and each feed's fetches in
    time order (neighbouring hours included)."""

    lineage: Lineage
    window: tuple[int, int]
    slots: Sequence[int]
    polls: dict[GtfsRtSource, list[Poll]]


@dataclass(frozen=True, slots=True)
class _Stretch:
    start: int
    end: int
    state: CoverageState | None
    """None: Unknown, for ``reason``."""
    cause: Cause
    reason: str | None = None


def missing_slots(slots: Sequence[int]) -> list[int]:
    """Slots with no invocation between the first and the last one seen:
    every ``SLOT_S`` step inside a longer gap between two distinct
    ``scheduled_at`` values. Needs no phase: dispatch jitter of up to half a
    slot reads as present."""
    out: list[int] = []
    for a, b in pairwise(sorted(set(slots))):
        n = (b - a + SLOT_S // 2) // SLOT_S - 1
        out += [a + SLOT_S * k for k in range(1, n + 1)]
    return out


def _suspect_pairs(polls: Sequence[Poll]) -> set[int]:
    """Indices ``i`` where polls ``i`` and ``i + 1`` lie in one run of at
    least ``SUSPECT_RUN`` good polls with one identical non-empty row set."""
    pairs: set[int] = set()
    start = 0
    for i in range(1, len(polls) + 1):
        same = (
            i < len(polls)
            and polls[i].kind == "good"
            and polls[i - 1].kind == "good"
            and bool(polls[i].rows)
            and polls[i].rows == polls[i - 1].rows
        )
        if same:
            continue
        if i - start >= SUSPECT_RUN:
            pairs.update(range(start, i - 1))
        start = i
    return pairs


def _transport(polls: Sequence[Poll]) -> CoverageState | None:
    kinds = {p.kind for p in polls}
    return next((k for k in _TRANSPORT if k in kinds), None)


def _feed_stretches(
    polls: Sequence[Poll],
    grid: Sequence[int],
    missing: Sequence[int],
    window: tuple[int, int],
) -> list[_Stretch]:
    """The stretches of one feed tiling ``window``, before merging. ``grid``:
    the sorted cron slots; ``missing``: those no invocation was recorded for."""
    w0, w1 = window
    if not polls:
        return [_Stretch(w0, w1, None, "unknown", NO_FETCH)]
    if any(a.at > b.at for a, b in pairwise(polls)):
        raise ValueError("polls must be in time order")
    out: list[_Stretch] = []
    goods = [i for i, p in enumerate(polls) if p.kind == "good"]
    suspect = _suspect_pairs(polls)
    first, last = polls[0].at, polls[-1].at
    if first > w0:
        out.append(_Stretch(w0, first, None, "unknown", NO_EARLIER_FETCH))
    # Before the first good fetch (or with none at all), the others.
    lead = polls[: goods[0]] if goods else polls
    if lead:
        end = polls[goods[0]].at if goods else last
        out.append(_Stretch(first, end, _transport(lead), "fetch"))
    for i, j in pairwise(goods):
        a, b = polls[i].at, polls[j].at
        state = _transport(polls[i + 1 : j])
        cause: Cause = "fetch"
        if state is None:
            # A slot's window [S, S + SLOT_S) with no fetch: no good fetch
            # lies strictly between a and b, so it is one inside (a, b].
            k = bisect_right(grid, a)
            if (k < len(grid) and grid[k] + SLOT_S <= b) or b - a > MAX_GAP_S:
                m = bisect_left(missing, a)
                cause = (
                    "missing_slot" if m < len(missing) and missing[m] < b else "late"
                )
                state = "no_invocation"
            elif j == i + 1 and i in suspect:
                state, cause = "suspect_stale", "suspect"
            else:
                state, cause = "covered", "covered"
        out.append(_Stretch(a, b, state, cause))
    if goods and goods[-1] < len(polls) - 1:
        tail = polls[goods[-1] + 1 :]
        out.append(_Stretch(polls[goods[-1]].at, last, _transport(tail), "fetch"))
    if last < w1:
        out.append(_Stretch(last, w1, None, "unknown", NO_LATER_FETCH))
    clipped = [
        _Stretch(max(s.start, w0), min(s.end, w1), s.state, s.cause, s.reason)
        for s in out
    ]
    return [s for s in clipped if s.start < s.end]


@dataclass
class FeedCoverage:
    """One feed's day: seconds per state, ``no_invocation`` by cause, and
    the seconds left ``Unknown``."""

    seconds: dict[str, int] = field(default_factory=dict[str, int])
    no_invocation_missing_slot_s: int = 0
    no_invocation_late_s: int = 0
    unknown_s: int = 0
    intervals: int = 0
    fetches: int = 0

    @property
    def uncovered_s(self) -> int:
        """Transport-uncovered seconds (``suspect_stale`` is not one)."""
        return sum(
            s
            for state, s in self.seconds.items()
            if state not in ("covered", "suspect_stale")
        )


def coverage_intervals(
    evidence: Evidence,
) -> tuple[list[CoverageInterval], dict[GtfsRtSource, FeedCoverage]]:
    """The day's ``CoverageInterval`` records per feed, and their tally."""
    w0, w1 = evidence.window
    missing = missing_slots(evidence.slots)
    grid = sorted({*evidence.slots, *missing})
    records: list[CoverageInterval] = []
    tally: dict[GtfsRtSource, FeedCoverage] = {}
    for feed in _FEEDS:
        polls = evidence.polls.get(feed, [])
        ats = [p.at for p in polls]
        stats = tally[feed] = FeedCoverage(
            fetches=bisect_left(ats, w1) - bisect_left(ats, w0)
        )
        merged: list[_Stretch] = []
        for s in _feed_stretches(polls, grid, missing, evidence.window):
            length = s.end - s.start
            if s.state is None:
                stats.unknown_s += length
            else:
                stats.seconds[s.state] = stats.seconds.get(s.state, 0) + length
            stats.no_invocation_missing_slot_s += length * (s.cause == "missing_slot")
            stats.no_invocation_late_s += length * (s.cause == "late")
            prev = merged[-1] if merged else None
            if prev is not None and (prev.state, prev.reason) == (s.state, s.reason):
                merged[-1] = _Stretch(prev.start, s.end, s.state, prev.cause, s.reason)
            else:
                merged.append(s)
        stats.seconds = dict(sorted(stats.seconds.items()))
        stats.intervals = len(merged)
        for s in merged:
            state: CoverageState | Unknown = (
                s.state
                if s.state is not None
                else Unknown(unknown=True, reason=s.reason or NO_LATER_FETCH)
            )
            records.append(
                CoverageInterval(
                    source=feed,
                    lineage=evidence.lineage,
                    interval=Interval(start=s.start, end=s.end),
                    state=state,
                    evidence_count=bisect_left(ats, s.end) - bisect_left(ats, s.start),
                )
            )
    return records, tally


# --- reading layer-2 inputs ------------------------------------------------------


@dataclass
class CoverageReport:
    day: str
    lineage: Lineage
    key: str
    window: tuple[int, int]
    builder_version: int
    invocations: int = 0
    """Invocations (raw) or trace objects (legacy) scheduled in the window."""
    from_bundle_index: int = 0
    """Raw invocations known from their bundle index alone."""
    missing_slots: list[int] = field(default_factory=list[int])
    """Slots of the window no invocation was recorded for (the cron grid's
    holes). Coverage keys presence to fetch instants (rule 2), so a late
    invocation fills its slot here, never in the coverage file."""
    undecoded_fetches: int = 0
    """Raw fetches stored and parsed whose header layer 2 did not decode."""
    unattributed_rows: int = 0
    """Legacy rows whose trip id names no ``RouteFeed`` route."""
    dropped_edge_hours: list[int] = field(default_factory=list[int])
    """Neighbouring hours left out because their layer-2 file is missing:
    raw ``feed_header``, legacy ``vehicle``."""
    feeds: dict[str, FeedCoverage] = field(default_factory=dict[str, FeedCoverage])
    rows: int = 0
    outcome: WriteOutcome | None = None
    missing_inputs: list[InputProblem] = field(default_factory=list[InputProblem])
    inconsistent_inputs: list[InputProblem] = field(default_factory=list[InputProblem])
    conflicts: list[InputProblem] = field(default_factory=list[InputProblem])

    @property
    def failed(self) -> bool:
        return bool(self.missing_inputs or self.inconsistent_inputs or self.conflicts)


class CoverageWindowError(ValueError):
    """A day the lineage holds no evidence for, by the cutover."""


def coverage_window(day: str, lineage: Lineage) -> tuple[int, int]:
    """The part of UTC ``day`` a lineage's coverage file covers."""
    d0 = int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp())
    if lineage == "raw":
        window = (max(d0, RECORDER_CUTOVER), d0 + _DAY)
    else:
        window = (d0, min(d0 + _DAY, RECORDER_CUTOVER))
    if window[0] >= window[1]:
        side = "before" if lineage == "raw" else "from"
        raise CoverageWindowError(
            f"{day}: no {lineage} lineage {side} the recorder cutover {utc_day(RECORDER_CUTOVER)}"
        )
    return window


def _hours(start: int, end: int) -> range:
    """Hour starts of every hour that meets ``[start, end)``."""
    return range(start - start % _HOUR, end, _HOUR)


class _Inputs:
    """Objects read, their digests and data instants, and the problems."""

    def __init__(self, storage: Storage, report: CoverageReport) -> None:
        self.storage = storage
        self.report = report
        self.read: dict[str, tuple[str, int]] = {}

    def get(self, key: str) -> bytes | None:
        try:
            return self.storage.get(key)
        except KeyError:
            return None

    def keep(self, key: str, data: bytes, captured_at: int) -> None:
        self.read[key] = (hashlib.sha256(data).hexdigest(), captured_at)

    def missing(self, key: str, problem: str) -> None:
        self.report.missing_inputs.append(InputProblem(key, problem))

    def inconsistent(self, key: str, problem: str) -> None:
        self.report.inconsistent_inputs.append(InputProblem(key, problem))

    def derived[M: BaseModel](
        self, key: str, model: type[M], lineage: Lineage, required: bool
    ) -> list[M] | None:
        """The records of a layer-2 derived file of ``lineage``; None (and a
        missing input when ``required``) when there is none."""
        data = self.get(key)
        if data is None:
            if required:
                self.missing(key, "no layer-2 file")
            return None
        try:
            meta = read_file_metadata(data)
            rows: list[dict[str, object]] = pq.read_table(
                pa.BufferReader(data)
            ).to_pylist()
            records = [from_row(model, r) for r in rows]
        except (pa.ArrowException, KeyError, ValueError) as err:
            self.inconsistent(key, f"not a readable derived file: {err}")
            return None
        if file_lineage(meta) != lineage:
            self.inconsistent(key, f"not a {lineage}-lineage file")
            return None
        self.keep(key, data, meta.provenance.built_at)
        return records


def _fetch_at(f: FeedFetch) -> int:
    """``observed_at`` as layer 2 sets it; a timeout's start otherwise."""
    return (f.started_at_ms if f.finished_at_ms is None else f.finished_at_ms) // 1000


def _invocations(
    inputs: _Inputs, hour_start: int
) -> list[tuple[int, bool, list[FeedFetch]]]:
    """``(scheduled_at, from the bundle index, GTFS-RT fetches)`` of every
    invocation filed under an hour: the tick record, else the bundle index."""
    day, hh = utc_day(hour_start), f"{hour_start % _DAY // _HOUR:02d}"
    found: dict[tuple[int, int], dict[str, str]] = {}
    for kind in ("ticks", "raw/gtfsrt"):
        for key in inputs.storage.list(f"{PREFIX}{kind}/{day}/{hh}/"):
            try:
                parsed = parse_key(key)
            except KeyLayoutError as err:
                inputs.inconsistent(key, str(err))
                continue
            if not isinstance(parsed, TickKey | BundleKey):
                inputs.inconsistent(key, "not a tick or bundle key")
                continue
            found.setdefault((parsed.scheduled_at, parsed.started_at_ms), {})[kind] = (
                key
            )
    out: list[tuple[int, bool, list[FeedFetch]]] = []
    for ident, keys in sorted(found.items()):
        fetches: list[FeedFetch] | None = None
        from_index = False
        tick_key = keys.get("ticks")
        data = None if tick_key is None else inputs.get(tick_key)
        if tick_key is not None and data is not None:
            try:
                tick = CollectorTick.model_validate_json(data)
            except ValidationError as err:
                inputs.inconsistent(tick_key, f"not a CollectorTick: {err}")
            else:
                if (tick.scheduled_at, tick.started_at_ms) != ident:
                    inputs.inconsistent(tick_key, f"holds tick {tick.tick_id}")
                else:
                    inputs.keep(tick_key, data, tick.finished_at_ms // 1000)
                    fetches = [f for f in tick.fetches if f.source in GTFSRT_SOURCES]
        bundle_key = keys.get("raw/gtfsrt")
        if fetches is None and bundle_key is not None:
            data = inputs.get(bundle_key)
            if data is None:
                inputs.missing(bundle_key, "listed but not readable")
            else:
                try:
                    index = read_bundle(bundle_key, data).index
                except RawObjectError as err:
                    inputs.inconsistent(bundle_key, str(err))
                else:
                    ats = [_fetch_at(f) for f in index.fetches]
                    inputs.keep(bundle_key, data, max(ats, default=index.scheduled_at))
                    fetches = list(index.fetches)
                    from_index = True
        if fetches is not None:
            out.append((ident[0], from_index, fetches))
    return out


def raw_evidence(
    storage: Storage, day: str, report: CoverageReport
) -> tuple[Evidence, dict[str, tuple[str, int]]]:
    """The raw lineage's evidence for ``day``: recorder ticks (or bundle
    indexes) and layer-2 ``feed_header`` files, from the cutover on, with
    an hour either side of the window."""
    window = report.window
    span = (max(window[0] - _HOUR, RECORDER_CUTOVER), window[1] + _HOUR)
    inputs = _Inputs(storage, report)
    slots: list[int] = []
    fetches: list[tuple[int, FeedFetch]] = []
    # A tick is filed by its scheduled hour; its fetches finish after it.
    for hour_start in _hours(span[0] - _HOUR, span[1]):
        for scheduled_at, from_index, listed in _invocations(inputs, hour_start):
            slots.append(scheduled_at)
            in_window = window[0] <= scheduled_at < window[1]
            report.invocations += in_window
            report.from_bundle_index += in_window and from_index
            for f in listed:
                at = _fetch_at(f)
                if span[0] <= at < span[1]:
                    fetches.append((at, f))
    if not report.invocations:
        report.missing_inputs.append(
            InputProblem(
                f"{PREFIX}ticks/{day}/", "no recorded invocation in the window"
            )
        )

    def needs_header(f: FeedFetch) -> bool:
        return f.outcome in _PARSED and f.body_bytes != 0 and f.payload_key is not None

    headers: dict[str, FeedHeaderObservation] = {}
    usable: set[int] = set()
    for hour_start in _hours(*span):
        required = window[0] <= hour_start < window[1]
        wanted = any(
            needs_header(f) and at - at % _HOUR == hour_start for at, f in fetches
        )
        key = DerivedHourKey("feed_header", hour_start).key
        records = (
            inputs.derived(key, FeedHeaderObservation, "raw", required)
            if wanted
            else []
        )
        if records is None:
            if not required:
                report.dropped_edge_hours.append(hour_start)
            continue
        usable.add(hour_start)
        for r in records:
            headers[r.fetch_id] = r
    polls: dict[GtfsRtSource, list[Poll]] = {feed: [] for feed in _FEEDS}
    for at, f in sorted(fetches, key=lambda x: (x[0], x[1].fetch_id)):
        if at - at % _HOUR not in usable:
            continue
        kind: FetchKind
        if f.outcome not in _PARSED or f.body_bytes == 0:
            kind = "fetch_failed"
        elif f.payload_key is None:
            kind = "not_stored"
        else:
            h = headers.get(f.fetch_id)
            if h is None:
                report.undecoded_fetches += window[0] <= at < window[1]
                kind = "fetch_failed"
            elif h.source != f.source or h.observed_at != at:
                inputs.inconsistent(
                    DerivedHourKey("feed_header", at - at % _HOUR).key,
                    f"{f.fetch_id} names another source or instant",
                )
                continue
            elif (
                h.header_timestamp is None or at - h.header_timestamp > MAX_HEADER_AGE_S
            ):
                kind = "stale_header"
            else:
                kind = "good"
        polls[cast(GtfsRtSource, f.source)].append(Poll(at, kind))
    report.missing_slots = [
        s for s in missing_slots(slots) if window[0] <= s < window[1]
    ]
    return Evidence("raw", window, sorted(set(slots)), polls), inputs.read


def _is_placeholder(
    feed: GtfsRtSource, stopped: bool, vts: int | None, at: int
) -> bool:
    """§2.2 legacy rule: a row that is not stopped and reports from the
    future; on the B division (not SIR) any row that is not stopped."""
    if stopped:
        return False
    if FEED_FAMILY[feed] == "last" and feed != "si":
        return True
    return vts is not None and vts > at


@dataclass
class _TraceObject:
    observed_at: int
    scheduled_at: int
    fresh: dict[GtfsRtSource, bool] = field(default_factory=dict[GtfsRtSource, bool])
    rows: dict[GtfsRtSource, set[RowKey]] = field(
        default_factory=dict[GtfsRtSource, set[RowKey]]
    )


def legacy_evidence(
    storage: Storage,
    day: str,
    report: CoverageReport,
    route_feed: RouteFeedTable,
    route_feed_digest: str,
) -> tuple[Evidence, dict[str, tuple[str, int]]]:
    """The legacy lineage's evidence for ``day``: the ``archive/trace``
    fetch outcomes and vehicle rows of layer 2, before the cutover, with an
    hour either side of the window."""
    window = report.window
    span = (window[0] - _HOUR, min(window[1] + _HOUR, RECORDER_CUTOVER))
    inputs = _Inputs(storage, report)
    objects: dict[str, _TraceObject] = {}
    for d in sorted({utc_day(span[0]), day, utc_day(span[1] - 1)}):
        key = DerivedDayKey("legacy_fetch_outcome", d).key
        records = inputs.derived(key, LegacyFetchOutcome, "legacy", d == day)
        for r in records or []:
            if r.lineage.archive != "archive/trace" or r.source not in GTFSRT_SOURCES:
                continue
            if not span[0] <= r.observed_at < span[1]:
                continue
            if isinstance(r.scheduled_at, Unknown):
                inputs.inconsistent(key, f"{r.legacy_fetch_id} has no scheduled_at")
                continue
            obj = objects.setdefault(
                r.lineage.source_key, _TraceObject(r.observed_at, r.scheduled_at)
            )
            if (obj.observed_at, obj.scheduled_at) != (r.observed_at, r.scheduled_at):
                inputs.inconsistent(
                    key, f"{r.legacy_fetch_id} disagrees with its object"
                )
            obj.fresh[cast(GtfsRtSource, r.source)] = r.succeeded

    def hour_of(obj: _TraceObject) -> int:
        return obj.observed_at - obj.observed_at % _HOUR

    # Layer 2 records no hour as holding zero rows, so an hour with a fresh
    # poll needs its vehicle file: an absent one is never read as empty
    # polls. A missing neighbouring hour is left out, as raw does.
    dropped: set[int] = set()
    for hour_start in sorted({hour_of(o) for o in objects.values()}):
        key = DerivedHourKey("vehicle", hour_start).key
        data = inputs.get(key)
        if data is not None:
            _legacy_rows(inputs, key, data, objects, route_feed)
            continue
        in_hour = [o for o in objects.values() if hour_of(o) == hour_start]
        if not any(any(o.fresh.values()) for o in in_hour):
            continue
        if window[0] <= hour_start < window[1]:
            inputs.missing(key, "no layer-2 file for an hour with fresh trace objects")
        else:
            dropped.add(hour_start)
            report.dropped_edge_hours.append(hour_start)
    by_order = sorted(
        ((k, o) for k, o in objects.items() if hour_of(o) not in dropped),
        key=lambda kv: (kv[1].observed_at, kv[0]),
    )
    polls: dict[GtfsRtSource, list[Poll]] = {feed: [] for feed in _FEEDS}
    for _, obj in by_order:
        for feed in _FEEDS:
            if feed not in obj.fresh:
                continue
            kind: FetchKind = "good" if obj.fresh[feed] else "fetch_failed"
            polls[feed].append(
                Poll(obj.observed_at, kind, frozenset(obj.rows.get(feed, ())))
            )
    slots = sorted({o.scheduled_at for o in objects.values()})
    report.invocations = sum(
        window[0] <= o.scheduled_at < window[1] for o in objects.values()
    )
    if not report.invocations and not report.missing_inputs:
        inputs.missing(
            DerivedDayKey("legacy_fetch_outcome", day).key,
            "no archive/trace object in the window",
        )
    report.missing_slots = [
        s for s in missing_slots(slots) if window[0] <= s < window[1]
    ]
    inputs.read[ROUTE_FEED_INPUT] = (route_feed_digest, route_feed.source.observed_to)
    return Evidence("legacy", window, slots, polls), inputs.read


def _legacy_rows(
    inputs: _Inputs,
    key: str,
    data: bytes,
    objects: dict[str, _TraceObject],
    route_feed: RouteFeedTable,
) -> None:
    """Add each legacy vehicle row of one hourly file to its trace object's
    row set for the feed ``RouteFeed`` gives its trip id."""
    try:
        meta = read_file_metadata(data)
        table = pq.read_table(
            pa.BufferReader(data),
            columns=[
                "lineage",
                "observed_at",
                "trip",
                "stop_id",
                "current_status",
                "vehicle_timestamp",
            ],
        )
    except (pa.ArrowException, KeyError, ValueError) as err:
        inputs.inconsistent(key, f"not a readable derived file: {err}")
        return
    if file_lineage(meta) != "legacy":
        inputs.inconsistent(key, "not a legacy-lineage file")
        return
    inputs.keep(key, data, meta.provenance.built_at)
    lineage = cast(pa.StructArray, table.column("lineage").combine_chunks())
    legacy = cast(pa.StructArray, lineage.field("legacy"))
    trip = cast(pa.StructArray, table.column("trip").combine_chunks())
    source_keys = cast(list[str], legacy.field("source_key").to_pylist())
    trip_ids = cast(list[str | None], trip.field("trip_id").to_pylist())
    observed = cast(list[int], table.column("observed_at").to_pylist())
    stops = cast(list[str | None], table.column("stop_id").to_pylist())
    status = cast(list[int | None], table.column("current_status").to_pylist())
    vts = cast(list[int | None], table.column("vehicle_timestamp").to_pylist())
    feeds: dict[str, GtfsRtSource | None] = {}
    orphans: set[str] = set()
    for sk, trip_id, at, stop, st, ts in zip(
        source_keys, trip_ids, observed, stops, status, vts, strict=True
    ):
        obj = objects.get(sk)
        if obj is None:
            if sk not in orphans:
                orphans.add(sk)
                inputs.inconsistent(key, f"rows of {sk} have no fetch outcome")
            continue
        if trip_id is None:
            inputs.report.unattributed_rows += 1
            continue
        if trip_id not in feeds:
            parsed = parse_realtime(trip_id)
            feeds[trip_id] = (
                None if parsed is None else route_feed.feed_of(parsed.route_field)
            )
        feed = feeds[trip_id]
        if feed is None:
            inputs.report.unattributed_rows += 1
            continue
        if _is_placeholder(feed, st == _STOPPED_AT, ts, at):
            continue
        obj.rows.setdefault(feed, set()).add((trip_id, stop, ts))


# --- the build -------------------------------------------------------------------


def build_coverage(
    storage: Storage,
    out: Storage,
    day: str,
    lineage: Lineage,
    code_sha: str,
    code_dirty: bool | None,
    *,
    overwrite: bool = False,
) -> CoverageReport:
    """Build the ``lineage`` coverage file of UTC ``day`` from the layer-2
    store ``storage`` into ``out``. Nothing is written when an input is
    missing or inconsistent, or when the stored file conflicts. Raises
    ``CoverageWindowError`` for a day the lineage does not hold."""
    key = UtcDayKey(lineage, "coverage", day).key
    report = CoverageReport(
        day, lineage, key, coverage_window(day, lineage), COVERAGE_VERSION
    )
    if lineage == "raw":
        evidence, read = raw_evidence(storage, day, report)
        retroactive: list[str] = []
    else:
        table, digest = load_route_feed()
        evidence, read = legacy_evidence(storage, day, report, table, digest)
        # The table was measured on raw days, after these legacy ones.
        retroactive = (
            [ROUTE_FEED_INPUT] if table.source.observed_to >= report.window[0] else []
        )
    records, tally = coverage_intervals(evidence)
    report.feeds = {feed: tally[feed] for feed in _FEEDS}
    report.rows = len(records)
    if report.failed:
        return report
    metadata = event_file_metadata(
        key, COVERAGE_VERSION, len(records), read, code_sha, code_dirty, retroactive
    )
    stored = out.get(key) if key in out.list(key) else None
    try:
        resolution = resolve_write(stored, records, metadata, overwrite)
    except EventFileError as err:
        report.inconsistent_inputs.append(InputProblem(key, str(err)))
        return report
    report.outcome = resolution.outcome
    if resolution.problem is not None:
        report.conflicts.append(InputProblem(key, resolution.problem))
    if resolution.data is not None:
        out.put(key, resolution.data)
    return report
