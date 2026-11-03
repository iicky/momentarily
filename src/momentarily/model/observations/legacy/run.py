"""Backfill the legacy archives of a store (a local mirror or R2) into
derived parquet files under ``model/observations/s1/derived/`` of a store.

The source holds the bucket's keys (``derive.storage``). A run covers a range
of UTC days, by default every day the source holds, one day at a time: every
archive's objects of that day in turn, listed under the day's folder. Each
record goes to the file its contract is partitioned into (``keys``): vehicles
and equipment hourly by ``observed_at``; alert versions daily by
``first_seen``; the legacy-only records daily by ``observed_at``. A file is
written once nothing later can add to it, so a run holds about one day of
daily records and one hour of hourly ones. Hourly files close on the
archive's watermark (an object's key bounds the time of every later record),
daily files at the end of their day.

A run writes the files of the days in its range only, so a day's files are
the same whichever range covers it. A trace object is filed under the day of
its ``scheduled_at`` but its records are observed at ``observed_at``: seconds
later as a rule, but a late Worker invocation is observed long after the
minute it is scheduled for (more than 67 minutes on 2026-10-25). So any trace
object of a day can fill the next day's files, and none can reach further:
the first day of a range also reads the whole previous day of trace objects
(``Source.spills``), and files after an explicit last day are left
unwritten (``deferred``). Without a last day, the run writes those too, as a
run over a whole mirror always has. A trace object that reaches past the
day after its own stops the run, because no range reads it there. Alert
versions need every sighting (``alerts``), so the whole ``archive/alerts`` up
to the recorder cutover's day is read first.

The recorder owns every record it writes from ``keys.RECORDER_CUTOVER`` on:
the adapters emit vehicles, equipment and alert versions observed before it
only, and alert versions of days before its UTC day only. The legacy-only
records continue past it.

A record a contract rejects stops the run. A day that reads an object it
cannot use is aborted: nothing of it is written, the report lists the day and
the key, and the run goes on with the next day as a new range would. That is
a listed object the store does not return (after the store's retries), a
store error, or a malformed one (listed with its reason), whose rows are
unknown. Both count for every day that reads the object, its own and, for a
trace object, the next (whose lead-in reads it), so no day is ever written
without an object that may hold its rows. A malformed or unreadable
``archive/alerts`` object stops the run before any day: every version depends
on every sighting. Files are rendered as they close and stored only once
their day is complete.

**Quarantine** (``quarantine``): an object the reviewed quarantine file lists
whose bytes hash to the listed sha256 is read as absent. Its days are
written without it, the report lists it under ``excluded``, and every file
it could have filled names it under ``parquet.EXCLUDED_METADATA_KEY``: the
files of its archive's record types whose span ends after the start of its
folder day and no later than the last day it may fill (``Source.spills``).
Its key is not trusted for more: its bytes are not what the archive writes.
An
``archive/alerts`` object is named in every ``alert_version`` file of the
run, since every version depends on every sighting. A listed object whose
bytes hash to anything else changed since the review: it fails as a
malformed one does.

**Writing.** A file is compared with what the store holds under its key:

* nothing: written with ``put_new``, which refuses a key written meanwhile;
  such a file is then compared again, so a concurrent writer is never
  overwritten;
* a raw-lineage file (the derive job's, read from its provenance), a mixed
  one, or an unreadable one: kept, never overwritten. A raw file at a key
  wholly before the cutover is listed as ``obsolete`` (the derive job prunes
  it on request), any other as a conflict;
* a legacy file with identical bytes, or bytes that differ only in the code
  provenance (``code_sha``, ``code_dirty``: this build rendered under the
  stored file's values gives its bytes): unchanged, kept;
* a legacy file with other content: a conflict, kept, unless ``overwrite``
  replaces it.

Nothing is ever deleted. A rerun over the same objects and code skips every
file, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import tempfile
import time
from collections import deque
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import date, timedelta
from itertools import islice
from pathlib import Path

import pyarrow as pa
from pydantic import BaseModel

from momentarily.model.observations.contracts import (
    AlertVersion,
    DerivedFileMetadata,
    EquipmentObservation,
    LegacyArchive,
    LegacyCollectorHealth,
    LegacyFetchOutcome,
    LegacyRouteAlertState,
    VehicleObservation,
)
from momentarily.model.observations.derive.storage import ConditionalStorage, Storage
from momentarily.model.observations.keys import (
    RECORDER_CUTOVER,
    RECORDER_CUTOVER_DAY,
    DailyRecord,
    DerivedDayKey,
    DerivedHourKey,
    HourlyRecord,
    utc_day,
)
from momentarily.model.observations.legacy import (
    alerts,
    alerts_liveness,
    ene,
    fresh_feeds,
    health,
    predictions,
    quarantine,
    trace,
)
from momentarily.model.observations.legacy.common import (
    DECODER_VERSION,
    Emitted,
    LegacyObject,
    MalformedObjectError,
)
from momentarily.model.observations.parquet import (
    file_lineage,
    file_metadata,
    read_file_metadata,
    write_records,
)

Adapter = Callable[[LegacyObject], list[Emitted]]

_DAY_SECONDS = 86400
_NEVER = 2**62


@dataclass(frozen=True)
class Source:
    archive: LegacyArchive
    adapt: Adapter
    watermark: Callable[[str], int] | None
    records: tuple[str, ...]
    """The record types (``keys`` names) its objects fill."""
    spills: bool = False
    """An object's records may land on the day after the one it is filed
    under (never later); False when every record is in its object's day."""


# Day order within a run; the alerts archive is read up front (alerts.py).
_FETCH = "legacy_fetch_outcome"
SOURCES: tuple[Source, ...] = (
    Source(
        "archive/trace", trace.adapt, trace.watermark, ("vehicle", _FETCH), spills=True
    ),
    Source("archive/trip_updates", fresh_feeds.adapt, None, (_FETCH,)),
    Source("archive/vehicles", fresh_feeds.adapt, None, (_FETCH,)),
    Source("archive/alerts_liveness", alerts_liveness.adapt, None, (_FETCH,)),
    Source("archive/ene", ene.adapt, ene.watermark, ("equipment", _FETCH)),
    Source("archive/health", health.adapt, None, ("legacy_collector_health",)),
    Source("v1/predictions", predictions.adapt, None, ("legacy_route_alert_state",)),
)
ALERTS: LegacyArchive = "archive/alerts"
ARCHIVES: tuple[LegacyArchive, ...] = (
    *(s.archive for s in SOURCES[:3]),
    ALERTS,
    *(s.archive for s in SOURCES[3:]),
)


@dataclass
class ArchiveStats:
    inputs: int = 0
    rows: dict[str, int] = field(default_factory=dict[str, int])
    """Rows bound for files of the run's days."""
    failures: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    seconds: float = 0.0


@dataclass
class RecordStats:
    files: int = 0
    rows: int = 0
    bytes: int = 0
    seconds: float = 0.0


@dataclass
class DayReport:
    day: str
    objects: int
    malformed: int
    written: int
    unchanged: int
    replaced: int
    conflicts: int
    obsolete: int
    seconds: float
    aborted: str | None = None
    """The listed object that could not be read, or was malformed; nothing of
    the day was written."""


@dataclass
class Report:
    archives: dict[str, ArchiveStats]
    records: dict[str, RecordStats]
    files: list[str]
    """Files written, new or ``replaced``."""
    unchanged: list[str] = field(default_factory=list[str])
    replaced: list[str] = field(default_factory=list[str])
    conflicts: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    """(key, reason): stored files kept that this run would have changed."""
    obsolete: list[str] = field(default_factory=list[str])
    """Raw-lineage files at keys the backfill owns (before the recorder
    cutover): kept, not written; ``derive --prune-obsolete`` removes them."""
    deferred: list[str] = field(default_factory=list[str])
    """Files after the range that its last day's objects begin: not written;
    the run that covers their day writes them."""
    days: list[DayReport] = field(default_factory=list[DayReport])
    aborted: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    """(day, key): days left unwritten because an object they read could not
    be read or was malformed (``archive/alerts`` for the up-front read, which
    stops the run)."""
    excluded: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    """(key, reason): quarantined objects read as absent, their days written
    without them."""

    @property
    def failed(self) -> bool:
        malformed = any(a.failures for a in self.archives.values())
        return bool(malformed or self.conflicts or self.obsolete or self.aborted)


class MissingObjectError(Exception):
    """A listed object the store did not return, even after retries. The
    archives are never pruned, so this is a transient store error, not bad
    data."""

    def __init__(self, key: str) -> None:
        super().__init__(f"{key} is listed but could not be read")
        self.key = key


class _MalformedInputError(Exception):
    """A malformed object a day reads. Its rows are unknown, so every day
    that reads it is aborted, as for an unreadable one."""

    def __init__(self, key: str) -> None:
        super().__init__(f"{key} is malformed")
        self.key = key


@dataclass(frozen=True)
class _Exclusion:
    """A quarantined object and the files it could have filled: those of
    ``records`` whose span ends in ``(after, by]``."""

    sha256: str
    reason: str
    records: tuple[str, ...]
    after: int
    by: int


@dataclass
class _Bucket:
    record: str
    end: int
    records: list[BaseModel] = field(default_factory=list[BaseModel])
    inputs: dict[str, tuple[str, int]] = field(
        default_factory=dict[str, tuple[str, int]]
    )
    archives: set[str] = field(default_factory=set[str])


def _day_start(day: str) -> int:
    return (date.fromisoformat(day) - date(1970, 1, 1)).days * _DAY_SECONDS


def _hourly(record: HourlyRecord, at: int) -> tuple[str, str, int]:
    start = at - at % 3600
    return DerivedHourKey(record, start).key, record, start + 3600


def _daily(record: DailyRecord, at: int) -> tuple[str, str, int]:
    day = utc_day(at)
    return DerivedDayKey(record, day).key, record, _day_start(day) + _DAY_SECONDS


def _partition(record: BaseModel) -> tuple[str, str, int]:
    """(file key, record name, end of the span the file covers)."""
    match record:
        case VehicleObservation():
            return _hourly("vehicle", record.observed_at)
        case EquipmentObservation():
            return _hourly("equipment", record.observed_at)
        case AlertVersion() if isinstance(record.first_seen, int):
            return _daily("alert_version", record.first_seen)
        case LegacyRouteAlertState():
            return _daily("legacy_route_alert_state", record.observed_at)
        case LegacyFetchOutcome():
            return _daily("legacy_fetch_outcome", record.observed_at)
        case LegacyCollectorHealth():
            return _daily("legacy_collector_health", record.observed_at)
        case _:
            raise TypeError(f"{type(record).__name__} has no legacy partition")


@dataclass(frozen=True)
class _Staged:
    """A rendered file not yet stored."""

    key: str
    record: str
    rows: int
    end: int
    data: bytes
    seconds: float
    same_as: bytes | None
    """The stored file, when it differs from ``data`` in code provenance
    alone (decided at render time, while the records are held)."""


def _stored_metadata(data: bytes) -> DerivedFileMetadata | None:
    try:
        return read_file_metadata(data)
    except (pa.ArrowException, KeyError, ValueError):
        return None


class _Writer:
    def __init__(
        self,
        out: ConditionalStorage,
        code: tuple[str, bool | None],
        overwrite: bool,
        span: tuple[int, int],
        staging: Path,
        report: Report,
    ) -> None:
        self._out = out
        self._code = code
        self._overwrite = overwrite
        self._span = span
        self._staging = staging
        self._report = report
        self._open: dict[str, _Bucket] = {}
        self._written: set[str] = set()
        self._pending: list[_Staged] = []
        self._excluded: dict[str, _Exclusion] = {}

    def _in_span(self, end: int) -> bool:
        return self._span[0] < end <= self._span[1]

    def add(self, archive: str, emitted: Iterable[Emitted], by: int | None) -> None:
        """``by``: no record may be in a file that ends after it."""
        rows = self._report.archives[archive].rows
        for e in emitted:
            key, name, end = _partition(e.record)
            if key in self._written:
                raise RuntimeError(f"{archive} adds to {key} after it was written")
            if by is not None and end > by:
                raise RuntimeError(
                    f"{archive}: {e.inputs[0].key} adds to {key}, beyond the "
                    "day after the one it is filed under, which a range reading "
                    "only the previous day as its lead-in would miss"
                )
            bucket = self._open.get(key)
            if bucket is None:
                bucket = self._open[key] = _Bucket(name, end)
            bucket.records.append(e.record)
            bucket.archives.add(archive)
            for s in e.inputs:
                bucket.inputs[s.key] = (s.sha256, s.observed_at)
            if self._in_span(end):
                rows[name] = rows.get(name, 0) + 1

    def exclude(self, key: str, exclusion: _Exclusion) -> None:
        """Name ``key`` in every file it could have filled. Kept across
        ``discard``: the same object always gives the same exclusion."""
        self._excluded[key] = exclusion

    def _excluded_from(self, bucket: _Bucket) -> dict[str, tuple[str, str]]:
        return {
            k: (x.sha256, x.reason)
            for k, x in self._excluded.items()
            if bucket.record in x.records and x.after < bucket.end <= x.by
        }

    def flush(self, before: int, archive: str | None = None) -> None:
        """Render every file whose span ends by ``before``; with ``archive``,
        only the files that archive alone fills. A file before the run's
        days (filled by the lead-in) is dropped. Rendered files reach the
        store at ``commit``."""
        for key in sorted(self._open):
            b = self._open[key]
            if b.end <= before and (archive is None or b.archives == {archive}):
                self._written.add(key)
                self._open.pop(key)
                if self._in_span(b.end):
                    self._stage(key, b)

    def defer(self) -> None:
        """Leave every file still open (after the run's days) unwritten."""
        self._report.deferred += sorted(self._open)
        self._open.clear()

    def discard(self, start: int) -> None:
        """Drop everything not yet committed and start over at ``start``, as
        a new range would."""
        self._open.clear()
        self._pending.clear()
        self._written.clear()
        self._span = (start, self._span[1])

    def _render(
        self, key: str, bucket: _Bucket, code: tuple[str, bool | None]
    ) -> bytes:
        meta = file_metadata(
            key, DECODER_VERSION, len(bucket.records), bucket.inputs, *code
        )
        path = self._staging / key
        write_records(path, bucket.records, meta, self._excluded_from(bucket))
        data = path.read_bytes()
        path.unlink()
        return data

    def _same_as(self, key: str, bucket: _Bucket, data: bytes) -> bytes | None:
        """The stored legacy file under ``key`` when it is this file built
        by other code: its bytes equal this build's under its own
        ``code_sha`` and ``code_dirty`` (the derive job's rule)."""
        stored = self._stored(key)
        if stored is None or stored == data:
            return None
        meta = _stored_metadata(stored)
        if meta is None or file_lineage(meta) != "legacy":
            return None
        code = (meta.provenance.code_sha, meta.provenance.code_dirty)
        if code == self._code or self._render(key, bucket, code) != stored:
            return None
        return stored

    def _stage(self, key: str, bucket: _Bucket) -> None:
        t0 = time.perf_counter()
        data = self._render(key, bucket, self._code)
        same_as = self._same_as(key, bucket, data)
        seconds = time.perf_counter() - t0
        self._pending.append(
            _Staged(
                key,
                bucket.record,
                len(bucket.records),
                bucket.end,
                data,
                seconds,
                same_as,
            )
        )

    def commit(self) -> None:
        """Store every rendered file."""
        for staged in self._pending:
            self._store(staged)
        self._pending.clear()

    def _stored(self, key: str) -> bytes | None:
        # List first: R2Storage.get retries a missing key for seconds.
        if key not in self._out.list(key):
            return None
        try:
            return self._out.get(key)
        except KeyError:
            return None

    def _conflict(self, key: str, reason: str) -> None:
        self._report.conflicts.append((key, reason))

    def _store(self, s: _Staged) -> None:
        t0 = time.perf_counter()
        stored = self._stored(s.key)
        if stored is None:
            if self._out.put_new(s.key, s.data):
                self._count(s, t0)
            elif self._stored(s.key) == s.data:
                self._report.unchanged.append(s.key)
            else:
                self._conflict(s.key, "written by another run meanwhile")
            return
        meta = _stored_metadata(stored)
        lineage = None if meta is None else file_lineage(meta) or "mixed"
        if lineage == "raw" and s.end <= RECORDER_CUTOVER:
            self._report.obsolete.append(s.key)
        elif lineage != "legacy":
            self._conflict(s.key, f"stored file is {lineage or 'unreadable'}")
        elif stored in (s.data, s.same_as):
            self._report.unchanged.append(s.key)
        elif self._overwrite:
            self._out.put(s.key, s.data)
            self._report.replaced.append(s.key)
            self._count(s, t0)
        else:
            self._conflict(s.key, "stored legacy file differs (--overwrite replaces)")

    def _count(self, s: _Staged, t0: float) -> None:
        stats = self._report.records.setdefault(s.record, RecordStats())
        stats.files += 1
        stats.rows += s.rows
        stats.bytes += len(s.data)
        stats.seconds += s.seconds + time.perf_counter() - t0
        self._report.files.append(s.key)


def _folder(archive: str, key: str) -> str | None:
    """The day folder of ``key`` under ``archive``; None when it has none."""
    folder, slash, _ = key.removeprefix(f"{archive}/").partition("/")
    return folder if slash else None


def _is_day(text: str) -> bool:
    try:
        return date.fromisoformat(text).isoformat() == text
    except ValueError:
        return False


def _fetch(source: Storage, keys: list[str], readers: int) -> Iterator[LegacyObject]:
    """Each key's object, in order, with at most ``2 * readers`` read
    ahead. Raises ``MissingObjectError`` for a key the store does not return
    (``R2Storage.get`` retries R2's spurious not-found answers first)."""

    def get(key: str) -> LegacyObject:
        try:
            return LegacyObject(key, source.get(key))
        except KeyError as err:
            raise MissingObjectError(key) from err

    if readers <= 1:
        yield from map(get, keys)
        return
    todo = iter(keys)
    with ThreadPoolExecutor(readers) as pool:
        ahead: deque[Future[LegacyObject]] = deque(
            pool.submit(get, k) for k in islice(todo, 2 * readers)
        )
        while ahead:
            yield ahead.popleft().result()
            for k in islice(todo, 1):
                ahead.append(pool.submit(get, k))


def _changed(entry: quarantine.Entry, obj: LegacyObject) -> str | None:
    """Why a quarantined object may not be read as absent; None when its
    bytes are the reviewed ones."""
    if obj.sha256 == entry.sha256:
        return None
    return (
        f"listed in the quarantine file with sha256 {entry.sha256}, but its "
        f"bytes hash to {obj.sha256}: it changed since the review"
    )


def _fail(stats: ArchiveStats, key: str, reason: str) -> _MalformedInputError:
    """List ``key``'s failure (once) and the error that aborts its days."""
    if (key, reason) not in stats.failures:
        stats.failures.append((key, reason))
    return _MalformedInputError(key)


def _exclude(report: Report, key: str, entry: quarantine.Entry) -> None:
    if (key, entry.reason) not in report.excluded:
        report.excluded.append((key, entry.reason))


def _alert_versions(
    source: Storage,
    report: Report,
    readers: int,
    quarantined: Mapping[str, quarantine.Entry],
) -> tuple[dict[str, list[Emitted]], dict[str, _Exclusion]]:
    """Alert versions by the day of their first sighting, from every
    ``archive/alerts`` object filed up to the recorder cutover's day (none
    after it is read: ``alerts.versions``), and the quarantined objects read
    as absent."""
    stats = report.archives[ALERTS]
    t0 = time.perf_counter()
    keys = [
        k
        for k in source.list(f"{ALERTS}/")
        if (day := _folder(ALERTS, k)) is not None and day <= RECORDER_CUTOVER_DAY
    ]
    sightings: list[alerts.Sighting] = []
    excluded: dict[str, _Exclusion] = {}
    for obj in _fetch(source, keys, readers):
        stats.inputs += 1
        entry = quarantined.get(obj.key)
        if entry is not None:
            changed = _changed(entry, obj)
            if changed is not None:
                raise _fail(stats, obj.key, changed)
            _exclude(report, obj.key, entry)
            excluded[obj.key] = _Exclusion(
                entry.sha256, entry.reason, ("alert_version",), 0, _NEVER
            )
            continue
        try:
            sightings.append(alerts.parse(obj))
        except MalformedObjectError as err:
            raise _fail(stats, obj.key, str(err)) from err
    by_day: dict[str, list[Emitted]] = {}
    for e in alerts.versions(sightings):
        first = e.record.first_seen if isinstance(e.record, AlertVersion) else None
        if not isinstance(first, int):
            raise TypeError(
                "alerts.versions yields alert versions with a first sighting"
            )
        by_day.setdefault(utc_day(first), []).append(e)
    stats.seconds += time.perf_counter() - t0
    return by_day, excluded


def held_days(source: Storage) -> tuple[str, str] | None:
    """The first and last UTC day folder of any legacy archive in ``source``;
    None when it holds none. Lists every archive in full."""
    days = {
        day
        for archive in ARCHIVES
        for key in source.list(f"{archive}/")
        if (day := _folder(archive, key)) is not None and _is_day(day)
    }
    return (min(days), max(days)) if days else None


def _calendar(first: str, last: str) -> list[str]:
    start, end = date.fromisoformat(first), date.fromisoformat(last)
    return [(start + timedelta(n)).isoformat() for n in range((end - start).days + 1)]


class _Run:
    def __init__(
        self,
        source: Storage,
        writer: _Writer,
        report: Report,
        readers: int,
        quarantined: Mapping[str, quarantine.Entry],
    ):
        self._source = source
        self._writer = writer
        self._report = report
        self._readers = readers
        self._quarantined = quarantined

    def objects(
        self, src: Source, keys: list[str], end: int, lead_in: bool = False
    ) -> None:
        """Adapt ``keys`` of ``src``, filed under the day that ends at
        ``end``, in order. ``lead_in``: read for the next day only; the run
        of their own day counts them. A malformed object is listed (once)
        and aborts the day reading it; a quarantined one is read as absent
        and named in the files it could have filled."""
        stats = self._report.archives[src.archive]
        t0 = time.perf_counter()
        by = end + _DAY_SECONDS if src.spills else end
        # Named before any object is read: the watermark flushes close files
        # by the keys read so far, and a quarantined object's key may name
        # any instant. The folder day bounds it instead, never the key's
        # minute (a bound before the folder day would reach files already
        # written). Its bytes are checked when it is read; a mismatch aborts
        # the day, so nothing named here is stored for it.
        for key in keys:
            entry = self._quarantined.get(key)
            if entry is not None:
                after = end - _DAY_SECONDS
                self._writer.exclude(
                    key,
                    _Exclusion(entry.sha256, entry.reason, src.records, after, by),
                )
        for obj in _fetch(self._source, keys, self._readers):
            key = obj.key
            if not lead_in:
                stats.inputs += 1
            try:
                entry = self._quarantined.get(key)
                if entry is not None:
                    changed = _changed(entry, obj)
                    if changed is not None:
                        raise _fail(stats, key, changed)
                    _exclude(self._report, key, entry)
                    continue
                emitted = src.adapt(obj)
            except MalformedObjectError as err:
                raise _fail(stats, key, str(err)) from err
            if not emitted:
                continue
            self._writer.add(src.archive, emitted, by)
            if src.watermark is not None:
                self._writer.flush(src.watermark(key), src.archive)
        stats.seconds += time.perf_counter() - t0

    def lead_in(self, day: str) -> None:
        """Every object filed the day before ``day`` that may hold records
        of it: the whole day of each archive that ``spills``."""
        start = _day_start(day)
        before = utc_day(start - 1)
        for src in SOURCES:
            if src.spills:
                keys = self._source.list(f"{src.archive}/{before}/")
                self.objects(src, keys, start, lead_in=True)
        self._writer.flush(start)

    def day(self, day: str, versions: list[Emitted]) -> None:
        end = _day_start(day) + _DAY_SECONDS
        for src in SOURCES:
            self.objects(src, self._source.list(f"{src.archive}/{day}/"), end)
        self._writer.add(ALERTS, versions, None)
        self._writer.flush(end)


class _Tally:
    """A day's share of the report's running totals."""

    def __init__(self, report: Report) -> None:
        self._report = report
        self._t0 = time.perf_counter()
        self._at = self._now()

    def _now(self) -> tuple[int, ...]:
        r = self._report
        return (
            sum(a.inputs for a in r.archives.values()),
            sum(len(a.failures) for a in r.archives.values()),
            len(r.files) - len(r.replaced),
            len(r.unchanged),
            len(r.replaced),
            len(r.conflicts),
            len(r.obsolete),
        )

    def day(self, day: str) -> DayReport:
        n = [b - a for a, b in zip(self._at, self._now(), strict=True)]
        return DayReport(day, *n, seconds=time.perf_counter() - self._t0)


def run(
    source: Storage,
    out: ConditionalStorage,
    code_sha: str,
    code_dirty: bool | None,
    *,
    first: str | None = None,
    last: str | None = None,
    overwrite: bool = False,
    readers: int = 1,
    on_day: Callable[[DayReport], None] | None = None,
    quarantined: Mapping[str, quarantine.Entry] | None = None,
) -> Report:
    """Backfill the legacy archives in ``source`` into ``out``, for the UTC
    days ``first`` through ``last`` (default: the days ``source`` holds).
    ``readers`` objects are read concurrently; ``on_day`` sees each day's
    report as it ends. Without ``last`` every file is written, including
    those after the last day held; with it, those are ``deferred``.
    ``quarantined``: the objects to read as absent (default: the packaged
    quarantine file, ``quarantine.load``)."""
    if quarantined is None:
        quarantined = quarantine.load()
    report = Report({a: ArchiveStats() for a in ARCHIVES}, {}, [])
    open_end = last is None
    if first is None or last is None:
        held = held_days(source)
        if held is None:
            return report
        first, last = first or held[0], last or held[1]
    days = _calendar(first, last)
    if not days:
        return report
    end = _NEVER if open_end else _day_start(days[-1]) + _DAY_SECONDS
    try:
        versions, excluded = (
            _alert_versions(source, report, readers, quarantined)
            if days[0] < RECORDER_CUTOVER_DAY
            else ({}, {})
        )
    except (MissingObjectError, _MalformedInputError) as err:
        report.aborted.append((ALERTS, err.key))
        return report
    with tempfile.TemporaryDirectory(prefix="legacy-") as staging:
        span = (_day_start(days[0]), end)
        writer = _Writer(
            out, (code_sha, code_dirty), overwrite, span, Path(staging), report
        )
        for key, exclusion in excluded.items():
            writer.exclude(key, exclusion)
        job = _Run(source, writer, report, readers, quarantined)
        lead_in = True
        for day in days:
            tally = _Tally(report)
            try:
                if lead_in:
                    job.lead_in(day)
                job.day(day, versions.pop(day, []))
            except (MissingObjectError, _MalformedInputError) as err:
                # Nothing of the day is stored; the next day starts afresh.
                writer.discard(_day_start(day) + _DAY_SECONDS)
                report.aborted.append((day, err.key))
                report.days.append(replace(tally.day(day), aborted=err.key))
                lead_in = True
            else:
                writer.commit()
                report.days.append(tally.day(day))
                lead_in = False
            if on_day is not None:
                on_day(report.days[-1])
        if open_end:
            writer.flush(_NEVER)
            writer.commit()
        writer.defer()
    return report
