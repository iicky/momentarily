"""Backfill every legacy archive of a local R2 mirror into derived parquet
files under ``<out>/model/observations/s1/derived/``.

The mirror holds the bucket's keys as paths. Objects are read one UTC day at
a time, every archive in turn, and each record goes to the file its contract
is partitioned into (``keys``): vehicles and equipment hourly by
``observed_at``; alert versions daily by ``first_seen``; the legacy-only
records daily by ``observed_at``. A file is written once nothing later can
add to it, so a run holds about one day of daily records and one hour of
hourly ones. Hourly files close on the archive's watermark (an object's key
bounds the time of every later record), daily files at the end of their day.

A malformed object is skipped and listed in the report; a record a contract
rejects stops the run. A stored file at a target key that is not a readable
legacy file (raw lineage, read from its provenance, or unreadable) is kept
and listed as a conflict, never overwritten.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pyarrow as pa
from pydantic import BaseModel

from momentarily.model.observations.contracts import (
    AlertVersion,
    EquipmentObservation,
    LegacyArchive,
    LegacyCollectorHealth,
    LegacyFetchOutcome,
    LegacyRouteAlertState,
    VehicleObservation,
)
from momentarily.model.observations.keys import (
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


@dataclass(frozen=True)
class Source:
    archive: LegacyArchive
    adapt: Adapter
    watermark: Callable[[str], int] | None


# Day order within a run; the alerts archive is read up front (alerts.py).
SOURCES: tuple[Source, ...] = (
    Source("archive/trace", trace.adapt, trace.watermark),
    Source("archive/trip_updates", fresh_feeds.adapt, None),
    Source("archive/vehicles", fresh_feeds.adapt, None),
    Source("archive/alerts_liveness", alerts_liveness.adapt, None),
    Source("archive/ene", ene.adapt, ene.watermark),
    Source("archive/health", health.adapt, None),
    Source("v1/predictions", predictions.adapt, None),
)
ALERTS: LegacyArchive = "archive/alerts"
ARCHIVES: tuple[LegacyArchive, ...] = (
    *(s.archive for s in SOURCES[:3]),
    ALERTS,
    *(s.archive for s in SOURCES[3:]),
)

_DAY_SECONDS = 86400


@dataclass
class ArchiveStats:
    inputs: int = 0
    rows: dict[str, int] = field(default_factory=dict[str, int])
    failures: list[tuple[str, str]] = field(default_factory=list[tuple[str, str]])
    seconds: float = 0.0


@dataclass
class RecordStats:
    files: int = 0
    rows: int = 0
    bytes: int = 0
    seconds: float = 0.0


@dataclass
class Report:
    archives: dict[str, ArchiveStats]
    records: dict[str, RecordStats]
    files: list[str]
    conflicts: list[str] = field(default_factory=list[str])


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


class _Writer:
    def __init__(
        self, out: Path, code_sha: str, code_dirty: bool | None, report: Report
    ) -> None:
        self._out = out
        self._code = (code_sha, code_dirty)
        self._report = report
        self._open: dict[str, _Bucket] = {}
        self._written: set[str] = set()

    def add(self, archive: str, emitted: Iterable[Emitted]) -> None:
        rows = self._report.archives[archive].rows
        for e in emitted:
            key, name, end = _partition(e.record)
            if key in self._written:
                raise RuntimeError(f"{archive} adds to {key} after it was written")
            bucket = self._open.get(key)
            if bucket is None:
                bucket = self._open[key] = _Bucket(name, end)
            bucket.records.append(e.record)
            bucket.archives.add(archive)
            for s in e.inputs:
                bucket.inputs[s.key] = (s.sha256, s.observed_at)
            rows[name] = rows.get(name, 0) + 1

    def flush(self, before: int, archive: str | None = None) -> None:
        """Write every file whose span ends by ``before``; with ``archive``,
        only the files that archive alone fills."""
        for key in sorted(self._open):
            b = self._open[key]
            if b.end <= before and (archive is None or b.archives == {archive}):
                self._write(key, self._open.pop(key))

    def _foreign(self, key: str) -> bool:
        """Whether a file stored under ``key`` is not a readable legacy
        derived file (raw lineage, or unreadable)."""
        path = self._out / key
        if not path.is_file():
            return False
        try:
            meta = read_file_metadata(path.read_bytes())
        except (pa.ArrowException, KeyError, ValueError):
            return True
        return file_lineage(meta) != "legacy"

    def _write(self, key: str, bucket: _Bucket) -> None:
        self._written.add(key)
        if self._foreign(key):
            self._report.conflicts.append(key)
            return
        t0 = time.perf_counter()
        meta = file_metadata(
            key, DECODER_VERSION, len(bucket.records), bucket.inputs, *self._code
        )
        size = write_records(self._out / key, bucket.records, meta)
        stats = self._report.records.setdefault(bucket.record, RecordStats())
        stats.files += 1
        stats.rows += len(bucket.records)
        stats.bytes += size
        stats.seconds += time.perf_counter() - t0
        self._report.files.append(key)


def _listing(mirror: Path, archive: str) -> dict[str, list[Path]]:
    """Objects of ``archive`` by UTC day folder, in key order."""
    root = mirror / archive
    if not root.is_dir():
        return {}
    days: dict[str, list[Path]] = {}
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        days[folder.name] = sorted(p for p in folder.iterdir() if p.is_file())
    return days


def _read(mirror: Path, path: Path) -> LegacyObject:
    return LegacyObject(path.relative_to(mirror).as_posix(), path.read_bytes())


def _alert_versions(mirror: Path, stats: ArchiveStats) -> dict[str, list[Emitted]]:
    t0 = time.perf_counter()
    sightings: list[alerts.Sighting] = []
    for paths in _listing(mirror, ALERTS).values():
        for path in paths:
            obj = _read(mirror, path)
            stats.inputs += 1
            try:
                sightings.append(alerts.parse(obj))
            except MalformedObjectError as err:
                stats.failures.append((obj.key, str(err)))
    by_day: dict[str, list[Emitted]] = {}
    for e in alerts.versions(sightings):
        first = e.record.first_seen if isinstance(e.record, AlertVersion) else None
        if not isinstance(first, int):
            raise TypeError(
                "alerts.versions yields alert versions with a first sighting"
            )
        by_day.setdefault(utc_day(first), []).append(e)
    stats.seconds += time.perf_counter() - t0
    return by_day


def run(mirror: Path, out: Path, code_sha: str, code_dirty: bool | None) -> Report:
    """Backfill every legacy archive under ``mirror`` into ``out``."""
    report = Report({a: ArchiveStats() for a in ARCHIVES}, {}, [])
    writer = _Writer(out, code_sha, code_dirty, report)
    versions = _alert_versions(mirror, report.archives[ALERTS])
    listings = {s.archive: _listing(mirror, s.archive) for s in SOURCES}
    days = sorted(set(versions).union(*listings.values()))
    for day in days:
        for source in SOURCES:
            stats = report.archives[source.archive]
            for path in listings[source.archive].get(day, []):
                t0 = time.perf_counter()
                obj = _read(mirror, path)
                stats.inputs += 1
                try:
                    emitted = source.adapt(obj)
                except MalformedObjectError as err:
                    stats.failures.append((obj.key, str(err)))
                    emitted = []
                writer.add(source.archive, emitted)
                stats.seconds += time.perf_counter() - t0
                if source.watermark is not None and emitted:
                    writer.flush(source.watermark(obj.key), source.archive)
        writer.add(ALERTS, versions.get(day, []))
        writer.flush(_day_start(day) + _DAY_SECONDS)
    writer.flush(2**62)
    return report
