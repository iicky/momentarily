"""Derive one UTC day of recorded raw objects into derived parquet files.

Reads, from a ``Storage`` laid out as R2 (``keys``), every ``CollectorTick``
and GTFS-RT tick bundle filed under the hours that can hold the day's
observations, and the blobs they point at. Writes the hourly ``vehicle``,
``trip_update``, ``feed_header``, ``alert_presence`` and ``equipment`` files
and the daily ``alert_version`` file (``alert_index``).

**Joining an invocation.** Ticks and bundles are filed under the hour of
``scheduled_at``, records under the hour of ``observed_at``, so the hour
before the scope is read too. An invocation's fetches come from its tick
record. Without one, its GTFS-RT fetches come from the bundle's own index
and the minute's other fetches are lost (doc §4); ``from_bundle_index``
counts such invocations. When both exist, every fetch the index lists must
be in the tick as is, and every tick fetch with a bundle member must be
listed. Overlapping invocations of one minute are separate observations
and are all kept.

**Report.** Fetch, decode and record counts cover the fetches observed in
the scope (a timeout counts at its start). Input problems and decode
failures cover everything the run read, including the carry, because the
day's index depends on it.

**Decoding.** Only a stored payload of a fetch with outcome ``ok`` or
``empty`` is decoded. ``unparseable`` (the recorder could not parse it),
zero-byte GTFS-RT bodies (``empty`` to the recorder, but no FeedMessage),
unstored bodies and sources with no layer-2 record (``ene_equipments``,
Socrata, the static GTFS HEAD) are counted, not decoded. A ``DecodeError``
is reported per fetch and the run goes on.

**Memory.** Each fetch's records become arrow rows (``parquet.Rows``) at
once, and each hour's files are written and released as soon as no later
tick folder can add to them.

**Files.** Each file's metadata names, as inputs, every tick, bundle and
blob its records were read from, with the sha256 of the stored object and
the latest ``observed_at`` it contributed, so ``built_at`` is the latest
observation in the file and a rebuild from the same objects gives the same
bytes. ``Output`` applies the overwrite rule. A run that observes no fetch in
its scope (``no_ticks``) writes and removes nothing.
"""

from __future__ import annotations

import hashlib
import tempfile
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal, cast

import pyarrow as pa
from google.transit import gtfs_realtime_pb2
from pydantic import BaseModel, JsonValue, ValidationError

from momentarily.model.observations.contracts import (
    GTFSRT_SOURCES,
    AlertPresence,
    AlertVersion,
    CollectorTick,
    DerivedFileMetadata,
    EquipmentObservation,
    FeedFetch,
    FeedHeaderObservation,
    RawLineage,
    TripUpdateObservation,
    VehicleObservation,
    alert_content_digest,
)
from momentarily.model.observations.decode import (
    DECODER_VERSION,
    DecodeError,
    RawFetch,
    decode_alerts_json,
    decode_ene,
    decode_gtfsrt,
    numbered_alert_entity,
    parse_feed,
)
from momentarily.model.observations.decode.common import load_json
from momentarily.model.observations.derive.alert_index import (
    CARRY_SECONDS,
    Evidence,
    alert_versions,
    day_start,
)
from momentarily.model.observations.derive.bundle import (
    Bundle,
    RawObjectError,
    read_blob,
    read_bundle,
)
from momentarily.model.observations.derive.storage import Storage
from momentarily.model.observations.keys import (
    HOURLY_RECORDS,
    PREFIX,
    BlobKey,
    BundleKey,
    DerivedDayKey,
    DerivedHourKey,
    HourlyRecord,
    KeyLayoutError,
    TickKey,
    parse_key,
    utc_day,
)
from momentarily.model.observations.parquet import (
    Rows,
    file_lineage,
    file_metadata,
    read_file_metadata,
    write_rows,
)

_HOUR = 3600
_DAY = 86400
_PARSED = ("ok", "empty")
_ALERT_SOURCES = ("alerts", "numbered")
_DECODED = (*GTFSRT_SOURCES, "alerts", "ene_current", "ene_upcoming")
_MODELS: dict[HourlyRecord, type[BaseModel]] = {
    "vehicle": VehicleObservation,
    "trip_update": TripUpdateObservation,
    "feed_header": FeedHeaderObservation,
    "alert_presence": AlertPresence,
    "equipment": EquipmentObservation,
}

FileOutcome = Literal["written", "replaced", "unchanged", "removed", "conflict"]


@dataclass
class RecordStats:
    inputs: int = 0
    rows: int = 0
    files: int = 0
    outcomes: dict[str, int] = field(default_factory=dict[str, int])


@dataclass(frozen=True)
class DecodeFailure:
    fetch_id: str
    source: str
    payload_key: str
    payload_member: int | None
    error: str


@dataclass(frozen=True)
class InputProblem:
    key: str
    problem: str


@dataclass
class NotDecoded:
    """Fetches observed in the scope that were not decoded, by reason."""

    unparseable: int = 0
    """The recorder could not parse the body."""
    empty_body: int = 0
    """A zero-byte GTFS-RT body, which the recorder calls ``empty``."""
    not_stored: int = 0
    """The body was read but not stored (``store_error``)."""
    no_record: int = 0
    """The source has no layer-2 record (ENE catalog, Socrata, static HEAD)."""


@dataclass
class Report:
    day: str
    hours: tuple[int, int]
    decoder_version: str
    status: Literal["derived", "no_ticks"] = "no_ticks"
    from_tick_record: int = 0
    from_bundle_index: int = 0
    fetches: dict[str, int] = field(default_factory=dict[str, int])
    not_decoded: NotDecoded = field(default_factory=NotDecoded)
    records: dict[str, RecordStats] = field(default_factory=dict[str, RecordStats])
    alert_versions_reappearing: int = 0
    decode_failures: list[DecodeFailure] = field(default_factory=list[DecodeFailure])
    missing_inputs: list[InputProblem] = field(default_factory=list[InputProblem])
    inconsistent_inputs: list[InputProblem] = field(default_factory=list[InputProblem])
    conflicts: list[InputProblem] = field(default_factory=list[InputProblem])
    foreign_kept: list[str] = field(default_factory=list[str])

    @property
    def failed(self) -> bool:
        return bool(
            self.decode_failures
            or self.missing_inputs
            or self.inconsistent_inputs
            or self.conflicts
        )


# --- reading -----------------------------------------------------------------


@dataclass(frozen=True)
class _Invocation:
    """One Worker invocation as stored: the fetches, the key that listed
    them, and its bundle when one could be read."""

    fetches: list[FeedFetch]
    meta_key: str
    bundle: Bundle | None


class _Reader:
    def __init__(self, storage: Storage, report: Report) -> None:
        self._storage = storage
        self._report = report
        self.digests: dict[str, str] = {}
        self._blob_days: dict[str, set[str]] = {}
        self._blob_cache: tuple[str, bytes | None] | None = None
        self._reported: set[str] = set()

    def _get(self, key: str) -> bytes | None:
        """The bytes of a listed key; None (reported missing) when the store
        lists it but cannot return it."""
        try:
            data = self._storage.get(key)
        except KeyError:
            self.missing(key, "listed but not readable")
            return None
        self.digests[key] = hashlib.sha256(data).hexdigest()
        return data

    def missing(self, key: str, problem: str) -> None:
        if key not in self._reported:
            self._reported.add(key)
            self._report.missing_inputs.append(InputProblem(key, problem))

    def inconsistent(self, key: str, problem: str) -> None:
        if key not in self._reported:
            self._reported.add(key)
            self._report.inconsistent_inputs.append(InputProblem(key, problem))

    def hour(self, hour_start: int) -> Iterator[_Invocation]:
        """Every invocation filed under the hour starting at ``hour_start``,
        in (scheduled_at, started_at_ms) order."""
        day, hh = utc_day(hour_start), f"{hour_start % _DAY // _HOUR:02d}"
        ticks: dict[tuple[int, int], str] = {}
        bundles: dict[tuple[int, int], str] = {}
        for kind, found in (("ticks", ticks), ("raw/gtfsrt", bundles)):
            for key in self._storage.list(f"{PREFIX}{kind}/{day}/{hh}/"):
                try:
                    parsed = parse_key(key)
                except KeyLayoutError as err:
                    self.inconsistent(key, str(err))
                    continue
                if not isinstance(parsed, TickKey | BundleKey):
                    self.inconsistent(key, "not a tick or bundle key")
                    continue
                found[parsed.scheduled_at, parsed.started_at_ms] = key
        for ident in sorted(ticks.keys() | bundles.keys()):
            invocation = self._invocation(ticks.get(ident), bundles.get(ident), ident)
            if invocation is not None:
                yield invocation

    def _invocation(
        self, tick_key: str | None, bundle_key: str | None, ident: tuple[int, int]
    ) -> _Invocation | None:
        bundle: Bundle | None = None
        data = None if bundle_key is None else self._get(bundle_key)
        if bundle_key is not None and data is not None:
            try:
                bundle = read_bundle(bundle_key, data)
            except RawObjectError as err:
                self.inconsistent(bundle_key, str(err))
        tick: CollectorTick | None = None
        data = None if tick_key is None else self._get(tick_key)
        if tick_key is not None and data is not None:
            try:
                tick = CollectorTick.model_validate_json(data)
            except ValidationError as err:
                self.inconsistent(tick_key, f"not a CollectorTick: {err}")
            else:
                if (tick.scheduled_at, tick.started_at_ms) != ident:
                    self.inconsistent(tick_key, f"holds tick {tick.tick_id}")
                    tick = None
        if tick is None:
            if bundle is None or bundle_key is None:
                return None
            self._report.from_bundle_index += 1
            return _Invocation(list(bundle.index.fetches), bundle_key, bundle)
        self._report.from_tick_record += 1
        assert tick_key is not None
        own = BundleKey(*ident).key
        gtfsrt = {f.fetch_id: f for f in tick.fetches if f.source in GTFSRT_SOURCES}
        if bundle is not None:
            # A body that reported after the bundle was written is in the tick
            # only, with no member; every indexed fetch is in the tick as is.
            listed = {f.fetch_id for f in bundle.index.fetches}
            if any(gtfsrt.get(f.fetch_id) != f for f in bundle.index.fetches) or any(
                f.payload_member is not None and i not in listed
                for i, f in gtfsrt.items()
            ):
                self.inconsistent(own, f"index disagrees with {tick_key}")
        elif any(f.payload_key == own for f in gtfsrt.values()):
            self.missing(own, f"named by {tick_key}")
        return _Invocation(tick.fetches, tick_key, bundle)

    def payload(self, fetch: FeedFetch, invocation: _Invocation) -> bytes | None:
        """The stored body of ``fetch``, or None (reported) when it cannot
        be read."""
        key = fetch.payload_key
        assert key is not None
        if fetch.payload_member is not None:
            if invocation.bundle is None:
                return None  # reported with the bundle
            body = invocation.bundle.payloads.get(fetch.payload_member)
            if body is None:
                self.inconsistent(key, f"no member {fetch.payload_member}")
            return body
        parsed = parse_key(key)
        if not isinstance(parsed, BlobKey):
            self.inconsistent(key, f"{fetch.fetch_id} names no blob")
            return None
        listed = self._blob_days.get(parsed.day)
        if listed is None:
            listed = self._blob_days[parsed.day] = set(
                self._storage.list(f"{PREFIX}raw/blob/{parsed.day}/")
            )
        if key not in listed:
            self.missing(key, f"named by {fetch.fetch_id}")
            return None
        # Consecutive fetches mostly name the same blob (alerts rarely change).
        if self._blob_cache is not None and self._blob_cache[0] == key:
            return self._blob_cache[1]
        body: bytes | None = None
        data = self._get(key)
        try:
            body = None if data is None else read_blob(key, data)
        except RawObjectError as err:
            self.inconsistent(key, str(err))
        self._blob_cache = (key, body)
        return body


# --- decoding ----------------------------------------------------------------


def _alert_entities(source: str, payload: bytes) -> dict[str, dict[str, JsonValue]]:
    """Entity by content digest, as the decoders digest them."""
    found: list[dict[str, JsonValue]] = []
    if source == "numbered":
        msg = parse_feed(payload)
        entity: gtfs_realtime_pb2.FeedEntity
        for entity in msg.entity:
            if entity.HasField("alert"):
                found.append(numbered_alert_entity(entity))
    else:
        doc = load_json(payload)
        entities: JsonValue = doc.get("entity", []) if isinstance(doc, dict) else None
        if isinstance(entities, list):
            found += [e for e in entities if isinstance(e, dict)]
    return {alert_content_digest(e): e for e in found}


def _decode(payload: bytes, fetch: RawFetch) -> dict[HourlyRecord, list[BaseModel]]:
    out: dict[HourlyRecord, list[BaseModel]] = {}
    if fetch.source in GTFSRT_SOURCES:
        decoded = decode_gtfsrt(payload, fetch)
        out["feed_header"] = [decoded.header]
        out["vehicle"] = list(decoded.vehicles)
        out["trip_update"] = list(decoded.trip_updates)
        if decoded.alerts is not None:
            out["alert_presence"] = [decoded.alerts]
    elif fetch.source == "alerts":
        out["alert_presence"] = [decode_alerts_json(payload, fetch)]
    else:
        out["equipment"] = list(decode_ene(payload, fetch))
    return out


# --- writing -----------------------------------------------------------------


class Output:
    """Writes derived files under the overwrite rule:

    * no file under the key: written;
    * a file with identical bytes: unchanged;
    * a file whose provenance does not name raw inputs alone (a legacy
      backfill file): a conflict, kept even with ``overwrite``; this job
      never replaces or removes another writer's file;
    * a raw file written by another ``DECODER_VERSION``: replaced;
    * a file whose bytes equal this build's under the existing file's
      ``code_sha`` and ``code_dirty`` (only the code provenance differs):
      unchanged, the existing file is kept;
    * any other difference, or a stored file that cannot be read: a
      conflict. The existing file is kept and the run fails, unless
      ``overwrite`` replaces it.

    A stored file the rebuild has no rows for: another writer's file is
    kept and listed in ``foreign_kept`` (not a conflict); a raw file is
    removed when ``overwrite`` is set or its ``decoder_version`` differs,
    else a conflict.
    """

    def __init__(
        self,
        storage: Storage,
        code_sha: str,
        code_dirty: bool | None,
        overwrite: bool,
        report: Report,
    ) -> None:
        self._storage = storage
        self._code = (code_sha, code_dirty)
        self._overwrite = overwrite
        self._report = report
        self._staging = tempfile.TemporaryDirectory(prefix="derive-")
        self.produced: set[str] = set()

    def close(self) -> None:
        self._staging.cleanup()

    def list(self, prefix: str) -> list[str]:
        return self._storage.list(prefix)

    def _render(
        self,
        rows: Rows,
        inputs: dict[str, tuple[str, int]],
        code: tuple[str, bool | None],
    ) -> bytes:
        key = rows.file_key
        meta = file_metadata(key, DECODER_VERSION, len(rows), inputs, *code)
        path = Path(self._staging.name) / key
        write_rows(path, rows, meta)
        data = path.read_bytes()
        path.unlink()
        return data

    def _stored(self, key: str) -> tuple[bytes, DerivedFileMetadata | None] | None:
        """The stored file under ``key`` and its metadata (None when it
        cannot be read); None when there is no file."""
        if key not in self._storage.list(key):
            return None
        try:
            old = self._storage.get(key)
        except KeyError:
            return None
        try:
            return old, read_file_metadata(old)
        except (pa.ArrowException, KeyError, ValueError):
            return old, None

    def _foreign(self, key: str) -> FileOutcome:
        """A file another writer owns: kept whatever ``overwrite`` says."""
        self._report.conflicts.append(InputProblem(key, "not a raw-lineage file"))
        return "conflict"

    def _conflict(self, key: str, problem: str) -> FileOutcome:
        if self._overwrite:
            return "replaced"
        self._report.conflicts.append(InputProblem(key, problem))
        return "conflict"

    def _count(self, record: str, outcome: FileOutcome, rows: int, inputs: int) -> None:
        stats = self._report.records.setdefault(record, RecordStats())
        stats.files += outcome != "removed"
        stats.rows += rows
        stats.inputs += inputs
        stats.outcomes[outcome] = stats.outcomes.get(outcome, 0) + 1

    def write(
        self, record: str, rows: Rows, inputs: dict[str, tuple[str, int]]
    ) -> FileOutcome:
        key = rows.file_key
        self.produced.add(key)
        data = self._render(rows, inputs, self._code)
        stored = self._stored(key)
        outcome: FileOutcome
        if stored is None:
            outcome = "written"
        elif stored[0] == data:
            outcome = "unchanged"
        elif stored[1] is None:
            outcome = self._conflict(key, "stored file is not a readable derived file")
        elif file_lineage(stored[1]) != "raw":
            outcome = self._foreign(key)
        elif stored[1].decoder_version != DECODER_VERSION:
            outcome = "replaced"
        else:
            p = stored[1].provenance
            same = self._render(rows, inputs, (p.code_sha, p.code_dirty)) == stored[0]
            outcome = "unchanged" if same else self._conflict(key, "content differs")
        if outcome in ("written", "replaced"):
            self._storage.put(key, data)
        self._count(record, outcome, len(rows), len(inputs))
        return outcome

    def stale(self, record: str, key: str) -> None:
        """``key`` is stored, but the rebuild has no rows for it."""
        stored = self._stored(key)
        if stored is None:
            return
        meta = stored[1]
        if meta is not None and file_lineage(meta) != "raw":
            self._report.foreign_kept.append(key)
        elif self._overwrite or (
            meta is not None and meta.decoder_version != DECODER_VERSION
        ):
            self._storage.delete(key)
            self._count(record, "removed", 0, 0)
        else:
            self._report.conflicts.append(InputProblem(key, "the rebuild has no rows"))
            self._count(record, "conflict", 0, 0)


@dataclass
class _Bucket:
    rows: Rows
    inputs: dict[str, tuple[str, int]] = field(
        default_factory=dict[str, tuple[str, int]]
    )


def _add_inputs(
    inputs: dict[str, tuple[str, int]],
    keys: tuple[str, ...],
    digests: dict[str, str],
    at: int,
) -> None:
    for k in keys:
        seen = inputs.get(k)
        inputs[k] = (digests[k], at if seen is None else max(seen[1], at))


# --- the run -----------------------------------------------------------------


def check_day(day: str) -> None:
    """Reject anything but an ISO ``YYYY-MM-DD`` calendar date."""
    try:
        ok = date.fromisoformat(day).isoformat() == day
    except ValueError:
        ok = False
    if not ok:
        raise ValueError(f"not a YYYY-MM-DD date: {day!r}")


class _Day:
    """One derive run: the scope's hourly buckets and the index's evidence."""

    def __init__(
        self, day: str, hours: tuple[int, int], reader: _Reader, output: Output
    ) -> None:
        start = day_start(day)
        self.day = day
        self.whole_day = hours == (0, 23)
        self.scope = (start + hours[0] * _HOUR, start + (hours[1] + 1) * _HOUR)
        self.carry_from = self.scope[0] - (CARRY_SECONDS if self.whole_day else 0)
        self.reader = reader
        self.output = output
        self.buckets: dict[tuple[HourlyRecord, int], _Bucket] = {}
        self.evidence: list[Evidence] = []
        self.evidence_inputs: dict[str, tuple[str, int]] = {}
        self.entities: dict[tuple[str, str], dict[str, JsonValue]] = {}
        self.fetches: Counter[str] = Counter()
        self.not_decoded = NotDecoded()

    def folders(self) -> range:
        """Tick folders to read: every hour whose invocations can observe
        inside the carry or the scope."""
        return range(self.carry_from - _HOUR, self.scope[1], _HOUR)

    def fetch(self, inv: _Invocation, f: FeedFetch, report: Report) -> None:
        obs = None if f.finished_at_ms is None else f.finished_at_ms // 1000
        at = obs if obs is not None else f.started_at_ms // 1000
        in_scope = self.scope[0] <= at < self.scope[1]
        carried = self.carry_from <= at < self.scope[0] and f.source in _ALERT_SOURCES
        if in_scope:
            self.fetches[f.outcome] += 1
        skipped = self.not_decoded if in_scope else NotDecoded()
        if not (in_scope or carried) or obs is None or f.outcome not in _PARSED:
            skipped.unparseable += f.outcome == "unparseable"
            return
        if f.source not in _DECODED:
            skipped.no_record += 1
            return
        if f.source in GTFSRT_SOURCES and f.body_bytes == 0:
            # The recorder's wire scan counts 0 entities in 0 bytes; it is no
            # FeedMessage (the header is required) and says nothing.
            skipped.empty_body += 1
            return
        if f.payload_key is None or f.payload_sha256 is None:
            skipped.not_stored += 1
            return
        payload = self.reader.payload(f, inv)
        if payload is None:
            return
        lineage = RawLineage(
            kind="raw",
            payload_key=f.payload_key,
            payload_member=f.payload_member,
            payload_sha256=f.payload_sha256,
        )
        try:
            decoded = _decode(payload, RawFetch(f.fetch_id, f.source, obs, lineage))
        except DecodeError as err:
            report.decode_failures.append(
                DecodeFailure(
                    f.fetch_id, f.source, f.payload_key, f.payload_member, str(err)
                )
            )
            return
        keys = tuple(dict.fromkeys((inv.meta_key, f.payload_key)))
        if self.whole_day:
            for presence in decoded.get("alert_presence", []):
                self._evidence(f, cast(AlertPresence, presence), payload, keys, obs)
        if in_scope:
            hour_start = obs - obs % _HOUR
            for name, records in decoded.items():
                if records:
                    b = self.buckets.get((name, hour_start))
                    if b is None:
                        key = DerivedHourKey(name, hour_start).key
                        b = self.buckets[name, hour_start] = _Bucket(
                            Rows(_MODELS[name], key)
                        )
                    b.rows.add(records)
                    _add_inputs(b.inputs, keys, self.reader.digests, obs)

    def _evidence(
        self,
        f: FeedFetch,
        presence: AlertPresence,
        payload: bytes,
        keys: tuple[str, ...],
        obs: int,
    ) -> None:
        self.evidence.append(Evidence(f.scheduled_at, presence))
        _add_inputs(self.evidence_inputs, keys, self.reader.digests, obs)
        source = presence.source
        if any(
            (source, e.content_digest) not in self.entities for e in presence.entries
        ):
            for digest, entity in _alert_entities(source, payload).items():
                self.entities.setdefault((source, digest), entity)

    def flush(self, through: int) -> None:
        """Write every hourly file whose hour starts by ``through``."""
        for name, hour_start in sorted(self.buckets):
            if hour_start <= through:
                b = self.buckets.pop((name, hour_start))
                self.output.write(name, b.rows, b.inputs)

    def index(self, report: Report) -> None:
        if not (self.whole_day and self.evidence):
            return
        versions, stats = alert_versions(self.day, self.evidence, self.entities)
        report.alert_versions_reappearing = stats.reappearing
        if versions:
            rows = Rows(AlertVersion, DerivedDayKey("alert_version", self.day).key)
            rows.add(versions)
            self.output.write("alert_version", rows, self.evidence_inputs)

    def sweep(self) -> None:
        """Stored files in the scope that this rebuild produced no rows for."""
        hours = range(self.scope[0], self.scope[1], _HOUR)
        expected = {
            DerivedHourKey(name, h).key: name for name in HOURLY_RECORDS for h in hours
        }
        if self.whole_day:
            expected[DerivedDayKey("alert_version", self.day).key] = "alert_version"
        prefixes = [f"{PREFIX}derived/{name}/{self.day}/" for name in HOURLY_RECORDS]
        prefixes.append(DerivedDayKey("alert_version", self.day).key)
        for prefix in prefixes:
            for key in self.output.list(prefix):
                if key in expected and key not in self.output.produced:
                    self.output.stale(expected[key], key)


def derive(
    source: Storage,
    out: Storage,
    day: str,
    code_sha: str,
    code_dirty: bool | None,
    *,
    hours: tuple[int, int] = (0, 23),
    overwrite: bool = False,
) -> Report:
    """Derive ``day`` (UTC hours ``hours``, inclusive) from ``source`` into
    ``out``. The daily ``alert_version`` file is built only for a whole day.
    A day with no fetch observed in the scope writes and removes nothing
    (status ``no_ticks``)."""
    check_day(day)
    if not 0 <= hours[0] <= hours[1] <= 23:
        raise ValueError(f"hours {hours} are not a range within one day")
    report = Report(day, hours, DECODER_VERSION)
    output = Output(out, code_sha, code_dirty, overwrite, report)
    run = _Day(day, hours, _Reader(source, report), output)
    try:
        for folder in run.folders():
            for inv in run.reader.hour(folder):
                for f in inv.fetches:
                    run.fetch(inv, f, report)
            run.flush(folder)
        if run.fetches:
            run.index(report)
            run.sweep()
    finally:
        output.close()
    report.fetches = dict(sorted(run.fetches.items()))
    report.not_decoded = run.not_decoded
    report.records = dict(sorted(report.records.items()))
    if run.fetches:
        report.status = "derived"
    return report
