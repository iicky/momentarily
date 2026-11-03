"""Coverage (docs/model-events.md §5.1 and the combined rule of the 10-21
verdict) and the event file writer.

Raw cases run on a scripted recorder store: tick records, bundles and
layer-2 ``feed_header`` files written here. The frozen-feed case runs the
layer-2 derive job on tests/fixtures/model/events/frozen_feed/, four real
invocations of 2026-10-21 cut by scripts/gen_frozen_feed_fixture.py. Legacy
cases write ``archive/trace`` objects and run the layer-2 legacy backfill on
them.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from momentarily.model.events import coverage as coverage_module
from momentarily.model.events.__main__ import main
from momentarily.model.events.contracts import CoverageInterval
from momentarily.model.events.coverage import (
    CoverageReport,
    build_coverage,
)
from momentarily.model.events.files import read_records
from momentarily.model.events.keys import UtcDayKey
from momentarily.model.observations.columns import parquet_metadata
from momentarily.model.observations.contracts import (
    GTFSRT_SOURCES,
    CollectorBuild,
    CollectorTick,
    FeedFetch,
    FeedHeaderObservation,
    RawLineage,
    TickBundleIndex,
)
from momentarily.model.observations.decode import DECODER_VERSION
from momentarily.model.observations.derive.run import derive
from momentarily.model.observations.derive.storage import LocalStorage
from momentarily.model.observations.keys import (
    RECORDER_CUTOVER,
    BundleKey,
    DerivedHourKey,
    TickKey,
)
from momentarily.model.observations.legacy import run as legacy_run
from momentarily.model.observations.parquet import (
    Rows,
    file_metadata,
    write_parquet,
    write_records,
)
from scripts.gen_recorded_hour_fixture import bundle_bytes, compact_json, gzip_member

FROZEN = (
    Path(__file__).resolve().parent / "fixtures" / "model" / "events" / "frozen_feed"
)
CODE = ("coverage-test", False)


def _midnight(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp())


def _intervals(root: Path, lineage: str, day: str) -> list[CoverageInterval]:
    key = UtcDayKey(lineage, "coverage", day).key  # pyright: ignore[reportArgumentType]
    records, meta = read_records((root / key).read_bytes(), CoverageInterval)
    assert meta.file_key == key
    return records


def _feed(records: list[CoverageInterval], feed: str) -> list[tuple[int, int, str]]:
    """(start, end, state) of one feed, ``unknown`` for an Unknown state."""
    return [
        (
            r.interval.start,
            r.interval.end,
            r.state if isinstance(r.state, str) else "unknown",
        )
        for r in records
        if r.source == feed
    ]


def _build(root: Path, out: Path, day: str, lineage: str, **kw: Any) -> CoverageReport:
    return build_coverage(
        LocalStorage(root),
        LocalStorage(out),
        day,
        lineage,  # pyright: ignore[reportArgumentType]
        *kw.pop("code", CODE),
        **kw,
    )


# --- a scripted raw store --------------------------------------------------------

DAY = "2026-10-23"
D0 = _midnight(DAY)
# Ten cron minutes from 23:57:19 on the day before, fetches 2 s later.
SLOTS = [D0 - 161 + 60 * k for k in range(10)]
MISSING = 4  # 00:01:19: no tick and no bundle
LATE = 6  # 00:03:19 runs 39 s late: 99 s after the fetch before it
BUNDLE_ONLY = 8  # the tick record was lost; the bundle index describes it
T = [s + 2 for s in SLOTS]
T[LATE] += 39


def _fetch(feed: str, k: int, outcome: str, member: int | None) -> FeedFetch:
    """One GTFS-RT fetch of minute ``k``: stored as bundle ``member``, or
    not stored (None) when its body was read."""
    started = (T[k] - 1) * 1000
    timed_out = outcome == "timeout"
    body = None if timed_out else 100
    return FeedFetch.model_validate(
        {
            "fetch_id": f"{feed}@{started}",
            "source": feed,
            "url": f"https://example.test/{feed}",
            "method": "GET",
            "scheduled_at": SLOTS[k],
            "started_at_ms": started,
            "finished_at_ms": None if timed_out else T[k] * 1000,
            "outcome": outcome,
            "http_status": None if timed_out else 200,
            "content_type": None if timed_out else "application/octet-stream",
            "http_date": None,
            "http_last_modified": None,
            "http_etag": None,
            "body_bytes": body,
            "payload_sha256": None
            if timed_out
            else hashlib.sha256(feed.encode()).hexdigest(),
            "payload_key": BundleKey(SLOTS[k], (T[k] - 1) * 1000).key
            if member
            else None,
            "payload_member": member,
            "header_timestamp": None,
            "entity_count": None if timed_out else 5,
            "store_error": None if member or timed_out else "put failed",
        }
    )


def _header_time(feed: str, k: int) -> int:
    """``bdfm``'s header stands still from minute 1 to minute 6."""
    if feed == "bdfm" and 1 <= k <= LATE:
        return T[1] - 10
    return T[k] - 10


def _scripted_store(root: Path) -> None:
    """Every feed fetched every minute, but: ``ace`` times out at minute 2;
    minute 4 never ran; ``bdfm``'s header is stuck from minute 1 (stale from
    minute 3); minute 6 runs late; ``nqrw``'s body at minute 7 was not
    stored; minute 8 survives only as its bundle."""
    headers: dict[int, list[FeedHeaderObservation]] = {}
    inputs: dict[int, dict[str, tuple[str, int]]] = {}
    for k in range(len(SLOTS)):
        if k == MISSING:
            continue
        started = (T[k] - 1) * 1000
        fetches: list[FeedFetch] = []
        for feed in sorted(GTFSRT_SOURCES):
            timed_out = (feed, k) == ("ace", 2)
            stored = not timed_out and (feed, k) != ("nqrw", 7)
            member = 1 + sum(f.payload_member is not None for f in fetches)
            fetches.append(
                _fetch(
                    feed,
                    k,
                    "timeout" if timed_out else "ok",
                    member if stored else None,
                )
            )
        bodies = {
            f.payload_member: gzip_member(f.source.encode())
            for f in fetches
            if f.payload_member is not None
        }
        index = TickBundleIndex.model_validate(
            {
                "schema": "observations/s1",
                "tick_id": f"{SLOTS[k]}-{started}",
                "scheduled_at": SLOTS[k],
                "started_at_ms": started,
                "fetches": [f.model_dump(mode="json") for f in fetches],
            }
        )
        bundle_key = BundleKey(SLOTS[k], started).key
        (root / bundle_key).parent.mkdir(parents=True, exist_ok=True)
        (root / bundle_key).write_bytes(bundle_bytes(index, bodies))
        if k != BUNDLE_ONLY:
            tick = CollectorTick(
                schema="observations/s1",
                tick_id=f"{SLOTS[k]}-{started}",
                scheduled_at=SLOTS[k],
                started_at_ms=started,
                finished_at_ms=T[k] * 1000,
                collector=CollectorBuild(code_sha="test", code_dirty=False),
                planned_sources=sorted(GTFSRT_SOURCES),  # pyright: ignore[reportArgumentType]
                fetches=fetches,
            )
            tick_key = TickKey(SLOTS[k], started).key
            (root / tick_key).parent.mkdir(parents=True, exist_ok=True)
            (root / tick_key).write_bytes(compact_json(tick))
        hour = T[k] - T[k] % 3600
        for f in fetches:
            if f.payload_key is None or f.payload_member is None:
                continue
            headers.setdefault(hour, []).append(
                FeedHeaderObservation(
                    lineage=RawLineage(
                        kind="raw",
                        payload_key=f.payload_key,
                        payload_member=f.payload_member,
                        payload_sha256=f.payload_sha256 or "",
                    ),
                    fetch_id=f.fetch_id,
                    source=f.source,  # pyright: ignore[reportArgumentType]
                    observed_at=T[k],
                    gtfs_realtime_version="2.0",
                    incrementality=None,
                    header_timestamp=_header_time(f.source, k),
                    feed_version=None,
                    nyct_feed_header=None,
                )
            )
        inputs.setdefault(hour, {})[bundle_key] = ("ab" * 32, T[k])
    for hour, records in headers.items():
        key = DerivedHourKey("feed_header", hour).key
        meta = file_metadata(key, DECODER_VERSION, len(records), inputs[hour], *CODE)
        write_records(root / key, records, meta)


@pytest.fixture
def scripted(tmp_path: Path) -> Path:
    root = tmp_path / "store"
    _scripted_store(root)
    return root


def test_scripted_fetch_failure_missing_tick_and_stale_header(
    scripted: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    for day in ("2026-10-22", DAY):
        report = _build(scripted, out, day, "raw")
        assert not report.failed, report
    after = _intervals(out, "raw", DAY)
    before = _intervals(out, "raw", "2026-10-22")
    end = D0 + 86400
    # A missing tick: no_invocation between the fetches around it, though
    # they are only 120 s apart. The 99 s gap after it stays covered.
    assert _feed(after, "g") == [
        (D0, T[3], "covered"),
        (T[3], T[5], "no_invocation"),
        (T[5], T[9], "covered"),
        (T[9], end, "unknown"),
    ]
    assert _feed(before, "g") == [
        (D0 - 86400, T[0], "unknown"),
        (T[0], D0, "covered"),
    ]
    # A timeout makes its stretch fetch_failed, split at 00:00Z.
    assert _feed(before, "ace")[-1] == (T[1], D0, "fetch_failed")
    assert _feed(after, "ace")[0] == (D0, T[3], "fetch_failed")
    # A stuck header is stale from the last fresh fetch until the next one,
    # over the missing tick too.
    assert _feed(before, "bdfm")[-1] == (T[2], D0, "stale_header")
    assert _feed(after, "bdfm")[:2] == [
        (D0, T[7], "stale_header"),
        (T[7], T[9], "covered"),
    ]
    # Bytes that were not stored.
    assert (T[6], T[8], "not_stored") in _feed(after, "nqrw")
    # The bundle-only invocation is presence: no stretch reads no_invocation
    # around minute 8, and 99 s is not a gap.
    assert {s for _, _, s in _feed(after, "l")} == {
        "covered",
        "no_invocation",
        "unknown",
    }


def test_every_record_holds_its_fetches_and_the_day_is_tiled(
    scripted: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    report = _build(scripted, out, DAY, "raw")
    records = _intervals(out, "raw", DAY)
    for feed in GTFSRT_SOURCES:
        spans = [
            (r.interval.start, r.interval.end) for r in records if r.source == feed
        ]
        assert spans[0][0] == D0
        assert spans[-1][1] == D0 + 86400
        assert all(a[1] == b[0] for a, b in pairwise(spans))
        held = sum(r.evidence_count for r in records if r.source == feed)
        assert held == report.feeds[feed].fetches
    assert report.invocations == 6  # minutes 3, 5, 6, 7, 8 (bundle only), 9
    assert report.from_bundle_index == 1
    assert report.missing_slots == [SLOTS[MISSING]]


def test_a_rebuild_is_byte_identical_and_leaves_the_file_alone(
    scripted: Path, tmp_path: Path
) -> None:
    key = UtcDayKey("raw", "coverage", DAY).key
    first, second = tmp_path / "a", tmp_path / "b"
    assert _build(scripted, first, DAY, "raw").outcome == "written"
    assert _build(scripted, second, DAY, "raw").outcome == "written"
    assert (first / key).read_bytes() == (second / key).read_bytes()
    assert _build(scripted, first, DAY, "raw").outcome == "unchanged"
    # Another code version only restamps provenance: the stored file stays.
    before = (first / key).read_bytes()
    report = _build(scripted, first, DAY, "raw", code=("other-sha", True))
    assert report.outcome == "unchanged"
    assert (first / key).read_bytes() == before


def test_changed_content_conflicts_unless_overwritten(
    scripted: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    key = UtcDayKey("raw", "coverage", DAY).key
    _build(scripted, out, DAY, "raw")
    stored = (out / key).read_bytes()
    (scripted / TickKey(SLOTS[9], (T[9] - 1) * 1000).key).unlink()
    (scripted / BundleKey(SLOTS[9], (T[9] - 1) * 1000).key).unlink()
    report = _build(scripted, out, DAY, "raw")
    assert (report.outcome, report.failed) == ("conflict", True)
    assert (out / key).read_bytes() == stored
    assert _build(scripted, out, DAY, "raw", overwrite=True).outcome == "replaced"
    assert (out / key).read_bytes() != stored


def test_a_file_of_another_key_is_never_replaced(
    scripted: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out"
    _build(scripted, out, "2026-10-22", "raw")
    other = (out / UtcDayKey("raw", "coverage", "2026-10-22").key).read_bytes()
    key = UtcDayKey("raw", "coverage", DAY).key
    (out / key).parent.mkdir(parents=True, exist_ok=True)
    (out / key).write_bytes(other)
    report = _build(scripted, out, DAY, "raw", overwrite=True)
    assert report.outcome == "conflict"
    assert (out / key).read_bytes() == other


def test_a_missing_header_file_fails_and_writes_nothing(
    scripted: Path, tmp_path: Path
) -> None:
    (scripted / DerivedHourKey("feed_header", D0).key).unlink()
    out = tmp_path / "out"
    status = main(
        [
            "coverage",
            "--root",
            str(scripted),
            "--out",
            str(out),
            "--day",
            DAY,
            "--lineage",
            "raw",
        ]
    )
    assert status == 1
    assert not (out / UtcDayKey("raw", "coverage", DAY).key).exists()


def test_a_day_the_lineage_does_not_hold_is_a_usage_error(tmp_path: Path) -> None:
    args = ["coverage", "--root", str(tmp_path), "--out", str(tmp_path), "--day"]
    assert main([*args, "2026-10-20", "--lineage", "raw"]) == 2
    assert main([*args, "2026-10-22", "--lineage", "legacy"]) == 2


# --- the 2026-10-21 freeze ---------------------------------------------------------


def test_the_frozen_feeds_read_stale_header_and_l_and_numbered_covered(
    tmp_path: Path,
) -> None:
    """Four real invocations, 03:59:19 to 04:02:19 UTC: six feeds serve a
    header stuck at 03:22. Raw coverage starts at the recorder cutover, so
    the 03:59 invocation adds nothing."""
    root = tmp_path / "store"
    shutil.copytree(FROZEN, root)
    day = "2026-10-21"
    derived = derive(LocalStorage(root), LocalStorage(root), day, *CODE, hours=(3, 4))
    assert not derived.failed
    out = tmp_path / "out"
    report = _build(root, out, day, "raw")
    assert not report.failed, report
    records = _intervals(out, "raw", day)
    assert min(r.interval.start for r in records) == RECORDER_CUTOVER
    for feed in ("ace", "bdfm", "g", "jz", "nqrw", "si"):
        states = [s for _, _, s in _feed(records, feed)]
        assert states == ["unknown", "stale_header", "unknown"], feed
    for feed in ("l", "numbered"):
        states = [s for _, _, s in _feed(records, feed)]
        assert states == ["unknown", "covered", "unknown"], feed
        assert report.feeds[feed].fetches == 3


# --- legacy ------------------------------------------------------------------------

LEGACY_DAY = "2026-09-20"
L0 = _midnight(LEGACY_DAY) + 12 * 3600
ALL_FEEDS = ["ace", "bdfm", "g", "jz", "nqrw", "l", "numbered", "si"]


def _row(trip: str, stop: str, ts: int, stopped: bool = True) -> dict[str, Any]:
    return {
        "trip_id": trip,
        "route_id": trip.split("_")[1].split(".")[0],
        "direction": "south",
        "stop_id": stop,
        "stop_seq": 5 if stopped else None,
        "stopped": stopped,
        "vehicle_ts": ts,
    }


def _trace(mirror: Path, k: int, rows: list[dict[str, Any]], fresh: list[str]) -> None:
    at = L0 + 60 * k
    key = f"archive/trace/{LEGACY_DAY}/{at}.json"
    doc = {"observed_at": at, "scheduled_at": at, "fresh_feeds": fresh, "rows": rows}
    (mirror / key).parent.mkdir(parents=True, exist_ok=True)
    (mirror / key).write_text(json.dumps(doc))


def _legacy_store(tmp_path: Path) -> Path:
    """Eight trace minutes from 12:00 UTC. ``ace`` and ``numbered`` move on
    every minute; ``g`` and ``jz`` are fresh with no rows; ``si`` is thin
    (one or two rows) and changing; ``l`` reports one stopped train whose
    timestamp advances. Minute 3 holds no rows at all, every feed fresh, and
    minute 6 lists ``l`` as not fresh."""
    mirror = tmp_path / "mirror"
    for k in range(8):
        at = L0 + 60 * k
        rows: list[dict[str, Any]] = []
        if k != 3:
            rows += [
                _row("072000_A..S55R", f"A{10 + k}S", at - 20),
                _row("071000_SI.S", f"S{10 + k}S", at - 30),
                _row("071500_L..S", "L03S", at - 5),
            ]
            if k % 2:
                rows.append(_row("072500_SI.N", "S20N", at - 15))
            rows.append(_row("070000_1..S03R", f"R0{k}S", at - 10))
        fresh = [f for f in ALL_FEEDS if (f, k) != ("l", 6)]
        _trace(mirror, k, rows, fresh)
    store = tmp_path / "store"
    legacy_run.run(LocalStorage(mirror), LocalStorage(store), *CODE)
    return store


def test_legacy_empty_thin_and_fully_empty_trace_objects_stay_covered(
    tmp_path: Path,
) -> None:
    store = _legacy_store(tmp_path)
    out = tmp_path / "out"
    report = _build(store, out, LEGACY_DAY, "legacy")
    assert not report.failed, report
    records = _intervals(out, "legacy", LEGACY_DAY)
    t = [L0 + 60 * k for k in range(8)]
    end = _midnight(LEGACY_DAY) + 86400
    # Fresh with no rows, and the fully empty minute 3: covered. A thin
    # feed whose rows change: covered.
    for feed in ("ace", "g", "si", "jz", "numbered"):
        assert _feed(records, feed) == [
            (_midnight(LEGACY_DAY), t[0], "unknown"),
            (t[0], t[7], "covered"),
            (t[7], end, "unknown"),
        ], feed
    assert (t[5], t[7], "fetch_failed") in _feed(records, "l")


def test_a_stationary_fresh_feed_gives_suspect_stale_only(tmp_path: Path) -> None:
    """One stopped train with a frozen timestamp in four fresh objects. A
    placeholder whose timestamp moves does not hide it: placeholders are not
    part of the row set."""
    mirror = tmp_path / "mirror"
    for k in range(6):
        at = L0 + 60 * k
        ts = L0 + 30 if 1 <= k <= 4 else at - 10
        stop = "R01S" if 1 <= k <= 4 else f"R0{k}S"
        rows = [
            _row("070000_1..S03R", stop, ts),
            _row("080000_1..S03R", "101S", at + 600, stopped=False),
        ]
        _trace(mirror, k, rows, ALL_FEEDS)
    store = tmp_path / "store"
    legacy_run.run(LocalStorage(mirror), LocalStorage(store), *CODE)
    out = tmp_path / "out"
    report = _build(store, out, LEGACY_DAY, "legacy")
    assert not report.failed, report
    t = [L0 + 60 * k for k in range(6)]
    assert _feed(_intervals(out, "legacy", LEGACY_DAY), "numbered")[1:-1] == [
        (t[0], t[1], "covered"),
        (t[1], t[4], "suspect_stale"),
        (t[4], t[5], "covered"),
    ]
    assert report.feeds["numbered"].uncovered_s == 0


def test_an_hour_of_rowless_trace_objects_has_no_vehicle_file_and_stays_covered(
    tmp_path: Path,
) -> None:
    """The backfill writes no vehicle file for an hour whose trace objects
    hold no rows at all; those objects are fresh, empty polls."""
    mirror = tmp_path / "mirror"
    for k in range(-3, 3):  # 11:57 to 12:02; hour 11 holds rows, hour 12 none
        rows = (
            [_row("072000_A..S55R", f"A{20 + k}S", L0 + 60 * k - 20)] if k < 0 else []
        )
        _trace(mirror, k, rows, ALL_FEEDS)
    store = tmp_path / "store"
    legacy_run.run(LocalStorage(mirror), LocalStorage(store), *CODE)
    assert not (store / DerivedHourKey("vehicle", L0).key).exists()
    out = tmp_path / "out"
    report = _build(store, out, LEGACY_DAY, "legacy")
    assert not report.failed, report
    t = [L0 + 60 * k for k in range(-3, 3)]
    for feed in ("ace", "g"):
        assert _feed(_intervals(out, "legacy", LEGACY_DAY), feed)[1] == (
            t[0],
            t[-1],
            "covered",
        )


# --- the writer --------------------------------------------------------------------


def test_a_new_builder_version_replaces_the_stored_file(
    scripted: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    key = UtcDayKey("raw", "coverage", DAY).key
    _build(scripted, out, DAY, "raw")
    before = (out / key).read_bytes()
    monkeypatch.setattr(coverage_module, "COVERAGE_VERSION", 2)
    report = _build(scripted, out, DAY, "raw")
    assert (report.outcome, report.failed) == ("replaced", False)
    _, meta = read_records((out / key).read_bytes(), CoverageInterval)
    assert meta.builder_version == 2
    assert (out / key).read_bytes() != before


# sha256 of a layer-2 feed_header file written from the observation fixture
# by write_rows before its writer settings moved into write_parquet (pyarrow
# as locked in uv.lock). The shared settings must not change layer-2 bytes.
LAYER2_FILE_SHA256 = "b8d5ddd3e3dd10c5ff6ed839b0c18c3032ff51267890d769851602e00cb6402e"


def test_the_shared_writer_keeps_layer_two_bytes(tmp_path: Path) -> None:
    fixture = (
        Path(__file__).resolve().parent
        / "fixtures"
        / "model"
        / "observations"
        / "feed_header_observation.json"
    )
    records = [
        FeedHeaderObservation.model_validate(r) for r in json.loads(fixture.read_text())
    ]
    hour = records[0].observed_at - records[0].observed_at % 3600
    key = DerivedHourKey("feed_header", hour).key
    inputs = {
        r.lineage.payload_key: (r.lineage.payload_sha256, r.observed_at)
        for r in records
    }
    meta = file_metadata(key, "gtfsrt-1", len(records), inputs, "pin", False)
    write_records(tmp_path / key, records, meta)
    written = (tmp_path / key).read_bytes()
    assert hashlib.sha256(written).hexdigest() == LAYER2_FILE_SHA256
    rows = Rows(FeedHeaderObservation, key)
    rows.add(records)
    sink = pa.BufferOutputStream()
    write_parquet(rows.table().replace_schema_metadata(parquet_metadata(meta)), sink)
    assert sink.getvalue().to_pybytes() == written
