"""The offline derive job: recorded ticks, bundles and blobs to derived files.

tests/fixtures/model/observations/recorded_hour/ is ten cron minutes of real
2026-10-20 payloads packaged exactly as the recorder stores them
(scripts/gen_recorded_hour_fixture.py documents what it synthesizes: a
failed fetch, an overlapping invocation, a lost tick record).
tests/fixtures/recorder_output.json and worker/test/fixtures/observations/
are what the Worker's own test wrote, so the bundle reader is checked
against the recorder itself.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError
from pydantic import BaseModel

from momentarily.model.observations.contracts import (
    AlertEntry,
    AlertPresence,
    AlertVersion,
    CollectorTick,
    EquipmentObservation,
    FeedFetch,
    FeedHeaderObservation,
    RawLineage,
    TickBundleIndex,
    TripUpdateObservation,
    VehicleObservation,
    alert_content_digest,
)
from momentarily.model.observations.decode import DECODER_VERSION
from momentarily.model.observations.derive import run as derive_run
from momentarily.model.observations.derive.__main__ import main
from momentarily.model.observations.derive.alert_index import (
    Evidence,
    alert_versions,
    day_start,
)
from momentarily.model.observations.derive.bundle import read_bundle
from momentarily.model.observations.derive.run import NotDecoded, Report, derive
from momentarily.model.observations.derive.storage import LocalStorage
from momentarily.model.observations.keys import (
    BlobKey,
    BundleKey,
    DerivedDayKey,
    DerivedHourKey,
    TickKey,
    parse_key,
    utc_day,
)
from momentarily.model.observations.legacy import run as legacy_run
from momentarily.model.observations.legacy.__main__ import main as legacy_main
from momentarily.model.observations.parquet import read_records
from scripts.gen_recorded_hour_fixture import bundle_bytes, compact_json, gzip_member
from training.derive_observations import R2Storage

FIXTURES = Path(__file__).resolve().parent / "fixtures"
RECORDED = FIXTURES / "model" / "observations" / "recorded_hour"
WORKER_BODIES = FIXTURES.parent.parent / "worker" / "test" / "fixtures" / "observations"
DAY = "2026-10-20"
CODE = ("replay-test", False)

# sha256 of the canonical JSON of every derived file of the fixture day
# (records and metadata, code provenance aside), per DECODER_VERSION. A
# decoder change that alters output must bump DECODER_VERSION and add a pin;
# never edit an old version's pin.
REPLAY_DIGEST: dict[str, str] = {
    "1": "69e3e139087d0114460250ce8bd534cc884085c1d31a9b0bf466fb19b76a3d48",
}

MODELS: dict[str, type[BaseModel]] = {
    "vehicle": VehicleObservation,
    "trip_update": TripUpdateObservation,
    "feed_header": FeedHeaderObservation,
    "alert_presence": AlertPresence,
    "equipment": EquipmentObservation,
    "alert_version": AlertVersion,
}


def _derive(out: Path, source: Path = RECORDED, day: str = DAY, **kw: Any) -> Report:
    return derive(LocalStorage(source), LocalStorage(out), day, *CODE, **kw)


def _files(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*.parquet"))
    }


def _records(root: Path, record: str) -> list[Any]:
    found: list[Any] = []
    for key in _files(root):
        if key.split("/")[4] == record:
            found += read_records(root / key, MODELS[record])[0]
    return found


def _canonical(root: Path) -> str:
    doc: dict[str, Any] = {}
    for key in _files(root):
        record = key.split("/")[4]
        records, meta = read_records(root / key, MODELS[record])
        meta_doc = meta.model_dump(mode="json", by_alias=True)
        del meta_doc["provenance"]["code_sha"], meta_doc["provenance"]["code_dirty"]
        doc[key] = {
            "metadata": meta_doc,
            "records": [r.model_dump(mode="json", by_alias=True) for r in records],
        }
    text = json.dumps(doc, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture(scope="module")
def derived(tmp_path_factory: pytest.TempPathFactory) -> tuple[Report, Path]:
    """The fixture day derived once, for the tests that only read it."""
    out = tmp_path_factory.mktemp("derived")
    return _derive(out), out


# --- replay ------------------------------------------------------------------


def test_the_fixture_hour_replays_to_the_pinned_records(
    derived: tuple[Report, Path],
) -> None:
    """If this fails, derived output changed: bump DECODER_VERSION and pin
    the new digest under it."""
    report, out = derived
    assert not report.failed, report
    assert REPLAY_DIGEST.get(DECODER_VERSION) == _canonical(out)


def test_a_second_run_writes_identical_bytes(
    derived: tuple[Report, Path], tmp_path: Path
) -> None:
    _derive(tmp_path)
    first, second = _files(derived[1]), _files(tmp_path)
    assert first
    assert first == second


def test_a_minute_without_its_tick_record_is_read_from_its_bundle(
    derived: tuple[Report, Path],
) -> None:
    report, out = derived
    assert report.from_bundle_index == 1
    lost = {
        BundleKey(p.scheduled_at, p.started_at_ms).key
        for p in (
            parse_key(k.relative_to(RECORDED).as_posix())
            for k in RECORDED.rglob("*.bin")
        )
        if isinstance(p, BundleKey)
        and not (RECORDED / TickKey(p.scheduled_at, p.started_at_ms).key).exists()
    }
    assert len(lost) == 1
    headers = _records(out, "feed_header")
    from_lost = [h for h in headers if h.lineage.payload_key in lost]
    assert len(from_lost) == 8  # every GTFS-RT feed of that minute
    presences = _records(out, "alert_presence")
    # Its alerts fetch was only in the tick record, so that minute has none.
    assert sum(p.source == "alerts" for p in presences) == 10
    assert sum(p.source == "numbered" for p in presences) == 11


def test_overlapping_invocations_of_one_minute_are_both_kept(
    derived: tuple[Report, Path],
) -> None:
    out = derived[1]
    by_minute: dict[int, set[str]] = {}
    for tick_path in RECORDED.rglob("ticks/**/*.json"):
        tick = CollectorTick.model_validate_json(tick_path.read_bytes())
        by_minute.setdefault(tick.scheduled_at // 60, set()).add(tick.tick_id)
    (minute,) = [m for m, ids in by_minute.items() if len(ids) == 2]
    si = [
        h
        for h in _records(out, "feed_header")
        if h.source == "si" and h.observed_at // 60 in (minute, minute + 1)
    ]
    assert len({h.fetch_id for h in si}) >= 2


def test_failed_and_unparsed_fetches_are_counted_not_decoded(
    derived: tuple[Report, Path],
) -> None:
    report = derived[0]
    assert report.fetches["timeout"] == 3
    assert report.fetches["http_error"] == 1
    # The static GTFS HEADs and the ENE equipment catalog.
    assert report.not_decoded == NotDecoded(no_record=11)
    assert report.decode_failures == []


# --- the recorder's own output -------------------------------------------------

GARBAGE = b"<html><body>Service Unavailable</body></html>"


def _recorder_store(
    root: Path, alerts_body: bytes | None = None
) -> tuple[CollectorTick, TickBundleIndex]:
    """The tick, bundle and blob the Worker test wrote, rebuilt from the
    fixture JSON and the bodies its upstream stub served (an empty 200 for
    g, a garbage 200 for jz). ``alerts_body`` replaces the alerts payload,
    as the recorder would have recorded it."""
    output = json.loads((FIXTURES / "recorder_output.json").read_text())
    tick = CollectorTick.model_validate(output["collector_tick"])
    index = TickBundleIndex.model_validate(output["tick_bundle_index"])
    store = LocalStorage(root)
    if alerts_body is not None:
        fetches: list[FeedFetch] = []
        for f in tick.fetches:
            if f.source == "alerts" and f.finished_at_ms is not None:
                sha = hashlib.sha256(alerts_body).hexdigest()
                at = f.finished_at_ms // 1000
                f = FeedFetch.model_validate(
                    f.model_dump()
                    | {
                        "body_bytes": len(alerts_body),
                        "payload_sha256": sha,
                        "payload_key": BlobKey.for_fetch(at, sha, "json").key,
                        "entity_count": len(json.loads(alerts_body)["entity"]),
                    }
                )
            fetches.append(f)
        tick = CollectorTick.model_validate(
            tick.model_dump(by_alias=True) | {"fetches": fetches}
        )

    def body(source: str) -> bytes:
        if source == "g":
            return b""
        if source == "jz":
            return GARBAGE
        if source == "alerts" and alerts_body is not None:
            return alerts_body
        ext = "pb" if source != "alerts" else "json"
        return gzip.decompress((WORKER_BODIES / f"{source}.{ext}.gz").read_bytes())

    members = {
        f.payload_member: body(f.source) for f in index.fetches if f.payload_member
    }
    store.put(
        BundleKey(index.scheduled_at, index.started_at_ms).key,
        bundle_bytes(index, members),
    )
    store.put(TickKey(tick.scheduled_at, tick.started_at_ms).key, compact_json(tick))
    for f in tick.fetches:
        if f.payload_key is not None and isinstance(parse_key(f.payload_key), BlobKey):
            store.put(f.payload_key, gzip_member(body(f.source)))
    return tick, index


def test_the_generator_writes_records_as_the_recorder_serializes_them() -> None:
    output = json.loads((FIXTURES / "recorder_output.json").read_text())
    for name, model in (
        ("collector_tick", CollectorTick),
        ("tick_bundle_index", TickBundleIndex),
    ):
        stringified = json.dumps(
            output[name], separators=(",", ":"), ensure_ascii=False
        )
        assert compact_json(model.model_validate(output[name])) == stringified.encode()


def test_the_bundle_reader_numbers_members_as_the_recorder_does(tmp_path: Path) -> None:
    _, index = _recorder_store(tmp_path)
    key = BundleKey(index.scheduled_at, index.started_at_ms).key
    bundle = read_bundle(key, (tmp_path / key).read_bytes())
    assert bundle.index == index
    stored = [f for f in index.fetches if f.payload_member is not None]
    assert len(stored) == len(bundle.payloads) == 5
    for f in stored:
        assert f.payload_member is not None
        sha = hashlib.sha256(bundle.payloads[f.payload_member]).hexdigest()
        assert sha == f.payload_sha256, f.fetch_id


def test_the_recorders_own_minute_derives_and_its_empty_body_is_skipped(
    tmp_path: Path,
) -> None:
    """The recorder calls a zero-byte 200 outcome empty. It is no
    FeedMessage, so it is skipped: neither a record nor a failure."""
    _recorder_store(tmp_path / "store")
    report = _derive(tmp_path / "out", tmp_path / "store", hours=(16, 16))
    assert (report.from_tick_record, report.from_bundle_index) == (1, 0)
    assert report.missing_inputs == report.inconsistent_inputs == []
    assert report.decode_failures == []
    # g's empty body, jz's garbage, the static HEAD.
    assert report.not_decoded == NotDecoded(empty_body=1, unparseable=1, no_record=1)
    headers = _records(tmp_path / "out", "feed_header")
    assert {h.source for h in headers} == {"nqrw", "numbered", "si"}
    assert {p.source for p in _records(tmp_path / "out", "alert_presence")} == {
        "alerts",
        "numbered",
    }
    args = ["--root", str(tmp_path / "store"), "--out", str(tmp_path / "cli")]
    assert main([*args, "--day", DAY, "--hours", "16-16"]) == 0


def test_a_decode_failure_is_reported_per_fetch_and_the_rest_derives(
    tmp_path: Path,
) -> None:
    """An alert entity without an id: the recorder counts it ok, the decoder
    cannot read it."""
    tick, _ = _recorder_store(tmp_path / "store", b'{"entity":[{"alert":{}}]}')
    report = _derive(tmp_path / "out", tmp_path / "store", hours=(16, 16))
    (alerts,) = [f for f in tick.fetches if f.source == "alerts"]
    assert [f.fetch_id for f in report.decode_failures] == [alerts.fetch_id]
    assert {p.source for p in _records(tmp_path / "out", "alert_presence")} == {
        "numbered"
    }
    assert _records(tmp_path / "out", "vehicle")
    args = ["--root", str(tmp_path / "store"), "--out", str(tmp_path / "cli")]
    assert main([*args, "--day", DAY, "--hours", "16-16"]) == 1


# --- missing inputs and empty days --------------------------------------------


def test_missing_and_corrupt_inputs_are_reported_and_the_rest_derives(
    tmp_path: Path,
) -> None:
    store = tmp_path / "store"
    shutil.copytree(RECORDED, store)
    first_tick = CollectorTick.model_validate_json(
        sorted(store.rglob("ticks/**/*.json"))[0].read_bytes()
    )
    (alerts,) = [f for f in first_tick.fetches if f.source == "alerts"]
    assert alerts.payload_key is not None
    blob = store / alerts.payload_key
    blob.unlink()
    bundles = sorted(store.rglob("*.bin"))
    bundles[0].write_bytes(bundles[0].read_bytes()[:-10])
    report = _derive(tmp_path / "out", store)
    assert [p.key for p in report.missing_inputs] == [
        blob.relative_to(store).as_posix()
    ]
    assert [p.key for p in report.inconsistent_inputs] == [
        bundles[0].relative_to(store).as_posix()
    ]
    assert report.failed
    assert _records(tmp_path / "out", "vehicle")
    assert not any(
        p.source == "alerts" for p in _records(tmp_path / "out", "alert_presence")
    )
    args = ["--root", str(store), "--out", str(tmp_path / "cli"), "--day", DAY]
    assert main(args) == 1


def test_a_day_without_recorded_ticks_is_a_clean_no_op(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "out"
    assert (
        main(["--root", str(RECORDED), "--out", str(out), "--day", "2026-10-21"]) == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "no_ticks"
    assert not out.exists()


def test_the_cli_rejects_a_missing_root_or_a_bad_day(tmp_path: Path) -> None:
    assert main(["--root", str(tmp_path / "absent"), "--day", DAY]) == 2
    assert main(["--root", str(RECORDED), "--day", "20261020"]) == 2


# --- the overwrite rule ----------------------------------------------------------


def _outcomes(report: Report) -> set[str]:
    return {o for s in report.records.values() for o in s.outcomes}


def test_a_rebuild_keeps_files_whose_content_is_unchanged(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _derive(out)
    before = _files(out)
    again = derive(LocalStorage(RECORDED), LocalStorage(out), DAY, "other-commit", True)
    assert _outcomes(again) == {"unchanged"}
    assert not again.failed
    assert _files(out) == before


def test_different_content_is_a_conflict_unless_overwritten(tmp_path: Path) -> None:
    store, out = tmp_path / "store", tmp_path / "out"
    shutil.copytree(RECORDED, store)
    _derive(out, store)
    before = _files(out)
    # The store lost a minute since: the hour's content differs.
    sorted(store.rglob("raw/gtfsrt/**/*.bin"))[-1].unlink()
    changed = _derive(out, store)
    assert changed.conflicts
    assert changed.failed
    assert _files(out) == before
    forced = _derive(out, store, overwrite=True)
    assert "replaced" in _outcomes(forced)
    assert not forced.conflicts
    assert _files(out) != before


def test_a_new_decoder_version_replaces_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "out"
    _derive(out)
    monkeypatch.setattr(derive_run, "DECODER_VERSION", "test-next")
    report = _derive(out)
    assert _outcomes(report) == {"replaced"}
    assert not report.failed


@pytest.mark.parametrize("resolve", ["overwrite", "decoder_version"])
def test_a_stored_file_the_rebuild_has_no_rows_for_is_not_kept_silently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resolve: str
) -> None:
    store, out = tmp_path / "store", tmp_path / "out"
    shutil.copytree(RECORDED, store)
    _derive(out, store)
    (equipment,) = [k for k in _files(out) if "/equipment/" in k]
    # Minute 0's tick record held the only ENE fetches; its bundle stays.
    sorted(store.rglob("ticks/**/*.json"))[0].unlink()
    kept = _derive(out, store)
    assert equipment in {c.key for c in kept.conflicts}
    assert equipment in _files(out)
    if resolve == "overwrite":
        resolved = _derive(out, store, overwrite=True)
    else:
        monkeypatch.setattr(derive_run, "DECODER_VERSION", "test-next")
        resolved = _derive(out, store)
    assert resolved.records["equipment"].outcomes == {"removed": 1}
    assert equipment not in _files(out)
    assert not resolved.conflicts


def test_an_unreadable_stored_file_is_a_conflict_not_a_crash(tmp_path: Path) -> None:
    out = tmp_path / "out"
    _derive(out)
    (key,) = [k for k in _files(out) if "/feed_header/" in k]
    (out / key).write_bytes(b"not parquet")
    report = _derive(out)
    assert [c.key for c in report.conflicts] == [key]
    assert (out / key).read_bytes() == b"not parquet"
    forced = _derive(out, overwrite=True)
    assert not forced.failed
    assert read_records(out / key, FeedHeaderObservation)[0]


# --- legacy backfill files share the derived/ keys -----------------------------------

LEGACY_MIRROR = FIXTURES / "model" / "observations" / "legacy_mirror"
HOUR_14 = day_start(DAY) + 14 * 3600
VEHICLE_13 = DerivedHourKey("vehicle", HOUR_14 - 3600).key
VEHICLE_14 = DerivedHourKey("vehicle", HOUR_14).key
ALERT_VERSION = DerivedDayKey("alert_version", DAY).key


def _shift(mirror: Path, key: str, to: int, origin: int) -> None:
    """Copy the fixture object ``key`` into ``mirror``, every timestamp within
    a day of ``origin`` moved by ``to - origin``."""

    def moved(text: str) -> str:
        return re.sub(
            r"\b\d{10}\b",
            lambda m: (
                str(int(m[0]) + to - origin)
                if abs(int(m[0]) - origin) < 86400
                else m[0]
            ),
            text,
        ).replace(utc_day(origin), utc_day(to))

    path = mirror / moved(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(moved((LEGACY_MIRROR / key).read_text()))


def _legacy_mirror(root: Path) -> Path:
    """Legacy objects moved onto the fixture day: trace ticks in hours 13 and
    14 (vehicle files) and two alert sightings (the alert_version file)."""
    trace = "archive/trace/2026-09-15/1789477995.json"
    for at in (HOUR_14 - 3000, HOUR_14 + 600):
        _shift(root, trace, at, 1789477995)
    for name in ("1791893575", "1791893817"):
        key = f"archive/alerts/2026-10-13/{name}-lmm_alert_271625_26.json"
        _shift(root, key, 1791893575 + 7 * 86400, 1791893575)
    return root


def _backfilled(out: Path, tmp_path: Path) -> dict[str, bytes]:
    legacy_run.run(_legacy_mirror(tmp_path / "mirror"), out, *CODE)
    files = _files(out)
    assert {VEHICLE_13, VEHICLE_14, ALERT_VERSION} <= set(files)
    return files


@pytest.mark.parametrize("overwrite", [False, True])
def test_a_day_without_ticks_leaves_backfill_files_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], overwrite: bool
) -> None:
    out, empty = tmp_path / "out", tmp_path / "empty"
    empty.mkdir()
    before = _backfilled(out, tmp_path)
    args = ["--root", str(empty), "--out", str(out), "--day", DAY]
    assert main(args + ["--overwrite"] * overwrite) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "no_ticks"
    assert report["records"] == {}
    assert _files(out) == before


@pytest.mark.parametrize("overwrite", [False, True])
def test_recorded_rows_never_replace_or_remove_backfill_files(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], overwrite: bool
) -> None:
    """Hour 14 and the daily index collide with recorded rows (conflicts);
    hour 13 has none, so the sweep keeps it without failing."""
    out = tmp_path / "out"
    before = _backfilled(out, tmp_path)
    args = ["--root", str(RECORDED), "--out", str(out), "--day", DAY]
    assert main(args + ["--overwrite"] * overwrite) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["failed"]
    assert {c["key"] for c in report["conflicts"]} == {VEHICLE_14, ALERT_VERSION}
    assert report["foreign_kept"] == [VEHICLE_13]
    assert report["records"]["vehicle"]["outcomes"] == {"conflict": 1}
    assert report["records"]["alert_version"]["outcomes"] == {"conflict": 1}
    after = _files(out)
    assert {k: after[k] for k in before} == before
    assert DerivedHourKey("trip_update", HOUR_14).key in after


def test_the_backfill_never_overwrites_a_raw_file(tmp_path: Path) -> None:
    out, saved = tmp_path / "out", tmp_path / "report.json"
    _derive(out)
    raw = _files(out)
    mirror = _legacy_mirror(tmp_path / "mirror")
    args = ["--mirror", str(mirror), "--out", str(out), "--report", str(saved)]
    assert legacy_main(args) == 1
    report = json.loads(saved.read_text())
    assert sorted(report["conflicts"]) == sorted([VEHICLE_14, ALERT_VERSION])
    assert VEHICLE_13 in report["files"]
    after = _files(out)
    assert {k: after[k] for k in raw} == raw
    assert VEHICLE_13 in after


def test_the_backfill_never_overwrites_an_unreadable_file(tmp_path: Path) -> None:
    out, saved = tmp_path / "out", tmp_path / "report.json"
    (out / VEHICLE_13).parent.mkdir(parents=True)
    (out / VEHICLE_13).write_bytes(b"not parquet")
    mirror = _legacy_mirror(tmp_path / "mirror")
    args = ["--mirror", str(mirror), "--out", str(out), "--report", str(saved)]
    assert legacy_main(args) == 1
    report = json.loads(saved.read_text())
    assert report["conflicts"] == [VEHICLE_13]
    assert VEHICLE_14 in report["files"]
    assert (out / VEHICLE_13).read_bytes() == b"not parquet"


# --- the R2 store ------------------------------------------------------------------


class _FakeS3:
    """The S3 calls R2Storage makes, over a dict, listing two keys a page."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def get_paginator(self, name: str) -> _FakeS3:
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket: str, Prefix: str) -> Iterator[dict[str, Any]]:
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        for i in range(0, max(len(keys), 1), 2):
            yield {"Contents": [{"Key": k} for k in keys[i : i + 2]]} if keys else {}

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(self, Bucket: str, Key: str, Body: bytes, ContentType: str) -> None:
        self.objects[Key] = Body

    def delete_object(self, Bucket: str, Key: str) -> None:
        self.objects.pop(Key, None)


class _VanishingS3(_FakeS3):
    """Lists ``vanished`` but cannot return it, as R2 does for a key deleted
    after the listing."""

    def __init__(self, objects: dict[str, bytes], vanished: str) -> None:
        super().__init__(objects)
        self.vanished = vanished

    def paginate(self, Bucket: str, Prefix: str) -> Iterator[dict[str, Any]]:
        keys = sorted(k for k in {*self.objects, self.vanished} if k.startswith(Prefix))
        yield {"Contents": [{"Key": k} for k in keys]}


def test_the_r2_store_derives_what_the_local_store_does(tmp_path: Path) -> None:
    _derive(tmp_path / "local")
    objects = {
        p.relative_to(RECORDED).as_posix(): p.read_bytes()
        for p in RECORDED.rglob("*")
        if p.is_file()
    }
    fake = _FakeS3(objects)
    r2 = R2Storage(fake, "bucket")  # pyright: ignore[reportArgumentType]
    report = derive(r2, r2, DAY, *CODE)
    assert not report.failed
    derived = {k: v for k, v in fake.objects.items() if k.endswith(".parquet")}
    assert derived == _files(tmp_path / "local")


# --- the AlertVersion index ----------------------------------------------------------

D = day_start(DAY)


def _entity(alert_id: str, text: str) -> dict[str, Any]:
    return {
        "id": alert_id,
        "alert": {
            "header_text": {"translation": [{"text": text}]},
            "transit_realtime.mercury_alert": {
                "alert_type": "Delays",
                "created_at": D - 7200,
                "updated_at": D - 7200,
            },
        },
    }


class _Script:
    """Alerts presences at chosen instants, one per cron minute."""

    def __init__(self) -> None:
        self.evidence: list[Evidence] = []
        self.entities: dict[tuple[str, str], dict[str, Any]] = {}

    def poll(self, at: int, *entities: dict[str, Any]) -> None:
        sha = hashlib.sha256(str(at).encode()).hexdigest()
        entries: list[AlertEntry] = []
        for e in entities:
            digest = alert_content_digest(e)
            self.entities["alerts", digest] = e
            entries.append(
                AlertEntry(alert_id=e["id"], content_digest=digest, updated_at=D - 7200)
            )
        presence = AlertPresence(
            lineage=RawLineage(
                kind="raw",
                payload_key=BlobKey.for_fetch(at, sha, "json").key,
                payload_member=None,
                payload_sha256=sha,
            ),
            fetch_id=f"alerts@{at * 1000}",
            source="alerts",
            observed_at=at,
            header_timestamp=None,
            entries=sorted(entries, key=lambda x: (x.alert_id, x.content_digest)),
        )
        self.evidence.append(Evidence(at - 5, presence))

    def versions(self, day: str = DAY) -> dict[tuple[str, str], AlertVersion]:
        """Versions with one run each, by (alert_id, digest)."""
        found, _ = alert_versions(day, self.evidence, self.entities)
        by_key = {(v.alert_id, v.content_digest): v for v in found}
        assert len(by_key) == len(found)
        return by_key

    def runs(self, entity: dict[str, Any], day: str = DAY) -> list[tuple[Any, ...]]:
        """Bounds of every row of ``entity``'s version, by first_seen."""
        found, _ = alert_versions(day, self.evidence, self.entities)
        digest = alert_content_digest(entity)
        rows = [
            v for v in found if (v.alert_id, v.content_digest) == (entity["id"], digest)
        ]
        return [_bounds(v) for v in sorted(rows, key=lambda v: str(v.first_seen))]


def _bounds(v: AlertVersion) -> tuple[Any, ...]:
    gaps = (
        [(g.start, g.end) for g in v.unobserved]
        if isinstance(v.unobserved, list)
        else v.unobserved
    )
    return (v.absent_before, v.first_seen, v.last_seen, v.gone_by, v.gone_kind, gaps)


def test_alert_versions_bound_appear_persist_gap_supersede_and_removal() -> None:
    a1, a2, b1 = (
        _entity("A", "delays"),
        _entity("A", "delays, now worse"),
        _entity("B", "x"),
    )
    t = [D + 3600 + 60 * m + 20 for m in range(8)]
    s = _Script()
    s.poll(t[0])  # an empty feed: evidence that nothing is up
    s.poll(t[1], a1)  # A appears
    s.poll(t[2], a1)  # persists
    # minute 3: the fetch failed, so no presence: a gap, not a removal
    s.poll(t[4], a1, b1)
    s.poll(t[5], a2, b1)  # A's content changed: superseded
    s.poll(t[6], a2)  # B removed
    s.poll(t[7])  # an empty feed again: A removed
    v = s.versions()
    digest = alert_content_digest
    assert _bounds(v["A", digest(a1)]) == (
        t[0], t[1], t[4], t[5], "superseded", [(t[2] + 1, t[4])]
    )  # fmt: skip
    assert _bounds(v["B", digest(b1)]) == (t[2], t[4], t[5], t[6], "removed", [])
    assert _bounds(v["A", digest(a2)]) == (t[4], t[5], t[6], t[7], "removed", [])


def test_a_version_that_leaves_and_returns_is_one_row_per_run() -> None:
    a, b = _entity("A", "delays"), _entity("B", "other")
    t = [D + 3600 + 60 * m + 20 for m in range(9)]
    s = _Script()
    for at, present in zip(t, ([a], [a], [], [], [a], [], [a], [b], [a]), strict=True):
        s.poll(at, *present)
    # A, A, -, -, A, - then A, B, A: an empty feed or another alert is
    # evidence that A was gone, so each return starts a new row.
    assert s.runs(a) == [
        (None, t[0], t[1], t[2], "removed", []),
        (t[3], t[4], t[4], t[5], "removed", []),
        (t[5], t[6], t[6], t[7], "removed", []),
        (t[7], t[8], t[8], None, None, [(t[8] + 1, D + 86400)]),
    ]
    _, stats = alert_versions(DAY, s.evidence, s.entities)
    assert stats.reappearing == 1


def test_a_content_flip_and_back_supersedes_each_run() -> None:
    a1, a2 = _entity("A", "delays"), _entity("A", "delays, now worse")
    t = [D + 3600 + 60 * m + 20 for m in range(4)]
    s = _Script()
    for at, present in zip(t, ([a1], [a2], [a1], []), strict=True):
        s.poll(at, *present)
    assert s.runs(a1) == [
        (None, t[0], t[0], t[1], "superseded", []),
        (t[1], t[2], t[2], t[3], "removed", []),
    ]


def test_a_run_gone_just_after_midnight_is_closed_in_the_new_day() -> None:
    a = _entity("A", "ends at midnight")
    s = _Script()
    s.poll(D - 120 + 20)  # 23:58, empty
    s.poll(D - 60 + 20, a)  # 23:59, last sighting
    s.poll(D + 20)  # 00:00, gone
    assert s.runs(a, "2026-10-19") == [(D - 100, D - 40, D - 40, None, None, [])]
    assert s.runs(a) == [(D - 100, D - 40, D - 40, D + 20, "removed", [])]


def test_a_version_spanning_midnight_keeps_its_bounds_in_the_next_day() -> None:
    c, early = _entity("C", "overnight"), _entity("E", "carry only")
    s = _Script()
    s.poll(D - 1800 + 20)  # 23:30, empty
    s.poll(D - 1200 + 20, early)  # 23:40, seen in the carry only
    for m in range(-29, 11):  # 23:31 through 00:10
        s.poll(D + 60 * m + 20, c)
    s.poll(D + 660 + 20)  # 00:11, empty
    previous = "2026-10-19"
    v_prev = s.versions(previous)[("C", alert_content_digest(c))]
    assert _bounds(v_prev) == (D - 1800 + 20, D - 1740 + 20, D - 40, None, None, [])
    today = s.versions()
    assert ("E", alert_content_digest(early)) not in today
    assert _bounds(today["C", alert_content_digest(c)]) == (
        D - 1800 + 20, D - 1740 + 20, D + 600 + 20, D + 660 + 20, "removed", []
    )  # fmt: skip
    # The next day's presences change nothing in this day's index.
    s.poll(D + 86400 + 20, c)
    assert s.versions() == today


def _no_sleep(_: float) -> None:
    return None


def test_a_listed_key_r2_cannot_return_is_a_missing_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("training.r2_client.time.sleep", _no_sleep)
    objects = {
        p.relative_to(RECORDED).as_posix(): p.read_bytes()
        for p in RECORDED.rglob("*")
        if p.is_file()
    }
    blob = next(k for k in objects if "/raw/blob/" in k)
    del objects[blob]
    r2 = R2Storage(_VanishingS3(objects, blob), "bucket")  # pyright: ignore[reportArgumentType]
    report = derive(r2, r2, DAY, *CODE)
    assert [p.key for p in report.missing_inputs] == [blob]
    assert report.records["vehicle"].rows
