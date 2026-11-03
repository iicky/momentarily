"""The legacy backfill over a range of days and over R2 (a fake S3 client):
per-day files whatever the range, resume, the overwrite rule, the
conditional put, and reads that go one day at a time."""

from __future__ import annotations

import io
import json
import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

from momentarily.model.observations.derive.storage import LocalStorage
from momentarily.model.observations.keys import (
    DerivedDayKey,
    DerivedHourKey,
    utc_day,
)
from momentarily.model.observations.legacy.__main__ import main as local_main
from momentarily.model.observations.legacy.run import Report, run
from training.derive_observations import R2Storage
from training.legacy_backfill import main as r2_main
from training.legacy_backfill import unsettled

REPO = Path(__file__).resolve().parent.parent
MIRROR = REPO / "tests" / "fixtures" / "model" / "observations" / "legacy_mirror"
CODE = ("f" * 40, False)
DERIVED = "model/observations/s1/derived/"

TRACE = "archive/trace/2026-09-15/1789477995.json"
DAY, NEXT = "2026-09-15", "2026-09-16"
MIDNIGHT = 1789516800  # 2026-09-16T00:00:00Z


def _fixture() -> dict[str, bytes]:
    return {
        p.relative_to(MIRROR).as_posix(): p.read_bytes()
        for p in sorted(MIRROR.rglob("*"))
        if p.is_file()
    }


def _trace(scheduled: int, observed: int) -> tuple[str, bytes]:
    """The fixture trace object scheduled and observed at other times."""
    doc = json.loads((MIRROR / TRACE).read_bytes())
    doc |= {"scheduled_at": scheduled, "observed_at": observed}
    return f"archive/trace/{utc_day(scheduled)}/{scheduled}.json", json.dumps(
        doc
    ).encode()


def _midnight() -> dict[str, bytes]:
    """The fixture plus a trace minute scheduled 20 s before midnight and
    observed 10 s after it, and one at 00:05."""
    objects = _fixture()
    for scheduled, observed in ((MIDNIGHT - 20, MIDNIGHT + 10), (MIDNIGHT + 300,) * 2):
        key, body = _trace(scheduled, observed)
        objects[key] = body
    return objects


class _FakeS3:
    """The S3 calls R2Storage makes, over a dict, with conditional puts; it
    logs every list, get and put."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = dict(objects)
        self.log: list[tuple[str, str]] = []

    def get_paginator(self, name: str) -> _FakeS3:
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket: str, Prefix: str) -> Iterator[dict[str, Any]]:
        self.log.append(("list", Prefix))
        keys = sorted(k for k in self.objects if k.startswith(Prefix))
        for i in range(0, max(len(keys), 1), 2):
            yield {"Contents": [{"Key": k} for k in keys[i : i + 2]]} if keys else {}

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        self.log.append(("get", Key))
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self.objects[Key])}

    def put_object(
        self,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
        IfNoneMatch: str | None = None,
    ) -> None:
        self.log.append(("put", Key))
        if IfNoneMatch == "*" and Key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body

    def derived(self) -> dict[str, bytes]:
        return {k: v for k, v in self.objects.items() if k.startswith(DERIVED)}


def _r2(fake: _FakeS3) -> R2Storage:
    return R2Storage(fake, "bucket")  # pyright: ignore[reportArgumentType]


def _backfill(fake: _FakeS3, *days: str, **kw: Any) -> Report:
    store = _r2(fake)
    first, last = (days[0], days[-1]) if days else (None, None)
    return run(store, store, *CODE, first=first, last=last, readers=4, **kw)


def _local(root: Path) -> dict[str, bytes]:
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_r2_gives_the_local_bytes_on_the_fixture(tmp_path: Path) -> None:
    run(LocalStorage(MIRROR), LocalStorage(tmp_path), *CODE)
    fake = _FakeS3(_fixture())
    report = _backfill(fake)
    assert not report.failed
    assert fake.derived() == _local(tmp_path)


def test_a_day_s_files_are_the_same_whichever_range_covers_it() -> None:
    whole = _FakeS3(_midnight())
    assert not _backfill(whole, DAY, NEXT).failed
    by_day = _FakeS3(_midnight())
    first = _backfill(by_day, DAY)
    # The minute observed after midnight fills the next day's files, which a
    # run of DAY alone leaves to the run of NEXT.
    deferred = {
        DerivedHourKey("vehicle", MIDNIGHT).key,
        DerivedDayKey("legacy_fetch_outcome", NEXT).key,
    }
    assert set(first.deferred) == deferred
    assert not deferred & set(by_day.derived())
    second = _backfill(by_day, NEXT)
    assert not second.failed
    assert second.deferred == []
    assert by_day.derived() == whole.derived()
    assert [d.day for d in first.days] == [DAY]


def test_a_range_writes_only_its_own_days() -> None:
    fake = _FakeS3(_fixture())
    report = _backfill(fake, "2026-08-12", "2026-09-11")
    days = {_day_of(k) for k in fake.derived()}
    assert days == {"2026-08-12", "2026-09-02", "2026-09-11"}
    assert (report.days[0].day, report.days[-1].day) == ("2026-08-12", "2026-09-11")
    assert len(report.days) == 31
    assert sum(d.written for d in report.days) == len(fake.derived())


def _late() -> dict[str, bytes]:
    """The midnight fixture plus a late invocation: scheduled 22:52, 68
    minutes before midnight, and observed 00:05 the next day (the Worker's
    missed crons of 2026-10-25)."""
    objects = _midnight()
    key, body = _trace(MIDNIGHT - 68 * 60, MIDNIGHT + 300 + 5)
    objects[key] = body
    return objects


def test_a_late_trace_minute_fills_the_next_day_whichever_range_covers_it() -> None:
    whole = _FakeS3(_late())
    assert not _backfill(whole, DAY, NEXT).failed
    outcome = DerivedDayKey("legacy_fetch_outcome", NEXT).key
    assert outcome in whole.derived()
    alone = _FakeS3(_late())
    assert not _backfill(alone, NEXT).failed
    assert alone.derived() == {
        k: v for k, v in whole.derived().items() if _day_of(k) == NEXT
    }
    split = _FakeS3(_late())
    assert not _backfill(split, DAY).failed
    assert not _backfill(split, NEXT).failed
    assert split.derived() == whole.derived()
    # Without the late minute the next day's outcomes differ: it is in them.
    plain = _FakeS3(_midnight())
    _backfill(plain, NEXT)
    assert plain.derived()[outcome] != whole.derived()[outcome]


def test_a_trace_minute_observed_at_the_end_of_the_next_day_is_read() -> None:
    objects = _midnight()
    key, body = _trace(MIDNIGHT - 68 * 60, MIDNIGHT + 86400 - 1)
    objects[key] = body
    whole = _FakeS3(objects)
    assert not _backfill(whole, DAY, NEXT).failed
    alone = _FakeS3(objects)
    assert not _backfill(alone, NEXT).failed
    assert alone.derived() == {
        k: v for k, v in whole.derived().items() if _day_of(k) == NEXT
    }


def test_a_trace_minute_reaching_past_the_next_day_stops_the_run() -> None:
    objects = _fixture()
    key, body = _trace(MIDNIGHT - 7200, MIDNIGHT + 86400 + 10)
    objects[key] = body
    with pytest.raises(RuntimeError, match="lead-in"):
        _backfill(_FakeS3(objects), DAY, NEXT)


def test_a_lead_in_minute_reaching_past_the_first_day_stops_the_run() -> None:
    """A range's lead-in object whose records reach past the range's first
    day: no range reads it there."""
    objects = _fixture()
    after = MIDNIGHT + 86400  # 2026-09-17T00:00:00Z
    key, body = _trace(after + 86400 - 3600, after + 2 * 86400 + 10)
    objects[key] = body
    assert key.startswith("archive/trace/2026-09-17/")
    with pytest.raises(RuntimeError, match="lead-in"):
        _backfill(_FakeS3(objects), "2026-09-18")


def test_a_rerun_skips_every_file_and_puts_nothing() -> None:
    fake = _FakeS3(_midnight())
    _backfill(fake, DAY, NEXT)
    before = fake.derived()
    fake.log.clear()
    report = _backfill(fake, DAY, NEXT)
    assert not report.failed
    assert report.files == []
    assert sorted(report.unchanged) == sorted(before)
    assert not [op for op in fake.log if op[0] == "put"]
    assert fake.derived() == before


def test_a_file_built_by_other_code_is_unchanged() -> None:
    fake = _FakeS3(_midnight())
    _backfill(fake, DAY, NEXT)
    before = fake.derived()
    fake.log.clear()
    store = _r2(fake)
    report = run(store, store, "e" * 40, True, first=DAY, last=NEXT)
    assert not report.failed
    assert sorted(report.unchanged) == sorted(before)
    assert not [op for op in fake.log if op[0] == "put"]
    assert fake.derived() == before


@pytest.mark.parametrize("overwrite", [False, True])
def test_a_legacy_file_with_other_content_is_replaced_only_on_request(
    overwrite: bool,
) -> None:
    fake = _FakeS3(_fixture())
    _backfill(fake, "2026-09-17")
    key = DerivedDayKey("legacy_collector_health", "2026-09-17").key
    old = fake.objects[key]
    health = "archive/health/2026-09-17/1789682655.json"
    doc = json.loads(fake.objects[health])
    doc["write_failures"] = {"arrivals_publish": 2}
    fake.objects[health] = json.dumps(doc).encode()
    report = _backfill(fake, "2026-09-17", overwrite=overwrite)
    assert report.failed is not overwrite
    if overwrite:
        assert report.replaced == [key]
        assert fake.objects[key] != old
    else:
        assert report.conflicts == [(key, report.conflicts[0][1])]
        assert "differs" in report.conflicts[0][1]
        assert fake.objects[key] == old


@pytest.mark.parametrize("overwrite", [False, True])
def test_an_unreadable_file_is_never_replaced(overwrite: bool) -> None:
    fake = _FakeS3(_fixture())
    key = DerivedDayKey("legacy_collector_health", "2026-09-17").key
    fake.objects[key] = b"not parquet"
    report = _backfill(fake, "2026-09-17", overwrite=overwrite)
    assert [k for k, _ in report.conflicts] == [key]
    assert fake.objects[key] == b"not parquet"


class _Racing(_FakeS3):
    """Another run writes ``body`` under the first key this one creates,
    just before this one's conditional put arrives."""

    def __init__(self, objects: dict[str, bytes], body: bytes | None) -> None:
        super().__init__(objects)
        self.body = body
        self.raced: str | None = None

    def put_object(
        self,
        Bucket: str,
        Key: str,
        Body: bytes,
        ContentType: str,
        IfNoneMatch: str | None = None,
    ) -> None:
        if self.raced is None and IfNoneMatch == "*":
            self.raced = Key
            self.objects[Key] = Body if self.body is None else self.body
        super().put_object(Bucket, Key, Body, ContentType, IfNoneMatch)


@pytest.mark.parametrize("same", [False, True])
def test_the_conditional_put_never_replaces_a_file_written_meanwhile(
    same: bool,
) -> None:
    fake = _Racing(_fixture(), None if same else b"theirs")
    report = _backfill(fake, "2026-09-17", overwrite=True)
    assert fake.raced is not None
    if same:
        assert not report.failed
        assert fake.raced in report.unchanged
    else:
        assert report.conflicts == [(fake.raced, "written by another run meanwhile")]
        assert fake.objects[fake.raced] == b"theirs"


def _day_of(key: str) -> str | None:
    m = re.search(r"\d{4}-\d{2}-\d{2}", key)
    return m[0] if m else None


def test_reads_go_one_day_at_a_time() -> None:
    """Each day's objects are listed under the day's folder, and read only
    after every earlier day's files were put; ``archive/alerts`` alone is
    listed whole, before any day."""
    fake = _FakeS3(_midnight())
    _backfill(fake, "2026-08-12", NEXT)
    lists = [p for op, p in fake.log if op == "list" and not p.startswith(DERIVED)]
    assert lists[0] == "archive/alerts/"
    assert all(_day_of(p) is not None and p.endswith("/") for p in lists[1:])
    last_put: dict[str, int] = {}
    for i, (op, key) in enumerate(fake.log):
        if op == "put":
            last_put[_day_of(key) or ""] = i
    for i, (op, key) in enumerate(fake.log):
        day = _day_of(key)
        if op == "get" and not key.startswith(("archive/alerts/", DERIVED)):
            assert day is not None
            assert all(at < i for d, at in last_put.items() if d < day)


def test_the_local_cli_resumes_and_takes_a_range(tmp_path: Path) -> None:
    args = ["--mirror", str(MIRROR), "--out", str(tmp_path)]
    assert local_main([*args, "--from", "2026-09-17", "--through", "2026-09-17"]) == 0
    assert {k.split("/")[-1] for k in _local(tmp_path)} == {"2026-09-17.parquet"}
    assert local_main(args) == 0
    whole = tmp_path / "whole"
    assert local_main(["--mirror", str(MIRROR), "--out", str(whole)]) == 0
    assert _local(tmp_path / DERIVED) == _local(whole / DERIVED)
    assert local_main([*args, "--from", "2026-09-18", "--through", "2026-09-17"]) == 2


def test_the_r2_cli_refuses_a_day_that_has_not_ended() -> None:
    assert r2_main(["--from", "2026-10-01", "--through", "2999-01-01"]) == 2


def test_the_last_day_settles_a_full_day_after_it_ends() -> None:
    at = datetime(2026, 11, 2, 0, 0, tzinfo=UTC)
    assert unsettled("2026-10-31", at) is None
    late = unsettled("2026-10-31", at - timedelta(seconds=1))
    assert late is not None
    assert "2026-11-02T00:00Z" in late
    assert unsettled("2026-11-01", at) is not None


class _Missing(_FakeS3):
    """Lists ``missing`` but answers NoSuchKey for it ``times`` times (every
    time when None), as R2 does spuriously under concurrent GETs."""

    def __init__(
        self, objects: dict[str, bytes], missing: str, times: int | None
    ) -> None:
        super().__init__(objects)
        self.missing, self.times = missing, times

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        if Key == self.missing and (self.times is None or self.times > 0):
            if self.times is not None:
                self.times -= 1
            self.log.append(("get", Key))
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return super().get_object(Bucket, Key)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_wait(_: float) -> None:
        pass

    monkeypatch.setattr("training.r2_client.time.sleep", no_wait)


@pytest.mark.usefixtures("no_sleep")
def test_a_spurious_not_found_is_retried() -> None:
    whole = _FakeS3(_midnight())
    _backfill(whole, DAY, NEXT)
    fake = _Missing(_midnight(), TRACE, times=2)
    report = _backfill(fake, DAY, NEXT)
    assert not report.failed
    assert fake.derived() == whole.derived()


@pytest.mark.usefixtures("no_sleep")
def test_an_unreadable_trace_object_aborts_its_day_and_the_next() -> None:
    """The next day's lead-in needs every trace object of the day, so it is
    aborted too; the day after starts afresh."""
    after = "2026-09-17"
    whole = _FakeS3(_midnight())
    _backfill(whole, DAY, after)
    fake = _Missing(_midnight(), TRACE, times=None)
    report = _backfill(fake, DAY, after)
    assert report.failed
    assert report.aborted == [(DAY, TRACE), (NEXT, TRACE)]
    assert [d.aborted for d in report.days] == [TRACE, TRACE, None]
    assert report.archives["archive/trace"].failures == []
    assert {_day_of(k) for k in fake.derived()} == {after}
    assert fake.derived() == {
        k: v for k, v in whole.derived().items() if _day_of(k) == after
    }


@pytest.mark.usefixtures("no_sleep")
def test_an_unreadable_alert_sighting_writes_nothing() -> None:
    key = "archive/alerts/2026-07-22/1784592096-lmm_alert_556649.json"
    fake = _Missing(_fixture(), key, times=None)
    report = _backfill(fake, "2026-07-21", "2026-07-22")
    assert report.aborted == [("archive/alerts", key)]
    assert report.failed
    assert fake.derived() == {}


def _late_malformed() -> tuple[dict[str, bytes], str]:
    """The late fixture with its 22:52 trace minute, which fills the next
    day's files, replaced by bytes that are not JSON."""
    objects = _late()
    key = f"archive/trace/{DAY}/{MIDNIGHT - 68 * 60}.json"
    objects[key] = b"{not-json"
    return objects, key


@pytest.mark.parametrize("days", [(NEXT,), (DAY, NEXT)])
def test_a_malformed_trace_object_aborts_every_day_that_reads_it(
    days: tuple[str, ...],
) -> None:
    """Read in-day or only as the next day's lead-in, it may hold rows of
    both days, so neither is written without it, and the run fails."""
    after = "2026-09-17"
    objects, key = _late_malformed()
    fake = _FakeS3(objects)
    report = _backfill(fake, *days, after)
    assert report.failed
    assert report.aborted == [(day, key) for day in days]
    assert [k for k, _ in report.archives["archive/trace"].failures] == [key]
    whole = _FakeS3(_late())
    _backfill(whole, *days, after)
    assert fake.derived() == {
        k: v for k, v in whole.derived().items() if _day_of(k) == after
    }


def test_a_malformed_alert_sighting_writes_nothing() -> None:
    key = "archive/alerts/2026-07-22/1784592096-lmm_alert_556649.json"
    objects = _fixture()
    objects[key] = b"{not-json"
    fake = _FakeS3(objects)
    report = _backfill(fake, "2026-07-21", "2026-07-22")
    assert report.aborted == [("archive/alerts", key)]
    assert [k for k, _ in report.archives["archive/alerts"].failures] == [key]
    assert fake.derived() == {}


def test_without_a_last_day_the_files_after_the_last_held_day_are_written(
    tmp_path: Path,
) -> None:
    """A trace minute of the last held day observed after midnight: the
    default run writes the next day's files, as the whole-mirror run always
    has; a run through that day defers them."""
    mirror = tmp_path / "mirror"
    for key, body in _fixture().items():
        (mirror / key).parent.mkdir(parents=True, exist_ok=True)
        (mirror / key).write_bytes(body)
    late = 1791935980  # 2026-10-13T23:59:40Z, the last day the mirror holds
    key, body = _trace(late, late + 30)
    (mirror / key).parent.mkdir(parents=True, exist_ok=True)
    (mirror / key).write_bytes(body)
    after = {
        DerivedHourKey("vehicle", late + 20).key,
        DerivedDayKey("legacy_fetch_outcome", "2026-10-14").key,
    }
    whole = run(LocalStorage(mirror), LocalStorage(tmp_path / "a"), *CODE)
    assert whole.deferred == []
    assert after <= set(whole.files)
    ranged = run(
        LocalStorage(mirror), LocalStorage(tmp_path / "b"), *CODE, last="2026-10-13"
    )
    assert sorted(ranged.deferred) == sorted(after)


def test_local_writes_never_replace_a_new_key(tmp_path: Path) -> None:
    store = LocalStorage(tmp_path)
    assert store.put_new("a/b", b"1")
    assert not store.put_new("a/b", b"2")
    assert store.get("a/b") == b"1"
    assert store.list("a/") == ["a/b"]
