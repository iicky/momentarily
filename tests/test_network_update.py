"""Extending the day->feed map from recorder evidence, building new feeds, and
publishing without replacing different bytes."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from botocore.exceptions import ClientError

from momentarily.model.network.contracts import (
    DayFeed,
    DayFeedMap,
    IndexEntry,
    NetworkIndex,
    NetworkSnapshot,
)
from momentarily.model.network.stations import digest_of
from momentarily.model.network.update import (
    BY_ETAG_PREFIX,
    DAY_FEED_MAP_KEY,
    GTFS_PREFIX,
    INDEX_KEY,
    NETWORK_PREFIX,
    TICKS_PREFIX,
    PublishConflictError,
    UpdateReport,
    publish,
    run,
    update,
)
from momentarily.model.observations.contracts import CollectorTick
from scripts.gen_model_network_fixture import (
    FIXTURE_CODE,
    GTFS_DIR,
    STATIONS_PATH,
    zip_dir,
)
from training.network_update import NetworkR2Storage

LAST_DAY = "2026-10-18"


class MemStore:
    """A bucket as a dict."""

    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = objects if objects is not None else {}

    def list(self, prefix: str) -> list[str]:
        return sorted(k for k in self.objects if k.startswith(prefix))

    def get(self, key: str) -> bytes:
        return self.objects[key]

    def put(self, key: str, data: bytes) -> None:
        self.objects[key] = data

    def delete(self, key: str) -> None:
        self.objects.pop(key, None)


def _feed(tmp_path: Path, version: str) -> bytes:
    """The fixture feed under another version. Its two-way platform gets an N
    suffix, so it builds from the stations dataset alone, as the NYCT feed
    does (the nightly build uses no overrides)."""
    out = tmp_path / version
    out.mkdir()
    for src in GTFS_DIR.glob("*.txt"):
        text = src.read_text().replace("106M", "106N")
        if src.name == "feed_info.txt":
            text = text.replace("20260601-fixture", version)
        (out / src.name).write_text(text)
    return zip_dir(out)


def _start(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp())


def _tick(at: int, etag: str | None) -> tuple[str, bytes]:
    ms = at * 1000 + 500
    doc: dict[str, Any] = {
        "schema": "observations/s1",
        "tick_id": f"{at}-{ms}",
        "scheduled_at": at,
        "started_at_ms": ms,
        "finished_at_ms": ms + 100,
        "collector": {"code_sha": "c" * 40, "code_dirty": False},
        "planned_sources": ["gtfs_static"],
        "fetches": [
            {
                "fetch_id": f"gtfs_static@{ms}",
                "source": "gtfs_static",
                "url": "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip",
                "method": "HEAD",
                "scheduled_at": at,
                "started_at_ms": ms,
                "finished_at_ms": ms + 10,
                "outcome": "ok" if etag else "http_error",
                "http_status": 200 if etag else 503,
                "content_type": "application/zip",
                "http_date": at,
                "http_last_modified": at - 86_400,
                "http_etag": etag,
                "body_bytes": None,
                "payload_sha256": None,
                "payload_key": None,
                "payload_member": None,
                "header_timestamp": None,
                "entity_count": None,
                "store_error": None,
            }
        ],
    }
    data = json.dumps(doc).encode()
    CollectorTick.model_validate_json(data)
    hour = datetime.fromtimestamp(at, UTC).strftime("%Y-%m-%d/%H")
    return f"{TICKS_PREFIX}{hour}/{at}-{ms}.json", data


def _record(store: MemStore, day: str, etags: Sequence[str | None]) -> None:
    """One tick a minute from midnight, the n-th carrying ``etags[n]``."""
    for i, etag in enumerate(etags):
        key, data = _tick(_start(day) + 60 * i, etag)
        store.put(key, data)


class World:
    """Feed A built and mapped through LAST_DAY; feed B captured, not built."""

    def __init__(self, tmp_path: Path) -> None:
        self.store = MemStore()
        self.a = _feed(tmp_path, "v-a")
        self.b = _feed(tmp_path, "v-b")
        self.fa = hashlib.sha256(self.a).hexdigest()
        self.fb = hashlib.sha256(self.b).hexdigest()
        for blob, fid, etag in ((self.a, self.fa, "ea"), (self.b, self.fb, "eb")):
            self.store.put(f"{GTFS_PREFIX}{fid}.zip", blob)
            self.capture(etag, fid)
        stations = STATIONS_PATH.read_bytes()
        self.store.put(
            f"{NETWORK_PREFIX}stations-{digest_of(stations)[:16]}.json", stations
        )
        day_map = DayFeedMap(
            days={
                LAST_DAY: DayFeed(
                    feed_id=self.fa,
                    verified=True,
                    reason=None,
                    feed_version="v-a",
                    inferred_feed_id=None,
                )
            }
        )
        index = NetworkIndex(
            schema="network/s1",
            feeds={
                self.fa: IndexEntry(
                    build_id="0" * 16,
                    content_digest="0" * 64,
                    feed_version="v-a",
                    valid_from=None,
                    valid_to=None,
                    inputs={},
                    applied_retroactively=[],
                )
            },
        )
        self.store.put(DAY_FEED_MAP_KEY, _dump(day_map.model_dump(mode="json")))
        self.store.put(INDEX_KEY, _dump(index.model_dump(mode="json")))

    def capture(self, etag: str, feed: str) -> None:
        self.store.put(
            f"{BY_ETAG_PREFIX}{etag}.json", json.dumps({"digest": feed}).encode()
        )

    def update(self, through: str) -> UpdateReport:
        return update(self.store, self.store, through, FIXTURE_CODE)


def _dump(doc: object) -> bytes:
    return (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode()


FULL = 1440


def _q(etag: str) -> str:
    return f'"{etag}"'


def test_a_day_on_one_feed_is_verified(tmp_path: Path) -> None:
    w = World(tmp_path)
    _record(w.store, "2026-10-19", [_q("ea")] * FULL)
    day = w.update("2026-10-19").days["2026-10-19"]
    assert (day.feed_id, day.verified, day.feed_version) == (w.fa, True, "v-a")


def test_a_mid_day_change_names_no_feed_and_a_new_feed_is_built(
    tmp_path: Path,
) -> None:
    w = World(tmp_path)
    _record(w.store, "2026-10-19", [_q("ea")] * 720 + [_q("eb")] * 720)
    _record(w.store, "2026-10-20", [_q("eb")] * FULL)
    report = w.update("2026-10-20")

    changed = report.days["2026-10-19"]
    assert (changed.feed_id, changed.verified, changed.inferred_feed_id) == (
        None,
        False,
        None,
    )
    assert report.days["2026-10-20"].feed_id == w.fb

    index = NetworkIndex.model_validate_json(w.store.get(INDEX_KEY))
    assert set(index.feeds) == {w.fa, w.fb}
    entry = index.feeds[w.fb]
    assert entry.feed_version == "v-b"
    build = NetworkSnapshot.model_validate_json(
        gzip.decompress(w.store.get(f"{NETWORK_PREFIX}{entry.build_id}.json.gz"))
    )
    assert build.feed.feed_id == w.fb
    assert build.provenance.observed_from_utc_day == "2026-10-20"
    assert build.provenance.inputs["day_feed_map"] == digest_of(
        w.store.get(DAY_FEED_MAP_KEY)
    )


def test_a_day_without_ticks_is_unresolved(tmp_path: Path) -> None:
    w = World(tmp_path)
    _record(w.store, "2026-10-20", [_q("ea")] * FULL)
    report = w.update("2026-10-20")
    empty = report.days["2026-10-19"]
    assert (empty.feed_id, empty.verified, empty.inferred_feed_id) == (
        None,
        False,
        None,
    )
    assert empty.reason
    assert report.days["2026-10-20"].verified


def test_a_day_covered_only_in_part_keeps_a_lead(tmp_path: Path) -> None:
    w = World(tmp_path)
    _record(w.store, "2026-10-19", [None] * 240 + [_q("ea")] * 600)
    day = w.update("2026-10-19").days["2026-10-19"]
    assert (day.feed_id, day.verified, day.inferred_feed_id) == (None, False, w.fa)


def test_an_etag_change_resolves_through_the_capture_record(tmp_path: Path) -> None:
    w = World(tmp_path)
    # Same bytes re-served under a new ETag: still one feed, nothing to build.
    w.capture("ea2", w.fa)
    _record(w.store, "2026-10-19", [_q("ea")] * 720 + [_q("ea2")] * 720)
    # A new ETag with no captured zip: the day cannot be named.
    _record(w.store, "2026-10-20", [_q("ea")] * 720 + [_q("ex")] * 720)
    report = w.update("2026-10-20")
    same = report.days["2026-10-19"]
    assert (same.feed_id, same.verified) == (w.fa, True)
    uncaptured = report.days["2026-10-20"]
    assert (uncaptured.feed_id, uncaptured.inferred_feed_id) == (None, None)
    assert not report.built


def test_rerun_is_byte_identical_and_keeps_existing_days(tmp_path: Path) -> None:
    w = World(tmp_path)
    before_day: Any = json.loads(w.store.get(DAY_FEED_MAP_KEY))["days"][LAST_DAY]
    _record(w.store, "2026-10-19", [_q("ea")] * 720 + [_q("eb")] * 720)
    _record(w.store, "2026-10-20", [_q("eb")] * FULL)
    w.update("2026-10-20")
    first = dict(w.store.objects)

    again = w.update("2026-10-20")
    assert w.store.objects == first
    assert {action for _, action in again.published} == {"unchanged"}
    assert not again.days
    after: Any = json.loads(w.store.get(DAY_FEED_MAP_KEY))
    assert after["days"][LAST_DAY] == before_day


def test_publish_refuses_to_rewrite_a_mapped_day(tmp_path: Path) -> None:
    w = World(tmp_path)
    doc: Any = json.loads(w.store.get(DAY_FEED_MAP_KEY))
    doc["days"][LAST_DAY]["feed_id"] = w.fb
    doc["days"]["2026-10-19"] = doc["days"][LAST_DAY]
    before = dict(w.store.objects)
    with pytest.raises(PublishConflictError):
        publish(w.store, [(DAY_FEED_MAP_KEY, _dump(doc))])
    assert w.store.objects == before


class _FakeS3:
    """The S3 calls NetworkR2Storage makes, over a dict. Keys in ``unlisted``
    exist but are not yet visible to a listing or a GET, as an object written
    by another run after this one checked."""

    def __init__(self, objects: dict[str, bytes], unlisted: set[str]) -> None:
        self.objects = objects
        self.unlisted = unlisted

    def get_paginator(self, name: str) -> _FakeS3:
        assert name == "list_objects_v2"
        return self

    def paginate(self, Bucket: str, Prefix: str) -> Iterator[dict[str, Any]]:
        keys = sorted(
            k for k in self.objects if k.startswith(Prefix) and k not in self.unlisted
        )
        yield {"Contents": [{"Key": k} for k in keys]}

    def get_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        if Key not in self.objects or Key in self.unlisted:
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
        if IfNoneMatch == "*" and Key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body


@pytest.mark.parametrize("visible", [True, False])
def test_the_r2_upload_never_replaces_different_bytes(
    tmp_path: Path, visible: bool
) -> None:
    w = World(tmp_path)
    _record(w.store, "2026-10-19", [_q("eb")] * FULL)
    objects = dict(w.store.objects)
    build_id = w.update("2026-10-19").built[w.fb]

    # Something else sits at the new build's key: seen by the check, or
    # written after it (only the conditional put can catch that one).
    build_key = f"{NETWORK_PREFIX}{build_id}.json.gz"
    objects[build_key] = b"other bytes"
    before = dict(objects)
    fake = _FakeS3(objects, set() if visible else {build_key})
    r2 = NetworkR2Storage(fake, "bucket")  # pyright: ignore[reportArgumentType]
    assert run(r2, r2, "2026-10-19", FIXTURE_CODE) == 1
    assert objects == before


def test_a_capture_that_is_not_a_zip_leaves_its_days_unresolved(
    tmp_path: Path,
) -> None:
    w = World(tmp_path)
    junk = b"not a zip"
    fj = hashlib.sha256(junk).hexdigest()
    w.store.put(f"{GTFS_PREFIX}{fj}.zip", junk)
    w.capture("ej", fj)
    _record(w.store, "2026-10-19", [_q("ej")] * FULL)
    _record(w.store, "2026-10-20", [_q("ea")] * FULL)

    assert run(w.store, w.store, "2026-10-20", FIXTURE_CODE) == 1
    days = DayFeedMap.model_validate_json(w.store.get(DAY_FEED_MAP_KEY)).days
    assert (days["2026-10-19"].feed_id, days["2026-10-19"].inferred_feed_id) == (
        None,
        None,
    )
    assert days["2026-10-20"].feed_id == w.fa


def test_a_feed_that_fails_to_build_still_extends_the_map(tmp_path: Path) -> None:
    w = World(tmp_path)
    broken_dir = tmp_path / "broken"
    broken_dir.mkdir()
    for src in GTFS_DIR.glob("*.txt"):
        (broken_dir / src.name).write_text(src.read_text().replace("106M", "106N"))
    (broken_dir / "agency.txt").write_text(
        "agency_id,agency_name,agency_url\nMTA NYCT,MTA,http://www.mta.info\n"
    )
    broken = zip_dir(broken_dir)
    fc = hashlib.sha256(broken).hexdigest()
    w.store.put(f"{GTFS_PREFIX}{fc}.zip", broken)
    w.capture("ec", fc)
    _record(w.store, "2026-10-19", [_q("ec")] * FULL)
    _record(w.store, "2026-10-20", [_q("eb")] * FULL)

    assert run(w.store, w.store, "2026-10-20", FIXTURE_CODE) == 1
    days = DayFeedMap.model_validate_json(w.store.get(DAY_FEED_MAP_KEY)).days
    assert days["2026-10-19"].feed_id == fc
    index = NetworkIndex.model_validate_json(w.store.get(INDEX_KEY))
    assert set(index.feeds) == {w.fa, w.fb}
