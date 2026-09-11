"""The content-addressed static-feed store (training/gtfs_archive.py).

A version label cannot rebuild a timetable, and MTA serves no superseded
snapshots, so replaying a historical grade needs the feed ARTIFACT. These cases
pin the addressing, the corruption check, and the rule that a digest is only ever
recorded when it can be proven — never inferred from whatever feed is current.
"""

from __future__ import annotations

import hashlib
from datetime import date

from training.gtfs_archive import (
    BY_ETAG_PREFIX,
    LATEST_KEY,
    digest_of,
    etag_key,
    key_for,
    store,
)
from training.traversal_archive import (
    EXTRACTOR_VERSION,
    SCHEMA_VERSION,
    DayProvenance,
    ReadResult,
)


def test_the_key_is_the_content_hash():
    """Content addressing is what makes the nightly store idempotent: an
    unchanged feed writes nothing, and a republish cannot overwrite the snapshot
    that priced last month's hops."""
    blob = b"not really a zip"
    sha = hashlib.sha256(blob).hexdigest()
    assert digest_of(blob) == sha
    assert key_for(sha) == f"archive/gtfs/{sha}.zip"


def test_different_bytes_never_share_a_key():
    """Two feeds can carry the same `feed_version` label; only the bytes decide
    what a scheduled time was."""
    assert digest_of(b"feed-a") != digest_of(b"feed-b")


def _prov(*, digest: str | None) -> DayProvenance:
    return DayProvenance(
        schema=SCHEMA_VERSION,
        extractor=EXTRACTOR_VERSION,
        feed_version="20260807",
        feed_digest=digest,
        n_rows=1,
        n_source_objects=1,
        source_manifest="x",
        code_sha="deadbeef",
        written_at=0,
    )


def test_days_with_and_without_a_proven_digest_do_not_pool_silently():
    """The rule this store exists to protect.

    A backfill can only fetch the CURRENT feed, so it cannot know which bytes
    were live weeks ago and records None. A day that DOES carry a proven digest
    is making a stronger claim, and pooling the two without comment would let an
    unverified day ride on a verified one's provenance. `feed_digest` is part of
    the comparability identity precisely so that cannot happen quietly.
    """
    mixed = ReadResult(
        traversals=[],
        provenance={
            date(2026, 8, 12): _prov(digest=None),
            date(2026, 8, 13): _prov(digest="abc123"),
        },
    )
    assert not mixed.homogeneous

    both_unknown = ReadResult(
        traversals=[],
        provenance={
            date(2026, 8, 12): _prov(digest=None),
            date(2026, 8, 13): _prov(digest=None),
        },
    )
    assert both_unknown.homogeneous


def test_two_different_proven_digests_are_a_version_boundary():
    """Same version label, different bytes: a republish between two days changes
    what the scheduled reference was, and the digest is the only field that can
    see it."""
    result = ReadResult(
        traversals=[],
        provenance={
            date(2026, 8, 12): _prov(digest="aaa"),
            date(2026, 8, 20): _prov(digest="bbb"),
        },
    )
    assert not result.homogeneous


# --- store() writes the latest-digest pointer --------------------------------


class _FakeClient:
    """Minimal S3Client stub that records put_object calls in order."""

    def __init__(self, existing_keys: set[str] | None = None) -> None:
        self.puts: list[str] = []  # keys in write order
        self.objects: dict[str, bytes | str] = {}
        self._existing = existing_keys or set()

    def put_object(self, *, Bucket: str, Key: str, Body: object, **_: object) -> None:
        self.puts.append(Key)
        self.objects[Key] = Body if isinstance(Body, (bytes, str)) else b""

    def list_objects_v2(self, **_: object) -> dict[str, object]:
        return {"Contents": [{"Key": k} for k in self._existing]}


class _FakeConfig:
    bucket = "test-bucket"
    endpoint_url = ""
    access_key_id = ""
    secret_access_key = ""


def test_store_writes_zip_before_latest_pointer() -> None:
    """The pointer must never name a digest whose artifact hasn't landed yet."""
    from .conftest import make_gtfs_bytes

    blob = make_gtfs_bytes(["A,t1,Weekday,X,0,s1"], ["t1,s1N,00:00:00,00:00:00,1"])
    client = _FakeClient()
    snap = store(blob, etag='"e-1"', config=_FakeConfig(), client=client)  # type: ignore[arg-type]

    assert snap.stored is True
    # The zip is written FIRST, the by_etag mapping SECOND, the pointer LAST.
    assert len(client.puts) == 3
    assert client.puts[0] == key_for(snap.digest)
    assert client.puts[1] == etag_key('"e-1"')
    assert client.puts[2] == LATEST_KEY

    # Pointer carries the correct digest
    import json

    latest = json.loads(client.objects[LATEST_KEY])
    assert latest["digest"] == snap.digest
    assert latest["version"] == snap.version
    assert "stored_at" in latest


def test_store_refreshes_latest_pointer_on_idempotent_run() -> None:
    """When the artifact already exists, the zip is NOT re-uploaded but the
    pointer and the by_etag mapping ARE refreshed — a prior run may have
    stored the artifact but crashed before writing either."""
    from .conftest import make_gtfs_bytes

    blob = make_gtfs_bytes(["A,t1,Weekday,X,0,s1"], ["t1,s1N,00:00:00,00:00:00,1"])
    sha = digest_of(blob)
    existing = {key_for(sha)}  # artifact already in the bucket
    client = _FakeClient(existing_keys=existing)
    snap = store(blob, etag='"e-1"', config=_FakeConfig(), client=client)  # type: ignore[arg-type]

    assert snap.stored is False  # no new zip upload
    # Only the by_etag mapping and the pointer were written, not the zip
    assert client.puts == [etag_key('"e-1"'), LATEST_KEY]

    import json

    latest = json.loads(client.objects[LATEST_KEY])
    assert latest["digest"] == sha


def test_store_records_etag_in_latest_and_by_etag() -> None:
    """latest.json carries the ETag/Last-Modified validators, and by_etag/
    resolves that exact etag back to the digest with one R2 read."""
    import json

    from .conftest import make_gtfs_bytes

    blob = make_gtfs_bytes(["A,t1,Weekday,X,0,s1"], ["t1,s1N,00:00:00,00:00:00,1"])
    client = _FakeClient()
    snap = store(
        blob,
        etag='"e-42"',
        last_modified="Wed, 21 Oct 2026 07:28:00 GMT",
        config=_FakeConfig(),  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
    )

    latest = json.loads(client.objects[LATEST_KEY])
    assert latest["etag"] == '"e-42"'
    assert latest["last_modified"] == "Wed, 21 Oct 2026 07:28:00 GMT"

    by_etag = json.loads(client.objects[etag_key('"e-42"')])
    assert by_etag == {"digest": snap.digest}


def test_store_without_etag_omits_by_etag() -> None:
    """No etag in hand -> no by_etag object is written at all, and latest.json
    carries null validators rather than a fabricated one."""
    import json

    from .conftest import make_gtfs_bytes

    blob = make_gtfs_bytes(["A,t1,Weekday,X,0,s1"], ["t1,s1N,00:00:00,00:00:00,1"])
    client = _FakeClient()
    store(blob, config=_FakeConfig(), client=client)  # type: ignore[arg-type]

    assert not any(k.startswith(BY_ETAG_PREFIX) for k in client.puts)
    latest = json.loads(client.objects[LATEST_KEY])
    assert latest["etag"] is None
    assert latest["last_modified"] is None
