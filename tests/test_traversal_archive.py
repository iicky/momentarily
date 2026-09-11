"""The long-lived derived traversal archive (training/traversal_archive.py).

The archive outlives its inputs: once `archive/trace/` is pruned at 30 days, a
row in here cannot be regenerated or checked against anything. So these cases
are mostly about refusing to read or write something incomparable, and about
making a change in extraction semantics FAIL rather than quietly produce rows
that get averaged with older ones years from now.
"""

from __future__ import annotations

import gzip
import io
import json
from datetime import date
from typing import Any, cast

import pytest

from training.gtfs_static import FeedVersion
from training.r2_client import R2Config
from training.trace import EXACT, RIGHT, Traversal, traversals_from_trace
from training.traversal_archive import (
    EXTRACTOR_VERSION,
    FIELDS,
    SCHEMA_VERSION,
    DayProvenance,
    FeedDigestRange,
    ReadResult,
    decode_day,
    encode_day,
    is_closed,
    key_for,
    read_days,
    resolve_feed_version,
    write_day,
)

AT = 1786551646


def _t(seconds: int, *, to: str | None = "A2S", censoring: str = EXACT) -> Traversal:
    return Traversal(
        trip_id="072000_A..S01R",
        route_id="A",
        direction="south",
        from_stop="A1S",
        to_stop=to,
        at=AT,
        seconds=seconds,
        moving_seconds=None,
        n_hops=1 if censoring == EXACT else None,
        censoring=censoring,
    )


def _doc(rows: list[Traversal] | None = None) -> dict[str, object]:
    blob = encode_day(rows or [_t(90)], feed_version="F", source_keys=["k"])
    return json.loads(gzip.decompress(blob))


def test_a_day_round_trips_every_field_exactly():
    """Rows are positional, so a field-order mistake would silently shift values
    between columns rather than fail. Round-tripping whole objects catches it."""
    rows = [_t(90), _t(400, to=None, censoring=RIGHT)]
    got, prov = decode_day(
        encode_day(rows, feed_version="TEST-20260807", source_keys=["a", "b"])
    )
    assert got == rows
    assert prov.schema == SCHEMA_VERSION
    assert prov.extractor == EXTRACTOR_VERSION
    assert prov.feed_version == "TEST-20260807"
    assert (prov.n_rows, prov.n_source_objects) == (2, 2)


def test_provenance_travels_with_the_rows_not_beside_them():
    """A day that cannot say which extractor and which static feed produced it is
    comparable to nothing once the raw trace behind it has been pruned."""
    doc = _doc()
    assert set(doc) == {"provenance", "fields", "rows"}
    prov = doc["provenance"]
    assert isinstance(prov, dict)
    for field in ("schema", "extractor", "feed_version", "source_manifest", "code_sha"):
        assert field in prov


def test_an_unreadable_schema_raises_instead_of_parsing_partially():
    """A silently mis-parsed historical day is worse than a loud failure: the
    input that could have checked it no longer exists."""
    doc = _doc()
    assert isinstance(doc["provenance"], dict)
    doc["provenance"]["schema"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="schema"):
        decode_day(gzip.compress(json.dumps(doc).encode()))


def test_reordered_fields_are_rejected_rather_than_silently_transposed():
    """The header names the layout; a mismatch means the values are not where the
    reader believes they are."""
    doc = _doc()
    doc["fields"] = list(reversed(FIELDS))
    with pytest.raises(ValueError, match="field order"):
        decode_day(gzip.compress(json.dumps(doc).encode()))


def test_the_source_manifest_distinguishes_days_built_from_different_inputs():
    """Two writes of one day from different trace objects must not look
    identical, or a partial day could masquerade as a complete one."""
    _r1, p1 = decode_day(encode_day([_t(90)], feed_version="F", source_keys=["a"]))
    _r2, p2 = decode_day(encode_day([_t(90)], feed_version="F", source_keys=["a", "b"]))
    assert p1.source_manifest != p2.source_manifest


def test_missing_provenance_counts_raise_rather_than_defaulting_to_zero():
    """How a format rename turns history into plausible nonsense.

    If a required count is absent and the reader defaults it to 0, every day
    written under an older field name decodes as "0 rows from 0 sources" and no
    check fires. The inputs that could rebuild it are gone, so absence is fatal.
    """
    doc = _doc()
    assert isinstance(doc["provenance"], dict)
    del doc["provenance"]["n_rows"]
    with pytest.raises(ValueError, match="missing"):
        decode_day(gzip.compress(json.dumps(doc).encode()))


def _prov(
    *, extractor: int = EXTRACTOR_VERSION, feed: str | None = "F"
) -> DayProvenance:
    return DayProvenance(
        schema=SCHEMA_VERSION,
        extractor=extractor,
        feed_version=feed,
        feed_digest=None,
        n_rows=1,
        n_source_objects=1,
        source_manifest="x",
        code_sha="deadbeef",
        written_at=AT,
    )


def test_a_span_crossing_an_extractor_change_reports_itself_as_mixed():
    """Averaging across a definitional change is the failure this archive exists
    to prevent, and after the raw inputs expire nothing else could detect it."""
    mixed = ReadResult(
        traversals=[],
        provenance={
            date(2026, 8, 12): _prov(extractor=1),
            date(2026, 8, 13): _prov(extractor=2),
        },
    )
    assert not mixed.homogeneous
    assert len(mixed.versions) == 2

    same = ReadResult(
        traversals=[],
        provenance={date(2026, 8, 12): _prov(), date(2026, 8, 13): _prov()},
    )
    assert same.homogeneous


def test_a_feed_republish_also_counts_as_a_version_boundary():
    """The static feed decides scheduled times and which hops are bypasses, so a
    republish changes what the numbers mean, not merely their inputs."""
    result = ReadResult(
        traversals=[],
        provenance={
            date(2026, 8, 12): _prov(feed="20260807"),
            date(2026, 8, 20): _prov(feed="20260814"),
        },
    )
    assert not result.homogeneous


def test_only_closed_days_are_finalized():
    """Writing the day still in progress would store a truncated one, and because
    a present day is skipped by default every later run would skip it forever —
    unrepairable once the raw trace prunes. Closure is a UTC question because the
    trace is partitioned on the UTC date."""
    today = date(2026, 8, 17)
    assert is_closed(date(2026, 8, 16), now=today)
    assert not is_closed(today, now=today)
    assert not is_closed(date(2026, 8, 18), now=today)


def test_keys_are_one_immutable_object_per_day_under_a_date_segment():
    """The date-segment layout is what training.prune's matcher recognises, so
    this prefix is policed by the same retention machinery as every other stream.
    Day-addressable and immutable is also what makes a holdout a date filter."""
    assert (
        key_for(date(2026, 8, 12)) == "archive/traversals/2026-08-12/traversals.json.gz"
    )


def test_a_feed_is_only_stamped_on_days_it_claims_to_describe():
    """A backfill fetches ONE static feed but spans up to 28 days, and MTA
    republishes every few weeks. Stamping today's version on a day it does not
    cover records provenance that is WRONG rather than missing — strictly worse,
    because `homogeneous` would then call a span poolable across a real feed
    boundary. Unknown is recorded as None, which is its own bucket."""
    feed = FeedVersion(
        version="20260807", start=date(2026, 8, 7), end=date(2026, 8, 21)
    )
    assert resolve_feed_version(feed, date(2026, 8, 12)) == "20260807"
    # Before the feed existed: the day ran under a version we did not archive.
    assert resolve_feed_version(feed, date(2026, 8, 1)) is None
    # After it lapsed.
    assert resolve_feed_version(feed, date(2026, 8, 25)) is None
    assert resolve_feed_version(None, date(2026, 8, 12)) is None


def test_known_and_unknown_feed_days_do_not_pool_silently():
    """None is not a wildcard. A span mixing days whose feed is known with days
    whose feed is not must still report itself as mixed."""
    result = ReadResult(
        traversals=[],
        provenance={
            date(2026, 8, 12): _prov(feed="20260807"),
            date(2026, 8, 1): _prov(feed=None),
        },
    )
    assert not result.homogeneous


def test_allow_partial_without_overwrite_is_refused_before_any_work():
    """The one combination that would cause permanent truncation.

    A partial day written once is skipped by every later run, and after the raw
    trace prunes there is nothing left to repair it from. The pairing is enforced
    rather than documented, and it fails before any network fetch so a bad
    invocation costs nothing.
    """
    from training.traversal_archive import main

    with pytest.raises(SystemExit) as exc:
        main(["--allow-partial", "--no-feed-version"])
    assert exc.value.code != 0


def test_the_extractor_semantics_are_pinned_so_a_change_cannot_pass_silently():
    """THE GUARD ON EXTRACTOR_VERSION.

    `trace.traversals_from_trace` decides what an arrival is, what one hop is,
    and which censoring kind a span gets. If those semantics change, every day
    already written becomes incomparable with every day written afterwards — and
    the raw trace that could have re-derived the old days is gone. Nothing in the
    type system notices, and no downstream measure would fail; they would all
    keep producing numbers.

    So this pins the extractor's output over a fixed synthetic trace. When it
    fails, the change is real, and the correct response is to bump
    EXTRACTOR_VERSION and re-derive whatever raw trace still survives — NOT to
    update the expectation in place.
    """
    bodies: list[dict[str, object]] = [
        {
            "scheduled_at": AT,
            "rows": [
                {
                    "trip_id": "T1",
                    "route_id": "A",
                    "direction": "south",
                    "stop_id": "A1S",
                    "stop_seq": 1,
                    "stopped": True,
                    "vehicle_ts": AT,
                }
            ],
        },
        {
            "scheduled_at": AT + 60,
            "rows": [
                {
                    "trip_id": "T1",
                    "route_id": "A",
                    "direction": "south",
                    "stop_id": "A2S",
                    "stop_seq": 2,
                    "stopped": True,
                    "vehicle_ts": AT + 60,
                }
            ],
        },
    ]
    got, _stats = traversals_from_trace(bodies)
    exact = [t for t in got if t.censoring == EXACT]
    assert len(exact) == 1
    hop = exact[0]
    assert (hop.trip_id, hop.route_id, hop.from_stop, hop.to_stop) == (
        "T1",
        "A",
        "A1S",
        "A2S",
    )
    assert (hop.at, hop.seconds, hop.n_hops) == (AT, 60, 1)
    assert EXTRACTOR_VERSION == 1, (
        "extractor semantics changed above; bump EXTRACTOR_VERSION and re-derive "
        "surviving raw trace rather than editing the expectation"
    )


def _stop_body(
    at: int, *, stop_id: str, stop_seq: int, trip: str = "T1"
) -> dict[str, object]:
    return {
        "scheduled_at": at,
        "rows": [
            {
                "trip_id": trip,
                "route_id": "A",
                "direction": "south",
                "stop_id": stop_id,
                "stop_seq": stop_seq,
                "stopped": True,
                "vehicle_ts": at,
            }
        ],
    }


class _FakeArchiveClient:
    """R2 stand-in: a literal key -> bytes map, listing by prefix and serving
    puts, so write_day's full path (list trace keys, fetch bodies, derive,
    encode, put) is exercised without R2. Mirrors the client surface
    load_r2/r2_client actually touch: list_objects_v2, get_object, put_object.
    """

    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self._objects = dict(objects or {})

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        prefix = str(kwargs["Prefix"])
        return {
            "Contents": [
                {"Key": k} for k in sorted(self._objects) if k.startswith(prefix)
            ],
            "IsTruncated": False,
        }

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        return {"Body": io.BytesIO(self._objects[str(kwargs["Key"])])}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, **_: Any) -> None:
        del Bucket
        self._objects[Key] = bytes(Body)


def _r2_config() -> R2Config:
    return R2Config(
        account_id="acct",
        access_key_id="key",
        secret_access_key="secret",
        bucket="test-bucket",
    )


def test_write_day_then_read_days_round_trips_through_the_client_path():
    """encode_day/decode_day agreeing in isolation does not prove write_day's
    full path -- list trace keys, fetch bodies, derive, encode, put -- hands a
    later read_days back exactly what was derived. Exercised end to end
    against a fake R2 client instead of a real trace archive, which is the
    on-disk contract every replay-grade depends on."""
    day = date(2026, 8, 12)
    bodies = [
        _stop_body(AT, stop_id="A1S", stop_seq=1),
        _stop_body(AT + 60, stop_id="A2S", stop_seq=2),
    ]
    expected, _stats = traversals_from_trace(bodies)
    assert expected  # sanity: the fixture actually derives a traversal

    objects = {
        f"archive/trace/{day.isoformat()}/{i}.json": json.dumps(b).encode()
        for i, b in enumerate(bodies)
    }
    client = cast(Any, _FakeArchiveClient(objects))
    cfg = _r2_config()
    feed = FeedVersion(
        version="20260807", start=date(2026, 8, 7), end=date(2026, 8, 21)
    )

    prov = write_day(day, feed=feed, config=cfg, client=client, now=date(2026, 8, 13))

    assert prov is not None
    assert prov.feed_version == "20260807"
    result = read_days(day, day, config=cfg, client=client)
    assert result.traversals == expected
    assert result.provenance == {day: prov}


def test_write_day_skips_a_day_already_present_without_overwrite():
    """A day present at the destination key is left untouched unless
    overwrite is set -- the guard that keeps a finalized day from ever being
    silently shortened once the raw trace prunes."""
    day = date(2026, 8, 12)
    key = key_for(day)
    client = cast(Any, _FakeArchiveClient({key: b"already-written"}))
    cfg = _r2_config()

    prov = write_day(day, feed=None, config=cfg, client=client, now=date(2026, 8, 13))

    assert prov is None
    assert client._objects[key] == b"already-written"


# --- feed_digest extraction from trace bodies (2a3.15) -------------------------
#
# These exercise write_day end-to-end through a fake R2 client so the consensus
# logic, the decode round-trip, and the persisted DayProvenance are all covered.


def _trace_body(
    minute: int,
    *,
    feed_digest: str | None = None,
    feed_etag: str | None = None,
) -> dict[str, Any]:
    """A minimal trace body with one stopped row, optionally carrying feed identity."""
    at = 1_786_550_000 + minute * 60
    body: dict[str, Any] = {
        "observed_at": at + 3,
        "scheduled_at": at,
        "fresh_feeds": ["ace"],
        "feed_digest": feed_digest,
        "feed_etag": feed_etag,
        "rows": [
            {
                "trip_id": "t1",
                "route_id": "A",
                "direction": "south",
                "stop_id": "A1S",
                "stop_seq": 1,
                "stopped": True,
                "vehicle_ts": at,
            },
        ],
    }
    return body


class _FakeR2:
    """Minimal S3Client stub for write_day: supports list, get, and put.

    Optionally seeds ``archive/gtfs/by_etag/`` objects so ``_consensus_digest``
    can resolve mismatch-tick etags to digests.
    """

    def __init__(
        self,
        trace_bodies: list[dict[str, Any]],
        day: date,
        *,
        etag_map: dict[str, str] | None = None,
    ) -> None:
        self._objects: dict[str, bytes] = {}
        prefix = f"archive/trace/{day.isoformat()}/"
        for i, body in enumerate(trace_bodies):
            key = f"{prefix}{body.get('scheduled_at', i)}.json"
            self._objects[key] = json.dumps(body).encode()
        # Seed by_etag mappings (etag string -> digest)
        for etag_val, digest_val in (etag_map or {}).items():
            stripped = etag_val.strip()
            if stripped[:2] in ("W/", "w/"):
                stripped = stripped[2:]
            stripped = stripped.strip('"')
            etag_key = f"archive/gtfs/by_etag/{stripped}.json"
            self._objects[etag_key] = json.dumps({"digest": digest_val}).encode()
        self.puts: list[str] = []

    def list_objects_v2(
        self, *, Bucket: str, Prefix: str, MaxKeys: int = 1000, **_: object
    ) -> dict[str, object]:
        contents = [{"Key": k} for k in self._objects if k.startswith(Prefix)]
        return {"Contents": contents, "IsTruncated": False}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        import io

        if Key not in self._objects:
            from botocore.exceptions import ClientError

            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        return {"Body": io.BytesIO(self._objects[Key])}

    def put_object(self, *, Bucket: str, Key: str, Body: object, **_: object) -> None:
        self.puts.append(Key)
        self._objects[Key] = Body if isinstance(Body, bytes) else b""


class _FakeCfg:
    bucket = "b"
    endpoint_url = ""
    access_key_id = ""
    secret_access_key = ""


_DAY = date(2026, 8, 10)  # safely in the past
# A feed version that covers _DAY
_FEED = FeedVersion(version="V1", start=date(2026, 1, 1), end=date(2026, 12, 31))


def test_write_day_extracts_unanimous_feed_digest() -> None:
    """When every trace body carries the same feed_digest and no explicit digest
    is passed, write_day stamps it on the DayProvenance."""
    digest = "a" * 64
    client = _FakeR2(
        [_trace_body(0, feed_digest=digest), _trace_body(1, feed_digest=digest)],
        _DAY,
    )
    prov = write_day(
        _DAY,
        feed=_FEED,
        config=_FakeCfg(),  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        allow_partial=True,
        present=set(),
    )
    assert prov is not None
    assert prov.feed_digest == digest
    assert prov.feed_digests is None


def test_write_day_mismatch_tick_resolved_via_by_etag() -> None:
    """A mismatch tick carries feed_digest=null + feed_etag set. By the time
    write_day runs the capture has completed and by_etag maps the etag to its
    digest. The resolved digest participates in the consensus."""
    digest = "a" * 64
    etag = '"etag-new"'
    client = _FakeR2(
        [
            _trace_body(0, feed_digest=digest),
            _trace_body(1, feed_etag=etag),  # mismatch tick: digest null, etag set
            _trace_body(2, feed_digest=digest),
        ],
        _DAY,
        etag_map={etag: digest},  # capture completed: same feed
    )
    prov = write_day(
        _DAY,
        feed=_FEED,
        config=_FakeCfg(),  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        allow_partial=True,
        present=set(),
    )
    assert prov is not None
    # All three bodies resolve to the same digest (two directly, one via etag)
    assert prov.feed_digest == digest
    assert prov.feed_digests is None


def test_write_day_mismatch_tick_resolves_to_different_digest_yields_ranges() -> None:
    """A mid-day feed change: first ticks carry digest_a, the mismatch tick's
    etag resolves to digest_b, and subsequent ticks carry digest_b directly.
    write_day must produce two ranges spanning the transition."""
    digest_a, digest_b = "a" * 64, "b" * 64
    etag_b = '"etag-new-b"'
    client = _FakeR2(
        [
            _trace_body(0, feed_digest=digest_a),
            _trace_body(1, feed_digest=digest_a),
            _trace_body(2, feed_etag=etag_b),  # mismatch tick -> resolves to digest_b
            _trace_body(3, feed_digest=digest_b),
        ],
        _DAY,
        etag_map={etag_b: digest_b},
    )
    prov = write_day(
        _DAY,
        feed=_FEED,
        config=_FakeCfg(),  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        allow_partial=True,
        present=set(),
    )
    assert prov is not None
    assert prov.feed_digest is None
    assert prov.feed_digests == [
        FeedDigestRange(digest=digest_a, first_ts=1_786_550_000, last_ts=1_786_550_060),
        FeedDigestRange(digest=digest_b, first_ts=1_786_550_120, last_ts=1_786_550_180),
    ]


def test_write_day_ranges_are_contiguous_runs_broken_by_reverts_and_unresolved_ticks() -> (
    None
):
    """A -> B -> A must yield three runs, never one A range whose span covers
    B's ticks; and an unresolved tick (HEAD failed, capture never landed)
    breaks a run the same way, so no range claims a tick whose feed is unknown."""
    digest_a, digest_b = "a" * 64, "b" * 64
    client = _FakeR2(
        [
            _trace_body(0, feed_digest=digest_a),
            _trace_body(1, feed_digest=digest_b),
            _trace_body(2, feed_digest=digest_a),
            _trace_body(3, feed_digest=None, feed_etag=None),  # HEAD failed
            _trace_body(4, feed_digest=digest_a),
        ],
        _DAY,
    )
    prov = write_day(
        _DAY,
        feed=_FEED,
        config=_FakeCfg(),  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        allow_partial=True,
        present=set(),
    )
    assert prov is not None
    assert prov.feed_digest is None
    assert prov.feed_digests == [
        FeedDigestRange(digest=digest_a, first_ts=1_786_550_000, last_ts=1_786_550_000),
        FeedDigestRange(digest=digest_b, first_ts=1_786_550_060, last_ts=1_786_550_060),
        FeedDigestRange(digest=digest_a, first_ts=1_786_550_120, last_ts=1_786_550_120),
        FeedDigestRange(digest=digest_a, first_ts=1_786_550_240, last_ts=1_786_550_240),
    ]


def test_write_day_mixed_digest_resolves_to_ranges() -> None:
    """Two different digests directly in trace bodies -> ranges."""
    digest_a, digest_b = "a" * 64, "b" * 64
    client = _FakeR2(
        [
            _trace_body(0, feed_digest=digest_a),
            _trace_body(1, feed_digest=digest_a),
            _trace_body(2, feed_digest=digest_b),
            _trace_body(3, feed_digest=digest_b),
        ],
        _DAY,
    )
    prov = write_day(
        _DAY,
        feed=_FEED,
        config=_FakeCfg(),  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        allow_partial=True,
        present=set(),
    )
    assert prov is not None
    assert prov.feed_digest is None
    assert prov.feed_digests == [
        FeedDigestRange(digest=digest_a, first_ts=1_786_550_000, last_ts=1_786_550_060),
        FeedDigestRange(digest=digest_b, first_ts=1_786_550_120, last_ts=1_786_550_180),
    ]


def test_write_day_no_digest_in_any_body() -> None:
    """All bodies lack feed_digest (pre-change archive) → both None."""
    client = _FakeR2([_trace_body(0), _trace_body(1)], _DAY)
    prov = write_day(
        _DAY,
        feed=_FEED,
        config=_FakeCfg(),  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        allow_partial=True,
        present=set(),
    )
    assert prov is not None
    assert prov.feed_digest is None
    assert prov.feed_digests is None


def test_write_day_some_unresolved_ticks_prevent_unanimous() -> None:
    """A day where most ticks carry a digest but some couldn't resolve (HEAD
    failure, no etag) must NOT stamp feed_digest — it would falsely claim
    whole-day coverage. Instead it gets runs for the resolved ticks only."""
    digest = "a" * 64
    client = _FakeR2(
        [
            _trace_body(0, feed_digest=digest),
            _trace_body(1),  # HEAD failure: no digest, no etag
            _trace_body(2, feed_digest=digest),
        ],
        _DAY,
    )
    prov = write_day(
        _DAY,
        feed=_FEED,
        config=_FakeCfg(),  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        allow_partial=True,
        present=set(),
    )
    assert prov is not None
    assert prov.feed_digest is None  # NOT unanimous — one body unresolved
    # The unresolved tick splits the run: no range may claim coverage over a
    # tick whose feed is unknown.
    assert prov.feed_digests == [
        FeedDigestRange(digest=digest, first_ts=1_786_550_000, last_ts=1_786_550_000),
        FeedDigestRange(digest=digest, first_ts=1_786_550_120, last_ts=1_786_550_120),
    ]


def test_write_day_explicit_digest_overrides_bodies() -> None:
    """An explicit feed_digest parameter takes precedence over bodies, and no
    ranges are derived when the caller already hands over a proven value."""
    explicit = "f" * 64
    client = _FakeR2([_trace_body(0, feed_digest="a" * 64)], _DAY)
    prov = write_day(
        _DAY,
        feed=_FEED,
        feed_digest=explicit,
        config=_FakeCfg(),  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        allow_partial=True,
        present=set(),
    )
    assert prov is not None
    assert prov.feed_digest == explicit
    assert prov.feed_digests is None
