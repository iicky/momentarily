"""The content-addressed static-feed store (training/gtfs_archive.py).

A version label cannot rebuild a timetable, and MTA serves no superseded
snapshots, so replaying a historical grade needs the feed ARTIFACT. These cases
pin the addressing, the corruption check, and the rule that a digest is only ever
recorded when it can be proven — never inferred from whatever feed is current.
"""

from __future__ import annotations

import hashlib
from datetime import date

from training.gtfs_archive import digest_of, key_for
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
