"""The long-lived derived traversal archive (training/traversal_archive.py).

The archive outlives its inputs: once `archive/trace/` is pruned at 30 days, a
row in here cannot be regenerated or checked against anything. So these cases
are mostly about refusing to read or write something incomparable, and about
making a change in extraction semantics FAIL rather than quietly produce rows
that get averaged with older ones years from now.
"""

from __future__ import annotations

import gzip
import json
from datetime import date

import pytest

from training.gtfs_static import FeedVersion
from training.trace import EXACT, RIGHT, Traversal, traversals_from_trace
from training.traversal_archive import (
    EXTRACTOR_VERSION,
    FIELDS,
    SCHEMA_VERSION,
    DayProvenance,
    ReadResult,
    decode_day,
    encode_day,
    is_closed,
    key_for,
    resolve_feed_version,
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
