"""Journey span chaining (training/journey.py).

Synthetic traversal rows against a synthetic timetable — no R2. The contract
under test: chains break on RIGHT censoring and on non-contiguous stops,
windows enumerate every contiguous run of hops up to max_hops, scheduled_sec
sums the trip's OWN observed hops and one unknown hop poisons the whole span,
and the JSONL layout round-trips losslessly.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import asdict, fields
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from training.gtfs_archive import digest_of
from training.gtfs_static import Timetable, timetable
from training.journey import load_spans, read_spans, spans_from_traversals, write_spans
from training.journey_types import SpanSample
from training.trace import EXACT, INTERVAL, RIGHT, Traversal
from training.traversal_archive import DayProvenance, FeedDigestRange, ReadResult

from .conftest import make_gtfs_bytes, make_gtfs_zip

# A weekday noon and a trip whose id says it left at noon, so the observation
# lands on the weekday timetable by the same rule the live pipeline uses.
WEEKDAY = date(2026, 8, 12)
AT = int(
    datetime.combine(
        WEEKDAY, time(12, 0), tzinfo=ZoneInfo("America/New_York")
    ).timestamp()
)
TRIP = "072000_A..S01R"

# Scheduled seconds along the pattern A1S -> A2S -> A3S -> A4S -> A5S.
SCHED = {
    ("A1S", "A2S"): 90,
    ("A2S", "A3S"): 400,
    ("A3S", "A4S"): 100,
    ("A4S", "A5S"): 200,
}


_TRIPS = [f"A,{TRIP},Weekday,X,1,A..S01R"]
_STOP_TIMES = [
    f"{TRIP},A1S,12:00:00,12:00:00,1",
    f"{TRIP},A2S,12:01:30,12:01:30,2",
    f"{TRIP},A3S,12:08:10,12:08:10,3",
    f"{TRIP},A4S,12:09:50,12:09:50,4",
    f"{TRIP},A5S,12:13:10,12:13:10,5",
]


def _feed_bytes() -> bytes:
    return make_gtfs_bytes(_TRIPS, _STOP_TIMES)


def _feed() -> zipfile.ZipFile:
    return make_gtfs_zip(_TRIPS, _STOP_TIMES)


def _timetable() -> Timetable:
    return timetable(_feed())


def _prov(
    feed_version: str,
    feed_digest: str | None = None,
    feed_digests: list[FeedDigestRange] | None = None,
) -> DayProvenance:
    return DayProvenance(
        schema=1,
        extractor=1,
        feed_version=feed_version,
        feed_digest=feed_digest,
        feed_digests=feed_digests,
        n_rows=0,
        n_source_objects=0,
        source_manifest="m",
        code_sha=None,
        written_at=0,
    )


def _row(
    frm: str,
    to: str | None,
    *,
    at: int,
    seconds: int = 100,
    n_hops: int | None = 1,
    censoring: str = EXACT,
) -> Traversal:
    return Traversal(
        trip_id=TRIP,
        route_id="A",
        direction="south",
        from_stop=frm,
        to_stop=to,
        at=at,
        seconds=seconds,
        moving_seconds=None,
        n_hops=n_hops,
        censoring=censoring,
    )


def _chain4() -> list[Traversal]:
    """Four contiguous exact hops A1S..A5S, arrival-ordered."""
    stops = ["A1S", "A2S", "A3S", "A4S", "A5S"]
    return [
        _row(stops[k], stops[k + 1], at=AT + 100 * k, seconds=100) for k in range(4)
    ]


def test_right_censoring_breaks_the_chain():
    """A RIGHT row is never part of a span and no span crosses it, even though
    the stops on either side of it would abut."""
    rows = [
        _row("A1S", "A2S", at=AT),
        _row("A2S", None, at=AT + 100, n_hops=None, censoring=RIGHT),
        _row("A2S", "A3S", at=AT + 200),
    ]
    got = spans_from_traversals(rows, _timetable())
    assert sorted((s.origin, s.destination, s.n_hops) for s in got) == [
        ("A1S", "A2S", 1),
        ("A2S", "A3S", 1),
    ]
    assert all(s.service_day == "2026-08-12" for s in got)


def test_non_contiguous_stops_break_the_chain():
    rows = [_row("A1S", "A2S", at=AT), _row("A3S", "A4S", at=AT + 100)]
    got = spans_from_traversals(rows, _timetable())
    assert sorted((s.origin, s.destination) for s in got) == [
        ("A1S", "A2S"),
        ("A3S", "A4S"),
    ]
    assert all(s.n_hops == 1 for s in got)


def test_window_enumeration_over_a_four_hop_chain():
    """n rows of one hop each give n*(n+1)/2 windows under a loose cap and
    fewer once the cap bites: 10 at max_hops=8, 7 at max_hops=2."""
    rows = _chain4()
    loose = spans_from_traversals(rows, _timetable(), max_hops=8)
    assert len(loose) == 10
    tight = spans_from_traversals(rows, _timetable(), max_hops=2)
    assert len(tight) == 7
    assert all(s.n_hops <= 2 for s in tight)

    two_hop = next(s for s in loose if s.origin == "A1S" and s.destination == "A3S")
    assert two_hop.depart_at == AT
    assert two_hop.seconds == 200
    assert two_hop.scheduled_sec == SCHED[("A1S", "A2S")] + SCHED[("A2S", "A3S")]
    full = next(s for s in loose if s.origin == "A1S" and s.destination == "A5S")
    assert full.n_hops == 4
    assert full.scheduled_sec == sum(SCHED.values())


def test_an_interval_row_counts_by_its_own_n_hops():
    """A 2-hop INTERVAL row chains, and the cap is on summed realtime hops,
    not on row count."""
    rows = [
        _row("A1S", "A2S", at=AT),
        _row("A2S", "A4S", at=AT + 100, seconds=500, n_hops=2, censoring=INTERVAL),
    ]
    got = spans_from_traversals(rows, _timetable(), max_hops=2)
    assert sorted((s.origin, s.destination, s.n_hops) for s in got) == [
        ("A1S", "A2S", 1),
        ("A2S", "A4S", 2),
    ]


def test_one_unknown_hop_makes_the_spans_scheduled_sec_none():
    """The timetable never scheduled A2S -> B9S, so every window containing
    that hop abstains from a scheduled comparison; the window before it keeps
    its own."""
    rows = [_row("A1S", "A2S", at=AT), _row("A2S", "B9S", at=AT + 100)]
    got = {
        (s.origin, s.destination): s.scheduled_sec
        for s in spans_from_traversals(rows, _timetable())
    }
    assert got == {
        ("A1S", "A2S"): 90,
        ("A2S", "B9S"): None,
        ("A1S", "B9S"): None,
    }


def test_jsonl_round_trip_is_lossless(tmp_path: Path):
    spans = spans_from_traversals(_chain4(), _timetable())
    for name in ("spans.jsonl", "spans.jsonl.gz"):
        path = str(tmp_path / name)
        write_spans(path, spans)
        assert read_spans(path) == spans

    # The layout is the contract: exactly the SpanSample field names, one
    # object per line.
    first = json.loads((tmp_path / "spans.jsonl").read_text().splitlines()[0])
    assert set(first) == {f.name for f in fields(SpanSample)}
    assert first == asdict(spans[0])


def test_an_unverified_day_loses_scheduled_sec_but_keeps_seconds():
    """Schedule numbers come only from byte-verified days; realized times
    never depend on schedule bytes and survive."""
    rows = [_row("A1S", "A2S", at=AT, seconds=123)]
    unverified = spans_from_traversals(rows, _timetable(), verified_days=frozenset())
    assert [(s.seconds, s.scheduled_sec) for s in unverified] == [(123, None)]
    verified = spans_from_traversals(rows, _timetable(), verified_days={WEEKDAY})
    assert [(s.seconds, s.scheduled_sec) for s in verified] == [(123, 90)]


def _patched_load(monkeypatch: pytest.MonkeyPatch, result: ReadResult) -> None:
    def _read_days(*args: object, **kwargs: object) -> ReadResult:
        return result

    def _fetch_gtfs_zip(*args: object, **kwargs: object) -> bytes:
        return _feed_bytes()

    monkeypatch.setattr("training.journey.read_days", _read_days)
    monkeypatch.setattr("training.journey.fetch_gtfs_zip", _fetch_gtfs_zip)


def test_load_spans_refuses_a_mixed_feed_version(monkeypatch: pytest.MonkeyPatch):
    result = ReadResult(
        traversals=[],
        provenance={
            WEEKDAY: _prov("20260807-H"),
            WEEKDAY + timedelta(days=1): _prov("20260826-X"),
        },
    )
    _patched_load(monkeypatch, result)
    with pytest.raises(ValueError, match="not comparable"):
        load_spans(WEEKDAY, WEEKDAY + timedelta(days=1))


def test_load_spans_pools_a_digest_mix_under_one_version_and_verifies_per_day(
    monkeypatch: pytest.MonkeyPatch,
):
    """A republish under the same version label does not refuse the window; it
    only decides, day by day, whether scheduled_sec may be read. Only the day
    whose stamped digest matches the parsed bytes is verified — an unstamped
    day and a mid-day mix are not."""
    live = digest_of(_feed_bytes())
    # A Monday, so all four days run the Weekday service and a missing
    # scheduled_sec can only mean the verification gate, not the calendar.
    monday = date(2026, 8, 10)
    at0 = int(
        datetime.combine(
            monday, time(12, 0), tzinfo=ZoneInfo("America/New_York")
        ).timestamp()
    )
    verified_day = monday
    other = monday + timedelta(days=1)
    unstamped = monday + timedelta(days=2)
    mixed = monday + timedelta(days=3)
    result = ReadResult(
        traversals=[
            _row("A1S", "A2S", at=at0 + 86400 * k, seconds=123) for k in range(4)
        ],
        provenance={
            verified_day: _prov("V", feed_digest=live),
            other: _prov("V", feed_digest="deadbeef"),
            unstamped: _prov("V"),
            mixed: _prov(
                "V", feed_digests=[FeedDigestRange(digest=live, first_ts=0, last_ts=1)]
            ),
        },
    )
    _patched_load(monkeypatch, result)
    spans, got = load_spans(verified_day, mixed)
    assert got is result
    # Only the byte-verified day keeps a scheduled comparison; every day keeps
    # its realized seconds.
    assert {s.service_day: (s.seconds, s.scheduled_sec) for s in spans} == {
        verified_day.isoformat(): (123, 90),
        other.isoformat(): (123, None),
        unstamped.isoformat(): (123, None),
        mixed.isoformat(): (123, None),
    }
