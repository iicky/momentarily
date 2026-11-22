"""Layer 3, slice 4: runs, pattern fitting, trip matching and supply
(docs/model-events.md §2, §5.2, §10 slice 4).

A synthetic network (``_synthetic``) gives exact schedules: route 1 north on
two patterns (101N–105N, and 103N–105N), route 2, J and a Z pattern whose
static ids carry ``J`` (as NYCT's do), and G, on ordinary days and on both
2026/27 DST nights. The recorded-hour fixture, derived by layer 2, drives the
prefix-replay and rebuild tests.
"""

from __future__ import annotations

import gzip
import json
import random
from collections.abc import Iterable
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import blake3
import pytest

from momentarily.model.conventions import wall_instant_of
from momentarily.model.events import match_build
from momentarily.model.events.__main__ import main as events_main
from momentarily.model.events.contracts import SupplyObservation, TripMatch
from momentarily.model.events.coverage import Evidence, Poll, coverage_intervals
from momentarily.model.events.files import read_records
from momentarily.model.events.match_build import read_window
from momentarily.model.events.match_index import (
    BuildChoice,
    IndexInputs,
    MatchIndex,
    build_match_index,
)
from momentarily.model.events.matching import (
    DRIFT_TOL_S,
    Matcher,
    _Candidate,  # pyright: ignore[reportPrivateUsage]
    assign_min_cost,
    service_day_window,
    trip_matches,
)
from momentarily.model.events.route_feed import load_route_feed
from momentarily.model.events.runs import (
    PatternBook,
    Row,
    Run,
    build_runs,
    is_legacy_placeholder,
    is_raw_placeholder,
)
from momentarily.model.events.supply import supply_observations
from momentarily.model.network.contracts import NetworkSnapshot, ScheduledTrip
from momentarily.model.observations.contracts import GtfsRtSource
from momentarily.model.observations.derive.run import derive
from momentarily.model.observations.derive.storage import LocalStorage

FIXTURES = Path(__file__).parent / "fixtures" / "model"
RECORDED = FIXTURES / "observations" / "recorded_hour"
TZ = "America/New_York"
BASE_BUILD = "606d9f640f5aad67"

DAY = "2026-06-01"  # a Monday
DATES = [
    "2026-05-31",
    "2026-06-01",
    "2026-06-02",
    "2026-10-30",
    "2026-10-31",
    "2026-11-01",
    "2026-11-02",
    "2027-03-12",
    "2027-03-13",
    "2027-03-14",
    "2027-03-15",
]
PATTERNS = {
    "1-N-local": ("1", ["101N", "102N", "103N", "104N", "105N"]),
    "1-N-short": ("1", ["103N", "104N", "105N"]),
    "2-N": ("2", ["201N", "202N", "203N"]),
    "J-N": ("J", ["J12N", "J13N", "J14N", "J15N"]),
    "Z-N": ("Z", ["J12N", "J14N", "J15N"]),
    "G-N": ("G", ["G22N", "G24N", "G26N"]),
}
DAYTIME = list(range(6 * 3600, 8 * 3600 + 1, 600))
DST_ORIGINS = [14 * 60, 180 * 60]
# Trips that run on one DST date only, so no two candidate days offer a trip
# at the same instant: service id -> (date, departures in GTFS seconds).
DST_ONLY = {
    "Late1031": ("2026-10-31", [1590 * 60]),  # 26:30 = 02:30 EST on 11-01
    "Early1101": ("2026-11-01", [90 * 60]),  # 01:30, first occurrence
    "Late0313": ("2027-03-13", [1530 * 60]),  # 25:30 = 01:30 EST on 03-14
    "Early0314": ("2027-03-14", [150 * 60]),  # 02:30: does not exist
}


def _base() -> NetworkSnapshot:
    raw = gzip.decompress(
        (FIXTURES / "events" / f"network_{BASE_BUILD}_subset.json.gz").read_bytes()
    )
    return NetworkSnapshot.model_validate_json(raw)


def _origin(secs: int) -> str:
    return f"{secs // 60 * 100:06d}"


@cache
def _synthetic() -> NetworkSnapshot:
    base = _base()
    template = base.patterns[0]
    patterns = [
        template.model_copy(update={"pattern_id": pid, "route_id": r, "stops": s})
        for pid, (r, s) in PATTERNS.items()
    ]
    stops = sorted({s for _r, ss in PATTERNS.values() for s in ss})
    platforms = [
        base.platforms[0].model_copy(update={"stop_id": s, "station_id": s[:-1]})
        for s in stops
    ]
    trips: list[ScheduledTrip] = []

    def add(pid: str, field: str, secs: int, path: str, service: str = "Daily") -> None:
        route, ss = PATTERNS[pid]
        trips.append(
            base.trips[0].model_copy(
                update={
                    "trip_id": f"SYN-Daily_{_origin(secs)}_{field}..N{path}",
                    "route_id": route,
                    "pattern_id": pid,
                    "service_id": service,
                    "stop_times": [
                        (s, secs + 120 * i, secs + 120 * i) for i, s in enumerate(ss)
                    ],
                }
            )
        )

    for secs in DAYTIME:
        add("1-N-local", "1", secs, "01R")
        add("2-N", "2", secs, "01R")
        add("G-N", "G", secs, "01R")
    for secs in (*DST_ORIGINS, 1441 * 60, 1450 * 60):
        add("1-N-local", "1", secs, "01R")
    for service, (_date, departures) in DST_ONLY.items():
        for secs in departures:
            add("1-N-local", "1", secs, "01R", service)
    add("J-N", "J", 8 * 3600, "01R")
    add("Z-N", "J", 8 * 3600 + 600, "01Z")
    days = [
        base.service_days[0].model_copy(
            update={
                "date": d,
                "active_service_ids": [
                    "Daily",
                    *(s for s, (on, _) in DST_ONLY.items() if on == d),
                ],
                "day_type": "mixed",
                "is_exception": False,
            }
        )
        for d in DATES
    ]
    return base.model_copy(
        update={
            "patterns": patterns,
            "platforms": platforms,
            "trips": trips,
            "service_days": days,
        }
    )


@cache
def _index(service_date: str) -> MatchIndex | None:
    if service_date not in DATES:
        return None
    net = _synthetic()
    table, digest = load_route_feed()
    doc = build_match_index(
        service_date,
        IndexInputs(
            network=net,
            network_digest=blake3.blake3(b"synthetic").hexdigest(),
            choice=BuildChoice(net.feed.feed_id, net.build_id, "explicit"),
            day_map_digest=None,
            route_feed=table,
            route_feed_digest=digest,
            code_sha="test",
            code_dirty=None,
        ),
    )
    return MatchIndex(doc)


def _at(local: str) -> int:
    """An instant from a New York wall time (first occurrence)."""
    return int(datetime.fromisoformat(local).replace(tzinfo=ZoneInfo(TZ)).timestamp())


def _utc(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp())


def _row(
    at: int,
    trip: str,
    stop: str,
    *,
    feed: GtfsRtSource = "numbered",
    poll: str | None = None,
    entity: int = 0,
    vts: int | None = None,
    train: str | None = None,
    route: str | None = None,
    start_date: str | None = None,
    raw: bool = True,
) -> Row:
    return Row(
        at=at,
        poll=poll or f"{feed}@{at}000",
        entity=entity,
        feed=feed,
        trip_id=trip,
        stop_id=stop,
        status=1,
        vts=at if vts is None else vts,
        route=route if route is not None or not raw else trip.split("_")[1][0],
        start_date=start_date,
        train_id=train,
    )


def _walk(
    trip: str, start: int, stops: list[str], step: int = 120, **kw: Any
) -> list[Row]:
    """A train reaching ``stops`` ``step`` seconds apart, polled each minute."""
    rows: list[Row] = []
    for i, s in enumerate(stops):
        for k in range(step // 60):
            rows.append(_row(start + i * step + 60 * k, trip, s, **kw))
    return rows


def _runs(
    rows: Iterable[Row], lineage: str = "raw", last_poll: int | None = None
) -> list[Run]:
    rows = list(rows)
    end = max(r.at for r in rows) if last_poll is None else last_poll
    last: dict[GtfsRtSource, int] = {r.feed: end for r in rows}
    book = PatternBook(_index(DAY) or _never())
    return build_runs("raw" if lineage == "raw" else "legacy", rows, [], last, book)


def _never() -> MatchIndex:
    raise AssertionError("no index")


def _match(
    rows: Iterable[Row], *, last_poll: int | None = None, withheld: bool = False
) -> list[TripMatch]:
    runs = _runs(rows, last_poll=last_poll)
    return trip_matches(runs, Matcher(_index, TZ, withheld), "")


def _one(matches: list[TripMatch]) -> TripMatch:
    assert len(matches) == 1, [m.run_id for m in matches]
    return matches[0]


# --- tiers, service day, DST -------------------------------------------------------

LOCAL = ["101N", "102N", "103N", "104N", "105N"]


def test_exact_id_matches_with_the_offset_at_the_first_common_stop() -> None:
    m = _one(_match(_walk("036000_1..N01R", _at(f"{DAY}T06:01:00"), LOCAL)))
    assert (m.outcome, m.tier, m.scheduled_trip_id) == (
        "matched",
        "exact_id",
        "SYN-Daily_036000_1..N01R",
    )
    assert (m.service_date, m.service_date_basis, m.offset_s) == (
        DAY,
        "scheduled_trip",
        60,
    )
    assert (m.pattern_attribution, m.pattern_id, m.deviation) == (
        "scheduled_trip",
        "1-N-local",
        "consistent",
    )
    assert m.compatible_pattern_ids == ["1-N-local"]


def test_a_truncated_path_matches_at_tier_2_and_a_drifted_id_at_tier_5() -> None:
    prefix = _one(_match(_walk("037000_1..N", _at(f"{DAY}T06:10:00"), LOCAL)))
    assert (prefix.tier, prefix.scheduled_trip_id) == (
        "path_prefix",
        "SYN-Daily_037000_1..N01R",
    )
    # No trip has origin 06:18; tier 5 times it at 102N, after the origin.
    drift = _one(_match(_walk("037800_1..N01R", _at(f"{DAY}T06:22:30"), LOCAL)))
    assert (drift.tier, drift.scheduled_trip_id, drift.offset_s) == (
        "drift",
        "SYN-Daily_038000_1..N01R",
        150,
    )


def test_a_wrapped_id_at_0005_takes_the_previous_service_date() -> None:
    m = _one(_match(_walk("000100_1..N01R", _at("2026-06-02T00:01:30"), LOCAL)))
    assert (m.scheduled_trip_id, m.service_date, m.origin_wrapped) == (
        "SYN-Daily_144100_1..N01R",
        "2026-06-01",
        True,
    )


@pytest.mark.parametrize(
    ("start_date", "flagged"), [("20260602", False), ("20260601", True)]
)
def test_a_trip_at_2410_carries_the_next_calendar_date(
    start_date: str, flagged: bool
) -> None:
    rows = _walk(
        "001000_1..N01R", _at("2026-06-02T00:10:00"), LOCAL, start_date=start_date
    )
    m = _one(_match(rows))
    assert (m.scheduled_trip_id, m.service_date) == (
        "SYN-Daily_145000_1..N01R",
        "2026-06-01",
    )
    assert ("start_date_mismatch" in m.flags) is flagged


def test_an_unmatched_run_at_0030_gets_its_id_origin_date_and_is_ambiguous() -> None:
    rows = _walk("003000_1..S01R", _at("2026-06-02T00:30:00"), ["105S", "104S"])
    m = _one(_match(rows))
    assert (m.outcome, m.service_date, m.service_date_basis) == (
        "unscheduled",
        "2026-06-02",
        "id_origin",
    )
    assert "service_date_ambiguous" in m.flags
    assert m.alternative_service_date == "2026-06-01"
    assert (m.route_id, m.route_basis) == ("1", "published")


def test_an_unparseable_id_dates_by_the_clock_rule() -> None:
    m = _one(
        _match(_walk("not-a-trip", _at("2026-06-02T03:00:00"), LOCAL[:2], route="1"))
    )
    assert (m.outcome, m.miss_class, m.service_date, m.service_date_basis) == (
        "unparseable",
        "unparseable_id",
        "2026-06-01",
        "clock_rule",
    )


@pytest.mark.parametrize(
    ("trip", "seen_utc", "service_date", "scheduled", "offset", "wrapped"),
    [
        # Fall back, 2026-11-01: trips start on the wall clock.
        ("001400_1..N01R", "2026-11-01T04:14:00", "2026-11-01", "001400", 0, False),
        ("015000_1..N01R", "2026-11-01T07:30:00", "2026-10-31", "159000", 0, True),
        ("018000_1..N01R", "2026-11-01T08:00:00", "2026-11-01", "018000", 0, False),
        # Spring forward, 2027-03-14: 02:30 does not exist; the id still
        # matches, untimed. 25:30 on 03-13 is 01:30 EST.
        ("015000_1..N01R", "2027-03-14T07:30:00", "2027-03-14", "015000", None, False),
        ("009000_1..N01R", "2027-03-14T06:30:00", "2027-03-13", "153000", 0, True),
        ("018000_1..N01R", "2027-03-14T07:00:00", "2027-03-14", "018000", 0, False),
        # The days before at an ordinary hour.
        ("036000_1..N01R", "2026-10-31T10:00:00", "2026-10-31", "036000", 0, False),
        ("036000_1..N01R", "2027-03-13T11:00:00", "2027-03-13", "036000", 0, False),
    ],
)
def test_dst_nights_match_on_the_anchored_wall_clock(
    trip: str,
    seen_utc: str,
    service_date: str,
    scheduled: str,
    offset: int | None,
    wrapped: bool,
) -> None:
    m = _one(_match(_walk(trip, _utc(seen_utc), LOCAL)))
    assert m.scheduled_trip_id == f"SYN-Daily_{scheduled}_1..N01R"
    assert (m.service_date, m.offset_s, m.origin_wrapped) == (
        service_date,
        offset,
        wrapped,
    )


def test_the_repeated_hour_marker_matches_the_trip_an_hour_later() -> None:
    # NYCT ids a trip running in the repeated 01:00 hour of 2026-11-01 with
    # its origin + 1 (009001 for 01:30); it runs at the second 01:30 (EST).
    m = _one(_match(_walk("009001_1..N01R", _utc("2026-11-01T06:30:00"), LOCAL)))
    assert (m.tier, m.scheduled_trip_id, m.service_date, m.offset_s) == (
        "exact_id",
        "SYN-Daily_009000_1..N01R",
        "2026-11-01",
        0,
    )
    # The same residue outside 01:00-01:59 is no marker.
    m = _one(_match(_walk("036001_1..N01R", _at(f"{DAY}T06:00:00"), LOCAL)))
    assert m.tier == "drift"


@pytest.mark.parametrize(("late_s", "matched"), [(3600, True), (3 * 3600, False)])
def test_an_id_hit_without_a_common_stop_is_timed_by_its_departure(
    late_s: int, matched: bool
) -> None:
    # Seen only at route-2 stops, which the 06:00 route-1 trip does not serve.
    rows = _walk("036000_1..N01R", _at(f"{DAY}T06:00:00") + late_s, ["201N", "202N"])
    m = _one(_match(rows))
    assert (m.scheduled_trip_id is not None) is matched
    assert m.offset_s is None


def test_tier_5_skips_unresolved_spring_forward_trips() -> None:
    # 02:30 on 2027-03-14 does not exist: no time to drift-match against.
    rows = _walk("015500_1..N01R", _utc("2027-03-14T07:35:00"), LOCAL)
    m = _one(_match(rows))
    assert m.outcome == "unscheduled"
    # The same drifted id on the fall-back night matches 01:30 (first) by time.
    rows = _walk("009500_1..N01R", _utc("2026-11-01T05:35:00"), LOCAL)
    m = _one(_match(rows))
    assert (m.tier, m.scheduled_trip_id, m.service_date) == (
        "drift",
        "SYN-Daily_009000_1..N01R",
        "2026-11-01",
    )


def test_withheld_mode_matches_by_time_alone() -> None:
    m = _one(
        _match(_walk("036000_1..N01R", _at(f"{DAY}T06:01:00"), LOCAL), withheld=True)
    )
    assert (m.tier, m.scheduled_trip_id) == ("drift", "SYN-Daily_036000_1..N01R")


# --- one-to-one tier 5 -----------------------------------------------------------


def _cand(cost: int, trip: str) -> _Candidate:
    return _Candidate(cost, trip, DAY, cost)


def test_min_cost_assignment_is_one_to_one_and_leaves_far_runs_unassigned() -> None:
    items = [
        ("a", [_cand(100, "t1"), _cand(500, "t2")]),
        ("b", [_cand(120, "t1"), _cand(880, "t3")]),
        ("c", [_cand(50, "t1")]),
    ]
    got = {k: c.trip_id for k, c in assign_min_cost(items, DRIFT_TOL_S).items()}
    # c takes t1; a falls back to t2 (500 < 900); b would cost 880 for t3,
    # below the 900 leave-unassigned cost, so it takes it.
    assert got == {"a": "t2", "b": "t3", "c": "t1"}
    assert len(set(got.values())) == len(got)
    # A candidate costing more than leaving the run unassigned is never taken.
    got = assign_min_cost([("a", [_cand(50, "t1")]), ("b", [_cand(60, "t1")])], 900)
    assert {k: c.trip_id for k, c in got.items()} == {"a": "t1"}


def test_equal_costs_go_to_the_trip_first_in_id_order() -> None:
    for order in ([0, 1], [1, 0]):
        cands = [_cand(75, "t2"), _cand(75, "t1")]
        got = assign_min_cost([("a", [cands[i] for i in order])], 900)
        assert got["a"].trip_id == "t1"


def test_no_scheduled_trip_is_held_by_two_tier_5_runs() -> None:
    # Two drifted copies near one trip: one gets it, the other the next trip
    # or nothing; id-tier holders are never taken from.
    rows = [
        *_walk("036200_1..N01R", _at(f"{DAY}T06:03:00"), LOCAL, train="A"),
        *_walk("036300_1..N01R", _at(f"{DAY}T06:04:00"), LOCAL, train="B"),
        *_walk("037000_1..N01R", _at(f"{DAY}T06:10:00"), LOCAL, train="C"),
    ]
    ms = _match(rows)
    drift = [m for m in ms if m.tier == "drift"]
    trips = [m.scheduled_trip_id for m in ms if m.outcome == "matched"]
    assert len(trips) == len(set(trips))
    assert len(drift) == 1
    loser = next(m for m in ms if m.outcome != "matched")
    assert loser.miss_class == "drift_unassigned"
    assert loser.nearest_scheduled_trip_id is not None


# --- runs: placeholders, tails, window end -----------------------------------------


def test_placeholder_rules() -> None:
    at = _at(f"{DAY}T06:00:00")
    assert is_raw_placeholder(None, 1, at, at)
    assert is_raw_placeholder(False, 1, at, at)
    assert not is_raw_placeholder(True, 1, at, at)
    assert is_raw_placeholder(True, None, at + 61, at)  # SIR flags every trip
    assert is_legacy_placeholder("ace", False, at - 600, at)  # B division
    assert not is_legacy_placeholder("si", False, at - 600, at)
    assert is_legacy_placeholder("numbered", False, at + 30, at)
    # A placeholder-only trip gives no run; it is counted on a real run of
    # the same trip id when one holds it.
    assert build_runs("raw", [], [("numbered", "036000_1..N01R", at)], {}, None) == []


def test_a_trailing_old_block_keeps_its_arrival_and_ends_in_a_stale_tail() -> None:
    t0 = _at(f"{DAY}T06:00:00")
    rows = _walk("036000_1..N01R", t0, LOCAL[:2])
    # The train reaches 103N at +240 s and its timestamp freezes there.
    tail = [
        _row(t0 + 240 + 60 * k, "036000_1..N01R", "103N", vts=t0 + 240)
        for k in range(10)
    ]
    (run,) = _runs([*rows, *tail])
    # Rows older than 300 s add nothing: the block's fresh rows are kept.
    assert [r.at for r in run.kept()][-1] == tail[5].at
    assert run.stops() == ("101N", "102N", "103N")
    assert run.end_state == "stale_tail"
    assert run.n_old_rows == 4
    m = _one(trip_matches([run], Matcher(_index, TZ), ""))
    assert (m.end_state, m.n_old_timestamp_rows, m.stale_only) == (
        "stale_tail",
        4,
        False,
    )


def test_a_held_old_block_followed_by_movement_stays_a_stop() -> None:
    t0 = _at(f"{DAY}T06:00:00")
    held = [_row(t0 + 60 * k, "036000_1..N01R", "102N", vts=t0 - 400) for k in range(5)]
    moved = _walk("036000_1..N01R", t0 + 300, ["103N", "104N"])
    (run,) = _runs([*held, *moved])
    assert run.stops() == ("102N", "103N", "104N")
    assert run.end_state == "window_end"
    assert not run.stale_only


def test_runs_split_at_30_minutes_and_left_feed_versus_window_end() -> None:
    t0 = _at(f"{DAY}T06:00:00")
    a = _walk("036000_1..N01R", t0, LOCAL[:2])
    b = _walk("036000_1..N01R", t0 + 240 + 1801, LOCAL[2:4])
    runs = _runs([*a, *b], last_poll=t0 + 4000)
    assert [r.end_state for r in runs] == ["left_feed", "left_feed"]
    runs = _runs([*a, *b])
    assert [r.end_state for r in runs] == ["left_feed", "window_end"]


# --- identity ----------------------------------------------------------------------


def test_two_trains_sharing_a_trip_id_and_first_poll_get_distinct_runs() -> None:
    t0 = _at(f"{DAY}T06:00:00")
    trip = "036000_1..N01R"
    a = _walk(trip, t0, LOCAL[:3], train="01 0600+ 242/SFY")
    b = [
        _row(r.at, trip, s, entity=1, train="01 0601+ 242/SFY")
        for r, s in zip(
            a, ["101N", "101N", "101N", "101N", "102N", "102N"], strict=True
        )
    ]
    runs = _runs([*a, *b])
    assert len({r.run_id for r in runs}) == 2
    assert {r.track for r in runs} == {0, 1}
    assert all(r.first_at == t0 for r in runs)


def test_a_split_run_joins_its_continuation_under_another_id() -> None:
    t0 = _at(f"{DAY}T06:10:00")
    first = _walk("037000_1..N01R", t0, LOCAL[:2], train="T")
    rest = _walk("037000_1..N", t0 + 300, LOCAL[2:], train="T")
    runs = _runs([*first, *rest])
    (run,) = runs
    assert run.trip_id == "037000_1..N01R"
    assert run.member_run_ids == [f"raw:numbered:037000_1..N@{t0 + 300}#0"]
    m = _one(trip_matches(runs, Matcher(_index, TZ), ""))
    assert "joined_split_run" in m.flags
    assert m.scheduled_trip_id == "SYN-Daily_037000_1..N01R"
    # Different train ids never join.
    rest = _walk("037000_1..N", t0 + 300, LOCAL[2:], train="U")
    assert len(_runs([*first, *rest])) == 2


def _diverging(t0: int, trip: str) -> list[list[Row]]:
    """Polls of two legacy trains with one trip id: both at 101N in the
    first poll (identical rows), then one runs ahead."""
    polls: list[list[Row]] = []
    stops_a = ["101N", "102N", "103N", "104N"]
    stops_b = ["101N", "101N", "102N", "103N"]
    for k, (sa, sb) in enumerate(zip(stops_a, stops_b, strict=True)):
        at = t0 + 60 * k
        vts_b = t0 if k < 2 else at
        polls.append(
            [
                _row(
                    at,
                    trip,
                    sa,
                    raw=False,
                    poll=f"archive/trace/{at}.json",
                    vts=at if k else t0,
                ),
                _row(
                    at, trip, sb, raw=False, poll=f"archive/trace/{at}.json", vts=vts_b
                ),
            ]
        )
    return polls


def test_identity_tie_is_independent_of_entity_order() -> None:
    t0 = _at(f"{DAY}T06:00:00")
    trip = "036000_1..N01R"
    seen: set[tuple[tuple[str, tuple[str, ...]], ...]] = set()
    rng = random.Random(7)
    for _ in range(40):
        rows: list[Row] = []
        for poll in _diverging(t0, trip):
            order = list(range(len(poll)))
            rng.shuffle(order)
            rows += [_reindex(poll[i], e) for e, i in enumerate(order)]
        rng.shuffle(rows)
        runs = _runs(rows, lineage="legacy")
        seen.add(tuple(sorted((r.run_id, r.stops()) for r in runs)))
    assert len(seen) == 1
    ((a, sa), (b, sb)) = next(iter(seen))
    assert a != b
    assert {sa, sb} == {("101N", "102N", "103N", "104N"), ("101N", "102N", "103N")}


def _reindex(r: Row, entity: int) -> Row:
    return Row(
        r.at,
        r.poll,
        entity,
        r.feed,
        r.trip_id,
        r.stop_id,
        r.status,
        r.vts,
        r.route,
        r.start_date,
        r.train_id,
    )


@pytest.mark.parametrize(("trains", "runs"), [(("X", "X"), 1), (("X", "Y"), 2)])
def test_identical_tracks_merge_only_with_equal_train_ids(
    trains: tuple[str, str], runs: int
) -> None:
    t0 = _at(f"{DAY}T06:00:00")
    trip = "036000_1..N01R"
    rows = [
        _reindex(r, e)
        for e, train in enumerate(trains)
        for r in _walk(trip, t0, LOCAL[:3], train=train)
    ]
    got = _runs(rows)
    assert len(got) == runs
    ms = trip_matches(got, Matcher(_index, TZ), "")
    if runs == 1:
        assert ms[0].flags == ["deduplicated_by_position"]
        assert ms[0].track == 0
    else:
        a, b = ms
        assert a.identity_uncertain_with == [b.run_id]
        assert b.identity_uncertain_with == [a.run_id]


def test_a_run_the_continuity_rule_cannot_split_is_flagged() -> None:
    t0 = _at(f"{DAY}T06:00:00")
    trip = "036000_1..N01R"
    rows: list[Row] = []
    # Two rows in every poll, both moving backwards: no continuity.
    for k, (a, b) in enumerate([("103N", "104N"), ("101N", "102N"), ("101N", "102N")]):
        at = t0 + 60 * k
        rows += [
            _row(at, trip, a, raw=False, poll=f"p{at}"),
            _row(at, trip, b, raw=False, poll=f"p{at}", entity=1),
        ]
    rows.append(_row(t0 + 180, trip, "105N", raw=False, poll=f"p{t0 + 180}"))
    (run,) = _runs(rows, lineage="legacy")
    assert "duplicate_trip_id" in run.flags
    assert run.track == 0
    assert run.stops() == ("105N",)  # the shared polls give no stops


# --- pattern fit -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("stops", "attribution", "compatible"),
    [
        (["101N", "102N"], "unique_fit", ["1-N-local"]),
        (["104N", "102N"], "off_pattern", []),
        (["103N", "104N"], "ambiguous", ["1-N-local", "1-N-short"]),
    ],
)
def test_pattern_fit_from_observations_alone(
    stops: list[str], attribution: str, compatible: list[str]
) -> None:
    # 12:00: no trip near, so the run stays unmatched and keeps its fit.
    m = _one(_match(_walk("072000_1..N01R", _at(f"{DAY}T12:00:00"), stops)))
    assert m.outcome == "unscheduled"
    assert (m.pattern_attribution, m.compatible_pattern_ids) == (
        attribution,
        compatible,
    )
    assert m.pattern_id == (compatible[0] if attribution == "unique_fit" else None)


# --- supply ------------------------------------------------------------------------


def _supply(
    rows: list[Row], polls: dict[GtfsRtSource, list[int]], span: tuple[int, int]
) -> list[SupplyObservation]:
    runs = _runs(rows, last_poll=span[1]) if rows else []
    ms = trip_matches(runs, Matcher(_index, TZ), "")
    return supply_observations(
        "raw",
        {f: [(at, f"{f}@{at}000") for at in ats] for f, ats in polls.items()},
        runs,
        ms,
        {},
        _index,
        TZ,
        span,
    )


@pytest.mark.parametrize("feed", ["numbered", "g"])
def test_a_fresh_feed_with_a_route_at_zero_stays_covered_and_shows_zero_supply(
    feed: GtfsRtSource,
) -> None:
    t0 = _at(f"{DAY}T06:20:00")
    polls = [t0 + 60 * k for k in range(50)]
    # Route 2 keeps running on numbered; route 1, or the whole g feed, has no
    # train for 50 minutes while every fetch is good.
    rows = (
        _walk("036000_2..N01R", t0, ["201N", "202N", "203N"], step=1000)
        if feed == "numbered"
        else []
    )
    zeroed = "1" if feed == "numbered" else "G"
    evidence = Evidence(
        "raw", (polls[0], polls[-1]), polls, {feed: [Poll(at, "good") for at in polls]}
    )
    intervals, _ = coverage_intervals(evidence)
    assert [i.state for i in intervals if i.source == feed] == ["covered"]
    recs = _supply(rows, {feed: polls}, (t0, t0 + 3600))
    zero = [r for r in recs if r.route_id == zeroed]
    scheduled = [at for at in polls if (_index(DAY) or _never()).supply(zeroed, at)]
    # Every fetch in a scheduled minute records the route at 0, none else.
    assert [r.observed_at for r in zero] == scheduled
    assert len(scheduled) >= 20
    assert all(r.vehicles == 0 and r.vehicles_stale_tail == 0 for r in zero)
    assert all(r.scheduled_in_service > 0 for r in zero)
    if feed == "numbered":
        assert sum(r.vehicles for r in recs if r.route_id == "2") > 0


def test_z_trains_count_toward_route_z_not_j() -> None:
    t0 = _at(f"{DAY}T08:10:00")
    rows = _walk("049000_J..N01Z", t0, ["J12N", "J14N", "J15N"], feed="jz")
    m = _one(_match(rows))
    assert (m.route_id, m.route_basis, m.scheduled_trip_id) == (
        "Z",
        "scheduled_trip",
        "SYN-Daily_049000_J..N01Z",
    )
    recs = _supply(rows, {"jz": [t0 + 60]}, (t0, t0 + 3600))
    by_route = {r.route_id: r for r in recs}
    assert (by_route["Z"].vehicles, by_route["Z"].route_basis) == (1, "scheduled_trip")
    assert by_route.get("J") is None or by_route["J"].vehicles == 0


def test_old_rows_count_as_stale_tail_supply() -> None:
    t0 = _at(f"{DAY}T06:00:00")
    rows = [
        _row(t0, "036000_1..N01R", "101N"),
        _row(t0 + 60, "036000_1..N01R", "101N", vts=t0 - 600),
    ]
    recs = _supply(rows, {"numbered": [t0, t0 + 60]}, (t0, t0 + 120))
    route1 = [r for r in recs if r.route_id == "1"]
    assert [(r.vehicles, r.vehicles_stale_tail) for r in route1] == [(1, 0), (0, 1)]


def test_legacy_supply_counts_each_feed_from_its_own_rows() -> None:
    # One trace object holds every feed's rows under one poll id.
    t0 = _at(f"{DAY}T06:00:00")
    poll = f"archive/trace/{t0}.json"
    rows = [
        _row(t0, "036000_1..N01R", "101N", raw=False, poll=poll),
        _row(t0, "036000_G..N01R", "G22N", feed="g", raw=False, poll=poll, entity=1),
    ]
    runs = _runs(rows, lineage="legacy")
    ms = trip_matches(runs, Matcher(_index, TZ), "")
    polls: dict[GtfsRtSource, list[tuple[int, str]]] = {
        "numbered": [(t0, poll)],
        "g": [(t0, poll)],
    }
    recs = supply_observations(
        "legacy", polls, runs, ms, None, _index, TZ, (t0, t0 + 60)
    )
    seen = {(r.source, r.route_id): r.vehicles for r in recs if r.vehicles}
    assert seen == {("numbered", "1"): 1, ("g", "G"): 1}
    assert {r.route_id for r in recs if r.source == "g"} == {"G"}


# --- recorded hour: prefix replay and rebuild -----------------------------------------

RECORDED_DAY = "2026-10-20"


@pytest.fixture(scope="module")
def recorded(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("recorded")
    derive(
        LocalStorage(RECORDED),
        LocalStorage(out),
        RECORDED_DAY,
        "test",
        False,
        cutover=0,
    )
    return out


def _recorded_rows(root: Path, monkeypatch: pytest.MonkeyPatch) -> list[Row]:
    monkeypatch.setattr(match_build, "RECORDER_CUTOVER", 0)
    table, digest = load_route_feed()
    start = _utc(f"{RECORDED_DAY}T14:00:00")
    inputs = read_window(LocalStorage(root), "raw", start, start + 3600, table, digest)
    assert not inputs.missing
    assert not inputs.inconsistent
    assert inputs.rows
    return inputs.rows


def _legacy(rows: list[Row]) -> list[Row]:
    """The same polls reduced to legacy fields: no published descriptor."""
    return [
        Row(
            r.at,
            r.poll,
            r.entity,
            r.feed,
            r.trip_id,
            r.stop_id,
            r.status,
            r.vts,
            None,
            None,
            None,
        )
        for r in rows
    ]


def _state(runs: list[Run], until: int) -> dict[str, tuple[object, ...]]:
    """What a replay up to ``until`` must agree on, per run id: rows, kept
    rows and the tail's state."""
    out: dict[str, tuple[object, ...]] = {}
    for r in runs:
        rows = [x for x in r.rows if x.at <= until]
        if not rows:
            continue
        kept = [(x.poll, x.entity) for x in r.kept() if x.at <= until]
        out[r.run_id] = (
            tuple((x.poll, x.entity) for x in rows),
            tuple(kept),
            rows[-1].old,
        )
    return out


@pytest.mark.parametrize("lineage", ["raw", "legacy"])
def test_prefix_replay_with_restart_and_out_of_order_duplicates(
    recorded: Path, monkeypatch: pytest.MonkeyPatch, lineage: str
) -> None:
    rows = _recorded_rows(recorded, monkeypatch)
    if lineage == "legacy":
        rows = _legacy(rows)
    full = _runs(rows, lineage=lineage)
    instants = sorted({r.at for r in rows})
    rng = random.Random(lineage)
    for until in instants:
        prefix = [r for r in rows if r.at <= until]
        # Restart mid-prefix: a first delivery up to the middle, then a
        # second that repeats part of it, every row shuffled.
        cut = until - (until - instants[0]) // 2
        first = [r for r in prefix if r.at <= cut]
        second = [r for r in prefix if r.at >= cut - 120]
        delivered = first + second
        rng.shuffle(delivered)
        replay = _runs(delivered, lineage=lineage, last_poll=until)
        assert _state(replay, until) == _state(full, until)
        # A run cut by the prefix end is open at window_end or in a stale
        # tail, never decided to have left.
        for r in replay:
            if r.last_at == until:
                assert r.end_state in ("window_end", "stale_tail")


def test_a_later_second_train_never_renumbers_the_first() -> None:
    t0 = _at(f"{DAY}T06:00:00")
    trip = "036000_1..N01R"
    # A starts at 102N; B appears later at 101N, earlier in the pattern,
    # which a pattern-position-first order would number before A.
    a = [
        _row(r.at, trip, r.stop_id, raw=False, poll=f"p{r.at}")
        for r in _walk(trip, t0, LOCAL[1:], raw=False)
    ]
    b = [
        _row(r.at + 240, trip, r.stop_id, raw=False, poll=f"p{r.at + 240}", entity=1)
        for r in _walk(trip, t0, LOCAL[:3], raw=False)
    ]
    rows = [*a, *b]
    full = _runs(rows, lineage="legacy")
    assert [r.run_id for r in full] == [
        f"legacy:numbered:{trip}@{t0}#0",
        f"legacy:numbered:{trip}@{t0 + 240}#1",
    ]
    for until in range(t0, t0 + 600, 60):
        prefix = _runs([r for r in rows if r.at <= until], lineage="legacy")
        first = min(prefix, key=lambda r: r.first_at)
        assert first.run_id == f"legacy:numbered:{trip}@{t0}#0"
        assert _state(prefix, until) == _state(full, until)


def test_duplicate_invocations_in_one_second_are_both_kept_in_order(
    recorded: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = _recorded_rows(recorded, monkeypatch)
    r0 = rows[0]
    twin = Row(
        r0.at,
        r0.poll + "x",
        0,
        r0.feed,
        r0.trip_id,
        r0.stop_id,
        r0.status,
        r0.vts,
        r0.route,
        r0.start_date,
        r0.train_id,
    )
    forward = _runs([*rows, twin])
    backward = _runs([twin, *reversed(rows)])
    assert [(r.run_id, [(x.poll, x.entity) for x in r.rows]) for r in forward] == [
        (r.run_id, [(x.poll, x.entity) for x in r.rows]) for r in backward
    ]


def _network_root(tmp_path: Path) -> Path:
    root = tmp_path / "network"
    root.mkdir()
    raw = gzip.decompress(
        (FIXTURES / "events" / f"network_{BASE_BUILD}_subset.json.gz").read_bytes()
    )
    net = NetworkSnapshot.model_validate_json(raw)
    (root / f"{BASE_BUILD}.json.gz").write_bytes(gzip.compress(raw, mtime=0))
    feed = {
        "build_id": BASE_BUILD,
        "content_digest": blake3.blake3(raw).hexdigest(),
        "feed_version": net.feed.feed_version,
        "valid_from": net.feed.valid_from,
        "valid_to": net.feed.valid_to,
        "inputs": net.provenance.inputs,
        "applied_retroactively": net.provenance.applied_retroactively,
    }
    (root / "index.json").write_text(
        json.dumps({"schema": "network/s1", "feeds": {net.feed.feed_id: feed}})
    )
    return root


def test_trip_match_and_supply_files_rebuild_byte_identically(
    recorded: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(match_build, "RECORDER_CUTOVER", 0)
    network = _network_root(tmp_path)
    files: dict[str, list[bytes]] = {}
    for n in range(2):
        out = tmp_path / f"out{n}"
        for cmd, flag, period in (
            ("trip-match", "--service-date", RECORDED_DAY),
            ("supply", "--day", RECORDED_DAY),
        ):
            argv = [
                cmd,
                "--root",
                str(recorded),
                "--network",
                str(network),
                "--out",
                str(out),
                flag,
                period,
                "--lineage",
                "raw",
                "--build-id",
                BASE_BUILD,
            ]
            assert events_main(argv) == 1  # the window's other hours are missing
            capsys.readouterr()
            assert events_main([*argv, "--partial"]) == 0
            report = json.loads(capsys.readouterr().out)
            assert report["reports"][0]["outcome"] == "written"
            assert events_main([*argv, "--partial"]) == 0
            report = json.loads(capsys.readouterr().out)
            assert report["reports"][0]["outcome"] == "unchanged"
        for p in sorted(out.rglob("*.parquet")):
            files.setdefault(p.relative_to(out).as_posix(), []).append(p.read_bytes())
    assert set(files) == {
        f"model/events/s1/raw/trip_match/{RECORDED_DAY}.parquet",
        f"model/events/s1/raw/supply/{RECORDED_DAY}.parquet",
    }
    assert all(a == b for a, b in files.values())
    matches, meta = read_records(
        files[f"model/events/s1/raw/trip_match/{RECORDED_DAY}.parquet"][0], TripMatch
    )
    assert matches
    assert all(m.service_date == RECORDED_DAY for m in matches)
    assert meta.provenance.period_start == RECORDED_DAY
    assert any(
        k.startswith("model/events/s1/match_index/") for k in meta.provenance.inputs
    )
    supply, _ = read_records(
        files[f"model/events/s1/raw/supply/{RECORDED_DAY}.parquet"][0],
        SupplyObservation,
    )
    assert supply
    assert all(isinstance(s.listed_trips, int) for s in supply)


def test_service_day_window_spans_local_midnight_minus_2h_to_0800_next_day() -> None:
    start, end = service_day_window("2026-11-01", TZ)
    assert start == (wall_instant_of("2026-11-01", 0, TZ) or 0) - 7200
    assert end == _utc("2026-11-02T13:00:00")  # 08:00 EST
