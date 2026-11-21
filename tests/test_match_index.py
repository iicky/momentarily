"""The per-service-day match index (layer 3, slice 1).

Agreement with the prototype. The scoping prototype's ``Net`` (the network
lookups the matching figures in docs/model-events.md were measured with,
kept outside the repo in ``<work>/step3_scope/holdup_lib.py``) was run on the
two full layer-1 builds. ``tests/fixtures/model/events/net_agreement.json``
holds, per build and service date, the size and blake3 digest of each lookup
family it gave, restricted to the routes in ``routes``. The fixture networks
``network_<build_id>_subset.json.gz`` are those builds with every pattern,
platform, service and service day, but only the trips of those routes, and no
stations or segments. Each family here is computed from the index alone and
must hash to the prototype's value.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import blake3
import pytest

from momentarily.gtfs.static import base_route
from momentarily.model.events.__main__ import main as events_main
from momentarily.model.events.keys import MatchIndexKey, parse_key
from momentarily.model.events.match_index import (
    INDEX_VERSION,
    BuildChoice,
    IndexInputs,
    MatchIndex,
    MatchIndexBuildError,
    MatchIndexDocument,
    build_match_index,
    candidate_service_dates,
    deserialize,
    read_day_map,
    read_network,
    read_network_index,
    resolve_write,
    select_build,
    serialize,
)
from momentarily.model.events.route_feed import (
    RouteFeedError,
    derive_route_feed,
    load_route_feed,
)
from momentarily.model.events.trip_ids import WRAP, TripIdKey, parse_realtime
from momentarily.model.network.contracts import (
    DayFeed,
    DayFeedMap,
    DirectionCode,
    NetworkSnapshot,
    ScheduledTrip,
)
from momentarily.model.observations.keys import KeyLayoutError

FIXTURES = Path(__file__).parent / "fixtures" / "model" / "events"
AGREEMENT: dict[str, Any] = json.loads((FIXTURES / "net_agreement.json").read_text())
TZ = "America/New_York"


def _network(build_id: str) -> tuple[NetworkSnapshot, bytes]:
    raw = gzip.decompress(
        (FIXTURES / f"network_{build_id}_subset.json.gz").read_bytes()
    )
    return NetworkSnapshot.model_validate_json(raw), raw


def _index(net: NetworkSnapshot, service_date: str) -> MatchIndex:
    table, digest = load_route_feed()
    doc = build_match_index(
        service_date,
        IndexInputs(
            network=net,
            network_digest=blake3.blake3(net.build_id.encode()).hexdigest(),
            choice=BuildChoice(net.feed.feed_id, net.build_id, "explicit"),
            day_map_digest=None,
            route_feed=table,
            route_feed_digest=digest,
            code_sha="test",
            code_dirty=None,
        ),
    )
    return MatchIndex(deserialize(serialize(doc)))


def index_families(ix: MatchIndex, routes: set[str] | None) -> dict[str, Any]:
    """Every lookup the matcher makes, as sorted JSON-able lists. ``routes``
    limits trip-level families to those routes (None: all)."""

    def keep(trip_route: str) -> bool:
        return routes is None or trip_route in routes

    trips = [t for t in ix.trips if keep(t.route_id)]
    keys: set[tuple[int, str, DirectionCode]] = {
        (t.origin, t.route_field, t.direction) for t in trips
    }
    pids = {t.pattern_id for t in trips}
    supply: list[Any] = []
    folded = None if routes is None else {base_route(r) for r in routes}
    for route, s in ix.document.supply.items():
        if folded is not None and route not in folded:
            continue
        minutes = range(s.start_minute, s.start_minute + len(s.counts))
        supply.append(
            [
                route,
                [
                    [m, ix.supply(route, m * 60)]
                    for m in minutes
                    if ix.supply(route, m * 60)
                ],
            ]
        )
    return {
        "active": sorted(ix.active_service_ids),
        "keys": sorted(
            [
                o,
                f,
                d,
                sorted(t.trip_id for t in ix.static_key(o, f, d) if keep(t.route_id)),
            ]
            for o, f, d in keys
        ),
        "wrapped": sorted(
            [
                o - WRAP,
                f,
                d,
                sorted(
                    h.trip.trip_id
                    for h in ix.realtime_key(TripIdKey(o - WRAP, f, d, ""))
                    if h.wrapped and keep(h.trip.route_id)
                ),
            ]
            for o, f, d in keys
            if o >= WRAP
        ),
        "origin_direction": sorted(
            [
                o,
                d,
                sorted(
                    t.trip_id for t in ix.origin_direction(o, d) if keep(t.route_id)
                ),
            ]
            for o, d in {(o, d) for o, _f, d in keys}
        ),
        "meta": sorted(
            [
                t.trip_id,
                t.origin,
                t.route_field,
                t.direction,
                t.path,
                t.route_id,
                t.service_id,
                t.block,
            ]
            for t in trips
        ),
        "arrivals": sorted(
            [t.trip_id, s, ix.arrival(t.trip_id, s)]
            for t in trips
            for s in ix.pattern_stops(t.pattern_id)
        ),
        "stop_times": sorted(
            [p, s, [[a, t] for a, t in zip(st.instants, st.trip_ids, strict=True)]]
            for p in pids
            for s in ix.pattern_stops(p)
            for st in [ix.stop_times(p, s)]
        ),
        "patterns_of": sorted(
            [br, d, sorted(ix.patterns_of(br, d))]
            for br, d in {
                (base_route(p.route_id), p.direction) for p in ix.document.patterns
            }
        ),
        "pattern_stops": sorted(
            [p.pattern_id, list(ix.pattern_stops(p.pattern_id))]
            for p in ix.document.patterns
        ),
        "platforms": sorted(ix.platforms),
        "supply": sorted(supply),
    }


def family_digest(value: Any) -> dict[str, Any]:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return {"n": len(value), "blake3": blake3.blake3(data).hexdigest()}


CASES = [
    (build, day)
    for build, days in sorted(AGREEMENT["expected"].items())
    for day in sorted(days)
]


@pytest.mark.parametrize(("build_id", "service_date"), CASES)
def test_lookups_agree_with_prototype_net(build_id: str, service_date: str) -> None:
    net, _ = _network(build_id)
    ix = _index(net, service_date)
    got = {
        k: family_digest(v)
        for k, v in index_families(ix, set(AGREEMENT["routes"])).items()
    }
    assert got == AGREEMENT["expected"][build_id][service_date]


@pytest.mark.parametrize(
    ("at", "days"), [tuple(x) for x in AGREEMENT["candidate_days"]]
)
def test_candidate_days_agree_with_prototype_net(at: int, days: list[str]) -> None:
    assert list(candidate_service_dates(at, TZ)) == days


# --- synthetic network: the 24:00 wrap and DST ------------------------------------


def _utc(text: str) -> int:
    return int(datetime.fromisoformat(text).replace(tzinfo=UTC).timestamp())


# GTFS times of the synthetic trips (one per origin), each running 101N, 102N,
# 103N two minutes apart.
SYNTH_ORIGINS = {
    "00:14": 14 * 60,
    "00:30": 30 * 60,
    "01:30": 90 * 60,
    "01:59": 119 * 60,
    "02:30": 150 * 60,
    "03:00": 180 * 60,
    "06:00": 360 * 60,
    "25:30": 1530 * 60,
    "26:30": 1590 * 60,
}
# Fall back on 2026-11-01 at 06:00 UTC (02:00 EDT -> 01:00 EST); spring
# forward on 2027-03-14 at 07:00 UTC (02:00 EST -> 03:00 EDT).
SYNTH_DATES = ["2026-10-31", "2026-11-01", "2027-03-13", "2027-03-14"]


def _trip_id(gtfs: str) -> str:
    return f"SYN-Daily_{SYNTH_ORIGINS[gtfs] // 60 * 100:06d}_1..N03R"


def _synthetic() -> NetworkSnapshot:
    base, _ = _network("606d9f640f5aad67")
    stops = ["101N", "102N", "103N"]
    pattern = base.patterns[0].model_copy(
        update={"pattern_id": "1-N-synthetic", "route_id": "1", "stops": stops}
    )
    trips: list[ScheduledTrip] = []
    for secs in SYNTH_ORIGINS.values():
        origin = f"{secs // 60 * 100:06d}"
        trips.append(
            base.trips[0].model_copy(
                update={
                    "trip_id": f"SYN-Daily_{origin}_1..N03R",
                    "route_id": "1",
                    "pattern_id": pattern.pattern_id,
                    "service_id": "Daily",
                    "stop_times": [
                        (s, secs + 120 * i, secs + 120 * i) for i, s in enumerate(stops)
                    ],
                }
            )
        )
    days = [
        base.service_days[0].model_copy(
            update={
                "date": d,
                "active_service_ids": ["Daily"],
                "day_type": "mixed",
                "is_exception": False,
            }
        )
        for d in SYNTH_DATES
    ]
    return base.model_copy(
        update={"patterns": [pattern], "trips": trips, "service_days": days}
    )


@pytest.mark.parametrize(
    ("service_date", "gtfs", "arrivals_utc", "spans_dst"),
    [
        # Fall back. The feed starts trips on the wall clock: 00:14 on 11-01 is
        # 00:14 EDT (instant_of's noon-minus-12h origin, 01:00 EDT, gives 05:14).
        ("2026-11-01", "00:14", ["04:14", "04:16", "04:18"], False),
        ("2026-11-01", "01:30", ["05:30", "05:32", "05:34"], False),  # first 01:30
        # Departs 01:59 EDT and runs straight through the change: no hour gap.
        ("2026-11-01", "01:59", ["05:59", "06:01", "06:03"], True),
        ("2026-11-01", "06:00", ["11:00", "11:02", "11:04"], False),  # 06:00 EST
        ("2026-10-31", "25:30", ["05:30", "05:32", "05:34"], False),  # first 01:30
        ("2026-10-31", "26:30", ["07:30", "07:32", "07:34"], False),  # 02:30 EST
        # Spring forward: 02:00-02:59 on 03-14 does not exist. A trip departing
        # then is unresolved; one departing before runs on by elapsed time.
        ("2027-03-14", "00:30", ["05:30", "05:32", "05:34"], False),  # 00:30 EST
        ("2027-03-14", "01:59", ["06:59", "07:01", "07:03"], True),  # 01:59 EST
        ("2027-03-14", "02:30", [None, None, None], False),
        ("2027-03-14", "03:00", ["07:00", "07:02", "07:04"], False),  # 03:00 EDT
        ("2027-03-13", "25:30", ["06:30", "06:32", "06:34"], False),  # 01:30 EST
        ("2027-03-13", "26:30", [None, None, None], False),  # 02:30 on 03-14
    ],
)
def test_dst_nights_anchor_trips_on_the_wall_clock(
    service_date: str, gtfs: str, arrivals_utc: list[str | None], spans_dst: bool
) -> None:
    ix = _index(_synthetic(), service_date)
    utc_date = "2027-03-14" if service_date.startswith("2027") else "2026-11-01"
    expected = [
        None if hm is None else _utc(f"{utc_date}T{hm}:00") for hm in arrivals_utc
    ]
    trip_id = _trip_id(gtfs)
    assert [ix.arrival(trip_id, s) for s in ("101N", "102N", "103N")] == expected
    for stop, instant in zip(("101N", "102N", "103N"), expected, strict=True):
        st = ix.stop_times("1-N-synthetic", stop)
        assert (trip_id in st.trip_ids) == (instant is not None)
        if instant is not None:
            assert st.instants[st.trip_ids.index(trip_id)] == instant
    trip = ix.trip(trip_id)
    assert trip is not None
    assert (trip.first_departure, trip.last_arrival) == (expected[0], expected[-1])
    assert trip.spans_dst_transition is spans_dst


@pytest.mark.parametrize(
    ("service_date", "arrivals_utc"),
    [
        # 02:02 EST exists once: the anchor is 07:02 UTC, and the 01:57
        # arrival lies 5 min before it, the second 01:57 (EST), not the first.
        ("2026-11-01", ["06:57", "07:04", "07:06"]),
        # 02:02 does not exist: the whole trip is unresolved, though its
        # 01:57 arrival does exist on the wall clock.
        ("2027-03-14", [None, None, None]),
    ],
)
def test_a_trip_is_anchored_on_its_first_departure_not_its_first_arrival(
    service_date: str, arrivals_utc: list[str | None]
) -> None:
    """A dwell at the first stop: arrives 01:57, departs 02:02. The realtime
    feeds start trips by their departure, so the anchor is the departure and
    every other time (the first arrival included) is placed by elapsed
    scheduled time."""
    net = _synthetic()
    stop_times = [(stop, m * 60, m * 60) for stop, m in (("102N", 124), ("103N", 126))]
    trip = net.trips[0].model_copy(
        update={
            "trip_id": "SYN-Daily_011700_1..N03R",
            "stop_times": [("101N", 117 * 60, 122 * 60), *stop_times],
        }
    )
    ix = _index(net.model_copy(update={"trips": [trip]}), service_date)
    expected = [
        None if hm is None else _utc(f"{service_date}T{hm}:00") for hm in arrivals_utc
    ]
    assert [ix.arrival(trip.trip_id, s) for s in ("101N", "102N", "103N")] == expected
    found = ix.trip(trip.trip_id)
    assert found is not None
    departure = None if expected[0] is None else expected[0] + 300
    assert found.first_departure == departure


def test_an_unresolved_trip_still_answers_its_realtime_key() -> None:
    """The 02:30 trip on 2027-03-14 has no instant, but its realtime id
    ``015000_1..N`` still finds it, so the matcher can tell an unresolved
    trip from an unknown id."""
    ix = _index(_synthetic(), "2027-03-14")
    key = parse_realtime("015000_1..N")
    assert key is not None
    hits = [h for h in ix.realtime_key(key) if not h.wrapped]
    assert [h.trip.trip_id for h in hits] == [_trip_id("02:30")]
    assert (hits[0].trip.first_departure, hits[0].trip.last_arrival) == (None, None)
    assert ix.arrival(_trip_id("02:30"), "101N") is None


def test_unresolved_times_are_null_in_the_stored_document() -> None:
    ix = _index(_synthetic(), "2027-03-14")
    pattern = next(p for p in ix.document.patterns if p.trips)
    row = [ix.trips[k].trip_id for k in pattern.trips].index(_trip_id("02:30"))
    assert [arr[row] for arr in pattern.arrivals] == [None, None, None]
    # Only the 02:30 trip; the 01:59 trip departs before the gap.
    assert ix.metadata.entries["unresolved_stop_times"] == 3
    assert ix.metadata.entries["unresolved_trips"] == 1
    assert ix.metadata.entries["trips_spanning_dst"] == 1
    stored = json.loads(gzip.decompress(serialize(ix.document)))
    assert [arr[row] for arr in stored["patterns"][0]["arrivals"]] == [None] * 3


def test_spring_forward_keeps_resolved_times_in_schedule_order() -> None:
    ix = _index(_synthetic(), "2027-03-14")
    by_schedule = sorted(SYNTH_ORIGINS, key=SYNTH_ORIGINS.__getitem__)
    for stop in ("101N", "102N", "103N"):
        st = ix.stop_times("1-N-synthetic", stop)
        assert list(st.instants) == sorted(set(st.instants))
        resolved = [
            _trip_id(g)
            for g in by_schedule
            if ix.arrival(_trip_id(g), stop) is not None
        ]
        assert list(st.trip_ids) == resolved


def test_spring_forward_gap_has_no_scheduled_supply() -> None:
    """On 03-14 the 02:30 trip is in no minute. The 01:59 trip runs 06:59 to
    07:03 UTC by elapsed time, overlapping the 03:00 trip from 07:00 UTC."""
    ix = _index(_synthetic(), "2027-03-14")
    s = ix.document.supply["1"]
    busy = Counter(
        datetime.fromtimestamp((s.start_minute + k) * 60, UTC).strftime("%m-%d %H:%M")
        for k, n in enumerate(s.counts)
        for _ in range(n)
    )
    spans = {  # trip: first and last UTC minute on 03-14 or 03-15
        "00:14": ("03-14", 5 * 60 + 14, 5 * 60 + 18),
        "00:30": ("03-14", 5 * 60 + 30, 5 * 60 + 34),
        "01:30": ("03-14", 6 * 60 + 30, 6 * 60 + 34),
        "01:59": ("03-14", 6 * 60 + 59, 7 * 60 + 3),
        "03:00": ("03-14", 7 * 60, 7 * 60 + 4),
        "06:00": ("03-14", 10 * 60, 10 * 60 + 4),
        "25:30": ("03-15", 5 * 60 + 30, 5 * 60 + 34),  # 01:30 EDT
        "26:30": ("03-15", 6 * 60 + 30, 6 * 60 + 34),  # 02:30 EDT
    }
    assert busy == Counter(
        f"{d} {m // 60:02d}:{m % 60:02d}"
        for d, lo, hi in spans.values()
        for m in range(lo, hi + 1)
    )


def test_fall_back_hour_supply_counts_first_occurrence_only() -> None:
    """At 05:31 UTC on 11-01 (the first 01:31, EDT) the 10-31 25:30 trip and
    the 11-01 01:30 trip are both in service. At 06:02 UTC (the second 01:02,
    EST) only the 11-01 01:59 trip is, finishing its run; at 06:31 UTC (the
    second 01:31) none is. At 07:31 UTC (02:31 EST) 10-31's 26:30 trip and
    11-01's 02:30 trip are, both departing 02:30 on the wall clock."""
    net = _synthetic()
    sat, sun = _index(net, "2026-10-31"), _index(net, "2026-11-01")
    first, through, second, after = (
        _utc(f"2026-11-01T{hm}:00") for hm in ("05:31", "06:02", "06:31", "07:31")
    )
    assert (sat.supply("1", first), sun.supply("1", first)) == (1, 1)
    assert (sat.supply("1", through), sun.supply("1", through)) == (0, 1)
    assert (sat.supply("1", second), sun.supply("1", second)) == (0, 0)
    assert (sat.supply("1", after), sun.supply("1", after)) == (1, 1)


def test_wrapped_origin_maps_to_previous_service_date() -> None:
    """A realtime id ``009000_1..N`` (01:30) seen at 01:31 on 2027-03-14 names
    the 25:30 trip (static origin 153000) of every day that runs it, wrapped.
    Only the previous service date's one is near in time. (It also names each
    day's own 01:30 trip, unwrapped.)"""
    net = _synthetic()
    seen = _utc("2027-03-14T06:31:00")  # 01:31 EST on 03-14
    key = parse_realtime("009000_1..N")
    assert key is not None
    built = {d.date for d in net.service_days}
    offsets: dict[tuple[str, str, bool], int] = {}
    for day in candidate_service_dates(seen, TZ):
        if day not in built:
            continue
        ix = _index(net, day)
        for h in ix.realtime_key(key):
            if not h.wrapped:
                continue
            arrival = ix.arrival(h.trip.trip_id, "101N")
            assert arrival is not None
            offsets[(day, h.trip.trip_id, h.wrapped)] = seen - arrival
    nearest = min(offsets, key=lambda k: abs(offsets[k]))
    assert nearest == ("2027-03-13", "SYN-Daily_153000_1..N03R", True)
    assert offsets[nearest] == 60
    assert all(abs(v) > 12 * 3600 for k, v in offsets.items() if k != nearest)


def test_missing_route_feed_entry_fails_the_build() -> None:
    net = _synthetic()
    trip = net.trips[0].model_copy(
        update={"trip_id": "SYN-Daily_003000_9Q..N03R", "route_id": "9Q"}
    )
    with pytest.raises(MatchIndexBuildError):
        _index(net.model_copy(update={"trips": [trip]}), "2026-11-01")


# --- the CLI on a layer-1 root: build per day, byte-identical rebuild -----------------


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "network"
    root.mkdir()
    feeds: dict[str, Any] = {}
    for build_id in ("2faf6350456df7f6", "606d9f640f5aad67"):
        net, raw = _network(build_id)
        (root / f"{build_id}.json.gz").write_bytes(gzip.compress(raw, mtime=0))
        feeds[net.feed.feed_id] = {
            "build_id": build_id,
            "content_digest": blake3.blake3(raw).hexdigest(),
            "feed_version": net.feed.feed_version,
            "valid_from": net.feed.valid_from,
            "valid_to": net.feed.valid_to,
            "inputs": net.provenance.inputs,
            "applied_retroactively": net.provenance.applied_retroactively,
        }
    (root / "index.json").write_text(
        json.dumps({"schema": "network/s1", "feeds": feeds})
    )
    (root / "day_feed_map.json").write_bytes(
        (FIXTURES / "day_feed_map.json").read_bytes()
    )
    return root


def test_cli_chooses_the_build_per_day_and_rebuilds_identically(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, out = _root(tmp_path), tmp_path / "out"
    argv = [
        "match-index",
        "--network",
        str(root),
        "--date",
        "2026-08-26",
        "--date",
        "2026-08-27",
        "--out",
        str(out),
    ]
    assert events_main(argv) == 0
    first = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    keys = [k for r in first if isinstance(k := parse_key(r["key"]), MatchIndexKey)]
    assert len(keys) == len(first)
    _, rf_digest = load_route_feed()
    assert [(k.service_date, k.build_id) for k in keys] == [
        ("2026-08-26", "2faf6350456df7f6"),
        ("2026-08-27", "606d9f640f5aad67"),
    ]
    assert {(k.index_version, k.route_feed) for k in keys} == {
        (INDEX_VERSION, rf_digest[:16])
    }
    before = {r["key"]: (out / r["key"]).read_bytes() for r in first}
    assert events_main(argv) == 0
    second = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["outcome"] for r in second] == ["unchanged", "unchanged"]
    assert {k: (out / k).read_bytes() for k in before} == before
    # Other bytes under a key are a conflict until --overwrite.
    k26, k27 = before
    (out / k26).write_bytes(before[k27])
    assert events_main(argv) == 1
    assert (out / k26).read_bytes() == before[k27]
    assert events_main([*argv, "--overwrite"]) == 0
    assert (out / k26).read_bytes() == before[k26]


def test_cli_fails_for_a_day_the_map_does_not_cover(tmp_path: Path) -> None:
    root = _root(tmp_path)
    argv = [
        "match-index",
        "--network",
        str(root),
        "--date",
        "2026-07-03",
        "--out",
        str(tmp_path / "o"),
    ]
    assert events_main(argv) == 1
    argv += ["--build-id", "2faf6350456df7f6"]
    assert events_main(argv) == 0


# --- RouteFeed and keys ----------------------------------------------------------------


def test_route_feed_takes_the_majority_feed_per_route_key() -> None:
    table = derive_route_feed(
        [
            ("bdfm", "F", "010000_F..N", 1),
            ("bdfm", "F", "020000_F..N", 2),
            ("ace", "F", "030000_F..N", 3),
            ("nqrw", "W", "040000_N..N", 4),
        ],
        ["vehicle"],
    )
    assert table.routes["F"].feed == "bdfm"
    assert table.routes["F"].also_in == ["ace"]
    assert table.feed_of("N") == "nqrw"  # W trains carry N in the trip id
    with pytest.raises(RouteFeedError):
        derive_route_feed([("bdfm", "F", "1", 1), ("ace", "F", "2", 2)], ["vehicle"])


def test_packaged_route_feed_covers_every_route_of_both_builds() -> None:
    table, _ = load_route_feed()
    for build_id in AGREEMENT["expected"]:
        net, _ = _network(build_id)
        assert {r.route_id for r in net.routes} <= set(table.routes)


def test_match_index_key_round_trips_and_rejects_bad_parts() -> None:
    key = MatchIndexKey(1, "2026-09-15", "606d9f640f5aad67", "0123456789abcdef")
    assert parse_key(key.key) == key
    with pytest.raises(KeyLayoutError):
        MatchIndexKey(1, "2026-02-30", "606d9f640f5aad67", "0123456789abcdef")
    with pytest.raises(KeyLayoutError):
        MatchIndexKey(0, "2026-09-15", "606d9f640f5aad67", "0123456789abcdef")
    with pytest.raises(KeyLayoutError):
        parse_key("model/events/s1/match_index/2026-09-15/606d9f640f5aad67.json.gz")


# --- choosing the build and reading a layer-1 root ----------------------------------

OLD_FEED = "056d1fc821f26e859c41ffb227197bdd45dbf4a0c3ef41b3509a5ff6dda4602e"
NEW_FEED = "2c9e69b82ac596a2fe660b56656509570cfba3c2e4c50be75eaf624eee1bc6f8"


def _day(feed_id: str | None, inferred: str | None) -> DayFeed:
    return DayFeed(
        feed_id=feed_id,
        verified=feed_id is not None,
        reason=None if feed_id else "unstamped",
        feed_version=None,
        inferred_feed_id=inferred,
    )


def test_select_build_reads_the_day_map_and_refuses_what_it_cannot_resolve(
    tmp_path: Path,
) -> None:
    root = _root(tmp_path)
    index = read_network_index(root)
    unbuilt = "f" * 64
    day_map = DayFeedMap(
        days={
            "2026-09-15": _day(NEW_FEED, None),
            "2026-08-26": _day(None, OLD_FEED),
            "2026-08-01": _day(None, None),
            "2026-08-02": _day(unbuilt, None),
        }
    )

    def snapshot(feed_id: str) -> NetworkSnapshot:
        return read_network(root, feed_id, index)

    def select(
        service_date: str, dm: DayFeedMap | None, build: str | None = None
    ) -> BuildChoice:
        return select_build(service_date, index, dm, build, snapshot=snapshot)

    assert select("2026-09-15", day_map) == BuildChoice(
        NEW_FEED, "606d9f640f5aad67", "live"
    )
    assert select("2026-08-26", day_map) == BuildChoice(
        OLD_FEED, "2faf6350456df7f6", "live_inferred"
    )
    assert select("2026-08-26", None, "606d9f640f5aad67") == BuildChoice(
        NEW_FEED, "606d9f640f5aad67", "explicit"
    )
    for service_date, dm, build in [
        ("2026-10-21", day_map, None),  # not in the map
        ("2026-08-01", day_map, None),  # no map feed, and none published by then
        ("2026-08-02", day_map, None),  # the feed is not built
        ("2026-09-15", None, None),  # no map and no build named
        ("2026-09-15", day_map, "0000000000000000"),  # unknown build
    ]:
        with pytest.raises(MatchIndexBuildError):
            select(service_date, dm, build)


def test_a_feed_live_before_its_calendar_starts_leaves_those_days_to_the_previous_feed(
    tmp_path: Path,
) -> None:
    # The new feed is live from 09-15 but its calendar starts on 09-18, as the
    # MTA's fall 2026 pick was live from 10-29 with service from 11-01.
    root = _root(tmp_path)
    index = read_network_index(root)
    old, _ = _network("2faf6350456df7f6")
    full, _ = _network("606d9f640f5aad67")
    new = full.model_copy(
        update={
            "service_days": [d for d in full.service_days if d.date >= "2026-09-18"]
        }
    )
    nets = {OLD_FEED: old, NEW_FEED: new}
    live = ["2026-09-15", "2026-09-16", "2026-09-17", "2026-09-18", "2026-11-05"]
    day_map = DayFeedMap(
        days={"2026-09-01": _day(OLD_FEED, None)}
        | {d: _day(NEW_FEED, None) for d in live}
    )
    chosen = {
        d: select_build(d, index, day_map, snapshot=nets.__getitem__) for d in live[:4]
    }
    assert chosen == {
        "2026-09-15": BuildChoice(OLD_FEED, "2faf6350456df7f6", "previous_covering"),
        "2026-09-16": BuildChoice(OLD_FEED, "2faf6350456df7f6", "previous_covering"),
        "2026-09-17": BuildChoice(OLD_FEED, "2faf6350456df7f6", "previous_covering"),
        "2026-09-18": BuildChoice(NEW_FEED, "606d9f640f5aad67", "live"),
    }
    # The chosen build really schedules the day it was chosen for.
    assert _index(old, "2026-09-15").trips
    # Neither build's calendar reaches 11-05.
    with pytest.raises(MatchIndexBuildError):
        select_build("2026-11-05", index, day_map, snapshot=nets.__getitem__)


def test_the_fallback_skips_non_covering_feeds_and_refuses_an_unbuilt_one(
    tmp_path: Path,
) -> None:
    # Three feeds: OLD schedules 09-15; MID and LATE start their calendars
    # on 09-20. 09-15 is unverified with LATE as its lead.
    index = read_network_index(_root(tmp_path))
    old, _ = _network("2faf6350456df7f6")
    full, _ = _network("606d9f640f5aad67")
    later = full.model_copy(
        update={
            "service_days": [d for d in full.service_days if d.date >= "2026-09-20"]
        }
    )
    mid_feed, late_feed = "a" * 64, "b" * 64
    entry = index.feeds[NEW_FEED]
    feeds = {
        OLD_FEED: index.feeds[OLD_FEED],
        mid_feed: entry.model_copy(update={"build_id": "1111111111111111"}),
        late_feed: entry.model_copy(update={"build_id": "2222222222222222"}),
    }
    nets = {OLD_FEED: old, mid_feed: later, late_feed: later}
    day_map = DayFeedMap(
        days={
            "2026-09-01": _day(OLD_FEED, None),
            "2026-09-05": _day(mid_feed, None),
            "2026-09-10": _day(late_feed, None),
            "2026-09-15": _day(None, late_feed),
        }
    )

    def select(feeds: dict[str, Any]) -> BuildChoice:
        return select_build(
            "2026-09-15",
            index.model_copy(update={"feeds": feeds}),
            day_map,
            snapshot=nets.__getitem__,
        )

    assert select(feeds) == BuildChoice(
        OLD_FEED, "2faf6350456df7f6", "previous_covering"
    )
    # MID unbuilt: it might schedule 09-15, so OLD is not taken in its place.
    with pytest.raises(MatchIndexBuildError):
        select({f: e for f, e in feeds.items() if f != mid_feed})


def test_a_day_the_map_names_no_feed_for_takes_the_latest_feed_published_by_then(
    tmp_path: Path,
) -> None:
    # 09-14 and 09-16 are mid-day changes (no feed in the map), like 10-28.
    index = read_network_index(_root(tmp_path))
    old, _ = _network("2faf6350456df7f6")
    new, _ = _network("606d9f640f5aad67")
    nets = {OLD_FEED: old, NEW_FEED: new}
    unmapped = ["2026-08-31", "2026-09-14", "2026-09-16", "2026-11-05"]
    day_map = DayFeedMap(
        days={
            "2026-09-01": _day(OLD_FEED, None),
            "2026-09-15": _day(NEW_FEED, None),
        }
        | {d: _day(None, None) for d in unmapped}
    )

    def select(service_date: str) -> BuildChoice:
        return select_build(service_date, index, day_map, snapshot=nets.__getitem__)

    # Both builds schedule 09-14, but the new feed was first named on 09-15.
    assert select("2026-09-14") == BuildChoice(
        OLD_FEED, "2faf6350456df7f6", "covering_unmapped"
    )
    assert select("2026-09-16") == BuildChoice(
        NEW_FEED, "606d9f640f5aad67", "covering_unmapped"
    )
    # No feed published by 08-31; no build's calendar reaches 11-05.
    for service_date in ("2026-08-31", "2026-11-05"):
        with pytest.raises(MatchIndexBuildError):
            select(service_date)


def test_missing_layer1_files_are_build_errors(tmp_path: Path) -> None:
    with pytest.raises(MatchIndexBuildError):
        read_network_index(tmp_path)
    with pytest.raises(MatchIndexBuildError):
        read_day_map(tmp_path / "day_feed_map.json")
    root = _root(tmp_path)
    (root / "2faf6350456df7f6.json.gz").unlink()
    index = read_network_index(root)
    with pytest.raises(MatchIndexBuildError):
        read_network(root, OLD_FEED, index)
    argv = ["match-index", "--network", str(tmp_path / "none"), "--date", "2026-09-15"]
    assert events_main([*argv, "--out", str(tmp_path / "o")]) == 1


# --- supply ---------------------------------------------------------------------------


def test_supply_folds_express_variants_and_sums_by_feed() -> None:
    net = _synthetic()
    local = next(t for t in net.trips if t.trip_id.endswith("_036000_1..N03R"))
    express = local.model_copy(
        update={"trip_id": "SYN-Daily_036000_6..N03R", "route_id": "6X"}
    )
    ix = _index(net.model_copy(update={"trips": [*net.trips, express]}), "2026-11-01")
    at = _utc("2026-11-01T11:01:00")  # both 06:00 trips are running
    assert ix.supply("6", at) == ix.supply("6X", at) == 1
    assert ix.supply("1", at) == 1
    assert "6X" not in ix.document.supply
    assert ix.supply_by_feed(at) == {"numbered": 2}
    assert ix.supply_by_feed(at + 3600) == {}


# --- the stored-file rule ---------------------------------------------------------------


def _doc(code_sha: str) -> MatchIndexDocument:
    table, digest = load_route_feed()
    net = _synthetic()
    return build_match_index(
        "2026-11-01",
        IndexInputs(
            network=net,
            network_digest="0" * 64,
            choice=BuildChoice(net.feed.feed_id, net.build_id, "explicit"),
            day_map_digest=None,
            route_feed=table,
            route_feed_digest=digest,
            code_sha=code_sha,
            code_dirty=False,
        ),
    )


def test_stored_file_rule() -> None:
    doc = _doc("a")
    data = serialize(doc)
    assert resolve_write(None, doc, False) == ("written", data)
    assert resolve_write(data, doc, False) == ("unchanged", None)
    # Only the code stamp differs: keep the stored file.
    assert resolve_write(serialize(_doc("b")), doc, False) == ("unchanged", None)
    other = serialize(_index(_synthetic(), "2026-10-31").document)
    for stored in (other, b"not gzip"):
        assert resolve_write(stored, doc, False) == ("conflict", None)
        assert resolve_write(stored, doc, True) == ("replaced", data)
