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
    "00:30": 30 * 60,
    "02:30": 150 * 60,
    "06:00": 360 * 60,
    "25:30": 1530 * 60,
    "26:30": 1590 * 60,
}
SYNTH_DATES = ["2026-03-07", "2026-03-08", "2026-10-31", "2026-11-01"]


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
    ("service_date", "gtfs", "first_arrival_utc"),
    [
        # Spring forward, 2026-03-08: GTFS time counts from 23:00 EST on 03-07.
        ("2026-03-08", "00:30", "2026-03-08T04:30:00"),
        ("2026-03-08", "02:30", "2026-03-08T06:30:00"),  # 01:30 EST
        ("2026-03-08", "06:00", "2026-03-08T10:00:00"),  # 06:00 EDT
        ("2026-03-07", "25:30", "2026-03-08T06:30:00"),  # 01:30 EST, before the jump
        ("2026-03-07", "26:30", "2026-03-08T07:30:00"),  # 03:30 EDT, after it
        # Fall back, 2026-11-01: GTFS time counts from 01:00 EDT on 11-01.
        ("2026-11-01", "00:30", "2026-11-01T05:30:00"),  # the first 01:30 (EDT)
        ("2026-11-01", "06:00", "2026-11-01T11:00:00"),  # 06:00 EST
        ("2026-10-31", "25:30", "2026-11-01T05:30:00"),  # the first 01:30 (EDT)
        ("2026-10-31", "26:30", "2026-11-01T06:30:00"),  # the second 01:30 (EST)
    ],
)
def test_dst_nights_give_utc_instants(
    service_date: str, gtfs: str, first_arrival_utc: str
) -> None:
    ix = _index(_synthetic(), service_date)
    origin = f"{SYNTH_ORIGINS[gtfs] // 60 * 100:06d}"
    trip_id = f"SYN-Daily_{origin}_1..N03R"
    start = _utc(first_arrival_utc)
    assert [ix.arrival(trip_id, s) for s in ("101N", "102N", "103N")] == [
        start,
        start + 120,
        start + 240,
    ]
    assert start in ix.stop_times("1-N-synthetic", "101N").instants
    assert ix.supply("1", start + 30) >= 1


def test_fall_back_hour_supply_counts_both_days() -> None:
    """At 05:31 UTC on 11-01 (01:31 EDT) the 10-31 25:30 trip and the 11-01
    00:30 trip are both in service; at 06:31 UTC (01:31 EST) only 10-31's
    26:30 trip is."""
    net = _synthetic()
    sat, sun = _index(net, "2026-10-31"), _index(net, "2026-11-01")
    edt, est = _utc("2026-11-01T05:31:00"), _utc("2026-11-01T06:31:00")
    assert (sat.supply("1", edt), sun.supply("1", edt)) == (1, 1)
    assert (sat.supply("1", est), sun.supply("1", est)) == (1, 0)


def test_wrapped_origin_maps_to_previous_service_date() -> None:
    """A realtime id ``009000_1..N`` (01:30) seen at 01:31 on 03-08 names the
    25:30 trip (static origin 153000) of every day that runs it. Only the
    previous service date's one is near in time; no 03-08 trip starts at
    01:30."""
    net = _synthetic()
    seen = _utc("2026-03-08T06:31:00")  # 01:31 EST on 03-08
    key = parse_realtime("009000_1..N")
    assert key is not None
    built = {d.date for d in net.service_days}
    offsets: dict[tuple[str, str, bool], int] = {}
    for day in candidate_service_dates(seen, TZ):
        if day not in built:
            continue
        ix = _index(net, day)
        for h in ix.realtime_key(key):
            arrival = ix.arrival(h.trip.trip_id, "101N")
            assert arrival is not None
            offsets[(day, h.trip.trip_id, h.wrapped)] = seen - arrival
    assert all(wrapped for _d, _t, wrapped in offsets)
    nearest = min(offsets, key=lambda k: abs(offsets[k]))
    assert nearest == ("2026-03-07", "SYN-Daily_153000_1..N03R", True)
    assert offsets[nearest] == 60
    assert all(abs(v) > 12 * 3600 for k, v in offsets.items() if k != nearest)


def test_missing_route_feed_entry_fails_the_build() -> None:
    net = _synthetic()
    trip = net.trips[0].model_copy(
        update={"trip_id": "SYN-Daily_003000_9Q..N03R", "route_id": "9Q"}
    )
    with pytest.raises(MatchIndexBuildError):
        _index(net.model_copy(update={"trips": [trip]}), "2026-03-08")


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
    keys = [parse_key(r["key"]) for r in first]
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
    index = read_network_index(_root(tmp_path))
    unbuilt = "f" * 64
    day_map = DayFeedMap(
        days={
            "2026-09-15": _day(NEW_FEED, None),
            "2026-08-26": _day(None, OLD_FEED),
            "2026-08-01": _day(None, None),
            "2026-08-02": _day(unbuilt, None),
        }
    )
    assert select_build("2026-09-15", index, day_map) == BuildChoice(
        NEW_FEED, "606d9f640f5aad67", "verified"
    )
    assert select_build("2026-08-26", index, day_map) == BuildChoice(
        OLD_FEED, "2faf6350456df7f6", "inferred"
    )
    assert select_build("2026-08-26", index, None, "606d9f640f5aad67") == BuildChoice(
        NEW_FEED, "606d9f640f5aad67", "explicit"
    )
    for service_date, dm, build in [
        ("2026-10-21", day_map, None),  # not in the map
        ("2026-08-01", day_map, None),  # the map names no feed
        ("2026-08-02", day_map, None),  # the feed is not built
        ("2026-09-15", None, None),  # no map and no build named
        ("2026-09-15", day_map, "0000000000000000"),  # unknown build
    ]:
        with pytest.raises(MatchIndexBuildError):
            select_build(service_date, index, dm, build)


def test_missing_layer1_files_are_build_errors(tmp_path: Path) -> None:
    with pytest.raises(MatchIndexBuildError):
        read_network_index(tmp_path)
    with pytest.raises(MatchIndexBuildError):
        read_day_map(tmp_path / "day_feed_map.json")
    root = _root(tmp_path)
    (root / "2faf6350456df7f6.json.gz").unlink()
    index = read_network_index(root)
    with pytest.raises(MatchIndexBuildError):
        read_network(root, BuildChoice(OLD_FEED, "2faf6350456df7f6", "explicit"), index)
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
        "2026-03-08",
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
    other = serialize(_index(_synthetic(), "2026-03-07").document)
    for stored in (other, b"not gzip"):
        assert resolve_write(stored, doc, False) == ("conflict", None)
        assert resolve_write(stored, doc, True) == ("replaced", data)
