"""Network layer (schema network/s1): contracts, builder invariants, and the
cross-language guards for the generated schemas and fixtures.

The fixture feed (tests/fixtures/model/gtfs_mini) is small but deliberately
awkward: a holiday that swaps Weekday for Sunday service, a trip running past
24:00, two routes sharing a segment, an express hop, a branch, a trip with no
direction_id, a direction_id 0 trip with no N/S code, a platform used in both
directions (labelled through the overrides file), and a two-station complex.
"""

from __future__ import annotations

import gzip
import json
import random
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest

from momentarily.archive.traversals import DayProvenance, FeedDigestRange
from momentarily.model.conventions import ContractDefaultError, Model, instant_of
from momentarily.model.network.__main__ import BuildConflictError, write_build
from momentarily.model.network.build import (
    BUILDER_VERSION,
    CodeStamp,
    FirstObserved,
    Inputs,
    NetworkBuildError,
    build_network,
    canonical_json,
    check_chains,
    content_digest,
    first_observed,
    load_overrides,
    network_digest,
    serialize,
)
from momentarily.model.network.contracts import (
    DayFeed,
    DayFeedMap,
    IndexEntry,
    NetworkIndex,
    NetworkSnapshot,
    PatternSegment,
)
from momentarily.model.network.day_feed import resolve_day
from momentarily.model.network.stations import (
    NonCanonicalSnapshotError,
    canonical_bytes,
    parse_snapshot,
)
from scripts.export_model_schema import committed_schemas, render_schemas
from scripts.gen_model_network_fixture import (
    FIXTURE_CODE,
    GTFS_DIR,
    STATIONS_PATH,
    fixture_inputs,
    render_fixtures,
    zip_dir,
)

FEED_ID_A = "a" * 64
FEED_ID_B = "b" * 64
# blake3 of the fixture's canonical JSON without provenance, under
# BUILDER_VERSION "1". Changing either means old build_ids name other data.
PINNED_NETWORK_DIGEST = (
    "40231254b0a3c15663125112bf1b8c396d99d3c564b35d61f57d82f4da5369df"
)
WEEKDAY_NAMES = (
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
)


@pytest.fixture(scope="module")
def snap() -> NetworkSnapshot:
    return build_network(fixture_inputs(), stamp=FIXTURE_CODE)


def _stations_rows() -> tuple[int, list[dict[str, Any]]]:
    doc = json.loads(STATIONS_PATH.read_bytes())
    return int(doc["retrieved_at"]), list(doc["rows"])


def _with_stations(rows: list[dict[str, Any]], retrieved_at: int) -> Inputs:
    return replace(
        fixture_inputs(), stations=parse_snapshot(canonical_bytes(retrieved_at, rows))
    )


def _gtfs_with(tmp_path: Path, **files: str) -> bytes:
    """The fixture feed with some files replaced."""
    for src in GTFS_DIR.glob("*.txt"):
        (tmp_path / src.name).write_bytes(src.read_bytes())
    for name, text in files.items():
        (tmp_path / f"{name}.txt").write_text(text)
    return zip_dir(tmp_path)


def _header_only(name: str) -> str:
    return (GTFS_DIR / f"{name}.txt").read_text().splitlines()[0] + "\n"


def _problems(inputs: Inputs) -> list[str]:
    with pytest.raises(NetworkBuildError) as err:
        build_network(inputs, stamp=FIXTURE_CODE)
    return err.value.problems


# --- cross-language guards ----------------------------------------------


def test_committed_model_schemas_match_pydantic() -> None:
    """If this fails, run: uv run python -m scripts.export_model_schema
    and then (cd worker && bun run gen:model)."""
    rendered = render_schemas()
    assert committed_schemas() == rendered, "schema/model/s1 is stale"


def test_committed_fixtures_are_fresh() -> None:
    """If this fails, run: uv run python -m scripts.gen_model_network_fixture"""
    for path, text in render_fixtures().items():
        assert path.read_text() == text, f"{path.name} is stale"


# --- conventions --------------------------------------------------------


def test_contract_fields_cannot_default() -> None:
    with pytest.raises(ContractDefaultError) as err:

        class _Defaulted(Model):  # pyright: ignore[reportUnusedClass]
            value: int = 0
            other: str

    assert err.value.fields == ["value"]


@pytest.mark.parametrize(
    ("service_date", "seconds", "expected"),
    [
        # Spring forward: noon EDT is 16:00Z, so the GTFS origin is 04:00Z
        # (23:00 EST the night before), not local midnight (05:00Z).
        ("2026-03-08", 0, 1772942400),
        ("2026-03-08", 24 * 3600 + 4 * 60, 1772942400 + 86640),
        # Fall back: noon EST is 17:00Z, origin 05:00Z (01:00 EDT), not 04:00Z.
        ("2026-11-01", 0, 1793509200),
        ("2026-11-01", 8 * 3600, 1793509200 + 28800),
        # Ordinary day: origin is local midnight, 04:00Z under EDT.
        ("2026-06-19", 0, 1781841600),
    ],
)
def test_instant_of_follows_the_noon_minus_12h_rule(
    service_date: str, seconds: int, expected: int
) -> None:
    assert instant_of(service_date, seconds, "America/New_York") == expected


# --- determinism ----------------------------------------------------------


def test_identical_inputs_give_identical_content() -> None:
    first = build_network(fixture_inputs(), stamp=FIXTURE_CODE)
    second = build_network(fixture_inputs(), stamp=FIXTURE_CODE)
    assert content_digest(first) == content_digest(second)
    assert gzip.decompress(serialize(first)) == canonical_json(second)


def test_row_order_does_not_change_the_network(tmp_path: Path) -> None:
    """Only the zip bytes (hence feed_id and build_id) may change when the
    feed's rows are reordered; every network fact must stay put."""
    rng = random.Random(7)
    shuffled: dict[str, str] = {}
    for name in ("trips", "stop_times", "stops", "routes"):
        header, *rows = (GTFS_DIR / f"{name}.txt").read_text().splitlines()
        rng.shuffle(rows)
        shuffled[name] = "\n".join([header, *rows]) + "\n"
    inputs = replace(fixture_inputs(), gtfs=_gtfs_with(tmp_path, **shuffled))
    base = build_network(fixture_inputs(), stamp=FIXTURE_CODE)
    moved = build_network(inputs, stamp=FIXTURE_CODE)
    assert moved.feed.feed_id != base.feed.feed_id
    ignore = {"build_id", "provenance", "feed"}
    assert moved.model_dump(exclude=ignore) == base.model_dump(exclude=ignore)


def test_stations_digest_ignores_portal_columns_and_row_order() -> None:
    retrieved_at, rows = _stations_rows()
    noisy = [{**r, ":@computed_region_xyz": "42"} for r in reversed(rows)]
    assert canonical_bytes(retrieved_at, noisy) == STATIONS_PATH.read_bytes()


def test_non_canonical_stations_snapshot_is_refused() -> None:
    doc = json.loads(STATIONS_PATH.read_bytes())
    with pytest.raises(NonCanonicalSnapshotError):
        parse_snapshot(json.dumps(doc, indent=2).encode())


# --- invariants -----------------------------------------------------------


def test_every_trip_follows_its_pattern_and_patterns_chain(
    snap: NetworkSnapshot,
) -> None:
    patterns = {p.pattern_id: p for p in snap.patterns}
    for trip in snap.trips:
        pattern = patterns[trip.pattern_id]
        assert [s for s, _, _ in trip.stop_times] == pattern.stops
        assert (trip.route_id, trip.direction) == (pattern.route_id, pattern.direction)
    assert all(len(p.stops) >= 2 for p in snap.patterns)
    assert check_chains(snap.patterns, snap.pattern_segments, snap.segments) == []
    assert sum(p.n_trips for p in snap.patterns) == len(snap.trips)


def test_check_chains_catches_a_broken_chain(snap: NetworkSnapshot) -> None:
    target = snap.pattern_segments[1]
    other = next(
        s.segment_id
        for s in snap.segments
        if s.from_stop != target.segment_id.split(">")[0]
    )
    corrupted = [
        PatternSegment(pattern_id=ps.pattern_id, seq=ps.seq, segment_id=other)
        if ps == target
        else ps
        for ps in snap.pattern_segments
    ]
    problems = check_chains(snap.patterns, corrupted, snap.segments)
    assert len(problems) == 1
    assert target.pattern_id in problems[0]
    gapped = [ps for ps in snap.pattern_segments if ps != target]
    assert check_chains(snap.patterns, gapped, snap.segments) != []


def test_service_days_follow_the_declared_services(snap: NetworkSnapshot) -> None:
    """Reconstruct each day's active set from the Service rows alone
    (half-open [start, end), weekdays, added, removed)."""
    for day in snap.service_days:
        d = date.fromisoformat(day.date)
        expected = {
            s.service_id
            for s in snap.services
            if (
                (
                    s.start is not None
                    and s.end is not None
                    and s.start <= day.date < s.end
                    and WEEKDAY_NAMES[d.weekday()] in s.weekdays
                )
                or day.date in s.added_dates
            )
            and day.date not in s.removed_dates
        }
        assert set(day.active_service_ids) == expected, day.date


def test_service_days_cover_the_feed_window(snap: NetworkSnapshot) -> None:
    dates = [date.fromisoformat(d.date) for d in snap.service_days]
    assert snap.feed.valid_from == dates[0].isoformat()
    assert snap.feed.valid_to == (dates[-1] + timedelta(days=1)).isoformat()
    assert dates == [dates[0] + timedelta(days=i) for i in range(len(dates))]


# --- fixture cases --------------------------------------------------------


def test_holiday_runs_sunday_service(snap: NetworkSnapshot) -> None:
    days = {d.date: d for d in snap.service_days}
    holiday = days["2026-06-19"]
    assert holiday.active_service_ids == ["Sunday"]
    assert holiday.day_type == "sunday"
    assert holiday.is_exception
    assert days["2026-06-18"].day_type == "weekday"
    assert not days["2026-06-18"].is_exception
    weekday = next(s for s in snap.services if s.service_id == "Weekday")
    assert weekday.removed_dates == ["2026-06-19"]
    assert weekday.end == "2026-07-01"  # half-open: calendar end_date + 1


def test_trip_past_midnight_keeps_its_service_day(snap: NetworkSnapshot) -> None:
    trip = next(t for t in snap.trips if t.trip_id == "A-Weekday-2358-local-S")
    assert trip.service_id == "Weekday"
    assert trip.stop_times[0][1] == 23 * 3600 + 58 * 60
    assert trip.stop_times[-1][1] == 24 * 3600 + 6 * 60


def test_routes_share_a_segment(snap: NetworkSnapshot) -> None:
    segments = {s.segment_id: s for s in snap.segments}
    assert segments["102S>103S"].route_ids == ["A", "B"]
    assert segments["103S>201S"].route_ids == ["B"]


def test_express_hop_is_its_own_segment(snap: NetworkSnapshot) -> None:
    segments = {s.segment_id: s for s in snap.segments}
    express = segments["101S>103S"]
    assert len(express.pattern_ids) == 1
    pattern = next(p for p in snap.patterns if p.pattern_id == express.pattern_ids[0])
    assert pattern.stops == ["101S", "103S", "104S", "105S"]
    assert pattern.shape_ids == ["A..S02", "A..S03"]  # shapes don't split it


def test_branch_gives_two_patterns_for_one_route_direction(
    snap: NetworkSnapshot,
) -> None:
    ends = {
        p.stops[-1]
        for p in snap.patterns
        if p.route_id == "A" and p.direction.direction_id == 1
    }
    assert ends == {"105S", "106M"}


def test_pattern_id_is_stable(snap: NetworkSnapshot) -> None:
    """Pattern ids are keys later layers store; this one must never move."""
    local = next(
        p for p in snap.patterns if p.stops == ["101S", "102S", "103S", "104S", "105S"]
    )
    assert local.pattern_id == "A-S-45f4d05ed81b"


def test_pattern_tag_falls_back_from_code_to_direction_id(
    snap: NetworkSnapshot,
) -> None:
    tags = {t.trip_id: t.pattern_id.split("-")[1] for t in snap.trips}
    assert tags["A-Weekday-0810-local-N"] == "N"  # code wins over direction_id 0
    assert tags["A-Weekday-0950-shortturn-N"] == "0"  # no code, direction_id 0
    assert tags["A-Weekday-0850-branch-N"] == "x"  # neither


def test_trip_without_direction_id(snap: NetworkSnapshot) -> None:
    trip = next(t for t in snap.trips if t.trip_id == "A-Weekday-0850-branch-N")
    assert trip.direction.direction_id is None
    assert trip.direction.code is None  # 106M has no N/S suffix


def test_every_platform_has_a_label(snap: NetworkSnapshot) -> None:
    assert all(p.direction_label for p in snap.platforms)
    platform = next(p for p in snap.platforms if p.stop_id == "106M")
    assert platform.direction_code is None
    assert platform.direction_label == "All trains"  # from the overrides file


def test_platform_serving_both_directions_needs_an_override() -> None:
    inputs = replace(fixture_inputs(), direction_overrides={}, overrides_digest=None)
    assert any("106M" in p for p in _problems(inputs))


def test_complex_groups_two_stations(snap: NetworkSnapshot) -> None:
    complexes = {c.complex_id: c for c in snap.complexes}
    assert complexes["900"].station_ids == ["103", "201"]
    stations = {s.station_id: s for s in snap.stations}
    assert stations["103"].complex_id == stations["201"].complex_id == "900"
    assert stations["103"].ada == "partial"


def test_station_missing_from_dataset_fails_the_build() -> None:
    retrieved_at, rows = _stations_rows()
    rows = [r for r in rows if r["gtfs_stop_id"] != "104"]
    assert any("104" in p for p in _problems(_with_stations(rows, retrieved_at)))


def test_dataset_row_without_gtfs_station_fails_the_build() -> None:
    retrieved_at, rows = _stations_rows()
    rows = [*rows, {**rows[0], "gtfs_stop_id": "999"}]
    assert any("999" in p for p in _problems(_with_stations(rows, retrieved_at)))


def test_missing_direction_label_fails_unless_overridden() -> None:
    retrieved_at, rows = _stations_rows()
    rows = [
        {**r, "south_direction_label": ""} if r["gtfs_stop_id"] == "102" else r
        for r in rows
    ]
    inputs = _with_stations(rows, retrieved_at)
    assert any("102S" in p for p in _problems(inputs))

    overrides, digest = load_overrides(b'{"102S": "Downtown", "106M": "All trains"}')
    filled = build_network(
        replace(inputs, direction_overrides=overrides, overrides_digest=digest),
        stamp=FIXTURE_CODE,
    )
    label = next(p.direction_label for p in filled.platforms if p.stop_id == "102S")
    assert label == "Downtown"
    assert filled.provenance.inputs["direction_overrides"] == digest


def test_override_that_fills_no_gap_fails_the_build() -> None:
    overrides, digest = load_overrides(b'{"101S": "Downtown", "106M": "All trains"}')
    inputs = replace(
        fixture_inputs(), direction_overrides=overrides, overrides_digest=digest
    )
    assert any("101S" in p for p in _problems(inputs))


def test_trip_with_unknown_stop_fails_the_build(tmp_path: Path) -> None:
    text = (GTFS_DIR / "stop_times.txt").read_text().replace(",105S,", ",999S,", 1)
    inputs = replace(fixture_inputs(), gtfs=_gtfs_with(tmp_path, stop_times=text))
    assert any("999S" in p for p in _problems(inputs))


@pytest.mark.parametrize("name", ["routes", "trips", "stop_times"])
def test_empty_core_file_fails_the_build(tmp_path: Path, name: str) -> None:
    gtfs = _gtfs_with(tmp_path, **{name: _header_only(name)})
    assert any(
        f"{name}.txt" in p for p in _problems(replace(fixture_inputs(), gtfs=gtfs))
    )


def test_duplicate_trip_id_fails_the_build(tmp_path: Path) -> None:
    text = (GTFS_DIR / "trips.txt").read_text()
    dup = text.splitlines()[1]
    gtfs = _gtfs_with(tmp_path, trips=text + dup + "\n")
    trip_id = dup.split(",")[1]
    assert any(trip_id in p for p in _problems(replace(fixture_inputs(), gtfs=gtfs)))


# --- provenance and identity ----------------------------------------------


def test_stations_snapshot_is_flagged_retroactive_only_when_newer() -> None:
    retrieved_at, rows = _stations_rows()  # captured 2026-06-15
    late = build_network(_with_stations(rows, retrieved_at), stamp=FIXTURE_CODE)
    assert late.provenance.applied_retroactively == ["stations"]
    assert late.provenance.captured_at == {"stations": retrieved_at}
    early_at = 1780272000  # 2026-06-01T00:00Z = 2026-05-31 in New York
    early = build_network(_with_stations(rows, early_at), stamp=FIXTURE_CODE)
    assert early.provenance.applied_retroactively == []


def _observed(utc_day: str) -> FirstObserved:
    return FirstObserved(utc_day=date.fromisoformat(utc_day), source_digest="c" * 64)


def test_first_observed_utc_day_moves_the_retroactive_boundary() -> None:
    """Stations captured 2026-06-15T12:00Z; the archive day is a UTC day."""

    def build(utc_day: str) -> NetworkSnapshot:
        inputs = replace(fixture_inputs(), first_observed=_observed(utc_day))
        return build_network(inputs, stamp=FIXTURE_CODE)

    later = build("2026-06-16")
    assert later.provenance.observed_from_utc_day == "2026-06-16"
    assert later.provenance.applied_retroactively == []
    assert later.provenance.inputs["day_feed_map"] == "c" * 64
    # Captured during the first observed day: part of it ran without the data.
    assert build("2026-06-15").provenance.applied_retroactively == ["stations"]
    assert build("2026-06-10").provenance.applied_retroactively == ["stations"]


def test_first_observed_reads_verified_and_inferred_days() -> None:
    gtfs = fixture_inputs().gtfs
    feed_id = build_network(fixture_inputs(), stamp=FIXTURE_CODE).feed.feed_id

    def day(fid: str | None, inferred: str | None) -> DayFeed:
        return DayFeed(
            feed_id=fid,
            verified=fid is not None,
            reason=None,
            feed_version=None,
            inferred_feed_id=inferred,
        )

    day_map = (
        DayFeedMap(
            days={
                "2026-06-03": day(FEED_ID_A, None),
                "2026-06-05": day(None, feed_id),
                "2026-06-09": day(feed_id, None),
            }
        )
        .model_dump_json()
        .encode()
    )
    seen = first_observed(day_map, gtfs)
    assert seen is not None
    assert seen.utc_day == date(2026, 6, 5)
    assert first_observed(DayFeedMap(days={}).model_dump_json().encode(), gtfs) is None


def test_build_id_tracks_data_inputs_system_and_builder_not_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    a = build_network(fixture_inputs(), stamp=FIXTURE_CODE)
    b = build_network(fixture_inputs(), stamp=CodeStamp(sha="other", dirty=True))
    assert a.build_id == b.build_id
    assert a.provenance.code_sha != b.provenance.code_sha
    retrieved_at, rows = _stations_rows()
    c = build_network(_with_stations(rows, retrieved_at + 1), stamp=FIXTURE_CODE)
    assert c.build_id != a.build_id
    d = build_network(fixture_inputs(), system_id="other", stamp=FIXTURE_CODE)
    assert d.build_id != a.build_id
    no_overrides = replace(
        fixture_inputs(),
        direction_overrides={"106M": "All trains"},
        overrides_digest=None,
    )
    assert build_network(no_overrides, stamp=FIXTURE_CODE).build_id != a.build_id
    monkeypatch.setattr("momentarily.model.network.build.BUILDER_VERSION", "next")
    assert build_network(fixture_inputs(), stamp=FIXTURE_CODE).build_id != a.build_id


def test_network_data_is_pinned_to_builder_version(snap: NetworkSnapshot) -> None:
    assert BUILDER_VERSION == "1"
    assert network_digest(snap) == PINNED_NETWORK_DIGEST, (
        "the fixture's network data changed: bump BUILDER_VERSION in "
        "momentarily/model/network/build.py and update both pins here"
    )


def test_rebuild_refreshes_provenance_but_never_changes_network_data(
    tmp_path: Path,
) -> None:
    first = build_network(fixture_inputs(), stamp=FIXTURE_CODE)
    path, entry, written = write_build(tmp_path, first)
    assert written
    assert entry.content_digest == content_digest(first)

    # Identical rebuild: nothing to write.
    assert not write_build(tmp_path, first)[2]

    # Different network data under the same build_id: refused, file untouched.
    original = path.read_bytes()
    changed = first.model_copy(update={"routes": first.routes[:1]})
    with pytest.raises(BuildConflictError) as err:
        write_build(tmp_path, changed)
    assert err.value.path == path
    assert path.read_bytes() == original


def test_day_map_provenance_is_refreshed_in_both_directions(tmp_path: Path) -> None:
    plain = build_network(fixture_inputs(), stamp=FIXTURE_CODE)
    mapped = build_network(
        replace(fixture_inputs(), first_observed=_observed("2026-06-16")),
        stamp=FIXTURE_CODE,
    )
    assert plain.build_id == mapped.build_id
    assert plain.provenance.applied_retroactively == ["stations"]
    assert mapped.provenance.applied_retroactively == []

    for i, (before, after) in enumerate(((plain, mapped), (mapped, plain))):
        out = tmp_path / str(i)
        write_build(out, before)
        path, entry, written = write_build(out, after)
        assert written
        on_disk = NetworkSnapshot.model_validate_json(
            gzip.decompress(path.read_bytes())
        )
        assert on_disk.provenance == after.provenance
        assert entry.content_digest == content_digest(after)
        assert entry.applied_retroactively == after.provenance.applied_retroactively


def test_unreadable_existing_build_is_a_conflict(tmp_path: Path) -> None:
    snap = build_network(fixture_inputs(), stamp=FIXTURE_CODE)
    path = tmp_path / f"{snap.build_id}.json.gz"
    for junk in (b"not gzip", gzip.compress(b'{"schema": "network/s0"}')):
        path.write_bytes(junk)
        with pytest.raises(BuildConflictError) as err:
            write_build(tmp_path, snap)
        assert err.value.path == path
        assert path.read_bytes() == junk


# --- day -> feed map ------------------------------------------------------


def _prov(**kw: Any) -> DayProvenance:
    base: dict[str, Any] = {
        "schema": 1,
        "extractor": 1,
        "feed_version": "v-new",
        "feed_digest": None,
        "n_rows": 1,
        "n_source_objects": 1,
        "source_manifest": "",
        "code_sha": None,
        "written_at": 0,
    }
    return DayProvenance(**{**base, **kw})


_INDEX = NetworkIndex(
    schema="network/s1",
    feeds={
        fid: IndexEntry(
            build_id="0" * 16,
            content_digest="0" * 64,
            feed_version=version,
            valid_from=None,
            valid_to=None,
            inputs={},
            applied_retroactively=[],
        )
        for fid, version in ((FEED_ID_A, "v-old"), (FEED_ID_B, "v-new"))
    },
)


def test_stamped_day_is_verified() -> None:
    day = resolve_day(_prov(feed_digest=FEED_ID_A), _INDEX)
    assert (day.feed_id, day.verified) == (FEED_ID_A, True)


def test_unstamped_day_is_unverified_with_a_version_lead() -> None:
    day = resolve_day(_prov(), _INDEX)
    assert (day.feed_id, day.verified, day.inferred_feed_id) == (None, False, FEED_ID_B)
    assert day.reason


def test_partially_stamped_day_is_unverified() -> None:
    ranges = [FeedDigestRange(digest=FEED_ID_B, first_ts=10, last_ts=20)]
    day = resolve_day(_prov(feed_digests=ranges), _INDEX)
    assert (day.feed_id, day.verified, day.inferred_feed_id) == (None, False, FEED_ID_B)
    mixed = [*ranges, FeedDigestRange(digest=FEED_ID_A, first_ts=30, last_ts=40)]
    day = resolve_day(_prov(feed_digests=mixed), _INDEX)
    assert (day.verified, day.inferred_feed_id) == (False, None)
