"""Network layer (schema network/s1): contracts, builder invariants, and the
cross-language guards for the generated schema and fixture.

The fixture feed (tests/fixtures/model/gtfs_mini) is small but deliberately
awkward: a holiday that swaps Weekday for Sunday service, a trip running past
24:00, two routes sharing a segment, an express hop, a branch, a trip with no
direction_id, a platform used in both directions, and a two-station complex.
"""

from __future__ import annotations

import gzip
import io
import json
import random
import zipfile
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from momentarily.model.conventions import Model
from momentarily.model.network.build import (
    CodeStamp,
    Inputs,
    NetworkBuildError,
    build_network,
    check_chains,
    load_overrides,
    serialize,
)
from momentarily.model.network.contracts import (
    IndexEntry,
    NetworkIndex,
    NetworkSnapshot,
)
from momentarily.model.network.day_feed import resolve_day
from momentarily.model.network.stations import canonical_bytes, parse_snapshot
from scripts.export_model_schema import SCHEMA_DIR, render_schemas
from scripts.gen_model_network_fixture import (
    FIXTURE_CODE,
    FIXTURE_PATH,
    GTFS_DIR,
    STATIONS_PATH,
    fixture_inputs,
    render_fixture,
    zip_dir,
)
from training.gtfs_static import read_calendar
from training.traversal_archive import DayProvenance, FeedDigestRange

FEED_ID_A = "a" * 64
FEED_ID_B = "b" * 64


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


# --- cross-language guards ----------------------------------------------


def test_committed_model_schemas_match_pydantic() -> None:
    """If this fails, run: uv run python -m scripts.export_model_schema
    and then (cd worker && bun run gen:model)."""
    rendered = render_schemas()
    committed = {p.name: p.read_text() for p in SCHEMA_DIR.glob("*.schema.json")}
    assert committed == rendered, "schema/model/s1 is stale"


def test_committed_network_fixture_is_fresh() -> None:
    """If this fails, run: uv run python -m scripts.gen_model_network_fixture"""
    assert FIXTURE_PATH.read_text() == render_fixture()


# --- conventions --------------------------------------------------------


def test_contract_fields_cannot_default() -> None:
    with pytest.raises(TypeError, match="defaults"):

        class _Defaulted(Model):  # pyright: ignore[reportUnusedClass]
            value: int = 0


# --- determinism ----------------------------------------------------------


def test_identical_inputs_give_identical_bytes() -> None:
    first = serialize(build_network(fixture_inputs(), stamp=FIXTURE_CODE))
    second = serialize(build_network(fixture_inputs(), stamp=FIXTURE_CODE))
    assert first == second
    assert gzip.decompress(first)  # a real gzip stream


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
    with pytest.raises(ValueError, match="canonical"):
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


def test_service_days_match_gtfs_calendar(snap: NetworkSnapshot) -> None:
    with zipfile.ZipFile(io.BytesIO(zip_dir(GTFS_DIR))) as zf:
        calendar = read_calendar(zf)
    dates = [d.date for d in snap.service_days]
    assert dates[0] == "2026-06-01"
    assert dates[-1] == "2026-06-30"
    assert len(dates) == 30
    for day in snap.service_days:
        assert set(day.active_service_ids) == calendar.active(
            date.fromisoformat(day.date)
        )


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


def test_trip_without_direction_id(snap: NetworkSnapshot) -> None:
    trip = next(t for t in snap.trips if t.trip_id == "A-Weekday-0850-branch-N")
    assert trip.direction.direction_id is None
    assert trip.direction.code is None  # 106M has no N/S suffix
    assert trip.pattern_id.startswith("A-x-")


def test_platform_serving_both_directions(snap: NetworkSnapshot) -> None:
    platform = next(p for p in snap.platforms if p.stop_id == "106M")
    assert platform.direction_code is None
    assert platform.station_id == "106"
    labelled = [p for p in snap.platforms if p.direction_code is not None]
    assert all(p.direction_label for p in labelled)


def test_complex_groups_two_stations(snap: NetworkSnapshot) -> None:
    complexes = {c.complex_id: c for c in snap.complexes}
    assert complexes["900"].station_ids == ["103", "201"]
    stations = {s.station_id: s for s in snap.stations}
    assert stations["103"].complex_id == stations["201"].complex_id == "900"
    assert stations["103"].ada == "partial"


def test_station_missing_from_dataset_fails_the_build() -> None:
    retrieved_at, rows = _stations_rows()
    rows = [r for r in rows if r["gtfs_stop_id"] != "104"]
    with pytest.raises(NetworkBuildError) as err:
        build_network(_with_stations(rows, retrieved_at), stamp=FIXTURE_CODE)
    assert any("104" in p for p in err.value.problems)


def test_dataset_row_without_gtfs_station_fails_the_build() -> None:
    retrieved_at, rows = _stations_rows()
    rows = [*rows, {**rows[0], "gtfs_stop_id": "999"}]
    with pytest.raises(NetworkBuildError) as err:
        build_network(_with_stations(rows, retrieved_at), stamp=FIXTURE_CODE)
    assert any("999" in p for p in err.value.problems)


def test_missing_direction_label_fails_unless_overridden() -> None:
    retrieved_at, rows = _stations_rows()
    rows = [
        {**r, "south_direction_label": ""} if r["gtfs_stop_id"] == "102" else r
        for r in rows
    ]
    inputs = _with_stations(rows, retrieved_at)
    with pytest.raises(NetworkBuildError) as err:
        build_network(inputs, stamp=FIXTURE_CODE)
    assert any("102S" in p for p in err.value.problems)

    overrides, digest = load_overrides(b'{"102S": "Downtown"}')
    filled = build_network(
        replace(inputs, direction_overrides=overrides, overrides_digest=digest),
        stamp=FIXTURE_CODE,
    )
    label = next(p.direction_label for p in filled.platforms if p.stop_id == "102S")
    assert label == "Downtown"
    assert filled.provenance.inputs["direction_overrides"] == digest
    assert (
        filled.build_id != build_network(fixture_inputs(), stamp=FIXTURE_CODE).build_id
    )


def test_override_that_fills_no_gap_fails_the_build() -> None:
    overrides, digest = load_overrides(b'{"101S": "Downtown"}')
    inputs = replace(
        fixture_inputs(), direction_overrides=overrides, overrides_digest=digest
    )
    with pytest.raises(NetworkBuildError) as err:
        build_network(inputs, stamp=FIXTURE_CODE)
    assert any("101S" in p for p in err.value.problems)


def test_trip_with_unknown_stop_fails_the_build(tmp_path: Path) -> None:
    text = (GTFS_DIR / "stop_times.txt").read_text().replace(",105S,", ",999S,", 1)
    inputs = replace(fixture_inputs(), gtfs=_gtfs_with(tmp_path, stop_times=text))
    with pytest.raises(NetworkBuildError) as err:
        build_network(inputs, stamp=FIXTURE_CODE)
    assert any("999S" in p for p in err.value.problems)


# --- provenance -----------------------------------------------------------


def test_stations_snapshot_is_flagged_retroactive_only_when_newer() -> None:
    retrieved_at, rows = _stations_rows()
    late = build_network(_with_stations(rows, retrieved_at), stamp=FIXTURE_CODE)
    assert late.provenance.applied_retroactively == ["stations"]
    early_at = 1780272000  # 2026-06-01T00:00Z = 2026-05-31 in New York
    early = build_network(_with_stations(rows, early_at), stamp=FIXTURE_CODE)
    assert early.provenance.applied_retroactively == []


def test_build_id_tracks_data_inputs_not_code() -> None:
    a = build_network(fixture_inputs(), stamp=FIXTURE_CODE)
    b = build_network(fixture_inputs(), stamp=CodeStamp(sha="other", dirty=True))
    assert a.build_id == b.build_id
    assert a.provenance.code_sha != b.provenance.code_sha
    retrieved_at, rows = _stations_rows()
    c = build_network(_with_stations(rows, retrieved_at + 1), stamp=FIXTURE_CODE)
    assert c.build_id != a.build_id


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
