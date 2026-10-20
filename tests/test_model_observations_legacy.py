"""Backfill adapters: legacy archives -> legacy-lineage observation records.

tests/fixtures/model/observations/legacy_mirror/ is a tiny mirror cut from
the R2 mirror of 2026-10-20: each archive's first object (the first day of
its legacy period) and a few chosen ones, with trace rows and ENE payloads
truncated. One object is synthetic: archive/alerts/2026-07-22/ holds a
re-sighting of an identical 07-21 entity, made by copying that object with a
later observed_at (the Worker writes such a copy when it loses its dedupe
state; the real mirror holds none).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from momentarily.model.conventions import Unknown
from momentarily.model.observations.columns import NORMALIZED
from momentarily.model.observations.contracts import (
    AlertVersion,
    DerivedFileMetadata,
    EquipmentObservation,
    LegacyFetchOutcome,
    LegacyLineage,
    LegacyRouteAlertState,
    VehicleObservation,
    alert_content_digest,
)
from momentarily.model.observations.keys import PREFIX, DerivedHourKey, utc_day
from momentarily.model.observations.legacy import (
    alerts,
    alerts_liveness,
    ene,
    fresh_feeds,
    health,
    predictions,
    trace,
)
from momentarily.model.observations.legacy.__main__ import main
from momentarily.model.observations.legacy.common import (
    PERIODS,
    LegacyObject,
)
from momentarily.model.observations.legacy.run import run
from momentarily.model.observations.parquet import (
    DerivedFileError,
    file_layout,
    file_metadata,
    read_records,
    write_records,
)

REPO = Path(__file__).resolve().parent.parent
MIRROR = REPO / "tests" / "fixtures" / "model" / "observations" / "legacy_mirror"
LAYOUTS = REPO / "schema" / "model" / "s1" / "observations" / "parquet"
MODELS: dict[str, type[BaseModel]] = {str(k): v for k, v in NORMALIZED.items()}


def _objects(archive: str) -> list[LegacyObject]:
    return [
        LegacyObject(p.relative_to(MIRROR).as_posix(), p.read_bytes())
        for p in sorted((MIRROR / archive).rglob("*"))
        if p.is_file()
    ]


def _paths(record: BaseModel, prefix: str = "") -> dict[str, Any]:
    """Every leaf field of ``record`` by dotted path; lineage excluded."""
    out: dict[str, Any] = {}
    for name in type(record).model_fields:
        value = getattr(record, name)
        if isinstance(value, LegacyLineage):
            continue
        if isinstance(value, BaseModel) and not isinstance(value, Unknown):
            out |= _paths(value, f"{prefix}{name}.")
        else:
            out[prefix + name] = value
    return out


def _assert_kept(record: BaseModel, kept: set[str]) -> None:
    paths = _paths(record)
    assert kept <= set(paths)
    unknown = {p for p, v in paths.items() if isinstance(v, Unknown)}
    assert unknown == set(paths) - kept


_VEHICLE_KEPT = {"observed_at", "trip.trip_id", "stop_id", "vehicle_timestamp"}
_STOPPED_KEPT = {"current_status", "current_stop_sequence"}
_FETCH_KEPT = {"legacy_fetch_id", "source", "observed_at", "succeeded"}


# --- what each adapter keeps (doc §5) ----------------------------------------


def test_trace_rows_keep_only_what_the_trace_kept() -> None:
    stopped = moving = 0
    for obj in _objects("archive/trace"):
        for e in trace.adapt(obj):
            r = e.record
            if isinstance(r, LegacyFetchOutcome):
                _assert_kept(r, _FETCH_KEPT | {"scheduled_at"})
                continue
            assert isinstance(r, VehicleObservation)
            if r.current_status == 1:
                stopped += 1
                _assert_kept(r, _VEHICLE_KEPT | _STOPPED_KEPT)
            else:
                moving += 1
                _assert_kept(r, _VEHICLE_KEPT)
    assert stopped
    assert moving


def test_a_stopped_row_without_stop_seq_keeps_the_omission() -> None:
    rows = [
        e.record
        for obj in _objects("archive/trace")
        for e in trace.adapt(obj)
        if isinstance(e.record, VehicleObservation) and e.record.current_status == 1
    ]
    assert any(r.current_stop_sequence is None for r in rows)
    assert any(isinstance(r.current_stop_sequence, int) for r in rows)


@pytest.mark.parametrize("archive", ["archive/trip_updates", "archive/vehicles"])
def test_fresh_feed_archives_keep_per_feed_success_only(archive: str) -> None:
    for obj in _objects(archive):
        records = [e.record for e in fresh_feeds.adapt(obj)]
        fresh = set(json.loads(obj.body)["fresh_feeds"])
        assert {
            r.source
            for r in records
            if isinstance(r, LegacyFetchOutcome) and r.succeeded
        } == fresh
        for r in records:
            _assert_kept(r, _FETCH_KEPT)


def test_a_feed_missing_from_fresh_feeds_is_a_failure() -> None:
    (obj,) = [o for o in _objects("archive/trip_updates") if "2026-09-22" in o.key]
    failed = {
        r.source
        for e in fresh_feeds.adapt(obj)
        if isinstance(r := e.record, LegacyFetchOutcome) and not r.succeeded
    }
    assert len(failed) == 4


def test_alerts_liveness_keeps_outcome_and_fetched_at() -> None:
    for obj in _objects("archive/alerts_liveness"):
        (e,) = alerts_liveness.adapt(obj)
        _assert_kept(e.record, _FETCH_KEPT | {"last_success_at"})


def test_alert_versions_keep_the_entity_and_first_sighting_only() -> None:
    emitted = alerts.versions(alerts.parse(o) for o in _objects("archive/alerts"))
    superseded = 0
    for e in emitted:
        r = e.record
        assert isinstance(r, AlertVersion)
        kept = {"source", "alert_id", "content_digest", "entity", "alert_type"}
        kept |= {"created_at", "updated_at", "first_seen"}
        if r.gone_kind == "superseded":
            superseded += 1
            kept.add("gone_kind")
        _assert_kept(r, kept)
    assert superseded == 1


def test_only_a_later_updated_at_of_the_same_id_proves_superseded() -> None:
    emitted = alerts.versions(alerts.parse(o) for o in _objects("archive/alerts"))
    by_id: dict[str, list[AlertVersion]] = {}
    for e in emitted:
        assert isinstance(e.record, AlertVersion)
        by_id.setdefault(e.record.alert_id, []).append(e.record)
    (pair,) = [v for v in by_id.values() if len(v) == 2]
    early, late = sorted(pair, key=lambda v: v.updated_at or 0)
    assert early.gone_kind == "superseded"
    assert isinstance(late.gone_kind, Unknown)


def test_ene_keeps_records_verbatim_and_lacks_only_the_fetch() -> None:
    sources: set[str] = set()
    for obj in _objects("archive/ene"):
        for e in ene.adapt(obj):
            r = e.record
            if isinstance(r, LegacyFetchOutcome):
                assert r.succeeded
                sources.add(r.source)
                _assert_kept(r, _FETCH_KEPT)
            else:
                assert isinstance(r, EquipmentObservation)
                _assert_kept(r, set(_paths(r)) - {"fetch_id"})
    assert sources == {"ene_current", "ene_upcoming", "ene_equipments"}


@pytest.mark.parametrize(
    ("archive", "adapt"),
    [("archive/health", health.adapt), ("v1/predictions", predictions.adapt)],
)
def test_legacy_only_records_lose_nothing_they_have_a_field_for(
    archive: str, adapt: Any
) -> None:
    for obj in _objects(archive):
        for e in adapt(obj):
            _assert_kept(e.record, set(_paths(e.record)))


@pytest.mark.parametrize(
    ("text", "utc"),
    [
        ("10/30/2026 10:00:00 PM", datetime(2026, 10, 31, 2, tzinfo=UTC)),  # EDT
        ("12/01/2026 06:00:00 AM", datetime(2026, 12, 1, 11, tzinfo=UTC)),  # EST
        ("01/02/2027 12:00:00 AM", datetime(2027, 1, 2, 5, tzinfo=UTC)),  # midnight
    ],
)
def test_ene_wall_times_parse_in_new_york(text: str, utc: datetime) -> None:
    assert ene.eastern_instant(text) == int(utc.timestamp())


def test_ene_wall_times_that_name_no_single_instant_are_unknown() -> None:
    repeated = ene.eastern_instant("11/01/2026 01:30:00 AM")
    skipped = ene.eastern_instant("03/08/2026 02:30:00 AM")
    garbled = ene.eastern_instant("2026-10-30 22:00")
    assert isinstance(repeated, Unknown)
    assert isinstance(skipped, Unknown)
    assert isinstance(garbled, Unknown)
    assert len({repeated.reason, skipped.reason, garbled.reason}) == 3
    assert ene.eastern_instant(None) is None


@pytest.mark.parametrize("blank", ["", " ", "\t "])
def test_a_blank_ene_wall_time_is_absent(blank: str) -> None:
    assert ene.eastern_instant(blank) is None


# --- lineage -----------------------------------------------------------------


def test_trace_rows_name_their_object_and_row() -> None:
    for obj in _objects("archive/trace"):
        rows = json.loads(obj.body)["rows"]
        vehicles = [
            e.record
            for e in trace.adapt(obj)
            if isinstance(e.record, VehicleObservation)
        ]
        assert len(vehicles) == len(rows)
        for v in vehicles:
            assert isinstance(v.lineage, LegacyLineage)
            assert v.lineage.source_key == obj.key
            assert v.lineage.row_index is not None
            row = rows[v.lineage.row_index]
            assert v.trip is not None
            assert (v.trip.trip_id, v.stop_id) == (row["trip_id"], row["stop_id"])


def test_ene_records_name_their_object_and_record() -> None:
    for obj in _objects("archive/ene"):
        payload = json.loads(obj.body)["payload"]
        for e in ene.adapt(obj):
            r = e.record
            if isinstance(r, EquipmentObservation):
                assert isinstance(r.lineage, LegacyLineage)
                assert r.lineage.source_key == obj.key
                assert r.lineage.row_index == r.record_index
                assert r.record == payload[r.record_index]
            else:
                assert isinstance(r, LegacyFetchOutcome)
                assert (r.lineage.source_key, r.lineage.row_index) == (obj.key, None)


def test_prediction_rows_name_their_object_and_line() -> None:
    for obj in _objects("v1/predictions"):
        lines = [json.loads(line) for line in obj.body.decode().splitlines()]
        emitted = predictions.adapt(obj)
        assert len(emitted) == len(lines)
        for e in emitted:
            r = e.record
            assert isinstance(r, LegacyRouteAlertState)
            assert r.lineage.source_key == obj.key
            assert r.lineage.row_index is not None
            line = lines[r.lineage.row_index]
            assert (r.route_id, r.primary_alert_type) == (
                line["route"],
                line["primary_alert_type"],
            )


# --- counts --------------------------------------------------------------------


def _alert_bodies() -> list[dict[str, Any]]:
    return [json.loads(o.body) for o in _objects("archive/alerts")]


def test_alert_versions_are_objects_minus_identical_resightings() -> None:
    bodies = _alert_bodies()
    distinct = {(b["alert"]["id"], alert_content_digest(b["alert"])) for b in bodies}
    assert len(distinct) < len(bodies)  # the fixture holds a re-sighting
    emitted = alerts.versions(alerts.parse(o) for o in _objects("archive/alerts"))
    assert len(emitted) == len(distinct)


def test_a_resighted_version_keeps_its_first_sighting() -> None:
    bodies = _alert_bodies()
    first: dict[tuple[str, str], int] = {}
    for b in bodies:
        key = (b["alert"]["id"], alert_content_digest(b["alert"]))
        first[key] = min(first.get(key, b["observed_at"]), b["observed_at"])
    for e in alerts.versions(alerts.parse(o) for o in _objects("archive/alerts")):
        r = e.record
        assert isinstance(r, AlertVersion)
        assert r.first_seen == first[(r.alert_id, r.content_digest)]
        assert isinstance(r.lineage, LegacyLineage)
        assert (
            json.loads((MIRROR / r.lineage.source_key).read_bytes())["observed_at"]
            == r.first_seen
        )


def _run(out: Path, mirror: Path = MIRROR) -> dict[str, bytes]:
    run(mirror, out, "f" * 40, False)
    return {
        p.relative_to(out).as_posix(): p.read_bytes()
        for p in sorted(out.rglob("*"))
        if p.is_file()
    }


def _read(out: Path, record: str) -> list[BaseModel]:
    rows: list[BaseModel] = []
    for p in sorted((out / PREFIX / "derived" / record).rglob("*.parquet")):
        rows += read_records(p, MODELS[record])[0]
    return rows


def test_trace_observations_count_trace_rows(tmp_path: Path) -> None:
    objects = _objects("archive/trace")
    _run(tmp_path)
    vehicles = _read(tmp_path, "vehicle")
    assert len(vehicles) == sum(len(json.loads(o.body)["rows"]) for o in objects)
    outcomes = [
        r
        for r in _read(tmp_path, "legacy_fetch_outcome")
        if isinstance(r, LegacyFetchOutcome) and r.lineage.archive == "archive/trace"
    ]
    assert len(outcomes) == 8 * len(objects)


def test_vehicles_are_filed_under_the_hour_they_were_observed(tmp_path: Path) -> None:
    _run(tmp_path)
    for p in (tmp_path / PREFIX / "derived" / "vehicle").rglob("*.parquet"):
        day, hour = p.parent.name, int(p.stem)
        for r in read_records(p, VehicleObservation)[0]:
            assert (utc_day(r.observed_at), r.observed_at // 3600 % 24) == (day, hour)


# --- the files -------------------------------------------------------------------


def test_the_cli_twice_gives_identical_files_in_the_committed_layout(
    tmp_path: Path,
) -> None:
    outs = [tmp_path / "a", tmp_path / "b"]
    for out in outs:
        assert main(["--mirror", str(MIRROR), "--out", str(out)]) == 0
    files = [
        {p.relative_to(o).as_posix(): p.read_bytes() for p in o.rglob("*.parquet")}
        for o in outs
    ]
    assert files[0] == files[1]
    records = {Path(k).relative_to(f"{PREFIX}derived").parts[0] for k in files[0]}
    assert records == set(NORMALIZED) - {"trip_update", "feed_header", "alert_presence"}
    for key in files[0]:
        record = Path(key).relative_to(f"{PREFIX}derived").parts[0]
        assert file_layout(outs[0] / key) == (LAYOUTS / f"{record}.txt").read_text()


def test_file_metadata_names_every_input_object(tmp_path: Path) -> None:
    _run(tmp_path)
    named: set[str] = set()
    for p in (tmp_path / PREFIX / "derived").rglob("*.parquet"):
        record = p.relative_to(tmp_path / PREFIX / "derived").parts[0]
        rows, meta = read_records(p, MODELS[record])
        assert meta.row_count == len(rows)
        assert meta.file_key == p.relative_to(tmp_path).as_posix()
        for key, digest in meta.provenance.inputs.items():
            assert hashlib.sha256((MIRROR / key).read_bytes()).hexdigest() == digest
        named |= set(meta.provenance.inputs)
    every = {p.relative_to(MIRROR).as_posix() for p in MIRROR.rglob("*") if p.is_file()}
    assert named == every


def test_a_malformed_object_is_skipped_and_reported(tmp_path: Path) -> None:
    mirror = tmp_path / "mirror"
    shutil.copytree(MIRROR, mirror)
    bad = mirror / "archive" / "health" / "2026-09-11" / "1789094708.json"
    bad.write_bytes(b'{"observed_at": 1789094708, "write_fail')
    report = run(mirror, tmp_path / "out", "f" * 40, False)
    failures = report.archives["archive/health"].failures
    assert [k for k, _ in failures] == ["archive/health/2026-09-11/1789094708.json"]
    assert report.archives["archive/health"].rows == {"legacy_collector_health": 2}
    assert main(["--mirror", str(mirror), "--out", str(tmp_path / "cli")]) == 1


def test_the_cli_fails_without_a_mirror(tmp_path: Path) -> None:
    out = tmp_path / "out"
    assert main(["--mirror", str(tmp_path / "absent"), "--out", str(out)]) == 2
    assert not out.exists()


def _trace_vehicles() -> list[VehicleObservation]:
    return [
        e.record
        for obj in _objects("archive/trace")
        for e in trace.adapt(obj)
        if isinstance(e.record, VehicleObservation)
    ]


def _hour(v: VehicleObservation) -> int:
    return v.observed_at // 3600 * 3600


def _hour_meta(records: list[VehicleObservation]) -> tuple[str, DerivedFileMetadata]:
    key = DerivedHourKey("vehicle", _hour(records[0])).key
    inputs: dict[str, tuple[str, int]] = {}
    for r in records:
        assert isinstance(r.lineage, LegacyLineage)
        body = (MIRROR / r.lineage.source_key).read_bytes()
        inputs[r.lineage.source_key] = (hashlib.sha256(body).hexdigest(), r.observed_at)
    return key, file_metadata(key, "legacy-1", len(records), inputs, "f", False)


def _one_hour() -> list[VehicleObservation]:
    by_hour: dict[int, list[VehicleObservation]] = {}
    for v in _trace_vehicles():
        by_hour.setdefault(_hour(v), []).append(v)
    return max(by_hour.values(), key=len)


def test_the_writer_sorts_so_input_order_does_not_change_the_bytes(
    tmp_path: Path,
) -> None:
    records = _one_hour()
    key, meta = _hour_meta(records)
    write_records(tmp_path / "a" / key, records, meta)
    write_records(tmp_path / "b" / key, list(reversed(records)), meta)
    assert (tmp_path / "a" / key).read_bytes() == (tmp_path / "b" / key).read_bytes()


def test_null_unknown_and_a_value_survive_parquet_distinct(tmp_path: Path) -> None:
    vehicles = _trace_vehicles()
    seqs = [v.current_stop_sequence for v in vehicles]
    assert None in seqs
    assert any(isinstance(s, Unknown) for s in seqs)
    assert any(isinstance(s, int) for s in seqs)
    for i, v in enumerate(vehicles):
        key, meta = _hour_meta([v])
        path = tmp_path / str(i) / key
        write_records(path, [v], meta)
        assert read_records(path, VehicleObservation)[0] == [v]


def test_the_writer_rejects_records_outside_the_file_or_a_wrong_path(
    tmp_path: Path,
) -> None:
    records = _one_hour()
    key, meta = _hour_meta(records)
    other = next(v for v in _trace_vehicles() if _hour(v) != _hour(records[0]))
    with pytest.raises(DerivedFileError):
        write_records(tmp_path / key, [*records[:-1], other], meta)
    with pytest.raises(DerivedFileError):
        write_records(tmp_path / "x.parquet", records, meta)


def test_the_writer_rejects_two_records_with_one_natural_key(tmp_path: Path) -> None:
    (obj,) = [o for o in _objects("archive/health") if "09-11" in o.key]
    (e,) = health.adapt(obj)
    key = f"{PREFIX}derived/legacy_collector_health/2026-09-11.parquet"
    source = e.inputs[0]
    meta = file_metadata(
        key,
        "legacy-1",
        2,
        {source.key: (source.sha256, source.observed_at)},
        "f",
        False,
    )
    with pytest.raises(DerivedFileError):
        write_records(tmp_path / key, [e.record, e.record], meta)


# --- legacy periods ------------------------------------------------------------


def test_every_archives_first_fixture_day_is_its_period_start() -> None:
    for archive, period in PERIODS.items():
        days = sorted(p.name for p in (MIRROR / archive).iterdir())
        assert days[0] == period.start, archive


@pytest.mark.parametrize(
    ("adapt", "key", "body"),
    [
        (
            health.adapt,
            "archive/health/2026-09-10/1789041600.json",
            {"observed_at": 1789041600, "write_failures": {}},
        ),
        (
            alerts_liveness.adapt,
            "archive/alerts_liveness/2026-09-01/1788264000.json",
            {"observed_at": 1788264000, "outcome": "success", "fetched_at": 1788264000},
        ),
        (
            fresh_feeds.adapt,
            "archive/vehicles/2026-06-20/1781956800.json",
            {"observed_at": 1781956800, "fresh_feeds": ["g"], "rows": {}},
        ),
    ],
)
def test_an_object_dated_before_its_legacy_period_is_rejected(
    adapt: Any, key: str, body: dict[str, Any]
) -> None:
    obj = LegacyObject(key, json.dumps(body).encode())
    with pytest.raises(ValidationError) as err:
        adapt(obj)
    assert "legacy_period" in {e["type"] for e in err.value.errors()}
