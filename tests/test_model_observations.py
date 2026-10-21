"""Observations layer (schema observations/s1): contracts, lineage, the R2 key
layout, and the pinned parquet column layout.

The fixtures under tests/fixtures/model/observations/ are hand-written from
the live feeds of 2026-10-20 (and, for legacy rows, the archive formats the
Worker writes): one file per exported document, each a list of records.
worker/test/model_observations.test.ts validates the same files with ajv.

Validators raise typed errors; tests match the error type, never wording.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from momentarily.model.conventions import Unknown
from momentarily.model.observations.columns import (
    LAYOUT_VERSION,
    NORMALIZED,
    UNKNOWN_SUFFIX,
    from_row,
    layout,
    parquet_metadata,
    read_parquet_metadata,
    to_row,
)
from momentarily.model.observations.contracts import (
    AlertVersion,
    CollectorTick,
    DerivedFileMetadata,
    EquipmentObservation,
    FeedFetch,
    FeedHeaderObservation,
    LegacyCollectorHealth,
    LegacyFetchOutcome,
    LegacyRouteAlertState,
    TickBundleIndex,
    TripUpdateObservation,
    VehicleObservation,
    alert_content_digest,
    canonical_entity_json,
)
from momentarily.model.observations.keys import (
    DAILY_RECORDS,
    HOURLY_RECORDS,
    BlobKey,
    BundleKey,
    DerivedDayKey,
    DerivedHourKey,
    GtfsArchiveKey,
    KeyLayoutError,
    LegacyPredictionsKey,
    ObservationKey,
    TickKey,
    parse_key,
)
from scripts.export_model_schema import (
    DOCUMENTS,
    SCHEMA_DIR,
    committed_layouts,
    render_layouts,
)

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "model" / "observations"
OBSERVATION_DOCS = {
    name.removeprefix("observations/"): model
    for name, model in DOCUMENTS.items()
    if name.startswith("observations/")
}
SHA = "0f" * 32
# sha256 over the committed parquet layouts, per LAYOUT_VERSION. Parquet files
# are kept indefinitely: a layout change must bump LAYOUT_VERSION and add a
# new pin here, never edit an old one.
PINNED_LAYOUTS = {
    1: "05664fcfdec2e713911ce69ef4deb02df3e13bd7b60646526dd6ab8971d1dd65",
    2: "9ad00e5cb9c23c34e336084c6a4a13022737cc181e08aa4a0a739f4b183d5fe1",
}


def _records(name: str) -> list[dict[str, Any]]:
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text())


def _record(name: str, index: int = 0, **changes: Any) -> dict[str, Any]:
    return {**_records(name)[index], **changes}


def _error_types(model: type[BaseModel], doc: dict[str, Any]) -> set[str]:
    with pytest.raises(ValidationError) as err:
        model.model_validate(doc)
    return {e["type"] for e in err.value.errors()}


def _unknown(reason: str = "not kept") -> dict[str, Any]:
    return {"unknown": True, "reason": reason}


# --- fixtures and generated schema -----------------------------------------


def test_every_document_has_a_fixture_and_nothing_else() -> None:
    assert {p.stem for p in FIXTURE_DIR.glob("*.json")} == set(OBSERVATION_DOCS)


@pytest.mark.parametrize("name", sorted(OBSERVATION_DOCS))
def test_fixtures_round_trip_through_pydantic(name: str) -> None:
    model = OBSERVATION_DOCS[name]
    for record in _records(name):
        parsed = model.model_validate_json(json.dumps(record))
        assert parsed.model_dump(mode="json", by_alias=True) == record


def _objects(node: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if "properties" in node:
            found.append(node)  # pyright: ignore[reportUnknownArgumentType]
        for value in node.values():  # pyright: ignore[reportUnknownVariableType]
            found += _objects(value)
    elif isinstance(node, list):
        for value in node:  # pyright: ignore[reportUnknownVariableType]
            found += _objects(value)
    return found


@pytest.mark.parametrize("name", sorted(OBSERVATION_DOCS))
def test_exported_schemas_make_every_field_required(name: str) -> None:
    """A default would surface as an optional property: a consumer reading a
    record without it would see 'absent' where the contract promises an
    explicit null or Unknown."""
    schema = json.loads(
        (SCHEMA_DIR / "observations" / f"{name}.schema.json").read_text()
    )
    for obj in _objects(schema):
        assert sorted(obj.get("required", [])) == sorted(obj["properties"])
        assert obj.get("additionalProperties") is False


# --- three states: value, None, Unknown -------------------------------------


def test_an_omitted_field_is_rejected_not_defaulted() -> None:
    vehicle = _record("vehicle_observation")
    trip = {k: v for k, v in vehicle["trip"].items() if k != "nyct_is_assigned"}
    with pytest.raises(ValidationError) as err:
        VehicleObservation.model_validate({**vehicle, "trip": trip})
    assert [e["type"] for e in err.value.errors()] == ["missing"]


def test_a_missing_assigned_flag_is_not_false() -> None:
    placeholder = VehicleObservation.model_validate(_record("vehicle_observation", 1))
    assert placeholder.trip is not None
    assert placeholder.trip.nyct_is_assigned is None
    row = to_row(placeholder)
    assert row["trip"]["nyct_is_assigned"] is None
    assert row["trip"]["nyct_is_assigned" + UNKNOWN_SUFFIX] is None
    assert from_row(VehicleObservation, row) == placeholder
    assigned = VehicleObservation.model_validate(_record("vehicle_observation"))
    assert assigned.trip is not None
    unassigned = assigned.model_copy(
        update={"trip": assigned.trip.model_copy(update={"nyct_is_assigned": False})}
    )
    assert to_row(unassigned)["trip"]["nyct_is_assigned"] is False


def test_a_placeholder_vehicle_keeps_its_future_timestamp_and_no_status() -> None:
    placeholder = VehicleObservation.model_validate(_record("vehicle_observation", 1))
    assert placeholder.vehicle_timestamp is not None
    assert placeholder.vehicle_timestamp > placeholder.observed_at
    assert placeholder.current_status is None
    assert from_row(VehicleObservation, to_row(placeholder)) == placeholder


@pytest.mark.parametrize("delay", [-1, -49, -3600])
def test_a_negative_delay_stays_negative(delay: int) -> None:
    doc = _record("trip_update_observation")
    stops = [
        {**s, "arrival_delay": delay, "departure_delay": delay}
        for s in doc["stop_times"]
    ]
    update = TripUpdateObservation.model_validate_json(
        json.dumps({**doc, "delay": delay, "stop_times": stops})
    )
    back = from_row(TripUpdateObservation, to_row(update))
    assert back.delay == delay
    assert {s.arrival_delay for s in back.stop_times} == {delay}
    assert {s.departure_delay for s in back.stop_times} == {delay}


def test_none_unknown_and_value_stay_distinct_in_columns() -> None:
    live, legacy = (AlertVersion.model_validate(r) for r in _records("alert_version"))
    live_row, legacy_row = to_row(live), to_row(legacy)
    assert (live_row["gone_by"], live_row["gone_by" + UNKNOWN_SUFFIX]) == (
        1792514580,
        None,
    )
    assert legacy_row["gone_by"] is None
    assert legacy_row["gone_by" + UNKNOWN_SUFFIX] is not None
    still_present = live.model_copy(update={"gone_by": None, "gone_kind": None})
    row = to_row(still_present)
    assert (row["gone_by"], row["gone_by" + UNKNOWN_SUFFIX]) == (None, None)
    assert from_row(AlertVersion, row).gone_by is None
    assert isinstance(from_row(AlertVersion, legacy_row).gone_by, Unknown)


def test_an_opaque_entity_holding_unknown_true_stays_data() -> None:
    doc = _record("alert_version")
    entity = {**doc["entity"], "unknown": True}
    version = AlertVersion.model_validate(
        {**doc, "entity": entity, "content_digest": alert_content_digest(entity)}
    )
    assert from_row(AlertVersion, to_row(version)) == version


def test_a_row_with_both_a_value_and_an_unknown_reason_is_rejected() -> None:
    row = to_row(AlertVersion.model_validate(_record("alert_version")))
    row["gone_by" + UNKNOWN_SUFFIX] = "not captured"
    with pytest.raises(ValueError, match="both a value and an unknown reason"):
        from_row(AlertVersion, row)


# --- lineage ---------------------------------------------------------------------


def test_legacy_rows_are_filterable_by_lineage_kind() -> None:
    rows = [
        to_row(VehicleObservation.model_validate(r))
        for r in _records("vehicle_observation")
    ]
    kinds = [r["lineage"]["kind"] for r in rows]
    assert kinds == ["raw", "raw", "raw", "legacy"]
    legacy = rows[-1]["lineage"]
    assert legacy["raw"] is None
    assert legacy["legacy"]["archive"] == "archive/trace"


def test_a_raw_record_cannot_hold_a_legacy_unknown() -> None:
    doc = _record("vehicle_observation")
    assert "legacy_only_unknown" in _error_types(
        VehicleObservation, {**doc, "source": _unknown()}
    )
    trip = {**doc["trip"], "route_id": _unknown()}
    assert "legacy_only_unknown" in _error_types(
        VehicleObservation, {**doc, "trip": trip}
    )


def test_a_raw_equipment_record_may_hold_a_decode_unknown() -> None:
    doc = _record("equipment_observation")
    parsed = EquipmentObservation.model_validate(
        {**doc, "outage_start": _unknown("ambiguous")}
    )
    assert isinstance(parsed.outage_start, Unknown)
    assert "legacy_only_unknown" in _error_types(
        EquipmentObservation, {**doc, "fetch_id": _unknown()}
    )


def test_a_legacy_row_must_come_from_an_archive_that_holds_it() -> None:
    doc = _record("vehicle_observation", 3)
    lineage = {
        **doc["lineage"],
        "archive": "archive/ene",
        "source_key": "archive/ene/x.json",
    }
    assert "lineage" in _error_types(VehicleObservation, {**doc, "lineage": lineage})


def test_a_legacy_row_outside_its_period_is_rejected() -> None:
    doc = _record("vehicle_observation", 3)
    lineage = {**doc["lineage"], "period": {"start": "2026-08-12", "end": "2026-10-01"}}
    assert "legacy_period" in _error_types(
        VehicleObservation, {**doc, "lineage": lineage}
    )


def test_a_legacy_period_needs_real_dates() -> None:
    doc = _record("vehicle_observation", 3)
    lineage = {**doc["lineage"], "period": {"start": "2026-08-12", "end": "2026-99-99"}}
    assert "legacy_period" in _error_types(
        VehicleObservation, {**doc, "lineage": lineage}
    )


def test_trip_updates_have_no_legacy_lineage() -> None:
    doc = _record("trip_update_observation")
    legacy = _record("vehicle_observation", 3)["lineage"]
    assert "literal_error" in _error_types(
        TripUpdateObservation, {**doc, "lineage": legacy}
    )


def test_a_gtfsrt_record_must_come_from_a_bundle() -> None:
    doc = _record("vehicle_observation")
    blob = {
        "kind": "raw",
        "payload_key": BlobKey("2026-10-20", SHA, "pb").key,
        "payload_member": None,
        "payload_sha256": SHA,
    }
    assert "lineage" in _error_types(VehicleObservation, {**doc, "lineage": blob})


# --- legacy-only records ---------------------------------------------------------


def test_legacy_route_alert_state_requires_lineage_and_granularity() -> None:
    record = _record("legacy_route_alert_state")
    for field in ("lineage", "granularity"):
        without = {k: v for k, v in record.items() if k != field}
        assert "missing" in _error_types(LegacyRouteAlertState, without)
    undated = {k: v for k, v in record["lineage"].items() if k != "period"}
    assert "missing" in _error_types(
        LegacyRouteAlertState, {**record, "lineage": undated}
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("scope", "alert"), ("tick_seconds", 60), ("content", "alert_versions")],
)
def test_legacy_granularity_cannot_claim_more_than_the_source_holds(
    field: str, value: object
) -> None:
    record = _record("legacy_route_alert_state")
    granularity = {**record["granularity"], field: value}
    assert "literal_error" in _error_types(
        LegacyRouteAlertState, {**record, "granularity": granularity}
    )


def test_legacy_route_alert_state_must_name_the_object_holding_its_tick() -> None:
    record = _record("legacy_route_alert_state")
    moved = {**record, "observed_at": record["observed_at"] + 300}
    assert "legacy_key" in _error_types(LegacyRouteAlertState, moved)
    before = 1784505600  # 2026-07-20T00:00Z
    early_lineage = {
        **record["lineage"],
        "source_key": LegacyPredictionsKey(before).key,
    }
    early = {**record, "observed_at": before, "lineage": early_lineage}
    assert "legacy_period" in _error_types(LegacyRouteAlertState, early)


def test_legacy_fetch_outcome_ids_and_archives() -> None:
    record = _record("legacy_fetch_outcome")
    assert "legacy_key" in _error_types(
        LegacyFetchOutcome, {**record, "legacy_fetch_id": "ace@1790856000000"}
    )
    health = {
        **record["lineage"],
        "archive": "archive/health",
        "source_key": "archive/health/x",
    }
    assert "lineage" in _error_types(LegacyFetchOutcome, {**record, "lineage": health})


def test_legacy_collector_health_keys_on_observed_at() -> None:
    record = _record("legacy_collector_health")
    assert "legacy_key" in _error_types(
        LegacyCollectorHealth, {**record, "observed_at": record["observed_at"] + 1}
    )


# --- alert end bounds ------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "error_type"),
    [
        ({"gone_by": 1792514520}, "alert_bounds"),  # == last_seen
        ({"gone_by": 1792514460}, "alert_bounds"),  # < last_seen
        ({"absent_before": 1792511880}, "alert_bounds"),  # == first_seen
        ({"last_seen": 1792511820}, "alert_bounds"),  # < first_seen
        ({"gone_by": None}, "alert_gone_kind"),  # still present, kind set
        ({"gone_kind": None}, "alert_gone_kind"),  # observed end, no kind
    ],
)
def test_alert_version_bounds(changes: dict[str, Any], error_type: str) -> None:
    assert error_type in _error_types(AlertVersion, _record("alert_version", **changes))


def test_a_legacy_alert_version_with_unknown_end_cannot_claim_removed() -> None:
    record = _record("alert_version", 1, gone_kind="removed")
    assert "alert_gone_kind" in _error_types(AlertVersion, record)
    superseded = AlertVersion.model_validate(
        _record("alert_version", 1, gone_kind="superseded")
    )
    assert superseded.gone_kind == "superseded"
    observed_end = _record("alert_version", 1, gone_by=1792514580)
    assert "alert_gone_kind" in _error_types(AlertVersion, observed_end)


@pytest.mark.parametrize(
    "gaps",
    [
        [{"start": 1792512600, "end": 1792512600}],  # empty
        [
            {"start": 1792512800, "end": 1792512900},
            {"start": 1792512600, "end": 1792512700},
        ],
        [
            {"start": 1792512600, "end": 1792512800},
            {"start": 1792512700, "end": 1792512900},
        ],
        [{"start": 1792514580, "end": 1792514700}],  # after gone_by
    ],
)
def test_unobserved_gaps_are_sorted_disjoint_and_inside(
    gaps: list[dict[str, int]],
) -> None:
    assert "alert_gaps" in _error_types(
        AlertVersion, _record("alert_version", unobserved=gaps)
    )


def test_alert_fields_must_match_the_entity() -> None:
    record = _record("alert_version")
    assert "alert_entity" in _error_types(
        AlertVersion, {**record, "alert_type": "Delays!"}
    )
    assert "alert_entity" in _error_types(
        AlertVersion, {**record, "updated_at": record["updated_at"] + 1}
    )
    entity = {**record["entity"], "extra": 1}
    assert "alert_digest" in _error_types(AlertVersion, {**record, "entity": entity})


def test_content_digest_writes_numbers_as_javascript_does() -> None:
    assert canonical_entity_json({"a": 1.0, "b": [2.0, 0.5]}) == b'{"a":1,"b":[2,0.5]}'
    with pytest.raises(ValueError, match="JSON compliant"):
        canonical_entity_json({"a": float("nan")})


# --- fetch records ----------------------------------------------------------------


def _fetch(source: str) -> dict[str, Any]:
    return next(
        f for f in _record("collector_tick")["fetches"] if f["source"] == source
    )


@pytest.mark.parametrize(
    ("source", "changes", "error_type"),
    [
        ("si", {"finished_at_ms": 1792514530000}, "fetch_timing"),
        ("g", {"http_status": 200}, "fetch_status"),
        ("si", {"http_status": 503}, "fetch_status"),
        ("si", {"content_type": "text/html"}, "fetch_headers"),
        ("jz", {"entity_count": 3}, "fetch_parse"),
        ("l", {"payload_sha256": None}, "fetch_body"),
        ("si", {"body_bytes": 10}, "fetch_body"),
        ("g", {"store_error": "R2 put failed"}, "fetch_body"),
        ("l", {"payload_key": None, "payload_member": None}, "fetch_storage"),
        (
            "nqrw",
            {"payload_key": BlobKey("2026-10-20", SHA, "pb").key},
            "fetch_storage",
        ),
        ("alerts", {"payload_sha256": "1" * 64}, "fetch_storage"),
        (
            "l",
            {"payload_key": BundleKey(1792514580, 1792514580412).key},
            "fetch_storage",
        ),
        ("l", {"fetch_id": "ace@1792514520416"}, "fetch_identity"),
    ],
)
def test_fetch_records_reject_contradictions(
    source: str, changes: dict[str, Any], error_type: str
) -> None:
    assert error_type in _error_types(FeedFetch, {**_fetch(source), **changes})


def test_a_blob_is_filed_under_its_fetch_day() -> None:
    fetch = _fetch("alerts")
    next_day = BlobKey("2026-10-21", fetch["payload_sha256"], "json").key
    assert "fetch_storage" in _error_types(
        FeedFetch, {**fetch, "payload_key": next_day}
    )


@pytest.mark.parametrize(
    ("outcome", "finished_at_ms"),
    [("timeout", None), ("network_error", 1792514521300)],
)
def test_a_body_read_failure_after_a_200_keeps_the_status(
    outcome: str, finished_at_ms: int | None
) -> None:
    """Headers can arrive before the body times out or the connection resets;
    the status that did arrive is a fact the record must be able to hold."""
    failed = FeedFetch.model_validate(
        {
            **_fetch("si"),
            "outcome": outcome,
            "finished_at_ms": finished_at_ms,
            "http_status": 200,
            "content_type": "application/octet-stream",
        }
    )
    assert failed.http_status == 200
    assert failed.payload_key is None


def test_a_tick_rejects_an_unplanned_fetch() -> None:
    tick = _record("collector_tick")
    planned = [s for s in tick["planned_sources"] if s != "alerts"]
    assert "tick_fetches" in _error_types(
        CollectorTick, {**tick, "planned_sources": planned}
    )


def test_a_tick_rejects_a_fetch_outside_its_run() -> None:
    tick = _record("collector_tick")
    assert "tick_timing" in _error_types(
        CollectorTick, {**tick, "finished_at_ms": tick["started_at_ms"] + 500}
    )


def test_a_tick_rejects_a_fetch_in_another_invocations_bundle() -> None:
    """Overlapping invocations of one cron minute share scheduled_at; each
    writes its own bundle."""
    tick = _record("collector_tick")
    later = tick["started_at_ms"] + 2000
    shifted = {
        **tick,
        "tick_id": f"{tick['scheduled_at']}-{later}",
        "started_at_ms": later,
    }
    fetches = [{**f} for f in tick["fetches"]]
    for f in fetches:
        f["started_at_ms"] += 2000
        f["fetch_id"] = f"{f['source']}@{f['started_at_ms']}"
        if f["finished_at_ms"] is not None:
            f["finished_at_ms"] += 2000
    shifted["finished_at_ms"] = tick["finished_at_ms"] + 2000
    assert "tick_fetches" in _error_types(
        CollectorTick, {**shifted, "fetches": fetches}
    )


def test_a_bundle_index_holds_only_its_own_gtfsrt_members() -> None:
    index = _record("tick_bundle_index")
    assert "tick_fetches" in _error_types(
        TickBundleIndex, {**index, "fetches": [*index["fetches"], _fetch("alerts")]}
    )
    dup = {
        **index["fetches"][1],
        "payload_member": index["fetches"][0]["payload_member"],
    }
    assert "tick_fetches" in _error_types(
        TickBundleIndex, {**index, "fetches": [index["fetches"][0], dup]}
    )


# --- key layout -------------------------------------------------------------------

KEYS: list[tuple[ObservationKey, str]] = [
    (
        TickKey(1792514520, 1792514520412),
        "model/observations/s1/ticks/2026-10-20/16/1792514520-1792514520412.json",
    ),
    (
        BundleKey(1792514520, 1792514520412),
        "model/observations/s1/raw/gtfsrt/2026-10-20/16/1792514520-1792514520412.bin",
    ),
    (
        BlobKey("2026-10-20", SHA, "json"),
        f"model/observations/s1/raw/blob/2026-10-20/{SHA}.json.gz",
    ),
    (
        DerivedHourKey("trip_update", 1792512000),
        "model/observations/s1/derived/trip_update/2026-10-20/16.parquet",
    ),
    (
        DerivedDayKey("alert_version", "2026-10-20"),
        "model/observations/s1/derived/alert_version/2026-10-20.parquet",
    ),
    (GtfsArchiveKey(SHA), f"archive/gtfs/{SHA}.zip"),
    (LegacyPredictionsKey(1792514400), "v1/predictions/2026-10-20/1792514400.jsonl"),
]


@pytest.mark.parametrize(("typed", "text"), KEYS)
def test_keys_round_trip(typed: ObservationKey, text: str) -> None:
    assert typed.key == text
    assert parse_key(text) == typed


def test_a_tick_one_minute_before_midnight_files_under_its_own_day() -> None:
    late = TickKey(1792540799 - 59, 1792540800123)
    assert late.key.startswith("model/observations/s1/ticks/2026-10-20/23/")
    assert parse_key(late.key) == late


def test_blob_day_is_the_utc_day_of_the_fetch() -> None:
    assert BlobKey.for_fetch(1792540800, SHA, "pb").day == "2026-10-21"


@pytest.mark.parametrize(
    "text",
    [
        # directory disagrees with the scheduled_at in the file name
        "model/observations/s1/ticks/2026-10-20/15/1792514520-1792514520412.json",
        "model/observations/s1/ticks/2026-10-21/16/1792514520-1792514520412.json",
        # leading zero: parses to a different canonical key
        "model/observations/s1/raw/gtfsrt/2026-10-20/16/01792514520-1792514520412.bin",
        # beyond any calendar
        "model/observations/s1/ticks/2026-10-20/16/99999999999999999999-1.json",
        f"model/observations/s1/raw/blob/2026-10-20/{SHA.upper()}.json.gz",
        f"model/observations/s1/raw/blob/2026-10-20/{SHA}.csv.gz",
        f"model/observations/s1/raw/blob/2026-02-30/{SHA}.json.gz",
        "model/observations/s1/derived/vehicle/2026-10-20.parquet",
        "model/observations/s1/derived/alert_version/2026-10-20/16.parquet",
        "model/observations/s1/derived/trip_update/2026-10-20/24.parquet",
        "v1/predictions/2026-10-21/1792514400.jsonl",
        "archive/trace/2026-10-20/1792514520.json",
    ],
)
def test_keys_outside_the_layout_are_rejected(text: str) -> None:
    with pytest.raises(KeyLayoutError):
        parse_key(text)


def test_derived_hour_must_start_on_the_hour() -> None:
    with pytest.raises(KeyLayoutError):
        DerivedHourKey("vehicle", 1792512060)


# --- parquet layout and file metadata --------------------------------------------


def test_every_derived_record_kind_has_a_column_layout() -> None:
    assert set(NORMALIZED) == {*HOURLY_RECORDS, *DAILY_RECORDS}


def test_the_parquet_layout_is_pinned() -> None:
    """If this fails, the column layout changed. Stored parquet is kept
    indefinitely: bump LAYOUT_VERSION (and the decoder version if decoding
    changed), add its digest to PINNED_LAYOUTS, then regenerate with
    uv run python -m scripts.export_model_schema."""
    rendered = render_layouts()
    assert committed_layouts() == rendered, (
        "parquet layout drifted from the committed files"
    )
    digest = hashlib.sha256(
        "".join(f"{name}\n{text}" for name, text in sorted(rendered.items())).encode()
    ).hexdigest()
    assert PINNED_LAYOUTS.get(LAYOUT_VERSION) == digest, (
        "layout changed: bump LAYOUT_VERSION"
    )


FIXTURE_OF: dict[str, str] = {
    "vehicle": "vehicle_observation",
    "trip_update": "trip_update_observation",
    "feed_header": "feed_header_observation",
    "alert_presence": "alert_presence",
    "equipment": "equipment_observation",
    "alert_version": "alert_version",
    "legacy_route_alert_state": "legacy_route_alert_state",
    "legacy_fetch_outcome": "legacy_fetch_outcome",
    "legacy_collector_health": "legacy_collector_health",
}


@pytest.mark.parametrize("record", sorted(NORMALIZED))
def test_fixtures_round_trip_through_columns(record: str) -> None:
    model: type[BaseModel] = NORMALIZED[record]  # pyright: ignore[reportArgumentType]
    names = [c.name for c in layout(model)]
    for doc in _records(FIXTURE_OF[record]):
        parsed = model.model_validate(doc)
        row = to_row(parsed)
        assert list(row) == names
        assert from_row(model, row) == parsed


def test_feed_header_keeps_replacement_periods_as_published() -> None:
    """Start omitted, route omitted, range omitted, an empty range, and a
    repeated route all survive pydantic and the columns, in published order."""
    periods = [
        {"route_id": "6", "replacement_period": {"start": None, "end": 1792516308}},
        {"route_id": None, "replacement_period": {"start": 1792514508, "end": None}},
        {"route_id": "6", "replacement_period": {"start": 1792514508, "end": None}},
        {"route_id": "S", "replacement_period": None},
        {"route_id": "1", "replacement_period": {"start": None, "end": None}},
    ]
    doc = _record(
        "feed_header_observation",
        2,
        nyct_feed_header={
            "nyct_subway_version": "1.0",
            "trip_replacement_periods": periods,
        },
    )
    parsed = FeedHeaderObservation.model_validate_json(json.dumps(doc))
    assert (
        parsed.model_dump(mode="json")["nyct_feed_header"]["trip_replacement_periods"]
        == periods
    )
    row = to_row(parsed)
    assert row["nyct_feed_header"]["trip_replacement_periods"] == periods
    assert from_row(FeedHeaderObservation, row) == parsed


def test_feed_header_without_nyct_extension_differs_from_one_without_periods() -> None:
    absent = FeedHeaderObservation.model_validate(
        _record("feed_header_observation", nyct_feed_header=None)
    )
    empty = FeedHeaderObservation.model_validate(
        _record(
            "feed_header_observation",
            nyct_feed_header={
                "nyct_subway_version": "1.0",
                "trip_replacement_periods": [],
            },
        )
    )
    assert to_row(absent)["nyct_feed_header"] is None
    assert to_row(empty)["nyct_feed_header"] == {
        "nyct_subway_version": "1.0",
        "trip_replacement_periods": [],
    }
    assert from_row(FeedHeaderObservation, to_row(absent)) == absent
    assert from_row(FeedHeaderObservation, to_row(empty)) == empty


def test_file_metadata_round_trips_through_parquet_key_values() -> None:
    meta = DerivedFileMetadata.model_validate(_record("derived_file_metadata"))
    stored = {k.encode(): v.encode() for k, v in parquet_metadata(meta).items()}
    assert read_parquet_metadata(stored) == meta


@pytest.mark.parametrize(
    "changes",
    [
        {"file_key": DerivedDayKey("alert_version", "2026-10-21").key},
        {"file_key": TickKey(1792514520, 1792514520412).key},
    ],
)
def test_file_metadata_must_describe_its_own_file(changes: dict[str, Any]) -> None:
    assert "metadata" in _error_types(
        DerivedFileMetadata, _record("derived_file_metadata", **changes)
    )


def test_file_metadata_built_at_is_the_latest_input() -> None:
    record = _record("derived_file_metadata")
    provenance = {
        **record["provenance"],
        "built_at": record["provenance"]["built_at"] + 60,
    }
    assert "metadata" in _error_types(
        DerivedFileMetadata, {**record, "provenance": provenance}
    )
