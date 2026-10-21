"""Decoders: raw payload bytes of 2026-10-20 to observation records.

tests/fixtures/model/observations/raw/ holds real payloads (gzipped as
stored, manifest.json naming the fetch each came from) and one object of the
legacy archive/alerts store. Decoded output is pinned by digest per
DECODER_VERSION, so any change to what the decoders emit fails here.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from google.transit import gtfs_realtime_pb2 as rt
from pydantic import BaseModel

from momentarily.model.conventions import Unknown
from momentarily.model.observations.contracts import (
    AlertPresence,
    EquipmentObservation,
    RawLineage,
    SourceId,
    alert_content_digest,
)
from momentarily.model.observations.decode import (
    DECODER_VERSION,
    DecodeError,
    GtfsRtRecords,
    RawFetch,
    decode_alerts_json,
    decode_ene,
    decode_gtfsrt,
    new_york_instant,
)
from momentarily.model.observations.decode import nyct_subway_pb2 as nyct

RAW_DIR = (
    Path(__file__).resolve().parent / "fixtures" / "model" / "observations" / "raw"
)
MANIFEST: list[dict[str, Any]] = json.loads((RAW_DIR / "manifest.json").read_text())

# sha256 of the canonical JSON every fixture decodes to, per DECODER_VERSION.
# A decoder change that alters output must bump DECODER_VERSION and add new
# pins; never edit an old version's pins.
GOLDEN: dict[str, dict[str, str]] = {
    "1": {
        "numbered-1792507017657.pb.gz": "d1ff24ce2c6c4d0c3cd23037240ca62e53c7d38570df5101e620c3197b00844e",
        "g-1792507017429.pb.gz": "e10dc164e59a23f1ee5c1784e2a05e85dbb28f6153448d0c891a63d367d46616",
        "l-1792507017609.pb.gz": "a1fc3877bb5db92460c643b909350d84c3e9ccb11bc8c7a37345d27ddb8eeab9",
        "si-1792507017716.pb.gz": "5cd5a8d309e8a478655cbee3daa86a58f356d00d24a98eb3df1076a264b63e86",
        "alerts-1792507017768.json.gz": "4eb514817feef0252d3ec2ff33746f73bd0f2e3c28c8c346525a789669e2d6ea",
        "ene_current-1792507017910.json.gz": "eb967ee7dc5497ebf403134f18f7ff39999cae74a755672392d664cab63106ce",
        "ene_upcoming-1792507017956.json.gz": "d87076f2493a0fd2f1c7a4788743d784548909120fd54b61e9e2ea471b2287a1",
    },
}


def _entry(source: str) -> dict[str, Any]:
    (entry,) = [e for e in MANIFEST if e["source"] == source]
    return entry


def _fetch(source: SourceId, payload: bytes) -> RawFetch:
    """The manifest's fetch of ``source``, pointing at ``payload``: the same
    bundle member, or the blob its digest names."""
    entry = _entry(source)
    sha = hashlib.sha256(payload).hexdigest()
    key = entry["payload_key"]
    if entry["payload_member"] is None:
        key = f"model/observations/s1/raw/blob/2026-10-20/{sha}.json.gz"
    lineage = RawLineage(
        kind="raw",
        payload_key=key,
        payload_member=entry["payload_member"],
        payload_sha256=sha,
    )
    return RawFetch(entry["fetch_id"], source, entry["observed_at"], lineage)


def _fixture(source: SourceId) -> tuple[bytes, RawFetch]:
    payload = gzip.decompress((RAW_DIR / _entry(source)["file"]).read_bytes())
    return payload, _fetch(source, payload)


def _decode(payload: bytes, fetch: RawFetch) -> Any:
    if fetch.source == "alerts":
        return decode_alerts_json(payload, fetch)
    if fetch.source.startswith("ene_"):
        return decode_ene(payload, fetch)
    return decode_gtfsrt(payload, fetch)


def _gtfsrt(source: SourceId, payload: bytes | None = None) -> GtfsRtRecords:
    """``source``'s fixture decoded, or ``payload`` as if it were that fetch."""
    if payload is None:
        payload, _ = _fixture(source)
    return decode_gtfsrt(payload, _fetch(source, payload))


def _json(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, GtfsRtRecords):
        return {
            "header": _json(value.header),
            "vehicles": _json(value.vehicles),
            "trip_updates": _json(value.trip_updates),
            "alerts": _json(value.alerts),
        }
    if isinstance(value, list):
        return [_json(v) for v in value]  # pyright: ignore[reportUnknownVariableType]
    assert value is None
    return None


def _records(value: Any) -> list[BaseModel]:
    if isinstance(value, GtfsRtRecords):
        found: list[BaseModel] = [value.header, *value.vehicles, *value.trip_updates]
        return found + ([value.alerts] if value.alerts else [])
    return value if isinstance(value, list) else [value]  # pyright: ignore[reportUnknownVariableType]


# --- golden output -------------------------------------------------------------


@pytest.mark.parametrize("entry", MANIFEST, ids=[e["source"] for e in MANIFEST])
def test_fixtures_decode_to_the_pinned_records(entry: dict[str, Any]) -> None:
    """If this fails, decoded output changed: bump DECODER_VERSION and pin the
    new digests under it."""
    payload, fetch = _fixture(entry["source"])
    decoded = _decode(payload, fetch)
    for record in _records(decoded):
        dumped = record.model_dump_json(by_alias=True)
        assert type(record).model_validate_json(dumped) == record
    canonical = json.dumps(_json(decoded), sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    assert GOLDEN[DECODER_VERSION].get(entry["file"]) == digest, (
        "decoded output changed"
    )


def test_every_fixture_has_a_pin_and_nothing_else() -> None:
    assert set(GOLDEN[DECODER_VERSION]) == {e["file"] for e in MANIFEST}


# --- GTFS-RT rules ------------------------------------------------------------


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte, value = value & 0x7F, value >> 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _field(number: int, body: bytes) -> bytes:
    """A length-delimited field."""
    return _varint(number << 3 | 2) + _varint(len(body)) + body


def _unknown_varint(number: int) -> bytes:
    return _varint(number << 3) + _varint(1)


def _minimal_feed() -> rt.FeedMessage:
    """Only what GTFS-RT requires, plus one trip update and one vehicle that
    carry a trip id and a stop id and nothing else."""
    feed = rt.FeedMessage()
    feed.header.gtfs_realtime_version = "2.0"
    entity = feed.entity.add(id="e")
    entity.trip_update.trip.trip_id = "t"
    entity.trip_update.stop_time_update.add(stop_id="101N")
    entity.vehicle.trip.trip_id = "t"
    return feed


def test_absent_fields_decode_to_none_never_a_proto_default() -> None:
    decoded = _gtfsrt("si", _minimal_feed().SerializeToString())
    header = decoded.header.model_dump(exclude={"lineage", "fetch_id", "source"})
    assert header == {
        "observed_at": decoded.header.observed_at,
        "gtfs_realtime_version": "2.0",
        "incrementality": None,
        "header_timestamp": None,
        "feed_version": None,
        "nyct_feed_header": None,
    }
    (update,) = decoded.trip_updates
    (vehicle,) = decoded.vehicles
    for trip in (update.trip, vehicle.trip):
        assert trip is not None
        assert set(trip.model_dump(exclude={"trip_id"}).values()) == {None}
    (stop,) = update.stop_times
    assert set(stop.model_dump(exclude={"index", "stop_id"}).values()) == {None}
    assert (update.trip_update_timestamp, update.delay) == (None, None)
    moving = ("stop_id", "current_stop_sequence", "current_status", "vehicle_timestamp")
    assert set(vehicle.model_dump(include={*moving, "congestion_level"}).values()) == {
        None
    }
    assert decoded.alerts is None


def test_explicit_zero_and_false_stay_values() -> None:
    feed = _minimal_feed()
    feed.header.incrementality = rt.FeedHeader.FULL_DATASET
    vehicle = feed.entity[0].vehicle
    vehicle.current_status = rt.VehiclePosition.INCOMING_AT
    vehicle.trip.direction_id = 0
    decoded = _gtfsrt("si", feed.SerializeToString())
    assert decoded.header.incrementality == 0
    (observed,) = decoded.vehicles
    assert observed.current_status == 0
    assert observed.trip is not None
    assert observed.trip.direction_id == 0


def test_missing_assigned_flag_stays_none_on_real_b_division_trips() -> None:
    trips = [u.trip for u in _gtfsrt("g").trip_updates]
    flags = [t.nyct_is_assigned for t in trips]
    assert None in flags
    assert True in flags
    assert False not in flags


def test_negative_delays_come_out_signed() -> None:
    delays = [
        d
        for u in _gtfsrt("l").trip_updates
        for s in u.stop_times
        for d in (s.arrival_delay, s.departure_delay)
        if d is not None
    ]
    assert min(delays) < 0
    assert all(-(2**31) <= d < 2**31 for d in delays)


def test_placeholder_vehicles_keep_their_future_timestamp() -> None:
    decoded = _gtfsrt("g")
    published = decoded.header.header_timestamp
    assert published is not None
    placeholders = [
        v
        for v in decoded.vehicles
        if v.current_status is None
        and v.vehicle_timestamp is not None
        and v.vehicle_timestamp > published + 60
    ]
    assert placeholders


def test_route_ids_are_kept_verbatim() -> None:
    payload, _ = _fixture("numbered")
    feed = rt.FeedMessage.FromString(payload)
    updates = [e for e in feed.entity if e.HasField("trip_update")]
    updates[0].trip_update.trip.route_id = "6X"
    updates[1].trip_update.trip.route_id = "SS"
    decoded = _gtfsrt("numbered", feed.SerializeToString())
    routes = [u.trip.route_id for u in decoded.trip_updates]
    assert routes[:2] == ["6X", "SS"]
    assert "GS" in routes


def test_injected_unknown_fields_show_in_unmodeled_fields() -> None:
    """An entity appended to a real payload, carrying an unknown field on the
    entity, in the vehicle, and in its trip, plus a position the contract
    does not hold."""
    payload, _ = _fixture("l")
    trip = rt.TripDescriptor(trip_id="injected").SerializeToString()
    position = rt.Position(latitude=40.7, longitude=-73.9).SerializeToString()
    vehicle = (
        _field(rt.VehiclePosition.TRIP_FIELD_NUMBER, trip + _unknown_varint(50))
        + _field(rt.VehiclePosition.POSITION_FIELD_NUMBER, position)
        + _unknown_varint(99)
    )
    entity = (
        rt.FeedEntity(id="injected").SerializeToString()
        + _field(rt.FeedEntity.VEHICLE_FIELD_NUMBER, vehicle)
        + _unknown_varint(20)
    )
    decoded = _gtfsrt("l", payload + _field(rt.FeedMessage.ENTITY_FIELD_NUMBER, entity))
    injected = decoded.vehicles[-1]
    assert injected.entity_id == "injected"
    assert injected.entity_index == len(rt.FeedMessage.FromString(payload).entity)
    assert injected.unmodeled_fields == ["..20", "1.50", "2", "99"]
    assert {tuple(v.unmodeled_fields) for v in decoded.vehicles[:-1]} == {("..2",)}


def test_injected_unknown_fields_inside_a_trip_update_show_their_paths() -> None:
    """Unknown tags in the trip, its NYCT extension, a stop-time update, its
    arrival and departure events and its NYCT extension, plus two known
    fields the contract does not hold (TripUpdate.vehicle and
    stop_time_properties)."""
    payload, _ = _fixture("l")
    nyct_trip = nyct.NyctTripDescriptor(train_id="x").SerializeToString()
    trip = (
        rt.TripDescriptor(trip_id="injected").SerializeToString()
        + _field(
            nyct.NYCT_TRIP_DESCRIPTOR_FIELD_NUMBER, nyct_trip + _unknown_varint(51)
        )
        + _unknown_varint(50)
    )
    event = rt.TripUpdate.StopTimeEvent(time=1792507100).SerializeToString()
    nyct_stop = nyct.NyctStopTimeUpdate(scheduled_track="1").SerializeToString()
    stu = rt.TripUpdate.StopTimeUpdate
    stop = (
        stu(stop_id="L01N").SerializeToString()
        + _field(stu.ARRIVAL_FIELD_NUMBER, event + _unknown_varint(52))
        + _field(stu.DEPARTURE_FIELD_NUMBER, event + _unknown_varint(53))
        + _field(stu.STOP_TIME_PROPERTIES_FIELD_NUMBER, b"")
        + _field(
            nyct.NYCT_STOP_TIME_UPDATE_FIELD_NUMBER, nyct_stop + _unknown_varint(54)
        )
        + _unknown_varint(55)
    )
    vehicle = rt.VehicleDescriptor(id="car").SerializeToString()
    update = (
        _field(rt.TripUpdate.TRIP_FIELD_NUMBER, trip)
        + _field(rt.TripUpdate.STOP_TIME_UPDATE_FIELD_NUMBER, stop)
        + _field(rt.TripUpdate.VEHICLE_FIELD_NUMBER, vehicle)
    )
    entity = rt.FeedEntity(id="injected").SerializeToString() + _field(
        rt.FeedEntity.TRIP_UPDATE_FIELD_NUMBER, update
    )
    decoded = _gtfsrt("l", payload + _field(rt.FeedMessage.ENTITY_FIELD_NUMBER, entity))
    injected = decoded.trip_updates[-1]
    assert injected.entity_id == "injected"
    assert injected.unmodeled_fields == [
        "1.50",
        "1.1001.51",
        "2.2.52",
        "2.3.53",
        "2.6",
        "2.55",
        "2.1001.54",
        "3",
    ]
    assert injected.trip.nyct_train_id == "x"
    assert injected.stop_times[0].nyct_scheduled_track == "1"


@pytest.mark.parametrize(
    "payload",
    [
        pytest.param(b"\xff\xff\xff", id="garbage"),
        pytest.param(b"", id="no-header"),
        pytest.param(b"\x0a\x00", id="header-without-version"),
    ],
)
def test_a_payload_that_does_not_parse_raises(payload: bytes) -> None:
    with pytest.raises(DecodeError):
        _gtfsrt("si", payload)


def test_a_truncated_real_payload_raises_rather_than_decoding_part() -> None:
    payload, _ = _fixture("si")
    with pytest.raises(DecodeError):
        _gtfsrt("si", payload[: len(payload) // 2])


def test_bytes_that_are_not_the_lineage_payload_are_rejected() -> None:
    payload, fetch = _fixture("si")
    with pytest.raises(DecodeError):
        decode_gtfsrt(payload + b"\x00", fetch)


def test_values_the_contracts_reject_raise_decode_errors() -> None:
    bad_direction = _minimal_feed()
    bad_direction.entity[0].vehicle.trip.direction_id = 2
    bad_time = _minimal_feed()
    bad_time.entity[0].trip_update.stop_time_update[0].arrival.time = -5
    for feed in (bad_direction, bad_time):
        with pytest.raises(DecodeError):
            _gtfsrt("si", feed.SerializeToString())


# --- alerts ---------------------------------------------------------------------


def test_alert_digest_matches_the_archived_reserialized_entity() -> None:
    (archived_file,) = RAW_DIR.glob("archive-alerts-*.json")
    archived = json.loads(archived_file.read_text(encoding="utf-8"))["alert"]
    payload, fetch = _fixture("alerts")
    presence = decode_alerts_json(payload, fetch)
    (entry,) = [e for e in presence.entries if e.alert_id == archived["id"]]
    assert entry.content_digest == alert_content_digest(archived)


def test_alert_presence_is_sorted_and_carries_updated_at() -> None:
    payload, fetch = _fixture("alerts")
    live = json.loads(payload)["entity"]
    presence = decode_alerts_json(payload, fetch)
    assert [e.alert_id for e in presence.entries] == sorted(e["id"] for e in live)
    published = {
        e["id"]: e["alert"]["transit_realtime.mercury_alert"]["updated_at"]
        for e in live
    }
    assert {e.alert_id: e.updated_at for e in presence.entries} == published


def test_a_feed_with_no_entities_is_an_empty_presence() -> None:
    payload = json.dumps({"header": {"timestamp": 1792507000}}).encode()
    presence = decode_alerts_json(payload, _fetch("alerts", payload))
    assert isinstance(presence, AlertPresence)
    assert (presence.header_timestamp, presence.entries) == (1792507000, [])


@pytest.mark.parametrize(
    ("header", "timestamp"),
    [
        ({"timestamp": "1792507000"}, 1792507000),
        ({"timestamp": 1792507000.0}, 1792507000),
        ({"timestamp": "soon"}, None),
        ({"timestamp": -1}, None),
        ("not an object", None),
    ],
)
def test_alerts_header_timestamp_is_read_as_the_recorder_reads_it(
    header: Any, timestamp: int | None
) -> None:
    payload = json.dumps({"header": header}).encode()
    presence = decode_alerts_json(payload, _fetch("alerts", payload))
    assert presence.header_timestamp == timestamp


@pytest.mark.parametrize(
    "payload",
    [
        b"not json",
        b"[]",
        b'{"entity": {}}',
        b'{"entity": [{"alert": {}}]}',
        b"NaN",
        b'{"entity": ' + b"[" * 100_000 + b"]" * 100_000 + b"}",
    ],
    ids=["text", "array", "entity-object", "no-id", "nan", "deeply-nested"],
)
def test_alerts_json_that_is_not_a_feed_raises(payload: bytes) -> None:
    with pytest.raises(DecodeError):
        decode_alerts_json(payload, _fetch("alerts", payload))


def test_numbered_alerts_become_a_presence_of_their_entities() -> None:
    payload, _ = _fixture("numbered")
    alerts = [
        e for e in rt.FeedMessage.FromString(payload).entity if e.HasField("alert")
    ]
    presence = _gtfsrt("numbered").alerts
    assert presence is not None
    assert [e.alert_id for e in presence.entries] == sorted(e.id for e in alerts)
    assert {e.updated_at for e in presence.entries} == {None}


# --- ENE --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("published", "instant"),
    [
        ("10/20/2026 11:41:00 AM", 1792510860),
        ("10/20/2026 12:00:00 AM", 1792468800),
        ("10/20/2026 12:00:00 PM", 1792512000),
        ("11/01/2026 01:30:00 AM", Unknown),
        ("03/08/2026 02:30:00 AM", Unknown),
        ("11/01/2026 03:00:00 AM", 1793520000),
        ("02/30/2026 10:00:00 AM", Unknown),
        ("10/20/2026 13:00:00 PM", Unknown),
        ("2026-10-20 11:41", Unknown),
        ("", None),
        ("  ", None),
        ("12/31/1969 06:00:00 PM", Unknown),
        ("01/01/0001 12:00:00 AM", Unknown),
        ("\u0661\u0660/\u0662\u0660/\u0662\u0660\u0662\u0666 11:41:00 AM", Unknown),
        (None, None),
    ],
)
def test_ene_wall_times_parse_in_new_york(
    published: str | None, instant: int | type[Unknown] | None
) -> None:
    parsed = new_york_instant(published)
    if instant is Unknown:
        assert isinstance(parsed, Unknown)
    else:
        assert parsed == instant


def test_ene_records_keep_their_order_and_omitted_times_are_none() -> None:
    payload, fetch = _fixture("ene_current")
    published = json.loads(payload)
    observed = decode_ene(payload, fetch)
    assert [o.record for o in observed] == published
    assert [o.record_index for o in observed] == list(range(len(published)))

    record = {k: v for k, v in published[0].items() if k != "outagedate"}
    stripped = json.dumps([record]).encode()
    (only,) = decode_ene(stripped, _fetch("ene_current", stripped))
    assert isinstance(only, EquipmentObservation)
    assert (only.outage_start_local, only.outage_start) == (None, None)


@pytest.mark.parametrize(
    "payload",
    [b"{}", b'[{"equipment": 1}]', b'[{"station": "Grand St"}]', b"[1]"],
)
def test_ene_payload_that_is_not_a_record_list_raises(payload: bytes) -> None:
    with pytest.raises(DecodeError):
        decode_ene(payload, _fetch("ene_upcoming", payload))
