"""Tests asserting the snapshot serializes to the documented JSON shape."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from momentarily.schema import (
    SCHEMA_VERSION,
    Bridge,
    Compat,
    Crossing,
    Inference,
    Observation,
    ObservationSample,
    RouteStatus,
    ScheduledHeadway,
    SegmentFlow,
    SegmentRecovery,
    SegmentStatus,
    Snapshot,
    StationServiceFlow,
    StationStatus,
    TrainPosition,
    Trains,
    Tunnel,
)


def test_minimal_snapshot_serializes() -> None:
    snap = Snapshot(generated_at=1_700_000_000)
    payload = json.loads(snap.model_dump_json())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["generated_at"] == 1_700_000_000
    assert payload["alerts"] == []
    assert payload["observations"] == []
    assert payload["routes"] == {}
    assert payload["bridges"] == []
    assert payload["tunnels"] == []
    assert payload["supported_modes"] == []
    assert payload["compat"]["subwaynow_routes"] == {}


def test_attribution_present() -> None:
    snap = Snapshot(generated_at=0)
    assert "Momentarily" in snap.attribution
    assert "MTA" in snap.attribution
    assert "Not affiliated" in snap.attribution


def test_compat_default_empty() -> None:
    compat = Compat()
    assert compat.subwaynow_routes == {}


def test_observation_round_trips() -> None:
    """Observation is the continuous-measurement slot for headway/bridges/etc."""
    obs = Observation(
        entity_ref="bridge:verrazano",
        kind="travel_time",
        value=22.5,
        unit="minutes",
        observed_at=1_700_000_000,
        source="google_directions",
    )
    payload = json.loads(obs.model_dump_json())
    assert payload["entity_ref"] == "bridge:verrazano"
    assert payload["kind"] == "travel_time"
    assert payload["value"] == 22.5
    # A measurement whose entity_ref already locates it carries neither.
    assert payload["direction"] is None
    assert payload["stop_id"] is None
    # No rolling window on a measurement that carries no series.
    assert payload["window"] is None


def test_headway_observation_carries_its_measurement_point() -> None:
    """The shape worker/src/headway.ts publishes: a raw headway reading, per
    (route, direction), located at that cell's canonical reference stop.

    direction and stop_id are what keep a route's two directional readings
    apart — they share entity_ref, so without them the pair is ambiguous —
    and stop_id is the measurement point a headway only means anything
    relative to.
    """
    obs = Observation(
        entity_ref="subway_route:1",
        kind="headway",
        value=240,
        unit="seconds",
        observed_at=1_700_000_300,
        source="gtfs_rt_vehicle_positions",
        direction="north",
        stop_id="121N",
    )
    payload = json.loads(obs.model_dump_json())
    assert payload == {
        "entity_ref": "subway_route:1",
        "kind": "headway",
        "value": 240,
        "unit": "seconds",
        "observed_at": 1_700_000_300,
        "source": "gtfs_rt_vehicle_positions",
        "direction": "north",
        "stop_id": "121N",
        # A bare reading with no history yet carries no window (None, not []).
        "window": None,
        # No timetable baseline attached by hand: observed alone, on-reference.
        "scheduled": None,
        "off_reference": False,
    }


def test_headway_observation_carries_its_rolling_window() -> None:
    """The historical N-car chain: the last hour of one cell's headways in the
    single Observation, oldest first, each entry a gap between successive trains.

    The newest window entry restates the Observation's value/observed_at, and
    the whole series shares its stop_id/direction — so a consumer renders the
    chain from this one object, with no archive and no unbounded array.
    """
    obs = Observation(
        entity_ref="subway_route:1",
        kind="headway",
        value=240,
        unit="seconds",
        observed_at=1_700_000_300,
        source="gtfs_rt_vehicle_positions",
        direction="north",
        stop_id="121N",
        window=[
            ObservationSample(value=300, observed_at=1_700_000_000),
            ObservationSample(value=240, observed_at=1_700_000_300),
        ],
    )
    payload = json.loads(obs.model_dump_json())
    assert payload["window"] == [
        {"value": 300, "observed_at": 1_700_000_000},
        {"value": 240, "observed_at": 1_700_000_300},
    ]
    # The last car is exactly the published single reading.
    assert payload["window"][-1] == {
        "value": payload["value"],
        "observed_at": payload["observed_at"],
    }


def test_headway_observation_normalised_against_the_scheduled_baseline() -> None:
    """The observed-vs-scheduled read the Worker embeds at tick time: the
    timetable median for this cell's hour-of-week, so a consumer states "every 9
    min, scheduled 6" straight off the snapshot with no second fetch."""
    obs = Observation(
        entity_ref="subway_route:1",
        kind="headway",
        value=540,
        unit="seconds",
        observed_at=1_700_000_300,
        source="gtfs_rt_vehicle_positions",
        direction="north",
        stop_id="121N",
        scheduled=ScheduledHeadway(median_headway_s=360, n_trips=10),
    )
    payload = json.loads(obs.model_dump_json())
    assert payload["scheduled"] == {"median_headway_s": 360, "n_trips": 10}
    assert payload["off_reference"] is False


def test_off_reference_reading_withholds_the_scheduled_comparison() -> None:
    """A reroute-fallback reading is taken at a different stop than the baseline
    is keyed on, so `scheduled` stays None and `off_reference` is set — a
    consumer labels the point moved rather than comparing the wrong cell."""
    obs = Observation(
        entity_ref="subway_route:1",
        kind="headway",
        value=540,
        unit="seconds",
        observed_at=1_700_000_300,
        source="gtfs_rt_vehicle_positions",
        direction="north",
        stop_id="118N",  # a fallback stop, not the cell's canonical reference
        scheduled=None,
        off_reference=True,
    )
    payload = json.loads(obs.model_dump_json())
    assert payload["scheduled"] is None
    assert payload["off_reference"] is True


def test_observation_direction_is_a_closed_vocabulary() -> None:
    """Unlike kind/unit/source, direction is not an open string: NYCT runs two
    directions and a third would be a schema change, so it is rejected rather
    than passed through. The ignore is the point — the vocabulary is closed at
    type-check time too, and this asserts it is also closed at runtime, where
    a malformed trainer doc or hand-written payload actually arrives."""
    with pytest.raises(ValidationError):
        Observation(
            entity_ref="subway_route:1",
            kind="headway",
            value=240,
            unit="seconds",
            observed_at=1_700_000_300,
            source="gtfs_rt_vehicle_positions",
            direction="northbound",  # pyright: ignore[reportArgumentType]
        )


def test_snapshot_observations_default_empty() -> None:
    """A snapshot with no measured headway publishes an empty surface — the
    honest reading on a cold start or a vehicle-feed outage, never a zero."""
    snap = Snapshot(generated_at=0)
    assert snap.observations == []
    assert snap.freshness.vehicle_positions is None


def test_bridge_with_crossings() -> None:
    bridge = Bridge(
        id="verrazano",
        name="Verrazzano-Narrows Bridge",
        operator="MTA-BT",
        crossings=[
            Crossing(id="verrazano:upper:westbound", name="Upper level westbound"),
            Crossing(id="verrazano:upper:eastbound", name="Upper level eastbound"),
        ],
    )
    assert len(bridge.crossings) == 2
    assert bridge.crossings[0].id == "verrazano:upper:westbound"


def test_tunnel_minimal() -> None:
    tunnel = Tunnel(
        id="brooklyn_battery", name="Brooklyn-Battery Tunnel", operator="MTA-BT"
    )
    assert tunnel.crossings == []


def test_inference_field_defaults_none_on_status() -> None:
    """During shadow Phase 1 the HMM doesn't populate inference; should be None."""
    route_status = RouteStatus(route_id="Q", label="Good Service")
    assert route_status.inference is None

    station_status = StationStatus(station_complex_id="Q05")
    assert station_status.inference is None


def test_inference_serializes() -> None:
    """When the publisher does populate Inference (Phase 3+), shape is documented."""
    inf = Inference(
        condition="disrupted",
        recovery_minutes=47,
        is_disrupted=True,
        p_normal=0.05,
        p_disrupted=0.83,
        p_suspended=0.12,
        regime_entered_at=1_700_000_000,
        regime_age_seconds=1800,
        recovery_minutes_low=28,
        recovery_minutes_high=71,
        p_normal_in_30min=0.34,
        p_normal_in_60min=0.51,
        p_normal_in_120min=0.71,
    )
    payload = json.loads(inf.model_dump_json())
    assert payload["condition"] == "disrupted"
    assert payload["recovery_minutes"] == 47
    assert payload["model_warming_up"] is False  # default


def test_snapshot_with_supported_modes() -> None:
    snap = Snapshot(generated_at=0, supported_modes=["subway", "ene"])
    payload = json.loads(snap.model_dump_json())
    assert payload["supported_modes"] == ["subway", "ene"]


def test_snapshot_segment_flow_defaults_none() -> None:
    """Absent before the first vehicle tick after deploy, or when stale."""
    snap = Snapshot(generated_at=0)
    assert snap.segment_flow is None


def test_segment_status_recovery_optional() -> None:
    """A cell with no trained curve, or whose clock never started, publishes
    status without a fabricated recovery."""
    seg = SegmentStatus(
        route="F",
        direction="south",
        from_stop="A09S",
        to="A10S",
        status="disrupted",
        entered_at=0,
    )
    assert seg.recovery is None
    payload = json.loads(seg.model_dump_json())
    assert payload["recovery"] is None


def test_segment_flow_serializes() -> None:
    recovery = SegmentRecovery(
        recovery_minutes=40,
        recovery_minutes_low=9,
        recovery_minutes_high=180,
        p_normal_in_30min=0.2,
        p_normal_in_60min=0.5,
        p_normal_in_120min=0.8,
    )
    flow = SegmentFlow(
        observed_at=1_700_000_000,
        segments={
            "F|south|A09S": SegmentStatus(
                route="F",
                direction="south",
                from_stop="A09S",
                to="A10S",
                status="disrupted",
                entered_at=1_699_998_200,
                recovery=recovery,
            )
        },
    )
    payload = json.loads(flow.model_dump_json())
    assert payload["observed_at"] == 1_700_000_000
    seg = payload["segments"]["F|south|A09S"]
    assert seg["to"] == "A10S"
    assert seg["recovery"]["recovery_minutes"] == 40
    assert seg["recovery"]["recovery_indeterminate"] is False  # default


def test_station_service_flow_worst_recovery_optional() -> None:
    """worst_recovery is additive — a doc built before this landed (or a
    consumer on the old shape) must still validate with it absent."""
    flow = StationServiceFlow(status="degraded", worst_deficit=0.9, n_segments=1)
    assert flow.worst_recovery is None
    payload = json.loads(flow.model_dump_json())
    assert payload["worst_recovery"] is None
    # existing fields untouched
    assert payload["status"] == "degraded"
    assert payload["worst_deficit"] == 0.9
    assert payload["n_segments"] == 1


def test_snapshot_has_no_trains_field() -> None:
    """trains lives at its own published artifact (v1/trains.json) now, not
    on Snapshot — the whole point of splitting it out so the canonical
    snapshot consumer (homeassistant-mta-subway) never pays bandwidth for a
    payload it can't use. See Trains's docstring for the full rationale."""
    snap = Snapshot(generated_at=0)
    assert not hasattr(snap, "trains")
    payload = json.loads(snap.model_dump_json())
    assert "trains" not in payload


def test_trains_is_self_describing_like_snapshot() -> None:
    """Trains (the v1/trains.json shape) carries its own observed_at and the
    same provenance block Snapshot does, so a consumer holding only this
    object can still say which build produced it."""
    trains = Trains(observed_at=1_700_000_000)
    payload = json.loads(trains.model_dump_json())
    assert payload["observed_at"] == 1_700_000_000
    assert payload["provenance"] == {
        "code_sha": "unknown",
        "dirty": None,
        "producer": "unknown",
        "params": None,
        "prov_ref": None,
    }
    assert payload["positions"] == []


def test_train_position_serializes() -> None:
    """One dot per (route, direction, stop, stopped) tuple, with the fold
    count `n` — the map overlay's whole point."""
    trains = Trains(
        observed_at=1_700_000_000,
        fresh_feeds=["ace", "bdfm"],
        expected_feeds=["ace", "bdfm"],
        positions=[
            TrainPosition(
                route="F", direction="north", stop="A09N", stopped=False, n=3
            ),
        ],
    )
    payload = json.loads(trains.model_dump_json())
    assert payload["observed_at"] == 1_700_000_000
    assert payload["positions"][0] == {
        "route": "F",
        "direction": "north",
        "stop": "A09N",
        "stopped": False,
        "n": 3,
    }


def test_trains_fresh_feeds_shorter_than_expected_flags_a_partial_read() -> None:
    """A rejected NYCT line-group feed is a silent skip, not an exception —
    fresh_feeds vs. expected_feeds is how a consumer tells "zero trains" from
    "some lines are missing" without hardcoding the feed grouping itself."""
    trains = Trains(
        observed_at=1_700_000_000,
        fresh_feeds=["bdfm", "g"],
        expected_feeds=["ace", "bdfm", "g"],
    )
    payload = json.loads(trains.model_dump_json())
    assert len(payload["fresh_feeds"]) < len(payload["expected_feeds"])
    assert "ace" not in payload["fresh_feeds"]


def test_train_position_direction_optional() -> None:
    """direction is null when NYCT's feed gives no N/S signal at all on
    either the stop_id suffix or the trip_id — never fabricated."""
    pos = TrainPosition(route="L", stop="L06", stopped=True, n=1)
    assert pos.direction is None
    payload = json.loads(pos.model_dump_json())
    assert payload["direction"] is None
