"""Tests asserting the snapshot serializes to the documented JSON shape."""

from __future__ import annotations

import json

from momentarily.schema import (
    SCHEMA_VERSION,
    Bridge,
    Compat,
    Crossing,
    Inference,
    Observation,
    RouteStatus,
    Snapshot,
    StationStatus,
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
    """Observation is the new continuous-measurement slot for bridges/headway/etc."""
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
