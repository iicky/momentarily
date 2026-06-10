"""Flag derivation in build_tick_observations (training/load_r2.py).

Synthetic alert-version bodies — no R2 access.
"""

from __future__ import annotations

from typing import Any

from training.load_r2 import build_tick_observations

TICK = 300
T0 = 1_700_000_100  # tick-aligned


def _body(
    alert_id: str,
    alert_type: str,
    route_id: str = "1",
    start: int = T0,
    end: int = T0 + 600,
) -> dict[str, Any]:
    return {
        "observed_at": start,
        "alert": {
            "id": alert_id,
            "alert": {
                "active_period": [{"start": start, "end": end}],
                "informed_entity": [
                    {
                        "route_id": route_id,
                        "transit_realtime.mercury_entity_selector": {
                            "sort_order": f"MTASBWY:{route_id}:20"
                        },
                    }
                ],
                "transit_realtime.mercury_alert": {"alert_type": alert_type},
            },
        },
    }


def test_no_scheduled_service_is_not_suspension():
    """Scheduled absence (overnight/weekend non-service) is normal operations,
    not a suspension — it must not trip the suspended flag. It still counts
    toward alert_count/severity. See momentarily-vk0.3."""
    obs = build_tick_observations([_body("a1", "No Scheduled Service")])
    assert obs
    assert all(not o.observation.has_suspended_alert for o in obs)
    assert all(o.observation.alert_count == 1 for o in obs)


def test_suspended_and_no_trains_set_flag():
    for alert_type in ("Suspended", "Part Suspended", "No Trains"):
        obs = build_tick_observations([_body("a1", alert_type)])
        assert obs
        assert all(o.observation.has_suspended_alert for o in obs), alert_type


def test_planned_suspension_excluded():
    obs = build_tick_observations([_body("a1", "Planned - Part Suspended")])
    assert obs
    assert all(not o.observation.has_suspended_alert for o in obs)
    assert all(o.observation.has_planned for o in obs)
