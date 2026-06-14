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


def test_no_scheduled_service_is_invisible_to_the_hmm():
    """Scheduled absence (overnight/weekend non-service, rush-only lines) is a
    planned non-disruption, not something to recover from — like Extra Service
    it drops out of the HMM observation entirely so the filter stays normal and
    is ready at resume. The not_scheduled condition is applied downstream."""
    obs = build_tick_observations([_body("a1", "No Scheduled Service")])
    assert obs
    for o in obs:
        assert o.observation.alert_count == 0
        assert o.observation.severity_sum == 0
        assert not o.observation.has_suspended_alert
    # ...and it doesn't mask a real disruption alongside it.
    obs = build_tick_observations(
        [_body("a1", "No Scheduled Service"), _body("a2", "Delays")]
    )
    assert obs
    for o in obs:
        assert o.observation.alert_count == 1
        assert o.observation.has_delays


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


def test_extra_service_is_invisible_to_the_hmm():
    """Extra Service is good news — it must not contribute to any observation
    channel (count, severity, flags). See momentarily-vk0.11."""
    obs = build_tick_observations([_body("a1", "Extra Service")])
    assert obs
    for o in obs:
        assert o.observation.alert_count == 0
        assert o.observation.severity_sum == 0
        assert not o.observation.has_service_change
    # ...and it doesn't mask a real disruption alongside it.
    obs = build_tick_observations([_body("a1", "Extra Service"), _body("a2", "Delays")])
    assert obs
    for o in obs:
        assert o.observation.alert_count == 1
        assert o.observation.has_delays
