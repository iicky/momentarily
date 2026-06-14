"""Observation flag derivation in training/load.py (the local JSONL loader).

Synthetic poll records — no filesystem access. Mirrors the parity rules locked
for the R2 path in test_load_r2.py.
"""

from __future__ import annotations

from typing import Any

from training.load import build_observations

T0 = 1_700_000_100  # tick-aligned


def _rec(
    alert_type: str,
    *,
    route: str = "1",
    alert_id: str = "a1",
    observed_at: int = T0,
) -> dict[str, Any]:
    return {
        "observed_at": observed_at,
        "alert": {
            "id": alert_id,
            "alert": {
                "informed_entity": [
                    {
                        "route_id": route,
                        "transit_realtime.mercury_entity_selector": {
                            "sort_order": f"MTASBWY:{route}:20"
                        },
                    }
                ],
                "transit_realtime.mercury_alert": {"alert_type": alert_type},
            },
        },
    }


def test_no_scheduled_service_is_invisible_to_the_hmm():
    obs = build_observations([_rec("No Scheduled Service")])
    assert obs
    for o in obs:
        assert o.observation.alert_count == 0
        assert o.observation.severity_sum == 0
        assert not o.observation.has_suspended_alert
    # ...and it doesn't mask a real disruption alongside it.
    obs = build_observations(
        [_rec("No Scheduled Service"), _rec("Delays", alert_id="a2")]
    )
    assert obs
    for o in obs:
        assert o.observation.alert_count == 1
        assert o.observation.has_delays


def test_extra_service_is_invisible_to_the_hmm():
    obs = build_observations([_rec("Extra Service")])
    assert obs
    for o in obs:
        assert o.observation.alert_count == 0
        assert o.observation.severity_sum == 0


def test_suspended_and_no_trains_still_set_flag():
    for alert_type in ("Suspended", "No Trains"):
        obs = build_observations([_rec(alert_type)])
        assert obs
        assert all(o.observation.has_suspended_alert for o in obs), alert_type


def test_planned_suspension_excluded_from_suspended_flag():
    obs = build_observations([_rec("Planned - Part Suspended")])
    assert obs
    for o in obs:
        assert not o.observation.has_suspended_alert
        assert o.observation.has_planned
