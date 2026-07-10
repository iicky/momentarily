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
    obs = build_observations(
        [_rec("No Scheduled Service", alert_id="lmm:planned_work:1")]
    )
    assert obs
    for o in obs:
        assert o.observation.alert_count == 0
        assert o.observation.severity_sum == 0
        assert not o.observation.has_suspended_alert
    # ...and it doesn't mask a real disruption alongside it.
    obs = build_observations(
        [
            _rec("No Scheduled Service", alert_id="lmm:planned_work:1"),
            _rec("Delays", alert_id="lmm:alert:2"),
        ]
    )
    assert obs
    for o in obs:
        assert o.observation.alert_count == 1
        assert o.observation.has_delays


def test_extra_service_is_invisible_to_the_hmm():
    obs = build_observations([_rec("Extra Service", alert_id="lmm:planned_work:1")])
    assert obs
    for o in obs:
        assert o.observation.alert_count == 0
        assert o.observation.severity_sum == 0


def test_suspended_and_no_trains_still_set_flag():
    for i, alert_type in enumerate(("Suspended", "No Trains")):
        obs = build_observations([_rec(alert_type, alert_id=f"lmm:alert:{i}")])
        assert obs
        assert all(o.observation.has_suspended_alert for o in obs), alert_type


def test_planned_suspension_excluded_from_suspended_flag():
    obs = build_observations(
        [_rec("Planned - Part Suspended", alert_id="lmm:planned_work:1")]
    )
    assert obs
    for o in obs:
        assert not o.observation.has_suspended_alert
        # Planned work is excluded from the observation entirely now, not just
        # the suspended flag — it never contributes to any channel.
        assert not o.observation.has_planned


def test_planned_only_route_is_quiet_observation():
    """A route whose only active alert is planned/scheduled work drops out of
    the HMM observation entirely: count, severity, and every flag (including
    has_planned) read as if nothing were active."""
    planned_types = (
        "Planned - Part Suspended",
        "Planned - Stops Skipped",
        "Reduced Service",
        "Special Schedule",
    )
    for i, alert_type in enumerate(planned_types):
        obs = build_observations([_rec(alert_type, alert_id=f"lmm:planned_work:{i}")])
        assert obs
        for o in obs:
            assert o.observation.alert_count == 0
            assert o.observation.severity_sum == 0
            assert not o.observation.has_suspended_alert
            assert not o.observation.has_delays
            assert not o.observation.has_service_change
            assert not o.observation.has_planned


def test_realtime_disruption_counts_and_sets_flag():
    """A real-time (lmm:alert:) Delays/Suspended/Service Change sets
    alert_count == 1 and its corresponding flag."""
    cases = (
        ("Delays", "has_delays"),
        ("Suspended", "has_suspended_alert"),
        ("Service Change", "has_service_change"),
    )
    for alert_type, flag in cases:
        obs = build_observations([_rec(alert_type, alert_id="lmm:alert:1")])
        assert obs
        for o in obs:
            assert o.observation.alert_count == 1
            assert getattr(o.observation, flag), alert_type


def test_mixed_realtime_and_planned_only_realtime_counts():
    """A real-time Delays alongside a planned suspension: only the real-time
    alert counts, and the planned alert's flag never sets."""
    obs = build_observations(
        [
            _rec("Delays", alert_id="lmm:alert:1"),
            _rec("Planned - Part Suspended", alert_id="lmm:planned_work:2"),
        ]
    )
    assert obs
    for o in obs:
        assert o.observation.alert_count == 1
        assert o.observation.has_delays
        assert not o.observation.has_suspended_alert
