"""Flag derivation in build_tick_observations (training/load_r2.py).

Synthetic alert-version bodies — no R2 access.
"""

from __future__ import annotations

from typing import Any

from momentarily.hmm import tod_bin
from training.load_r2 import (
    PresenceMask,
    advance_baseline_to_json,
    build_movement_series_by_direction,
    build_tick_observations,
    compute_advance_baseline,
    input_manifest_hash,
    presence_mask_from_predictions,
)

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


def _pred(ts: int, route: str, primary: str | None) -> Any:
    """Minimal PredictionRecord via from_json — only ts/route/primary matter
    for the presence mask."""
    from training.eval import PredictionRecord

    return PredictionRecord.from_json(
        {
            "ts": ts,
            "route": route,
            "condition": "disrupted",
            "regime_entered_at": ts,
            "p_normal": 0.1,
            "p_disrupted": 0.8,
            "p_suspended": 0.1,
            "p_normal_in_30min": 0.2,
            "p_normal_in_60min": 0.3,
            "p_normal_in_120min": 0.4,
            "recovery_minutes": 30,
            "recovery_minutes_low": 15,
            "recovery_minutes_high": 60,
            "primary_alert_type": primary,
        }
    )


def test_presence_mask_drops_over_extended_tail():
    # Alert archived active T0..T0+600 (3 ticks), but the live Worker only saw
    # it at T0 — the later ticks are the over-extended tail and must drop.
    mask = PresenceMask(
        active=frozenset({("1", T0)}),
        covered=frozenset({T0, T0 + TICK, T0 + 2 * TICK}),
    )
    obs = build_tick_observations(
        [_body("a1", "Delays", start=T0, end=T0 + 600)], active_mask=mask
    )
    assert [o.tick for o in obs] == [T0]
    assert obs[0].observation.has_delays


def test_presence_mask_keeps_ticks_it_does_not_cover():
    # Mask only covers T0; T0+TICK / T0+2*TICK are outside the stream, so they
    # fall back to the raw reconstruction (no wrongful drop).
    mask = PresenceMask(active=frozenset({("1", T0)}), covered=frozenset({T0}))
    obs = build_tick_observations(
        [_body("a1", "Delays", start=T0, end=T0 + 600)], active_mask=mask
    )
    assert [o.tick for o in obs] == [T0, T0 + TICK, T0 + 2 * TICK]


def test_presence_mask_none_is_unchanged_behavior():
    # Without a mask the reconstruction fills the whole active_period.
    obs = build_tick_observations([_body("a1", "Delays", start=T0, end=T0 + 600)])
    assert [o.tick for o in obs] == [T0, T0 + TICK, T0 + 2 * TICK]


def test_presence_mask_from_predictions_uses_primary_alert_type():
    mask = presence_mask_from_predictions(
        [_pred(T0, "1", "Delays"), _pred(T0, "2", None), _pred(T0 + TICK, "1", None)]
    )
    assert mask.is_active("1", T0)
    assert not mask.is_active("2", T0)  # primary None → not active
    assert mask.covers(T0)
    assert mask.covers(T0 + TICK)
    assert not mask.is_active("1", T0 + TICK)


# --- input_manifest_hash: deterministic lineage fingerprint over object keys ---


def test_manifest_hash_is_order_independent():
    a = ["archive/alerts/2026-06-01/100.json", "archive/alerts/2026-06-01/200.json"]
    assert input_manifest_hash(a) == input_manifest_hash(list(reversed(a)))


def test_manifest_hash_changes_with_key_set():
    base = ["archive/alerts/2026-06-01/100.json"]
    added = [*base, "archive/alerts/2026-06-01/200.json"]
    assert input_manifest_hash(base) != input_manifest_hash(added)


def test_manifest_hash_is_blake3_hex():
    h = input_manifest_hash(["archive/alerts/2026-06-01/100.json"])
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_manifest_hash_empty_is_stable():
    assert input_manifest_hash([]) == input_manifest_hash([])
    # A key boundary follows every key, so the empty set is NOT the same as a
    # single empty-string key.
    assert input_manifest_hash([]) != input_manifest_hash([""])


# --- Per-(route,direction,tod_bin) advance-rate baseline (momentarily-vhh.3) ---


def _movement_body(
    tick: int,
    route: str,
    *,
    north: tuple[int, int, int] = (0, 0, 0),  # (vehicles_n, advanced_n, stalled_n)
    south: tuple[int, int, int] = (0, 0, 0),
) -> dict[str, Any]:
    def dir_row(t: tuple[int, int, int]) -> dict[str, int]:
        return {"vehicles_n": t[0], "advanced_n": t[1], "stalled_n": t[2]}

    return {
        "observed_at": tick,
        "rows": {
            route: {
                "vehicles_n": north[0] + south[0],
                "advanced_n": north[1] + south[1],
                "stalled_n": north[2] + south[2],
                "by_direction": {"north": dir_row(north), "south": dir_row(south)},
            }
        },
    }


def test_movement_series_by_direction_splits_north_south():
    bodies = [_movement_body(T0, "A", north=(8, 6, 2), south=(7, 3, 4))]
    series = build_movement_series_by_direction(bodies)
    assert series[("A", "north", T0)] == {"vehicles_n": 8, "advanced_n": 6, "stalled_n": 2}
    assert series[("A", "south", T0)] == {"vehicles_n": 7, "advanced_n": 3, "stalled_n": 4}


def test_movement_series_skips_rows_without_by_direction():
    # A pre-vhh.2 archive row (no by_direction) contributes nothing.
    bodies = [{"observed_at": T0, "rows": {"A": {"advanced_n": 5, "stalled_n": 1}}}]
    assert build_movement_series_by_direction(bodies) == {}


def test_advance_baseline_median_resists_disrupted_minority():
    """Mostly-healthy north ticks (advance ~0.9) with a frozen minority should
    still yield a high p0 — the median ignores the disrupted tail."""
    bodies: list[dict[str, Any]] = []
    # 24 healthy ticks: 9 of 10 advanced. 6 frozen ticks: 0 of 10 advanced.
    for i in range(24):
        bodies.append(_movement_body(T0 + i * TICK, "A", north=(10, 9, 1)))
    for i in range(24, 30):
        bodies.append(_movement_body(T0 + i * TICK, "A", north=(10, 0, 10)))
    series = build_movement_series_by_direction(bodies)
    baseline = compute_advance_baseline(series, prior_strength=50.0, min_samples=20)
    cell = baseline[("A", "north", tod_bin(T0))]
    assert cell.p0 == 0.9  # median of the per-tick fractions, frozen tail ignored
    assert cell.n == 30
    # Beta prior carries p0 at the chosen strength (alpha+beta = prior_strength).
    assert abs(cell.alpha - 45.0) < 1e-9
    assert abs(cell.beta - 5.0) < 1e-9


def test_advance_baseline_keeps_beta_shapes_positive_at_endpoints():
    """A perfectly healthy line (every matched trip advances → median 1.0) must
    not produce a degenerate Beta(strength, 0); p0 is clamped off the endpoint."""
    bodies = [_movement_body(T0 + i * TICK, "A", north=(10, 10, 0)) for i in range(24)]
    series = build_movement_series_by_direction(bodies)
    cell = compute_advance_baseline(series, prior_strength=50.0, min_samples=20)[
        ("A", "north", tod_bin(T0))
    ]
    assert cell.p0 < 1.0
    assert cell.alpha > 0.0
    assert cell.beta > 0.0


def test_advance_baseline_omits_thin_cells_and_low_match_ticks():
    # 5 ticks (< min_samples) and one tick below the matched floor.
    bodies = [_movement_body(T0 + i * TICK, "A", north=(10, 8, 2)) for i in range(5)]
    bodies.append(_movement_body(T0 + 99 * TICK, "A", north=(2, 1, 1)))  # matched=2 < 3
    series = build_movement_series_by_direction(bodies)
    assert compute_advance_baseline(series, min_samples=20) == {}


def test_advance_baseline_to_json_nests_route_direction_todbin():
    bodies = [_movement_body(T0 + i * TICK, "A", north=(10, 7, 3)) for i in range(24)]
    series = build_movement_series_by_direction(bodies)
    baseline = compute_advance_baseline(series, prior_strength=50.0, min_samples=20)
    doc = advance_baseline_to_json(baseline)
    tod = str(tod_bin(T0))
    cell = doc["A"]["north"][tod]
    assert cell["p0"] == 0.7
    assert cell["n"] == 24
    assert abs(cell["alpha"] - 35.0) < 1e-9
    assert abs(cell["beta"] - 15.0) < 1e-9
    # JSON object keys must be strings (tod_bin stringified for delivery).
    assert all(isinstance(k, str) for k in doc["A"]["north"])
