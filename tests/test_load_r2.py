"""Flag derivation in build_tick_observations (training/load_r2.py).

Synthetic alert-version bodies — no R2 access.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from itertools import pairwise
from typing import Any

import pytest

from momentarily.hmm import schedule_bin, tod_bin
from training.load_r2 import (
    AdvanceBaseline,
    PresenceMask,
    ServiceQuantiles,
    _binom_lower_tail,  # pyright: ignore[reportPrivateUsage]
    _snap_tick,  # pyright: ignore[reportPrivateUsage]
    advance_baseline_to_json,
    build_movement_series,
    build_movement_series_by_direction,
    build_segment_series,
    build_tick_observations,
    classify_direction,
    compute_advance_baseline,
    compute_baseline,
    compute_service_quantiles,
    input_manifest_hash,
    presence_mask_from_predictions,
    service_baseline_to_json,
    service_quantiles_to_json,
)


def _approx(expected: float) -> object:
    """Typed wrapper around ``pytest.approx`` (pins the Unknown return type)."""
    return pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]


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
    obs = build_tick_observations([_body("lmm:planned_work:1", "No Scheduled Service")])
    assert obs
    for o in obs:
        assert o.observation.alert_count == 0
        assert o.observation.severity_sum == 0
        assert not o.observation.has_suspended_alert
    # ...and it doesn't mask a real disruption alongside it.
    obs = build_tick_observations(
        [
            _body("lmm:planned_work:1", "No Scheduled Service"),
            _body("lmm:alert:2", "Delays"),
        ]
    )
    assert obs
    for o in obs:
        assert o.observation.alert_count == 1
        assert o.observation.has_delays


def test_suspended_and_no_trains_set_flag():
    for i, alert_type in enumerate(("Suspended", "Part Suspended", "No Trains")):
        obs = build_tick_observations([_body(f"lmm:alert:{i}", alert_type)])
        assert obs
        assert all(o.observation.has_suspended_alert for o in obs), alert_type


def test_planned_suspension_excluded():
    obs = build_tick_observations(
        [_body("lmm:planned_work:1", "Planned - Part Suspended")]
    )
    assert obs
    assert all(not o.observation.has_suspended_alert for o in obs)
    # Planned work is excluded from the observation entirely now, not just the
    # suspended flag — it never contributes to any channel.
    assert all(not o.observation.has_planned for o in obs)


def test_extra_service_is_invisible_to_the_hmm():
    """Extra Service is good news — it must not contribute to any observation
    channel (count, severity, flags)."""
    obs = build_tick_observations([_body("lmm:planned_work:1", "Extra Service")])
    assert obs
    for o in obs:
        assert o.observation.alert_count == 0
        assert o.observation.severity_sum == 0
        assert not o.observation.has_service_change
    # ...and it doesn't mask a real disruption alongside it.
    obs = build_tick_observations(
        [_body("lmm:planned_work:1", "Extra Service"), _body("lmm:alert:2", "Delays")]
    )
    assert obs
    for o in obs:
        assert o.observation.alert_count == 1
        assert o.observation.has_delays


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
        obs = build_tick_observations([_body(f"lmm:planned_work:{i}", alert_type)])
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
        obs = build_tick_observations([_body("lmm:alert:1", alert_type)])
        assert obs
        for o in obs:
            assert o.observation.alert_count == 1
            assert getattr(o.observation, flag), alert_type


def test_mixed_realtime_and_planned_only_realtime_counts():
    """A real-time Delays alongside a planned suspension: only the real-time
    alert counts, and the planned alert's flag never sets."""
    obs = build_tick_observations(
        [
            _body("lmm:alert:1", "Delays"),
            _body("lmm:planned_work:2", "Planned - Part Suspended"),
        ]
    )
    assert obs
    for o in obs:
        assert o.observation.alert_count == 1
        assert o.observation.has_delays
        assert not o.observation.has_suspended_alert


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
        [_body("lmm:alert:1", "Delays", start=T0, end=T0 + 600)], active_mask=mask
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


# --- Per-(route,direction,tod_bin) advance-rate baseline ---


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
    assert series[("A", "north", T0)] == {
        "vehicles_n": 8,
        "advanced_n": 6,
        "stalled_n": 2,
    }
    assert series[("A", "south", T0)] == {
        "vehicles_n": 7,
        "advanced_n": 3,
        "stalled_n": 4,
    }


def test_movement_series_skips_rows_without_by_direction():
    # An older archive row (no by_direction) contributes nothing.
    bodies = [{"observed_at": T0, "rows": {"A": {"advanced_n": 5, "stalled_n": 1}}}]
    assert build_movement_series_by_direction(bodies) == {}


def _counted_body(tick: int, route: str, transitions: dict[str, int]) -> dict[str, Any]:
    """A row whose counters and transitions agree, as the Worker archives them."""
    advanced = sum(
        n for p, n in transitions.items() if p.split(">")[0] != p.split(">")[1]
    )
    stalled = sum(
        n for p, n in transitions.items() if p.split(">")[0] == p.split(">")[1]
    )
    return {
        "observed_at": tick,
        "rows": {
            route: {
                "vehicles_n": advanced + stalled,
                "stopped_n": stalled,
                "moving_n": advanced,
                "advanced_n": advanced,
                "stalled_n": stalled,
                "by_direction": {
                    "north": {
                        "vehicles_n": advanced + stalled,
                        "advanced_n": advanced,
                        "stalled_n": stalled,
                        "transitions": transitions,
                    }
                },
            }
        },
    }


_TRANSITIONS = {"A01N>A02N": 4, "A02N>A03N": 3, "A01N>A01N": 9, "A02N>A02N": 1}


def _through_only(route: str, direction: str, frm: str) -> bool:
    return (route, direction, frm) == ("A", "north", "A02N")


def test_movement_series_by_direction_unfiltered_reads_the_archived_counters():
    bodies = [_counted_body(T0, "A", _TRANSITIONS)]
    assert build_movement_series_by_direction(bodies)[("A", "north", T0)] == {
        "vehicles_n": 17,
        "advanced_n": 7,
        "stalled_n": 10,
    }


def test_movement_series_by_direction_counts_only_admitted_from_stops():
    """A01N's 9 stalls and 4 advances drop out; A02N's own 3 and 1 stay. The
    terminal layover is what the archived counters blend in."""
    bodies = [_counted_body(T0, "A", _TRANSITIONS)]
    series = build_movement_series_by_direction(bodies, counts_from_stop=_through_only)
    assert series[("A", "north", T0)] == {
        "vehicles_n": 17,  # presence is unaffected
        "advanced_n": 3,
        "stalled_n": 1,
    }


def test_movement_series_route_level_filters_and_keeps_presence():
    bodies = [_counted_body(T0, "A", _TRANSITIONS)]
    row = build_movement_series(bodies, counts_from_stop=_through_only)[("A", T0)]
    assert row == {
        "vehicles_n": 17,
        "stopped_n": 10,
        "moving_n": 7,
        "advanced_n": 3,
        "stalled_n": 1,
    }


def test_filtered_movement_series_ignores_a_row_with_no_transitions():
    """Counters alone can't be narrowed after the fact, so a row that carries no
    transitions map contributes nothing rather than its unfiltered totals."""
    bodies = [_movement_body(T0, "A", north=(8, 6, 2))]
    series = build_movement_series_by_direction(bodies, counts_from_stop=_through_only)
    assert series[("A", "north", T0)] == {
        "vehicles_n": 8,
        "advanced_n": 0,
        "stalled_n": 0,
    }


# --- Per-(route,direction,from,to,tick) segment leaf ---


def _segment_body(
    tick: int,
    route: str,
    *,
    north: dict[str, int] | None = None,
    south: dict[str, int] | None = None,
) -> dict[str, Any]:
    def dir_row(transitions: dict[str, int] | None) -> dict[str, Any]:
        return {"transitions": transitions or {}}

    return {
        "observed_at": tick,
        "rows": {
            route: {"by_direction": {"north": dir_row(north), "south": dir_row(south)}}
        },
    }


def test_segment_series_round_trips_with_snapped_tick():
    observed_at = T0 + 37  # not tick-aligned; _snap_tick floors it to T0
    assert _snap_tick(observed_at) == T0
    bodies = [
        _segment_body(observed_at, "A", north={"A09N>A10N": 3}, south={"A09S>A09S": 1})
    ]
    assert build_segment_series(bodies) == {
        ("A", "north", "A09N", "A10N", T0): 3,
        ("A", "south", "A09S", "A09S", T0): 1,
    }


def test_segment_series_sums_same_key_across_bodies_and_keeps_ticks_separate():
    bodies = [
        _segment_body(T0, "A", north={"A09N>A10N": 2}),
        _segment_body(T0, "A", north={"A09N>A10N": 5}),
        _segment_body(T0 + TICK, "A", north={"A09N>A10N": 1}),
    ]
    series = build_segment_series(bodies)
    assert series[("A", "north", "A09N", "A10N", T0)] == 7
    assert series[("A", "north", "A09N", "A10N", T0 + TICK)] == 1


def test_segment_series_skips_malformed_empty_and_nonpositive_entries():
    bodies = [
        _segment_body(
            T0,
            "A",
            north={
                "no-arrow": 5,
                ">A01N": 4,
                "A01N>": 3,
                "A01N>A02N": 0,
                "A01N>A03N": -2,
                "A05N>A06N": 2,  # the only entry that should survive
            },
        )
    ]
    assert build_segment_series(bodies) == {("A", "north", "A05N", "A06N", T0): 2}


def test_segment_series_missing_transitions_or_by_direction_yields_nothing():
    bodies: list[dict[str, Any]] = [
        {
            "observed_at": T0,
            "rows": {"A": {"by_direction": {"north": {}, "south": {}}}},
        },  # no transitions key
        {"observed_at": T0, "rows": {"A": {}}},  # no by_direction at all
        {"observed_at": T0, "rows": {"A": None}},  # malformed row, not a dict
    ]
    assert build_segment_series(bodies) == {}


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


def test_service_baseline_to_json_nests_route_tod():
    baseline = {("A", 1): 6.0, ("A", 3): 4.5, ("B", 1): 2.0}
    doc = service_baseline_to_json(baseline)
    assert doc == {"A": {"1": 6.0, "3": 4.5}, "B": {"1": 2.0}}
    # JSON object keys must be strings (tod_bin stringified for delivery).
    assert all(isinstance(k, str) for k in doc["A"])


def _assigned_series(route: str, vals: list[int]) -> dict[tuple[str, int], int]:
    return {(route, T0 + i * TICK): v for i, v in enumerate(vals)}


def test_compute_service_quantiles_nearest_rank_p10_p90():
    """Nearest-rank on the cell's own sorted assigned_n samples: for the 10
    values 1..10, p10 is sorted[n // 10] (the 2nd-smallest) and p90 is
    sorted[int(n * 0.9)] (the largest) — both an OBSERVED assigned_n value,
    never interpolated between two samples."""
    series = _assigned_series("A", list(range(1, 11)))  # 1..10, n=10
    cells = compute_service_quantiles(series, min_samples=10)
    cell = cells[("A", tod_bin(T0))]
    assert cell.p10 == 2.0
    assert cell.p90 == 10.0


def test_compute_service_quantiles_omits_cells_below_min_samples():
    # route A: 8 samples (< min_samples); route B: a single sample.
    series = _assigned_series("A", [10] * 8)
    series[("B", T0)] = 5
    cells = compute_service_quantiles(series, min_samples=10)
    # Below the sample floor -> omitted, same as compute_baseline (caller
    # treats a missing cell as "can't judge").
    assert cells == {}


def test_service_quantiles_bracket_the_baseline_median_cell_for_cell():
    """p10 <= median <= p90 for every cell, so the Worker's derived low/high
    ratios always straddle 1.0. The median and the quantiles are computed by two
    separate functions over the same series; nothing in the types stops one of
    them from drifting (a different min_samples, bin_fn, or an added filter), and
    the failure is silent — a high ratio under 1.0 would mark a route as running
    notably high while it sits below its own median."""
    rng = random.Random(20260823)
    series: dict[tuple[str, int], int] = {}
    # Gate at the production floor of 20, which is the point separate filtering
    # could drift. TICK is 300s, so only 12 ticks fit one wall-clock hour and a
    # single hour can never reach 20 for an hourly schedule_bin. So stack four
    # same-weekday dates at the SAME local hour: 4 x 12 = 48 samples in one
    # (route, bin) under both bin functions. Flooring the base to 3600 aligns it
    # to a UTC hour, which is also an ET hour (ET is a whole-hour offset), and a
    # 7-day step is a whole number of hours, so every tick lands in that same
    # local hour and daytype. The one-cell assertion below fails loudly if that
    # ever stops holding.
    floor = 20
    base = (T0 // 3600) * 3600
    for route in ("A", "B", "C"):
        for week in range(4):
            for i in range(12):
                tick = base + week * 7 * 86400 + i * TICK
                # Skewed and bimodal cells, which is the shape that actually
                # occurs: a route-hour with two operating modes, not noise
                # around one.
                series[(route, tick)] = rng.choice([1, 2, 2, 3, 9, 40])

    # BinKey is a constrained TypeVar (int or str, never a union), so a loop
    # variable holding both bin functions cannot satisfy it. Take one function
    # per call and keep the parameter generic.
    def check[BinKey: (int, str)](bin_fn: Callable[[int], BinKey]) -> None:
        baseline = compute_baseline(series, bin_fn=bin_fn, min_samples=floor)
        quantiles = compute_service_quantiles(series, bin_fn=bin_fn, min_samples=floor)
        assert len(baseline) == 3, (
            f"{bin_fn.__name__}: expected one cell per route, got {sorted(baseline)}"
        )
        assert set(baseline) == set(quantiles), "cells must be gated identically"
        for key, cell in quantiles.items():
            median = baseline[key]
            assert cell.p10 <= median <= cell.p90, (
                f"{key}: p10={cell.p10} median={median} p90={cell.p90}"
            )
            assert cell.p10 / median <= 1.0 <= cell.p90 / median

    check(tod_bin)
    check(schedule_bin)


def test_service_quantiles_to_json_nests_route_bin_and_stringifies_keys():
    quantiles = {
        ("A", 1): ServiceQuantiles(p10=8.0, p90=13.0),
        ("A", 3): ServiceQuantiles(p10=4.0, p90=9.0),
        ("B", 1): ServiceQuantiles(p10=1.0, p90=2.0),
    }
    doc = service_quantiles_to_json(quantiles)
    assert doc == {
        "A": {"1": {"p10": 8.0, "p90": 13.0}, "3": {"p10": 4.0, "p90": 9.0}},
        "B": {"1": {"p10": 1.0, "p90": 2.0}},
    }
    # JSON object keys must be strings (bin key stringified for delivery).
    assert all(isinstance(k, str) for k in doc["A"])


# --- classify_direction: three-way significance-gated call ---
#
# Case math (baseline p0 as given; prior_strength=8, disrupted_ratio=0.5, alpha=0.05):
#   case1 p0=0.125 advanced=0 matched=8:  post=0.0625==0.5*p0 (<=); tail=0.875**8~=0.3436>alpha
#         -> None. THE FIX: a short shuttle's degenerate baseline no longer misfires
#         disrupted on an ordinary zero-advance tick.
#   case2 p0=0.125 advanced=0 matched=25: post~=0.0303<=0.0625; tail=0.875**25~=0.0356<=alpha
#         -> disrupted. The same degenerate baseline still fires once there's enough evidence.
#   case3 p0=0.55  advanced=0 matched=17: post=0.176<=0.275; tail=0.45**17~=1.2e-6<=alpha
#         -> disrupted (a healthy trunk frozen solid).
#   case4 p0=0.55  advanced=8 matched=17: post=0.496>0.275 -> normal (posterior clears the
#         cutoff outright, no significance test needed).
#
# This exact (advanced, stalled, p0) -> expected-label table is reused verbatim in
# viz's classifyDirection tests (viz/tests/movement.test.ts) and the worker's
# deriveMovementState tests (worker/test/movement_state.test.ts) as a cross-mirror
# parity spot-check: the same inputs must resolve to the same label everywhere.
@pytest.mark.parametrize(
    ("advanced", "stalled", "p0", "expected"),
    [
        pytest.param(0, 8, 0.125, None, id="case1_shuttle_false_positive_now_abstains"),
        pytest.param(
            0, 25, 0.125, "disrupted", id="case2_sustained_shuttle_freeze_still_fires"
        ),
        pytest.param(0, 17, 0.55, "disrupted", id="case3_trunk_freeze_still_fires"),
        pytest.param(8, 9, 0.55, "normal", id="case4_normal_above_ratio"),
    ],
)
def test_classify_direction_three_way_cases(
    advanced: int, stalled: int, p0: float, expected: str | None
) -> None:
    baseline = AdvanceBaseline(p0=p0, n=50, alpha=50 * p0, beta=50 * (1 - p0))
    assert classify_direction(advanced, stalled, baseline) == expected


def test_classify_direction_below_min_matched_is_none():
    # Below MIN_MATCHED_TRIPS=3: the matched-floor guard short-circuits before the
    # posterior/significance path is ever evaluated, unchanged by the three-way rewrite.
    baseline = AdvanceBaseline(p0=0.9, n=50, alpha=45.0, beta=5.0)
    assert classify_direction(1, 1, baseline) is None


def test_classify_direction_no_baseline_is_none():
    assert classify_direction(8, 1, None) is None


# --- _binom_lower_tail: exact P(X<=k), boundaries, monotonicity ---


def test_binom_lower_tail_exact_values():
    assert _binom_lower_tail(0, 8, 0.125) == _approx(0.875**8)
    assert _binom_lower_tail(0, 10, 0.5) == _approx(9.765625e-4)


def test_binom_lower_tail_k_at_or_above_n_saturates_to_one():
    assert _binom_lower_tail(17, 17, 0.55) == 1.0
    assert _binom_lower_tail(20, 17, 0.55) == 1.0  # k > n also saturates


def test_binom_lower_tail_negative_k_is_zero():
    assert _binom_lower_tail(-1, 10, 0.5) == 0.0


def test_binom_lower_tail_monotonic_nondecreasing_in_k():
    n, p = 20, 0.3
    tails = [_binom_lower_tail(k, n, p) for k in range(n + 1)]
    assert all(b >= a for a, b in pairwise(tails))
