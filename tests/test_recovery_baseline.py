"""The causal recovery-duration climatology: population, pooling, conditional
serve, and the sidecar writer.

The conditional-quantile math itself is pinned cross-language in
test_recovery_parity.py; here we grade the pieces that only live in Python —
the shared population builder (the one review.py also grades against), the
pooling fallback and its recorded level, the primary-alert-type gate, and the
transactional writer (versioned snapshot durable before the live pointer flips).
"""

from __future__ import annotations

from datetime import date
from types import SimpleNamespace
from typing import Any, cast

import pytest

from momentarily.hmm import Observation
from training import recovery_baseline as rb
from training.episodes import TICK_SECONDS, Episode
from training.recovery_baseline import (
    MIN_CELL_SAMPLES,
    PopulationEpisode,
    RecoveryPopulation,
    build_climatology_cells,
    climatology_recovery,
    conditional_remaining_quantiles,
    resolve_climatology_cell,
)

WS = 1_000_000
WE = 4_000_000


def _episode(route: str, duration_min: float) -> Episode:
    onset = 1_000_000
    return Episode(
        route=route,
        onset=onset,
        recovery=onset + int(duration_min * 60),
        peak_state="disrupted",
        cause="delays",
        n_ticks=max(1, int(duration_min * 60 // TICK_SECONDS)),
        left_censored=False,
        right_censored=False,
    )


def _pop(episodes: list[PopulationEpisode]) -> RecoveryPopulation:
    return RecoveryPopulation(
        episodes=episodes, window_start=WS, window_end=WE, severity_floor=2
    )


def _pe(route: str, alert_type: str | None, duration_min: float) -> PopulationEpisode:
    return PopulationEpisode(
        episode=_episode(route, duration_min), alert_type=alert_type
    )


# --- pooling + resolution ----------------------------------------------------


def test_route_cell_stands_alone_at_the_min_boundary() -> None:
    # Exactly MIN samples on (Q, Delays) -> a route-level cell.
    pop = _pop([_pe("Q", "Delays", 10 + i) for i in range(MIN_CELL_SAMPLES)])
    cells = build_climatology_cells(pop)
    cell = resolve_climatology_cell(cells, "Q", "Delays")
    assert cell is not None
    assert cell["level"] == "route"
    assert cell["n"] == MIN_CELL_SAMPLES


def test_thin_route_cell_pools_to_alert_type() -> None:
    # (Q, Delays) has MIN-1 (thin); (R, Delays) adds enough that the Delays pool
    # clears MIN. Q's published cell records level 'alert_type' and the pool's n.
    eps = [_pe("Q", "Delays", 10 + i) for i in range(MIN_CELL_SAMPLES - 1)]
    eps += [_pe("R", "Delays", 20 + i) for i in range(MIN_CELL_SAMPLES)]
    cells = build_climatology_cells(_pop(eps))
    q = resolve_climatology_cell(cells, "Q", "Delays")
    assert q is not None
    assert q["level"] == "alert_type"
    assert q["n"] == 2 * MIN_CELL_SAMPLES - 1


def test_thin_everywhere_pools_to_system() -> None:
    # A handful of episodes, no cell clearing MIN at route or alert_type level.
    eps = [
        _pe("Q", "Delays", 10.0),
        _pe("R", "Reroute", 20.0),
        _pe("S", "Weather", 30.0),
    ]
    cells = build_climatology_cells(_pop(eps))
    cell = resolve_climatology_cell(cells, "Q", "Delays")
    assert cell is not None
    assert cell["level"] == "system"
    assert cell["n"] == 3


def test_resolve_requires_a_primary_alert_type() -> None:
    cells = build_climatology_cells(
        _pop([_pe("Q", "Delays", 10.0 + i) for i in range(9)])
    )
    # A disrupted route with no active alert keys no cell — climatology abstains.
    assert resolve_climatology_cell(cells, "Q", None) is None


def test_unseen_route_alert_falls_back_then_system() -> None:
    eps = [_pe("Q", "Delays", 10 + i) for i in range(MIN_CELL_SAMPLES)]
    cells = build_climatology_cells(_pop(eps))
    # Route never seen with Delays -> alert_type pool (Delays cleared MIN).
    at = resolve_climatology_cell(cells, "ZZ", "Delays")
    assert at is not None
    assert at["level"] == "alert_type"
    # An alert_type never seen at all -> system pool.
    sysc = resolve_climatology_cell(cells, "ZZ", "Weather")
    assert sysc is not None
    assert sysc["level"] == "system"


def test_alert_type_none_episodes_only_reach_the_system_pool() -> None:
    # An episode whose onset carried no prediction primary_alert_type still
    # counts system-wide but keys no route/alert_type cell.
    eps = [_pe("Q", None, 15.0), _pe("Q", "Delays", 25.0)]
    cells = build_climatology_cells(_pop(eps))
    assert cells["system"]["n"] == 2
    # The None-alert episode keys no (route, alert_type) cell but is counted in
    # the system pool (n=2). The lone (Q, Delays) episode is thin, so its
    # resolved cell pools to system too.
    assert cells["cells"]["Q"]["Delays"]["level"] == "system"
    assert "by_alert_type" in cells
    assert cells["by_alert_type"] == {}


def test_each_cell_records_its_window() -> None:
    cells = build_climatology_cells(
        _pop([_pe("Q", "Delays", 10.0 + i) for i in range(9)])
    )
    cell = resolve_climatology_cell(cells, "Q", "Delays")
    assert cell is not None
    assert cell["window_start"] == WS
    assert cell["window_end"] == WE


# --- conditional serve -------------------------------------------------------


def test_conditional_quantiles_indeterminate_when_outlived() -> None:
    samples = [5.0, 10.0, 15.0, 20.0, 25.0]
    # Only {20,25} exceed 18 -> fewer than min_samples=5 -> None.
    assert conditional_remaining_quantiles(samples, 18.0, min_samples=5) is None
    # Everything survives elapsed 0.
    rq = conditional_remaining_quantiles(samples, 0.0, min_samples=5)
    assert rq is not None
    assert rq.n == 5
    assert rq.p50 == 15.0


def test_climatology_recovery_serves_then_abstains_as_a_disruption_persists() -> None:
    pop = _pop([_pe("Q", "Delays", 5.0 * (i + 1)) for i in range(10)])  # 5..50 min
    cells = build_climatology_cells(pop)
    at_onset = climatology_recovery(cells, "Q", "Delays", 0.0)
    assert at_onset is not None
    assert not at_onset.indeterminate
    assert at_onset.minutes is not None
    assert at_onset.minutes > 0
    # Once the disruption outlives all but a few samples, abstain (indeterminate).
    outlived = climatology_recovery(cells, "Q", "Delays", 45.0)
    assert outlived is not None
    assert outlived.indeterminate
    assert outlived.minutes is None
    assert outlived.n == 10  # still records which cell was consulted


def test_climatology_recovery_none_without_alert_or_cell() -> None:
    cells = build_climatology_cells(
        _pop([_pe("Q", "Delays", 10.0 + i) for i in range(9)])
    )
    assert climatology_recovery(cells, "Q", None, 0.0) is None
    assert (
        climatology_recovery(
            {"min_samples": 5, "cells": {}, "by_alert_type": {}}, "Q", "Delays", 0.0
        )
        is None
    )


# --- shared population builder (the one review.py also grades against) --------


def _obs(route: str, tick: int, types: tuple[str, ...]) -> Any:
    from training.load import TickObservation

    return TickObservation(
        route_id=route,
        tick=tick,
        observation=Observation(
            alert_count=len(types), severity_sum=0, has_suspended_alert=False
        ),
        disruptive_types=types,
    )


def test_build_recovery_population_keeps_uncensored_nonstanding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start, end = date(2026, 1, 10), date(2026, 1, 11)
    ws, we = rb.window_bounds(start, end)
    first = (ws // TICK_SECONDS) * TICK_SECONDS
    last = (we // TICK_SECONDS) * TICK_SECONDS

    # A clean 3-tick severe incident in the interior -> kept (15 min).
    o = first + 100 * TICK_SECONDS
    obs = [_obs("1", o + k * TICK_SECONDS, ("Severe Delays",)) for k in range(3)]
    # A left-censored run (active at the very first grid tick) -> dropped.
    obs += [_obs("2", first + k * TICK_SECONDS, ("Severe Delays",)) for k in range(3)]
    # A right-censored run (active at the last grid tick) -> dropped.
    obs += [_obs("3", last - k * TICK_SECONDS, ("Severe Delays",)) for k in range(3)]
    # A sub-floor-only run (Delays, tier 1) never becomes severe truth -> no episode.
    obs += [_obs("4", o + k * TICK_SECONDS, ("Delays",)) for k in range(3)]

    preds = [
        SimpleNamespace(route="1", ts=o, primary_alert_type="Delays"),  # onset primary
        SimpleNamespace(
            route="1", ts=o + TICK_SECONDS, primary_alert_type="Severe Delays"
        ),
    ]

    def _fake_load_predictions(*_a: object, **_k: object) -> list[Any]:
        return preds

    def _fake_load_truth(*_a: object, **_k: object) -> list[Any]:
        return obs

    monkeypatch.setattr(rb, "load_predictions", _fake_load_predictions)
    monkeypatch.setattr(rb, "load_truth_observations", _fake_load_truth)

    pop = rb.build_recovery_population(cast(Any, None), "bucket", start, end)
    assert pop.window_start == ws
    assert pop.window_end == we
    routes = {pe.episode.route for pe in pop.episodes}
    assert routes == {"1"}  # only the interior severe incident survives
    kept = pop.episodes[0]
    assert kept.episode.duration_sec == 3 * TICK_SECONDS
    # alert_type is the Worker's primary_alert_type at the ONSET tick.
    assert kept.alert_type == "Delays"


# --- the transactional writer ------------------------------------------------


class _RecordingClient:
    def __init__(self) -> None:
        self.puts: list[str] = []

    def put_object(self, **kwargs: Any) -> None:
        self.puts.append(kwargs["Key"])


def test_write_recovery_baseline_versioned_before_pointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from training import publish_params as pp

    pop = _pop([_pe("Q", "Delays", 10.0 + i) for i in range(MIN_CELL_SAMPLES)])

    def _fake_pop(*_a: object, **_k: object) -> RecoveryPopulation:
        return pop

    monkeypatch.setattr(pp, "build_recovery_population", _fake_pop)
    client = _RecordingClient()
    trained_at = 1788_000_000

    n = pp.write_recovery_baseline(
        cast(Any, client), "bucket", date(2026, 1, 1), date(2026, 2, 4), trained_at
    )

    assert n == 1  # one observed (route, alert_type) cell
    # Standalone path (pending=None) writes BOTH; the immutable versioned
    # snapshot must be durable before the live pointer is flipped.
    assert client.puts == [
        f"{pp.VERSIONED_RECOVERY_BASELINE_PREFIX}v{trained_at}.json",
        pp.RECOVERY_BASELINE_KEY,
    ]


def test_write_recovery_baseline_defers_pointer_when_transactional(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from training import publish_params as pp

    pop = _pop([_pe("Q", "Delays", 10.0 + i) for i in range(MIN_CELL_SAMPLES)])

    def _fake_pop(*_a: object, **_k: object) -> RecoveryPopulation:
        return pop

    monkeypatch.setattr(pp, "build_recovery_population", _fake_pop)
    client = _RecordingClient()
    pending: list[pp.DeferredPointer] = []
    trained_at = 1788_000_001

    pp.write_recovery_baseline(
        cast(Any, client),
        "bucket",
        date(2026, 1, 1),
        date(2026, 2, 4),
        trained_at,
        pending=pending,
    )

    # Only the versioned snapshot is written now; the live pointer is deferred
    # for the transactional flush at the end of the run.
    assert client.puts == [f"{pp.VERSIONED_RECOVERY_BASELINE_PREFIX}v{trained_at}.json"]
    assert [p.key for p in pending] == [pp.RECOVERY_BASELINE_KEY]
