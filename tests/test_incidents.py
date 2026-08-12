"""Incident clustering, identity tracking, and duration fitting
(training/incidents.py). Hermetic — no R2/network; the archive-touching
pieces live behind incidents.main() and are never exercised here."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from datetime import UTC, date, datetime, timedelta

from training.hierarchical import PooledCell
from training.incidents import (
    DEFAULT_MAX_GAP,
    IncidentState,
    SegmentKey,
    _week_windows,  # pyright: ignore[reportPrivateUsage]
    advance_incidents,
    cluster_disrupted,
    fit_incident_duration,
    measure_premise,
    open_incident_regimes,
    parse_topology,
    path_incident_durations,
    premise_report,
    published_baseline_cells,
    replay_incidents,
    route_of,
    segment_ticks_with_baseline,
    topology_from_successors,
)
from training.movement_backfill import segment_ticks_from_vehicle_bodies
from training.regime import RegimeChange

TICK = 300
T0 = 1_700_000_000


def t(i: int) -> int:
    return T0 + i * TICK


# A linear F-south chain A0S -> B0S -> C0S -> D0S -> E0S, plus an unrelated
# Q-north segment — the same adjacency-doc shape state/segment_params.json
# publishes once retrained on the static GTFS feed.
_ADJACENCY = {
    "F|south|A0S": {"to": "B0S", "successors": [{"to": "B0S", "n_trips": 10}]},
    "F|south|B0S": {"to": "C0S", "successors": [{"to": "C0S", "n_trips": 10}]},
    "F|south|C0S": {"to": "D0S", "successors": [{"to": "D0S", "n_trips": 10}]},
    "F|south|D0S": {"to": "E0S", "successors": [{"to": "E0S", "n_trips": 10}]},
    "Q|north|X0N": {"to": "Y0N", "successors": [{"to": "Y0N", "n_trips": 5}]},
}
TOPOLOGY = parse_topology(_ADJACENCY)

A, B, C, D = (
    "F|south|A0S",
    "F|south|B0S",
    "F|south|C0S",
    "F|south|D0S",
)
X = "Q|north|X0N"


def _change(seq: int, entered: int, exited: int) -> RegimeChange:
    return RegimeChange(
        key=f"F|south#{seq}",
        prev_state="active",
        new_state="ended",
        entered_at=entered,
        exited_at=exited,
        dwell_sec=exited - entered,
    )


# --- Topology parsing ---


def test_parse_topology_uses_full_successor_list_when_present():
    doc = {
        "F|south|A0S": {
            "to": "B0S",
            "successors": [{"to": "B0S", "n_trips": 8}, {"to": "C0S", "n_trips": 2}],
        }
    }
    graph = parse_topology(doc)
    assert graph["F|south|A0S"] == ["F|south|B0S", "F|south|C0S"]


def test_parse_topology_falls_back_to_dominant_to_without_successors():
    """The observed-adjacency shape state/segment_params.json still publishes
    when the static GTFS fetch fails — no `successors` key at all."""
    doc = {"F|south|A0S": {"to": "B0S", "share": 0.7, "n": 40}}
    assert parse_topology(doc) == {"F|south|A0S": ["F|south|B0S"]}


def test_topology_from_successors_matches_parse_topology_shape():
    succ = {("F", "south", "A0S"): [("B0S", 8), ("C0S", 2)]}
    assert topology_from_successors(succ) == {
        "F|south|A0S": ["F|south|B0S", "F|south|C0S"]
    }


def test_topology_from_successors_skips_dead_ends():
    assert topology_from_successors({("F", "south", "E0S"): []}) == {}


def test_route_of_splits_on_first_pipe():
    assert route_of("F|south|A0S") == "F"


# --- Clustering ---


def test_two_adjacent_disrupted_segments_cluster_into_one_incident():
    assert cluster_disrupted({A, B}, TOPOLOGY) == [frozenset({A, B})]


def test_disrupted_segments_on_different_routes_do_not_merge():
    clusters = cluster_disrupted({A, X}, TOPOLOGY)
    assert set(clusters) == {frozenset({A}), frozenset({X})}


def test_gap_boundary_zero_keeps_incidents_separate():
    """A and C are one hop apart through B, which is NOT disrupted. Default
    max_gap=0 keeps them as two incidents: B itself reads normal, i.e. a
    train IS getting through there, so it's evidence against one blockage
    spanning both sides."""
    clusters = cluster_disrupted({A, C}, TOPOLOGY, max_gap=0)
    assert set(clusters) == {frozenset({A}), frozenset({C})}


def test_gap_boundary_one_bridges_a_single_healthy_segment():
    clusters = cluster_disrupted({A, C}, TOPOLOGY, max_gap=1)
    assert clusters == [frozenset({A, C})]
    assert B not in clusters[0]  # the bridge segment never joins the incident


def test_default_max_gap_is_zero():
    assert DEFAULT_MAX_GAP == 0


def test_no_disrupted_segments_produces_no_clusters():
    assert cluster_disrupted([], TOPOLOGY) == []


def test_unknown_segment_clusters_alone():
    """A segment absent from the topology (no static adjacency) has no
    neighbors — it can't merge with anything."""
    assert set(cluster_disrupted({"F|south|ZZZ", A}, TOPOLOGY)) == {
        frozenset({"F|south|ZZZ"}),
        frozenset({A}),
    }


# --- Identity tracking across ticks ---


def test_cold_start_opens_an_incident_without_debounce():
    state, changes = advance_incidents(None, {A, B}, TOPOLOGY, t(0))
    assert changes == []
    (incident_id,) = state.regimes.keys()
    assert state.regimes[incident_id].entered_at == t(0)
    assert state.footprints[incident_id] == frozenset({A, B})


def test_growing_and_shrinking_footprint_keeps_identity():
    state, changes0 = advance_incidents(None, {A, B}, TOPOLOGY, t(0))
    (incident_id,) = state.regimes.keys()
    assert changes0 == []

    state, changes1 = advance_incidents(state, {A, B, C}, TOPOLOGY, t(1))
    assert changes1 == []  # same call ('active'), no transition at all
    assert set(state.regimes) == {incident_id}
    assert state.regimes[incident_id].entered_at == t(0)  # unchanged
    assert state.footprints[incident_id] == frozenset({A, B, C})

    state, changes2 = advance_incidents(state, {B, C}, TOPOLOGY, t(2))
    assert changes2 == []
    assert set(state.regimes) == {incident_id}
    assert state.regimes[incident_id].entered_at == t(0)
    assert state.footprints[incident_id] == frozenset({B, C})


def test_default_debounce_closes_an_incident_on_the_first_ended_tick():
    """DEBOUNCE_TICKS now defaults to 1 (training/regime.py) — an incident
    closes as soon as its footprint stops matching, no second confirming
    tick required. The debounce mechanism itself still exists (see the
    explicit debounce_ticks=2 tests below); the default just doesn't use it."""
    state, _ = advance_incidents(None, {A, B}, TOPOLOGY, t(0))
    (incident_id,) = state.regimes.keys()

    state, changes = advance_incidents(state, set(), TOPOLOGY, t(1))
    assert [c.key for c in changes] == [incident_id]
    assert changes[0].new_state == "ended"
    assert changes[0].entered_at == t(0)
    assert changes[0].exited_at == t(1)
    assert changes[0].dwell_sec == t(1) - t(0)


def test_incident_closes_after_debounced_absence():
    """debounce_ticks passed explicitly: the mechanism is still available to
    a caller (and still pinned by tests/fixtures/parity_regime.json) even
    though the production default no longer uses it."""
    state, _ = advance_incidents(None, {A, B}, TOPOLOGY, t(0), debounce_ticks=2)
    (incident_id,) = state.regimes.keys()

    state, changes = advance_incidents(state, set(), TOPOLOGY, t(1), debounce_ticks=2)
    assert changes == []  # one 'ended' tick isn't debounced yet
    assert state.regimes[incident_id].state == "active"
    assert state.regimes[incident_id].pending == "ended"

    state, changes = advance_incidents(state, set(), TOPOLOGY, t(2), debounce_ticks=2)
    assert [c.key for c in changes] == [incident_id]
    assert changes[0].new_state == "ended"
    assert changes[0].entered_at == t(0)
    assert changes[0].exited_at == t(1)  # back-dated to the first 'ended' tick
    assert changes[0].dwell_sec == t(1) - t(0)


def test_reappearing_footprint_before_debounce_cancels_the_close():
    """At debounce_ticks=2, a one-tick gap doesn't end the incident — the
    same protection abstention gives a route/segment regime, reached here
    through an explicit re-match instead of silence."""
    state, _ = advance_incidents(None, {A, B}, TOPOLOGY, t(0), debounce_ticks=2)
    (incident_id,) = state.regimes.keys()

    state, changes = advance_incidents(state, set(), TOPOLOGY, t(1), debounce_ticks=2)
    assert changes == []
    state, changes = advance_incidents(state, {A, B}, TOPOLOGY, t(2), debounce_ticks=2)
    assert changes == []
    assert state.regimes[incident_id].state == "active"
    assert state.regimes[incident_id].pending is None
    assert state.regimes[incident_id].entered_at == t(0)


def test_two_incidents_merging_keeps_the_lower_id_and_ends_the_other():
    state, _ = advance_incidents(None, {A, D}, TOPOLOGY, t(0), debounce_ticks=2)
    winner, loser = sorted(state.regimes)
    assert state.footprints[winner] == frozenset({A})
    assert state.footprints[loser] == frozenset({D})

    state, changes = advance_incidents(
        state, {A, B, C, D}, TOPOLOGY, t(1), debounce_ticks=2
    )
    assert changes == []  # loser is pending-ended, nothing committed yet
    assert state.footprints[winner] == frozenset({A, B, C, D})
    assert state.regimes[winner].entered_at == t(0)
    assert state.regimes[loser].state == "active"
    assert state.regimes[loser].pending == "ended"

    state, changes = advance_incidents(
        state, {A, B, C, D}, TOPOLOGY, t(2), debounce_ticks=2
    )
    assert [c.key for c in changes] == [loser]
    assert changes[0].new_state == "ended"


def test_split_incident_keeps_id_on_one_piece_and_mints_a_fresh_id():
    state, _ = advance_incidents(None, {A, B, C, D}, TOPOLOGY, t(0))
    (original_id,) = state.regimes.keys()

    state, changes = advance_incidents(state, {A, D}, TOPOLOGY, t(1))
    assert changes == []
    assert state.footprints[original_id] == frozenset({A})
    (new_id,) = set(state.regimes) - {original_id}
    assert state.footprints[new_id] == frozenset({D})
    assert state.regimes[new_id].entered_at == t(1)  # genuinely new, not back-dated


def test_replay_matches_tick_by_tick_folding():
    ticks: list[tuple[int, Iterable[SegmentKey]]] = [
        (t(0), {A, B}),
        (t(1), {A, B, C}),
        (t(2), set()),
        (t(3), set()),
    ]
    folded_state: IncidentState | None = None
    folded_changes: list[RegimeChange] = []
    for observed_at, disrupted in ticks:
        folded_state, changes = advance_incidents(
            folded_state, disrupted, TOPOLOGY, observed_at
        )
        folded_changes.extend(changes)
    replayed_state, replayed_changes = replay_incidents(ticks, TOPOLOGY)
    assert replayed_changes == folded_changes
    assert folded_state is not None
    assert replayed_state.regimes == folded_state.regimes


def test_still_open_incident_at_window_end_is_censored_not_completed():
    ticks: list[tuple[int, Iterable[SegmentKey]]] = [
        (t(0), {A, B}),
        (t(1), {A, B}),
        (t(2), {A, B}),
    ]
    state, changes = replay_incidents(ticks, TOPOLOGY)
    assert all(c.new_state != "ended" for c in changes)  # never completed
    open_regimes = open_incident_regimes(state)
    assert open_regimes["F"] == ("active", t(0))


# --- Fitting duration at the incident level ---


def test_fit_incident_duration_pools_sparse_routes():
    changes = [_change(0, t(0), t(2)), _change(1, t(5), t(9))]
    cells = fit_incident_duration(changes)
    assert cells["F"]["n"] == 2
    assert cells["F"]["n_censored"] == 0


def test_fit_incident_duration_censors_a_still_open_incident():
    changes = [_change(0, t(0), t(2))]
    still_open = {"F": ("active", t(10))}
    cells = fit_incident_duration(changes, window_end=t(20), still_open=still_open)
    assert cells["F"]["n"] == 1
    assert cells["F"]["n_censored"] == 1


def test_fit_incident_duration_empty_input_is_empty_output():
    assert fit_incident_duration([]) == {}


# --- Path composition: incident-level, not per-segment ---


def _quantiles():
    return fit_incident_duration([_change(0, t(0), t(2)), _change(1, t(5), t(9))])["F"]


def test_path_incident_durations_collapses_one_incident_not_per_segment():
    """Two adjacent disrupted segments on the path share ONE incident — the
    result has ONE entry, not one per segment."""
    result = path_incident_durations([A, B, C], {A, B}, TOPOLOGY, {"F": _quantiles()})
    assert result == [_quantiles()]


def test_path_incident_durations_returns_two_for_two_separate_incidents():
    result = path_incident_durations(
        [A, B, C, D], {A, D}, TOPOLOGY, {"F": _quantiles()}, max_gap=0
    )
    assert len(result) == 2


def test_path_incident_durations_empty_when_nothing_disrupted():
    assert path_incident_durations([A, B], set(), TOPOLOGY, {"F": _quantiles()}) == []


def test_path_incident_durations_ignores_incidents_off_the_path():
    """An incident on a different route entirely mustn't leak into the path
    result just because SOME segment somewhere is disrupted."""
    result = path_incident_durations([A, B], {X}, TOPOLOGY, {"F": _quantiles()})
    assert result == []


# --- measure_premise ---


def test_measure_premise_reports_gap0_and_gap1_side_by_side():
    """A and C are one hop apart through B, which isn't disrupted: they
    cluster at gap1 but not gap0 — same boundary as
    test_gap_boundary_one_bridges_a_single_healthy_segment, checked through
    the aggregate stats instead of cluster_disrupted directly."""
    result = measure_premise([(t(0), [A, C])], TOPOLOGY)
    assert result["ticks_with_2plus_disrupted"] == 1
    assert result["ticks_with_adjacent_cluster_gap0"] == 0
    assert result["ticks_with_adjacent_cluster_gap1"] == 1
    assert result["share_ticks_adjacent_gap0"] == 0.0
    assert result["share_ticks_adjacent_gap1"] == 1.0
    assert result["mean_cluster_size_gap0"] == 1.0
    assert result["mean_cluster_size_gap1"] == 2.0
    assert result["median_cluster_size_gap0"] == 1
    assert result["median_cluster_size_gap1"] == 2
    assert result["share_disrupted_segments_in_multi_segment_incident_gap0"] == 0.0
    assert result["share_disrupted_segments_in_multi_segment_incident_gap1"] == 1.0


def test_measure_premise_ignores_ticks_with_fewer_than_two_disrupted():
    result = measure_premise([(t(0), [A]), (t(1), [])], TOPOLOGY)
    assert result["ticks_with_2plus_disrupted"] == 0


def test_measure_premise_empty_ticks_returns_none_for_every_share():
    result = measure_premise([], TOPOLOGY)
    assert result["ticks_with_2plus_disrupted"] == 0
    assert result["share_ticks_adjacent_gap0"] is None
    assert result["share_ticks_adjacent_gap1"] is None
    assert result["mean_cluster_size_gap0"] is None
    assert result["median_cluster_size_gap1"] is None
    assert result["share_disrupted_segments_in_multi_segment_incident_gap0"] is None


# --- published_baseline_cells ---


def test_published_baseline_cells_parses_the_state_segment_params_shape():
    doc = {"cells": {"F|south|A0S": {"p0": 0.85, "n": 120}}}
    cells = published_baseline_cells(doc)
    cell = cells[("F", "south", "A0S")]
    assert cell.p0 == 0.85
    assert cell.n == 120
    assert cell.source == "published"


def test_published_baseline_cells_skips_keys_without_three_parts():
    doc = {"cells": {"not-a-segment-key": {"p0": 0.5, "n": 1}}}
    assert published_baseline_cells(doc) == {}


def test_published_baseline_cells_empty_doc_is_empty():
    assert published_baseline_cells({}) == {}


# --- segment_ticks_with_baseline ---


def _segment_body(
    tick: int, transitions: dict[str, int], route: str = "F", direction: str = "south"
) -> dict[str, object]:
    return {
        "observed_at": tick,
        "rows": {route: {"by_direction": {direction: {"transitions": transitions}}}},
    }


def test_segment_ticks_with_baseline_uses_the_supplied_baseline_not_a_self_trained_one():
    """4 ticks of {"A>A": 2} accumulate to 8 matched trips, 0 advanced, by the
    window's last tick. At matched=8 the DISRUPTED_RATIO=0.5 posterior gate
    trips for ANY p0 (post = 8*p0/(8+8) = 0.5*p0 exactly), so the call comes
    down entirely to the significance test: (1-p0)**8 <= 0.05 needs
    p0 >= ~0.259.

    Self-training this single leaf on its own 8-count sample has no siblings
    to pool against, so partially_pool (hierarchical.py) falls back to its
    system default (mu=0.5, kappa=50): p0 = (50*0.5 + 0) / (50 + 8) = 25/58
    ~= 0.431, which clears 0.259 => 'disrupted'. A published baseline with a
    genuinely low p0=0.1 (a shuttle that's slow even when healthy) does NOT
    clear 0.259 => abstains (None, dropped from the tick entirely). Same
    bodies, same tick, opposite calls depending on which baseline is
    supplied — proving segment_ticks_with_baseline actually scores against
    the injected baseline, not one it quietly recomputes from `bodies`."""
    bodies = [_segment_body(t(i), {"A>A": 2}) for i in range(4)]

    self_ticks = segment_ticks_from_vehicle_bodies(bodies, window_ticks=4)
    last_tick = self_ticks[-1][0]
    assert dict(self_ticks)[last_tick]["F|south|A"] == "disrupted"

    published_low = {
        ("F", "south", "A"): PooledCell(
            p0=0.1, raw=0.1, n=100, alpha=5.0, beta=45.0, source="published"
        )
    }
    published_ticks = dict(
        segment_ticks_with_baseline(bodies, published_low, window_ticks=4)
    )
    assert "F|south|A" not in published_ticks.get(last_tick, {})


def test_segment_ticks_with_baseline_skips_leaves_missing_from_the_baseline():
    bodies = [_segment_body(t(i), {"A>B": 2}) for i in range(2)]
    ticks = segment_ticks_with_baseline(bodies, {}, window_ticks=2)
    assert ticks == []


# --- _week_windows / premise_report ---


def test_week_windows_bins_by_seven_days_with_a_short_final_bin():
    windows = _week_windows(date(2026, 1, 1), date(2026, 1, 10))
    assert windows == [
        (date(2026, 1, 1), date(2026, 1, 7)),
        (date(2026, 1, 8), date(2026, 1, 10)),
    ]


def test_week_windows_covers_the_53_day_archive_range_contiguously():
    start, end = date(2026, 6, 21), date(2026, 8, 12)
    windows = _week_windows(start, end)
    assert windows[0][0] == start
    assert windows[-1][1] == end
    for prev, cur in itertools.pairwise(windows):
        assert cur[0] == prev[1] + timedelta(days=1)
    assert all((b - a).days <= 6 for a, b in windows)
    assert sum((b - a).days + 1 for a, b in windows) == (end - start).days + 1


def _epoch(d: date, hour: int = 12) -> int:
    return int(datetime(d.year, d.month, d.day, hour, tzinfo=UTC).timestamp())


def test_premise_report_splits_ticks_into_weekly_bins_and_pools_overall():
    start, end = date(2026, 1, 1), date(2026, 1, 10)
    week1_tick = (_epoch(date(2026, 1, 2)), [A, B])  # adjacent at gap0
    week2_tick = (_epoch(date(2026, 1, 9)), [A, C])  # not adjacent at gap0
    report = premise_report([week1_tick, week2_tick], TOPOLOGY, start, end)

    assert report["overall"]["ticks_with_2plus_disrupted"] == 2
    assert report["overall"]["ticks_with_adjacent_cluster_gap0"] == 1

    assert len(report["weekly"]) == 2
    week1, week2 = report["weekly"]
    assert week1["start"] == "2026-01-01"
    assert week1["end"] == "2026-01-07"
    assert week1["ticks_with_2plus_disrupted"] == 1
    assert week1["ticks_with_adjacent_cluster_gap0"] == 1
    assert week2["start"] == "2026-01-08"
    assert week2["end"] == "2026-01-10"
    assert week2["ticks_with_2plus_disrupted"] == 1
    assert week2["ticks_with_adjacent_cluster_gap0"] == 0
