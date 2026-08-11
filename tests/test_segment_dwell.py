"""Hierarchical partial-pooling behaviour for per-segment dwell curves."""

from __future__ import annotations

from training.eval import MovementTransitionRecord
from training.hierarchical import MIN_LEAF_N, MIN_PARENT_LEAVES
from training.segment_dwell import build_segment_dwell

START = 1_700_000_000
# Deterministic multiplicative spread around a centre — real dwells are
# heavy-tailed and never identical; a fixture built from one repeated
# duration collapses the fit and stops exercising anything real.
_SPREAD = (0.25, 0.5, 0.8, 1.0, 1.3, 2.0, 3.5, 6.0)


def _events(
    key: str,
    route: str,
    n: int,
    *,
    state: str = "disrupted",
    centre_sec: int = 1800,
) -> list[MovementTransitionRecord]:
    """n completed `state` -> normal dwells for one segment cell."""
    out: list[MovementTransitionRecord] = []
    ts = START
    for i in range(n):
        dwell = max(30, round(centre_sec * _SPREAD[i % len(_SPREAD)]))
        out.append(
            MovementTransitionRecord(
                ts=ts,
                scope="segment",
                key=key,
                route=route,
                prev_state=state,
                new_state="normal",
                regime_entered_at=ts - dwell,
                exited_at=ts,
                dwell_sec=dwell,
            )
        )
        ts += dwell + 60
    return out


def _hierarchy_fixture() -> list[MovementTransitionRecord]:
    """One coherent (route, segment) population exercising every level:

    - route F: 5 rich segments (>= MIN_LEAF_N events each) -> each votes for
      route/system and each gets its own standalone fit.
    - route Q: 1 thin segment + MIN_PARENT_LEAVES rich siblings in the same
      (route, direction) -> the thin one backs off to the route aggregate.
    - route R: 1 thin segment, no siblings at all -> backs off past its own
      (empty) route straight to the system aggregate, which route F/Q's rich
      leaves supply.
    """
    n_rich = MIN_LEAF_N + 4
    out: list[MovementTransitionRecord] = []
    for i in range(5):
        out += _events(f"F|south|RICH{i}", "F", n_rich, centre_sec=1800 + i * 100)
    for i in range(MIN_PARENT_LEAVES):
        out += _events(f"Q|south|RICH{i}", "Q", n_rich, centre_sec=900)
    out += _events("Q|south|THIN1", "Q", 2, centre_sec=900)
    out += _events("R|south|LONE1", "R", 2, centre_sec=3600)
    return out


def test_data_rich_leaf_gets_its_own_standalone_fit():
    cells, stats = build_segment_dwell(_hierarchy_fixture())
    n_rich = MIN_LEAF_N + 4
    cell = cells["F|south|RICH0"]["disrupted"]
    # cell_from_fit reports n as completed-events-only, matching this leaf's
    # own sample count exactly — proof it was not merged with anything else.
    assert cell["n"] == n_rich
    assert stats.n_cells_own >= 5 + MIN_PARENT_LEAVES  # F's 5 + Q's 4 rich siblings


def test_thin_leaf_with_rich_siblings_falls_back_to_its_route():
    cells, stats = build_segment_dwell(_hierarchy_fixture())
    cell = cells["Q|south|THIN1"]["disrupted"]
    # The route aggregate pools every Q segment's samples (4 * rich + 2 of its
    # own): far more than THIN1's 2 completed events alone, proof it borrowed
    # the route curve rather than fitting standalone.
    assert cell["n"] > 2
    assert stats.n_cells_route >= 1


def test_thin_leaf_on_a_thin_route_falls_back_to_system():
    cells, stats = build_segment_dwell(_hierarchy_fixture())
    cell = cells["R|south|LONE1"]["disrupted"]
    # No siblings on R at all: the aggregate can only have come from the
    # system-wide pool (F + Q's rich leaves), not from R's own 2 events.
    assert cell["n"] > 2
    assert stats.n_cells_system >= 1


def test_hierarchical_pooling_falls_back_segment_route_system_when_thin():
    """The end-to-end claim: a thin leaf's curve traces back to whichever
    level actually had support, never to a standalone fit of its own 2
    events, and the three levels are mutually exclusive per cell."""
    cells, stats = build_segment_dwell(_hierarchy_fixture())
    assert stats.n_cells_own > 0
    assert stats.n_cells_route > 0
    assert stats.n_cells_system > 0
    assert stats.n_cells_total == len(
        [1 for by_state in cells.values() for _ in by_state]
    )


def test_route_scope_records_are_ignored():
    route_scope = [
        MovementTransitionRecord(
            ts=START,
            scope="route",
            key="F",
            route="F",
            prev_state="disrupted",
            new_state="normal",
            regime_entered_at=START - 1800,
            exited_at=START,
            dwell_sec=1800,
        )
    ]
    cells, stats = build_segment_dwell(route_scope)
    assert cells == {}
    assert stats.n_cells_total == 0


def test_empty_input_returns_empty():
    cells, stats = build_segment_dwell([])
    assert cells == {}
    assert stats.n_cells_own == 0
    assert stats.n_cells_route == 0
    assert stats.n_cells_system == 0
    assert stats.n_cells_skipped == 0


def test_still_open_regime_censors_the_pooled_curve_instead_of_dropping_it():
    # Q|south|OPEN1 never completes a disrupted regime in this window — its
    # last transition instead ENTERS disrupted — so on its own it would
    # contribute nothing (dwell.dwell_samples_by_cell only infers a censored
    # observation for the state a cell's last transition entered). With
    # window_end past that entry, it must still show up as a right-censored
    # sample feeding whichever aggregate this thin leaf resolves to.
    transitions = [
        *_hierarchy_fixture(),
        MovementTransitionRecord(
            ts=START,
            scope="segment",
            key="Q|south|OPEN1",
            route="Q",
            prev_state="normal",
            new_state="disrupted",
            regime_entered_at=START,
            exited_at=START,
            dwell_sec=0,
        ),
    ]
    window_end = START + 7200
    cells, _stats = build_segment_dwell(transitions, window_end=window_end)
    cell = cells["Q|south|OPEN1"]["disrupted"]
    assert cell["n_censored"] >= 1
