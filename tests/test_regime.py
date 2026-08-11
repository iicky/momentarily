"""Regime clock: debounce, back-dating, abstention, eviction, and the
cross-language parity pin."""

from __future__ import annotations

import json
from pathlib import Path

from training.regime import (
    DEBOUNCE_TICKS,
    MAX_IDLE_SEC,
    RegimeChange,
    RegimeEntry,
    advance_regimes,
    replay_regimes,
)

TICK = 300
T0 = 1_700_000_000


def t(i: int) -> int:
    return T0 + i * TICK


def test_cold_start_opens_a_regime_without_debounce() -> None:
    entries, changes = advance_regimes(None, {"A": "disrupted"}, t(0))
    assert changes == []
    assert entries["A"] == RegimeEntry(
        state="disrupted", entered_at=t(0), last_seen_at=t(0)
    )


# The debounce mechanism is exercised at debounce_ticks=2 explicitly. The
# production default is 1 (see DEBOUNCE_TICKS) — these pin the machinery, not
# the shipped setting.
D2 = {"debounce_ticks": 2}


def test_single_tick_blip_does_not_commit() -> None:
    entries, _ = advance_regimes(None, {"A": "normal"}, t(0), **D2)
    entries, changes = advance_regimes(entries, {"A": "disrupted"}, t(1), **D2)
    assert changes == []
    assert entries["A"].state == "normal"
    assert entries["A"].entered_at == t(0)
    assert entries["A"].pending == "disrupted"


def test_commit_back_dates_to_the_first_tick_of_the_run() -> None:
    """The regime starts when the evidence started, not when it convinced us.
    A curve fitted on the later tick would understate every dwell by the
    debounce."""
    entries, _ = advance_regimes(None, {"A": "normal"}, t(0), **D2)
    entries, _ = advance_regimes(entries, {"A": "disrupted"}, t(1), **D2)
    entries, changes = advance_regimes(entries, {"A": "disrupted"}, t(2), **D2)
    assert entries["A"].state == "disrupted"
    assert entries["A"].entered_at == t(1)
    assert [(c.prev_state, c.new_state, c.exited_at, c.dwell_sec) for c in changes] == [
        ("normal", "disrupted", t(1), TICK)
    ]


def test_interrupted_run_restarts_the_debounce() -> None:
    entries, _ = advance_regimes(None, {"A": "normal"}, t(0), **D2)
    entries, _ = advance_regimes(entries, {"A": "disrupted"}, t(1), **D2)
    entries, _ = advance_regimes(entries, {"A": "normal"}, t(2), **D2)
    entries, changes = advance_regimes(entries, {"A": "disrupted"}, t(3), **D2)
    assert changes == []
    assert entries["A"].state == "normal"
    assert entries["A"].pending_run == 1


def test_abstention_resets_the_run_but_holds_the_regime_open() -> None:
    """No reading is not a reading of change."""
    entries, _ = advance_regimes(None, {"A": "normal"}, t(0), **D2)
    entries, _ = advance_regimes(entries, {"A": "disrupted"}, t(1), **D2)
    entries, changes = advance_regimes(entries, {}, t(2), **D2)
    assert changes == []
    assert entries["A"].state == "normal"
    assert entries["A"].entered_at == t(0)
    assert entries["A"].pending is None
    # The next disrupted tick is run 1 again, so it takes two more to commit.
    entries, changes = advance_regimes(entries, {"A": "disrupted"}, t(3), **D2)
    assert changes == []
    entries, changes = advance_regimes(entries, {"A": "disrupted"}, t(4), **D2)
    assert [c.exited_at for c in changes] == [t(3)]


def test_abstention_holds_the_regime_open_at_the_shipped_default() -> None:
    """The default commits on the first disagreeing call, but an abstention is
    not a disagreeing call — it must still not end the regime."""
    entries, _ = advance_regimes(None, {"A": "normal"}, t(0))
    entries, changes = advance_regimes(entries, {}, t(1))
    assert changes == []
    assert entries["A"].state == "normal"
    assert entries["A"].entered_at == t(0)


def test_cell_evicts_after_max_idle_and_reopens_as_new() -> None:
    entries, _ = advance_regimes(None, {"A": "normal"}, t(0))
    entries, changes = advance_regimes(entries, {}, t(0) + MAX_IDLE_SEC + 1)
    assert entries == {}
    assert changes == []


def test_regime_survives_an_abstention_shorter_than_max_idle() -> None:
    entries, _ = advance_regimes(None, {"A": "normal"}, t(0))
    entries, _ = advance_regimes(entries, {}, t(0) + MAX_IDLE_SEC - 1)
    assert entries["A"].entered_at == t(0)


def test_debounce_of_one_commits_immediately() -> None:
    entries, _ = advance_regimes(None, {"A": "normal"}, t(0), debounce_ticks=1)
    entries, changes = advance_regimes(
        entries, {"A": "disrupted"}, t(1), debounce_ticks=1
    )
    assert entries["A"].state == "disrupted"
    assert entries["A"].entered_at == t(1)
    assert [c.dwell_sec for c in changes] == [TICK]


def test_changes_are_ordered_by_key() -> None:
    """Both languages must emit the same list; sorted keys is the guarantee."""
    prev = {
        k: RegimeEntry(state="normal", entered_at=t(0), last_seen_at=t(0))
        for k in ("C", "A", "B")
    }
    _, changes = advance_regimes(
        prev, dict.fromkeys(("C", "A", "B"), "disrupted"), t(1), debounce_ticks=1
    )
    assert [c.key for c in changes] == ["A", "B", "C"]


def test_clock_is_key_agnostic_across_route_and_segment_scope() -> None:
    """The whole point of the abstraction: a segment cell clocks identically to
    a route."""
    route, segment = "A", "Q|north|Q05N"
    ticks = [
        (t(0), {route: "normal", segment: "normal"}),
        (t(1), {route: "disrupted", segment: "disrupted"}),
        (t(2), {route: "disrupted", segment: "disrupted"}),
    ]
    entries, changes = replay_regimes(ticks)
    assert entries[route].entered_at == entries[segment].entered_at == t(1)
    assert {c.key for c in changes} == {route, segment}
    assert {c.dwell_sec for c in changes} == {TICK}


def test_replay_matches_tick_by_tick_folding() -> None:
    ticks = [
        (t(0), {"A": "normal"}),
        (t(1), {"A": "disrupted"}),
        (t(2), {"A": "disrupted"}),
        (t(3), {"A": "normal"}),
        (t(4), {"A": "normal"}),
    ]
    replayed_entries, replayed_changes = replay_regimes(ticks)

    entries: dict[str, RegimeEntry] = {}
    changes: list[RegimeChange] = []
    for observed_at, calls in ticks:
        entries, tick_changes = advance_regimes(entries, calls, observed_at)
        changes.extend(tick_changes)
    assert replayed_entries == entries
    assert replayed_changes == changes


def test_replay_sorts_ticks_out_of_order() -> None:
    ordered = replay_regimes(
        [
            (t(0), {"A": "normal"}),
            (t(1), {"A": "disrupted"}),
            (t(2), {"A": "disrupted"}),
        ]
    )
    shuffled = replay_regimes(
        [
            (t(2), {"A": "disrupted"}),
            (t(0), {"A": "normal"}),
            (t(1), {"A": "disrupted"}),
        ]
    )
    assert ordered == shuffled


def test_parity_fixture_matches_this_implementation() -> None:
    """The TS port replays the same fixture in worker/test/regime_parity.test.ts.
    Regenerate with: uv run python -m scripts.gen_regime_parity_fixture"""
    path = Path(__file__).parent / "fixtures" / "parity_regime.json"
    fixture = json.loads(path.read_text())
    entries: dict[str, RegimeEntry] = {}
    for step in fixture["steps"]:
        entries, changes = advance_regimes(
            entries,
            step["observed"],
            step["observed_at"],
            debounce_ticks=fixture["debounce_ticks"],
            max_idle_sec=fixture["max_idle_sec"],
        )
        assert {k: vars(v) for k, v in sorted(entries.items())} == step["entries"]
        assert [vars(c) for c in changes] == step["changes"]
    assert {k: vars(v) for k, v in sorted(entries.items())} == fixture["final_entries"]


def test_shipped_debounce_commits_on_the_first_change() -> None:
    """Pinned because the trainer and the Worker must agree on it without
    passing it around, and because raising it to 2 measurably erased 76% of the
    route episode population."""
    assert DEBOUNCE_TICKS == 1
    entries, _ = advance_regimes(None, {"A": "normal"}, t(0))
    entries, changes = advance_regimes(entries, {"A": "disrupted"}, t(1))
    assert entries["A"].state == "disrupted"
    assert entries["A"].entered_at == t(1)
    assert [c.dwell_sec for c in changes] == [TICK]
