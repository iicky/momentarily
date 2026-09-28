"""Run splitting in the span-coverage tool (training/journey_coverage.py).

A trip's run must survive UTC midnight (8 pm in New York) as one run, and a
reused trip_id must not merge two runs. Synthetic traversals only.
"""

from __future__ import annotations

from datetime import UTC, datetime

from training.journey_coverage import split_runs
from training.trace import EXACT, TRIP_GAP_SECONDS, Traversal

MIDNIGHT = int(datetime(2026, 9, 26, tzinfo=UTC).timestamp())


def _hop(trip: str, at: int, frm: str, to: str, seconds: int = 90) -> Traversal:
    return Traversal(
        trip_id=trip,
        route_id="F",
        direction="S",
        from_stop=frm,
        to_stop=to,
        at=at,
        seconds=seconds,
        moving_seconds=None,
        n_hops=1,
        censoring=EXACT,
    )


def test_a_run_crossing_utc_midnight_stays_one_run() -> None:
    hops = [
        _hop("t", MIDNIGHT - 180, "F01S", "F02S"),
        _hop("t", MIDNIGHT - 90, "F02S", "F03S"),
        _hop("t", MIDNIGHT, "F03S", "F04S"),
        _hop("t", MIDNIGHT + 90, "F04S", "F05S"),
    ]
    runs = split_runs(hops)
    assert len(runs) == 1
    assert [h.from_stop for h in runs[0]] == ["F01S", "F02S", "F03S", "F04S"]


def test_a_reused_trip_id_after_a_gap_is_a_new_run() -> None:
    first = _hop("t", MIDNIGHT, "F01S", "F02S")
    # Starts exactly TRIP_GAP_SECONDS after `first` ended: still the same run.
    same = _hop("t", first.at + first.seconds + TRIP_GAP_SECONDS, "F02S", "F03S")
    # One second more than the gap after `same` ended: a different run.
    later = _hop("t", same.at + same.seconds + TRIP_GAP_SECONDS + 1, "F01S", "F02S")
    runs = split_runs([later, first, same])  # input order must not matter
    assert [[h.at for h in r] for r in runs] == [[first.at, same.at], [later.at]]
