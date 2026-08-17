"""Grading the segment deviation score against planned work (segment_grade.py).

Synthetic windows and traversals against a synthetic timetable. Each case pins
one rule about what enters an arm, what the baseline is allowed to see, when a
window abstains, and what the summary is allowed to pool.
"""

from __future__ import annotations

import zipfile
from datetime import date, datetime, time
from math import comb
from zoneinfo import ZoneInfo

from training.gtfs_static import HopKey, Timetable, timetable
from training.planned_work import Window
from training.segment_grade import (
    GRADED,
    NO_AFFECTED_BASELINE,
    THIN_AFFECTED,
    baseline_outside,
    grade_window,
    sign_test,
    summarise,
)
from training.trace import EXACT, Traversal
from training.traversal import TraversalCell

from .conftest import make_gtfs_zip

ET = ZoneInfo("America/New_York")
WEEKDAY = date(2026, 8, 12)
TRIP = "072000_A..S01R"
AT = int(datetime.combine(WEEKDAY, time(12, 0), tzinfo=ET).timestamp())

# A1S -> A2S -> A3S -> A4S. The window below names A2, so A1S->A2S and
# A2S->A3S are boundary hops and A3S->A4S is untouched.
BOUNDARY = ("A", "south", "A1S", "A2S")
UNTOUCHED = ("A", "south", "A3S", "A4S")


def _feed() -> zipfile.ZipFile:
    return make_gtfs_zip(
        [f"A,{TRIP},Weekday,X,1,A..S01R"],
        [
            f"{TRIP},A1S,12:00:00,12:00:00,1",
            f"{TRIP},A2S,12:01:30,12:01:30,2",
            f"{TRIP},A3S,12:03:00,12:03:00,3",
            f"{TRIP},A4S,12:04:30,12:04:30,4",
        ],
    )


def _timetable() -> Timetable:
    return timetable(_feed())


def _window(start: int = AT, end: int = AT + 3600) -> Window:
    return Window(
        alert_type="Planned - Part Suspended",
        routes=frozenset({"A"}),
        stops=frozenset({"A2"}),
        start=start,
        end=end,
    )


def _t(key: tuple[str, str, str, str], seconds: int, at: int) -> Traversal:
    route, direction, frm, to = key
    return Traversal(
        trip_id=TRIP,
        route_id=route,
        direction=direction,
        from_stop=frm,
        to_stop=to,
        at=at,
        seconds=seconds,
        moving_seconds=None,
        n_hops=1,
        censoring=EXACT,
    )


def _cells(median: float = 90.0) -> dict[tuple[str, str, str, str], TraversalCell]:
    cell = TraversalCell(
        n=50, median_sec=median, p90_sec=median * 1.5, scheduled_sec=90
    )
    return {BOUNDARY: cell, UNTOUCHED: cell}


def test_the_baseline_never_sees_movement_inside_announced_work():
    """A cell fitted through the closure would learn the disrupted times as
    normal and the deviation would read 1.0 by construction. Traversals inside
    any announced window are withheld and counted."""
    window = _window()
    inside = [_t(BOUNDARY, 900, AT + 60 * i) for i in range(30)]
    outside = [_t(BOUNDARY, 90, AT - 7200 + 60 * i) for i in range(30)]
    cells, withheld = baseline_outside(inside + outside, _timetable(), [window])
    assert withheld == 30
    assert cells[BOUNDARY].median_sec < 200  # the 900s crawls never reached the fit


def test_the_blackout_covers_every_announced_window_not_just_the_graded_one():
    """A cell contaminated by a DIFFERENT closure carries a disrupted level into
    the baseline the graded window is measured against."""
    graded = _window()
    other = _window(start=AT - 7200, end=AT - 3600)
    during_other = [_t(BOUNDARY, 900, AT - 7200 + 60 * i) for i in range(30)]
    clean = [_t(BOUNDARY, 90, AT - 20000 + 60 * i) for i in range(30)]
    _cells_out, withheld = baseline_outside(
        during_other + clean, _timetable(), [graded, other]
    )
    assert withheld == 30


def test_a_traversal_that_runs_into_the_window_is_kept_out_of_the_baseline():
    """A hop has DURATION. One that began before the work opened but arrived
    after it did is partly disrupted movement, and it sits nearest the work, so
    admitting it drags the fitted normal toward the disrupted level and makes
    the deviation read closer to 1.0 — biased against detection, the direction
    that quietly passes a broken model. A start-time test would keep it."""
    window = _window()
    crossing = [_t(BOUNDARY, 1200, window.start - 300) for _ in range(30)]
    clean = [_t(BOUNDARY, 90, window.start - 20000 + 60 * i) for i in range(30)]
    cells, withheld = baseline_outside(crossing + clean, _timetable(), [window])
    assert withheld == 30
    assert cells[BOUNDARY].median_sec < 200


def test_boundary_hops_go_to_the_affected_arm_and_untouched_hops_to_the_control():
    """The window names A2. A hop with exactly one endpoint named keeps running
    beside the work and is the thing being timed; a hop that never touches the
    named stops is the control. Both are observed inside the window, so time of
    day cancels without a matched control period."""
    window = _window()
    rows = [_t(BOUNDARY, 180, AT + 60 * i) for i in range(12)]
    rows += [_t(UNTOUCHED, 90, AT + 60 * i) for i in range(12)]
    got = grade_window(window, rows, _cells())
    assert got.state == GRADED
    assert (got.n_affected, got.n_control) == (12, 12)
    # Affected ran 2x its 90s normal, control ran exactly at it.
    assert got.median_affected == 2.0
    assert got.median_control == 1.0
    assert got.auc == 1.0


def test_a_hop_with_no_fitted_cell_abstains_rather_than_defaulting():
    """The segment cannot speak for itself yet. That is an abstention, not a
    reading of 1.0 that would drag the arm toward no-effect."""
    window = _window()
    rows = [_t(BOUNDARY, 180, AT + 60 * i) for i in range(12)]
    rows += [_t(UNTOUCHED, 90, AT + 60 * i) for i in range(12)]
    only_control: dict[HopKey, TraversalCell] = {UNTOUCHED: _cells()[UNTOUCHED]}
    got = grade_window(window, rows, only_control)
    assert got.state == NO_AFFECTED_BASELINE
    assert got.n_affected == 0
    assert got.auc is None


def test_a_thin_arm_abstains_and_says_which_arm_was_thin():
    """An AUC over three hops is not a weak result, it is not a result."""
    window = _window()
    rows = [_t(BOUNDARY, 180, AT + 60 * i) for i in range(3)]
    rows += [_t(UNTOUCHED, 90, AT + 60 * i) for i in range(40)]
    got = grade_window(window, rows, _cells())
    assert got.state == THIN_AFFECTED
    assert got.auc is None


def test_movement_outside_the_window_is_not_graded():
    """The claim is about what happened during the announced hours.

    `Window.contains` is inclusive of `end`, so the sweep starts one step past
    it rather than on it.
    """
    window = _window()
    after = window.end + 60
    rows = [_t(BOUNDARY, 180, after + 60 * i) for i in range(20)]
    rows += [_t(UNTOUCHED, 90, after + 60 * i) for i in range(20)]
    before = window.start - 60
    rows += [_t(BOUNDARY, 180, before - 60 * i) for i in range(20)]
    got = grade_window(window, rows, _cells())
    assert (got.n_affected, got.n_control) == (0, 0)


def test_the_summary_counts_windows_not_hops():
    """One closure contributing thousands of traversals is one piece of
    evidence. Pooling hops across windows would re-inflate a handful of
    closures into thousands of observations."""
    window = _window()
    strong = grade_window(
        window,
        [_t(BOUNDARY, 180, AT + 60 * i) for i in range(20)]
        + [_t(UNTOUCHED, 90, AT + 60 * i) for i in range(2000)],
        _cells(),
    )
    weak = grade_window(
        window,
        [_t(BOUNDARY, 45, AT + 60 * i) for i in range(20)]
        + [_t(UNTOUCHED, 90, AT + 60 * i) for i in range(20)],
        _cells(),
    )
    report = summarise([strong, weak])
    assert report.n_graded == 2
    assert report.windows_above_half == 1
    assert report.sign_test_p == 1.0


def test_sign_test_matches_the_exact_binomial():
    """The summary statistic across windows, checked against a direct
    enumeration rather than trusted."""

    def brute(k: int, n: int) -> float:
        d = abs(k - n / 2)
        return sum(comb(n, i) for i in range(n + 1) if abs(i - n / 2) >= d) / 2**n

    for k, n in [(13, 13), (0, 13), (7, 13), (10, 10), (9, 10), (3, 4)]:
        got = sign_test(k, n)
        assert got is not None
        assert abs(got - min(1.0, brute(k, n))) < 1e-12
    assert sign_test(0, 0) is None
