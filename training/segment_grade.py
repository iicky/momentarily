"""Grade the segment deviation score against announced planned work.

WHAT IS BEING GRADED, AND WHY IT IS NOT WHAT planned_work GRADES. That module
grades the raw traversal DATA: do the named hops stop appearing, do the hops
beside them take longer. It validated itself over 2026-08-12..16 by ranking the
announced types monotonically in how much of the route they remove, with no
knowledge of that ordering. What has never been graded is the thing we would
actually SHIP — `traversal.deviation`, a live hop scored against its own
segment's fitted normal. A signal being present in the data does not mean the
model reads it, and every movement score graded in this repo so far has come
out at or below chance.

THE ANSWER KEY. Announced planned work is pre-registered: the stations and the
hours are published days ahead, so the hops that should slow are named BEFORE
any measurement. That is the property the alerts feed cannot otherwise offer,
and it is why this is the one label able to grade slowness at all — the
assigned_n label measures service being WITHDRAWN, and the trains that remain
under a withdrawal run to their booked times (measured 2026-08-13: 1.017x
against 1.000x).

THE UNIT IS THE ANNOUNCED WINDOW. Not the hop, and not the local-day row.
`planned_work` splits a period per local day because a Friday-to-Monday closure
is not one comparison for ITS purposes; those fragments are the same announced
work and are not independent evidence about whether a model detects it. Thirteen
graded rows over that window are far fewer announced windows, and treating the
rows as independent is the same error as counting a disruption's five-minute
ticks as independent draws — which cost three corrections on the label grade
before it was caught. So this module iterates UNSPLIT windows and emits exactly
one row each.

BOTH ARMS ARE OBSERVED INSIDE THE WINDOW. The affected arm is the boundary hops
(one endpoint named — the ones that keep running beside the work); the control
arm is the same routes' hops that the window does not touch at all. Because both
are measured over the same minutes, time of day cancels without needing a
matched control PERIOD, which is what starves `pattern_shift` of comparisons.

THE BASELINE MUST NOT SEE THE WORK. Cells are fitted from traversals outside
EVERY announced window on that route, not merely outside the one being graded.
A baseline that included the closure would learn the disrupted times as normal
and the deviation would read 1.0 by construction — the model would be graded
against itself and would pass no matter how bad it was.
"""

from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta

from training.gtfs_static import (
    HopKey,
    Timetable,
    fetch_gtfs_zip,
)
from training.gtfs_static import (
    timetable as parse_timetable,
)
from training.planned_work import (
    Window,
    fetch_alert_bodies,
    overlaps,
    windows_from_alerts,
)
from training.progress import auc_from
from training.trace import EXACT, Traversal, fetch_trace_bodies, traversals_from_trace
from training.traversal import TraversalCell, deviation, traversal_baseline

# Scored traversals each arm needs before a window is graded. Below it the
# window ABSTAINS and is counted under its own state -- an AUC over three hops
# is not a weak result, it is not a result.
MIN_ARM_HOPS = 10

GRADED = "graded"
NO_AFFECTED_BASELINE = "no_affected_baseline"
THIN_AFFECTED = "thin_affected"
THIN_CONTROL = "thin_control"


def _key(t: Traversal) -> HopKey:
    return (t.route_id, t.direction or "", t.from_stop, t.to_stop or "")


def _overlaps_work(t: Traversal, windows: Sequence[Window]) -> bool:
    """Whether a traversal's own span touches any announced window on its route.

    A HOP HAS DURATION, and a start-time test is not enough. A traversal that
    began at 21:25 and arrived at 21:45 spent most of itself inside a window
    that opened at 21:30, and it is partly disrupted movement. Those are also
    exactly the traversals nearest the work, so admitting them into a "normal"
    fit drags the normal toward the disrupted level and makes the deviation read
    closer to 1.0 — the bias runs AGAINST detection, which is the direction that
    quietly passes a broken model.
    """
    finish = t.at + t.seconds
    return any(
        w.covers_route(t.route_id)
        and finish >= w.start
        and (w.end == 0 or t.at <= w.end)
        for w in windows
    )


def baseline_outside(
    traversals: Sequence[Traversal],
    timetable: Timetable,
    windows: Sequence[Window],
) -> tuple[dict[HopKey, TraversalCell], int]:
    """Per-hop cells fitted only from movement outside every announced window.

    Returns the cells and how many traversals were withheld. The blackout spans
    all announced work on the route, not just the window under test: a cell
    contaminated by ANY closure carries a disrupted level into a normal
    baseline, and the deviation it produces is measured against the wrong
    normal in both directions. Overlap is judged on the traversal's whole span,
    not its start — see `_overlaps_work`.
    """
    clean: list[Traversal] = []
    withheld = 0
    for t in traversals:
        if _overlaps_work(t, windows):
            withheld += 1
            continue
        clean.append(t)
    cells, _stats = traversal_baseline(clean, timetable)
    return cells, withheld


def baseline_before(
    traversals: Sequence[Traversal],
    timetable: Timetable,
    windows: Sequence[Window],
    window: Window,
) -> tuple[dict[HopKey, TraversalCell], int]:
    """Cells fitted only from clean movement that FINISHED before `window` began.

    The difference from `baseline_outside` is causality, and it decides what the
    grade is a statement about. Fitting from the whole archive lets a cell be
    informed by traversals that had not happened yet when the window opened, so
    the result is retrospective: it measures whether the signal is IN the data.
    A model running live at the moment the work starts has only the past, and a
    segment whose cell is built entirely from later days simply has no cell yet.

    Both are worth reporting and they answer different questions, so neither is
    allowed to stand in for the other. This one is the deployable claim, and it
    is strictly harsher: fewer hops clear MIN_HOP_SAMPLES on a truncated
    history, so more windows abstain.
    """
    clean: list[Traversal] = []
    withheld = 0
    for t in traversals:
        if t.at + t.seconds > window.start:
            continue
        if _overlaps_work(t, windows):
            withheld += 1
            continue
        clean.append(t)
    cells, _stats = traversal_baseline(clean, timetable)
    return cells, withheld


@dataclass(frozen=True)
class WindowGrade:
    """One announced window, graded once."""

    alert_type: str
    routes: tuple[str, ...]
    start: int
    end: int
    state: str
    n_affected: int  # scored traversals on boundary hops, inside the window
    n_control: int
    n_affected_keys: int
    n_control_keys: int
    # P(a boundary hop deviates more than an untouched hop of the same routes,
    # over the same minutes). 0.5 is no signal; below is inverted.
    auc: float | None
    median_affected: float | None
    median_control: float | None


def grade_window(
    window: Window,
    traversals: Iterable[Traversal],
    cells: Mapping[HopKey, TraversalCell],
    *,
    min_arm: int = MIN_ARM_HOPS,
) -> WindowGrade:
    """Score one announced window: do its boundary hops deviate further from
    their own normal than the untouched hops of the same routes do, over the
    same minutes?

    A hop with no fitted cell is skipped rather than defaulted — the segment
    cannot speak for itself yet, which is an abstention, not a reading of 1.0.
    """
    affected: list[float] = []
    control: list[float] = []
    affected_keys: set[HopKey] = set()
    control_keys: set[HopKey] = set()
    for t in traversals:
        if t.censoring != EXACT or t.to_stop is None or t.n_hops != 1:
            continue
        if not window.covers_route(t.route_id) or not window.contains(t.at):
            continue
        key = _key(t)
        cell = cells.get(key)
        if cell is None:
            continue
        if window.at_boundary(key):
            affected.append(deviation(cell, t.seconds))
            affected_keys.add(key)
        elif not window.touches(key):
            control.append(deviation(cell, t.seconds))
            control_keys.add(key)

    if not affected_keys:
        state = NO_AFFECTED_BASELINE
    elif len(affected) < min_arm:
        state = THIN_AFFECTED
    elif len(control) < min_arm:
        state = THIN_CONTROL
    else:
        state = GRADED

    return WindowGrade(
        alert_type=window.alert_type,
        routes=tuple(sorted(window.routes)),
        start=window.start,
        end=window.end,
        state=state,
        n_affected=len(affected),
        n_control=len(control),
        n_affected_keys=len(affected_keys),
        n_control_keys=len(control_keys),
        auc=auc_from(affected, control) if state == GRADED else None,
        median_affected=statistics.median(affected) if affected else None,
        median_control=statistics.median(control) if control else None,
    )


def sign_test(wins: int, n: int) -> float | None:
    """Two-sided exact binomial p for `wins` of `n` windows landing above 0.5.

    The windows are the independent units, so the summary across them is a
    count, not a pooled AUC. Pooling every hop from every window back into one
    ranking would re-inflate a handful of closures into thousands of
    observations, which is exactly the mistake this module's unit rule exists
    to prevent.
    """
    if n == 0:
        return None
    coef = 1
    tail = 0.0
    total = float(2**n)
    for k in range(n + 1):
        if k > 0:
            coef = coef * (n - k + 1) // k
        if abs(k - n / 2) >= abs(wins - n / 2):
            tail += coef
    return min(1.0, tail / total)


@dataclass(frozen=True)
class Report:
    n_windows_considered: int
    n_graded: int
    states: dict[str, int]
    windows_above_half: int
    sign_test_p: float | None
    median_auc: float | None
    median_affected_deviation: float | None
    median_control_deviation: float | None
    by_type: dict[str, dict[str, float | int]]
    rows: list[dict[str, object]]


def summarise(grades: Sequence[WindowGrade]) -> Report:
    """Aggregate at the window level, and only at the window level."""
    graded = [g for g in grades if g.state == GRADED and g.auc is not None]
    aucs = [g.auc for g in graded if g.auc is not None]
    wins = sum(1 for a in aucs if a > 0.5)

    by_type: dict[str, dict[str, float | int]] = {}
    for alert_type in sorted({g.alert_type for g in graded}):
        rows = [g for g in graded if g.alert_type == alert_type]
        vals = [g.auc for g in rows if g.auc is not None]
        by_type[alert_type] = {
            "n_windows": len(rows),
            "median_auc": round(statistics.median(vals), 4),
            "median_affected_deviation": round(
                statistics.median(
                    [g.median_affected for g in rows if g.median_affected is not None]
                ),
                4,
            ),
        }

    states: dict[str, int] = {}
    for g in grades:
        states[g.state] = states.get(g.state, 0) + 1

    return Report(
        n_windows_considered=len(grades),
        n_graded=len(graded),
        states=states,
        windows_above_half=wins,
        sign_test_p=sign_test(wins, len(aucs)),
        median_auc=round(statistics.median(aucs), 4) if aucs else None,
        median_affected_deviation=(
            round(
                statistics.median(
                    [g.median_affected for g in graded if g.median_affected is not None]
                ),
                4,
            )
            if graded
            else None
        ),
        median_control_deviation=(
            round(
                statistics.median(
                    [g.median_control for g in graded if g.median_control is not None]
                ),
                4,
            )
            if graded
            else None
        ),
        by_type=by_type,
        rows=[asdict(g) for g in grades],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Grade the segment deviation score against announced planned work"
    )
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument("--min-arm", type=int, default=MIN_ARM_HOPS)
    parser.add_argument(
        "--causal",
        action="store_true",
        help="fit each window's baseline from its own past only; the deployable "
        "claim, and strictly harsher than the retrospective default",
    )
    args = parser.parse_args(argv)

    today = datetime.now(UTC).date()
    end = args.end_date or today
    start = args.start_date or (end - timedelta(days=1))

    bodies = fetch_trace_bodies(start_date=start, end_date=end)
    traversals, _trace_stats = traversals_from_trace(bodies)
    print(
        f"{len(bodies)} trace snapshots, {len(traversals)} traversals", file=sys.stderr
    )
    if not traversals:
        raise SystemExit(f"no traversals in {start}..{end}")
    span_lo = min(t.at for t in traversals)
    span_hi = max(t.at + t.seconds for t in traversals)

    with zipfile.ZipFile(io.BytesIO(fetch_gtfs_zip())) as zf:
        timetable = parse_timetable(zf)

    announced = windows_from_alerts(fetch_alert_bodies(start_date=start, end_date=end))
    live = [w for w in announced if overlaps(w, span_lo, span_hi)]
    gradeable = [w for w in live if w.gradeable]
    print(
        f"{len(announced)} announced, {len(live)} over the span, "
        f"{len(gradeable)} gradeable",
        file=sys.stderr,
    )

    if args.causal:
        # One fit per window, from that window's own past only. Slower by the
        # number of windows, and the only mode whose result describes a model
        # that could have run live.
        grades = [
            grade_window(
                w,
                traversals,
                baseline_before(traversals, timetable, live, w)[0],
                min_arm=args.min_arm,
            )
            for w in gradeable
        ]
    else:
        cells, withheld = baseline_outside(traversals, timetable, live)
        print(
            f"{len(cells)} hop cells fitted, {withheld} traversals withheld as "
            f"inside announced work",
            file=sys.stderr,
        )
        grades = [
            grade_window(w, traversals, cells, min_arm=args.min_arm) for w in gradeable
        ]
    report = summarise(grades)
    print(json.dumps(asdict(report), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
