"""Observed-headway reconstruction and typical-actual baselines at reference stops.

This is the offline substrate for the headway/wait surface and for a headway-gap
severity reading. It answers, per (route, direction), one question the supply and
movement axes are both structurally blind to: how long is a rider actually
waiting between trains, and how bunched is the service — measured where every
train of the route passes, against that cell's OWN typical delivered service
rather than the timetable.

WHERE THE EVENTS COME FROM. Not the trip-updates archive: that stream is a
compact per-route service metric (assigned_n and siblings) with no stop-level
timing. Observed arrivals live in archive/trace/ — the per-minute vehicle
census, reconstructed into (trip, stop) arrivals by training.trace. Successive
arrivals of distinct trains at one reference stop are the observed headway
series. That substrate begins 2026-08-12; there is no stop-level history before
it, however far back assigned_n reaches.

WHY A REFERENCE STOP. Headway is measured at a point. To be comparable across
ticks, days and routes, every route/direction needs one stable, documented stop.
The rule (select_reference_stops): the through-stop carrying the most scheduled
trips. That stop is served by the most trains, which on a branching or
express/local route is the trunk where every pattern overlaps — so the series
misses no train — and a through-stop excludes terminal turnbacks and yard leads,
whose dwell distorts the gap. It is deterministic from the static feed and, on
the current feed, resolves to a full-coverage (served-by-all-patterns) stop for
every route/direction.

THE BASELINE IS OWN-CELL, NEVER SCHEDULE. The published reading is deviation
from a cell's own typical delivered AWT/CV quantiles, per (route, direction,
weekday/weekend x hour). Scheduled SWT is computed alongside, for comparison
only: a route whose delivered headways chronically differ from its timetable
would read as permanently degraded against schedule while it is running its
ordinary service. See the weekday-service Saturday case for the concrete failure
a schedule baseline makes and this one does not.

THE WAIT FORMULA. AWT = sum(h^2) / (2 * sum(h)) over the headways in a period —
the average wait of a rider arriving at random, which weights the long gaps a
rider is more likely to fall into. Scheduled SWT uses the same formula on the
scheduled headways; for perfectly even service both reduce to headway / 2.

Three things the reconstruction has to honour, all measured, none assumed:
  * A gap in the FEED is not a gap in SERVICE. A trace coverage gap inflates the
    apparent headway across it; such headways are flagged and kept out of the
    wait statistics.
  * Trip ids repeat and trains are re-reported. A second arrival of the same
    trip at the same stop within DUP_ARRIVAL_SECONDS is the same train, not a
    zero headway.
  * ET, not UTC. Every time bucket is America/New_York local, matching
    schedule_bin; the night unit for the independent-night gate is the ET date.
"""

from __future__ import annotations

import argparse
import csv
import io
import statistics
import zipfile
from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from itertools import pairwise

from momentarily.hmm import schedule_bin
from training.eval_common import et_date, nearest_rank
from training.gtfs_static import (
    GTFS_STATIC_URL,
    Calendar,
    base_route,
    direction_of,
    fetch_gtfs_zip,
    read_calendar,
    route_patterns,
    successors,
    through_stops,
)
from training.trace import Passing

# --- reference-stop selection ---


@dataclass(frozen=True)
class ReferenceStop:
    """One route/direction's canonical headway measurement point.

    `coverage` is the share of the route/direction's scheduled trips whose
    pattern includes this stop — 1.0 means served by every pattern (express,
    local, and every branch), which is what a headway series needs to miss no
    train. `n_scheduled_trips` is how many trips serve it (the density criterion);
    `position`/`pattern_len` place it along the most-run pattern (mid-line, away
    from terminals)."""

    route: str
    direction: str
    stop_id: str
    n_scheduled_trips: int
    coverage: float
    n_patterns: int
    position: int
    pattern_len: int


def select_reference_stops(
    zf: zipfile.ZipFile,
) -> dict[tuple[str, str], ReferenceStop]:
    """Pick the canonical reference stop for every (route, direction).

    RULE, deterministic from the static feed: among a route/direction's
    through-stops (stops with both a scheduled predecessor and successor, so
    terminals and yard leads are excluded), take the one carrying the most
    scheduled trips. Ties break toward the stop nearest the middle of the most-run
    pattern, then the smaller stop id. The maximum-trips through-stop is the trunk
    where express/local and every branch overlap, so it is served by all patterns
    and the observed series contains one entry per train that ran the line.
    """
    patterns = route_patterns(zf)
    thru = through_stops(successors(zf))
    thru_by_group: dict[tuple[str, str], set[str]] = defaultdict(set)
    for route, direction, stop in thru:
        thru_by_group[(route, direction)].add(stop)

    out: dict[tuple[str, str], ReferenceStop] = {}
    for key, pats in patterns.items():
        candidates = thru_by_group.get(key)
        if not candidates:
            continue
        total = sum(p.n_trips for p in pats)
        trips_through = {
            stop: sum(p.n_trips for p in pats if stop in p.stops) for stop in candidates
        }
        dominant = pats[0].stops
        mid = len(dominant) / 2
        # rank each candidate by (most trips, nearest the pattern middle,
        # smaller id) via a plain tuple sort — no closure over the loop var.
        best = min(
            (
                -trips_through[stop],
                abs(
                    (dominant.index(stop) if stop in dominant else len(dominant)) - mid
                ),
                stop,
            )
            for stop in candidates
        )[2]
        route, direction = key
        out[key] = ReferenceStop(
            route=route,
            direction=direction,
            stop_id=best,
            n_scheduled_trips=trips_through[best],
            coverage=trips_through[best] / total if total else 0.0,
            n_patterns=len(pats),
            position=dominant.index(best) if best in dominant else -1,
            pattern_len=len(dominant),
        )
    return out


# --- observed headway reconstruction ---

# Two passings of the same trip at the same reference stop closer than this are
# one train re-reported by the feed, not two trains a zero headway apart.
DUP_ARRIVAL_SECONDS = 120

# A trace coverage gap at least this long overlapping a headway means the feed,
# not the service, may account for the interval. The trace polls every ~60s, so
# this is several missed polls, not one late one.
FEED_GAP_SECONDS = 240


@dataclass(frozen=True)
class HeadwayEvent:
    """One observed gap between two successive trains at a reference stop.

    `at` is the passing that closed the gap; `headway_sec` is that passing minus
    the previous train's. `feed_gap` marks a headway whose interval overlaps a
    trace coverage gap, so the wait statistics can exclude it: the long gap may
    be missing observation, not missing service.
    """

    route: str
    direction: str
    stop_id: str
    trip_id: str
    prev_trip_id: str
    at: int
    headway_sec: int
    feed_gap: bool


def _coverage_gaps(covered_seconds: Sequence[int]) -> list[tuple[int, int]]:
    """Runs of missing trace time as (start, end) intervals, from the sorted
    scheduled seconds actually present. A gap runs from one snapshot to the next
    whenever they are more than FEED_GAP_SECONDS apart."""
    gaps: list[tuple[int, int]] = []
    for a, b in pairwise(covered_seconds):
        if b - a > FEED_GAP_SECONDS:
            gaps.append((a, b))
    return gaps


def reference_arrivals(
    passings: Iterable[Passing],
    reference_stops: Mapping[tuple[str, str], ReferenceStop],
) -> dict[tuple[str, str], list[Passing]]:
    """Group transition-keyed passings at each route/direction's reference stop,
    in time order. (Name kept for the downstream handoff seam; the elements are
    now Passing departures, not first-STOPPED_AT Arrivals — see below.)

    Only passings whose (route_id, direction, stop_id) is that route/direction's
    reference stop are kept, so this is already the per-cell departure stream a
    headway series is built from. Keying on the departure (Passing) rather than
    the first STOPPED_AT sighting (Arrival) is what makes this series measure the
    same event as the live Worker and catch sub-poll dwells — see
    training.trace.passings_from_trace. Return shape is unchanged: one time-
    ordered per-cell list keyed by (route, direction)."""
    want = {
        (rs.route, rs.direction, rs.stop_id): key for key, rs in reference_stops.items()
    }
    out: dict[tuple[str, str], list[Passing]] = defaultdict(list)
    for p in passings:
        key = want.get((p.route_id, p.direction or "", p.stop_id))
        if key is not None:
            out[key].append(p)
    for series in out.values():
        series.sort(key=lambda p: (p.at, p.trip_id))  # (at, trip): the live tie-break
    return dict(out)


def headway_events(
    ref_passings: Mapping[tuple[str, str], list[Passing]],
    covered_seconds: Sequence[int],
) -> dict[tuple[str, str], list[HeadwayEvent]]:
    """Successive-train headways at each reference stop, with feed-gap flags.

    Same-trip re-reports within DUP_ARRIVAL_SECONDS collapse to the first
    sighting. A headway whose interval overlaps a trace coverage gap is flagged
    rather than dropped, so a caller decides whether the wait was real or
    unobserved."""
    gaps = _coverage_gaps(sorted(covered_seconds))
    gap_starts = [g[0] for g in gaps]
    out: dict[tuple[str, str], list[HeadwayEvent]] = {}
    for key, passings in ref_passings.items():
        events: list[HeadwayEvent] = []
        prev: Passing | None = None
        for a in passings:
            if (
                prev is not None
                and a.trip_id == prev.trip_id
                and (a.at - prev.at) < DUP_ARRIVAL_SECONDS
            ):
                continue  # same train re-reported
            if prev is not None:
                headway = a.at - prev.at
                if headway <= 0:
                    continue
                # a coverage gap starting inside (prev.at, a.at) makes this
                # interval feed-uncertain
                i = bisect_left(gap_starts, prev.at)
                feed_gap = i < len(gaps) and gaps[i][0] < a.at
                events.append(
                    HeadwayEvent(
                        route=key[0],
                        direction=key[1],
                        stop_id=a.stop_id,
                        trip_id=a.trip_id,
                        prev_trip_id=prev.trip_id,
                        at=a.at,
                        headway_sec=headway,
                        feed_gap=feed_gap,
                    )
                )
            prev = a
        out[key] = events
    return out


# --- wait statistics ---


def awt(headways: Sequence[float]) -> float | None:
    """Average wait time: sum(h^2) / (2 * sum(h)), the mean wait of a rider
    arriving at a random moment. Weights long gaps by how likely a rider is to
    land in one. None when there is no positive headway to divide by."""
    denom = 2 * sum(headways)
    if denom <= 0:
        return None
    return sum(h * h for h in headways) / denom


def headway_cv(headways: Sequence[float]) -> float | None:
    """Coefficient of variation of the headways — the bunching measure. 0 is
    perfectly even; the EWT identity AWT = (H/2)(1 + CV^2) makes it the excess
    wait driver. None below two headways (undefined stdev)."""
    if len(headways) < 2:
        return None
    mean = statistics.fmean(headways)
    if mean <= 0:
        return None
    return statistics.stdev(headways) / mean


# Trailing window over which a tick's AWT/CV is read, and the fewest headways in
# it for a reading. An hour of service is the standard AWT period; three
# headways is the floor below which CV is noise.
AWT_WINDOW_SECONDS = 3600
MIN_HEADWAYS_FOR_AWT = 3
TICK_SECONDS = 300


@dataclass(frozen=True)
class TickWait:
    """A reference stop's rolling wait reading at one 5-minute tick, over the
    trailing AWT_WINDOW_SECONDS of clean (non-feed-gap) headways."""

    route: str
    direction: str
    tick: int
    awt_sec: float
    cv: float | None
    n_headways: int


def tick_aligned_waits(
    events: Sequence[HeadwayEvent],
    *,
    window: int = AWT_WINDOW_SECONDS,
    min_headways: int = MIN_HEADWAYS_FOR_AWT,
    tick_seconds: int = TICK_SECONDS,
    include_feed_gap: bool = False,
) -> list[TickWait]:
    """5-minute-tick-aligned rolling AWT and CV for one reference stop.

    At each tick the window is the trailing `window` seconds of headways that
    closed in it; feed-gap headways are excluded unless `include_feed_gap`. A
    tick with fewer than `min_headways` clean headways gets no reading — the
    cell abstains rather than reporting a wait off one or two gaps."""
    clean = [e for e in events if include_feed_gap or not e.feed_gap]
    if not clean:
        return []
    times = [e.at for e in clean]
    hws = [e.headway_sec for e in clean]
    first_tick = ((clean[0].at) // tick_seconds) * tick_seconds
    last_tick = ((clean[-1].at) // tick_seconds) * tick_seconds
    out: list[TickWait] = []
    tick = first_tick
    while tick <= last_tick:
        lo = bisect_right(times, tick - window)
        hi = bisect_right(times, tick)
        w = hws[lo:hi]
        if len(w) >= min_headways:
            value = awt(w)
            if value is not None:
                out.append(
                    TickWait(
                        route=clean[0].route,
                        direction=clean[0].direction,
                        tick=tick,
                        awt_sec=value,
                        cv=headway_cv(w),
                        n_headways=len(w),
                    )
                )
        tick += tick_seconds
    return out


# --- typical-actual baseline (own-cell quantiles, night-gated) ---

# Mirror the supply axis's independent-night gate (load_r2.SERVICE_MIN_NIGHTS): a
# cell must span this many distinct ET dates before its quantiles publish, so a
# handful of autocorrelated ticks from one or two nights cannot certify a cell.
HEADWAY_MIN_NIGHTS = 8
HEADWAY_MIN_SAMPLES = 20


@dataclass(frozen=True)
class WaitCell:
    """One (route, direction, schedule_bin) cell's own typical delivered wait.

    Quantiles are of the tick AWT (seconds) over the cell's confirmed-normal
    ticks; `cv_p90` is the same for the bunching measure. `n_nights` is the
    independent-night count the gate is applied to, reported beside `n_ticks`
    because the ticks are autocorrelated within a night."""

    p10: float
    p50: float
    p90: float
    cv_p50: float | None
    cv_p90: float | None
    n_ticks: int
    n_nights: int


def typical_actual_baseline(
    waits: Mapping[tuple[str, str], Sequence[TickWait]],
    *,
    normal_ticks: Mapping[tuple[str, int], bool] | None = None,
    min_samples: int = HEADWAY_MIN_SAMPLES,
    min_nights: int = HEADWAY_MIN_NIGHTS,
) -> dict[tuple[str, str, str], WaitCell]:
    """Per (route, direction, schedule_bin) own-cell AWT/CV quantiles.

    `normal_ticks` maps (route, tick) -> True on ticks confirmed normal by an
    independent axis; when given, only those ticks feed the baseline, so the
    typical reading is of ordinary service, not of every disruption the window
    happened to contain. A cell below `min_samples` ticks OR spanning fewer than
    `min_nights` distinct ET dates is omitted — the reading is withheld, never
    fabricated. Nearest-rank on the cell's own samples, so a published quantile is
    always an observed value."""
    awt_by_cell: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    cv_by_cell: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    nights: dict[tuple[str, str, str], set[date]] = defaultdict(set)
    for (route, direction), series in waits.items():
        for tw in series:
            if normal_ticks is not None and not normal_ticks.get((route, tw.tick)):
                continue
            cell = (route, direction, schedule_bin(tw.tick))
            awt_by_cell[cell].append(tw.awt_sec)
            if tw.cv is not None:
                cv_by_cell[cell].append(tw.cv)
            nights[cell].add(et_date(tw.tick))

    out: dict[tuple[str, str, str], WaitCell] = {}
    for cell, values in awt_by_cell.items():
        if len(values) < min_samples or len(nights[cell]) < min_nights:
            continue
        ordered = sorted(values)
        cvs = sorted(cv_by_cell.get(cell, []))
        out[cell] = WaitCell(
            p10=nearest_rank(ordered, 0.10),
            p50=nearest_rank(ordered, 0.50),
            p90=nearest_rank(ordered, 0.90),
            cv_p50=nearest_rank(cvs, 0.50) if cvs else None,
            cv_p90=nearest_rank(cvs, 0.90) if cvs else None,
            n_ticks=len(values),
            n_nights=len(nights[cell]),
        )
    return out


# --- scheduled SWT, for comparison only (never the published reference) ---


def _pick_service_day(calendar: Calendar, *, weekend: bool) -> date:
    """A representative in-window date of the requested class whose calendar
    actually runs service — the anchor a single clean day's schedule is read
    from, avoiding the double-count of overlapping same-class calendars."""
    starts = [
        date(int(w.start[:4]), int(w.start[4:6]), int(w.start[6:8]))
        for w in calendar.weekly
        if len(w.start) == 8
    ]
    d0 = max(starts) if starts else date.today()
    for i in range(28):
        d = d0 + timedelta(days=i)
        if (d.weekday() >= 5) == weekend and calendar.active(d):
            return d
    return d0


def scheduled_swt(
    zf: zipfile.ZipFile,
    reference_stops: Mapping[tuple[str, str], ReferenceStop],
    *,
    weekday: date | None = None,
    weekend: date | None = None,
) -> dict[tuple[str, str, str], float]:
    """Scheduled SWT per (route, direction, schedule_bin) at the reference stop.

    SWT = sum(h^2)/(2*sum(h)) over the scheduled headways in the cell — the same
    wait formula the observed AWT uses, so the two are directly comparable. This
    is the timetable's promise, computed ONLY for the schedule-vs-typical
    comparison; it is never the baseline a deviation is read against.

    ONE representative service day per class. NYCT carries several overlapping
    weekend calendars (`Saturday`, `Saturday-H-<range>`, ...) that each describe
    the same service on a different date range; pooling every weekend service_id
    would count one Saturday's trains two or three times and halve the headway.
    So the schedule is read for the service_ids Calendar.active names on one
    representative weekday and one weekend date within the feed window.

    Departures at or past 24:00 (service-day time) are abstained rather than
    wrapped into the wrong weekday/weekend class — the same overnight caution the
    supply axis uses.
    """
    ref_stop_ids = {rs.stop_id for rs in reference_stops.values()}
    calendar = read_calendar(zf)
    weekday = weekday or _pick_service_day(calendar, weekend=False)
    weekend = weekend or _pick_service_day(calendar, weekend=True)
    cls_of: dict[str, str] = {}
    for sid in calendar.active(weekday):
        cls_of[sid] = "wd"
    for sid in calendar.active(weekend):
        cls_of.setdefault(sid, "we")  # if a sid runs both days, weekday wins

    # trip_id -> (base route, service class) for every trip on an active service
    trip_meta: dict[str, tuple[str, str | None]] = {}
    with zf.open("trips.txt") as raw:
        rows = csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
        for row in rows:
            trip_meta[row["trip_id"]] = (
                base_route(row["route_id"]),
                cls_of.get(row["service_id"]),
            )

    # (route, direction, class, hour) -> list of scheduled departure seconds
    departures: dict[tuple[str, str, str, int], list[int]] = defaultdict(list)
    with zf.open("stop_times.txt") as raw:
        reader = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
        header = next(reader)
        trip_col = header.index("trip_id")
        stop_col = header.index("stop_id")
        dep_col = header.index("departure_time")
        for row in reader:
            stop = row[stop_col]
            if stop not in ref_stop_ids:
                continue
            route, cls = trip_meta.get(row[trip_col], ("", None))
            if not route or cls is None:
                continue
            direction = direction_of(stop, row[trip_col])
            if direction is None:
                continue
            # count a departure only at this route/direction's OWN reference
            # stop — trunk stops belong to several routes' stop sequences, and a
            # through-running train would otherwise inflate a route it shares track
            # with (the 1/2/3 share 7th Ave stops).
            ref = reference_stops.get((route, direction))
            if ref is None or ref.stop_id != stop:
                continue
            parts = row[dep_col].split(":")
            if len(parts) != 3:
                continue
            sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            if sec >= 86400:  # after-midnight: abstain, don't misclassify
                continue
            departures[(route, direction, cls, sec // 3600)].append(sec)

    out: dict[tuple[str, str, str], float] = {}
    for (route, direction, cls, hour), secs in departures.items():
        if len(secs) < 2:
            continue
        secs.sort()
        headways = [b - a for a, b in pairwise(secs) if b - a > 0]
        value = awt(headways)
        if value is not None:
            out[(route, direction, f"{cls}{hour:02d}")] = value
    return out


def load_gtfs_zip(path: str | None = None) -> zipfile.ZipFile:
    """Open a GTFS static zip from a local path (for a window-matched archived
    feed) or fetch the current one."""
    if path:
        return zipfile.ZipFile(path)
    return zipfile.ZipFile(io.BytesIO(fetch_gtfs_zip(GTFS_STATIC_URL)))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Reference-stop selection")
    parser.add_argument("--gtfs-zip", default=None)
    args = parser.parse_args(argv)
    zf = load_gtfs_zip(args.gtfs_zip)
    stops = select_reference_stops(zf)
    for key in sorted(stops):
        rs = stops[key]
        print(
            f"{rs.route:>3} {rs.direction:<5} {rs.stop_id:<5} "
            f"trips={rs.n_scheduled_trips:>4} cov={rs.coverage:.3f} "
            f"pos={rs.position}/{rs.pattern_len} npat={rs.n_patterns}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
