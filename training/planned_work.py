"""Grade the segment traversal measure against announced planned work.

The measure has never had an answer key. The alerts feed cannot supply one: 95%
of route-minutes on an ordinary weekday carry a Delays or Suspended alert, and
every UNPLANNED alert type is published route-wide with no stop scope at all
(measured 2026-08-13 over 255 alert versions: 180 bare route-level `Delays`;
`Part Suspended`, `Reroute` and `Stops Skipped` name zero stops).

Planned work is the exception, and it is the only exception. It arrives with the
stations it affects and the hours it runs, days ahead. That makes it a
pre-registerable natural experiment: we know now which segments should slow and
when, so a measure that cannot see it is not measuring service.

WHAT THIS GRADES. Detection, not prediction — the work is announced, so nothing
here is a forecast. Detection is simply the prerequisite, and it is the part
currently unverifiable. Unplanned disruption still has no truth source.

WHAT COUNTS AS SUPPLY. 388 geo-scoped windows were announced as of 2026-08-13,
but 278 are `Express to Local` naming a median of ONE stop, and those are close
to untestable here for two structural reasons: vehicles.ts folds 6X/7X to their
base route, so an express running local is indistinguishable from a local in the
cell keys, and one named station does not describe a stretch. The gradeable
subset is the 99 windows that change WHICH stops a train serves — Part
Suspended, Stops Skipped, Suspended, Reroute, Special Schedule — at a median of
5 named stops.

MOVEMENT ONLY. Nothing here reads the timetable. The effect is each segment
measured against ITSELF outside the window, so the comparison needs no external
reference and inherits no schedule.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo

from training.gtfs_static import (
    HopKey,
    base_route,
    is_express_variant,
    parent_station,
)
from training.load_r2 import date_range, fetch_objects, list_keys
from training.r2_client import R2Config, load_config, make_client
from training.trace import EXACT, Traversal

# WHAT PLANNED WORK ACTUALLY DOES TO MOVEMENT, which is not what the first cut of
# this module assumed. Nearly every announced type changes WHICH PAIRS a train
# serves rather than how long it takes over the pairs it keeps: a skipped station
# makes the hops into and out of it disappear and a longer nonstop hop appear; a
# part suspension does the same at the scale of a stretch; an express running
# local stops making its own long hops. Scoring any of those by hop DURATION on
# the named segments reports the announced outcome as a failed detection, because
# the named segments have no traversals inside the window at all.
#
# So coverage is the primary grade for every type that changes service, and
# duration is a SECONDARY, weaker question about spillover onto the segments that
# keep running beside the work. Both apply to the same windows.
SERVICE_CHANGING = frozenset(
    {
        "Planned - Part Suspended",
        "Planned - Stops Skipped",
        "Planned - Suspended",
        "Planned - Reroute",
        "Planned - Express to Local",
        "Special Schedule",
    }
)

# Types that move headways or signage rather than the path a train takes. Nothing
# in a station-to-station traversal can see them, so they are excluded rather than
# graded and counted as misses.
HEADWAY_ONLY = frozenset({"Extra Service", "Reduced Service", "Boarding Change"})

# Traversals a segment needs on each side of the window boundary before its own
# before/after ratio is worth anything. Deliberately low: a single overnight
# closure is a few hours, and the point of the difference-in-differences is that
# noise in the affected arm is cancelled by the same noise in the control arm.
# Local time: a service day and an overnight work window are both wall-clock
# facts, and New York observes DST.
_ET = ZoneInfo("America/New_York")

MIN_SIDE_SAMPLES = 5


@dataclass(frozen=True)
class Window:
    """One announced planned-work period, scoped to the stations it names.

    `stops` are PARENT station ids exactly as the alert publishes them (`J20`),
    while the trace reports directional platforms (`J20N`, `J20S`). Nothing in
    either feed flags the mismatch — a naive join finds zero of 42 affected
    platforms and reads as "the measure detected nothing", so the strip happens
    here, once.
    """

    alert_type: str
    routes: frozenset[str]
    stops: frozenset[str]
    start: int
    end: int  # 0 when the alert left it open-ended

    def contains(self, at: int) -> bool:
        return at >= self.start and (self.end == 0 or at <= self.end)

    def covers_route(self, route_id: str) -> bool:
        """Alerts name the base route, so a window on the 7 covers 7 and 7X."""
        return base_route(route_id) in self.routes

    def touches(self, key: HopKey) -> bool:
        """Whether a hop runs into, out of, or between the named stations."""
        return self._named(key) > 0

    def at_boundary(self, key: HopKey) -> bool:
        """Exactly one endpoint named: the hops that keep running beside the work.

        A hop with BOTH endpoints named lies inside the closed stretch and simply
        stops appearing, so including it in a duration test selects for absence
        and reports no effect. It is `pattern_shift`'s business, not this one's.
        """
        return self._named(key) == 1

    def _named(self, key: HopKey) -> int:
        _route, _direction, from_stop, to_stop = key
        return (parent_station(from_stop) in self.stops) + (
            parent_station(to_stop) in self.stops
        )

    @property
    def gradeable(self) -> bool:
        """Whether a station-to-station traversal can see this type of work at
        all. Headway and signage changes cannot be graded here and must not be
        counted as misses."""
        return self.alert_type in SERVICE_CHANGING


def windows_from_alerts(bodies: Iterable[dict[str, Any]]) -> list[Window]:
    """Every geo-scoped planned-work window in the archived alert versions.

    One Window per (alert, active_period): an alert republished unchanged yields
    duplicates, so the result is deduplicated. Alerts naming no stop are skipped
    entirely rather than widened to the whole route — a route-wide window has no
    control arm, which is the same reason the unplanned feed is useless here.
    """
    out: set[Window] = set()
    for body in bodies:
        alert = cast(dict[str, Any], body.get("alert") or {})
        payload = cast(dict[str, Any], alert.get("alert") or {})
        mercury = cast(
            dict[str, Any],
            payload.get("transit_realtime.mercury_alert") or {},
        )
        alert_type = str(mercury.get("alert_type") or "")
        entities = cast(list[Any], payload.get("informed_entity") or [])
        stops: set[str] = set()
        routes: set[str] = set()
        for raw in entities:
            if not isinstance(raw, dict):
                continue
            entity = cast(dict[str, Any], raw)
            if stop_id := entity.get("stop_id"):
                stops.add(str(stop_id))
            if route_id := entity.get("route_id"):
                routes.add(base_route(str(route_id)))
        if not stops or not routes:
            continue
        for raw_period in cast(list[Any], payload.get("active_period") or []):
            if not isinstance(raw_period, dict):
                continue
            period = cast(dict[str, Any], raw_period)
            start = int(period.get("start") or 0)
            if not start:
                continue
            out.add(
                Window(
                    alert_type=alert_type,
                    routes=frozenset(routes),
                    stops=frozenset(stops),
                    start=start,
                    end=int(period.get("end") or 0),
                )
            )
    return sorted(out, key=lambda w: (w.start, w.alert_type))


def fetch_alert_bodies(
    config: R2Config | None = None,
    *,
    start_date: date,
    end_date: date,
) -> list[dict[str, Any]]:
    """Every archived alert version in the window, one object per version."""
    cfg = config or load_config()
    client = make_client(cfg)
    keys: list[str] = []
    for day in date_range(start_date, end_date):
        keys.extend(list_keys(client, cfg.bucket, f"archive/alerts/{day.isoformat()}/"))
    return fetch_objects(client, cfg.bucket, keys)


@dataclass(frozen=True)
class Effect:
    """What a window did to the segments it named, against the segments it did not.

    `affected_lift` is the median over named segments of (median seconds inside
    the window / median seconds outside it). `control_lift` is the same over the
    route's other segments, and `effect` is the ratio of the two.

    The control arm is the whole point. A raw before/after on the affected
    segments cannot tell planned work from the evening rush, a slow news day, or
    the archive simply covering different hours on each side. Dividing by what
    the same route's untouched segments did over the same clock cancels
    everything the two arms share, so `effect` above 1.0 means the named
    segments slowed BY MORE than the rest of their own route.
    """

    alert_type: str
    routes: tuple[str, ...]
    start: int
    end: int
    n_affected_keys: int
    n_control_keys: int
    n_inside: int  # traversals on affected segments inside the window
    affected_lift: float
    control_lift: float
    effect: float


def _lift(
    keyed: Sequence[tuple[HopKey, list[int], list[int]]],
) -> tuple[float, int] | None:
    """Median per-segment inside/outside ratio, and how many segments backed it."""
    ratios = [
        statistics.median(inside) / statistics.median(outside)
        for _key, inside, outside in keyed
        if len(inside) >= MIN_SIDE_SAMPLES and len(outside) >= MIN_SIDE_SAMPLES
    ]
    return (statistics.median(ratios), len(ratios)) if ratios else None


def _effect_one(
    window: Window,
    rows: Sequence[Traversal],
    blackout: Sequence[Window],
) -> Effect | None:
    affected: dict[HopKey, tuple[list[int], list[int]]] = {}
    control: dict[HopKey, tuple[list[int], list[int]]] = {}
    for t in rows:
        if t.to_stop is None or not window.covers_route(t.route_id):
            continue
        key: HopKey = (t.route_id, t.direction or "", t.from_stop, t.to_stop)
        if window.at_boundary(key):
            bucket = affected
        elif window.touches(key):
            continue  # inside the closed stretch: it vanishes, it does not slow
        else:
            bucket = control
        inside, outside = bucket.setdefault(key, ([], []))
        if window.contains(t.at):
            inside.append(t.seconds)
        elif not any(w.contains(t.at) and w.covers_route(t.route_id) for w in blackout):
            outside.append(t.seconds)

    got_affected = _lift([(k, i, o) for k, (i, o) in affected.items()])
    got_control = _lift([(k, i, o) for k, (i, o) in control.items()])
    if got_affected is None or got_control is None:
        return None
    affected_lift, n_affected = got_affected
    control_lift, n_control = got_control
    return Effect(
        alert_type=window.alert_type,
        routes=tuple(sorted(window.routes)),
        start=window.start,
        end=window.end,
        n_affected_keys=n_affected,
        n_control_keys=n_control,
        n_inside=sum(len(i) for i, _o in affected.values()),
        affected_lift=round(affected_lift, 4),
        control_lift=round(control_lift, 4),
        effect=round(affected_lift / control_lift, 4),
    )


def measure(
    window: Window,
    traversals: Iterable[Traversal],
    *,
    other_windows: Iterable[Window] = (),
) -> list[Effect]:
    """The SECONDARY, weaker grade: did the segments beside the work slow down?

    One Effect per local day the window covers. Not a test of the named segments
    themselves — a closure or a skip removes their traversals entirely, and that
    absence is what `pattern_shift` measures as the primary claim. What survives
    for a duration test is spillover: a hop with ONE endpoint in the named set
    still runs, and is where single-tracking, merging and reduced frequency show
    up as time.

    Difference-in-differences, from movement alone, because a raw before/after on
    the boundary hops cannot tell planned work from the evening rush. Dividing by
    what the same route's distant segments did cancels everything the two arms
    share.

    THE OUTSIDE ARM IS DELIBERATELY UNRESTRICTED, and narrowing it to matched
    hours would only cost samples. Both arms are measured against the same
    comparison period, so a period that runs systematically faster or slower than
    the window scales both lifts by the same factor and drops out of the ratio.
    That is exactly what `pattern_shift` cannot do -- it has one arm, so its
    control has to be built matched instead. It is still split per local day: over a Friday-to-Monday
    closure the boundary hops and the distant ones are not observed in the same
    proportions across the weekend, so the cancellation is only approximate when
    the days are pooled.

    `other_windows` is excluded from the OUTSIDE arm on the same route: a second
    closure sitting in the control period would be compared against as though it
    were normal service, dragging the baseline toward the disrupted state and
    shrinking whatever effect exists. Overlapping planned work is the norm.

    Empty when the type cannot be seen in traversals at all, or when neither arm
    clears MIN_SIDE_SAMPLES on enough segments. Neither is a failed detection.
    """
    if not window.gradeable:
        return []
    rows = [
        t
        for t in traversals
        if t.censoring == EXACT and t.to_stop is not None and t.n_hops == 1
    ]
    # The UNSPLIT window goes in the blackout, not just the other alerts. Each
    # piece's control arm is "outside this piece", and without the original a
    # Friday piece would treat the Saturday and Sunday of its own closure as
    # normal service -- contaminating the baseline with the very disruption
    # being measured.
    blackout = [*(w for w in other_windows if w != window), window]
    out: list[Effect] = []
    for piece in split_by_local_day(window):
        got = _effect_one(piece, rows, blackout)
        if got is not None:
            out.append(got)
    return out


@dataclass(frozen=True)
class PatternShift:
    """Which pairs ONE service served inside a window against outside it.

    The measure for `Express to Local`, which is 278 of the 388 announced
    windows and which a duration test cannot see. An express running local keeps
    moving at ordinary speed; what changes is that its own long hops stop
    appearing while short local ones show up in their place.

    Per service, not per route, and that is the whole design. The Worker folds 7X
    to 7 before writing, so a route-wide aggregate mixes the express with the
    local: a handful of vanished express pairs against hundreds of local pairs
    running normally averages to nothing. Reported separately, the 7X row carries
    the signal and the 7 row is a control that came free — the same window, the
    same clock, a service that was not told to change.

    `vanished` are pairs this service ran outside the window and not inside it, as
    a share of the pairs it ran outside. `appeared` is the mirror. Neither is a
    detection alone: a key can vanish because the train stopped making that hop,
    or because the archive thinned out, which is what `n_inside` is for.
    """

    alert_type: str
    service: str  # the route as the trip id declares it: '7X', not the folded '7'
    routes: tuple[str, ...]
    start: int
    end: int
    n_keys_inside: int
    n_keys_outside: int
    n_inside: int
    n_outside: int
    vanished: float
    appeared: float


# Service classes as the calendar defines them, which is all this validator
# needs: NYCT runs one timetable Monday-Friday, one Saturday, one Sunday. Derived
# from the date rather than read off the schedule, so no timetable enters here.
# The known gap is holidays — a Monday running Sunday service is classed a
# weekday, and only calendar_dates.txt knows otherwise.
def _service_class(at: int) -> int:
    """0 for any weekday, 1 Saturday, 2 Sunday."""
    weekday = datetime.fromtimestamp(at, _ET).weekday()
    return 0 if weekday < 5 else weekday - 4


def _clock(at: int) -> int:
    """Seconds since local midnight."""
    local = datetime.fromtimestamp(at, _ET)
    return local.hour * 3600 + local.minute * 60 + local.second


def _matches_clock(at: int, window: Window) -> bool:
    """Whether a moment falls in the window's span of the LOCAL clock, on any day.
    Wrap-aware, because planned work is overnight far more often than not."""
    if window.end == 0:
        return True
    start, end, now = _clock(window.start), _clock(window.end), _clock(at)
    return start <= now <= end if start <= end else now >= start or now <= end


def split_by_local_day(window: Window) -> list[Window]:
    """One sub-window per local calendar day the period spans.

    An alert period is one row, but a closure is not one comparison. The J part
    suspension announced for 2026-08-14 runs Fri 23:45 straight through to Mon
    05:00: a Friday night, a whole Saturday, a whole Sunday and a Monday morning,
    on three different timetables. Graded whole, the inside arm is every weekend
    train while the control arm is whatever ran in the 23:45-05:00 band — not a
    comparison, just two different populations.

    Split, each piece has one service class and one clock band, which is what the
    matched control in `pattern_shift` needs. A window already inside one local
    day is returned unchanged.
    """
    if window.end == 0 or window.end <= window.start:
        return [window]
    first = datetime.fromtimestamp(window.start, _ET).date()
    last = datetime.fromtimestamp(window.end, _ET).date()
    if first == last:
        return [window]
    out: list[Window] = []
    day = first
    while day <= last:
        midnight = int(datetime.combine(day, time.min, tzinfo=_ET).timestamp())
        nxt = int(
            datetime.combine(day + timedelta(days=1), time.min, tzinfo=_ET).timestamp()
        )
        start = max(window.start, midnight)
        end = min(window.end, nxt - 1)
        # `>=`, not `>`: Window intervals are closed at both ends, so a piece one
        # second wide is a real piece. A closure ending exactly at local midnight
        # has a final piece of start == end, and dropping it loses the traversals
        # at that instant.
        if end >= start:
            out.append(
                Window(
                    alert_type=window.alert_type,
                    routes=window.routes,
                    stops=window.stops,
                    start=start,
                    end=end,
                )
            )
        day += timedelta(days=1)
    return out or [window]


def _is_control(
    traversal: Traversal,
    window: Window,
    want_class: int,
    blackout: Sequence[Window],
) -> bool:
    """Whether a traversal outside the window belongs in the matched control arm:
    a comparable service day, the same band of the local clock, and no other
    announced work running on its route at the time.

    Shared with `control_supply` so that what the grade counts and what the
    diagnosis counts cannot drift apart.
    """
    return (
        _service_class(traversal.at) == want_class
        and _matches_clock(traversal.at, window)
        and not any(
            w.contains(traversal.at) and w.covers_route(traversal.route_id)
            for w in blackout
        )
    )


def _service_of(traversal: Traversal) -> str:
    """The service as the trip id declares it: '7X', not the '7' the Worker
    already folded it to.

    Shared with `control_supply` so the grade and the diagnosis cannot disagree
    about which service a traversal belongs to.
    """
    return (
        traversal.route_id + "X"
        if is_express_variant(traversal.trip_id)
        else traversal.route_id
    )


def _shift_one(
    window: Window,
    rows: Sequence[Traversal],
    blackout: Sequence[Window],
) -> list[PatternShift]:
    want_class = _service_class(window.start)
    inside: dict[str, set[HopKey]] = defaultdict(set)
    outside: dict[str, set[HopKey]] = defaultdict(set)
    n_in: Counter[str] = Counter()
    n_out: Counter[str] = Counter()
    for t in rows:
        if t.to_stop is None or not window.covers_route(t.route_id):
            continue
        service = _service_of(t)
        key: HopKey = (service, t.direction or "", t.from_stop, t.to_stop)
        if window.contains(t.at):
            inside[service].add(key)
            n_in[service] += 1
            continue
        if not _is_control(t, window, want_class, blackout):
            continue
        outside[service].add(key)
        n_out[service] += 1
    out: list[PatternShift] = []
    for service in sorted(set(inside) | set(outside)):
        was, now = outside[service], inside[service]
        if not was or not now:
            continue
        out.append(
            PatternShift(
                alert_type=window.alert_type,
                service=service,
                routes=tuple(sorted(window.routes)),
                start=window.start,
                end=window.end,
                n_keys_inside=len(now),
                n_keys_outside=len(was),
                n_inside=n_in[service],
                n_outside=n_out[service],
                vanished=round(len(was - now) / len(was), 4),
                appeared=round(len(now - was) / len(now), 4),
            )
        )
    return out


def pattern_shift(
    window: Window,
    traversals: Iterable[Traversal],
    *,
    other_windows: Iterable[Window] = (),
) -> list[PatternShift]:
    """One PatternShift per (local day, service) the window covers, from movement
    alone. Empty when the type cannot be seen in traversals, or when no matched
    control period exists in the data yet.

    Services are recovered from the trip id, because the archived route_id has
    already folded 7X into 7 and a folded aggregate cannot see an express-to-local
    change at all. Each service is reported on its own so the express's vanishing
    pairs are not averaged against the local's surviving ones.

    THE CONTROL ARM IS THE SAME HOURS ON A COMPARABLE DAY, not the rest of the
    same day, and that is not a refinement — it is what makes the measure mean
    anything. Three ways the obvious comparison lies:

      * Express service only runs at rush hour, which is exactly when this work
        is scheduled. Measured 2026-08-13 against a 15:00-22:00 window, the 7X had
        13 traversals outside it against 790 inside, so "outside" really meant
        "when no express runs" and every pair would read as appearing from
        nowhere.
      * The 7X does not run at all on a weekend, so a Saturday control against a
        weekday window would manufacture the entire signal.
      * A single alert period is not a single comparison. Multi-day closures are
        graded per local day for the reason `split_by_local_day` gives.

    Unlike `measure`, this has no second arm to cancel time of day against, so the
    comparison has to be built matched rather than corrected afterwards.
    """
    if not window.gradeable:
        return []
    rows = [
        t
        for t in traversals
        if t.censoring == EXACT and t.to_stop is not None and t.n_hops == 1
    ]
    # See `measure` on why the unsplit window is in the blackout.
    blackout = [*(w for w in other_windows if w != window), window]
    return [
        shift
        for piece in split_by_local_day(window)
        for shift in _shift_one(piece, rows, blackout)
    ]


def control_supply(
    window: Window,
    traversals: Iterable[Traversal],
    *,
    other_windows: Iterable[Window] = (),
) -> dict[str, int]:
    """Control-arm traversals PER SERVICE, summed over the window's local-day
    pieces.

    `pattern_shift` emits a row whenever one service holds both arms, including
    when nothing moved -- an unchanged pairing is a row of zeros, not an absence.
    So an empty result never means "measured, no effect"; it means no service was
    in a position to be measured, and this is how the caller tells which way.

    Per service, not per route, for the same reason `PatternShift` is: a row is
    emitted only when ONE service has both arms, so a route-level total says
    "compared" on the strength of a 7 local while the 7X the work actually named
    had no control at all. Counted by route, the diagnostic would certify a
    comparison that never happened -- which is the exact failure it exists to
    catch.

    An empty result means no service had a control arm; an archive shorter than
    two comparable days cannot produce one for any window, by construction.
    """
    if not window.gradeable:
        return {}
    rows = [
        t
        for t in traversals
        if t.censoring == EXACT and t.to_stop is not None and t.n_hops == 1
    ]
    blackout = [*(w for w in other_windows if w != window), window]
    out: Counter[str] = Counter()
    for piece in split_by_local_day(window):
        want_class = _service_class(piece.start)
        for t in rows:
            if t.to_stop is None or not piece.covers_route(t.route_id):
                continue
            if piece.contains(t.at) or not _is_control(t, piece, want_class, blackout):
                continue
            out[_service_of(t)] += 1
    return dict(out)


# The three outcomes a coverage grade can reach. Only the first is a measurement;
# the other two are the archive saying it could not supply one, and collapsing
# them into "detected nothing" is how a supply gap gets published as a result.
GRADED = "graded"
NO_PAIRED_SERVICE = "no_paired_service"
NO_CONTROL_PERIOD = "no_control_period"


def coverage_state(
    window: Window,
    shifts: Sequence[PatternShift],
    traversals: Iterable[Traversal],
    *,
    other_windows: Iterable[Window] = (),
) -> str:
    """Which of the three outcomes this window's coverage grade reached.

    `shifts` is the window's own `pattern_shift` result, passed in rather than
    recomputed: a non-empty list already establishes that some service had both
    arms, so that question is never asked twice. What is left to distinguish is
    whether an empty list means the archive held no comparable period at all, or
    held one for some service while none of them paired.
    """
    if shifts:
        return GRADED
    supply = control_supply(window, traversals, other_windows=other_windows)
    return NO_PAIRED_SERVICE if any(supply.values()) else NO_CONTROL_PERIOD


def _band_on(day: date, piece: Window) -> tuple[int, int]:
    """A PIECE's span of the local clock projected onto `day`, absolute seconds.

    The band, not the period: projecting a closure's whole duration onto a
    candidate day would sweep months for the long ones. The caller passes a
    `split_by_local_day` piece rather than the announced window, because a piece
    is what carries one clock band and one service class — the unsplit Friday-to-
    Monday closure has neither, and reading a single band off it is how a
    diagnostic starts certifying comparisons the grade never makes. Wrap and
    open-endedness are handled exactly as `_matches_clock` handles them, so this
    selects the moments `_is_control` would admit.

    Both endpoints are built from the wall clock with `datetime.combine`, the way
    `split_by_local_day` builds its day boundaries, rather than by adding seconds
    to a midnight. New York observes DST, so on a transition day the offset from
    midnight to a wall-clock hour is not the number of seconds that hour names,
    and a band projected arithmetically lands an hour off — enough to certify a
    day as free against the wrong overlap twice a year.
    """
    if piece.end == 0:
        return (
            int(datetime.combine(day, time.min, tzinfo=_ET).timestamp()),
            int(
                datetime.combine(
                    day + timedelta(days=1), time.min, tzinfo=_ET
                ).timestamp()
            )
            - 1,
        )
    start_local = datetime.fromtimestamp(piece.start, _ET).time()
    end_local = datetime.fromtimestamp(piece.end, _ET).time()
    # A band ending earlier on the clock than it started wraps past midnight, so
    # its close belongs to the NEXT calendar day — the same rule `_matches_clock`
    # applies, spelled as a date rather than as 86,400 seconds.
    end_day = day + timedelta(days=1) if end_local < start_local else day
    return (
        int(datetime.combine(day, start_local, tzinfo=_ET).timestamp()),
        int(datetime.combine(end_day, end_local, tzinfo=_ET).timestamp()),
    )


def _free_instant(start: int, end: int, route: str, blackout: Sequence[Window]) -> bool:
    """Whether any instant in [start, end] on `route` escapes every blackout.

    AN INSTANT, NOT THE BAND. `_is_control` admits a traversal when no blackout
    window `contains` its own timestamp, so work occupying PART of a band leaves
    the rest admissible: against a 12:00-16:00 band, a 12:00-13:00 closure still
    permits 13:00-16:00 controls and the grade will use them. Rejecting the whole
    band on any overlap makes this diagnostic stricter than the measure it
    describes, which reports no reach for a grade that can run — the same
    disagreement, in a third place, as reading one clock band off a multi-piece
    window and as ignoring routes in the blackout test.

    Swept rather than summed because the blackout intervals may overlap each
    other; a gap survives only where none of them reaches. Both ends are closed,
    matching `Window.contains`, so a blocked span resumes at `hi + 1`.
    """
    spans = sorted(
        (w.start, end if w.end == 0 else w.end)
        for w in blackout
        if w.covers_route(route) and w.start <= end and (w.end == 0 or w.end >= start)
    )
    cursor = start
    for lo, hi in spans:
        if lo > cursor:
            return True
        cursor = max(cursor, hi + 1)
        if cursor > end:
            return False
    return cursor <= end


@dataclass(frozen=True)
class ControlReach:
    """Whether a matched control could exist YET, judged only where the answer
    key is in a position to answer.

    `NO_CONTROL_PERIOD` says the archive held no comparable period, and on its
    own it cannot tell a window whose control is one week out of reach from one
    whose control cannot exist. Both read as a permanent dead end, so a report
    that only carries the state invites the same class of mistake this module
    already corrected once for an empty `pattern_shift`: an absence of supply
    published as a finding.

    CERTIFIED ONLY WHERE THE ANSWER KEY CAN ANSWER. A day is free when no
    announced work runs in the window's clock band on it, and absence of an
    announcement is evidence of that ONLY on days we hold alert snapshots for.
    Before the first publication day the work that ran and ended is simply not in
    the record; after the last, work is not yet announced; and between them,
    coverage can have holes. Measured 2026-08-17, searching +/-60 days against
    the loaded record named 2026-08-04 as the 7's nearest free weekday when the
    record began 2026-08-12 and could not see that day at all. A countdown to a
    contaminated control is worse than no countdown.
    """

    day: date | None
    # Signed: negative when the free day precedes the window, so the sign says
    # whether the answer is "reach further back" or "wait".
    lag_days: int | None
    certified: int  # comparable days inside the coverage span
    covered: int  # of those, how many carry announced work in the band


def control_reach(
    window: Window,
    announced: Iterable[Window],
    coverage: Iterable[date],
) -> ControlReach:
    """The nearest day a matched control could come from, or nothing when the
    record cannot certify one.

    PER LOCAL-DAY PIECE, because that is what gets graded. `pattern_shift` and
    `control_supply` both iterate `split_by_local_day`, and a piece is the only
    thing carrying one clock band and one service class. Read off the unsplit
    window, a Friday-23:45-to-Monday-05:00 closure yields a single 23:45-05:00
    band on Friday's weekday class — while the grade actually wants a Friday late
    night, a whole Saturday, a whole Sunday and a Monday morning, on three
    different timetables. The diagnostic would then certify a free weekday as the
    control for a Sunday piece and promise a grade that cannot run.

    The nearest free day across pieces wins, mirroring `coverage_state`: a window
    counts as graded when ANY piece produced a row, so it becomes reachable as
    soon as any one of them has a control. `certified` and `covered` are summed
    over (piece, candidate day) pairs for the same reason.

    `coverage` is the SET of publication days held, never a first..last span.
    `load_windows` supports non-contiguous coverage, and collapsing it to its
    endpoints would read a day whose snapshots are missing as evidence of free
    service — the same unbacked promise as searching before the record begins,
    one level in.

    THE WINDOW BLOCKS ITS OWN DATES. The unsplit window goes in the blackout,
    exactly as `pattern_shift` puts it there: a closure running Friday to Monday
    is one announced period, and without it every day of that period but the
    first would read free. The 4's announced work spans 2026-04-27 to 2026-08-18,
    so the difference is 113 days of its own closure certified as a control for
    itself.

    `day is None` with `certified == covered` is the informative case: every
    comparable day the answer key can speak for carries work on the route, so the
    remedy is more coverage rather than a different comparison. That is the state
    all fifteen ungraded windows were in on 2026-08-17, and it is why recurring
    work is not a defect in the control definition — `_is_control` admits a free
    day untouched the moment the archive holds one.
    """
    blackout = [*(w for w in announced if w != window and (w.routes & window.routes))]
    blackout.append(window)
    days = sorted(set(coverage))
    certified = 0
    covered = 0
    nearest: tuple[int, date] | None = None
    for piece in split_by_local_day(window):
        want = _service_class(piece.start)
        own = datetime.fromtimestamp(piece.start, _ET).date()
        for day in days:
            start, end = _band_on(day, piece)
            if _service_class(start) != want:
                continue
            certified += 1
            # Free when ANY route the window names has an admissible instant in
            # the band -- `coverage_state` calls a window graded when ANY service
            # pairs, so blocking the day on one of two named routes would report
            # no reach for a grade that can run.
            if not any(
                _free_instant(start, end, route, blackout) for route in window.routes
            ):
                covered += 1
                continue
            lag = (day - own).days
            if nearest is None or abs(lag) < abs(nearest[0]):
                nearest = (lag, day)
    return ControlReach(
        day=nearest[1] if nearest else None,
        lag_days=nearest[0] if nearest else None,
        certified=certified,
        covered=covered,
    )


def unknown_types(windows: Iterable[Window]) -> set[str]:
    """Alert types this module classifies as neither service-changing nor
    headway-only.

    MTA adds alert types without notice, and the failure mode is silent: a new
    stop-changing type would be excluded from every grade and read as an absence
    of supply rather than a gap in this taxonomy. Surface it instead.
    """
    known = SERVICE_CHANGING | HEADWAY_ONLY
    return {w.alert_type for w in windows if w.alert_type not in known}


def overlaps(window: Window, start: int, end: int) -> bool:
    """Whether an announced period intersects the span the trace actually covers.

    The alert archive is 91 days deep and carries work announced up to 180 days
    out, while the trace is a rolling few. Grading every announced window against
    a span it does not touch would report hundreds of empty results and call the
    denominator supply.
    """
    return window.start <= end and (window.end == 0 or window.end >= start)


def _median_by_type(pairs: Sequence[tuple[str, float]]) -> dict[str, dict[str, Any]]:
    """Grouped medians, because the types are different experiments: a stretch
    closed for the weekend and an express dropped for a night are not one
    population and pooling them reports neither."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for alert_type, value in pairs:
        grouped[alert_type].append(value)
    return {
        alert_type: {
            "n_rows": len(values),
            "median": round(statistics.median(values), 4),
        }
        for alert_type, values in sorted(grouped.items())
    }


def main(argv: list[str] | None = None) -> int:
    """Grade the traversal measure against the work announced over the same span
    the trace covers, and report the supply that produced the grade."""
    parser = argparse.ArgumentParser(
        description="Grade segment traversal against announced planned work"
    )
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    args = parser.parse_args(argv)

    today = datetime.now(UTC).date()
    start_date = args.start_date or today - timedelta(days=1)
    end_date = args.end_date or today

    # Imported here, not at module scope: archive_read imports this module for
    # Window and windows_from_alerts, so a top-level import would be circular.
    from training.archive_read import load_traversals, load_windows

    loaded = load_traversals(start_date, end_date)
    traversals = loaded.rows
    print(f"traversals — {loaded.summary()}", file=sys.stderr)
    # This grade pools every window's rows into per-type medians
    # (`_median_by_type`), so a span crossing an extraction or feed boundary is
    # not one measurement. Same guard the other two graders apply.
    loaded.require_pooled("traversals")
    # Hard fail rather than an empty report: no trace means the archive or the
    # window is wrong, and every count below would read as "detected nothing".
    if not traversals:
        raise SystemExit(f"no traversals in {start_date}..{end_date}")

    span_start = min(t.at for t in traversals)
    span_end = max(t.at for t in traversals)

    loaded_windows = load_windows(start_date, end_date)
    windows = loaded_windows.rows
    print(f"windows — {loaded_windows.summary()}", file=sys.stderr)
    # The answer key is pooled across the span too — per-type medians here, a
    # per-type summary in segment_grade — so a window archive written by two
    # different parsers is two answer keys, not one. Guarded on the same terms as
    # the traversals above.
    loaded_windows.require_pooled("windows")
    live = [w for w in windows if overlaps(w, span_start, span_end)]
    print(
        f"{len(windows)} announced windows, {len(live)} over the trace span",
        file=sys.stderr,
    )

    # The days the answer key can speak for: absence of an announcement is
    # evidence of a free day only where we hold alert snapshots. A SET, not a
    # span — coverage can have holes, and reading it as first..last would certify
    # a day whose snapshots are missing. Unioned across archive and raw because
    # both are days we captured.
    key_days = set(loaded_windows.archived_days) | set(loaded_windows.raw_days)

    coverage: list[PatternShift] = []
    duration: list[Effect] = []
    states: Counter[str] = Counter()
    blocked: list[dict[str, object]] = []
    graded_duration = 0
    for window in live:
        shifts = pattern_shift(window, traversals, other_windows=live)
        effects = measure(window, traversals, other_windows=live)
        graded_duration += bool(effects)
        if window.gradeable:
            state = coverage_state(window, shifts, traversals, other_windows=live)
            states[state] += 1
            if state == NO_CONTROL_PERIOD and key_days:
                reach = control_reach(window, windows, key_days)
                blocked.append(
                    {
                        "alert_type": window.alert_type,
                        "routes": sorted(window.routes),
                        "start": window.start,
                        "end": window.end,
                        **asdict(reach),
                        "day": reach.day.isoformat() if reach.day else None,
                    }
                )
        coverage.extend(shifts)
        duration.extend(effects)

    print(
        json.dumps(
            {
                "source": {
                    "traversal_archived_days": [
                        d.isoformat() for d in loaded.archived_days
                    ],
                    "traversal_raw_days": [d.isoformat() for d in loaded.raw_days],
                    "window_archived_days": [
                        d.isoformat() for d in loaded_windows.archived_days
                    ],
                    # Reported per stream, never unioned: traversal days carry a
                    # feed version and window days cannot (alerts have no static
                    # feed dependence), so a combined set always looks mixed even
                    # when each stream is internally consistent.
                    "traversal_provenance": sorted(str(v) for v in loaded.versions),
                    "window_provenance": sorted(
                        str(v) for v in loaded_windows.versions
                    ),
                },
                "supply": {
                    "span": [span_start, span_end],
                    "announced": len(windows),
                    "announced_by_type": dict(
                        Counter(w.alert_type for w in windows).most_common()
                    ),
                    "unknown_types": sorted(unknown_types(windows)),
                    "over_span": len(live),
                    "over_span_by_type": dict(
                        Counter(w.alert_type for w in live).most_common()
                    ),
                    "over_span_gradeable": sum(1 for w in live if w.gradeable),
                    "graded_coverage": states[GRADED],
                    "coverage_no_paired_service": states[NO_PAIRED_SERVICE],
                    "coverage_no_control_period": states[NO_CONTROL_PERIOD],
                    # Of those, how many have no free comparable day anywhere the
                    # answer key can certify one. A window here is not waiting on
                    # a cleverer comparison; it is waiting on coverage.
                    "coverage_reach_unknown": sum(
                        1 for row in blocked if row["day"] is None
                    ),
                    "answer_key_days": len(key_days),
                    "graded_duration": graded_duration,
                },
                # Every window that reached NO_CONTROL_PERIOD, with how far the
                # record would have to reach to grade it. Without this the state
                # reads the same whether a control is a week out of reach or
                # cannot exist, which are opposite conclusions about whether
                # waiting helps.
                "coverage_blocked": blocked,
                "coverage": {
                    "vanished_by_type": _median_by_type(
                        [(r.alert_type, r.vanished) for r in coverage]
                    ),
                    "rows": [asdict(r) for r in coverage],
                },
                "duration": {
                    "effect_by_type": _median_by_type(
                        [(r.alert_type, r.effect) for r in duration]
                    ),
                    "rows": [asdict(r) for r in duration],
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
