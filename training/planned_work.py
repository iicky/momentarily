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

import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime
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


def measure(
    window: Window,
    traversals: Iterable[Traversal],
    *,
    other_windows: Iterable[Window] = (),
) -> Effect | None:
    """The SECONDARY, weaker grade: did the segments beside the work slow down?

    Not a test of the named segments themselves. A closure or a skip removes their
    traversals entirely — that absence is what `pattern_shift` measures, and it is
    the primary claim. What survives for a duration test is spillover: a hop with
    ONE endpoint in the named set still runs, and is where single-tracking,
    merging and reduced frequency show up as time.

    Difference-in-differences, from movement alone, because a raw before/after on
    the boundary hops cannot tell planned work from the evening rush. Dividing by
    what the same route's distant segments did over the same clock cancels
    everything the two arms share — which is also why this one does not need the
    day-matched control `pattern_shift` does.

    `other_windows` is excluded from the OUTSIDE arm on the same route: a second
    closure sitting in the control period would be compared against as though it
    were normal service, dragging the baseline toward the disrupted state and
    shrinking whatever effect exists. Overlapping planned work is the norm.

    None when the type cannot be seen in traversals at all, or when neither arm
    clears MIN_SIDE_SAMPLES on enough segments. Neither is a failed detection.
    """
    if not window.gradeable:
        return None
    blackout = [w for w in other_windows if w != window]
    affected: dict[HopKey, tuple[list[int], list[int]]] = {}
    control: dict[HopKey, tuple[list[int], list[int]]] = {}
    for t in traversals:
        if t.censoring != EXACT or t.to_stop is None or t.n_hops != 1:
            continue
        if not window.covers_route(t.route_id):
            continue
        key: HopKey = (t.route_id, t.direction or "", t.from_stop, t.to_stop)
        if window.at_boundary(key):
            bucket = affected
        elif window.touches(key):
            continue  # inside the closed stretch: it vanishes, not slows
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


def pattern_shift(
    window: Window,
    traversals: Iterable[Traversal],
    *,
    other_windows: Iterable[Window] = (),
) -> list[PatternShift]:
    """One PatternShift per service running on the window's routes, from movement
    alone. Empty when the window is not a coverage intervention, or when no
    matched control period exists in the data yet.

    Services are recovered from the trip id, because the archived route_id has
    already folded 7X into 7 and a folded aggregate cannot see an express-to-local
    change at all. Each service is reported on its own so the express's vanishing
    pairs are not averaged against the local's surviving ones.

    THE CONTROL ARM IS THE SAME HOURS ON A COMPARABLE DAY, not the rest of the
    same day, and that is not a refinement — it is what makes the measure mean
    anything. Two ways the obvious comparison lies:

      * Express service only runs at rush hour, which is exactly when this work
        is scheduled. Measured 2026-08-13 against a 15:00-22:00 window, the 7X had
        13 traversals outside it against 790 inside, so "outside" really meant
        "when no express runs" and every pair would read as appearing from
        nowhere.
      * The 7X does not run at all on a weekend, so a Saturday control against a
        weekday window would manufacture the entire signal.

    Unlike `measure`, this has no second arm to cancel time of day against, so the
    comparison has to be built matched rather than corrected afterwards.
    """
    if not window.gradeable:
        return []
    blackout = [w for w in other_windows if w != window]
    want_class = _service_class(window.start)
    inside: dict[str, set[HopKey]] = defaultdict(set)
    outside: dict[str, set[HopKey]] = defaultdict(set)
    n_in: Counter[str] = Counter()
    n_out: Counter[str] = Counter()
    for t in traversals:
        if t.censoring != EXACT or t.to_stop is None or t.n_hops != 1:
            continue
        if not window.covers_route(t.route_id):
            continue
        service = t.route_id + "X" if is_express_variant(t.trip_id) else t.route_id
        key: HopKey = (service, t.direction or "", t.from_stop, t.to_stop)
        if window.contains(t.at):
            inside[service].add(key)
            n_in[service] += 1
            continue
        if _service_class(t.at) != want_class or not _matches_clock(t.at, window):
            continue
        if any(w.contains(t.at) and w.covers_route(t.route_id) for w in blackout):
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


def unknown_types(windows: Iterable[Window]) -> set[str]:
    """Alert types this module classifies as neither service-changing nor
    headway-only.

    MTA adds alert types without notice, and the failure mode is silent: a new
    stop-changing type would be excluded from every grade and read as an absence
    of supply rather than a gap in this taxonomy. Surface it instead.
    """
    known = SERVICE_CHANGING | HEADWAY_ONLY
    return {w.alert_type for w in windows if w.alert_type not in known}
