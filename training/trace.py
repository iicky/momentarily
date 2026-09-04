"""Arrival and traversal reconstruction from the per-minute vehicle trace.

`archive/trace/<date>/<scheduled_at>.json` is a full census of every in-service
trip, one object per minute (worker/src/vehicles.ts deriveTrace). This module is
what reads it: consecutive snapshots are diffed per trip_id into arrivals, and
consecutive arrivals into station-to-station traversal times.

That measurement is the point of the whole trace. The 5-minute advance signal
cannot make it: at 5-minute polling the mean observed "move" spans 2.7 stations,
so its (from_stop, to_stop) pairs are multi-station jumps and the stations in
between are never observed at all.

Three properties of the raw stream the diff has to honour, all of them from
deriveTrace's own contract:

- stop_id means the stop a train is HEADING TO while in transit and the stop it
  is AT once stopped. One hop into station N therefore appears as
  (stop_id=N, stopped=false) and then (stop_id=N, stopped=true) — same stop,
  different status. The second is the arrival, and stop_seq is populated only
  then.
- vehicle_ts is the feed's own per-vehicle measurement time, which is finer than
  the 1-minute poll and is what makes arrival timing better than the cadence.
- Snapshots are keyed on the scheduled second and are idempotent, so a retried
  minute overwrites and there is at most one object per minute.
"""

from __future__ import annotations

import argparse
import itertools
import statistics
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from training.dwell import DwellSample
from training.gtfs_static import Timetable, load_timetable
from training.load_r2 import date_range, fetch_objects, list_keys
from training.r2_client import R2Config, load_config, make_client

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

# Censoring kinds on a Traversal. Named rather than boolean because the three
# are statistically different and the survival machinery only handles two of
# them (see to_dwell_samples).
EXACT = "exact"  # both ends observed, consecutive stop_seq
INTERVAL = "interval"  # an arrival was missed: n_hops covered in one span
RIGHT = "right"  # in progress when the trip was last seen


@dataclass(frozen=True)
class Arrival:
    """A trip's first snapshot stopped at a stop — the moment it got there,
    timed by the feed's own vehicle_ts where the feed supplied one.

    `departed_at` is the first sighting that is no longer stopped here, so the
    train left at some point in (last stopped sighting, departed_at]. None when
    the trip was never seen leaving.
    """

    trip_id: str
    route_id: str
    direction: str | None
    stop_id: str
    stop_seq: int | None
    at: int
    departed_at: int | None = None


@dataclass(frozen=True)
class Passing:
    """A trip's DEPARTURE from a stop, detected by the trip's reported stop_id
    moving off that stop rather than by a STOPPED_AT sighting.

    This is the transition-keyed measurement the live Worker uses
    (worker/src/headway.detectPassings and worker/src/crowding.ts): stop_id
    names the stop a train is heading to or standing at, so the poll at which a
    trip first reports a DIFFERENT stop is the poll by which it has served and
    left the previous one. Unlike an Arrival — the first STOPPED_AT sighting —
    this catches a train whose dwell was shorter than the 1-minute poll and so
    was never caught standing at the stop at all. A missed STOPPED_AT loses no
    datum for dwell/traversal work; for headways it MERGES two real gaps into
    one double-length reading, which is what keying on the transition fixes.

    `at` is stamped at the departing sighting — the first row reporting a stop
    other than this one — preferring that row's own vehicle_ts when it is close
    enough to the poll to be believable (via _passing_time), else the poll. This
    is the same stamp, and the same wild-clock clamp, the live worker's
    passingTime applies, so the two series agree on the departure TIME, not only
    the event. Only rows carrying a direction are considered, matching the live
    carry, which cannot place a directionless row at a reference cell.

    `stop_seq` is the best sequence seen while the trip was at this stop, which
    is None when it was only ever caught in transit toward it (stop_seq is
    populated only on a STOPPED_AT sighting). Headways do not read it; it is
    kept for parity with Arrival.
    """

    trip_id: str
    route_id: str
    direction: str | None
    stop_id: str
    stop_seq: int | None
    at: int


@dataclass(frozen=True)
class Traversal:
    """One trip's movement from one arrival to the next.

    `at` is the arrival at from_stop — when the span started. It is what lets a
    traversal be matched to a service day, and so to the timetable that was
    actually running.

    Two durations, because a 1-minute poll brackets rather than pins the hop:
    `seconds` is arrival to arrival, which includes the dwell at from_stop and
    is the quantity the baselines fit; `moving_seconds` runs from the first
    sighting that had left from_stop, which starts the clock late and is
    therefore a LOWER bound on travel. `moving_seconds` is None when the
    departure was never observed, which is three quarters of the time.

    `n_hops` is how many stops the span covered in the REALTIME feed's own stop
    sequence: 1 is a station-to-station traversal, more means an arrival went
    unobserved between polls and the individual hop times are only known to sum
    to `seconds`. It is not the timetable's opinion — a train that bypasses a
    station reports consecutive stops the schedule puts a station between, which
    is what gtfs_static.Span.n_hops is for. `to_stop`/`n_hops` are None when the
    traversal was still in progress the last time the trip was seen.
    """

    trip_id: str
    route_id: str
    direction: str | None
    from_stop: str
    to_stop: str | None
    at: int
    seconds: int
    moving_seconds: int | None
    n_hops: int | None
    censoring: str


@dataclass(frozen=True)
class TraceStats:
    """What the reconstruction saw and what it had to discard. Reported rather
    than logged: the discard counts are the honest denominator for anything
    fitted on the output."""

    n_bodies: int
    n_rows: int
    n_trips: int
    n_arrivals: int
    n_exact: int
    n_interval: int
    n_right: int
    n_backwards: int  # stop_seq went down or stood still between arrivals
    n_unknown_seq: int  # an arrival without stop_seq, so the span is unmeasurable
    # Traversals with a departure observed strictly before the next arrival, so
    # travel time is bounded below by something better than zero. Seeing the
    # train leave at the same moment it turns up at the next stop says nothing.
    n_with_moving_time: int


def _row_time(row: dict[str, Any], fallback: int) -> int:
    """The feed's own measurement time for this vehicle, or the poll's scheduled
    second when the feed omitted one."""
    ts = row.get("vehicle_ts")
    return int(ts) if ts else fallback


# A departing vehicle_ts this far from the poll is a frozen or running-ahead
# clock; trusting it would poison this passing and the next, so it is clamped to
# the poll. Matches the live worker's passingTime guard
# (worker/src/headway.passingTime) and training/headway.FEED_GAP_SECONDS, both
# 240 — the passing definition, not just the capture rate, is what converges.
PASSING_CLOCK_TOLERANCE_SECONDS = 240


def _passing_time(row: dict[str, Any], poll: int) -> int:
    """The departing sighting's own vehicle_ts when believable, else the poll —
    the same stamp the live worker's passingTime picks for a departure. Unlike
    _row_time (used by the arrival/traversal path) this rejects a wild clock, so
    the offline and live headway series agree on the stamp, not only the event."""
    ts = row.get("vehicle_ts")
    if ts and abs(int(ts) - poll) <= PASSING_CLOCK_TOLERANCE_SECONDS:
        return int(ts)
    return poll


# A trip absent from the trace for longer than this is treated as finished, and
# any later sighting of the same trip_id as a fresh run. NYCT reuses trip_ids
# (they encode origin time, route and direction, so the same id comes back the
# next day and can come back sooner), and without a break the reused id inherits
# the old one's position: a new run that starts stopped where the old one ended
# would have its first arrival swallowed as "still standing here", and a
# right-censored hop could be stretched across the whole gap. At a 1-minute
# cadence a live trip appears every minute, so ten missed polls is not a gap in
# observation, it is a different train.
TRIP_GAP_SECONDS = 600


@dataclass(frozen=True)
class _Run:
    """One continuous appearance of a trip_id in the trace."""

    trip_id: str
    arrivals: list[Arrival]
    last_stop: str
    last_stopped: bool
    last_at: int


def _runs_from_trace(bodies: list[dict[str, Any]]) -> list[_Run]:
    """Split the trace into per-trip runs and reconstruct each one's arrivals.

    An arrival is the FIRST snapshot of a consecutive stopped run at one stop: a
    train sits stopped for several polls, and only the first of them is the
    moment it arrived. The departure is the first snapshot after that which is no
    longer stopped there — either in transit to the next stop, or already stopped
    at it when the poll cadence missed the trip in between.
    """
    # trip_id -> (arrivals so far, last (stop_id, stopped), last poll, last time,
    # index of an arrival still awaiting its departure)
    open_runs: dict[str, _RunState] = {}
    done: list[_Run] = []

    for body in sorted(bodies, key=lambda b: int(b.get("scheduled_at") or 0)):
        poll = int(body.get("scheduled_at") or body.get("observed_at") or 0)
        for raw in cast(list[Any], body.get("rows") or []):
            if not isinstance(raw, dict):
                continue
            row = cast(dict[str, Any], raw)
            trip_id = str(row.get("trip_id") or "")
            stop_id = str(row.get("stop_id") or "")
            if not trip_id or not stop_id:
                continue
            stopped = bool(row.get("stopped"))
            at = _row_time(row, poll)

            state = open_runs.get(trip_id)
            if state is not None and poll - state.last_poll > TRIP_GAP_SECONDS:
                done.append(state.close())
                state = None
            if state is None:
                state = _RunState(trip_id)
                open_runs[trip_id] = state
            state.observe(row, stop_id, stopped, at, poll)

    done.extend(state.close() for state in open_runs.values())
    return [run for run in done if run.arrivals]


class _RunState:
    """Mutable accumulator for one run; frozen into a _Run when it ends."""

    def __init__(self, trip_id: str) -> None:
        self.trip_id = trip_id
        self.arrivals: list[Arrival] = []
        self.last: tuple[str, bool] | None = None
        self.last_poll = 0
        self.last_at = 0
        self.pending: int | None = None

    def observe(
        self, row: dict[str, Any], stop_id: str, stopped: bool, at: int, poll: int
    ) -> None:
        previous = self.last
        self.last = (stop_id, stopped)
        self.last_poll = poll
        self.last_at = at

        if self.pending is not None:
            waiting = self.arrivals[self.pending]
            still_here = stopped and stop_id == waiting.stop_id
            if not still_here and at > waiting.at:
                self.arrivals[self.pending] = replace(waiting, departed_at=at)
                self.pending = None

        if not stopped or previous == (stop_id, True):
            return  # in transit, or still standing where it already arrived
        seq = row.get("stop_seq")
        self.arrivals.append(
            Arrival(
                trip_id=self.trip_id,
                route_id=str(row.get("route_id") or ""),
                direction=cast(str | None, row.get("direction")),
                stop_id=stop_id,
                stop_seq=None if seq is None else int(cast(int, seq)),
                at=at,
            )
        )
        self.pending = len(self.arrivals) - 1

    def close(self) -> _Run:
        stop_id, stopped = self.last or ("", False)
        return _Run(
            trip_id=self.trip_id,
            arrivals=self.arrivals,
            last_stop=stop_id,
            last_stopped=stopped,
            last_at=self.last_at,
        )


def arrivals_from_trace(bodies: list[dict[str, Any]]) -> list[Arrival]:
    """Every (trip, stop) arrival in the window, in time order, each carrying the
    first sighting that had left it again. See _runs_from_trace."""
    out = [a for run in _runs_from_trace(bodies) for a in run.arrivals]
    out.sort(key=lambda a: a.at)
    return out


@dataclass(frozen=True)
class CatchStats:
    """How the transition-keyed reconstruction relates to the old STOPPED_AT
    rule, over one window. `caught` is the count of departures whose immediately
    preceding sighting had the trip STOPPED_AT the departing stop — the ones the
    old arrival-keyed series also saw. The two `missed_*` are what it recovers:
    a sub-poll dwell never caught standing at all, and the rarer case of a trip
    seen standing then moving-off the same stop before the next poll. This is
    the honest denominator behind the "~7% of headways were doubles" claim —
    every missed departure merged two real headways into one."""

    n_transitions: int
    caught: int
    missed_never_stopped: int
    missed_stopped_then_moved: int

    @property
    def missed(self) -> int:
        return self.missed_never_stopped + self.missed_stopped_then_moved

    @property
    def caught_fraction(self) -> float:
        return self.caught / self.n_transitions if self.n_transitions else 0.0


class _PassState:
    """Mutable per-trip accumulator for transition-keyed passings.

    Tracks the stop a trip is currently reporting; a change to a different stop
    is the departure from the previous one. Mirrors the live carry
    (worker/src/headway.updateHeadwayState): the point of interest is where the
    trip is now, and the passing fires the moment that moves. Alongside the
    passings it tallies whether each departure would also have been caught by
    the old STOPPED_AT rule — see CatchStats."""

    def __init__(self) -> None:
        self.stop: str | None = None
        self.route = ""
        self.direction: str | None = None
        self.seq: int | None = None
        self.last_poll = 0
        self.last_stopped = False  # most recent sighting at self.stop
        self.stopped_seen = False  # any sighting at self.stop was STOPPED_AT
        self.passings: list[Passing] = []
        self.caught = 0
        self.missed_never_stopped = 0
        self.missed_stopped_then_moved = 0

    def observe(
        self,
        row: dict[str, Any],
        stop_id: str,
        stopped: bool,
        at: int,
        poll: int,
    ) -> None:
        # A long absence is a different train (TRIP_GAP_SECONDS, same reasoning
        # as _runs_from_trace): the pending departure is dropped unfired, which
        # is what the live carry does when a stale trip expires.
        if self.stop is not None and poll - self.last_poll > TRIP_GAP_SECONDS:
            self.stop = None
        self.last_poll = poll

        if self.stop is None:
            self._enter(row, stop_id, stopped)
            return
        if stop_id != self.stop:
            # Reporting a new stop: it served and left the previous one.
            self.passings.append(
                Passing(
                    trip_id=str(row.get("trip_id") or ""),
                    route_id=self.route,
                    direction=self.direction,
                    stop_id=self.stop,
                    stop_seq=self.seq,
                    at=at,
                )
            )
            if self.last_stopped:
                self.caught += 1
            elif self.stopped_seen:
                self.missed_stopped_then_moved += 1
            else:
                self.missed_never_stopped += 1
            self._enter(row, stop_id, stopped)
            return
        # Still at the same stop; keep the sharper stop_seq if this poll has one.
        self.last_stopped = stopped
        if stopped:
            self.stopped_seen = True
            if self.seq is None:
                seq = row.get("stop_seq")
                self.seq = None if seq is None else int(cast(int, seq))

    def _enter(self, row: dict[str, Any], stop_id: str, stopped: bool) -> None:
        self.stop = stop_id
        self.route = str(row.get("route_id") or "")
        self.direction = cast(str | None, row.get("direction"))
        seq = row.get("stop_seq") if stopped else None
        self.seq = None if seq is None else int(cast(int, seq))
        self.last_stopped = stopped
        self.stopped_seen = stopped


def passings_and_catch_stats(
    bodies: list[dict[str, Any]],
) -> tuple[list[Passing], CatchStats]:
    """The transition-keyed passings plus the CatchStats that say how many of
    them the old STOPPED_AT rule would have missed. One walk, so the fraction
    and the series it produces are computed from the same data."""
    open_trips: dict[str, _PassState] = {}
    for body in sorted(bodies, key=lambda b: int(b.get("scheduled_at") or 0)):
        poll = int(body.get("scheduled_at") or body.get("observed_at") or 0)
        for raw in cast(list[Any], body.get("rows") or []):
            if not isinstance(raw, dict):
                continue
            row = cast(dict[str, Any], raw)
            trip_id = str(row.get("trip_id") or "")
            stop_id = str(row.get("stop_id") or "")
            if not trip_id or not stop_id:
                continue
            # A null-direction row cannot be placed at a directional reference
            # cell, so the live carry skips it (worker/src/headway.detectPassings
            # and updateHeadwayState both continue on row.direction === null);
            # match that here rather than firing a directionless passing.
            if row.get("direction") is None:
                continue
            state = open_trips.get(trip_id)
            if state is None:
                state = _PassState()
                open_trips[trip_id] = state
            state.observe(
                row, stop_id, bool(row.get("stopped")), _passing_time(row, poll), poll
            )
    out = [p for state in open_trips.values() for p in state.passings]
    # (at, then trip) — the live detectPassings tie-break, so two trains that
    # clear a stop in the same second order identically offline and on the Worker.
    out.sort(key=lambda p: (p.at, p.trip_id))
    states = open_trips.values()
    stats = CatchStats(
        n_transitions=sum(len(s.passings) for s in states),
        caught=sum(s.caught for s in states),
        missed_never_stopped=sum(s.missed_never_stopped for s in states),
        missed_stopped_then_moved=sum(s.missed_stopped_then_moved for s in states),
    )
    return out, stats


def passings_from_trace(bodies: list[dict[str, Any]]) -> list[Passing]:
    """Every (trip, stop) DEPARTURE in the window, in time order — the
    transition-keyed measurement that converges the offline headway series with
    the live Worker's (worker/src/headway.detectPassings).

    A passing is emitted for every stop a trip serves EXCEPT the last one it was
    seen at, whose departure was never observed — exactly like the live carry,
    which holds a trip at its reference stop until it reports the next one. The
    stop a train skips is never reported as its stop_id, so an express is not
    credited with passing a stop it ran through. See Passing and
    passings_and_catch_stats."""
    return passings_and_catch_stats(bodies)[0]


def traversals_from_trace(
    bodies: list[dict[str, Any]],
) -> tuple[list[Traversal], TraceStats]:
    """Per-(trip, hop) traversal durations with their censoring kind.

    A right-censored record needs the trip to have been IN TRANSIT the last time
    it was seen, heading to a stop other than the one it last arrived at: only
    then is "this hop had run at least this long" a true statement. Two nearby
    situations are not censoring and must not be recorded as it — a trip last
    seen standing where it arrived may simply have finished its run, and a trip
    last seen standing somewhere FURTHER ALONG has already completed the hop
    (we just couldn't time the arrival, e.g. the feed omitted stop_seq). Calling
    either "still running at T" is false, not merely unverified, and biases
    every fit upward.

    A feed gap after that last sighting does not weaken the bound: the train
    demonstrably was still in transit when we last saw it, and the endpoint is
    the feed's own vehicle_ts rather than a poll time. A gap that swallows an
    arrival shows up instead as a stop_seq jump, which is recorded as
    interval-censored. A gap long enough to be a different train entirely
    (TRIP_GAP_SECONDS) ends the run, so nothing is carried across it.
    """
    runs = _runs_from_trace(bodies)
    out: list[Traversal] = []
    backwards = unknown_seq = 0
    n_exact = n_interval = n_right = 0
    n_arrivals = sum(len(run.arrivals) for run in runs)

    for run in runs:
        trip_id, seen = run.trip_id, run.arrivals
        for prev, nxt in itertools.pairwise(seen):
            if prev.stop_seq is None or nxt.stop_seq is None:
                unknown_seq += 1
                continue
            hops = nxt.stop_seq - prev.stop_seq
            if hops <= 0:
                backwards += 1
                continue
            seconds = nxt.at - prev.at
            if seconds <= 0:
                backwards += 1
                continue
            censoring = EXACT if hops == 1 else INTERVAL
            if hops == 1:
                n_exact += 1
            else:
                n_interval += 1
            moving = (
                None
                if prev.departed_at is None or nxt.at <= prev.departed_at
                else nxt.at - prev.departed_at
            )
            out.append(
                Traversal(
                    trip_id=trip_id,
                    route_id=prev.route_id,
                    direction=prev.direction,
                    from_stop=prev.stop_id,
                    to_stop=nxt.stop_id,
                    at=prev.at,
                    seconds=seconds,
                    moving_seconds=moving,
                    n_hops=hops,
                    censoring=censoring,
                )
            )

        final = seen[-1]
        at = run.last_at
        if run.last_stopped or run.last_stop == final.stop_id or at <= final.at:
            continue  # not in transit toward a further stop when last seen
        out.append(
            Traversal(
                trip_id=trip_id,
                route_id=final.route_id,
                direction=final.direction,
                from_stop=final.stop_id,
                to_stop=None,
                at=final.at,
                seconds=at - final.at,
                moving_seconds=(
                    None
                    if final.departed_at is None or at <= final.departed_at
                    else at - final.departed_at
                ),
                n_hops=None,
                censoring=RIGHT,
            )
        )
        n_right += 1

    stats = TraceStats(
        n_bodies=len(bodies),
        n_rows=sum(len(cast(list[Any], b.get("rows") or [])) for b in bodies),
        n_trips=len({run.trip_id for run in runs}),
        n_arrivals=n_arrivals,
        n_exact=n_exact,
        n_interval=n_interval,
        n_right=n_right,
        n_backwards=backwards,
        n_unknown_seq=unknown_seq,
        n_with_moving_time=sum(1 for t in out if t.moving_seconds is not None),
    )
    return out, stats


def to_dwell_samples(traversals: list[Traversal]) -> list[DwellSample]:
    """(duration, completed) pairs for the survival machinery in dwell.py /
    survival.py, timed departure to arrival.

    Records without an observed departure are dropped rather than falling back
    to arrival-to-arrival: that would fold the origin's station dwell into
    travel time, which is a different quantity and a systematic overstatement.

    Interval-censored spans are dropped too, and not approximated. Those fitters
    carry a right-censored likelihood only, and a multi-hop span says each hop
    took at most the total, which is an upper bound they cannot express —
    splitting it evenly would fabricate observations. TraceStats.n_interval and
    n_with_moving_time size what this discards; check both before trusting a fit.
    """
    return [
        (t.moving_seconds, t.censoring == EXACT)
        for t in traversals
        if t.censoring in (EXACT, RIGHT) and t.moving_seconds is not None
    ]


def fetch_trace_bodies(
    config: R2Config | None = None,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
    client: S3Client | None = None,
) -> list[dict[str, Any]]:
    """Every archived trace snapshot in the window (one object per minute, so
    ~1440 per day against archive/vehicles' 288). Mirrors
    load_r2.fetch_vehicle_metrics, including its yesterday-through-today
    default."""
    cfg = config or load_config()
    client = client or make_client(cfg)
    today = datetime.now(UTC).date()
    start = start_date or (today - timedelta(days=1))
    end = end_date or today
    keys: list[str] = []
    for d in date_range(start, end):
        keys.extend(list_keys(client, cfg.bucket, f"archive/trace/{d.isoformat()}/"))
    return fetch_objects(client, cfg.bucket, keys)


def _ratio_stats(pairs: list[tuple[int, int]], unmatched: int) -> dict[str, Any]:
    if not pairs:
        return {"n": 0, "unmatched": unmatched}
    ratios = sorted(o / s for o, s in pairs)
    return {
        "n": len(pairs),
        "unmatched": unmatched,
        "observed_median": statistics.median([o for o, _s in pairs]),
        "scheduled_median": statistics.median([s for _o, s in pairs]),
        "ratio_median": round(statistics.median(ratios), 3),
        "ratio_p10": round(ratios[len(ratios) // 10], 3),
        "ratio_p90": round(ratios[int(len(ratios) * 0.9)], 3),
    }


@dataclass(frozen=True)
class Scheduled:
    """What the timetable allows for one observed traversal.

    `n_hops` is how many of the trip's OWN scheduled stops the span covers, and
    it is None when the trip could not be matched to a stopping pattern and the
    service day's median for the observed pair was used instead. A traversal the
    realtime feed calls one hop but whose pattern says two is a bypass: the train
    skipped a station, and treating it as a direct hop is what makes a disrupted
    train read as a fast one.
    """

    seconds: int
    n_hops: int | None


def scheduled_for(traversal: Traversal, timetable: Timetable) -> Scheduled | None:
    """What the timetable allowed for one completed traversal, from the trip's own
    scheduled stops where the static feed names the trip, and from the service
    day's median for that pair otherwise.

    None when no honest comparison exists, which is two different situations and
    both are reported by callers rather than papered over: a hop the timetable
    never scheduled at all (1.2% of live single hops, concentrated in the thin
    keys), or a traversal from outside the loaded feed's validity window.

    The window guard lives HERE, not in each caller. The vehicle archive reaches
    back further than any one GTFS snapshot, so every comparison in the repo is
    one replay away from measuring trains against a schedule that was not in
    force when they ran.
    """
    if traversal.to_stop is None:
        return None
    if not timetable.covers(traversal.at, traversal.trip_id):
        return None
    day = timetable.day_for(traversal.at, traversal.trip_id)
    span = day.span(traversal.trip_id, traversal.from_stop, traversal.to_stop)
    if span is not None:
        return Scheduled(seconds=span.seconds, n_hops=span.n_hops)
    key = (
        traversal.route_id,
        traversal.direction or "",
        traversal.from_stop,
        traversal.to_stop,
    )
    want = day.hops.get(key)
    return None if not want or want <= 0 else Scheduled(seconds=want, n_hops=None)


def schedule_comparison(
    traversals: list[Traversal], timetable: Timetable
) -> dict[str, Any]:
    """Observed single-hop times against what the timetable allowed the same
    trip, both ways round.

    The reference is arrival-to-arrival, so `arrival_to_arrival` is the like-for-
    like reading. `departure_to_arrival` is kept because the two bracket the
    truth from opposite sides at a 1-minute cadence: it starts its clock at the
    first sighting that had already left, which is late, and covers only the
    quarter of hops slow enough to be caught in transit at all.

    Ratios, not the marginal medians: the two cuts cover different hops.

    `outside_feed_window` is split out of `unmatched` because the two mean
    opposite things: a hop the timetable never scheduled is a fact about the
    network, while a traversal from outside the feed's validity window means this
    whole report is measuring against the wrong timetable. Pooled into one
    counter, the second would look like the first.
    """
    arrival_pairs: list[tuple[int, int]] = []
    moving_pairs: list[tuple[int, int]] = []
    unmatched = outside = 0
    for t in traversals:
        if t.censoring != EXACT or t.to_stop is None:
            continue
        if not timetable.covers(t.at, t.trip_id):
            outside += 1
            continue
        want = scheduled_for(t, timetable)
        if want is None:
            unmatched += 1
            continue
        arrival_pairs.append((t.seconds, want.seconds))
        if t.moving_seconds is not None:
            moving_pairs.append((t.moving_seconds, want.seconds))
    return {
        "feed_version": timetable.version.version,
        "outside_feed_window": outside,
        "arrival_to_arrival": _ratio_stats(arrival_pairs, unmatched),
        "departure_to_arrival": _ratio_stats(moving_pairs, unmatched),
    }


def main(argv: list[str] | None = None) -> int:
    """Reconstruct a window and report what it yields, against the timetable."""
    parser = argparse.ArgumentParser(description="Trace reconstruction report")
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    args = parser.parse_args(argv)

    today = datetime.now(UTC).date()
    bodies = fetch_trace_bodies(
        start_date=args.start_date or today, end_date=args.end_date or today
    )
    traversals, stats = traversals_from_trace(bodies)
    total = stats.n_exact + stats.n_interval + stats.n_right
    print(
        f"{stats.n_bodies} snapshots, {stats.n_rows} rows, {stats.n_trips} trips, "
        f"{stats.n_arrivals} arrivals"
    )
    print(
        f"{total} traversals: {stats.n_exact} exact, {stats.n_interval} "
        f"interval-censored, {stats.n_right} right-censored; "
        f"{stats.n_with_moving_time} with a departure observed in time to bound "
        f"travel; dropped {stats.n_backwards} backwards, {stats.n_unknown_seq} "
        f"without stop_seq"
    )
    comparison = schedule_comparison(traversals, load_timetable())
    for label, cut in comparison.items():
        print(f"vs the timetable, {label}: {cut}")
    samples = to_dwell_samples(traversals)
    events = sum(1 for _d, completed in samples if completed)
    print(
        f"{len(samples)} survival samples ({events} events, {len(samples) - events} censored)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
