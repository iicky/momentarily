"""Trip matching: which scheduled trip each run is, and its service day (§2.3,
§2.4, decision 5).

Every tier needs a candidate service day (D−1, D, D+1 around the local date of
the run's first row) whose match index holds the trip, so the trip's service
is active then. The first tier that hits wins:

- **tiers 1–4** (``exact_id``, ``path_prefix``, ``line_code``,
  ``path_differs``): the realtime id's (origin, route field, direction), also
  with the origin + 24 h (a wrapped id belongs to the day before the
  observation), and the path equal, a prefix either way, the same two-letter
  line code, or different. The time check is at the first common stop among
  the run's first ``BLOCK_LOOKAHEAD`` stop blocks: within ``ID_TOL_S`` for
  tiers 1–3 and ``PATH_TOL_S`` for tier 4 (which needs a common stop); with
  no common stop, the trip's first departure against the run's first row.
  Among hits: the lowest tier, then the smallest offset, then trip id and
  day. A trip whose first departure does not exist on the wall clock
  (spring forward) has no times: the id tiers 1–3 still match it, untimed.
  An origin one hundredth past a scheduled one in 01:00–01:59 is NYCT's
  marker for the repeated fall-back hour: it also names origin − 1, timed
  at the second occurrence (``REPEATED_HOUR_S`` later);
- **tier 5** (``drift``): runs with no id hit, through one min-cost
  one-to-one assignment (decision 5, revised 2026-10-22). A run's candidates
  are the trips of its compatible patterns within ``DRIFT_TOL_S`` at the
  first common stop **after the pattern origin** (trains wait at the
  origin), or at the origin when no later stop of the pattern is among the
  run's first blocks (the rule decision 5 was measured with), the trips
  id-tier runs hold left out. The cost is the absolute
  offset, and leaving a run unassigned costs ``DRIFT_TOL_S``. Equal total
  costs resolve to the assignment whose trips come first in (trip id, day)
  order. The assignment is solved per connected component of the
  candidate graph, over inputs sorted by run id and trip, so a rebuild gives
  the same result.

A run with no trip is ``unscheduled`` (or ``unparseable``), with its miss
class and the nearest candidate within ``NEAR_S``: ``drift_unassigned`` (it
had tier-5 candidates but lost the assignment or was left out),
``unknown_route``, ``service_day_mismatch`` (its id names trips only more
than ``WRONG_DAY_S`` away), ``id_time_off``, ``path_code_differs``,
``route_field_differs``, ``unscheduled`` or ``unparseable_id``.

Service day: the matched trip's. Unmatched with a parseable id: the old
``service_dates`` rule (``id_origin``), flagged ``service_date_ambiguous``
with the other date when first seen 00:00–04:00 local; unparseable: the
04:00 clock rule.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Literal, cast
from zoneinfo import ZoneInfo

from momentarily.gtfs.static import base_route, service_dates
from momentarily.model.conventions import Unknown, wall_instant_of
from momentarily.model.events.contracts import (
    LINEAGE_SOURCE_BASIS,
    TripMatch,
)
from momentarily.model.events.match_index import (
    IndexTrip,
    MatchIndex,
    candidate_service_dates,
)
from momentarily.model.events.runs import BLOCK_LOOKAHEAD, Block, Run
from momentarily.model.events.trip_ids import TripIdKey

MATCH_VERSION = 1
"""Builder version of TripMatch and SupplyObservation files."""

ID_TOL_S = 7200
PATH_TOL_S = 1800
DRIFT_TOL_S = 900
NEAR_S = 3600
WRONG_DAY_S = 21600
NIGHT_END_HOUR = 4
"""Unmatched runs first seen before this local hour get service_date_ambiguous;
the clock rule cuts service days here."""
REPEATED_HOUR_S = 3600
"""The repeated hour of a fall-back night: a marked trip runs this much later
than its scheduled time's first occurrence."""
REPEATED_HOUR_FROM, REPEATED_HOUR_TO = 6000, 12000
"""01:00 and 02:00 as trip-id origins (hundredths of a minute)."""

Tier = Literal["exact_id", "path_prefix", "line_code", "path_differs", "drift"]
TIERS: tuple[Tier, ...] = (
    "exact_id",
    "path_prefix",
    "line_code",
    "path_differs",
    "drift",
)

IndexFor = Callable[[str], MatchIndex | None]
"""The match index of a service date, or None when there is none."""


@dataclass(frozen=True)
class Assignment:
    trip: IndexTrip
    service_date: str
    tier: Tier
    offset_s: int | None
    wrapped: bool
    build_id: str


@dataclass(frozen=True)
class _Candidate:
    cost: int
    trip_id: str
    service_date: str
    offset: int


@dataclass
class _Miss:
    miss_class: str
    nearest: str | None


def _path_tier(static: str, realtime: str) -> int:
    if static == realtime:
        return 0
    if static.startswith(realtime) or realtime.startswith(static):
        return 1
    if static[:2] == realtime[:2]:
        return 2
    return 3


def _first_common(
    ix: MatchIndex, trip_id: str, blocks: Sequence[Block]
) -> tuple[int, int] | None:
    for b in blocks[:BLOCK_LOOKAHEAD]:
        a = ix.arrival(trip_id, b.stop_id)
        if a is not None:
            return b.time, a
    return None


def _repeated_hour_marker(key: TripIdKey) -> bool:
    """A realtime origin one hundredth of a minute past a scheduled origin
    (``006001`` for ``006000``) in 01:00–01:59: NYCT's marker for a trip in
    the repeated hour of a fall-back night. Scheduled origins are multiples
    of 50 (half minutes); this residue was seen only on 2026-11-01."""
    base = key.origin - 1
    return key.origin % 50 == 1 and REPEATED_HOUR_FROM <= base < REPEATED_HOUR_TO


class Matcher:
    """Matches runs of one input window against the indexes of their
    candidate service days."""

    def __init__(self, index_for: IndexFor, tz: str, withheld: bool = False) -> None:
        self.index_for = index_for
        self.tz = tz
        self.withheld = withheld
        """Withheld-input mode (§10 slice 4 evaluation): the id tiers are
        skipped and the route is the trip-id field alone."""
        self._fits: dict[tuple[str, str, tuple[str, ...]], bool] = {}

    # -- id tiers ----------------------------------------------------------------
    def _id_tiers(
        self, run: Run, key: TripIdKey
    ) -> tuple[Assignment | None, bool, int | None]:
        """The best id-tier hit; whether the id names any trip of a
        candidate day; the smallest offset of a tier 1–3 hit rejected on
        time."""
        blocks = run.blocks()
        best: tuple[tuple[object, ...], Assignment] | None = None
        any_key = False
        rejected: int | None = None
        keys: list[tuple[TripIdKey, int]] = [(key, 0)]
        if _repeated_hour_marker(key):
            keys.append((replace(key, origin=key.origin - 1), REPEATED_HOUR_S))
        for day in candidate_service_dates(run.first_at, self.tz):
            ix = self.index_for(day)
            if ix is None:
                continue
            for lookup, shift in keys:
                for hit in ix.realtime_key(lookup):
                    any_key = True
                    t = hit.trip
                    tier = _path_tier(t.path, key.path)
                    fc = _first_common(ix, t.trip_id, blocks)
                    off: int | None = None
                    if fc is not None:
                        off = fc[0] - fc[1] - shift
                        rank = abs(off)
                    elif t.first_departure is not None:
                        rank = abs(run.first_at - t.first_departure - shift)
                    else:
                        rank = 1 << 40
                    tol = PATH_TOL_S if tier == 3 else ID_TOL_S
                    timed = off is not None or t.first_departure is not None
                    if timed and rank > tol:
                        if tier < 3 and off is not None:
                            rejected = rank if rejected is None else min(rejected, rank)
                        continue
                    if tier == 3 and off is None:
                        continue
                    order = (tier, shift, rank, t.trip_id, day, hit.wrapped)
                    if best is None or order < best[0]:
                        best = (
                            order,
                            Assignment(
                                t, day, TIERS[tier], off, hit.wrapped, ix.build_id
                            ),
                        )
        return (None if best is None else best[1]), any_key, rejected

    # -- tier 5 ------------------------------------------------------------------
    def _fits_pattern(
        self, ix: MatchIndex, pattern_id: str, stops: tuple[str, ...]
    ) -> bool:
        k = (ix.build_id, pattern_id, stops)
        hit = self._fits.get(k)
        if hit is None:
            last, hit = -1, False
            for s in stops:
                if s not in ix.platforms:
                    continue
                i = ix.stop_position(pattern_id, s)
                if i is None or i <= last:
                    hit = False
                    break
                last, hit = i, True
            self._fits[k] = hit
        return hit

    def drift_candidates(self, run: Run, tol: int) -> list[_Candidate]:
        """Scheduled trips of the run's compatible patterns within ``tol`` at
        the first common stop after the pattern origin, best per (trip, day)."""
        key = run.parsed
        if key is None:
            return []
        blocks = run.blocks()[:BLOCK_LOOKAHEAD]
        stops = run.stops()
        keys = run.keys(self.withheld)
        best: dict[tuple[str, str], _Candidate] = {}
        for day in candidate_service_dates(run.first_at, self.tz):
            ix = self.index_for(day)
            if ix is None:
                continue
            pids: set[str] = set()
            for rk in keys:
                pids.update(ix.patterns_of(rk, key.direction))
            for pid in sorted(pids):
                if not self._fits_pattern(ix, pid, stops):
                    continue
                anchor = next(
                    (b for b in blocks if (ix.stop_position(pid, b.stop_id) or 0) > 0),
                    None,
                ) or next(
                    (b for b in blocks if ix.stop_position(pid, b.stop_id) is not None),
                    None,
                )
                if anchor is None:
                    continue
                st = ix.stop_times(pid, anchor.stop_id)
                lo = bisect_left(st.instants, anchor.time - tol)
                hi = bisect_right(st.instants, anchor.time + tol)
                for a, trip_id in zip(
                    st.instants[lo:hi], st.trip_ids[lo:hi], strict=True
                ):
                    c = _Candidate(abs(anchor.time - a), trip_id, day, anchor.time - a)
                    k = (trip_id, day)
                    if k not in best or (c.cost, c.offset) < (
                        best[k].cost,
                        best[k].offset,
                    ):
                        best[k] = c
        return sorted(best.values(), key=lambda c: (c.cost, c.trip_id, c.service_date))

    def _miss(
        self, run: Run, key: TripIdKey, any_key: bool, rejected: int | None
    ) -> str:
        keys = run.keys(self.withheld)
        ix = self.index_for(candidate_service_dates(run.first_at, self.tz)[1])
        if ix is not None and not any(ix.patterns_of(k, key.direction) for k in keys):
            return "unknown_route"
        if any_key and rejected is not None and rejected > WRONG_DAY_S:
            return "service_day_mismatch"
        if any_key and rejected is not None:
            return "id_time_off"
        if any_key:
            return "path_code_differs"
        if ix is not None and not self.withheld:
            folded = set(keys)
            for t in ix.origin_direction(key.origin, key.direction):
                if base_route(t.route_id) in folded:
                    return "route_field_differs"
        return "unscheduled"

    # -- all runs ----------------------------------------------------------------
    def match(
        self, runs: Sequence[Run]
    ) -> tuple[dict[str, Assignment], dict[str, _Miss], set[str]]:
        """Assignments and misses by run id, and the run ids that share an
        id-tier trip with another run."""
        assigned: dict[str, Assignment] = {}
        misses: dict[str, _Miss] = {}
        pending: list[tuple[Run, TripIdKey, bool, int | None]] = []
        for run in sorted(runs, key=lambda r: r.run_id):
            key = run.parsed
            if key is None:
                misses[run.run_id] = _Miss("unparseable_id", None)
                continue
            hit, any_key, rejected = (None, False, None)
            if not self.withheld:
                hit, any_key, rejected = self._id_tiers(run, key)
            if hit is not None:
                assigned[run.run_id] = hit
            else:
                pending.append((run, key, any_key, rejected))
        holders: dict[tuple[str, str], list[str]] = defaultdict(list)
        for rid, a in assigned.items():
            holders[(a.trip.trip_id, a.service_date)].append(rid)
        shared = {rid for ids in holders.values() if len(ids) > 1 for rid in ids}
        held = set(holders)
        items: list[tuple[str, list[_Candidate]]] = []
        by_id = {r.run_id: r for r, *_ in pending}
        with_candidates: set[str] = set()
        for run, _key, _ak, _rej in pending:
            every = self.drift_candidates(run, DRIFT_TOL_S)
            if every:
                with_candidates.add(run.run_id)
            cs = [c for c in every if (c.trip_id, c.service_date) not in held]
            if cs:
                items.append((run.run_id, cs))
        for rid, c in assign_min_cost(items, DRIFT_TOL_S).items():
            ix = self.index_for(c.service_date)
            assert ix is not None
            trip = ix.trip(c.trip_id)
            assert trip is not None
            assigned[rid] = Assignment(
                trip, c.service_date, "drift", c.offset, False, ix.build_id
            )
        # A run with tier-5 candidates that holds none lost the assignment
        # (to another tier-5 run, an id-tier run, or the leave-unassigned cost).
        for run, key, any_key, rejected in pending:
            if run.run_id in assigned:
                continue
            near_c = self.drift_candidates(run, NEAR_S)
            near = near_c[0].trip_id if near_c else None
            cls = (
                "drift_unassigned"
                if run.run_id in with_candidates
                else self._miss(by_id[run.run_id], key, any_key, rejected)
            )
            misses[run.run_id] = _Miss(cls, near)
        return assigned, misses, shared


# --- the assignment ---------------------------------------------------------------


def assign_min_cost(
    items: Sequence[tuple[str, list[_Candidate]]], unassigned_cost: int
) -> dict[str, _Candidate]:
    """Min-cost one-to-one assignment of runs to (trip, day) with a cost of
    ``unassigned_cost`` for leaving a run without a trip. Equal totals go to
    the assignment whose trips sort first. Solved per connected component."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for rid, cs in items:
        for c in cs:
            parent[find("r:" + rid)] = find(f"t:{c.trip_id}|{c.service_date}")
    comps: dict[str, list[tuple[str, list[_Candidate]]]] = defaultdict(list)
    for rid, cs in sorted(items):
        comps[find("r:" + rid)].append((rid, cs))
    out: dict[str, _Candidate] = {}
    for comp in sorted(comps.values(), key=lambda c: c[0][0]):
        trips = sorted({(c.trip_id, c.service_date) for _, cs in comp for c in cs})
        col = {t: j for j, t in enumerate(trips)}
        n, m = len(comp), len(trips)
        scale = n * (m + 1) + 1
        big = (unassigned_cost + 1) * scale * (n + 1)
        width = m + n
        cost = [[big] * width for _ in range(n)]
        choice: list[dict[int, _Candidate]] = [{} for _ in range(n)]
        for i, (_rid, cs) in enumerate(comp):
            for c in cs:
                j = col[(c.trip_id, c.service_date)]
                v = c.cost * scale + j
                if v < cost[i][j]:
                    cost[i][j] = v
                    choice[i][j] = c
            for k in range(n):
                cost[i][m + k] = unassigned_cost * scale + m
        for i, j in enumerate(_hungarian(cost)):
            if j < m:
                out[comp[i][0]] = choice[i][j]
    return out


def _hungarian(cost: list[list[int]]) -> list[int]:
    """Columns of a min-cost assignment of every row (rows <= columns),
    integer costs (shortest augmenting paths with potentials)."""
    n, m = len(cost), len(cost[0])
    inf = float("inf")
    u = [0] * (n + 1)
    v = [0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [inf] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            row = cost[i0 - 1]
            delta = inf
            j1 = 0
            ui0 = u[i0]
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = row[j - 1] - ui0 - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            d = cast(int, delta)
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += d
                    v[j] -= d
                else:
                    minv[j] -= d
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    result = [0] * n
    for j in range(1, m + 1):
        if p[j]:
            result[p[j] - 1] = j - 1
    return result


# --- records ----------------------------------------------------------------------


def clock_rule_date(at: int, tz: str) -> str:
    local = datetime.fromtimestamp(at, ZoneInfo(tz))
    day = local.date() - timedelta(days=1 if local.hour < NIGHT_END_HOUR else 0)
    return day.isoformat()


def _deviation(ix: MatchIndex, pattern_id: str, stops: tuple[str, ...]) -> str:
    last = -1
    order = False
    for s in stops:
        if s not in ix.platforms:
            continue
        i = ix.stop_position(pattern_id, s)
        if i is None:
            return "off_pattern"
        if i <= last:
            order = True
        last = max(last, i)
    return "order_violation" if order else "consistent"


def trip_matches(
    runs: Sequence[Run], matcher: Matcher, fallback_build_id: str
) -> list[TripMatch]:
    """TripMatch records of every run of a window, sorted by run id."""
    assigned, misses, shared = matcher.match(runs)
    tz = matcher.tz
    out: list[TripMatch] = []
    for run in sorted(runs, key=lambda r: r.run_id):
        a = assigned.get(run.run_id)
        flags = set(run.flags) & {
            "duplicate_trip_id",
            "joined_split_run",
            "deduplicated_by_position",
        }
        raw = run.lineage == "raw"
        legacy_unknown = Unknown(
            unknown=True, reason="archive/trace keeps the trip id alone"
        )
        published: dict[str, object] = {
            "route_published": run.route_published if raw else legacy_unknown,
            "start_date_published": run.start_date if raw else legacy_unknown,
            "nyct_train_ids": run.train_ids if raw else legacy_unknown,
        }
        compatible = list(run.compatible_pattern_ids)
        alternative: str | None = None
        if a is not None:
            ix = matcher.index_for(a.service_date)
            assert ix is not None
            if (
                raw
                and run.start_date is not None
                and a.trip.first_departure is not None
            ):
                expected = datetime.fromtimestamp(
                    a.trip.first_departure, ZoneInfo(tz)
                ).strftime("%Y%m%d")
                if run.start_date != expected:
                    flags.add("start_date_mismatch")
            fields: dict[str, object] = {
                "outcome": "matched",
                "tier": a.tier,
                "miss_class": None,
                "network_build_id": a.build_id,
                "scheduled_trip_id": a.trip.trip_id,
                "nearest_scheduled_trip_id": None,
                "pattern_id": a.trip.pattern_id,
                "pattern_attribution": "scheduled_trip",
                "route_id": a.trip.route_id,
                "route_basis": "scheduled_trip",
                "service_date": a.service_date,
                "service_date_basis": "scheduled_trip",
                "origin_wrapped": a.wrapped,
                "offset_s": a.offset_s,
                "deviation": _deviation(ix, a.trip.pattern_id, run.stops()),
            }
        else:
            miss = misses[run.run_id]
            parsed = run.parsed
            if parsed is None:
                outcome, basis = "unparseable", "clock_rule"
                service_date = clock_rule_date(run.first_at, tz)
            else:
                outcome, basis = "unscheduled", "id_origin"
                days = service_dates(run.first_at, run.trip_id)
                service_date = days[0].isoformat()
                local = datetime.fromtimestamp(run.first_at, ZoneInfo(tz))
                if local.hour < NIGHT_END_HOUR and len(days) > 1:
                    flags.add("service_date_ambiguous")
                    alternative = days[1].isoformat()
            if raw and run.route_published is not None:
                route_id, route_basis = run.route_published, "published"
            else:
                route_id = None if parsed is None else parsed.route_field
                route_basis = "trip_id_field"
            ix = matcher.index_for(service_date) or matcher.index_for(
                candidate_service_dates(run.first_at, tz)[1]
            )
            if len(compatible) == 1:
                attribution, pattern_id = "unique_fit", compatible[0]
            elif compatible:
                attribution, pattern_id = "ambiguous", None
            else:
                attribution, pattern_id = "off_pattern", None
            fields = {
                "outcome": outcome,
                "tier": None,
                "miss_class": miss.miss_class,
                "network_build_id": fallback_build_id if ix is None else ix.build_id,
                "scheduled_trip_id": None,
                "nearest_scheduled_trip_id": miss.nearest,
                "pattern_id": pattern_id,
                "pattern_attribution": attribution,
                "route_id": route_id,
                "route_basis": route_basis,
                "service_date": service_date,
                "service_date_basis": basis,
                "origin_wrapped": False,
                "offset_s": None,
                "deviation": None,
            }
        out.append(
            TripMatch.model_validate(
                {
                    "run_id": run.run_id,
                    "track": run.track,
                    "member_run_ids": run.member_run_ids,
                    "supersedes_run_ids": [],
                    "lineage": run.lineage,
                    "source": run.feed,
                    "source_basis": LINEAGE_SOURCE_BASIS[run.lineage],
                    "realtime_trip_id": run.trip_id,
                    **published,
                    "first_observed_at": run.first_at,
                    "last_observed_at": run.last_at,
                    "n_observations": len(run.rows),
                    "n_placeholder_rows": run.n_placeholder_rows,
                    "n_old_timestamp_rows": run.n_old_rows,
                    "stale_only": run.stale_only,
                    "stationary": run.stationary,
                    "flags": sorted(flags),
                    "compatible_pattern_ids": compatible,
                    "shares_scheduled_trip": run.run_id in shared,
                    "identity_uncertain_with": run.identity_uncertain_with,
                    "alternative_service_date": alternative,
                    "end_state": run.end_state,
                    **fields,
                }
            )
        )
    return out


def service_day_window(service_date: str, tz: str) -> tuple[int, int]:
    """Observations a service day needs: local midnight − 2 h to the next
    day's 08:00 local (§6, Partitioning)."""
    start = wall_instant_of(service_date, 0, tz)
    end = wall_instant_of(service_date, 32 * 3600, tz)
    if start is None or end is None:
        raise ValueError(f"{service_date}: local midnight or 08:00 does not exist")
    return start - 7200, end
