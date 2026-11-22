"""Runs: realtime vehicle rows grouped into the trains they are (§2.2, §2.3).

Steps, in order, each a function of the rows alone (never of their delivery
order, never of ``entity_index``):

1. **Rows.** Placeholders are removed (``is_placeholder``: raw, NYCT
   ``is_assigned`` not true, or no status with a timestamp more than 60 s
   ahead; legacy, the §2.2 rule). Rows without a trip id or a stop are not
   positions of a train. A row delivered twice (the same poll and entity) is
   one row.
2. **Gap split.** Per (lineage, feed, trip id), rows sorted by (observed_at,
   poll, content) split where two consecutive rows lie more than
   ``RUN_GAP_S`` apart.
3. **Tracks.** A run with a poll holding two or more rows of its trip id is
   two vehicles sharing the id. Raw: split by ``train_id``, then by stop
   continuity; legacy: by stop continuity (``_continuity``). Tracks whose
   sampled positions are identical merge when their published identities
   agree (equal ``train_id``, or legacy), flagged
   ``deduplicated_by_position``; otherwise both stay, each naming the other in
   ``identity_uncertain_with``. Several tracks are numbered by first
   instant, then the §2.3 key (first stop's pattern position, first stop, full sampled
   sequence, train ids). A run the continuity rule cannot split is kept
   whole, flagged ``duplicate_trip_id``, and its shared polls give no stops.
4. **Split-run join.** Consecutive runs of one feed with the same origin,
   route field and direction, the second starting within ``JOIN_GAP_S`` of
   the first's end at the same or the next pattern stop, join (raw: unless
   both carry train ids and they differ). The joined run keeps the earliest
   member's identity and lists the others.
5. **Old timestamps.** A row whose ``vehicle_timestamp`` is more than
   ``OLD_TIMESTAMP_S`` old carries no new position: it is kept only when it
   opens a stop block (it is the arrival there). Nothing is deleted; the
   counts stay on the run. A run whose last row is old ended in a
   ``stale_tail``; a run still in the last poll of its feed ended at
   ``window_end``; any other run ``left_feed``. Every rule here looks only at
   a row and the rows before it, so a prefix of the input gives the same
   kept rows as the whole input up to the prefix end.
6. **Pattern fit.** ``compatible_pattern_ids``: the patterns of the run's
   route keys (published route and trip-id field, 6X/7X/FX folded) and
   direction whose stop order contains the run's observed platform stops in
   order, at least one of them. Computed from observations alone.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from itertools import pairwise
from typing import Literal

from momentarily.gtfs.static import base_route
from momentarily.model.events.contracts import FEED_FAMILY, Lineage
from momentarily.model.events.match_index import MatchIndex
from momentarily.model.events.trip_ids import TripIdKey, parse_realtime
from momentarily.model.observations.contracts import GtfsRtSource

RUN_GAP_S = 1800
"""Decision 6: a gap longer than this between two rows of a trip id starts a
new run."""
OLD_TIMESTAMP_S = 300
"""Decision 6: a row whose vehicle_timestamp is older than this is old."""
JOIN_GAP_S = 120
"""Split-run join: the second run starts at most this long after the first
ends."""
FUTURE_PLACEHOLDER_S = 60
"""Raw placeholder rule for SIR: no status and a timestamp this far ahead."""
EVENT_TIME_SKEW_S = 240
"""A stop block is timed by its first row's vehicle_timestamp when that lies
this close to the poll, else by the poll (the prototype's rule)."""
BLOCK_LOOKAHEAD = 8
"""Matching looks for a common stop among a run's first this many blocks."""

_STOPPED_AT = 1

EndState = Literal["left_feed", "stale_tail", "window_end"]


@dataclass(frozen=True, slots=True)
class Row:
    """One vehicle row, reduced to what layer 3 reads. ``poll`` names the
    fetch (raw ``fetch_id``) or the trace object (legacy ``source_key``);
    ``entity`` is its position there, used only to drop a row delivered
    twice. Published fields are None on legacy rows."""

    at: int
    poll: str
    entity: int
    feed: GtfsRtSource
    trip_id: str
    stop_id: str
    status: int | None
    vts: int | None
    route: str | None
    start_date: str | None
    train_id: str | None

    @property
    def old(self) -> bool:
        return self.vts is not None and self.at - self.vts > OLD_TIMESTAMP_S

    def content(self) -> tuple[object, ...]:
        """The row's content, for ordering rows of one poll without their
        entity position."""
        return (
            self.stop_id,
            -1 if self.vts is None else self.vts,
            -1 if self.status is None else self.status,
            self.train_id or "",
            self.route or "",
            self.start_date or "",
        )

    def order(self) -> tuple[object, ...]:
        return (self.at, self.poll, *self.content(), self.entity)


def is_raw_placeholder(
    assigned: bool | None, status: int | None, vts: int | None, at: int
) -> bool:
    """§2.2 raw rule: NYCT does not flag the trip assigned, or (SIR, which
    flags every trip) it has no status and a timestamp more than 60 s ahead."""
    if assigned is not True:
        return True
    return status is None and vts is not None and vts - at > FUTURE_PLACEHOLDER_S


def is_legacy_placeholder(
    feed: GtfsRtSource, stopped: bool, vts: int | None, at: int
) -> bool:
    """§2.2 legacy rule: a row that is not stopped and reports from the
    future; on the B division (not SIR) any row that is not stopped."""
    if stopped:
        return False
    if FEED_FAMILY[feed] == "last" and feed != "si":
        return True
    return vts is not None and vts > at


# --- stop blocks ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Block:
    """Consecutive kept rows of a run at one stop."""

    stop_id: str
    first_at: int
    last_at: int
    time: int
    """The block's observed time for matching (``EVENT_TIME_SKEW_S``)."""


def _blocks(rows: Sequence[Row]) -> list[Block]:
    out: list[Block] = []
    for r in rows:
        if out and out[-1].stop_id == r.stop_id:
            out[-1] = replace(out[-1], last_at=r.at)
            continue
        t = (
            r.vts
            if r.vts is not None and abs(r.vts - r.at) <= EVENT_TIME_SKEW_S
            else r.at
        )
        out.append(Block(r.stop_id, r.at, r.at, t))
    return out


# --- patterns ---------------------------------------------------------------------


class PatternBook:
    """Pattern stop lists of one build, by folded route and direction, for
    fitting and stop continuity."""

    def __init__(self, index: MatchIndex) -> None:
        self.build_id = index.build_id
        self.platforms = index.platforms
        self._index = index
        self._fit: dict[tuple[str, tuple[str, ...]], bool] = {}

    def patterns(self, route_keys: Iterable[str], direction: str) -> list[str]:
        found: set[str] = set()
        for key in route_keys:
            found.update(self._index.patterns_of(key, direction))
        return sorted(found)

    def position(self, pattern_id: str, stop: str) -> int | None:
        return self._index.stop_position(pattern_id, stop)

    def fits(self, pattern_id: str, stops: tuple[str, ...]) -> bool:
        """The platform stops appear in the pattern's order, at least one."""
        hit = self._fit.get((pattern_id, stops))
        if hit is not None:
            return hit
        last = -1
        ok = False
        for s in stops:
            if s not in self.platforms:
                continue
            i = self._index.stop_position(pattern_id, s)
            if i is None or i <= last:
                ok = False
                break
            last = i
            ok = True
        self._fit[(pattern_id, stops)] = ok
        return ok

    def compatible(
        self, route_keys: Iterable[str], direction: str, stops: tuple[str, ...]
    ) -> list[str]:
        return [p for p in self.patterns(route_keys, direction) if self.fits(p, stops)]

    def hops(
        self, route_keys: Iterable[str], direction: str, a: str, b: str
    ) -> int | None:
        """Fewest forward pattern positions from ``a`` to ``b``; 0 when equal."""
        if a == b:
            return 0
        best: int | None = None
        for p in self.patterns(route_keys, direction):
            i, j = self.position(p, a), self.position(p, b)
            if (
                i is not None
                and j is not None
                and j > i
                and (best is None or j - i < best)
            ):
                best = j - i
        return best

    def first_position(
        self, route_keys: Iterable[str], direction: str, stop: str
    ) -> int | None:
        found = [
            i
            for p in self.patterns(route_keys, direction)
            if (i := self.position(p, stop)) is not None
        ]
        return min(found) if found else None


def route_keys(
    trip: TripIdKey | None, published: str | None, withheld: bool = False
) -> list[str]:
    """Folded routes a run may belong to: its published route (raw) and its
    trip-id route field. ``withheld`` keeps the field alone."""
    keys: set[str] = set()
    if trip is not None:
        keys.add(base_route(trip.route_field))
    if published and not withheld:
        keys.add(base_route(published))
    return sorted(keys)


# --- runs -------------------------------------------------------------------------


@dataclass
class Run:
    lineage: Lineage
    feed: GtfsRtSource
    trip_id: str
    rows: list[Row]
    """Every row of the run, in order (old ones included)."""
    track: int = 0
    flags: set[str] = field(default_factory=set[str])
    identity_uncertain_with: list[str] = field(default_factory=list[str])
    member_run_ids: list[str] = field(default_factory=list[str])
    shared_polls: frozenset[tuple[int, str]] = frozenset()
    """Polls of an unsplittable run that hold several rows; they give no stops."""
    n_placeholder_rows: int = 0
    end_state: EndState = "left_feed"
    compatible_pattern_ids: list[str] = field(default_factory=list[str])

    @property
    def first_at(self) -> int:
        return self.rows[0].at

    @property
    def last_at(self) -> int:
        return self.rows[-1].at

    @property
    def run_id(self) -> str:
        return f"{self.lineage}:{self.feed}:{self.trip_id}@{self.first_at}#{self.track}"

    @property
    def parsed(self) -> TripIdKey | None:
        return parse_realtime(self.trip_id)

    @property
    def route_published(self) -> str | None:
        return self.rows[0].route

    @property
    def start_date(self) -> str | None:
        return self.rows[0].start_date

    @property
    def train_ids(self) -> list[str]:
        return sorted({r.train_id for r in self.rows if r.train_id is not None})

    @property
    def n_old_rows(self) -> int:
        return sum(r.old for r in self.rows)

    @property
    def stale_only(self) -> bool:
        return all(r.old for r in self.rows)

    def kept(self) -> list[Row]:
        """Rows that carry a position: fresh rows, and an old row only when it
        opens a stop block (§2.2, retained tails). Shared polls of an
        unsplittable run are left out."""
        out: list[Row] = []
        prev: str | None = None
        for r in self.rows:
            if (r.at, r.poll) in self.shared_polls:
                continue
            if not r.old or r.stop_id != prev:
                out.append(r)
            prev = r.stop_id
        return out

    def blocks(self) -> list[Block]:
        return _blocks(self.kept())

    def stops(self) -> tuple[str, ...]:
        return tuple(b.stop_id for b in self.blocks())

    @property
    def stationary(self) -> bool:
        return len({r.stop_id for r in self.kept()}) < 2

    def keys(self, withheld: bool = False) -> list[str]:
        return route_keys(self.parsed, self.route_published, withheld)


def _polls(rows: Sequence[Row]) -> list[list[Row]]:
    out: list[list[Row]] = []
    for r in rows:
        if out and (out[-1][0].at, out[-1][0].poll) == (r.at, r.poll):
            out[-1].append(r)
        else:
            out.append([r])
    return out


def _continuity(
    rows: Sequence[Row], hops: Callable[[str, str], int | None]
) -> list[list[Row]] | None:
    """Split rows into tracks by stop continuity, poll by poll. Each poll's
    rows go to the open tracks they continue (forward pattern hops from the
    track's last stop): as many rows as possible, then the fewest hops in
    total, then the smallest single hop (trains move evenly rather than one
    overtaking a standing one), then rows in content order to tracks in the
    order they opened. A row that continues no track opens one. None when
    that opens more tracks than the most rows any poll holds (the rule cannot
    tell the trains apart)."""
    tracks: list[list[Row]] = []
    widest = 0
    for poll in _polls(rows):
        widest = max(widest, len(poll))
        cost = [[hops(t[-1].stop_id, r.stop_id) for t in tracks] for r in poll]
        best = _best_assignment(cost)
        for ri, ti in enumerate(best):
            if ti is None:
                tracks.append([poll[ri]])
            else:
                tracks[ti].append(poll[ri])
    if len(tracks) > widest:
        return None
    return tracks


def _best_assignment(cost: list[list[int | None]]) -> list[int | None]:
    """For each row, its track (or None): the most rows assigned, then the
    least total cost, then the least largest cost, then the first in
    (row, track) order. Exhaustive: a poll holds a few rows of one trip id."""
    best: tuple[tuple[int, int, int], list[int | None]] | None = None

    def walk(i: int, used: frozenset[int], chosen: list[int | None]) -> None:
        nonlocal best
        if i == len(cost):
            costs = [cost[r][t] or 0 for r, t in enumerate(chosen) if t is not None]
            key = (-len(costs), sum(costs), max(costs, default=0))
            if best is None or key < best[0]:
                best = (key, list(chosen))
            return
        for t, c in enumerate(cost[i]):
            if c is not None and t not in used:
                walk(i + 1, used | {t}, [*chosen, t])
        walk(i + 1, used, [*chosen, None])

    walk(0, frozenset(), [])
    assert best is not None
    return best[1]


def _sequence(rows: Sequence[Row]) -> tuple[tuple[int, str, int], ...]:
    return tuple((r.at, r.stop_id, -1 if r.vts is None else r.vts) for r in rows)


def _split(
    lineage: Lineage,
    feed: GtfsRtSource,
    trip_id: str,
    rows: list[Row],
    book: PatternBook | None,
) -> list[Run]:
    polls = _polls(rows)
    if all(len(p) == 1 for p in polls):
        return [Run(lineage, feed, trip_id, rows)]
    parsed = parse_realtime(trip_id)
    keys = route_keys(parsed, rows[0].route)
    direction = "" if parsed is None else parsed.direction

    def hops(a: str, b: str) -> int | None:
        if book is None or parsed is None:
            return 0 if a == b else None
        return book.hops(keys, direction, a, b)

    groups: list[list[Row]]
    if lineage == "raw":
        by_train: dict[str, list[Row]] = defaultdict(list)
        for r in rows:
            by_train[r.train_id or ""].append(r)
        groups = [by_train[k] for k in sorted(by_train)]
    else:
        groups = [rows]
    tracks: list[list[Row]] = []
    for g in groups:
        if all(len(p) == 1 for p in _polls(g)):
            tracks.append(g)
            continue
        split = _continuity(g, hops)
        if split is None:
            shared = frozenset((p[0].at, p[0].poll) for p in polls if len(p) > 1)
            return [
                Run(
                    lineage,
                    feed,
                    trip_id,
                    rows,
                    flags={"duplicate_trip_id"},
                    shared_polls=shared,
                )
            ]
        tracks.extend(split)
    # identical sampled positions: merge when the published identity agrees
    merged: list[tuple[list[Row], bool]] = []
    for t in tracks:
        twin = next(
            (
                i
                for i, (m, _) in enumerate(merged)
                if _sequence(m) == _sequence(t)
                and (
                    lineage == "legacy"
                    or {r.train_id for r in m} == {r.train_id for r in t}
                )
            ),
            None,
        )
        if twin is None:
            merged.append((t, False))
        else:
            merged[twin] = (merged[twin][0], True)

    def order(t: list[Row]) -> tuple[object, ...]:
        pos = (
            None
            if book is None or parsed is None
            else book.first_position(keys, direction, t[0].stop_id)
        )
        # The first instant leads, so a track that appears later never
        # renumbers an earlier one (a live id stays the nightly id); the
        # §2.3 keys order tracks that start in the same poll.
        return (
            t[0].at,
            1 << 30 if pos is None else pos,
            t[0].stop_id,
            _sequence(t),
            tuple(sorted(r.train_id or "" for r in t)),
        )

    merged.sort(key=lambda m: order(m[0]))
    single = len(merged) == 1
    runs = [
        Run(
            lineage,
            feed,
            trip_id,
            t,
            track=0 if single else i,
            flags={"deduplicated_by_position"} if dedup else set(),
        )
        for i, (t, dedup) in enumerate(merged)
    ]
    for a in runs:
        for b in runs:
            if a is not b and _sequence(a.rows) == _sequence(b.rows):
                a.identity_uncertain_with.append(b.run_id)
        a.identity_uncertain_with.sort()
    return runs


def _join(runs: list[Run], book: PatternBook | None) -> list[Run]:
    """Join consecutive runs of one train split by a trip-id change."""
    groups: dict[tuple[object, ...], list[Run]] = defaultdict(list)
    alone: list[Run] = []
    for r in runs:
        p = r.parsed
        if (
            p is None
            or r.track != 0
            or r.flags & {"duplicate_trip_id", "deduplicated_by_position"}
        ):
            alone.append(r)
            continue
        groups[(r.lineage, r.feed, p.origin, p.route_field, p.direction)].append(r)
    out = list(alone)
    for members in groups.values():
        members.sort(key=lambda r: (r.first_at, r.trip_id))
        current = members[0]
        for nxt in members[1:]:
            if _joins(current, nxt, book):
                current = Run(
                    current.lineage,
                    current.feed,
                    current.trip_id,
                    current.rows + nxt.rows,
                    flags=current.flags | {"joined_split_run"},
                    member_run_ids=sorted(
                        [*current.member_run_ids, nxt.run_id, *nxt.member_run_ids]
                    ),
                )
            else:
                out.append(current)
                current = nxt
        out.append(current)
    return out


def _joins(a: Run, b: Run, book: PatternBook | None) -> bool:
    gap = b.first_at - a.last_at
    if not 0 < gap <= JOIN_GAP_S:
        return False
    ta, tb = a.train_ids, b.train_ids
    if ta and tb and not set(ta) & set(tb):
        return False
    s, t = a.rows[-1].stop_id, b.rows[0].stop_id
    if s == t:
        return True
    p = a.parsed
    if book is None or p is None:
        return False
    return book.hops(route_keys(p, a.route_published), p.direction, s, t) == 1


def build_runs(
    lineage: Lineage,
    rows: Iterable[Row],
    placeholders: Iterable[tuple[GtfsRtSource, str, int]],
    last_poll: dict[GtfsRtSource, int],
    book: PatternBook | None,
) -> list[Run]:
    """Runs from real rows (placeholders already removed). ``placeholders``
    are the removed rows as (feed, trip id, observed_at), counted on the run
    of their trip id whose span holds them. ``last_poll``: each feed's last
    good poll in the input, for ``window_end``. Sorted by run id."""
    seen: dict[tuple[object, ...], Row] = {}
    for r in rows:
        # A row delivered twice (one poll, entity and content) is one row.
        seen.setdefault((r.poll, r.entity, r.trip_id, *r.content()), r)
    by_trip: dict[tuple[GtfsRtSource, str], list[Row]] = defaultdict(list)
    for r in seen.values():
        by_trip[(r.feed, r.trip_id)].append(r)
    runs: list[Run] = []
    for (feed, trip_id), trip_rows in sorted(by_trip.items()):
        trip_rows.sort(key=Row.order)
        start = 0
        for i in range(1, len(trip_rows) + 1):
            if i == len(trip_rows) or trip_rows[i].at - trip_rows[i - 1].at > RUN_GAP_S:
                runs.extend(_split(lineage, feed, trip_id, trip_rows[start:i], book))
                start = i
    runs = _join(runs, book)
    ph: dict[tuple[GtfsRtSource, str], list[int]] = defaultdict(list)
    for feed, trip_id, at in placeholders:
        ph[(feed, trip_id)].append(at)
    for r in runs:
        r.rows.sort(key=Row.order)
        r.n_placeholder_rows = sum(
            r.first_at <= at <= r.last_at for at in ph.get((r.feed, r.trip_id), ())
        )
        if r.rows[-1].old:
            r.end_state = "stale_tail"
        elif r.last_at >= last_poll.get(r.feed, r.last_at + 1):
            r.end_state = "window_end"
        else:
            r.end_state = "left_feed"
        p = r.parsed
        if book is not None and p is not None:
            r.compatible_pattern_ids = book.compatible(r.keys(), p.direction, r.stops())
    runs.sort(key=lambda r: r.run_id)
    for a, b in pairwise(runs):
        if a.run_id == b.run_id:
            raise ValueError(f"two runs share the id {a.run_id}")
    return runs
