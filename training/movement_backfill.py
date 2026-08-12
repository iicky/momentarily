"""Offline reconstruction of the movement regime-transition stream.

The Worker clocks movement regimes online (worker/src/regime.ts) straight from
live vehicle positions and archives every commit to
v1/movement_transitions/<date>/<ts>.jsonl (training.eval.load_movement_transitions).
That stream only exists from the day the regime clock shipped, and only at
route scope — segment-scope regimes have never been wired up online
(worker/src/index.ts calls advanceRegimes with scope='route' only; segment_flow.ts
is a separate per-station roll-up, not a regime clock).

This module reconstructs the SAME transition stream offline from archives that
already exist, through the identical debounce (training.regime.replay_regimes)
so online and offline curves describe the same regimes, from two independent
sources:

    published_condition   v1/predictions/<date>/*.jsonl (training.eval.
                           load_predictions) — what the Worker published each
                           tick, route scope only. The archived window predates
                           this session's regime clock (state/movement_state.json
                           carried no debounce before it), so these rows are the
                           RAW per-tick classifier call, not a debounced regime —
                           the higher-fidelity source where it exists, because it
                           went through the Worker's own classifier rather than a
                           Python re-derivation.
    archive/vehicles       archive/vehicles/<date>/<ts>.json — raw vehicle
                           positions, re-classified here with the Python mirror
                           of the Worker's classifier (training.load_r2.
                           derive_movement_state / training.segments.
                           classify_segment). The ONLY source that reaches
                           segment scope, and the only one that covers the three
                           weeks before the movement arm shipped (published_condition
                           read ~100% `unknown` there — journal.md 2026-08-11,
                           "the movement arm was blind for three weeks").

Run with:
    murk exec -- uv run python -m training.movement_backfill --days 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict, deque
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from training.eval import (
    NOT_NORMAL,
    TICK_SECONDS,
    MovementTransitionRecord,
    PredictionRecord,
    load_predictions,
)
from training.gtfs_static import load_successors, through_stops
from training.load_r2 import (
    StopFilter,
    build_movement_series_by_direction,
    build_movement_truth,
    build_segment_baseline,
    build_segment_series,
    compute_advance_baseline,
    fetch_objects,
)
from training.pooled_dwell import MIN_VOTER_EVENTS
from training.r2_client import load_config, make_client
from training.regime import DEBOUNCE_TICKS, MAX_IDLE_SEC, replay_regimes
from training.segments import classify_segment

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

# A single 5-minute tick sees ~1 tracked train per segment leaf — far too few
# for classify_segment's significance gate (measured on real archive data:
# median matched-per-leaf-tick is 1, only 10.7% of leaf-ticks clear
# MIN_MATCHED_TRIPS=3). Each tick's segment call is judged against a trailing
# sum instead of the raw tick. 6 ticks (30 min) approximates worker/src/
# segment_flow.ts's ~1/(1-SEGMENT_DECAY)=5-tick effective decay window as a
# plain trailing sum — simpler to reason about and test than porting the EWMA
# state machine for a one-shot offline backfill.
SEGMENT_WINDOW_TICKS = 6

_VALID_SCOPES = ("route", "segment")


# --- Pure reconstruction: ticks -> calls, independent of R2 --------------
#
# Everything below takes/returns plain data (PredictionRecord lists, vehicle
# archive dicts, or (tick, {key: state}) pairs) so it is hermetically testable
# on synthetic input. The R2 fetch lives only in `_fetch_vehicle_bodies` and
# the `client`/`bucket` entry points at the bottom of this module.


def ticks_from_predictions(
    predictions: list[PredictionRecord],
) -> list[tuple[int, Mapping[str, str]]]:
    """Per-tick {route: published_condition} calls, route scope.

    `unknown` is excluded: an absence of reading is an abstention, not a
    state — the regime clock already treats an absent key correctly. Rows
    where the arm never shipped (published_condition is None, pre-dating
    escalation-arm grading) are excluded too. Every other value — including
    `not_scheduled` — is a real call: the Worker's own regime clock
    (deriveMovementStates) feeds it the same way.
    """
    by_tick: dict[int, dict[str, str]] = {}
    for p in predictions:
        cond = p.published_condition
        if cond is None or cond == "unknown":
            continue
        by_tick.setdefault(p.ts, {})[p.route] = cond
    return cast("list[tuple[int, Mapping[str, str]]]", sorted(by_tick.items()))


def route_ticks_from_vehicle_bodies(
    bodies: list[dict[str, Any]],
    *,
    counts_from_stop: StopFilter | None = None,
) -> list[tuple[int, Mapping[str, str]]]:
    """Per-tick {route: state} calls from the vehicle archive, route scope.

    Reuses build_movement_truth (the existing (route, tick) -> state
    reconstruction) rather than re-deriving it. Its docstring recommends a
    baseline from a clean/earlier window for causal TRUTH; this is not truth
    grading, it's regime reconstruction over the one window C1 hands us (no
    separate baseline period in the contracted signature). compute_advance_baseline's
    per-cell MEDIAN already resists a disrupted minority, so a typical outage
    (the minority case) doesn't drag down its own baseline.

    `counts_from_stop` goes to the baseline and the scored counts together — a
    through-stop baseline runs higher, so scoring unfiltered counts against it
    reads spuriously disrupted.
    """
    dir_series = build_movement_series_by_direction(
        bodies, counts_from_stop=counts_from_stop
    )
    baseline = compute_advance_baseline(dir_series)
    truth = build_movement_truth(
        bodies, movement_baseline=baseline, counts_from_stop=counts_from_stop
    )
    by_tick: dict[int, dict[str, str]] = {}
    for (route, tick), state in truth.items():
        by_tick.setdefault(tick, {})[route] = state
    return cast("list[tuple[int, Mapping[str, str]]]", sorted(by_tick.items()))


def segment_ticks_from_vehicle_bodies(
    bodies: list[dict[str, Any]],
    *,
    window_ticks: int = SEGMENT_WINDOW_TICKS,
    counts_from_stop: StopFilter | None = None,
) -> list[tuple[int, Mapping[str, str]]]:
    """Per-tick {`route|direction|from_stop`: state} calls from the vehicle
    archive, segment scope — the only source that reaches this granularity.

    classify_segment needs ACCUMULATED counts (segments.py docstring); each
    tick's call sums advanced/stalled over the trailing `window_ticks` ticks
    for that leaf, then classifies. The baseline is computed over the same
    window's bodies — same self-window rationale as
    route_ticks_from_vehicle_bodies, and its partial pooling already shrinks
    thin leaves toward their line/route/system normal.

    `counts_from_stop` restricts both the window accumulation and the
    baseline to leaves whose from_stop it admits — see training.load_r2.
    StopFilter / training.gtfs_static.through_stops. A terminal leaf gets
    neither accumulated counts nor a baseline cell, so it's never classified.
    """
    series = build_segment_series(bodies)
    baseline = build_segment_baseline(bodies, counts_from_stop=counts_from_stop)

    per_leaf: dict[tuple[str, str, str], dict[int, tuple[int, int]]] = defaultdict(dict)
    for (route, direction, frm, to, tick), n in series.items():
        if counts_from_stop is not None and not counts_from_stop(route, direction, frm):
            continue
        leaf = (route, direction, frm)
        adv, stall = per_leaf[leaf].get(tick, (0, 0))
        if frm == to:
            stall += n
        else:
            adv += n
        per_leaf[leaf][tick] = (adv, stall)

    window_sec = window_ticks * TICK_SECONDS
    by_tick: dict[int, dict[str, str]] = {}
    for leaf, tick_counts in per_leaf.items():
        cell = baseline.get(leaf)
        if cell is None:
            continue
        route, direction, frm = leaf
        key = f"{route}|{direction}|{frm}"
        window: deque[tuple[int, int, int]] = deque()
        adv_sum = stall_sum = 0
        for tick in sorted(tick_counts):
            a, s = tick_counts[tick]
            window.append((tick, a, s))
            adv_sum += a
            stall_sum += s
            while window and tick - window[0][0] >= window_sec:
                _, old_adv, old_stall = window.popleft()
                adv_sum -= old_adv
                stall_sum -= old_stall
            call = classify_segment(adv_sum, stall_sum, cell)
            if call is not None:
                by_tick.setdefault(tick, {})[key] = call
    return cast("list[tuple[int, Mapping[str, str]]]", sorted(by_tick.items()))


def transitions_from_ticks(
    ticks: list[tuple[int, Mapping[str, str]]],
    scope: str,
    *,
    debounce_ticks: int = DEBOUNCE_TICKS,
    max_idle_sec: int = MAX_IDLE_SEC,
) -> list[MovementTransitionRecord]:
    """Segment `ticks` via replay_regimes and shape the commits as
    MovementTransitionRecord. `ts` is set to the back-dated `exited_at`
    boundary: replay_regimes folds the whole history in one pass and doesn't
    expose which observed_at tick detected each commit (unlike the Worker,
    which stamps `ts` with that tick) — neither dwell.py nor scorecard.py
    reads `.ts` off a transition record, so this is descriptive, not
    load-bearing.
    """
    if scope not in _VALID_SCOPES:
        raise ValueError(f"scope must be one of {_VALID_SCOPES}, got {scope!r}")
    _, changes = replay_regimes(
        ticks, debounce_ticks=debounce_ticks, max_idle_sec=max_idle_sec
    )
    return [
        MovementTransitionRecord(
            ts=c.exited_at,
            scope=scope,
            key=c.key,
            route=c.key if scope == "route" else c.key.split("|", 1)[0],
            prev_state=c.prev_state,
            new_state=c.new_state,
            regime_entered_at=c.entered_at,
            exited_at=c.exited_at,
            dwell_sec=c.dwell_sec,
        )
        for c in changes
    ]


def open_regimes_from_ticks(
    ticks: list[tuple[int, Mapping[str, str]]],
    *,
    debounce_ticks: int = DEBOUNCE_TICKS,
    max_idle_sec: int = MAX_IDLE_SEC,
) -> dict[str, tuple[str, int]]:
    """{key: (state, entered_at)} for every regime still open at the last
    tick in `ticks` — the OpenRegimes shape training/dwell.py already
    consumes, so these censored observations reach the curves."""
    entries, _ = replay_regimes(
        ticks, debounce_ticks=debounce_ticks, max_idle_sec=max_idle_sec
    )
    return {key: (entry.state, entry.entered_at) for key, entry in entries.items()}


# --- R2 fetch --------------------------------------------------------------


def _date_range(start: date, end: date) -> Iterator[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _list_keys(client: S3Client, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix, "MaxKeys": 1000}
        if token:
            kwargs["ContinuationToken"] = token
        resp = client.list_objects_v2(**kwargs)
        for obj in resp.get("Contents") or []:
            key = obj.get("Key")
            if key is not None:
                keys.append(key)
        if not resp.get("IsTruncated"):
            return keys
        token = resp.get("NextContinuationToken")


def _fetch_vehicle_bodies(
    client: S3Client, bucket: str, start_date: date, end_date: date
) -> list[dict[str, Any]]:
    keys: list[str] = []
    for d in _date_range(start_date, end_date):
        keys.extend(_list_keys(client, bucket, f"archive/vehicles/{d.isoformat()}/"))
    return fetch_objects(client, bucket, keys)


def resolve_stop_filter(scope: str) -> StopFilter | None:
    """`--stop-scope` CLI value -> a counts_from_stop callable, fail-soft.

    "all" returns None (count every stop — the pre-filter numbers) without a
    fetch. "through" fetches the static GTFS topology and restricts counting
    to (route, direction, stop) triples with both a scheduled predecessor and
    successor (training.gtfs_static.through_stops) — a chain endpoint or a
    stop the timetable never names stalls by schedule, not disruption, and
    would otherwise pollute the advance signal. Mirrors train_em.py's
    _static_topology: a fetch failure degrades to None with the reason
    printed, never raises.
    """
    if scope != "through":
        return None
    try:
        through = through_stops(load_successors())
    except Exception as exc:
        print(
            f"gtfs static topology unavailable, counting every stop ({exc})",
            file=sys.stderr,
        )
        return None
    return lambda route, direction, stop: (route, direction, stop) in through


def _ticks_for(
    client: S3Client,
    bucket: str,
    start_date: date,
    end_date: date,
    *,
    scope: str,
    source: str,
    counts_from_stop: StopFilter | None = None,
) -> list[tuple[int, Mapping[str, str]]]:
    if scope not in _VALID_SCOPES:
        raise ValueError(f"scope must be one of {_VALID_SCOPES}, got {scope!r}")
    resolved = source
    if resolved == "auto":
        # published_condition has no segment granularity; vehicles is the only
        # source there. Route scope defaults to published_condition — the
        # higher-fidelity source where it exists (see module docstring).
        resolved = "vehicles" if scope == "segment" else "predictions"
    if resolved == "predictions":
        if scope == "segment":
            raise ValueError("segment scope has no published_condition source")
        predictions = load_predictions(client, bucket, start_date, end_date)
        return ticks_from_predictions(predictions)
    if resolved == "vehicles":
        bodies = _fetch_vehicle_bodies(client, bucket, start_date, end_date)
        return (
            route_ticks_from_vehicle_bodies(bodies, counts_from_stop=counts_from_stop)
            if scope == "route"
            else segment_ticks_from_vehicle_bodies(
                bodies, counts_from_stop=counts_from_stop
            )
        )
    raise ValueError(f"source must be one of auto/predictions/vehicles, got {source!r}")


# --- C1 contract -------------------------------------------------------


def reconstruct_movement_transitions(
    *,
    client: S3Client,
    bucket: str,
    start_date: date,
    end_date: date,
    scope: str = "route",
    debounce_ticks: int = DEBOUNCE_TICKS,
    source: str = "auto",
    counts_from_stop: StopFilter | None = None,
) -> list[MovementTransitionRecord]:
    """Reconstruct the movement-regime transition stream the Worker would have
    emitted over [start_date, end_date], at route or segment scope.

    `source` picks the reconstruction source ("predictions" | "vehicles");
    "auto" (default) resolves to published_condition for route scope (the
    higher-fidelity source where it exists) and archive/vehicles for segment
    scope (the only source that reaches it).

    `counts_from_stop` restricts vehicle-derived counting, at both scopes, to
    admitted from_stops (see segment_ticks_from_vehicle_bodies). The
    published-condition source has no per-stop counts and is unaffected.
    """
    ticks = _ticks_for(
        client,
        bucket,
        start_date,
        end_date,
        scope=scope,
        source=source,
        counts_from_stop=counts_from_stop,
    )
    return transitions_from_ticks(ticks, scope, debounce_ticks=debounce_ticks)


def movement_open_regimes(
    *,
    client: S3Client,
    bucket: str,
    start_date: date,
    end_date: date,
    scope: str = "route",
    debounce_ticks: int = DEBOUNCE_TICKS,
    source: str = "auto",
    counts_from_stop: StopFilter | None = None,
) -> dict[str, tuple[str, int]]:
    """Each key's regime still open at end_date — see reconstruct_movement_transitions
    for the `source` and `counts_from_stop` resolution rules."""
    ticks = _ticks_for(
        client,
        bucket,
        start_date,
        end_date,
        scope=scope,
        source=source,
        counts_from_stop=counts_from_stop,
    )
    return open_regimes_from_ticks(ticks, debounce_ticks=debounce_ticks)


# --- Debounce sensitivity + cross-source agreement report -----------------
#
# The analysis this backfill exists to unblock: how much of the raw episode
# population survives at each debounce_ticks, and how well the two
# independent sources agree at route scope. Not part of C1; a reproducible
# instrument for the numbers reported below.


@dataclass(frozen=True)
class EpisodeCensus:
    n_episodes: int
    median_duration_min: float | None
    min_duration_min: float | None
    max_duration_min: float | None
    distinct_cells: int
    cells_with_ge_min_voter_events: int  # >= MIN_VOTER_EVENTS (3) completed episodes


def episode_census(transitions: list[MovementTransitionRecord]) -> EpisodeCensus:
    """Count completed NOT_NORMAL runs (disrupted/suspended) as "episodes" —
    the incident-census vocabulary already used in journal.md and
    training/eval.py's NOT_NORMAL. `normal`/`not_scheduled` exits are real
    transitions (feed dwell_movement's own-state curves) but are not
    incidents, so they're excluded from this census."""
    eps = [t for t in transitions if t.prev_state in NOT_NORMAL]
    durations = sorted(t.dwell_sec / 60.0 for t in eps)
    by_key = Counter(t.key for t in eps)
    return EpisodeCensus(
        n_episodes=len(eps),
        median_duration_min=statistics.median(durations) if durations else None,
        min_duration_min=durations[0] if durations else None,
        max_duration_min=durations[-1] if durations else None,
        distinct_cells=len(by_key),
        cells_with_ge_min_voter_events=sum(
            1 for n in by_key.values() if n >= MIN_VOTER_EVENTS
        ),
    )


def matched_transition_count(
    a: list[MovementTransitionRecord],
    b: list[MovementTransitionRecord],
    *,
    tick_sec: int = TICK_SECONDS,
) -> int:
    """How many of `a`'s transitions have some (greedy, one-to-one) match in
    `b`: same route, same new_state, entered_at within one tick. Not a
    globally-optimal assignment, but transitions are sparse enough over a
    week that greedy nearest-available is not the thing that would move this
    number."""
    pool = list(b)
    matched = 0
    for ta in a:
        for i, tb in enumerate(pool):
            if (
                ta.route == tb.route
                and ta.new_state == tb.new_state
                and abs(ta.regime_entered_at - tb.regime_entered_at) <= tick_sec
            ):
                matched += 1
                del pool[i]
                break
    return matched


def debounce_sensitivity_report(
    client: S3Client,
    bucket: str,
    start_date: date,
    end_date: date,
    *,
    debounce_values: tuple[int, ...] = (1, 2, 3),
    recommended_debounce_ticks: int = DEBOUNCE_TICKS,
    counts_from_stop: StopFilter | None = None,
) -> dict[str, Any]:
    """The debounce sensitivity table (route + segment scope) and the
    cross-source agreement check, computed once over [start_date, end_date].
    Fetches predictions and vehicle bodies exactly once and reuses them
    across every debounce_ticks value.

    `counts_from_stop` restricts both vehicle-derived reconstructions to
    admitted from_stops. The published-condition side has no per-stop counts."""
    predictions = load_predictions(client, bucket, start_date, end_date)
    bodies = _fetch_vehicle_bodies(client, bucket, start_date, end_date)

    route_ticks_pub = ticks_from_predictions(predictions)
    route_ticks_veh = route_ticks_from_vehicle_bodies(
        bodies, counts_from_stop=counts_from_stop
    )
    segment_ticks_veh = segment_ticks_from_vehicle_bodies(
        bodies, counts_from_stop=counts_from_stop
    )

    def sweep(
        ticks: list[tuple[int, Mapping[str, str]]], scope: str
    ) -> dict[int, dict[str, Any]]:
        out: dict[int, dict[str, Any]] = {}
        baseline_n: int | None = None
        for d in debounce_values:
            transitions = transitions_from_ticks(ticks, scope, debounce_ticks=d)
            census = episode_census(transitions)
            if baseline_n is None:
                baseline_n = census.n_episodes
            out[d] = {
                **asdict(census),
                "survival_vs_debounce_1": (
                    census.n_episodes / baseline_n if baseline_n else None
                ),
            }
        return out

    route_sweep = sweep(route_ticks_pub, "route")
    segment_sweep = sweep(segment_ticks_veh, "segment")

    pub_transitions = transitions_from_ticks(
        route_ticks_pub, "route", debounce_ticks=recommended_debounce_ticks
    )
    veh_transitions = transitions_from_ticks(
        route_ticks_veh, "route", debounce_ticks=recommended_debounce_ticks
    )
    pub_matched = matched_transition_count(pub_transitions, veh_transitions)
    veh_matched = matched_transition_count(veh_transitions, pub_transitions)

    return {
        "window": {"start": start_date.isoformat(), "end": end_date.isoformat()},
        "n_predictions": len(predictions),
        "n_vehicle_bodies": len(bodies),
        "route_debounce_sensitivity": route_sweep,
        "segment_debounce_sensitivity": segment_sweep,
        "source_agreement": {
            "debounce_ticks": recommended_debounce_ticks,
            "published_to_vehicles": {
                "matched": pub_matched,
                "total": len(pub_transitions),
            },
            "vehicles_to_published": {
                "matched": veh_matched,
                "total": len(veh_transitions),
            },
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Movement transition backfill report")
    parser.add_argument("--start-date", type=date.fromisoformat, default=None)
    parser.add_argument("--end-date", type=date.fromisoformat, default=None)
    parser.add_argument("--days", type=int, default=8, help="window length in days")
    parser.add_argument(
        "--debounce-ticks",
        type=int,
        default=DEBOUNCE_TICKS,
        help="debounce_ticks the source_agreement check reconstructs both sides at",
    )
    parser.add_argument(
        "--stop-scope",
        choices=("through", "all"),
        default="through",
        help="through: vehicle-derived counting admits only stops with a "
        "scheduled predecessor AND successor (training.gtfs_static."
        "through_stops), excluding terminal layovers. all: count every stop "
        "(pre-filter behavior, for a direct comparison run). The "
        "published-condition source has no per-stop counts and is unaffected.",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    client = make_client(cfg)
    today = datetime.now(UTC).date()
    end_date = args.end_date or today
    start_date = args.start_date or (end_date - timedelta(days=args.days - 1))
    counts_from_stop = resolve_stop_filter(args.stop_scope)

    report = debounce_sensitivity_report(
        client,
        cfg.bucket,
        start_date,
        end_date,
        recommended_debounce_ticks=args.debounce_ticks,
        counts_from_stop=counts_from_stop,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
