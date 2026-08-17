"""Grade a movement score against the assigned_n degradation label.

WHY THIS EXISTS. Two movement scores compete for the "is this route degraded
now" job and only one of them has ever been graded. The advance rate — did each
tracked train move to a new stop between polls — was scored against the
assigned_n label over ten days and came out anti-correlated: 7 firings in 53,993
route-ticks, 0 on a labelled degradation, and 0.003 agreement when loosened
enough to fire at all, against a 1.795% base rate. The other arm, traversal
time against what the timetable allots the same hop, was prototyped offline and
never graded, because until the minute trace existed there was nothing to
compute it from honestly. This module closes that.

WHY THE FIVE-MINUTE ARCHIVE CANNOT ANSWER IT. `archive/vehicles/` stores
per-(route, direction) transition COUNTS, not per-trip positions, so a duration
score built there has to price the stretch between two observed stops with the
route's MODAL scheduled chain. That credits a train with the scheduled time of
every stop it skipped, so a bypassing train reads as running FAST exactly when
service is worst. The minute trace carries trip identity and stop sequence, so
each observation is priced against the trip's OWN pattern, which makes a bypass
IDENTIFIABLE per observation and therefore excludable, instead of silently
absorbed into every stretch the modal chain mispriced. The score below is
therefore trace-only, and its window is the trace's window.

WHAT THE SCORE IS. Per admitted hop, observed arrival-to-arrival seconds over
the seconds the trip's own timetable allows for that same hop; per (route,
tick), the median over the hops finishing in it. ONE over the timetable is
on-time, TWO is twice as long as booked. Nothing here is fitted, which is the
point: a baseline fitted on the trace and graded over the same trace window
would be scored partly against itself, and the trace is only a day old. The
timetable is the outside reference that makes a one-day window legitimate.

WHICH TICK AN OBSERVATION LANDS IN. The one containing the hop's COMPLETION,
not its start. A hop's duration is not known until it ends, so completion is
the only attribution a live nowcast could actually compute without lookahead.
It costs the score latency on exactly the slow hops it most wants to catch —
a 15-minute crawl is reported at minute 15 — and that is a real property of
the measure, not an artifact to correct away.

ORIENTATION. Every arm here is stated so HIGHER MEANS WORSE, so an AUC below
0.5 is an inverted score rather than a sign convention to squint at. The
advance arm is therefore reported as the STALLED share, not the advance rate.

WHAT AN HONEST COMPARISON NEEDS BESIDES AUC. The advance arm's failure was not
mainly discrimination, it was coverage: assigned_n degradation is service
WITHDRAWN, the movement arms measure whether the trains still out there MOVE,
and withdrawing service removes the observations a movement call needs. Only
276 of 1,249 labelled degraded ticks were judgeable by the advance arm at all.
`degraded_coverage` below reports that share for each arm, because a score that
abstains on the degraded ticks has not been beaten on skill, it has been
prevented from playing.
"""

from __future__ import annotations

import argparse
import bisect
import io
import json
import random
import statistics
import sys
import zipfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from training.degradation_label import BIN_FN, build_labels
from training.gtfs_static import (
    Timetable,
    base_route,
    fetch_gtfs_zip,
    successors,
    through_stops,
)
from training.gtfs_static import (
    timetable as parse_timetable,
)
from training.load import TICK_SECONDS
from training.load_r2 import (
    MIN_MATCHED_TRIPS,
    StopFilter,
    build_movement_series,
    build_service_series,
    compute_baseline,
    fetch_trip_update_metrics,
    fetch_vehicle_metrics,
)
from training.r2_client import load_config, make_client
from training.trace import (
    EXACT,
    Traversal,
    scheduled_for,
)

# The label's two degraded states. "normal" is the only other value.
DEGRADED = ("acute", "chronic")

# Hops a route-tick needs before its median is taken. Five is the floor the
# dwell fits already use for "trust the empirical quantity rather than a
# fallback" (dwell.MIN_SAMPLES_FOR_EMPIRICAL), and it is doing the same job
# here: below it the median is one or two trains, and a single bad run would
# read as a route-wide slowdown. A route-tick under the floor is ABSENT, never
# scored as normal — the same can't-judge contract the movement baseline and
# the label itself use.
MIN_TICK_SAMPLES = 5


def _snap(epoch: int) -> int:
    return (epoch // TICK_SECONDS) * TICK_SECONDS


@dataclass(frozen=True)
class RatioStats:
    """The honest denominator for the progress score: what the trace offered
    and why each discarded observation was discarded."""

    n_traversals: int
    # Admitted: an EXACT single-hop span, inside the loaded feed's validity
    # window, that the trip's own timetable prices.
    n_attributable: int
    n_censored: int  # RIGHT or INTERVAL: no destination, or a span of many hops
    n_multi_hop: int  # EXACT by kind but the feed's stop sequence spans a gap
    n_outside_feed_window: int
    n_no_schedule: int  # covered by the feed, but no scheduled time for the pair
    n_bypass: int  # priced, and the trip's own pattern puts a station inside
    n_route_ticks: int
    n_route_ticks_thin: int  # had observations, under MIN_TICK_SAMPLES


def hop_ratios(
    traversals: Sequence[Traversal], timetable: Timetable
) -> tuple[dict[tuple[str, int], list[float]], RatioStats]:
    """(route, completion tick) -> observed/scheduled for every hop that both
    the trace and the timetable can speak to.

    Admission starts from traversal.hop_samples — EXACT, a destination, one hop
    by the feed's own stop sequence — and then adds two rules a RATIO needs
    that a baseline fit does not.

    An observation the timetable cannot price is dropped rather than kept with
    a null: there is no ratio without a denominator.

    A BYPASS IS DROPPED, and this is the one place this module deliberately
    parts company with the baseline fit. There, a bypass is a real measurement
    of the pair and is kept, because the level is fitted from movement alone
    and excluding it would let the schedule choose the training set. Here the
    denominator is the trip's scheduled time for a pattern the train DID NOT
    RUN: one realtime hop priced against two booked hops and the station dwell
    between them. Numerator and denominator would describe different journeys,
    which is not a slow-or-fast measurement at any bias. It would also fail in
    the worst possible direction — a skipped stop makes the run legitimately
    quicker than the booking, bypasses cluster on bad days, so keeping them
    would push the score toward FAST exactly on the ticks this grade is trying
    to detect, and manufacture the inversion it exists to test for.

    TWO DIFFERENT `n_hops` MEET HERE, and they are not interchangeable. The
    traversal's own `t.n_hops` is what the realtime stop sequence spanned; the
    Scheduled's `want.n_hops` is how many booked hops the timetable puts in that
    same stretch. A bypass is the two disagreeing.

    `want.n_hops is None` is ADMITTED: that is the day-median fallback for a
    trip the static feed cannot name, and it prices the observed pair itself.
    `t.n_hops` is required to be exactly 1 — never None. That is defensive
    rather than restrictive: trace.py only labels a span EXACT when it covers
    one hop, and only right-censored spans carry a null hop count, both of
    which the checks above have already excluded. The requirement is stated on
    `n_hops` anyway because nothing in the Traversal type guarantees it, which
    is the same belt-and-braces traversal.hop_samples uses. Measured over the
    2026-08-12..13 window it rejects nothing.
    """
    out: dict[tuple[str, int], list[float]] = defaultdict(list)
    censored = multi_hop = outside = no_schedule = bypass = 0
    for t in traversals:
        if t.censoring != EXACT or t.to_stop is None:
            censored += 1
            continue
        if t.n_hops != 1:
            multi_hop += 1
            continue
        if not timetable.covers(t.at, t.trip_id):
            outside += 1
            continue
        want = scheduled_for(t, timetable)
        if want is None or want.seconds <= 0:
            no_schedule += 1
            continue
        if want.n_hops not in (None, 1):
            bypass += 1
            continue
        out[(base_route(t.route_id), _snap(t.at + t.seconds))].append(
            t.seconds / want.seconds
        )
    thin = sum(1 for v in out.values() if len(v) < MIN_TICK_SAMPLES)
    return dict(out), RatioStats(
        n_traversals=len(traversals),
        n_attributable=sum(len(v) for v in out.values()),
        n_censored=censored,
        n_multi_hop=multi_hop,
        n_outside_feed_window=outside,
        n_no_schedule=no_schedule,
        n_bypass=bypass,
        n_route_ticks=len(out),
        n_route_ticks_thin=thin,
    )


def progress_ratio(
    traversals: Sequence[Traversal],
    timetable: Timetable,
    *,
    min_samples: int = MIN_TICK_SAMPLES,
) -> tuple[dict[tuple[str, int], float], RatioStats]:
    """(route, tick) -> median observed/scheduled over the hops finishing in it.

    Median rather than mean: one train held at a signal for ten minutes should
    not move a route-tick that fifty other trains ran on time.
    """
    ratios, stats = hop_ratios(traversals, timetable)
    scored = {
        key: statistics.median(values)
        for key, values in ratios.items()
        if len(values) >= min_samples
    }
    return scored, stats


def stalled_share(
    bodies: list[dict[str, Any]],
    *,
    counts_from_stop: StopFilter | None = None,
    min_matched: int = MIN_MATCHED_TRIPS,
) -> dict[tuple[str, int], float]:
    """(route, tick) -> stalled/(advanced+stalled), the advance arm stated so
    higher means worse.

    The CONTINUOUS rate, not the trip-wire built on top of it. The wire's
    thresholded verdict was already graded against this label and the sweep
    established there is no operating point to find; grading the raw rate asks
    the prior question of whether the underlying quantity separates the classes
    at all, and puts the arm on the same footing as the progress ratio, which
    has no threshold either.
    """
    series = build_movement_series(bodies, counts_from_stop=counts_from_stop)
    out: dict[tuple[str, int], float] = {}
    for (route, tick), row in series.items():
        matched = int(row.get("advanced_n") or 0) + int(row.get("stalled_n") or 0)
        if matched < min_matched:
            continue
        out[(route, tick)] = int(row.get("stalled_n") or 0) / matched
    return out


NORMAL = ("normal",)


def _split(
    scores: Mapping[tuple[str, int], float],
    labels: Mapping[tuple[str, int], str],
    positive: Sequence[str],
    negative: Sequence[str],
) -> tuple[list[tuple[str, int]], list[tuple[str, int]]]:
    """The scored ticks split into the two named classes, over the keys both the
    arm and the label can speak to.

    Keys, not values, because the interval below has to know which ticks are
    neighbours.

    A tick in NEITHER list is dropped rather than swept into the negative class.
    That matters for the acute cut: "can this score flag a fresh onset" has to
    be asked against NORMAL ticks, and a chronic tick is a route already known
    to be down. Letting chronic fall through to the negative side would score
    the arm for failing to rank a fresh collapse above a standing one, which is
    not the question and depresses the number for a reason unrelated to onset
    detection.
    """
    pos: list[tuple[str, int]] = []
    neg: list[tuple[str, int]] = []
    for key in scores.keys() & labels.keys():
        label = labels[key]
        if label in positive:
            pos.append(key)
        elif label in negative:
            neg.append(key)
    return pos, neg


def _episode_ids(labels: Mapping[tuple[str, int], str]) -> dict[tuple[str, int], int]:
    """(route, tick) -> the id of the episode it belongs to.

    THE INDEPENDENT UNIT IS NOT THE TICK. A disruption occupies a run of
    consecutive 5-minute ticks on one route, and so does the normal service
    either side of it: the score at 14:05 and the score at 14:10 on the same
    line are close to the same observation. Resampling ticks would treat fifty
    ticks of one incident as fifty independent draws and report an interval
    several times too narrow, which is how a handful of episodes comes to look
    like decisive evidence.

    EPISODES ARE CUT FROM THE FULL LABEL, not from the subset an arm managed to
    score, and that distinction is load-bearing here. A movement arm judges
    about a quarter of the degraded ticks, so a single continuous disruption
    reaches it as a handful of scattered ticks with unscored gaps between them.
    Cutting runs on the scored keys alone would break that one incident into
    several "episodes" and re-inflate exactly the independence this function
    exists to deny. The label has every tick, so it defines the boundaries and
    the arm's ticks are then filed under them.

    An episode is a maximal run of consecutive ticks on one route that are all
    degraded or all normal.
    """
    ids: dict[tuple[str, int], int] = {}
    prev: tuple[str, int] | None = None
    episode = -1
    for key in sorted(labels):
        degraded = labels[key] in DEGRADED
        contiguous = (
            prev is not None
            and key[0] == prev[0]
            and key[1] - prev[1] <= TICK_SECONDS
            and (labels[prev] in DEGRADED) == degraded
        )
        if not contiguous:
            episode += 1
        ids[key] = episode
        prev = key
    return ids


def _grouped(
    keys: Sequence[tuple[str, int]], episodes: Mapping[tuple[str, int], int]
) -> list[list[tuple[str, int]]]:
    """The arm's keys bucketed by the episode each belongs to."""
    out: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for key in keys:
        out[episodes[key]].append(key)
    return list(out.values())


def auc_from(pos: Sequence[float], neg: Sequence[float]) -> float | None:
    """P(a positive scores above a negative), ties counted half."""
    if not pos or not neg:
        return None
    neg_sorted = sorted(neg)
    total = 0.0
    for p in pos:
        lo = bisect.bisect_left(neg_sorted, p)
        hi = bisect.bisect_right(neg_sorted, p)
        total += lo + (hi - lo) / 2
    return total / (len(pos) * len(neg))


@dataclass(frozen=True)
class Auc:
    """One AUC with the sample that produced it and a bootstrap interval.

    The three travel together on purpose. This grade's positive class is a few
    dozen route-ticks against several thousand negatives, and at that size a
    point estimate alone invites exactly the over-reading the epic's own notes
    warn about ("do not interpret the 0.6025 vs 0.5744 acute AUC as a result;
    n=24 positives"). An AUC published without its n and its interval is not a
    result.
    """

    value: float | None
    n_pos: int
    n_neg: int
    # The independent units behind those tick counts: runs of consecutive ticks
    # on one route. The ratio between n_pos and n_pos_clusters is how much of
    # the apparent sample is repetition of the same episode.
    n_pos_clusters: int
    n_neg_clusters: int
    lo: float | None  # 2.5th percentile
    hi: float | None  # 97.5th percentile


# Resamples behind each interval. Enough for a stable 2.5/97.5 percentile pair
# at these class sizes, and cheap: the whole grade re-runs in well under a
# minute.
N_BOOTSTRAP = 2000


def auc(
    scores: Mapping[tuple[str, int], float],
    labels: Mapping[tuple[str, int], str],
    *,
    positive: Sequence[str] = DEGRADED,
    negative: Sequence[str] = NORMAL,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 0,
) -> Auc:
    """P(score of a positive-class tick > score of a negative-class tick), with
    a stratified bootstrap interval.

    0.5 is no signal, below 0.5 is an INVERTED score — the failure mode three
    movement scores already showed against the alert truth, so it has to be
    representable rather than folded away by an absolute value. A None value
    when either class is empty is a different answer from 0.5 and must not be
    reported as one.

    The interval resamples the two classes INDEPENDENTLY with replacement,
    holding both class sizes fixed. That is the right stratification here
    because the question is about the score's ability to separate the classes,
    not about how often a degradation happens.

    It resamples CLUSTERS, not ticks — see `_clusters`. A tick-level bootstrap
    on this data reports intervals several times too narrow, because a
    disruption is one episode observed every five minutes rather than dozens of
    independent degradations. Percentile bootstrap rather than a closed form:
    no assumption about the score's distribution, which matters when one arm's
    normal median sits exactly on 1.0.
    """
    pos_keys, neg_keys = _split(scores, labels, positive, negative)
    point = auc_from([scores[k] for k in pos_keys], [scores[k] for k in neg_keys])
    episodes = _episode_ids(labels)
    pos_groups = _grouped(pos_keys, episodes)
    neg_groups = _grouped(neg_keys, episodes)
    # A single cluster on either side makes the interval a lie in the most
    # dangerous direction: resampling one cluster with replacement returns that
    # same cluster every time, so the band collapses to a point and reads as
    # maximum confidence when the truth is one episode of evidence. Report the
    # estimate with NO interval instead — the same "can't judge" contract used
    # everywhere else here, applied to the uncertainty rather than the value.
    if point is None or len(pos_groups) < 2 or len(neg_groups) < 2:
        return Auc(
            value=point,
            n_pos=len(pos_keys),
            n_neg=len(neg_keys),
            n_pos_clusters=len(pos_groups),
            n_neg_clusters=len(neg_groups),
            lo=None,
            hi=None,
        )
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(n_boot):
        rp = [
            scores[k]
            for _ in range(len(pos_groups))
            for k in pos_groups[rng.randrange(len(pos_groups))]
        ]
        rn = [
            scores[k]
            for _ in range(len(neg_groups))
            for k in neg_groups[rng.randrange(len(neg_groups))]
        ]
        drawn = auc_from(rp, rn)
        if drawn is not None:
            draws.append(drawn)
    draws.sort()
    return Auc(
        value=point,
        n_pos=len(pos_keys),
        n_neg=len(neg_keys),
        n_pos_clusters=len(pos_groups),
        n_neg_clusters=len(neg_groups),
        lo=draws[int(0.025 * (len(draws) - 1))],
        hi=draws[int(0.975 * (len(draws) - 1))],
    )


def _centred(scores: Mapping[tuple[str, int], float]) -> dict[tuple[str, int], float]:
    """Each route's scores minus that route's own median.

    Pooled AUC over route-ticks can be won by knowing which ROUTES run slow
    rather than which TICKS are degraded: a line that habitually runs 1.3x its
    timetable would score every one of its ticks above a punctual line's worst.
    Centring per route removes the between-route level and asks the question
    the label actually poses — is THIS route worse than usual right now.
    """
    by_route: dict[str, list[float]] = defaultdict(list)
    for (route, _tick), value in scores.items():
        by_route[route].append(value)
    centre = {route: statistics.median(v) for route, v in by_route.items()}
    return {key: value - centre[key[0]] for key, value in scores.items()}


@dataclass(frozen=True)
class Grade:
    """One arm scored against the label over one window."""

    arm: str
    n_scored: int  # route-ticks the arm produced a number for
    n_graded: int  # of those, ones the label can also speak to
    n_degraded: int  # of those, ones the label calls degraded
    base_rate: float | None
    auc: Auc  # degraded (acute or chronic) against normal
    auc_within_route: Auc  # the same, after removing each route's own level
    # Fresh onsets against NORMAL ticks only. Chronic ticks are excluded from
    # both sides rather than counted as negatives: they are routes already known
    # to be down, and ranking a fresh collapse against a standing one is a
    # different question from detecting the onset.
    auc_acute: Auc
    # Of every degraded tick in the label, the share this arm could judge. The
    # coverage number, and the one that sank the advance arm.
    n_label_degraded: int
    degraded_coverage: float | None
    median_normal: float | None
    median_degraded: float | None


def grade(
    arm: str,
    scores: Mapping[tuple[str, int], float],
    labels: Mapping[tuple[str, int], str],
) -> Grade:
    """Score one arm against the label, on the intersection of what both can
    speak to, with the size of that intersection reported beside the skill."""
    shared = scores.keys() & labels.keys()
    degraded = [k for k in shared if labels[k] in DEGRADED]
    normal = [k for k in shared if labels[k] not in DEGRADED]
    label_degraded = sum(1 for v in labels.values() if v in DEGRADED)
    return Grade(
        arm=arm,
        n_scored=len(scores),
        n_graded=len(shared),
        n_degraded=len(degraded),
        base_rate=(len(degraded) / len(shared)) if shared else None,
        auc=auc(scores, labels),
        auc_within_route=auc(_centred(dict(scores)), labels),
        auc_acute=auc(scores, labels, positive=("acute",)),
        n_label_degraded=label_degraded,
        degraded_coverage=(len(degraded) / label_degraded) if label_degraded else None,
        median_normal=(
            statistics.median([scores[k] for k in normal]) if normal else None
        ),
        median_degraded=(
            statistics.median([scores[k] for k in degraded]) if degraded else None
        ),
    )


def _within(
    labels: Mapping[tuple[str, int], str], lo: int, hi: int
) -> dict[tuple[str, int], str]:
    """The label restricted to the graded span. The label's baseline wants more
    history than the trace covers, so it is fitted over a longer window and then
    cut to the window a movement arm can actually be graded over — otherwise
    every label tick outside the trace counts against `degraded_coverage` as an
    abstention the arm was never offered the chance to make."""
    return {k: v for k, v in labels.items() if lo <= k[1] <= hi}


def main(argv: list[str] | None = None) -> int:
    """Grade both movement arms against the assigned_n label over the trace's
    own window."""
    parser = argparse.ArgumentParser(
        description="Grade the progress ratio and the advance rate against the "
        "assigned_n degradation label"
    )
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument(
        "--baseline-days",
        type=int,
        default=11,
        help="trailing trip-updates history the label's baseline is fitted over",
    )
    parser.add_argument("--min-samples", type=int, default=MIN_TICK_SAMPLES)
    args = parser.parse_args(argv)

    today = datetime.now(UTC).date()
    end = args.end_date or today
    start = args.start_date or (end - timedelta(days=1))

    cfg = load_config()
    client = make_client(cfg)

    from training.archive_read import load_traversals

    loaded = load_traversals(start, end)
    traversals = loaded.rows
    print(f"traversals — {loaded.summary()}", file=sys.stderr)
    loaded.require_pooled("traversals")
    # An empty trace would make every count below read as "detected nothing"
    # rather than "never ran", the same confusion the planned-work grade had to
    # be taught to distinguish.
    if not traversals:
        raise SystemExit(f"no traversals in {start}..{end}")
    span_lo = _snap(min(t.at for t in traversals))
    span_hi = _snap(max(t.at + t.seconds for t in traversals))

    # One fetch, both parses: the progress arm needs the timetable to price a
    # hop, and the advance arm needs the successor skeleton to know which stops
    # are through stops. Fetching the zip twice would risk the two arms being
    # graded against different feed versions.
    data = fetch_gtfs_zip()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        timetable = parse_timetable(zf)
        through = through_stops(successors(zf))
    print(
        f"feed {timetable.version.version}, {len(through)} through stops",
        file=sys.stderr,
    )

    # The label's baseline is a per-(route, schedule_bin) median and wants more
    # history than the trace covers, so it is fitted over its own trailing
    # window and the resulting label is then cut to the trace span.
    label_start = end - timedelta(days=args.baseline_days - 1)
    series = build_service_series(
        fetch_trip_update_metrics(
            cfg, start_date=label_start, end_date=end, client=client
        )
    )
    baseline = compute_baseline(series, bin_fn=BIN_FN)
    _disruptions, all_labels = build_labels(series, baseline)
    labels = _within(all_labels, span_lo, span_hi)
    print(
        f"{len(series)} service ticks, {len(baseline)} baseline cells, "
        f"{len(labels)} labelled ticks over the trace span",
        file=sys.stderr,
    )

    progress, ratio_stats = progress_ratio(
        traversals, timetable, min_samples=args.min_samples
    )
    advance = stalled_share(
        fetch_vehicle_metrics(cfg, start_date=start, end_date=end, client=client),
        counts_from_stop=lambda route, direction, stop: (
            (route, direction, stop) in through
        ),
    )

    print(
        json.dumps(
            {
                "window": {
                    "trace": [start.isoformat(), end.isoformat()],
                    "span": [span_lo, span_hi],
                    "label_baseline_from": label_start.isoformat(),
                    "feed_version": timetable.version.version,
                    "min_tick_samples": args.min_samples,
                },
                "ratio_supply": asdict(ratio_stats),
                "label": {
                    "ticks_over_span": len(labels),
                    "degraded_ticks": sum(1 for v in labels.values() if v in DEGRADED),
                    "acute_ticks": sum(1 for v in labels.values() if v == "acute"),
                },
                "arms": [
                    asdict(grade("progress_ratio", progress, labels)),
                    asdict(grade("stalled_share", advance, labels)),
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
