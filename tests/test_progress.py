"""Grading movement arms against the assigned_n label (training/progress.py).

Synthetic traversals against a synthetic timetable, and synthetic scores against
synthetic labels — no R2, no trace reconstruction, no GTFS fetch. Each case pins
one rule about which observations reach the score, which tick they land in, how
an AUC is allowed to be reported, and what the coverage number counts.
"""

from __future__ import annotations

import zipfile
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

from training.gtfs_static import Timetable, timetable
from training.load import TICK_SECONDS
from training.progress import (
    Auc,
    auc,
    grade,
    hop_ratios,
    progress_ratio,
    stalled_share,
)
from training.trace import EXACT, INTERVAL, RIGHT, Traversal

from .conftest import make_gtfs_zip

ET = ZoneInfo("America/New_York")
WEEKDAY = date(2026, 8, 12)
TRIP = "072000_A..S01R"
# A weekday noon, snapped to a tick boundary so a test can reason about which
# tick an observation lands in without arithmetic.
AT = (
    int(datetime.combine(WEEKDAY, time(12, 0), tzinfo=ET).timestamp())
    // TICK_SECONDS
    * TICK_SECONDS
)


def _feed() -> zipfile.ZipFile:
    """One southbound weekday pattern: A1S -> A2S is 90 scheduled seconds,
    A2S -> A3S is 400, so A1S -> A3S is 490 across two hops."""
    return make_gtfs_zip(
        [f"A,{TRIP},Weekday,X,1,A..S01R"],
        [
            f"{TRIP},A1S,12:00:00,12:00:00,1",
            f"{TRIP},A2S,12:01:30,12:01:30,2",
            f"{TRIP},A3S,12:08:10,12:08:10,3",
        ],
    )


def _timetable() -> Timetable:
    return timetable(_feed())


def _exact(
    seconds: int,
    *,
    at: int = AT,
    frm: str = "A1S",
    to: str = "A3S",
    n_hops: int = 1,
    route: str = "A",
) -> Traversal:
    """One completed hop. Defaults to A1S -> A2S priced at 90 scheduled
    seconds."""
    return Traversal(
        trip_id=TRIP,
        route_id=route,
        direction="south",
        from_stop=frm,
        to_stop=to,
        at=at,
        seconds=seconds,
        moving_seconds=None,
        n_hops=n_hops,
        censoring=EXACT,
    )


def _hop(seconds: int, *, at: int = AT, route: str = "A") -> Traversal:
    return _exact(seconds, at=at, frm="A1S", to="A2S", route=route)


def test_ratio_is_observed_over_what_the_trips_own_timetable_allows():
    """The score's unit: 180 seconds on a hop the trip is booked 90 for is 2.0,
    and an on-time hop is exactly 1.0."""
    scores, stats = hop_ratios([_hop(180), _hop(90)], _timetable())
    assert sorted(v for values in scores.values() for v in values) == [1.0, 2.0]
    assert stats.n_attributable == 2


def test_a_bypass_is_dropped_rather_than_priced_against_a_run_it_did_not_make():
    """One realtime hop that the trip's own pattern says covers two booked hops
    means a station was skipped. Pricing 200 observed seconds against the 490 the
    two hops are booked for would read as running fast — the artificial speedup
    that clusters on bad days — so it is excluded and counted."""
    scores, stats = hop_ratios([_exact(200, frm="A1S", to="A3S")], _timetable())
    assert scores == {}
    assert stats.n_bypass == 1
    assert stats.n_attributable == 0


def test_an_observation_lands_in_the_tick_it_finished_in():
    """A hop's duration is unknown until it ends, so a live nowcast could only
    score it at completion. One starting at a tick boundary and running past it
    belongs to the NEXT tick, not the one it started in."""
    scores, _ = hop_ratios([_hop(TICK_SECONDS + 60, at=AT)], _timetable())
    assert list(scores) == [("A", AT + TICK_SECONDS)]


def test_censored_and_unpriceable_observations_never_reach_the_score():
    """A right-censored span has no destination, an interval span covers several
    hops, and a hop the timetable cannot price has no denominator. Each is
    counted under its own reason rather than pooled into one discard bucket."""
    right = Traversal(
        trip_id=TRIP,
        route_id="A",
        direction="south",
        from_stop="A1S",
        to_stop=None,
        at=AT,
        seconds=400,
        moving_seconds=None,
        n_hops=None,
        censoring=RIGHT,
    )
    interval = Traversal(
        trip_id=TRIP,
        route_id="A",
        direction="south",
        from_stop="A1S",
        to_stop="A3S",
        at=AT,
        seconds=500,
        moving_seconds=None,
        n_hops=2,
        censoring=INTERVAL,
    )
    unpriced = _exact(120, frm="Z9S", to="Z8S")
    scores, stats = hop_ratios([right, interval, unpriced], _timetable())
    assert scores == {}
    assert (stats.n_censored, stats.n_no_schedule) == (2, 1)


def test_a_route_tick_under_the_sample_floor_is_absent_not_normal():
    """Abstention, not a default: four trains is one bad run away from reading as
    a route-wide slowdown, so the tick carries no score at all."""
    thin, _ = progress_ratio([_hop(90) for _ in range(4)], _timetable())
    assert thin == {}
    enough, stats = progress_ratio([_hop(90) for _ in range(5)], _timetable())
    assert enough == {("A", AT): 1.0}
    assert stats.n_route_ticks_thin == 0


def test_the_tick_score_is_a_median_so_one_held_train_does_not_move_it():
    """Four on-time hops and one that took over three times its booked time, all
    finishing inside the same tick: the tick reads on time, because it was.

    The crawl has to still FINISH in the tick to be in its median at all — a hop
    long enough to spill past the boundary is scored in the tick it lands in,
    which is the completion rule above, not a robustness property.
    """
    scores, _ = progress_ratio(
        [_hop(90), _hop(90), _hop(90), _hop(90), _hop(290)], _timetable()
    )
    assert list(scores.values()) == [1.0]


def _vehicle_body(at: int, *, advanced: int, stalled: int) -> dict[str, object]:
    return {
        "observed_at": at,
        "rows": {"A": {"advanced_n": advanced, "stalled_n": stalled}},
    }


def test_the_advance_arm_is_oriented_so_higher_means_worse():
    """Stated as the STALLED share, so an inverted AUC is a real inversion rather
    than a sign convention: three stalls in four matched trips is 0.75."""
    got = stalled_share([_vehicle_body(AT, advanced=1, stalled=3)])
    assert got == {("A", AT): 0.75}


def test_the_advance_arm_abstains_below_the_matched_trip_floor():
    """Two matched trips cannot support a rate, so the tick is absent."""
    assert stalled_share([_vehicle_body(AT, advanced=1, stalled=1)]) == {}


def _labelled(
    scores: dict[tuple[str, int], float], degraded: set[tuple[str, int]]
) -> dict[tuple[str, int], str]:
    return {k: ("chronic" if k in degraded else "normal") for k in scores}


def test_auc_separates_perfect_inverted_and_tied_scores():
    """Three cases that must be distinguishable: a score that ranks every
    degraded tick above every normal one, one that does the exact opposite, and
    one that says the same thing everywhere."""
    keys = [("A", AT), ("A", AT + 300), ("B", AT), ("B", AT + 300)]
    degraded = {keys[0], keys[1]}
    perfect = dict(zip(keys, [2.0, 1.9, 1.0, 1.1], strict=True))
    assert auc(perfect, _labelled(perfect, degraded)).value == 1.0
    inverted = dict(zip(keys, [1.0, 1.1, 2.0, 1.9], strict=True))
    assert auc(inverted, _labelled(inverted, degraded)).value == 0.0
    flat = dict.fromkeys(keys, 1.0)
    assert auc(flat, _labelled(flat, degraded)).value == 0.5


def test_the_acute_cut_excludes_chronic_ticks_from_both_sides():
    """An onset must be graded against NORMAL, not against routes already known
    to be down. Chronic ticks score high, so sweeping them into the negative
    class would penalise the arm for failing to rank a fresh collapse above a
    standing one — a different question, and one that drags the number down for
    a reason that has nothing to do with onset detection.

    Here the score is perfect at acute-vs-normal and the chronic ticks outrank
    everything: counted as negatives the AUC would be 0.0, excluded it is 1.0.
    """
    scores = {
        ("A", AT): 1.0,  # normal
        ("A", AT + 300): 2.0,  # acute
        ("A", AT + 600): 9.0,  # chronic
        ("A", AT + 900): 9.0,  # chronic
    }
    labels = {
        ("A", AT): "normal",
        ("A", AT + 300): "acute",
        ("A", AT + 600): "chronic",
        ("A", AT + 900): "chronic",
    }
    got = auc(scores, labels, positive=("acute",))
    assert got.value == 1.0
    assert (got.n_pos, got.n_neg) == (1, 1)


def test_an_empty_class_is_reported_as_no_answer_not_as_no_skill():
    """None and 0.5 are different claims. A window with no degraded tick has not
    shown the score to be useless, it has failed to test it — and the class
    counts still have to say so."""
    scores = {("A", AT): 1.0, ("A", AT + 300): 2.0}
    got = auc(scores, _labelled(scores, set()))
    assert got == Auc(
        value=None,
        n_pos=0,
        n_neg=2,
        n_pos_clusters=0,
        n_neg_clusters=1,
        lo=None,
        hi=None,
    )


def test_the_interval_resamples_episodes_not_ticks():
    """One long incident observed every five minutes is ONE draw, not fifty.

    Two arrangements of the SAME 40 positive scores against the same negatives:
    two incidents of 20 consecutive ticks each, versus 40 isolated one-tick
    incidents. Identical point estimate, identical n — and the scattered
    arrangement must produce a strictly narrower interval, because it genuinely
    carries more independent evidence. A tick-level bootstrap reports the two
    identically and so overstates the first.

    The scores overlap on purpose. Under perfect separation every resample
    returns 1.0 whatever the clustering, and the test could not fail.
    """
    # Negatives spread across the unit interval, offset off the positives' values
    # so no comparison lands on a tie. Split over two routes because a single
    # block is one episode and earns no interval at all; both arrangements share
    # these same negatives, so the contrast stays on the positive side.
    neg = {(f"N{i % 2}", AT + (i // 2) * 300): (i + 0.5) / 60 for i in range(60)}
    # Each incident is internally homogeneous — the realistic case, and the one
    # where treating its ticks as independent is most wrong.
    runs = dict(neg)
    runs.update({("A", AT + i * 300): 0.6 for i in range(20)})
    runs.update({("B", AT + i * 300): 0.4 for i in range(20)})
    scattered = dict(neg)
    scattered.update({("A", AT + i * 900): 0.6 for i in range(20)})
    scattered.update({("B", AT + i * 900 + 300): 0.4 for i in range(20)})

    incidents = {k for k in runs if not k[0].startswith("N")}
    clustered = auc(runs, _labelled(runs, incidents))
    spread = auc(
        scattered,
        _labelled(scattered, {k for k in scattered if not k[0].startswith("N")}),
    )

    assert clustered.n_pos == spread.n_pos == 40
    assert clustered.value == spread.value
    assert clustered.n_pos_clusters == 2
    assert spread.n_pos_clusters == 40
    assert clustered.lo is not None
    assert spread.lo is not None
    assert clustered.lo < spread.lo


def test_the_interval_brackets_the_point_estimate_and_records_both_class_sizes():
    """The guardrail against reading a few dozen positives as a result: the
    reported interval must contain the point estimate and carry the n that
    produced it.

    Positives are spaced apart so they form several episodes; a single run of
    consecutive ticks is one episode and gets no interval at all, which the next
    case covers.
    """
    keys = [("A", AT + i * 300) for i in range(40)]
    scores = {k: float(i % 5) for i, k in enumerate(keys)}
    positives = {keys[i] for i in range(0, 40, 5)}
    labels = _labelled(scores, positives)
    got = auc(scores, labels, n_boot=200)
    assert got.value is not None
    assert got.lo is not None
    assert got.hi is not None
    assert got.lo <= got.value <= got.hi
    assert (got.n_pos, got.n_neg) == (8, 32)
    assert got.n_pos_clusters == 8


def test_one_episode_gets_a_point_estimate_and_no_interval():
    """A single episode resampled with replacement returns itself every time, so
    a bootstrap band around it collapses to a point and reads as certainty
    earned from one incident. The estimate still stands; the interval is
    withheld rather than fabricated."""
    keys = [("A", AT + i * 300) for i in range(20)]
    scores = {k: float(i) for i, k in enumerate(keys)}
    # Five consecutive ticks on one route: one episode.
    labels = _labelled(scores, set(keys[10:15]))
    got = auc(scores, labels, n_boot=200)
    assert got.n_pos_clusters == 1
    assert got.value is not None
    assert got.lo is None
    assert got.hi is None


def test_a_partly_observed_disruption_is_still_one_episode():
    """Episode boundaries come from the LABEL, not from the ticks an arm
    happened to score.

    A movement arm judges roughly a quarter of the degraded ticks, so one
    continuous disruption reaches it as scattered ticks with unscored gaps. If
    the runs were cut on the scored keys those gaps would split one incident
    into several, re-inflating the independence the clustering exists to deny —
    and the interval would tighten on evidence that does not exist.
    """
    labels = {("A", AT + i * 300): "normal" for i in range(40)}
    # One unbroken disruption: twenty consecutive degraded ticks.
    for i in range(10, 30):
        labels[("A", AT + i * 300)] = "chronic"
    # The arm can only judge four of them, spread across the disruption.
    scored_degraded = [("A", AT + i * 300) for i in (11, 17, 23, 28)]
    scores = dict.fromkeys(scored_degraded, 2.0)
    scores.update({("A", AT + i * 300): 1.0 for i in range(10)})
    got = auc(scores, labels)
    assert got.n_pos == 4
    assert got.n_pos_clusters == 1


def test_centring_asks_whether_this_route_is_worse_than_usual():
    """A habitually slow line that is rarely degraded, and a punctual line that
    usually is. Every tick of the slow line outscores every DEGRADED tick of the
    punctual one, so the pooled AUC reads badly inverted while the score is in
    fact ordered correctly inside each route.

    The class imbalance across routes is the point, not a convenience: with the
    same normal/degraded mix on both routes, pooling cannot invert, so a level
    confound only bites when the routes that run slow are not the routes that
    are degraded. That is the case both numbers exist to tell apart.
    """
    scores: dict[tuple[str, int], float] = {}
    degraded: set[tuple[str, int]] = set()

    def put(route: str, i: int, value: float, is_degraded: bool) -> None:
        key = (route, AT + i * 300)
        scores[key] = value + i * 0.001
        if is_degraded:
            degraded.add(key)

    for i in range(18):
        put("SLOW", i, 1.300, False)
    for i in range(18, 20):
        put("SLOW", i, 1.332, True)
    for i in range(2):
        put("FAST", i, 1.000, False)
    for i in range(2, 20):
        put("FAST", i, 1.048, True)

    got = grade("test", scores, _labelled(scores, degraded))
    assert got.auc.value is not None
    assert got.auc.value < 0.5
    assert got.auc_within_route.value is not None
    assert got.auc_within_route.value > 0.6


def test_coverage_counts_the_degraded_ticks_the_arm_could_not_judge():
    """The number that sank the advance arm. A score that abstains on three
    quarters of the degradation has not been beaten on skill, and the grade has
    to say so rather than reporting a clean AUC over the quarter it saw."""
    scores = {("A", AT): 1.0, ("A", AT + 300): 2.0}
    labels = {
        ("A", AT): "normal",
        ("A", AT + 300): "chronic",
        ("A", AT + 600): "chronic",  # never scored
        ("A", AT + 900): "acute",  # never scored
    }
    got = grade("test", scores, labels)
    assert (got.n_degraded, got.n_label_degraded) == (1, 3)
    assert got.degraded_coverage is not None
    assert abs(got.degraded_coverage - 1 / 3) < 1e-9
