"""Synthetic-series tests for training.degradation_label (no R2).

T0 sits at 10:00 ET (winter, UTC-5). The label bins by schedule_bin (ET
weekday/weekend x hour), so a multi-hour synthetic series spans several
baseline cells. Rather than generate days of ticks just to clear
compute_baseline's min_samples in every one of them, these tests pin the
baseline directly with `_level_baseline` — what a long clean window converges
to. compute_baseline's own median/min_samples behaviour is covered in
test_trip_validation.py.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

from momentarily.hmm import tod_bin
from training.degradation_label import (
    ACUTE_ONSET_TICKS,
    AT_BASELINE_LOOKBACK_TICKS,
    AT_BASELINE_RATIO,
    BIN_FN,
    DEGRADE_DEBOUNCE_TICKS,
    DEGRADE_RATIO,
    RECOVER_RATIO,
    _hourly_prevalence,  # pyright: ignore[reportPrivateUsage]
    _onsets_by_et_hour,  # pyright: ignore[reportPrivateUsage]
    _week_windows,  # pyright: ignore[reportPrivateUsage]
    label_ticks,
    measure_degradation,
    was_at_baseline,
)
from training.load import TICK_SECONDS
from training.load_r2 import compute_baseline, derive_actual_recovery

# 2024-01-10 15:00 UTC == 10:00 ET.
T0 = 1_704_898_800


def _flat(
    route: str, n: int, value: int, *, t0: int = T0
) -> dict[tuple[str, int], int]:
    return {(route, t0 + i * TICK_SECONDS): value for i in range(n)}


def _merge(
    *series: dict[tuple[str, int], int],
) -> dict[tuple[str, int], int]:
    out: dict[tuple[str, int], int] = {}
    for s in series:
        out.update(s)
    return out


def _level_baseline(
    levels: dict[str, float], *, ticks: int, t0: int = T0
) -> dict[tuple[str, str], float]:
    """A schedule_bin baseline holding each route at a fixed level across every
    bin the [t0, t0 + ticks) window touches."""
    return {
        (route, BIN_FN(t0 + i * TICK_SECONDS)): level
        for route, level in levels.items()
        for i in range(ticks)
    }


def _recovery(
    series: dict[tuple[str, int], int], baseline: dict[tuple[str, str], float]
):
    return derive_actual_recovery(
        series,
        baseline,
        bin_fn=BIN_FN,
        degrade_ratio=DEGRADE_RATIO,
        recover_ratio=RECOVER_RATIO,
        debounce=DEGRADE_DEBOUNCE_TICKS,
    )


# --- label_ticks: clean / sustained / fresh / shuttle / debounce ----------


def test_clean_route_yields_no_label():
    """A route that never deviates from its own baseline stays "normal" at
    every judgeable tick -- no acute or chronic label anywhere."""
    series = _flat("A", 40, 8)
    baseline = _level_baseline({"A": 8}, ticks=40)
    disruptions = _recovery(series, baseline)
    assert disruptions == []

    labels = label_ticks(series, baseline, disruptions)
    assert len(labels) == 40
    assert set(labels.values()) == {"normal"}


def test_sustained_collapse_yields_chronic_label():
    """A collapse that outlasts the onset window reads acute for the first
    ACUTE_ONSET_TICKS ticks, then chronic for the remainder -- not one label
    for the whole interval."""
    normal = _flat("A", 40, 8)
    collapse_start = T0 + 40 * TICK_SECONDS
    collapse = _flat("A", 15, 1, t0=collapse_start)  # ratio 0.125 << 0.5
    recover_start = collapse_start + 15 * TICK_SECONDS
    recover = _flat("A", 4, 8, t0=recover_start)
    series = _merge(normal, collapse, recover)
    baseline = _level_baseline({"A": 8}, ticks=59)

    disruptions = _recovery(series, baseline)
    assert len(disruptions) == 1
    d = disruptions[0]
    assert d.start_tick == collapse_start
    assert d.recovered_tick == recover_start

    labels = label_ticks(series, baseline, disruptions)
    onset_end = collapse_start + ACUTE_ONSET_TICKS * TICK_SECONDS
    assert onset_end < recover_start  # the collapse really does outlast onset
    for i in range(ACUTE_ONSET_TICKS):
        assert labels[("A", collapse_start + i * TICK_SECONDS)] == "acute"
    assert labels[("A", onset_end)] == "chronic"
    assert labels[("A", recover_start - TICK_SECONDS)] == "chronic"
    assert (
        labels[("A", recover_start)] == "normal"
    )  # recovered_tick itself is not degraded


def test_fresh_collapse_acute_onset_at_right_tick():
    """A short collapse (fully inside the onset window) reads acute at
    EXACTLY derive_actual_recovery's start_tick -- not one tick early or
    late -- and never reaches chronic."""
    normal = _flat("A", 40, 8)
    collapse_start = T0 + 40 * TICK_SECONDS
    collapse = _flat("A", 4, 1, t0=collapse_start)  # 4 ticks < ACUTE_ONSET_TICKS (6)
    recover_start = collapse_start + 4 * TICK_SECONDS
    recover = _flat("A", 4, 8, t0=recover_start)
    series = _merge(normal, collapse, recover)
    baseline = _level_baseline({"A": 8}, ticks=48)

    disruptions = _recovery(series, baseline)
    assert len(disruptions) == 1
    d = disruptions[0]
    # debounce=2: the run OPENS on the first low tick, not the tick debounce
    # confirms on -- an off-by-one here would misplace the onset by one tick.
    assert d.start_tick == collapse_start

    labels = label_ticks(series, baseline, disruptions)
    assert labels[("A", collapse_start - TICK_SECONDS)] == "normal"
    for i in range(4):
        assert labels[("A", collapse_start + i * TICK_SECONDS)] == "acute"
    assert "chronic" not in labels.values()


def test_shuttle_low_baseline_not_flagged():
    """A shuttle whose own normal running level is low must not be flagged
    for running low -- judged against its own baseline, never a global
    floor. A trunk route collapsing to that SAME absolute count (2) is a
    real event; the shuttle running at 2 always is not."""
    shuttle = _flat("S", 40, 2)  # always 2 trains -- ratio 1.0 forever
    trunk_normal = _flat("T", 40, 20)
    trunk_collapse_start = T0 + 40 * TICK_SECONDS
    trunk_collapse = _flat("T", 10, 2, t0=trunk_collapse_start)  # ratio 0.1
    trunk_recover = _flat("T", 4, 20, t0=trunk_collapse_start + 10 * TICK_SECONDS)
    series = _merge(shuttle, trunk_normal, trunk_collapse, trunk_recover)
    baseline = _level_baseline({"S": 2, "T": 20}, ticks=54)

    disruptions = _recovery(series, baseline)
    assert {d.route for d in disruptions} == {"T"}

    labels = label_ticks(series, baseline, disruptions)
    assert all(labels[("S", tick)] == "normal" for _route, tick in shuttle)
    assert labels[("T", trunk_collapse_start)] == "acute"


def test_debounce_rejects_one_tick_dip():
    """A single-tick dip below the degrade floor that recovers on the very
    next tick never accumulates DEGRADE_DEBOUNCE_TICKS consecutive low
    ticks, so it must not open a disruption or produce any non-normal
    label."""
    before = _flat("A", 20, 8)
    dip_tick = T0 + 20 * TICK_SECONDS
    dip = {("A", dip_tick): 1}  # ratio 0.125, exactly one tick
    after = _flat("A", 20, 8, t0=dip_tick + TICK_SECONDS)
    series = _merge(before, dip, after)
    baseline = _level_baseline({"A": 8}, ticks=41)

    disruptions = _recovery(series, baseline)
    assert disruptions == []

    labels = label_ticks(series, baseline, disruptions)
    assert set(labels.values()) == {"normal"}
    assert labels[("A", dip_tick)] == "normal"


# --- was_at_baseline --------------------------------------------------


def test_was_at_baseline_true_when_lookback_window_is_all_normal():
    lead_in = _flat("A", AT_BASELINE_LOOKBACK_TICKS + 4, 8)
    start_tick = T0 + (AT_BASELINE_LOOKBACK_TICKS + 4) * TICK_SECONDS
    baseline = _level_baseline({"A": 8}, ticks=40)
    assert was_at_baseline(lead_in, baseline, "A", start_tick)


def test_was_at_baseline_false_when_a_lookback_tick_dips_below_the_ratio():
    """Below AT_BASELINE_RATIO but still above DEGRADE_RATIO -- a state
    derive_actual_recovery's own machine would call "not yet disrupted", but
    is not "genuinely at baseline". A caller that reused DEGRADE_RATIO here
    instead of AT_BASELINE_RATIO would wrongly pass this case."""
    baseline = _level_baseline({"A": 8}, ticks=40)
    start_tick = T0 + 40 * TICK_SECONDS
    series = _flat("A", AT_BASELINE_LOOKBACK_TICKS, 8, t0=T0 + 34 * TICK_SECONDS)
    limping_tick = start_tick - 2 * TICK_SECONDS
    ratio = 0.6
    assert DEGRADE_RATIO < ratio < AT_BASELINE_RATIO
    series[("A", limping_tick)] = round(8 * ratio)
    assert not was_at_baseline(series, baseline, "A", start_tick)


def test_was_at_baseline_false_when_data_missing():
    """No baseline / no series entry in the lookback window fails closed --
    "at baseline" is never assumed absent evidence."""
    baseline = _level_baseline({"A": 8}, ticks=40)
    start_tick = T0 + 40 * TICK_SECONDS
    assert not was_at_baseline({}, baseline, "A", start_tick)


# --- _week_windows / _hourly_prevalence / measure_degradation -------------


def test_week_windows_bins_by_seven_days_with_a_short_final_bin():
    windows = _week_windows(date(2026, 8, 1), date(2026, 8, 11))
    assert windows == [
        (date(2026, 8, 1), date(2026, 8, 7)),
        (date(2026, 8, 8), date(2026, 8, 11)),
    ]


def test_hourly_prevalence_pools_across_routes_and_reports_the_share():
    hour0 = int(datetime(2026, 1, 1, 0, 0, tzinfo=UTC).timestamp())
    hour1 = int(datetime(2026, 1, 1, 1, 0, tzinfo=UTC).timestamp())
    labels = {
        ("A", hour0): "normal",
        ("A", hour0 + TICK_SECONDS): "acute",
        ("B", hour0): "chronic",
        ("A", hour1): "normal",
    }
    out = _hourly_prevalence(labels)
    assert out[0] == {"prevalence": 2 / 3, "degraded_ticks": 2, "total_ticks": 3}
    assert out[1] == {"prevalence": 0.0, "degraded_ticks": 0, "total_ticks": 1}


def test_measure_degradation_reports_events_prevalence_and_baseline_survival():
    normal = _flat("A", 40, 8)
    collapse_start = T0 + 40 * TICK_SECONDS
    collapse = _flat("A", 15, 1, t0=collapse_start)
    recover = _flat("A", 4, 8, t0=collapse_start + 15 * TICK_SECONDS)
    series = _merge(normal, collapse, recover)
    baseline = _level_baseline({"A": 8}, ticks=59)

    start = datetime.fromtimestamp(T0, tz=UTC).date()
    end = datetime.fromtimestamp(collapse_start + 19 * TICK_SECONDS, tz=UTC).date()
    report = measure_degradation(series, baseline, start, end)

    assert report["total_events"] == 1
    assert report["distinct_routes"] == 1
    assert report["events_reaching_chronic"] == 1
    assert report["acute_ticks"] == ACUTE_ONSET_TICKS
    assert report["chronic_ticks"] == 15 - ACUTE_ONSET_TICKS
    # the collapse follows 40 baseline ticks, comfortably >= the 30-min lookback
    assert report["events_surviving_30min_at_baseline"] == 1
    assert sum(w["acute_events"] for w in report["weekly_acute_events"]) == 1
    assert sum(v["degraded_ticks"] for v in report["hourly_prevalence"].values()) == 15
    assert report["onsets_by_et_hour"] == {13: 1}  # collapse_start is 13:20 ET


# --- the bin-edge artifact this label's granularity exists to avoid --------


def test_quiet_edge_of_a_wide_bin_is_not_a_collapse():
    """A route running its GENUINE early-morning service level must read
    normal. Under the HMM's tod_bin that hour shares a bucket with the rush
    peak, so the bucket median lands far above real 06:00 service and the
    route reads collapsed at the top of every weekday -- the artifact that put
    26.8% of degraded ticks in one UTC hour. Hourly bins judge 06:00 against
    06:00."""
    # 2024-01-10 (a Wednesday). ET 06:00 through 09:59 = tod_bin 1.
    six_am_et = int(datetime(2024, 1, 10, 11, 0, tzinfo=UTC).timestamp())
    ticks_per_hour = 3600 // TICK_SECONDS
    # Real service ramping into the rush: 6 trains at 06:00, 30 by 08:00.
    by_hour = {0: 6, 1: 12, 2: 30, 3: 30}
    series: dict[tuple[str, int], int] = {}
    # Two weeks of identical weekday mornings, so every hourly cell clears
    # compute_baseline's min_samples the way a real training window does.
    for day in range(14):
        day_t0 = six_am_et + day * 86_400
        for hour, level in by_hour.items():
            for i in range(ticks_per_hour):
                series[("A", day_t0 + hour * 3600 + i * TICK_SECONDS)] = level

    wide = compute_baseline(series, bin_fn=tod_bin, min_samples=20)
    fine = compute_baseline(series, bin_fn=BIN_FN, min_samples=20)
    # The wide bucket pools 06:00-09:59 into one median, pulled up by its
    # 30-train core; the fine bucket measures 06:00 against itself.
    assert wide[("A", 1)] == 21.0
    assert fine[("A", "wd06")] == 6.0

    wide_onsets = derive_actual_recovery(
        series,
        wide,
        bin_fn=tod_bin,
        degrade_ratio=DEGRADE_RATIO,
        recover_ratio=RECOVER_RATIO,
        debounce=DEGRADE_DEBOUNCE_TICKS,
    )
    # 6/30 = 0.2 < DEGRADE_RATIO: every morning opens a phantom disruption.
    assert len(wide_onsets) >= 13
    assert set(_onsets_by_et_hour(wide_onsets)) == {6}

    assert _recovery(series, fine) == []
