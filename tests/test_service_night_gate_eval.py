"""Gate-eval logic for the supply baseline's independent-night gate: the
late-night filter, the alert-confirmed-normal night labels, the gate partition,
the night-clustered false-alarm rate, and the abstention confusion."""

from __future__ import annotations

from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from momentarily.hmm import Observation, schedule_bin
from training.eval_common import et_midnight
from training.load import TickObservation
from training.load_r2 import ServiceQuantiles, compute_service_quantiles
from training.service_night_gate_eval import (
    abstention_tradeoff,
    false_alarm_rate,
    is_weekend_late_tick,
    night_counts_by_cell,
    night_labels,
    partition_by_gate,
    service_night_ticks,
)

_ET = ZoneInfo("America/New_York")


def _tick(y: int, mo: int, d: int, h: int) -> int:
    """Epoch seconds for a wall-clock ET hour — schedule_bin and the service-night
    key both read ET, so build the fixtures in ET to land in the intended bin
    and night."""
    return int(datetime(y, mo, d, h, tzinfo=_ET).timestamp())


def _obs(**flags: bool) -> Observation:
    return Observation(
        alert_count=0, severity_sum=0, has_suspended_alert=False, **flags
    )


def test_short_fit_cutoff_is_et_midnight_not_utc():
    """The short paired-fit is sliced at ET midnight. A 23:00 ET tick on the eve
    of short_start (still we-late-band of the PRIOR ET date) must fall BELOW the
    cutoff and be excluded; the 00:00 ET tick of short_start is at the cutoff and
    kept. A UTC-midnight cut would keep the eve's 20:00-23:59 ET, folding the very
    weekend-late band under study into the short window's p90."""
    short_start = date(2026, 8, 8)  # a Saturday
    cutoff = et_midnight(short_start)
    eve_2300 = _tick(2026, 8, 7, 23)  # 23:00 ET the night before
    start_0000 = _tick(2026, 8, 8, 0)  # 00:00 ET of short_start
    assert eve_2300 < cutoff  # excluded from the short fit
    assert start_0000 == cutoff  # kept
    # A UTC-midnight cutoff would sit 4-5h earlier and wrongly admit eve_2300.
    utc_cut = int(datetime(2026, 8, 8, tzinfo=UTC).timestamp())
    assert eve_2300 >= utc_cut  # the bug: eve's late-night band leaks in


def test_late_night_filter_is_weekend_late_service_by_service_night():
    # 2026-08-15 Sat, 08-16 Sun, 08-17 Mon.
    assert is_weekend_late_tick(_tick(2026, 8, 15, 23))  # Sat 23 -> Sat night
    assert is_weekend_late_tick(_tick(2026, 8, 16, 1))  # Sun 01 -> Sat night
    assert is_weekend_late_tick(_tick(2026, 8, 16, 22))  # Sun 22 -> Sun night
    assert is_weekend_late_tick(_tick(2026, 8, 17, 1))  # Mon 01 -> Sun night, IN
    assert not is_weekend_late_tick(_tick(2026, 8, 15, 1))  # Sat 01 -> Fri night, OUT
    assert not is_weekend_late_tick(_tick(2026, 8, 15, 12))  # Sat midday
    assert not is_weekend_late_tick(_tick(2026, 8, 18, 1))  # Tue 01 -> Mon night


def test_night_labels_key_by_service_night_merges_across_midnight():
    """Sat 23:00 and Sun 01:00 are ONE service night (Saturday); a Sunday-morning
    acute alert taints Saturday's night, not a separate 'Sunday' one."""
    obs = [
        TickObservation("1", _tick(2026, 8, 15, 23), _obs()),  # Sat eve, clean
        TickObservation("1", _tick(2026, 8, 16, 1), _obs(has_delays=True)),  # Sun 01
    ]
    labels = night_labels(obs)
    assert labels[("1", date(2026, 8, 15))] == "disrupted"  # tainted by the 01:00
    assert ("1", date(2026, 8, 16)) not in labels  # no separate Sunday-am night


def test_night_labels_alert_free_needs_both_snapshot_and_alert_witness():
    """An alert-free service night is normal only when BOTH the trip-updates cron
    ran (snapshot_ticks) AND the separate alerts fetch was live that night
    (service night in alert_nights). Cron alone leaves an alerts-outage night
    falsely quiet, so it stays excluded."""
    sat_23 = _tick(2026, 8, 15, 23)
    svc = service_night_ticks({("2", sat_23): 20})
    sat = date(2026, 8, 15)
    # Both witnesses -> normal.
    both = night_labels(
        [], service_ticks=svc, snapshot_ticks={sat_23}, alert_nights={sat}
    )
    assert both[("2", sat)] == "normal"
    # Cron ran but NO alert archived system-wide that night -> alerts outage, excluded.
    outage = night_labels(
        [], service_ticks=svc, snapshot_ticks={sat_23}, alert_nights=set()
    )
    assert ("2", sat) not in outage
    # Alerts live but NO snapshot (cron gap) -> excluded.
    nogap = night_labels(
        [], service_ticks=svc, snapshot_ticks=set(), alert_nights={sat}
    )
    assert ("2", sat) not in nogap


def test_night_labels_own_acute_alert_beats_alert_free_augmentation():
    """A route with its own acute alert that night stays disrupted even when both
    witnesses are supplied — the augmentation only fills genuinely alert-free
    nights, never overrides a disruption."""
    sat_23 = _tick(2026, 8, 15, 23)
    obs = [TickObservation("2", sat_23, _obs(has_delays=True))]
    svc = service_night_ticks({("2", sat_23): 20})
    labels = night_labels(
        obs,
        service_ticks=svc,
        snapshot_ticks={sat_23},
        alert_nights={date(2026, 8, 15)},
    )
    assert labels[("2", date(2026, 8, 15))] == "disrupted"


def test_night_labels_routine_planned_advisory_stays_normal():
    """A routine "Planned -" advisory does NOT disqualify a night: planned notices
    blanket weekend nights whether or not service ran reduced, so gating on them
    would silence the full-service nights whose surplus is the false alarm. Only
    acute alerts (delays/suspension/unplanned service-change) mark a night."""
    obs = [TickObservation("1", _tick(2026, 8, 15, 22), _obs(has_planned=True))]
    assert night_labels(obs)[("1", date(2026, 8, 15))] == "normal"


def test_night_counts_only_weekend_late_night_cells():
    # Two distinct Saturdays at 23:00 ET (we23) plus a weekday tick that must not
    # count toward any weekend cell.
    series = {
        ("1", _tick(2026, 8, 15, 23)): 20,
        ("1", _tick(2026, 8, 22, 23)): 22,
        ("1", _tick(2026, 8, 17, 23)): 24,  # Monday -> wd23, excluded
    }
    counts = night_counts_by_cell(series)
    assert counts[("1", "we23")] == {date(2026, 8, 15), date(2026, 8, 22)}
    assert ("1", "wd23") not in counts


def _cell_over_nights(
    route: str, sbin_hour: int, saturdays: list[int]
) -> dict[tuple[str, int], int]:
    """assigned_n series for one weekend-late cell across the given August
    Saturdays, 12 near-constant 5-min ticks per night (one autocorrelated draw)."""
    series: dict[tuple[str, int], int] = {}
    for day in saturdays:
        base = _tick(2026, 8, day, sbin_hour)
        for i in range(12):
            series[(route, base + i * 300)] = 20
    return series


def test_partition_splits_cells_at_the_night_gate():
    thin = _cell_over_nights("J", 23, [1, 8])  # 2 nights
    thick = _cell_over_nights("1", 23, [1, 8, 15, 22, 29, 2, 9, 16])  # 8 nights
    series = {**thin, **thick}
    quant = compute_service_quantiles(series, bin_fn=schedule_bin)
    part = partition_by_gate(series, quant, min_nights=8)
    assert ("1", "we23") in part.night_pass
    assert ("J", "we23") in part.tick_only
    assert part.night_pass.isdisjoint(part.tick_only)


def test_false_alarm_rate_scores_high_score_nights_above_a_thin_fit_p90():
    """The amplifier is out-of-sample: a thin fit p90 drawn from a few reduced
    nights is exceeded by later confirmed-normal high-mode nights far more than
    the ~0.10 a percentile flags of its own distribution. Scoring a night that
    fed its own p90 would read ~0.10 by construction — so the fit and score
    nights must be disjoint."""
    # Fit p90 from a low/bimodal sample (reduced trackwork nights dominate).
    quant = {("2", "we23"): ServiceQuantiles(p10=6.0, p90=16.0)}
    # Score two later, disjoint confirmed-normal nights at a normal-service level
    # of 24 — above the fit p90 of 16 on every tick.
    score: dict[tuple[str, int], int] = {}
    for day in (29, 30):  # a later weekend, not in the fit sample
        base = _tick(2026, 8, day, 23)
        for i in range(12):
            score[("2", base + i * 300)] = 24
    labels = {("2", _et_date_of(day)): "normal" for day in (29, 30)}
    rate = false_alarm_rate(score, quant, [("2", "we23")], labels, n_boot=200)
    assert rate.rate == 1.0  # every score tick clears the thin fit p90
    assert rate.n_units == 2  # two (route, night) clusters


def test_false_alarm_rate_clusters_hourly_cells_of_one_night_into_one_unit():
    """A route's adjacent weekend-late hours (we22, we23) on the SAME night share
    a service regime and must resample together — one (route, night) unit, not two
    per-hour units, or the CI would read them as independent draws."""
    night = _tick(2026, 8, 29, 23)  # Saturday
    score: dict[tuple[str, int], int] = {}
    for i in range(12):
        score[("2", night + i * 300)] = 24  # we23
        score[("2", night - 3600 + i * 300)] = 24  # we22, same night
    quant = {
        ("2", "we23"): ServiceQuantiles(p10=6.0, p90=16.0),
        ("2", "we22"): ServiceQuantiles(p10=6.0, p90=16.0),
    }
    labels = {("2", date(2026, 8, 29)): "normal"}
    rate = false_alarm_rate(
        score, quant, [("2", "we22"), ("2", "we23")], labels, n_boot=200
    )
    assert rate.n_units == 1  # one route-night, both hours aggregated
    assert rate.n_ticks == 24  # 12 we22 + 12 we23 ticks in the one cluster
    assert rate.rate == 1.0


def test_abstention_tradeoff_separates_spurious_from_real_silenced_flags():
    # Three nights on a silenced cell: two confirmed-normal high-mode nights that
    # flag (spurious), one disrupted low-mode night that flags (real signal lost).
    series: dict[tuple[str, int], int] = {}
    spec = ((15, 24, "normal"), (22, 24, "normal"), (29, 1, "disrupted"))
    for day, level, _label in spec:
        base = _tick(2026, 8, day, 23)
        for i in range(12):
            series[("J", base + i * 300)] = level

    quant = {("J", "we23"): ServiceQuantiles(p10=5.0, p90=10.0)}
    labels = {("J", _et_date_of(day)): label for day, _level, label in spec}
    trade = abstention_tradeoff(series, quant, [("J", "we23")], labels)
    assert trade.spurious_flags_silenced == 2  # normal nights above p90
    assert trade.real_flags_silenced == 1  # disrupted night below p10
    assert trade.normal_nights == 2
    assert trade.disrupted_nights == 1


def _et_date_of(day: int) -> date:
    return (
        datetime.fromtimestamp(_tick(2026, 8, day, 23), tz=UTC).astimezone(_ET).date()
    )
