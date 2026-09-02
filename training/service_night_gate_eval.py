"""Does the supply baseline's independent-night gate tighten false alarms on the
bimodal weekend late-night cells that motivated it?

Background. The published supply axis divides live `assigned_n` by a per-(route,
schedule_bin) median and reads it against the cell's own p10/p90. On a short
training window a weekend-hourly (`we<hour>`) cell clears the raw `min_samples`
tick floor on only ~2-4 independent nights, and late-night weekend service is
bimodal (reduced/trackwork nights vs normal nights): the median lands in the
empty gap between the modes, so a genuinely-normal weekend night reads 1.7-2.1x
its own baseline and clears p90 — a false surplus alarm. `compute_baseline` /
`compute_service_quantiles` now take a `min_nights` gate (the published axis
passes `SERVICE_MIN_NIGHTS`); a cell spanning too few DISTINCT ET dates abstains
rather than publish a between-modes median.

This tool measures whether that gate earns its abstention. It rebuilds the
baseline+quantiles over an archive window, partitions the weekend late-night
cells by whether they clear the night gate, and on alert-confirmed-normal nights
measures the above-p90 false-alarm rate per partition, resampling NIGHTS (never
the autocorrelated 5-min ticks — a whole weekend night is one draw) and
withholding the interval below two clusters. The gate's benefit is the spurious
false alarms it silences on confirmed-normal nights; its cost is the genuine
off-distribution nights it silences with them. Both are counted.

Confirmed-normal is night-witnessed ALERT-QUIET, not tick-certified normal: no
per-tick alert-fetch liveness record exists in the archive, so the witness
proves at least one successful alerts fetch per service night, and a mid-night
alerts outage could mislabel later quiet ticks. The paired short-vs-long
comparison scores identical ticks under identical labels in both arms, so any
label error is shared by construction (direction indeterminate without
per-tick liveness); absolute FA levels are alert-quiet rates, not
certified-normal rates.

Confirmed-normal is an alert-feed consistency reference, not an independent one:
a (route, night) is confirmed-normal when NO tick that night carried an ACUTE
alert — delays, suspension, or an unplanned service change. Routine "Planned -"
trackwork advisories are NOT excluded: they blanket nearly every weekend night
whether or not service actually ran reduced, so gating on them would silence the
full-service (high) nights whose surplus reading is the false alarm under test.
A genuinely reduced night reads LOW, not high, so it never fires the above-p90
surplus flag regardless. The reference shares the operator's real-service state
with the `assigned_n` dispatch count but is a separate MTA product, so this is a
bound on how often the supply flag contradicts the alert feed on shared-normal
service.

The night unit is the SERVICE night (ET hours 0-3 roll back to the prior date),
so a night does not split at midnight and the weekend-late set follows service,
not the calendar (Mon 00-03 is in Sunday's night; Sat 00-03 is Friday's, out).

An alert-FREE route-night (no observation at all) is called normal only when two
liveness witnesses both hold, because the alerts fetch is a separate request from
the trip-updates cron (worker index.ts) and the archive keeps NO per-tick alert-
fetch record (the trip-updates body's fresh_feeds is the vehicle feeds only):
  (1) a trip-updates snapshot existed over the night (cron ran), AND
  (2) some alert-version body was archived system-wide that service night (the
      alerts fetch was live) — NYC nights are never system-wide alert-silent
      given standing planned advisories, so a witnessed-cron night with no alert
      archived is a fetch outage and is EXCLUDED, not read as quiet-normal.
Snapshot presence is used rather than reconstructed alert observation ticks,
which a long-running alert's active_period extends across a collection gap.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from statistics import median
from typing import TYPE_CHECKING

from momentarily.hmm import schedule_bin
from training.eval_common import (
    NYC_TZ as _NYC_TZ,
)
from training.eval_common import (
    alert_night_witness,
    et_midnight,
    snapshot_tick_witness,
)
from training.eval_common import (
    service_night as _service_night,
)
from training.headway_eval import Rate, Unit, night_bootstrap
from training.load_r2 import (
    ServiceQuantiles,
    compute_baseline,
    compute_service_quantiles,
)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

    from training.load import TickObservation
    from training.r2_client import R2Config


# ET weekend hours treated as "late night" — the evening-through-overnight band
# where reduced trackwork service makes the per-cell distribution bimodal. `we23`
# (the journal's named case) and its neighbours, plus the after-midnight hours
# 0-3 that belong to the SAME service night (eval_common.service_night).
LATE_NIGHT_HOURS: frozenset[int] = frozenset({20, 21, 22, 23, 0, 1, 2, 3})


def is_weekend_late_tick(
    epoch_seconds: int, *, hours: frozenset[int] = LATE_NIGHT_HOURS
) -> bool:
    """True when a tick is weekend late-night SERVICE: its ET hour is in the
    late-night band AND its service night (0-3 rolled to the prior date) falls on
    a Saturday or Sunday. So Sun 00-03 (Saturday's night) and Mon 00-03 (Sunday's
    night) are in; Sat 00-03 (Friday's night) is out — membership follows the
    service night, not the raw calendar date schedule_bin encodes."""
    dt = datetime.fromtimestamp(epoch_seconds, tz=_NYC_TZ)
    if dt.hour not in hours:
        return False
    return _service_night(epoch_seconds).weekday() >= 5  # Sat=5, Sun=6


# The named bimodal targets from the motivating investigation.
NAMED_TARGET_ROUTES: tuple[str, ...] = ("1", "2", "J")

# Acute alert flags whose presence on any tick disqualifies a night from being
# called normal service. Routine "Planned -" advisories are deliberately NOT here
# (has_service_change already excludes planned changes via its 'Planned -' prefix
# guard): planned work blankets weekend nights and gating on it would drop the
# full-service nights whose surplus reading is the false alarm being measured.
_DISRUPTIVE = (
    "has_delays",
    "has_suspended_alert",
    "has_service_change",
)


def night_labels(
    obs: Iterable[TickObservation],
    *,
    service_ticks: Mapping[tuple[str, date], Sequence[int]] | None = None,
    snapshot_ticks: frozenset[int] | set[int] | None = None,
    alert_nights: frozenset[date] | set[date] | None = None,
) -> dict[tuple[str, date], str]:
    """(route, service night) -> "normal" | "disrupted" from the alert feed. A
    night with an observation is normal only if EVERY observed tick on that route
    that night was free of ACUTE alerts (delays/suspension/unplanned
    service-change); one such tick marks the whole night disrupted. Routine
    planned advisories do not disqualify a night. Nights are keyed by
    _service_night, so the after-midnight tail groups with its own evening.

    build_tick_observations only emits a row for a (route, tick) that some alert's
    informed_entity named, so an entirely ALERT-FREE route-night — the quietest,
    most-normal service — has no observation and would otherwise vanish. When
    `service_ticks`, `snapshot_ticks`, and `alert_nights` are given, such a night
    is labelled "normal" only when BOTH liveness witnesses hold:
      1. a real trip-updates snapshot existed over it (one of its ticks is in
         `snapshot_ticks`, the snapped observed_at of the fetched bodies — proof
         the cron ran), AND
      2. its service night is in `alert_nights` (some alert-version body was
         archived system-wide that night — proof the SEPARATE alerts fetch was
         live, not outaged).
    Both are required because the alerts fetch is a distinct request from the
    trip-updates cron (worker index.ts steps 2 vs later), so cron liveness alone
    would read an alerts-outage tick as falsely quiet. A night failing either
    witness is a possible fetch gap and is left absent (excluded), never
    fabricated normal — NYC nights are never system-wide alert-silent given the
    standing planned advisories, so a witnessed-cron span with no alert archived
    is an outage, not true quiet."""
    labels: dict[tuple[str, date], str] = {}
    for o in obs:
        key = (o.route_id, _service_night(o.tick))
        if labels.get(key) == "disrupted":
            continue
        disruptive = any(getattr(o.observation, f) for f in _DISRUPTIVE)
        labels[key] = "disrupted" if disruptive else labels.get(key, "normal")
    if (
        service_ticks is not None
        and snapshot_ticks is not None
        and alert_nights is not None
    ):
        for key, ticks in service_ticks.items():
            if key in labels:  # already labelled from this route's own alerts
                continue
            night = key[1]
            cron_live = any(t in snapshot_ticks for t in ticks)
            if cron_live and night in alert_nights:  # both fetches live -> quiet
                labels[key] = "normal"
    return labels


def service_night_ticks(
    series: Mapping[tuple[str, int], int],
) -> dict[tuple[str, date], list[int]]:
    """(route, service night) -> the service ticks on it — the coverage a
    night_labels alert-free-normal decision is witnessed against."""
    out: dict[tuple[str, date], list[int]] = {}
    for route, tick in series:
        out.setdefault((route, _service_night(tick)), []).append(tick)
    return out


def night_counts_by_cell(
    series: Mapping[tuple[str, int], int],
    *,
    hours: frozenset[int] = LATE_NIGHT_HOURS,
) -> dict[tuple[str, str], set[date]]:
    """Per weekend-late (route, schedule_bin) cell, the set of distinct SERVICE
    nights it ran on — the count the independent-night gate is applied to. Only
    weekend-late-SERVICE ticks count (is_weekend_late_tick: hour in band and the
    service night is Sat/Sun), so Mon 00-03 (a wd0x cell on Sunday's night) is in
    and Sat 00-03 (Friday's night) is out."""
    out: dict[tuple[str, str], set[date]] = {}
    for route, tick in series:
        if not is_weekend_late_tick(tick, hours=hours):
            continue
        out.setdefault((route, schedule_bin(tick)), set()).add(_service_night(tick))
    return out


@dataclass(frozen=True)
class Partition:
    """Weekend-late-night cells split by the night gate. `night_pass` clears
    `min_nights` distinct nights (survives the gate); `tick_only` clears the raw
    tick floor but not the night gate (the gate silences it)."""

    night_pass: frozenset[tuple[str, str]]
    tick_only: frozenset[tuple[str, str]]


def partition_by_gate(
    series: Mapping[tuple[str, int], int],
    quantiles: Mapping[tuple[str, str], ServiceQuantiles],
    *,
    min_nights: int,
    hours: frozenset[int] = LATE_NIGHT_HOURS,
) -> Partition:
    """Partition the weekend-late cells that cleared the tick floor (are present
    in `quantiles`, built with min_nights=1) by whether they also clear
    `min_nights` distinct SERVICE nights. A cell is weekend-late iff it ran on any
    weekend-late-service tick (present in night_counts_by_cell) — tick-derived,
    since a schedule_bin string alone no longer decides membership (we0x mixes
    Fri- and Sat-night ticks, wd0x mixes weekday and Sunday-night ticks)."""
    counts = night_counts_by_cell(series, hours=hours)
    late_published = {cell for cell in quantiles if cell in counts}
    night_pass = frozenset(
        c for c in late_published if len(counts.get(c, set())) >= min_nights
    )
    tick_only = frozenset(late_published - night_pass)
    return Partition(night_pass=night_pass, tick_only=tick_only)


def _night_series(
    series: Mapping[tuple[str, int], int],
    cells: Iterable[tuple[str, str]],
    *,
    hours: frozenset[int] = LATE_NIGHT_HOURS,
) -> dict[tuple[str, str], dict[date, list[int]]]:
    """Regroup a subset of cells' ticks into per-cell, per-SERVICE-night assigned_n
    lists — the shape the night-clustered bootstrap and the per-night confusion
    read. Only weekend-late-service ticks are kept, keyed by _service_night."""
    wanted = set(cells)
    out: dict[tuple[str, str], dict[date, list[int]]] = {}
    for (route, tick), assigned in series.items():
        cell = (route, schedule_bin(tick))
        if cell not in wanted or not is_weekend_late_tick(tick, hours=hours):
            continue
        out.setdefault(cell, {}).setdefault(_service_night(tick), []).append(assigned)
    return out


def false_alarm_rate(
    series: Mapping[tuple[str, int], int],
    quantiles: Mapping[tuple[str, str], ServiceQuantiles],
    cells: Iterable[tuple[str, str]],
    labels: Mapping[tuple[str, date], str],
    *,
    hours: frozenset[int] = LATE_NIGHT_HOURS,
    n_boot: int = 2000,
    seed: int = 0,
) -> Rate:
    """Above-p90 false-alarm rate on confirmed-normal weekend-late-night ticks for
    a set of cells, night-clustered. A tick fires when its assigned_n exceeds its
    own cell's p90; the resampling unit is the (route, NIGHT), so ALL of a
    route-night's weekend-late hourly cells (we20..we03, each with its own p90)
    aggregate into ONE draw — adjacent hours of one night share a service regime
    and are not independent, mirroring headway_eval's night unit."""
    grouped = _night_series(series, cells, hours=hours)
    per_route_night: dict[tuple[str, date], list[int]] = {}
    for (route, sbin), by_night in grouped.items():
        q = quantiles.get((route, sbin))
        if q is None:
            continue
        for night, vals in by_night.items():
            if labels.get((route, night)) != "normal":
                continue
            acc = per_route_night.setdefault((route, night), [0, 0])
            acc[0] += sum(1 for v in vals if v > q.p90)
            acc[1] += len(vals)
    units = [Unit(alarmed=a, total=t) for a, t in per_route_night.values()]
    return night_bootstrap(units, n_boot=n_boot, seed=seed)


@dataclass(frozen=True)
class AbstentionTradeoff:
    """Per-night confusion for the cells the gate silences (`tick_only`). A night
    "flags" when its median assigned_n crosses the cell's p90 or falls under p10.
    Split by the alert label: a flagged normal night is a spurious alarm the gate
    correctly removes; a flagged disrupted night is a genuine off-distribution
    reading the gate silences with it."""

    spurious_flags_silenced: int  # confirmed-normal nights that would have flagged
    real_flags_silenced: int  # disrupted nights that would have flagged
    normal_nights: int
    disrupted_nights: int
    n_cells: int


def abstention_tradeoff(
    series: Mapping[tuple[str, int], int],
    quantiles: Mapping[tuple[str, str], ServiceQuantiles],
    cells: Iterable[tuple[str, str]],
    labels: Mapping[tuple[str, date], str],
    *,
    hours: frozenset[int] = LATE_NIGHT_HOURS,
) -> AbstentionTradeoff:
    """Count the nights the gate silences on `cells`, split into spurious (a
    confirmed-normal night that would have flagged — correctly removed) and real
    (a disrupted night that would have flagged — a genuine signal lost)."""
    grouped = _night_series(series, cells, hours=hours)
    spurious = real = normal_n = disrupted_n = 0
    for (route, sbin), by_night in grouped.items():
        q = quantiles.get((route, sbin))
        if q is None:
            continue
        for night, vals in by_night.items():
            label = labels.get((route, night))
            if label is None:
                continue
            m = median(vals)
            flagged = m > q.p90 or m < q.p10
            if label == "normal":
                normal_n += 1
                spurious += flagged
            else:
                disrupted_n += 1
                real += flagged
    return AbstentionTradeoff(
        spurious_flags_silenced=spurious,
        real_flags_silenced=real,
        normal_nights=normal_n,
        disrupted_nights=disrupted_n,
        n_cells=len(grouped),
    )


def named_cell_detail(
    fit_series: Mapping[tuple[str, int], int],
    score_series: Mapping[tuple[str, int], int],
    quantiles: Mapping[tuple[str, str], ServiceQuantiles],
    baseline: Mapping[tuple[str, str], float],
    labels: Mapping[tuple[str, date], str],
    *,
    routes: Sequence[str] = NAMED_TARGET_ROUTES,
    hours: frozenset[int] = LATE_NIGHT_HOURS,
) -> list[dict[str, object]]:
    """Per-cell detail for the named bimodal targets: the fit-window median,
    p10/p90 and distinct-night count that the gate acts on, the fit per-night
    medians (so a bimodal spread is visible), and — measured on HELD-OUT score
    nights — the median confirmed-normal service ratio (a score normal-night
    median over the fit cell median) and how many of those nights cleared p90.
    The out-of-sample ratio is the 1.7-2.1x the gate is meant to stop publishing;
    an in-sample ratio would be ~1.0 by construction."""
    cells = [c for c in quantiles if c[0] in set(routes)]
    fit_grouped = _night_series(fit_series, cells, hours=hours)
    score_grouped = _night_series(score_series, cells, hours=hours)
    out: list[dict[str, object]] = []
    for (route, sbin), by_night in sorted(fit_grouped.items()):
        q = quantiles[(route, sbin)]
        cell_med = baseline.get((route, sbin))
        per_night = {n.isoformat(): median(v) for n, v in sorted(by_night.items())}
        score_nights = score_grouped.get((route, sbin), {})
        normal_ratios = [
            median(v) / cell_med
            for n, v in score_nights.items()
            if cell_med and labels.get((route, n)) == "normal"
        ]
        normal_above_p90 = sum(
            1
            for n, v in score_nights.items()
            if labels.get((route, n)) == "normal" and median(v) > q.p90
        )
        out.append(
            {
                "route": route,
                "bin": sbin,
                "fit_n_nights": len(by_night),
                "fit_cell_median": cell_med,
                "fit_p10": q.p10,
                "fit_p90": q.p90,
                "fit_per_night_median": per_night,
                "score_median_confirmed_normal_ratio": round(median(normal_ratios), 3)
                if normal_ratios
                else None,
                "score_normal_nights": len(normal_ratios),
                "score_normal_nights_above_p90": normal_above_p90,
            }
        )
    return out


# --- end-to-end run over the R2 archive ---


def _rate_json(r: Rate) -> dict[str, object]:
    return {
        "rate": None if r.rate is None else round(r.rate, 4),
        "lo": None if r.lo is None else round(r.lo, 4),
        "hi": None if r.hi is None else round(r.hi, 4),
        "n_nights": r.n_units,
        "n_ticks": r.n_ticks,
        "n_alarmed": r.n_alarmed,
    }


def _service_series_over(
    cfg: R2Config, client: S3Client, start: date, end: date
) -> tuple[dict[tuple[str, int], int], set[int]]:
    """Returns the (route, tick) -> assigned_n series AND the coverage-witness
    tick set: the snapped observed_at of every fetched trip-updates snapshot. The
    collector writes one snapshot per cron tick regardless of alert quietness, so
    this set is the ground truth for "a snapshot existed here" — unlike alert
    observation ticks, which are reconstructed from active_periods and can span a
    collection gap."""
    from training.load_r2 import build_service_series, fetch_trip_update_metrics

    print(f"fetching trip_updates {start}..{end}", file=sys.stderr)
    bodies = fetch_trip_update_metrics(
        cfg, start_date=start, end_date=end, client=client
    )
    return build_service_series(bodies), snapshot_tick_witness(bodies)


def _alerts_over(
    cfg: R2Config, client: S3Client, start: date, end: date
) -> tuple[list[TickObservation], set[date]]:
    """Returns the per-tick alert observations AND the set of service-nights with
    any alert-version archived system-wide — the alerts-fetch liveness witness.
    The alerts fetch is a separate request from the trip-updates cron, so a night
    whose cron ran but archived NO alert version anywhere is an alerts outage
    reading falsely quiet, not genuine silence."""
    from training.load_r2 import build_tick_observations, fetch_alert_versions

    print(f"fetching alert_versions {start}..{end}", file=sys.stderr)
    bodies = fetch_alert_versions(cfg, start_date=start, end_date=end, client=client)
    return build_tick_observations(bodies), alert_night_witness(bodies)


def run_split(
    cfg: R2Config,
    client: S3Client,
    *,
    fit_start: date,
    fit_end: date,
    score_start: date,
    score_end: date,
    min_nights: int,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    """Fit the baseline+quantiles on the fit window, score confirmed-normal
    above-p90 false alarms on the DISJOINT score window. Out-of-sample by
    construction: the amplifier only bites nights the baseline did not see."""
    fit_series, _ = _service_series_over(cfg, client, fit_start, fit_end)
    score_series, score_snapshots = _service_series_over(
        cfg, client, score_start, score_end
    )
    obs, alert_nights = _alerts_over(cfg, client, score_start, score_end)
    labels = night_labels(
        obs,
        service_ticks=service_night_ticks(score_series),
        snapshot_ticks=score_snapshots,
        alert_nights=alert_nights,
    )

    # Tick-floor baseline (min_nights=1) publishes every fit cell, so the
    # tick_only partition still has a p90 to score against; the gate only decides
    # which partition a cell lands in.
    baseline = compute_baseline(fit_series, bin_fn=schedule_bin, min_nights=1)
    quantiles = compute_service_quantiles(fit_series, bin_fn=schedule_bin, min_nights=1)
    part = partition_by_gate(fit_series, quantiles, min_nights=min_nights)

    fa_pass = false_alarm_rate(
        score_series, quantiles, part.night_pass, labels, n_boot=n_boot, seed=seed
    )
    fa_tick = false_alarm_rate(
        score_series, quantiles, part.tick_only, labels, n_boot=n_boot, seed=seed
    )
    trade = abstention_tradeoff(score_series, quantiles, part.tick_only, labels)

    n_normal = sum(1 for v in labels.values() if v == "normal")
    return {
        "fit_window": {"start": fit_start.isoformat(), "end": fit_end.isoformat()},
        "score_window": {
            "start": score_start.isoformat(),
            "end": score_end.isoformat(),
        },
        "min_nights": min_nights,
        "score_route_nights_confirmed_normal": n_normal,
        "score_route_nights_total": len(labels),
        "partition": {
            "night_pass_cells": len(part.night_pass),
            "tick_only_cells_gate_silences": len(part.tick_only),
        },
        "score_false_alarm_above_p90": {
            "night_pass": _rate_json(fa_pass),
            "tick_only_gate_silences": _rate_json(fa_tick),
        },
        "abstention_tradeoff_on_silenced_cells": {
            "spurious_flags_silenced": trade.spurious_flags_silenced,
            "real_flags_silenced": trade.real_flags_silenced,
            "confirmed_normal_nights": trade.normal_nights,
            "disrupted_nights": trade.disrupted_nights,
            "n_silenced_cells": trade.n_cells,
        },
        "named_targets": named_cell_detail(
            fit_series, score_series, quantiles, baseline, labels
        ),
    }


def run_paired(
    cfg: R2Config,
    client: S3Client,
    *,
    fit_end: date,
    fit_days_long: int,
    fit_days_short: int,
    score_start: date,
    score_end: date,
    min_nights: int,
    n_boot: int,
    seed: int,
) -> dict[str, object]:
    """The exact gate counterfactual: on the SAME weekend-late cells and the SAME
    out-of-sample confirmed-normal score nights, the above-p90 false-alarm rate
    when their p90 came from a LONG fit (the window the night gate forces) versus
    a SHORT fit (what a tick-gate would have shipped on a thin window). The cell
    set is the cells the night gate publishes on the long window, so nothing but
    the fit-window breadth differs — the only lever the gate pulls.

    Also reports how many of those cells the night gate would have SILENCED on the
    short window (had too few distinct nights there), i.e. the abstention the gate
    buys instead of shipping the thin p90."""
    long_start = fit_end - timedelta(days=fit_days_long - 1)
    short_start = fit_end - timedelta(days=fit_days_short - 1)
    long_series, _ = _service_series_over(cfg, client, long_start, fit_end)
    short_cut = et_midnight(short_start)
    short_series = {k: v for k, v in long_series.items() if k[1] >= short_cut}
    score_series, score_snapshots = _service_series_over(
        cfg, client, score_start, score_end
    )
    obs, alert_nights = _alerts_over(cfg, client, score_start, score_end)
    labels = night_labels(
        obs,
        service_ticks=service_night_ticks(score_series),
        snapshot_ticks=score_snapshots,
        alert_nights=alert_nights,
    )

    q_long = compute_service_quantiles(long_series, bin_fn=schedule_bin, min_nights=1)
    q_short = compute_service_quantiles(short_series, bin_fn=schedule_bin, min_nights=1)
    part_long = partition_by_gate(long_series, q_long, min_nights=min_nights)
    # The cells the gate publishes on the adequate window — score both fits here.
    shared = frozenset(part_long.night_pass) & frozenset(q_short)
    short_counts = night_counts_by_cell(short_series)
    silenced_on_short = sum(
        1 for c in shared if len(short_counts.get(c, set())) < min_nights
    )

    fa_long = false_alarm_rate(
        score_series, q_long, shared, labels, n_boot=n_boot, seed=seed
    )
    fa_short = false_alarm_rate(
        score_series, q_short, shared, labels, n_boot=n_boot, seed=seed
    )
    return {
        "fit_end": fit_end.isoformat(),
        "score_window": {
            "start": score_start.isoformat(),
            "end": score_end.isoformat(),
        },
        "fit_days": {"long": fit_days_long, "short": fit_days_short},
        "min_nights": min_nights,
        "shared_cells_scored": len(shared),
        "shared_cells_gate_would_silence_on_short_fit": silenced_on_short,
        "score_false_alarm_above_p90_same_cells": {
            "long_fit_gate_satisfied": _rate_json(fa_long),
            "short_fit_tick_gate_would_ship": _rate_json(fa_short),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure whether the supply baseline's independent-night gate "
        "lowers confirmed-normal false alarms on the bimodal weekend late-night "
        "cells, and at what abstention cost — fit on a trailing window, scored "
        "out-of-sample and night-bootstrapped."
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help="run the same-cell long-vs-short fit counterfactual",
    )
    parser.add_argument(
        "--fit-days-short", type=int, default=13, help="short fit for --paired"
    )
    parser.add_argument(
        "--score-end", type=date.fromisoformat, default=None, help="default: yesterday"
    )
    parser.add_argument(
        "--score-days", type=int, default=21, help="score window length"
    )
    parser.add_argument(
        "--fit-days", type=int, default=35, help="leading fit window before the score"
    )
    parser.add_argument("--min-nights", type=int, default=8, help="the gate under test")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from training.r2_client import load_config, make_client

    cfg = load_config()
    client = make_client(cfg)
    score_end = args.score_end or (datetime.now(UTC).date() - timedelta(days=1))
    score_start = score_end - timedelta(days=args.score_days - 1)
    fit_end = score_start - timedelta(days=1)
    if args.paired:
        report = run_paired(
            cfg,
            client,
            fit_end=fit_end,
            fit_days_long=args.fit_days,
            fit_days_short=args.fit_days_short,
            score_start=score_start,
            score_end=score_end,
            min_nights=args.min_nights,
            n_boot=args.bootstrap,
            seed=args.seed,
        )
    else:
        fit_start = fit_end - timedelta(days=args.fit_days - 1)
        report = run_split(
            cfg,
            client,
            fit_start=fit_start,
            fit_end=fit_end,
            score_start=score_start,
            score_end=score_end,
            min_nights=args.min_nights,
            n_boot=args.bootstrap,
            seed=args.seed,
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
