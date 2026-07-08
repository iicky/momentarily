"""HMM shadow-log validation review: calibration, regime confusion, changepoint alignment.

Joins v1/predictions + v1/regime_transitions with MTA-derived ground truth from
the alerts archive, then emits reliability diagrams, a regime confusion matrix,
a changepoint alignment histogram, and a per-route recovery summary. All
artifacts plus summary.json land in docs/review/<date>-shadow-hmm/ so the
go/no-go memo can reference committed images.

Run with:
    PYTHONPATH=. murk exec -- .venv/bin/python -m training.review [--days N]
"""

# matplotlib lacks complete pyright stubs — silence the partially-unknown noise
# in this file only. Real type errors still surface.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from momentarily.hmm import Observation
from momentarily.mapping import (
    CANONICAL_SEVERITY_FLOOR,
    TRUTH_VERSION,
    category_for_label,
    coarse_status,
    severity_tier,
)
from training.episodes import (
    disruptive_types_by_key,
    episodes_summary,
    extract_episodes,
)
from training.eval import (
    PARAMS_KEY,
    TICK_SECONDS,
    PredictionRecord,
    TransitionRecord,
    build_eval,
    independent_recovery_metrics,
    load_predictions,
    load_transitions,
    recovery_as_dict,
)
from training.load import TickObservation
from training.load_r2 import (
    Disruption,
    build_movement_truth,
    build_tick_observations,
    fetch_alert_versions,
    fetch_vehicle_metrics,
)
from training.r2_client import load_config, make_client
from training.scorecard import dwell_lookup_from_params, episode_scorecard

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

HMM_STATES = ("normal", "disrupted", "suspended")
MTA_STATES = ("normal", "disrupted", "suspended")
CHANGEPOINT_WINDOW_MIN = 30


def derive_mta_state(obs: Observation) -> str:
    """MTA-derived ground-truth state for a tick — mirrors the HMM training
    flags (planned-work explicitly excluded upstream in build_tick_observations)."""
    if obs.has_suspended_alert:
        return "suspended"
    if obs.has_delays or obs.has_service_change:
        return "disrupted"
    return "normal"


def derive_graded_mta_state(alert_types: tuple[str, ...], *, floor: int) -> str:
    """Severity-graded ground-truth state from a route-tick's active alert_types.

    suspended if any suspension alert (tier 3); disrupted if any alert reaches
    `floor`; otherwise normal — so sub-floor alerts (minor delays, routine
    reroutes) read normal and the HMM filtering them is scored as correct, not a
    miss. floor=1 reproduces the breadth-dominated truth; floor=2 is severe-only.
    """
    tiers = [severity_tier(at) for at in alert_types]
    if any(t == 3 for t in tiers):
        return "suspended"
    if any(t >= floor for t in tiers):
        return "disrupted"
    return "normal"


def snap_tick(ts: int) -> int:
    return ((ts + TICK_SECONDS // 2) // TICK_SECONDS) * TICK_SECONDS


def load_truth_observations(
    client: S3Client,
    bucket: str,
    start_date: Any,
    end_date: Any,
) -> list[TickObservation]:
    """Per-(route, tick) observations from the alerts archive, with the active
    alert_types retained for severity grading. Fetched once; both the broad and
    graded truths derive from it."""
    del bucket  # client carries its own configured bucket via load_config
    bodies = fetch_alert_versions(
        start_date=start_date, end_date=end_date, client=client
    )
    return build_tick_observations(bodies)


def mta_truth(
    obs_list: list[TickObservation], *, severity_floor: int = CANONICAL_SEVERITY_FLOOR
) -> dict[tuple[str, int], str]:
    """(route, tick) → MTA-derived state. Defaults to the canonical severe-only
    truth (severity_floor >= 2: only Severe Delays / suspension count as
    disrupted). severity_floor <= 1 is the legacy breadth truth (any
    delays/service-change = disrupted), kept as a sensitivity. Ticks not in the
    dict had no active alerts → 'normal'."""
    if severity_floor <= 1:
        return {(o.route_id, o.tick): derive_mta_state(o.observation) for o in obs_list}
    return {
        (o.route_id, o.tick): derive_graded_mta_state(
            o.disruptive_types, floor=severity_floor
        )
        for o in obs_list
    }


def build_mta_truth(
    client: S3Client,
    bucket: str,
    start_date: Any,
    end_date: Any,
    *,
    severity_floor: int = CANONICAL_SEVERITY_FLOOR,
) -> dict[tuple[str, int], str]:
    """Convenience: fetch + derive the truth in one call."""
    obs_list = load_truth_observations(client, bucket, start_date, end_date)
    return mta_truth(obs_list, severity_floor=severity_floor)


# Alert categories that put the HMM in a disrupted condition — the others
# (planned work, information, no active alert) never do, so a recovery
# prediction is never graded against them. Clearance of one of these is the
# feed-side recovery event.
_DISRUPTIVE_CATEGORIES = frozenset(
    {"delays", "service_change", "service_suspension", "slow_speeds"}
)


def is_disruptive(alert_type: str | None) -> bool:
    if alert_type is None:
        return False
    return category_for_label(coarse_status(alert_type)) in _DISRUPTIVE_CATEGORIES


def clearance_disruptions(
    predictions: list[PredictionRecord], *, debounce: int = 2
) -> list[Disruption]:
    """Independent disruption intervals from alert-feed clearance: a route is
    disrupted while its primary_alert_type maps to a disruptive category and
    recovered when that clears. The signal is the raw feed (primary_alert_type in
    v1/predictions), upstream of and independent of the HMM's own argmax — a
    lighter recovery truth than the trip-updates service level (momentarily-xum).

    This validates against "when the alert cleared in the feed," NOT true service
    recovery (trains actually moving) — label it a feed-clearance proxy. Same
    debounce/hysteresis shape as derive_actual_recovery; disruptions still open at
    the window end are censored (dropped). See momentarily-up0.
    """
    by_route: dict[str, dict[int, bool]] = {}
    for p in predictions:
        by_route.setdefault(p.route, {})[snap_tick(p.ts)] = is_disruptive(
            p.primary_alert_type
        )

    out: list[Disruption] = []
    for route, ticks in by_route.items():
        in_disruption = False
        start: int | None = None
        cand_start: int | None = None
        cand_recover: int | None = None
        on_run = 0
        off_run = 0
        for tick in sorted(ticks):
            present = ticks[tick]
            if not in_disruption:
                if present:
                    if on_run == 0:
                        cand_start = tick
                    on_run += 1
                    if on_run >= debounce:
                        in_disruption = True
                        start = cand_start
                        off_run = 0
                else:
                    on_run = 0
            elif not present:
                if off_run == 0:
                    cand_recover = tick
                off_run += 1
                if off_run >= debounce and start is not None:
                    out.append(Disruption(route, start, cand_recover or tick))
                    in_disruption = False
                    start = None
                    on_run = 0
            else:
                off_run = 0
    return out


def confusion(
    preds: list[PredictionRecord],
    truth: dict[tuple[str, int], str],
) -> dict[str, dict[str, int]]:
    """Confusion matrix: HMM `condition` (rows) vs MTA-derived state (cols)."""
    matrix: dict[str, dict[str, int]] = {
        h: dict.fromkeys(MTA_STATES, 0) for h in HMM_STATES
    }
    for p in preds:
        mta = truth.get((p.route, snap_tick(p.ts)), "normal")
        if p.condition not in matrix:
            continue
        matrix[p.condition][mta] += 1
    return matrix


def changepoint_alignment(
    transitions: list[TransitionRecord],
    truth: dict[tuple[str, int], str],
    *,
    window_start: int,
    window_end: int,
    window_minutes: int = CHANGEPOINT_WINDOW_MIN,
) -> list[float | None]:
    """For each HMM regime transition, return signed minutes to the nearest
    MTA-state change for that route. None when no MTA change is within ±window.

    The truth dict only contains (route, tick) pairs where alerts were active —
    a route with no alerts has no entry. Walking only present ticks therefore
    never sees the change *back* to normal when alerts clear, making every
    recovery changepoint invisible (the 2026-06-09 review matched 25/1401
    transitions largely because of this). Walk the full tick grid instead,
    treating absent ticks as 'normal'. See momentarily-vk0.2.
    """
    routes = {route for route, _tick in truth}
    first_tick = snap_tick(window_start)
    last_tick = snap_tick(window_end)

    mta_change_ticks: dict[str, list[int]] = {}
    for route in routes:
        change_ticks: list[int] = []
        prev = "normal"
        tick = first_tick
        while tick <= last_tick:
            state = truth.get((route, tick), "normal")
            if state != prev:
                change_ticks.append(tick)
                prev = state
            tick += TICK_SECONDS
        mta_change_ticks[route] = change_ticks

    window_sec = window_minutes * 60
    deltas: list[float | None] = []
    for t in transitions:
        ticks = mta_change_ticks.get(t.route, [])
        if not ticks:
            deltas.append(None)
            continue
        nearest = min(ticks, key=lambda x: abs(x - t.exited_at))
        gap = nearest - t.exited_at
        deltas.append(gap / 60.0 if abs(gap) <= window_sec else None)
    return deltas


# === Plotting ===


def plot_reliability(eval_doc: dict[str, Any], out: Path) -> None:
    horizons = eval_doc["calibration"]
    fig, axes = plt.subplots(
        1, len(horizons), figsize=(4 * len(horizons), 4), sharey=True
    )
    for ax, cal in zip(axes, horizons, strict=True):
        bins = cal["bins"]
        xs = [b["mean_pred"] for b in bins if b["mean_pred"] is not None]
        ys = [b["mean_outcome"] for b in bins if b["mean_pred"] is not None]
        ns = [b["n"] for b in bins if b["mean_pred"] is not None]
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        if xs:
            ax.scatter(
                xs,
                ys,
                s=[max(n / 50.0, 8.0) for n in ns],
                color="steelblue",
                alpha=0.85,
            )
            for x, y, n in zip(xs, ys, ns, strict=True):
                ax.annotate(f"n={n}", (x, y), fontsize=7, ha="left", va="bottom")
        brier = cal["brier"]
        if brier is None:
            ax.set_title(f"horizon = {cal['horizon_min']} min (n=0)")
        else:
            bss_p = cal.get("bss_persistence")
            bss_c = cal.get("bss_climatology")
            skill = (
                f"\nBSS vs persist={bss_p:.2f}, vs climo={bss_c:.2f}"
                if bss_p is not None and bss_c is not None
                else ""
            )
            ax.set_title(
                f"horizon = {cal['horizon_min']} min\n"
                f"n={cal['n']}, Brier={brier:.3f}{skill}"
            )
        ax.set_xlabel("predicted P(normal)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("observed P(normal)")
    fig.suptitle("Reliability — p_normal_in_<horizon> vs realized condition")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_recovery_by_route(eval_doc: dict[str, Any], out: Path) -> None:
    by_route = eval_doc["recovery"]["by_route"]
    items = [(r, s) for r, s in by_route.items() if s["mae_min"] is not None]
    items.sort(key=lambda kv: -kv[1]["mae_min"])
    routes = [r for r, _ in items]
    maes = [s["mae_min"] for _, s in items]
    iqrs = [s["iqr_coverage"] for _, s in items]
    ns = [s["n"] for _, s in items]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(6, len(routes) * 0.35), 6))
    ax1.bar(routes, maes, color="firebrick")
    ax1.set_ylabel("MAE (minutes)")
    ax1.set_title("Recovery MAE per route (lower = better)")
    ax1.grid(True, axis="y", alpha=0.3)
    ax2.bar(routes, iqrs, color="steelblue")
    ax2.set_ylabel("IQR coverage")
    ax2.set_title("Recovery IQR coverage per route (target ≈ 0.5)")
    ax2.axhline(0.5, color="gray", ls="--", lw=1)
    ax2.set_ylim(0, 1)
    ax2.grid(True, axis="y", alpha=0.3)
    for ax in (ax1, ax2):
        for i, n in enumerate(ns):
            ax.annotate(
                f"n={n}",
                (i, 0),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=6,
            )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_confusion(
    conf: dict[str, dict[str, int]],
    out: Path,
    *,
    truth_label: str = "MTA-derived state (from alerts)",
    title: str = "Regime confusion: HMM vs MTA alerts (row-normalized)",
) -> None:
    mat = np.array(
        [[conf[h][m] for m in MTA_STATES] for h in HMM_STATES],
        dtype=float,
    )
    row_sums = mat.sum(axis=1, keepdims=True)
    pct = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums > 0)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(pct, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i in range(len(HMM_STATES)):
        for j in range(len(MTA_STATES)):
            ax.text(
                j,
                i,
                f"{int(mat[i, j])}\n({pct[i, j]:.0%})",
                ha="center",
                va="center",
                color="white" if pct[i, j] > 0.5 else "black",
                fontsize=9,
            )
    ax.set_xticks(range(len(MTA_STATES)))
    ax.set_xticklabels(MTA_STATES)
    ax.set_yticks(range(len(HMM_STATES)))
    ax.set_yticklabels(HMM_STATES)
    ax.set_xlabel(truth_label)
    ax.set_ylabel("HMM-published condition")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="row fraction")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_changepoint_alignment(deltas: list[float | None], out: Path) -> None:
    matched = [d for d in deltas if d is not None]
    unmatched = sum(1 for d in deltas if d is None)
    fig, ax = plt.subplots(figsize=(6, 4))
    if matched:
        ax.hist(matched, bins=21, color="steelblue", edgecolor="black", alpha=0.85)
        ax.axvline(0, color="firebrick", ls="--", lw=1)
        ax.set_xlabel("(HMM transition) - (nearest MTA changepoint) minutes")
        ax.set_ylabel("count")
        ax.grid(True, axis="y", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "no matched changepoints", ha="center", va="center")
        ax.set_xticks([])
        ax.set_yticks([])
    total = len(deltas)
    ax.set_title(
        f"Changepoint alignment ({len(matched)}/{total} matched within "
        f"±{CHANGEPOINT_WINDOW_MIN} min, {unmatched} unmatched)"
    )
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HMM shadow-log validation review")
    parser.add_argument("--days", type=int, default=5, help="window length in days")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output dir (default docs/review/<today>-shadow-hmm/)",
    )
    parser.add_argument(
        "--severity-floor",
        type=int,
        default=CANONICAL_SEVERITY_FLOOR,
        help="alert-severity tier for the canonical truth (2=severe-only)",
    )
    args = parser.parse_args(argv)

    today = datetime.now(UTC).date()
    start_date = today - timedelta(days=args.days - 1)
    out_dir = args.out or Path("docs/review") / f"{today.isoformat()}-shadow-hmm"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config()
    client = make_client(cfg)

    print(f"loading predictions/transitions {start_date}..{today}")
    preds = load_predictions(client, cfg.bucket, start_date, today)
    trans = load_transitions(client, cfg.bucket, start_date, today)
    print(f"  {len(preds)} predictions, {len(trans)} transitions")
    print("loading alerts archive for MTA-state truth")
    truth_obs = load_truth_observations(client, cfg.bucket, start_date, today)
    truth = mta_truth(truth_obs, severity_floor=args.severity_floor)
    truth_breadth = mta_truth(truth_obs, severity_floor=1)
    reclassified = sum(1 for k, v in truth_breadth.items() if truth.get(k) != v)
    print(f"  {len(truth)} (route, tick) ground-truth states")
    print(
        f"  canonical truth (severity_floor={args.severity_floor}): "
        f"{reclassified} ticks read normal vs the breadth truth"
    )
    print("loading vehicle-movement archive for independent current-state truth")
    movement_truth = build_movement_truth(
        fetch_vehicle_metrics(cfg, start_date=start_date, end_date=today, client=client)
    )
    print(f"  {len(movement_truth)} (route, tick) movement-derived states")

    window_end = int(datetime.now(UTC).timestamp())
    window_start = int(
        datetime(
            start_date.year, start_date.month, start_date.day, tzinfo=UTC
        ).timestamp()
    )
    eval_doc = build_eval(
        preds, trans, window_start=window_start, window_end=window_end
    )
    # Canonical: HMM condition vs severe-only MTA truth. Sub-floor alerts (minor
    # delays, routine reroutes) and planned work read normal, so predicted-normal
    # cells are judged on whether real disruption was missed, not alert breadth.
    conf = confusion(preds, truth)
    # Legacy breadth truth (any alert = disrupted), kept as a labeled sensitivity.
    conf_breadth = confusion(preds, truth_breadth)
    # Same HMM condition, scored against where trains physically are instead of
    # the alert feed — an independent-in-derivation cross-check. Empty until the
    # vehicle archive has accumulated ticks (Worker side ships first).
    conf_movement = confusion(preds, movement_truth)
    deltas = changepoint_alignment(
        trans, truth, window_start=window_start, window_end=window_end
    )
    clearance = clearance_disruptions(preds)
    recovery_clearance = independent_recovery_metrics(preds, clearance)
    print(
        f"  recovery vs feed-clearance: {len(clearance)} disruptions, "
        f"n={recovery_clearance.overall.n} graded ticks"
    )

    episode_types = disruptive_types_by_key(truth_obs)
    episodes = extract_episodes(
        truth, episode_types, window_start=window_start, window_end=window_end
    )
    print(f"  {len(episodes)} incident episodes (severe-only truth)")

    try:
        params_doc: dict[str, Any] = json.loads(
            client.get_object(Bucket=cfg.bucket, Key=PARAMS_KEY)["Body"].read()
        )
    except Exception:
        params_doc = {}
    scorecard = episode_scorecard(
        episodes,
        preds,
        movement_truth,
        dwell_lookup_from_params(params_doc),
        window_start=window_start,
        window_end=window_end,
    )
    onset = scorecard["onset_latency"]
    print(
        f"  scorecard: onset {onset['n_detected']}/{onset['n_episodes']} detected, "
        f"recovery scored n={scorecard['recovery']['n_scored']}, "
        f"false alarms {scorecard['false_alarms']['n_false_alarm']}"
    )

    plot_reliability(eval_doc, out_dir / "reliability.png")
    plot_recovery_by_route(eval_doc, out_dir / "recovery_by_route.png")
    plot_confusion(
        conf,
        out_dir / "confusion.png",
        truth_label=f"severe-only MTA state (floor={args.severity_floor})",
        title="Regime confusion: HMM vs severe-only MTA truth (row-normalized)",
    )
    plot_confusion(
        conf_breadth,
        out_dir / "confusion_breadth.png",
        truth_label="breadth MTA state (any alert = disrupted)",
        title="Regime confusion: HMM vs breadth MTA truth (sensitivity)",
    )
    plot_confusion(
        conf_movement,
        out_dir / "confusion_movement.png",
        truth_label="movement-derived state (from vehicle positions)",
        title="Regime confusion: HMM vs vehicle movement (row-normalized)",
    )
    plot_changepoint_alignment(deltas, out_dir / "changepoint_alignment.png")

    matched = [d for d in deltas if d is not None]
    abs_sorted = sorted(abs(d) for d in matched)
    summary = {
        "generated_at": int(datetime.now(UTC).timestamp()),
        "truth_version": TRUTH_VERSION,
        "truth_definition": {
            "severity_floor": args.severity_floor,
            "source": "mta_alerts_severity_graded",
            "planned_work": "deterministic schedule overlay (tier 0 -> normal)",
            "note": (
                "canonical truth grades disrupted only at severity tier "
                f">= {args.severity_floor}; breadth (floor 1) kept as a sensitivity"
            ),
        },
        "window": {"start": window_start, "end": window_end, "days": args.days},
        "counts": {
            "predictions": len(preds),
            "transitions": len(trans),
            "mta_truth_ticks": len(truth),
            "movement_truth_ticks": len(movement_truth),
        },
        "calibration": eval_doc["calibration"],
        "recovery": eval_doc["recovery"],
        # Independent recovery truth from alert-feed clearance, beside the
        # argmax-based `recovery`. A feed-clearance proxy, not true service
        # recovery (that's the trip-updates signal).
        "recovery_clearance": {
            **recovery_as_dict(recovery_clearance),
            "truth_source": "alert_feed_clearance",
            "n_disruptions": len(clearance),
        },
        "current_params": eval_doc["current_params"],
        # Canonical confusion matrix: HMM condition vs severe-only MTA truth
        # (truth_version + severity_floor recorded in truth_definition above).
        "confusion": conf,
        # Breadth truth (any alert = disrupted), kept as a labeled sensitivity.
        # reclassified_ticks = ticks the canonical truth reads normal that the
        # breadth truth called disrupted.
        "confusion_breadth": {
            "matrix": conf_breadth,
            "truth_source": "mta_alerts_breadth",
            "severity_floor": 1,
            "reclassified_ticks": reclassified,
        },
        # HMM condition vs vehicle-movement truth — independent in derivation from
        # the alert feed above (where trains physically are, not what the feed
        # says). truth_source documents the column axis.
        "confusion_movement": {
            "matrix": conf_movement,
            "truth_source": "vehicle_movement",
        },
        "episodes": episodes_summary(episodes),
        "episode_scorecard": scorecard,
        "changepoint_alignment": {
            "window_minutes": CHANGEPOINT_WINDOW_MIN,
            "n_total": len(deltas),
            "n_matched": len(matched),
            "n_unmatched": sum(1 for d in deltas if d is None),
            "mean_delta_min": (sum(matched) / len(matched)) if matched else None,
            "median_abs_delta_min": (
                abs_sorted[len(abs_sorted) // 2] if abs_sorted else None
            ),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {out_dir}/ (summary.json + 6 PNGs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
