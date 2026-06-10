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
from training.eval import (
    TICK_SECONDS,
    PredictionRecord,
    TransitionRecord,
    build_eval,
    load_predictions,
    load_transitions,
)
from training.load_r2 import build_tick_observations, fetch_alert_versions
from training.r2_client import load_config, make_client

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


def snap_tick(ts: int) -> int:
    return ((ts + TICK_SECONDS // 2) // TICK_SECONDS) * TICK_SECONDS


def build_mta_truth(
    client: S3Client,
    bucket: str,
    start_date: Any,
    end_date: Any,
) -> dict[tuple[str, int], str]:
    """For each (route, tick) with any active alert in the window, return MTA
    state. Ticks not in the dict had no active alerts → treat as 'normal'."""
    del bucket  # client carries its own configured bucket via load_config
    bodies = fetch_alert_versions(
        start_date=start_date, end_date=end_date, client=client
    )
    obs_list = build_tick_observations(bodies)
    return {(o.route_id, o.tick): derive_mta_state(o.observation) for o in obs_list}


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


def plot_confusion(conf: dict[str, dict[str, int]], out: Path) -> None:
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
    ax.set_xlabel("MTA-derived state (from alerts)")
    ax.set_ylabel("HMM-published condition")
    ax.set_title("Regime confusion: HMM vs MTA alerts (row-normalized)")
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
    truth = build_mta_truth(client, cfg.bucket, start_date, today)
    print(f"  {len(truth)} (route, tick) ground-truth states")

    window_end = int(datetime.now(UTC).timestamp())
    window_start = int(
        datetime(
            start_date.year, start_date.month, start_date.day, tzinfo=UTC
        ).timestamp()
    )
    eval_doc = build_eval(
        preds, trans, window_start=window_start, window_end=window_end
    )
    conf = confusion(preds, truth)
    deltas = changepoint_alignment(
        trans, truth, window_start=window_start, window_end=window_end
    )

    plot_reliability(eval_doc, out_dir / "reliability.png")
    plot_recovery_by_route(eval_doc, out_dir / "recovery_by_route.png")
    plot_confusion(conf, out_dir / "confusion.png")
    plot_changepoint_alignment(deltas, out_dir / "changepoint_alignment.png")

    matched = [d for d in deltas if d is not None]
    abs_sorted = sorted(abs(d) for d in matched)
    summary = {
        "generated_at": int(datetime.now(UTC).timestamp()),
        "window": {"start": window_start, "end": window_end, "days": args.days},
        "counts": {
            "predictions": len(preds),
            "transitions": len(trans),
            "mta_truth_ticks": len(truth),
        },
        "calibration": eval_doc["calibration"],
        "recovery": eval_doc["recovery"],
        "confusion": conf,
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
    print(f"wrote {out_dir}/ (summary.json + 4 PNGs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
