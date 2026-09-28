"""The journey-time gate: score every registered Forecaster on one holdout.

Measurement contract: spans come from a JSONL file (one SpanSample-shaped JSON
object per line, `.gz` means gzip — the layout journey_types documents).
Train/holdout split is by service_day against the TRAIN_*/HOLDOUT_* windows;
the scored set is the holdout spans whose trip_id passes the deterministic
HOLDOUT_TRIP_MODULUS rule. Every forecaster is fitted on the same train spans
and handed the SAME holdout samples, each prediction seeing only the causal
history the contract allows: spans (train or holdout, any route) that
COMPLETED within the trailing HISTORY_SECONDS before the sample departs.

Skill is only valid on a COMMON cohort (conductor amendment): each
forecaster's Report covers its own answered set (coverage and abstention are
per forecaster), but every skill number is computed on the samples answered
by ALL forecasters being compared — the forecaster itself plus both
baselines. A third forecaster's abstentions never move that cohort, so
adding or dropping an abstain-happy model cannot change anyone else's
denominator. summary.json carries, per forecaster: the Report fields,
common_cohort_n, crps_common, and skill_vs_schedule_common /
skill_vs_historical_common computed from crps_common (the baseline's CRPS is
re-averaged over the same cohort). Positive common-cohort skill over BOTH
baselines is the epic's gate.

Usage:
    python -m training.journey_gate --spans spans.jsonl.gz --out summary.json \
        [--forecaster name ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from training.journey import read_spans
from training.journey_baselines import HistoricalBaseline, ScheduleBaseline
from training.journey_score import Report, crps_from_quantiles, score
from training.journey_types import (
    HISTORY_SECONDS,
    HOLDOUT_END,
    HOLDOUT_START,
    HOLDOUT_TRIP_MODULUS,
    TRAIN_END,
    TRAIN_START,
    Forecaster,
    Quantiles,
    SpanSample,
)

# The forecasters every skill comparison is anchored to.
BASELINE_NAMES = ("schedule", "historical")


def in_scored_subsample(trip_id: str) -> bool:
    """The contract's deterministic holdout thinning rule."""
    digest = hashlib.sha1(trip_id.encode()).hexdigest()
    return int(digest, 16) % HOLDOUT_TRIP_MODULUS == 0


class HistoryIndex:
    """Completed-span lookups over the trailing window, via bisect.

    Spans are held sorted by completion time (depart_at + seconds); a query
    returns those with sample.depart_at - HISTORY_SECONDS < completion
    <= sample.depart_at, in ascending completion order.
    """

    def __init__(self, spans: Sequence[SpanSample]) -> None:
        self._spans = sorted(spans, key=lambda s: s.depart_at + s.seconds)
        self._completions = [s.depart_at + s.seconds for s in self._spans]

    def before(self, depart_at: int) -> list[SpanSample]:
        hi = bisect_right(self._completions, depart_at)
        lo = bisect_right(self._completions, depart_at - HISTORY_SECONDS)
        return self._spans[lo:hi]

    @property
    def spans(self) -> Sequence[SpanSample]:
        """Every indexed span, ascending by completion."""
        return self._spans


def registry() -> dict[str, Callable[[], Forecaster]]:
    """Every known forecaster factory. journey_model is a sibling deliverable;
    absent, the gate still runs the two baselines."""
    out: dict[str, Callable[[], Forecaster]] = {
        "schedule": ScheduleBaseline,
        "historical": HistoricalBaseline,
    }
    try:
        from training.journey_model import FORECASTERS
    except ImportError:
        pass
    else:
        out.update(FORECASTERS)
    try:
        from training.journey_state import FORECASTERS as STATE_FORECASTERS
    except ImportError:
        pass
    else:
        out.update(STATE_FORECASTERS)
    return out


def common_metrics(
    samples: Sequence[SpanSample],
    predictions: Mapping[str, Sequence[Quantiles | None]],
) -> dict[str, dict[str, float | int | None]]:
    """Common-cohort CRPS and skill, per forecaster.

    A forecaster's cohort is the samples answered by itself AND every baseline
    in BASELINE_NAMES present in `predictions`. Only those forecasters enter
    the intersection — another forecaster's abstentions never shrink this
    cohort. Baseline CRPS in each skill number is re-averaged over the SAME
    cohort, so numerator and denominator always share a denominator count.
    Empty cohort or absent baseline -> None for the affected fields.
    """
    crps: dict[str, list[float | None]] = {
        name: [
            None if q is None else crps_from_quantiles(q, float(s.seconds))
            for s, q in zip(samples, preds, strict=True)
        ]
        for name, preds in predictions.items()
    }
    baselines = [b for b in BASELINE_NAMES if b in predictions]
    out: dict[str, dict[str, float | int | None]] = {}
    for name in predictions:
        anchor = {name, *baselines}
        cohort = [
            i
            for i in range(len(samples))
            if all(crps[a][i] is not None for a in anchor)
        ]
        n = len(cohort)

        def mean_over(who: str, cohort: Sequence[int] = cohort) -> float | None:
            if not cohort:
                return None
            return sum(crps[who][i] for i in cohort) / len(cohort)  # type: ignore[misc]

        crps_common = mean_over(name)
        row: dict[str, float | int | None] = {
            "common_cohort_n": n,
            "crps_common": crps_common,
        }
        for b in BASELINE_NAMES:
            key = f"skill_vs_{b}_common"
            base = mean_over(b) if b in predictions else None
            row[key] = (
                None
                if crps_common is None or base is None or base == 0
                else 1.0 - crps_common / base
            )
        out[name] = row
    return out


def run_gate(
    spans: Sequence[SpanSample],
    names: Sequence[str] | None = None,
    *,
    train_window: tuple[str, str] = (TRAIN_START, TRAIN_END),
    holdout_window: tuple[str, str] = (HOLDOUT_START, HOLDOUT_END),
) -> tuple[
    dict[str, Report], dict[str, dict[str, float | int | None]], dict[str, int | str]
]:
    """Fit and score the selected forecasters.

    Returns per-forecaster Reports (own answered set), common-cohort metrics,
    and the split counts. The windows default to the contract's; a caller
    scoring a different week passes its own, and the split records which.
    """
    if not train_window[1] < holdout_window[0]:
        raise ValueError(
            f"train {train_window} must end before holdout {holdout_window}"
        )
    train = [s for s in spans if train_window[0] <= s.service_day <= train_window[1]]
    holdout = [
        s for s in spans if holdout_window[0] <= s.service_day <= holdout_window[1]
    ]
    scored = sorted(
        (s for s in holdout if in_scored_subsample(s.trip_id)),
        key=lambda s: (s.depart_at, s.trip_id, s.origin, s.destination),
    )
    counts: dict[str, int | str] = {
        "train_start": train_window[0],
        "train_end": train_window[1],
        "holdout_start": holdout_window[0],
        "holdout_end": holdout_window[1],
        "n_spans": len(spans),
        "n_train": len(train),
        "n_holdout": len(holdout),
        "n_scored_samples": len(scored),
    }
    # Nothing below reads the full list; dropping this binding lets a caller
    # that passed a temporary release every out-of-window span now.
    del spans
    # The contract's history set is train + holdout only: a span from an
    # out-of-window service day never enters anyone's causal window. Of the
    # train spans, only those completing inside the first scored sample's
    # window can ever be returned, so the index holds just those — the rest
    # would only be sorted and never read.
    earliest = scored[0].depart_at - HISTORY_SECONDS if scored else 0
    index = HistoryIndex(
        [s for s in train if s.depart_at + s.seconds > earliest] + holdout
    )
    factories = registry()
    if names:
        missing = sorted(set(names) - factories.keys())
        if missing:
            raise SystemExit(f"unknown forecaster(s): {', '.join(missing)}")
        factories = {n: factories[n] for n in names}
    # Fit everything before predicting so the train list can be released
    # before the prediction pass allocates.
    forecasters: dict[str, Forecaster] = {}
    for name, factory in factories.items():
        forecasters[name] = factory()
        forecasters[name].fit(train)
    del train, holdout
    predictions: dict[str, list[Quantiles | None]] = {}
    reports: dict[str, Report] = {}
    for name, forecaster in forecasters.items():
        index_history = getattr(forecaster, "index_history", None)
        if index_history is not None:
            # IndexedForecaster: sees the whole ordered history once and
            # applies the window rule itself; no per-sample slice.
            index_history(index.spans)
            preds = [forecaster.predict(s, ()) for s in scored]
        else:
            preds = [forecaster.predict(s, index.before(s.depart_at)) for s in scored]
        predictions[name] = preds
        reports[name] = score(name, list(zip(scored, preds, strict=True)))
    return reports, common_metrics(scored, predictions), counts


def _summary(
    reports: dict[str, Report],
    commons: dict[str, dict[str, float | int | None]],
    counts: dict[str, int | str],
) -> dict[str, Any]:
    forecasters: dict[str, Any] = {}
    for name, r in reports.items():
        row = asdict(r)
        row["pit_bins"] = list(r.pit_bins)
        row.update(commons[name])
        forecasters[name] = row
    return {"split": counts, "forecasters": forecasters}


def _print_table(
    reports: dict[str, Report],
    commons: dict[str, dict[str, float | int | None]],
    counts: dict[str, int | str],
) -> None:
    print(
        f"train={counts['train_start']}..{counts['train_end']} "
        f"holdout={counts['holdout_start']}..{counts['holdout_end']} "
        f"route={counts['route']} spans={counts['n_spans']} train={counts['n_train']} "
        f"holdout={counts['n_holdout']} scored={counts['n_scored_samples']}"
    )
    print(
        f"{'forecaster':<14}{'scored':>8}{'abstain':>8}{'crps':>10}"
        f"{'cov10-90':>10}{'common_n':>10}{'crps_com':>10}{'sk_sched':>10}{'sk_hist':>10}"
    )
    for name, r in reports.items():
        c = commons[name]

        def cell(v: float | int | None, spec: str) -> str:
            return "        --" if v is None else format(v, spec)

        print(
            f"{name:<14}{r.n_scored:>8}{r.n_abstained:>8}"
            f"{r.crps_mean:>10.1f}{r.coverage_10_90:>10.3f}"
            f"{c['common_cohort_n']:>10}"
            f"{cell(c['crps_common'], '>10.1f')}"
            f"{cell(c['skill_vs_schedule_common'], '>10.3f')}"
            f"{cell(c['skill_vs_historical_common'], '>10.3f')}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score every registered journey-time forecaster on the holdout."
    )
    parser.add_argument("--spans", type=Path, required=True, help="spans JSONL[.gz]")
    parser.add_argument("--out", type=Path, required=True, help="summary.json path")
    parser.add_argument(
        "--forecaster",
        action="append",
        default=None,
        help="restrict to named forecaster(s); default: all registered",
    )
    for flag, default in (
        ("--train-start", TRAIN_START),
        ("--train-end", TRAIN_END),
        ("--holdout-start", HOLDOUT_START),
        ("--holdout-end", HOLDOUT_END),
    ):
        parser.add_argument(
            flag, default=default, help=f"YYYY-MM-DD (default {default})"
        )
    parser.add_argument(
        "--route",
        default=None,
        help="score one route_id only (fit and holdout); default: all",
    )
    args = parser.parse_args(argv)
    spans = read_spans(str(args.spans))
    if args.route is not None:
        spans = [s for s in spans if s.route_id == args.route]
    reports, commons, counts = run_gate(
        spans,
        args.forecaster,
        train_window=(args.train_start, args.train_end),
        holdout_window=(args.holdout_start, args.holdout_end),
    )
    counts["route"] = args.route or "all"
    args.out.write_text(json.dumps(_summary(reports, commons, counts), indent=2) + "\n")
    _print_table(reports, commons, counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
