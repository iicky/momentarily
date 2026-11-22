"""Withheld-input evaluation of the trip matcher (docs/model-events.md §10,
slice 4).

On a raw-lineage UTC day, the runs are built once. The full matcher gives
each run its label: the trip its raw descriptor identifies through the id
tiers (1–4), kept only when the run carries one ``start_date`` and it agrees
with the trip. The matcher then runs again with the trip id's origin and
path, ``start_time``, ``start_date`` and ``train_id`` withheld (the
legacy-equivalent inputs: route field and direction, observed stops and
times), so every run goes through tier 5 and the one-to-one assignment.
The withholding covers the matcher only: the runs it sees were split by
``train_id`` and joined by id origin with the full inputs, so a legacy day,
which has neither, can fragment more than this probe shows.
Its pick is graded against the label, and every disagreement gets one
conflict class:

- ``correct``;
- ``fragmentation``: the label trip went to another run with the same label
  (one train split into runs);
- ``extra_train``: the label trip went to a run without a label;
- ``chain_shift``: another labelled run took the label trip;
- ``overtaking``: two labelled runs swapped trips;
- ``equal_costs``: the pick and the label are equally near;
- ``competing_service_dates``: the pick lies on another service day;
- ``sustained_lateness``: the label lies beyond the 900 s tolerance;
- ``no_candidate``: no pattern of the route field fits the run, or none has a
  trip within tolerance;
- ``label_not_a_candidate``: the label trip is not among the candidates
  (its pattern does not fit the observed stops);
- ``nearer_trip_preferred``: the run took a nearer free trip;
- ``unassigned``: the label trip was a candidate and stayed free, but the
  leave-unassigned cost won.

Usage (from the repo root; the stores are local directories):

    uv run python scripts/trip_match_probe.py --root <layer-2 store> \\
        --network <layer-1 dir> --day YYYY-MM-DD [--build-id ID] \\
        [--cases N] [--out PATH]

Prints a JSON report: ``assignment_accuracy_given_full_runs`` (assignment
accuracy given runs built from full descriptors, not legacy-equivalent
end-to-end accuracy), counts per class, accuracy per feed, service-day
agreement, and ``--cases`` hard cases spread over the classes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from momentarily.model.events.contracts import TripMatch
from momentarily.model.events.match_build import TZ, Indexes, match_window
from momentarily.model.events.matching import DRIFT_TOL_S, Matcher, trip_matches
from momentarily.model.events.runs import Run
from momentarily.model.observations.derive.storage import LocalStorage

ID_TIERS = ("exact_id", "path_prefix", "line_code", "path_differs")


def _et(at: int) -> str:
    return datetime.fromtimestamp(at, ZoneInfo(TZ)).strftime("%m-%d %H:%M")


def _label(run: Run, m: TripMatch) -> tuple[str, str] | None:
    if m.tier not in ID_TIERS or m.scheduled_trip_id is None:
        return None
    dates = {r.start_date for r in run.rows}
    if len(dates) != 1 or None in dates or "start_date_mismatch" in m.flags:
        return None
    return (m.scheduled_trip_id, m.service_date)


def probe(
    root: Path, network: Path, day: str, build_id: str | None, n_cases: int
) -> dict[str, Any]:
    indexes = Indexes(network, build_id)
    d0 = int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp())
    full = match_window(
        LocalStorage(root), indexes, "raw", d0, d0 + 86400, day, partial=True
    )
    runs = {r.run_id: r for r in full.runs}
    by_full = {m.run_id: m for m in full.matches}
    matcher = Matcher(indexes, TZ, withheld=True)
    withheld = {m.run_id: m for m in trip_matches(full.runs, matcher, "")}
    labels = {rid: lab for rid, m in by_full.items() if (lab := _label(runs[rid], m))}
    pick = {
        rid: (m.scheduled_trip_id, m.service_date)
        for rid, m in withheld.items()
        if m.scheduled_trip_id is not None
    }
    holder = {trip: rid for rid, trip in pick.items()}
    classes: Counter[str] = Counter()
    cases: dict[str, list[str]] = defaultdict(list)
    by_feed: dict[str, Counter[bool]] = defaultdict(Counter)
    sd: Counter[bool] = Counter()
    costs: dict[str, dict[tuple[str, str], int]] = {}
    for rid, lab in sorted(labels.items()):
        cand = {
            (c.trip_id, c.service_date): c.cost
            for c in matcher.drift_candidates(runs[rid], DRIFT_TOL_S)
        }
        costs[rid] = cand
        got = pick.get(rid)
        ok = got == lab
        by_feed[runs[rid].feed][ok] += 1
        if got is not None:
            sd[got[1] == lab[1]] += 1
        if ok:
            classes["correct"] += 1
            continue
        other = holder.get(lab)
        lab_off = by_full[rid].offset_s
        if not cand:
            cls = "no_candidate"
        elif got is not None and got[1] != lab[1]:
            cls = "competing_service_dates"
        elif lab not in cand:
            cls = (
                "sustained_lateness"
                if lab_off is not None and abs(lab_off) > DRIFT_TOL_S
                else "label_not_a_candidate"
            )
        elif other is not None and other in labels and labels[other] == lab:
            cls = "fragmentation"
        elif other is not None and other not in labels:
            cls = "extra_train"
        elif other is not None and got is not None and labels.get(other) == got:
            cls = "overtaking"
        elif other is not None:
            cls = "chain_shift"
        elif got is not None and cand.get(got) == cand[lab]:
            cls = "equal_costs"
        elif got is not None:
            cls = "nearer_trip_preferred"
        else:
            cls = "unassigned"
        classes[cls] += 1
        cases[cls].append(rid)

    def describe(rid: str, cls: str) -> dict[str, Any]:
        run, m, w = runs[rid], by_full[rid], withheld[rid]
        stops = list(run.stops())
        lab = labels[rid]
        return {
            "run": rid,
            "seen": f"{_et(run.first_at)}–{_et(run.last_at)[6:]}",
            "train_ids": run.train_ids,
            "label": f"{lab[0]} ({lab[1]}, {m.offset_s} s, {m.tier})",
            "probe": "unassigned"
            if w.scheduled_trip_id is None
            else f"{w.scheduled_trip_id} ({w.service_date}, {w.offset_s} s)",
            "probe_miss": w.miss_class,
            "candidates": len(costs[rid]),
            "label_cost": costs[rid].get(lab),
            "stops": stops[:3] + (["…", stops[-1]] if len(stops) > 3 else []),
            "class": cls,
        }

    hard: list[dict[str, Any]] = []
    order = sorted(cases, key=lambda c: (-len(cases[c]), c))
    depth = 0
    while len(hard) < n_cases and any(depth < len(cases[c]) for c in order):
        for c in order:
            xs = sorted(cases[c], key=lambda r: runs[r].first_at)
            if depth < len(xs) and len(hard) < n_cases:
                step = max(1, len(xs) // 3)
                hard.append(describe(xs[(depth * step + step // 2) % len(xs)], c))
        depth += 1
    n = len(labels)
    return {
        "day": day,
        "runs": len(runs),
        "labelled": n,
        "assignment_accuracy_given_full_runs": round(classes["correct"] / n, 4)
        if n
        else None,
        "given_a_trip": round(sum(rid in pick for rid in labels) / n, 4) if n else None,
        "precision": round(
            classes["correct"] / max(1, sum(rid in pick for rid in labels)), 4
        ),
        "service_day_right": f"{sd[True]}/{sd[True] + sd[False]}",
        "classes": dict(classes.most_common()),
        "accuracy_by_feed": {
            f: f"{c[True]}/{c[True] + c[False]}" for f, c in sorted(by_feed.items())
        },
        "full_match_rate": round(
            sum(m.outcome == "matched" for m in by_full.values()) / len(by_full), 4
        ),
        "hard_cases": hard,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Withheld-input evaluation of the trip matcher."
    )
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--network", type=Path, required=True)
    p.add_argument("--day", required=True)
    p.add_argument("--build-id")
    p.add_argument("--cases", type=int, default=8)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    text = (
        json.dumps(probe(a.root, a.network, a.day, a.build_id, a.cases), indent=1)
        + "\n"
    )
    sys.stdout.write(text)
    if a.out is not None:
        a.out.write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
