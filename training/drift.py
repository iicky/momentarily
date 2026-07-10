"""Input-drift signals for the eval job.

Output grading (Brier/reliability/recovery) tells us when the model is wrong;
input drift is the leading indicator that fires *before* calibration degrades —
the live feed moving away from what the model expects.

The sharpest such signal is the rate of observed alert_types we have no mapping
for. When the MTA introduces a new alert_type, the HMM can't place it and the
coarse status falls through to a passthrough label — emissions silently go
stale. This counts those, names the offenders (so the fix is "add these to
mapping.ALERT_TYPE_TO_STATUS"), and breaks the rate out per route.

Computed from the predictions stream the eval job already loads — primary_alert_type
is the raw MTA value, so no extra archive reads.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any, Protocol

from momentarily.mapping import is_known_alert_type, is_planned_or_scheduled_type
from training.load import TickObservation


class _Typed(Protocol):
    # Read-only properties so the frozen PredictionRecord dataclass satisfies it
    # structurally without drift importing eval (eval imports drift).
    @property
    def route(self) -> str: ...
    @property
    def primary_alert_type(self) -> str | None: ...


def unmapped_alert_type_drift(predictions: Sequence[_Typed]) -> dict[str, Any]:
    """Share of typed prediction-ticks whose alert_type has no mapping.

    Denominator is ticks with a non-null primary_alert_type — a tick with no
    active alert carries no type signal either way. Returns 0.0 with empty
    breakdowns when nothing is typed."""
    total = 0
    unmapped = 0
    by_route_total: Counter[str] = Counter()
    by_route_unmapped: Counter[str] = Counter()
    offenders: Counter[str] = Counter()

    for p in predictions:
        at = p.primary_alert_type
        if not at or is_planned_or_scheduled_type(at):
            # No type signal, or a type the HMM ignores (handled/scheduled) —
            # not input drift, so it stays out of the denominator entirely.
            continue
        total += 1
        by_route_total[p.route] += 1
        if not is_known_alert_type(at):
            unmapped += 1
            by_route_unmapped[p.route] += 1
            offenders[at] += 1

    return {
        "n_typed_ticks": total,
        "unmapped_rate": unmapped / total if total else 0.0,
        # The actual unrecognized strings, most frequent first — each is a row
        # to add to the mapping table.
        "unmapped_types": dict(offenders.most_common()),
        # Only routes that actually saw an unmapped type, as a rate.
        "by_route": {
            r: by_route_unmapped[r] / by_route_total[r]
            for r in sorted(by_route_unmapped)
        },
    }


# --- Emission-channel distribution drift ------------------------------------
#
# Has the per-(route, tod_bin) distribution of the HMM's observation channels
# moved away from what the model was trained on? params.json stores the
# training-window profile (the reference); eval builds the same profile over a
# recent window and compares. alert_count drift is a PSI over fixed bins; the
# Bernoulli flags compare as a rate delta.

# alert_count is long-tailed and mostly small — fixed bins keep the PSI stable
# and the stored profile tiny.
_AC_BIN_LABELS = ("0", "1", "2", "3", "4-5", "6+")
_FLAGS = ("suspended", "delays", "service_change", "planned")
# PSI convention: <0.1 no shift, 0.1-0.25 moderate, >0.25 significant.
PSI_SIGNIFICANT = 0.25
# Cells thinner than this in either window are too noisy to score.
MIN_CELL_N = 30


def _ac_bin(count: int) -> int:
    if count <= 3:
        return count
    return 4 if count <= 5 else 5


def build_input_profile(ticks: Iterable[TickObservation]) -> dict[str, Any]:
    """Per-(route, tod_bin) profile of the emission channels: an alert_count
    histogram (counts over fixed bins) and the four flag rates. Compact and
    JSON-able — the same builder runs at train time (reference, stored in
    params.json) and at eval time (current window)."""
    acc: dict[str, dict[str, dict[str, Any]]] = {}
    for t in ticks:
        o = t.observation
        cell = acc.setdefault(t.route_id, {}).setdefault(
            str(o.tod_bin),
            {
                "n": 0,
                "hist": [0] * len(_AC_BIN_LABELS),
                "flags": dict.fromkeys(_FLAGS, 0),
            },
        )
        cell["n"] += 1
        cell["hist"][_ac_bin(o.alert_count)] += 1
        cell["flags"]["suspended"] += int(o.has_suspended_alert)
        cell["flags"]["delays"] += int(o.has_delays)
        cell["flags"]["service_change"] += int(o.has_service_change)
        cell["flags"]["planned"] += int(o.has_planned)

    # Flag counts -> rates; histogram stays as counts (PSI normalizes itself).
    for by_bin in acc.values():
        for cell in by_bin.values():
            n = cell["n"]
            cell["flags"] = {k: v / n for k, v in cell["flags"].items()}
    return acc


def _psi(ref_counts: Sequence[int], cur_counts: Sequence[int]) -> float:
    """Population Stability Index between two histograms (count vectors)."""
    ref_total = sum(ref_counts) or 1
    cur_total = sum(cur_counts) or 1
    eps = 1e-6
    psi = 0.0
    for r, c in zip(ref_counts, cur_counts, strict=True):
        rp = max(r / ref_total, eps)
        cp = max(c / cur_total, eps)
        psi += (cp - rp) * math.log(cp / rp)
    return psi


def emission_channel_drift(
    reference: dict[str, Any], current: dict[str, Any], *, min_n: int = MIN_CELL_N
) -> dict[str, Any]:
    """Compare a current emission profile against the training reference.

    Scores only (route, tod_bin) cells with >= min_n ticks in both windows —
    thinner cells are too noisy. Per route: the worst alert_count PSI and the
    largest absolute flag-rate delta across its cells, plus whether the PSI
    crossed the significance line. Empty reference (pre-profile params) yields
    a null-ish result rather than spurious drift."""
    by_route: dict[str, Any] = {}
    cells_scored = 0
    cells_skipped = 0

    for route, ref_bins in reference.items():
        cur_bins = current.get(route, {})
        worst_psi = 0.0
        worst_flag_delta = 0.0
        worst_flag = None
        n_cells = 0
        for tod, ref_cell in ref_bins.items():
            cur_cell = cur_bins.get(tod)
            if cur_cell is None or ref_cell["n"] < min_n or cur_cell["n"] < min_n:
                cells_skipped += 1
                continue
            n_cells += 1
            cells_scored += 1
            psi = _psi(ref_cell["hist"], cur_cell["hist"])
            worst_psi = max(worst_psi, psi)
            for flag in _FLAGS:
                delta = abs(cur_cell["flags"][flag] - ref_cell["flags"][flag])
                if delta > worst_flag_delta:
                    worst_flag_delta = delta
                    worst_flag = flag
        if n_cells:
            by_route[route] = {
                "max_alert_count_psi": round(worst_psi, 4),
                "max_flag_delta": round(worst_flag_delta, 4),
                "max_flag_delta_channel": worst_flag,
                "n_cells": n_cells,
                "significant": worst_psi >= PSI_SIGNIFICANT,
            }

    drifted = sorted(r for r, v in by_route.items() if v["significant"])
    return {
        "cells_scored": cells_scored,
        "cells_skipped_thin": cells_skipped,
        "psi_threshold": PSI_SIGNIFICANT,
        "routes_drifted": drifted,
        "by_route": by_route,
    }
