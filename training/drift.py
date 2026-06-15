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

from collections import Counter
from collections.abc import Sequence
from typing import Any, Protocol

from momentarily.mapping import is_hmm_excluded, is_known_alert_type


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
        if not at or is_hmm_excluded(at):
            # No type signal, or a type the HMM deliberately ignores — neither
            # is input drift, so they stay out of the denominator entirely.
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
