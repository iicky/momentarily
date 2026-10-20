"""``archive/alerts_liveness`` -> ``LegacyFetchOutcome`` for ``alerts``.

Object: ``archive/alerts_liveness/<YYYY-MM-DD>/<observed_at>.json`` holding
``{observed_at, outcome, fetched_at}``, one per attempt, keyed on the wall
clock. ``outcome`` is ``success`` or ``fail``; ``fetched_at`` is the attempt
time on success and the last success carried forward on failure. The
scheduled minute, HTTP status and payload were not kept.
"""

from __future__ import annotations

import re

from momentarily.model.observations.contracts import LegacyArchive, LegacyFetchOutcome
from momentarily.model.observations.legacy.common import (
    Emitted,
    LegacyObject,
    MalformedObjectError,
    as_dict,
    check_day,
    get_int,
    get_str,
    lineage,
    load_json,
    match_key,
    unknown,
)

ARCHIVE: LegacyArchive = "archive/alerts_liveness"

_KEY = re.compile(
    r"archive/alerts_liveness/(?P<day>\d{4}-\d{2}-\d{2})/(?P<at>\d+)\.json"
)
_OUTCOMES = {"success": True, "fail": False}
_WALL_CLOCK_ONLY = unknown("archive/alerts_liveness keeps the wall clock only")


def adapt(obj: LegacyObject) -> list[Emitted]:
    m = match_key(_KEY, obj.key)
    doc = as_dict(load_json(obj), obj.key)
    observed_at = get_int(doc, "observed_at", obj.key)
    if int(m["at"]) != observed_at:
        raise MalformedObjectError(f"{obj.key} names another observed_at")
    check_day(obj.key, m["day"], observed_at)
    outcome = get_str(doc, "outcome", obj.key)
    if outcome not in _OUTCOMES:
        raise MalformedObjectError(f"{obj.key}: unknown outcome {outcome!r}")
    record = LegacyFetchOutcome(
        legacy_fetch_id=f"{obj.key}#alerts",
        lineage=lineage(ARCHIVE, obj.key, None),
        source="alerts",
        observed_at=observed_at,
        scheduled_at=_WALL_CLOCK_ONLY,
        succeeded=_OUTCOMES[outcome],
        last_success_at=get_int(doc, "fetched_at", obj.key),
    )
    return [Emitted(record, (obj.source(observed_at),))]
