"""``archive/health`` -> ``LegacyCollectorHealth``.

Object: ``archive/health/<YYYY-MM-DD>/<observed_at>.json`` holding
``{observed_at, write_failures}``, one per Worker invocation (wall clock).
Planned sources and fetch timing were not kept.
"""

from __future__ import annotations

import re

from momentarily.model.observations.contracts import (
    LegacyArchive,
    LegacyCollectorHealth,
)
from momentarily.model.observations.legacy.common import (
    Emitted,
    LegacyObject,
    MalformedObjectError,
    as_dict,
    check_day,
    get,
    get_int,
    lineage,
    load_json,
    match_key,
)

ARCHIVE: LegacyArchive = "archive/health"

_KEY = re.compile(r"archive/health/(?P<day>\d{4}-\d{2}-\d{2})/(?P<at>\d+)\.json")


def adapt(obj: LegacyObject) -> list[Emitted]:
    m = match_key(_KEY, obj.key)
    doc = as_dict(load_json(obj), obj.key)
    observed_at = get_int(doc, "observed_at", obj.key)
    if int(m["at"]) != observed_at:
        raise MalformedObjectError(f"{obj.key} names another observed_at")
    check_day(obj.key, m["day"], observed_at)
    where = f"{obj.key}.write_failures"
    failures = as_dict(get(doc, "write_failures", obj.key), where)
    counts = {name: get_int(failures, name, where) for name in failures}
    record = LegacyCollectorHealth(
        legacy_tick_id=f"archive/health@{observed_at}",
        lineage=lineage(ARCHIVE, obj.key, None),
        observed_at=observed_at,
        write_failures=counts,
    )
    return [Emitted(record, (obj.source(observed_at),))]
