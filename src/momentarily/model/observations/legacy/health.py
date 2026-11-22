"""``archive/health`` -> ``LegacyCollectorHealth``.

Object: ``archive/health/<YYYY-MM-DD>/<name>.json`` holding
``{observed_at, write_failures}``, one per Worker invocation (wall clock).
``name`` is the run's ``<scheduled_at>-<started_at_ms>``, or ``<observed_at>``
for objects written before per-run names. Planned sources and fetch timing
were not kept.
"""

from __future__ import annotations

from momentarily.model.observations.contracts import (
    LegacyArchive,
    LegacyCollectorHealth,
)
from momentarily.model.observations.legacy.common import (
    Emitted,
    LegacyObject,
    as_dict,
    check_day,
    get,
    get_int,
    lineage,
    load_json,
    match_key,
    run_key,
    run_scheduled_at,
)

ARCHIVE: LegacyArchive = "archive/health"

_KEY = run_key(ARCHIVE, ".json")


def adapt(obj: LegacyObject) -> list[Emitted]:
    m = match_key(_KEY, obj.key)
    doc = as_dict(load_json(obj), obj.key)
    observed_at = get_int(doc, "observed_at", obj.key)
    per_run = run_scheduled_at(m, obj.key, observed_at) is not None
    check_day(obj.key, m["day"], observed_at)
    where = f"{obj.key}.write_failures"
    failures = as_dict(get(doc, "write_failures", obj.key), where)
    counts = {name: get_int(failures, name, where) for name in failures}
    record = LegacyCollectorHealth(
        legacy_tick_id=f"archive/health@{m['name'] if per_run else observed_at}",
        lineage=lineage(ARCHIVE, obj.key, None, replaceable=False),
        observed_at=observed_at,
        write_failures=counts,
    )
    return [Emitted(record, (obj.source(observed_at),))]
