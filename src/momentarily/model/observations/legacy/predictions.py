"""``v1/predictions`` -> ``LegacyRouteAlertState``.

Object: ``v1/predictions/<YYYY-MM-DD>/<name>.jsonl``, one JSON line per route
with an inference row at that 5-minute tick, each line's ``ts`` the tick's
``observed_at``. ``name`` is the run's ``<scheduled_at>-<started_at_ms>``, or
``<ts>`` for objects written before per-run names. Only ``route`` and
``primary_alert_type`` (the highest-``sort_order`` active alert, planned
work included) are alert state; the inference itself is not layer 2.
"""

from __future__ import annotations

import json

from momentarily.model.observations.contracts import (
    LegacyArchive,
    LegacyRouteAlertState,
    RouteAlertGranularity,
)
from momentarily.model.observations.legacy.common import (
    Emitted,
    LegacyObject,
    MalformedObjectError,
    as_dict,
    check_day,
    get_int,
    get_opt_str,
    get_str,
    lineage,
    match_key,
    run_key,
    run_scheduled_at,
)

ARCHIVE: LegacyArchive = "v1/predictions"

_KEY = run_key(ARCHIVE, ".jsonl")
_GRANULARITY = RouteAlertGranularity(
    scope="route", tick_seconds=300, content="primary_alert_type_presence"
)


def adapt(obj: LegacyObject) -> list[Emitted]:
    m = match_key(_KEY, obj.key)
    # A name that is the tick's ts fixes observed_at before the body is read;
    # a run's name leaves it to the first line's ts.
    observed_at: int | None = None
    if m["at"] is not None:
        observed_at = int(m["at"])
        check_day(obj.key, m["day"], observed_at)
    try:
        lines = obj.body.decode().split("\n")
    except UnicodeDecodeError as err:
        raise MalformedObjectError(f"{obj.key}: not UTF-8") from err
    if lines and lines[-1] == "":
        lines.pop()
    out: list[Emitted] = []
    for i, line in enumerate(lines):
        where = f"{obj.key} line {i}"
        try:
            doc = as_dict(json.loads(line), where)
        except json.JSONDecodeError as err:
            raise MalformedObjectError(f"{where}: not JSON ({err})") from err
        if observed_at is None:
            observed_at = get_int(doc, "ts", where)
            run_scheduled_at(m, obj.key, observed_at)
            check_day(obj.key, m["day"], observed_at)
        if get_int(doc, "ts", where) != observed_at:
            raise MalformedObjectError(f"{where} belongs to another tick")
        record = LegacyRouteAlertState(
            lineage=lineage(ARCHIVE, obj.key, i, replaceable=False),
            granularity=_GRANULARITY,
            observed_at=observed_at,
            route_id=get_str(doc, "route", where),
            primary_alert_type=get_opt_str(doc, "primary_alert_type", where),
        )
        out.append(Emitted(record, (obj.source(observed_at),)))
    return out
