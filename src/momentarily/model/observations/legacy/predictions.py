"""``v1/predictions`` -> ``LegacyRouteAlertState``.

Object: ``v1/predictions/<YYYY-MM-DD>/<ts>.jsonl``, one JSON line per route
with an inference row at that 5-minute tick. Only ``route`` and
``primary_alert_type`` (the highest-``sort_order`` active alert, planned
work included) are alert state; the inference itself is not layer 2.
"""

from __future__ import annotations

import json
import re

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
)

ARCHIVE: LegacyArchive = "v1/predictions"

_KEY = re.compile(r"v1/predictions/(?P<day>\d{4}-\d{2}-\d{2})/(?P<at>\d+)\.jsonl")
_GRANULARITY = RouteAlertGranularity(
    scope="route", tick_seconds=300, content="primary_alert_type_presence"
)


def adapt(obj: LegacyObject) -> list[Emitted]:
    m = match_key(_KEY, obj.key)
    observed_at = int(m["at"])
    check_day(obj.key, m["day"], observed_at)
    try:
        lines = obj.body.decode().split("\n")
    except UnicodeDecodeError as err:
        raise MalformedObjectError(f"{obj.key}: not UTF-8") from err
    if lines and lines[-1] == "":
        lines.pop()
    source = obj.source(observed_at)
    out: list[Emitted] = []
    for i, line in enumerate(lines):
        where = f"{obj.key} line {i}"
        try:
            doc = as_dict(json.loads(line), where)
        except json.JSONDecodeError as err:
            raise MalformedObjectError(f"{where}: not JSON ({err})") from err
        if get_int(doc, "ts", where) != observed_at:
            raise MalformedObjectError(f"{where} belongs to another tick")
        record = LegacyRouteAlertState(
            lineage=lineage(ARCHIVE, obj.key, i),
            granularity=_GRANULARITY,
            observed_at=observed_at,
            route_id=get_str(doc, "route", where),
            primary_alert_type=get_opt_str(doc, "primary_alert_type", where),
        )
        out.append(Emitted(record, (source,)))
    return out
