"""Parse MTA elevator/escalator (E&E) feed records.

The nyct_ene.json feed publishes one record per current outage with shape:

    {
      "station": "61 St-Woodside",
      "borough": "",
      "trainno": "7/LIRR",
      "equipment": "ES448",
      "equipmenttype": "ES",          # "EL" elevator | "ES" escalator
      "serving": "...",
      "ADA": "Y" | "N",
      "outagedate": "MM/DD/YYYY HH:MM:SS AM/PM",   # ET wall time
      "estimatedreturntoservice": "MM/DD/YYYY HH:MM:SS AM/PM" | "",
      "reason": "...",
      "isupcomingoutage": "Y" | "N",
      "ismaintenanceoutage": "Y" | "N"
    }

We pass these through to the Equipment + EquipmentOutage schema, parsing the
ET wall times to epoch seconds. State is observable here — no HMM needed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast
from zoneinfo import ZoneInfo

from momentarily.schema import Equipment, EquipmentOutage

NY_TZ = ZoneInfo("America/New_York")
_DATE_FMT = "%m/%d/%Y %I:%M:%S %p"


def _parse_et_epoch(raw: str | None) -> int | None:
    """Parse an ET wall-time string from the feed to UTC epoch seconds."""
    if not raw or not raw.strip():
        return None
    try:
        dt_naive = datetime.strptime(raw.strip(), _DATE_FMT)
    except ValueError:
        return None
    dt_et = dt_naive.replace(tzinfo=NY_TZ)
    return int(dt_et.astimezone(UTC).timestamp())


def _equip_type(raw: str | None) -> str | None:
    """Map the feed's equipmenttype to our schema literal."""
    if raw == "EL":
        return "elevator"
    if raw == "ES":
        return "escalator"
    return None


def parse_outage_record(record: dict[str, Any]) -> Equipment | None:
    """Convert one E&E feed record into an Equipment with an active outage.

    Returns None for records we can't usefully represent (unknown equipment type,
    missing equipment id, etc.).
    """
    eq_type = _equip_type(record.get("equipmenttype"))
    if eq_type is None:
        return None
    eq_id = record.get("equipment")
    if not eq_id:
        return None

    since = _parse_et_epoch(record.get("outagedate"))
    est_return = _parse_et_epoch(record.get("estimatedreturntoservice"))
    reason = record.get("reason") or None

    outage = EquipmentOutage(
        reason=reason,
        est_return=est_return,
        since=since,
    )

    return Equipment(
        equipment_id=str(eq_id),
        type=cast(Any, eq_type),
        station_complex_id=record.get("station") or None,
        location_text=record.get("serving") or None,
        ada_pathway=(record.get("ADA") == "Y"),
        outage=outage,
    )


def parse_feed_payload(payload: Any) -> list[Equipment]:
    """Parse a full nyct_ene.json payload (a list of outage records)."""
    if not isinstance(payload, list):
        return []
    out: list[Equipment] = []
    for record in payload:
        if not isinstance(record, dict):
            continue
        eq = parse_outage_record(record)
        if eq is not None:
            out.append(eq)
    return out


def is_active_outage(outage: EquipmentOutage | None, *, now: int) -> bool:
    """Whether this outage should currently count against the station."""
    if outage is None or outage.since is None:
        return False
    if outage.est_return is not None and outage.est_return < now:
        # Estimated return has already passed; treat as resolved until feed
        # confirms otherwise. Adds a small forgive-window so a stuck feed
        # doesn't keep flagging "out" for hours past the ETA.
        return False
    return True


# Long-running outages (e.g. capital replacement) carry est_return values years
# out. Worth surfacing differently in UIs; this constant gives downstream
# consumers a default threshold.
LONG_RUNNING_DAYS = 30


def is_long_running(outage: EquipmentOutage, *, now: int) -> bool:
    if outage.since is None:
        return False
    return (now - outage.since) >= int(timedelta(days=LONG_RUNNING_DAYS).total_seconds())
