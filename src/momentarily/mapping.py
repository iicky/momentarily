"""MTA alert_type → coarse HA status string mapping.

Maintained as MTA's alerts feed adds new alert_type values over time. Unknown
values pass through as their own label rather than being silently dropped.

Status buckets are chosen to preserve the entity vocabulary the
homeassistant-mta-subway integration shipped before the Momentarily migration:
Good Service / Delays / Service Change / Suspended / Planned Work / Information.

Source of alert_type values: observed in production subway-alerts feed. See
also https://www.mta.info/developers/service-status-box for primitives.
"""

from __future__ import annotations

# When no active alerts apply to a route, this is the label we emit.
# Preserves backwards-compatible UX for users of homeassistant-mta-subway 0.x.
NO_ALERTS_FALLBACK = "Good Service"

# Coarse buckets the HA integration's entity state should resolve to.
COARSE_STATUSES = {
    "Good Service",
    "Delays",
    "Service Change",
    "Suspended",
    "Planned Work",
    "Information",
}

# Direct mappings from MTA alert_type strings to coarse buckets.
ALERT_TYPE_TO_STATUS: dict[str, str] = {
    # Delays family
    "Delays": "Delays",
    "Some Delays": "Delays",
    "Severe Delays": "Delays",
    "Slow Speeds": "Delays",
    # Service Change family (anything that alters routing)
    "Service Change": "Service Change",
    "Part Suspended": "Service Change",
    "Trains Rerouted": "Service Change",
    "Reroute": "Service Change",
    "Stations Skipped": "Service Change",
    "Stops Skipped": "Service Change",
    "Local to Express": "Service Change",
    "Express to Local": "Service Change",
    "Reduced Service": "Service Change",
    "Boarding Change": "Service Change",
    # Information family
    "Station Notice": "Information",
    "Special Schedule": "Information",
    # Suspended family
    "Suspended": "Suspended",
    "No Trains": "Suspended",
    # Information family
    "Information": "Information",
    "Other": "Information",
    # Railroad-specific (LIRR / MNR)
    "Cancellations": "Service Change",
    "Track Change": "Service Change",
    "Weather": "Service Change",
}


def coarse_status(alert_type: str | None) -> str:
    """Map a raw MTA alert_type string to one of the coarse status buckets.

    Rules:
      - None → fallback (Good Service)
      - exact match → that bucket
      - starts with "Planned" → "Planned Work"
      - "No <Direction> Service" → "Suspended"
      - unknown → pass through raw string so consumers see something
    """
    if alert_type is None:
        return NO_ALERTS_FALLBACK

    if alert_type in ALERT_TYPE_TO_STATUS:
        return ALERT_TYPE_TO_STATUS[alert_type]

    if alert_type.startswith("Planned"):
        return "Planned Work"

    if alert_type.startswith("No ") and "Service" in alert_type:
        return "Suspended"

    return alert_type


# The `category` axis — cause/kind of disruption in our own stable vocabulary,
# orthogonal to `condition` (severity). Derived from the coarse label so there's
# one mapping table to maintain, not two.
LABEL_TO_CATEGORY: dict[str, str] = {
    "Good Service": "none",
    "Planned Work": "planned_work",
    "Delays": "delays",
    "Service Change": "service_change",
    "Suspended": "service_suspension",
    "Slow Speeds": "slow_speeds",
    "Information": "information",
}


def category_for_label(label: str) -> str:
    """Coarse status label → stable category token. Unknown → 'other'."""
    return LABEL_TO_CATEGORY.get(label, "other")


def coarse_condition(category: str) -> str:
    """Non-model severity fallback for the Python publisher.

    The live (Worker) publisher sets `condition` from the HMM's hysteresis-stable
    published label. The Python path has no model, so it derives a coarse
    severity from the category: no alert → normal, an unplanned suspension →
    suspended, anything else → disrupted.
    """
    if category == "none":
        return "normal"
    if category == "service_suspension":
        return "suspended"
    return "disrupted"


def is_known_alert_type(alert_type: str) -> bool:
    """True if we have an explicit mapping for this alert_type.

    Use this in publisher logs/metrics to detect new MTA alert_type values
    that should be added to the table.
    """
    if alert_type in ALERT_TYPE_TO_STATUS:
        return True
    if alert_type.startswith("Planned"):
        return True
    return alert_type.startswith("No ") and "Service" in alert_type
