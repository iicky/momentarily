"""Pure functions deriving snapshot views from atomic alerts/equipment.

Kept separate from fetching and publishing so the derivation logic is
exhaustively testable in isolation.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from momentarily.mapping import NO_ALERTS_FALLBACK, coarse_status
from momentarily.schema import (
    Accessibility,
    Alert,
    CompatRoute,
    CompatRouteSummary,
    CompatServiceChangeSummary,
    DirectionStatus,
    Equipment,
    Mode,
    ModeRollup,
    Route,
    RouteStatus,
    SystemStatus,
)

type Direction = Literal["northbound", "southbound"]


def alerts_for_route(alerts: Iterable[Alert], route_id: str, now: int) -> list[Alert]:
    """Return alerts that mention this route AND are active at `now`."""
    matching: list[Alert] = []
    for alert in alerts:
        if not any(ie.route_id == route_id for ie in alert.informed_entities):
            continue
        if not _active_at(alert, now):
            continue
        matching.append(alert)
    return matching


def _active_at(alert: Alert, now: int) -> bool:
    if not alert.active_period:
        return True
    for period in alert.active_period:
        start = period.start if period.start is not None else 0
        end = period.end if period.end is not None else 9_999_999_999
        if start <= now <= end:
            return True
    return False


def _primary(alerts: list[Alert]) -> Alert | None:
    """Pick the highest-severity alert (largest sort_order)."""
    if not alerts:
        return None
    return max(alerts, key=lambda a: a.sort_order or 0)


def derive_route_status(route: Route, alerts: list[Alert], now: int) -> RouteStatus:
    """Build per-route derived status from currently-active alerts."""
    active = alerts_for_route(alerts, route.id, now)

    if not active:
        return RouteStatus(
            route_id=route.id,
            alerts=[],
            primary_alert_type=None,
            label=NO_ALERTS_FALLBACK,
            by_direction={},
        )

    primary = _primary(active)
    primary_type = primary.alert_type if primary else None

    return RouteStatus(
        route_id=route.id,
        alerts=[a.id for a in active],
        primary_alert_type=primary_type,
        label=coarse_status(primary_type),
        by_direction=_split_by_direction(active),
    )


def _split_by_direction(active: list[Alert]) -> dict[Direction, DirectionStatus]:
    """Split active alerts into northbound/southbound buckets.

    GTFS-RT direction_id: 0 = northbound (uptown for subway), 1 = southbound.
    Alerts without a direction apply to both.
    """
    north: list[Alert] = []
    south: list[Alert] = []
    for alert in active:
        dirs = {ie.direction_id for ie in alert.informed_entities}
        if 0 in dirs or None in dirs:
            north.append(alert)
        if 1 in dirs or None in dirs:
            south.append(alert)

    return {
        "northbound": _direction_status(north),
        "southbound": _direction_status(south),
    }


def _direction_status(alerts: list[Alert]) -> DirectionStatus:
    if not alerts:
        return DirectionStatus(alerts=[], primary_alert_type=None)
    primary = _primary(alerts)
    return DirectionStatus(
        alerts=[a.id for a in alerts],
        primary_alert_type=primary.alert_type if primary else None,
    )


def derive_compat_route(
    route: Route, status: RouteStatus, alerts: list[Alert]
) -> CompatRoute:
    """Project a derived RouteStatus into the legacy subwaynow_routes shape.

    Mirrors the field set in homeassistant-mta-subway/custom_components/mta_subway/models.py
    so existing installs can swap API_URL and read this view unchanged.
    """
    by_dir = status.by_direction
    north_dir = by_dir.get("northbound")
    south_dir = by_dir.get("southbound")

    direction_statuses = CompatRouteSummary(
        north=coarse_status(north_dir.primary_alert_type) if north_dir else None,
        south=coarse_status(south_dir.primary_alert_type) if south_dir else None,
    )

    delays = _summary_texts(alerts, "Delay")
    irreg = _summary_texts(alerts, "Slow", "Reroute", "Skip")
    changes_both = _summary_texts(
        alerts, "Service Change", "Suspend", "Express", "Local"
    )

    return CompatRoute(
        id=route.id,
        name=route.short_name,
        color=route.color or "#000000",
        status=status.label,
        scheduled=True,
        direction_statuses=direction_statuses,
        delay_summaries=CompatRouteSummary(
            north=delays[0] if delays else None,
            south=delays[1] if len(delays) > 1 else None,
        ),
        service_irregularity_summaries=CompatRouteSummary(
            north=irreg[0] if irreg else None,
            south=irreg[1] if len(irreg) > 1 else None,
        ),
        service_change_summaries=CompatServiceChangeSummary(
            both=changes_both,
            north=[],
            south=[],
        ),
    )


def _summary_texts(alerts: list[Alert], *keywords: str) -> list[str]:
    """Extract human-readable header text for alerts matching any keyword."""
    out: list[str] = []
    for alert in alerts:
        if not any(kw.lower() in alert.alert_type.lower() for kw in keywords):
            continue
        if alert.header_text and alert.header_text.translation:
            for t in alert.header_text.translation:
                if t.language in (None, "en"):
                    out.append(t.text)
                    break
    return out


def derive_system_status(
    route_statuses: Iterable[RouteStatus],
    equipment: Iterable[Equipment],
) -> SystemStatus:
    """Build the top-level system rollup."""
    routes_with_alerts: list[str] = []
    severity_max = 0
    alert_count = 0

    for rs in route_statuses:
        if rs.alerts:
            routes_with_alerts.append(rs.route_id)
            alert_count += len(rs.alerts)

    equipment_list = list(equipment)
    elevators_out = sum(
        1 for e in equipment_list if e.type == "elevator" and e.outage is not None
    )
    escalators_out = sum(
        1 for e in equipment_list if e.type == "escalator" and e.outage is not None
    )
    ada_degraded = sum(
        1
        for e in equipment_list
        if e.type == "elevator" and e.ada_pathway and e.outage is not None
    )

    by_mode: dict[Mode, ModeRollup] = {
        "subway": ModeRollup(
            routes_with_alerts=routes_with_alerts,
            alert_count=alert_count,
            severity_max=severity_max,
        ),
    }

    overall = _overall_label(routes_with_alerts, elevators_out, ada_degraded)

    return SystemStatus(
        by_mode=by_mode,
        accessibility=Accessibility(
            elevators_out=elevators_out,
            escalators_out=escalators_out,
            ada_pathways_degraded=ada_degraded,
        ),
        overall_label=overall,
    )


def _overall_label(
    routes_with_alerts: list[str], elevators_out: int, ada_degraded: int
) -> str:
    if not routes_with_alerts and elevators_out == 0:
        return "All systems normal"
    if routes_with_alerts and ada_degraded > 0:
        return (
            f"Alerts on {len(routes_with_alerts)} subway lines, "
            f"{ada_degraded} ADA pathway(s) degraded"
        )
    if routes_with_alerts:
        return f"Alerts on {len(routes_with_alerts)} subway lines"
    return f"{elevators_out} elevator(s) out of service"
