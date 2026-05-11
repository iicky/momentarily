"""Snapshot contract — the JSON shape Momentarily publishes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1"

type AlertSource = Literal["subway", "lirr", "mnr", "bus"]
type EquipmentType = Literal["elevator", "escalator"]
type Mode = Literal["subway", "lirr", "mnr", "bus"]


class TimeRange(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    start: int | None = None
    end: int | None = None


class InformedEntity(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    agency_id: str | None = None
    route_id: str | None = None
    stop_id: str | None = None
    trip_id: str | None = None
    direction_id: int | None = None


class TranslatedText(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    text: str
    language: str | None = None


class TranslatedString(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    translation: list[TranslatedText] = []


class Alert(BaseModel):
    """The atomic unit. Everything else is derived from these."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    alert_type: str
    sort_order: int | None = None
    active_period: list[TimeRange] = []
    created_at: int | None = None
    updated_at: int | None = None
    display_before_active: int | None = None
    header_text: TranslatedString | None = None
    description_text: TranslatedString | None = None
    informed_entities: list[InformedEntity] = []
    source: AlertSource


class DirectionLabels(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    north: str | None = None
    south: str | None = None


class Route(BaseModel):
    """Static-ish per-route metadata. From GTFS static + canonical lists."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    mode: Mode
    short_name: str
    long_name: str | None = None
    color: str | None = None
    text_color: str | None = None
    direction_labels: DirectionLabels | None = None


class DirectionStatus(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    alerts: list[str] = []
    primary_alert_type: str | None = None


class RouteStatus(BaseModel):
    """Derived per-route view from alerts + route metadata."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    route_id: str
    alerts: list[str] = []
    primary_alert_type: str | None = None
    label: str
    by_direction: dict[Literal["northbound", "southbound"], DirectionStatus] = {}


class Station(BaseModel):
    """Static-ish per-station metadata. From GTFS + NYS Open Data 39hk-dx4f."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    gtfs_stop_id: str
    station_complex_id: str | None = None
    name: str
    borough: str | None = None
    routes_served: list[str] = []
    ada: Literal[0, 1, 2] = 0
    ada_northbound: bool = False
    ada_southbound: bool = False


class EquipmentOutage(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    reason: str | None = None
    est_return: int | None = None
    since: int | None = None


class Equipment(BaseModel):
    """Elevator or escalator with optional active outage."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    equipment_id: str
    type: EquipmentType
    station_complex_id: str | None = None
    location_text: str | None = None
    ada_pathway: bool = False
    outage: EquipmentOutage | None = None


class StationStatus(BaseModel):
    """Derived per-station view from alerts + equipment + static metadata."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    station_complex_id: str
    alerts: list[str] = []
    ada_status: Literal["operational", "ada_degraded", "non_ada"] = "operational"
    elevators_total: int = 0
    elevators_out: int = 0
    escalators_total: int = 0
    escalators_out: int = 0


class ModeRollup(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    routes_with_alerts: list[str] = []
    alert_count: int = 0
    severity_max: int = 0


class Accessibility(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    elevators_out: int = 0
    escalators_out: int = 0
    ada_pathways_degraded: int = 0


class SystemStatus(BaseModel):
    """Top-of-dashboard rollup. The one-liner sensor."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    by_mode: dict[Mode, ModeRollup] = {}
    accessibility: Accessibility = Field(default_factory=Accessibility)
    overall_label: str = "All systems normal"


class Freshness(BaseModel):
    """When each upstream source was last successfully fetched (epoch seconds)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    subway_alerts: int | None = None
    lirr_alerts: int | None = None
    mnr_alerts: int | None = None
    bus_alerts: int | None = None
    ene: int | None = None
    stations_static: int | None = None


class CompatRouteSummary(BaseModel):
    """Legacy compat: matches homeassistant-mta-subway's DirectionalStatus."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    north: str | None = None
    south: str | None = None


class CompatServiceChangeSummary(BaseModel):
    """Legacy compat: matches homeassistant-mta-subway's ServiceChangeSummary."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    both: list[str] = []
    north: list[str] = []
    south: list[str] = []


class CompatRoute(BaseModel):
    """Legacy compat: matches homeassistant-mta-subway's Route exactly.

    Produced so existing HA installs can swap API_URL with zero code change
    and continue reading subway state from this snapshot.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    name: str
    color: str
    status: str
    scheduled: bool = True
    direction_statuses: CompatRouteSummary | None = None
    delay_summaries: CompatRouteSummary | None = None
    service_irregularity_summaries: CompatRouteSummary | None = None
    service_change_summaries: CompatServiceChangeSummary | None = None


class Compat(BaseModel):
    """Legacy surfaces, derived from canonical types above."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    subwaynow_routes: dict[str, CompatRoute] = Field(default_factory=dict)


class Snapshot(BaseModel):
    """The full published snapshot. The contract."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    schema_version: str = SCHEMA_VERSION
    generated_at: int
    attribution: str = (
        "Snapshot built from MTA GTFS-RT feeds via api.mta.info. "
        "Published by Momentarily (https://feed.momentarily.nyc). "
        "Not affiliated with the MTA."
    )
    freshness: Freshness = Field(default_factory=Freshness)

    alerts: list[Alert] = []
    routes: dict[str, Route] = Field(default_factory=dict)
    route_status: dict[str, RouteStatus] = Field(default_factory=dict)
    stations: dict[str, Station] = Field(default_factory=dict)
    station_status: dict[str, StationStatus] = Field(default_factory=dict)
    equipment: list[Equipment] = []
    system: SystemStatus = Field(default_factory=SystemStatus)

    compat: Compat = Field(default_factory=Compat)
