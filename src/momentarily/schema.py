"""Snapshot contract — the JSON shape Momentarily publishes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1"

# Open strings — adding new sources/modes/conditions doesn't break consumers.
# Documented value sets:
#   AlertSource:  "subway" | "lirr" | "mnr" | "bus" | "path" | "ferry" | ...
#   Mode:         same as AlertSource
#   Condition:    "normal" | "disrupted" | "suspended" (subject to future extension)
type AlertSource = str
type Mode = str
type EquipmentType = Literal["elevator", "escalator"]


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


class Observation(BaseModel):
    """Continuous / instantaneous measurement of an entity.

    Peer to Alert. Empty in v1 of the publisher; populated when we wire upstream
    sources for travel-time (bridges/tunnels), headway, ETAs, tolls, occupancy.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    entity_ref: str  # "<entity_type>:<id>" — e.g. "bridge:verrazano", "subway_route:1"
    kind: str  # open: "travel_time" | "headway" | "eta" | "toll" | "occupancy" | ...
    value: float | int | str
    unit: str  # open: "seconds" | "minutes" | "dollars" | "percent" | ...
    observed_at: int
    source: str


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
    agency: str | None = None  # e.g. "nyct_subway", "lirr", "mnr", "panynj_path"


class DirectionStatus(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    alerts: list[str] = []
    primary_alert_type: str | None = None


class Inference(BaseModel):
    """HMM-derived state inference.

    Per 5w0.6, populated only after the shadow review (Phase 3+). During Phase 1
    this field stays None on every entity status object.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    # Primary user-facing fields (graduate to sensor entities at Phase 4)
    #   "normal" | "disrupted" | "suspended" | "not_scheduled"
    # not_scheduled is a planned non-disruption (off-timetable, e.g. rush-only
    # lines off-hours); open for future regimes.
    condition: str
    recovery_minutes: int
    is_disrupted: bool

    # Probability vector (attribute-depth)
    p_normal: float
    p_disrupted: float
    p_suspended: float

    # Changepoint info
    regime_entered_at: int
    regime_age_seconds: int

    # Recovery posterior bounds (attribute-depth)
    recovery_minutes_low: int  # 25th percentile
    recovery_minutes_high: int  # 75th percentile

    # True when the dwell estimate saturated its ceiling — the regime is so
    # persistent (self-loop ≈ 1, typical of open-ended planned work) that the
    # model can't bound when it ends. recovery_minutes is clamped in that case.
    recovery_indeterminate: bool = False

    # Forward predictions
    p_normal_in_30min: float
    p_normal_in_60min: float
    p_normal_in_120min: float

    # Cold-start flag — true when the model is still warming up for this entity
    model_warming_up: bool = False

    # Where recovery_minutes comes from: "schedule" is a deterministic lookup of
    # the planned-work resume time (no model uncertainty); "hmm" is the dwell
    # estimate. Graders exclude "schedule" rows from HMM calibration.
    recovery_source: str = "hmm"  # "hmm" | "schedule"
    # Announced resume time (epoch s) for schedule recovery; None for "hmm".
    resumes_at: int | None = None
    # now has passed resumes_at but the planned alert is still active — recovery
    # is clamped to 0 rather than counting down past the announced time.
    overdue: bool = False


class RouteStatus(BaseModel):
    """Derived per-route view from alerts + route metadata + optional HMM inference."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    route_id: str
    alerts: list[str] = []
    # Severity axis. The Worker publisher sets this from the HMM's
    # hysteresis-stable published label; the Python path uses a coarse
    # alert-derived fallback (mapping.coarse_condition).
    #   "normal" | "disrupted" | "suspended" | "not_scheduled" | "unknown"
    condition: str = "unknown"
    # Cause axis — our stable vocabulary, derived from the MTA alert_type.
    #   "none" | "planned_work" | "delays" | "service_change" |
    #   "service_suspension" | "slow_speeds" | "information" | "other"
    category: str = "none"
    primary_alert_type: str | None = None
    # Soft-deprecated: derivable from condition + category. Kept for existing
    # consumers and the compat layer.
    label: str
    by_direction: dict[Literal["northbound", "southbound"], DirectionStatus] = {}
    inference: Inference | None = None


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
    """Derived per-station view from alerts + equipment + static + optional HMM inference."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    station_complex_id: str
    alerts: list[str] = []
    ada_status: Literal["operational", "ada_degraded", "non_ada"] = "operational"
    elevators_total: int = 0
    elevators_out: int = 0
    escalators_total: int = 0
    escalators_out: int = 0
    # Pass-through of MTA-provided estimated return time across all currently-out
    # equipment at this station. None when no equipment is out, or when none of
    # the active outages report an est_return.
    earliest_elevator_return: int | None = None
    # Epoch seconds of the longest-running equipment outage at this station.
    # Useful to surface "out for 6 months" indicators.
    oldest_outage_since: int | None = None
    inference: Inference | None = None


class Crossing(BaseModel):
    """One direction or segment of a bridge/tunnel crossing."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str  # e.g. "verrazano:upper:westbound"
    name: str


class Bridge(BaseModel):
    """Infrastructure asset. Schema scaffold; populated when a data source is wired."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    name: str
    operator: str  # "MTA-BT" | "PANYNJ" | "NYC-DOT" | ...
    crossings: list[Crossing] = []


class Tunnel(BaseModel):
    """Infrastructure asset. Schema scaffold; populated when a data source is wired."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    name: str
    operator: str
    crossings: list[Crossing] = []


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
    # Set in Phase 3+ by the HMM rollup; None during Phase 1 shadow
    condition: str | None = None  # "normal" | "degraded" | "severe"
    lines_disrupted_count: int = 0
    most_degraded_line: str | None = None
    most_recovered_line: str | None = None


class Freshness(BaseModel):
    """When each upstream source was last successfully fetched (epoch seconds)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    subway_alerts: int | None = None
    lirr_alerts: int | None = None
    mnr_alerts: int | None = None
    bus_alerts: int | None = None
    path_alerts: int | None = None
    ferry_alerts: int | None = None
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
    # Human status string. A not_scheduled route renders as "Not Scheduled" with
    # scheduled=false, so consumers see a planned gap rather than an outage.
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


class Provenance(BaseModel):
    """Which code produced this snapshot. code_sha is the git commit verbatim;
    dirty is null when it couldn't be determined (e.g. a clean-checkout build)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    code_sha: str = "unknown"
    dirty: bool | None = None
    producer: str = "unknown"


class Snapshot(BaseModel):
    """The full published snapshot. The contract."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    schema_version: str = SCHEMA_VERSION
    generated_at: int
    provenance: Provenance = Field(default_factory=Provenance)
    attribution: str = (
        "Snapshot built from MTA GTFS-RT feeds via api.mta.info. "
        "Published by Momentarily (https://feed.momentarily.nyc). "
        "Not affiliated with the MTA."
    )
    # Declares which sources are populated this run.
    # Lets consumers detect when LIRR/MNR/PATH/ferry/bridges land without schema bumps.
    supported_modes: list[str] = []
    freshness: Freshness = Field(default_factory=Freshness)

    # Atomic types
    alerts: list[Alert] = []
    observations: list[Observation] = []
    routes: dict[str, Route] = Field(default_factory=dict)
    stations: dict[str, Station] = Field(default_factory=dict)
    equipment: list[Equipment] = []
    bridges: list[Bridge] = []
    tunnels: list[Tunnel] = []

    # Derived views
    route_status: dict[str, RouteStatus] = Field(default_factory=dict)
    station_status: dict[str, StationStatus] = Field(default_factory=dict)
    system: SystemStatus = Field(default_factory=SystemStatus)

    # Legacy compat — preserves zero-breakage upgrade for HA 0.x consumers
    compat: Compat = Field(default_factory=Compat)
