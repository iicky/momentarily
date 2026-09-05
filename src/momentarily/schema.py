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

# The two directions NYCT runs, as the N/S suffix on every stop_id normalises
# to across the repo (worker/src/vehicles.ts directionOf,
# training/gtfs_static.direction_of). Closed, unlike the open strings above.
type ObservationDirection = Literal["north", "south"]


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


class ObservationSample(BaseModel):
    """One past reading in an Observation.window: a value and when it was taken.

    Compact on purpose — the window is a bounded series and repeating the
    parent's entity_ref/kind/unit/source/direction/stop_id on every entry would
    bloat the snapshot for nothing. Those are all fixed across the window, which
    is measured at one point; only value and observed_at move.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    value: float | int
    observed_at: int


class Observation(BaseModel):
    """Continuous / instantaneous measurement of an entity.

    Peer to Alert, and deliberately NOT an Inference: nothing here is fitted,
    baselined or graded. The value means only what it says it measured.

    Populated in v1 with observed subway headway, from the Worker's GTFS-RT
    vehicle-position decode (worker/src/headway.ts). Reserved for the sources
    still unwired: travel-time (bridges/tunnels), ETAs, tolls, occupancy.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    entity_ref: str  # "<entity_type>:<id>" — e.g. "bridge:verrazano", "subway_route:1"
    kind: str  # open: "travel_time" | "headway" | "eta" | "toll" | "occupancy" | ...
    value: float | int | str
    unit: str  # open: "seconds" | "minutes" | "dollars" | "percent" | ...
    observed_at: int
    source: str
    # Where the measurement was taken, when that is part of what it means. A
    # headway is measured at a point and per direction, so a route's two
    # directional readings share entity_ref and are told apart by these: the
    # canonical reference stop (one stable, documented stop per route and
    # direction — training/headway.select_reference_stops) and the direction.
    # Both None for a measurement whose entity_ref already locates it.
    #
    # `direction` is a CLOSED vocabulary, unlike the open strings above: NYCT
    # gives exactly two directions, as the N/S suffix on every stop_id, and
    # the whole repo normalises them to these two words (worker/src/vehicles.ts
    # directionOf, training/gtfs_static.direction_of). Adding a third would be
    # a schema change, not a new value in an open set — so it is a Literal and
    # reaches the published JSON Schema as an enum.
    direction: ObservationDirection | None = None
    stop_id: str | None = None
    # A bounded rolling window of this measurement's recent history, oldest
    # first — the historical N-car chain, one entry per gap between successive
    # trains. Populated for headway (worker/src/headway.ts cellWindow): the last
    # hour of the cell's readings, hard-capped in count so the snapshot cannot
    # bloat and bounded in time so it never stretches past the hour. The newest
    # entry restates this Observation's value/observed_at, and the whole window
    # shares its stop_id/direction, so a consumer renders the last hour of one
    # route's headways from this single object — no archive, no unbounded array.
    # None (not []) for a measurement that carries no series, matching the
    # optional-field precedent of direction/stop_id above.
    window: list[ObservationSample] | None = None


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

    Populated only after the shadow review (Phase 3+). During Phase 1 this field
    stays None on every entity status object.
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

    # True whenever recovery_minutes is NOT a prediction, in which case it and
    # its bounds all carry the ceiling. Three producers: the dwell estimate
    # saturated its ceiling or outlived every observed dwell (the regime is so
    # persistent — self-loop ≈ 1, typical of open-ended planned work — that the
    # model can't bound when it ends); no arm describing the published condition
    # could answer a live recovery question; or the arm that produced the
    # recovery block disagrees with is_disrupted about whether there is a
    # disruption at all.
    recovery_indeterminate: bool = False

    # Forward predictions.
    #
    # p_normal_in_30min is populated only when the forecast came from the same
    # arm (movement vs. alert-HMM) that produced `condition` above. Graded
    # against the condition actually published 30 minutes later (25,238
    # samples over 6 days), movement-sourced rows score AUC 0.856 and
    # hmm-sourced rows score AUC 0.261 — the two arms put probability on
    # different scales, so mixing them scores AUC 0.084, worse than either arm
    # alone, because the combined ranking then tracks which arm answered
    # rather than the risk. Null whenever the sourcing arm doesn't match.
    p_normal_in_30min: float | None = None
    # The 60- and 120-minute horizons lose to naive persistence in every cut
    # (BSS -0.00 to -1.30, AUC 0.395 and 0.352 — i.e. inverted, worse than a
    # coin flip), the loss is not a left-censoring artifact, and the horizon
    # projection itself was verified monotone, so the defect is the shape of
    # the fitted elapsed-conditional dwell curve. That needs a model-form
    # change and will not improve with more runtime, so rather than publish a
    # number we have measured to be anti-informative, the two longer horizons
    # are withheld: null whenever the value would come from a fitted curve.
    #
    # They stay populated when recovery_source == "schedule", where the answer
    # is a deterministic comparison against an announced resume time rather than
    # a forecast, and so carries none of the above defect.
    p_normal_in_60min: float | None = None
    p_normal_in_120min: float | None = None

    # Cold-start flag — true when the model is still warming up for this entity
    model_warming_up: bool = False

    # Where recovery_minutes comes from: "schedule" is a deterministic lookup of
    # the planned-work resume time (no model uncertainty); "movement" is the
    # movement-clock dwell curve; "hmm" is the alert-regime dwell estimate, the
    # fallback. Only the first two also decide the published condition, so only
    # they carry a forecast. Graders exclude "schedule" rows from HMM calibration.
    recovery_source: str = "hmm"  # "hmm" | "schedule" | "movement"
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
    # Severity axis — the published current state, movement-primary. The Worker
    # publisher sets it from observed train movement where judgeable, from a
    # planned "No Scheduled Service" alert where flagged, else "unknown" (an honest
    # coverage gap — alerts never assert disruption here). The alert-derived read
    # lives on as the shadow (inference.condition) and cause (category) axes.
    #   "normal" | "disrupted" | "suspended" | "not_scheduled" | "unknown"
    condition: str = "unknown"
    # Where `condition` came from. The Worker publisher emits "movement" (observed
    # from vehicle positions), "schedule" (a planned "No Scheduled Service" alert),
    # or "unknown" (movement can't judge — never an alert-derived fallback). The
    # Python alert-only path (derive_route_status) has no movement feed and emits
    # the default "hmm".
    condition_source: str = "hmm"
    # When the currently published `condition` began, epoch seconds — the badge's
    # own clock ("how long has it held"), NOT the model's argmax clock in
    # inference.regime_entered_at. Set only on the arm that can honestly time the
    # badge: the Worker publisher fills it from the movement regime's entered_at
    # when condition_source == "movement". None whenever no honest start exists —
    # "schedule" (a planned non-run whose start the Worker doesn't track), "unknown"
    # (movement declined to judge), and the alert-only Python "hmm" path. A reader
    # must never present inference.regime_entered_at as this clock; that one times
    # the HMM argmax, which flips independently of the badge.
    condition_entered_at: int | None = None
    # Supply axis — assigned_n against its own hourly baseline, one-tick lagged
    # like `condition`. Distinct from `condition` (flow): a route's trips can be
    # pulled (degraded) while the trains still running advance fine (normal), and
    # the reverse. The alert-only Python path can't judge it and leaves "unknown";
    # the Worker publisher sets it from the service regime.
    #   "normal" | "degraded" | "unknown"
    service_condition: str = "unknown"
    # Magnitude behind service_condition: assigned_n / its hourly baseline this
    # tick, or None when unjudgeable. Raw (not debounced); service_condition is
    # the debounced regime over it. The alert-only Python path leaves it None.
    service_ratio: float | None = None
    # Per-cell empirical spread ticks for the same meter, normalised onto the
    # same scale as service_ratio (cell p10/median, cell p90/median). None
    # whenever service_ratio would be None, plus whenever this cell has no
    # published quantiles (older or thin sidecar). The alert-only Python path
    # leaves both None, exactly like service_ratio.
    service_low_ratio: float | None = None
    service_high_ratio: float | None = None
    # Where service_ratio sits within this cell's own same-daypart baseline, as a
    # 0-100 percentile (worker movement_state.servicePercentile). Low = fewer
    # trains than usual for this daypart; exact at the cell's p10/median/p90,
    # saturating at 90 above its p90. A percentile of the baseline, NOT a
    # forecast. None under the same conditions as service_low_ratio.
    service_percentile: float | None = None
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
    # Last poll on which at least one GTFS-RT vehicle-position feed round-
    # tripped. The upstream behind `observations`, station_flow, segment_flow
    # and the crowding surface — published so a consumer can tell an absent
    # observation caused by a feed outage from one caused by a service gap.
    vehicle_positions: int | None = None


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


class ParamsProvenance(BaseModel):
    """Identity of the trained params.json behind this snapshot's inference — the
    one thing code_sha can't say, since the model version moves independently of
    the deployed Worker. trained_at is the trainer's own version stamp; key is the
    immutable versioned R2 object it maps to (state/params/v<trained_at>.json), so
    a consumer can pin the exact params without a LIST. Both null means the Worker
    is on BOOTSTRAP params (no params.json published yet), not that identity was
    unavailable."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    trained_at: int | None = None
    key: str | None = None


class Provenance(BaseModel):
    """Which code produced this snapshot. code_sha is the git commit verbatim;
    dirty is null when it couldn't be determined (e.g. a clean-checkout build)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    code_sha: str = "unknown"
    dirty: bool | None = None
    producer: str = "unknown"
    # Identity of the trained params behind the inference. Present on the snapshot
    # (always, even on bootstrap); absent on trains.json, which carries no model.
    params: ParamsProvenance | None = None


class SegmentRecovery(BaseModel):
    """Expected recovery off a dwell curve conditioned on a regime clock — same
    field names as Inference's recovery block, so a segment's recovery is
    directly comparable to a route's."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    recovery_minutes: int
    recovery_minutes_low: int
    recovery_minutes_high: int
    recovery_indeterminate: bool = False
    p_normal_in_30min: float | None
    p_normal_in_60min: float
    p_normal_in_120min: float


class StationServiceFlow(BaseModel):
    """Per-station service flow: is train service advancing through this station, rolled
    up from the segment movement model. Distinct from StationStatus
    (accessibility/alerts)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    # "quiet" when every segment touching the station is quiet-normal: the
    # timetable runs too little here right now for silence to mean anything, so
    # the station is neither provably flowing nor degraded.
    status: Literal["flowing", "quiet", "degraded"]
    # Worst incident segment's live advance shortfall vs its normal (0=normal,
    # 1=frozen), from the decay-smoothed movement signal.
    worst_deficit: float
    # (from_stop, to_stop) of the worst incident segment, or null.
    worst_segment: tuple[str, str] | None = None
    routes: list[str] = Field(default_factory=list)
    n_segments: int
    # Expected recovery of the worst_segment above, off the segment dwell
    # curve. Null when no curve is trained for it or its clock hasn't started
    # — never a fabricated number.
    worst_recovery: SegmentRecovery | None = None


class SegmentStatus(BaseModel):
    """Per-segment movement status: is train service advancing across this
    (route, direction, from_stop) cell, plus its expected recovery. Distinct
    from StationServiceFlow, the station-level roll-up."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    route: str
    direction: str
    from_stop: str
    # Successor stop from segment_params.json's adjacency, or null when the
    # topology doc is unavailable.
    to: str | None = None
    # "quiet" is the quiet-normal call: too little scheduled through this cell
    # right now for an empty window to be evidence of anything.
    status: Literal["normal", "quiet", "disrupted"]
    # When this regime began (the segment clock the recovery is conditioned
    # on). 0 before the clock has ever started.
    entered_at: int
    # Null on a NORMAL cell -- a healthy segment has nothing to forecast, so
    # this is never fabricated for one. Populated on a DISRUPTED cell only
    # once a trained dwell curve exists for it and its regime clock has
    # started (entered_at > 0); see segmentRecovery in the Worker.
    recovery: SegmentRecovery | None = None


class StationFlow(BaseModel):
    """The station-flow surface: per-station verdicts and when they were computed
    (one tick / ~5 min lagged, like the movement condition)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    observed_at: int
    stations: dict[str, StationServiceFlow] = Field(default_factory=dict)


class SegmentFlow(BaseModel):
    """The segment-flow surface: per-segment verdicts and when they were
    computed (one tick / ~5 min lagged, like station_flow). Sibling of
    StationFlow, keyed by the same `route|direction|from_stop` cell id used
    throughout the segment movement model.

    `segments` carries EVERY judged cell, normal and disrupted alike -- a
    key absent from it was never judged this tick, never a healthy read by
    omission. That single-dict membership check is the whole honesty
    property this surface rests on.

    SIZING, recorded so this isn't re-split without new evidence: a
    normal/disrupted split (full `segments` records for disrupted cells, a
    separate bare key -> successor collection for normal ones) shipped and
    was reverted the same day (2026-08-23) it was proposed, sized against a
    policy (decay=0.98, ~1199 judged cells/tick -- see
    worker/src/segment_flow.ts's module docstring) that was measured and
    REJECTED before it shipped. At the policy actually shipped (decay=0.94,
    ~701 judged cells/tick), the union fits: measured on the live feed the
    same day -- base snapshot minus segment_flow 179.9 KB, a bare segment
    record (incl. its key) 151 B, a record carrying a recovery block 382 B.
    A normal cell is always bare (SegmentStatus.recovery is null on healthy
    track), so 701 cells with their usual ~3 disrupted total ~284.0 KB,
    under the 300 KB line. The base (179.9 KB) leaves ~120 KB of budget --
    room for ~814 bare records before 300 KB is even in question, ~16%
    above today's population. Revisit the split only if judged volume
    climbs toward that ceiling, or the disrupted share grows well past
    today's ~0.4%; not before."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    observed_at: int
    segments: dict[str, SegmentStatus] = Field(default_factory=dict)


class TrainPosition(BaseModel):
    """One map dot: every train sharing this (route, direction, stop, stopped)
    tuple, folded into a single aggregated entry by the worker's
    vehicles.ts::trainPositions() from the ~700 concurrent in-service trips.

    `stop` carries NYCT's usual GTFS-RT duality, the same one documented on
    the worker's per-trip trace: it is the stop a train is *heading to*
    while still in transit (stopped=False) and the stop it is *at* once
    STOPPED_AT (stopped=True) — not a fixed "current location" in one sense.

    This surface deliberately does NOT report which segment a moving train
    occupies. Inferring that would mean guessing a direction of travel at
    every branch or express point (e.g. a train signed for a shared trunk
    could still be running local or express past the fork) where stop_id
    alone doesn't disambiguate — an assertion Momentarily isn't willing to
    fabricate. Consumers place the dot at the station for `stop` and use
    `stopped` to distinguish at-platform from approaching.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    route: str  # base route id; 6X/7X/FX fold to 6/7/F, same as route_status keys
    direction: Literal["north", "south"] | None = None
    stop: str  # directional stop_id exactly as the feed reports it
    stopped: bool  # True = at the platform, False = heading toward it
    n: int  # how many trains share this exact tuple


class Trains(BaseModel):
    """The published sibling artifact to Snapshot, served at
    https://feed.momentarily.nyc/v1/trains.json — aggregated live train
    positions for the /map overlay, kept OUT of the snapshot itself.

    At ~700 concurrent trips this would add ~36KB (measured on a realistic
    vehicle set, see the worker's vehicles.test.ts) to every snapshot fetch,
    and the canonical snapshot consumer (homeassistant-mta-subway, polling
    many installs every few minutes) never reads it — charging every
    install that bandwidth forever for a feature only the /map overlay uses
    is the wrong trade, so it is its own fetch instead.

    Self-describing like Snapshot itself: carries its own `observed_at` and
    the same `provenance` block (code_sha/dirty/producer), so a consumer
    holding only this object can still say which build produced it and how
    stale it is. Unlike station_flow/segment_flow this is NOT one-tick
    lagged: built fresh every tick from the same vehicle-position fetch the
    tick already made for movement inference, at zero extra fetch cost, and
    published on the same tick as snapshot.json.

    There is no established schema.py convention for a second published
    root distinct from Snapshot (scripts/export_schema.py hardcodes
    Snapshot as the JSON Schema root), so this model is not wired into the
    generated schema/snapshot.schema.json; it documents the v1/trains.json
    shape directly.

    fresh_feeds/expected_feeds exist because `positions` alone cannot
    distinguish "zero trains right now" from "some NYCT line-group feeds
    failed to decode, so those routes are silently missing" — the worker
    treats a rejected feed as a skip, not an exception, so a partial vehicle
    set is never itself an error. fresh_feeds names which feeds decoded this
    tick (the same convention the worker's archive/vehicles and
    archive/trip_updates objects already use); expected_feeds is the full
    constant set, same order, so a consumer can tell a partial read from a
    complete one without hardcoding NYCT's feed grouping itself.
    fresh_feeds shorter than expected_feeds means `positions` is a PARTIAL
    read. On a tick where NO feed decodes at all (fresh_feeds would be
    empty), the Worker skips publishing entirely rather than writing that
    fabrication — the object is simply left un-rewritten in R2, never a
    failed tick, never a false empty read.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    observed_at: int
    provenance: Provenance = Field(default_factory=Provenance)
    fresh_feeds: list[str] = []
    expected_feeds: list[str] = []
    positions: list[TrainPosition] = []


class PlatformCrowdingEstimate(BaseModel):
    """One platform's estimate. Carries the two inputs alongside the answer so a
    live consumer can re-derive it against its own clock — the crowd grows at
    `entries_per_min` and `waiting_riders` is only correct as of `observed_at`,
    which by publish time is already minutes old."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    # Last time a train was seen at, or seen leaving, this platform.
    last_train_at: int
    # This platform's ASSUMED share of its complex's entry rate for the current
    # (weekday/weekend, hour) cell. See PlatformCrowdingMethod.split_basis.
    entries_per_min: float
    # The estimate at observed_at, rounded to whole riders.
    waiting_riders: int


class PlatformCrowdingMethod(BaseModel):
    """The constants and admitted assumptions behind every estimate in this
    surface. Published rather than documented so a consumer can reproduce the
    arithmetic and see what it does not know."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    # waiting_riders = entries_per_min * (observed_at - last_train_at) / 60.
    formula: str = "entries_per_min * minutes_since_last_train"
    # The ridership feed counts entries per station COMPLEX, not per platform.
    # Splitting a complex's demand across the platforms currently in service is
    # an assumption, not a measurement: nothing in any feed says which platform
    # a rider walked to. "scheduled_service_over_served_platforms" splits it in
    # proportion to each platform's scheduled trains this hour (weighted where
    # the complex is fully covered, even otherwise); "uniform_over_served
    # _platforms" splits evenly and is what publishes when no service_weight
    # baseline is loaded.
    split_basis: Literal[
        "uniform_over_served_platforms",
        "scheduled_service_over_served_platforms",
    ] = "uniform_over_served_platforms"
    # Platforms whose last train is older than this get no estimate at all.
    # Beyond a few headways the linear accumulation stops describing a crowd —
    # people give up and leave, and the platform is usually out of service
    # rather than jammed.
    max_gap_minutes: int
    # A platform shares its complex's demand only if a train passed it within
    # this long. Keeps overnight and out-of-service platforms from absorbing
    # demand that went to the platforms actually running.
    served_window_minutes: int
    # What the entry counts structurally cannot see. Free in-system transfers
    # are never fare-swiped, and the hourly feed has no exits column at all, so
    # a crowd let out by an arriving train is invisible here.
    excludes: list[str] = Field(default_factory=list)
    # Provenance of the baseline these rates came from.
    baseline_generated_at: int
    baseline_window_start: str
    baseline_window_end: str
    # Provenance of the scheduled-service split weights: the service_weight
    # baseline's generated_at and GTFS feed_version, or null when it was absent
    # and the split fell back to uniform.
    service_weight_generated_at: int | None = None
    service_weight_feed_version: str | None = None


class PlatformCrowding(BaseModel):
    """Estimated riders waiting on each platform: the platform's assumed share
    of its complex's usual entry rate for this hour, times how long it has been
    since a train cleared it.

    This is an ESTIMATE built on an admitted assumption (see `method`), not a
    measurement — nothing counts people on platforms. Keys are DIRECTIONAL GTFS
    stop ids ('127N'), so the parent station is the key with its N/S suffix
    stripped and its metadata is in the `stations` surface under that id.

    Platforms that cannot be estimated are absent, not zeroed, and the reason
    is counted in `abstained`.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    observed_at: int
    method: PlatformCrowdingMethod
    platforms: dict[str, PlatformCrowdingEstimate] = Field(default_factory=dict)
    # Platforms carrying an estimate, and those that couldn't, by reason. The
    # honest denominator: a small n_platforms means most of the system is
    # unestimated, not uncrowded.
    n_platforms: int
    abstained: dict[str, int] = Field(default_factory=dict)


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
    # Per-station service flow, rolled up from the segment movement model.
    # Null before the first vehicle tick after deploy or when stale.
    station_flow: StationFlow | None = None
    # Per-segment service flow, the same segment movement model station_flow
    # rolls up. Null before the first vehicle tick after deploy or when stale.
    segment_flow: SegmentFlow | None = None
    # Estimated riders waiting per platform. Null before the ridership baseline
    # is published, before the first vehicle tick after deploy, or when stale.
    platform_crowding: PlatformCrowding | None = None
    system: SystemStatus = Field(default_factory=SystemStatus)

    # Legacy compat — preserves zero-breakage upgrade for HA 0.x consumers
    compat: Compat = Field(default_factory=Compat)
