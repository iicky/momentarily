"""Event contracts (schema ``events/s1``), docs/model-events.md §6.

Layer 3 turns layer-2 observations into facts about trains. Records:

  * ``TripMatch``: one realtime run, the scheduled trip it is (or why none),
    its service day and its observed pattern fit.
  * ``StopEvent``: one occurrence of a run at a platform, as feed-supported
    bounds with the feed's reported time and the service evidence it carries.
  * ``SegmentTraversal``: the run between two stop events of one pattern.
  * ``Headway``: consecutive passings at a stop, at two scopes (route,
    pattern), two evidence levels (service, physical) and, for the pattern
    scope, two ambiguity series (certain, with_ambiguous).
  * ``OpenState``: right-censored state at one instant (an open run, an open
    headway, or scheduled service with no observation yet).
  * ``CoverageInterval``: transport evidence per feed; train counts never
    make an interval uncovered.
  * ``SupplyObservation``: observed trains against schedule per good fetch.

Every time is a bound, never a point guess. ``EventBound`` is ``(after, by]``
in integer UTC seconds, exclusive below: ``after`` null is left-censored.
``DurationBound`` is interval-censored and right-censored at most.

Lineage is carried on every record as ``raw`` or ``legacy``; a run is never
built from both. Legacy records hold ``Unknown`` where the trace kept nothing
(published route, ``start_date``, train ids, listed trips) and name their
feed through the ``RouteFeed`` table (``source_basis = "route_feed_table"``).

Service evidence (Gate B, owner decision 2026-10-22): ``stopped`` on the
"next" family (``numbered``, ``l``), ``reported_planned`` on the "last"
family (the B division and SIR), a weaker statement that the train reached a
stop its trip update plans. Only those two are service evidence, and they
are never folded into each other. ``reported_not_stopped`` and
``inferred_passage`` are physical passage only.

Validation failures raise ``PydanticCustomError`` with a stable error type
(``event_bound``, ``stop_event_dwell``, ...), so callers and tests match on
the type, not the wording. Deferred records and fields (ScheduledTripOutcome,
``suspect_producer_outage``) are not part of the contract.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Literal, LiteralString, get_args

from pydantic import Field, model_validator
from pydantic_core import PydanticCustomError

from momentarily.model.conventions import (
    Instant,
    Interval,
    Model,
    ServiceDate,
    Unknown,
)
from momentarily.model.observations.contracts import GtfsRtSource
from momentarily.model.observations.keys import utc_day

SCHEMA_ID = "events/s1"

Lineage = Literal["raw", "legacy"]
Family = Literal["last", "next"]
SourceBasis = Literal["published", "route_feed_table"]
Direction = Literal["N", "S"]
ServiceEvidence = Literal[
    "stopped", "reported_planned", "reported_not_stopped", "inferred_passage"
]
PatternAttribution = Literal["scheduled_trip", "unique_fit", "ambiguous", "off_pattern"]
Scope = Literal["route", "pattern"]
EvidenceLevel = Literal["service", "physical"]
AmbiguitySet = Literal["certain", "with_ambiguous"]
DepartureBasis = Literal["vehicle", "next_arrival"]

SERVICE_EVIDENCE: tuple[ServiceEvidence, ...] = ("stopped", "reported_planned")
"""The values that show passenger service; the others are passage only."""
FAMILY_SERVICE_EVIDENCE: dict[Family, ServiceEvidence] = {
    "next": "stopped",
    "last": "reported_planned",
}
FAMILY_DEPARTURE_BASIS: dict[Family, DepartureBasis] = {
    "next": "vehicle",
    "last": "next_arrival",
}
LINEAGE_SOURCE_BASIS: dict[Lineage, SourceBasis] = {
    "raw": "published",
    "legacy": "route_feed_table",
}
CERTAIN_ATTRIBUTIONS: tuple[PatternAttribution, ...] = ("scheduled_trip", "unique_fit")
FEED_FAMILY: dict[GtfsRtSource, Family] = {
    "ace": "last",
    "bdfm": "last",
    "g": "last",
    "jz": "last",
    "nqrw": "last",
    "si": "last",
    "numbered": "next",
    "l": "next",
}
"""§3.1: what a vehicle row's stop means on each GTFS-RT feed."""
EVIDENCE_FAMILIES: dict[ServiceEvidence, tuple[Family, ...]] = {
    "stopped": ("next",),
    "reported_planned": ("last",),
    "reported_not_stopped": ("next",),
    "inferred_passage": ("last", "next"),
}
PassingBasis = Literal["departure", "arrival", "passage"]
EVIDENCE_BASIS: dict[ServiceEvidence, PassingBasis] = {
    "stopped": "departure",
    "reported_planned": "arrival",
    "reported_not_stopped": "passage",
    "inferred_passage": "passage",
}
"""Which bound times a passing at a stop: the departure where the feed shows
one, the arrival on the "last" family (§4), else the passage."""


def _fail(error_type: LiteralString, message: str) -> PydanticCustomError:
    """A validation error with a stable type; ``message`` is context, never
    a template."""
    return PydanticCustomError(error_type, "{detail}", {"detail": message})


def _check_flags(flags: list[str], error_type: LiteralString) -> None:
    if flags != sorted(set(flags)):
        raise _fail(error_type, "flags must be sorted and distinct")


def _check_source_basis(lineage: Lineage, basis: SourceBasis) -> None:
    if basis != LINEAGE_SOURCE_BASIS[lineage]:
        raise _fail(
            "source_basis",
            f"{lineage} records name their feed as {LINEAGE_SOURCE_BASIS[lineage]}",
        )


def _check_evidence(source: GtfsRtSource, evidence: ServiceEvidence) -> None:
    family = FEED_FAMILY[source]
    if family not in EVIDENCE_FAMILIES[evidence]:
        raise _fail(
            "service_evidence",
            f"{evidence} is not evidence of the {family} family ({source})",
        )


def _check_run_ids(
    lineage: Lineage, run_ids: list[str | None], source: GtfsRtSource
) -> None:
    """Every run id names the record's lineage and feed (§2.3: the run id
    starts ``{lineage}:{source}:``)."""
    prefix = f"{lineage}:{source}:"
    for run_id in run_ids:
        if run_id is not None and not run_id.startswith(prefix):
            raise _fail("run_lineage", f"{run_id} is not a {prefix} run")


def _check_family_run_id(
    lineage: Lineage, run_id: str | None, source: GtfsRtSource
) -> None:
    """A run id names the record's lineage and a feed of ``source``'s family:
    a headway leader may come from another feed of the route (raw ``ace``
    publishes some F trips), never from another family."""
    if run_id is None:
        return
    lineage_part, feed, *_ = [*run_id.split(":", 2), "", ""]
    family = next((f for name, f in FEED_FAMILY.items() if name == feed), None)
    if lineage_part != lineage or family != FEED_FAMILY[source]:
        raise _fail(
            "run_lineage",
            f"{run_id} is not a {lineage} run of a {FEED_FAMILY[source]}-family feed",
        )


def _overlap(a0: int, a1: int, b0: int, b1: int) -> int:
    """Seconds shared by ``(a0, a1]`` and ``(b0, b1]``."""
    return max(0, min(a1, b1) - max(a0, b0))


def _check_bound_lineage(lineage: Lineage, bounds: list[EventBound | None]) -> None:
    """suspect_stale is a legacy state: raw bounds carry no suspect_s."""
    for bound in bounds:
        if bound is None:
            continue
        expect_null = lineage == "raw" or bound.after is None
        if (bound.suspect_s is None) != expect_null:
            raise _fail(
                "event_bound_overlap",
                "suspect_s is set exactly on legacy bounds with an after",
            )


def _check_between(
    duration: DurationBound,
    start: EventBound,
    end: EventBound,
    error_type: LiteralString,
) -> None:
    """A duration from ``start`` to ``end`` follows from the two bounds (§3.2):
    ``lo = max(0, end.after - start.by)``, ``hi = end.by - start.after`` (open
    when the start is left-censored), and the reported difference when both
    reported times are kept.

    Its span is ``(start.after, end.by]``, ``hi`` seconds wide. Every second
    of either bound inside the span is a second of the span, so the span's
    ``uncovered_s`` is at least what the two bounds certainly hold there:
    each bound's ``uncovered_s`` less its seconds outside the span, ``s`` and
    ``e``, give ``max(s, e, s + e - overlap)`` with ``overlap`` the seconds the
    two bounds share. (``uncovered_s <= hi`` is ``DurationBound``'s.)"""
    lo = 0 if end.after is None else max(0, end.after - start.by)
    hi = None if start.after is None else end.by - start.after
    reported = (
        None
        if start.reported is None or end.reported is None
        else end.reported - start.reported
    )
    if (duration.lo, duration.hi, duration.reported) != (lo, hi, reported):
        raise _fail(error_type, "the duration does not follow from its bounds")
    if start.after is None or duration.uncovered_s is None:
        return
    span = (start.after, end.by)
    inside: list[int] = []
    for bound in (start, end):
        if bound.after is None or bound.uncovered_s is None:
            inside.append(0)
            continue
        outside = bound.by - bound.after - _overlap(bound.after, bound.by, *span)
        inside.append(max(0, bound.uncovered_s - outside))
    s, e = inside
    shared = (
        0 if end.after is None else _overlap(start.after, start.by, end.after, end.by)
    )
    least = max(s, e, s + e - shared)
    if duration.uncovered_s < least:
        raise _fail(
            error_type,
            f"uncovered_s {duration.uncovered_s} is below the {least} s its bounds hold",
        )


def _check_pattern_fit(
    attribution: PatternAttribution, pattern_id: str | None, compatible: list[str]
) -> None:
    """A matched trip fixes the pattern (which may lie outside the fit); else
    the observed fit decides: one pattern, several, or none."""
    if compatible != sorted(set(compatible)):
        raise _fail("pattern_fit", "compatible_pattern_ids must be sorted and distinct")
    if attribution == "scheduled_trip":
        ok = pattern_id is not None
    elif attribution == "unique_fit":
        ok = len(compatible) == 1 and pattern_id == compatible[0]
    elif attribution == "ambiguous":
        ok = len(compatible) >= 2 and pattern_id is None
    else:
        ok = not compatible and pattern_id is None
    if not ok:
        raise _fail(
            "pattern_fit", f"{attribution} disagrees with pattern_id and the fit"
        )


# --- bounds -------------------------------------------------------------------


class EventBound(Model):
    """Feed-supported bound: the event happened in ``(after, by]``, integer
    UTC seconds, exclusive below. ``after`` null is left-censored. ``after <
    by`` strictly. A kept ``reported`` satisfies ``after < reported <= by``
    (an earlier report widens ``after`` to ``reported - 1``, §3.2)."""

    after: Instant | None = Field(description="Exclusive; null = left-censored.")
    by: Instant = Field(description="Inclusive.")
    reported: Instant | None = Field(
        description="The feed's own time, kept only inside (after, by]."
    )
    uncovered_s: int | None = Field(
        ge=0,
        description="Seconds of (after, by] not covered; null when after is null.",
    )
    suspect_s: int | None = Field(
        ge=0,
        description="Seconds of (after, by] in legacy suspect_stale; null when after is null and on raw records.",
    )

    @model_validator(mode="after")
    def _consistent(self) -> EventBound:
        after, by = self.after, self.by
        if after is not None and not after < by:
            raise _fail("event_bound", "after must precede by")
        reported = self.reported
        if reported is not None and (
            reported > by or (after is not None and reported <= after)
        ):
            raise _fail("event_bound_reported", "reported lies outside (after, by]")
        if (self.uncovered_s is None) != (after is None):
            raise _fail(
                "event_bound_overlap", "uncovered_s is null exactly when after is"
            )
        if self.suspect_s is not None and after is None:
            raise _fail("event_bound_overlap", "suspect_s is null when after is")
        if (
            after is not None
            and (self.uncovered_s or 0) + (self.suspect_s or 0) > by - after
        ):
            # suspect_stale is its own coverage state, never an uncovered one,
            # so the two never share a second.
            raise _fail(
                "event_bound_overlap",
                "uncovered_s and suspect_s together exceed the bound's width",
            )
        return self


def bound_after(observed: Iterable[Instant], by: Instant) -> Instant | None:
    """``after`` for a bound closed at ``by`` (§3.2 tie rule): the latest
    observation instant strictly before ``by``, None (left-censored) when
    there is none. Two fetches of a feed can finish in one integer second
    with different content, so the fetch that did not show the event can
    share ``by``'s second. It then counts as at ``by``, and ``after`` falls
    back to the previous distinct instant: the bound only widens, and
    ``after < by`` stays strict."""
    return max((t for t in observed if t < by), default=None)


class DurationBound(Model):
    """Interval-censored duration in seconds; consumers fit distributions
    across events, never per-event midpoints. Nothing in layer 3 is
    left-censored: a left-censored start or an open end gives ``censoring =
    "right"`` with ``hi`` null."""

    lo: int = Field(ge=0, description="Clamped at 0.")
    hi: int | None = Field(description="null exactly when censoring is right.")
    censoring: Literal["none", "right"]
    reported: int | None = Field(
        description="Difference of the two events' reported times, when both are kept."
    )
    uncovered_s: int | None = Field(
        ge=0,
        description="Seconds of the span not covered; null exactly when right-censored.",
    )

    @model_validator(mode="after")
    def _consistent(self) -> DurationBound:
        if (self.hi is None) != (self.censoring == "right"):
            raise _fail(
                "duration_censoring", "hi is null exactly when censoring is right"
            )
        if (self.uncovered_s is None) != (self.censoring == "right"):
            raise _fail(
                "duration_censoring", "uncovered_s is null exactly when right-censored"
            )
        if self.hi is not None and self.hi < self.lo:
            raise _fail("duration_bound", "hi precedes lo")
        if (
            self.hi is not None
            and self.uncovered_s is not None
            and self.uncovered_s > self.hi
        ):
            raise _fail("duration_bound", "uncovered_s exceeds the span")
        return self


class GapInterval(Model):
    """Current elapsed gap at an evaluation instant; integer seconds, both
    ends inclusive. ``hi`` null only when the last passing is left-censored,
    or for ``scheduled_no_observation``. ``OpenState`` fixes both ends (§4):
    exactly ``[as_of - by, as_of - after - 1]`` with no outage."""

    lo: int = Field(ge=0)
    hi: int | None

    @model_validator(mode="after")
    def _consistent(self) -> GapInterval:
        if self.hi is not None and self.hi < self.lo:
            raise _fail("gap_interval", "hi precedes lo")
        return self


# --- runs -----------------------------------------------------------------------

TripMatchFlag = Literal[
    "deduplicated_by_position",
    "duplicate_trip_id",
    "joined_split_run",
    "service_date_ambiguous",
    "start_date_mismatch",
]
OUTCOME_SERVICE_DATE_BASIS: dict[str, str] = {
    "matched": "scheduled_trip",
    "unscheduled": "id_origin",
    "unparseable": "clock_rule",
}


class TripMatch(Model):
    """One realtime run and what it is (§2). Scheduled-trip identity and the
    observed pattern fit are separate results: ``compatible_pattern_ids`` is
    fitted from observations alone."""

    run_id: str = Field(
        description="f'{lineage}:{source}:{realtime_trip_id}@{first_observed_at}#{track}' (§2.3)."
    )
    track: int = Field(
        ge=0, description="0 for an unsplit run; else the track ordinal."
    )
    member_run_ids: list[str] = Field(
        description="Runs joined into this one (split-run join); [] when not joined."
    )
    supersedes_run_ids: list[str] = Field(
        description="Provisional live ids this nightly run absorbed."
    )
    lineage: Lineage
    source: GtfsRtSource
    source_basis: SourceBasis
    realtime_trip_id: str
    route_published: str | None | Unknown = Field(
        description="As published ('6X', 'SS'); Unknown on legacy runs."
    )
    start_date_published: str | None | Unknown = Field(
        description="Published 'YYYYMMDD', not a ServiceDate; Unknown on legacy runs."
    )
    nyct_train_ids: list[str] | Unknown = Field(
        description="Distinct NYCT train ids seen, sorted; Unknown on legacy runs."
    )
    first_observed_at: Instant
    last_observed_at: Instant
    n_observations: int = Field(ge=1)
    n_placeholder_rows: int = Field(ge=0)
    n_old_timestamp_rows: int = Field(ge=0)
    stale_only: bool = Field(description="Every row has an old vehicle_timestamp.")
    outcome: Literal["matched", "unscheduled", "unparseable"]
    stationary: bool = Field(description="Seen at one stop only; no traversal.")
    flags: list[TripMatchFlag] = Field(description="Sorted and distinct.")
    tier: (
        Literal["exact_id", "path_prefix", "line_code", "path_differs", "drift"] | None
    )
    miss_class: str | None = Field(
        description="Why no trip matched; null when matched."
    )
    network_build_id: str
    scheduled_trip_id: str | None
    nearest_scheduled_trip_id: str | None = Field(
        description="For an unmatched run: the trip it nearly matched."
    )
    compatible_pattern_ids: list[str] = Field(
        description="Patterns whose stop order contains the observed stops; sorted."
    )
    pattern_id: str | None = Field(
        description="The matched trip's pattern, or the unique fit."
    )
    pattern_attribution: PatternAttribution
    shares_scheduled_trip: bool = Field(
        description="Another id-tier run holds the same scheduled trip."
    )
    identity_uncertain_with: list[str] = Field(
        description="Runs with identical positions but a different train_id (§2.3)."
    )
    route_id: str | None = Field(
        description="Network route of the matched trip, else the route_basis route."
    )
    route_basis: Literal["scheduled_trip", "published", "trip_id_field"]
    service_date: ServiceDate
    service_date_basis: Literal["scheduled_trip", "id_origin", "clock_rule"]
    alternative_service_date: ServiceDate | None = Field(
        description="Set exactly when flagged service_date_ambiguous."
    )
    origin_wrapped: bool = Field(
        description="Matched only after adding 24 h to the origin."
    )
    offset_s: int | None = Field(
        description="Observed minus scheduled at the first common stop."
    )
    deviation: (
        Literal[
            "consistent",
            "skips_only",
            "extras_only",
            "skips_and_extras",
            "off_pattern",
            "order_violation",
        ]
        | None
    )
    end_state: Literal["left_feed", "stale_tail", "window_end"]

    @model_validator(mode="after")
    def _consistent(self) -> TripMatch:
        expected = (
            f"{self.lineage}:{self.source}:{self.realtime_trip_id}"
            f"@{self.first_observed_at}#{self.track}"
        )
        if self.run_id != expected:
            raise _fail("trip_match_identity", "run_id does not follow §2.3")
        if self.last_observed_at < self.first_observed_at:
            raise _fail(
                "trip_match_identity", "last_observed_at precedes first_observed_at"
            )
        _check_source_basis(self.lineage, self.source_basis)
        _check_flags(list(self.flags), "trip_match_flags")
        if ("joined_split_run" in self.flags) != bool(self.member_run_ids):
            raise _fail("trip_match_flags", "joined_split_run marks a run with members")
        self._check_identity()
        self._check_lineage()
        self._check_outcome()
        self._check_service_date()
        self._check_pattern()
        return self

    def _check_identity(self) -> None:
        own = self.run_id
        _check_run_ids(
            self.lineage,
            [*self.member_run_ids, *self.supersedes_run_ids],
            self.source,
        )
        if own in self.member_run_ids or own in self.identity_uncertain_with:
            raise _fail("trip_match_identity", "a run does not list itself")
        if self.identity_uncertain_with and self.lineage != "raw":
            raise _fail(
                "trip_match_identity",
                "identity uncertainty needs train_ids (raw runs only)",
            )
        same_poll = own.rsplit("#", 1)[0] + "#"
        for other in self.identity_uncertain_with:
            if not other.startswith(same_poll):
                raise _fail(
                    "trip_match_identity",
                    "identity_uncertain_with names other tracks of this run's trip id and first poll",
                )

    def _check_lineage(self) -> None:
        published = (
            self.route_published,
            self.start_date_published,
            self.nyct_train_ids,
        )
        unknown = [isinstance(v, Unknown) for v in published]
        if self.lineage == "raw" and any(unknown):
            raise _fail("lineage_unknown", "a raw run knows its published descriptor")
        if self.lineage == "legacy" and not all(unknown):
            raise _fail("lineage_unknown", "the trace kept no published descriptor")
        if self.lineage == "legacy" and self.route_basis == "published":
            raise _fail("lineage_unknown", "a legacy run has no published route")
        if isinstance(self.nyct_train_ids, list) and self.nyct_train_ids != sorted(
            set(self.nyct_train_ids)
        ):
            raise _fail(
                "trip_match_identity", "nyct_train_ids must be sorted and distinct"
            )

    def _check_outcome(self) -> None:
        matched = self.outcome == "matched"
        if (self.scheduled_trip_id is not None) != matched:
            raise _fail(
                "trip_match_outcome", "a scheduled trip is set exactly when matched"
            )
        if (self.tier is not None) != matched:
            raise _fail("trip_match_outcome", "a tier is set exactly when matched")
        if (self.miss_class is None) != matched:
            raise _fail(
                "trip_match_outcome", "a miss class is set exactly when unmatched"
            )
        if matched and self.nearest_scheduled_trip_id is not None:
            raise _fail("trip_match_outcome", "a matched run has no nearest candidate")
        if self.origin_wrapped and not matched:
            raise _fail("trip_match_outcome", "only a matched origin can wrap")
        if (self.route_basis == "scheduled_trip") != matched:
            raise _fail(
                "trip_match_outcome", "route_basis scheduled_trip means matched"
            )
        if matched and self.route_id is None:
            raise _fail("trip_match_outcome", "a matched run has its trip's route")

    def _check_service_date(self) -> None:
        if self.service_date_basis != OUTCOME_SERVICE_DATE_BASIS[self.outcome]:
            raise _fail(
                "service_date_basis",
                f"a {self.outcome} run dates by {OUTCOME_SERVICE_DATE_BASIS[self.outcome]}",
            )
        ambiguous = "service_date_ambiguous" in self.flags
        if ambiguous and self.outcome == "matched":
            raise _fail("service_date_basis", "a matched trip settles the service date")
        if (self.alternative_service_date is not None) != ambiguous:
            raise _fail(
                "service_date_basis",
                "alternative_service_date is set exactly when service_date_ambiguous",
            )
        if self.alternative_service_date == self.service_date:
            raise _fail("service_date_basis", "the alternative date is another date")

    def _check_pattern(self) -> None:
        if (self.pattern_attribution == "scheduled_trip") != (
            self.outcome == "matched"
        ):
            raise _fail("pattern_fit", "a matched trip fixes the pattern")
        _check_pattern_fit(
            self.pattern_attribution, self.pattern_id, self.compatible_pattern_ids
        )


# --- stop events and traversals --------------------------------------------------

StopEventFlag = Literal[
    "first_of_run",
    "held_old_timestamp",
    "liveness_uncertain",
    "order_violation",
    "plan_dropped",
    "reported_after_bound",
    "reported_widened_after",
]


class StopEvent(Model):
    """One occurrence of a run at a platform (§3). A service event
    (``stopped``, ``reported_planned``) has an arrival and a departure; a
    passage (``reported_not_stopped``, ``inferred_passage``) has only a
    passage bound. "last"-family departures are bounded only by the next
    arrival, and their dwell is ``Unknown`` (decision 4)."""

    run_id: str
    seq: int = Field(ge=0, description="Occurrence of this event within the run.")
    service_date: ServiceDate
    lineage: Lineage
    source: GtfsRtSource
    source_basis: SourceBasis
    family: Family
    stop_id: str = Field(description="Platform.")
    pattern_id: str | None = Field(description="null when ambiguous or off pattern.")
    pattern_index: int | None = Field(
        ge=0, description="null when the stop is off pattern or the run ambiguous."
    )
    pattern_attribution: PatternAttribution
    service_evidence: ServiceEvidence
    arrival: EventBound | None = Field(description="Service evidence only.")
    departure: EventBound | None | Unknown = Field(
        description="Service evidence only; Unknown: censored tail or run end."
    )
    passage: EventBound | None = Field(
        description="reported_not_stopped and inferred_passage only."
    )
    departure_basis: DepartureBasis | None = Field(
        description="vehicle ('next') or next_arrival ('last'); service evidence only."
    )
    dwell: DurationBound | Unknown | None = Field(
        description="Unknown for the 'last' family and for an Unknown departure."
    )
    flags: list[StopEventFlag] = Field(
        description="Sorted and distinct. plan_dropped (basis trip_update): the trip update removed this listed stop more than 60 s before its predicted arrival."
    )

    @model_validator(mode="after")
    def _consistent(self) -> StopEvent:
        _check_source_basis(self.lineage, self.source_basis)
        _check_flags(list(self.flags), "stop_event_flags")
        _check_run_ids(self.lineage, [self.run_id], self.source)
        if self.family != FEED_FAMILY[self.source]:
            raise _fail(
                "service_evidence",
                f"{self.source} is a {FEED_FAMILY[self.source]} feed",
            )
        _check_evidence(self.source, self.service_evidence)
        departure = self.departure if isinstance(self.departure, EventBound) else None
        _check_bound_lineage(self.lineage, [self.arrival, departure, self.passage])
        if self.pattern_attribution in ("ambiguous", "off_pattern"):
            if self.pattern_id is not None or self.pattern_index is not None:
                raise _fail("pattern_fit", "an unattributed event has no pattern")
        elif self.pattern_id is None:
            raise _fail("pattern_fit", "an attributed event names its pattern")
        if self.service_evidence in SERVICE_EVIDENCE:
            self._check_service()
        else:
            self._check_passage()
        first_of_run = self.arrival is not None and self.arrival.after is None
        if ("first_of_run" in self.flags) != first_of_run:
            raise _fail(
                "stop_event_flags", "first_of_run marks exactly a left-censored arrival"
            )
        if "liveness_uncertain" in self.flags and not isinstance(
            self.departure, Unknown
        ):
            raise _fail("stop_event_flags", "a censored tail has an Unknown departure")
        if "plan_dropped" in self.flags and self.lineage != "raw":
            raise _fail("stop_event_flags", "plan_dropped needs trip updates (raw)")
        self._check_reported_flags()
        return self

    def _check_reported_flags(self) -> None:
        """The §3.2 reported-time flags describe the event's first bound."""
        bound = self.arrival if self.arrival is not None else self.passage
        widened = "reported_widened_after" in self.flags
        dropped = "reported_after_bound" in self.flags
        if bound is None or not (widened or dropped):
            return
        if widened and dropped:
            raise _fail("stop_event_flags", "a reported time is widened or dropped")
        if dropped and bound.reported is not None:
            raise _fail("stop_event_flags", "a reported time after by is not kept")
        if widened and (
            bound.after is None
            or bound.reported is None
            or bound.reported != bound.after + 1
        ):
            raise _fail("stop_event_flags", "a widened bound has after = reported - 1")

    def _check_service(self) -> None:
        if self.arrival is None or self.departure is None or self.passage is not None:
            raise _fail(
                "stop_event_bounds",
                "a service event has an arrival and a departure, no passage",
            )
        if self.departure_basis != FAMILY_DEPARTURE_BASIS[self.family]:
            raise _fail(
                "stop_event_bounds",
                f"a {self.family} departure is by {FAMILY_DEPARTURE_BASIS[self.family]}",
            )
        if self.dwell is None:
            raise _fail("stop_event_dwell", "a service event has a dwell or Unknown")
        if self.family == "last" and not isinstance(self.dwell, Unknown):
            raise _fail("stop_event_dwell", "'last'-family dwell is not observable")
        if isinstance(self.departure, Unknown) and not isinstance(self.dwell, Unknown):
            raise _fail(
                "stop_event_dwell", "an Unknown departure gives an Unknown dwell"
            )
        if isinstance(self.departure, EventBound):
            if self.family == "last" and self.departure.after != self.arrival.after:
                raise _fail(
                    "stop_event_bounds",
                    "a next-arrival departure starts where the arrival does",
                )
            if isinstance(self.dwell, DurationBound):
                _check_between(
                    self.dwell, self.arrival, self.departure, "stop_event_dwell"
                )

    def _check_passage(self) -> None:
        if self.passage is None or any(
            v is not None
            for v in (self.arrival, self.departure, self.departure_basis, self.dwell)
        ):
            raise _fail(
                "stop_event_bounds",
                "a passage has a passage bound only: no arrival, departure or dwell",
            )
        if self.passage.after is None:
            raise _fail("stop_event_bounds", "a passage lies between two observations")


class SegmentTraversal(Model):
    """A run between two of its stop events on one pattern. ``run_time`` is
    departure(from) to arrival(to); ``stop_to_stop`` is arrival to arrival,
    dwell included, comparable with the old ``archive/traversals``."""

    run_id: str
    from_seq: int = Field(ge=0, description="StopEvent.seq, not a pattern index.")
    to_seq: int = Field(ge=0)
    service_date: ServiceDate
    lineage: Lineage
    source: GtfsRtSource
    pattern_id: str
    from_index: int = Field(ge=0)
    to_index: int = Field(
        ge=0, description="> from_index; > from_index + 1 only across unreported stops."
    )
    segment_ids: list[str] = Field(description="Layer-1 segments covered, in order.")
    hidden_realtime_stops: int = Field(ge=0)
    from_event: EventBound = Field(
        description="Departure ('next') or arrival ('last') at the from stop."
    )
    to_arrival: EventBound
    run_time: DurationBound | Unknown = Field(
        description="Unknown for the 'last' family."
    )
    stop_to_stop: DurationBound
    run_time_basis: DepartureBasis

    @model_validator(mode="after")
    def _consistent(self) -> SegmentTraversal:
        _check_run_ids(self.lineage, [self.run_id], self.source)
        _check_bound_lineage(self.lineage, [self.from_event, self.to_arrival])
        if not self.from_seq < self.to_seq:
            raise _fail("traversal_order", "to_seq must follow from_seq")
        if not self.from_index < self.to_index:
            raise _fail("traversal_order", "to_index must follow from_index")
        if len(self.segment_ids) != self.to_index - self.from_index:
            raise _fail("traversal_order", "one segment per pattern hop")
        basis = FAMILY_DEPARTURE_BASIS[FEED_FAMILY[self.source]]
        if self.run_time_basis != basis:
            raise _fail("traversal_run_time", f"a {self.source} run time is by {basis}")
        if isinstance(self.run_time, Unknown) != (
            self.run_time_basis == "next_arrival"
        ):
            raise _fail(
                "traversal_run_time",
                "run time is Unknown exactly when bounded by the next arrival",
            )
        if isinstance(self.run_time, DurationBound):
            _check_between(
                self.run_time, self.from_event, self.to_arrival, "traversal_run_time"
            )
        else:
            # "last": from_event is the arrival, so stop-to-stop is exact.
            _check_between(
                self.stop_to_stop,
                self.from_event,
                self.to_arrival,
                "traversal_run_time",
            )
        return self


# --- headways and open state ------------------------------------------------------


def _check_scope(
    scope: Scope | None,
    pattern_id: str | None,
    ambiguity_set: AmbiguitySet | None,
    attributions: list[PatternAttribution],
) -> None:
    pattern = scope == "pattern"
    if (pattern_id is not None) != pattern or (ambiguity_set is not None) != pattern:
        raise _fail(
            "headway_scope",
            "pattern_id and ambiguity_set are set exactly at pattern scope",
        )
    if not pattern:
        return
    allowed = (
        CERTAIN_ATTRIBUTIONS
        if ambiguity_set == "certain"
        else (*CERTAIN_ATTRIBUTIONS, "ambiguous")
    )
    if any(a not in allowed for a in attributions):
        raise _fail(
            "headway_scope", f"the {ambiguity_set} series admits only {list(allowed)}"
        )


def _check_service_level(
    evidence: EvidenceLevel | None, ends: list[ServiceEvidence]
) -> None:
    if evidence != "service":
        return
    if any(e not in SERVICE_EVIDENCE for e in ends):
        raise _fail(
            "headway_evidence",
            "a service-level series is built from stopped or reported_planned only",
        )
    if len(set(ends)) > 1:
        raise _fail(
            "headway_evidence", "stopped and reported_planned are never folded together"
        )


class Headway(Model):
    """Consecutive passings at a stop (§4), owned by the follower's service
    date. Adjacency is computed per scope and evidence level, never filtered
    from another series. ``leader_unknown``: no leader in the window and no
    valid checkpoint, so every leader field is null and ``headway.hi`` too."""

    service_date: ServiceDate = Field(description="The follower's.")
    leader_service_date: ServiceDate | None
    leader_unknown: bool
    lineage: Lineage
    source: GtfsRtSource = Field(
        description="The follower's feed; every feed of a route has one family."
    )
    stop_id: str
    route_id: str
    direction: Direction
    scope: Scope
    pattern_id: str | None = Field(description="Set exactly at pattern scope.")
    evidence: EvidenceLevel = Field(
        description="service: stopped or reported_planned ends only, never mixed."
    )
    leader_run_id: str | None
    follower_run_id: str
    leader_pattern_attribution: PatternAttribution | None
    follower_pattern_attribution: PatternAttribution
    leader_seq: int | None = Field(ge=0)
    follower_seq: int = Field(ge=0, description="Revisits of a stop differ here.")
    ambiguity_set: AmbiguitySet | None = Field(
        description="Set exactly at pattern scope."
    )
    leader_passing: EventBound | None
    follower_passing: EventBound
    leader_service_evidence: ServiceEvidence | None
    follower_service_evidence: ServiceEvidence
    basis: PassingBasis = Field(
        description="The follower's timed end: departure (stopped), arrival (reported_planned), else passage."
    )
    headway: DurationBound
    spans_no_service: bool = Field(
        description="A scheduled closure lies inside the span."
    )

    @model_validator(mode="after")
    def _consistent(self) -> Headway:
        _check_run_ids(self.lineage, [self.follower_run_id], self.source)
        _check_family_run_id(self.lineage, self.leader_run_id, self.source)
        _check_bound_lineage(self.lineage, [self.leader_passing, self.follower_passing])
        leader = (
            self.leader_service_date,
            self.leader_run_id,
            self.leader_pattern_attribution,
            self.leader_seq,
            self.leader_passing,
            self.leader_service_evidence,
        )
        if self.leader_unknown:
            if any(v is not None for v in leader) or self.headway.hi is not None:
                raise _fail(
                    "headway_leader",
                    "an unknown leader has null leader fields and no upper bound",
                )
        elif any(v is None for v in leader):
            raise _fail("headway_leader", "a known leader has every leader field")
        elif self.leader_passing is not None:
            _check_between(
                self.headway,
                self.leader_passing,
                self.follower_passing,
                "headway_bound",
            )
        attributions: list[PatternAttribution] = [self.follower_pattern_attribution]
        ends: list[ServiceEvidence] = [self.follower_service_evidence]
        if self.leader_pattern_attribution is not None:
            attributions.append(self.leader_pattern_attribution)
        if self.leader_service_evidence is not None:
            ends.append(self.leader_service_evidence)
        if self.scope == "pattern" and "off_pattern" in attributions:
            raise _fail(
                "headway_scope", "an off-pattern run counts at route scope only"
            )
        _check_scope(self.scope, self.pattern_id, self.ambiguity_set, attributions)
        for end in ends:
            _check_evidence(self.source, end)
        _check_service_level(self.evidence, ends)
        expected = EVIDENCE_BASIS[self.follower_service_evidence]
        if self.basis != expected:
            raise _fail(
                "headway_evidence",
                f"{self.follower_service_evidence} passings are timed by {expected}",
            )
        return self


OpenStateFlag = Literal["ambiguous_patterns_only", "possibly_stale", "uncovered_since"]


class OpenState(Model):
    """Right-censored state at one instant (the live store, §7; also the
    window-end tail and the next day's checkpoint of a nightly build). Never
    a closed record. ``current_gap`` is finite at ``as_of``; ``eventual`` is
    the still-open headway, right-censored from ``current_gap.lo``.

    ``current_gap`` is exact (§4): with the last passing in ``(after, by]``,
    ``lo = max(0, as_of - max(by, e))`` and ``hi = as_of - after - 1``, where
    ``e`` is ``uncovered_until`` (an uncovered interval of the feed ended
    after the last passing, so a train may have passed unseen) and drops
    out when null. ``scheduled_no_observation`` reads ``since`` for ``by``
    and has no ``hi``. ``uncovered_until`` is set exactly when the flag
    ``uncovered_since`` is; ``as_of`` while the feed is still uncovered,
    which gives ``lo = 0``."""

    as_of: Instant
    lineage: Lineage
    source: GtfsRtSource = Field(
        description="The run's feed (open_run) or the route's feed (clocks)."
    )
    kind: Literal["open_run", "open_headway", "scheduled_no_observation"]
    run_id: str | None = Field(description="open_run only.")
    last_event_seq: int | None = Field(
        ge=0, description="open_run only: StopEvent.seq of the last event."
    )
    pattern_attribution: PatternAttribution | None = Field(
        description="open_run only; scheduled_trip when matched (tiers 1-4 live)."
    )
    compatible_pattern_ids: list[str] = Field(
        description="open_run: the live pattern fit, sorted, which may exclude a matched pattern; [] otherwise."
    )
    last_event: EventBound | None = Field(
        description="null only for scheduled_no_observation."
    )
    last_event_evidence: ServiceEvidence | None = Field(
        description="Service evidence of last_event; null exactly when it is."
    )
    since: Instant = Field(
        description="Start of the open interval (§4 initialization)."
    )
    uncovered_until: Instant | None = Field(
        description="End of the last uncovered interval of the feed after the last passing (after since with no passing), as_of while still uncovered; null when none. Set exactly with the uncovered_since flag."
    )
    stop_id: str
    route_id: str
    direction: Direction
    scope: Scope | None = Field(description="null for open_run.")
    pattern_id: str | None = Field(
        description="open_run: the matched pattern or the unique fit; clocks: set exactly at pattern scope."
    )
    evidence: EvidenceLevel | None = Field(description="null for open_run.")
    ambiguity_set: AmbiguitySet | None = Field(
        description="Pattern scope: one clock per series."
    )
    current_gap: GapInterval = Field(
        description="Exactly [as_of - max(by, uncovered_until), as_of - after - 1], lo clamped at 0 (§4); since for by with no observation."
    )
    eventual: DurationBound = Field(
        description="The still-open headway: censoring right, hi null, lo = current_gap.lo."
    )
    flags: list[OpenStateFlag] = Field(description="Sorted and distinct.")

    @model_validator(mode="after")
    def _consistent(self) -> OpenState:
        _check_flags(list(self.flags), "open_state_flags")
        _check_run_ids(self.lineage, [self.run_id], self.source)
        _check_bound_lineage(self.lineage, [self.last_event])
        if self.last_event_evidence is not None:
            _check_evidence(self.source, self.last_event_evidence)
        if self.since > self.as_of:
            raise _fail("open_state_time", "since is after as_of")
        self._check_kind()
        last = self.last_event
        if (last is None) != (self.last_event_evidence is None):
            raise _fail("open_state_kind", "last_event_evidence goes with last_event")
        if self.scope is not None:
            _check_scope(self.scope, self.pattern_id, self.ambiguity_set, [])
            ends: list[ServiceEvidence] = (
                [self.last_event_evidence] if self.last_event_evidence else []
            )
            _check_service_level(self.evidence, ends)
        gap = self.current_gap
        until = self.uncovered_until
        if (until is not None) != ("uncovered_since" in self.flags):
            raise _fail(
                "open_state_gap",
                "uncovered_until is set exactly when uncovered_since is",
            )
        if until is not None and until > self.as_of:
            raise _fail("open_state_time", "uncovered_until is after as_of")
        if last is not None:
            if last.by > self.as_of:
                raise _fail("open_state_time", "the last event is after as_of")
            if until is not None and last.after is not None and until <= last.after:
                raise _fail(
                    "open_state_gap", "uncovered_until ends after the last passing"
                )
            hi = None if last.after is None else self.as_of - last.after - 1
            seen = last.by
        else:
            if until is not None and until <= self.since:
                raise _fail("open_state_gap", "uncovered_until ends after since")
            hi, seen = None, self.since
        lo = max(0, self.as_of - max(seen, until if until is not None else seen))
        if (gap.lo, gap.hi) != (lo, hi):
            raise _fail(
                "open_state_gap",
                f"current_gap must be [{lo}, {hi}] at as_of {self.as_of}",
            )
        eventual = self.eventual
        if eventual.censoring != "right" or eventual.lo != gap.lo:
            raise _fail(
                "open_state_gap", "eventual is right-censored from current_gap.lo"
            )
        return self

    def _check_kind(self) -> None:
        run = self.kind == "open_run"
        if (self.run_id is not None) != run:
            raise _fail("open_state_kind", "run_id is set exactly for open_run")
        if run:
            if self.scope is not None or self.evidence is not None:
                raise _fail("open_state_kind", "an open run has no headway scope")
            if self.ambiguity_set is not None:
                raise _fail("open_state_kind", "an open run has no headway scope")
            if self.pattern_attribution is None or self.last_event_seq is None:
                raise _fail(
                    "open_state_kind", "an open run has its attribution and last seq"
                )
            _check_pattern_fit(
                self.pattern_attribution, self.pattern_id, self.compatible_pattern_ids
            )
        else:
            if self.scope is None or self.evidence is None:
                raise _fail("open_state_kind", "a headway clock has a scope and level")
            if self.compatible_pattern_ids:
                raise _fail("open_state_kind", "only an open run carries a pattern fit")
            if self.pattern_attribution is not None or self.last_event_seq is not None:
                raise _fail(
                    "open_state_kind", "only an open run carries its run fields"
                )
        if (self.last_event is None) != (self.kind == "scheduled_no_observation"):
            raise _fail(
                "open_state_kind",
                "last_event is null only for scheduled_no_observation",
            )


# --- coverage and supply ----------------------------------------------------------

CoverageState = Literal[
    "covered",
    "fetch_failed",
    "no_invocation",
    "not_stored",
    "stale_header",
    "suspect_stale",
]
COVERAGE_STATES: tuple[CoverageState, ...] = get_args(CoverageState)
_RAW_ONLY_STATES = ("not_stored", "stale_header")
_LEGACY_ONLY_STATES = ("suspect_stale",)


class CoverageInterval(Model):
    """Transport evidence only, keyed to fetch instants (§5.1, combined rule
    of the 10-21 verdict). Train counts never make an interval uncovered.
    ``not_stored``: fetched, but the bytes were not stored, so nothing
    decodes. ``suspect_stale`` (legacy) is uncertain evidence, not a
    transport failure."""

    source: GtfsRtSource
    lineage: Lineage
    interval: Interval
    state: CoverageState | Unknown
    evidence_count: int = Field(
        ge=0, description="Fetches (raw) or trace objects (legacy) inside."
    )

    @model_validator(mode="after")
    def _consistent(self) -> CoverageInterval:
        if self.interval.end <= self.interval.start:
            raise _fail("coverage_interval", "a coverage interval is not empty")
        if utc_day(self.interval.start) != utc_day(self.interval.end - 1):
            raise _fail("coverage_interval", "coverage intervals are split at 00:00Z")
        state = self.state
        if self.lineage == "legacy" and state in _RAW_ONLY_STATES:
            raise _fail("coverage_state", f"{state} needs raw transport evidence")
        if self.lineage == "raw" and state in _LEGACY_ONLY_STATES:
            raise _fail("coverage_state", f"{state} is the legacy staleness proxy")
        return self


class SupplyObservation(Model):
    """Observed trains against schedule at one good fetch, per feed and
    route (§5.2). Evidence for layer 5, never coverage."""

    observed_at: Instant
    lineage: Lineage
    source: GtfsRtSource
    source_basis: SourceBasis
    route_id: str = Field(
        description="Network route: the matched trip's, else the trip-id field."
    )
    route_basis: Literal["scheduled_trip", "trip_id_field"]
    vehicles: int = Field(
        ge=0, description="Certain: real vehicles after §2.2, fresh positions."
    )
    vehicles_stale_tail: int = Field(
        ge=0, description="Uncertain: runs in a stale_tail block (held train or ghost)."
    )
    scheduled_in_service: int = Field(
        ge=0, description="Layer-1 trips in service at observed_at."
    )
    listed_trips: Annotated[int, Field(ge=0)] | Unknown = Field(
        description="Raw: trip updates listing the route; legacy: Unknown."
    )

    @model_validator(mode="after")
    def _consistent(self) -> SupplyObservation:
        _check_source_basis(self.lineage, self.source_basis)
        if isinstance(self.listed_trips, Unknown) != (self.lineage == "legacy"):
            raise _fail("lineage_unknown", "listed trips are Unknown exactly on legacy")
        return self
