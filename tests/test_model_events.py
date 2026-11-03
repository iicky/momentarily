"""Events layer (schema events/s1): contracts, the record keys, and the pinned
parquet column layout.

The fixtures under tests/fixtures/model/events/records/ use values from the
Gate A/B recorded days (docs/model-events.md §10): one file per exported
document, each a list of records. worker/test/model_events.test.ts validates
the same files with ajv. Validators raise typed errors; tests match the error
type, never wording.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError

from momentarily.model.conventions import ContractDefaultError, Unknown
from momentarily.model.events.columns import LAYOUT_VERSION, RECORDS
from momentarily.model.events.contracts import (
    CoverageInterval,
    DurationBound,
    EventBound,
    Headway,
    OpenState,
    SegmentTraversal,
    StopEvent,
    SupplyObservation,
    TripMatch,
    bound_after,
)
from momentarily.model.events.keys import (
    SERVICE_DAY_RECORDS,
    UTC_DAY_RECORDS,
    EventsKey,
    MatchIndexKey,
    ServiceDayKey,
    UtcDayKey,
    parse_key,
)
from momentarily.model.observations.columns import (
    UNKNOWN_SUFFIX,
    from_row,
    layout,
    to_row,
)
from momentarily.model.observations.keys import KeyLayoutError
from scripts.export_model_schema import (
    DOCUMENTS,
    SCHEMA_DIR,
    committed_layouts,
    render_layouts,
)

FIXTURE_DIR = (
    Path(__file__).resolve().parent / "fixtures" / "model" / "events" / "records"
)
EVENT_DOCS = {
    name.removeprefix("events/"): model
    for name, model in DOCUMENTS.items()
    if name.startswith("events/")
}
# sha256 over the committed parquet layouts, per LAYOUT_VERSION. Event files
# are kept: a layout change must bump LAYOUT_VERSION and add a new pin here,
# never edit an old one.
PINNED_LAYOUTS = {
    1: "78833180fb5238a31bf249f435fbf9b8011b7b961c3c04b1992489bad45bb8a4",
    2: "6184bf706ffa98fa39b349bcf2b191fef83369f1db63fa186e9ffb722a957916",
}
FIXTURE_OF: dict[str, str] = {
    "trip_match": "trip_match",
    "stop_event": "stop_event",
    "segment_traversal": "segment_traversal",
    "headway": "headway",
    "open_state": "open_state",
    "coverage": "coverage_interval",
    "supply": "supply_observation",
}


def _records(name: str) -> list[dict[str, Any]]:
    return json.loads((FIXTURE_DIR / f"{name}.json").read_text())


def _record(name: str, index: int = 0, **changes: Any) -> dict[str, Any]:
    return {**_records(name)[index], **changes}


def _error_types(model: type[BaseModel], doc: dict[str, Any]) -> set[str]:
    with pytest.raises(ValidationError) as err:
        model.model_validate(doc)
    return {e["type"] for e in err.value.errors()}


def _bound(after: int | None, by: int, reported: int | None = None) -> dict[str, Any]:
    known = None if after is None else 0
    return {
        "after": after,
        "by": by,
        "reported": reported,
        "uncovered_s": known,
        "suspect_s": known,
    }


# --- fixtures and generated schema -----------------------------------------


def test_every_document_has_a_fixture_and_nothing_else() -> None:
    assert {p.stem for p in FIXTURE_DIR.glob("*.json")} == set(EVENT_DOCS)


@pytest.mark.parametrize("name", sorted(EVENT_DOCS))
def test_fixtures_round_trip_through_pydantic(name: str) -> None:
    model = EVENT_DOCS[name]
    for record in _records(name):
        parsed = model.model_validate_json(json.dumps(record))
        assert parsed.model_dump(mode="json", by_alias=True) == record


def _objects(node: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(node, dict):
        if "properties" in node:
            found.append(node)  # pyright: ignore[reportUnknownArgumentType]
        for value in node.values():  # pyright: ignore[reportUnknownVariableType]
            found += _objects(value)
    elif isinstance(node, list):
        for value in node:  # pyright: ignore[reportUnknownVariableType]
            found += _objects(value)
    return found


@pytest.mark.parametrize("name", sorted(EVENT_DOCS))
def test_exported_schemas_make_every_field_required(name: str) -> None:
    """A default would surface as an optional property, so a record without
    the field would read as 'absent' instead of an explicit null or Unknown."""
    schema = json.loads((SCHEMA_DIR / "events" / f"{name}.schema.json").read_text())
    for obj in _objects(schema):
        assert sorted(obj.get("required", [])) == sorted(obj["properties"])
        assert obj.get("additionalProperties") is False


def test_an_event_contract_with_a_default_is_rejected() -> None:
    with pytest.raises(ContractDefaultError) as err:

        class _Defaulted(EventBound):  # pyright: ignore[reportUnusedClass]
            source_note: str = "feed"

    assert err.value.fields == ["source_note"]


# --- bounds -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("after", "by"), [(1792616107, 1792616107), (1792616108, 1792616107)]
)
def test_an_event_bound_needs_after_strictly_before_by(after: int, by: int) -> None:
    assert "event_bound" in _error_types(EventBound, _bound(after, by))


def test_a_left_censored_event_bound_has_no_after_and_no_overlap() -> None:
    bound = EventBound.model_validate(_bound(None, 1792616046, 1792616040))
    assert (bound.after, bound.uncovered_s, bound.suspect_s) == (None, None, None)
    assert "event_bound_overlap" in _error_types(
        EventBound, {**_bound(None, 1792616046), "uncovered_s": 0}
    )


@pytest.mark.parametrize(
    ("after", "reported"),
    [
        (1792616046, 1792616046),  # == after: §3.2 widens after to reported - 1
        (1792616046, 1792616000),  # before the bracket
        (1792616046, 1792616108),  # after by: dropped and flagged, never kept
        (None, 1792616108),
    ],
)
def test_a_reported_time_outside_the_bound_is_rejected(
    after: int | None, reported: int
) -> None:
    doc = _bound(after, 1792616107, reported)
    assert "event_bound_reported" in _error_types(EventBound, doc)


@pytest.mark.parametrize("reported", [1792616047, 1792616107])
def test_a_reported_time_inside_the_bound_is_kept(reported: int) -> None:
    bound = EventBound.model_validate(_bound(1792616046, 1792616107, reported))
    assert bound.reported == reported


def test_a_duration_without_an_upper_bound_must_be_right_censored() -> None:
    doc = {
        "lo": 60,
        "hi": None,
        "censoring": "none",
        "reported": None,
        "uncovered_s": 0,
    }
    assert "duration_censoring" in _error_types(DurationBound, doc)
    right = DurationBound.model_validate(
        {**doc, "censoring": "right", "uncovered_s": None}
    )
    assert right.hi is None


def test_a_duration_lower_bound_cannot_be_negative() -> None:
    doc = {"lo": -1, "hi": 120, "censoring": "none", "reported": None, "uncovered_s": 0}
    assert "greater_than_equal" in _error_types(DurationBound, doc)


# --- runs and stop events ------------------------------------------------------


def test_a_scheduled_trip_service_date_needs_a_scheduled_trip() -> None:
    unmatched = _record("trip_match", 1)
    assert unmatched["scheduled_trip_id"] is None
    doc = {**unmatched, "service_date_basis": "scheduled_trip"}
    assert "service_date_basis" in _error_types(TripMatch, doc)


def test_a_legacy_run_keeps_its_lineage_and_unknowns() -> None:
    legacy = TripMatch.model_validate(_record("trip_match", 2))
    assert legacy.lineage == "legacy"
    assert legacy.source_basis == "route_feed_table"
    assert isinstance(legacy.start_date_published, Unknown)
    raw = _record("trip_match")
    assert "lineage_unknown" in _error_types(
        TripMatch, {**raw, "start_date_published": {"unknown": True, "reason": "x"}}
    )


def test_a_last_family_stop_event_with_a_numeric_dwell_is_rejected() -> None:
    last = _record("stop_event")
    assert (last["family"], last["service_evidence"]) == ("last", "reported_planned")
    numeric = {
        "lo": 0,
        "hi": 182,
        "censoring": "none",
        "reported": None,
        "uncovered_s": 0,
    }
    assert "stop_event_dwell" in _error_types(StopEvent, {**last, "dwell": numeric})


def test_service_evidence_follows_the_feed_family() -> None:
    """Gate B: the "last" family reports planned stops (reported_planned),
    the "next" family shows the train stopped; neither stands in for the
    other."""
    last = _record("stop_event")
    nxt = _record("stop_event", 2)
    assert "service_evidence" in _error_types(
        StopEvent, {**last, "service_evidence": "stopped"}
    )
    assert "service_evidence" in _error_types(
        StopEvent, {**nxt, "service_evidence": "reported_planned"}
    )


def test_a_passage_has_no_arrival_or_departure() -> None:
    passage = _record("stop_event", 1)
    assert passage["service_evidence"] == "inferred_passage"
    assert "stop_event_bounds" in _error_types(
        StopEvent, {**passage, "arrival": passage["passage"]}
    )


def test_plan_dropped_needs_trip_updates() -> None:
    dropped = _record("stop_event", 1)
    assert dropped["flags"] == ["plan_dropped"]
    legacy = {
        **dropped,
        "lineage": "legacy",
        "source_basis": "route_feed_table",
        "run_id": dropped["run_id"].replace("raw:", "legacy:", 1),
        "passage": {**dropped["passage"], "suspect_s": 0},
    }
    assert "stop_event_flags" in _error_types(StopEvent, legacy)


# --- headways -------------------------------------------------------------------


@pytest.mark.parametrize("evidence", ["inferred_passage", "reported_not_stopped"])
def test_a_service_headway_rejects_a_passage_end(evidence: str) -> None:
    stopped = _record("headway")
    assert stopped["evidence"] == "service"
    for end in ("leader_service_evidence", "follower_service_evidence"):
        assert "headway_evidence" in _error_types(Headway, {**stopped, end: evidence})


def test_service_headways_accept_stopped_and_reported_planned_and_keep_them_apart() -> (
    None
):
    stopped = Headway.model_validate(_record("headway"))
    planned = Headway.model_validate(_record("headway", 1))
    assert (stopped.evidence, planned.evidence) == ("service", "service")
    assert (stopped.follower_service_evidence, stopped.basis) == (
        "stopped",
        "departure",
    )
    assert (planned.follower_service_evidence, planned.basis) == (
        "reported_planned",
        "arrival",
    )
    # A feed has one family, so a "last"-family series cannot take a stopped end.
    folded = {**_record("headway", 1), "leader_service_evidence": "stopped"}
    assert "service_evidence" in _error_types(Headway, folded)


def test_a_physical_headway_admits_passages() -> None:
    physical = {
        **_record("headway"),
        "evidence": "physical",
        "follower_service_evidence": "reported_not_stopped",
        "basis": "passage",
    }
    assert Headway.model_validate(physical).basis == "passage"


def test_the_certain_series_excludes_ambiguous_runs() -> None:
    mixed = _record("headway", 2)
    assert mixed["ambiguity_set"] == "with_ambiguous"
    assert Headway.model_validate(mixed).leader_pattern_attribution == "ambiguous"
    certain = {**mixed, "ambiguity_set": "certain"}
    assert "headway_scope" in _error_types(Headway, certain)


def test_an_unknown_leader_leaves_the_headway_open() -> None:
    carried = _record("headway", 3)
    assert Headway.model_validate(carried).headway.hi is None
    closed = {
        **carried,
        "headway": {
            **carried["headway"],
            "hi": 600,
            "censoring": "none",
            "uncovered_s": 0,
        },
    }
    assert "headway_leader" in _error_types(Headway, closed)


# --- rejected contradictions, by error type --------------------------------------


def _with(name: str, index: int, path: str, value: Any) -> dict[str, Any]:
    """A fixture record with one (dotted) field replaced."""
    doc = json.loads(json.dumps(_records(name)[index]))
    *parents, leaf = path.split(".")
    node = doc
    for part in parents:
        node = node[part]
    node[leaf] = value
    return doc


REJECTED: list[tuple[type[BaseModel], str, int, str, Any, str]] = [
    # the feed decides the family, not the record
    (StopEvent, "stop_event", 0, "family", "next", "service_evidence"),
    (
        SegmentTraversal,
        "segment_traversal",
        0,
        "run_time_basis",
        "vehicle",
        "traversal_run_time",
    ),
    # a feed's runs carry its id: retargeting the record breaks the follower
    (Headway, "headway", 0, "source", "jz", "run_lineage"),
    (OpenState, "open_state", 0, "source", "numbered", "run_lineage"),
    (OpenState, "open_state", 1, "source", "numbered", "service_evidence"),
    # run ids carry the record's lineage (and feed)
    (
        StopEvent,
        "stop_event",
        2,
        "run_id",
        "legacy:numbered:047450_6..S03R@1792583520#0",
        "run_lineage",
    ),
    (
        SegmentTraversal,
        "segment_traversal",
        1,
        "run_id",
        "raw:l:047450_6..S03R@1792583520#0",
        "run_lineage",
    ),
    (
        Headway,
        "headway",
        0,
        "leader_run_id",
        "legacy:numbered:046950_6..S03R@1792583221#0",
        "run_lineage",
    ),
    # a B-division run never joins a numbered series: follower, or leader of
    # another family
    (
        Headway,
        "headway",
        0,
        "follower_run_id",
        "raw:ace:047450_A..S@1792583520#0",
        "run_lineage",
    ),
    (
        Headway,
        "headway",
        0,
        "leader_run_id",
        "raw:bdfm:046950_F..S@1792583221#0",
        "run_lineage",
    ),
    (
        TripMatch,
        "trip_match",
        0,
        "supersedes_run_ids",
        ["raw:bdfm:101450_J..N47R@1792616046#0"],
        "run_lineage",
    ),
    (
        TripMatch,
        "trip_match",
        2,
        "member_run_ids",
        ["legacy:l:118650_1..N@1789516903#0"],
        "run_lineage",
    ),
    (
        TripMatch,
        "trip_match",
        2,
        "member_run_ids",
        ["raw:numbered:118650_1..N@1789516903#0"],
        "run_lineage",
    ),
    # identity (§2.3)
    (
        TripMatch,
        "trip_match",
        0,
        "run_id",
        "raw:jz:101450_J..N47R@1792616046#1",
        "trip_match_identity",
    ),
    (
        TripMatch,
        "trip_match",
        0,
        "identity_uncertain_with",
        ["raw:jz:101450_J..N47R@1792616046#0"],
        "trip_match_identity",
    ),
    (
        TripMatch,
        "trip_match",
        0,
        "identity_uncertain_with",
        ["raw:jz:101450_J..N47R@1792616107#1"],
        "trip_match_identity",
    ),
    (
        TripMatch,
        "trip_match",
        2,
        "identity_uncertain_with",
        ["legacy:numbered:118650_1..N17R@1789516001#1"],
        "trip_match_identity",
    ),
    (
        TripMatch,
        "trip_match",
        2,
        "member_run_ids",
        ["legacy:numbered:118650_1..N17R@1789516001#0"],
        "trip_match_identity",
    ),
    # outcome, service date and pattern fit
    (TripMatch, "trip_match", 0, "tier", None, "trip_match_outcome"),
    (TripMatch, "trip_match", 1, "pattern_attribution", "unique_fit", "pattern_fit"),
    (TripMatch, "trip_match", 0, "pattern_attribution", "unique_fit", "pattern_fit"),
    (
        TripMatch,
        "trip_match",
        0,
        "flags",
        ["service_date_ambiguous"],
        "service_date_basis",
    ),
    # durations follow from their bounds; reported flags describe the bound
    (StopEvent, "stop_event", 2, "dwell.hi", 120, "stop_event_dwell"),
    (
        StopEvent,
        "stop_event",
        0,
        "flags",
        ["reported_widened_after"],
        "stop_event_flags",
    ),
    (StopEvent, "stop_event", 2, "arrival.suspect_s", 0, "event_bound_overlap"),
    # uncovered seconds: inside the span, and at least its bounds'
    (StopEvent, "stop_event", 2, "arrival.uncovered_s", 30, "stop_event_dwell"),
    (Headway, "headway", 0, "leader_passing.uncovered_s", 60, "headway_bound"),
    (
        SegmentTraversal,
        "segment_traversal",
        1,
        "to_arrival.uncovered_s",
        61,
        "traversal_run_time",
    ),
    (StopEvent, "stop_event", 2, "dwell.uncovered_s", 122, "duration_bound"),
    (
        StopEvent,
        "stop_event",
        5,
        "arrival",
        {
            "after": 1789477261,
            "by": 1789477321,
            "reported": 1789477316,
            "uncovered_s": 31,
            "suspect_s": 30,
        },
        "event_bound_overlap",
    ),
    (
        SegmentTraversal,
        "segment_traversal",
        0,
        "stop_to_stop.lo",
        0,
        "traversal_run_time",
    ),
    (
        SegmentTraversal,
        "segment_traversal",
        1,
        "run_time.hi",
        120,
        "traversal_run_time",
    ),
    (
        SegmentTraversal,
        "segment_traversal",
        0,
        "segment_ids",
        ["M22N>M20N"],
        "traversal_order",
    ),
    (SegmentTraversal, "segment_traversal", 1, "to_seq", 1, "traversal_order"),
    (Headway, "headway", 1, "headway.lo", 540, "headway_bound"),
    (Headway, "headway", 3, "basis", "departure", "headway_evidence"),
    (
        DurationBound,
        "headway",
        0,
        "headway",
        {"lo": 1, "hi": None, "censoring": "right", "reported": None, "uncovered_s": 0},
        "duration_censoring",
    ),
    # open state
    (
        OpenState,
        "open_state",
        1,
        "last_event_evidence",
        "inferred_passage",
        "headway_evidence",
    ),
    (OpenState, "open_state", 0, "last_event_seq", None, "open_state_kind"),
    (OpenState, "open_state", 1, "last_event_seq", 1, "open_state_kind"),
    (OpenState, "open_state", 0, "current_gap", {"lo": 32, "hi": 93}, "open_state_gap"),
    (OpenState, "open_state", 1, "eventual.lo", 292, "open_state_gap"),
    (
        OpenState,
        "open_state",
        2,
        "current_gap",
        {"lo": 901, "hi": None},
        "open_state_gap",
    ),
    # supply and coverage
    (SupplyObservation, "supply_observation", 1, "listed_trips", 12, "lineage_unknown"),
    (
        SupplyObservation,
        "supply_observation",
        0,
        "source_basis",
        "route_feed_table",
        "source_basis",
    ),
    (SupplyObservation, "supply_observation", 0, "vehicles", -1, "greater_than_equal"),
    (
        CoverageInterval,
        "coverage_interval",
        1,
        "interval",
        {"start": 1792540790, "end": 1792540810},
        "coverage_interval",
    ),
]


@pytest.mark.parametrize(
    ("model", "name", "index", "path", "value", "error_type"),
    REJECTED,
    ids=[f"{r[1]}[{r[2]}].{r[3]}->{r[5]}" for r in REJECTED],
)
def test_contradictions_are_rejected(
    model: type[BaseModel],
    name: str,
    index: int,
    path: str,
    value: Any,
    error_type: str,
) -> None:
    doc = _with(name, index, path, value)
    if model is DurationBound:
        doc = doc["headway"]
    assert error_type in _error_types(model, doc)


def test_an_open_run_may_hold_a_matched_pattern_outside_its_fit() -> None:
    doc = _with("open_state", 0, "compatible_pattern_ids", ["Z-N-0000000000ff"])
    assert OpenState.model_validate(doc).pattern_id == "Z-N-bfb91b97d742"


_B = 1792616000


def _clock(lo: int, until: int | None, as_of: int = _B + 200) -> dict[str, Any]:
    """open_state[1] with its last passing in (B + 100, B + 160] (§4 and the
    slice-6 acceptance examples, shifted by B)."""
    doc = _record("open_state", 1)
    return {
        **doc,
        "as_of": as_of,
        "since": _B + 160,
        "last_event": {**doc["last_event"], "after": _B + 100, "by": _B + 160},
        "last_event_evidence": "reported_planned",
        "uncovered_until": until,
        "flags": [] if until is None else ["uncovered_since"],
        "current_gap": {"lo": lo, "hi": as_of - _B - 101},
        "eventual": {**doc["eventual"], "lo": lo},
    }


@pytest.mark.parametrize(
    ("lo", "until"),
    [
        (40, None),  # no outage: [t - by, t - after - 1] = [40, 99]
        (40, _B + 120),  # an outage ending inside the bound: lo 40, not 80
        (20, _B + 180),  # ending after it: t - e
        (0, _B + 200),  # still uncovered at as_of
    ],
)
def test_the_open_gap_is_exactly_what_the_evidence_allows(
    lo: int, until: int | None
) -> None:
    state = OpenState.model_validate(_clock(lo, until))
    assert (state.current_gap.lo, state.current_gap.hi) == (lo, 99)
    assert state.eventual.lo == lo


@pytest.mark.parametrize(
    ("lo", "until", "flags"),
    [
        (0, None, []),  # understated: the evidence gives 40
        (39, None, []),
        (41, None, []),  # overstated
        (0, _B + 120, ["uncovered_since"]),  # the outage ended inside the bound
        (40, _B + 120, []),  # an outage end without its flag
        (40, None, ["uncovered_since"]),  # a flag without the outage end
        (40, _B + 90, ["uncovered_since"]),  # ended before the last passing
    ],
)
def test_an_open_gap_off_the_evidence_is_rejected(
    lo: int, until: int | None, flags: list[str]
) -> None:
    doc = {**_clock(lo, until), "flags": flags}
    assert "open_state_gap" in _error_types(OpenState, doc)
    future = _clock(0, _B + 201)
    assert "open_state_time" in _error_types(OpenState, future)


def test_scheduled_no_observation_counts_exactly_from_since() -> None:
    doc = _record("open_state", 2)
    assert OpenState.model_validate(doc).current_gap.lo == doc["as_of"] - doc["since"]
    understated = {
        **doc,
        "current_gap": {"lo": 0, "hi": None},
        "eventual": {**doc["eventual"], "lo": 0},
    }
    assert "open_state_gap" in _error_types(OpenState, understated)
    # A feed outage that ended 300 s before as_of: a train may have passed.
    outage = {
        **understated,
        "uncovered_until": doc["as_of"] - 300,
        "flags": ["uncovered_since"],
        "current_gap": {"lo": 300, "hi": None},
        "eventual": {**doc["eventual"], "lo": 300},
    }
    assert OpenState.model_validate(outage).current_gap.lo == 300


def test_a_headway_leader_may_come_from_another_feed_of_the_family() -> None:
    """Raw ``ace`` publishes some F trips, so a route's runs can span the
    feeds of one family; never two families."""
    jz = _record("headway", 1)
    leader = jz["leader_run_id"].replace("raw:jz:", "raw:bdfm:", 1)
    assert Headway.model_validate({**jz, "leader_run_id": leader}).source == "jz"
    follower = jz["follower_run_id"].replace("raw:jz:", "raw:bdfm:", 1)
    assert "run_lineage" in _error_types(Headway, {**jz, "follower_run_id": follower})


def test_a_duration_carries_the_uncovered_seconds_of_its_bounds() -> None:
    """A dwell over an arrival with 30 uncovered seconds is not fully
    covered; with the 30 s (or more, up to its 121 s span) it is."""
    nxt = _record("stop_event", 2)
    arrival = {**nxt["arrival"], "uncovered_s": 30}
    for uncovered in (30, 121):
        dwell = {**nxt["dwell"], "uncovered_s": uncovered}
        event = StopEvent.model_validate({**nxt, "arrival": arrival, "dwell": dwell})
        assert isinstance(event.dwell, DurationBound)
    # Overlapping bounds: arrival (580, 641] and departure (600, 701] share
    # 41 s, so 30 + 30 uncovered seconds may be as few as 30 in the span.
    a = {
        "after": 580,
        "by": 641,
        "reported": None,
        "uncovered_s": 30,
        "suspect_s": None,
    }
    d = {**a, "after": 600, "by": 701}
    span = {"lo": 0, "hi": 121, "censoring": "none", "reported": None}
    ok = {**nxt, "arrival": a, "departure": d, "dwell": {**span, "uncovered_s": 30}}
    assert StopEvent.model_validate(ok).dwell is not None
    low = {**ok, "dwell": {**span, "uncovered_s": 29}}
    assert "stop_event_dwell" in _error_types(StopEvent, low)


def test_a_tie_in_one_second_falls_back_to_the_previous_distinct_fetch() -> None:
    """Real case: ``numbered@1792708401559`` and ``numbered@1792708402024``
    both finished in second 1792708402 and differ on 15 trips. When the
    second shows the event and the first does not, the first counts as at
    ``by``: ``after`` is the fetch a minute earlier, never ``by`` itself."""
    by = 1792708402
    observed = [1792708282, 1792708342, by, by]
    after = bound_after(observed, by)
    assert after == 1792708342
    bound = EventBound.model_validate(_bound(after, by))
    assert (bound.after, bound.by) == (1792708342, by)
    assert bound_after([by], by) is None
    assert "event_bound" in _error_types(EventBound, _bound(by, by))


# --- coverage -------------------------------------------------------------------


def test_coverage_accepts_not_stored() -> None:
    lost = CoverageInterval.model_validate(_record("coverage_interval", 3))
    assert lost.state == "not_stored"


@pytest.mark.parametrize(
    ("index", "state"),
    [(4, "stale_header"), (4, "not_stored"), (0, "suspect_stale")],
)
def test_coverage_states_need_their_lineage_evidence(index: int, state: str) -> None:
    doc = _record("coverage_interval", index, state=state)
    assert "coverage_state" in _error_types(CoverageInterval, doc)


# --- keys --------------------------------------------------------------------------

KEYS: list[tuple[EventsKey, str]] = [
    (
        ServiceDayKey("raw", "stop_event", "2026-10-21"),
        "model/events/s1/raw/stop_event/2026-10-21.parquet",
    ),
    (
        ServiceDayKey("legacy", "trip_match", "2026-09-15"),
        "model/events/s1/legacy/trip_match/2026-09-15.parquet",
    ),
    (
        UtcDayKey("raw", "coverage", "2026-10-22"),
        "model/events/s1/raw/coverage/2026-10-22.parquet",
    ),
    (
        MatchIndexKey(2, "2026-10-21", "606d9f640f5aad67", "0123456789abcdef"),
        "model/events/s1/match_index/v2/2026-10-21/606d9f640f5aad67-0123456789abcdef.json.gz",
    ),
]


@pytest.mark.parametrize(("typed", "text"), KEYS)
def test_keys_round_trip(typed: EventsKey, text: str) -> None:
    assert typed.key == text
    assert parse_key(text) == typed


def test_every_record_kind_has_a_column_layout() -> None:
    assert set(RECORDS) == {*SERVICE_DAY_RECORDS, *UTC_DAY_RECORDS}


@pytest.mark.parametrize(
    "text",
    [
        "model/events/s1/raw/coverage/2026-02-30.parquet",
        "model/events/s1/mixed/stop_event/2026-10-21.parquet",
        "model/events/s1/raw/scheduled_trip_outcome/2026-10-21.parquet",
        "model/events/s1/raw/stop_event/2026-10-21/00.parquet",
    ],
)
def test_keys_outside_the_layout_are_rejected(text: str) -> None:
    with pytest.raises(KeyLayoutError):
        parse_key(text)


# --- parquet layout --------------------------------------------------------------


def test_the_parquet_layout_is_pinned() -> None:
    """If this fails, the column layout changed. Event files are kept: bump
    LAYOUT_VERSION, add its digest to PINNED_LAYOUTS, then regenerate with
    uv run python -m scripts.export_model_schema."""
    rendered = render_layouts("events")
    assert committed_layouts("events") == rendered, (
        "parquet layout drifted from the committed files"
    )
    digest = hashlib.sha256(
        "".join(f"{name}\n{text}" for name, text in sorted(rendered.items())).encode()
    ).hexdigest()
    assert PINNED_LAYOUTS.get(LAYOUT_VERSION) == digest, (
        "layout changed: bump LAYOUT_VERSION"
    )


@pytest.mark.parametrize("record", sorted(RECORDS))
def test_fixtures_round_trip_through_columns(record: str) -> None:
    model: type[BaseModel] = RECORDS[record]  # pyright: ignore[reportArgumentType]
    names = [c.name for c in layout(model)]
    for doc in _records(FIXTURE_OF[record]):
        parsed = model.model_validate(doc)
        row = to_row(parsed)
        assert list(row) == names
        assert from_row(model, row) == parsed


def test_a_censored_departure_stays_distinct_from_none_in_columns() -> None:
    service, passage = (
        to_row(StopEvent.model_validate(_record("stop_event", i))) for i in (4, 3)
    )
    unknown = "departure" + UNKNOWN_SUFFIX
    assert service["departure"] is None
    assert service[unknown] is not None
    assert (passage["departure"], passage[unknown]) == (None, None)
