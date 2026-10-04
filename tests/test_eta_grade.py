"""Countdown-ETA grading (training/eta_grade.py): the matching and outcome rules.

Synthetic predictions and trace events only — no R2, no snapshots on disk.
"""

from __future__ import annotations

from training.eta_grade import (
    GRACE,
    MATCH_AFTER,
    Prediction,
    grade,
    predictions_from_archive_record,
    summarize,
)
from training.trace import Arrival, Passing

T = 1_790_000_000


def _pred(
    *, trip: str = "t1", stop: str = "A07N", at: int = T, eta: int = T + 300
) -> Prediction:
    return Prediction(
        observed_at=at, stop_id=stop, route="A", trip_id=trip, eta_epoch=eta
    )


def _arr(*, trip: str = "t1", stop: str = "A07N", at: int) -> Arrival:
    return Arrival(
        trip_id=trip, route_id="A", direction="N", stop_id=stop, stop_seq=None, at=at
    )


def _pas(*, trip: str = "t1", stop: str = "A07N", at: int) -> Passing:
    return Passing(
        trip_id=trip, route_id="A", direction="N", stop_id=stop, stop_seq=None, at=at
    )


def test_error_is_actual_minus_promised_and_binned_by_horizon() -> None:
    g = grade([_pred(eta=T + 300)], [_arr(at=T + 390)], [], trace_end=T + 10_000)
    assert list(g.errors["h|5-10 min"]) == [90]  # 90 s later than promised
    assert g.outcomes["matched|5-10 min"] == 1


def test_matches_the_same_trip_at_the_same_stop_only() -> None:
    g = grade(
        [_pred(trip="t1", stop="A07N")],
        [
            _arr(trip="t2", stop="A07N", at=T + 300),
            _arr(trip="t1", stop="A06N", at=T + 300),
        ],
        [],
        trace_end=T + 10_000,
    )
    assert "h|5-10 min" not in g.errors
    assert g.outcomes["never_seen|5-10 min"] == 1


def test_a_later_run_of_a_recurring_trip_id_is_not_matched() -> None:
    # trip_ids recur on later days; an arrival past the window is a different run.
    g = grade(
        [_pred()], [_arr(at=T + MATCH_AFTER + 1)], [], trace_end=T + 3 * MATCH_AFTER
    )
    assert g.outcomes["never_seen|5-10 min"] == 1


def test_a_passing_without_an_arrival_is_served_unobserved_never_scored() -> None:
    # Departure times are later than arrivals by the dwell; scoring one would
    # make the countdown look early. They only classify the miss.
    g = grade([_pred()], [], [_pas(at=T + 330)], trace_end=T + 10_000)
    assert g.outcomes["served_unobserved|5-10 min"] == 1
    assert not any(k.startswith("h|") for k in g.errors)


def test_outcome_is_pending_until_the_trace_runs_past_the_grace_window() -> None:
    g = grade([_pred(eta=T + 300)], [], [], trace_end=T + 300 + GRACE - 1)
    assert g.outcomes["pending"] == 1
    assert not any(k.startswith("never_seen") for k in g.outcomes)


def test_summary_reports_share_within_a_minute_and_late_rates() -> None:
    preds = [_pred(trip=f"t{i}", eta=T + 300) for i in range(4)]
    arrs = [_arr(trip=f"t{i}", at=T + 300 + d) for i, d in enumerate((0, 30, 200, -90))]
    s = summarize(grade(preds, arrs, [], trace_end=T + 10_000))["by_horizon"][
        "5-10 min"
    ]
    assert s["n"] == 4
    assert s["within_60s"] == 0.5
    assert s["late_over_120s"] == 0.25
    assert s["early_over_60s"] == 0.25


def test_an_arrival_before_the_snapshot_is_stale_not_scored() -> None:
    # The countdown still listed a train that had arrived 30 s earlier.
    g = grade([_pred(eta=T + 60)], [_arr(at=T - 30)], [], trace_end=T + 10_000)
    assert g.outcomes["already_arrived|0-2 min"] == 1
    assert not any(k.startswith("h|") for k in g.errors)


def test_archive_record_parses_schema_and_skips_null_trip_ids() -> None:
    # worker/src/archive.ts's archiveArrivalsSample shape: schema_version,
    # observed_at, fresh_feeds, expected_feeds, stops keyed by stop_id.
    doc = {
        "schema_version": 1,
        "observed_at": T,
        "fresh_feeds": ["ace"],
        "expected_feeds": ["ace"],
        "stops": {
            "A07N": [
                {"route": "A", "eta_epoch": T + 300, "trip_id": "t1"},
                # No trip_id (anonymous trip): never matchable, dropped same
                # as read_snapshots drops it.
                {"route": "A", "eta_epoch": T + 600, "trip_id": None},
            ],
            "A06N": [{"route": "A", "eta_epoch": T + 450, "trip_id": "t2"}],
        },
    }
    preds = list(predictions_from_archive_record(doc))
    assert preds == [
        Prediction(
            observed_at=T, stop_id="A07N", route="A", trip_id="t1", eta_epoch=T + 300
        ),
        Prediction(
            observed_at=T, stop_id="A06N", route="A", trip_id="t2", eta_epoch=T + 450
        ),
    ]
