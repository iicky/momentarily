"""Journey-time gate harness (training/journey_gate.py).

Synthetic spans, no R2. Pins the causal history window, the deterministic
holdout subsample, the common-cohort skill contract (an abstain-happy
forecaster cannot move anyone else's denominator), and one end-to-end run
that writes summary.json.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from training.journey_gate import (
    HistoryIndex,
    common_metrics,
    in_scored_subsample,
    main,
)
from training.journey_types import (
    HISTORY_SECONDS,
    HOLDOUT_START,
    HOLDOUT_TRIP_MODULUS,
    QUANTILES,
    TRAIN_END,
    TRAIN_START,
    SpanSample,
)

NY = ZoneInfo("America/New_York")
T0 = 1_760_000_000


def _span(
    *,
    trip: str = "t",
    depart_at: int = T0,
    seconds: int = 100,
    day: str = "2026-09-15",
    scheduled: int | None = 100,
) -> SpanSample:
    return SpanSample(
        trip_id=trip,
        route_id="Q",
        direction="S",
        origin="Q05S",
        destination="Q06S",
        depart_at=depart_at,
        seconds=seconds,
        n_hops=1,
        scheduled_sec=scheduled,
        service_day=day,
    )


def test_history_window_is_causal_and_ordered() -> None:
    inside_late = _span(trip="late", depart_at=T0 - 50, seconds=50)  # completes at T0
    inside_early = _span(
        trip="early", depart_at=T0 - HISTORY_SECONDS, seconds=1
    )  # completes T0 - 3599
    future = _span(trip="future", depart_at=T0 - 100, seconds=101)  # completes T0 + 1
    stale = _span(
        trip="stale", depart_at=T0 - HISTORY_SECONDS - 3601, seconds=3600
    )  # completes T0 - 3601
    boundary = _span(
        trip="boundary", depart_at=T0 - HISTORY_SECONDS - 100, seconds=100
    )  # completes exactly T0 - 3600: outside the open lower bound
    index = HistoryIndex([future, stale, inside_late, boundary, inside_early])
    got = index.before(T0)
    assert [s.trip_id for s in got] == ["early", "late"]  # ascending completion


def test_holdout_subsample_is_the_contract_hash_rule() -> None:
    for trip in ("072000_Q..S01R", "a", "b", "trip-42"):
        expected = (
            int(hashlib.sha1(trip.encode()).hexdigest(), 16) % HOLDOUT_TRIP_MODULUS == 0
        )
        assert in_scored_subsample(trip) is expected
        assert in_scored_subsample(trip) is in_scored_subsample(trip)


def test_abstainer_does_not_move_other_forecasters_common_cohort() -> None:
    samples = [_span(trip=f"t{i}", seconds=100 + i) for i in range(4)]
    point = dict.fromkeys(QUANTILES, 100.0)
    full = [point] * 4
    half = [point, None, point, None]
    with_abstainer = common_metrics(
        samples, {"schedule": full, "historical": full, "x": half}
    )
    without = common_metrics(samples, {"schedule": full, "historical": full})
    assert with_abstainer["historical"]["common_cohort_n"] == 4
    assert (
        with_abstainer["historical"]["crps_common"]
        == without["historical"]["crps_common"]
    )
    assert with_abstainer["schedule"]["common_cohort_n"] == 4
    # The abstainer itself is judged only where it answered.
    assert with_abstainer["x"]["common_cohort_n"] == 2


def _synthetic_spans() -> tuple[list[SpanSample], int]:
    """One hop: every train day x 5 trips, 7 holdout days x 8. Returns the
    spans and the train-day count so the assertions follow the contract window."""

    def at(day: date, hour: int, minute: int) -> int:
        return int(datetime.combine(day, time(hour, minute), tzinfo=NY).timestamp())

    # Holdout trips chosen so the modulus rule deterministically keeps some
    # and drops some.
    kept = [t for i in range(200) if in_scored_subsample(t := f"trip{i}")][:4]
    dropped = [t for i in range(200) if not in_scored_subsample(t := f"trip{i}")][:4]
    assert kept
    assert dropped
    train_days = (
        date.fromisoformat(TRAIN_END) - date.fromisoformat(TRAIN_START)
    ).days + 1
    out: list[SpanSample] = []
    for d in range(train_days):
        day = date.fromisoformat(TRAIN_START) + timedelta(days=d)
        for j in range(5):
            out.append(
                _span(
                    trip=f"train{d}_{j}",
                    depart_at=at(day, 12, 10 * j),
                    seconds=100 + (d + j) % 7,
                    day=day.isoformat(),
                )
            )
    for d in range(7):
        day = date.fromisoformat(HOLDOUT_START) + timedelta(days=d)
        for j, trip in enumerate(kept + dropped):
            out.append(
                _span(
                    trip=trip,
                    depart_at=at(day, 12, 5 * j),
                    seconds=100 + (d + j) % 7,
                    day=day.isoformat(),
                )
            )
    return out, train_days


def test_gate_end_to_end_writes_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spans, train_days = _synthetic_spans()
    spans_path = tmp_path / "spans.jsonl.gz"
    with gzip.open(spans_path, "wt", encoding="utf-8") as fh:
        for s in spans:
            fh.write(json.dumps(asdict(s)) + "\n")
    out_path = tmp_path / "summary.json"
    assert main(["--spans", str(spans_path), "--out", str(out_path)]) == 0
    summary = json.loads(out_path.read_text())
    split = summary["split"]
    assert split["n_train"] == train_days * 5
    assert split["n_holdout"] == 56
    assert split["n_scored_samples"] == 28  # 4 kept trips x 7 days
    for name in ("schedule", "historical"):
        row = summary["forecasters"][name]
        assert row["n_scored"] + row["n_abstained"] == 28
        assert row["common_cohort_n"] == 28
        assert row["crps_common"] is not None
        assert row["skill_vs_schedule_common"] is not None
        assert row["skill_vs_historical_common"] is not None
    # The schedule point forecast's CRPS is its MAE, end to end.
    sched = summary["forecasters"]["schedule"]
    assert sched["crps_mean"] == sched["mae_median"]
    assert "forecaster" in capsys.readouterr().out
