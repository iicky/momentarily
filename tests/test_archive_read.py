"""The shared grader read path (training/archive_read.py).

Pure cases only: the R2 reads behind `load_traversals` / `load_windows` are
exercised by running the graders. What is pinned here is the accounting that
decides which days a grade included, how a raw body is assigned to an archive
day, and when a span must refuse to pool.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from training.archive_read import (
    WINDOW_PUBLICATION_LOOKBACK_DAYS,
    Loaded,
    missing_days,
    utc_day,
)
from training.traversal_archive import (
    EXTRACTOR_VERSION,
    SCHEMA_VERSION,
    DayProvenance,
)
from training.traversal_archive import key_for as traversal_key


def _prov(*, feed: str | None = "F", digest: str | None = None) -> DayProvenance:
    return DayProvenance(
        schema=SCHEMA_VERSION,
        extractor=EXTRACTOR_VERSION,
        feed_version=feed,
        feed_digest=digest,
        n_rows=1,
        n_source_objects=1,
        source_manifest="x",
        code_sha="deadbeef",
        written_at=0,
    )


def test_a_raw_body_is_assigned_to_its_UTC_day_not_the_hosts():
    """Both the raw trace keys and the derived archive are partitioned on the UTC
    date. A host-local conversion would move everything between local midnight
    and UTC midnight into the neighbouring day, so on a New York host the small
    hours of each UTC day would be filtered against the wrong bucket — dropping
    real observations or double-counting ones the archive already served."""
    # 02:00 UTC on the 13th is still the 12th in New York.
    ts = int(datetime(2026, 8, 13, 2, 0, tzinfo=UTC).timestamp())
    assert utc_day({"scheduled_at": ts}) == date(2026, 8, 13)
    # observed_at is the documented fallback when scheduled_at is absent.
    assert utc_day({"observed_at": ts}) == date(2026, 8, 13)


def test_only_days_absent_from_the_archive_are_fetched_from_raw():
    """Fallback is per day, not per run: a span straddling the boundary works
    without a flag, because the archive only finalizes closed days."""
    present = {traversal_key(date(2026, 8, 12)), traversal_key(date(2026, 8, 13))}
    got = missing_days(date(2026, 8, 12), date(2026, 8, 15), present, traversal_key)
    assert got == [date(2026, 8, 14), date(2026, 8, 15)]


def test_nothing_is_fetched_when_the_archive_covers_the_span():
    present = {traversal_key(d) for d in (date(2026, 8, 12), date(2026, 8, 13))}
    assert (
        missing_days(date(2026, 8, 12), date(2026, 8, 13), present, traversal_key) == []
    )


def test_the_summary_says_which_half_a_result_came_from():
    """A grade must always be able to state how much of it was archived, because
    the archived half is the only part that survives the raw prune."""
    loaded = Loaded[int](
        rows=[1, 2, 3],
        archived_days=[date(2026, 8, 12)],
        raw_days=[date(2026, 8, 13)],
        provenance={date(2026, 8, 12): _prov()},
    )
    text = loaded.summary()
    assert "3 rows" in text
    assert "1 archived days" in text
    assert "1 from raw" in text
    assert "PROVENANCE" not in text  # homogeneous, so no warning


def test_a_mixed_span_says_so_in_its_summary_and_refuses_to_pool():
    """After the raw streams prune, provenance is the only remaining signal that
    two days are not the same measurement. A grade that aggregates across days
    must fail rather than average over a definitional change."""
    mixed = Loaded[int](
        rows=[1],
        archived_days=[date(2026, 8, 12), date(2026, 8, 13)],
        provenance={
            date(2026, 8, 12): _prov(feed="A"),
            date(2026, 8, 13): _prov(feed="B"),
        },
    )
    assert not mixed.homogeneous
    assert "PROVENANCE VERSIONS" in mixed.summary()
    with pytest.raises(SystemExit, match="provenance versions"):
        mixed.require_pooled("traversals")


def test_a_homogeneous_span_pools_without_complaint():
    same = Loaded[int](
        rows=[1],
        archived_days=[date(2026, 8, 12), date(2026, 8, 13)],
        provenance={date(2026, 8, 12): _prov(), date(2026, 8, 13): _prov()},
    )
    same.require_pooled("traversals")  # must not raise


def test_an_all_raw_span_has_no_provenance_and_pools_freely():
    """Days served from raw carry no archive provenance. That is not a mixed
    span — there is nothing to be inconsistent with — so it must not trip the
    pooling guard and block a grade over today's data."""
    raw_only = Loaded[int](rows=[1], raw_days=[date(2026, 8, 17)])
    assert raw_only.homogeneous
    raw_only.require_pooled("traversals")


def test_the_window_lookback_reaches_past_the_evaluation_span():
    """Windows are keyed by PUBLICATION day, not work day. A closure announced in
    July for August work is archived under July, so reading only the evaluation
    span's own days would omit exactly the long-lead closures this measure most
    wants."""
    assert WINDOW_PUBLICATION_LOOKBACK_DAYS >= 90
