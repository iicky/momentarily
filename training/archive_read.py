"""The read path every grader uses: derived archive first, raw feeds as fallback.

WHY A SHARED LOADER. Three graders (`planned_work`, `segment_grade`, `progress`)
each need the same two inputs over the same span — traversals and the announced
windows they are graded against — and each used to rebuild both from the raw
streams on every run. That is why a grade cost minutes and why it could never
reach past `archive/trace/`'s 30-day prune. Reading the derived archives fixes
both, but only if all three read them the SAME way: a loader per grader would
drift, and two graders disagreeing about which days they included is exactly the
class of bug this repo has spent the week correcting.

FALLBACK IS PER DAY, NOT PER RUN. A day is served from the archive when it is
there and derived from raw otherwise, so a run spanning the boundary works
without a flag: today and yesterday come from raw trace (the archive only
finalizes closed days), everything older comes from the archive. `raw_days` and
`archived_days` are reported so a result always says which half it came from.

PROVENANCE IS SURFACED, NEVER AVERAGED OVER. `homogeneous` is False when the
archived span crosses a change in extraction, in the static feed, or in whether
that feed's bytes are known. Callers are expected to print it; `require_pooled`
turns it into a hard failure for the grades that pool across days. The archive
outlives its inputs, so after the raw streams prune this is the only remaining
signal that two days are not the same measurement.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

from training.load_r2 import date_range
from training.planned_work import (
    Window,
    fetch_alert_bodies,
    windows_from_alerts,
)
from training.trace import Traversal, fetch_trace_bodies, traversals_from_trace
from training.traversal_archive import DayProvenance
from training.traversal_archive import key_for as traversal_key
from training.traversal_archive import read_days as read_traversal_days
from training.window_archive import key_for as window_key
from training.window_archive import read_days as read_window_days


@dataclass(frozen=True)
class Loaded[T]:
    """Rows over a span, and where each day came from."""

    rows: list[T]
    archived_days: list[date] = field(default_factory=lambda: list[date]())
    raw_days: list[date] = field(default_factory=lambda: list[date]())
    provenance: dict[date, DayProvenance] = field(
        default_factory=lambda: dict[date, DayProvenance]()
    )

    @property
    def versions(self) -> set[tuple[int, int, str | None, str | None]]:
        return {p.comparable_key for p in self.provenance.values()}

    @property
    def homogeneous(self) -> bool:
        return len(self.versions) <= 1

    def summary(self) -> str:
        return (
            f"{len(self.rows)} rows: {len(self.archived_days)} archived days, "
            f"{len(self.raw_days)} from raw"
            + (
                ""
                if self.homogeneous
                else f", {len(self.versions)} PROVENANCE VERSIONS"
            )
        )

    def require_pooled(self, what: str) -> None:
        """Fail rather than pool rows whose days are not the same measurement.

        For grades that aggregate across days. The versions are listed because
        the useful next step is almost always to narrow the span to one of them,
        not to override the check.
        """
        if not self.homogeneous:
            raise SystemExit(
                f"{what} spans {len(self.versions)} provenance versions and cannot "
                f"be pooled: {sorted(str(v) for v in self.versions)}"
            )


def missing_days(
    start: date, end: date, present: set[str], key_of: Callable[[date], str]
) -> list[date]:
    return [d for d in date_range(start, end) if key_of(d) not in present]


def utc_day(body: dict[str, object]) -> date:
    """The archive day a raw trace body belongs to.

    UTC, because that is what partitions both the raw trace keys
    (worker/src/archive.ts writes utcDate) and the derived archive. A host-local
    conversion would move every body between local midnight and UTC midnight into
    the neighbouring day, so on a New York host the small hours of each UTC day
    would be filtered against the wrong bucket — dropping real observations or
    double-counting ones the archive already served.
    """
    ts = int(body.get("scheduled_at") or body.get("observed_at") or 0)  # type: ignore[arg-type]
    return datetime.fromtimestamp(ts, UTC).date()


def load_traversals(start: date, end: date) -> Loaded[Traversal]:
    """Traversals over [start, end], archive first and raw trace for the rest."""
    archived = read_traversal_days(start, end)
    have = {traversal_key(d) for d in archived.provenance}
    missing = missing_days(start, end, have, traversal_key)

    rows = list(archived.traversals)
    raw_days: list[date] = []
    if missing:
        # Contiguity is not assumed: fetch_trace_bodies takes a span, so the
        # missing days are fetched as one range and the ones already served from
        # the archive are filtered back out by date rather than re-derived.
        bodies = fetch_trace_bodies(start_date=min(missing), end_date=max(missing))
        wanted = set(missing)
        kept = [b for b in bodies if utc_day(b) in wanted]
        derived, _stats = traversals_from_trace(kept)
        rows.extend(derived)
        raw_days = sorted(wanted)

    return Loaded(
        rows=rows,
        archived_days=sorted(archived.provenance),
        raw_days=raw_days,
        provenance=archived.provenance,
    )


# How far before an evaluation span to look for windows. Publication dates are
# not work dates: an alert announcing a closure is archived on the day it was
# PUBLISHED, which can be weeks or months before the work runs. Reading only the
# evaluation span's own days silently omits every window announced earlier —
# exactly the omission the nightly job's 88-day alert scan exists to prevent,
# and it would bite hardest on the long-lead closures this measure most wants.
# 180 days covers MTA's longest observed lead with headroom, and the window
# archive is small enough (a few KB a day) that the extra reads are free.
WINDOW_PUBLICATION_LOOKBACK_DAYS = 180


def load_windows(
    start: date,
    end: date,
    *,
    lookback_days: int = WINDOW_PUBLICATION_LOOKBACK_DAYS,
) -> Loaded[Window]:
    """Every announced window KNOWN to us that could touch [start, end].

    Returns windows published from `lookback_days` before `start` through `end`,
    de-duplicated and NOT filtered by overlap — callers apply their own
    `overlaps` test against the span they actually observed, because a window
    known to us is not the same claim as a window we have movement for.

    Raw alert fallback covers only [start, end]: those are the days recent enough
    to be unarchived, and `archive/alerts/` is pruned at 90 days so a 180-day
    lookback could not be served from it anyway. Older publication days come from
    the archive or not at all, which is precisely what the archive is for.
    """
    pub_start = start - timedelta(days=lookback_days)
    archived = read_window_days(pub_start, end)
    have = {window_key(d) for d in archived.provenance}
    missing = missing_days(start, end, have, window_key)

    seen: set[Window] = set(archived.windows)
    raw_days: list[date] = []
    if missing:
        # Fetched one day at a time, not as min..max. A span fetch would pull
        # alert versions from days that ARE archived whenever coverage is
        # non-contiguous, re-parsing them with the CURRENT parser and mixing that
        # output with the archived output for the same publication day — while
        # `provenance` still claimed those days came from the archive. Alert
        # volumes are a couple of hundred objects a day and `missing` is normally
        # one or two days, so the per-day loop costs nothing.
        for day in missing:
            seen.update(
                windows_from_alerts(fetch_alert_bodies(start_date=day, end_date=day))
            )
        raw_days = sorted(missing)

    return Loaded(
        rows=sorted(seen, key=lambda w: (w.start, w.alert_type, sorted(w.routes))),
        archived_days=sorted(archived.provenance),
        raw_days=raw_days,
        provenance=archived.provenance,
    )
