"""Shared seams for the offline evaluation and baseline tools.

WHY THIS MODULE EXISTS. Four independently-written eval tools grew their own
copies of the same handful of primitives, and every copy drifted. One day's
review pass caught four real bugs, all of them in the same four seam classes:

  1. ET-vs-UTC window cutoffs, and what counts as ONE night
  2. the clustering unit a bootstrap resamples (nights, not ticks)
  3. label-coverage semantics (an absent row means "quiet" or "gap"?)
  4. witness construction (a reconstructed activity tick is NOT proof the
     archive was live)

Two of those bugs reversed a headline verdict. The lesson is that these are not
tool-local details: they are the seams where an eval silently stops measuring
what it claims to. So they live here once, with their semantics documented and
pinned by tests, and the tools import them.

WHAT DOES *NOT* BELONG HERE. Each tool's choice of analysis unit stays with the
tool. A wait-time eval clusters by night; an episode-scored detector clusters by
episode. Those are different questions, and unifying them would be a silent
behaviour change dressed up as a refactor. What is shared is the DEFINITION of a
night and of a witness, not the decision about which unit to count.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, timedelta
from math import ceil
from typing import Any
from zoneinfo import ZoneInfo

from training.load import TICK_SECONDS

NYC_TZ = ZoneInfo("America/New_York")

# ET hours after midnight that belong to the PRIOR date's service night. The
# subway runs continuously across midnight, so a calendar date is the wrong
# boundary for anything reasoning about a night of service: it splits one
# evening in half and files the halves under different labels.
AFTER_MIDNIGHT_HOURS: frozenset[int] = frozenset({0, 1, 2, 3})


def et_date(epoch_seconds: int) -> date:
    """The America/New_York CALENDAR date a tick falls on.

    The zone schedule_bin/tod_bin already bucket by, so a cell and its night
    agree. Use this only where the calendar date is genuinely the unit wanted
    (e.g. a per-hour cell, where each date contributes exactly one observation
    of that hour). For anything that reasons about a night of service as a
    whole, use `service_night` instead — see its note.
    """
    return datetime.fromtimestamp(epoch_seconds, tz=NYC_TZ).date()


def service_night(epoch_seconds: int) -> date:
    """The ET SERVICE night a tick belongs to: its ET date, rolled back one day
    when the ET hour is 0-3.

    So Sat 23:00 and Sun 01:00 are one night (Saturday's), and Mon 01:00 belongs
    to Sunday's night rather than starting a bogus Monday one. This is the
    correct resampling and labelling unit whenever observations from several
    hours of the same evening are pooled: keying such a pool by calendar date
    splits one night into two, which then get counted as two independent draws.
    That is the pseudo-replication that inflated a false-alarm comparison by 7x
    until it was corrected to nothing.
    """
    dt = datetime.fromtimestamp(epoch_seconds, tz=NYC_TZ)
    d = dt.date()
    return d - timedelta(days=1) if dt.hour in AFTER_MIDNIGHT_HOURS else d


def et_midnight(d: date) -> int:
    """Epoch seconds at ET midnight starting date `d`.

    Must be ET, not UTC: every night concept here is ET, and a UTC-midnight cut
    would fold 20:00-23:59 ET of the PRIOR date — exactly the late-night band
    most of these evals study — into the wrong window.
    """
    return int(datetime(d.year, d.month, d.day, tzinfo=NYC_TZ).timestamp())


def nearest_rank(ordered: Sequence[float], q: float) -> float:
    """The nearest-rank q-quantile of an already-sorted sequence.

    Nearest-rank means the returned value is always an OBSERVED sample, never
    interpolated between two of them — so a published quantile is a reading the
    system actually saw.

    The definition is 1-indexed rank ceil(q*n), i.e. 0-indexed ceil(q*n)-1. That
    `-1` is the whole point: `int(q*n)` agrees with it for every non-integer
    product and is one rank TOO HIGH exactly when q*n lands on an integer. For
    n=10, q=0.9 that is the difference between the 9th value and the maximum,
    which biases a p90 threshold outward and understates every false-alarm rate
    measured against it. Both earlier copies of this helper had that bug.

    `round(q*n, 9)` snaps a product that is an integer in exact arithmetic but a
    hair above it in binary floating point, which would otherwise re-introduce
    the same off-by-one through the ceiling. (Measured: no such case arises for
    the q values this codebase actually uses, at any n up to 200k — the snap is
    here so that stays true of a q someone adds later.)
    """
    n = len(ordered)
    if n == 0:
        raise ValueError("nearest_rank on an empty sequence")
    idx = min(n - 1, max(0, ceil(round(q * n, 9)) - 1))
    return ordered[idx]


# --- liveness witnesses ---
#
# A witness answers "was the archive actually recording when this looked calm?"
# It must be built from the observed_at of a body we really fetched. A
# reconstructed activity tick is NOT a witness: those are interpolated from
# active_periods and happily span a collection gap, so using them makes an
# outage look like quiet service — the exact failure that capped a whole eval's
# labels at night granularity.


def snap_tick(epoch_seconds: int) -> int:
    """Floor an observed_at to its 5-minute tick, the grid every archive body and
    every truth map is keyed on."""
    return (epoch_seconds // TICK_SECONDS) * TICK_SECONDS


def snapshot_tick_witness(bodies: Iterable[Mapping[str, Any]]) -> set[int]:
    """The ticks a fetched body actually exists for — proof the cron ran.

    Built from each body's own observed_at, so membership means "this tick was
    collected", not "this tick was inferred".
    """
    return {snap_tick(int(b.get("observed_at") or 0)) for b in bodies}


def alert_night_witness(bodies: Iterable[Mapping[str, Any]]) -> set[date]:
    """The service nights with any alert-version body archived system-wide —
    proof the ALERTS fetch was live, which the cron witness cannot establish.

    The alerts feed is fetched in a separate request from the trip-updates cron,
    and it has a stale-fallback: on failure the prior value is carried forward
    and nothing distinguishes it from a genuinely quiet tick. So cron liveness
    alone would read an alerts outage as calm service. A night with a witnessed
    cron and no alert archived anywhere on the system is an outage, not silence:
    NYC nights are never system-wide alert-free given the standing planned
    advisories.
    """
    return {service_night(snap_tick(int(b.get("observed_at") or 0))) for b in bodies}
