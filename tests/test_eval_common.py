"""Pinning tests for the shared eval seams.

Each of these seams caused a real, verdict-changing bug in a tool that had its
own copy. The point of these tests is that the semantics are now asserted in one
place, so a future edit to the shared definition has to break a test rather than
silently shift a published number.
"""

from __future__ import annotations

from datetime import date, datetime

from training.eval_common import (
    NYC_TZ,
    et_date,
    et_midnight,
    nearest_rank,
    service_night,
)


def _tick(y: int, mo: int, d: int, h: int) -> int:
    """Epoch seconds for a wall-clock ET hour. Fixtures are built in ET because
    every night concept here reads ET."""
    return int(datetime(y, mo, d, h, tzinfo=NYC_TZ).timestamp())


# --- nearest_rank: the integer-product boundary ---


def test_nearest_rank_integer_product_takes_the_lower_rank():
    """The whole bug in one assertion. For n=10, q=0.9 the product is exactly 9,
    so the 1-indexed rank is 9 and the value is the 9th — NOT the maximum, which
    is what int(q*n) selected."""
    ordered = [float(v) for v in range(1, 11)]  # 1..10
    assert nearest_rank(ordered, 0.90) == 9.0
    assert nearest_rank(ordered, 0.10) == 1.0
    assert nearest_rank(ordered, 0.50) == 5.0


def test_nearest_rank_non_integer_product_unchanged():
    """Non-integer products were always right: ceil(q*n)-1 == floor(q*n) there.
    Pins that the fix is confined to the integer boundary."""
    ordered = [float(v) for v in range(1, 8)]  # 1..7
    assert nearest_rank(ordered, 0.90) == 7.0  # ceil(6.3) = 7th
    assert nearest_rank(ordered, 0.10) == 1.0  # ceil(0.7) = 1st
    assert nearest_rank(ordered, 0.50) == 4.0  # ceil(3.5) = 4th


def test_nearest_rank_returns_an_observed_sample_at_the_extremes():
    ordered = [2.0, 4.0, 8.0]
    assert nearest_rank(ordered, 0.0) == 2.0  # clamped to the first rank
    assert nearest_rank(ordered, 1.0) == 8.0  # exactly the last rank
    assert nearest_rank([5.0], 0.90) == 5.0  # single sample


def test_nearest_rank_never_interpolates():
    """A published quantile must be a reading the system actually saw. With a
    gap in the samples, no q may produce a value from inside the gap."""
    ordered = [1.0, 100.0]
    assert nearest_rank(ordered, 0.50) == 1.0
    assert nearest_rank(ordered, 0.51) == 100.0
    for q in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert nearest_rank(ordered, q) in (1.0, 100.0)


def test_nearest_rank_rejects_an_empty_sequence():
    """Withhold rather than fabricate: an empty cell has no observed value, so
    there is nothing honest to return."""
    try:
        nearest_rank([], 0.50)
    except ValueError:
        return
    raise AssertionError("expected ValueError on an empty sequence")


# --- the night seams ---


def test_service_night_rolls_after_midnight_hours_to_the_prior_date():
    """2026-08-15 is a Saturday. Sat 23:00 and Sun 01:00 are ONE service night
    (Saturday's); Mon 01:00 belongs to Sunday's night, not a new Monday one."""
    assert service_night(_tick(2026, 8, 15, 23)) == date(2026, 8, 15)
    assert service_night(_tick(2026, 8, 16, 1)) == date(2026, 8, 15)
    assert service_night(_tick(2026, 8, 17, 1)) == date(2026, 8, 16)
    assert service_night(_tick(2026, 8, 16, 4)) == date(2026, 8, 16)  # 04:00 stays


def test_et_date_splits_at_midnight_where_service_night_does_not():
    """The two definitions differ exactly on the after-midnight hours. This is
    the divergence that let one evening be counted as two independent draws."""
    sat_late = _tick(2026, 8, 15, 23)
    sun_small_hours = _tick(2026, 8, 16, 1)
    assert et_date(sat_late) != et_date(sun_small_hours)
    assert service_night(sat_late) == service_night(sun_small_hours)


def test_et_midnight_is_eastern_not_utc():
    """A UTC-midnight cut would fold the prior ET evening into the wrong window.
    ET midnight must land on ET hour 0 of the requested date."""
    cut = et_midnight(date(2026, 8, 16))
    local = datetime.fromtimestamp(cut, tz=NYC_TZ)
    assert (local.year, local.month, local.day) == (2026, 8, 16)
    assert local.hour == 0
    assert cut == _tick(2026, 8, 16, 0)


def test_night_seams_survive_a_dst_transition():
    """2026-11-01 is the ET fall-back date. The service-night roll and the ET
    midnight cutoff must both stay on the intended calendar date through it,
    since a naive 24-hour arithmetic step lands an hour off."""
    assert service_night(_tick(2026, 11, 1, 1)) == date(2026, 10, 31)
    assert service_night(_tick(2026, 11, 1, 23)) == date(2026, 11, 1)
    local = datetime.fromtimestamp(et_midnight(date(2026, 11, 1)), tz=NYC_TZ)
    assert (local.month, local.day, local.hour) == (11, 1, 0)
