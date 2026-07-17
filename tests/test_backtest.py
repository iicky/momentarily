from datetime import UTC, date, datetime

from training.backtest import compute_window, grade_recovery_timing
from training.eval import TICK_SECONDS, TransitionRecord


def _now(y: int, m: int, d: int, h: int = 0) -> datetime:
    return datetime(y, m, d, h, tzinfo=UTC)


def _midnight(y: int, m: int, d: int) -> int:
    return int(datetime(y, m, d, tzinfo=UTC).timestamp())


def test_live_window_anchors_on_today() -> None:
    now = _now(2026, 7, 16, 14)
    w = compute_window(eval_days=5, train_days=6, eval_end=None, now=now)
    assert w.eval_end == date(2026, 7, 16)
    assert w.eval_start == date(2026, 7, 12)
    assert w.train_end == date(2026, 7, 11)
    assert w.train_start == date(2026, 7, 6)
    assert w.warmup_start == date(2026, 7, 11)
    assert not w.is_historical
    # live: truth stops at eval_end, outcomes capped at wall-clock now (not
    # end-of-day) so futures that have not elapsed are never scored.
    assert w.truth_end == date(2026, 7, 16)
    assert w.outcome_bound_epoch == int(now.timestamp())


def test_eval_end_equal_today_stays_live() -> None:
    now = _now(2026, 7, 16, 9)
    w = compute_window(eval_days=3, train_days=6, eval_end=date(2026, 7, 16), now=now)
    assert not w.is_historical
    assert w.truth_end == date(2026, 7, 16)
    assert w.outcome_bound_epoch == int(now.timestamp())


def test_historical_window_extends_truth_and_bounds_ticks() -> None:
    now = _now(2026, 7, 16, 14)
    w = compute_window(eval_days=9, train_days=14, eval_end=date(2026, 6, 18), now=now)
    assert w.eval_start == date(2026, 6, 10)
    assert w.eval_end == date(2026, 6, 18)
    assert w.train_end == date(2026, 6, 9)
    assert w.train_start == date(2026, 5, 27)
    assert w.warmup_start == date(2026, 6, 9)
    assert w.is_historical
    # truth loaded one day past eval_end so the last day's +max-horizon futures
    # resolve to real outcomes instead of defaulting normal.
    assert w.truth_end == date(2026, 6, 19)
    # scored ticks bounded to end of eval_end (midnight 06-19), dropping any
    # reconstruction spillover synthesized into 06-19.
    assert w.eval_end_epoch == _midnight(2026, 6, 19)
    # outcomes scored through the end of the loaded truth day (midnight 06-20).
    assert w.outcome_bound_epoch == _midnight(2026, 6, 20)


def test_historical_bounds_are_independent_of_wall_clock() -> None:
    early = compute_window(9, 14, date(2026, 6, 18), now=_now(2026, 7, 16, 2))
    late = compute_window(9, 14, date(2026, 6, 18), now=_now(2026, 8, 1, 23))
    assert early.outcome_bound_epoch == late.outcome_bound_epoch
    assert early.eval_end_epoch == late.eval_end_epoch
    # the historical cutoff never leaks the current time.
    assert early.outcome_bound_epoch < int(_now(2026, 7, 16, 2).timestamp())


def test_epoch_bounds_are_midnight_utc() -> None:
    w = compute_window(9, 14, date(2026, 6, 18), now=_now(2026, 7, 16, 14))
    assert w.eval_start_epoch == _midnight(2026, 6, 10)
    # train_end_epoch is the exclusive upper bound: midnight after train_end,
    # which coincides with eval_start's midnight.
    assert w.train_end_epoch == _midnight(2026, 6, 10)
    assert w.eval_start_epoch == w.train_end_epoch


def _disrupted_run(
    truth: dict[tuple[str, int], str],
    types: dict[tuple[str, int], tuple[str, ...]],
    route: str,
    onset: int,
    n: int,
) -> None:
    for i in range(n):
        tick = onset + i * TICK_SECONDS
        truth[(route, tick)] = "disrupted"
        types[(route, tick)] = ("Delays",)


def _completed_disrupted_transitions(base: int) -> list[TransitionRecord]:
    # Six completed disrupted->normal transitions so a dwell curve exists at the
    # (route, state) and pooled levels.
    return [
        TransitionRecord(
            ts=base - 100_000 + i * 10_000,
            route="A",
            prev_state="disrupted",
            new_state="normal",
            regime_entered_at=base - 100_000 + i * 10_000 - 600 * (i + 1),
            exited_at=base - 100_000 + i * 10_000,
            dwell_sec=600 * (i + 1),
            alert_type_at_entry="Delays",
        )
        for i in range(6)
    ]


def test_recovery_timing_scores_only_onset_in_window() -> None:
    base = (1_700_000_000 // TICK_SECONDS) * TICK_SECONDS
    eval_start = base
    eval_end = base + 24 * TICK_SECONDS  # 2h eval window
    window_end = base + 48 * TICK_SECONDS  # +2h recovery tail
    truth: dict[tuple[str, int], str] = {}
    types: dict[tuple[str, int], tuple[str, ...]] = {}
    # Onset 1h into the window, recovers 30min later -> uncensored, scored.
    _disrupted_run(truth, types, "A", base + 12 * TICK_SECONDS, 6)
    # Onset past eval_end (in the tail) -> extracted but filtered out.
    _disrupted_run(truth, types, "A", base + 30 * TICK_SECONDS, 4)
    rec = grade_recovery_timing(
        _completed_disrupted_transitions(base),
        truth,
        types,
        train_end_epoch=base,
        eval_start_epoch=eval_start,
        eval_end_epoch=eval_end,
        window_end_epoch=window_end,
    )
    assert rec["n_eval_episodes"] == 1
    accounted = rec["n_scored"] + rec["n_censored_excluded"] + rec["n_no_curve"]
    assert accounted == 1
    assert rec["n_scored"] == 1


def test_recovery_timing_excludes_right_censored_incidents() -> None:
    base = (1_700_000_000 // TICK_SECONDS) * TICK_SECONDS
    eval_start = base
    eval_end = base + 24 * TICK_SECONDS
    window_end = base + 48 * TICK_SECONDS
    truth: dict[tuple[str, int], str] = {}
    types: dict[tuple[str, int], tuple[str, ...]] = {}
    # Onset in-window but never recovers through the loaded tail -> right-censored.
    _disrupted_run(truth, types, "A", base + 12 * TICK_SECONDS, 40)
    rec = grade_recovery_timing(
        _completed_disrupted_transitions(base),
        truth,
        types,
        train_end_epoch=base,
        eval_start_epoch=eval_start,
        eval_end_epoch=eval_end,
        window_end_epoch=window_end,
    )
    assert rec["n_eval_episodes"] == 1
    assert rec["n_scored"] == 0
    assert rec["n_censored_excluded"] == 1


def test_recovery_timing_empty_without_incidents() -> None:
    base = (1_700_000_000 // TICK_SECONDS) * TICK_SECONDS
    rec = grade_recovery_timing(
        [],
        {},
        {},
        train_end_epoch=base,
        eval_start_epoch=base,
        eval_end_epoch=base + 24 * TICK_SECONDS,
        window_end_epoch=base + 48 * TICK_SECONDS,
    )
    assert rec["n_eval_episodes"] == 0
    assert rec["n_scored"] == 0
