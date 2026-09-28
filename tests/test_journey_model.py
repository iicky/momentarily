"""State-conditioned journey forecasters (training/journey_model.py).

Synthetic SpanSamples only — no R2, no trace. Each case pins one rule of the
predict contract: causality against a polluted history, neutral state on empty
history, shrunken and clamped state factors, abstention below MIN_N, identical
abstention sets between "state" and its history-only ablation, and the fixed
quantile grid.
"""

from __future__ import annotations

from datetime import datetime, time
from zoneinfo import ZoneInfo

from training.journey_model import (
    F_MAX,
    FORECASTERS,
    MIN_N,
    K,
    StateConditionedForecaster,
)
from training.journey_types import QUANTILES, SpanSample

_ET = ZoneInfo("America/New_York")

# A weekday (Wed 2026-08-12) noon departure: tod_bin 3, weekend False.
T = int(
    datetime.combine(datetime(2026, 8, 12).date(), time(12, 0), tzinfo=_ET).timestamp()
)

OD = ("Q", "S", "Q05S", "Q03S")


def _span(
    *,
    depart_at: int = T,
    seconds: int = 600,
    route: str = "Q",
    direction: str = "S",
    origin: str = "Q05S",
    destination: str = "Q03S",
    scheduled: int | None = 550,
) -> SpanSample:
    return SpanSample(
        trip_id="072000_Q..S01R",
        route_id=route,
        direction=direction,
        origin=origin,
        destination=destination,
        depart_at=depart_at,
        seconds=seconds,
        n_hops=2,
        scheduled_sec=scheduled,
        service_day="2026-08-12",
    )


def _train() -> list[SpanSample]:
    """30 spans on one leaf, seconds 500..790 — clears MIN_N with room. One
    extra short OD (normal 20 s) exists only to anchor high state ratios that
    still complete inside the trailing window."""
    spans = [
        _span(depart_at=T - 86400 * (i + 1), seconds=500 + 10 * i) for i in range(30)
    ]
    spans.append(
        _span(depart_at=T - 86400, seconds=20, origin="Q09S", destination="Q08S")
    )
    return spans


def _fitted(name: str = "state") -> StateConditionedForecaster:
    f = FORECASTERS[name]()
    assert isinstance(f, StateConditionedForecaster)
    f.fit(_train())
    return f


def _hist_span(*, ratio: float, normal: float, depart_at: int) -> SpanSample:
    return _span(depart_at=depart_at, seconds=round(ratio * normal))


def test_causality_future_span_in_history_is_ignored() -> None:
    model = _fitted()
    sample = _span()
    hist_only = _fitted("state_hist_only").predict(sample, [])
    assert hist_only is not None
    normal = hist_only[0.5]
    legit = _hist_span(ratio=1.5, normal=normal, depart_at=T - 3000)
    base = model.predict(sample, [legit])
    # Departs before the sample but COMPLETES after it: must change nothing,
    # whatever order the caller appends it in.
    future = _span(depart_at=T - 10, seconds=5000)
    assert model.predict(sample, [legit, future]) == base
    assert model.predict(sample, [future, legit]) == base


def test_state_factor_is_one_on_empty_history() -> None:
    state, hist = _fitted(), _fitted("state_hist_only")
    sample = _span()
    p_state, p_hist = state.predict(sample, []), hist.predict(sample, [])
    assert p_state is not None
    assert p_hist is not None
    for q in QUANTILES:
        assert abs(p_state[q] - p_hist[q]) < 1e-9


def test_state_factor_rises_with_slow_history_and_is_clamped() -> None:
    state, hist = _fitted(), _fitted("state_hist_only")
    sample = _span()
    hist_pred = hist.predict(sample, [])
    assert hist_pred is not None
    normal = hist_pred[0.5]
    one_slow = [_hist_span(ratio=2.0, normal=normal, depart_at=T - 3000)]
    one_slow_pred = state.predict(sample, one_slow)
    assert one_slow_pred is not None
    f = one_slow_pred[0.5] / normal
    # exp(1/(1+K) * ln 2), shrunk toward 1 by thin support.
    assert abs(f - 2.0 ** (1 / (1 + K))) < 1e-6
    assert 1.0 < f < 2.0
    # Ratio 50 on the short OD (normal 20 s): each span completes in-window,
    # raw f = 50**(20/25) >> F_MAX.
    absurd = [
        _span(
            depart_at=T - 3000 + 60 * i, seconds=1000, origin="Q09S", destination="Q08S"
        )
        for i in range(20)
    ]
    # Well over the clamp, so f lands exactly on F_MAX.
    absurd_pred = state.predict(sample, absurd)
    assert absurd_pred is not None
    f_clamped = absurd_pred[0.5] / normal
    assert abs(f_clamped - F_MAX) < 1e-9


def test_abstains_without_leaf_support() -> None:
    model = FORECASTERS["state"]()
    # Under MIN_N everywhere, and no scheduled_sec to fall back on.
    model.fit(
        [_span(depart_at=T - 86400 * i, scheduled=None) for i in range(MIN_N - 1)]
    )
    assert model.predict(_span(), []) is None
    # A route the train never saw abstains too, whatever the history says.
    fitted = _fitted()
    assert (
        fitted.predict(_span(route="Z", origin="Z01S", destination="Z02S"), []) is None
    )


def test_state_and_hist_only_abstain_identically() -> None:
    state, hist = _fitted(), _fitted("state_hist_only")
    hist_pred = hist.predict(_span(), [])
    assert hist_pred is not None
    normal = hist_pred[0.5]
    history = [_hist_span(ratio=1.4, normal=normal, depart_at=T - 3000)]
    samples = [
        _span(),  # answered
        _span(route="Z", origin="Z01S", destination="Z02S"),  # unseen leaf
        _span(origin="Q07S", scheduled=None),  # unseen OD, no schedule fallback
    ]
    for s in samples:
        assert (state.predict(s, history) is None) == (hist.predict(s, history) is None)


def test_quantiles_on_fixed_grid_and_non_decreasing() -> None:
    model = _fitted()
    sample = _span()
    base_pred = model.predict(sample, [])
    assert base_pred is not None
    normal = base_pred[0.5]
    for history in ([], [_hist_span(ratio=1.7, normal=normal, depart_at=T - 3000)]):
        p = model.predict(sample, history)
        assert p is not None
        assert tuple(p.keys()) == QUANTILES
        values = [p[q] for q in QUANTILES]
        assert values == sorted(values)


def test_indexed_history_matches_the_scan_including_both_window_edges() -> None:
    """index_history must be observably identical to the per-sample scan: same
    route filter, open lower bound, closed upper bound, other routes ignored."""
    scanned = StateConditionedForecaster()
    scanned.fit(_train())
    indexed = StateConditionedForecaster()
    indexed.fit(_train())
    history = [
        _span(
            depart_at=T - 3600 - 100, seconds=100
        ),  # completes exactly T-3600: excluded
        _span(depart_at=T - 3600 - 99, seconds=100),  # completes T-3599: included
        _span(depart_at=T - 1000, seconds=900),  # ratio > 1, included
        _span(depart_at=T - 800, seconds=800, route="N"),  # other route: ignored
        _span(depart_at=T - 100, seconds=100),  # completes exactly T: included
        _span(depart_at=T - 50, seconds=100),  # completes after T: excluded
    ]
    sample = _span(seconds=0)
    want = scanned.predict(sample, history)
    # Deliberately NOT sorted by completion: the index must sort for itself.
    indexed.index_history(list(reversed(history)))
    got = indexed.predict(sample, ())
    assert want is not None
    assert got is not None
    assert got != scanned.predict(sample, [])  # the state term actually moved it
    for q in QUANTILES:
        assert abs(got[q] - want[q]) < 1e-9
