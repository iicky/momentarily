"""Segment-local state forecasters (training/journey_state.py).

The parent's route-wide factor is the thing these forecasters exist to fix,
so the tests pin what changed: only exact-OD spans (or, failing those,
contained spans) move the factor; a span elsewhere on the same route does
nothing. The window, gating, and short-window switches are pinned at their
boundaries.
"""

from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from training.journey_model import FORECASTERS as MODEL_FORECASTERS
from training.journey_model import MIN_N, K
from training.journey_state import FORECASTERS, SegmentStateForecaster
from training.journey_types import QUANTILES, SpanSample

_ET = ZoneInfo("America/New_York")

# A weekday (Wed 2026-08-12) noon departure: tod_bin 3, weekend False.
T = int(
    datetime.combine(datetime(2026, 8, 12).date(), time(12, 0), tzinfo=_ET).timestamp()
)

# Stop order on the fixture route: Q06S -> Q05S -> Q04S -> Q03S -> Q02S.
OD = ("Q", "S", "Q05S", "Q03S")
EDGES = [("Q06S", "Q05S"), ("Q05S", "Q04S"), ("Q04S", "Q03S"), ("Q03S", "Q02S")]


def _span(
    *,
    depart_at: int = T,
    seconds: int = 600,
    route: str = "Q",
    direction: str = "S",
    origin: str = "Q05S",
    destination: str = "Q03S",
    scheduled: int | None = 550,
    hops: int = 2,
    trip: str = "072000_Q..S01R",
) -> SpanSample:
    return SpanSample(
        trip_id=trip,
        route_id=route,
        direction=direction,
        origin=origin,
        destination=destination,
        depart_at=depart_at,
        seconds=seconds,
        n_hops=hops,
        scheduled_sec=scheduled,
        service_day="2026-08-12",
    )


def _train() -> list[SpanSample]:
    """30 spans on the sample's OD (seconds 500..790, clears MIN_N) plus one
    1-hop span per edge (300 s each): the edges define the stop order and
    give each 1-hop OD a pooled normal of 300."""
    spans = [
        _span(depart_at=T - 86400 * (i + 1), seconds=500 + 10 * i) for i in range(30)
    ]
    for o, d in EDGES:
        spans.append(
            _span(depart_at=T - 86400, seconds=300, origin=o, destination=d, hops=1)
        )
    return spans


def _fitted(name: str) -> SegmentStateForecaster:
    f = FORECASTERS[name]()
    assert isinstance(f, SegmentStateForecaster)
    f.fit(_train())
    return f


def _exact(*, ratio: float, normal: float, depart_at: int) -> SpanSample:
    return _span(depart_at=depart_at, seconds=round(ratio * normal))


def test_exact_od_moves_f_and_other_od_on_same_route_does_not() -> None:
    model = _fitted("state_od")
    sample = _span()
    base = model.predict(sample, [])
    assert base is not None
    pooled = model.normal(OD)
    assert pooled is not None
    exact = _exact(ratio=2.0, normal=pooled, depart_at=T - 3000)
    # Same route+direction, but past the sample's destination: not contained.
    other = _span(
        depart_at=T - 2900, seconds=900, origin="Q03S", destination="Q02S", hops=1
    )
    p_exact = model.predict(sample, [exact])
    assert p_exact is not None
    f = p_exact[0.5] / base[0.5]
    expected = math.exp(1 / (1 + K) * math.log(exact.seconds / pooled))
    assert abs(f - expected) < 1e-9
    # Alone, the off-segment span does nothing — the parent would count it.
    assert model.predict(sample, [other]) == base
    # Beside an exact span it still adds nothing.
    assert model.predict(sample, [exact, other]) == p_exact


def test_containment_fallback_counts_inside_spans_only() -> None:
    model = _fitted("state_od")
    sample = _span()
    base = model.predict(sample, [])
    assert base is not None
    # Q04S -> Q03S lies inside Q05S..Q03S; its pooled normal is 300.
    inside = _span(
        depart_at=T - 3000, seconds=600, origin="Q04S", destination="Q03S", hops=1
    )
    # Q06S -> Q05S ends at the sample's origin but starts before it: outside.
    outside = _span(
        depart_at=T - 3000, seconds=600, origin="Q06S", destination="Q05S", hops=1
    )
    p_inside = model.predict(sample, [inside])
    assert p_inside is not None
    expected = math.exp(1 / (1 + K) * math.log(600 / 300))
    assert abs(p_inside[0.5] / base[0.5] - expected) < 1e-9
    assert model.predict(sample, [outside]) == base
    # An exact span pre-empts the fallback entirely.
    pooled = model.normal(OD)
    assert pooled is not None
    exact = _exact(ratio=2.0, normal=pooled, depart_at=T - 2000)
    assert model.predict(sample, [exact, inside]) == model.predict(sample, [exact])


def test_containment_needs_the_same_stopping_pattern() -> None:
    # Geographically inside Q05S..Q03S, but run by a trip on another pattern
    # (another branch, or the local beside an express): it must not count.
    model = _fitted("state_od")
    sample = _span()
    base = model.predict(sample, [])
    other_pattern = _span(
        depart_at=T - 3000,
        seconds=600,
        origin="Q04S",
        destination="Q03S",
        hops=1,
        trip="073000_Q..S02R",
    )
    assert model.predict(sample, [other_pattern]) == base


def test_a_pattern_whose_train_edges_branch_gets_no_containment() -> None:
    model = FORECASTERS["state_od"]()
    assert isinstance(model, SegmentStateForecaster)
    # Q05S has two observed successors on the same pattern: no stop order.
    model.fit(
        [
            *_train(),
            _span(
                depart_at=T - 86400,
                seconds=300,
                origin="Q05S",
                destination="Q14S",
                hops=1,
            ),
        ]
    )
    sample = _span()
    base = model.predict(sample, [])
    inside = _span(
        depart_at=T - 3000, seconds=600, origin="Q04S", destination="Q03S", hops=1
    )
    assert model.predict(sample, [inside]) == base


def test_window_edges_open_lower_closed_upper() -> None:
    model = _fitted("state_od")
    sample = _span()
    base = model.predict(sample, [])
    assert base is not None
    pooled = model.normal(OD)
    assert pooled is not None
    sec = round(2.0 * pooled)
    history = [
        _span(depart_at=T - 3600 - sec, seconds=sec),  # completes T-3600: excluded
        _span(depart_at=T - 3599 - sec, seconds=sec),  # completes T-3599: included
        _span(depart_at=T - sec, seconds=sec),  # completes exactly T: included
        _span(depart_at=T + 1 - sec, seconds=sec),  # completes T+1: excluded
    ]
    model.index_history(sorted(history, key=lambda s: s.depart_at + s.seconds))
    got = model.predict(sample, ())
    assert got is not None
    expected = math.exp(2 / (2 + K) * math.log(sec / pooled))
    assert abs(got[0.5] / base[0.5] - expected) < 1e-9


def test_indexed_history_matches_the_scan() -> None:
    history = [
        _span(depart_at=T - 3000, seconds=900),  # exact OD
        _span(
            depart_at=T - 2500, seconds=600, origin="Q04S", destination="Q03S", hops=1
        ),
        _span(
            depart_at=T - 2000,
            seconds=800,
            route="N",
            origin="R01S",
            destination="R02S",
        ),
        _span(depart_at=T - 50, seconds=100),  # completes after T: excluded
    ]
    sample = _span(seconds=0)
    for name in FORECASTERS:
        scanned, indexed = _fitted(name), _fitted(name)
        want = scanned.predict(sample, history)
        # Deliberately NOT sorted by completion: the index must sort for itself.
        indexed.index_history(list(reversed(history)))
        got = indexed.predict(sample, ())
        assert want is not None
        assert got is not None
        for q in QUANTILES:
            assert abs(got[q] - want[q]) < 1e-9
        if name in ("state_od", "state_od_leaf"):
            assert want != scanned.predict(sample, [])  # the state term moved it


def test_gated_variant_needs_n_and_magnitude() -> None:
    gated, leaf = _fitted("state_od_gated"), _fitted("state_od_leaf")
    sample = _span()
    base = gated.predict(sample, [])
    pooled = gated.normal(OD)
    assert pooled is not None
    slow = [
        _exact(ratio=2.0, normal=pooled, depart_at=T - 3000 + 60 * i) for i in range(3)
    ]
    # n = 2: gate holds f at 1 while the ungated leaf variant moves.
    assert gated.predict(sample, slow[:2]) == base
    assert leaf.predict(sample, slow[:2]) != base
    # n = 3 but |mean log ratio| <= 0.05: still 1.
    mild = [
        _exact(ratio=1.02, normal=pooled, depart_at=T - 3000 + 60 * i) for i in range(3)
    ]
    assert gated.predict(sample, mild) == base
    # n = 3 with a large ratio: identical to the ungated leaf variant.
    assert gated.predict(sample, slow) == leaf.predict(sample, slow)
    assert gated.predict(sample, slow) != base


def test_short_window_ignores_a_span_from_25_minutes_ago() -> None:
    short, leaf = _fitted("state_od_20m"), _fitted("state_od_leaf")
    sample = _span()
    base = short.predict(sample, [])
    pooled = short.normal(OD)
    assert pooled is not None
    sec = round(2.0 * pooled)
    far = _span(depart_at=T - 1500 - sec, seconds=sec)  # completed 1500 s before
    near = _span(depart_at=T - 600 - sec, seconds=sec)  # completed 600 s before
    assert short.predict(sample, [far]) == base
    p_near = short.predict(sample, [near])
    assert p_near != base
    # Inside both windows the 20m variant matches the hour-window leaf one.
    assert p_near == leaf.predict(sample, [near])


def test_leaf_normal_differs_from_pooled_when_the_leaf_does() -> None:
    """Morning spans (bin 2, weekday) at a constant 1200 s give the history
    span's leaf a median of 1200 while the pooled OD median stays far below:
    the leaf-normal variant reads the span as normal, the pooled one as slow."""
    extra: list[SpanSample] = []
    d = date(2026, 8, 11)
    while len(extra) < MIN_N:
        if d.weekday() < 5:
            at = int(datetime.combine(d, time(9, 0), tzinfo=_ET).timestamp())
            extra.append(_span(depart_at=at, seconds=1200))
        d -= timedelta(days=1)
    train = _train() + extra
    pooled_model = FORECASTERS["state_od"]()
    pooled_model.fit(train)
    leaf_model = FORECASTERS["state_od_leaf"]()
    leaf_model.fit(train)
    # Departs 11:10 ET (bin 2, weekday), completes in-window at 1200 s.
    h = _span(depart_at=T - 3000, seconds=1200)
    sample = _span()
    assert leaf_model.predict(sample, [h]) == leaf_model.predict(sample, [])
    assert pooled_model.predict(sample, [h]) != pooled_model.predict(sample, [])


def test_all_variants_abstain_exactly_like_state_hist_only() -> None:
    hist = MODEL_FORECASTERS["state_hist_only"]()
    hist.fit(_train())
    history = [_span(depart_at=T - 3000, seconds=900)]
    samples = [
        _span(),  # answered
        _span(route="Z", origin="Z01S", destination="Z02S"),  # unseen leaf
        _span(origin="Q07S", scheduled=None),  # unseen OD, no schedule fallback
    ]
    for name in FORECASTERS:
        model = _fitted(name)
        for s in samples:
            for h in ([], history):
                assert (model.predict(s, h) is None) == (hist.predict(s, h) is None)


def test_quantiles_on_fixed_grid_and_non_decreasing() -> None:
    history = [_span(depart_at=T - 3000, seconds=900)]
    for name in FORECASTERS:
        model = _fitted(name)
        for h in ([], history):
            p = model.predict(_span(), h)
            assert p is not None
            assert tuple(p.keys()) == QUANTILES
            values = [p[q] for q in QUANTILES]
            assert values == sorted(values)
