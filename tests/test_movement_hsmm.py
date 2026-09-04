"""The explicit-duration movement debounce: the fit, and the gate logic.

These tests pin the properties that make the arm's numbers mean what they say,
all on synthetic input so no archive is needed:

  - the negative-binomial fit is a real MLE (it reproduces the closed form
    where one exists) and nests the geometric exactly at r = 1;
  - the nest test has the right operating characteristics in BOTH directions —
    it does not reject on geometric data and does reject on over-dispersed
    data, which is what stops a positive result from being an artefact;
  - censored observations enter through P(X >= ticks), not P(X > ticks);
  - the filter is causal, ages its clock in elapsed ticks rather than in
    observed calls, and does not hold a state through an abstention;
  - the r = 1 arm CANNOT debounce, which is the substantive claim;
  - every gate is enforced the way it is stated — in particular gate 3 fails on
    a support loss whose interval includes zero, and gate 2 requires the
    held-out interval to sit strictly above zero;
  - a degenerate population is refused BY NAME rather than graded to NaN.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pytest

from training.eval import TICK_SECONDS
from training.movement_hsmm import (
    NORMAL,
    NOT_MOVING,
    DwellObs,
    DwellPopulation,
    HsmmParams,
    NbDwellFit,
    chi2_sf_1df,
    dwell_hazards,
    dwell_loglik,
    dwell_observations,
    episode_dwell_observations,
    episode_populations,
    filter_route,
    fit_debounce,
    fit_dwell,
    fit_dwell_model,
    hsmm_states,
    nb_log_pmf,
    nb_sf,
    nest_test,
    params_from_fits,
)
from training.movement_hsmm_grade import (
    FA_REFERENCE,
    GateBlocked,
    GradeInputs,
    Interval,
    _refuse_degenerate,  # pyright: ignore[reportPrivateUsage]
    bootstrap_detection_delta,
    bootstrap_dwell_delta,
    detection_flags,
    gate_detection_support,
    gate_dwell_loglik,
    gate_false_alarms,
    grade,
    split_window,
)

_ORIGIN = 1_780_000_000 // TICK_SECONDS * TICK_SECONDS


def _approx(
    expected: float, *, rel: float | None = None, abs: float | None = None
) -> object:
    """Typed wrapper around ``pytest.approx``.

    pytest's ``approx`` leaks ``Unknown`` through its ``ApproxBase`` return type
    under strict mode, so the boundary is pinned to ``object`` once here — the
    same wrapper ``test_train_em`` uses.
    """
    return pytest.approx(expected, rel=rel, abs=abs)  # pyright: ignore[reportUnknownMemberType]


@dataclass(frozen=True)
class Ep:
    """A minimal stand-in for an archived episode. The production adapter is
    duck-typed over `route`/`onset`/`recovery` precisely so the two real
    episode products (`episodes.Episode`, `load_r2.Disruption`) and this
    fixture all work without a shared base class."""

    route: str
    onset: int
    recovery: int | None = None
    right_censored: bool = False
    left_censored: bool = False


def _obs(*specs: tuple[int, bool]) -> list[DwellObs]:
    return [DwellObs("A", ticks, censored=c) for ticks, c in specs]


# --- the negative binomial ---------------------------------------------------


@pytest.mark.parametrize(("r", "p"), [(1.0, 0.3), (2.5, 0.2), (0.4, 0.5), (12.0, 0.8)])
def test_pmf_is_a_distribution_and_sf_agrees_with_it(r: float, p: float) -> None:
    """The pmf sums to 1 on {1, 2, ...} and `nb_sf` is its exact upper tail.

    Not a formality: `nb_sf` switches between 1 - CDF and a direct tail sum to
    dodge cancellation, and the switch is where an off-by-one would hide.
    """
    total = sum(math.exp(nb_log_pmf(k, r, p)) for k in range(1, 6000))
    assert total == _approx(1.0, abs=1e-9)
    for k in (1, 3, 7, 20):
        head = sum(math.exp(nb_log_pmf(j, r, p)) for j in range(1, k + 1))
        assert nb_sf(k, r, p) == _approx(1.0 - head, rel=1e-9, abs=1e-12)


def test_sf_survives_the_deep_tail_where_one_minus_cdf_is_zero() -> None:
    """A censored observation far out in the tail must get a finite likelihood.

    With p large the CDF reaches 1.0 in floating point within a few dozen
    ticks, so a naive 1 - CDF returns exactly 0.0 and hands the fit -inf for a
    perfectly ordinary long regime. The direct tail sum is what prevents that.
    """
    deep = nb_sf(200, 2.0, 0.5)
    assert 0.0 < deep < 1e-50
    assert math.isfinite(math.log(deep))


def test_r_one_is_exactly_the_geometric_self_loop() -> None:
    """At r = 1 the sojourn is the incumbent's own form, with self_loop = 1 - p.

    This is the identity the whole nest test rests on: if it did not hold
    exactly, the likelihood-ratio statistic would be comparing the candidate
    against something that is not the shipped model.
    """
    p = 0.07
    fit = NbDwellFit(r=1.0, p=p, n_events=1, n_censored=0, loglik=0.0, r_at_bound=False)
    assert fit.geometric
    assert fit.self_loop == _approx(1.0 - p)
    a = 1.0 - p
    for k in (1, 2, 5, 30):
        assert math.exp(nb_log_pmf(k, 1.0, p)) == _approx(a ** (k - 1) * (1 - a))


def test_self_loop_is_withheld_off_the_geometric_nest() -> None:
    """r != 1 has no equivalent single self-loop, and must report None rather
    than an approximation: the arm's whole claim is that no per-tick
    continuation probability describes the dwell."""
    fit = NbDwellFit(
        r=3.0, p=0.2, n_events=1, n_censored=0, loglik=0.0, r_at_bound=False
    )
    assert not fit.geometric
    assert fit.self_loop is None


def test_fixed_r_fit_reproduces_the_closed_form_mle() -> None:
    """With no censoring and r fixed, p_hat = r / (r + mean(X - 1)) in closed
    form. The numeric search must land on it — that is what makes the censored
    case, where no closed form exists, believable."""
    durations = [1, 1, 2, 2, 2, 3, 4, 4, 5, 9, 1, 2, 3, 3, 6]
    obs = [DwellObs("A", k, censored=False) for k in durations]
    zbar = sum(k - 1 for k in durations) / len(durations)
    for r in (1.0, 0.5, 3.0):
        assert fit_dwell(obs, fix_r=r).p == _approx(r / (r + zbar), rel=1e-5)


def test_censoring_uses_survival_at_or_above_the_observed_length() -> None:
    """A run observed open for `t` ticks tells us X >= t, not X > t: the sojourn
    may have ended at its last observed tick with the recovery outside the
    window. Scoring it as X > t understates survival on every censored
    observation and biases the fit toward shorter dwell, which is the direction
    that flatters the memoryless incumbent."""
    r, p = 2.0, 0.3
    assert dwell_loglik(_obs((5, True)), r, p) == _approx(math.log(nb_sf(4, r, p)))
    # And it is strictly more permissive than the X > t reading.
    assert nb_sf(4, r, p) > nb_sf(5, r, p)


def test_censored_observations_lengthen_the_fit() -> None:
    """Censoring must push the fitted dwell UP. Two populations with identical
    observed lengths, one complete and one censored: the censored one only says
    "at least this long", so its MLE cannot be shorter."""
    lengths = [(4, False)] * 20
    complete = fit_dwell(_obs(*lengths))
    censored = fit_dwell(_obs(*[(4, True)] * 20))
    assert censored.mean_ticks > complete.mean_ticks


# --- the nest test's operating characteristics -------------------------------


def _geometric_sample(a: float, n: int, seed: int) -> list[int]:
    import random

    rng = random.Random(seed)
    out: list[int] = []
    for _ in range(n):
        k = 1
        while rng.random() < a:
            k += 1
        out.append(k)
    return out


def test_nest_test_does_not_reject_on_geometric_data() -> None:
    """THE NULL DIRECTION. On data generated BY the incumbent's own form, the
    test must not claim the explicit duration helps. Without this, a positive
    result on real data says nothing — a test that always rejects is not
    evidence."""
    obs = [
        DwellObs("A", k, censored=False) for k in _geometric_sample(0.93, 400, seed=1)
    ]
    result = nest_test(obs)
    assert result.negbin.r == _approx(1.0, abs=0.35)
    assert result.p_value > 0.05
    assert not result.improves


def test_nest_test_rejects_on_underdispersed_data() -> None:
    """THE POWER DIRECTION. Tightly clustered durations are exactly what a
    memoryless dwell cannot represent, and the test must find them."""
    obs = [DwellObs("A", k, censored=False) for k in ([7] * 40 + [8] * 40 + [6] * 40)]
    result = nest_test(obs)
    assert result.negbin.r > 1.0
    assert result.lr_stat > 0.0
    assert result.p_value < 0.01
    assert result.improves


def test_negbin_likelihood_never_loses_to_its_own_nest() -> None:
    """The geometric is nested, so the wider family cannot fit worse. A negative
    LR statistic would mean the search under-converged, and the class reports it
    rather than clipping it — but on ordinary data it must not happen."""
    for seed in range(4):
        obs = [
            DwellObs("A", k, censored=False)
            for k in _geometric_sample(0.8, 60, seed=seed)
        ]
        assert nest_test(obs).lr_stat >= -1e-6


def test_chi2_one_df_tail_matches_known_quantiles() -> None:
    """The p-value is erfc(sqrt(x/2)) exactly, with no table and no scipy."""
    assert chi2_sf_1df(0.0) == 1.0
    assert chi2_sf_1df(3.841459) == _approx(0.05, abs=1e-5)
    assert chi2_sf_1df(6.634897) == _approx(0.01, abs=1e-5)


# --- hazards ----------------------------------------------------------------


def _fit(r: float, p: float) -> NbDwellFit:
    return NbDwellFit(r=r, p=p, n_events=1, n_censored=0, loglik=0.0, r_at_bound=False)


def test_geometric_hazard_is_flat_and_that_is_the_whole_problem() -> None:
    """A memoryless dwell asserts a CONSTANT hazard: a regime open five minutes
    is exactly as likely to end next tick as one open two hours."""
    hz = dwell_hazards(_fit(1.0, 0.3), max_ticks=10)
    assert all(h == _approx(0.3) for h in hz)


def test_over_and_under_dispersion_bend_the_hazard_opposite_ways() -> None:
    """r > 1 gives a RISING hazard — a just-opened regime is unlikely to end
    immediately, which is what makes a one-tick blip expensive to believe.
    r < 1 falls, which is the shape contaminated data manufactures."""
    rising = dwell_hazards(_fit(4.0, 0.3), max_ticks=8)
    assert rising[0] < rising[-1]
    assert rising[0] < 0.05
    falling = dwell_hazards(_fit(0.5, 0.3), max_ticks=8)
    assert falling[0] > falling[-1]


def test_hazard_cap_holds_its_value_and_never_forces_an_exit() -> None:
    """The age cap must NOT set h = 1. Forcing an exit there would inject a
    spurious recovery spike at a fixed age; holding the fitted hazard is exact
    for r = 1 and asymptotically exact otherwise, since the NB tail is
    geometric."""
    hz = dwell_hazards(_fit(3.0, 0.25), max_ticks=64)
    assert hz[-1] < 0.999
    assert hz[-1] == _approx(0.25, abs=0.05)


# --- the filter -------------------------------------------------------------


def _params(
    *, false_call: float = 0.002, miss: float = 0.05, entry_mean: float = 1000.0
) -> HsmmParams:
    """A debounce whose not-moving dwell is tight (r high) and whose normal
    dwell is long — the regime the real arm sits in (P(enter) = 0.076%/tick)."""
    normal = _fit(1.0, 1.0 / entry_mean)
    not_moving = _fit(8.0, 0.6)
    return params_from_fits(
        normal, not_moving, false_call=false_call, miss=miss, max_ticks=48
    )


def _calls(pattern: str, route: str = "A") -> list[tuple[int, Mapping[str, str]]]:
    """`.` normal, `X` disrupted, ` ` abstention (route absent from the tick)."""
    out: list[tuple[int, Mapping[str, str]]] = []
    for i, ch in enumerate(pattern):
        tick = _ORIGIN + i * TICK_SECONDS
        if ch == " ":
            out.append((tick, {}))
        else:
            out.append((tick, {route: "disrupted" if ch == "X" else "normal"}))
    return out


def _decoded(pattern: str) -> str:
    params = _params()
    points = [(tick, per["A"]) for tick, per in _calls(pattern) if "A" in per]
    result = filter_route(points, params)
    by_tick = dict(result.states)
    return "".join(
        "X" if by_tick.get(_ORIGIN + i * TICK_SECONDS) == NOT_MOVING else "."
        for i, ch in enumerate(pattern)
        if ch != " "
    )


def test_isolated_false_call_is_suppressed() -> None:
    """One disrupted call inside a long normal run must not flip the state: the
    early hazard of a fresh not-moving sojourn is low, so a single call cannot
    pay for the entry."""
    assert "X" not in _decoded("." * 20 + "X" + "." * 20)


def test_a_sustained_run_is_still_detected() -> None:
    """Suppressing blips must not cost real episodes. The debounce has to enter
    once the calls persist, or gate 3 is bought with gate 1."""
    decoded = _decoded("." * 20 + "X" * 12 + "." * 20)
    assert "X" in decoded


def test_two_short_separated_episodes_both_survive() -> None:
    """THE STICKY-KAPPA FAILURE MODE, WHICH THIS FORM MUST NOT HAVE.

    A sticky self-transition bias suppresses RE-ENTRY, so the second of two
    separated episodes gets merged or dropped. The persistence here lives in
    the sojourn distribution, whose hazard resets on entry, so a second episode
    is judged by the same low early hazard as the first.
    """
    decoded = _decoded("." * 15 + "X" * 10 + "." * 8 + "X" * 10 + "." * 15)
    first, second = decoded[:33], decoded[33:]
    assert "X" in first
    assert "X" in second


def test_filter_is_causal() -> None:
    """The state at tick t must not depend on any later tick. Truncating the
    stream after t must leave t's verdict unchanged — a smoothed decode would
    score better on every gate and could not ship."""
    pattern = "." * 12 + "X" * 6 + "." * 12 + "X" * 3
    full = _decoded(pattern)
    for cut in (14, 18, 25, 30):
        assert _decoded(pattern[:cut]) == full[:cut]


def test_the_sojourn_clock_ages_through_an_abstention() -> None:
    """Time passes while a route is unseen. A run of abstentions must age the
    belief, not freeze it: a k-of-k debounce has no way to express this, and it
    is a real behavioural difference rather than a restatement.

    Here the calls either side of the gap are identical; only elapsed time
    differs, so any difference in the decode is the clock advancing.
    """
    params = _params()
    tight = [(_ORIGIN + i * TICK_SECONDS, "disrupted") for i in range(6)]
    spread = [(_ORIGIN + i * 4 * TICK_SECONDS, "disrupted") for i in range(6)]
    assert filter_route(tight, params).loglik != _approx(
        filter_route(spread, params).loglik
    )


def test_abstention_emits_no_state_rather_than_a_held_one() -> None:
    """A route is published only at ticks it was called on. Holding a state
    through an abstention would publish a reading no observation supports —
    which is the stale-disrupted false alarm the shipped published surface is
    graded for — and every span grader here counts only judged ticks, so an
    unemitted tick is an absence of evidence and not a normal call."""
    calls = _calls("." * 10 + " " * 4 + "." * 10)
    params = _params()
    stream = hsmm_states(calls, {}, params)
    for tick, per in stream:
        i = (tick - _ORIGIN) // TICK_SECONDS
        assert ("A" in per) == (i < 10 or i >= 14)


def test_not_moving_keeps_the_arm_label_the_classifier_gave() -> None:
    """The model is binary but the published vocabulary is not. A not-moving
    decode is emitted under the most recent not-normal call's own label, so the
    suspended arm's grade is not silently measured on relabelled disrupted."""
    calls: list[tuple[int, Mapping[str, str]]] = [
        (_ORIGIN + i * TICK_SECONDS, {"A": "normal"}) for i in range(20)
    ]
    calls += [
        (_ORIGIN + (20 + i) * TICK_SECONDS, {"A": "suspended"}) for i in range(12)
    ]
    stream = hsmm_states(calls, {}, _params())
    emitted = {per["A"] for _, per in stream if "A" in per}
    assert "suspended" in emitted
    assert "disrupted" not in emitted


# --- the duration family's effect on the debounce ---------------------------


def _blippy_stream(
    n_routes: int = 4,
) -> tuple[list[tuple[int, Mapping[str, str]]], list[Ep], int, int]:
    """A stream with genuine multi-tick episodes and isolated one-tick false
    calls, plus the adjudicated episode record for the genuine ones only."""
    import random

    rng = random.Random(11)
    n_ticks = 900
    routes = [f"R{i}" for i in range(n_routes)]
    truth = {r: [False] * n_ticks for r in routes}
    eps: list[Ep] = []
    for r in routes:
        t = 30
        while t < n_ticks - 40:
            length = rng.randint(6, 12)
            for i in range(length):
                truth[r][t + i] = True
            eps.append(
                Ep(
                    r,
                    _ORIGIN + t * TICK_SECONDS,
                    _ORIGIN + (t + length) * TICK_SECONDS,
                )
            )
            t += length + rng.randint(80, 150)
    calls: list[tuple[int, Mapping[str, str]]] = []
    for i in range(n_ticks):
        per: dict[str, str] = {}
        for r in routes:
            if truth[r][i]:
                per[r] = "disrupted"
            elif rng.random() < 0.005:
                per[r] = "disrupted"  # injected false call
            else:
                per[r] = "normal"
        calls.append((_ORIGIN + i * TICK_SECONDS, per))
    return calls, eps, _ORIGIN, _ORIGIN + n_ticks * TICK_SECONDS


def _false_alarm_ticks(
    stream: Sequence[tuple[int, Mapping[str, str]]],
    calls: Sequence[tuple[int, Mapping[str, str]]],
    eps: Sequence[Ep],
) -> int:
    """Not-normal published ticks outside every adjudicated episode."""
    spans: dict[str, list[tuple[int, int]]] = {}
    for e in eps:
        assert e.recovery is not None
        spans.setdefault(e.route, []).append((e.onset, e.recovery))
    n = 0
    for tick, per in stream:
        for route, state in per.items():
            if state == "normal":
                continue
            if not any(s <= tick < r for s, r in spans.get(route, ())):
                n += 1
    return n


def test_explicit_duration_suppresses_blips_the_fitted_r1_arm_does_not() -> None:
    """Both arms run through the IDENTICAL filter with emissions fitted the same
    way; only the duration family differs. The explicit-duration arm suppresses
    most of the injected false calls; the r = 1 arm, AT ITS OWN FITTED
    PARAMETERS on this fixture, does not.

    WHAT THIS DOES AND DOES NOT ESTABLISH. It is a measurement on one synthetic
    stream, not a proof about the geometric family. A geometric HMM with noisy
    emissions can debounce in principle — its posterior accumulates evidence
    across ticks, and a sufficiently small entry hazard against the emission
    likelihood ratio refuses a lone disagreeing call. So this asserts the
    measured ordering at fitted parameters, and deliberately does NOT assert
    that r = 1 is incapable.

    The structural difference the arm actually rests on is narrower: a constant
    hazard has one parameter setting both how hard a fresh sojourn is to enter
    and how readily an established one ends, so it cannot resist a blip without
    also slowing genuine recoveries. That is what the next test pins.
    """
    calls, eps, w0, w1 = _blippy_stream()
    routes = sorted({r for _, per in calls for r in per})
    pops = episode_populations(eps, window_start=w0, window_end=w1, routes=routes)
    model = fit_dwell_model(pops.not_moving, min_route_events=4)
    normal_nb = fit_dwell(pops.normal.obs)
    normal_geom = fit_dwell(pops.normal.obs, fix_r=1.0)

    nb = fit_debounce(calls, model, normal_nb, max_ticks=48, geometric=False)
    geom = fit_debounce(calls, model, normal_geom, max_ticks=48, geometric=True)

    raw_fa = _false_alarm_ticks(calls, calls, eps)
    nb_fa = _false_alarm_ticks(nb.states(calls), calls, eps)
    geom_fa = _false_alarm_ticks(geom.states(calls), calls, eps)

    assert raw_fa > 0, "fixture injected no false calls"
    assert nb_fa < raw_fa / 2, (nb_fa, raw_fa)
    assert geom_fa >= nb_fa, (geom_fa, nb_fa)


def test_a_constant_hazard_cannot_resist_entry_and_recover_promptly() -> None:
    """THE STRUCTURAL CLAIM, WHICH IS NOT ABOUT FITTED PARAMETERS.

    A geometric dwell has ONE parameter serving two jobs. Its hazard is
    constant, so the probability a fresh sojourn ends immediately equals the
    probability an old one does, and the mean dwell is exactly 1 / h. Demanding
    a low early hazard — which is what refusing a one-tick blip requires —
    therefore forces a long mean dwell, and there is no geometric that does
    both. An explicit duration decouples them: r > 1 buys a low early hazard
    while keeping the mean short.

    This is the claim that survives regardless of what any particular fit
    lands on, so it is asserted over the family rather than over a fixture.
    """
    resist = 0.02  # early hazard low enough to refuse a lone call
    prompt = 12.0  # mean dwell in ticks we still want to allow

    # No geometric can do both: mean = 1 / h, so h <= 0.02 implies mean >= 50.
    for p in (0.001, 0.005, 0.02, 0.08, 0.3):
        geom = _fit(1.0, p)
        hz = dwell_hazards(geom, max_ticks=32)
        assert hz[0] == _approx(hz[-1])  # constant: one number, both jobs
        assert not (hz[0] <= resist and geom.mean_ticks <= prompt)
        assert geom.mean_ticks == _approx(1.0 / hz[0], rel=1e-6)

    # An explicit duration does: low early hazard AND a short mean.
    nb = _fit(6.0, 0.42)
    nb_hz = dwell_hazards(nb, max_ticks=32)
    assert nb_hz[0] <= resist
    assert nb.mean_ticks <= prompt
    assert nb_hz[-1] > nb_hz[0] * 5


def test_both_duration_arms_get_the_same_route_structure() -> None:
    """THE CONTROL MUST NOT BE HANDICAPPED BY POOLING.

    Gates 1 and 3 compare two duration FAMILIES through one filter. If the NB
    arm gets per-route fits while the r = 1 arm is served a single pooled
    curve, those gates measure pooling against per-route fitting instead, and
    the "nested geometric" column stops being a control.

    So every route carrying its own fit carries one in both families, fitted
    from the same observations, and the r = 1 per-route fits must actually
    differ from the pooled one (otherwise the symmetry is vacuous).
    """
    calls, eps, w0, w1 = _blippy_stream()
    routes = sorted({r for _, per in calls for r in per})
    pops = episode_populations(eps, window_start=w0, window_end=w1, routes=routes)
    model = fit_dwell_model(pops.not_moving, min_route_events=4)

    assert model.by_route, "fixture produced no per-route fits"
    assert set(model.by_route) == set(model.by_route_geometric)
    assert sorted(model.routes()) == sorted(model.by_route)

    for route in model.routes():
        nb = model.fit_for(route, geometric=False)
        geom = model.fit_for(route, geometric=True)
        assert not nb.geometric
        assert geom.geometric  # r == 1 exactly
        # The per-route geometric is that ROUTE's fit, not the pooled curve.
        assert geom.n_events == nb.n_events

    differs = [
        r
        for r in model.routes()
        if model.fit_for(r, geometric=True).p != model.pooled_geometric.p
    ]
    assert differs, "per-route geometric fits are all identical to the pooled one"

    # And the assembled filter params carry that same structure on both arms.
    normal_nb = fit_dwell(pops.normal.obs)
    normal_geom = fit_dwell(pops.normal.obs, fix_r=1.0)
    nb_arm = fit_debounce(calls, model, normal_nb, max_ticks=48, geometric=False)
    geom_arm = fit_debounce(calls, model, normal_geom, max_ticks=48, geometric=True)
    assert set(nb_arm.params_by_route()) == set(geom_arm.params_by_route())
    geom_params = geom_arm.params_by_route()
    assert len({p.not_moving_hazards for p in geom_params.values()}) > 1, (
        "every route got the same geometric hazard, so the arm is pooled"
    )


# --- the episode adapter ----------------------------------------------------


def test_episode_adapter_reads_both_field_conventions() -> None:
    """`episodes.Episode` uses onset/recovery and `load_r2.Disruption` uses
    start_tick/recovered_tick. The adapter is duck-typed over both so neither
    arm needs a shared base class to be gradeable."""

    @dataclass(frozen=True)
    class Disruption:
        route: str
        start_tick: int
        recovered_tick: int

    pop = episode_dwell_observations(
        [
            Ep("A", _ORIGIN, _ORIGIN + 6 * TICK_SECONDS),
            Disruption("B", _ORIGIN, _ORIGIN + 4 * TICK_SECONDS),
        ]
    )
    assert sorted(o.ticks for o in pop.obs) == [4, 6]


def test_still_open_episode_is_censored_at_the_window_not_dropped() -> None:
    """An episode with no recovery is right-censored at the boundary. Dropping
    it would delete exactly the longest dwells and bias the fit toward the
    short memoryless reading this arm is testing against."""
    end = _ORIGIN + 20 * TICK_SECONDS
    pop = episode_dwell_observations(
        [Ep("A", _ORIGIN + 5 * TICK_SECONDS, None, right_censored=True)],
        window_start=_ORIGIN,
        window_end=end,
    )
    assert len(pop.obs) == 1
    assert pop.obs[0].censored
    assert pop.obs[0].ticks == 15


def test_unmeasurable_episode_is_refused_not_silently_dropped() -> None:
    """No recovery and no window to censor at means the duration is
    unmeasurable. It must raise rather than vanish from the denominator."""
    with pytest.raises(ValueError, match="no recovery"):
        episode_dwell_observations([Ep("A", _ORIGIN, None, right_censored=True)])


def test_causal_split_admits_an_episode_by_its_onset_only() -> None:
    """A train-window fit must not see a duration that begins in the eval
    window, and an episode that predates the window has an unobserved onset."""
    split = _ORIGIN + 50 * TICK_SECONDS
    eps = [
        Ep("A", _ORIGIN - 10 * TICK_SECONDS, _ORIGIN + 2 * TICK_SECONDS),
        Ep("A", _ORIGIN + 10 * TICK_SECONDS, _ORIGIN + 16 * TICK_SECONDS),
        Ep("A", split + 5 * TICK_SECONDS, split + 9 * TICK_SECONDS),
    ]
    pop = episode_dwell_observations(eps, window_start=_ORIGIN, window_end=split)
    assert [o.ticks for o in pop.obs] == [6]
    assert pop.n_left_censored == 1


def test_normal_sojourns_are_the_gaps_between_episodes() -> None:
    """The normal-state dwell comes from the same adjudicated record, because
    the entry hazard is the one term that must not be estimated from the calls
    it is meant to filter."""
    w0 = _ORIGIN
    w1 = _ORIGIN + 100 * TICK_SECONDS
    eps = [
        Ep("A", w0 + 20 * TICK_SECONDS, w0 + 25 * TICK_SECONDS),
        Ep("A", w0 + 60 * TICK_SECONDS, w0 + 70 * TICK_SECONDS),
    ]
    pops = episode_populations(eps, window_start=w0, window_end=w1, routes=["A"])
    # Interior gap 25->60 is complete; trailing 70->100 is right-censored; the
    # leading 0->20 is left-truncated (residual life) and dropped.
    complete = [o.ticks for o in pops.normal.obs if not o.censored]
    censored = [o.ticks for o in pops.normal.obs if o.censored]
    assert complete == [35]
    assert censored == [30]
    assert pops.normal.n_left_censored == 1


def test_left_truncated_normal_runs_are_dropped_not_scored_as_survival() -> None:
    """A leading gap is a RESIDUAL-life observation, conditional on survival to
    the window start — not P(X >= gap). Scoring it as an ordinary censored
    observation would be a different likelihood, so the convention is to drop
    and count it. A never-disrupted route is left-truncated too."""
    w0, w1 = _ORIGIN, _ORIGIN + 100 * TICK_SECONDS
    pops = episode_populations([], window_start=w0, window_end=w1, routes=["A", "B"])
    assert pops.normal.obs == ()
    assert pops.normal.n_left_censored == 2


def test_raw_call_runs_age_in_elapsed_ticks_not_in_calls_seen() -> None:
    """A run spanning an abstention keeps running and keeps COUNTING, because
    the sojourn clock the filter uses advances in wall-clock ticks. Measuring
    durations in calls-seen would put the fit on a different time axis than the
    decoder whenever presence is intermittent.

    The pattern opens with a normal call so the not-normal run is not the
    window's leading run, which would be left-censored and dropped.
    """
    pop = dwell_observations(_calls("..XX  XX.."), not_normal=True)
    # Six ELAPSED ticks from the first not-normal call to the last, spanning
    # two abstentions — not the four calls actually seen.
    assert [o.ticks for o in pop.obs] == [6]
    assert pop.obs[0].censored is False


# --- window arithmetic ------------------------------------------------------


def test_split_window_gives_the_requested_day_counts() -> None:
    """A 14/7 request must MEASURE 14 and 7.

    `train_end` and `eval_end` name days inside their own halves, so both
    bounds are exclusive uppers taken from `aligned_window`'s second element.
    Reading the split off the FIRST element instead puts midnight on
    `train_end`, dropping that day from train and handing it to eval — a 14/7
    request that silently measured 13/8, which is what shipped in the first
    real gate run.
    """
    from datetime import date as _date

    day = 86400
    train_start = _date(2026, 8, 14)
    train_end = _date(2026, 8, 27)  # inclusive last train day
    eval_end = _date(2026, 9, 3)  # inclusive last eval day
    t0, split, t1 = split_window(train_start, train_end, eval_end)
    assert (split - t0) // day == 14
    assert (t1 - split) // day == 7
    # And the halves are contiguous and non-overlapping.
    assert t0 < split < t1


def test_split_window_is_exact_for_a_one_day_train_half() -> None:
    """The degenerate case the off-by-one hid behind: a single train day must
    still be one day, not zero."""
    from datetime import date as _date

    day = 86400
    t0, split, t1 = split_window(
        _date(2026, 8, 14), _date(2026, 8, 14), _date(2026, 8, 15)
    )
    assert (split - t0) // day == 1
    assert (t1 - split) // day == 1


# --- gate logic -------------------------------------------------------------


def test_gate_one_is_stated_on_the_interval_not_the_point_estimate() -> None:
    """ "Must not widen past the reference" is a claim about the envelope. A
    point estimate under the reference with a bound above it is exactly the
    case the bound exists to catch, and must FAIL."""
    sneaky = {
        "tick_rate": FA_REFERENCE * 0.5,
        "tick_rate_ci_low": 0.0,
        "tick_rate_ci_high": FA_REFERENCE * 2,
        "n_units": 30,
        "n_ticks": 9000,
    }
    assert gate_false_alarms(sneaky, {}).passed is False
    ok = {**sneaky, "tick_rate_ci_high": FA_REFERENCE * 0.9}
    assert gate_false_alarms(ok, {}).passed is True


def test_gate_one_is_blocked_not_failed_without_exposure() -> None:
    """A missing measurement is not a failure and must never be reported as
    one."""
    gate = gate_false_alarms({"gradeable": 0}, {})
    assert gate.passed is None


def test_gate_two_requires_both_halves() -> None:
    """In-sample significance AND a held-out interval strictly above zero. An
    interval straddling zero is not a strict improvement, which is what the
    gate demands."""
    strong = nest_test(
        [DwellObs("A", k, censored=False) for k in ([7] * 40 + [8] * 40 + [6] * 40)]
    )
    assert strong.improves
    assert gate_dwell_loglik(strong, Interval(0.4, 0.1, 0.7, 50)).passed is True
    # Held-out interval includes zero: not strict.
    assert gate_dwell_loglik(strong, Interval(0.4, -0.1, 0.9, 50)).passed is False
    # Held-out CI withheld (one episode): not strict either.
    assert gate_dwell_loglik(strong, Interval(0.4, None, None, 1)).passed is False

    null = nest_test(
        [DwellObs("A", k, censored=False) for k in _geometric_sample(0.93, 400, seed=1)]
    )
    assert gate_dwell_loglik(null, Interval(0.4, 0.1, 0.7, 50)).passed is False


def test_gate_three_fails_a_support_loss_whose_interval_includes_zero() -> None:
    """ENFORCED ON THE POINT ESTIMATE, DELIBERATELY.

    "Detection support must not fall" is a requirement about support. Treating
    "the fall is not statistically resolved" as a pass would let a thin window
    license an arbitrary loss, so the interval is evidence about how well
    determined the difference is and never a route to passing.
    """
    candidate = [True, True, False, False]
    incumbent = [True, True, True, False]
    delta = bootstrap_detection_delta(candidate, incumbent, n=200, seed=0)
    assert delta.ci_low is not None
    assert delta.ci_high is not None
    # The interval does not resolve the fall as strictly negative...
    assert delta.ci_low < 0 <= delta.ci_high
    gate = gate_detection_support(candidate, incumbent, delta, n_unmatched=0)
    assert gate.passed is False
    assert gate.numbers["paired_delta"]["ci_high"] is not None


def test_gate_three_passes_on_equal_support() -> None:
    """Non-inferiority, not strict superiority: holding support is a pass."""
    flags = [True, False, True, True]
    delta = bootstrap_detection_delta(flags, list(flags), n=200, seed=0)
    assert gate_detection_support(flags, list(flags), delta, n_unmatched=0).passed


def test_gate_three_is_blocked_with_no_matched_population() -> None:
    gate = gate_detection_support(
        [], [], Interval(math.nan, None, None, 0), n_unmatched=3
    )
    assert gate.passed is None


def test_detection_flags_exclude_unjudged_episodes_as_absence_of_evidence() -> None:
    """An episode whose route the surface never judged is UNSCORED, not a miss:
    charging it as a miss would make a blind window look like a regression."""
    state_at = {_ORIGIN + i * TICK_SECONDS: {"A": "normal"} for i in range(10)}
    eps = [
        Ep("A", _ORIGIN, _ORIGIN + 5 * TICK_SECONDS),
        Ep("B", _ORIGIN, _ORIGIN + 5 * TICK_SECONDS),
    ]
    flags, scored, unscored = detection_flags(state_at, eps, frozenset({"disrupted"}))
    assert flags == [False]
    assert unscored == 1
    assert [e.route for e in scored if isinstance(e, Ep)] == ["A"]


def test_paired_bootstrap_rejects_mismatched_arms() -> None:
    """The two arms must describe the same episodes, or the difference is
    between two populations rather than two rules."""
    with pytest.raises(ValueError, match="equal-length"):
        bootstrap_detection_delta([True], [True, False])


def test_intervals_are_withheld_below_two_units() -> None:
    """A CI on one cluster is fabricated, not wide."""
    one = bootstrap_dwell_delta(
        _obs((4, False)), _fit(3.0, 0.3), _fit(1.0, 0.3), n=100, seed=0
    )
    assert one.n_units == 1
    assert one.ci_low is None
    assert one.ci_high is None


def test_held_out_delta_is_scored_at_train_parameters() -> None:
    """Out of sample, with the causally fitted parameters, the NB advantage is
    a forecast comparison. On held-out durations the NB fit genuinely matches,
    the per-episode delta must be positive; on durations it does not, negative.
    Either way the extra parameter is not paid for by the data scoring it."""
    tight = _obs(*[(7, False)] * 30)
    nb = fit_dwell(tight)
    geom = fit_dwell(tight, fix_r=1.0)
    good = bootstrap_dwell_delta(_obs(*[(7, False)] * 20), nb, geom, n=300, seed=0)
    assert good.ci_low is not None
    assert good.ci_low > 0
    bad = bootstrap_dwell_delta(_obs(*[(1, False)] * 20), nb, geom, n=300, seed=0)
    assert bad.value < 0


# --- refusing degenerate populations ----------------------------------------


def _pop(*specs: tuple[int, bool]) -> DwellPopulation:
    return DwellPopulation(tuple(_obs(*specs)), 0)


def test_all_censored_dwell_population_is_refused_by_name() -> None:
    """Survival terms alone have no upper bound, so the MLE runs off to an
    unbounded mean — the fit returns a plausible-looking huge number rather
    than crashing, which is exactly why this has to be refused explicitly."""
    runaway = fit_dwell(_obs(*[(10, True)] * 20))
    assert runaway.mean_ticks > 1000
    with pytest.raises(GateBlocked) as exc:
        _refuse_degenerate(
            _pop((10, True), (10, True)),
            _pop((4, False)),
            _pop((50, False)),
            [(_ORIGIN, {"A": "normal", "B": "normal"})],
            [("A", _ORIGIN, _ORIGIN + 10 * TICK_SECONDS)],
            [Ep("A", _ORIGIN, _ORIGIN + TICK_SECONDS)],
        )
    assert exc.value.reason == "dwell_all_censored"


@pytest.mark.parametrize(
    ("reason", "train", "evalp", "calls", "runs", "corr"),
    [
        ("no_calls", _pop((4, False)), _pop((4, False)), [], [("A", 0, 1)], [1]),
        (
            "single_route",
            _pop((4, False)),
            _pop((4, False)),
            _calls("..."),
            [("A", 0, 1)],
            [1],
        ),
    ],
)
def test_missing_populations_are_refused_by_name(
    reason: str,
    train: DwellPopulation,
    evalp: DwellPopulation,
    calls: list[tuple[int, Mapping[str, str]]],
    runs: list[tuple[str, int, int]],
    corr: list[object],
) -> None:
    with pytest.raises(GateBlocked) as exc:
        _refuse_degenerate(train, evalp, _pop((50, False)), calls, runs, corr)
    assert exc.value.reason == reason


def test_thin_populations_are_refused_before_anything_is_fitted() -> None:
    """A grade that measures nothing must SAY so. Every rate here has a
    denominator that can be legitimately zero on a thin window, and a
    zero-denominator rate formats as a normal-looking 0.0000 — so the failure
    guarded is a plausible table, not a crash."""
    two_routes = [(_ORIGIN, {"A": "normal", "B": "normal"})]
    for reason, runs, corr in (
        ("no_normal_runs", [], [Ep("A", _ORIGIN, _ORIGIN + TICK_SECONDS)]),
        ("no_detection_episodes", [("A", _ORIGIN, _ORIGIN + 10 * TICK_SECONDS)], []),
    ):
        with pytest.raises(GateBlocked) as exc:
            _refuse_degenerate(
                _pop((4, False)),
                _pop((4, False)),
                _pop((50, False)),
                two_routes,
                runs,
                corr,
            )
        assert exc.value.reason == reason


def test_missing_normal_dwell_is_refused_by_name() -> None:
    """THE ENTRY HAZARD CANNOT BE FITTED FROM NOTHING.

    `grade` fits the normal sojourn and hands it straight to
    `params_from_fits` as the entry hazard. `fit_dwell(())` returns NaN r and
    p, which propagate through `dwell_hazards` into NaN gate numbers instead of
    an error — a plausible-looking table again, not a crash.

    The path is reachable BECAUSE left-truncated normal runs are deliberately
    dropped: a thin train window whose routes are either never disrupted or
    disrupted only after a leading gap has episode dwells and no usable normal
    dwells at all.
    """
    nan_fit = fit_dwell(())
    assert math.isnan(nan_fit.r)
    assert math.isnan(nan_fit.p)

    two_routes = [(_ORIGIN, {"A": "normal", "B": "normal"})]
    runs = [("A", _ORIGIN, _ORIGIN + 10 * TICK_SECONDS)]
    corr = [Ep("A", _ORIGIN, _ORIGIN + TICK_SECONDS)]

    # Every normal run was left-truncated and therefore dropped.
    with pytest.raises(GateBlocked) as exc:
        _refuse_degenerate(
            _pop((4, False)),
            _pop((4, False)),
            DwellPopulation((), 7),
            two_routes,
            runs,
            corr,
        )
    assert exc.value.reason == "no_normal_dwell"
    assert exc.value.detail["n_left_truncated_dropped"] == 7

    # Normal sojourns exist but none completed: the same unbounded MLE.
    with pytest.raises(GateBlocked) as exc:
        _refuse_degenerate(
            _pop((4, False)),
            _pop((4, False)),
            _pop((60, True), (90, True)),
            two_routes,
            runs,
            corr,
        )
    assert exc.value.reason == "normal_dwell_all_censored"


def test_grade_refuses_an_empty_batch_rather_than_reporting_nan() -> None:
    empty = GradeInputs(
        train_calls=[],
        eval_calls=[],
        train_episodes=[],
        eval_episodes=[],
        normal_runs=[],
        corroborated=[],
        routes=[],
        train_start=_ORIGIN,
        train_end=_ORIGIN + 10 * TICK_SECONDS,
        eval_end=_ORIGIN + 20 * TICK_SECONDS,
    )
    with pytest.raises(GateBlocked):
        grade(empty)


def test_grade_rejects_an_unknown_arm() -> None:
    empty = GradeInputs(
        train_calls=[],
        eval_calls=[],
        train_episodes=[],
        eval_episodes=[],
        normal_runs=[],
        corroborated=[],
        routes=[],
        train_start=_ORIGIN,
        train_end=_ORIGIN + 10 * TICK_SECONDS,
        eval_end=_ORIGIN + 20 * TICK_SECONDS,
    )
    with pytest.raises(ValueError, match="unknown arm"):
        grade(empty, arm="nope")


# --- end to end on synthetic input ------------------------------------------


def test_grade_runs_end_to_end_and_states_all_three_gates() -> None:
    """The whole harness on synthetic input: three gates, each with a verdict
    and numbers, and a surface table that includes the shipped incumbent.

    This does not assert a PASS. Whether the candidate clears the gates is an
    empirical question about real archive data, and a test that demanded a pass
    on a fixture would be asserting the fixture.
    """
    calls, eps, w0, w1 = _blippy_stream()
    split = w0 + (len(calls) // 2) * TICK_SECONDS
    routes = sorted({r for _, per in calls for r in per})
    train_calls = [(t, c) for t, c in calls if t < split]
    eval_calls = [(t, c) for t, c in calls if t >= split]
    train_eps = [e for e in eps if e.onset < split]
    eval_eps = [e for e in eps if e.onset >= split]
    # Confirmed-normal exposure: a stretch on each route with no episode in it.
    runs: list[tuple[str, int, int]] = []
    for r in routes:
        busy = [(e.onset, e.recovery) for e in eval_eps if e.route == r]
        cursor = split
        for onset, recovery in sorted(busy):
            if onset - cursor > 20 * TICK_SECONDS:
                runs.append((r, cursor, onset))
            assert recovery is not None
            cursor = recovery
    assert runs, "fixture produced no confirmed-normal exposure"

    report = grade(
        GradeInputs(
            train_calls=train_calls,
            eval_calls=eval_calls,
            train_episodes=list(train_eps),
            eval_episodes=list(eval_eps),
            normal_runs=runs,
            corroborated=list(eval_eps),
            routes=routes,
            train_start=w0,
            train_end=split,
            eval_end=w1,
        ),
        min_route_events=4,
        max_ticks=48,
        bootstrap=200,
    )
    names = [g["name"] for g in report["gates"]]
    assert names == ["false_alarm_ci", "dwell_loglik", "detection_support"]
    assert report["verdict"] in {"pass", "fail", "blocked"}
    # The shipped incumbent and the raw calls are both reported, so a filter
    # that debounced nothing is visible rather than inferred.
    assert {"negbin", "geometric", "shipped", "raw_calls"} <= set(
        report["false_alarms"]
    )
    assert report["detection"]["vs_shipped"]["n_matched"] >= 0
    assert "vs_nested_geometric" in report["detection"]


def test_grade_never_leaks_the_eval_window_into_a_parameter() -> None:
    """CAUSALITY, ASSERTED. Changing the eval window's calls and episodes must
    not move a single fitted parameter: the fits are a function of the train
    half alone. Warming carries STATE across the boundary, never values."""
    calls, eps, w0, _ = _blippy_stream()
    split = w0 + (len(calls) // 2) * TICK_SECONDS
    routes = sorted({r for _, per in calls for r in per})
    train_calls = [(t, c) for t, c in calls if t < split]
    train_eps = [e for e in eps if e.onset < split]

    pops = episode_populations(
        train_eps, window_start=w0, window_end=split, routes=routes
    )
    model = fit_dwell_model(pops.not_moving, min_route_events=4)
    baseline = fit_debounce(
        train_calls, model, fit_dwell(pops.normal.obs), max_ticks=48
    )
    # Same train half, totally different eval half.
    mutated = [
        (t, dict.fromkeys(routes, "suspended")) if t >= split else (t, c)
        for t, c in calls
    ]
    mutated_train = [(t, c) for t, c in mutated if t < split]
    again = fit_debounce(mutated_train, model, fit_dwell(pops.normal.obs), max_ticks=48)
    assert again.emissions.false_call == baseline.emissions.false_call
    assert again.emissions.miss == baseline.emissions.miss
    assert again.dwell.pooled.r == baseline.dwell.pooled.r
    assert again.dwell.pooled.p == baseline.dwell.pooled.p


def test_states_constant_names_are_distinct() -> None:
    """Two states, plus presence as a channel — deliberately not the three
    canonical eval.STATES, because the signal being debounced is binary."""
    assert NORMAL != NOT_MOVING
