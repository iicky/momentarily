"""The pure half of the online-FDR replay (training/online_fdr.py): the LORD++
and ADDIS recursions, the candidate p-value stream reduction, and the
three-stream comparison. Every R2 / alert-truth read lives behind `main`;
everything here runs on synthetic streams and synthetic truth.
"""

from __future__ import annotations

import random

from training.episodes import Episode
from training.load import TICK_SECONDS
from training.load_r2 import CLASSIFY_ALPHA
from training.online_fdr import (
    Addis,
    Candidate,
    LordPlusPlus,
    _FixedGate,  # pyright: ignore[reportPrivateUsage]
    addis_gamma,
    build_comparison,
    candidate_stream,
    corroborated_retention,
    false_alarm_bound,
    gate_verdict,
    grade_stream,
    lord_gamma,
    realized_fdp,
    replay,
)

T0 = 1_700_000_100


def _t(i: int) -> int:
    return T0 + i * TICK_SECONDS


# --- LORD++ -----------------------------------------------------------------


def test_lord_first_level_is_gamma1_times_w0() -> None:
    """The opening test level is the whole recursion at i=1: gamma_1 * w0."""
    level1 = lord_gamma(1) * 0.025
    assert LordPlusPlus(alpha=0.05, w0=0.025).test(level1) is True
    assert LordPlusPlus(alpha=0.05, w0=0.025).test(level1 * 1.0001) is False


def test_lord_pp_returns_full_wealth_at_first_rejection() -> None:
    """The `++` term: after the first rejection the coefficient on gamma_{i-tau_1}
    is (alpha - w0), so the second level is gamma_2*w0 + (alpha-w0)*gamma_1."""
    lp = LordPlusPlus(alpha=0.05, w0=0.025)
    assert lp.test(0.0) is True  # reject h1
    level2 = lord_gamma(2) * 0.025 + (0.05 - 0.025) * lord_gamma(1)
    assert lp.test(level2) is True
    other = LordPlusPlus(alpha=0.05, w0=0.025)
    other.test(0.0)  # same first rejection
    assert other.test(level2 * 1.001) is False


def test_lord_second_rejection_uses_alpha_coefficient() -> None:
    """A second rejection contributes alpha*gamma (not alpha-w0): distinct from
    the first-rejection term."""
    lp = LordPlusPlus(alpha=0.05, w0=0.025)
    lp.test(0.0)  # tau_1 = 1
    lp.test(0.0)  # tau_2 = 2
    # level_3 = g(3)*w0 + (alpha-w0)*g(2) + alpha*g(1)
    level3 = (
        lord_gamma(3) * 0.025 + (0.05 - 0.025) * lord_gamma(2) + 0.05 * lord_gamma(1)
    )
    assert lp.test(level3) is True
    lp2 = LordPlusPlus(alpha=0.05, w0=0.025)
    lp2.test(0.0)
    lp2.test(0.0)
    assert lp2.test(level3 * 1.001) is False


def test_lord_never_rejects_a_pure_null_stream_much() -> None:
    """Under uniform nulls the realized rejection count stays tiny — the whole
    point of the wealth constraint (a loose sanity bound, seeded)."""
    rng = random.Random(1)
    lp = LordPlusPlus(alpha=0.05)
    rejects = sum(lp.test(rng.random()) for _ in range(5000))
    assert rejects <= 5  # ~alpha-controlled; nowhere near 5%*5000=250


# --- ADDIS ------------------------------------------------------------------


def test_addis_first_level_capped_at_tau_lambda() -> None:
    w0 = 0.5 * 0.5 * 0.05 / 2
    level1 = min(0.25, w0 * addis_gamma(1))
    assert Addis(alpha=0.05).test(level1) is True
    assert Addis(alpha=0.05).test(level1 * 1.0001) is False


def test_addis_discards_conservative_nulls_without_spending() -> None:
    """A p-value above tau is discarded: it does not advance the selected count,
    so the next candidate's base level is unchanged from the opening one."""
    w0 = 0.5 * 0.5 * 0.05 / 2
    opening = min(0.25, w0 * addis_gamma(1))
    ad = Addis(alpha=0.05)
    for _ in range(4):
        assert ad.test(0.9) is False  # discarded (p > tau = 0.5)
    # selected count still 0, so the level for the next hypothesis is the opening
    assert ad.test(opening) is True


def test_addis_rejects_a_run_of_tiny_pvalues() -> None:
    ad = Addis(alpha=0.05)
    assert [ad.test(1e-6) for _ in range(5)] == [True] * 5


def test_addis_controls_a_pure_null_stream() -> None:
    rng = random.Random(2)
    ad = Addis(alpha=0.05)
    rejects = sum(ad.test(rng.random()) for _ in range(5000))
    assert rejects <= 5


# The onlineFDR ADDIS.Rd reference example p-values (synchronous case).
_ADDIS_REF_PVALS = [
    2.90e-08,
    0.06743,
    0.01514,
    0.08174,
    0.00171,
    3.60e-05,
    0.79149,
    0.27201,
    0.28295,
    7.59e-08,
    0.69274,
    0.30443,
    0.00136,
    0.72342,
    0.54757,
]


def _addis_paper_algorithm1(
    pvals: list[float],
    *,
    alpha: float = 0.05,
    lam_p: float = 0.25,
    tau: float = 0.5,
    w0: float | None = None,
) -> list[bool]:
    """An INDEPENDENT batch reference for Tian & Ramdas (2019) Algorithm 1
    (ADDIS*), written in the PAPER's convention — unscaled candidate threshold
    lam_p < tau, an explicit outer (tau - lam_p) factor, initial wealth W0 <= alpha
    (default alpha/2) — recomputing every count from scratch each step, sharing no
    code path with the streaming Addis under test. If the two agree, the R-ratio
    parameterization Addis ports is provably the paper's algorithm."""
    if w0 is None:
        w0 = alpha / 2.0

    def g(k: int) -> float:
        # paper indexes gamma from 0; addis_gamma from 1 (hence the +1 at call).
        return addis_gamma(k) if k >= 1 else 0.0

    n = len(pvals)
    rejected = [False] * n
    for t in range(1, n + 1):
        s_t = sum(1 for i in range(t - 1) if pvals[i] <= tau)
        kappas = [i for i in range(1, t) if rejected[i - 1]]  # 1-indexed reject times
        c0 = sum(1 for i in range(1, t) if pvals[i - 1] <= lam_p)
        term = (tau - lam_p) * w0 * g(s_t - c0 + 1)
        for j, kap in enumerate(kappas, start=1):
            kstar = sum(1 for i in range(1, kap + 1) if pvals[i - 1] <= tau)
            cj = sum(1 for i in range(kap + 1, t) if pvals[i - 1] <= lam_p)
            coef = (alpha - w0) if j == 1 else alpha
            term += (tau - lam_p) * coef * g(s_t - kstar - cj + 1)
        level = min(lam_p, term)
        rejected[t - 1] = pvals[t - 1] <= level
    return rejected


def test_addis_matches_paper_algorithm1() -> None:
    """The streaming Addis (onlineFDR ratio convention: lambda=tau=0.5,
    w0=tau*lambda*alpha/2) reproduces the paper's Algorithm 1 (unscaled lambda_p =
    tau*lambda = 0.25, W0 = alpha/2) rejection-for-rejection on the reference
    example — the scaled/unscaled equivalence, executable. The five rejections are
    exactly the five decisive p-values (< 2e-3)."""
    ad = Addis(alpha=0.05)
    mine = [ad.test(p) for p in _ADDIS_REF_PVALS]
    paper = _addis_paper_algorithm1(_ADDIS_REF_PVALS)
    assert mine == paper
    assert [i for i, r in enumerate(mine) if r] == [0, 4, 5, 9, 12]


# --- replay -----------------------------------------------------------------


def test_replay_collects_rejected_route_ticks() -> None:
    stream = [
        Candidate(_t(0), "A", 1e-6),
        Candidate(_t(0), "B", 0.9),
        Candidate(_t(1), "A", 1e-6),
    ]
    out = replay(stream, _FixedGate(0.05))
    assert out == {("A", _t(0)), ("A", _t(1))}


def test_online_streams_are_a_subset_of_the_fixed_gate() -> None:
    """LORD++/ADDIS thresholds sit at/under the fixed alpha, so every online
    rejection is also a fixed-gate rejection — the monotone containment the
    comparison rests on."""
    rng = random.Random(3)
    stream = [Candidate(_t(i), "A", rng.random() * 0.05) for i in range(200)]
    fixed = replay(stream, _FixedGate(CLASSIFY_ALPHA))
    lord = replay(stream, LordPlusPlus(alpha=0.05))
    addis = replay(stream, Addis(alpha=0.05))
    assert lord <= fixed
    assert addis <= fixed


# --- candidate_stream -------------------------------------------------------


def _baseline(p0: float, n: int = 50):
    from training.load_r2 import AdvanceBaseline

    return AdvanceBaseline(p0=p0, n=n, alpha=50 * p0, beta=50 * (1 - p0))


def test_candidate_stream_takes_the_worst_direction() -> None:
    """A route candidate at a tick carries the smallest p over its candidate
    directions — the worse direction, as derive_movement_state reduces."""
    from training.load_r2 import tod_bin

    tick = _t(0)
    dir_series = {
        ("A", "north", tick): {"advanced_n": 0, "stalled_n": 10},  # frozen -> tiny p
        ("A", "south", tick): {"advanced_n": 3, "stalled_n": 7},  # milder
    }
    baseline = {
        ("A", "north", tod_bin(tick)): _baseline(0.9),
        ("A", "south", tod_bin(tick)): _baseline(0.9),
    }
    stream = candidate_stream(dir_series, baseline)
    assert len(stream) == 1
    c = stream[0]
    assert c.route == "A"
    assert c.tick == tick
    # north (0/10 vs p0=0.9) is the more extreme lower tail
    assert c.p < 1e-3


def test_candidate_stream_excludes_non_candidates_and_unjudgeable() -> None:
    tick = _t(0)
    from training.load_r2 import tod_bin

    dir_series = {
        # posterior normal (advancing near baseline) -> not a candidate
        ("A", "north", tick): {"advanced_n": 9, "stalled_n": 1},
        # too few matches -> unjudgeable
        ("B", "north", tick): {"advanced_n": 0, "stalled_n": 2},
        # no baseline -> unjudgeable
        ("C", "north", tick): {"advanced_n": 0, "stalled_n": 10},
    }
    baseline = {
        ("A", "north", tod_bin(tick)): _baseline(0.9),
        ("B", "north", tod_bin(tick)): _baseline(0.9),
    }
    assert candidate_stream(dir_series, baseline) == []


def test_candidate_stream_is_causally_ordered() -> None:
    from training.load_r2 import tod_bin

    dir_series = {
        ("B", "north", _t(1)): {"advanced_n": 0, "stalled_n": 10},
        ("A", "north", _t(1)): {"advanced_n": 0, "stalled_n": 10},
        ("A", "north", _t(0)): {"advanced_n": 0, "stalled_n": 10},
    }
    baseline = {(r, "north", tod_bin(_t(0))): _baseline(0.9) for r in ("A", "B")}
    order = [(c.tick, c.route) for c in candidate_stream(dir_series, baseline)]
    assert order == [(_t(0), "A"), (_t(1), "A"), (_t(1), "B")]


# --- grading ----------------------------------------------------------------


def test_false_alarm_bound_reuses_run_bootstrap() -> None:
    judged = {("A", _t(i)) for i in range(10)}
    rejections = {("A", _t(3))}
    runs = [("A", _t(0), _t(10))]
    fa = false_alarm_bound(rejections, judged, runs, bootstrap=100)
    assert fa["gradeable"] == 1
    assert fa["alarmed_ticks"] == 1
    assert fa["n_ticks"] == 10
    assert fa["tick_rate"] == 0.1


def test_false_alarm_bound_counts_only_judged_ticks() -> None:
    """A run tick the route was never judged on is not in the denominator."""
    judged = {("A", _t(0)), ("A", _t(1))}  # only 2 of 10 judged
    fa = false_alarm_bound({("A", _t(0))}, judged, [("A", _t(0), _t(10))], bootstrap=50)
    assert fa["n_ticks"] == 2
    assert fa["tick_rate"] == 0.5


def _ep(route: str, onset_i: int, recovery_i: int) -> Episode:
    return Episode(
        route=route,
        onset=_t(onset_i),
        recovery=_t(recovery_i),
        peak_state="disrupted",
        cause="delays",
        n_ticks=recovery_i - onset_i,
        left_censored=False,
        right_censored=False,
    )


def test_corroborated_retention_excludes_unscored_episodes() -> None:
    judged = {("A", _t(i)) for i in range(6)}  # B never judged
    corroborated = [_ep("A", 0, 4), _ep("B", 0, 4)]
    rej = {("A", _t(1))}
    out = corroborated_retention(rej, corroborated, judged, bootstrap=100)
    assert out["n_unscored"] == 1  # B dropped
    assert out["n"] == 1
    assert out["n_true"] == 1
    assert out["rate"] == 1.0


def test_realized_fdp_classifies_and_withholds_thin_ci() -> None:
    """A rejection on a normal run is false; one inside a corroborated episode is
    true. With <2 clusters bearing a discovery on a side, the CI is withheld."""
    runs = [("A", _t(0), _t(4))]
    corroborated = [_ep("A", 10, 14)]
    rej = {("A", _t(1)), ("A", _t(11))}
    out = realized_fdp(rej, runs, corroborated, bootstrap=100)
    assert out["false_discoveries"] == 1
    assert out["true_discoveries"] == 1
    assert out["fdp"] == 0.5
    assert "fdp_ci_low" not in out  # only 1 cluster each side


def test_realized_fdp_bootstraps_with_enough_clusters() -> None:
    runs = [("A", _t(0), _t(4)), ("B", _t(0), _t(4))]
    corroborated = [_ep("C", 10, 14), _ep("D", 10, 14)]
    rej = {
        ("A", _t(1)),
        ("B", _t(1)),  # 2 false clusters
        ("C", _t(11)),
        ("D", _t(11)),  # 2 true clusters
    }
    out = realized_fdp(rej, runs, corroborated, bootstrap=200)
    assert out["fdp"] == 0.5
    assert out["n_false_clusters"] == 2
    assert out["n_true_clusters"] == 2
    assert out["fdp_ci_low"] <= 0.5 <= out["fdp_ci_high"]


def test_gate_verdict_reads_both_bounds() -> None:
    stream = grade_stream(
        {("A", _t(1))},
        judged={("A", _t(i)) for i in range(100)},
        runs=[("A", _t(0), _t(100))],
        corroborated=[],
        n_days=1.0,
        bootstrap=50,
    )
    gv = gate_verdict(stream)
    # 1 alarm in 100 judged normal ticks = 0.01/tick, above the 0.00101 bound
    assert gv["per_route_bound_met"] is False
    # the lone rejection is on a confirmed-normal run with no true discovery to
    # offset it: an all-false adjudicable universe, FDP 1.0, gate not met
    assert gv["fleet_fdp"] == 1.0
    assert gv["fleet_fdp_met"] is False


# --- build_comparison (the three-stream core) -------------------------------


def test_build_comparison_orders_streams_by_strictness() -> None:
    """On a stream mixing decisive (tiny-p) and marginal (just-under-alpha)
    candidates on confirmed-normal supply, the fixed gate fires most; LORD++ and
    ADDIS fire a subset and cut the fleet false-alarm count."""
    # One route, a long confirmed-normal run; movement fires a scatter of
    # marginal false alarms plus nothing decisive.
    n = 60
    movement_truth = {("A", _t(i)): "normal" for i in range(n)}
    runs = [("A", _t(0), _t(n))]
    # candidate stream: every 5th tick a marginal candidate just under alpha.
    stream = [Candidate(_t(i), "A", 0.04) for i in range(n) if i % 5 == 0]
    report = build_comparison(
        stream,
        movement_truth=movement_truth,
        runs=runs,
        alert_disrupted=set(),
        window_start=_t(0),
        window_end=_t(n),
        n_days=1.0,
        bootstrap=100,
    )
    fixed = report["streams"]["fixed_gate"]["n_rejections"]
    lord = report["streams"]["lord_pp"]["n_rejections"]
    addis = report["streams"]["addis"]["n_rejections"]
    assert fixed == 12  # every marginal candidate clears 0.05
    assert lord < fixed  # online procedures reject far fewer marginals
    assert addis < fixed
    assert report["streams"]["fixed_gate"]["false_alarms_per_day"] == 12.0


def test_build_comparison_retains_decisive_corroborated_episode() -> None:
    """A genuine freeze (p ~ 0) inside an escalation-corroborated episode is
    retained by every stream — the strong-evidence rejection survives even the
    strict online threshold, while diffuse marginal alarms do not."""
    n = 40
    # Movement truth: normal, except a disrupted episode ticks 20..24 on A.
    movement_truth: dict[tuple[str, int], str] = {}
    for i in range(n):
        movement_truth[("A", _t(i))] = "disrupted" if 20 <= i < 25 else "normal"
    # Alert feed corroborates the episode onset at tick 20.
    alert_disrupted = {("A", _t(20))}
    runs = [("A", _t(0), _t(20))]
    # Stream: marginal false alarms early, decisive tiny-p across the episode.
    stream = [Candidate(_t(i), "A", 0.045) for i in (2, 6, 10)]
    stream += [Candidate(_t(i), "A", 1e-9) for i in range(20, 25)]
    report = build_comparison(
        stream,
        movement_truth=movement_truth,
        runs=runs,
        alert_disrupted=alert_disrupted,
        window_start=_t(0),
        window_end=_t(n),
        n_days=1.0,
        bootstrap=100,
    )
    assert report["reference"]["n_corroborated"] == 1
    for name in ("fixed_gate", "lord_pp", "addis"):
        ret = report["streams"][name]["corroborated_retention"]
        assert ret["n_true"] == 1  # decisive episode kept by all three
        onset = report["streams"][name]["onset_backdating"]
        assert onset["n_detected"] == 1
        assert onset["median_latency_min"] == 0.0  # fires at onset tick 20
