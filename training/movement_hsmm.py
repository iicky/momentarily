"""Explicit-duration (HSMM-style) debounce over the binary movement signal.

WHAT THIS REPLACES, AND WHY THE INCUMBENT IS NOT A MODEL
--------------------------------------------------------
The movement arm's dwell is memoryless. `train_em.MAX_SELF_LOOP` clamps the
disrupted diagonal at 0.93 and the fitted diagonal sits exactly ON that ceiling
for every route measured (28/28), so the published dwell is not something EM
learned from durations — it is the clamp, converted to a median by
`train_em.implied_median_dwell_minutes`. A geometric dwell also asserts a
CONSTANT hazard: a regime that has been open five minutes is exactly as likely
to end in the next tick as one open two hours. Neither claim survives contact
with the durations (P(stay)=63.9% against P(enter)=0.076%, ~14 min mean), and
the constant-hazard form is precisely the assumption an explicit-duration
sojourn distribution drops.

So this module fits the dwell as a distribution over durations, not as a
transition-matrix diagonal, and drives the debounce off its HAZARD.

THE FORM: 2 STATES + PRESENCE
-----------------------------
Two latent states over the binary movement signal — `normal` and `not_normal`
(disrupted/suspended collapsed; the arm distinction is carried through from the
raw calls, see `hsmm_states`) — each with its own explicit sojourn
distribution. Presence is the third channel and NOT a third state: a route
absent from a tick's calls is an abstention, whose emission likelihood is
state-independent and therefore contributes no evidence about the state. That
is `regime.py`'s "no reading is not a reading of change" restated as a
modelling assumption rather than a special case in the clock.

One consequence is a real behavioural difference from the incumbent, not a
restatement of it: the sojourn CLOCK advances through an abstention even though
the posterior does not move. A k-of-k debounce cannot express that, because it
has no notion of how long the regime has been open.

WHY NOT STICKY-KAPPA (it was rejected, and this is not it)
----------------------------------------------------------
An HDP-HSMM-style sticky self-transition bias was considered and REJECTED: a
kappa boost on the self-transition suppresses re-entry, so two genuinely
separate short episodes a few ticks apart get merged or the second is dropped.
Nothing here adds a self-transition bonus. The persistence lives entirely in
the sojourn distribution, whose hazard RESETS on entry — a fresh episode is
judged by the same low early hazard as the first one, so short separated
episodes stay separate. Sticky-kappa and an explicit duration are not two
strengths of the same knob; they are opposite treatments of re-entry.

NEGATIVE BINOMIAL, AND THE NEST THAT MAKES IT FALSIFIABLE
---------------------------------------------------------
The sojourn is a shifted negative binomial on {1, 2, ...}: X = 1 + Z with
Z ~ NB(r, p). At r = 1 this is EXACTLY the geometric dwell the self-loop
already asserts, with `self_loop = 1 - p`:

    r = 1:  P(X = k) = p (1 - p)^(k-1)   <->   a^(k-1) (1 - a),  a = 1 - p

so the incumbent is a point in the parameter space of the successor, and "is
the explicit duration worth anything" is a likelihood-ratio test against
r = 1 with one degree of freedom. r = 1 is an INTERIOR point of r > 0, not a
boundary, so the ordinary chi-squared(1) reference distribution applies and no
mixture correction is needed. `nest_test` reports it.

r < 1 is over-dispersed relative to geometric (a heavier short-and-long mix);
r > 1 is under-dispersed with a rising early hazard, which is the shape a real
debounce wants: a just-opened regime is UNLIKELY to end immediately, so a
one-tick blip is expensive to explain as a genuine episode.

THE TRUNCATED HAZARD TAIL IS EXACT IN THE LIMIT
-----------------------------------------------
The filter carries elapsed time in the state, capped at `DWELL_MAX_TICKS`. The
cap does NOT force an exit; it holds the hazard constant at its fitted value
there. That is not a convenience: the NB hazard converges to p as d grows (its
tail IS geometric with ratio 1 - p), so a constant-hazard tail beyond the cap
is exact for r = 1 and asymptotically exact otherwise. Forcing an exit at the
cap instead would inject a spurious recovery spike at a fixed age.

WHERE THE DWELL COMES FROM, AND WHY IT CANNOT COME FROM THE CALLS
----------------------------------------------------------------
Both sojourn distributions are fitted on the ARCHIVED EPISODE DURATIONS, whose
onsets and recoveries the assigned_n supply feed adjudicates —
independent-in-derivation from the vehicle-position flow signal being
debounced. Only the emission channel is fitted on the calls.

That split is forced, not stylistic. Fitting the dwell on the calls does not
identify the model: "a genuine one-tick episode" and "a false call" predict the
same single tick, so the likelihood is free to prefer whichever is cheaper, and
it prefers "genuine" — an over-dispersed sojourn absorbs a blip more cheaply
than any error channel can. Measured on a synthetic stream carrying 29-32
known-false one-tick calls: a joint fit reached a self-consistent fixed point
with r_hat = 0.315 of manufactured over-dispersion and a decode that reproduced
the raw calls tick for tick, and adjudicating between starts on the marginal
likelihood (the one criterion a hard-EM fixed point cannot game) still chose it,
-485.5 against -874.4. Reading only the NORMAL sojourn off the calls fails the
same way and for a sharper reason: the entry hazard is the single term deciding
whether one call can flip the state, and a false call fragments a long normal
run, raises that hazard, and makes the next false call easier to believe.

With both dwells pinned externally the debounce works. Same stream, same
filter, emissions fitted the same way on each arm: the explicit-duration arm
cut false-alarm ticks from 29 to 1, while the r = 1 arm AT ITS OWN FITTED
PARAMETERS reproduced the raw calls exactly.

That last result is a measurement on one synthetic fixture, not a theorem, and
it is worth stating at the strength it carries. A geometric HMM with noisy
emissions is NOT incapable of debouncing: its posterior still accumulates
evidence across ticks, and a small enough entry hazard against the emission
likelihood ratio will refuse a single disagreeing call. The pass-through here
is what that arm's fitted parameters happened to produce on this stream.

What IS structural, and is the actual claim of this arm, is narrower: a
constant hazard has one number serving two jobs. The same parameter sets how
hard a fresh sojourn is to enter and how readily an established one ends, so a
geometric dwell cannot resist a one-tick blip without also making genuine
recoveries slow, and cannot permit prompt recoveries without also accepting
blips. An explicit duration separates those, which is why the r = 1 nest is
carried through the identical filter rather than compared against `regime.py`,
which differs in more than the dwell.

CAUSALITY
---------
Everything in this module is a pure function of durations or of a call stream.
Nothing fetches, and nothing looks at a tick later than the one it is emitting:
`hsmm_states` is a FILTER (forward pass only, no smoothing), so its published
state at tick t is a function of calls up to t. A smoothed decode would score
better and could not ship. The train/eval split and the gates live in
`movement_hsmm_grade`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from training.eval import NOT_NORMAL, TICK_SECONDS
from training.regime import MAX_IDLE_SEC

# Latent states. Two, plus presence as an observation channel (module
# docstring) — deliberately NOT the three canonical `eval.STATES`, because the
# signal being debounced is binary.
NORMAL = 0
NOT_MOVING = 1

_NOT_NORMAL_CALLS = frozenset(NOT_NORMAL)

# Elapsed-time cap in the filter's augmented state. Beyond it the hazard is
# held at its fitted value rather than forced to 1 — exact for the geometric
# nest and asymptotically exact for any r, since the NB tail is geometric
# (module docstring). 96 ticks is 8 hours: far past where either state's hazard
# is still moving, and small enough that the forward pass stays linear-cheap.
DWELL_MAX_TICKS = 96

# A route needs this many UNCENSORED dwell events before it gets its own fit;
# thinner routes inherit the pooled train fit. Matches
# pooled_dwell.MIN_VOTER_EVENTS in spirit — the point is that a two-episode
# route cannot support a two-parameter duration fit — but is named separately
# because it gates a different fit and must be tunable without touching the
# published mixture.
MIN_ROUTE_DWELL_EVENTS = 8

# Search bounds for the dispersion parameter. Wide enough that a hard shoulder
# is visible as r_hat sitting on it (reported, never silently clamped away).
_R_MIN = 0.05
_R_MAX = 50.0

# Numerical floor/ceiling for the NB success probability.
_P_EPS = 1e-9


# --- censored duration observations -----------------------------------------


@dataclass(frozen=True)
class DwellObs:
    """One sojourn, in ticks, and whether its end was observed.

    `censored` means right-censored: the run was still open at the last tick of
    the window, so its true duration is > `ticks`. Such an observation carries
    real information (a survival term) and is kept, unlike a LEFT-censored run
    — already open when the window opened — whose onset is unobserved and whose
    duration cannot be recovered at all. Left-censored runs are dropped by
    `dwell_observations` and counted, never silently folded in: including them
    as if complete biases every duration statistic downward, which is the exact
    direction that would flatter a memoryless fit.
    """

    route: str
    ticks: int
    censored: bool


@dataclass(frozen=True)
class DwellPopulation:
    """The dwell observations of one window plus what was refused."""

    obs: tuple[DwellObs, ...]
    n_left_censored: int

    @property
    def n_events(self) -> int:
        return sum(1 for o in self.obs if not o.censored)

    @property
    def n_censored(self) -> int:
        return sum(1 for o in self.obs if o.censored)

    def by_route(self) -> dict[str, list[DwellObs]]:
        out: dict[str, list[DwellObs]] = {}
        for o in self.obs:
            out.setdefault(o.route, []).append(o)
        return out


def _route_series(
    calls: Sequence[tuple[int, Mapping[str, str]]],
) -> dict[str, list[tuple[int, str]]]:
    """{route: [(tick, call)]} in tick order, abstentions simply absent."""
    out: dict[str, list[tuple[int, str]]] = {}
    for tick, per_route in sorted(calls, key=lambda t: t[0]):
        for route, state in per_route.items():
            out.setdefault(route, []).append((tick, state))
    return out


def dwell_observations(
    calls: Sequence[tuple[int, Mapping[str, str]]],
    *,
    not_normal: bool,
    max_idle_sec: int = MAX_IDLE_SEC,
) -> DwellPopulation:
    """Sojourn durations read off the RAW per-tick calls, per route.

    `not_normal=True` collects runs of not-normal calls, `False` runs of normal
    calls.

    THIS IS A DIAGNOSTIC, NOT THE DWELL THE MODEL USES. Both fitted sojourn
    distributions come from the episode archive instead (`episode_populations`),
    because a duration read off the calls is contaminated by exactly the false
    calls the debounce exists to suppress — measured at r_hat = 0.315 of
    manufactured over-dispersion on a synthetic stream, and a pass-through
    filter as the result. What this function is good for is describing the raw
    call stream's own run structure, which is worth reporting beside a fit but
    is not evidence about either duration family.

    DURATIONS ARE ELAPSED TICKS, not counts of agreeing calls. A run that spans
    an abstention keeps running and keeps counting: no reading is not a reading
    of change, so the sojourn clock advances through it. That is the same unit
    `filter_route` and `viterbi_route` age their state in, and measuring
    durations in calls-seen instead would hand the fit a different time axis
    than the decoder's whenever presence is intermittent.

    A run breaks on a call of the other kind and on a gap longer than
    `max_idle_sec` — an hour of blindness means the run that resumes is not
    knowably the one that stopped, the same rule the regime clock applies. A
    run broken by a GAP is right-censored, not completed: nothing observed its
    end.

    Minimum duration is 1 tick.
    """
    obs: list[DwellObs] = []
    n_left = 0
    for route, points in _route_series(calls).items():
        first: int | None = None
        last = 0
        left_censored = False
        prev_tick: int | None = None
        for i, (tick, state) in enumerate(points):
            stale = prev_tick is not None and tick - prev_tick > max_idle_sec
            in_state = (state in _NOT_NORMAL_CALLS) == not_normal
            # A run ends either because the other kind of call arrived
            # (completed: its end was observed) or because the route went blind
            # (censored: it was not).
            ends = first is not None and (stale or not in_state)
            if ends:
                assert first is not None
                if left_censored:
                    n_left += 1
                else:
                    obs.append(
                        DwellObs(
                            route,
                            (last - first) // TICK_SECONDS + 1,
                            censored=stale,
                        )
                    )
                first = None
                left_censored = False
            if in_state:
                if first is None:
                    first = tick
                    left_censored = i == 0
                last = tick
            prev_tick = tick
        if first is not None:
            if left_censored:
                n_left += 1
            else:
                obs.append(
                    DwellObs(route, (last - first) // TICK_SECONDS + 1, censored=True)
                )
    return DwellPopulation(tuple(obs), n_left)


# --- the shifted negative binomial ------------------------------------------


def nb_log_pmf(k: int, r: float, p: float) -> float:
    """log P(X = k) for X = 1 + Z, Z ~ NB(r, p), on k >= 1."""
    if k < 1:
        return -math.inf
    z = k - 1
    return (
        math.lgamma(z + r)
        - math.lgamma(r)
        - math.lgamma(z + 1.0)
        + r * math.log(p)
        + z * math.log1p(-p)
    )


def nb_sf(k: int, r: float, p: float) -> float:
    """P(X > k) for X = 1 + Z, Z ~ NB(r, p).

    Computed as 1 - CDF while that is well conditioned, and by summing the tail
    directly once cancellation would dominate. The tail terms shrink by a factor
    approaching (1 - p) so the sum terminates; a censored observation deep in
    the tail is exactly where a naive 1 - CDF returns 0.0 and hands the fit an
    infinite penalty for a perfectly ordinary long regime.
    """
    if k < 1:
        return 1.0
    term = p**r  # pmf of z = 0
    cdf = term
    z = 0
    while z < k - 1:
        z += 1
        term *= (z + r - 1.0) / z * (1.0 - p)
        cdf += term
    sf = 1.0 - cdf
    if sf > 1e-8:
        return sf
    # Cancellation regime: sum the tail from z = k directly.
    tail = 0.0
    zz = z
    tterm = term
    while zz < k:
        zz += 1
        tterm *= (zz + r - 1.0) / zz * (1.0 - p)
    tail = tterm
    guard = 0
    while tterm > tail * 1e-16 and guard < 1_000_000:
        zz += 1
        tterm *= (zz + r - 1.0) / zz * (1.0 - p)
        tail += tterm
        guard += 1
    return max(tail, 1e-300)


def dwell_loglik(obs: Sequence[DwellObs], r: float, p: float) -> float:
    """Total log-likelihood of `obs` under the shifted NB, right-censored
    observations entering through their survival term.

    A run observed open for `ticks` ticks and never seen to end tells us
    X >= ticks, NOT X > ticks: the sojourn may have ended exactly at its last
    observed tick with the recovery falling outside the window. So the censored
    term is P(X >= ticks) = nb_sf(ticks - 1), the same convention
    `dwell_hazards` uses for the survival in its denominator. Using
    nb_sf(ticks) instead would assert strictly longer, understate survival on
    every censored observation, and bias the fit toward shorter dwell — the
    direction that flatters the memoryless incumbent.
    """
    total = 0.0
    for o in obs:
        if o.censored:
            total += math.log(nb_sf(o.ticks - 1, r, p))
        else:
            total += nb_log_pmf(o.ticks, r, p)
    return total


@dataclass(frozen=True)
class NbDwellFit:
    """One fitted sojourn distribution and what it was fitted on.

    `self_loop` is the geometric self-loop this fit is equivalent to, and is
    populated ONLY at r == 1 where that equivalence holds exactly. A single
    self-loop number for r != 1 would be a category error — the whole claim of
    this arm is that no single per-tick continuation probability describes the
    dwell — so it is None rather than an approximation.
    """

    r: float
    p: float
    n_events: int
    n_censored: int
    loglik: float
    r_at_bound: bool

    @property
    def geometric(self) -> bool:
        return abs(self.r - 1.0) < 1e-9

    @property
    def self_loop(self) -> float | None:
        return 1.0 - self.p if self.geometric else None

    @property
    def mean_ticks(self) -> float:
        return 1.0 + self.r * (1.0 - self.p) / self.p

    @property
    def mean_minutes(self) -> float:
        return self.mean_ticks * TICK_SECONDS / 60.0

    @property
    def n_obs(self) -> int:
        return self.n_events + self.n_censored

    @property
    def mean_loglik(self) -> float:
        """Per-observation log-likelihood — the scale the gate is stated on, so
        that populations of different size stay comparable."""
        return self.loglik / self.n_obs if self.n_obs else math.nan


def _maximize(
    fn: Callable[[float], float],
    lo: float,
    hi: float,
    *,
    grid: int = 24,
    iters: int = 60,
) -> float:
    """Argmax of a scalar function on [lo, hi] by coarse scan then golden
    section. The scan is what makes this safe on a profile likelihood whose
    unimodality is asserted rather than proven — a bracket picked from a grid
    cannot land on the wrong side of a shoulder the way a bare derivative
    search can.
    """
    best_x = lo
    best_v = -math.inf
    step = (hi - lo) / (grid - 1)
    for i in range(grid):
        x = lo + i * step
        v = float(fn(x))
        if v > best_v:
            best_v, best_x = v, x
    a = max(lo, best_x - step)
    b = min(hi, best_x + step)
    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = float(fn(c)), float(fn(d))
    for _ in range(iters):
        if b - a < 1e-10:
            break
        if fc > fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = float(fn(c))
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = float(fn(d))
    x = (a + b) / 2.0
    return x if float(fn(x)) >= best_v else best_x


def _best_p(obs: Sequence[DwellObs], r: float) -> float:
    return _maximize(
        lambda p: dwell_loglik(obs, r, p),
        _P_EPS,
        1.0 - _P_EPS,
    )


def fit_dwell(obs: Sequence[DwellObs], *, fix_r: float | None = None) -> NbDwellFit:
    """MLE of the shifted NB sojourn, right-censored observations included.

    `fix_r=1.0` fits the GEOMETRIC nest — the incumbent's own form — so the two
    arms of the nest test come out of one code path and cannot diverge by
    implementation. With no censoring and r fixed, the maximiser has the closed
    form p = r / (r + mean(X - 1)); the numeric search reproduces it, which is
    what this module's test asserts, and censoring is what makes the numeric
    path necessary at all.

    An empty population yields a degenerate fit with NaN likelihood rather than
    a plausible-looking number; callers must refuse it by name (see
    `movement_hsmm_grade.GateBlocked`).
    """
    if not obs:
        return NbDwellFit(
            r=fix_r if fix_r is not None else math.nan,
            p=math.nan,
            n_events=0,
            n_censored=0,
            loglik=math.nan,
            r_at_bound=False,
        )
    n_events = sum(1 for o in obs if not o.censored)
    n_censored = len(obs) - n_events
    if fix_r is not None:
        r = fix_r
        at_bound = False
    else:
        log_r = _maximize(
            lambda lr: dwell_loglik(obs, math.exp(lr), _best_p(obs, math.exp(lr))),
            math.log(_R_MIN),
            math.log(_R_MAX),
        )
        r = math.exp(log_r)
        at_bound = r <= _R_MIN * 1.001 or r >= _R_MAX * 0.999
    p = _best_p(obs, r)
    return NbDwellFit(
        r=r,
        p=p,
        n_events=n_events,
        n_censored=n_censored,
        loglik=dwell_loglik(obs, r, p),
        r_at_bound=at_bound,
    )


def chi2_sf_1df(x: float) -> float:
    """P(chi-squared with 1 df > x), exactly erfc(sqrt(x/2)) — no table, no
    approximation, and no scipy (this repo has none)."""
    if x <= 0.0:
        return 1.0
    return math.erfc(math.sqrt(x / 2.0))


@dataclass(frozen=True)
class NestTest:
    """The r = 1 likelihood-ratio test: is the explicit duration worth its one
    extra parameter, against the geometric the self-loop already asserts?

    r = 1 is interior to r > 0, so `p_value` is a plain chi-squared(1) tail and
    needs no boundary mixture correction (module docstring).

    `lr_stat` can only be >= 0 up to search error: the geometric is nested, so
    the wider family cannot fit worse. A negative value therefore does not mean
    "geometric wins" — it means the NB search under-converged, and it is
    reported rather than clipped to 0 so that failure is visible instead of
    disguised as a null result.
    """

    geometric: NbDwellFit
    negbin: NbDwellFit
    lr_stat: float
    p_value: float
    df: int = 1

    @property
    def delta_mean_loglik(self) -> float:
        return self.negbin.mean_loglik - self.geometric.mean_loglik

    @property
    def improves(self) -> bool:
        """Strict in-sample improvement AND significant at 0.05."""
        return self.lr_stat > 0.0 and self.p_value < 0.05


def nest_test(obs: Sequence[DwellObs]) -> NestTest:
    """Fit both arms on the same observations and test r = 1."""
    geom = fit_dwell(obs, fix_r=1.0)
    nb = fit_dwell(obs)
    lr = 2.0 * (nb.loglik - geom.loglik)
    return NestTest(
        geometric=geom,
        negbin=nb,
        lr_stat=lr,
        p_value=chi2_sf_1df(lr) if lr > 0.0 else 1.0,
    )


# --- hazards and the filter -------------------------------------------------


def dwell_hazards(
    fit: NbDwellFit, *, max_ticks: int = DWELL_MAX_TICKS
) -> tuple[float, ...]:
    """h[d - 1] = P(sojourn ends at age d | still open at d), for d = 1..D.

    The final entry is the hazard AT the cap and is held for every greater age
    by the filter, which is exact for r = 1 and asymptotically exact otherwise
    (module docstring). It is never forced to 1.0.
    """
    out: list[float] = []
    for d in range(1, max_ticks + 1):
        surv = nb_sf(d - 1, fit.r, fit.p)  # P(X >= d)
        if surv <= 0.0:
            out.append(1.0)
            continue
        h = math.exp(nb_log_pmf(d, fit.r, fit.p)) / surv
        out.append(min(max(h, 1e-12), 1.0))
    return tuple(out)


@dataclass(frozen=True)
class HsmmParams:
    """One route's debounce: a sojourn hazard per state plus the binary
    emission's two error rates.

    `false_call` is P(call reads not-normal | truly normal) and `miss` is
    P(call reads normal | truly not-normal). They are what makes this a
    debounce at all: with both at 0 the filter copies the raw calls.

    There is deliberately NO emission-tempering exponent. Down-weighting the
    call evidence against the duration prior is a separate modelling change
    with its own free parameter, and carrying one here — even defaulted to
    inert — would put a knob next to the gates that a later run could tune,
    confounding the duration-form comparison these gates exist to make.
    """

    normal_hazards: tuple[float, ...]
    not_moving_hazards: tuple[float, ...]
    false_call: float
    miss: float
    prior_not_moving: float = 0.5

    def emission(self, state: int, call_not_normal: bool) -> float:
        """P(observation | state). An abstention never reaches here: its
        likelihood is state-independent, so `filter_route` skips the emission
        entirely rather than multiplying both states by the same number
        (module docstring)."""
        if state == NOT_MOVING:
            b = (1.0 - self.miss) if call_not_normal else self.miss
        else:
            b = self.false_call if call_not_normal else (1.0 - self.false_call)
        return min(max(b, 1e-12), 1.0)


def params_from_fits(
    normal: NbDwellFit,
    not_moving: NbDwellFit,
    *,
    false_call: float,
    miss: float,
    max_ticks: int = DWELL_MAX_TICKS,
) -> HsmmParams:
    """Assemble a route's debounce from its two fitted sojourn distributions.

    The state prior is the two fits' own duty cycle — mean not-moving sojourn
    over the sum — so a route's first observed tick is judged against how much
    of its time that route actually spends not moving, not against a 50/50
    coin. There is no prior regime to protect at a route's first tick
    (`regime.py`), which is exactly why the prior has to come from somewhere
    principled.
    """
    mn, mm = normal.mean_ticks, not_moving.mean_ticks
    prior = mm / (mn + mm) if math.isfinite(mn + mm) and mn + mm > 0 else 0.5
    return HsmmParams(
        normal_hazards=dwell_hazards(normal, max_ticks=max_ticks),
        not_moving_hazards=dwell_hazards(not_moving, max_ticks=max_ticks),
        false_call=false_call,
        miss=miss,
        prior_not_moving=min(max(prior, 1e-6), 1.0 - 1e-6),
    )


@dataclass
class _Belief:
    """Forward message over the augmented state (state, elapsed ticks).

    Dense over elapsed age because the age distribution has support everywhere
    once abstentions and long regimes are in play; the last slot absorbs every
    age past the cap and keeps its hazard (see `dwell_hazards`).
    """

    alpha: tuple[list[float], list[float]]
    last_tick: int
    loglik: float


def _new_belief(params: HsmmParams, tick: int) -> _Belief:
    d = len(params.normal_hazards)
    a_normal = [0.0] * d
    a_moving = [0.0] * d
    a_normal[0] = 1.0 - params.prior_not_moving
    a_moving[0] = params.prior_not_moving
    return _Belief((a_normal, a_moving), tick, 0.0)


def _advance(belief: _Belief, params: HsmmParams) -> None:
    """One tick of the sojourn clock: age every open regime, and move the
    hazard mass into age 1 of the other state.

    This runs on EVERY tick including abstentions, which is the behavioural
    difference from a k-of-k debounce: time passes while the route is unseen.
    """
    hz = (params.normal_hazards, params.not_moving_hazards)
    cap = len(hz[0]) - 1
    out: tuple[list[float], list[float]] = ([0.0] * (cap + 1), [0.0] * (cap + 1))
    leaving = [0.0, 0.0]
    for s in (NORMAL, NOT_MOVING):
        a = belief.alpha[s]
        h = hz[s]
        dst = out[s]
        for d in range(cap + 1):
            m = a[d]
            if m == 0.0:
                continue
            hazard = h[d]
            leaving[s] += m * hazard
            stay = m * (1.0 - hazard)
            # The cap absorbs: age cap+1 and beyond share the cap's hazard.
            dst[min(d + 1, cap)] += stay
    out[NORMAL][0] += leaving[NOT_MOVING]
    out[NOT_MOVING][0] += leaving[NORMAL]
    belief.alpha = out


def _apply_call(belief: _Belief, params: HsmmParams, call_not_normal: bool) -> None:
    """Multiply in the binary emission and renormalize, accumulating the
    marginal log-likelihood of the observation."""
    for s in (NORMAL, NOT_MOVING):
        b = params.emission(s, call_not_normal)
        a = belief.alpha[s]
        for d in range(len(a)):
            if a[d]:
                a[d] *= b
    total = sum(belief.alpha[NORMAL]) + sum(belief.alpha[NOT_MOVING])
    if total <= 0.0:
        # Both states assign the observation zero mass. Restart the clock
        # rather than propagate a dead message.
        fresh = _new_belief(params, belief.last_tick)
        belief.alpha = fresh.alpha
        belief.loglik += math.log(1e-300)
        return
    belief.loglik += math.log(total)
    for s in (NORMAL, NOT_MOVING):
        a = belief.alpha[s]
        for d in range(len(a)):
            if a[d]:
                a[d] /= total


def _map_state(belief: _Belief) -> int:
    return (
        NOT_MOVING
        if sum(belief.alpha[NOT_MOVING]) > sum(belief.alpha[NORMAL])
        else NORMAL
    )


@dataclass(frozen=True)
class RouteFilterResult:
    """One route's filtered state per observed tick, plus the marginal
    log-likelihood of its call sequence under the params."""

    states: tuple[tuple[int, int], ...]
    loglik: float
    n_scored: int


def filter_route(
    points: Sequence[tuple[int, str]],
    params: HsmmParams,
    *,
    max_idle_sec: int = MAX_IDLE_SEC,
) -> RouteFilterResult:
    """Filter one route's raw calls into a published state per observed tick.

    FORWARD ONLY. The state emitted at tick t uses calls up to and including t
    and nothing later, so this is a rule that could run online. Smoothing would
    improve every number here and could not ship, which is why it is absent
    rather than optional.

    A gap longer than `max_idle_sec` drops the route's belief entirely and the
    next call opens a fresh regime — the regime clock's dropout rule, kept so
    that offline and online cannot disagree about what a blind hour means.
    """
    belief: _Belief | None = None
    out: list[tuple[int, int]] = []
    loglik = 0.0
    for tick, call in points:
        if belief is not None and tick - belief.last_tick > max_idle_sec:
            loglik += belief.loglik
            belief = None
        if belief is None:
            belief = _new_belief(params, tick)
        else:
            gap_ticks = max(1, (tick - belief.last_tick) // TICK_SECONDS)
            for _ in range(gap_ticks):
                _advance(belief, params)
        _apply_call(belief, params, call in _NOT_NORMAL_CALLS)
        belief.last_tick = tick
        out.append((tick, _map_state(belief)))
    if belief is not None:
        loglik += belief.loglik
    return RouteFilterResult(tuple(out), loglik, len(out))


def hsmm_states(
    calls: Sequence[tuple[int, Mapping[str, str]]],
    params_by_route: Mapping[str, HsmmParams],
    default_params: HsmmParams,
    *,
    max_idle_sec: int = MAX_IDLE_SEC,
) -> list[tuple[int, dict[str, str]]]:
    """The debounced published-state stream, shaped exactly like
    `movement_validation.published_states` so the same graders score both.

    The model is binary but the published vocabulary is not, so a not-moving
    decode is emitted under the LABEL of the most recent not-normal raw call on
    that route (`suspended` if that is what the classifier said, else
    `disrupted`). Collapsing both arms to `disrupted` would quietly change what
    the suspended arm's own grade is measured on; the debounce decides WHETHER
    the route is not-moving, and has no business relabelling which kind.

    Unlike the regime clock this does NOT hold a route's state through an
    abstention in the output: a route is emitted only at ticks it was called
    on. Holding it would mean publishing a state no observation supports, which
    is the stale-disrupted false alarm `movement_validation` grades the
    published surface for. Every span grader here counts only ticks the route
    was actually judged on, so an unemitted tick is an absence of evidence
    rather than a normal call.
    """
    series = _route_series(calls)
    decoded: dict[str, dict[int, int]] = {}
    for route, points in series.items():
        params = params_by_route.get(route, default_params)
        res = filter_route(points, params, max_idle_sec=max_idle_sec)
        decoded[route] = dict(res.states)

    out: list[tuple[int, dict[str, str]]] = []
    last_label: dict[str, str] = {}
    for tick, per_route in sorted(calls, key=lambda t: t[0]):
        emitted: dict[str, str] = {}
        for route, call in per_route.items():
            if call in _NOT_NORMAL_CALLS:
                last_label[route] = call
            state = decoded.get(route, {}).get(tick)
            if state is None:
                continue
            if state == NOT_MOVING:
                emitted[route] = last_label.get(route, "disrupted")
            else:
                emitted[route] = "normal"
        out.append((tick, emitted))
    return out


def stream_loglik(
    calls: Sequence[tuple[int, Mapping[str, str]]],
    params_by_route: Mapping[str, HsmmParams],
    default_params: HsmmParams,
    *,
    max_idle_sec: int = MAX_IDLE_SEC,
) -> tuple[float, int]:
    """(marginal log-likelihood, n scored route-ticks) of a whole call stream.

    This is what the emission error rates are profiled against
    (`fit_emissions`): they are the only free parameters left once the sojourn
    fits are in hand, so maximising the stream's own marginal likelihood over
    them is an MLE and not a tuning pass.
    """
    total = 0.0
    n = 0
    for route, points in _route_series(calls).items():
        res = filter_route(
            points,
            params_by_route.get(route, default_params),
            max_idle_sec=max_idle_sec,
        )
        total += res.loglik
        n += res.n_scored
    return total, n


# --- fitting: where the sojourn distribution has to come from ---------------
#
# THE CALL STREAM ALONE DOES NOT IDENTIFY THIS MODEL. MEASURED.
# -------------------------------------------------------------
# The natural-looking approach is to fit the sojourn distribution and the
# emission error rates jointly on the movement calls, and it does not work —
# not because of an optimiser, but because the two explanations of a one-tick
# not-normal call are observationally equivalent. "A genuine one-tick episode"
# and "a false call" predict exactly the same single tick, so the data cannot
# separate them and the likelihood is free to prefer whichever is cheaper.
#
# It prefers "genuine". On a synthetic stream carrying 32 known-false one-tick
# calls against 156 genuine not-moving ticks, a joint hard-EM fit from a
# blip-trusting start reached a self-consistent fixed point with the emission
# error rate driven to its floor and an over-dispersed dwell (r_hat = 0.315,
# early hazard 0.55) that treats one-tick episodes as ordinary. Its decode
# reproduced the raw calls tick for tick — a debounce that debounces nothing.
# Adjudicating between starts on the MARGINAL likelihood, which is the only
# criterion a hard-EM fixed point cannot game, did not rescue it: trusting
# -485.5 against sceptical -874.4. The pass-through is not an artefact of a bad
# start or a hard assignment. It is the higher-likelihood explanation, because
# an over-dispersed sojourn absorbs blips more cheaply than a false-call
# channel can.
#
# SO THE DWELL IS PINNED FROM OUTSIDE THE SIGNAL
# ----------------------------------------------
# The sojourn distribution is fitted on the ARCHIVED EPISODE DURATIONS, whose
# onsets and recoveries are adjudicated by the assigned_n supply feed —
# independent-in-derivation from the vehicle-position flow signal being
# debounced (movement_validation's TRUTH section). That is not a convenience:
# a duration model taken from the calls is a duration model of the calls'
# noise, and the whole purpose of the debounce is to disagree with the calls
# sometimes.
#
# With the dwell pinned externally, the emission channel is the only free
# parameter left and IS identified: given a fixed hazard, the rate at which
# calls disagree with a fitted persistence is a measurable quantity, and
# `fit_emissions` is a one-criterion profile MLE over it on the train window.
#
# This is also what keeps Gate 2 honest. The nest test is run on the same
# fixed, externally adjudicated durations for both arms, so the geometric null
# is never asked to account for a segmentation an NB alternative produced.


def episode_dwell_observations(
    episodes: Sequence[object],
    *,
    window_start: int | None = None,
    window_end: int | None = None,
) -> DwellPopulation:
    """Archived episodes -> sojourn observations, the externally adjudicated
    dwell population.

    Accepts anything carrying `route`, `onset`, `recovery` and — optionally —
    `right_censored` / `left_censored`, which covers both `episodes.Episode`
    and `load_r2.Disruption` (whose fields are `start_tick` / `recovered_tick`)
    without this module importing either. Duck-typed on purpose: the two
    episode products are built by different arms and neither should have to
    grow a shared base class to be gradeable here.

    `window_start` / `window_end` make the CAUSAL split: an episode is admitted
    only if its onset falls inside the window, so a train-window fit cannot see
    a duration that begins in the eval window. An episode still open at
    `window_end` is right-censored at the boundary rather than dropped —
    dropping it would delete exactly the long episodes and bias the fit toward
    the memoryless short-dwell reading.
    """
    obs: list[DwellObs] = []
    n_left = 0
    for ep in episodes:
        route = getattr(ep, "route", None)
        onset = getattr(ep, "onset", None)
        recovery = getattr(ep, "recovery", None)
        if onset is None:
            onset = getattr(ep, "start_tick", None)
        if recovery is None:
            recovery = getattr(ep, "recovered_tick", None)
        if not isinstance(route, str) or not isinstance(onset, int):
            raise ValueError(f"episode carries no route/onset to measure: {ep!r}")
        if window_start is not None and onset < window_start:
            # Onset outside the window: its duration is not this window's to
            # measure, and if it predates the window its onset is unobserved.
            n_left += 1
            continue
        if window_end is not None and onset >= window_end:
            continue
        if getattr(ep, "left_censored", False):
            n_left += 1
            continue
        censored = bool(getattr(ep, "right_censored", False))
        if isinstance(recovery, int):
            end = recovery
        elif window_end is not None:
            # A still-open episode has no recovery to read. It is censored AT
            # the window boundary, not unusable: skipping it would drop exactly
            # the longest dwells in the population and bias the fit toward the
            # short memoryless reading this arm is testing against.
            end = window_end
            censored = True
        else:
            raise ValueError(
                "episode has no recovery and no window_end to censor it at, so "
                f"its duration is unmeasurable: {ep!r}"
            )
        if window_end is not None and end > window_end:
            end = window_end
            censored = True
        ticks = max(1, (end - onset) // TICK_SECONDS)
        obs.append(DwellObs(route, ticks, censored=censored))
    return DwellPopulation(tuple(obs), n_left)


@dataclass(frozen=True)
class EpisodePopulations:
    """Both states' sojourn populations, both read off the SAME episode
    archive over the same window."""

    not_moving: DwellPopulation
    normal: DwellPopulation


def episode_populations(
    episodes: Sequence[object],
    *,
    window_start: int,
    window_end: int,
    routes: Sequence[str] | None = None,
) -> EpisodePopulations:
    """Both sojourn populations from the archived episodes and the window.

    THE NORMAL SOJOURNS ARE THE GAPS. A route's episodes partition its window,
    so the intervals between one episode's recovery and the next one's onset
    are its normal sojourns, and they are known from the same adjudicated
    record — no movement call is consulted for either state.

    That matters because the alternative fails in a measurable way. Reading the
    normal dwell off the CALLS lets a false call split one long normal run into
    two short ones, which inflates the fitted hazard of leaving normal, which
    makes entering not-moving cheap, which makes the next false call easier to
    believe. Measured on a synthetic stream with 32 injected false calls and
    the normal dwell taken from the calls: the filter stayed a pass-through,
    reproducing the raw confusion matrix exactly (156/32/10/5802). The entry
    hazard is the term that decides whether a blip can flip the state, so it is
    the one term that must not be estimated from the blips.

    `routes` names the routes whose window should contribute a normal sojourn
    even though they had NO episode at all. Those are the steadiest routes and
    they carry the longest normal dwell in the fleet; leaving them out because
    the episode archive has no row for them would bias the entry hazard upward
    exactly as the call-derived version does. Omit it and only routes with at
    least one episode contribute.

    The leading gap on each route is left-censored and dropped — the route was
    already running normally before the window, so that sojourn's onset is
    unobserved. The trailing gap is right-censored and kept.
    """
    not_moving = episode_dwell_observations(
        episodes, window_start=window_start, window_end=window_end
    )
    by_route: dict[str, list[tuple[int, int]]] = {}
    for r in routes or ():
        by_route.setdefault(r, [])
    for ep in episodes:
        route = getattr(ep, "route", None)
        onset = getattr(ep, "onset", None) or getattr(ep, "start_tick", None)
        recovery = getattr(ep, "recovery", None) or getattr(ep, "recovered_tick", None)
        if not isinstance(route, str) or not isinstance(onset, int):
            continue
        if onset < window_start or onset >= window_end:
            continue
        end = recovery if isinstance(recovery, int) else window_end
        by_route.setdefault(route, []).append((onset, min(end, window_end)))

    obs: list[DwellObs] = []
    n_left = 0
    for route, spans in by_route.items():
        spans.sort()
        cursor = window_start
        for i, (onset, end) in enumerate(spans):
            if onset > cursor:
                ticks = (onset - cursor) // TICK_SECONDS
                if ticks >= 1:
                    if i == 0:
                        # LEFT-TRUNCATED, AND NOT A SURVIVAL TERM.
                        #
                        # This run was already in progress at window_start, so
                        # what is observed is its RESIDUAL life. The likelihood
                        # of a residual life is conditional on having survived
                        # to the boundary — an integral over the unobserved
                        # elapsed age — and is NOT P(X >= ticks), which is the
                        # survival of a sojourn that starts fresh. Scoring it
                        # as an ordinary censored observation would be a
                        # different model, so it is dropped and counted, which
                        # is the convention the rest of this repo's duration
                        # fits already use for left-censored runs.
                        n_left += 1
                    else:
                        obs.append(DwellObs(route, ticks, censored=False))
            cursor = max(cursor, end)
        if window_end > cursor:
            ticks = (window_end - cursor) // TICK_SECONDS
            if ticks >= 1:
                if spans:
                    # Onset observed (the preceding recovery), end cut by the
                    # window: an ordinary right-censored observation.
                    obs.append(DwellObs(route, ticks, censored=True))
                else:
                    # No episode all window: onset unobserved as well, so this
                    # is left-truncated too and dropped for the same reason.
                    n_left += 1
    # WHICH WAY THIS BIASES, STATED.
    #
    # The dropped runs are the leading gaps and the never-disrupted routes,
    # i.e. the longest quiet stretches in the fleet. Fitting the normal sojourn
    # without them shortens it, which RAISES the fitted hazard of leaving
    # normal, which makes the filter readier to enter not-moving on thin
    # evidence. That is a bias against the candidate on the false-alarm gate
    # and toward it on detection, so a candidate that holds the false-alarm
    # bound here holds it conservatively. `n_left_censored` reports how much
    # evidence this convention refused.
    return EpisodePopulations(not_moving, DwellPopulation(tuple(obs), n_left))


@dataclass(frozen=True)
class DwellModel:
    """The pinned sojourn distributions: per route where a route can support
    its own two parameters, pooled everywhere else.

    BOTH FAMILIES CARRY THE SAME ROUTE STRUCTURE, and that symmetry is
    load-bearing rather than tidy. `by_route` holds the NB fits and
    `by_route_geometric` the r = 1 fits of the SAME per-route observations, so
    a route that gets its own negative binomial also gets its own geometric.
    Serving the pooled geometric to every route while NB got per-route
    resolution would hand the control arm a handicap and quietly turn gates 1
    and 3 into a comparison of pooling against per-route fitting rather than of
    one duration family against another.

    Both arms of the nest test are carried, per route and pooled, because the
    gate is stated on the pooled population but a pooled positive built out of
    one route's shape is a different claim from a broad one — and only the
    per-route table can tell them apart.
    """

    pooled: NbDwellFit
    pooled_geometric: NbDwellFit
    by_route: dict[str, NbDwellFit]
    by_route_geometric: dict[str, NbDwellFit]
    nest: NestTest
    nest_by_route: dict[str, NestTest]
    population: DwellPopulation
    min_route_events: int

    @property
    def n_own_fits(self) -> int:
        return len(self.by_route)

    def fit_for(self, route: str, *, geometric: bool = False) -> NbDwellFit:
        """That route's own fit in the requested family, else the pooled one.

        The two families are looked up through one function so a caller cannot
        accidentally pair a per-route fit on one arm with a pooled fit on the
        other.
        """
        if geometric:
            return self.by_route_geometric.get(route, self.pooled_geometric)
        return self.by_route.get(route, self.pooled)

    def routes(self) -> list[str]:
        """Routes carrying their own fit, in both families by construction."""
        return sorted(self.by_route)


def fit_dwell_model(
    population: DwellPopulation,
    *,
    min_route_events: int = MIN_ROUTE_DWELL_EVENTS,
) -> DwellModel:
    """Fit both duration arms on one fixed, externally adjudicated population.

    Per-route fits require `min_route_events` COMPLETED episodes on that route:
    a route whose every episode is still open at a window edge has no observed
    duration to fit a hazard shape to, however many ticks it contributes.

    A route that clears that bar gets its own fit in BOTH families, from the
    same observations. The r = 1 arm is the control, and a control served a
    pooled curve while the candidate gets per-route resolution is not a control
    (see `DwellModel`).
    """
    by_route: dict[str, NbDwellFit] = {}
    by_route_geometric: dict[str, NbDwellFit] = {}
    nest_by_route: dict[str, NestTest] = {}
    for route, obs in population.by_route().items():
        if sum(1 for o in obs if not o.censored) >= min_route_events:
            by_route[route] = fit_dwell(obs)
            by_route_geometric[route] = fit_dwell(obs, fix_r=1.0)
            nest_by_route[route] = nest_test(obs)
    return DwellModel(
        pooled=fit_dwell(population.obs),
        pooled_geometric=fit_dwell(population.obs, fix_r=1.0),
        by_route=by_route,
        by_route_geometric=by_route_geometric,
        nest=nest_test(population.obs),
        nest_by_route=nest_by_route,
        population=population,
        min_route_events=min_route_events,
    )


# Grid the emission error rates are profiled over. Log-spaced because the
# interesting region is small rates — a false-call rate of 0.5 is not a
# hypothesis anyone holds — and coarse-then-refine because each evaluation is a
# full forward pass over the train window.
# The `miss` rate genuinely can be large: it is P(the call reads normal | the
# route is not moving), and on a signal whose not-normal calls are 0.13% of
# ticks a real episode is mostly read as normal. The first real run maximised
# at 0.3, which was the top of a grid that stopped there, so the reported fit
# was CLAMPED rather than converged. The ceiling is now 0.8: past that the
# emission carries essentially no information about the state and the filter is
# running on its duration prior alone, which is a conclusion rather than a
# parameter.
_EMISSION_GRID: tuple[float, ...] = (
    0.0005,
    0.002,
    0.008,
    0.03,
    0.1,
    0.3,
    0.55,
    0.8,
)


@dataclass(frozen=True)
class EmissionFit:
    """The emission channel's two error rates and the fit that produced them.

    `at_bound` flags a rate that maximised on the first or last grid point. A
    parameter sitting on its search boundary did not converge inside the space
    the search offered, so the number is a clamp and not an estimate — the same
    distinction `NbDwellFit.r_at_bound` draws, reported for the same reason.
    """

    false_call: float
    miss: float
    loglik: float
    n_scored: int
    n_evaluations: int
    at_bound: tuple[bool, bool] = (False, False)


def _neighbours(x: float, grid: Sequence[float]) -> tuple[float, ...]:
    """`x` plus the geometric midpoints either side of it in `grid` — the
    refinement round's candidates."""
    i = grid.index(x)
    out: list[float] = [x]
    if i > 0:
        out.append(math.sqrt(grid[i - 1] * x))
    if i < len(grid) - 1:
        out.append(math.sqrt(grid[i + 1] * x))
    return tuple(out)


def fit_emissions(
    calls: Sequence[tuple[int, Mapping[str, str]]],
    dwell: DwellModel,
    normal_dwell: NbDwellFit,
    *,
    max_ticks: int = DWELL_MAX_TICKS,
    grid: Sequence[float] = _EMISSION_GRID,
    refine: bool = True,
    max_idle_sec: int = MAX_IDLE_SEC,
    geometric: bool = False,
) -> EmissionFit:
    """Profile-likelihood MLE of the two emission error rates on the train
    window's calls, holding the externally pinned sojourn fits fixed.

    Two parameters, one criterion (the call stream's marginal log-likelihood
    under the forward filter), a fixed deterministic grid with one refinement
    round. Identified precisely because the dwell is not free here: the
    degeneracy that sank the joint fit was the sojourn distribution's freedom
    to absorb blips, and pinning it externally is what removes that freedom.

    `geometric=True` profiles the emissions against the r = 1 arm instead, so
    the incumbent's own duration form gets its emission channel fitted on the
    same criterion. Grading the two duration families with one family's
    emissions would confound the comparison with a handicap.
    """
    evaluated = 0

    def build(fa: float, miss: float) -> tuple[dict[str, HsmmParams], HsmmParams]:
        # `fit_for(route, geometric=...)` keeps the route structure IDENTICAL
        # across the two families. Substituting the pooled geometric here
        # (which this did) gave the candidate per-route resolution and the
        # control a single pooled curve, so gates 1 and 3 would have compared
        # pooling against per-route fitting rather than one duration family
        # against the other.
        by_route = {
            r: params_from_fits(
                normal_dwell,
                dwell.fit_for(r, geometric=geometric),
                false_call=fa,
                miss=miss,
                max_ticks=max_ticks,
            )
            for r in dwell.routes()
        }
        default = params_from_fits(
            normal_dwell,
            dwell.pooled_geometric if geometric else dwell.pooled,
            false_call=fa,
            miss=miss,
            max_ticks=max_ticks,
        )
        return by_route, default

    def score(fa: float, miss: float) -> float:
        nonlocal evaluated
        evaluated += 1
        by_route, default = build(fa, miss)
        ll, _ = stream_loglik(calls, by_route, default, max_idle_sec=max_idle_sec)
        return ll

    best = (grid[0], grid[0], -math.inf)
    for fa in grid:
        for miss in grid:
            ll = score(fa, miss)
            if ll > best[2]:
                best = (fa, miss, ll)
    if refine:
        # Neighbours are taken around the COARSE best, which is the point that
        # is actually on the grid. Reading them off `best` after it has already
        # been refined asks the grid for the index of a value that is by
        # construction between two of its entries.
        coarse_fa, coarse_miss = best[0], best[1]
        for fa in _neighbours(coarse_fa, grid):
            for miss in _neighbours(coarse_miss, grid):
                if fa == coarse_fa and miss == coarse_miss:
                    continue
                ll = score(fa, miss)
                if ll > best[2]:
                    best = (fa, miss, ll)
    by_route, default = build(best[0], best[1])
    _, n_scored = stream_loglik(calls, by_route, default, max_idle_sec=max_idle_sec)
    edges = (grid[0], grid[-1])
    return EmissionFit(
        best[0],
        best[1],
        best[2],
        n_scored,
        evaluated,
        at_bound=(best[0] in edges, best[1] in edges),
    )


@dataclass(frozen=True)
class Debounce:
    """A fitted debounce, ready to publish a state stream.

    `geometric` marks the r = 1 arm — the incumbent's memoryless form carried
    through the identical filter, emissions refitted on the same criterion. It
    exists so gates 1 and 3 can be read as a comparison between two duration
    families rather than between this module and `regime.py`, which differ in
    more than the dwell.
    """

    dwell: DwellModel
    normal_dwell: NbDwellFit
    emissions: EmissionFit
    max_ticks: int
    geometric: bool

    def params_by_route(self) -> dict[str, HsmmParams]:
        # Same route structure on both arms — see `DwellModel`. This used to
        # rebuild one pooled geometric fit and serve it to every route, which
        # made the control arm a pooling comparison instead of a duration-form
        # comparison.
        return {
            r: params_from_fits(
                self.normal_dwell,
                self.dwell.fit_for(r, geometric=self.geometric),
                false_call=self.emissions.false_call,
                miss=self.emissions.miss,
                max_ticks=self.max_ticks,
            )
            for r in self.dwell.routes()
        }

    def default_params(self) -> HsmmParams:
        return params_from_fits(
            self.normal_dwell,
            self.dwell.pooled_geometric if self.geometric else self.dwell.pooled,
            false_call=self.emissions.false_call,
            miss=self.emissions.miss,
            max_ticks=self.max_ticks,
        )

    def states(
        self,
        calls: Sequence[tuple[int, Mapping[str, str]]],
        *,
        max_idle_sec: int = MAX_IDLE_SEC,
    ) -> list[tuple[int, dict[str, str]]]:
        return hsmm_states(
            calls,
            self.params_by_route(),
            self.default_params(),
            max_idle_sec=max_idle_sec,
        )


def fit_debounce(
    calls: Sequence[tuple[int, Mapping[str, str]]],
    dwell: DwellModel,
    normal_dwell: NbDwellFit,
    *,
    max_ticks: int = DWELL_MAX_TICKS,
    max_idle_sec: int = MAX_IDLE_SEC,
    geometric: bool = False,
) -> Debounce:
    """Assemble a debounce from two externally pinned sojourn fits plus the
    train-window calls.

    BOTH sojourn distributions come from the episode archive
    (`episode_populations`), and only the emission channel is fitted on the
    calls. The normal-state fit is not a decoration: its hazard is the term
    that decides whether one disagreeing call can flip the state, and taking it
    from the calls lets a false call fragment a long normal run, raise that
    hazard, and make the next false call easier to believe. That loop was
    measured — the filter stayed a pass-through on a stream with 32 known-false
    calls in it. The entry hazard must not be estimated from the blips.
    """
    emissions = fit_emissions(
        calls,
        dwell,
        normal_dwell,
        max_ticks=max_ticks,
        max_idle_sec=max_idle_sec,
        geometric=geometric,
    )
    return Debounce(
        dwell=dwell,
        normal_dwell=normal_dwell,
        emissions=emissions,
        max_ticks=max_ticks,
        geometric=geometric,
    )
