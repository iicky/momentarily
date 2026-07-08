"""Aalen-Johansen cause-specific cumulative incidence for disrupted-regime
dwell.

A disrupted regime ends one of two ways: it clears back to `normal`, or it
escalates to `suspended`. That's a competing-risks problem — one clock (time
since regime entry) races two causes, and only the first one to fire is
observed; the other becomes hypothetical at that instant. `training.dwell`'s
Kaplan-Meier curve answers "when does the regime end, of any cause" but
throws the cause away once it's estimated. The Worker currently recovers a
cause split with a *homogeneous* approximation (see worker/src/snapshot.ts):
the all-cause exit probability (from the KM/tail curve) times a fixed "share
of exits that go to normal", read off the trained HMM's stationary
transition ratios. That flat multiplier is only exact under proportional
hazards — i.e. if the two causes' hazards stay in constant proportion for
the whole regime, so the destination mix never shifts as the regime ages. In
practice it does shift: quick delay clearances often resolve to `normal`
early, while regimes that survive long enough skew toward `suspended` (or
the reverse, depending on the route/alert mix), so a single flat ratio
either over- or under-states P(normal) depending on how far into the dwell
distribution the query lands.

The Aalen-Johansen estimator fixes this by tracking each cause's cumulative
incidence function (CIF) directly and non-parametrically from the observed
exit-cause counts at each event time, instead of factoring "when" and
"which" apart. Over ordered distinct event times t_1 < ... < t_k (times at
which one or more completed exits, of any cause, are observed):

    n(t)    = number at risk just before t (duration >= t; a censored
              observation counts toward the risk set right up through its
              own duration, the same convention as the plain KM estimator).
    d_c(t)  = completed exits to cause c at exactly t.
    d(t)    = sum_c d_c(t), all-cause exits at t.

    S(t)     = S(t_prev) * (1 - d(t)/n(t))                    S(0) = 1
    CIF_c(t) = CIF_c(t_prev) + S(t_prev) * d_c(t)/n(t)        CIF_c(0) = 0

`S` is the ordinary Kaplan-Meier all-cause survival curve — the complement
of what `training.dwell.km_cdf_points` computes. The increment for cause c
uses `S(t_prev)`, survival just *before* t: of the mass still un-exited an
instant before t, the d_c(t)/n(t) fraction of it that exits to c at t is
what gets credited to CIF_c(t). Using S(t) instead of S(t_prev) would
double-count the very mass that's leaving at t. At every t, CIF_c(t) summed
over all causes plus S(t) equals 1 — nothing is lost between "exited" and
"still disrupted", it's just partitioned finer than plain KM.

Standalone module: deliberately does not import training.dwell or
training.survival (both model all-cause dwell) to avoid a cycle, since this
is their finer-grained sibling estimator, not a consumer of theirs.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any

# One observation for the estimator: (duration_sec, cause). `cause` is the
# destination state a completed exit landed in ("normal" or "suspended");
# None means the regime was still open at the observation boundary
# (right-censored) — we know dwell > duration_sec, not which cause or when.
CompetingSample = tuple[int, str | None]


@dataclass(frozen=True)
class CIFResult:
    """Aalen-Johansen output for one batch of competing-risks dwell samples.

    `causes` is the distinct non-None cause labels observed, sorted. `cif`
    holds one ascending (event_time_sec, CIF_c(t)) step-point list per
    cause — every key has at least one point, since a cause only enters
    `causes` by occurring at least once. `survival` is the shared ascending
    (event_time_sec, S(t)) overall KM step-point list. All three are empty
    together when `samples` had no completed events at all (empty input, or
    every observation censored).
    """

    causes: tuple[str, ...]
    cif: dict[str, list[tuple[int, float]]]
    survival: list[tuple[int, float]]


def cif_curves(samples: list[CompetingSample]) -> CIFResult:
    """Aalen-Johansen cause-specific CIF curves plus the shared overall KM
    survival curve they're built from — see the module docstring for the
    increment formula.

    Ties (multiple samples at the same duration) are resolved together: the
    at-risk count only drops once the whole tied group has been charged
    against it, matching `training.dwell.km_cdf_points`. A censored-only
    tail (or a wholly-censored input) never adds a step, since it never
    supplies an event to charge — the curves just stop at the last real
    event and hold there under `cif_at`/`survival_at`'s step lookup.
    """
    if not samples:
        return CIFResult(causes=(), cif={}, survival=[])

    causes = tuple(sorted({cause for _, cause in samples if cause is not None}))
    # Sort by duration only: cause is `str | None`, and None isn't orderable
    # against str, so a plain full-tuple sort would raise on a tie between a
    # censored and a completed observation at the same duration.
    ordered = sorted(samples, key=lambda sample: sample[0])
    n_total = len(ordered)

    survival = 1.0
    cif_running = dict.fromkeys(causes, 0.0)
    cif_points: dict[str, list[tuple[int, float]]] = {cause: [] for cause in causes}
    survival_points: list[tuple[int, float]] = []

    at_risk = n_total
    i = 0
    while i < n_total:
        t = ordered[i][0]
        deaths: Counter[str] = Counter()
        ties = 0
        while i < n_total and ordered[i][0] == t:
            cause = ordered[i][1]
            if cause is not None:
                deaths[cause] += 1
            ties += 1
            i += 1
        d_all = sum(deaths.values())
        if d_all > 0:
            survival_prev = survival
            survival = survival_prev * (1.0 - d_all / at_risk)
            survival_points.append((t, survival))
            for cause, d_c in deaths.items():
                cif_running[cause] += survival_prev * d_c / at_risk
                cif_points[cause].append((t, cif_running[cause]))
        at_risk -= ties

    return CIFResult(causes=causes, cif=cif_points, survival=survival_points)


def _step_at(points: list[tuple[int, float]], t: float, before: float) -> float:
    """Value of the ascending step function `points` at time t: the value of
    the last step at or before t, or `before` if t precedes the first step
    (or `points` is empty)."""
    value = before
    for step_t, step_v in points:
        if step_t > t:
            break
        value = step_v
    return value


def cif_at(points: list[tuple[int, float]], t: float) -> float:
    """CIF_c(t) from one cause's point list: 0.0 before that cause's first
    event."""
    return _step_at(points, t, before=0.0)


def survival_at(points: list[tuple[int, float]], t: float) -> float:
    """S(t) from the overall KM survival point list: 1.0 before the first
    event anywhere (nobody has exited yet)."""
    return _step_at(points, t, before=1.0)


def conditional_cif(
    result: CIFResult, cause: str, elapsed_sec: float, horizon_sec: float
) -> float:
    """P(exits to `cause` within horizon_sec | still disrupted at elapsed_sec).

    = (CIF_cause(elapsed+horizon) - CIF_cause(elapsed)) / S(elapsed): of the
    mass still un-exited at `elapsed`, the fraction the AJ curves say exits
    specifically to `cause` in the next `horizon_sec`. The competing-risks
    analogue of `training.dwell.conditional_recover_by`, which answers the
    all-cause version of the same question.

    0.0 if nobody survives to `elapsed` (S(elapsed) <= 0, so the conditional
    probability is undefined) or `cause` was never observed in `result`.
    Clamped to [0, 1] to absorb finite-sample estimator noise (e.g. a
    horizon reaching past the last observed event, where the numerator is a
    stale plateau that roundoff could otherwise nudge past S(elapsed)).
    """
    points = result.cif.get(cause)
    if points is None:
        return 0.0
    denom = survival_at(result.survival, elapsed_sec)
    if denom <= 0.0:
        return 0.0
    numer = cif_at(points, elapsed_sec + horizon_sec) - cif_at(points, elapsed_sec)
    return min(max(numer / denom, 0.0), 1.0)


def result_as_dict(result: CIFResult) -> dict[str, Any]:
    """JSON-serializable form of a CIFResult (params.json sidecar shape)."""
    return asdict(result)
