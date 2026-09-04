"""Causal grading harness for the movement recovery-dwell arm.

WHAT THIS ANSWERS
-----------------
Does the published movement dwell forecast beat climatology on CRPS, and if
not, is the residual a fixable model-form gap or the floor predictability of a
population that is mostly one-tick partial freezes?

The numbers this arm has been steered by so far (journal 2026-08-12: continuous
-0.1391, single global atom -0.0582, per-route shrunk atom -0.0606) were
produced by throwaway scripts that no longer exist, so they cannot be refreshed
when the archive grows or the movement truth changes -- and it HAS changed since
(journal 2026-08-26: the baseline desaturation plus through-stop filtering took
the same window from 14 episodes to 56, adding the multi-tick tail the old cut
was blind to). This module is that grading, committed.

THE SPLIT IS CAUSAL, AND IT IS THE WHOLE POINT
----------------------------------------------
Cells are fitted on [train_start, train_end] and graded on episodes that begin
after it. Nothing from the eval window reaches a fitted curve. The one
deliberate exception is the ORACLE CRPS baseline: recovery_dist_report's
`oracle_baseline_crps` is the graded population's OWN empirical duration CDF,
which is hindsight the model does not get. It is reported because every number
already on the record was quoted in it -- but it means "oracle_skill < 0" is
not by itself proof of a bad model. `km_pooled` below is what separates the
two: it is the same climatology fitted CAUSALLY, so the gap between its skill
and zero is the hindsight advantage, and only what is left over is the model's.

VARIANTS
--------
Every variant differs ONLY in how a cell is built from the same train-window
samples, and is graded through the same unmodified seam
(scorecard.episode_recovery -> recovery_dist.recovery_dist_report):

  shipped      per-route shrunk one-tick point mass + conditional log-logistic
               tail, exactly as train_em._movement_dwell publishes it
  global_atom  same, with every route's atom rate replaced by the population
               rate -- the "does per-route resolution buy anything" control
  continuous   the pre-mixture form: one log-logistic over the whole population
  pooled       shipped form fitted on all routes as a single cell, then served
               to every route -- the unconditional-median hypothesis in
               distribution form (journal 2026-08-26)
  km_pooled    Kaplan-Meier body over pooled train durations with a fitted
               log-logistic tail: nonparametric causal climatology

PIT COMPARABILITY, WHICH IS NOT SYMMETRIC ACROSS VARIANTS
---------------------------------------------------------
episode_recovery hands the grader a jump left-limit (`pred_left`) only for a
cell carrying an explicit atom, so the mixture variants get their one-tick
episodes spread across the jump while `km_pooled` -- whose quantile curve has a
flat run at one tick, i.e. the same point mass expressed differently -- does
not, and stacks them in one bin. That deflates km_pooled's PIT shape and NOT
its CRPS. So skill is compared across all variants; PIT shape only among the
atom-carrying ones. Reported rather than silently patched: fixing it means
teaching episode_recovery to read a left limit off a repeated knot, which is a
change to a shared grading file, not to this harness.

THE FIT DEPENDS ON A PER-TICK CENSUS, AND THE DEFICIT IS NARROWER THAN IT LOOKS
------------------------------------------------------------------------------
Right-censored observations come from the replay's open-regime map
(movement_backfill.open_regimes_from_ticks), built from the per-tick census.
Withholding it does NOT turn censoring off: dwell.dwell_samples_by_cell falls
back to dwell._open_regimes, which reads each route's open regime off the
new_state of its LAST transition record. So censoring survives for every route
that moved inside the window, and `--no-census` means "censoring restricted to
routes that moved".

The real deficit is routes with NO transition in the window — there is no
record to read a regime off, so they contribute neither an event nor a censored
observation, and since a route completes a `normal` regime only by leaving
normal, that blind spot covers exactly the steadiest routes. It bites the
mixture specifically: pooled_dwell._atom_counts credits a censored observation
as evidence ABOUT the point mass once it has outlived the atom and never as an
atom itself, so a never-transitioned route sitting in a disrupted regime is
pure NON-atom evidence, and dropping it can only push the fitted one-tick rate
UP while also costing that route its cell.

That distinction matters for reading any --no-census number: transition-derived
inference can also hand a route a SPURIOUS censored observation, so a fixture
whose movers still sit in open not-normal regimes at the boundary measures two
effects at once. The isolation check that separates them is asserting that a
censused fit with the never-transitioned routes withheld reproduces
`--no-census` exactly; only then is the remaining delta theirs. This module's
test does exactly that.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from training.dwell import (
    DwellQuantiles,
    DwellSample,
    dwell_samples_by_cell,
    make_cell,
)
from training.episodes import Episode, extract_episodes
from training.eval import NOT_NORMAL, STATES, TICK_SECONDS, MovementTransitionRecord
from training.eval_common import snap_tick
from training.movement_backfill import (
    open_regimes_from_ticks,
    ticks_for,
    transitions_from_ticks,
)
from training.pooled_dwell import (
    ATOM_MIN_P,
    MIN_VOTER_EVENTS,
    AtomFit,
    atom_fits,
    cell_from_fit,
    mixture_cell,
    partially_pooled_dwell,
    pooled_dwell_cells,
)
from training.r2_client import load_config, make_client
from training.recovery_dist import (
    RecoveryDistSample,
    predicted_recovery_curve,
    recovery_dist_report,
)
from training.regime import DEBOUNCE_TICKS
from training.scorecard import movement_dwell_lookup_from_params
from training.survival import loglogistic_tail

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

# The key every pooled variant fits under. Not a real route id, and never
# published — `pooled`/`km_pooled` serve this one cell to every route.
_POOLED_KEY = "*"

VARIANTS = ("shipped", "global_atom", "continuous", "pooled", "km_pooled")

# States the movement arm can produce an episode in. `normal` is fitted too
# (routes dwell in it for hours) but is never a graded episode: an episode IS a
# not-normal run.
_NOT_NORMAL = tuple(s for s in STATES if s != "normal")


def aligned_window(start: date, end: date) -> tuple[int, int]:
    """Tick-aligned UTC epochs covering [start, end+1day). Same convention as
    train_em._aligned_window, so a window named here means the same span it
    would mean to the trainer."""
    start_dt = datetime(start.year, start.month, start.day, tzinfo=UTC)
    end_dt = datetime(end.year, end.month, end.day, tzinfo=UTC) + timedelta(days=1)
    return (
        int(start_dt.timestamp()) // TICK_SECONDS * TICK_SECONDS,
        int(end_dt.timestamp()) // TICK_SECONDS * TICK_SECONDS,
    )


# --- cell construction: one function per variant, all on train samples ------


def _samples_by_route(
    transitions: Sequence[MovementTransitionRecord],
    state: str,
    *,
    window_end: int,
    open_regimes: Mapping[str, tuple[str, int]] | None,
) -> dict[str, list[DwellSample]]:
    by_cell = dwell_samples_by_cell(
        transitions, window_end=window_end, open_regimes=open_regimes
    )
    return {r: s for (r, st), s in by_cell.items() if st == state}


def _mixture_cells(
    samples_by_route: dict[str, list[DwellSample]],
    *,
    atom_sec: int,
    force_parent_p: bool,
) -> tuple[dict[str, DwellQuantiles], dict[str, AtomFit]]:
    """The shipped mixture, optionally with per-route resolution switched off.

    Deliberately re-composes the same public pieces pooled_dwell_cells uses
    rather than calling it, because `global_atom` needs to reach between the
    atom fit and the cell render. `shipped` does NOT come through here — it
    calls pooled_dwell_cells directly, so what is graded is the production path
    and not a re-implementation of it that could drift from it.
    """
    atoms = atom_fits(samples_by_route, atom_sec)
    if not atoms:
        return {}, {}
    parent_p = next(iter(atoms.values())).parent_p
    if parent_p < ATOM_MIN_P:
        return {}, atoms
    if force_parent_p:
        atoms = {
            r: AtomFit(
                route=a.route,
                p=parent_p,
                raw=a.raw,
                n_atom=a.n_atom,
                n_informative=a.n_informative,
                parent_p=parent_p,
                source="global",
            )
            for r, a in atoms.items()
        }
    tail_by_route = {
        r: [(d, c) for d, c in s if float(d) > atom_sec]
        for r, s in samples_by_route.items()
    }
    if sum(len(s) for s in tail_by_route.values()) < 2:
        return {}, atoms
    tail_fits = partially_pooled_dwell(tail_by_route, truncate_at=float(atom_sec))
    cells = {
        route: mixture_cell(fit, atoms[route], atom_sec)
        for route, fit in tail_fits.items()
        if route in atoms
    }
    return cells, atoms


def _continuous_cells(
    samples_by_route: dict[str, list[DwellSample]],
) -> dict[str, DwellQuantiles]:
    return {
        r: cell_from_fit(f) for r, f in partially_pooled_dwell(samples_by_route).items()
    }


def _pooled_to_one_cell(
    samples_by_route: dict[str, list[DwellSample]], *, atom_sec: int
) -> tuple[dict[str, DwellQuantiles], dict[str, AtomFit]]:
    """Collapse every route's samples into one cell, then serve it to all of
    them. Tests whether per-route conditioning adds anything at all over the
    unconditional distribution."""
    pooled = [s for samples in samples_by_route.values() for s in samples]
    if not pooled:
        return {}, {}
    cells, atoms = _mixture_cells(
        {_POOLED_KEY: pooled}, atom_sec=atom_sec, force_parent_p=False
    )
    cell = cells.get(_POOLED_KEY)
    if cell is None:
        # No atom here — same fall-through pooled_dwell_cells takes, so this
        # variant stays a pooling change rather than also becoming a coverage
        # change.
        cell = _continuous_cells({_POOLED_KEY: pooled}).get(_POOLED_KEY)
        if cell is None:
            return {}, atoms
    return dict.fromkeys(samples_by_route, cell), atoms


def _km_pooled_cell(
    samples_by_route: dict[str, list[DwellSample]],
) -> dict[str, DwellQuantiles]:
    """Nonparametric causal climatology: one Kaplan-Meier body over every
    route's train durations, with a fitted log-logistic tail for past-the-curve
    horizons. No atom — the point mass shows up as a flat run in the quantile
    curve instead, which is why its PIT is not comparable (see module docstring).
    """
    pooled = [s for samples in samples_by_route.values() for s in samples]
    if not pooled:
        return {}
    cell = make_cell(pooled, tail_fn=loglogistic_tail)
    return dict.fromkeys(samples_by_route, cell)


def _empty_cells() -> dict[str, dict[str, DwellQuantiles]]:
    return {}


def _empty_atoms() -> dict[str, dict[str, AtomFit]]:
    return {}


@dataclass
class FittedVariant:
    """One variant's dwell_movement block plus the atom diagnostics behind it."""

    name: str
    cells: dict[str, dict[str, DwellQuantiles]] = field(default_factory=_empty_cells)
    atoms: dict[str, dict[str, AtomFit]] = field(default_factory=_empty_atoms)


def fit_variant(
    variant: str,
    transitions: Sequence[MovementTransitionRecord],
    *,
    window_end: int,
    open_regimes: Mapping[str, tuple[str, int]] | None,
    atom_sec: int = TICK_SECONDS,
) -> FittedVariant:
    """Fit one variant's {route: {state: cell}} block off train transitions."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    out = FittedVariant(name=variant)
    for state in STATES:
        if variant == "shipped":
            # The production path itself, not a copy of it.
            cells = pooled_dwell_cells(
                transitions,
                state=state,
                window_end=window_end,
                open_regimes=open_regimes,
                atom_sec=atom_sec,
            )
            atoms: dict[str, AtomFit] = {}
        else:
            samples = _samples_by_route(
                transitions, state, window_end=window_end, open_regimes=open_regimes
            )
            if not samples:
                continue
            if variant == "continuous":
                cells = _continuous_cells(samples)
                atoms = {}
            elif variant == "global_atom":
                cells, atoms = _mixture_cells(
                    samples, atom_sec=atom_sec, force_parent_p=True
                )
                if not cells:
                    # Same fall-through pooled_dwell_cells takes where the
                    # population has no spike, so this variant differs from
                    # `shipped` only in atom resolution and not in coverage.
                    cells = _continuous_cells(samples)
            elif variant == "pooled":
                cells, atoms = _pooled_to_one_cell(samples, atom_sec=atom_sec)
            else:  # km_pooled
                cells, atoms = _km_pooled_cell(samples), {}
        for route, cell in cells.items():
            out.cells.setdefault(route, {})[state] = cell
        if atoms:
            out.atoms[state] = atoms
    return out


# --- the graded population --------------------------------------------------


def eval_episodes(
    ticks: Sequence[tuple[int, Mapping[str, str]]],
    *,
    window_start: int,
    window_end: int,
) -> list[Episode]:
    """Graded episodes over the eval window, segmented the way the scorecard's
    movement arm segments them (scorecard.episode_scorecard): raw per-tick
    movement calls through episodes.extract_episodes, standing advisories held
    out. Censored episodes stay in — episode_recovery counts and drops them
    itself, and its count is worth seeing.

    Ticks are SNAPPED to the 5-minute grid, and that is load-bearing rather
    than defensive. extract_episodes walks the grid and reads truth[(route,
    tick)] at exact grid epochs, but the prediction stream's `ts` is the actual
    publish time (…210, …132 — the cron's own jitter), so an unsnapped truth
    map never gets hit once and every population silently grades as zero
    episodes. The vehicles source hides the bug: build_movement_truth is
    already grid-keyed, so it was only the production `predictions` source that
    came back empty.

    An episode is a NOT_NORMAL run — disrupted or suspended — and NOT merely
    "anything that is not normal". The prediction stream also carries
    `not_scheduled`, which is the absence of scheduled service rather than a
    disruption, and admitting it grades overnight service gaps as recovery
    forecasts: measured on 2026-08-28..09-03 it took the one-tick share from
    0.61 to 0.034 and the distinct-duration count from 4 to 19, i.e. it
    replaces the population this arm models with a different one. The vehicles
    source never exposed this either, because build_movement_truth emits only
    normal/disrupted/suspended.
    """
    truth = {
        (route, snap_tick(tick)): state
        for tick, calls in ticks
        for route, state in calls.items()
        if window_start <= tick < window_end and state in NOT_NORMAL
    }
    eps = extract_episodes(
        truth, {}, window_start=window_start, window_end=window_end - TICK_SECONDS
    )
    return [e for e in eps if not e.standing]


def duration_histogram(eps: Sequence[Episode]) -> dict[str, Any]:
    """Tick-count histogram of the graded population, with the one-tick share.

    This is the number the floor argument rests on: if the population is
    overwhelmingly one tick, no amount of conditional structure can beat
    predicting one tick, because there is nothing left to condition on.
    """
    completed = [e for e in eps if not (e.left_censored or e.right_censored)]
    ticks = Counter(e.duration_sec // TICK_SECONDS for e in completed)
    n = len(completed)
    return {
        "n_completed": n,
        "n_distinct_durations": len(ticks),
        "one_tick_share": (ticks.get(1, 0) / n) if n else None,
        "by_ticks": {str(k): ticks[k] for k in sorted(ticks)},
    }


# --- grading ----------------------------------------------------------------
#
# TWO SKILLS, AND ONLY ONE OF THEM IS A FAIR COMPARISON BETWEEN VARIANTS.
#
# recovery_dist_report's `oracle_skill` is measured against the empirical CDF
# of the graded population's own durations — perfect hindsight the model never
# gets. It is carried through under the same name because it is the figure
# every prior number for this arm was quoted in, and dropping it would make the
# refresh incomparable to the record. It is NOT the causal comparison.
#
# `causal_skill` is 1 - CRPS_variant / CRPS_km_pooled: the same climatology,
# fitted on the train window only, so both sides are forecasts.
#
# Both are computed on the MATCHED population — the episodes every graded
# variant can score. Variants disagree on coverage (a route with no train
# observation gets no cell, and which routes those are depends on the variant),
# and both a CRPS mean and an oracle baseline built from a different set of
# episodes are a different measurement, not a comparable one.


def _episode_samples(
    cells: dict[str, dict[str, DwellQuantiles]], eps: Sequence[Episode]
) -> tuple[dict[str, RecoveryDistSample], int, int]:
    """{episode key: sample} for the episodes this cell block can score, plus
    (n_censored, n_no_curve).

    Mirrors scorecard.episode_recovery's sample construction exactly — same
    censoring rule, same n_no_curve rule, same elapsed-0 forecast, same atom
    left-limit — and is pinned to it by a test that asserts the two agree on
    full coverage. It exists only because the matched-population comparison
    above needs the samples keyed per episode, which episode_recovery
    aggregates away.
    """
    lookup = movement_dwell_lookup_from_params({"dwell_movement": cells})
    out: dict[str, RecoveryDistSample] = {}
    n_censored = 0
    n_no_curve = 0
    for e in eps:
        if e.left_censored or e.right_censored:
            n_censored += 1
            continue
        cell = lookup(e.route, e.peak_state, e.cause)
        if cell is None or len(cell[0]) < 2:
            n_no_curve += 1
            continue
        curve_sec, tail_ll, atom = cell
        pred_left = (
            0.0 if atom is not None and abs(e.duration_sec - atom[1]) < 1.0 else None
        )
        key = f"{e.route}:{e.onset}"
        out[key] = RecoveryDistSample(
            pred_curve=predicted_recovery_curve(0.0, curve_sec, tail_ll, atom),
            actual_min=e.duration_sec / 60.0,
            regime_key=key,
            pred_left=pred_left,
        )
    return out, n_censored, n_no_curve


# A run that grades nothing, or grades everything on an empty population, must
# SAY SO rather than print a table of NaN. recovery_dist_report([]) returns a
# full report object whose every field is NaN, which formats as a normal-looking
# row — so the failure mode this guards is not a crash, it is a plausible table
# that means nothing. One sparse variant is enough to cause it: the matched
# population is an intersection, so a single variant with zero coverable
# episodes empties it for everyone.
#
# That is not a hypothetical regime. It is what a just-recovering transition
# stream looks like, which is exactly when this harness gets pointed at a thin
# window.


@dataclass(frozen=True)
class GradeBlocked(Exception):
    """The batch cannot produce comparable numbers, and why.

    Carries the per-variant coverage so the message can name WHICH variant
    emptied the intersection — with five variants and one culprit, "no matched
    population" alone sends the reader back to re-run them one at a time.
    """

    reason: str
    coverage: dict[str, int]
    culprits: tuple[str, ...]

    def __str__(self) -> str:
        if self.reason == "no_fits":
            return "no variants to grade"
        cover = ", ".join(f"{n}={c}" for n, c in sorted(self.coverage.items()))
        if self.reason == "variant_covers_nothing":
            return (
                "cannot grade: "
                + ", ".join(self.culprits)
                + " can score 0 of the eval window's episodes, which empties the "
                f"matched population for every variant (coverage: {cover}). "
                "Drop the variant with --variants, or widen the train window so "
                "it has cells for the routes that appear in eval."
            )
        return (
            "cannot grade: the variants have no episode in common, so the "
            f"matched population is empty (coverage: {cover}). Widen the train "
            "window, or grade fewer variants."
        )


def grade_variants(
    fits: Sequence[FittedVariant], eps: Sequence[Episode]
) -> dict[str, Any]:
    """Grade every variant on the episodes all of them can score.

    Raises GradeBlocked when no comparable population exists — see the note
    above on why an empty population is more dangerous than an exception here.

    `causal_skill` needs km_pooled in the batch; without it that column is
    None rather than silently falling back to the oracle baseline, and
    `causal_baseline` reads None so the omission is visible in the output.
    """
    if not fits:
        raise GradeBlocked(reason="no_fits", coverage={}, culprits=())
    built: dict[str, tuple[dict[str, RecoveryDistSample], int, int]] = {
        f.name: _episode_samples(f.cells, eps) for f in fits
    }
    coverage: dict[str, int] = {
        name: len(samples) for name, (samples, _c, _n) in built.items()
    }
    barren = tuple(sorted(n for n, c in coverage.items() if c == 0))
    if barren:
        raise GradeBlocked(
            reason="variant_covers_nothing", coverage=coverage, culprits=barren
        )
    # Folded rather than set.intersection(*...): the varargs form erases the
    # element type, and these keys are the episode ids every diagnostic below
    # reports on.
    key_sets: list[set[str]] = [set(samples) for samples, _c, _n in built.values()]
    matched: set[str] = set(key_sets[0])
    for keys in key_sets[1:]:
        matched &= keys
    if not matched:
        raise GradeBlocked(reason="no_common_episode", coverage=coverage, culprits=())
    ordered = sorted(matched)
    # None: this harness takes its causal comparison from km_pooled's own CRPS
    # (see the note above), not from an ECDF baseline inside the report, so the
    # report contributes only the oracle column here.
    reports = {
        name: recovery_dist_report(
            [samples[k] for k in ordered], baseline_durations_min=None
        )
        for name, (samples, _c, _n) in built.items()
    }
    climatology = reports.get("km_pooled")
    rows: list[dict[str, Any]] = []
    for fitted in fits:
        samples, n_censored, n_no_curve = built[fitted.name]
        report = reports[fitted.name]
        causal = (
            1.0 - report.mean_crps / climatology.mean_crps
            if climatology is not None and climatology.mean_crps > 0
            else None
        )
        rows.append(
            {
                "variant": fitted.name,
                "n_coverable": len(samples),
                "n_censored_excluded": n_censored,
                "n_no_curve": n_no_curve,
                "mean_crps": report.mean_crps,
                "oracle_baseline_crps": report.oracle_baseline_crps,
                "oracle_skill": report.oracle_skill,
                "causal_skill": causal,
                "mean_pit": report.mean_pit,
                "pit": report.pit,
                "atoms": atom_report(fitted),
            }
        )
    return {
        "n_matched": len(matched),
        "causal_baseline": "km_pooled" if climatology is not None else None,
        "variants": rows,
    }


def atom_report(fitted: FittedVariant) -> dict[str, Any]:
    """Per-route atom shrinkage for the not-normal states, or None where the
    variant publishes no atom. Answers what the shrinkage actually did: whether
    each route's own rate was trusted (`own`) or pulled to the population rate
    (`pooled`), and how far it moved."""
    out: dict[str, Any] = {}
    for state, by_route in fitted.atoms.items():
        if state not in _NOT_NORMAL or not by_route:
            continue
        any_fit = next(iter(by_route.values()))
        out[state] = {
            "parent_p": any_fit.parent_p,
            "routes": {
                r: {
                    "p": a.p,
                    "raw": a.raw,
                    "n_atom": a.n_atom,
                    "n_informative": a.n_informative,
                    "source": a.source,
                    "shrinkage": a.raw - a.p,
                }
                for r, a in sorted(by_route.items())
            },
        }
    return out


def published_atom_report(fitted: FittedVariant) -> dict[str, Any]:
    """`shipped` fits through pooled_dwell_cells, which returns cells and not
    AtomFits, so its atom behaviour is read back off the published cells."""
    out: dict[str, Any] = {}
    for route, by_state in sorted(fitted.cells.items()):
        for state, cell in by_state.items():
            if state not in _NOT_NORMAL:
                continue
            atom_p = cell.get("atom_p")
            atom_sec = cell.get("atom_sec")
            # Both keys or neither: a cell carrying only one is not a usable
            # mixture (dwell._atom_params rejects it), so it is not an atom to
            # report either.
            if atom_p is None or atom_sec is None:
                continue
            out.setdefault(state, {})[route] = {
                "atom_p": atom_p,
                "atom_sec": atom_sec,
                "n": cell["n"],
            }
    return out


def cell_quality(fitted: FittedVariant) -> dict[str, int]:
    """Cell COUNT is the wrong health metric for a dwell block; these are the
    quantities that move.

    Comparing the replay reconstruction against the committed transition
    stream, cell count held up (41 vs 44, inside a 10% bar) while `n_own` fell
    16 -> 7 and `n_atom` 42 -> 18: nearly as many cells, but fewer than half
    carrying a one-tick point mass and most shrunk to the population rather
    than self-supported. A criterion gated on count alone is blind to exactly
    that.

    Attribution matters here and is easy to get wrong, because this module also
    has a census section and the two effects look alike. Those figures are a
    SOURCE-VOLUME effect — that comparison ran 273 transitions against 93 — and
    NOT a censoring effect. The controlled census measurement holds transitions
    fixed and varies only the open-regime map, and it moves these counts barely
    at all (43 -> 41 cells, n_own 20 -> 20, n_atom 42 -> 40, every grade
    identical to three decimals). Thin evidence per cell is what degrades cell
    quality; withholding the census is not.
    """
    n_cells = n_own = n_atom = 0
    for by_state in fitted.cells.values():
        for cell in by_state.values():
            n_cells += 1
            if cell["n"] >= MIN_VOTER_EVENTS:
                n_own += 1
            if "atom_p" in cell:
                n_atom += 1
    return {"n_cells": n_cells, "n_own": n_own, "n_atom": n_atom}


@dataclass(frozen=True)
class GradeInputs:
    """Everything the grade needs, fetched once."""

    ticks: Sequence[tuple[int, Mapping[str, str]]]
    train_end_epoch: int
    eval_start_epoch: int
    eval_end_epoch: int


def load_inputs(
    client: S3Client,
    bucket: str,
    *,
    train_start: date,
    train_end: date,
    eval_end: date,
    source: str,
    scope: str = "route",
) -> GradeInputs:
    """One archive pass over the whole span, split by tick afterwards. Fetching
    train and eval separately would decode the boundary day twice and risk the
    two halves disagreeing about it."""
    train_start_epoch, train_end_epoch = aligned_window(train_start, train_end)
    _, eval_end_epoch = aligned_window(train_end, eval_end)
    ticks = ticks_for(
        client,
        bucket,
        train_start,
        eval_end,
        scope=scope,
        source=source,
    )
    return GradeInputs(
        ticks=[(t, c) for t, c in ticks if train_start_epoch <= t < eval_end_epoch],
        train_end_epoch=train_end_epoch,
        eval_start_epoch=train_end_epoch,
        eval_end_epoch=eval_end_epoch,
    )


def grade(
    inputs: GradeInputs,
    *,
    variants: Sequence[str] = VARIANTS,
    debounce_ticks: int = DEBOUNCE_TICKS,
    census: bool = True,
) -> dict[str, Any]:
    """Fit every variant on the train half and grade all of them on the same
    eval episodes. One episode population, one baseline, one difference.

    `census=False` withholds the open-regime map. It does NOT turn censoring
    off — dwell_samples_by_cell falls back to inferring each mover's open
    regime from its last transition record — so what it measures is the loss of
    the routes that never transitioned. See the census section of the module
    docstring, including the isolation check that makes such a delta
    attributable.
    """
    train_ticks = [(t, c) for t, c in inputs.ticks if t < inputs.train_end_epoch]
    eval_ticks = [(t, c) for t, c in inputs.ticks if t >= inputs.eval_start_epoch]

    transitions = transitions_from_ticks(
        train_ticks, "route", debounce_ticks=debounce_ticks
    )
    open_regimes = (
        open_regimes_from_ticks(train_ticks, debounce_ticks=debounce_ticks) or None
        if census
        else None
    )
    eps = eval_episodes(
        eval_ticks,
        window_start=inputs.eval_start_epoch,
        window_end=inputs.eval_end_epoch,
    )

    fits = [
        fit_variant(
            variant,
            transitions,
            window_end=inputs.train_end_epoch,
            open_regimes=open_regimes,
        )
        for variant in variants
    ]
    graded = grade_variants(fits, eps)
    empty_atoms: dict[str, Any] = {}
    published = next(
        (published_atom_report(f) for f in fits if f.name == "shipped"),
        empty_atoms,
    )

    return {
        "train": {
            "n_ticks": len(train_ticks),
            "n_transitions": len(transitions),
            "censoring": "census" if census else "transition_inferred",
            # What actually reached the fit, not the size of the map that was
            # withheld: with the census off, censoring is still inferred for
            # every route that moved, and reporting 0 here would misdescribe it.
            "n_censored_samples": sum(
                1
                for samples in dwell_samples_by_cell(
                    transitions,
                    window_end=inputs.train_end_epoch,
                    open_regimes=open_regimes,
                ).values()
                for _d, completed in samples
                if not completed
            ),
            "cells": {f.name: cell_quality(f) for f in fits},
        },
        "eval": {
            "n_ticks": len(eval_ticks),
            "n_episodes": len(eps),
            "n_matched": graded["n_matched"],
            "causal_baseline": graded["causal_baseline"],
            "durations": duration_histogram(eps),
        },
        "published_atoms": published,
        "variants": graded["variants"],
    }


# --- CLI --------------------------------------------------------------------


def _fmt(x: Any) -> str:
    if x is None:
        return "-"
    return f"{x:+.4f}" if isinstance(x, float) else str(x)


def _print_report(report: dict[str, Any]) -> None:
    tr = report["train"]
    ev = report["eval"]
    dur = ev["durations"]
    print(
        f"train: {tr['n_ticks']} ticks, {tr['n_transitions']} transitions, "
        f"censoring={tr['censoring']} ({tr['n_censored_samples']} censored "
        f"samples)"
    )
    for name, q in tr["cells"].items():
        print(
            f"       {name:<13} cells={q['n_cells']:>3} own={q['n_own']:>3} "
            f"atom={q['n_atom']:>3}"
        )
    print()
    share = dur["one_tick_share"]
    print(
        f"eval:  {ev['n_ticks']} ticks, {ev['n_episodes']} episodes, "
        f"{ev['n_matched']} scored on the matched population "
        f"({dur['n_completed']} completed, {dur['n_distinct_durations']} distinct "
        f"durations, one-tick share "
        f"{'-' if share is None else format(share, '.3f')})"
    )
    print(f"       duration ticks: {dur['by_ticks']}")
    print(f"       causal baseline: {ev['causal_baseline']}")
    print()
    print(
        f"{'variant':<13} {'cover':>6} {'nocurve':>7} {'CRPS':>7} "
        f"{'causal':>9} {'oracle':>9} {'PIT':>6}  pit histogram"
    )
    for row in report["variants"]:
        print(
            f"{row['variant']:<13} {row['n_coverable']:>6} {row['n_no_curve']:>7} "
            f"{row['mean_crps']:>7.3f} {_fmt(row['causal_skill']):>9} "
            f"{_fmt(row['oracle_skill']):>9} {row['mean_pit']:>6.3f}  {row['pit']}"
        )
    print()
    for state, by_route in report["published_atoms"].items():
        print(f"published atom_p ({state}):")
        for route, a in sorted(by_route.items()):
            print(f"  {route:<4} atom_p={a['atom_p']:.4f} n={a['n']}")
    for row in report["variants"]:
        for state, block in row["atoms"].items():
            print(
                f"{row['variant']} atom shrinkage ({state}), parent_p="
                f"{block['parent_p']:.4f}:"
            )
            for route, a in block["routes"].items():
                print(
                    f"  {route:<4} raw={a['raw']:.3f} -> p={a['p']:.3f} "
                    f"({a['source']}, {a['n_atom']}/{a['n_informative']})"
                )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Causal CRPS/PIT grade of the movement dwell arm against "
        "climatology, one variant per cell-construction form"
    )
    p.add_argument("--train-start", required=True, help="YYYY-MM-DD")
    p.add_argument(
        "--train-end", required=True, help="YYYY-MM-DD, inclusive; eval starts after"
    )
    p.add_argument("--eval-end", required=True, help="YYYY-MM-DD, inclusive")
    p.add_argument(
        "--source",
        default="predictions",
        choices=("predictions", "vehicles", "auto"),
        help="tick source; predictions (default) is the replay source the "
        "published movement dwell fit reads",
    )
    p.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help=f"comma-separated subset of {','.join(VARIANTS)}",
    )
    p.add_argument("--json", action="store_true", help="emit the raw report as JSON")
    p.add_argument(
        "--no-census",
        action="store_true",
        help="withhold the open-regime map and infer censoring from transition "
        "records alone — the model-form cost of a transitions-only source",
    )
    args = p.parse_args(argv)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    unknown = [v for v in variants if v not in VARIANTS]
    if unknown:
        print(f"unknown variants: {unknown}", file=sys.stderr)
        return 2

    cfg = load_config()
    client = make_client(cfg)
    inputs = load_inputs(
        client,
        cfg.bucket,
        train_start=date.fromisoformat(args.train_start),
        train_end=date.fromisoformat(args.train_end),
        eval_end=date.fromisoformat(args.eval_end),
        source=args.source,
    )
    try:
        report = grade(inputs, variants=variants, census=not args.no_census)
    except GradeBlocked as blocked:
        # Nonzero and on stderr: this tool is run from scripts and its silence
        # would otherwise be read as "graded fine, all NaN".
        print(f"movement dwell grade: {blocked}", file=sys.stderr)
        print(
            json.dumps(
                {
                    "blocked": blocked.reason,
                    "culprits": list(blocked.culprits),
                    "coverage": blocked.coverage,
                },
                indent=2,
            )
        )
        return 1
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
