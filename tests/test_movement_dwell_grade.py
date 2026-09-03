"""The causal grading harness for the movement dwell arm.

These tests pin the three properties that make its numbers trustworthy, all on
synthetic ticks so no archive is needed:

  - the train/eval split is causal — no eval-window duration reaches a cell;
  - `_episode_samples` really does mirror scorecard.episode_recovery, so the
    harness is grading through the production seam rather than beside it;
  - every variant is scored on the SAME episodes, because a CRPS mean over a
    different population is a different measurement.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest

from training.dwell import DwellQuantiles
from training.episodes import Episode
from training.eval import TICK_SECONDS
from training.eval import snap_tick as _round_snap_tick
from training.eval_common import snap_tick
from training.load_r2 import (
    _snap_tick as _prod_snap_tick,  # pyright: ignore[reportPrivateUsage]
)
from training.movement_backfill import (
    open_regimes_from_ticks,
    transitions_from_ticks,
)
from training.movement_dwell_grade import (
    VARIANTS,
    FittedVariant,
    GradeBlocked,
    GradeInputs,
    _episode_samples,  # pyright: ignore[reportPrivateUsage]
    aligned_window,
    duration_histogram,
    eval_episodes,
    fit_variant,
    grade,
    grade_variants,
)
from training.recovery_dist import recovery_dist_report
from training.scorecard import episode_recovery, movement_dwell_lookup_from_params

_ROUTES = ("A", "B", "C")


def _episode_ticks(
    start: int, route: str, n_ticks: int, state: str = "disrupted"
) -> list[tuple[int, Mapping[str, str]]]:
    """One disruption episode: `n_ticks` not-normal calls, then a normal call so
    the regime clock and the episode segmenter both see the recovery."""
    ticks: list[tuple[int, Mapping[str, str]]] = [
        (start + i * TICK_SECONDS, {route: state}) for i in range(n_ticks)
    ]
    ticks.append((start + n_ticks * TICK_SECONDS, {route: "normal"}))
    return ticks


def _merge(
    *groups: Sequence[tuple[int, Mapping[str, str]]],
) -> list[tuple[int, Mapping[str, str]]]:
    by_tick: dict[int, dict[str, str]] = {}
    for group in groups:
        for tick, calls in group:
            by_tick.setdefault(tick, {}).update(calls)
    return sorted(by_tick.items())


def _population(
    origin: int, spacing_ticks: int = 12
) -> list[tuple[int, Mapping[str, str]]]:
    """A one-tick-dominated episode population on three routes: the shape the
    real movement arm produces (journal 2026-08-12: 70% one tick, a thin
    multi-tick tail), which is what every variant here exists to model."""
    groups: list[list[tuple[int, Mapping[str, str]]]] = []
    slot = 0
    for route in _ROUTES:
        for n_ticks in (1, 1, 1, 1, 1, 2, 3, 6):
            groups.append(
                _episode_ticks(
                    origin + slot * spacing_ticks * TICK_SECONDS, route, n_ticks
                )
            )
            slot += 1
    return _merge(*groups)


def _inputs() -> GradeInputs:
    origin = 1_780_000_000 // TICK_SECONDS * TICK_SECONDS
    train = _population(origin)
    train_end = max(t for t, _c in train) + TICK_SECONDS
    ev = _population(train_end)
    eval_end = max(t for t, _c in ev) + TICK_SECONDS
    return GradeInputs(
        ticks=_merge(train, ev),
        train_end_epoch=train_end,
        eval_start_epoch=train_end,
        eval_end_epoch=eval_end,
    )


def _fitted_and_episodes():
    inputs = _inputs()
    train_ticks = [(t, c) for t, c in inputs.ticks if t < inputs.train_end_epoch]
    eval_ticks = [(t, c) for t, c in inputs.ticks if t >= inputs.eval_start_epoch]
    transitions = transitions_from_ticks(train_ticks, "route")
    open_regimes = open_regimes_from_ticks(train_ticks) or None
    fitted = fit_variant(
        "shipped",
        transitions,
        window_end=inputs.train_end_epoch,
        open_regimes=open_regimes,
    )
    eps = eval_episodes(
        eval_ticks,
        window_start=inputs.eval_start_epoch,
        window_end=inputs.eval_end_epoch,
    )
    return fitted, eps


def test_episode_samples_mirrors_episode_recovery():
    """The harness reaches inside the seam to key samples per episode; if that
    mirror drifts from scorecard.episode_recovery, every number the harness
    prints is measuring a different model than production grades. Pin it: on
    full coverage the two must agree on the counts AND on the CRPS."""
    fitted, eps = _fitted_and_episodes()
    samples, n_censored, n_no_curve = _episode_samples(fitted.cells, eps)

    graded = episode_recovery(
        eps, movement_dwell_lookup_from_params({"dwell_movement": fitted.cells})
    )
    assert graded["n_scored"] == len(samples)
    assert graded["n_censored_excluded"] == n_censored
    assert graded["n_no_curve"] == n_no_curve
    mine = recovery_dist_report([samples[k] for k in samples])
    assert mine.mean_crps == graded["report"]["mean_crps"]
    assert mine.pit == graded["report"]["pit"]


def test_no_eval_window_duration_reaches_a_fitted_cell():
    """Causality, stated as a measurement rather than an intention: refit with
    the eval half's episodes made absurdly long and every published cell must be
    byte-identical, because the fit never saw them."""
    inputs = _inputs()
    train_ticks = [(t, c) for t, c in inputs.ticks if t < inputs.train_end_epoch]

    def cells_for(
        extra: Sequence[tuple[int, Mapping[str, str]]],
    ) -> dict[str, dict[str, DwellQuantiles]]:
        merged = _merge(train_ticks, extra)
        merged = [(t, c) for t, c in merged if t < inputs.train_end_epoch]
        return fit_variant(
            "shipped",
            transitions_from_ticks(merged, "route"),
            window_end=inputs.train_end_epoch,
            open_regimes=open_regimes_from_ticks(merged) or None,
        ).cells

    base = cells_for([])
    # A 40-tick monster entirely inside the eval half.
    monster = _episode_ticks(inputs.eval_start_epoch + TICK_SECONDS, "A", 40)
    assert cells_for(monster) == base


def test_every_variant_is_scored_on_the_same_episodes():
    """Variants disagree on coverage, and a CRPS mean over a different episode
    set is not comparable. The matched population is the fix; this pins that it
    is actually applied — one n_matched, and every variant's oracle baseline
    identical because the baseline is built from the shared actuals."""
    _fitted, eps = _fitted_and_episodes()
    inputs = _inputs()
    train_ticks = [(t, c) for t, c in inputs.ticks if t < inputs.train_end_epoch]
    transitions = transitions_from_ticks(train_ticks, "route")
    open_regimes = open_regimes_from_ticks(train_ticks) or None
    fits = [
        fit_variant(
            v, transitions, window_end=inputs.train_end_epoch, open_regimes=open_regimes
        )
        for v in VARIANTS
    ]
    graded = grade_variants(fits, eps)
    assert graded["n_matched"] > 0
    assert graded["causal_baseline"] == "km_pooled"
    baselines = {row["oracle_baseline_crps"] for row in graded["variants"]}
    assert len(baselines) == 1


def test_causal_skill_is_measured_against_the_causal_climatology():
    """km_pooled is the causal baseline, so its own causal_skill must be exactly
    zero — the identity that proves the ratio is taken against it and not
    against the report's hindsight baseline."""
    _fitted, eps = _fitted_and_episodes()
    inputs = _inputs()
    train_ticks = [(t, c) for t, c in inputs.ticks if t < inputs.train_end_epoch]
    transitions = transitions_from_ticks(train_ticks, "route")
    fits = [
        fit_variant(
            v,
            transitions,
            window_end=inputs.train_end_epoch,
            open_regimes=open_regimes_from_ticks(train_ticks) or None,
        )
        for v in ("shipped", "km_pooled")
    ]
    rows = {r["variant"]: r for r in grade_variants(fits, eps)["variants"]}
    assert rows["km_pooled"]["causal_skill"] == 0.0
    assert rows["shipped"]["causal_skill"] is not None


def test_shipped_publishes_a_one_tick_atom_on_this_population():
    """A one-tick-dominated population must produce a point mass at one tick —
    otherwise the harness would be grading the continuous form under the
    `shipped` label and silently reporting no change."""
    fitted, _eps = _fitted_and_episodes()
    disrupted = [
        cell
        for by_state in fitted.cells.values()
        for state, cell in by_state.items()
        if state == "disrupted"
    ]
    assert disrupted
    assert all(cell.get("atom_sec") == TICK_SECONDS for cell in disrupted)
    assert all(0.0 < (cell.get("atom_p") or 0.0) < 1.0 for cell in disrupted)


def test_duration_histogram_counts_only_completed_episodes():
    """The one-tick share is the number the floor argument rests on, so it must
    not be diluted by censored runs whose true duration is unknown."""
    onset = 1_780_000_000
    eps = [
        Episode(
            route="A",
            onset=onset,
            recovery=onset + TICK_SECONDS,
            peak_state="disrupted",
            cause="other",
            n_ticks=1,
            left_censored=False,
            right_censored=False,
        ),
        Episode(
            route="A",
            onset=onset + 10 * TICK_SECONDS,
            recovery=onset + 40 * TICK_SECONDS,
            peak_state="disrupted",
            cause="other",
            n_ticks=30,
            left_censored=False,
            right_censored=True,
        ),
    ]
    hist = duration_histogram(eps)
    assert hist["n_completed"] == 1
    assert hist["one_tick_share"] == 1.0
    assert hist["by_ticks"] == {"1": 1}


def test_grade_runs_end_to_end_on_synthetic_ticks():
    """The orchestrator, with no archive: every requested variant graded, on the
    one matched population, over a one-tick-dominated eval set."""
    report = grade(_inputs())
    assert {row["variant"] for row in report["variants"]} == set(VARIANTS)
    assert report["eval"]["n_matched"] > 0
    assert all(
        row["n_coverable"] >= report["eval"]["n_matched"] for row in report["variants"]
    )
    assert report["eval"]["durations"]["one_tick_share"] > 0.5
    assert report["published_atoms"]


def test_the_mixture_beats_the_continuous_form_on_a_one_tick_population():
    """The reason the mixture exists, as a test rather than a claim: on a
    population that is mostly one tick, front-loading a point mass there has to
    score better than a single continuous curve over the same samples. Compared
    on CRPS against the shared causal climatology, so it is not an artifact of
    either variant's own baseline."""
    _fitted, eps = _fitted_and_episodes()
    inputs = _inputs()
    train_ticks = [(t, c) for t, c in inputs.ticks if t < inputs.train_end_epoch]
    transitions = transitions_from_ticks(train_ticks, "route")
    open_regimes = open_regimes_from_ticks(train_ticks) or None
    fits = [
        fit_variant(
            v, transitions, window_end=inputs.train_end_epoch, open_regimes=open_regimes
        )
        for v in ("shipped", "continuous", "km_pooled")
    ]
    rows = {r["variant"]: r for r in grade_variants(fits, eps)["variants"]}
    assert rows["shipped"]["mean_crps"] < rows["continuous"]["mean_crps"]
    assert rows["shipped"]["causal_skill"] > rows["continuous"]["causal_skill"]


def test_a_cell_without_a_curve_lands_in_no_curve_not_a_crash():
    """Same convention as the production lookup: a route the train window never
    saw is a coverage gap, not an exception."""
    eps = [
        Episode(
            route="ZZZ",
            onset=1_780_000_000,
            recovery=1_780_000_000 + TICK_SECONDS,
            peak_state="disrupted",
            cause="other",
            n_ticks=1,
            left_censored=False,
            right_censored=False,
        )
    ]
    cells: dict[str, dict[str, DwellQuantiles]] = {}
    samples, n_censored, n_no_curve = _episode_samples(cells, eps)
    assert (len(samples), n_censored, n_no_curve) == (0, 0, 1)


def test_withholding_the_census_inflates_the_one_tick_rate():
    """The model-form cost of a transitions-only source, as a measurement.

    pooled_dwell._atom_counts credits a right-censored observation as evidence
    ABOUT the point mass only once it has outlived the atom, and never as an
    atom itself — so an open disrupted regime is pure non-atom evidence. Drop
    the open-regime map and that evidence goes with it, so the published
    one-tick rate can only rise, and routes that never transitioned lose their
    cell entirely. Independently reproduced against the transition-stream
    source on a fixture built to isolate it: atom_p 0.645 -> 0.750.
    """
    inputs = _inputs()
    # A route sitting in disrupted at the train boundary with no transition to
    # read it off: present in the census, invisible to the stream.
    open_route = [
        (inputs.train_end_epoch - i * TICK_SECONDS, {"OPEN": "disrupted"})
        for i in range(1, 6)
    ]
    ticks = _merge(inputs.ticks, open_route)

    def run(
        ticks_in: Sequence[tuple[int, Mapping[str, str]]], census: bool
    ) -> dict[str, Any]:
        return grade(
            GradeInputs(
                ticks=ticks_in,
                train_end_epoch=inputs.train_end_epoch,
                eval_start_epoch=inputs.eval_start_epoch,
                eval_end_epoch=inputs.eval_end_epoch,
            ),
            variants=("shipped",),
            census=census,
        )

    with_census = run(ticks, True)
    without = run(ticks, False)
    # Isolation check: the censused fit with the never-transitioned route
    # withheld must reproduce the no-census fit EXACTLY. Without this the delta
    # below could be a mix of two effects, because transition-derived inference
    # can also hand a mover a spurious censored observation.
    withheld = run(inputs.ticks, True)

    def atoms(report: dict[str, Any]) -> dict[str, float]:
        return {
            r: a["atom_p"] for r, a in report["published_atoms"]["disrupted"].items()
        }

    assert atoms(without) == atoms(withheld)
    assert with_census["train"]["censoring"] == "census"
    assert without["train"]["censoring"] == "transition_inferred"
    # Censoring is not off — the movers still get theirs inferred.
    assert without["train"]["n_censored_samples"] > 0
    # The never-transitioned route's non-atom evidence is what moves the rate.
    assert atoms(without)[_ROUTES[0]] > atoms(with_census)[_ROUTES[0]]
    assert "OPEN" in atoms(with_census)
    assert "OPEN" not in atoms(without)


def test_unaligned_publish_times_still_produce_episodes():
    """Regression, and it cost a whole measurement round.

    The prediction stream stamps `ts` with the actual publish time, not the
    5-minute grid — real values end …210, …132. extract_episodes walks the grid
    and looks truth up at exact grid epochs, so an unsnapped truth map scores
    ZERO episodes on a window that plainly contains disruptions, silently. It
    was invisible against the vehicles source, whose truth is already
    grid-keyed, and only the production `predictions` source came back empty.
    """
    origin = 1_780_000_000 // TICK_SECONDS * TICK_SECONDS
    jitter = 37  # cron lateness, well inside one tick
    ticks = [(origin + i * TICK_SECONDS + jitter, {"A": "disrupted"}) for i in range(3)]
    ticks.append((origin + 3 * TICK_SECONDS + jitter, {"A": "normal"}))

    eps = eval_episodes(
        ticks,
        window_start=origin,
        window_end=origin + 5 * TICK_SECONDS,
    )
    assert len(eps) == 1
    assert eps[0].duration_sec == 3 * TICK_SECONDS


def test_episode_grid_matches_the_production_truth_grid_not_the_rounding_snap():
    """Which snap_tick this harness uses is load-bearing, and the repo has two.

    training/eval.py:486 and training/review.py:125 ROUND to the nearest tick;
    training/eval_common.py:120 and training/load_r2.py:57 FLOOR. The grid that
    matters here is the one PRODUCTION TRUTH is keyed on, and that is the floor
    one: every production truth map snaps with load_r2._snap_tick, whose own
    docstring (load_r2.py:200-202) says it uses "the reconstruction's floor grid
    so the keys line up".

    So flooring is correct and rounding would be the bug: a publish 210s past a
    boundary would land on the NEXT tick and shift episode membership away from
    the population production grades. The rounding snap DOES appear in
    episodes.extract_episodes, but only on window_start/window_end, which this
    module passes pre-aligned — where rounding and flooring agree exactly.

    Pinned because a reviewer comparing the two definitions in isolation will
    reasonably conclude the harness should adopt the production scorecard's
    rounding, and that change would silently move every number.
    """
    boundary = 1_787_875_200
    assert boundary % TICK_SECONDS == 0

    # Agreement with the production truth grid, across a whole tick.
    assert all(
        snap_tick(boundary + off) == _prod_snap_tick(boundary + off)
        for off in range(TICK_SECONDS)
    )
    # And the rounding variant genuinely differs — so this is a real choice,
    # not two spellings of one thing.
    assert any(
        _round_snap_tick(boundary + off) != _prod_snap_tick(boundary + off)
        for off in range(TICK_SECONDS)
    )
    # On the pre-aligned bounds this module actually passes, they agree.
    start, end = aligned_window(date(2026, 8, 25), date(2026, 9, 3))
    for bound in (start, end):
        assert bound % TICK_SECONDS == 0
        assert _round_snap_tick(bound) == snap_tick(bound) == bound


def test_grading_no_variants_is_refused_not_a_typeerror():
    """set.intersection(*()) raises a bare TypeError from deep inside the
    grader; an empty batch is a caller mistake and should say so."""
    with pytest.raises(GradeBlocked) as caught:
        grade_variants([], [])
    assert caught.value.reason == "no_fits"
    assert "no variants to grade" in str(caught.value)


def test_one_barren_variant_does_not_silently_poison_every_grade():
    """The dangerous edge, and the reason this refuses rather than warns.

    The matched population is an INTERSECTION, so a single variant that can
    score nothing empties it for everyone — and recovery_dist_report([]) does
    not crash, it returns a full report whose every field is NaN. That formats
    as an ordinary-looking table of ordinary-looking rows that mean nothing. So
    the failure being pinned here is a plausible wrong answer, not an error.

    The diagnostic must NAME the culprit: with five variants and one at fault,
    "no matched population" alone sends the reader back to re-run them singly.
    """
    fitted, eps = _fitted_and_episodes()
    barren = FittedVariant(name="km_pooled")  # fitted nothing: no cells at all

    with pytest.raises(GradeBlocked) as caught:
        grade_variants([fitted, barren], eps)

    blocked = caught.value
    assert blocked.reason == "variant_covers_nothing"
    assert blocked.culprits == ("km_pooled",)
    # The healthy variant's coverage is still reported, so the message shows
    # that the batch was fine apart from the named culprit.
    assert blocked.coverage["km_pooled"] == 0
    assert blocked.coverage[fitted.name] > 0
    assert "km_pooled" in str(blocked)
    assert "--variants" in str(blocked)

    # And the same batch minus the culprit grades normally — i.e. the refusal
    # is about that variant, not about the population.
    graded = grade_variants([fitted], eps)
    assert graded["n_matched"] > 0


def test_disjoint_coverage_is_refused_with_its_own_reason_code():
    """Every variant covers something, but they share no episode.

    Distinct from the barren case and worth its own reason, because the remedy
    differs: here no single variant is at fault, so dropping one does not help
    and the train window is what has to change. Reached by giving two variants
    cells for disjoint route sets — a configuration the aligned fall-throughs
    make unlikely in practice, which is exactly why it is worth pinning rather
    than trusting.
    """
    fitted, eps = _fitted_and_episodes()
    routes = sorted({e.route for e in eps})
    assert len(routes) >= 2
    left, right = routes[0], routes[1]

    only_left = FittedVariant(name="shipped", cells={left: fitted.cells[left]})
    only_right = FittedVariant(name="km_pooled", cells={right: fitted.cells[right]})

    with pytest.raises(GradeBlocked) as caught:
        grade_variants([only_left, only_right], eps)

    blocked = caught.value
    assert blocked.reason == "no_common_episode"
    assert blocked.culprits == ()
    # Both covered something — that is what makes this not the barren case.
    assert blocked.coverage["shipped"] > 0
    assert blocked.coverage["km_pooled"] > 0
    assert "no episode in common" in str(blocked)


def test_causal_baseline_is_none_when_km_pooled_is_absent():
    """Grading without the causal climatology is allowed but must be VISIBLE:
    causal_skill None on every row and causal_baseline None, never a silent
    fallback to the hindsight oracle baseline."""
    fitted, eps = _fitted_and_episodes()
    graded = grade_variants([fitted], eps)
    assert graded["causal_baseline"] is None
    assert all(row["causal_skill"] is None for row in graded["variants"])


def test_a_thin_window_with_no_episodes_is_refused_through_grade():
    """Propagation through the orchestrator, on the realistic case: a window
    whose eval half holds no episode at all. Every variant covers 0, so this
    must refuse rather than return a table of NaN rows."""
    inputs = _inputs()
    empty_eval = GradeInputs(
        ticks=[(t, c) for t, c in inputs.ticks if t < inputs.train_end_epoch],
        train_end_epoch=inputs.train_end_epoch,
        eval_start_epoch=inputs.eval_start_epoch,
        eval_end_epoch=inputs.eval_end_epoch,
    )
    with pytest.raises(GradeBlocked) as caught:
        grade(empty_eval)
    assert caught.value.reason == "variant_covers_nothing"
    assert set(caught.value.culprits) == set(VARIANTS)


def test_cli_exits_nonzero_when_the_grade_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The exit code is the contract for callers. A refusal that exited 0 would
    read to a script as 'graded fine', which is the whole failure this guards."""
    import training.movement_dwell_grade as mod

    inputs = _inputs()
    empty_eval = GradeInputs(
        ticks=[(t, c) for t, c in inputs.ticks if t < inputs.train_end_epoch],
        train_end_epoch=inputs.train_end_epoch,
        eval_start_epoch=inputs.eval_start_epoch,
        eval_end_epoch=inputs.eval_end_epoch,
    )

    def fake_config() -> SimpleNamespace:
        return SimpleNamespace(bucket="test-bucket")

    def fake_client(_cfg: object) -> object:
        return object()

    def fake_inputs(*_args: object, **_kwargs: object) -> GradeInputs:
        return empty_eval

    monkeypatch.setattr(mod, "load_config", fake_config)
    monkeypatch.setattr(mod, "make_client", fake_client)
    monkeypatch.setattr(mod, "load_inputs", fake_inputs)

    rc = mod.main(
        [
            "--train-start",
            "2026-06-01",
            "--train-end",
            "2026-06-10",
            "--eval-end",
            "2026-06-12",
        ]
    )
    assert rc == 1
    captured = capsys.readouterr()
    assert "cannot grade" in captured.err
    assert "variant_covers_nothing" in captured.out
