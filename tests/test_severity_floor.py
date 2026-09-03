"""The severity floor on the HMM training observation.

The latent 'disrupted' regime was measured at 94.3% ordinary tier-1 Delays while
the grading truth counts severity >= 2 only, so the model learned a dwell from a
population the yardstick never asks about. These tests lock the transform that
closes that gap, and the two properties that make it safe to land: the legacy
arm is byte-identical to the pre-severity build, and a floored fit cannot be
published while the Worker still counts every non-planned alert.

Synthetic records only — no filesystem, no R2.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from momentarily.hmm import EmissionParams, HMMParams, Observation
from momentarily.mapping import (
    CANONICAL_SEVERITY_FLOOR,
    LEGACY_SEVERITY_FLOOR,
    severity_tier,
)
from training.load import alert_observation, build_observations, fill_quiet_ticks
from training.load_r2 import build_tick_observations
from training.train_em import (
    MAX_SELF_LOOP,
    geometric_dwell_fit,
    implied_median_dwell_minutes,
    main,
    self_loop_diagonal,
    self_loop_excess,
    severe_episode_ticks,
    train,
)

T0 = 1_700_000_100  # tick-aligned
TICK = 300


def _approx(expected: object, abs: float | None = None) -> object:
    """Typed wrapper around ``pytest.approx``.

    pytest's ``approx`` leaks ``Unknown`` through its ``ApproxBase`` return type
    under strict mode, so we pin the boundary to ``object`` once here.
    """
    return pytest.approx(expected, abs=abs)  # pyright: ignore[reportUnknownMemberType]


def _quiet(n: int) -> list[Observation]:
    """A run of ticks with nothing active — the shape that drives EM's self-loop
    to 1.0 and makes the clamp fire."""
    return [Observation(alert_count=0, severity_sum=0, has_suspended_alert=False)] * n


def _rec(
    alert_type: str,
    *,
    route: str = "1",
    alert_id: str = "lmm:alert:1",
    observed_at: int = T0,
) -> dict[str, Any]:
    """One collector poll record naming a single route."""
    return {
        "observed_at": observed_at,
        "alert": {
            "id": alert_id,
            "alert": {
                "informed_entity": [
                    {
                        "route_id": route,
                        "transit_realtime.mercury_entity_selector": {
                            "sort_order": f"MTASBWY:{route}:20"
                        },
                    }
                ],
                "transit_realtime.mercury_alert": {"alert_type": alert_type},
            },
        },
    }


def _version(
    alert_type: str,
    *,
    route: str = "1",
    alert_id: str = "lmm:alert:1",
    start: int = T0,
    end: int = T0,
) -> dict[str, Any]:
    """One archived alert version with an explicit active_period."""
    body = _rec(alert_type, route=route, alert_id=alert_id, observed_at=start)
    body["alert"]["alert"]["active_period"] = [{"start": start, "end": end}]
    return body


# --- the tier mapping the floor rests on -------------------------------------


def test_tier_assumptions_the_floor_depends_on() -> None:
    """The floor is only meaningful if these tiers hold. Pinned here so a change
    to the mapping fails against the transform that consumes it, rather than
    silently redefining which alerts count as disruption evidence."""
    assert severity_tier("Delays") == 1
    assert severity_tier("Trains Rerouted") == 1
    assert severity_tier("Severe Delays") == 2
    assert severity_tier("Suspended") == 3
    assert severity_tier("Station Notice") == 0
    assert LEGACY_SEVERITY_FLOOR == 1
    assert CANONICAL_SEVERITY_FLOOR == 2


# --- the legacy arm applies no filter at all --------------------------------


def test_legacy_floor_counts_every_alert_unfiltered() -> None:
    """The legacy branch must not filter anything: it is the build the Worker
    serves and the baseline arm of every pre/post comparison. Tier-0 alerts in
    particular were always counted here."""
    counted = [(20, "Station Notice"), (20, "Delays"), (20, "Severe Delays")]
    obs = alert_observation(counted, T0, severity_floor=LEGACY_SEVERITY_FLOOR)
    assert obs.alert_count == 3
    assert obs.severity_sum == 60
    assert obs.has_delays
    assert not obs.has_minor_alert  # nothing is sub-floor in the legacy arm
    assert obs.max_severity_tier == 2
    assert obs.severity_floor == LEGACY_SEVERITY_FLOOR


def test_legacy_floor_is_the_default_everywhere() -> None:
    """An unparameterised call anywhere in the loaders yields the serving build,
    so no existing caller silently acquires a severity floor."""
    obs = alert_observation([(20, "Delays")], T0)
    assert obs.severity_floor == LEGACY_SEVERITY_FLOOR
    assert obs.alert_count == 1
    tick_obs = build_observations([_rec("Delays")])
    assert [o.observation.severity_floor for o in tick_obs] == [LEGACY_SEVERITY_FLOOR]


# --- floor 2 scores exactly the population the grader treats as disrupted ---


def test_canonical_floor_drops_tier_one_from_scored_channels() -> None:
    """Ordinary Delays stops being disruption evidence: it leaves alert_count,
    severity_sum and has_delays, and survives only as has_minor_alert. This is
    the whole intervention."""
    obs = alert_observation(
        [(20, "Delays")], T0, severity_floor=CANONICAL_SEVERITY_FLOOR
    )
    assert obs.alert_count == 0
    assert obs.severity_sum == 0
    assert not obs.has_delays
    assert obs.has_minor_alert
    # The tier itself is read off the UNFILTERED list, so the severe-episode
    # yardstick is identical under both arms.
    assert obs.max_severity_tier == 1
    assert obs.severity_floor == CANONICAL_SEVERITY_FLOOR


def test_canonical_floor_also_drops_tier_zero() -> None:
    """A Station Notice is not severity >= 2, so under floor 2 it must not
    inflate alert_count either — otherwise the scored population is not the one
    truth_version 2 grades and the state definition no longer matches the
    yardstick. It is not 'minor': has_minor_alert tracks discarded DISRUPTIVE
    mass, and tier 0 was never disruption."""
    obs = alert_observation(
        [(20, "Station Notice")], T0, severity_floor=CANONICAL_SEVERITY_FLOOR
    )
    assert obs.alert_count == 0
    assert obs.severity_sum == 0
    assert not obs.has_minor_alert
    assert obs.max_severity_tier == 0


def test_canonical_floor_keeps_severe_delays_and_suspensions() -> None:
    severe = alert_observation(
        [(20, "Severe Delays")], T0, severity_floor=CANONICAL_SEVERITY_FLOOR
    )
    assert severe.alert_count == 1
    assert severe.has_delays
    assert not severe.has_minor_alert
    assert severe.max_severity_tier == 2

    suspended = alert_observation(
        [(20, "Suspended")], T0, severity_floor=CANONICAL_SEVERITY_FLOOR
    )
    assert suspended.alert_count == 1
    assert suspended.has_suspended_alert
    assert suspended.max_severity_tier == 3


def test_severe_alert_masks_a_concurrent_minor_one() -> None:
    """A severe alert and an ordinary one on the same tick: the tick counts once
    for the severe alert, and has_minor_alert still records that tier-1 mass was
    discarded so the diagnostic can see how much the floor removed."""
    obs = alert_observation(
        [(20, "Delays"), (30, "Severe Delays")],
        T0,
        severity_floor=CANONICAL_SEVERITY_FLOOR,
    )
    assert obs.alert_count == 1
    assert obs.severity_sum == 30
    assert obs.has_delays
    assert obs.has_minor_alert
    assert obs.max_severity_tier == 2


def test_service_changes_are_sub_floor_at_the_canonical_floor() -> None:
    """Every Service Change type maps to tier 1, so floor 2 silences the
    has_service_change channel entirely. That is a real consequence of aligning
    with truth_version 2, not an oversight — pinned so it is a decision on
    record rather than a surprise in a refit."""
    obs = alert_observation(
        [(20, "Trains Rerouted")], T0, severity_floor=CANONICAL_SEVERITY_FLOOR
    )
    assert not obs.has_service_change
    assert obs.alert_count == 0
    assert obs.has_minor_alert


def test_planned_work_stays_out_under_every_floor() -> None:
    """The floor must not resurrect planned work, which is excluded by alert id
    before the floor is ever consulted."""
    for floor in (LEGACY_SEVERITY_FLOOR, CANONICAL_SEVERITY_FLOOR):
        obs = build_observations(
            [_rec("Planned - Part Suspended", alert_id="lmm:planned_work:1")],
            severity_floor=floor,
        )
        assert obs
        for o in obs:
            assert o.observation.alert_count == 0
            assert not o.observation.has_suspended_alert
            assert not o.observation.has_planned
            assert not o.observation.has_minor_alert


def test_has_planned_reads_the_unfiltered_list() -> None:
    """A 'Planned -' TYPE arriving under a non-planned id is tier 0, so the
    floor drops it from the scored channels — but has_planned reports whether
    planned work is up, not how severe it is, and must still fire."""
    obs = alert_observation(
        [(20, "Planned - Stops Skipped")],
        T0,
        severity_floor=CANONICAL_SEVERITY_FLOOR,
    )
    assert obs.has_planned
    assert obs.alert_count == 0


# --- both builders, one transform -------------------------------------------


def test_local_builder_populates_disruptive_types() -> None:
    """The local loader left disruptive_types empty, so severity grading and the
    HMM path read different sources depending on which loader ran."""
    obs = build_observations([_rec("Severe Delays")])
    assert obs
    assert [o.disruptive_types for o in obs] == [("Severe Delays",)]


def test_disruptive_types_is_floor_independent() -> None:
    """Raising the floor changes what the HMM trains on, never the truth built
    from the same call."""
    for floor in (LEGACY_SEVERITY_FLOOR, CANONICAL_SEVERITY_FLOOR):
        obs = build_observations([_rec("Delays")], severity_floor=floor)
        assert [o.disruptive_types for o in obs] == [("Delays",)]


@pytest.mark.parametrize(
    "alert_type", ["Delays", "Severe Delays", "Suspended", "Station Notice"]
)
@pytest.mark.parametrize("floor", [LEGACY_SEVERITY_FLOOR, CANONICAL_SEVERITY_FLOOR])
def test_local_and_archive_builders_agree(alert_type: str, floor: int) -> None:
    """The two loaders were duplicate copies of this logic. They now share one
    builder, and this is what stops them drifting again."""
    local = build_observations([_rec(alert_type)], severity_floor=floor)
    archive = build_tick_observations([_version(alert_type)], severity_floor=floor)
    assert local
    assert archive
    assert local[0].observation == archive[0].observation
    assert local[0].disruptive_types == archive[0].disruptive_types


def test_quiet_ticks_carry_the_series_floor() -> None:
    """A quiet tick has no alerts and so is identical under every floor, but it
    must still report the floor its series was built under — otherwise a series
    reports a mix and no diagnostic can tell which arm produced it."""
    obs = build_observations(
        [_rec("Severe Delays", observed_at=T0)],
        severity_floor=CANONICAL_SEVERITY_FLOOR,
    )
    filled = fill_quiet_ticks(
        obs,
        "1",
        start_tick=T0,
        end_tick=T0 + 2 * TICK,
        severity_floor=CANONICAL_SEVERITY_FLOOR,
    )
    assert len(filled) == 3
    assert all(o.observation.severity_floor == CANONICAL_SEVERITY_FLOOR for o in filled)
    assert filled[1].observation.alert_count == 0
    assert filled[1].observation.max_severity_tier == 0


# --- severe-episode extraction ----------------------------------------------


def _series(tiers: list[int]) -> list[Observation]:
    return [
        Observation(
            alert_count=1 if t else 0,
            severity_sum=0,
            has_suspended_alert=False,
            max_severity_tier=t,
        )
        for t in tiers
    ]


def test_severe_episodes_are_maximal_runs_at_or_above_the_tier() -> None:
    durations, censored = severe_episode_ticks(_series([0, 2, 2, 2, 0, 1, 0, 3, 0]))
    assert durations == [3, 1]
    assert censored == 0


def test_severe_episodes_exclude_both_censored_ends() -> None:
    """A run touching either end of the window has an unobserved duration. Both
    are excluded rather than truncated — counting a truncated run would bias
    every dwell statistic downward, the direction that would flatter the fit."""
    durations, censored = severe_episode_ticks(_series([2, 2, 0, 2, 0, 2, 2]))
    assert durations == [1]
    assert censored == 2


def test_severe_episodes_ignore_sub_tier_ticks() -> None:
    """The yardstick is the severe population regardless of which arm built the
    series, so tier-1 runs must never open an episode."""
    durations, censored = severe_episode_ticks(_series([0, 1, 1, 1, 0]))
    assert durations == []
    assert censored == 0


def test_severe_episodes_survive_a_floored_build() -> None:
    """End to end: a floor-2 series scores its tier-1 ticks as quiet in the
    likelihood channels, yet the episode extractor still sees the severe run —
    that is what makes the two arms comparable on one yardstick."""
    records = [
        _rec("Delays", observed_at=T0),
        _rec("Severe Delays", alert_id="lmm:alert:2", observed_at=T0 + TICK),
    ]
    obs = build_observations(records, severity_floor=CANONICAL_SEVERITY_FLOOR)
    filled = fill_quiet_ticks(
        obs,
        "1",
        start_tick=T0,
        end_tick=T0 + 2 * TICK,
        severity_floor=CANONICAL_SEVERITY_FLOOR,
    )
    series = [o.observation for o in filled]
    assert [o.alert_count for o in series] == [0, 1, 0]
    durations, censored = severe_episode_ticks(series)
    assert durations == [1]
    assert censored == 0


# --- geometric dwell scoring ------------------------------------------------


def test_implied_median_dwell_inverts_the_self_loop() -> None:
    """The cap on the disrupted row is what makes clamp pressure interesting: it
    permits ~48 minutes, and EM wanting ~97 is the measured tension. Pin the
    arithmetic that converts between the two."""
    assert implied_median_dwell_minutes(0.93) == _approx(47.7, abs=0.1)
    assert implied_median_dwell_minutes(0.965) == _approx(97.3, abs=0.1)
    assert implied_median_dwell_minutes(1.0) == math.inf


def test_geometric_fit_prefers_the_self_loop_that_matches_the_durations() -> None:
    """The self-loop IS a geometric dwell model, so the honest score is how well
    it describes the observed durations. A self-loop matching the sample's mean
    must beat one far from it on both statistics, or the diagnostic cannot
    distinguish the arms."""
    durations = [2, 3, 3, 4, 5, 6, 8, 10]
    mean = sum(durations) / len(durations)
    matched = geometric_dwell_fit(durations, 1.0 - 1.0 / mean, 0)
    too_sticky = geometric_dwell_fit(durations, 0.99, 0)
    assert matched.mean_loglik > too_sticky.mean_loglik
    assert matched.ks < too_sticky.ks
    assert matched.n == len(durations)
    assert matched.empirical_median_ticks == _approx(4.5)


def test_geometric_fit_reports_no_episodes_without_inventing_a_score() -> None:
    """A route with no severe episode must not read as a good fit."""
    fit = geometric_dwell_fit([], 0.93, 3)
    assert fit.n == 0
    assert fit.n_censored == 3
    assert math.isnan(fit.mean_loglik)
    assert math.isnan(fit.ks)


def test_ks_is_near_zero_for_a_matching_distribution() -> None:
    """A sample drawn from the exact geometric the fit scores should sit near
    zero KS; otherwise the metric is not measuring agreement. Deterministic
    quantiles rather than a random sample so it cannot flake."""
    a = 0.8
    durations = [
        max(1, math.ceil(math.log(1.0 - i / 200.0) / math.log(a)))
        for i in range(1, 200)
    ]
    assert geometric_dwell_fit(durations, a, 0).ks < 0.05


# --- clamp-pressure instrumentation -----------------------------------------


def _params(diag: tuple[float, float, float]) -> HMMParams:
    rows: list[tuple[float, float, float]] = []
    for s in range(3):
        row = [(1.0 - diag[s]) / 2.0] * 3
        row[s] = diag[s]
        rows.append((row[0], row[1], row[2]))
    return HMMParams(
        transition=(rows[0], rows[1], rows[2]),
        initial=(1.0, 0.0, 0.0),
        emissions=EmissionParams(
            poisson_lambda=(0.1, 1.0, 2.0),
            gamma_alpha=(1.0, 1.0, 1.0),
            gamma_beta=(1.0, 1.0, 1.0),
            bernoulli_p=(0.01, 0.05, 0.9),
        ),
    )


def test_self_loop_diagonal_reads_the_three_self_transitions() -> None:
    assert self_loop_diagonal(_params((0.99, 0.5, 0.2))) == _approx((0.99, 0.5, 0.2))


def test_self_loop_excess_never_goes_negative() -> None:
    """A state under its cap must contribute zero, not a negative number that
    would cancel out a state over its cap when these are averaged across
    routes."""
    excess = self_loop_excess(_params((0.99, 0.5, 0.95)), (0.975, 0.93, 0.93))
    assert excess[0] == _approx(0.015)
    assert excess[1] == 0.0
    assert excess[2] == _approx(0.02)


def test_train_records_the_diagonal_em_wanted_before_the_clamp() -> None:
    """The pre-clamp diagonal is the whole primary gate, and it had been
    hand-patched in twice because nothing recorded it. A long quiet series drives
    the normal self-loop past its ceiling, so the sink must show the excess while
    the returned params show the clamped value."""
    series = {"R1": _quiet(600)}
    pre_clamp: dict[str | None, tuple[float, float, float]] = {}
    global_prior, per_route = train(
        series, min_ticks=100, prior_strength=10.0, pre_clamp_diagonals=pre_clamp
    )
    assert None in pre_clamp  # the pooled global-prior fit
    assert "R1" in pre_clamp
    assert pre_clamp["R1"][0] > MAX_SELF_LOOP[0]
    assert per_route["R1"].transition[0][0] == _approx(MAX_SELF_LOOP[0])
    assert global_prior.transition[0][0] == _approx(MAX_SELF_LOOP[0])


def test_train_omits_routes_that_inherited_the_prior() -> None:
    """A route under min_ticks has no fit of its own, so recording a diagonal for
    it would report the prior's clamp pressure as if it were the route's."""
    pre_clamp: dict[str | None, tuple[float, float, float]] = {}
    train(
        {"R1": _quiet(50)},
        min_ticks=1000,
        prior_strength=10.0,
        pre_clamp_diagonals=pre_clamp,
    )
    assert set(pre_clamp) == {None}


def test_train_without_the_sink_is_unaffected() -> None:
    """The sink is an out-parameter precisely so no existing caller changes."""
    series = {"R1": _quiet(600)}
    _, with_sink = train(series, min_ticks=100, prior_strength=10.0)
    _, without = train(
        series, min_ticks=100, prior_strength=10.0, pre_clamp_diagonals={}
    )
    assert with_sink["R1"].transition == without["R1"].transition


# --- the serving-parity guard -----------------------------------------------


def test_a_floored_fit_refuses_to_publish(monkeypatch: pytest.MonkeyPatch) -> None:
    """A floored fit's alert-count distribution is one the Worker never produces,
    so publishing it would ship a train/serve emission mismatch. The refusal must
    land before any R2 credential or fetch is touched."""

    def _boom() -> object:
        raise AssertionError("main() reached R2 config despite the parity guard")

    monkeypatch.setattr("training.train_em.load_config", _boom)
    assert main(["--severity-floor", str(CANONICAL_SEVERITY_FLOOR)]) == 1


def test_a_floored_diagnostic_run_is_allowed_past_the_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The guard blocks publishing, not measuring — otherwise the floored arm
    could never be evaluated at all."""
    reached: list[str] = []

    def _stop() -> object:
        reached.append("config")
        raise RuntimeError("stop here")

    monkeypatch.setattr("training.train_em.load_config", _stop)
    for flag in ("--diagnose-severity", "--dry-run"):
        reached.clear()
        with pytest.raises(RuntimeError):
            main(["--severity-floor", str(CANONICAL_SEVERITY_FLOOR), flag])
        assert reached == ["config"]


def test_the_serving_floor_publishes_without_the_guard_firing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default arm must be entirely unaffected by the guard."""
    reached: list[str] = []

    def _stop() -> object:
        reached.append("config")
        raise RuntimeError("stop here")

    monkeypatch.setattr("training.train_em.load_config", _stop)
    with pytest.raises(RuntimeError):
        main([])
    assert reached == ["config"]


# --- the diagnostic end to end ----------------------------------------------


def _episodic_series(floor: int) -> list[Observation]:
    """A synthetic day whose two alert populations have the shape that was
    measured on the archive: chronic ordinary Delays with a long tail, and rare,
    tight Severe Delays runs. Built through alert_observation at `floor`, so the
    two arms differ exactly as they do in a real run.

    SYNTHETIC — this exercises the diagnostic, it is not evidence about the
    archive.
    """
    plan: list[tuple[str | None, int]] = []
    for _ in range(3):
        plan.append((None, 12))  # quiet hour
        plan.append(("Delays", 60))  # 5h ordinary run — the heavy tail
        plan.append((None, 12))
        plan.append(("Severe Delays", 10))  # 50 min severe run
        plan.append((None, 24))
    out: list[Observation] = []
    tick = T0
    for alert_type, n in plan:
        counted = [] if alert_type is None else [(20, alert_type)]
        for _ in range(n):
            out.append(alert_observation(counted, tick, severity_floor=floor))
            tick += TICK
    return out


def _run_diagnostic(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], floor: int
) -> tuple[int, str]:
    from training.r2_client import R2Config
    from training.train_em import CorpusStats, MovementInputs

    cfg = R2Config(
        account_id="acct",
        access_key_id="key",
        secret_access_key="secret",
        bucket="test-bucket",
    )
    corpus = CorpusStats(start_tick=T0, end_tick=T0 + 30 * 86_400, n_observations=1)
    published: list[str] = []

    def _series_by_route(
        cfg_arg: object, start: object, end: object, **kwargs: object
    ) -> tuple[dict[str, list[Observation]], CorpusStats, dict[str, Any]]:
        # The floor main() computed is the one the corpus is built at — that is
        # the wiring the arms depend on, so assert it rather than assume it.
        assert kwargs["severity_floor"] == floor
        return {"R1": _episodic_series(floor)}, corpus, {}

    monkeypatch.setattr("training.train_em.load_config", lambda: cfg)

    def _fake_make_client(config: object = None) -> object:
        return object()

    monkeypatch.setattr("training.train_em.make_client", _fake_make_client)
    monkeypatch.setattr(
        "training.train_em._static_topology", lambda: (None, None, "observed")
    )

    def _fake_movement_baseline(*a: object, **k: object) -> MovementInputs:
        return MovementInputs({}, 0, {}, set(), {})

    monkeypatch.setattr("training.train_em._movement_baseline", _fake_movement_baseline)
    monkeypatch.setattr("training.train_em.load_series_by_route", _series_by_route)

    def _fake_write_params(*a: object, **k: object) -> str:
        published.append("wrote")
        return "state/params/v1.json"

    monkeypatch.setattr("training.train_em.write_params", _fake_write_params)

    code = main(
        [
            "--start",
            "2026-06-01",
            "--end",
            "2026-06-14",
            "--min-ticks",
            "100",
            "--severity-floor",
            str(floor),
            "--diagnose-severity",
        ]
    )
    assert published == []  # a diagnostic run never writes params
    return code, capsys.readouterr().out


@pytest.mark.parametrize("floor", [LEGACY_SEVERITY_FLOOR, CANONICAL_SEVERITY_FLOOR])
def test_diagnostic_reports_both_gates_under_either_arm(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], floor: int
) -> None:
    """Both primary gates must be printed under both arms — a null result has to
    be as legible as a positive one, or the comparison cannot be recorded."""
    code, out = _run_diagnostic(monkeypatch, capsys, floor)
    assert code == 0
    assert f"severity_floor={floor}" in out
    assert "clamp pressure" in out
    assert "disrupted dwell fit" in out
    assert "POOLED" in out
    # Every state's clamp pressure, and a per-route dwell row with the pre-clamp
    # diagonal beside the one that ships.
    for state in ("normal", "disrupted", "suspended"):
        assert state in out
    assert "a11_pre" in out


def test_diagnostic_scores_the_same_severe_episodes_under_both_arms(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The yardstick must not move between the arms. The severe episode count is
    read off max_severity_tier, which the floor never touches, so both arms score
    the identical episode set — without that the pre/post numbers would be
    comparing different populations and would mean nothing."""
    legacy = severe_episode_ticks(_episodic_series(LEGACY_SEVERITY_FLOOR))
    floored = severe_episode_ticks(_episodic_series(CANONICAL_SEVERITY_FLOOR))
    assert legacy == floored
    assert legacy[0] == [10, 10, 10]  # the three 50-minute severe runs


def test_floored_arm_scores_only_severe_ticks(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The point of the floor: the ordinary-Delays ticks that made up the bulk of
    the old disrupted regime must read quiet in the likelihood channels, while
    the severe ticks still carry an alert."""
    legacy = _episodic_series(LEGACY_SEVERITY_FLOOR)
    floored = _episodic_series(CANONICAL_SEVERITY_FLOOR)
    assert sum(1 for o in legacy if o.alert_count) == 3 * 70
    assert sum(1 for o in floored if o.alert_count) == 3 * 10
    assert sum(1 for o in floored if o.has_minor_alert) == 3 * 60
    assert all(
        o.max_severity_tier == 0
        for o in floored
        if not o.alert_count and not o.has_minor_alert
    )
