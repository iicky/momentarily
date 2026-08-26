"""Fit the grade-driven recovery recalibration factor (`dwell.recalibrate_cell`'s
gamma) from held-out incident episodes.

The published recovery quantiles run systematically optimistic: graded against
observed episode clearances the recover-by-H forecast sits above the realised
rate, worst in the 60-120min band. `dwell.recalibrate_cell` corrects this by
warping every cell's predictive dwell CDF as F' = F**gamma (gamma >= 1). This
module estimates gamma from grades.

gamma is fit to the recover-by CALIBRATION rather than the raw PIT mean: for each
horizon h in a grid, the recalibrated mean forecast mean_i F_i(h)**gamma is driven
onto the observed clearance fraction O(h). Aggregating over episodes per horizon
makes the fit robust to the one-tick curve-floor quantisation that pins short
episodes at PIT 0 regardless of fit quality (see dwell.dwell_cdf) — an artifact of
the atom-free quantile representation, not a tail-shape error gamma should chase.
So the grid is anchored in the 30-120min band, away from the point mass.

Caller owns the temporal split (fit on the earlier window, grade on the later)
and the episode population (full-label boundaries, severe-peak criterion).
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from momentarily.mapping import (
    CANONICAL_SEVERITY_FLOOR,
    category_for_label,
    coarse_status,
    severity_tier,
)
from training.dwell import DwellQuantiles, recalibrate_cell
from training.episodes import Episode, extract_episodes
from training.eval import TICK_SECONDS
from training.recovery_dist import RecoveryDistSample, predicted_recovery_curve
from training.scorecard import DwellLookup, cause_dwell_lookup

if TYPE_CHECKING:
    from datetime import date

    from mypy_boto3_s3 import S3Client

    from training.r2_client import R2Config


# Any-disruption label used for episode BOUNDARIES: a severe incident that dips
# to a minor delay and back is one episode, not two with a fake recovery between
# them. Planned/info alerts are tier 0 and stay normal, so they never open one.
FULL_LABEL_FLOOR = 1

# Horizons (minutes) the recover-by calibration is fit over. Anchored in the
# band where the optimism concentrates; the sub-30 region is dominated by the
# one-tick point mass and its curve-floor quantisation, not by tail shape.
FIT_HORIZONS_MIN: tuple[int, ...] = (30, 45, 60, 90, 120)

# gamma is a correction for optimism, so it is clamped at 1.0 below (a window
# that grades pessimistic asks for no change rather than for extra optimism) and
# capped above so a thin or pathological window cannot ship a violent warp.
GAMMA_MIN = 1.0
GAMMA_MAX = 3.0


def _severe_peak_and_selector(
    episode: Episode, types: dict[tuple[str, int], tuple[str, ...]]
) -> tuple[int, str, str]:
    """Peak severity tier plus the (peak_state, cause) selector derived from the
    episode's SEVERE ticks only. Voting the selector over the full-label run
    would let a delays-dominated minor tail relabel a suspension incident and
    pull the wrong dwell cell — the diluted-selector failure this repo measured.
    """
    votes: dict[str, int] = {}
    peak_by_cause: dict[str, int] = {}
    peak = 0
    suspended = False
    tick = episode.onset
    while tick < episode.recovery:
        for at in types.get((episode.route, tick), ()):
            tier = severity_tier(at)
            peak = max(peak, tier)
            if tier == 3:
                suspended = True
            if tier >= CANONICAL_SEVERITY_FLOOR:
                cause = category_for_label(coarse_status(at))
                votes[cause] = votes.get(cause, 0) + 1
                peak_by_cause[cause] = max(peak_by_cause.get(cause, 0), tier)
        tick += TICK_SECONDS
    cause = (
        max(votes, key=lambda c: (votes[c], peak_by_cause[c], c)) if votes else "other"
    )
    return peak, ("suspended" if suspended else "disrupted"), cause


def severe_recovery_episodes(
    full_truth: dict[tuple[str, int], str],
    types: dict[tuple[str, int], tuple[str, ...]],
    *,
    window_start: int,
    window_end: int,
    onset_from: int,
    onset_to: int,
) -> list[Episode]:
    """The gradeable severe-incident population, built the way the hard rules
    require: boundaries cut from the FULL alert label (`full_truth`, floor 1) so a
    severe incident is one episode across its minor dips; the severe-peak
    criterion (peak tier >= CANONICAL_SEVERITY_FLOOR) applied AFTER segmentation;
    and the (peak_state, cause) cell selector overwritten with the severe-derived
    one via `replace`, so downstream `episode_samples` reads the right cell.

    `window_start/window_end` bound the label walk (load a recovery tail past the
    onset window so late incidents resolve rather than censor); `onset_from/
    onset_to` bound which onsets are kept, so the graded set is exactly the
    incidents that began in the intended window.
    """
    eps = extract_episodes(
        full_truth, types, window_start=window_start, window_end=window_end
    )
    out: list[Episode] = []
    for e in eps:
        if not (onset_from <= e.onset < onset_to):
            continue
        if e.left_censored or e.right_censored:
            continue
        peak, peak_state, cause = _severe_peak_and_selector(e, types)
        if peak < CANONICAL_SEVERITY_FLOOR:
            continue
        out.append(replace(e, peak_state=peak_state, cause=cause))
    return out


def episode_samples(
    episodes: list[Episode], lookup: DwellLookup
) -> list[RecoveryDistSample]:
    """Build recovery-grade samples for uncensored episodes with a dwell curve,
    identical in construction to scorecard.episode_recovery (elapsed-0 forecast
    vs realised duration) so the fit sees exactly what the grade will."""
    samples: list[RecoveryDistSample] = []
    for e in episodes:
        if e.left_censored or e.right_censored:
            continue
        cell = lookup(e.route, e.peak_state, e.cause)
        if cell is None or len(cell[0]) < 2:
            continue
        curve_sec, tail_ll, atom = cell
        pred_left = (
            0.0 if atom is not None and abs(e.duration_sec - atom[1]) < 1.0 else None
        )
        samples.append(
            RecoveryDistSample(
                pred_curve=predicted_recovery_curve(0.0, curve_sec, tail_ll, atom),
                actual_min=e.duration_sec / 60.0,
                regime_key=f"{e.route}:{e.onset}",
                pred_left=pred_left,
            )
        )
    return samples


def _calibration_error(
    samples: list[RecoveryDistSample],
    horizons: tuple[int, ...],
    gamma: float,
) -> float:
    """Sum over horizons of (recalibrated mean forecast - observed clearance)^2."""
    n = len(samples)
    err = 0.0
    for h in horizons:
        observed = sum(1 for s in samples if s.actual_min <= h) / n
        tmax = len(samples[0].pred_curve) - 1
        idx = min(h, tmax)
        predicted = sum(s.pred_curve[idx] ** gamma for s in samples) / n
        err += (predicted - observed) ** 2
    return err


def fit_recovery_gamma(
    samples: list[RecoveryDistSample],
    *,
    horizons: tuple[int, ...] = FIT_HORIZONS_MIN,
    gamma_min: float = GAMMA_MIN,
    gamma_max: float = GAMMA_MAX,
) -> float:
    """Fit the recovery-recalibration gamma minimising the recover-by calibration
    error over `horizons`. Returns gamma in [gamma_min, gamma_max]; 1.0 (identity)
    when there is nothing to grade.

    A coarse grid then a local refine over a smooth 1-D objective — no dependency
    on a minimiser, and deterministic so the published factor is reproducible.
    """
    if not samples:
        return 1.0

    def best_on(lo: float, hi: float, steps: int) -> tuple[float, float]:
        step = (hi - lo) / steps
        best_g, best_e = lo, _calibration_error(samples, horizons, lo)
        for i in range(1, steps + 1):
            g = lo + i * step
            e = _calibration_error(samples, horizons, g)
            if e < best_e:
                best_g, best_e = g, e
        return best_g, best_e

    coarse, _ = best_on(gamma_min, gamma_max, 100)
    span = (gamma_max - gamma_min) / 100
    fine, _ = best_on(max(gamma_min, coarse - span), min(gamma_max, coarse + span), 40)
    return round(fine, 4)


# --- Train-time orchestration ------------------------------------------------
#
# train_em publishes on a short trailing window (e.g. 14d), too short to both fit
# dwell cells and hold out a calibration tail. So gamma is fit on a SHADOW window
# of the SAME length immediately before the publish window, graded on the publish
# window's own incidents — which the shadow cells never saw. This keeps the
# calibrator matched to the estimator: same-length dwell fit, same one-window-
# ahead relationship the published cells will face once deployed. gamma is then
# applied to the published cells.

# Extra days of truth loaded past the calibration window so a late-onset incident
# resolves rather than right-censoring; mirrors backtest.RECOVERY_TAIL_DAYS (kept
# local to avoid importing backtest, which imports train_em — an import cycle).
_RECOVERY_TAIL_DAYS = 2
# Below this many gradeable calibration incidents the window is too thin to trust
# a warp; publish the identity (gamma 1.0) rather than a noisy correction.
MIN_CALIB_EPISODES = 50


def _full_label_state(alert_types: tuple[str, ...]) -> str:
    """Full-label truth state for episode boundaries — the floor-1 twin of
    review.derive_graded_mta_state, inlined so the publish path need not import
    review.py (which pulls in matplotlib). Suspended > any-disruption > normal."""
    tiers = [severity_tier(at) for at in alert_types]
    if any(t == 3 for t in tiers):
        return "suspended"
    if any(t >= FULL_LABEL_FLOOR for t in tiers):
        return "disrupted"
    return "normal"


def fit_published_recovery_gamma(
    cfg: R2Config, client: S3Client, start_date: date, end_date: date
) -> tuple[float, dict[str, Any]]:
    """Fit the recovery-recalibration gamma for a retrain, on held-out episodes.

    `start_date`/`end_date` are the publish window. gamma is fit on a SHADOW
    window of the same length immediately before it: dwell cells from the shadow
    window, calibration incidents onsetting in the publish window itself (which
    the shadow cells never saw), boundaries from the full label, severe-peak
    criterion. Returns (gamma, diagnostics). gamma is 1.0 (identity) when the
    shadow history is unavailable or the calibration window is too thin to grade.
    """
    from datetime import UTC, datetime, timedelta

    from training.dwell import compute_dwell_quantiles, compute_dwell_quantiles_by_cause
    from training.eval import TICK_SECONDS, load_predictions, load_transitions
    from training.load_r2 import (
        build_tick_observations,
        fetch_alert_versions,
        presence_mask_from_predictions,
    )
    from training.survival import loglogistic_tail

    def _midnight(d: date) -> int:
        return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp())

    window_len = (end_date - start_date).days + 1
    shadow_end = start_date - timedelta(days=1)
    shadow_start = start_date - timedelta(days=window_len)

    # Shadow cells: a same-length dwell fit on the window_len days before the
    # publish window, so the calibrator matches the published estimator.
    shadow_trans = load_transitions(client, cfg.bucket, shadow_start, shadow_end)
    if not shadow_trans:
        return 1.0, {"gamma": 1.0, "reason": "no shadow-window history to fit on"}
    shadow_end_epoch = _midnight(start_date)  # censor open regimes at publish start
    by_cause = compute_dwell_quantiles_by_cause(shadow_trans, tail_fn=loglogistic_tail)
    by_state = compute_dwell_quantiles(
        shadow_trans, window_end=shadow_end_epoch, tail_fn=loglogistic_tail
    )
    pooled = compute_dwell_quantiles(
        [replace(t, route="*") for t in shadow_trans],
        window_end=shadow_end_epoch,
        tail_fn=loglogistic_tail,
    ).get("*", {})
    lookup = cause_dwell_lookup(by_cause, by_state, pooled)

    # Calibration incidents: onset in the publish window (held out from the shadow
    # cells), full-label boundaries + severe-peak criterion.
    truth_end = end_date + timedelta(days=_RECOVERY_TAIL_DAYS)
    mask = presence_mask_from_predictions(
        load_predictions(client, cfg.bucket, start_date, truth_end)
    )
    obs = build_tick_observations(
        fetch_alert_versions(
            config=cfg, start_date=start_date, end_date=truth_end, client=client
        ),
        active_mask=mask,
    )
    full_truth = {
        (o.route_id, o.tick): _full_label_state(o.disruptive_types) for o in obs
    }
    types = {(o.route_id, o.tick): o.disruptive_types for o in obs}
    wee = _midnight(truth_end + timedelta(days=1)) - TICK_SECONDS
    episodes = severe_recovery_episodes(
        full_truth,
        types,
        window_start=_midnight(start_date),
        window_end=wee,
        onset_from=_midnight(start_date),
        onset_to=_midnight(end_date + timedelta(days=1)),
    )
    samples = episode_samples(episodes, lookup)
    if len(samples) < MIN_CALIB_EPISODES:
        return 1.0, {
            "gamma": 1.0,
            "reason": "too few calibration episodes",
            "n_calib_episodes": len(samples),
        }
    gamma = fit_recovery_gamma(samples)
    return gamma, {
        "gamma": gamma,
        "n_calib_episodes": len(samples),
        "shadow_cells_from": shadow_start.isoformat(),
        "shadow_cells_to": shadow_end.isoformat(),
        "calib_onset_from": start_date.isoformat(),
        "calib_onset_to": end_date.isoformat(),
    }


# States the recalibration touches: the disruption-recovery curves gamma was fit
# on. `normal`-state dwell (how long a route STAYS normal) is a different forecast
# and must not be warped by a recovery-optimism factor.
RECALIBRATED_STATES = ("disrupted", "suspended")


def recalibrate_dwell_cells(
    block: dict[str, dict[str, DwellQuantiles]],
    gamma: float,
    *,
    states: tuple[str, ...] = RECALIBRATED_STATES,
) -> dict[str, dict[str, DwellQuantiles]]:
    """Recalibrate a {route: {state: cell}} block (the dwell_quantiles shape).
    Only the disruption `states` are warped; every other state (notably `normal`)
    passes through unchanged. gamma 1.0 is a whole-block no-op."""
    if gamma == 1.0:
        return block
    return {
        route: {
            state: (recalibrate_cell(cell, gamma) if state in states else cell)
            for state, cell in by_state.items()
        }
        for route, by_state in block.items()
    }


def recalibrate_dwell_cells_by_key(
    block: dict[str, dict[str, dict[str, DwellQuantiles]]],
    gamma: float,
    *,
    states: tuple[str, ...] = RECALIBRATED_STATES,
) -> dict[str, dict[str, dict[str, DwellQuantiles]]]:
    """Recalibrate a {route: {state: {key: cell}}} block (the by-alert / by-cause
    shape). Same state gating as recalibrate_dwell_cells; gamma 1.0 is a no-op."""
    if gamma == 1.0:
        return block
    return {
        route: {
            state: (
                {key: recalibrate_cell(cell, gamma) for key, cell in by_key.items()}
                if state in states
                else by_key
            )
            for state, by_key in by_state.items()
        }
        for route, by_state in block.items()
    }
