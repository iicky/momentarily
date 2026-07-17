"""Tier-1 decision-gate backtest: KM-residual (explicit-duration) p_normal vs geometric.

Question this answers: is the geometric-dwell *forecast* what makes p_normal_in_H
lose to persistence? We replay a held-out window through the existing forward
filter once, then at every tick produce two forecasts of P(normal at t+H):

  geometric : project_forward() — the current production path (repeated matmul
              of the transition matrix; dwell is implicitly geometric).
  km-residual: condition on the most-likely current regime and how long we've
              already been in it, and read the leave-probability off the same
              Kaplan-Meier dwell curves that power recovery_minutes:
                  p_leave = P(dwell <= elapsed + H | dwell > elapsed)
              A long-calm regime has a small p_leave (heavy tail) so the forecast
              stays confident where persistence is right; just after a transition
              p_leave is large so it drops fast. That elapsed-conditioning is the
              whole mechanism, ported straight from the recovery surface's
              conditional-survival approach.

Both arms are scored against ONE fixed target — the alert-derived MTA state at
t+H (build_mta_truth) — so the comparison is apples-to-apples: same filter, same
observations, same truth, only the projection differs. Note this truth is derived
from the same alert feed the model observes (it is NOT an external ground truth
like trip-updates); it is independent of the *projection choice*, which is what a
projection A/B needs. The geometric arm here is the control, not the production
review number (which grades against the model's own published condition).

The temporal train/eval split is honored: params + dwell curves are fit on the
TRAIN window only; replay + scoring happen on a later held-out window, with a
one-day warmup lead so the filter isn't cold-started inside the scored region.

Read-only, offline. No R2 writes, no deploy.

Run with:
    PYTHONPATH=. murk exec -- .venv/bin/python -m training.backtest \
        [--eval-days 3] [--train-days 6] [--eval-end YYYY-MM-DD]

--eval-end anchors the window on a past day (default today) so a historical
incident-rich window can be replayed; truth is loaded one day past it so the
last day's futures resolve to real outcomes, and scored ticks are bounded to
the eval window.

v1 approximations (deliberately crude — this is a go/no-go, not the HSMM itself):
  * single-jump: after the current regime ends we don't re-apply a duration, we
    fall to a one-step jump (to-normal transition prob, or train climatology when
    leaving 'normal'). The real HSMM forward filter (Tier 2) drops these.
  * argmax-conditioned: forecast conditions on the most-likely state + its
    elapsed, mirroring how recovery_minutes already works.
  * cells below the dwell min-samples floor (or a regime that has outlived every
    observed dwell) fall back to the geometric forecast for that tick; the
    fallback rate is reported.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

from momentarily.hmm import (
    STATES,
    FilterState,
    HMMParams,
    forward_step,
    initial_published_state,
    project_forward,
)
from momentarily.mapping import CANONICAL_SEVERITY_FLOOR, TRUTH_VERSION
from training.competing_risks import (
    CIFResult,
    CompetingSample,
    cif_curves,
    conditional_cif,
)
from training.dwell import (
    compute_dwell_quantiles,
    compute_dwell_quantiles_by_cause,
    dwell_cdf,
    dwell_samples_by_cell,
)
from training.episodes import disruptive_types_by_key, extract_episodes
from training.eval import (
    TICK_SECONDS,
    TransitionRecord,
    load_predictions,
    load_transitions,
    snap_tick,
)
from training.load_r2 import load_route_series_r2, presence_mask_from_predictions
from training.r2_client import load_config, make_client
from training.review import derive_mta_state, load_truth_observations, mta_truth
from training.scorecard import cause_dwell_lookup, episode_recovery
from training.survival import (
    ParametricFit,
    fit_loglogistic,
    loglogistic_survival,
    loglogistic_tail,
)
from training.train_em import load_series_by_route, train

HORIZONS_MIN = (30, 60, 120)
# Arms compared, all scored against the same fixed truth:
#   geom  : geometric filter + geometric projection (production path)
#   km    : geometric filter + KM-residual projection (the shipped Tier-1 change)
#   km_ll : KM-residual, but the past-the-curve tail extrapolation swaps the
#           constant-hazard exponential patch for a fitted log-logistic tail
#           (the body stays empirical, only the tail beyond the last observed
#           quantile differs).
#   cif   : disrupted-now only — the competing-risks cumulative incidence of the
#           normal exit (D->Normal vs D->Suspended, elapsed-conditioned) instead
#           of km's flat normal-share split; falls back to km_ll elsewhere.
# A full HSMM filter was tested and shelved — it added nothing to the forecast
# (filtering is emission-dominated).
MODELS = ("geom", "km", "km_ll", "cif")


def _argmax(probs: tuple[float, float, float]) -> int:
    return max(range(len(probs)), key=lambda i: probs[i])


def _p_leave(curve_sec: list[int], elapsed: float, horizon: float) -> float:
    """P(dwell <= elapsed+horizon | dwell > elapsed), with an exponential-tail
    extrapolation past the last observed quantile instead of bailing to None.

    Outliving every observed dwell is the heavy-tail regime the geometric model
    can't see — extrapolating the top-segment hazard keeps the KM arm engaged
    (and, for a long-calm normal stretch, *more* confident) rather than handing
    the tick back to geometric. That handoff is what made v1's KM arm a no-op."""
    pe = dwell_cdf(curve_sec, elapsed)
    if pe < 1.0:
        ph = dwell_cdf(curve_sec, elapsed + horizon)
        return (ph - pe) / (1.0 - pe)
    # elapsed is at/beyond the curve's max: estimate a constant tail hazard from
    # the last segment (top 1/(k-1) of mass lost over its width) and project it.
    k = len(curve_sec)
    seg = curve_sec[-1] - curve_sec[-2] if k >= 2 else 0
    lam = (1.0 / (k - 1)) / seg if seg > 0 else 1.0 / max(1.0, float(curve_sec[-1]))
    return 1.0 - math.exp(-max(lam, 1e-12) * horizon)


def _p_leave_ll(
    curve_sec: list[int], fit: ParametricFit | None, elapsed: float, horizon: float
) -> float:
    """Like _p_leave, but past the last observed quantile the tail is the fitted
    log-logistic conditional survival 1 - S(elapsed+h)/S(elapsed) rather than a
    constant-hazard exponential. The body (elapsed within the curve) is still the
    empirical KM curve — this is a tail splice, not a parametric body fit, which
    gtq.4 showed is a worse in-body match. Falls back to the exponential patch
    when no fit converged."""
    pe = dwell_cdf(curve_sec, elapsed)
    if pe < 1.0:
        ph = dwell_cdf(curve_sec, elapsed + horizon)
        return (ph - pe) / (1.0 - pe)
    if fit is None:
        return _p_leave(curve_sec, elapsed, horizon)
    s_now = loglogistic_survival(elapsed, fit.shape, fit.scale)
    s_fut = loglogistic_survival(elapsed + horizon, fit.shape, fit.scale)
    if s_now <= 0.0:
        return 1.0
    return max(0.0, min(1.0, 1.0 - s_fut / s_now))


def _km_residual_p_normal(
    state: FilterState,
    params: HMMParams,
    route: str,
    dwell_curves: dict[str, dict[str, Any]],
    pooled: dict[str, Any],
    dwell_fits: dict[str, dict[str, ParametricFit]],
    pooled_fits: dict[str, ParametricFit],
    horizon_sec: float,
    clim_normal: float,
    tail: str,
) -> float | None:
    """Explicit-duration P(normal at t+H). None only when no curve exists at all
    (route cell missing AND no pooled fallback) — then caller uses geometric.
    `tail` selects the past-the-curve extrapolation: "exp" (constant hazard) or
    "ll" (fitted log-logistic). The fit is drawn from the same source as the cell
    so curve and tail stay on the same samples."""
    s = _argmax(state.probabilities)
    state_name = STATES[s]
    cell = dwell_curves.get(route, {}).get(state_name)
    fit = dwell_fits.get(route, {}).get(state_name)
    if cell is None:
        cell = pooled.get(state_name)
        fit = pooled_fits.get(state_name)
    if cell is None:
        return None
    elapsed = max(0, state.last_updated_at - state.regime_entered_at)
    if tail == "ll":
        p_leave = _p_leave_ll(cell["curve_sec"], fit, elapsed, horizon_sec)
    else:
        p_leave = _p_leave(cell["curve_sec"], elapsed, horizon_sec)
    if s == 0:  # currently normal: stay normal, or leave then revert to climatology
        return (1.0 - p_leave) + p_leave * clim_normal
    # currently disrupted/suspended: only normal if we've left AND jumped to normal
    self_loop = params.transition[s][s]
    denom = 1.0 - self_loop
    to_normal = params.transition[s][0] / denom if denom > 1e-9 else clim_normal
    return p_leave * min(1.0, max(0.0, to_normal))


def _cif_p_normal(
    state: FilterState,
    route: str,
    cif_by_route: dict[str, CIFResult],
    pooled_cif: CIFResult,
    horizon_sec: float,
) -> float | None:
    """Competing-risks CIF of the normal exit for a currently-disrupted route,
    conditioned on elapsed dwell. None for non-disrupted states (the caller falls
    back to the km_ll arm) — the CIF decomposition only refines disrupted-now
    p_normal, where km's flat normal-share approximation is loosest."""
    if _argmax(state.probabilities) != 1:  # disrupted state index
        return None
    result = cif_by_route.get(route)
    if result is None or "normal" not in result.cif:
        result = pooled_cif
    if "normal" not in result.cif:
        return None  # no normal-exit incidence anywhere — let the caller use km_ll
    elapsed = max(0, state.last_updated_at - state.regime_entered_at)
    return conditional_cif(result, "normal", elapsed, horizon_sec)


def _brier(samples: list[tuple[float, float]]) -> float | None:
    """Mean squared error of (pred, outcome) pairs."""
    if not samples:
        return None
    return sum((p - o) ** 2 for p, o in samples) / len(samples)


def _bss(model: float | None, base: float | None) -> float | None:
    if model is None or base is None or base <= 0:
        return None
    return 1.0 - model / base


@dataclass(frozen=True)
class BacktestWindow:
    """Resolved train/eval bounds for one backtest run (see compute_window)."""

    train_start: date
    train_end: date
    eval_start: date
    eval_end: date
    warmup_start: date
    truth_end: date
    is_historical: bool
    eval_start_epoch: int
    train_end_epoch: int
    eval_end_epoch: int
    outcome_bound_epoch: int


def _midnight_epoch(d: date) -> int:
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp())


def compute_window(
    eval_days: int,
    train_days: int,
    eval_end: date | None,
    *,
    now: datetime,
) -> BacktestWindow:
    """Derive every date/epoch bound for a backtest from the eval-end anchor.

    A live run (eval_end omitted or today) scores only futures that have already
    elapsed. A historical run loads truth one day past eval_end so the last day's
    +max-horizon futures resolve to real outcomes instead of defaulting normal,
    and bounds scored ticks to [eval_start, eval_end] so reconstruction spillover
    into eval_end+1 is never scored.
    """
    today = now.date()
    eval_end = eval_end or today
    eval_start = eval_end - timedelta(days=eval_days - 1)
    train_end = eval_start - timedelta(days=1)
    train_start = train_end - timedelta(days=train_days - 1)
    warmup_start = eval_start - timedelta(days=1)
    is_historical = eval_end < today
    truth_end = eval_end + timedelta(days=1) if is_historical else eval_end
    outcome_bound_epoch = (
        int(now.timestamp())
        if not is_historical
        else _midnight_epoch(truth_end + timedelta(days=1))
    )
    return BacktestWindow(
        train_start=train_start,
        train_end=train_end,
        eval_start=eval_start,
        eval_end=eval_end,
        warmup_start=warmup_start,
        truth_end=truth_end,
        is_historical=is_historical,
        eval_start_epoch=_midnight_epoch(eval_start),
        train_end_epoch=_midnight_epoch(train_end + timedelta(days=1)),
        eval_end_epoch=_midnight_epoch(eval_end + timedelta(days=1)),
        outcome_bound_epoch=outcome_bound_epoch,
    )


# Extra days of truth loaded past eval_end so an incident that onsets late in the
# eval window and clears shortly after is graded as a recovery, not censored.
RECOVERY_TAIL_DAYS = 2


def grade_recovery_timing(
    train_trans: list[TransitionRecord],
    truth: dict[tuple[str, int], str],
    types: dict[tuple[str, int], tuple[str, ...]],
    *,
    train_end_epoch: int,
    eval_start_epoch: int,
    eval_end_epoch: int,
    window_end_epoch: int,
) -> dict[str, Any]:
    """Grade the current recovery model's timing on held-out incident episodes.

    Dwell curves are fit on the TRAIN transitions only (cause -> state -> pooled,
    each with a log-logistic tail), matching the episode grader's cause buckets
    and the production tail splice -- no leakage. Episodes are the severe-only
    truth incidents whose onset falls in the eval window; recovery may land in the
    loaded tail past eval_end, so incidents that clear soon after aren't censored.
    """
    by_cause = compute_dwell_quantiles_by_cause(train_trans, tail_fn=loglogistic_tail)
    by_state = compute_dwell_quantiles(
        train_trans, window_end=train_end_epoch, tail_fn=loglogistic_tail
    )
    pooled = compute_dwell_quantiles(
        [replace(t, route="*") for t in train_trans],
        window_end=train_end_epoch,
        tail_fn=loglogistic_tail,
    ).get("*", {})
    lookup = cause_dwell_lookup(by_cause, by_state, pooled)
    eps = extract_episodes(
        truth, types, window_start=eval_start_epoch, window_end=window_end_epoch
    )
    eval_eps = [e for e in eps if eval_start_epoch <= e.onset < eval_end_epoch]
    rec = episode_recovery(eval_eps, lookup)
    return {"n_eval_episodes": len(eval_eps), **rec}


def run(
    eval_days: int,
    train_days: int,
    out_dir: Path | None,
    eval_end: date | None = None,
) -> dict[str, Any]:
    now = datetime.now(UTC)
    now_epoch = int(now.timestamp())
    w = compute_window(eval_days, train_days, eval_end, now=now)
    train_start, train_end = w.train_start, w.train_end
    eval_start, eval_end = w.eval_start, w.eval_end
    warmup_start = w.warmup_start
    eval_start_epoch = w.eval_start_epoch
    train_end_epoch = w.train_end_epoch
    eval_end_epoch = w.eval_end_epoch
    outcome_bound_epoch = w.outcome_bound_epoch

    print(
        f"train {train_start}..{train_end} ({train_days}d)  |  "
        f"eval {eval_start}..{eval_end} ({eval_days}d, +1d warmup lead)",
        file=sys.stderr,
    )

    cfg = load_config()
    client = make_client(cfg)

    # --- fit params + dwell curves on TRAIN only (temporal split) ---
    by_route_train, corpus, _ = load_series_by_route(cfg, train_start, train_end)
    if not by_route_train:
        raise SystemExit("no training observations in window")
    _global, params_by_route = train(by_route_train)
    clim_normal = {
        r: (
            sum(derive_mta_state(o) == "normal" for o in obs) / len(obs) if obs else 0.8
        )
        for r, obs in by_route_train.items()
    }
    train_trans = load_transitions(client, cfg.bucket, train_start, train_end)
    dwell_curves = compute_dwell_quantiles(train_trans, window_end=train_end_epoch)
    # Pooled-across-routes curve per state: the fallback when a route's own cell
    # is below the min-samples floor, so the KM arm engages on ~every tick.
    pooled_trans = [replace(t, route="*") for t in train_trans]
    pooled = compute_dwell_quantiles(pooled_trans, window_end=train_end_epoch).get(
        "*", {}
    )
    # Log-logistic tail fits for the km_ll arm, on the same censored samples that
    # back each curve. A cell with no fit (no events) falls back to the
    # exponential tail inside _p_leave_ll.
    dwell_fits: dict[str, dict[str, ParametricFit]] = defaultdict(dict)
    for (route_, state_), cell_samples in dwell_samples_by_cell(
        train_trans, window_end=train_end_epoch
    ).items():
        fit = fit_loglogistic(cell_samples)
        if fit is not None:
            dwell_fits[route_][state_] = fit
    pooled_fits: dict[str, ParametricFit] = {}
    for (_route, state_), cell_samples in dwell_samples_by_cell(
        pooled_trans, window_end=train_end_epoch
    ).items():
        fit = fit_loglogistic(cell_samples)
        if fit is not None:
            pooled_fits[state_] = fit

    # Competing-risks CIF on TRAIN transitions: the cif arm reads the elapsed-
    # conditioned cumulative incidence of the D->Normal exit (accounting for the
    # competing D->Suspended off-ramp) instead of km's flat normal-share split.
    cif_samples: dict[str, list[CompetingSample]] = defaultdict(list)
    for tr in train_trans:
        if tr.prev_state == "disrupted":
            cif_samples[tr.route].append((int(tr.dwell_sec), tr.new_state))
    last_seen: dict[str, TransitionRecord] = {}
    for tr in train_trans:
        if tr.route not in last_seen or tr.exited_at > last_seen[tr.route].exited_at:
            last_seen[tr.route] = tr
    for route_, tr in last_seen.items():  # censor each route's open disrupted tail
        if tr.new_state == "disrupted":
            cif_samples[route_].append((max(0, train_end_epoch - tr.exited_at), None))
    cif_by_route: dict[str, CIFResult] = {
        r: cif_curves(s) for r, s in cif_samples.items()
    }
    pooled_cif: CIFResult = cif_curves(
        [s for samples in cif_samples.values() for s in samples]
    )
    print(
        f"fit {len(params_by_route)} routes; dwell cells for "
        f"{sum(len(v) for v in dwell_curves.values())} (route,state) pairs; "
        f"pooled states: {sorted(pooled)}; "
        f"log-logistic tail fits: {sum(len(v) for v in dwell_fits.values())} route cells, "
        f"{sorted(pooled_fits)} pooled",
        file=sys.stderr,
    )

    # --- truth over the eval window, plus a recovery tail so incidents that clear
    # shortly after eval_end aren't censored (truth also feeds outcome lookup) ---
    episode_truth_end = min(eval_end + timedelta(days=RECOVERY_TAIL_DAYS), now.date())
    mask_preds = load_predictions(client, cfg.bucket, eval_start, episode_truth_end)
    mask = presence_mask_from_predictions(mask_preds)
    print(
        f"presence mask: {len(mask.covered)} covered ticks, "
        f"{len(mask.active)} active cells (from {len(mask_preds)} predictions)",
        file=sys.stderr,
    )
    truth_obs = load_truth_observations(
        client, cfg.bucket, eval_start, episode_truth_end, mask=mask
    )
    truth = mta_truth(truth_obs, severity_floor=CANONICAL_SEVERITY_FLOOR)

    # --- replay held-out window per route, collect samples per horizon ---
    # samples[h] : list of (route, {model: p_normal}, persistence, outcome, disrupted_now)
    samples: dict[int, list[tuple[str, dict[str, float], float, float, bool]]] = {
        h: [] for h in HORIZONS_MIN
    }
    fallback = dict.fromkeys(HORIZONS_MIN, 0)
    total = dict.fromkeys(HORIZONS_MIN, 0)

    for route, params in sorted(params_by_route.items()):
        series = load_route_series_r2(
            route, start_date=warmup_start, end_date=eval_end, config=cfg
        )
        if not series:
            continue
        cn = clim_normal.get(route, 0.8)
        geom_state = FilterState(
            probabilities=params.initial,
            regime_entered_at=series[0].tick,
            last_updated_at=series[0].tick,
        )
        published = initial_published_state(geom_state)
        for tick_obs in series:
            geom_state, published = forward_step(
                geom_state, published, tick_obs.observation, params, now=tick_obs.tick
            )
            tick = tick_obs.tick
            if tick < eval_start_epoch or tick >= eval_end_epoch:
                continue  # warmup lead or reconstruction spillover, don't score
            cur_state = truth.get((route, snap_tick(tick)), "normal")
            persistence = 1.0 if cur_state == "normal" else 0.0
            for h in HORIZONS_MIN:
                future = snap_tick(tick) + h * 60
                if future > outcome_bound_epoch:
                    continue  # no observed outcome yet
                outcome = (
                    1.0 if truth.get((route, future), "normal") == "normal" else 0.0
                )
                hsec = h * 60
                geom = project_forward(geom_state, params, hsec // TICK_SECONDS)[0]
                km = _km_residual_p_normal(
                    geom_state,
                    params,
                    route,
                    dwell_curves,
                    pooled,
                    dwell_fits,
                    pooled_fits,
                    hsec,
                    cn,
                    "exp",
                )
                km_ll = _km_residual_p_normal(
                    geom_state,
                    params,
                    route,
                    dwell_curves,
                    pooled,
                    dwell_fits,
                    pooled_fits,
                    hsec,
                    cn,
                    "ll",
                )
                total[h] += 1
                if km is None:
                    fallback[h] += 1
                    km = geom
                if km_ll is None:
                    km_ll = geom
                cif = _cif_p_normal(geom_state, route, cif_by_route, pooled_cif, hsec)
                if cif is None:
                    cif = km_ll
                preds = {"geom": geom, "km": km, "km_ll": km_ll, "cif": cif}
                samples[h].append(
                    (route, preds, persistence, outcome, cur_state != "normal")
                )

    # --- score: pooled Brier + BSS per horizon, overall and disrupted-now ---
    def score(
        rows: list[tuple[str, dict[str, float], float, float, bool]],
    ) -> dict[str, Any] | None:
        if not rows:
            return None
        # per-route climatology = base rate of the event (normal at t+H)
        by_route_out: dict[str, list[float]] = defaultdict(list)
        for route, _pr, _p, o, _d in rows:
            by_route_out[route].append(o)
        base = {r: sum(v) / len(v) for r, v in by_route_out.items()}
        b_per = _brier([(p, o) for _r, _pr, p, o, _d in rows])
        b_clim = _brier([(base[r], o) for r, _pr, _p, o, _d in rows])
        out: dict[str, Any] = {
            "n": len(rows),
            "brier_persistence": b_per,
            "brier_climatology": b_clim,
        }
        for m in MODELS:
            b = _brier([(pr[m], o) for _r, pr, _p, o, _d in rows])
            out[f"brier_{m}"] = b
            out[f"bss_persist_{m}"] = _bss(b, b_per)
            out[f"bss_clim_{m}"] = _bss(b, b_clim)
        return out

    results: list[dict[str, Any]] = []
    for h in HORIZONS_MIN:
        rows = samples[h]
        if not rows:
            continue
        results.append(
            {
                "horizon_min": h,
                "km_fallback_rate": fallback[h] / total[h] if total[h] else 0.0,
                "all": score(rows),
                "disrupted_now": score([r for r in rows if r[4]]),
                "normal_now": score([r for r in rows if not r[4]]),
            }
        )

    types = disruptive_types_by_key(truth_obs)
    # extract_episodes loops tick <= snap_tick(window_end): use the last grid tick
    # of the loaded truth so the next day's first tick isn't swept in.
    episode_window_end_epoch = min(
        now_epoch,
        _midnight_epoch(episode_truth_end + timedelta(days=1)) - TICK_SECONDS,
    )
    recovery = grade_recovery_timing(
        train_trans,
        truth,
        types,
        train_end_epoch=train_end_epoch,
        eval_start_epoch=eval_start_epoch,
        eval_end_epoch=eval_end_epoch,
        window_end_epoch=episode_window_end_epoch,
    )
    recovery["window"] = {
        "recovery_tail_days": RECOVERY_TAIL_DAYS,
        "episode_truth_end": str(episode_truth_end),
        "onset_from": str(eval_start),
        "onset_to": str(eval_end),
    }

    doc = {
        "generated_at": now_epoch,
        "truth_version": TRUTH_VERSION,
        "truth_severity_floor": CANONICAL_SEVERITY_FLOOR,
        "train_window": {
            "start": str(train_start),
            "end": str(train_end),
            "days": train_days,
        },
        "eval_window": {
            "start": str(eval_start),
            "end": str(eval_end),
            "days": eval_days,
        },
        "train_observations": corpus.n_observations,
        "horizons": results,
        "recovery_timing": recovery,
    }

    _print_report(doc)

    out_dir = out_dir or Path("docs/review") / f"{eval_end.isoformat()}-backtest-hsmm"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(
        json.dumps(_json_safe(doc), indent=2, allow_nan=False)
    )
    print(f"\nwrote {out_dir}/summary.json", file=sys.stderr)
    return doc


def _json_safe(obj: Any) -> Any:
    """Recursively replace non-finite floats (NaN/inf) with None so the summary
    is valid strict JSON -- json.dumps writes bare NaN/Infinity otherwise, which
    JS and strict parsers reject."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        items = cast("dict[Any, Any]", obj).items()
        return {k: _json_safe(v) for k, v in items}
    if isinstance(obj, list | tuple):
        seq = cast("list[Any] | tuple[Any, ...]", obj)
        return [_json_safe(v) for v in seq]
    return obj


def _stratum_table(title: str, doc: dict[str, Any], key: str) -> None:
    print(f"\n{title}")
    cols = "".join(f"{m:>9}" for m in MODELS) + f"{'persist':>9}"
    hdr = f"{'H(min)':>6} {'n':>7} | Brier {cols}"
    print(hdr)
    print("-" * len(hdr))
    for r in doc["horizons"]:
        s = r.get(key)
        if not s:
            continue
        briers = "".join(f"{s['brier_' + m]:>9.4f}" for m in MODELS)
        print(
            f"{r['horizon_min']:>6} {s['n']:>7} |       {briers}{s['brier_persistence']:>9.4f}"
        )


def _print_report(doc: dict[str, Any]) -> None:
    print(
        "\n=== Backtest: geometric vs KM-residual p_normal (scored vs fixed alert-truth) ==="
    )
    print(
        "  geom = geometric filter + geometric projection | km = geometric filter + KM-residual projection"
    )
    print(
        f"train {doc['train_window']['start']}..{doc['train_window']['end']}  "
        f"eval {doc['eval_window']['start']}..{doc['eval_window']['end']}"
    )
    print(
        "KM fallback (no curve at all): "
        + ", ".join(
            f"h{r['horizon_min']}={r['km_fallback_rate'] * 100:.0f}%"
            for r in doc["horizons"]
        )
    )

    _stratum_table(
        "ALL ticks (persistence near-perfect here — sticky truth, low discrimination):",
        doc,
        "all",
    )
    _stratum_table(
        "DISRUPTED-now ticks (where dwell timing matters — the meaningful stratum):",
        doc,
        "disrupted_now",
    )
    _stratum_table(
        "NORMAL-now ticks (persistence predicts stay-normal ~1.0; the km-residual "
        "normal branch must beat it here to graduate normal-condition p_normal):",
        doc,
        "normal_now",
    )

    # Lower Brier = better. When disruptions persist through every horizon,
    # persistence is a degenerate 0 yardstick, so rank the arms against each
    # other on the disrupted-now stratum.
    strata = [r["disrupted_now"] for r in doc["horizons"] if r.get("disrupted_now")]

    def _mean(model: str) -> float | None:
        vals = [
            s[f"brier_{model}"] for s in strata if s.get(f"brier_{model}") is not None
        ]
        return sum(vals) / len(vals) if vals else None

    means = {m: _mean(m) for m in MODELS}
    # Persistence Brier == 0 across the stratum means no disrupted tick recovered
    # within any horizon: the arm ranking then only reflects how confidently each
    # predicts CONTINUED disruption, not recovery timing.
    persist_vals = [
        s["brier_persistence"] for s in strata if s.get("brier_persistence") is not None
    ]
    no_recoveries = bool(persist_vals) and max(persist_vals) < 1e-9
    if no_recoveries:
        print(
            "\n  NOTE: no disrupted-now tick recovered within any horizon "
            "(persistence Brier 0)."
        )
        print(
            "  The ranking below measures continued-disruption prediction, "
            "NOT recovery timing."
        )
    print("\n  mean Brier (disrupted-now, lower=better):")
    for m, v in sorted(means.items(), key=lambda kv: (kv[1] is None, kv[1])):
        print(f"    {m:>8}: {v:.4f}" if v is not None else f"    {m:>8}:   n/a")
    valid = {m: v for m, v in means.items() if v is not None}
    best = min(valid, key=valid.__getitem__)
    best_label = (
        "BEST ARM (continued-disruption)" if no_recoveries else "BEST FORECAST ARM"
    )
    print(f"\n  {best_label}: {best} (mean Brier {means[best]:.4f})")
    if means.get("geom") and means.get("km"):
        print(
            f"  projection effect (geom proj -> KM-residual proj): "
            f"{means['geom']:.4f} -> {means['km']:.4f}"
        )
    m_km, m_ll = means.get("km"), means.get("km_ll")
    if m_km is not None and m_ll is not None:
        delta = m_ll - m_km
        if no_recoveries:
            direction = "lower" if delta < 0 else "higher" if delta > 0 else "equal"
            verdict = f"{direction} on continued-disruption only"
        else:
            verdict = "ship" if delta < 0 else "no ship"
        print(
            f"  tail splice effect (exp tail -> log-logistic tail): "
            f"{m_km:.4f} -> {m_ll:.4f} (Δ {delta:+.4f}, {verdict})"
        )
    rec: dict[str, Any] = doc.get("recovery_timing") or {}
    print("\nRECOVERY TIMING (current model, no-leakage temporal split):")
    n_scored = rec.get("n_scored", 0)
    print(
        f"  episodes: {rec.get('n_eval_episodes', 0)} onset-in-window, "
        f"{n_scored} scored, {rec.get('n_censored_excluded', 0)} censored, "
        f"{rec.get('n_no_curve', 0)} no-curve"
    )
    report: dict[str, Any] = rec.get("report") or {}
    pr: dict[str, Any] = report.get("per_regime") or {}
    if n_scored and pr:
        print(
            f"  CRPS/min per-incident {pr['mean_crps']:.1f} "
            f"(climatology {pr['baseline_crps']:.1f}, skill {pr['skill']:+.2f})"
        )
        print(
            f"  PIT mean per-incident {pr['mean_pit']:.2f} "
            "(<0.5 pessimistic / >0.5 optimistic / 0.5 calibrated)"
        )
        horizons: list[dict[str, float]] = report.get("horizons") or []
        for hz in horizons:
            print(
                f"    recover-by-{int(hz['h'])}min: predicted "
                f"{hz['predicted']:.2f} vs observed {hz['observed']:.2f}"
            )
        rec_verdict: dict[str, Any] = rec.get("verdict") or {}
        if rec_verdict.get("verdict"):
            print(
                f"  verdict: {rec_verdict['verdict']} — "
                f"{rec_verdict.get('explain', '')}"
            )
    else:
        print("  not enough uncensored episodes with a dwell curve to grade timing.")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Tier-1 HSMM decision-gate backtest")
    p.add_argument(
        "--eval-days", type=int, default=3, help="held-out window length (most recent)"
    )
    p.add_argument(
        "--train-days",
        type=int,
        default=6,
        help="training window length (precedes eval)",
    )
    p.add_argument("--out", type=Path, default=None, help="output dir")
    p.add_argument(
        "--eval-end",
        type=date.fromisoformat,
        default=None,
        help="last eval day (YYYY-MM-DD); default today, for a historical window",
    )
    args = p.parse_args(argv)
    run(args.eval_days, args.train_days, args.out, eval_end=args.eval_end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
