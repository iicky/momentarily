"""Demo: run the HMM forward filter over one route's observation history.

Reads the local collector archive, builds per-tick observations for the chosen
route, runs the forward filter with hand-picked bootstrap parameters, and
prints a regime trajectory.

Usage:
    uv run python -m training.run_filter --route 6
    uv run python -m training.run_filter --route 1 --data-dir ./data
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from momentarily.hmm import (
    EmissionParams,
    FilterState,
    HMMParams,
    expected_dwell_ticks,
    fit_em,
    forward_step,
    initial_published_state,
    project_forward,
)
from training.load import TICK_SECONDS, load_route_series

# Bootstrap HMM parameters — values from the methodology spec (c72.5 / papers.md).
# Tuned for the alerts feed: normal = quiet, disrupted = elevated, suspended = severe.
BOOTSTRAP_PARAMS = HMMParams(
    transition=(
        (0.95, 0.04, 0.01),  # normal → ...
        (0.08, 0.90, 0.02),  # disrupted → ...
        (0.02, 0.10, 0.88),  # suspended → ...
    ),
    initial=(0.9, 0.08, 0.02),
    emissions=EmissionParams(
        poisson_lambda=(0.3, 4.0, 12.0),
        gamma_alpha=(1.0, 3.0, 6.0),
        gamma_beta=(2.0, 0.4, 0.2),
        bernoulli_p=(0.001, 0.05, 0.95),
    ),
)


def _fmt_time(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%d %H:%M UTC")


def _argmax_state(probs: tuple[float, float, float]) -> str:
    states = ("normal", "disrupted", "suspended")
    return states[max(range(3), key=lambda i: probs[i])]


def _fmt_params(params: HMMParams) -> str:
    lam = params.emissions.poisson_lambda
    gp = params.emissions.gamma_alpha, params.emissions.gamma_beta
    p_sus = params.emissions.bernoulli_p
    a = params.transition
    return (
        f"  poisson_lambda = ({lam[0]:.2f}, {lam[1]:.2f}, {lam[2]:.2f})\n"
        f"  gamma_alpha    = ({gp[0][0]:.2f}, {gp[0][1]:.2f}, {gp[0][2]:.2f})\n"
        f"  gamma_beta     = ({gp[1][0]:.3f}, {gp[1][1]:.3f}, {gp[1][2]:.3f})\n"
        f"  bernoulli_p    = ({p_sus[0]:.3f}, {p_sus[1]:.3f}, {p_sus[2]:.3f})\n"
        f"  self-loop diag = ({a[0][0]:.3f}, {a[1][1]:.3f}, {a[2][2]:.3f})"
    )


def run(route_id: str, data_dir: Path, train: bool = False) -> int:
    series = load_route_series(data_dir, route_id)
    if not series:
        print(f"No data for route {route_id!r} in {data_dir}")
        return 1

    print(
        f"Route {route_id}: {len(series)} ticks "
        f"({_fmt_time(series[0].tick)} → {_fmt_time(series[-1].tick)})"
    )

    params = BOOTSTRAP_PARAMS
    if train:
        print("\nBootstrap params:")
        print(_fmt_params(params))
        obs = [to.observation for to in series]
        params, log_liks = fit_em(obs, params, max_iterations=50, tolerance=1e-4)
        print(
            f"\nBaum-Welch EM: {len(log_liks)} iterations, "
            f"log-likelihood {log_liks[0]:.2f} → {log_liks[-1]:.2f}"
        )
        print("\nLearned params:")
        print(_fmt_params(params))
        print()

    print(
        f"{'tick':>20} {'alerts':>6} {'sev':>4} {'sus':>4}  "
        f"{'P(N)':>5} {'P(D)':>5} {'P(S)':>5}  {'state':<10} {'~recov':>8}"
    )
    print("-" * 100)

    state = FilterState(
        probabilities=params.initial,
        regime_entered_at=series[0].tick,
        last_updated_at=series[0].tick,
    )
    published = initial_published_state(state)

    # Print every Nth tick so a multi-hour history fits the terminal.
    every = max(1, len(series) // 60)

    raw_transitions: list[tuple[int, str, str]] = []
    published_transitions: list[tuple[int, str, str]] = []
    last_argmax = _argmax_state(state.probabilities)
    last_published = published.label

    for i, tick_obs in enumerate(series):
        state, published = forward_step(
            state, published, tick_obs.observation, params, now=tick_obs.tick
        )
        argmax = _argmax_state(state.probabilities)
        if argmax != last_argmax:
            raw_transitions.append((tick_obs.tick, last_argmax, argmax))
            last_argmax = argmax
        if published.label != last_published:
            published_transitions.append(
                (tick_obs.tick, last_published, published.label)
            )
            last_published = published.label

        if i % every == 0 or i == len(series) - 1:
            recov, _, _ = expected_dwell_ticks(state, params)
            obs = tick_obs.observation
            print(
                f"{_fmt_time(tick_obs.tick):>20} "
                f"{obs.alert_count:>6} {obs.severity_sum:>4} "
                f"{'Y' if obs.has_suspended_alert else '·':>4}  "
                f"{state.probabilities[0]:>5.2f} "
                f"{state.probabilities[1]:>5.2f} "
                f"{state.probabilities[2]:>5.2f}  "
                f"{published.label:<10} "
                f"{recov * TICK_SECONDS // 60:>5}min"
            )

    print()
    print(
        f"Raw argmax transitions:       {len(raw_transitions)}\n"
        f"Published (after hysteresis): {len(published_transitions)}"
    )
    for tick, prev, curr in published_transitions:
        print(f"  {_fmt_time(tick)}: {prev} → {curr}")

    # Projection from final state
    p30 = project_forward(state, params, ticks_ahead=30 // 5)
    p60 = project_forward(state, params, ticks_ahead=60 // 5)
    p120 = project_forward(state, params, ticks_ahead=120 // 5)
    print()
    print("Forward projection from final state:")
    print(f"  P(normal in  30 min): {p30[0]:.3f}")
    print(f"  P(normal in  60 min): {p60[0]:.3f}")
    print(f"  P(normal in 120 min): {p120[0]:.3f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route", required=True, help="GTFS route_id, e.g. 1, A, 7")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Collector data directory (default: ./data)",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run Baum-Welch EM on the data first to learn params, "
        "then run the filter with the learned params.",
    )
    args = parser.parse_args()
    return run(args.route, args.data_dir, train=args.train)


if __name__ == "__main__":
    raise SystemExit(main())
