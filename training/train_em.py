"""Per-route EM trainer with empirical-Bayes prior anchoring.

Each run:
  1. Pull the alerts archive over a window (default 14 days) from R2.
  2. Pool all routes into one corpus and fit a global HMM — the prior.
  3. For each route with enough data, fit again with `prior_params=global`
     and Dirichlet/Gamma/Beta pseudo-counts (`prior_strength`).
  4. Routes with thin data inherit the global prior as-is.
  5. Write state/params.json — the Worker picks it up on its next cron tick.

Run with:
    murk exec -- python -m training.train_em [--days 14] [--prior-strength 100]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from momentarily.hmm import HMMParams, Observation, fit_em
from training.load import TICK_SECONDS, TickObservation, fill_quiet_ticks
from training.load_r2 import build_tick_observations, fetch_alert_versions
from training.r2_client import R2Config, load_config, make_client
from training.run_filter import BOOTSTRAP_PARAMS

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


PARAMS_KEY = "state/params.json"
SCHEMA_VERSION = "1"

# A route needs at least this many ticks of data to fit per-route — under that,
# we fall back to the global prior.
MIN_TICKS_PER_ROUTE = 288  # one day at the 5-min grid


def _aligned_window(start: date, end: date) -> tuple[int, int]:
    """Tick-aligned UTC window covering [start, end+1day)."""
    start_dt = datetime(start.year, start.month, start.day, tzinfo=UTC)
    end_dt = datetime(end.year, end.month, end.day, tzinfo=UTC) + timedelta(days=1)
    start_epoch = (int(start_dt.timestamp()) // TICK_SECONDS) * TICK_SECONDS
    end_epoch = (int(end_dt.timestamp()) // TICK_SECONDS) * TICK_SECONDS
    return start_epoch, end_epoch


def load_series_by_route(
    cfg: R2Config,
    start: date,
    end: date,
) -> dict[str, list[Observation]]:
    """Single R2 pass: fetch alerts, build per-route quiet-filled series."""
    bodies = fetch_alert_versions(cfg, start_date=start, end_date=end)
    all_ticks = build_tick_observations(bodies)
    if not all_ticks:
        return {}

    start_tick, end_tick_excl = _aligned_window(start, end)
    last_tick = end_tick_excl - TICK_SECONDS

    by_route: dict[str, list[Observation]] = {}
    seen_routes = {t.route_id for t in all_ticks}
    for route in sorted(seen_routes):
        filled: list[TickObservation] = fill_quiet_ticks(
            all_ticks, route, start_tick=start_tick, end_tick=last_tick
        )
        by_route[route] = [t.observation for t in filled]
    return by_route


def train(
    series_by_route: dict[str, list[Observation]],
    *,
    prior_strength: float = 100.0,
    min_ticks: int = MIN_TICKS_PER_ROUTE,
) -> tuple[HMMParams, dict[str, HMMParams]]:
    """Returns (global_prior, per_route_params). Doesn't touch R2."""
    if not series_by_route:
        raise ValueError("no observations to train on")

    pooled: list[Observation] = []
    for series in series_by_route.values():
        pooled.extend(series)
    global_prior, _ = fit_em(pooled, BOOTSTRAP_PARAMS, max_iterations=50)

    out: dict[str, HMMParams] = {}
    for route, series in series_by_route.items():
        if len(series) < min_ticks:
            out[route] = global_prior
            continue
        fitted, _ = fit_em(
            series,
            global_prior,
            max_iterations=30,
            prior_params=global_prior,
            prior_strength=prior_strength,
        )
        out[route] = fitted
    return global_prior, out


def _params_to_json(params: HMMParams) -> dict[str, Any]:
    """Serialize HMMParams to the loose schema the Worker reads."""
    emissions = asdict(params.emissions)
    body: dict[str, Any] = {
        "transition": [list(row) for row in params.transition],
        "initial": list(params.initial),
        "emissions": emissions,
    }
    if params.emissions_by_bin is not None:
        body["emissions_by_bin"] = [asdict(e) for e in params.emissions_by_bin]
    return body


def write_params(
    client: "S3Client",
    bucket: str,
    per_route: dict[str, HMMParams],
    *,
    trained_at: int | None = None,
) -> None:
    trained_at = trained_at or int(datetime.now(UTC).timestamp())
    doc = {
        "schema_version": SCHEMA_VERSION,
        "trained_at": trained_at,
        "routes": {r: _params_to_json(p) for r, p in per_route.items()},
    }
    client.put_object(
        Bucket=bucket,
        Key=PARAMS_KEY,
        Body=json.dumps(doc).encode(),
        ContentType="application/json",
        CacheControl="public, max-age=300, s-maxage=900",
    )


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-route EM trainer")
    parser.add_argument("--days", type=int, default=14)
    parser.add_argument(
        "--prior-strength",
        type=float,
        default=100.0,
        help="pseudo-counts strength for per-route prior anchor (in tick units)",
    )
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config()
    today = datetime.now(UTC).date()
    start = today - timedelta(days=args.days - 1)
    series = load_series_by_route(cfg, start, today)
    if not series:
        print("no observations in archive — skipping training", file=sys.stderr)
        return 1

    global_prior, per_route = train(series, prior_strength=args.prior_strength)

    if args.no_publish:
        print(
            json.dumps(
                {
                    "global_prior": _params_to_json(global_prior),
                    "routes": {r: _params_to_json(p) for r, p in per_route.items()},
                },
                indent=2,
            )
        )
        return 0

    client = make_client(cfg)
    write_params(client, cfg.bucket, per_route)
    print(
        f"published {PARAMS_KEY}: {len(per_route)} routes fitted "
        f"(prior_strength={args.prior_strength}, window={args.days}d)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
