"""Per-route EM trainer with empirical-Bayes prior anchoring.

Each run:
  1. Pull the alerts archive over a window (default 14 days) from R2.
  2. Pool all routes into one corpus and fit a global HMM — the prior.
  3. For each route with enough data, fit again with `prior_params=global`
     and Dirichlet/Gamma/Beta pseudo-counts (`prior_strength`).
  4. Routes with thin data inherit the global prior as-is.
  5. Write state/params.json (live pointer) + state/params/v<epoch>.json
     (immutable per-run snapshot) — the Worker picks up params.json on its
     next cron tick; the versioned copies are the rollback trail.

Run with:
    murk exec -- python -m training.train_em [--days 14] [--start/--end DATE]
        [--routes A,C,E] [--min-ticks N] [--prior-strength 100] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
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
# Immutable per-run snapshots live under this prefix as v<trained_at>.json.
VERSIONED_PARAMS_PREFIX = "state/params/"
SCHEMA_VERSION = "1"

# A route needs at least this many ticks of data to fit per-route — under that,
# we fall back to the global prior.
MIN_TICKS_PER_ROUTE = 288  # one day at the 5-min grid


@dataclass(frozen=True)
class CorpusStats:
    """Audit metadata about the archive window a run actually trained on."""

    start_tick: int
    end_tick: int
    n_observations: int  # real (alert-bearing) tick-observations, pre-quiet-fill

    @property
    def span_seconds(self) -> int:
        return self.end_tick - self.start_tick

# EM on a thin or mostly-quiet corpus drives transition self-loops toward 1.0,
# which pins the forward filter so a route can never leave a regime. Cap the
# diagonal, and refuse to publish at all under a week of archive. The original
# bound was two weeks; once the EM variance/Bernoulli floors landed (momentarily-p8y)
# the dominant risk of thin data — degenerate emissions — was no longer in
# play, so we relaxed the gate. _cap_self_loops still bounds the transition
# self-loops independently. See momentarily-625.
MAX_SELF_LOOP = 0.97
MIN_DATA_DAYS = 5


def _cap_self_loops(params: HMMParams, max_self: float = MAX_SELF_LOOP) -> HMMParams:
    """Clamp each transition row's diagonal to `max_self`, redistributing the
    freed mass across that row's off-diagonal entries (proportionally, or
    evenly when they're all zero)."""
    rows: list[tuple[float, float, float]] = []
    for s in range(3):
        row = list(params.transition[s])
        if row[s] <= max_self:
            rows.append((row[0], row[1], row[2]))
            continue
        freed = row[s] - max_self
        row[s] = max_self
        off = [j for j in range(3) if j != s]
        off_sum = sum(row[j] for j in off)
        for j in off:
            share = row[j] / off_sum if off_sum > 0 else 1.0 / len(off)
            row[j] += freed * share
        rows.append((row[0], row[1], row[2]))
    return HMMParams(
        transition=tuple(rows),
        initial=params.initial,
        emissions=params.emissions,
        emissions_by_bin=params.emissions_by_bin,
    )


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
) -> tuple[dict[str, list[Observation]], CorpusStats]:
    """Single R2 pass: fetch alerts, build per-route quiet-filled series.

    Returns (by_route, corpus). `corpus` describes the *actual* observed ticks
    before quiet-filling — fill_quiet_ticks pads every series to the requested
    window, so series length can't tell us how much real archive we have. The
    publish gate and the params.json audit block both need the unpadded view.
    """
    bodies = fetch_alert_versions(cfg, start_date=start, end_date=end)
    all_ticks = build_tick_observations(bodies)
    if not all_ticks:
        return {}, CorpusStats(start_tick=0, end_tick=0, n_observations=0)

    ticks = [t.tick for t in all_ticks]
    corpus = CorpusStats(
        start_tick=min(ticks),
        end_tick=max(ticks),
        n_observations=len(all_ticks),
    )

    start_tick, end_tick_excl = _aligned_window(start, end)
    last_tick = end_tick_excl - TICK_SECONDS

    by_route: dict[str, list[Observation]] = {}
    seen_routes = {t.route_id for t in all_ticks}
    for route in sorted(seen_routes):
        filled: list[TickObservation] = fill_quiet_ticks(
            all_ticks, route, start_tick=start_tick, end_tick=last_tick
        )
        by_route[route] = [t.observation for t in filled]
    return by_route, corpus


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
    global_prior = _cap_self_loops(global_prior)

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
        out[route] = _cap_self_loops(fitted)
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
    client: S3Client,
    bucket: str,
    per_route: dict[str, HMMParams],
    *,
    corpus: CorpusStats,
    n_routes_trained: int,
    trained_at: int | None = None,
) -> str:
    """Write the live params pointer plus an immutable versioned snapshot.

    The Worker reads state/params.json; the state/params/v<epoch>.json copies
    give us a per-run rollback trail. Returns the versioned key.
    """
    trained_at = trained_at or int(datetime.now(UTC).timestamp())
    doc = {
        "schema_version": SCHEMA_VERSION,
        "trained_at": trained_at,
        "training_corpus": {
            "start_tick": corpus.start_tick,
            "end_tick": corpus.end_tick,
            "n_routes_trained": n_routes_trained,
            "n_observations": corpus.n_observations,
        },
        "routes": {r: _params_to_json(p) for r, p in per_route.items()},
    }
    body = json.dumps(doc).encode()
    versioned_key = f"{VERSIONED_PARAMS_PREFIX}v{trained_at}.json"
    for key in (PARAMS_KEY, versioned_key):
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            CacheControl="public, max-age=300, s-maxage=900",
        )
    return versioned_key


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Per-route EM trainer")
    parser.add_argument(
        "--days", type=int, default=14, help="trailing-window size when --start is unset"
    )
    parser.add_argument(
        "--start", help="window start date YYYY-MM-DD (overrides --days)"
    )
    parser.add_argument("--end", help="window end date YYYY-MM-DD (default: today UTC)")
    parser.add_argument(
        "--routes", help="comma-separated route whitelist (default: all observed)"
    )
    parser.add_argument(
        "--min-ticks",
        type=int,
        default=MIN_TICKS_PER_ROUTE,
        help="routes with fewer observations inherit the global prior",
    )
    parser.add_argument(
        "--prior-strength",
        type=float,
        default=100.0,
        help="pseudo-counts strength for per-route prior anchor (in tick units)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print learned params instead of writing to R2",
    )
    args = parser.parse_args(argv)

    cfg = load_config()
    end_date = date.fromisoformat(args.end) if args.end else datetime.now(UTC).date()
    start_date = (
        date.fromisoformat(args.start)
        if args.start
        else end_date - timedelta(days=args.days - 1)
    )
    series, corpus = load_series_by_route(cfg, start_date, end_date)
    if not series:
        print("no observations in archive — skipping training", file=sys.stderr)
        return 1

    if args.routes:
        whitelist = {r.strip() for r in args.routes.split(",") if r.strip()}
        series = {r: s for r, s in series.items() if r in whitelist}
        if not series:
            print(
                f"none of --routes {sorted(whitelist)} present in archive",
                file=sys.stderr,
            )
            return 1

    global_prior, per_route = train(
        series, prior_strength=args.prior_strength, min_ticks=args.min_ticks
    )

    if args.dry_run:
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

    if corpus.span_seconds < MIN_DATA_DAYS * 86_400:
        print(
            f"archive spans {corpus.span_seconds / 86_400:.1f}d "
            f"(< {MIN_DATA_DAYS}d minimum) — refusing to publish; thin data "
            "overfits transition self-loops",
            file=sys.stderr,
        )
        return 1

    n_routes_trained = sum(1 for p in per_route.values() if p is not global_prior)
    client = make_client(cfg)
    versioned_key = write_params(
        client,
        cfg.bucket,
        per_route,
        corpus=corpus,
        n_routes_trained=n_routes_trained,
    )
    print(
        f"published {PARAMS_KEY} + {versioned_key}: "
        f"{n_routes_trained}/{len(per_route)} routes fitted "
        f"(prior_strength={args.prior_strength})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
