"""Static IRT/BMT/IND division map, and the variance decomposition that tests
whether it earns a level in the dwell hierarchy.

Every route the trainer sees pools toward one system-wide centre today
(`training/pooled_dwell.py`). Before adding a division level between "route"
and "system" this module answers a narrower question first: do per-route
fitted scales actually cluster by division, or would a division level just be
extra machinery pooling noise?

Nothing here writes survival math. `decompose_variance` is a one-way ANOVA
(between vs within variance, eta-squared, and a distribution-free permutation
p-value) over whatever per-route statistic the caller already fitted --
`main()` below fits it via the existing `pooled_dwell.partially_pooled_dwell`,
unmodified.

--- The division map ---

Source: the real NYCT static GTFS feed (`training.gtfs_static.GTFS_STATIC_URL`,
`routes.txt`, fetched and inspected 2026-08-11) gives the definitive 29-row
route_id space -- 28 rows with route_type "1" (subway) and one, SI, with
route_type "2" (Staten Island Railway: administratively NYCT but never part
of the three-division subway network, physically disconnected). That fixes
which route ids exist; it does not carry IRT/BMT/IND, which GTFS has no field
for. Division assignment below is cross-checked against two independent
primary/secondary sources:

  * Wikipedia "A Division (New York City Subway)" / "B Division (New York
    City Subway)": A Division ("also known as the IRT Division") is the
    numbered routes plus the 42nd Street Shuttle; B Division is the lettered
    routes plus the Franklin Avenue and Rockaway Park shuttles, split into
    B1 (BMT) / B2 (IND) "for chaining purposes ... still sometimes referred
    to as the BMT Division and IND Division".
  * nycsubway.org's "Subway FAQ" ("Which Lines Were Former IRT, BMT, IND?")
    itemizes the post-1940-unification lineage route by route. Quoted
    verbatim where a route is named directly.

Express variants (6X, 7X, FX) inherit their base route's division -- same
rule as `gtfs_static.base_route` / worker's `baseRoute`, not a second one.

Four assignments are flagged below because the FAQ documents mixed heritage
or doesn't itemize the route by name; get these wrong and the whole test is
invalidated silently, so they're called out rather than asserted quietly:

  * M -- FAQ states "M: BMT along entire length" (its home yard and the
    Myrtle/Nassau St trackage are BMT-built). Flagged because the MTA's own
    route-bullet color scheme groups M with the IND Sixth Ave family (B, D,
    F -- all orange) since 2010, and M now interlines through IND Queens
    Blvd/6th Ave trackage on weekdays. Taking the FAQ's operational-lineage
    reading, not the bullet-color reading.
  * Q -- FAQ describes Q as IND-built north of 47-50th St, BMT (Broadway
    Manhattan trunk + Brighton Line) everywhere else; the IND-adjacent piece
    is the 63rd St/2nd Ave extension, built by the MTA well after the 1940
    divisions existed, not inherited from either predecessor company.
    Majority mileage and identity is BMT; going with BMT.
  * W -- not itemized in the FAQ (discontinued when the FAQ was written,
    reinstated 2010); it is the same physical Astoria/Broadway/Whitehall
    trackage as N, which the FAQ names BMT directly. Inferred by identity,
    not independently documented.
  * H (Rockaway Park Shuttle) -- not itemized in the FAQ's per-route list.
    Inferred IND from two independent sources: Wikipedia's Rockaway Park
    Shuttle article points its station listing at "IND Rockaway Line", and
    the FAQ's own narrative for A has the Rockaway branch "becom[ing] a
    subway line in 1956" via the IND connection.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, Literal

Division = Literal["IRT", "BMT", "IND"]

DIVISION_BY_ROUTE: dict[str, Division] = {
    # IRT (A Division): numbered routes + the Times Sq-Grand Central (42nd
    # St) Shuttle. High confidence -- FAQ names this group directly.
    "1": "IRT",
    "2": "IRT",
    "3": "IRT",
    "4": "IRT",
    "5": "IRT",
    "6": "IRT",
    "6X": "IRT",  # express variant of 6
    "7": "IRT",
    "7X": "IRT",  # express variant of 7
    "GS": "IRT",  # Times Sq-Grand Central Shuttle
    # BMT (B1 Division).
    "J": "BMT",
    "Z": "BMT",
    "L": "BMT",
    "M": "BMT",  # flagged -- see module docstring
    "N": "BMT",
    "Q": "BMT",  # flagged -- see module docstring
    "R": "BMT",
    "W": "BMT",  # flagged -- see module docstring
    "FS": "BMT",  # Franklin Avenue Shuttle
    # IND (B2 Division).
    "A": "IND",
    "B": "IND",
    "C": "IND",
    "D": "IND",
    "E": "IND",
    "F": "IND",
    "FX": "IND",  # express variant of F
    "G": "IND",
    "H": "IND",  # Rockaway Park Shuttle, flagged -- see module docstring
}

# Staten Island Railway: NYCT-operated but never merged into the IRT/BMT/IND
# network -- physically disconnected from the rest of the subway, and its
# own route_type ("2", rail, vs "1", subway) in the real GTFS feed. Excluded
# by name instead of forced into a division it was never part of.
ROUTES_WITHOUT_DIVISION: frozenset[str] = frozenset({"SI"})

# The full 29-route_id space from the real feed (verified 2026-08-11). A
# route present in the trainer's data but absent here is either new service
# or a mapping bug -- `main()` below warns on the difference rather than
# silently dropping the route from the analysis.
ALL_KNOWN_ROUTES: frozenset[str] = (
    frozenset(DIVISION_BY_ROUTE) | ROUTES_WITHOUT_DIVISION
)


# --- Variance decomposition -------------------------------------------------


@dataclass(frozen=True)
class DivisionGroupStats:
    division: str
    n: int
    mean: float
    # Sample variance (ddof=1); 0.0 for a singleton group, which contributes
    # nothing to within-group SS but still casts its one vote on the mean.
    variance: float


@dataclass(frozen=True)
class VarianceDecomposition:
    """One-way ANOVA of a per-route statistic grouped by division."""

    groups: tuple[DivisionGroupStats, ...]
    n_total: int
    grand_mean: float
    ss_between: float
    ss_within: float
    df_between: int
    df_within: int
    ms_between: float
    ms_within: float
    # MS_between / MS_within. None when MS_within is exactly 0 (degenerate:
    # every route agrees with its division's mean, ratio is undefined/infinite).
    f_ratio: float | None
    # SS_between / SS_total, in [0, 1] -- the share of variance in the
    # statistic that division membership explains. The requested effect size.
    eta_squared: float


def decompose_variance(
    values_by_route: Mapping[str, float],
    division_by_route: Mapping[str, str] = DIVISION_BY_ROUTE,
) -> VarianceDecomposition:
    """Between- vs within-division variance of `values_by_route` (e.g. a
    fitted log-scale per route).

    Routes absent from `division_by_route` (SI, or anything unmapped) are
    dropped rather than guessed into a group. Raises ValueError if that
    leaves nothing to decompose.
    """
    by_division: dict[str, list[float]] = defaultdict(list)
    for route, value in values_by_route.items():
        division = division_by_route.get(route)
        if division is not None:
            by_division[division].append(value)

    all_values = [v for values in by_division.values() for v in values]
    n_total = len(all_values)
    if n_total == 0:
        raise ValueError("no routes with both a value and a known division")
    grand_mean = statistics.fmean(all_values)

    groups: list[DivisionGroupStats] = []
    ss_between = 0.0
    ss_within = 0.0
    for division in sorted(by_division):
        values = by_division[division]
        n = len(values)
        mean = statistics.fmean(values)
        variance = statistics.variance(values) if n >= 2 else 0.0
        groups.append(DivisionGroupStats(division, n, mean, variance))
        ss_between += n * (mean - grand_mean) ** 2
        ss_within += sum((v - mean) ** 2 for v in values)

    k = len(groups)
    df_between = k - 1
    df_within = n_total - k
    ms_between = ss_between / df_between if df_between > 0 else 0.0
    ms_within = ss_within / df_within if df_within > 0 else 0.0
    f_ratio = (ms_between / ms_within) if ms_within > 0 else None
    ss_total = ss_between + ss_within
    eta_squared = (ss_between / ss_total) if ss_total > 0 else 0.0

    return VarianceDecomposition(
        groups=tuple(groups),
        n_total=n_total,
        grand_mean=grand_mean,
        ss_between=ss_between,
        ss_within=ss_within,
        df_between=df_between,
        df_within=df_within,
        ms_between=ms_between,
        ms_within=ms_within,
        f_ratio=f_ratio,
        eta_squared=eta_squared,
    )


def permutation_p_value(
    values_by_route: Mapping[str, float],
    division_by_route: Mapping[str, str] = DIVISION_BY_ROUTE,
    *,
    n_permutations: int = 10_000,
    seed: int = 0,
) -> float:
    """Empirical p-value for the observed eta-squared under the null that
    division is uninformative: shuffle division labels across the same
    routes' values, recompute eta-squared each draw, and report the fraction
    at least as extreme (+1/+1 smoothing, the standard permutation-test
    convention -- never reports exactly 0).

    Distribution-free by construction: no F-distribution assumption, which
    matters with ~9 routes per group. Deterministic for a fixed `seed`.
    """
    routes = [r for r in values_by_route if r in division_by_route]
    if len(routes) < 3:
        raise ValueError("need routes spanning at least one division to permute")
    values = {r: values_by_route[r] for r in routes}
    original_labels = [division_by_route[r] for r in routes]

    observed = decompose_variance(
        values, dict(zip(routes, original_labels, strict=True))
    ).eta_squared

    rng = random.Random(seed)
    labels = list(original_labels)
    at_least_as_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(labels)
        permuted = decompose_variance(
            values, dict(zip(routes, labels, strict=True))
        ).eta_squared
        if permuted >= observed:
            at_least_as_extreme += 1
    return (at_least_as_extreme + 1) / (n_permutations + 1)


# --- Measuring the premise against real data --------------------------------
#
# Everything below touches R2. None of it is called by the test suite;
# exercised by running this module directly (`murk exec -- uv run python -m
# training.divisions`), not by pytest.


def _date_window(days: int) -> tuple[date, date]:
    end = datetime.now(UTC).date()
    start = end - timedelta(days=days - 1)
    return start, end


def main(argv: Sequence[str] | None = None) -> int:
    from training.dwell import dwell_samples_by_cell
    from training.eval import (
        load_predictions,
        load_transitions,
        open_regimes_from_predictions,
    )
    from training.pooled_dwell import MIN_VOTER_EVENTS, partially_pooled_dwell
    from training.r2_client import load_config, make_client

    parser = argparse.ArgumentParser(
        description="Fit the flat pooled normal-dwell scale per route (alert "
        "arm) and test whether it clusters by IRT/BMT/IND division."
    )
    parser.add_argument("--days", type=int, default=14, help="trailing window")
    parser.add_argument("--min-voter-events", type=int, default=MIN_VOTER_EVENTS)
    parser.add_argument("--permutations", type=int, default=10_000)
    args = parser.parse_args(argv)

    start, end = _date_window(args.days)
    cfg = load_config()
    client = make_client(cfg)

    print(
        f"fetching alert-arm transitions + predictions {start}..{end}", file=sys.stderr
    )
    transitions = load_transitions(client, cfg.bucket, start, end)
    window_end = int(datetime.now(UTC).timestamp())
    open_regimes = (
        open_regimes_from_predictions(
            load_predictions(client, cfg.bucket, start, end), window_end=window_end
        )
        or None
    )
    print(f"{len(transitions)} transitions", file=sys.stderr)

    by_cell = dwell_samples_by_cell(
        transitions, window_end=window_end, open_regimes=open_regimes
    )
    samples_by_route = {
        route: samples
        for (route, state), samples in by_cell.items()
        if state == "normal"
    }
    fits = partially_pooled_dwell(
        samples_by_route, min_voter_events=args.min_voter_events
    )
    print(f"{len(fits)} routes with a fitted normal-dwell scale", file=sys.stderr)

    unmapped = sorted(set(fits) - ALL_KNOWN_ROUTES)
    if unmapped:
        print(f"WARNING: routes with no division mapping: {unmapped}", file=sys.stderr)

    log_scale_all = {route: math.log(fit.scale_sec) for route, fit in fits.items()}
    log_scale_own = {
        route: math.log(fit.scale_sec)
        for route, fit in fits.items()
        if fit.source == "own"
    }

    report: dict[str, Any] = {
        "window_days": args.days,
        "n_routes_fitted": len(fits),
        "n_own": len(log_scale_own),
        "n_pooled": len(fits) - len(log_scale_own),
        "unmapped_routes": unmapped,
    }
    for label, values in (("all_fits", log_scale_all), ("own_only", log_scale_own)):
        if len(values) < 3:
            report[label] = {
                "error": f"only {len(values)} mapped routes, too few to decompose"
            }
            continue
        decomposition = decompose_variance(values)
        p_value = permutation_p_value(values, n_permutations=args.permutations)
        report[label] = {
            "groups": [
                {
                    "division": g.division,
                    "n": g.n,
                    "mean_log_scale": g.mean,
                    "variance": g.variance,
                }
                for g in decomposition.groups
            ],
            "n_total": decomposition.n_total,
            "ms_between": decomposition.ms_between,
            "ms_within": decomposition.ms_within,
            "variance_ratio": decomposition.f_ratio,
            "eta_squared": decomposition.eta_squared,
            "permutation_p_value": p_value,
        }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
