"""Join the MTA's official Major Incidents log to our archive — the first
EXTERNAL truth source in this repo.

WHY THIS SOURCE. Every internal signal we grade against — the alert filter, the
movement arm, the supply axis, headway severity — is derived from the same
ATS-sourced GTFS-RT feeds and shares their common-mode failures. When the feed
is blind, all four go blind together, and nothing already in the archive can say
so. The MTA's Major Incidents log is compiled by NYCT from operational records,
not from the public real-time feed, so it is the one reference here whose misses
are independent of ours. That is the whole reason to want it.

WHAT IT ACTUALLY IS — the finding that bounds everything below. The published
dataset (NYS Open Data `ereg-mcvp`, "MTA Subway Major Incidents: Beginning
2015") is NOT an incident log. It is a MONTHLY AGGREGATE: one row per
(month, division, line, day_type, category) carrying a COUNT. There are no
timestamps, no per-incident rows, no line-level onset/clear. Granularity, per
the dataset's own metadata: "Subway line, division, month, day type, delay
category". The sibling `g937-7k7c` ("Delay-Causing Incidents", a superset that
also counts sub-major delays) has the identical monthly-aggregate shape. There
is no incident-level public companion — metrics.mta.info renders these same
aggregates.

CONSEQUENCE, stated plainly so the report cannot overclaim:

  * A month x line x category COUNT cannot be aligned to an archive window by
    timestamp. The join this module achieves is a PREVALENCE ANCHOR (how many
    major incidents the MTA logged on a line in a month, vs. how much severe
    disruption our signals recorded on that line that month), NOT per-episode
    miss adjudication. You cannot ask "did movement fire for THIS incident"
    because the source never names an incident.

  * Worse, for the one signal this was meant to adjudicate — the movement arm —
    there is ZERO temporal overlap. Movement condition and headway severity are
    reconstructed only from archive/trace, which begins 2026-08-12. The dataset's
    last published month is 2026-07. So the movement arm's misses remain
    STRUCTURALLY UNMEASURABLE against this source: not merely at the wrong
    granularity, but with no overlapping day at all. The alert feed and supply
    axis (archive from ~2026-06) do overlap 2026-06 and 2026-07, but they are
    the common-mode signals this external truth was supposed to check.

So this module ingests and normalizes the dataset, maps its line names to our
route ids, classifies how much of each incident-month our archive actually
covers per signal, and — where a signal overlaps — tabulates our severe-episode
prevalence beside the MTA count. It reports the miss-rate as WITHHELD, with the
numeric reason (n episode-alignable incidents = 0), rather than manufacturing a
per-episode number the source cannot support.

THE 2023->2024 METHODOLOGY BREAK. The MTA revised how Major Incidents are
counted between 2023 and 2024; counts are not comparable across that boundary.
Our archive window (2026-06 onward) sits entirely on the post-2024 side, so the
break does not affect the joined months — but `era_for` labels every row so a
longer historical pull can never silently pool across it.

The pure core (line mapping, row parsing, era labelling, month/coverage
classification) is hermetically testable. The R2-touching prevalence read lives
behind `main()` and is never exercised by the test suite.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from training.divisions import ALL_KNOWN_ROUTES

# --- Dataset identity ---------------------------------------------------------

MAJOR_INCIDENTS_DATASET = "ereg-mcvp"  # MTA Subway Major Incidents: Beginning 2015
DELAY_CAUSING_DATASET = "g937-7k7c"  # superset (sub-major too); same monthly shape
_RESOURCE_URL = "https://data.ny.gov/resource/{dataset}.json"

# The six expert categories, verbatim from the dataset. A Major Incident is one
# that delays 50+ trains; these are the causes NYCT files it under.
SIX_CATEGORIES: frozenset[str] = frozenset(
    {
        "Signals",
        "Track",
        "Persons on Trackbed/Police/Medical",
        "Subway Car",
        "Stations and Structure",
        "Other",
    }
)

# day_type is coded 1 = weekday, 2 = weekend (dataset metadata).
DAY_TYPE_LABEL: dict[str, str] = {"1": "weekday", "2": "weekend"}

# The MTA changed its Major-Incident counting methodology between 2023 and 2024;
# counts do not compare across this boundary. See module docstring.
METHODOLOGY_BREAK = date(2024, 1, 1)


def era_for(month: date) -> str:
    """Which side of the 2023->2024 methodology break a month falls on."""
    return "post_2024" if month >= METHODOLOGY_BREAK else "pre_2024"


# --- Line -> route-id mapping -------------------------------------------------
#
# The dataset's `line` vocabulary is not quite our route-id space. Three shuttles
# carry descriptive names and one entry ("JZ") bundles two routes the MTA runs as
# one skip-stop service. Everything else is an identity map onto ALL_KNOWN_ROUTES.
# These are the join's line-attribution blind spots, documented as data:
#
#   - "JZ" -> J and Z: the MTA counts the Nassau St skip-stop pair as one line, so
#     a JZ count cannot be attributed to J vs Z. We fan it out to BOTH; a per-route
#     comparison therefore double-counts JZ incidents across the pair (flagged).
#   - "S 42nd" -> GS (Times Sq-Grand Central 42nd St Shuttle).
#   - "S Rock" -> H  (Rockaway Park Shuttle).
#   - "S Fkln" -> FS (Franklin Avenue Shuttle).
#
# The dataset has no Staten Island Railway ("SI") — it is out of scope for Major
# Incidents — and no express variants (6X/7X/FX): those are our internal route
# ids, folded into their parent line here.
LINE_TO_ROUTES: dict[str, tuple[str, ...]] = {
    "JZ": ("J", "Z"),
    "S 42nd": ("GS",),
    "S Rock": ("H",),
    "S Fkln": ("FS",),
}

# Lines that fan out to more than one route id — a per-route join cannot
# attribute their counts to a single route. Reported as an explicit blind spot.
AMBIGUOUS_LINES: frozenset[str] = frozenset(
    line for line, routes in LINE_TO_ROUTES.items() if len(routes) > 1
)


def map_line(line: str | None) -> tuple[str, ...]:
    """MTA `line` value -> our route id(s). Empty tuple for an unmappable value
    (the dataset's occasional null row, or a line we do not model). Raises for a
    mapped route id that is not in ALL_KNOWN_ROUTES, so a feed change surfaces
    rather than silently dropping incidents."""
    if line is None:
        return ()
    line = line.strip()
    if not line:
        return ()
    routes = LINE_TO_ROUTES.get(line, (line,) if line in ALL_KNOWN_ROUTES else ())
    unknown = [r for r in routes if r not in ALL_KNOWN_ROUTES]
    if unknown:
        raise ValueError(f"line {line!r} maps to unknown route ids {unknown}")
    return routes


# --- Normalized rows ----------------------------------------------------------


@dataclass(frozen=True)
class IncidentCount:
    """One monthly-aggregate cell of the Major Incidents log.

    `routes` is the mapped route-id tuple (see map_line); a multi-route tuple
    means the source line is ambiguous (JZ) and this count applies to the pair,
    not to one route.
    """

    month: date
    division: str
    line: str
    routes: tuple[str, ...]
    day_type: str  # "weekday" | "weekend"
    category: str
    count: int

    @property
    def era(self) -> str:
        return era_for(self.month)

    @property
    def ambiguous_line(self) -> bool:
        return self.line in AMBIGUOUS_LINES


def _parse_month(raw: str) -> date:
    """Socrata floating-timestamp ('2026-07-01T00:00:00.000') -> first-of-month."""
    return datetime.fromisoformat(raw.split("T")[0]).date().replace(day=1)


def parse_rows(raw: Iterable[Mapping[str, object]]) -> list[IncidentCount]:
    """Normalize raw Socrata rows into IncidentCount, skipping unmappable/null
    rows (a `line`/`count` that does not resolve). Never raises on a null cell —
    the dataset carries an occasional all-null row — but does raise via map_line
    on a real line that maps to an unknown route id."""
    out: list[IncidentCount] = []
    for r in raw:
        line = r.get("line")
        month = r.get("month")
        count = r.get("count")
        category = r.get("category")
        if line is None or month is None or count is None or category is None:
            continue
        routes = map_line(str(line))
        if not routes:
            continue
        out.append(
            IncidentCount(
                month=_parse_month(str(month)),
                division=str(r.get("division", "")).strip(),
                line=str(line).strip(),
                routes=routes,
                day_type=DAY_TYPE_LABEL.get(
                    str(r.get("day_type", "")).strip(), "unknown"
                ),
                category=str(category).strip(),
                count=int(str(count)),
            )
        )
    return out


# --- Archive coverage classification -----------------------------------------


@dataclass(frozen=True)
class SignalBounds:
    """First and last archived day for one signal's substrate (inclusive)."""

    name: str
    first: date
    last: date


def month_span(month: date) -> tuple[date, date]:
    """First and last calendar day of `month` (month is first-of-month)."""
    first = month.replace(day=1)
    nxt = (first + timedelta(days=32)).replace(day=1)
    return first, nxt - timedelta(days=1)


def classify_coverage(month: date, bounds: SignalBounds) -> str:
    """How much of an incident-month a signal's archive covers.

    'full'    — the signal's archive spans the whole calendar month.
    'partial' — it covers some but not all days (an aggregate comparison over
                this month undercounts our side).
    'none'    — no overlapping day: the signal cannot speak to this month at all.
    """
    m_first, m_last = month_span(month)
    if bounds.last < m_first or bounds.first > m_last:
        return "none"
    if bounds.first <= m_first and bounds.last >= m_last:
        return "full"
    return "partial"


# --- I/O: fetch ---------------------------------------------------------------

_FETCH_TIMEOUT = httpx.Timeout(10.0, read=60.0)


def fetch_incidents(
    dataset: str = MAJOR_INCIDENTS_DATASET, *, limit: int = 50000
) -> list[Mapping[str, object]]:
    """Pull the full monthly-aggregate table (no credentials — public Socrata).
    The table is a few thousand rows, so one paged request suffices."""
    url = _RESOURCE_URL.format(dataset=dataset)
    with httpx.Client(timeout=_FETCH_TIMEOUT) as client:
        resp = client.get(url, params={"$limit": limit, "$order": "month"})
        resp.raise_for_status()
        return list(resp.json())


# --- I/O: our severe-condition prevalence from the archive --------------------


def severe_route_days(start: date, end: date) -> dict[str, int]:
    """Per route, the number of ET-days over [start, end] on which the canonical
    severe-only truth (tier>=2: Severe Delays / suspension, planned work excluded)
    was active at any tick. Reuses the review's own truth builder
    (`mta_truth`, severity_floor = CANONICAL_SEVERITY_FLOOR).

    WHY ROUTE-DAYS, NOT EPISODES. A severe-EPISODE count needs the predictions
    presence-mask to make open-ended alert tails close; without it (there is no
    predictions stream to mask against for these pre-trace months) every
    unclosed severe alert runs to the window end and is dropped as a >24h
    'standing advisory', collapsing the count toward zero and MEASURING THE MASK,
    NOT THE FEED. A route-day is immune: it asks only whether a severe condition
    was present that day, which needs no episode closure and no mask. It is a
    prevalence measure (how many route-days carried severe disruption), NOT an
    incident count — the units differ from the MTA's per-incident count and the
    two must not be subtracted, only read side by side. R2 read; not unit-tested.
    """
    from training.r2_client import load_config, make_client
    from training.review import load_truth_observations, mta_truth

    cfg = load_config()
    client = make_client(cfg)
    obs = load_truth_observations(client, cfg.bucket, start, end)
    truth = mta_truth(obs)  # canonical severe-only
    route_days: dict[str, set[date]] = {}
    for (route, tick), state in truth.items():
        if state == "normal":
            continue
        day = datetime.fromtimestamp(tick, UTC).date()
        route_days.setdefault(route, set()).add(day)
    return {route: len(days) for route, days in route_days.items()}


# --- Join + report ------------------------------------------------------------


@dataclass(frozen=True)
class JoinedMonth:
    """One incident-month, its MTA major counts by route, and per-signal coverage."""

    month: date
    mta_by_route: dict[str, int]  # route -> major-incident count that month
    mta_by_category: dict[str, int]
    coverage: dict[str, str]  # signal name -> full|partial|none
    our_severe_route_days: dict[str, int] = field(
        default_factory=lambda: dict[str, int]()
    )


# Archive substrate bounds, probed from R2 on 2026-09-01 (see journal). The
# movement arm and headway ride archive/trace; the alert feed and supply axis
# reach back to ~2026-06. Kept as data so `main()` can re-probe and warn on drift.
DEFAULT_BOUNDS: tuple[SignalBounds, ...] = (
    SignalBounds("alerts", date(2026, 6, 3), date(2026, 9, 2)),
    SignalBounds("supply", date(2026, 6, 15), date(2026, 9, 2)),
    SignalBounds("movement", date(2026, 8, 12), date(2026, 9, 2)),
    SignalBounds("headway", date(2026, 8, 12), date(2026, 9, 2)),
)


def join(
    incidents: Sequence[IncidentCount],
    bounds: Sequence[SignalBounds] = DEFAULT_BOUNDS,
    *,
    months: Sequence[date] | None = None,
) -> list[JoinedMonth]:
    """Aggregate MTA counts to month x route and month x category, and classify
    each signal's coverage of each month. Pure — the archive-prevalence field is
    filled by the caller (main) where a signal overlaps."""
    if months is None:
        months = sorted({inc.month for inc in incidents})
    out: list[JoinedMonth] = []
    for month in months:
        rows = [inc for inc in incidents if inc.month == month]
        by_route: dict[str, int] = {}
        by_cat: dict[str, int] = {}
        for inc in rows:
            by_cat[inc.category] = by_cat.get(inc.category, 0) + inc.count
            for route in inc.routes:  # JZ fans to J and Z (double-counts, flagged)
                by_route[route] = by_route.get(route, 0) + inc.count
        coverage = {b.name: classify_coverage(month, b) for b in bounds}
        out.append(
            JoinedMonth(
                month=month,
                mta_by_route=by_route,
                mta_by_category=by_cat,
                coverage=coverage,
            )
        )
    return out


def _overlapping_incident_months(
    incidents: Sequence[IncidentCount], bounds: Sequence[SignalBounds]
) -> dict[str, list[date]]:
    """Per signal, the incident-months it covers at all (partial or full)."""
    months = sorted({inc.month for inc in incidents})
    out: dict[str, list[date]] = {}
    for b in bounds:
        out[b.name] = [m for m in months if classify_coverage(m, b) != "none"]
    return out


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-archive",
        action="store_true",
        help="skip the R2 severe-episode read; ingest + coverage only",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON, not text")
    args = parser.parse_args(argv)

    raw = fetch_incidents()
    incidents = parse_rows(raw)
    months = sorted({inc.month for inc in incidents})
    joined = join(incidents)

    overlap = _overlapping_incident_months(incidents, DEFAULT_BOUNDS)

    # The joinable months for the alert feed: the only signal that overlaps AND
    # is not blocked by the trace start. Movement/headway overlap is 0 by design.
    alert_months = overlap["alerts"]
    our_severe: dict[date, dict[str, int]] = {}
    if not args.no_archive and alert_months:
        for month in alert_months:
            m_first, m_last = month_span(month)
            b = next(x for x in DEFAULT_BOUNDS if x.name == "alerts")
            lo = max(m_first, b.first)
            hi = min(m_last, b.last)
            our_severe[month] = severe_route_days(lo, hi)

    joined_entries: list[dict[str, Any]] = []
    for jm in joined:
        entry: dict[str, Any] = {
            "month": jm.month.isoformat(),
            "era": era_for(jm.month),
            "coverage": jm.coverage,
            "mta_major_total": sum(jm.mta_by_category.values()),
            "mta_by_category": jm.mta_by_category,
        }
        if jm.month in our_severe:
            sev = our_severe[jm.month]
            entry["mta_by_route"] = jm.mta_by_route
            entry["our_severe_alert_route_days_by_route"] = sev
            entry["our_severe_alert_route_days_total"] = sum(sev.values())
        joined_entries.append(entry)

    report: dict[str, Any] = {
        "dataset": {
            "id": MAJOR_INCIDENTS_DATASET,
            "granularity": "monthly aggregate: month x division x line x day_type x category (COUNT)",
            "incident_level": False,
            "months": [m.isoformat() for m in months],
            "n_rows": len(incidents),
            "methodology_break": METHODOLOGY_BREAK.isoformat(),
            "window_era": sorted({era_for(m) for m in months if m >= date(2026, 1, 1)}),
        },
        "coverage_overlap_months": {
            name: [m.isoformat() for m in ms] for name, ms in overlap.items()
        },
        "ambiguous_lines": sorted(AMBIGUOUS_LINES),
        "miss_rate": {
            "movement_arm": "WITHHELD",
            "reason": "0 episode-alignable incidents: source is monthly aggregate "
            "(no timestamps) AND movement/headway trace begins 2026-08-12, after "
            "the last published incident month (2026-07). n_incidents on a per-"
            "episode basis = 0 on both sides; interval withheld per convention.",
        },
        "joined_months": joined_entries,
    }

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return 0

    _print_text(report)
    return 0


def _print_text(report: dict[str, Any]) -> None:
    d = report["dataset"]  # type: ignore[index]
    print("MTA Major Incidents -> archive join")
    print(f"  dataset {d['id']}: {d['granularity']}")
    print(f"  incident-level: {d['incident_level']}  rows: {d['n_rows']}")
    print(f"  months: {d['months'][0]} .. {d['months'][-1]}  (era: {d['window_era']})")
    print(f"  2023->2024 methodology break at {d['methodology_break']}")
    print()
    print("coverage — incident-months each signal overlaps at all:")
    for name, ms in report["coverage_overlap_months"].items():  # type: ignore[union-attr]
        span = f"{ms[0]}..{ms[-1]}" if ms else "NONE"
        print(f"  {name:9s}: {len(ms):2d} months  {span}")
    print()
    mr = report["miss_rate"]  # type: ignore[index]
    print(f"movement-arm miss rate: {mr['movement_arm']}")
    print(f"  {mr['reason']}")
    print()
    print("joined months (MTA majors vs our severe alert route-days):")
    for jm in report["joined_months"]:  # type: ignore[union-attr]
        cov = ",".join(f"{k}={v}" for k, v in jm["coverage"].items())
        line = f"  {jm['month']}  MTA majors={jm['mta_major_total']:3d}  [{cov}]"
        if "our_severe_alert_route_days_total" in jm:
            line += f"  our severe alert route-days={jm['our_severe_alert_route_days_total']}"
        print(line)
        print(f"      by category: {jm['mta_by_category']}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
