"""Per-station maintenance scorecard from our own R2 archives (training/station_maintenance.py).

WHAT IT PRODUCES. One static sidecar, `viz/public/station_maintenance.json`,
keyed by GTFS parent stop id (the same key the Station page and
`station_facts.json` use), carrying for each station:

    elevator_outages / escalator_outages  — distinct equipment outages that were
                                             active at any point THIS MONTH.
    median_repair_hours                    — median time-to-repair over the
                                             outages that RESOLVED this month
                                             (null when none did).
    resolved_outages                       — how many outages back that median.
    planned_closures                       — announced planned-work closures
                                             touching the stop THIS YEAR.

WHY A COMMITTED STATIC FILE, NOT AN R2 BASELINE. The ridership and
service-weight baselines are read every tick by the Worker's inference path, so
they publish to R2 with a live pointer + a versioned copy. This sidecar is the
opposite: it is derived history the Station page renders, never an inference
input, and it is small (a few hundred stations x five small numbers, tens of
KB). The viz already reads exactly this class of slowly-changing derived
reference data as a committed asset fetched once and cached
(`station_facts.json`, `diagram.json`) — this joins them. Publishing it to R2
`state/` instead would put a new object beside the params the eval window
depends on and require a new public read path through the Worker; a committed
asset touches none of the inference surface. The slow cron regenerates and
commits this file, the same lifecycle `scripts/gen_station_facts.py` has.

READ-ONLY. Every byte read here comes from the immutable `archive/` prefixes
(`archive/ene/` for the hourly elevator/escalator feed, `archive/windows/` for
the parsed planned-work answer key). Nothing under `state/` or `params` is read
or written, and no new collection runs — this reduces what the Worker already
archived.

Run:  uv run python -m training.station_maintenance
      uv run python -m training.station_maintenance --dry-run
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from zoneinfo import ZoneInfo

from momentarily.ene import parse_feed_payload
from training.load_r2 import fetch_objects, list_keys
from training.planned_work import Window
from training.r2_client import R2Config, load_config, make_client
from training.window_archive import read_days

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

    from momentarily.schema import Equipment

# New York wall time. A calendar month and a calendar year are wall-clock facts
# and the city observes DST, so the window boundaries are anchored in ET rather
# than UTC — an outage "this month" is judged against local months.
ET = ZoneInfo("America/New_York")

ENE_PREFIX = "archive/ene/"
WINDOWS_PREFIX = "archive/windows/"
# One object per (hour, source); we only need the active-outage feed here. The
# equipment catalog (…-ene_equipments.json) is read separately for the
# equipment -> GTFS-stop map.
CURRENT_SUFFIX = "-ene_current.json"
CATALOG_SUFFIX = "-ene_equipments.json"

SCHEMA_VERSION = "1"

# A single missed hourly fetch must not split one outage into two, nor read as a
# repair. An equipment that vanishes for longer than this and returns is treated
# as genuinely cleared and re-broken; the MTA-reported `since` re-keys it as a
# new outage anyway (see reconstruct_episodes), so this only governs the
# resolved/ongoing call at the tail of the window.
GAP_SECONDS = 3 * 3600

# The planned-work types that actually remove service at the stops they name:
# a full or partial suspension, or a station skipped. Reroutes, express-to-local
# and extra-transfer notices keep the station served (the trains just take a
# different path or make different stops elsewhere), so they are announced work
# but not a closure of the named stop and are deliberately excluded. The
# archive's window rows carry the real-time counterparts of these types too
# ("Suspended" vs "Planned - Suspended"); only the planned ones count here.
CLOSURE_TYPES = frozenset(
    {
        "Planned - Suspended",
        "Planned - Part Suspended",
        "Planned - Stops Skipped",
    }
)

# training/ sits at the repo root, so parent.parent is the checkout root.
DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent
    / "viz"
    / "public"
    / "station_maintenance.json"
)


@dataclass
class Episode:
    """One elevator/escalator outage, reconstructed from the hourly feed.

    Keyed while building by (equipment_id, since): the MTA publishes the outage
    start (`since`) on every hourly record, so two observations that share it are
    the same outage even across a gap in our collection, and a re-broken unit
    gets a new `since` and a new episode. `first_seen`/`last_seen` are the bounds
    of when we actually observed it out.
    """

    equipment_id: str
    kind: str  # "elevator" | "escalator"
    since: int | None
    first_seen: int
    last_seen: int
    reason: str | None
    est_return: int | None


def _empty_hours() -> list[float]:
    return []


@dataclass
class StopOutages:
    """The month's outage tally accruing to one GTFS stop."""

    elevator: int = 0
    escalator: int = 0
    repair_hours: list[float] = field(default_factory=_empty_hours)


def list_day_prefixes(client: S3Client, bucket: str, prefix: str) -> list[date]:
    """Every YYYY-MM-DD day partition present under `prefix`, ascending.

    A delimiter listing so it returns ~one entry per archived day rather than
    every object, used to find the archive's coverage bounds cheaply.
    """
    days: list[date] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/"):
        for common in page.get("CommonPrefixes") or []:
            key = common.get("Prefix")
            if not key:
                continue
            segment = key.rstrip("/").rsplit("/", 1)[-1]
            try:
                days.append(date.fromisoformat(segment))
            except ValueError:
                continue
    return sorted(days)


def current_feed_keys(client: S3Client, bucket: str, days: Iterable[date]) -> list[str]:
    """Every archived active-outage snapshot key over `days`, in list order."""
    keys: list[str] = []
    for day in days:
        keys.extend(
            k
            for k in list_keys(client, bucket, f"{ENE_PREFIX}{day.isoformat()}/")
            if k.endswith(CURRENT_SUFFIX)
        )
    return keys


def latest_catalog_key(
    client: S3Client, bucket: str, days: Sequence[date]
) -> str | None:
    """The newest equipment-catalog snapshot key, walking days from the most
    recent. The catalog barely changes day to day, so the latest copy is the
    right equipment -> GTFS-stop map for the whole window."""
    for day in reversed(days):
        catalogs = sorted(
            k
            for k in list_keys(client, bucket, f"{ENE_PREFIX}{day.isoformat()}/")
            if k.endswith(CATALOG_SUFFIX)
        )
        if catalogs:
            return catalogs[-1]
    return None


def equipment_stop_map(catalog_payload: Any) -> dict[str, list[str]]:
    """equipment_id -> the GTFS parent stop ids it serves.

    `elevatorsgtfsstopid` is either a single stop id ("L06") or, for a unit that
    serves a multi-stop complex, a "/"-joined list ("112/A09"). Splitting it
    attributes an outage to every stop of the complex it belongs to — the same
    complex-level scope the live status block on the Station page shows, so the
    history reads consistently beneath it. Units with no GTFS id (non-NYCT
    equipment) map to nothing and their outages are dropped.
    """
    out: dict[str, list[str]] = {}
    if not isinstance(catalog_payload, list):
        return out
    for raw in cast(list[Any], catalog_payload):
        if not isinstance(raw, dict):
            continue
        record = cast(dict[str, Any], raw)
        equipment_id = record.get("equipmentno")
        gtfs = record.get("elevatorsgtfsstopid")
        if not equipment_id or not gtfs:
            continue
        stops = [s for s in str(gtfs).split("/") if s]
        if stops:
            out[str(equipment_id)] = stops
    return out


def _active_records(payload: Any) -> list[Any]:
    """The current-outage records in one snapshot payload, dropping any flagged
    as an upcoming (scheduled future) outage so it never counts as active."""
    if not isinstance(payload, list):
        return []
    records = cast(list[Any], payload)
    return [
        record
        for record in records
        if not (
            isinstance(record, dict)
            and cast(dict[str, Any], record).get("isupcomingoutage") == "Y"
        )
    ]


def reconstruct_episodes(
    snapshots: Sequence[tuple[int, Sequence[Equipment]]],
) -> list[Episode]:
    """Fold the ordered hourly snapshots into one Episode per outage.

    Pure: no network. `snapshots` is (observed_at, parsed active-outage list),
    which the reduction can build from fixtures. Episodes are keyed by
    (equipment_id, since) so a persistent outage stays one episode across a
    collection gap and a re-broken unit opens a new one.
    """
    episodes: dict[tuple[str, int | None], Episode] = {}
    for observed_at, equipment in sorted(snapshots, key=lambda s: s[0]):
        for unit in equipment:
            if unit.outage is None:
                continue
            since = unit.outage.since
            key = (unit.equipment_id, since)
            existing = episodes.get(key)
            if existing is None:
                episodes[key] = Episode(
                    equipment_id=unit.equipment_id,
                    kind=unit.type,
                    since=since,
                    first_seen=observed_at,
                    last_seen=observed_at,
                    reason=unit.outage.reason,
                    est_return=unit.outage.est_return,
                )
            else:
                existing.last_seen = max(existing.last_seen, observed_at)
                existing.first_seen = min(existing.first_seen, observed_at)
    return list(episodes.values())


def reduce_outages(
    episodes: Iterable[Episode],
    eq_stops: dict[str, list[str]],
    *,
    last_observed: int,
    gap: int = GAP_SECONDS,
) -> dict[str, StopOutages]:
    """Tally the month's outages per GTFS stop.

    Every episode here was observed inside the fetched month window, so each
    counts toward its stops' month tally. An episode is RESOLVED when its last
    sighting predates the final snapshot by more than `gap` — i.e. we kept
    collecting and it was gone — and only resolved episodes contribute a
    time-to-repair, measured from the MTA-reported `since` (falling back to first
    sighting) to the last time we saw it out. A still-open outage counts as an
    outage but carries no repair time.
    """
    per_stop: dict[str, StopOutages] = {}
    for episode in episodes:
        stops = eq_stops.get(episode.equipment_id)
        if not stops:
            continue
        resolved = episode.last_seen < last_observed - gap
        repair_hours: float | None = None
        if resolved:
            start = episode.since if episode.since is not None else episode.first_seen
            delta = episode.last_seen - start
            if delta >= 0:
                repair_hours = delta / 3600.0
        for stop in stops:
            tally = per_stop.setdefault(stop, StopOutages())
            if episode.kind == "elevator":
                tally.elevator += 1
            else:
                tally.escalator += 1
            if repair_hours is not None:
                tally.repair_hours.append(repair_hours)
    return per_stop


def reduce_closures(
    windows: Iterable[Window],
    *,
    year_start: int,
    now: int,
) -> dict[str, int]:
    """Count planned-work closures per GTFS stop over [year_start, now].

    `windows` is the de-duplicated answer key from the window archive; each names
    the parent stops it affects. A closure counts against a stop when its type is
    service-removing (CLOSURE_TYPES) and its announced active period overlaps the
    year so far. An open-ended window (end == 0) that has started overlaps.
    """
    per_stop: dict[str, int] = {}
    for window in windows:
        if window.alert_type not in CLOSURE_TYPES:
            continue
        if window.start > now:
            continue
        if window.end and window.end < year_start:
            continue
        for stop in window.stops:
            per_stop[stop] = per_stop.get(stop, 0) + 1
    return per_stop


def build_doc(
    outages: dict[str, StopOutages],
    closures: dict[str, int],
    *,
    generated_at: int,
    month_start: int,
    year_start: int,
    window_end: int,
    ene_coverage: tuple[date, date] | None,
    windows_coverage: tuple[date, date] | None,
) -> dict[str, Any]:
    """Assemble the published document. Pure — no network, no clock — so
    --dry-run prints exactly what a write would commit."""
    stations: dict[str, dict[str, Any]] = {}
    for stop in set(outages) | set(closures):
        tally = outages.get(stop)
        repair = tally.repair_hours if tally else []
        stations[stop] = {
            "elevator_outages": tally.elevator if tally else 0,
            "escalator_outages": tally.escalator if tally else 0,
            "resolved_outages": len(repair),
            "median_repair_hours": (
                round(statistics.median(repair), 1) if repair else None
            ),
            "planned_closures": closures.get(stop, 0),
        }

    def _iso(epoch: int) -> str:
        return datetime.fromtimestamp(epoch, ET).isoformat()

    def _cov(bounds: tuple[date, date] | None) -> dict[str, str] | None:
        if bounds is None:
            return None
        return {"start": bounds[0].isoformat(), "end": bounds[1].isoformat()}

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "source": {
            "ene": ENE_PREFIX,
            "windows": WINDOWS_PREFIX,
        },
        # The windows the numbers describe. `coverage` is what the archive
        # actually holds: our collection began in May 2026, so "this year" is
        # clipped to the earliest archived day, and a reader must not mistake a
        # short archive for a quiet station.
        "outage_window": {"start": _iso(month_start), "end": _iso(window_end)},
        "closure_window": {"start": _iso(year_start), "end": _iso(window_end)},
        "coverage": {"ene": _cov(ene_coverage), "windows": _cov(windows_coverage)},
        "stations": stations,
        "n_stations": len(stations),
    }


def _month_start(now: datetime) -> int:
    local = now.astimezone(ET)
    return int(datetime(local.year, local.month, 1, tzinfo=ET).timestamp())


def _year_start(now: datetime) -> int:
    local = now.astimezone(ET)
    return int(datetime(local.year, 1, 1, tzinfo=ET).timestamp())


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate the hourly E&E archive and the planned-work answer key "
            "into the committed per-station maintenance sidecar"
        )
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="path to write the sidecar JSON (default: viz/public/station_maintenance.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and print a summary; write nothing",
    )
    args = parser.parse_args(argv)

    cfg: R2Config = load_config()
    client = make_client(cfg)

    now = datetime.now(UTC)
    now_epoch = int(now.timestamp())
    month_start = _month_start(now)
    year_start = _year_start(now)
    month_start_day = datetime.fromtimestamp(month_start, ET).date()
    year_start_day = datetime.fromtimestamp(year_start, ET).date()
    today = now.astimezone(ET).date()

    ene_days = list_day_prefixes(client, cfg.bucket, ENE_PREFIX)
    if not ene_days:
        print("no archive/ene/ days present — nothing to reduce", file=sys.stderr)
        return 1
    ene_coverage = (ene_days[0], ene_days[-1])
    month_days = [d for d in ene_days if d >= month_start_day]

    snapshots: list[tuple[int, Sequence[Equipment]]] = []
    current_keys = current_feed_keys(client, cfg.bucket, month_days)
    for body in fetch_objects(client, cfg.bucket, current_keys):
        observed_at = int(body.get("observed_at") or 0)
        if not observed_at:
            continue
        # A record flagged upcoming is a scheduled future outage, not a current
        # one; drop it before parsing so it never counts as this month's outage.
        active = _active_records(body.get("payload"))
        snapshots.append((observed_at, parse_feed_payload(active)))

    if not snapshots:
        print(
            "no active-outage snapshots this month — nothing to reduce", file=sys.stderr
        )
        return 1
    last_observed = max(observed_at for observed_at, _ in snapshots)

    catalog_key = latest_catalog_key(client, cfg.bucket, ene_days)
    if catalog_key is None:
        print(
            "no equipment catalog in the archive — cannot map outages to stops",
            file=sys.stderr,
        )
        return 1
    catalog = fetch_objects(client, cfg.bucket, [catalog_key])[0]
    eq_stops = equipment_stop_map(catalog.get("payload"))

    episodes = reconstruct_episodes(snapshots)
    outages = reduce_outages(episodes, eq_stops, last_observed=last_observed)

    windows_result = read_days(year_start_day, today, config=cfg, client=client)
    windows_coverage = (
        (min(windows_result.provenance), max(windows_result.provenance))
        if windows_result.provenance
        else None
    )
    closures = reduce_closures(
        windows_result.windows, year_start=year_start, now=now_epoch
    )

    doc = build_doc(
        outages,
        closures,
        generated_at=now_epoch,
        month_start=month_start,
        year_start=year_start,
        window_end=now_epoch,
        ene_coverage=ene_coverage,
        windows_coverage=windows_coverage,
    )

    n_outages = sum(t.elevator + t.escalator for t in outages.values())
    n_closures = sum(closures.values())
    print(
        f"{doc['n_stations']} stations, "
        f"{n_outages} equipment-outage tallies over {len(episodes)} episodes "
        f"(month from {month_start_day.isoformat()}), "
        f"{n_closures} planned-closure tallies (year from {year_start_day.isoformat()}); "
        f"ene coverage {ene_coverage[0]}..{ene_coverage[1]}",
        file=sys.stderr,
    )

    if args.dry_run:
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
