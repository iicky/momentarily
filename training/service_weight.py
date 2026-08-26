"""GTFS static schedule -> per-directional-platform scheduled-service baseline
(training/service_weight.py).

The live platform-crowding estimate (worker/src/crowding.ts) splits a station
complex's entries/min across the platforms currently in service. It used to
split evenly by platform count, which over-allocates a low-demand platform at a
busy complex -- the 42 St Shuttle at Grand Central published ~950 waiting
riders because it received a full fifth of the complex's entry demand. Demand
tracks how much SERVICE a platform gets far better than it tracks platform
arithmetic, so this baseline gives the Worker a per-directional-platform weight:
how many trains the schedule runs at that exact platform in each
(weekday/weekend, local hour) cell.

Source is the same static GTFS zip the trainer already fetches for its movement
baselines (training/gtfs_static). Keyed by the DIRECTIONAL stop_id (`631N`,
`901S`) -- the same id the realtime trace and stop_times.txt both carry -- so it
joins to `state/station_wait.json`'s platform keys with no crosswalk. A cell is
a raw scheduled-departure count for a representative service day, not a rate:
the Worker only ever consumes RATIOS between a complex's platforms at one (day,
hour) cell, so no per-day normalization is needed.

Which services count as "weekday" vs "weekend" is resolved the same way the rest
of this package reads the calendar -- through gtfs_static.Calendar.active(),
which applies calendar_dates.txt exceptions -- for a representative weekday,
Saturday, and Sunday that the feed actually covers and that carry no holiday
exception. The `we` cell pools Saturday and Sunday, the same weekend pooling
the ridership baseline uses (schedule_bin maps both onto `we`). Unioning
calendar.txt by its weekday flags instead would fold a holiday timetable into
the regular one; a Saturday-service-on-a-Monday holiday is only visible through
calendar_dates, and the reference-day resolution honors it.

The Worker treats a complex/hour as scheduled-split ONLY when every one of its
currently-served platforms carries a positive count here; otherwise it falls
back to the even split for that complex. So a count of 0 (a platform the
schedule does not serve that hour) is meaningful and must be published as 0,
distinct from a stop absent from the document entirely (unknown to the
schedule).

Run with:
    murk exec -- python -m training.service_weight [--no-publish] [--out PATH]

Writes state/service_weight_baseline.json (live pointer) + a versioned sibling
under state/service_weight_baseline/, both stamped with
training/provenance.py::code_provenance(), mirroring training/ridership.py.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from training.gtfs_static import (
    GTFS_STATIC_URL,
    Calendar,
    FeedVersion,
    fetch_gtfs_zip,
    read_calendar,
    read_version,
)
from training.provenance import code_provenance
from training.r2_client import R2Config, load_config, make_client

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

BASELINE_KEY = "state/service_weight_baseline.json"
VERSIONED_BASELINE_PREFIX = "state/service_weight_baseline/"
SCHEMA_VERSION = "1"

# How far past the feed's start to scan for representative service days before
# giving up on the "covered, no holiday exception" preference. A GTFS pick is
# a few months wide, so a plain weekday and Saturday always fall inside this.
_REFERENCE_SCAN_DAYS = 60


def hour_of(departure_time: str) -> int | None:
    """Local hour 0..23 from a GTFS `HH:MM:SS` departure, or None for an
    after-midnight trip (hour >= 24, GTFS's encoding for service running past
    midnight on the prior service day).

    Those trips are EXCLUDED, not wrapped with `% 24`: they run on the NEXT
    wall-clock day, whose weekday/weekend class can differ from the service
    day's (Friday 25:10 runs Saturday 01:10; Sunday 25:10 runs Monday 01:10),
    and a two-bin wd/we artifact looked up by wall clock cannot carry that
    without landing counts in the wrong class. Omitting them can zero a
    platform's overnight cell, which the Worker's full-coverage gate turns
    into the even split for that complex-hour -- the pre-baseline behavior,
    never a wrong-class weight. Day-of-week resolution is the exact fix and
    needs a schema + lookup change.
    """
    head = departure_time.split(":", 1)[0].strip()
    if not head:
        return None
    try:
        hour = int(head)
    except ValueError:
        return None
    return hour if 0 <= hour <= 23 else None


def _has_exception(calendar: Calendar, day: date) -> bool:
    stamp = day.strftime("%Y%m%d")
    return stamp in calendar.added or stamp in calendar.removed


@dataclass(frozen=True)
class ReferenceDays:
    """The concrete service days the two classes are resolved on. `weekday` is
    a plain Mon-Fri; `saturday` and `sunday` are pooled into the `we` class the
    same way the ridership baseline pools weekend entries."""

    weekday: date
    saturday: date
    sunday: date


def select_reference_days(
    feed: FeedVersion, calendar: Calendar, *, start: date | None = None
) -> ReferenceDays:
    """A representative (weekday, Saturday, Sunday) triple to resolve the
    service classes against. Prefers days the feed covers, carrying no
    calendar_dates exception, with a non-empty active service set -- a plain,
    in-service day of each type. Falls back to the first day of each type with
    any active service when the strict scan finds none (a pathologically narrow
    feed window). Both weekend days are resolved because the Worker's `we` cell
    pools Saturday and Sunday, matching the ridership baseline.

    Resolving the classes on concrete days, rather than by calendar.txt weekday
    flags, is what routes the choice through Calendar.active and its
    calendar_dates exceptions -- see the module docstring.
    """
    origin = start or feed.start or datetime.now(UTC).date()

    def _scan(strict: bool) -> dict[int, date]:
        # weekday()==5 Saturday, ==6 Sunday, <5 a weekday. First match of each.
        found: dict[int, date] = {}
        day = origin
        for _ in range(_REFERENCE_SCAN_DAYS):
            active = calendar.active(day)
            ok = bool(active) and (
                not strict or (feed.covers(day) and not _has_exception(calendar, day))
            )
            if ok:
                slot = day.weekday() if day.weekday() >= 5 else 0
                found.setdefault(slot, day)
            if {0, 5, 6} <= found.keys():
                break
            day += timedelta(days=1)
        return found

    found = _scan(strict=True)
    if not ({0, 5, 6} <= found.keys()):
        for slot, day in _scan(strict=False).items():
            found.setdefault(slot, day)
    missing = {0, 5, 6} - found.keys()
    if missing:
        raise ValueError(f"no in-service reference day for weekday slots {missing}")
    return ReferenceDays(weekday=found[0], saturday=found[5], sunday=found[6])


def reduce_service_weights(
    stop_times: Iterable[Mapping[str, Any]],
    trip_service: Mapping[str, str],
    weekday_services: frozenset[str],
    weekend_services: frozenset[str],
) -> dict[str, dict[str, list[int]]]:
    """Pure reduction: stop_times rows -> per-directional-stop scheduled counts.

    Each row is one scheduled departure. A row counts toward `wd`/`we`[hour]
    when its trip's service_id is in the weekday/weekend class and its
    departure_time parses to that local hour. Rows whose trip is in neither
    class, or whose time is unparseable, are skipped. No network, no R2 --
    exercised directly by the tests.

    Every stop that appears in a counted row gets a full 24-slot pair, so a 0
    is an hour the schedule genuinely does not serve that platform, never a
    missing cell. A stop that never appears is simply absent from the result.
    """
    stops: dict[str, dict[str, list[int]]] = {}

    def _cell(stop_id: str) -> dict[str, list[int]]:
        entry = stops.get(stop_id)
        if entry is None:
            entry = {"wd": [0] * 24, "we": [0] * 24}
            stops[stop_id] = entry
        return entry

    for row in stop_times:
        service = trip_service.get(str(row["trip_id"]))
        if service is None:
            continue
        in_wd = service in weekday_services
        in_we = service in weekend_services
        if not (in_wd or in_we):
            continue
        hour = hour_of(str(row["departure_time"]))
        if hour is None:
            continue
        entry = _cell(str(row["stop_id"]))
        if in_wd:
            entry["wd"][hour] += 1
        if in_we:
            entry["we"][hour] += 1

    return stops


def build_doc(
    stops: dict[str, dict[str, list[int]]],
    *,
    generated_at: int,
    url: str,
    feed_version: str,
    reference_weekday: date,
    reference_saturday: date,
    reference_sunday: date,
    weekday_services: Sequence[str],
    weekend_services: Sequence[str],
) -> dict[str, Any]:
    """Wrap a reduce_service_weights() result with schema_version/provenance/
    source into the published document shape. Pure -- no network, no R2, so
    --out and --no-publish produce byte-identical JSON to what --publish
    writes."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "provenance": code_provenance(),
        "source": {
            "dataset": "gtfs_subway",
            "url": url,
            "feed_version": feed_version,
            "reference_weekday": reference_weekday.isoformat(),
            "reference_saturday": reference_saturday.isoformat(),
            "reference_sunday": reference_sunday.isoformat(),
            "weekday_services": sorted(weekday_services),
            "weekend_services": sorted(weekend_services),
        },
        "stops": stops,
        "n_stops": len(stops),
    }


def reduce_from_zip(
    data: bytes,
) -> tuple[dict[str, dict[str, list[int]]], dict[str, Any]]:
    """Parse a GTFS zip into (stops, meta). Resolves the weekday/weekend service
    classes on representative days via the calendar (honoring calendar_dates),
    then counts departures per directional stop. `meta` carries the feed
    version, the reference days, and the resolved service ids for provenance."""
    zf = zipfile.ZipFile(io.BytesIO(data))
    feed = read_version(zf)
    calendar = read_calendar(zf)
    ref = select_reference_days(feed, calendar)
    weekday_services = calendar.active(ref.weekday)
    # Pool both weekend days: the Worker's `we` cell is looked up on Saturday
    # AND Sunday (schedule_bin), so the split weight must reflect the weekend
    # service both days see, the same pooling the ridership `we` rate uses.
    weekend_services = calendar.active(ref.saturday) | calendar.active(ref.sunday)

    with zf.open("trips.txt") as raw:
        trip_service = {
            str(t["trip_id"]): str(t["service_id"])
            for t in csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))
        }

    def _stream_stop_times() -> Iterable[Mapping[str, str]]:
        with zf.open("stop_times.txt") as raw:
            yield from csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8-sig"))

    stops = reduce_service_weights(
        _stream_stop_times(), trip_service, weekday_services, weekend_services
    )
    meta = {
        "feed_version": feed.version,
        "reference_weekday": ref.weekday,
        "reference_saturday": ref.saturday,
        "reference_sunday": ref.sunday,
        "weekday_services": sorted(weekday_services),
        "weekend_services": sorted(weekend_services),
    }
    return stops, meta


def write_baseline(client: S3Client, bucket: str, doc: dict[str, Any]) -> str:
    """Write the live pointer + an immutable versioned snapshot, mirroring
    training/ridership.py::write_baseline: read every tick, so the same
    edge-cache window as params.json/ridership_baseline.json. Returns the
    versioned key."""
    body = json.dumps(doc).encode()
    versioned_key = f"{VERSIONED_BASELINE_PREFIX}v{doc['generated_at']}.json"
    for key in (BASELINE_KEY, versioned_key):
        client.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
            CacheControl="public, max-age=300, s-maxage=900",
        )
    return versioned_key


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Reduce the static GTFS schedule to a per-directional-platform "
            "scheduled-service baseline for the platform-crowding split"
        )
    )
    parser.add_argument("--url", default=GTFS_STATIC_URL)
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="compute and print a summary; write nothing",
    )
    parser.add_argument("--out", help="write the artifact JSON to this local path")
    args = parser.parse_args(argv)

    data = fetch_gtfs_zip(args.url)
    stops, meta = reduce_from_zip(data)
    if not stops:
        print(
            "no scheduled stop_times matched a weekday/weekend service -- "
            "refusing to publish an empty baseline",
            file=sys.stderr,
        )
        return 1

    generated_at = int(datetime.now(UTC).timestamp())
    doc = build_doc(
        stops,
        generated_at=generated_at,
        url=args.url,
        feed_version=meta["feed_version"],
        reference_weekday=meta["reference_weekday"],
        reference_saturday=meta["reference_saturday"],
        reference_sunday=meta["reference_sunday"],
        weekday_services=meta["weekday_services"],
        weekend_services=meta["weekend_services"],
    )

    busiest = max(stops.items(), key=lambda kv: max(kv[1]["wd"]))
    print(
        f"{len(stops)} directional platforms, feed_version={meta['feed_version']}, "
        f"weekday {meta['reference_weekday']} {meta['weekday_services']}, "
        f"weekend {meta['reference_saturday']}/{meta['reference_sunday']} "
        f"{meta['weekend_services']}, "
        f"busiest {busiest[0]} peak {max(busiest[1]['wd'])} trips/hr (wd)"
    )

    if args.out:
        Path(args.out).write_text(json.dumps(doc, indent=2))

    if not args.no_publish:
        cfg: R2Config = load_config()
        client = make_client(cfg)
        versioned_key = write_baseline(client, cfg.bucket, doc)
        print(f"published {BASELINE_KEY} + {versioned_key}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
