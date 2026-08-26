"""MTA hourly ridership -> per-station-complex entry-rate baseline (training/ridership.py).

The feed is NYS Open Data `5wq4-mkjj` ("MTA Subway Hourly Ridership: Beginning
2025"), a Socrata dataset of hourly (station_complex, fare_class_category,
payment_method) entry counts. It is ENTRY-SIDE ONLY -- there is no `exits`
column, so nothing downstream of this module may treat a cell as anything but
"riders who tapped in that hour". It is keyed by station COMPLEX
(`station_complex_id`, the same id `training/transfers.py` joins against NYS
Open Data `39hk-dx4f`), not by platform or even by individual physical stop --
a complex spanning several lines (Times Sq-42 St) publishes one entry rate for
the whole fare-controlled area, never a per-platform breakdown. The published
feed itself runs roughly 10 days behind real time, which is why this ingest
resolves its own trailing window against the dataset's own latest available
hour (`fetch_latest_hour`) rather than against today.

Run with:
    murk exec -- python -m training.ridership [--days 90] [--end DATE]
        [--no-publish] [--out PATH]

Two server-side aggregated SoQL queries (weekday, weekend) reduce the raw feed
to a (station_complex, hour-of-day) grid without ever pulling a raw row --
`$group=station_complex_id,station_complex,borough,hh` collapses the
fare_class_category/payment_method fan-out for free, since summing across an
un-grouped dimension is exactly what SQL SUM already does. Each grid cell
divides by how many weekday or weekend calendar dates the window actually
covers (`weekday_weekend_day_counts`) to land on entries-per-minute, the same
unit the live crowding estimate (worker/) multiplies by minutes-since-last-
train.

Writes state/ridership_baseline.json (live pointer) + a versioned sibling
under state/ridership_baseline/, both stamped with
`training/provenance.py::code_provenance()`, mirroring
`train_em.py::write_service_baseline`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx

from training.provenance import code_provenance
from training.r2_client import R2Config, load_config, make_client

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

DATASET_ID = "5wq4-mkjj"
FEED_URL = f"https://data.ny.gov/resource/{DATASET_ID}.json"
# Read timeout only, not connect -- these are server-side aggregations over a
# 90-day window (measured 2026-08-23: the weekday query alone takes ~55s
# against the live feed), an order of magnitude past what a static-file GET
# like gtfs_static.fetch_gtfs_zip needs. This runs from a weekly cron job,
# never a request path, so waiting is free; a slow response is not the same
# failure as a hung connection.
FETCH_TIMEOUT = httpx.Timeout(10.0, read=150.0)

BASELINE_KEY = "state/ridership_baseline.json"
# Immutable per-run snapshots live under this prefix as v<generated_at>.json.
VERSIONED_BASELINE_PREFIX = "state/ridership_baseline/"
SCHEMA_VERSION = "1"

# Socrata SODA 2.0's hard ceiling on $limit for an aggregated (GROUP BY)
# query. Both the weekday and weekend query here return at most
# n_complexes * 24 rows -- ~10,176 measured against the live feed for ~424
# subway complexes, comfortably under this. It is an ASSERTION that the real
# answer never quietly hit the ceiling and got truncated, not a paging
# budget: a truncated aggregate would silently bias every cell it touched, so
# fetch_hourly_rows refuses to return a response that reaches it rather than
# paging past it with $offset (paging a GROUP BY changes which groups are
# cut, not how many rows are, so it wouldn't even fix the problem).
QUERY_LIMIT = 50_000

DEFAULT_WINDOW_DAYS = 90


def _headers() -> dict[str, str]:
    """`X-App-Token`, when configured. Every query here works anonymously --
    absence of `DATA_NY_APP_TOKEN` is not an error, only a lower rate limit."""
    token = os.environ.get("DATA_NY_APP_TOKEN")
    return {"X-App-Token": token} if token else {}


def _floating(dt: datetime) -> str:
    """Format a naive datetime as the floating-timestamp string SoQL expects
    for `transit_timestamp` -- no timezone suffix; the feed's timestamps are
    already local Eastern wall-clock, not UTC instants."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S")


def _get_json(url: str, params: dict[str, Any]) -> Any:
    """The one place an HTTP GET happens. Kept separate from the two fetch
    functions below so tests can monkeypatch this instead of the network."""
    with httpx.Client(timeout=FETCH_TIMEOUT, follow_redirects=True) as client:
        response = client.get(url, params=params, headers=_headers())
        response.raise_for_status()
        return response.json()


# The modes we build baselines for. Staten Island Railway rides the same rails
# in our GTFS-RT trace (route 'SI'), so excluding it left every SIR platform
# abstaining with 'no_baseline' while we were happily tracking its trains.
#
# It only recovers TWO complexes, and that is not a bug in this filter: SIR
# collects fares at St George and Tompkinsville only, so those are the sole SIR
# complexes with any rows at all (measured July 2026: St George 249,980
# entries, Tompkinsville 20,481, and nothing anywhere else). The other 19 SIR
# complexes have no entry data in any dataset, and keep abstaining, which is
# the honest answer rather than a modelled one.
#
# The Roosevelt Island tram is deliberately absent: its complex ids ('TRAM1',
# 'TRAM2') join to no GTFS stop and it is not in the subway trace at all, so a
# baseline for it would have nothing to multiply.
TRANSIT_MODES = ("subway", "staten_island_railway")
_MODE_CLAUSE = "transit_mode IN (" + ",".join(f"'{m}'" for m in TRANSIT_MODES) + ")"


def fetch_latest_hour(url: str = FEED_URL) -> str:
    """The newest `transit_timestamp` the feed holds for the modes we baseline,
    as its raw floating string. Anchors the default trailing window to what the
    dataset actually holds rather than to today -- see the module docstring's
    ~10-day lag."""
    rows = _get_json(
        url,
        {
            "$select": "max(transit_timestamp) AS latest",
            "$where": _MODE_CLAUSE,
        },
    )
    if not rows or not rows[0].get("latest"):
        raise RuntimeError(f"{url}: max(transit_timestamp) returned no rows")
    return str(rows[0]["latest"])


def fetch_hourly_rows(
    url: str,
    *,
    window_start: datetime,
    window_end: datetime,
    weekend: bool,
) -> list[dict[str, Any]]:
    """One server-side aggregated SoQL query -- the weekday or weekend half of
    the (station_complex, hour-of-day) grid over the half-open
    [window_start, window_end) span. Raises rather than paging when the
    response reaches QUERY_LIMIT; see its docstring for why."""
    dow_clause = (
        "(date_extract_dow(transit_timestamp) = 0 "
        "OR date_extract_dow(transit_timestamp) = 6)"
        if weekend
        else "date_extract_dow(transit_timestamp) BETWEEN 1 AND 5"
    )
    where = (
        f"{_MODE_CLAUSE} "
        f"AND transit_timestamp >= '{_floating(window_start)}' "
        f"AND transit_timestamp < '{_floating(window_end)}' "
        f"AND {dow_clause}"
    )
    params = {
        "$select": (
            "station_complex_id,station_complex,borough,"
            "date_extract_hh(transit_timestamp) AS hh,"
            "sum(ridership) AS ridership,sum(transfers) AS transfers"
        ),
        "$where": where,
        "$group": "station_complex_id,station_complex,borough,hh",
        "$limit": QUERY_LIMIT,
    }
    rows: list[dict[str, Any]] = _get_json(url, params)
    if len(rows) >= QUERY_LIMIT:
        label = "weekend" if weekend else "weekday"
        raise RuntimeError(
            f"{url}: {label} aggregate returned >= {QUERY_LIMIT} rows -- "
            "Socrata's $limit would silently truncate it; narrow the window "
            "or raise QUERY_LIMIT rather than trust a partial baseline"
        )
    return rows


def resolve_window(
    *, days: int, end: str | None, latest_hour: str
) -> tuple[datetime, datetime]:
    """The half-open [window_start, window_end) span a run trains on. Pure --
    takes the already-discovered `latest_hour` rather than fetching it.

    `window_end` floors to local midnight so a partial day at the tail of the
    feed's publication lag never skews an hourly cell with fewer than a full
    complement of days behind it: `latest_hour + 1h` is the exclusive bound of
    confirmed-complete hours, and truncating that to midnight of its own date
    either lands exactly on the following midnight (when latest_hour is
    23:00, i.e. that whole day is complete) or falls back to the START of
    that day (any earlier hour), dropping the partial day entirely rather
    than averaging it in as if it were whole.

    `end`, given as a YYYY-MM-DD date, overrides the discovered latest_hour
    outright.
    """
    if end:
        window_end = datetime.fromisoformat(end)
    else:
        anchor = datetime.fromisoformat(latest_hour.split(".")[0]) + timedelta(hours=1)
        window_end = datetime(anchor.year, anchor.month, anchor.day)
    window_start = window_end - timedelta(days=days)
    return window_start, window_end


def weekday_weekend_day_counts(
    window_start: datetime, window_end: datetime
) -> tuple[int, int]:
    """(weekday_days, weekend_days): the count of Mon-Fri vs Sat/Sun calendar
    dates in the half-open [window_start, window_end) span. This is the exact
    denominator each entries_per_min cell divides by, so it has to match the
    SoQL `BETWEEN 1 AND 5` / `IN (0, 6)` day-of-week split date for date, not
    just approximate a 5/7 vs 2/7 ratio.

    Python's date.weekday() (Mon=0 .. Sun=6) and Socrata's
    date_extract_dow() (Sun=0 .. Sat=6, Postgres EXTRACT(DOW) convention)
    disagree on WHICH INTEGER a day gets, but agree on the partition:
    weekday() < 5 is exactly Mon-Fri, the same dates dow BETWEEN 1 AND 5
    admits.
    """
    if window_end <= window_start:
        raise ValueError("window_end must be after window_start")
    weekday_days = weekend_days = 0
    day = window_start.date()
    end_day = window_end.date()
    while day < end_day:
        if day.weekday() < 5:
            weekday_days += 1
        else:
            weekend_days += 1
        day += timedelta(days=1)
    return weekday_days, weekend_days


def reduce_baseline(
    weekday_rows: Sequence[Mapping[str, Any]],
    weekend_rows: Sequence[Mapping[str, Any]],
    *,
    weekday_days: int,
    weekend_days: int,
) -> dict[str, dict[str, Any]]:
    """Pure reduction: the two (complex, hour) aggregate row sets ->
    per-complex entries_per_min baseline. No network, no R2 -- exercised
    directly by the tests. The caller wraps the returned `complexes` mapping
    with schema_version/provenance/source into the published document
    (`build_doc`).

    Socrata's aggregate JSON stringifies every number, so every numeric field
    is coerced explicitly. A (complex, hour) cell absent from a response
    stays 0.0 in `entries_per_min` -- not a fabricated observation, just the
    zero a fixed-length 24-slot array needs -- and `n_cells` (of 48) is the
    honest count of how many the feed actually returned, so a consumer can
    tell "quiet cell" from "missing cell".
    """
    if weekday_days <= 0 or weekend_days <= 0:
        raise ValueError("weekday_days and weekend_days must be positive")

    accum: dict[str, dict[str, Any]] = {}

    def _cell(complex_id: str, name: str, borough: str | None) -> dict[str, Any]:
        entry = accum.get(complex_id)
        if entry is None:
            entry = {
                "name": name,
                "borough": borough,
                "wd": [0.0] * 24,
                "we": [0.0] * 24,
                "n_cells": 0,
                "entries_total": 0.0,
                "transfers_total": 0.0,
            }
            accum[complex_id] = entry
        return entry

    def _apply(rows: Sequence[Mapping[str, Any]], cls: str, n_days: int) -> None:
        for row in rows:
            complex_id = str(row["station_complex_id"])
            hour = int(float(row["hh"]))
            if not 0 <= hour < 24:
                raise ValueError(
                    f"complex {complex_id}: date_extract_hh returned {hour!r}, "
                    "expected 0..23"
                )
            ridership = float(row["ridership"])
            transfers = float(row["transfers"])
            borough = row.get("borough")
            entry = _cell(
                complex_id,
                str(row["station_complex"]),
                str(borough) if borough is not None else None,
            )
            entry[cls][hour] = round(ridership / n_days / 60.0, 3)
            entry["n_cells"] += 1
            entry["entries_total"] += ridership
            entry["transfers_total"] += transfers

    _apply(weekday_rows, "wd", weekday_days)
    _apply(weekend_rows, "we", weekend_days)

    ranked = sorted(accum.items(), key=lambda kv: (-kv[1]["entries_total"], kv[0]))
    complexes: dict[str, dict[str, Any]] = {}
    for rank, (complex_id, entry) in enumerate(ranked, start=1):
        complexes[complex_id] = {
            "name": entry["name"],
            "borough": entry["borough"],
            "entries_per_min": {"wd": entry["wd"], "we": entry["we"]},
            "entries_total": round(entry["entries_total"], 1),
            "transfers_total": round(entry["transfers_total"], 1),
            "rank": rank,
            "n_cells": entry["n_cells"],
        }
    return complexes


def build_doc(
    complexes: dict[str, dict[str, Any]],
    *,
    generated_at: int,
    url: str,
    window_start: datetime,
    window_end: datetime,
    latest_hour: str,
    weekday_days: int,
    weekend_days: int,
) -> dict[str, Any]:
    """Wrap a reduce_baseline() result with schema_version/provenance/source
    into the published document shape. Pure -- no network, no R2, so --out
    and --no-publish can produce byte-identical JSON to what --publish
    writes."""
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "provenance": code_provenance(),
        "source": {
            "dataset": DATASET_ID,
            "url": url,
            "transit_mode": ",".join(TRANSIT_MODES),
            "window_start": _floating(window_start),
            "window_end": _floating(window_end),
            "latest_hour": latest_hour,
            "weekday_days": weekday_days,
            "weekend_days": weekend_days,
        },
        "complexes": complexes,
        "n_complexes": len(complexes),
    }


def write_baseline(client: S3Client, bucket: str, doc: dict[str, Any]) -> str:
    """Write the live pointer + an immutable versioned snapshot, mirroring
    train_em.write_service_baseline: same content-type/cache-control
    convention as params.json (read every tick, so it gets that edge-cache
    window rather than service_baseline's no-store), keyed by this run's own
    `generated_at`. Returns the versioned key."""
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
            "Reduce the MTA hourly ridership feed (5wq4-mkjj) to a "
            "per-station-complex entry-rate baseline"
        )
    )
    parser.add_argument("--url", default=FEED_URL)
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help="trailing window size ending at the dataset's own latest hour",
    )
    parser.add_argument(
        "--end",
        help=(
            "window end date YYYY-MM-DD (default: the dataset's own latest "
            "available hour, floored to midnight)"
        ),
    )
    parser.add_argument(
        "--no-publish",
        action="store_true",
        help="compute and print a summary; write nothing",
    )
    parser.add_argument("--out", help="write the artifact JSON to this local path")
    args = parser.parse_args(argv)

    latest_hour = fetch_latest_hour(args.url)
    window_start, window_end = resolve_window(
        days=args.days, end=args.end, latest_hour=latest_hour
    )
    weekday_days, weekend_days = weekday_weekend_day_counts(window_start, window_end)

    weekday_rows = fetch_hourly_rows(
        args.url, window_start=window_start, window_end=window_end, weekend=False
    )
    weekend_rows = fetch_hourly_rows(
        args.url, window_start=window_start, window_end=window_end, weekend=True
    )
    complexes = reduce_baseline(
        weekday_rows,
        weekend_rows,
        weekday_days=weekday_days,
        weekend_days=weekend_days,
    )
    if not complexes:
        print(
            "no rows in the requested window -- refusing to publish an empty baseline",
            file=sys.stderr,
        )
        return 1

    generated_at = int(datetime.now(UTC).timestamp())
    doc = build_doc(
        complexes,
        generated_at=generated_at,
        url=args.url,
        window_start=window_start,
        window_end=window_end,
        latest_hour=latest_hour,
        weekday_days=weekday_days,
        weekend_days=weekend_days,
    )

    busiest_id, busiest = min(complexes.items(), key=lambda kv: kv[1]["rank"])
    peak_rate = max(
        *busiest["entries_per_min"]["wd"], *busiest["entries_per_min"]["we"]
    )
    n_cells = sum(c["n_cells"] for c in complexes.values())
    print(
        f"{len(complexes)} complexes, {n_cells} cells, "
        f"window {_floating(window_start)}..{_floating(window_end)} "
        f"(weekday_days={weekday_days}, weekend_days={weekend_days}), "
        f"busiest {busiest_id} ({busiest['name']}) "
        f"peak {peak_rate:.2f} riders/min"
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
