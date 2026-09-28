"""Journey spans chained from per-trip traversals.

The measurement: one SpanSample per contiguous window of a trip's observed
traversals — the trip's arrival at `origin` to its arrival at `destination`,
covering 1..MAX_HOPS realtime hops. A chain continues only through EXACT or
INTERVAL rows whose stops abut (row k's to_stop is row k+1's from_stop);
a RIGHT-censored row, a gap, or a row missing direction/to_stop/n_hops breaks
it, and no span crosses a break. `seconds` is the sum of arrival-to-arrival
times, so it includes the dwell at origin — the same quantity the traversal
baselines fit.

`scheduled_sec` is the sum, over the window's OBSERVED hops, of what the
timetable allowed each one, resolved exactly as traversal.hop_samples resolves
a hop's scheduled_sec (trace.scheduled_for: the trip's own stopping pattern
first, the service day's per-hop median as fallback, None outside the feed
window). One unresolvable hop makes the whole span's scheduled_sec None. This
deliberately follows the trip's OWN observed hops, never a modal chain — an
earlier model's known flaw that must not recur.

A scheduled number must also come from the schedule that was IN FORCE: when
`verified_days` is given, a hop whose service day is not in it gets None. The
loader builds that set by hashing the very bytes it parses into the Timetable
and keeping only archive days whose provenance carries that digest — an
unstamped or mid-day-mixed day is not verified. Realized seconds never depend
on schedule bytes and are kept for every day.

`service_day` comes from gtfs_static.service_dates on the first hop's arrival:
nearest candidate by the trip id's own scheduled origin, the rule the live
pipeline already uses. No midnight rule is invented here.

JSONL layout per training/journey_types.py: one JSON object per line with
exactly the SpanSample field names; `.gz` means gzip.
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import statistics
import sys
import zipfile
from collections import Counter, defaultdict
from collections.abc import Collection, Iterator, Sequence
from dataclasses import asdict
from datetime import date
from typing import IO, TYPE_CHECKING, Literal

from training.gtfs_archive import digest_of
from training.gtfs_static import Timetable, fetch_gtfs_zip, service_dates, timetable
from training.journey_types import MAX_HOPS, SpanSample
from training.trace import EXACT, INTERVAL, Traversal, scheduled_for
from training.traversal_archive import ReadResult, read_days

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

    from training.load_r2 import R2Config


def _eligible(t: Traversal) -> bool:
    """Whether a row may sit inside a chain at all. RIGHT-censored rows have no
    destination and no duration to their next arrival; rows without direction or
    n_hops cannot be attributed or bounded."""
    return (
        t.censoring in (EXACT, INTERVAL)
        and t.direction is not None
        and t.to_stop is not None
        and t.n_hops is not None
    )


def _chains(traversals: Sequence[Traversal]) -> Iterator[list[Traversal]]:
    """Maximal contiguous runs of one trip's eligible rows, in arrival order."""
    by_trip: dict[str, list[Traversal]] = defaultdict(list)
    for t in traversals:
        by_trip[t.trip_id].append(t)
    for rows in by_trip.values():
        rows.sort(key=lambda t: t.at)
        chain: list[Traversal] = []
        for t in rows:
            if not _eligible(t):
                if chain:
                    yield chain
                chain = []
                continue
            if chain and chain[-1].to_stop != t.from_stop:
                yield chain
                chain = []
            chain.append(t)
        if chain:
            yield chain


def _row_scheduled(
    t: Traversal,
    timetable: Timetable,
    verified_days: Collection[date] | None,
) -> int | None:
    """One observed hop's scheduled seconds, resolved exactly as
    traversal.hop_samples resolves it: nothing outside the feed window, and
    trace.scheduled_for's pattern-then-day-median lookup inside it. With
    `verified_days` given, nothing either for a hop whose service day's archived
    schedule bytes were not verified against the parsed timetable."""
    if (
        verified_days is not None
        and service_dates(t.at, t.trip_id)[0] not in verified_days
    ):
        return None
    if not timetable.covers(t.at, t.trip_id):
        return None
    want = scheduled_for(t, timetable)
    return None if want is None else want.seconds


def spans_from_traversals(
    traversals: Sequence[Traversal],
    timetable: Timetable,
    *,
    max_hops: int = MAX_HOPS,
    verified_days: Collection[date] | None = None,
) -> list[SpanSample]:
    """Every contiguous window of every chain whose realtime hops sum to at
    most `max_hops`, as SpanSamples. `verified_days`, when given, restricts
    scheduled_sec to hops on those service days; realized seconds are kept
    regardless."""
    out: list[SpanSample] = []
    for chain in _chains(traversals):
        sched = [_row_scheduled(t, timetable, verified_days) for t in chain]
        for i, first in enumerate(chain):
            assert first.direction is not None  # _eligible
            day = service_dates(first.at, first.trip_id)[0].isoformat()
            hops = 0
            seconds = 0
            total_sched: int | None = 0
            for j in range(i, len(chain)):
                row = chain[j]
                assert row.to_stop is not None
                assert row.n_hops is not None
                hops += row.n_hops
                if hops > max_hops:
                    break
                seconds += row.seconds
                s = sched[j]
                if total_sched is None or s is None:
                    total_sched = None
                else:
                    total_sched += s
                out.append(
                    SpanSample(
                        trip_id=first.trip_id,
                        route_id=first.route_id,
                        direction=first.direction,
                        origin=first.from_stop,
                        destination=row.to_stop,
                        depart_at=first.at,
                        seconds=seconds,
                        n_hops=hops,
                        scheduled_sec=total_sched,
                        service_day=day,
                    )
                )
    return out


def _require_comparable(result: ReadResult, start: date, end: date) -> None:
    """Realized times may pool across a mid-window feed REPUBLISH (same
    version label, new bytes) because they never read the schedule; they may
    not pool across a schema, extractor, or feed VERSION change. The digest
    difference is handled per day by `verified_days`, not by refusing the
    window. traversal_archive.homogeneous stays the strict check."""
    keys = {(p.schema, p.extractor, p.feed_version) for p in result.provenance.values()}
    if len(keys) > 1:
        raise ValueError(
            f"archive days {start}..{end} are not comparable: "
            f"{len(keys)} distinct (schema, extractor, feed_version) keys: "
            f"{sorted(keys, key=repr)}"
        )


def _build(
    start: date,
    end: date,
    *,
    config: R2Config | None = None,
    client: S3Client | None = None,
) -> tuple[list[SpanSample], ReadResult, frozenset[date]]:
    """load_spans plus the verified-day set, for the CLI's stats.

    The live feed is fetched ONCE; the digest is the sha256 of exactly the
    bytes parsed into the Timetable, so a verified day's scheduled_sec numbers
    are byte-identical to the schedule that stamped that day's trace. A day
    with no digest (backfilled) or a mid-day digest mix is not verified."""
    result = read_days(start, end, config=config, client=client)
    _require_comparable(result, start, end)
    data = fetch_gtfs_zip()
    live_digest = digest_of(data)
    tt = timetable(zipfile.ZipFile(io.BytesIO(data)))
    verified = frozenset(
        day
        for day, p in result.provenance.items()
        if p.feed_digests is None and p.feed_digest == live_digest
    )
    spans = spans_from_traversals(result.traversals, tt, verified_days=verified)
    return spans, result, verified


def load_spans(
    start: date,
    end: date,
    *,
    config: R2Config | None = None,
    client: S3Client | None = None,
) -> tuple[list[SpanSample], ReadResult]:
    """Spans over [start, end] from the derived archive, plus the ReadResult
    whose provenance says what was pooled. Scheduled numbers come only from
    byte-verified days — see _build."""
    spans, result, _verified = _build(start, end, config=config, client=client)
    return spans, result


def _open(path: str, mode: Literal["r", "w"]) -> IO[str]:
    if path.endswith(".gz"):
        if mode == "r":
            return gzip.open(path, "rt", encoding="utf-8")
        return gzip.open(path, "wt", encoding="utf-8")
    return open(path, mode, encoding="utf-8")


def write_spans(path: str, spans: Sequence[SpanSample]) -> None:
    """One JSON object per line, exactly the SpanSample field names."""
    with _open(path, "w") as f:
        for s in spans:
            f.write(json.dumps(asdict(s), separators=(",", ":")))
            f.write("\n")


def read_spans(path: str) -> list[SpanSample]:
    """Inverse of write_spans. String fields are interned: a day's spans repeat
    the same few thousand trip/stop/route ids millions of times, and a run
    over the full archive holds every span at once."""
    out: list[SpanSample] = []
    with _open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out.append(
                SpanSample(
                    trip_id=sys.intern(d["trip_id"]),
                    route_id=sys.intern(d["route_id"]),
                    direction=sys.intern(d["direction"]),
                    origin=sys.intern(d["origin"]),
                    destination=sys.intern(d["destination"]),
                    depart_at=d["depart_at"],
                    seconds=d["seconds"],
                    n_hops=d["n_hops"],
                    scheduled_sec=d["scheduled_sec"],
                    service_day=sys.intern(d["service_day"]),
                )
            )
    return out


def main(argv: list[str] | None = None) -> int:
    """Build the span dataset over a date range and report what it holds."""
    parser = argparse.ArgumentParser(
        description="Journey span dataset from the traversal archive"
    )
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    spans, result, verified = _build(args.start, args.end)
    write_spans(args.out, spans)

    hist = Counter(s.n_hops for s in spans)
    ratios = [
        s.seconds / s.scheduled_sec for s in spans if s.n_hops == 1 and s.scheduled_sec
    ]
    n_none = sum(1 for s in spans if s.scheduled_sec is None)
    print(
        json.dumps(
            {
                "n_traversals": len(result.traversals),
                "n_trips": len({t.trip_id for t in result.traversals}),
                "n_days": len(result.provenance),
                "n_days_schedule_verified": len(verified),
                "n_spans": len(spans),
                "spans_by_n_hops": {
                    str(h): hist.get(h, 0) for h in range(1, MAX_HOPS + 1)
                },
                "share_scheduled_none": (
                    round(n_none / len(spans), 4) if spans else None
                ),
                "one_hop_seconds_over_scheduled_median": (
                    round(statistics.median(ratios), 3) if ratios else None
                ),
                "out": args.out,
            },
            indent=2,
        )
    )
    print(f"{len(spans)} spans -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
