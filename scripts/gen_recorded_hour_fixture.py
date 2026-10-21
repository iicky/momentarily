"""Package real 2026-10-20 payloads as the Worker recorder stores them.

    uv run python -m scripts.gen_recorded_hour_fixture --sample <step2-scope>

``<step2-scope>`` is the 2026-10-20 sampling run: ``raw/<source>/<ms>.<ext>``
(every feed polled about every 15 s for 10 minutes, ENE once) and
``fetches.jsonl`` (one row per poll: start time, status, headers, elapsed
time, sha256). The output, ``tests/fixtures/model/observations/recorded_hour``,
is a store root holding the keys the recorder writes
(``worker/src/recorder.ts``) for ten cron minutes of hour 14 UTC:

* ``ticks/.../<scheduled_at>-<started_at_ms>.json``: the ``CollectorTick``,
  serialized like ``JSON.stringify`` (compact, contract field order);
* ``raw/gtfsrt/....bin``: gzip member 0 the ``TickBundleIndex``, then one
  gzip member per GTFS-RT body, numbered in index order;
* ``raw/blob/<day>/<sha256>.json.gz``: one gzip member per JSON body,
  written once per UTC day.

Minute ``k`` (0-9) is sample round ``4k``, so invocations are 60 s apart.
Fetch times, statuses, content types, ``Date`` headers and bodies are the
sample's; outcome, entity count and header timestamp are computed from the
bodies the way the recorder computes them. Every minute polls the eight
GTFS-RT feeds, the alerts JSON and the static GTFS HEAD; minute 0 also
records the hourly ENE fetches (outages, upcoming, equipment catalog).
Synthesized, because the sample has none of them:

* the static GTFS HEAD (40 ms, 200, ``application/zip``, no body);
* minute 3: ``si`` answers HTTP 503 (a failed fetch, no body);
* minute 5: a second, overlapping invocation 3 s after the first, built
  from sample round 21, whose ``ace``, ``bdfm`` and ``nqrw`` fetches time out;
* minute 7: the tick record is missing (its PUT failed), so only the
  bundle describes that invocation and its alerts fetch is lost.

The sample's alerts JSON did not change in the 10 minutes, so every alerts
fetch names one blob. The output is deterministic (gzip mtime 0, level 6 as
``CompressionStream`` uses): rerunning gives the same bytes.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from momentarily.model.observations.contracts import (
    GTFSRT_SOURCES,
    CollectorTick,
    FeedFetch,
    SourceId,
    TickBundleIndex,
)
from momentarily.model.observations.decode import parse_feed
from momentarily.model.observations.keys import BlobKey, BundleKey, TickKey, utc_day

OUT = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "model"
    / "observations"
    / "recorded_hour"
)

GATEWAY = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds"
URLS: dict[str, str] = {
    "ace": f"{GATEWAY}/nyct%2Fgtfs-ace",
    "bdfm": f"{GATEWAY}/nyct%2Fgtfs-bdfm",
    "g": f"{GATEWAY}/nyct%2Fgtfs-g",
    "jz": f"{GATEWAY}/nyct%2Fgtfs-jz",
    "nqrw": f"{GATEWAY}/nyct%2Fgtfs-nqrw",
    "l": f"{GATEWAY}/nyct%2Fgtfs-l",
    "numbered": f"{GATEWAY}/nyct%2Fgtfs",
    "si": f"{GATEWAY}/nyct%2Fgtfs-si",
    "alerts": f"{GATEWAY}/camsys%2Fsubway-alerts.json",
    "ene_current": f"{GATEWAY}/nyct%2Fnyct_ene.json",
    "ene_upcoming": f"{GATEWAY}/nyct%2Fnyct_ene_upcoming.json",
    "ene_equipments": f"{GATEWAY}/nyct%2Fnyct_ene_equipments.json",
    "gtfs_static": "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip",
}
# The order index.ts plans them in: alerts and the GTFS-RT feeds at the
# start, the static HEAD, then the pipeline's hourly ENE fetches.
MINUTE_SOURCES: tuple[SourceId, ...] = (
    "alerts", "ace", "bdfm", "g", "jz", "nqrw", "l", "numbered", "si", "gtfs_static",
)  # fmt: skip
ENE_SOURCES: tuple[SourceId, ...] = ("ene_current", "ene_upcoming", "ene_equipments")

MINUTES = 10
OVERLAP_MINUTE, OVERLAP_ROUND, OVERLAP_DELAY_S = 5, 21, 3
OVERLAP_TIMEOUTS = ("ace", "bdfm", "nqrw")
HTTP_ERROR_MINUTE, HTTP_ERROR_SOURCE = 3, "si"
MISSING_TICK_MINUTE = 7


def compact_json(model: BaseModel) -> bytes:
    """``model`` as the recorder's ``JSON.stringify`` writes it."""
    doc = model.model_dump(mode="json", by_alias=True)
    return json.dumps(doc, separators=(",", ":"), ensure_ascii=False).encode()


def gzip_member(body: bytes) -> bytes:
    return gzip.compress(body, compresslevel=6, mtime=0)


def bundle_bytes(index: TickBundleIndex, bodies: dict[int, bytes]) -> bytes:
    """A tick bundle: the index member, then the body of each member."""
    members = [compact_json(index), *(bodies[m] for m in sorted(bodies))]
    return b"".join(gzip_member(m) for m in members)


def _header_seconds(value: Any) -> int | None:
    """``epochSeconds`` in recorder.ts."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value if 0 <= value <= 2**53 - 1 else None
    if (
        isinstance(value, str)
        and value.isascii()
        and value.isdigit()
        and len(value) <= 15
    ):
        return int(value)
    return None


def _parsed(source: str, body: bytes) -> tuple[str, int, int | None]:
    """(outcome, entity_count, header_timestamp) as the recorder computes them."""
    if source in GTFSRT_SOURCES:
        msg = parse_feed(body)
        header = msg.header.timestamp if msg.header.HasField("timestamp") else None
        count = len(msg.entity)
    else:
        doc = json.loads(body)
        if source == "alerts":
            count = len(doc.get("entity", []))
            head = doc.get("header")
            header = (
                _header_seconds(cast(dict[str, Any], head).get("timestamp"))
                if isinstance(head, dict)
                else None
            )
        else:
            count, header = len(doc), None
    return ("ok" if count else "empty"), count, header


def _http_instant(text: str | None) -> int | None:
    return None if text is None else int(parsedate_to_datetime(text).timestamp())


class Sample:
    def __init__(self, root: Path) -> None:
        self.raw = root / "raw"
        self.rows: dict[tuple[str, int], dict[str, Any]] = {}
        for line in (root / "fetches.jsonl").read_text().splitlines():
            row = json.loads(line)
            self.rows[row["name"], row["round"]] = row

    def poll(self, source: str, round_: int) -> tuple[dict[str, Any], int, bytes]:
        row = self.rows[source, round_]
        started = int(row["fetched_at"] * 1000)
        ext = "pb" if source in GTFSRT_SOURCES else "json"
        body = (self.raw / source / f"{started}.{ext}").read_bytes()
        if hashlib.sha256(body).hexdigest() != row["sha256"]:
            raise SystemExit(f"{source} round {round_}: body differs from the log")
        return row, started, body


class Writer:
    def __init__(self, out: Path) -> None:
        self.out = out

    def put(self, key: str, data: bytes) -> None:
        path = self.out / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def _fetch(**fields: Any) -> FeedFetch:
    empty = dict.fromkeys(
        (
            "http_status", "content_type", "http_date", "http_last_modified",
            "http_etag", "body_bytes", "payload_sha256", "payload_key",
            "payload_member", "header_timestamp", "entity_count", "store_error",
        )
    )  # fmt: skip
    return FeedFetch.model_validate(empty | fields)


def invocation(
    sample: Sample,
    writer: Writer,
    round_: int,
    sources: tuple[SourceId, ...],
    *,
    scheduled_at: int | None = None,
    timeouts: tuple[str, ...] = (),
    http_errors: tuple[str, ...] = (),
    write_tick: bool = True,
) -> None:
    first = min(
        int(sample.rows[s, round_]["fetched_at"] * 1000)
        for s in sources
        if s != "gtfs_static"
    )
    started_ms = first - 40
    sched = scheduled_at if scheduled_at is not None else (started_ms - 1000) // 1000
    bundle_key = BundleKey(sched, started_ms).key
    fetches: list[FeedFetch] = []
    bodies: dict[int, bytes] = {}
    clock = started_ms
    for source in sources:
        common = {
            "fetch_id": "",
            "source": source,
            "url": URLS[source],
            "method": "GET",
            "scheduled_at": sched,
        }
        if source == "gtfs_static":
            started = clock + 5
            finished = started + 40
            f = _fetch(
                **(common | {"fetch_id": f"{source}@{started}", "method": "HEAD"}),
                started_at_ms=started,
                finished_at_ms=finished,
                outcome="ok",
                http_status=200,
                content_type="application/zip",
                http_date=finished // 1000,
            )
        else:
            row, started, body = sample.poll(source, round_)
            common["fetch_id"] = f"{source}@{started}"
            finished = started + row["elapsed_ms"]
            if source in timeouts:
                f = _fetch(
                    **common,
                    started_at_ms=started,
                    finished_at_ms=None,
                    outcome="timeout",
                )
                finished = started + 10_000
            elif source in http_errors:
                f = _fetch(
                    **common,
                    started_at_ms=started,
                    finished_at_ms=finished,
                    outcome="http_error",
                    http_status=503,
                )
            else:
                outcome, count, header = _parsed(source, body)
                sha = hashlib.sha256(body).hexdigest()
                headers = row["headers"]
                stored: dict[str, Any]
                if source in GTFSRT_SOURCES:
                    member = len(bodies) + 1
                    bodies[member] = body
                    stored = {"payload_key": bundle_key, "payload_member": member}
                else:
                    key = BlobKey(utc_day(finished // 1000), sha, "json").key
                    if not (writer.out / key).exists():
                        writer.put(key, gzip_member(body))
                    stored = {"payload_key": key}
                f = _fetch(
                    **common,
                    started_at_ms=started,
                    finished_at_ms=finished,
                    outcome=outcome,
                    http_status=row["status"],
                    content_type=headers.get("content-type"),
                    http_date=_http_instant(headers.get("date")),
                    http_last_modified=_http_instant(headers.get("last-modified")),
                    http_etag=headers.get("etag"),
                    body_bytes=len(body),
                    payload_sha256=sha,
                    header_timestamp=header,
                    entity_count=count,
                    **stored,
                )
        clock = max(clock, finished)
        fetches.append(f)
    # Fetches stay in planned order, so members are already numbered in the
    # order the index lists the stored fetches.
    index = TickBundleIndex.model_validate(
        {
            "schema": "observations/s1",
            "tick_id": f"{sched}-{started_ms}",
            "scheduled_at": sched,
            "started_at_ms": started_ms,
            "fetches": [f for f in fetches if f.source in GTFSRT_SOURCES],
        }
    )
    writer.put(bundle_key, bundle_bytes(index, bodies))
    if not write_tick:
        return
    tick = CollectorTick.model_validate(
        {
            "schema": "observations/s1",
            "tick_id": f"{sched}-{started_ms}",
            "scheduled_at": sched,
            "started_at_ms": started_ms,
            "finished_at_ms": clock + 25,
            "collector": {"code_sha": "0" * 40, "code_dirty": False},
            "planned_sources": list(sources),
            "fetches": fetches,
        }
    )
    writer.put(TickKey(sched, started_ms).key, compact_json(tick))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m scripts.gen_recorded_hour_fixture")
    parser.add_argument(
        "--sample", type=Path, required=True, help="the step-2 sampling run"
    )
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    sample = Sample(args.sample.expanduser())
    if args.out.exists():
        shutil.rmtree(args.out)
    writer = Writer(args.out)
    for minute in range(MINUTES):
        round_ = 4 * minute
        sources = MINUTE_SOURCES + (ENE_SOURCES if minute == 0 else ())
        invocation(
            sample,
            writer,
            round_,
            sources,
            http_errors=(HTTP_ERROR_SOURCE,) if minute == HTTP_ERROR_MINUTE else (),
            write_tick=minute != MISSING_TICK_MINUTE,
        )
        if minute == OVERLAP_MINUTE:
            first = min(
                int(sample.rows[s, round_]["fetched_at"] * 1000)
                for s in MINUTE_SOURCES
                if s != "gtfs_static"
            )
            invocation(
                sample,
                writer,
                OVERLAP_ROUND,
                MINUTE_SOURCES,
                scheduled_at=(first - 1040) // 1000 + OVERLAP_DELAY_S,
                timeouts=OVERLAP_TIMEOUTS,
            )
    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"wrote {args.out} ({size} bytes)")


if __name__ == "__main__":
    main()
