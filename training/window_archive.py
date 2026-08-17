"""A long-lived archive of the ANSWER KEY: parsed announced-work windows.

WHY, AND WHY IT IS NOT OPTIONAL. `traversal_archive` keeps the movement past
`archive/trace/`'s 30-day prune. On its own that buys nothing for retrospective
grading, because the thing movement is graded AGAINST — announced planned work —
is reconstructed by `planned_work.windows_from_alerts` from
`archive/alerts/`, and that prefix is pruned at 90 days. Keep the measurements
and drop the answer key and a grade over last year's traversals cannot run at
all; worse, it could silently run over the subset of history whose alerts happen
to survive, and report a number.

So both halves are persisted on the same nightly path, both BEFORE the prune
step that would otherwise remove their sources.

WHAT A WINDOW IS. One (alert, active_period) with its named stations, exactly as
`windows_from_alerts` produces it — this module stores that output and does not
reinterpret it. Windows are recorded per day OBSERVED, meaning the day whose
alert versions mentioned them: work is announced days ahead, so one window
appears in several days' alert files and readers de-duplicate. That is deliberate
— it preserves WHEN a window was announced, which a grade of a forecast (as
opposed to a detection) will need.

PROVENANCE, for the same reason as the traversal archive: this outlives
`archive/alerts/`, so once those are pruned a stored window cannot be
regenerated or checked. `PARSER_VERSION` tracks the semantics of
`windows_from_alerts` — which alert types count, how stops are scoped, how
active periods are split — and is guarded by a golden test, because a parse
change that forgets to bump it would make old and new answer keys disagree while
every grade kept producing plausible numbers.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any, cast

from training.load_r2 import date_range, input_manifest_hash, list_alert_keys, list_keys
from training.planned_work import Window, windows_from_alerts
from training.provenance import code_provenance
from training.r2_client import R2Config, get_object_bytes, load_config, make_client

# Reused rather than re-declared: the two derived archives must answer the
# "may these days be pooled" question the same way, and a second provenance type
# would be a second convention that could drift from the first.
from training.traversal_archive import DayProvenance, is_closed

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

PREFIX = "archive/windows/"

SCHEMA_VERSION = 1

# Semantics of planned_work.windows_from_alerts. Bump when the set of counted
# alert types, the stop scoping, or the active-period handling changes.
PARSER_VERSION = 1

FIELDS: tuple[str, ...] = ("alert_type", "routes", "stops", "start", "end")


def _row(w: Window) -> list[Any]:
    return [w.alert_type, sorted(w.routes), sorted(w.stops), w.start, w.end]


def _window(row: Sequence[Any]) -> Window:
    alert_type, routes, stops, start, end = row
    return Window(
        alert_type=str(alert_type),
        routes=frozenset(cast(list[str], routes)),
        stops=frozenset(cast(list[str], stops)),
        start=int(start),
        end=int(end),
    )


def key_for(day: date) -> str:
    """Date-segment layout, matching every other stream so `training.prune`'s
    date matcher polices this prefix too."""
    return f"{PREFIX}{day.isoformat()}/windows.json.gz"


def encode_day(
    windows: Sequence[Window],
    *,
    source_keys: Sequence[str],
) -> bytes:
    doc = {
        "provenance": {
            "schema": SCHEMA_VERSION,
            "extractor": PARSER_VERSION,
            "feed_version": None,  # alerts carry no static-feed dependence
            "n_rows": len(windows),
            "n_source_objects": len(source_keys),
            "source_manifest": input_manifest_hash(list(source_keys)),
            "code_sha": code_provenance()["code_sha"],
            "written_at": int(datetime.now(UTC).timestamp()),
        },
        "fields": list(FIELDS),
        "rows": [_row(w) for w in windows],
    }
    return gzip.compress(json.dumps(doc, separators=(",", ":")).encode(), 9)


def decode_day(blob: bytes) -> tuple[list[Window], DayProvenance]:
    """Inverse of `encode_day`, refusing anything it cannot read faithfully.

    Same contract as the traversal archive: an unknown schema or a reordered
    field list raises rather than best-efforting a parse, because the alert
    versions that could have checked the result are gone.
    """
    doc = cast(dict[str, Any], json.loads(gzip.decompress(blob)))
    prov_raw = cast(dict[str, Any], doc.get("provenance") or {})
    schema = int(prov_raw.get("schema") or 0)
    if schema != SCHEMA_VERSION:
        raise ValueError(
            f"window archive schema {schema} is not readable by this build "
            f"(expects {SCHEMA_VERSION})"
        )
    missing = {"extractor", "n_rows", "n_source_objects", "source_manifest"} - set(
        prov_raw
    )
    if missing:
        # Silently defaulting a missing count to 0 is how a format rename turns
        # historical rows into plausible nonsense. Absent provenance is a hard
        # failure, because the inputs that could have rebuilt it are gone.
        raise ValueError(f"provenance is missing {sorted(missing)}")
    fields = tuple(cast(list[str], doc.get("fields") or ()))
    if fields != FIELDS:
        raise ValueError(f"window archive field order {fields} != {FIELDS}")
    prov = DayProvenance(
        schema=schema,
        extractor=int(prov_raw.get("extractor") or 0),
        feed_version=cast(str | None, prov_raw.get("feed_version")),
        n_rows=int(prov_raw.get("n_rows") or 0),
        n_source_objects=int(prov_raw.get("n_source_objects") or 0),
        source_manifest=str(prov_raw.get("source_manifest") or ""),
        code_sha=cast(str | None, prov_raw.get("code_sha")),
        written_at=int(prov_raw.get("written_at") or 0),
    )
    rows = [_window(cast(Sequence[Any], r)) for r in cast(list[Any], doc["rows"])]
    return rows, prov


@dataclass(frozen=True)
class WindowReadResult:
    """The answer key over a span, de-duplicated, with its provenance.

    Windows are stored per day observed and a multi-day announcement appears in
    several of them, so `windows` is a de-duplicated set rather than a
    concatenation. `Window` is frozen and hashable, which is what makes that
    exact rather than approximate.
    """

    windows: list[Window]
    provenance: dict[date, DayProvenance]

    @property
    def versions(self) -> set[tuple[int, int, str | None]]:
        return {p.comparable_key for p in self.provenance.values()}

    @property
    def homogeneous(self) -> bool:
        return len(self.versions) <= 1


def read_days(
    start: date,
    end: date,
    *,
    config: R2Config | None = None,
    client: S3Client | None = None,
) -> WindowReadResult:
    cfg = config or load_config()
    client = client or make_client(cfg)
    present = set(list_keys(client, cfg.bucket, PREFIX))
    seen: set[Window] = set()
    prov: dict[date, DayProvenance] = {}
    for day in date_range(start, end):
        key = key_for(day)
        if key not in present:
            continue
        rows, p = decode_day(get_object_bytes(client, cfg.bucket, key))
        seen.update(rows)
        prov[day] = p
    return WindowReadResult(
        windows=sorted(seen, key=lambda w: (w.start, w.alert_type, sorted(w.routes))),
        provenance=prov,
    )


def write_day(
    day: date,
    *,
    config: R2Config | None = None,
    client: S3Client | None = None,
    overwrite: bool = False,
    allow_partial: bool = False,
    now: date | None = None,
    present: set[str] | None = None,
) -> DayProvenance | None:
    """Parse one day's alert versions into windows and store them.

    Closed days only, for the same reason as the traversal archive: a day still
    gaining alert versions would be finalized short and then skipped forever.
    """
    cfg = config or load_config()
    client = client or make_client(cfg)
    if not allow_partial and not is_closed(day, now=now):
        return None
    key = key_for(day)
    if present is None:
        present = set(list_keys(client, cfg.bucket, PREFIX))
    if not overwrite and key in present:
        return None

    alert_keys = list_alert_keys(client, cfg.bucket, day, day)
    if not alert_keys:
        return None
    bodies = [
        cast(dict[str, Any], json.loads(get_object_bytes(client, cfg.bucket, k)))
        for k in alert_keys
    ]
    windows = windows_from_alerts(bodies)
    blob = encode_day(windows, source_keys=alert_keys)
    client.put_object(
        Bucket=cfg.bucket,
        Key=key,
        Body=blob,
        ContentType="application/json",
        ContentEncoding="gzip",
    )
    _rows, prov = decode_day(blob)
    return prov


def backfill(
    days: Iterable[date],
    *,
    overwrite: bool = False,
    allow_partial: bool = False,
) -> list[tuple[date, DayProvenance | None]]:
    cfg = load_config()
    client = make_client(cfg)
    present = set(list_keys(client, cfg.bucket, PREFIX))
    out: list[tuple[date, DayProvenance | None]] = []
    for day in days:
        prov = write_day(
            day,
            config=cfg,
            client=client,
            overwrite=overwrite,
            allow_partial=allow_partial,
            present=present,
        )
        out.append((day, prov))
        if prov is not None:
            reason = f"{prov.n_rows} windows"
        elif not allow_partial and not is_closed(day):
            reason = "skipped: day still open"
        else:
            reason = "skipped: already present or no alerts"
        print(f"{day}: {reason}", file=sys.stderr)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write the long-lived announced-window archive from alert versions"
    )
    parser.add_argument("--start-date", type=date.fromisoformat)
    parser.add_argument("--end-date", type=date.fromisoformat)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rewrite days already present (needed after a parser bump)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="also write the day still in progress; requires --overwrite",
    )
    args = parser.parse_args(argv)

    if args.allow_partial and not args.overwrite:
        parser.error("--allow-partial requires --overwrite")

    today = datetime.now(UTC).date()
    end = args.end_date or (today - timedelta(days=1))
    start = args.start_date or end

    written = backfill(
        date_range(start, end),
        overwrite=args.overwrite,
        allow_partial=args.allow_partial,
    )
    print(
        json.dumps(
            {
                "schema": SCHEMA_VERSION,
                "parser": PARSER_VERSION,
                "days_written": sum(1 for _d, p in written if p is not None),
                "days_skipped": sum(1 for _d, p in written if p is None),
                "windows": sum(p.n_rows for _d, p in written if p is not None),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
