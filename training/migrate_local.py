"""Migrate the local collector's JSONL archive to R2 in the Worker's format.

The collector wrote one record per (poll × alert) — including redundant copies
of long-lived alerts at every 5-min poll. The Worker writes one R2 object per
(alert_id, updated_at) pair. This tool replays the local archive, dedupes by
the same key, and uploads to R2.

Idempotent — re-running is safe; existing keys are simply overwritten with
identical content.

Usage:
    python -m training.migrate_local                      # all local data
    python -m training.migrate_local --dry-run            # count, don't upload
    python -m training.migrate_local --kind alerts        # alerts only
    python -m training.migrate_local --data-dir ./data    # custom path
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from training.r2_client import R2Config, load_config, make_client

# Map collector feed_source strings to the Worker's source names so migrated
# objects use a single uniform vocabulary in the archive.
ENE_SOURCE_MAP: dict[str, str] = {
    "nyct/nyct_ene.json": "ene_current",
    "nyct/nyct_ene_upcoming.json": "ene_upcoming",
    "nyct/nyct_ene_equipments.json": "ene_equipments",
}

_SAFE_KEY_RE = re.compile(r"[^a-zA-Z0-9._-]")


def _safe(s: str) -> str:
    return _SAFE_KEY_RE.sub("_", s)


def _date_prefix(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%d")


def _time_prefix(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, UTC).strftime("%H%M%S")


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield cast(dict[str, Any], json.loads(line))
            except json.JSONDecodeError:
                continue


def collect_alert_uploads(
    paths: Iterable[Path],
) -> list[tuple[str, dict[str, Any]]]:
    """Walk local alert JSONL files, dedupe by (alert_id, updated_at), and
    return (r2_key, body_dict) tuples in chronological order.
    """
    seen: dict[tuple[str, int], int] = {}  # (alert_id, updated_at) → observed_at
    records_in_order: list[dict[str, Any]] = []

    for path in sorted(paths):
        for record in _iter_jsonl(path):
            alert_envelope = cast(dict[str, Any], record.get("alert") or {})
            alert_id = alert_envelope.get("id")
            if not isinstance(alert_id, str):
                continue
            inner = cast(dict[str, Any], alert_envelope.get("alert") or {})
            mercury = cast(
                dict[str, Any], inner.get("transit_realtime.mercury_alert") or {}
            )
            updated_at = mercury.get("updated_at")
            if not isinstance(updated_at, int):
                continue
            key_tuple = (alert_id, updated_at)
            if key_tuple in seen:
                continue
            observed_at = int(record.get("observed_at") or 0)
            seen[key_tuple] = observed_at
            records_in_order.append(record)

    uploads: list[tuple[str, dict[str, Any]]] = []
    for record in records_in_order:
        observed_at = int(record["observed_at"])
        alert_id = record["alert"]["id"]
        key = (
            f"archive/alerts/{_date_prefix(observed_at)}/"
            f"{_time_prefix(observed_at)}-{_safe(alert_id)}.json"
        )
        body = {"observed_at": observed_at, "alert": record["alert"]}
        uploads.append((key, body))
    return uploads


def collect_ene_uploads(
    paths: Iterable[Path],
) -> list[tuple[str, dict[str, Any]]]:
    """Walk local ENE JSONL files, return (r2_key, body) for each snapshot.

    No dedup — each line is one feed × one hourly poll. Keys differ by minute
    so consecutive polls don't collide.
    """
    uploads: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(paths):
        for record in _iter_jsonl(path):
            observed_at = int(record.get("observed_at") or 0)
            feed_source = record.get("feed_source") or ""
            source = ENE_SOURCE_MAP.get(feed_source, _safe(feed_source))
            payload = record.get("payload")
            if observed_at == 0 or payload is None:
                continue
            key = (
                f"archive/ene/{_date_prefix(observed_at)}/"
                f"{_time_prefix(observed_at)}-{source}.json"
            )
            body = {"observed_at": observed_at, "source": source, "payload": payload}
            uploads.append((key, body))
    return uploads


def upload_all(
    uploads: list[tuple[str, dict[str, Any]]],
    config: R2Config,
    *,
    progress_every: int = 50,
) -> int:
    client = make_client(config)
    written = 0
    for i, (key, body) in enumerate(uploads, start=1):
        client.put_object(
            Bucket=config.bucket,
            Key=key,
            Body=json.dumps(body).encode("utf-8"),
            ContentType="application/json",
        )
        written += 1
        if i % progress_every == 0 or i == len(uploads):
            print(f"  uploaded {i}/{len(uploads)}")
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="Collector data root"
    )
    parser.add_argument(
        "--kind",
        choices=("alerts", "ene", "all"),
        default="all",
        help="Which feed family to migrate",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count what would be uploaded; don't talk to R2",
    )
    args = parser.parse_args()

    alerts_paths = sorted((args.data_dir / "alerts").glob("*.jsonl"))
    ene_paths = sorted((args.data_dir / "ene").glob("*.jsonl"))
    print(
        f"Source: {args.data_dir}\n"
        f"  alerts files: {len(alerts_paths)}\n"
        f"  ene files:    {len(ene_paths)}"
    )

    config = None if args.dry_run else load_config()

    if args.kind in ("alerts", "all"):
        print("\nAlerts:")
        uploads = collect_alert_uploads(alerts_paths)
        print(f"  unique (alert_id, updated_at) versions: {len(uploads)}")
        if not args.dry_run and uploads:
            assert config is not None
            upload_all(uploads, config)

    if args.kind in ("ene", "all"):
        print("\nE&E:")
        uploads = collect_ene_uploads(ene_paths)
        print(f"  snapshot records: {len(uploads)}")
        if not args.dry_run and uploads:
            assert config is not None
            upload_all(uploads, config)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
