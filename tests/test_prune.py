"""Retention logic for the R2 pruning job (training/prune.py).

Stubs the S3 client — no R2 access.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from training.prune import PARAMS_RETENTION_DAYS, collect_expired

NOW = datetime(2026, 6, 11, tzinfo=UTC)


class _FakePaginator:
    def __init__(self, keys: list[str]) -> None:
        self._keys = keys

    def paginate(self, *, Bucket: str, Prefix: str) -> list[dict[str, Any]]:
        del Bucket
        matching = [k for k in self._keys if k.startswith(Prefix)]
        return [{"Contents": [{"Key": k} for k in matching]}]


class _FakeClient:
    def __init__(self, keys: list[str]) -> None:
        self._keys = keys

    def get_paginator(self, name: str) -> _FakePaginator:
        assert name == "list_objects_v2"
        return _FakePaginator(self._keys)


def test_only_aged_params_snapshots_expire() -> None:
    old_epoch = int(datetime(2025, 11, 1, tzinfo=UTC).timestamp())  # >180d
    new_epoch = int(datetime(2026, 6, 10, tzinfo=UTC).timestamp())
    client = _FakeClient(
        [
            f"state/params/v{old_epoch}.json",
            f"state/params/v{new_epoch}.json",
            "state/params.json",  # live pointer — never pruned
            # Dated history is kept indefinitely, however old.
            "archive/trace/2010-01-01/1000.json",
            "archive/alerts/2010-01-01/1-a.json",
            "v1/predictions/2010-01-01/1000.jsonl",
        ]
    )
    expired = collect_expired(client, "b", NOW)  # type: ignore[arg-type]
    assert expired == [f"state/params/v{old_epoch}.json"]


def test_params_snapshot_exactly_at_the_cutoff_is_kept() -> None:
    # Strictly-older deletes: a snapshot stamped at the cutoff survives.
    cutoff = int((NOW - timedelta(days=PARAMS_RETENTION_DAYS)).timestamp())
    client = _FakeClient(
        [f"state/params/v{cutoff}.json", f"state/params/v{cutoff - 1}.json"]
    )
    expired = collect_expired(client, "b", NOW)  # type: ignore[arg-type]
    assert expired == [f"state/params/v{cutoff - 1}.json"]
