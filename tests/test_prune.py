"""Retention logic for the R2 pruning job (training/prune.py).

Stubs the S3 client — no R2 access.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from training.prune import collect_expired

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


def test_dated_prefixes_expire_past_retention() -> None:
    client = _FakeClient(
        [
            "v1/predictions/2026-01-01/1000.jsonl",  # ~160d old → expired
            "v1/predictions/2026-06-01/2000.jsonl",  # 10d old → kept
            "archive/alerts/2026-02-15/1-a.json",  # ~115d old → expired
            "archive/alerts/2026-06-10/2-b.json",  # kept
            "v1/eval.json",  # no date segment → never pruned
        ]
    )
    expired = collect_expired(client, "b", NOW)  # type: ignore[arg-type]
    assert expired["v1/predictions/"] == ["v1/predictions/2026-01-01/1000.jsonl"]
    assert expired["archive/alerts/"] == ["archive/alerts/2026-02-15/1-a.json"]


def test_params_versions_expire_by_epoch() -> None:
    old_epoch = int(datetime(2025, 11, 1, tzinfo=UTC).timestamp())  # >180d
    new_epoch = int(datetime(2026, 6, 10, tzinfo=UTC).timestamp())
    client = _FakeClient(
        [
            f"state/params/v{old_epoch}.json",
            f"state/params/v{new_epoch}.json",
            "state/params.json",  # live pointer — never pruned
        ]
    )
    expired = collect_expired(client, "b", NOW)  # type: ignore[arg-type]
    assert expired["state/params/"] == [f"state/params/v{old_epoch}.json"]


def test_exact_boundary_is_kept() -> None:
    # An object exactly at the cutoff date is kept (strictly-older deletes).
    client = _FakeClient(["v1/predictions/2026-03-13/1.jsonl"])  # 90d before NOW
    expired = collect_expired(client, "b", NOW)  # type: ignore[arg-type]
    assert expired["v1/predictions/"] == []


def test_trace_prefix_uses_its_own_narrower_window() -> None:
    # archive/trace/ retains 30d, well short of the other dated prefixes'
    # 90d, because at ~125 MB/day it is two orders of magnitude bigger. A
    # 40d-old object only expires here if the prefix's own 30d window is in
    # effect — it would still be within any of the 90d windows.
    client = _FakeClient(
        [
            "archive/trace/2026-05-02/1000.json",  # 40d old → expired at 30d
            "archive/trace/2026-06-06/2000.json",  # 5d old → kept
        ]
    )
    expired = collect_expired(client, "b", NOW)  # type: ignore[arg-type]
    assert expired["archive/trace/"] == ["archive/trace/2026-05-02/1000.json"]
