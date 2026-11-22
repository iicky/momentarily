"""Readers of the archive objects a Worker run writes once.

Objects are named by the run's tick id, ``<scheduled_at>-<started_at_ms>``;
those written before are named by the run's execution second,
``<observed_at>``. Both stay in the bucket, so every reader takes both. The
old-named objects here are real ones, from the legacy backfill's fixture
mirror (cut from the R2 mirror); the run-named ones are the same bytes under
the name the Worker now gives them.
"""

from __future__ import annotations

import io
import json
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from training.alert_poll_latency import liveness_keys
from training.eval import (
    load_movement_census,
    load_movement_transitions,
    load_predictions,
)
from training.load_r2 import fetch_trip_update_metrics, fetch_vehicle_metrics
from training.r2_client import R2Config

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

MIRROR = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "model"
    / "observations"
    / "legacy_mirror"
)
CFG = R2Config("account", "key", "secret", "bucket")


def _mirror(prefix: str) -> dict[str, bytes]:
    return {
        p.relative_to(MIRROR).as_posix(): p.read_bytes()
        for p in sorted((MIRROR / prefix).rglob("*"))
        if p.is_file()
    }


def _run_name(key: str) -> str:
    """``<day>/<observed_at><ext>`` -> ``<day>/<scheduled_at>-<started_at_ms><ext>``."""
    folder, name = key.rsplit("/", 1)
    at, ext = name.split(".", 1)
    return f"{folder}/{int(at) - 41}-{int(at) * 1000 + 7}.{ext}"


def _renamed(objects: dict[str, bytes]) -> dict[str, bytes]:
    return {_run_name(k): v for k, v in objects.items()}


class _Bucket:
    """list_objects_v2 and get_object over a dict."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        prefix = str(kwargs["Prefix"])
        keys = sorted(k for k in self.objects if k.startswith(prefix))
        return {"Contents": [{"Key": k} for k in keys], "IsTruncated": False}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        return {"Body": io.BytesIO(self.objects[str(kwargs["Key"])])}


def _client(objects: dict[str, bytes]) -> S3Client:
    return cast("S3Client", _Bucket(objects))


def test_trip_update_and_vehicle_metrics_read_both_names() -> None:
    for prefix, fetch, start, end in (
        (
            "archive/trip_updates",
            fetch_trip_update_metrics,
            date(2026, 6, 15),
            date(2026, 9, 22),
        ),
        (
            "archive/vehicles",
            fetch_vehicle_metrics,
            date(2026, 6, 21),
            date(2026, 6, 21),
        ),
    ):
        old = _mirror(prefix)
        got = fetch(CFG, start_date=start, end_date=end, client=_client(old))
        assert got == [json.loads(v) for v in old.values()]
        assert (
            fetch(CFG, start_date=start, end_date=end, client=_client(_renamed(old)))
            == got
        )
        both = fetch(
            CFG, start_date=start, end_date=end, client=_client(old | _renamed(old))
        )
        assert sorted(json.dumps(b, sort_keys=True) for b in both) == sorted(
            json.dumps(b, sort_keys=True) for b in got + got
        )


def test_predictions_read_both_names() -> None:
    old = _mirror("v1/predictions")
    window = (date(2026, 7, 21), date(2026, 7, 21))
    got = load_predictions(_client(old), "bucket", *window)
    assert len(got) == sum(len(v.splitlines()) for v in old.values())
    assert load_predictions(_client(_renamed(old)), "bucket", *window) == got


def test_alerts_liveness_keys_select_both_names_by_day() -> None:
    (old,) = _mirror("archive/alerts_liveness")  # 2026-09-02
    new = _run_name(old)
    later = "archive/alerts_liveness/2026-09-03/1788465000-1788465060123.json"
    keys = [old, new, later, "archive/alerts_liveness/2026-09-02/not-a-run.json"]
    assert liveness_keys(keys, date(2026, 9, 2), date(2026, 9, 2)) == [old, new]
    assert liveness_keys(keys, date(2026, 9, 3), date(2026, 9, 3)) == [later]


def test_runs_observed_in_one_second_each_load() -> None:
    """A late run of one minute and the next minute's on-time run that start
    in the same second keep an object each, beside an old-named one."""
    at = 1_700_000_900
    census = {
        f"archive/movement_census/2023-11-14/{name}.json": json.dumps(
            {
                "observed_at": at,
                "regimes": {"A": {"state": s, "open_state": None, "open_since": None}},
            }
        ).encode()
        for name, s in (
            (str(at), "normal"),
            (f"{at - 41}-{at * 1000 + 7}", "disrupted"),
        )
    }
    loaded = load_movement_census(
        _client(census), "bucket", date(2023, 11, 14), date(2023, 11, 14)
    )
    assert sorted(r.regimes["A"].state for r in loaded) == ["disrupted", "normal"]

    def line(scope: str, key: str) -> bytes:
        return json.dumps(
            {
                "ts": at,
                "scope": scope,
                "key": key,
                "route": key.split("|")[0],
                "prev_state": "normal",
                "new_state": "disrupted",
                "regime_entered_at": at - 600,
                "exited_at": at,
                "dwell_sec": 600,
            }
        ).encode()

    folder = "v1/movement_transitions/2023-11-14"
    transitions = {
        f"{folder}/{at}-route.jsonl": line("route", "A"),
        f"{folder}/{at - 41}-{at * 1000 + 7}-route.jsonl": line("route", "B"),
        f"{folder}/{at - 101}-{at * 1000 + 9}-segment.jsonl": line(
            "segment", "C|north|C01N"
        ),
    }
    loaded_t = load_movement_transitions(
        _client(transitions), "bucket", date(2023, 11, 14), date(2023, 11, 14)
    )
    assert sorted((r.scope, r.key) for r in loaded_t) == [
        ("route", "A"),
        ("route", "B"),
        ("segment", "C|north|C01N"),
    ]
