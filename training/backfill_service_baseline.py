"""Backfill the service-baseline sidecar (state/service_baseline.json) from the
trip-updates archive, WITHOUT touching params.json.

The supply axis's denominator -- per-route, per-schedule_bin median assigned_n --
was historically folded into params.json, so it could only be refreshed by a
full retrain. A retrain moves the HMM artifact's trained_at, which reseeds the
Worker's filter and splits the grader's params-version window. This tool computes
just that baseline and publishes it to its own versioned object
(train_em.write_service_baseline), so a frozen model can light or refresh the
supply axis with its evaluation window and weights left untouched.

The sidecar carries its OWN `generated_at` stamp (a fresh timestamp per run, so
repeated backfills never overwrite a prior immutable snapshot) and records the
frozen model it pairs with as `params_trained_at`.

Usage:
    murk exec -- uv run python -m training.backfill_service_baseline --dry-run
    murk exec -- uv run python -m training.backfill_service_baseline \
        [--start YYYY-MM-DD] [--end YYYY-MM-DD]
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING

from training.r2_client import load_config, make_client
from training.train_em import (
    PARAMS_KEY,
    SERVICE_BASELINE_KEY,
    _service_baseline,  # pyright: ignore[reportPrivateUsage]
    write_service_baseline,
)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

# Trailing window used when no explicit dates are given. Matches the archive's
# retention headroom and is long enough for a stable per-cell normal.
DEFAULT_WINDOW_DAYS = 28


def _current_params_trained_at(client: S3Client, bucket: str) -> int | None:
    """trained_at of the live params.json, recorded on the sidecar as the model
    it accompanies. None (omit the provenance field) when the pointer is missing
    or unreadable."""
    try:
        obj = client.get_object(Bucket=bucket, Key=PARAMS_KEY)
        doc = json.loads(obj["Body"].read())
        raw = doc.get("trained_at")
        return int(raw) if raw is not None else None
    except Exception as exc:
        print(f"could not read {PARAMS_KEY} trained_at ({exc})", file=sys.stderr)
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Publish the service-baseline sidecar without retraining"
    )
    parser.add_argument("--start", help="window start YYYY-MM-DD (default: end - 28d)")
    parser.add_argument("--end", help="window end YYYY-MM-DD (default: yesterday)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="compute and report cell counts, but write nothing",
    )
    args = parser.parse_args(argv)

    end = (
        date.fromisoformat(args.end)
        if args.end
        else datetime.now(UTC).date() - timedelta(days=1)
    )
    start = (
        date.fromisoformat(args.start)
        if args.start
        else end - timedelta(days=DEFAULT_WINDOW_DAYS - 1)
    )

    cfg = load_config()
    client = make_client(cfg)

    (_tod, _n_tod, _sched, _n_sched, hourly, n_hourly) = _service_baseline(
        cfg, client, start, end
    )
    if not hourly:
        print(
            f"no hourly service baseline for {start}..{end}; nothing to write",
            file=sys.stderr,
        )
        return 1

    generated_at = int(datetime.now(UTC).timestamp())
    params_trained_at = _current_params_trained_at(client, cfg.bucket)
    routes = sorted(hourly)
    sample = {r: hourly[r] for r in routes[:3]}
    print(
        f"service baseline {start}..{end}: {len(routes)} routes, "
        f"{n_hourly} (route,bin) cells; generated_at={generated_at}, "
        f"params_trained_at={params_trained_at}"
    )
    print(f"sample: {json.dumps(sample)[:400]}")

    if args.dry_run:
        print(
            f"DRY RUN -- would write {SERVICE_BASELINE_KEY} "
            f"(+ versioned v{generated_at}); params.json untouched"
        )
        return 0

    n = write_service_baseline(
        client, cfg.bucket, hourly, generated_at, params_trained_at=params_trained_at
    )
    print(f"wrote {SERVICE_BASELINE_KEY} + versioned v{generated_at}: {n} routes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
