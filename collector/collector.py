"""MTA alert + E&E collector.

Polls the MTA developer gateway on a 5-minute cadence (alerts) and hourly
cadence (elevator/escalator) and appends every observation to a daily-rotated
JSONL log on a bind-mounted volume.

Designed for local-only operation under Colima or Docker Desktop. Survives
restarts gracefully by reading meta/last_fetched.json on startup.

Output goes to $DATA_DIR (default /data inside the container; bind-mounted
to ../data/ from the host via docker-compose).
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

# MTA gateway endpoints (JSON variants only — no protobuf needed in v1)
GATEWAY = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds"
SUBWAY_ALERTS_URL = f"{GATEWAY}/camsys%2Fsubway-alerts.json"
ENE_CURRENT_URL = f"{GATEWAY}/nyct%2Fnyct_ene.json"
ENE_UPCOMING_URL = f"{GATEWAY}/nyct%2Fnyct_ene_upcoming.json"

# Config — overridable via env
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL_SECONDS", "300"))
ENE_POLL_INTERVAL = int(os.environ.get("ENE_POLL_INTERVAL_SECONDS", "3600"))
REQUEST_TIMEOUT = httpx.Timeout(30.0)

# Derived paths
ALERTS_DIR = DATA_DIR / "alerts"
ENE_DIR = DATA_DIR / "ene"
META_DIR = DATA_DIR / "meta"
LAST_FETCHED_PATH = META_DIR / "last_fetched.json"
POLL_LOG_PATH = META_DIR / "poll_log.jsonl"

_LOGGER = logging.getLogger("momentarily.collector")
_SHUTDOWN = False


def _utc_now() -> int:
    return int(datetime.now(UTC).timestamp())


def _utc_today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def _ensure_dirs() -> None:
    for path in (ALERTS_DIR, ENE_DIR, META_DIR):
        path.mkdir(parents=True, exist_ok=True)


def _read_last_fetched() -> dict[str, int]:
    if not LAST_FETCHED_PATH.exists():
        return {}
    try:
        with LAST_FETCHED_PATH.open() as f:
            data = json.load(f)
            return {k: int(v) for k, v in data.items()}
    except (OSError, json.JSONDecodeError, ValueError) as err:
        _LOGGER.warning("Could not read last_fetched.json (%s); starting fresh", err)
        return {}


def _write_last_fetched(state: dict[str, int]) -> None:
    tmp = LAST_FETCHED_PATH.with_suffix(".json.tmp")
    with tmp.open("w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(LAST_FETCHED_PATH)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record, separators=(",", ":")) + "\n")


def _log_poll(
    feed: str, status: str, observed_at: int, *, detail: str | None = None
) -> None:
    record = {
        "observed_at": observed_at,
        "feed": feed,
        "status": status,
    }
    if detail is not None:
        record["detail"] = detail
    _append_jsonl(POLL_LOG_PATH, record)


def _fetch(url: str) -> Any:
    """Fetch a JSON feed from the MTA gateway.

    Per testing 2026-05-11, the camsys/* and nyct/* JSON variants are publicly
    accessible without authentication. Protobuf feeds (trip updates, vehicle
    positions) DO require an MTA API key — when we eventually add those, switch
    to an authenticated client.
    """
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def collect_alerts() -> int:
    """Fetch the subway-alerts JSON and append every entity as one JSONL row."""
    observed_at = _utc_now()
    poll_id = str(uuid.uuid4())
    try:
        payload = _fetch(SUBWAY_ALERTS_URL)
    except (httpx.HTTPError, httpx.HTTPStatusError) as err:
        _LOGGER.error("subway-alerts fetch failed: %s", err)
        _log_poll(
            "camsys/subway-alerts.json", "error", observed_at, detail=str(err)[:200]
        )
        return 0

    entities = payload.get("entity") or []
    daily = ALERTS_DIR / f"{_utc_today()}.jsonl"
    for entity in entities:
        record = {
            "observed_at": observed_at,
            "poll_id": poll_id,
            "feed_source": "camsys/subway-alerts.json",
            "alert": entity,
        }
        _append_jsonl(daily, record)

    _log_poll(
        "camsys/subway-alerts.json",
        "ok",
        observed_at,
        detail=f"appended {len(entities)} entities",
    )
    _LOGGER.info("subway-alerts: appended %s entities to %s", len(entities), daily.name)
    return observed_at


def collect_ene() -> int:
    """Fetch all three E&E feeds and append observations."""
    observed_at = _utc_now()
    poll_id = str(uuid.uuid4())
    daily = ENE_DIR / f"{_utc_today()}.jsonl"
    total = 0
    for url, source in (
        (ENE_CURRENT_URL, "nyct/nyct_ene.json"),
        (ENE_UPCOMING_URL, "nyct/nyct_ene_upcoming.json"),
    ):
        try:
            payload = _fetch(url)
        except (httpx.HTTPError, httpx.HTTPStatusError) as err:
            _LOGGER.error("%s fetch failed: %s", source, err)
            _log_poll(source, "error", observed_at, detail=str(err)[:200])
            continue

        record = {
            "observed_at": observed_at,
            "poll_id": poll_id,
            "feed_source": source,
            "payload": payload,
        }
        _append_jsonl(daily, record)
        _log_poll(source, "ok", observed_at)
        total += 1

    if total:
        _LOGGER.info("ene: appended %s feeds to %s", total, daily.name)
    return observed_at


def _handle_signal(signum: int, _frame: object) -> None:
    global _SHUTDOWN
    _LOGGER.info("Received signal %s; shutting down after current loop", signum)
    _SHUTDOWN = True


def main() -> int:
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _ensure_dirs()
    state = _read_last_fetched()
    _LOGGER.info(
        "Collector starting. data_dir=%s poll_interval=%ss ene_interval=%ss "
        "last_alerts_fetch=%s last_ene_fetch=%s",
        DATA_DIR,
        POLL_INTERVAL,
        ENE_POLL_INTERVAL,
        state.get("alerts", 0),
        state.get("ene", 0),
    )

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    while not _SHUTDOWN:
        now = _utc_now()

        # Always poll alerts on the 5-min cadence
        observed = collect_alerts()
        if observed:
            state["alerts"] = observed

        # Poll E&E only if the hour has elapsed since the last successful fetch
        last_ene = state.get("ene", 0)
        if now - last_ene >= ENE_POLL_INTERVAL:
            observed_ene = collect_ene()
            if observed_ene:
                state["ene"] = observed_ene

        _write_last_fetched(state)

        if _SHUTDOWN:
            break

        # Sleep in 1-second slices so SIGTERM is responsive
        elapsed = 0
        while elapsed < POLL_INTERVAL and not _SHUTDOWN:
            time.sleep(1)
            elapsed += 1

    _LOGGER.info("Collector shut down cleanly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
