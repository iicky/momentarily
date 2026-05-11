"""HTTP fetching of upstream MTA feeds.

Stub implementation — wire up to api.mta.info gateway once an MTA_API_KEY is
available. The actual feed URLs are documented in the README under "Upstream
sources".

Design notes:
- Sync httpx (single-shot 5-min cron, no need for async).
- All fetches return raw parsed JSON; transformation into Alert / Equipment
  objects happens in `transform.py` (not yet written — slot for v1).
- Each fetch returns (payload, fetched_at_epoch_seconds) so the snapshot can
  emit a per-source freshness timestamp.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

# MTA gateway base. All endpoints live under here.
MTA_GATEWAY = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds"

# URL-encoded paths for the JSON variants we actually consume in v1.
SUBWAY_ALERTS = f"{MTA_GATEWAY}/camsys%2Fsubway-alerts.json"
ENE_CURRENT = f"{MTA_GATEWAY}/nyct%2Fnyct_ene.json"
ENE_UPCOMING = f"{MTA_GATEWAY}/nyct%2Fnyct_ene_upcoming.json"
ENE_EQUIPMENTS = f"{MTA_GATEWAY}/nyct%2Fnyct_ene_equipments.json"

REQUEST_TIMEOUT = httpx.Timeout(20.0)


def _client(api_key: str) -> httpx.Client:
    return httpx.Client(
        timeout=REQUEST_TIMEOUT,
        headers={"x-api-key": api_key},
        follow_redirects=True,
    )


def fetch_json(url: str, api_key: str) -> tuple[Any, int]:
    """Return (parsed JSON, fetched_at_epoch_seconds). Raises on HTTP errors."""
    with _client(api_key) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json(), int(time.time())
