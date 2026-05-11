# Momentarily

> "We are being held momentarily by the train's dispatcher."

A normalized snapshot of NYC MTA service status, alerts, and elevator/escalator state — published every few minutes to a public URL for downstream consumers.

The [homeassistant-mta-subway](https://github.com/iicky/homeassistant-mta-subway) integration is the canonical consumer; the snapshot URL is public so anyone (other HA users, custom dashboards, transit hackers) can use it.

**Not affiliated with the MTA.** Built from official feeds at `api.mta.info` per the MTA developer agreement.

## Snapshot URL

```
https://feed.momentarily.nyc/v1/snapshot.json
```

Path-versioned. Breaking schema changes will publish to `/v2/`, `/v3/`, etc.

## What's in the snapshot

- **`alerts`** — every currently-active GTFS-RT alert, with route/stop/direction filtering metadata
- **`routes`** — static per-route metadata (id, color, name)
- **`route_status`** — per-route derived view: active alerts, severity, primary alert_type, per-direction breakdown
- **`stations`**, **`station_status`** — per-station metadata + derived view (alerts affecting the stop, ADA status, equipment outage counts)
- **`equipment`** — elevator/escalator outage state
- **`system`** — top-of-dashboard rollup; one human-readable `overall_label`
- **`compat.subwaynow_routes`** — legacy view matching homeassistant-mta-subway's pre-Momentarily `Route` shape, derived from canonical surfaces. Existing HA installs swap `API_URL` and read this view with zero code changes.

Full schema in [`src/momentarily/schema.py`](src/momentarily/schema.py).

## Upstream sources

All fetched from the MTA developer gateway (`api-endpoint.mta.info`):

| Source | URL | Cadence |
|---|---|---|
| Subway alerts | `…/camsys%2Fsubway-alerts.json` | every 5 min |
| Elevator/escalator (current) | `…/nyct%2Fnyct_ene.json` | hourly |
| Elevator/escalator (upcoming) | `…/nyct%2Fnyct_ene_upcoming.json` | hourly |
| Elevator/escalator (registry) | `…/nyct%2Fnyct_ene_equipments.json` | hourly |
| MTA Subway Stations | NYS Open Data `39hk-dx4f` | daily |

JSON-only for v1 — no protobuf parsing. Trip updates and vehicle positions (protobuf-only) deferred to a later milestone if station-arrival sensors get added downstream.

## Running it

### Local dev (without R2)

```bash
uv sync
uv run pytest
```

Tests exercise the derivation logic against synthetic fixtures — no MTA key required.

### Live publish (your own instance)

1. Get a free MTA API key at [api.mta.info](https://api.mta.info).
2. Create a Cloudflare account, enable R2, create a bucket. Bind it to a custom domain via Cloudflare DNS.
3. In this repo's GitHub Actions secrets, add:
   - `MTA_API_KEY`
   - `R2_ACCOUNT_ID`
   - `R2_ACCESS_KEY_ID`
   - `R2_SECRET_ACCESS_KEY`
   - `R2_BUCKET`
4. The publish workflow runs on cron (`.github/workflows/publish.yml`).

### Self-host without GitHub Actions

A Docker image is planned ([beads-1ct](https://github.com/iicky/homeassistant-mta-subway/issues)). For now, run `python -m momentarily` against any scheduler with the same env vars.

## Status mapping

MTA's alerts feed uses an open-set `alert_type` string. Momentarily maps observed values to a coarse status bucket so downstream consumers have a stable vocabulary; unknown values pass through as their raw label rather than being dropped. See [`src/momentarily/mapping.py`](src/momentarily/mapping.py) for the table.

When a new `alert_type` is seen in production, add a mapping and ship a release. The publisher logs unknown values so they're noticed.

## License & data attribution

**Code:** Apache License 2.0 — see [LICENSE](LICENSE). You can fork, modify, and redistribute the publisher (commercial use included). You must preserve the [NOTICE](NOTICE) file in your distribution so credit follows the code, and you must indicate any significant modifications. Matches the license used by [home-assistant/core](https://github.com/home-assistant/core) and [iicky/homeassistant-mta-subway](https://github.com/iicky/homeassistant-mta-subway).

**Data:** The snapshot content is derived from MTA-operated APIs. MTA owns the data and governs its use through the [MTA developer agreement](https://api.mta.info/#/DataFeedAgreement). MTA's terms are independent of this Apache 2.0 license — if you run any Momentarily instance, you're bound by MTA's terms (you need your own API key, your own attribution to MTA in your snapshot, etc.). See [NOTICE](NOTICE) for the full breakdown.

Momentarily is not affiliated with, endorsed by, or licensed by the MTA.
