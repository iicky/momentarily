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
- **`observations`** — continuous measurements (travel times, headways, ETAs); empty in v1
- **`routes`** — static per-route metadata (id, color, name)
- **`route_status`** — per-route derived view: active alerts, severity, primary alert_type, per-direction breakdown, optional HMM-inferred `condition` + `recovery_minutes`
- **`stations`**, **`station_status`** — per-station metadata + derived view (alerts affecting the stop, ADA status, equipment outage counts)
- **`equipment`** — elevator/escalator outage state
- **`bridges`**, **`tunnels`** — infrastructure scaffolds; populated when a travel-time data source is wired
- **`system`** — top-of-dashboard rollup; one human-readable `overall_label`
- **`compat.subwaynow_routes`** — legacy view matching homeassistant-mta-subway's pre-Momentarily `Route` shape, derived from canonical surfaces. Existing HA installs swap `API_URL` and read this view with zero code changes.

Full schema in [`src/momentarily/schema.py`](src/momentarily/schema.py).

## Method

Momentarily applies a per-line **Hidden Markov Model** with three regimes (normal / disrupted / suspended) to the GTFS-Realtime Mercury alerts stream, producing a probabilistic estimate of each line's current operational state plus expected recovery time. The forward algorithm filters per cron tick; Baum-Welch re-estimates transition matrices and emission parameters weekly from rolling history.

User-facing fields graduate from a shadow-logging phase to the published snapshot only after a calibration review. Grading is prequential and self-consistent: forecasts are scored against the model's own subsequently-published `condition` (the regime-transition stream the recovery metrics use is derived from that same filter output). This is anchored to real MTA alerts — both the forecast and the outcome are driven by the live feed — but it is not yet an independent measure of service recovery (e.g. against GTFS trip-updates). Calibration uses standard probabilistic-forecast tools — reliability diagrams, Brier scores with persistence/climatology skill baselines, quantile bracketing.

The live path runs on Cloudflare — a TypeScript Worker for per-tick inference, a weekly Python training container, R2 as the only state store. See [ADR 0001](docs/adr/0001-cloudflare-workers-r2-only-split-ts-python.md) for the full architecture and why.

Every published artifact records its own provenance. The snapshot, the eval/calibration outputs, and `params.json` each carry a `provenance` block — the git `code_sha` that produced them, a `dirty` flag, and the producer (worker / container / local). `params.json` additionally records the `hyperparams` it was fit with (resolved window, prior strength, min ticks) and a BLAKE3 hash of the exact input-manifest (the immutable alert-version objects that fed the fit). Together these make any model version traceable to a commit and re-derivable from the archive — "which build produced this?" is a one-field lookup, not an investigation.

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

### Local dev

```bash
uv sync
uv run pytest
```

Tests exercise the derivation logic against synthetic fixtures — no MTA key required.

### Live publish

The live publish path is a TypeScript Cloudflare Worker writing to R2 on a Workers Cron Trigger. The Worker is under active development; see the project tracker for status.

This Python package is the offline toolkit — used for HMM training (Baum-Welch EM), calibration notebooks, and as the reference implementation for the Worker's derivation logic. It is not the live publisher.

## Status mapping

MTA's alerts feed uses an open-set `alert_type` string. Momentarily maps observed values to a coarse status bucket so downstream consumers have a stable vocabulary; unknown values pass through as their raw label rather than being dropped. See [`src/momentarily/mapping.py`](src/momentarily/mapping.py) for the table.

When a new `alert_type` is seen in production, add a mapping and ship a release. The publisher logs unknown values so they're noticed.

## License & data attribution

**Code:** Apache License 2.0 — see [LICENSE](LICENSE). You can fork, modify, and redistribute the publisher (commercial use included). You must preserve the [NOTICE](NOTICE) file in your distribution so credit follows the code, and you must indicate any significant modifications. Matches the license used by [home-assistant/core](https://github.com/home-assistant/core) and [iicky/homeassistant-mta-subway](https://github.com/iicky/homeassistant-mta-subway).

**Data:** The snapshot content is derived from MTA-operated APIs. MTA owns the data and governs its use through the [MTA developer agreement](https://api.mta.info/#/DataFeedAgreement). MTA's terms are independent of this Apache 2.0 license — if you run any Momentarily instance, you're bound by MTA's terms (you need your own API key, your own attribution to MTA in your snapshot, etc.). See [NOTICE](NOTICE) for the full breakdown.

Momentarily is not affiliated with, endorsed by, or licensed by the MTA.
