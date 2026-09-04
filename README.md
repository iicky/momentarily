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

## Trains URL

Aggregated live train positions, for map overlays — published separately
from the snapshot so lightweight consumers never pay for it:

```
https://feed.momentarily.nyc/v1/trains.json
```

At ~700 concurrent trips this would add tens of kilobytes to every snapshot
fetch, and the canonical snapshot consumer (homeassistant-mta-subway) never
reads it. `positions` is one entry per (route, direction, stop, stopped)
tuple actually observed; `fresh_feeds`/`expected_feeds` say whether that set
is complete — a NYCT line-group feed can fail independently of the others,
and a consumer needs to tell "zero trains on that line" from "that line's
feed didn't decode this tick" apart. When every feed fails, the object is
left un-rewritten rather than published as a false empty read.

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

User-facing fields graduate from a shadow-logging phase to the published snapshot only after review, and grading is **event-based** — the model is scored per incident episode, not per tick: onset-detection latency, recovery time as a full predicted distribution (CRPS/PIT against a causally-fitted duration-climatology skill baseline), and false-alarm episodes cross-checked against the independent train-movement signal. The older tick-level view (prequential Brier of `p_normal_in_H` against the model's own subsequently-published `condition`, with persistence/climatology baselines and reliability diagrams) is kept only as a secondary calibration read: on the sticky severe-only truth persistence is near-optimal on the no-event majority, so tick Brier vs persistence is a degenerate yardstick and no longer drives graduation. This is anchored to real MTA alerts but is not yet an independent measure of service recovery (e.g. against GTFS trip-updates).

The ground truth for regime grading (the confusion matrix, changepoint alignment, and the offline projection backtest) is **severity-graded**: a route-tick counts as disrupted only when an active alert reaches a severe tier — Severe Delays or a service suspension. Chronic minor alerts (ordinary delays, routine reroutes) and planned work read *normal*, so planned windows are a deterministic schedule overlay rather than stochastic disruption episodes. The legacy breadth truth (any active alert = disrupted) is kept only as a labeled sensitivity. Every eval, backtest, and review artifact records a `truth_version` so any metric traces back to the truth definition that produced it.

A field graduates only if it clears its event-based gate: `condition` (the nowcast) graded as classification against the movement truth; `recovery_minutes` and its band graded per episode with positive CRPS skill over a duration-climatology baseline fitted CAUSALLY, on a window before the graded episodes, over enough incidents. That qualifier is load-bearing: scoring against the graded window's own duration CDF is hindsight, a causally-fitted climatology loses to that reference by 0.1157 skill, and a gate stated against it would fail forecasts for being causal rather than for being wrong. The hindsight figure is still reported for comparability with pre-2026-09 numbers, and never decides. Onset latency is reported but not gated — the alert feed is coincident-to-lagging by construction, so the model cannot lead it and isn't penalized for that. Changepoint alignment is likewise reported, not gated: severe-truth changepoints are far sparser than filter transitions, so a low match rate is mostly arithmetic and is read alongside reverse detection recall.

`p_normal_in_H` is **shadow-only by decision, not pending one**. The deciding diagnostic — the `normal_now` backtest stratum, graded against fixed severity-graded truth rather than the model's own published condition — returned Brier 0.877 (geometric projection) and 0.886 (KM-residual) at 30 minutes against 0.0009 for persistence over 21,071 route-ticks, and the same shape at 60 and 120 minutes. That persistence score pins the outcome: truth stays *normal* through the next 30 minutes on 99.91% of normal-now ticks. A Brier of 0.877 against an outcome that near-certain is 88% of the worst score attainable on the stratum and roughly 3.5× worse than a flat uninformative 0.5 — confidently wrong, not merely unskilled, and far outside the small negative skill a near-certain base rate produces mechanically. Re-baselining does not rescue it: the per-route climatology of "normal at t+H" on this stratum is ~0.999 by construction, and a forecast that constant scores ~0.0009 arithmetically, the same yardstick persistence sets. The KM-residual normal branch scores slightly worse than the plain geometric projection, so that arm is not the fix either. Re-scoping the field means a different primitive (conditional residual survival, length-bias corrected, with abstention in sparse elapsed-time regimes) cleared against its own stated gate, not another run of this one.

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
| MTA Subway Hourly Ridership | NYS Open Data `5wq4-mkjj` | weekly |

The ridership feed is entry-side only — it has no `exits` column — and is reduced offline (`training/ridership.py`) to a per-station-complex entry-rate baseline behind the live platform-crowding estimate; it publishes with roughly a 10-day lag, so the ingest resolves its own trailing window against the feed's own latest available hour rather than against today.

The published v1 snapshot is JSON-derived. The protobuf GTFS-RT feeds (trip updates and vehicle positions) are decoded too, but only for offline HMM validation — each tick archives a per-route service metric (assigned trips) and a movement metric (where trains are, advancing vs stalled across ticks), held out as independent truth for recovery and current-state classification. They do not feed the public snapshot.

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

### Self-hosting

You can run your own publisher — your own snapshot URL on your own Cloudflare account, independent of the iicky-operated instance. No MTA API key needed. See [docs/self-hosting.md](docs/self-hosting.md).

## Status mapping

MTA's alerts feed uses an open-set `alert_type` string. Momentarily maps observed values to a coarse status bucket so downstream consumers have a stable vocabulary; unknown values pass through as their raw label rather than being dropped. The live table is documented in [`worker/README.md`](worker/README.md); [`src/momentarily/mapping.py`](src/momentarily/mapping.py) is the offline reference implementation.

When a new `alert_type` is seen in production, add a mapping and ship a release. The offline drift job tracks the unmapped-`alert_type` rate so new values get noticed.

## License & data attribution

**Code:** Apache License 2.0 — see [LICENSE](LICENSE). You can fork, modify, and redistribute the publisher (commercial use included). You must preserve the [NOTICE](NOTICE) file in your distribution so credit follows the code, and you must indicate any significant modifications. Matches the license used by [home-assistant/core](https://github.com/home-assistant/core) and [iicky/homeassistant-mta-subway](https://github.com/iicky/homeassistant-mta-subway).

**Data:** The snapshot content is derived from MTA-operated APIs. MTA owns the data and governs its use through the [MTA developer agreement](https://api.mta.info/#/DataFeedAgreement). MTA's terms are independent of this Apache 2.0 license — if you run any Momentarily instance, you're bound by MTA's terms (you need your own API key, your own attribution to MTA in your snapshot, etc.). See [NOTICE](NOTICE) for the full breakdown.

Momentarily is not affiliated with, endorsed by, or licensed by the MTA.
