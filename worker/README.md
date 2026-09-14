# Momentarily publisher (Cloudflare Worker)

The live publish path: a TypeScript Worker that fetches MTA GTFS-RT feeds on a
Cron Trigger, runs the HMM derivation, and writes the snapshot to R2.
[`src/momentarily/`](../src/momentarily/) is the offline reference implementation
of the same derivation logic.

## R2 object layout

The Worker's only state store is one R2 bucket. Keys fall under three prefixes.
Only `v1/` is served publicly — the Worker is the auth boundary and the R2
custom domain must NOT be bound directly to the bucket, so `state/` and
`archive/` stay private ([`src/index.ts`](src/index.ts), `PUBLIC_PREFIX`).

The cron fires every minute ([`wrangler.toml`](wrangler.toml)); the full
derivation pipeline runs only when the minute is a 5-minute boundary, while the
1-minute trace, platform-wait, and headway carries run every fire. "Cadence"
below is the write rate, not the object count — the dated `archive/` and `v1/`
streams accumulate one object per write.

### `v1/` — public

| Key | Producer | Cadence |
| --- | --- | --- |
| `v1/snapshot.json` | [`snapshot.ts`](src/snapshot.ts) `publishSnapshot` | every 5-min tick |
| `v1/trains.json` | [`snapshot.ts`](src/snapshot.ts) `publishTrains` | every 5-min tick |
| `v1/arrivals.json` | [`snapshot.ts`](src/snapshot.ts) `publishArrivals` | every minute |
| `v1/predictions/<date>/<observed_at>.jsonl` | [`grading.ts`](src/grading.ts) `writePredictions` | every 5-min tick (skipped when no records) |
| `v1/regime_transitions/<date>/<observed_at>.jsonl` | [`grading.ts`](src/grading.ts) `writeTransitions` | every 5-min tick (only on a regime change) |
| `v1/movement_transitions/<date>/<observed_at>-<scope>.jsonl` | [`grading.ts`](src/grading.ts) `writeMovementTransitions` | every 5-min tick (only on a movement-regime change) |
| `v1/entrances.json` | [`entrances_static.ts`](src/entrances_static.ts) `publishEntrances` | daily |
| `v1/route_shapes.json` | weekly Python trainer ([`publish_params.py`](../training/publish_params.py) `write_route_shapes`) | weekly (on GTFS timetable change) |
| `v1/route_shapes/<feed_version>.json` | weekly Python trainer (immutable per feed_version) | once per feed_version |

`v1/snapshot.json` withholds curve-fitted recovery pending validation (nulled,
with `recovery_withheld: "pending_validation"`) and publishes the schedule
countdowns; `v1/predictions` keeps the full numbers for grading. The gate is
the `PUBLISH_FITTED_RECOVERY` constant in [`snapshot.ts`](src/snapshot.ts).

`deriveArrivals` ([`arrivals.ts`](src/arrivals.ts)) folds the decoded trip-update
stop times into a per-stop `arrivals` surface, and `buildArrivals`/`publishArrivals`
([`snapshot.ts`](src/snapshot.ts), beside `buildTrains`/`publishTrains`) wrap and
write it. It is published on its own `v1/arrivals.json` object every minute — a
countdown up to 5 min stale is not a countdown, so it rides the 1-minute cron
(built from the trip-updates the tick already decodes for the trace, no second
fetch or decode) rather than the 5-minute snapshot. Self-describing like
`trains.json`: its own `observed_at`,
`provenance`, and `fresh_feeds`/`expected_feeds` per-feed liveness, each row an
absolute `eta_epoch` a consumer recomputes against, with a short
`Cache-Control: public, max-age=30, s-maxage=30`. `buildSnapshot` can still
attach the same surface inline when passed, but the cron leaves it off
`v1/snapshot.json` and publishes the dedicated object instead.

`stations` now carries `lat`/`lon` (nullable, from NYS Open Data 39hk-dx4f
`gtfs_latitude`/`gtfs_longitude`).

`v1/prov/v<trained_at>.json` also lives under the public prefix but is written by
the weekly Python trainer, not the Worker; the Worker only derives its public URL
([`params.ts`](src/params.ts) `publicProvUrl`).

`v1/entrances.json` is a standalone daily object with every subway entrance/exit
from NYS Open Data i9wp-a4ja, keyed by `gtfs_stop_id` in `entrances` (direct
match, ~97% of stations) and by `complex_id` in `complex_entrances` (shared
entrances in large complexes whose compound stop id doesn't match a single
station). A consumer joins `complex_entrances` via `Station.station_complex_id`.
The `coverage` block reports both direct and effective station coverage so a
consumer can see completeness without computing the join.

### `state/` — private, read-modify-write carry

Each is read at the start of a tick, mutated, and written back. `last_seen.json`,
`alpha.json`, and `headway.json` use a compare-and-swap conditional put
([`r2.ts`](src/r2.ts) `conditionalPut`) so an overlapping or retried cron cannot
clobber a concurrent writer; the rest are single-writer-per-tick plain puts.

| Key | Producer | Cadence |
| --- | --- | --- |
| `state/last_seen.json` | [`state.ts`](src/state.ts) `writeLastSeen` | every 5-min tick |
| `state/alpha.json` | [`alpha.ts`](src/alpha.ts) `writeAlphaState` | every 5-min tick |
| `state/vehicle_stops.json` | [`state.ts`](src/state.ts) `writeVehicleStops` | every 5-min tick |
| `state/movement_state.json` | [`state.ts`](src/state.ts) `writeMovementState` | every 5-min tick |
| `state/movement_metric.json` | [`state.ts`](src/state.ts) `writeMovementMetric` | every 5-min tick |
| `state/service_metric.json` | [`state.ts`](src/state.ts) `writeServiceMetric` | every 5-min tick |
| `state/segment_flow.json` | [`state.ts`](src/state.ts) `writeSegmentFlow` | every 5-min tick |
| `state/station_flow.json` | [`state.ts`](src/state.ts) `writeStationFlow` | every 5-min tick |
| `state/station_wait.json` | [`state.ts`](src/state.ts) `writeStationWait` (folded by [`crowding.ts`](src/crowding.ts)) | every minute |
| `state/headway.json` | [`state.ts`](src/state.ts) `writeHeadway` (folded by [`headway.ts`](src/headway.ts)) | every minute |
| `state/stations.json` | [`stations_static.ts`](src/stations_static.ts) `writeStationsCache` | daily |

The remaining `state/` objects — `params.json` and `params/v<trained_at>.json`,
`prov/v<trained_at>.json`, `segment_params.json`, `segment_dwell.json`,
`scheduled_headway.json`, `service_baseline.json`, `ridership_baseline.json`,
`service_weight_baseline.json` — are written by the weekly trainer and only
read by the Worker.

### `archive/` — private, append-only corpus

The held-out record the offline HMM validation is graded against
([`archive.ts`](src/archive.ts)). Dated by UTC day; keyed by version or
observation time so an overlapping/retried run overwrites the same object rather
than duplicating it.

| Key | Producer | Cadence |
| --- | --- | --- |
| `archive/alerts/<date>/<updated_at>-<alert_id>.json` | [`archive.ts`](src/archive.ts) `archiveNewAlerts` | every 5-min tick, one object per new `(alert_id, updated_at)` version |
| `archive/alerts_liveness/<date>/<observed_at>.json` | [`archive.ts`](src/archive.ts) `archiveAlertsLiveness` | every 5-min tick |
| `archive/ene/<date>/HH0000-<source>.json` | [`archive.ts`](src/archive.ts) `archiveEneSnapshot` | hourly |
| `archive/trip_updates/<date>/<observed_at>.json` | [`archive.ts`](src/archive.ts) `archiveTripUpdateMetric` | every 5-min tick |
| `archive/vehicles/<date>/<observed_at>.json` | [`archive.ts`](src/archive.ts) `archiveVehicleMetric` | every 5-min tick |
| `archive/trace/<date>/<scheduled_at>.json` | [`archive.ts`](src/archive.ts) `archiveTraceRows` | every minute |

## alert_type → status mapping

MTA's alerts feed uses an open-set string for `alert_type` (`Delays`,
`Service Change`, `Slow Speeds`, `Trains Rerouted`, `Planned – Multiple Changes`,
…). New values can appear without versioning, so the mapping is maintained by
hand and unknown values pass through as their own raw label rather than being
dropped or coerced.

Two axes come out of the mapping, both in [`src/mapping.ts`](src/mapping.ts):

- **`coarseStatus(alertType)`** — a short human label, chosen to preserve the
  entity vocabulary the `homeassistant-mta-subway` integration shipped before the
  Momentarily migration. First substring match wins, so the table is ordered
  most-specific first.
- **`categoryForLabel(label)`** — a stable token for the coarse label, the
  `category` axis on the snapshot. Derived from the label so there is one table to
  maintain, not two.

### Coarse status table

The live table matches by substring (first match wins):

| Substring                | Status         |
| ------------------------ | -------------- |
| `Planned -`              | `Planned Work` |
| `Suspend`                | `Suspended`    |
| `No Trains`              | `Suspended`    |
| `No Scheduled Service`   | `Suspended`    |
| `Severe Delays`          | `Delays`       |
| `Delays`                 | `Delays`       |
| `Reroute`                | `Service Change` |
| `Trains Rerouted`        | `Service Change` |
| `Stops Skipped`          | `Service Change` |
| `Express to Local`       | `Service Change` |
| `Local to Express`       | `Service Change` |
| `Service Change`         | `Service Change` |
| `Boarding Change`        | `Service Change` |
| `Slow Speeds`            | `Slow Speeds`  |
| `Station Notice`         | `Information`   |
| `Special Schedule`       | `Information`   |
| `Information`            | `Information`   |
| _no match_               | raw `alert_type` (passed through) |
| _null / empty_           | `Good Service` (`NO_ALERTS_FALLBACK`) |

### Label → category

| Coarse label   | Category             |
| -------------- | -------------------- |
| `Good Service` | `none`               |
| `Planned Work` | `planned_work`       |
| `Delays`       | `delays`             |
| `Service Change` | `service_change`   |
| `Suspended`    | `service_suspension` |
| `Slow Speeds`  | `slow_speeds`        |
| `Information`  | `information`        |
| _anything else_ | `other`             |

## Relationship to mapping.py

[`src/momentarily/mapping.py`](../src/momentarily/mapping.py) is the offline
reference. It matches `alert_type` by exact dict lookup (plus `Planned*` and
`No … Service` prefix rules); the Worker matches by substring and is the live,
more-complete table. They agree on every `alert_type` currently observed in
production. Known intentional divergences:

- `Slow Speeds` → `Delays` in Python (locked by `tests/test_mapping.py`) vs its
  own `Slow Speeds` label in the Worker, which has a dedicated `slow_speeds`
  category.
- Python carries railroad-only types (`Cancellations`, `Track Change`,
  `Weather`) and `Some Delays`, which the subway-only Worker does not need.

## Maintenance

Revisit this table on each publisher release. The unmapped-`alert_type` rate is
tracked by the offline drift job
([`training/drift.py`](../training/drift.py), via `is_known_alert_type`) over the
predictions stream the Worker writes — a rising rate is the signal that MTA added
a value that needs a row here. When that happens, add the mapping in both
`mapping.ts` and `mapping.py`, extend the table above, and ship a release.
