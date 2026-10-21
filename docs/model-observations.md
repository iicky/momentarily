# Data model, layer 2: raw observations (`observations/s1`) — scoping proposal

Status: the owner's decisions are in §8. The contracts, schema export, TS
generation and fixtures (slice 2) are built; §3.4 lists where the build
changed the draft. Nothing else here is built yet.

Layer 2 records what each upstream source said, when we asked, and what came
back, including failures. It keeps upstream identities verbatim: realtime trip
ids, route ids as published (`6X`, `SS`), realtime-only stop ids (`R60N`), and
alert ids. Matching to the network, service-day assignment, and every judgment
("is this train real", "did the alert end") belong to layer 3.

All sizes below come from measurements made on 2026-10-20, unless the row
names another source, using local measurement scripts that are not part of
the repo. Live
sample: 40 polls at 15 s intervals of every source, 14:36–14:46 UTC
(10:36 ET, a Tuesday). Archive figures use the local R2 mirror, median of
2026-10-01..10-18. `[inference]` marks a claim that was not measured.

## 1. Inventory

### 1.1 Upstream sources the Worker reads

| Source (Worker name) | URL (gateway = `api-endpoint.mta.info/Dataservice/mtagtfsfeeds`) | Worker cadence | Measured payload (decoded / gzip) | What changes between polls |
|---|---|---|---|---|
| GTFS-RT `ace` | `nyct%2Fgtfs-ace` | 1 min | 112 KB / 21 KB, 257 entities, 2,434 stop-time updates | every poll (header ts moves every ≤15 s) |
| GTFS-RT `bdfm` | `nyct%2Fgtfs-bdfm` | 1 min | 176 KB / 34 KB | every poll |
| GTFS-RT `g` | `nyct%2Fgtfs-g` | 1 min | 20 KB / 3.8 KB | every poll |
| GTFS-RT `jz` | `nyct%2Fgtfs-jz` | 1 min | 23 KB / 4.5 KB | every poll |
| GTFS-RT `nqrw` | `nyct%2Fgtfs-nqrw` | 1 min | 132 KB / 25 KB | every poll |
| GTFS-RT `l` | `nyct%2Fgtfs-l` | 1 min | 31 KB / 5.9 KB | every poll |
| GTFS-RT `numbered` | `nyct%2Fgtfs` | 1 min | 180 KB / 40 KB | every poll |
| GTFS-RT `si` | `nyct%2Fgtfs-si` | 1 min | 6 KB / 1.5 KB | every poll |
| **All 8 GTFS-RT** | | 1 min | **679 KB / 136 KB per minute**, 660 vehicle entities, 16,081 stop-time updates | 0 of 320 payloads repeated |
| Alerts JSON `alerts` | `camsys%2Fsubway-alerts.json` | 5 min (boundary ticks only) | 908 KB / 73 KB, 179 entities (1 `lmm:alert`, 178 `lmm:planned_work`) | unchanged across all 40 polls (header ts 14:30:14) |
| ENE outages `ene_current` | `nyct%2Fnyct_ene.json` | 1 h | 60 KB / 4.5 KB, 146 records | 2,129 of 2,142 archived hours distinct |
| ENE upcoming `ene_upcoming` | `nyct%2Fnyct_ene_upcoming.json` | 1 h | 42 KB / 2.9 KB, 103 records | 365 distinct (as a set) of 2,141 |
| ENE catalog `ene_equipments` | `nyct%2Fnyct_ene_equipments.json` | 1 h | 645 KB / 65 KB, 707 records | **20 distinct of 2,141** |
| Static GTFS | `rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip` | HEAD every minute, GET on ETag change | 5.6 MB zip | ETag `bd592a…` since 2026-08-27 |
| Stations (data.ny.gov `39hk-dx4f`) | Socrata JSON | 1 day | 285 KB / 23 KB, 496 rows | Last-Modified 2026-10-20 04:17 |
| Entrances (data.ny.gov `i9wp-a4ja`) | Socrata JSON | 1 day | 901 KB / 54 KB, 2,120 rows | Last-Modified 2025-12-05 |

The MTA gateway sends no `ETag` or `Last-Modified`. Only `Date` and
`x-cache: Miss from cloudfront` come back, so the feed header timestamp is the
only upstream clock. The alerts feed is also published as protobuf
(`camsys%2Fsubway-alerts`, 442 KB). The Worker does not read it.

Per-day volume scales with trains in service. On 2026-10-13/14 the mean
vehicle count over the day was 0.80 of its value at the sample time. Weekend
days run lower (391 and 355 against 551). The weekly mean is **0.73 × the
sample**, so "per day" below means 1,440 × 0.73 = 1,051 sample-minutes.

### 1.2 What the live feeds contain (verified against the brief)

Field census of 320 GTFS-RT payloads (`pb_census.json`, `detail.json`):

- **Vehicle entities.** `ace`, `bdfm`, `g`, `jz`, `nqrw` and `si` send exactly
  one VehiclePosition for every TripUpdate. That is why `vehicles_n == trips_n`
  on those feeds. `l` is the exception among non-numbered feeds: it sends a
  vehicle only for assigned trips (850 vehicles, 1,371 trips). So does
  `numbered` (6,306 and 9,958). On the six one-per-trip feeds, a vehicle for an
  **unassigned** trip is a placeholder. It has no `current_status`, and its
  `timestamp` is a median 29 min in the future, which looks like the scheduled
  departure [inference]. Assigned vehicles report a timestamp a median 17–20 s
  old. On `si`, every trip is flagged assigned, placeholders included (median
  vehicle timestamp 955 s in the future). So "only 1–7 and L carry real
  positions" is too strong. Every feed carries real positions for dispatched
  trains, and six feeds add one placeholder per undispatched trip.
- **`current_status`.** On `ace`, `bdfm`, `jz` and `nqrw`, every assigned
  vehicle reported `STOPPED_AT` in every poll. On `g`, 27 of 442 had no status.
  `numbered` uses `INCOMING_AT` (842), `STOPPED_AT` (3,023) and absent (2,441).
  `l` uses `STOPPED_AT` and `IN_TRANSIT_TO`. The Worker's comment that NYCT
  "only ever emits STOPPED_AT" holds only for the B-division feeds.
- **Producer differences.** Only `l` sends `incrementality`, `is_deleted`,
  trip `schedule_relationship`, `direction_id`, `stop_sequence`, arrival and
  departure `delay` and `uncertainty`, `TripUpdate.timestamp`, and
  `congestion_level`. `l` and `numbered` omit `start_time`. The B-division
  feeds and `si` send `departure_occupancy_status` (field 7) on every
  stop-time update. Some `delay` values are negative and arrive as 64-bit
  two's-complement varints, so a decoder must read them as signed.
- **NYCT extensions.** The header carries `nyct_feed_header`: a required
  `nyct_subway_version` (`1.0`) and a repeated `trip_replacement_period`,
  each an optional `route_id` and an optional `TimeRange` within which the
  realtime feed replaces the route's scheduled trips. Every feed sends one
  period per route, in a fixed order (`ace`: A C E H; `numbered`: 1–7 S).
  `numbered` omits the start and its end equals the header timestamp, not
  the proto comment's "now + 30 minutes"; the B-division feeds and `si`
  omit the start and end about 60 minutes on; `l` sends both, 30 minutes
  apart. Trips carry `train_id` (e.g. `1A 0904 207/FAR`), `is_assigned`
  (absent on about half of the B-division trips), and `direction`. Stop-time
  updates carry `scheduled_track` and `actual_track`.
- **Trip-level alerts.** `numbered` carries about one Alert entity per poll
  ("Train delayed"), with a trip descriptor in `informed_entity`.
- **Path codes.** Trip ids with `X` in the path part: `bdfm` 12%, `numbered`
  12%, `jz` 47%, `si` 52%. No `6X`/`7X`/`FX` or `SS` route appeared in this
  off-peak sample. The holdup audit saw `SS` only at weekday rush hours.
- **Alerts JSON.** The header has `gtfs_realtime_version`, `incrementality`,
  `timestamp` and `mercury_feed_header`. Every entity has `active_period`,
  `informed_entity`, `header_text`, `description_text` and `mercury_alert`
  (`created_at`, `updated_at`, `alert_type`, `display_before_active`,
  `human_readable_active_period`, sometimes `station_alternative` and
  `affected_stations`). 952 of 1,625 informed entities carry `direction_id`,
  1,612 carry `stop_id`. Today's sample is almost all planned work, and it is
  consistent with the holdup finding that Delays alerts lost `stop_id` on
  2026-08-11.

### 1.3 Current archives and streams

Sizes are logical bytes from the mirror (median day, 10-01..10-18). Retention
comes from `training/prune.py`.

| R2 prefix | Writer, cadence | What it holds | Size/day | Objects/day | Retention |
|---|---|---|---|---|---|
| `archive/trace/` | Worker, 1 min | `{observed_at, scheduled_at, fresh_feeds, feed_digest, feed_etag, rows:[{trip_id, route_id, direction, stop_id, stop_seq, stopped, vehicle_ts}]}` | ~90 MB (prune.py) | 1,440 | 120 d |
| `archive/arrivals/` | Worker, 1 min | gzip, first 4 trains per stop: `{route, eta_epoch, trip_id}` | ~31 MB (prune.py) | 1,440 | 120 d |
| `archive/trip_updates/` | Worker, 5 min | per-route counts `assigned_n, trips_n, with_movement_n, dir_n, dir_s` + `fresh_feeds` | 0.56 MB | 288 | 3650 d |
| `archive/vehicles/` | Worker, 5 min | per-route counts + cross-tick `from>to` transition counts + `fresh_feeds` | 3.1 MB | 288 | 3650 d |
| `archive/alerts/` | Worker, 5 min | one object per new `(alert_id, updated_at)`: `{observed_at, alert: entity}` | 0.30 MB | 196 | 90 d |
| `archive/alerts_liveness/` | Worker, 5 min | `{observed_at, outcome, fetched_at}` | 20 KB | 288 | 3650 d |
| `archive/ene/` | Worker, 1 h | `{observed_at, source, payload}` for all 3 ENE feeds | **14.0 MB** | 69 | 90 d |
| `archive/health/` | Worker, 1 min | `{observed_at, write_failures}` | 66 KB | 1,434 | 3650 d |
| `archive/movement_census/` | Worker, 5 min | route regime census | 0.54 MB | 288 | 3650 d |
| `archive/gtfs/` | Worker on ETag change + nightly | static zip, sha256-addressed, `by_etag/`, `latest.json` | 5.6 MB per release | — | never |
| `archive/traversals/` | nightly trainer | per-hop traversals derived from trace | 2.0 MB | 1 | 3650 d |
| `archive/windows/` | nightly trainer | planned-work windows by observation day | 1.3 KB | 1 | 3650 d |
| `v1/predictions/` | Worker, 5 min | per-route inference incl. `primary_alert_type` | 5.1 MB | 288 | 90 d |
| `v1/regime_transitions/` | Worker | regime changes | 27 KB | 96 | 90 d |
| `v1/movement_transitions/` | Worker, 5 min | movement transitions | 0.78 MB | 1,232 | 90 d |
| `state/stations.json`, `v1/entrances.json` | Worker, 1 day | overwritten in place | — | — | none kept |

### 1.4 What the Worker's reduction drops

GTFS-RT (`gtfsrt.ts` reads only the fields listed in its header comment):

| Level | Dropped | Changed |
|---|---|---|
| FeedHeader | everything: `gtfs_realtime_version`, `incrementality`, `timestamp`, NYCT `trip_replacement_period` | — |
| FeedEntity | `id`, `is_deleted`, every `alert` (trip-level "Train delayed") | — |
| TripDescriptor | `start_time`, `start_date`, `schedule_relationship`, `direction_id`, NYCT `train_id` | `is_assigned` absent → `false`. `route_id` folded `6X/7X/FX` → `6/7/F` (`baseRoute`) before every archive |
| TripUpdate | `timestamp`, `delay`, `vehicle` | — |
| StopTimeUpdate | `stop_sequence`, arrival/departure `delay` and `uncertainty`, `departure_occupancy_status`, NYCT `scheduled_track`, `actual_track` | `schedule_relationship` absent → 0. Missing `stop_id` → `''` |
| VehiclePosition | `trip.start_*`, `is_assigned` (never carried in `VehicleLite`), NYCT `train_id`, `congestion_level`, entity id | `current_status` collapsed to `stopped` (0, 2 and absent all become `false`). `stop_seq` kept only when stopped |

On top of that, the trace drops vehicles with an empty `trip_id` or `stop_id`
and records which feeds succeeded but not when, why one failed, the header
timestamp, or the bytes. Because `is_assigned` is not carried, trace rows
include the placeholder vehicles of §1.2 and nothing marks them. The arrivals
sample keeps 4 trains per stop within 60 min, drops SKIPPED and past rows,
and merges arrival and departure (`arrival ?? departure`).

Alerts: the archive keeps each entity verbatim, but only the first time its
`(alert_id, updated_at)` is seen. It drops the feed header, the set of alerts
present at each fetch, the moment an alert leaves the feed, any content change
that does not bump `updated_at`, and every minute that is not a 5-minute
boundary. If `state/last_seen.json` is lost, a version is written again with a
later `observed_at`. So `observed_at` is "a sighting of this version", normally
the first.

ENE: payloads are complete but re-serialized (`JSON.stringify`), so the bytes
are not upstream's. A failed fetch writes nothing, and two runs in one hour
overwrite one key. 1,228 MB of the 1,319 MB under `archive/ene/` are identical
copies of the equipment catalog.

**Is any raw feed archived unreduced today?** No realtime payload is kept as
fetched. The only byte-exact raw archive is the static GTFS zip
(`archive/gtfs/`). ENE comes closest: full content, hourly, re-serialized,
with no failure record.

## 2. Storage options

Measured per sample-minute for all 8 GTFS-RT feeds, then per day (× 1,051):

| Option | Per minute | Per day | Notes |
|---|---|---|---|
| (a) raw protobuf, plain | 679 KB | 714 MB | |
| (a) raw, gzip-6 per payload | 136 KB | **143 MB** | what a Worker can write (`CompressionStream`) |
| (a) raw, zstd-19 per payload | 79 KB | 83 MB | Workers have no zstd |
| (a) raw, zstd-19 `--long`, one 10-min bundle per feed | 28.6 KB | **30 MB** | what a nightly repack reaches |
| (b) VehicleObservation, gzip JSONL | 15.8 KB | 17 MB | 660 rows/min |
| (b) VehicleObservation, parquet-zstd (10-min file) | 5.2 KB | 5.5 MB | |
| (b) TripUpdateObservation, gzip JSONL | 192 KB | 202 MB | 16,081 stop-time rows/min, bigger than raw |
| (b) TripUpdateObservation, parquet-zstd (10-min file) | 47.8 KB | 50 MB | |

Content addressing: none of 320 GTFS-RT payloads repeated, even after
removing the header. 31–62% of entity bytes repeat from one 15-s poll to the
next (0.5% on `l`), which is where the `--long` bundle gains. Whole-payload
dedupe pays off only for the alerts JSON (no change in 10 min) and ENE (the
catalog: 20 distinct payloads in 92 days, 11.5 MB instead of 1,228 MB).

Other sources, per day: alerts raw gzip, written only on change, 118–288
changes × 73 KB = **8.6–21 MB** (118 = median count of 5-min ticks per day
that saw a new alert version, a lower bound). ENE content-addressed: 1.0 MB
(the unique bytes of the 92-day archive). Fetch metadata: about 2.5 KB per
tick, 3.6 MB uncompressed [inference: size of the sketch in §3].

**Recommendation: (c), split by writer.**

- The Worker records **raw payload bytes for every fetch, plus fetch
  metadata**, and nothing else. GTFS-RT goes into one object per tick. Repeating
  sources (alerts, ENE, stations, entrances) go into content-addressed blobs.
- Python decodes the raw payloads **offline** into the normalized contracts of
  §3 and stores them as hourly parquet files.

Reasons:

1. One decoder. The normalized records come from one Python implementation,
   so no TypeScript/Python parity is needed for them. The current Worker
   decoder has three faults a replay could fix after the fact: `is_assigned`
   absent read as false, `current_status` collapsed, and the arrival/departure
   merge. With raw bytes kept, a decoder fix replays history. Without them,
   history stays wrong.
2. Normalized stop-time rows written per minute as JSON cost more than the raw
   protobuf (202 MB/day against 143 MB/day) and need far more Worker CPU.
3. The Worker stays a recorder. It adds about 5 ms of CPU per tick (§4) and no
   decoding beyond what it does today.
4. Fetch metadata separates "no data" from "nothing happening" (§3.1). A
   missing tick means the collector did not run. A failed fetch carries its
   outcome. An `ok` fetch with zero alerts means upstream said there were none.

## 3. Draft contracts

Conventions are layer 1's (`momentarily.model.conventions`): `Model` (frozen,
closed, no defaults), `Instant` (UTC epoch s), `Digest`, `Unknown`,
`Interval`, `Provenance`. Schema id `observations/s1`. Pydantic is the source
of truth (`momentarily.model.observations`). JSON Schema goes to
`schema/model/s1/observations/`, TypeScript to `shared/model/s1/observations/`.
The code blocks below are the draft; §3.4 lists what the build changed.

Three states for every upstream field:

- a value: the payload carried it;
- `None`: the payload omitted it (GTFS-RT is proto2, so presence is knowable);
- `Unknown(reason)`: a legacy archive did not keep it, or the decoder could
  not read it. A raw-lineage record may hold `Unknown` only in the fields its
  contract lists as decode failures (§3.4); collector metadata
  (`FeedFetch`, `CollectorTick`) never holds it.

Time fields, used the same way everywhere:

- `scheduled_at`: the cron minute this attempt served.
- `started_at_ms`, `finished_at_ms`: the Worker clock around the HTTP request.
- `observed_at`: `finished_at_ms // 1000`. It is the instant we learned the
  content, and it is the partition key.
- `header_timestamp`: the upstream's own claim of when it built the payload.
- Entity timestamps, kept verbatim: `vehicle_timestamp`,
  `trip_update_timestamp`, alert `created_at` and `updated_at`, and predicted
  `arrival_time` and `departure_time`.

Layer 2 never assigns a ServiceDate. `start_date` stays the published
`YYYYMMDD` string.

### 3.1 Written by the Worker

```python
SCHEMA_ID = "observations/s1"
SourceId = Literal[
    "ace", "bdfm", "g", "jz", "nqrw", "l", "numbered", "si",      # GTFS-RT
    "alerts",                                                      # camsys JSON
    "ene_current", "ene_upcoming", "ene_equipments",
    "gtfs_static", "stations", "entrances",
]
FetchOutcome = Literal[
    "ok",             # HTTP 2xx, body read, parsed as the expected format
    "empty",          # HTTP 2xx, zero entities / records (payload still stored)
    "unparseable",    # HTTP 2xx, body is not valid protobuf / JSON (payload stored)
    "http_error",     # non-2xx (status recorded)
    "timeout",        # the 10 s fetch budget expired
    "network_error",  # DNS / connect / reset
]

class FeedFetch(Model):
    fetch_id: str = Field(description="f'{source}@{started_at_ms}'; unique per attempt.")
    source: SourceId
    url: str
    method: Literal["GET", "HEAD"]
    scheduled_at: Instant
    started_at_ms: int
    finished_at_ms: int | None = Field(description="null on timeout.")
    outcome: FetchOutcome
    http_status: int | None
    content_type: str | None
    http_date: Instant | None
    http_last_modified: Instant | None
    http_etag: str | None
    body_bytes: int | None = Field(description="Decoded body length.")
    payload_sha256: Digest | None = Field(description="sha256 of the decoded body bytes.")
    payload_key: str | None = Field(description="R2 key holding the bytes; null if not stored.")
    payload_member: int | None = Field(description="Member index inside a tick bundle; null for blobs.")
    header_timestamp: Instant | None
    entity_count: int | None = Field(description="Top-level entities (GTFS-RT, alerts) or records (ENE, Socrata).")
    store_error: str | None = Field(description="Why the bytes were not stored, when they were not.")

class CollectorBuild(Model):
    code_sha: str
    code_dirty: bool | None

class CollectorTick(Model):
    """One Worker invocation. Absence of a tick for a minute = collector gap."""
    schema_: str = Field(alias="schema")
    tick_id: str = Field(description="f'{scheduled_at}-{started_at_ms}'; retries keep both.")
    scheduled_at: Instant
    started_at_ms: int
    finished_at_ms: int
    collector: CollectorBuild
    planned_sources: list[SourceId] = Field(description="What this tick was due to fetch.")
    fetches: list[FeedFetch]
```

`sha256` rather than layer 1's blake3: Workers have `crypto.subtle` SHA-256
natively and no blake3. It is also the `archive/gtfs/` key already.

### 3.2 Derived offline from raw payloads (Python)

```python
class TripDescriptorObs(Model):
    trip_id: str | None
    route_id: str | None = Field(description="As published: '6X', 'SS', never folded.")
    start_time: str | None = Field(description="Published 'HH:MM:SS'; may exceed 24h.")
    start_date: str | None = Field(description="Published 'YYYYMMDD'; not a ServiceDate.")
    schedule_relationship: int | None
    direction_id: Literal[0, 1] | None
    nyct_train_id: str | None
    nyct_is_assigned: bool | None = Field(description="null when the extension omits it.")
    nyct_direction: Literal[1, 2, 3, 4] | None

class VehicleObservation(Model):
    fetch_id: str
    source: SourceId
    observed_at: Instant
    entity_id: str | Unknown
    trip: TripDescriptorObs
    stop_id: str | None = Field(description="As published, realtime-only ids kept.")
    current_stop_sequence: int | None
    current_status: Literal[0, 1, 2] | None
    vehicle_timestamp: Instant | None = Field(description="Verbatim; may be in the future.")
    congestion_level: int | None
    unmodeled_fields: list[str] = Field(description="Field paths present but not decoded, e.g. '2' (position).")

class StopTimePrediction(Model):
    index: int = Field(ge=0, description="Position in the trip's stop_time_update list.")
    stop_id: str | None
    stop_sequence: int | None
    arrival_time: Instant | None
    arrival_delay: int | None
    arrival_uncertainty: int | None
    departure_time: Instant | None
    departure_delay: int | None
    departure_uncertainty: int | None
    schedule_relationship: int | None
    departure_occupancy_status: int | None
    nyct_scheduled_track: str | None
    nyct_actual_track: str | None

class TripUpdateObservation(Model):
    fetch_id: str
    source: SourceId
    observed_at: Instant
    entity_id: str | Unknown
    trip: TripDescriptorObs
    trip_update_timestamp: Instant | None
    delay: int | None
    stop_times: list[StopTimePrediction]
    unmodeled_fields: list[str]

class FeedHeaderObservation(Model):
    fetch_id: str
    source: SourceId
    observed_at: Instant
    gtfs_realtime_version: str  # required upstream
    incrementality: int | None
    header_timestamp: Instant | None
    feed_version: str | None
    nyct_feed_header: NyctFeedHeaderObs | None  # None: no NYCT extension

class NyctFeedHeaderObs(Model):
    nyct_subway_version: str  # required upstream
    trip_replacement_periods: list[TripReplacementPeriodObs] = Field(
        description="Every published period, in published order; a route may repeat.")

class TripReplacementPeriodObs(Model):
    route_id: str | None
    replacement_period: TimeRangeObs | None

class TimeRangeObs(Model):
    start: Instant | None  # None: omitted, an open bound
    end: Instant | None

class AlertEntry(Model):
    alert_id: str
    content_digest: Digest = Field(description="sha256 of the entity's canonical JSON.")
    updated_at: Instant | None

class AlertPresence(Model):
    """The full alert set at one successful fetch: the fix for alert end times."""
    fetch_id: str
    source: Literal["alerts", "numbered"]
    observed_at: Instant
    header_timestamp: Instant | None
    entries: list[AlertEntry] = Field(description="Sorted by alert_id; [] = feed said none.")

class AlertVersion(Model):
    """One distinct alert content, and the window in which it was seen."""
    source: Literal["alerts", "numbered"]
    alert_id: str
    content_digest: Digest
    entity: dict[str, JsonValue] = Field(description="Verbatim entity.")
    alert_type: str | None
    created_at: Instant | None
    updated_at: Instant | None
    absent_before: Instant | None | Unknown = Field(
        description="Last fetch (outcome ok or empty) before first_seen without this version; null = first fetch of the window.")
    first_seen: Instant | Unknown
    last_seen: Instant | Unknown
    gone_by: Instant | None | Unknown = Field(
        description="First fetch (outcome ok or empty) after last_seen without this version; null = still present at window end.")
    gone_kind: Literal["removed", "superseded"] | None
    unobserved: list[Interval] = Field(
        description="Gaps inside [first_seen, gone_by) with no fetch of outcome ok or empty.")
    window: Interval = Field(description="The presence window this index was built over.")

class EquipmentObservation(Model):
    fetch_id: str
    source: Literal["ene_current", "ene_upcoming"]
    observed_at: Instant
    equipment_id: str
    record: dict[str, str] = Field(description="The record verbatim.")
    outage_start_local: str | None = Field(description="'outagedate' as published (ET wall time).")
    estimated_return_local: str | None
    outage_start: Instant | None | Unknown = Field(description="Parsed in America/New_York; Unknown if ambiguous or unparseable.")
    estimated_return: Instant | None | Unknown
```

Keys and rules:

- Natural keys: VehicleObservation and TripUpdateObservation use
  `(fetch_id, entity_index)`. A trip id can repeat inside one payload, so it
  is not a key. AlertPresence uses `fetch_id`. AlertVersion uses
  `(source, alert_id, content_digest, first_seen)`: one row per run of
  sightings, so a version that leaves and comes back is two rows (§4.2).
  EquipmentObservation uses
  `(fetch_id, equipment_id, record_index)`.
- An alert's end is **bounded, not guessed**. The version left the feed in
  `(last_seen, gone_by]`. When `unobserved` overlaps that bound, the end is
  only known to lie in the wider span. A version gone only by `superseded`
  continues as the same alert id.
- `content_digest` hashes the parsed entity, so a content change without an
  `updated_at` bump becomes a new version (the current archive misses that).
  The digest also matches between live JSON and the re-serialized legacy
  archive. The canonical form is sorted keys, no whitespace, non-ASCII kept,
  UTF-8 (`canonical_entity_json`); AlertVersion rejects a digest that does
  not hash its `entity`.
- The ENE catalog (`ene_equipments`) is stored raw and referenced by
  `FeedFetch`. A typed catalog belongs to a layer-1 extension, not to layer 2.
- Provenance on every derived file: `inputs` maps raw keys to digests,
  `code_sha`, and a decoder version. `built_at` = the latest `observed_at` in
  the file, never the wall clock, so a rebuild gives the same bytes.

### 3.3 R2 layout and partitioning

All keys sit under `model/observations/s1/` and are partitioned by the **UTC
hour of observation**. Service day is assigned in layer 3.

```
ticks/<YYYY-MM-DD>/<HH>/<scheduled_at>-<started_at_ms>.json      CollectorTick            (Worker)
raw/gtfsrt/<YYYY-MM-DD>/<HH>/<scheduled_at>-<started_at_ms>.bin  8 gzip members + index   (Worker)
raw/blob/<YYYY-MM-DD>/<sha256>.<json|pb|zip>.gz                  alerts, ENE, Socrata     (Worker)
derived/<record>/<YYYY-MM-DD>/<HH>.parquet                       vehicle, trip_update, feed_header,
                                                                 alert_presence, equipment (Python)
derived/alert_version/<YYYY-MM-DD>.parquet                       AlertVersion, window = that day ± carry
derived/legacy_route_alert_state/<YYYY-MM-DD>.parquet            LegacyRouteAlertState     (Python)
derived/legacy_fetch_outcome/<YYYY-MM-DD>.parquet                LegacyFetchOutcome        (Python)
derived/legacy_collector_health/<YYYY-MM-DD>.parquet             LegacyCollectorHealth     (Python)
```

Ticks and bundles are filed under the UTC hour of `scheduled_at`, so the
directory follows from the file name. A blob is filed under the UTC day of
the fetch that stored it. `momentarily.model.observations.keys` formats and
parses every key, and rejects a key whose directory disagrees with its file
name.

Each GTFS-RT tick bundle is a series of independent gzip members, one per
feed, preceded by a small JSON index member (`TickBundleIndex`) that repeats
that tick's GTFS-RT `FeedFetch` rows. If the tick record is lost, the bundle
still describes itself. Blobs are deduplicated **within a UTC day** (HEAD,
then PUT if missing). A day-dated key lets prune age blobs like everything
else, with no reference-counting GC. The cost is one copy of the ENE catalog
per day (65 KB gzip). Static GTFS stays in `archive/gtfs/`, and its
`FeedFetch` points there.

### 3.4 As built (slice 2)

The build follows §3.1–3.3 except where listed here.

- **Natural-key fields.** `entity_index` (VehicleObservation,
  TripUpdateObservation) and `record_index` (EquipmentObservation) are
  fields. The draft named them as keys but had no field for them. Legacy
  rows key on `(lineage.source_key, lineage.row_index)` instead.
- **Narrower sources.** Vehicle, trip-update and header records take only a
  GTFS-RT source. `VehicleObservation.trip` is nullable, because
  VehiclePosition's trip is optional in GTFS-RT.
- **`TickBundleIndex`** is the contract of the bundle's index member: the
  tick id and the tick's GTFS-RT fetches, each pointing into this bundle.
- **Collector metadata is live only.** `FeedFetch` and `CollectorTick` keep
  every field required and known, and hold no `Unknown`. What the old
  archives kept about fetches and ticks goes to separate legacy contracts
  (below).
- **Lineage on every observation record.** Each record carries `lineage`,
  tagged by `kind`:
  - `raw`: `payload_key` (the tick bundle or blob), `payload_member` (bundle
    member, null for a blob) and `payload_sha256`. GTFS-RT records point into
    a bundle, alerts JSON and ENE records into a blob filed under the
    record's UTC day.
  - `legacy`: `archive` (which legacy prefix), `source_key` (the object),
    `row_index` (position in that object, null when it holds one record) and
    `period`.

  A consumer filters legacy rows on `lineage.kind` alone; in parquet the
  lineage is a struct with the `kind` tag and one nullable sub-struct per
  variant. Only records a legacy archive can supply take a legacy lineage:
  VehicleObservation (`archive/trace`), AlertVersion (`archive/alerts`) and
  EquipmentObservation (`archive/ene`). TripUpdateObservation,
  FeedHeaderObservation and AlertPresence are raw only.
- **Legacy unknowns.** The fields §5 lists as lost admit `Unknown`:
  VehicleObservation `fetch_id`, `source`, `entity_index`, `entity_id`,
  `current_stop_sequence`, `current_status`, `congestion_level`,
  `unmodeled_fields`; every TripDescriptorObs field except `trip_id`;
  EquipmentObservation `fetch_id`; AlertVersion `absent_before`,
  `first_seen`, `last_seen`, `gone_by`, `gone_kind`, `unobserved`, `window`.
  A raw-lineage record holding `Unknown` anywhere else than its decode
  failures is rejected. The only decode failures today are
  EquipmentObservation `outage_start` and `estimated_return` (ambiguous or
  unparseable ET wall times).
- **Legacy period** (decision 5, applied to every legacy record):
  `start` is the first UTC day the archive covers, `end` (exclusive) the
  first UTC day the recorder wrote the replacing record, null until it has.
  Both are checked as calendar dates. A legacy record whose day falls outside
  its period is rejected.
- **`LegacyRouteAlertState`** has a `v1/predictions` lineage (`source_key`
  is the prediction object holding its tick, `row_index` its line) and a
  `granularity` of fixed literals: `scope: "route"`, `tick_seconds: 300`,
  `content: "primary_alert_type_presence"`. `primary_alert_type` null means
  the route had an inference row and no alert.
- **`LegacyFetchOutcome`** (id `f"{source_key}#{source}"`): one source's
  fetch result as `archive/trace`, `archive/trip_updates`,
  `archive/vehicles` (`fresh_feeds`), `archive/alerts_liveness` or
  `archive/ene` (presence of a success) kept it: `succeeded`,
  `observed_at`, `scheduled_at` (trace only, else `Unknown`) and
  `last_success_at` (`alerts_liveness` `fetched_at`, else `Unknown`).
- **`LegacyCollectorHealth`** (id `f"archive/health@{observed_at}"`): one
  `archive/health` object, `observed_at` and `write_failures` (failed writes
  per kind, `{}` on a clean tick).
- **AlertVersion bounds.** "Fetch" in the bounds means outcome `ok` or
  `empty`: an empty feed is evidence that the alert is gone, not a gap.
  `absent_before < first_seen <= last_seen < gone_by`. `gone_kind` is null
  exactly when `gone_by` is null and known when `gone_by` is an instant.
  With `gone_by` `Unknown` it is `Unknown`, or `superseded` when a later
  `updated_at` of the same id proves it (§5); `removed` cannot be proven
  then. `unobserved` gaps are non-empty, sorted, disjoint, and inside the
  sighting. `alert_type`, `created_at` and `updated_at` must equal the
  entity's `mercury_alert` fields.
- **Alert digest.** The canonical form writes integral floats as ints
  (`1.0` -> `1`, as `JSON.stringify` does) and rejects NaN and infinities.
  Python and JavaScript format non-integral floats below 1e-4 or from 1e16
  up differently; alert entities carry none [inference].
- **`FeedFetch` invariants.** `fetch_id` and `tick_id` formats;
  `finished_at_ms` null exactly on timeout; a status of the class each
  outcome implies (a timeout or reset after 2xx headers keeps the 2xx);
  header fields only with a status; HEAD only for `gtfs_static`. A GET whose
  body was read has `body_bytes`, `payload_sha256` and exactly one of
  `payload_key` and `store_error`; any other attempt has none of them. A
  blob key carries the fetch's digest and UTC day. Every fetch in a tick
  starts and ends within it, and a bundle member points at that
  invocation's own bundle.
- **Error types.** Validators raise `PydanticCustomError` with a stable type
  (`fetch_storage`, `alert_bounds`, `legacy_period`, ...), so callers match
  types, not wording.
- **Digests.** `Sha256` (lowercase, 64 hex) is stricter than layer 1's
  `Digest` (16 to 64 hex) and types every payload and content digest here.
- **Parquet layout** (`momentarily.model.observations.columns`), derived from
  the contract: ints are `int64`, strings `string`, nested models `struct`,
  lists `list`, `dict[str, X]` a `map` written as key-sorted pairs, the
  verbatim alert entity a canonical-JSON `string`, and `lineage` a tagged
  struct. A field that can be `Unknown` gets a sibling `<name>__unknown`
  string column with the reason, so null, Unknown and a value stay distinct.
  The layout of every record is committed under
  `schema/model/s1/observations/parquet/` and pinned per `LAYOUT_VERSION`;
  any change fails a test until the version is bumped. `pyarrow` is not a
  dependency yet; the derive job adds it as an optional extra.
- **Feed header.** `FeedHeaderObservation` follows the proto: the required
  `gtfs_realtime_version` and `nyct_subway_version` are non-null, the
  optional `feed_version` is kept, and `nyct_feed_header` is a nullable
  struct, so a payload without the NYCT extension (null) stays distinct from
  one with zero periods (empty list). Replacement periods are a list in
  published order with duplicates kept; `route_id`, the `TimeRange` and each
  bound are null exactly when the payload omits them, and an empty
  `TimeRange` stays distinct from an absent one. Layout version 2.
- **Derived-file metadata.** Every parquet file carries one
  `DerivedFileMetadata` as canonical JSON under the key-value metadata key
  `momentarily.observations`: `file_key`, `layout_version`,
  `decoder_version`, `row_count`, and layer 1's `Provenance` (`inputs` and
  `captured_at` over the same raw or legacy keys, `built_at` the latest
  `captured_at`, `observed_from_utc_day` the file's day, no service day, never
  retroactive).
- **Daily legacy files.** `legacy_route_alert_state`,
  `legacy_fetch_outcome` and `legacy_collector_health` are daily
  `derived/<record>/<YYYY-MM-DD>.parquet` files.

### 3.5 As built (slice 3)

The decoders are `momentarily.model.observations.decode`: pure functions of
the payload bytes and a `RawFetch` (`fetch_id`, `source`, `observed_at`,
`RawLineage`), with no I/O and no clock. `DECODER_VERSION` (`"1"`) names
their output. Each decoder checks that the bytes hash to the lineage's
`payload_sha256`. A payload that does not parse, lacks a proto2 required
field, is not the JSON shape its source publishes, is nested too deep to
read, or holds a value that a contract rejects (a `direction_id` of 2, a
negative time) raises `DecodeError`, with the cause chained. Nothing is
returned for that payload.

- **Protobuf.** `gtfs-realtime-bindings` 3.0.0 on protobuf 7.36.2. The NYCT
  extension is vendored beside the decoder as the upstream `.proto` and its
  compiled module, whose header gives the source, its digest and the exact
  `protoc` run. An omitted optional field is `None`. An enum value that the
  proto does not define is kept by protobuf as an unknown field. The decoder
  therefore reads it as `None` and lists its path in `unmodeled_fields`.
- **`unmodeled_fields`.** Dotted field-number paths, relative to the
  record's message (VehiclePosition or TripUpdate): `2` is a vehicle's
  `position`, `2.6` a stop-time update's `stop_time_properties`. A leading
  `..` names a field of the enclosing FeedEntity other than `id`,
  `trip_update`, `vehicle` and `alert`. `l` sends `is_deleted` on every
  entity, so all its records carry `..2`. A path stops at the first
  unmodelled field. Extensions and unknown fields appear by number. Repeated
  messages give no index, so a field seen in any stop-time update is listed
  once. The list is sorted: `..` paths first, then by field numbers.
  Unmodelled fields of the FeedMessage and FeedHeader are not recorded,
  because no contract has a place for them. So an unknown tag appended to a
  payload's bytes, which lands on the FeedMessage, is dropped, and an
  undefined `incrementality` value reads as `None` with no path. On the
  2026-10-20 sample, every other feed decodes with an empty list.
- **Entities.** An entity that holds a trip update and a vehicle gives one
  record of each, with the same `entity_index`. Shapes, stops and trip
  modifications give no record. Alert entities are decoded only from
  `numbered`, because `AlertSource` admits no other GTFS-RT feed. In any
  other feed they are dropped. The sample had none elsewhere.
- **`numbered` alerts.** The digested entity JSON is protobuf
  `MessageToDict(entity, preserving_proto_field_name=True)` of the whole
  FeedEntity, with every other option at its default. So enums are names,
  64-bit integers are strings, omitted fields are left out, extensions are
  keyed `[<full name>]`, and unknown fields are dropped, so they are
  neither digested nor reported anywhere. `updated_at` is null. The entity
  id is a per-poll sequence number: the one "Train delayed" alert came back
  as `000401`–`000417` across 40 polls. So a `numbered` alert id does not
  identify an alert from one poll to the next.
- **Camsys alerts.** Each entity is digested exactly as parsed. A digest of
  the same entity read back from `archive/alerts/` is identical. An absent
  `entity` list is an empty presence. An entity published twice with the
  same content is one entry. The header timestamp is read exactly as the
  recorder reads it for `FeedFetch.header_timestamp`: a non-negative
  integer or a string of 1 to 15 digits, and otherwise `None`. An entity
  without a string `id`, or with an `updated_at` that is not epoch seconds,
  fails the decode even though the recorder counts the fetch as `ok`.
- **Silent alert edits.** Content does change without an `updated_at` bump.
  The committed fixture is the 14:36 UTC 2026-10-20 alerts payload. Of the
  170 `archive/alerts/` objects in the local mirror (2026-07-21 to
  2026-10-20) whose `(id, updated_at)` it still carries, 6 have different
  content. All 6 differences are in `active_period`. In 5, the feed dropped
  periods that had already ended. In 1, the far-future end of the alert's
  only period moved later. The digest records each such change as a new
  version.
- **ENE.** Records are kept in published order, and every value must be a
  string. `outagedate` and `estimatedreturntoservice` are read as
  `MM/DD/YYYY hh:mm:ss AM|PM` New York wall time, ASCII digits only. A key
  the record omits, or a blank or whitespace-only value, parses to `None`
  (the `_local` field keeps the published text), as in the backfill
  adapters. Any other text is `Unknown("unparseable: ...")`. A wall time
  that the fall-back hour repeats is `Unknown("ambiguous: ...")`, one that
  the spring-forward hour skips is `Unknown("nonexistent: ...")`, and one
  before 1970 is `Unknown("before 1970: ...")`, because no `Instant` holds
  it.

## 4. Writer placement

The Worker is the only always-on fetcher, so it must write `CollectorTick`,
the GTFS-RT tick bundles, and the blobs. Everything in §3.2 is derived
offline. A nightly Python job (the existing training workflow) derives the
previous UTC day, and an on-demand CLI derives any range.

Per tick, the Worker work is:

1. Keep the body bytes it already fetches. Today `fetchProtobuf` discards a
   200 with zero entities. The recorder must store those bytes and mark the
   fetch `empty`.
2. Hash each body with sha256, gzip it, and assemble the bundle.
3. PUT the bundle **before** decoding, so a decode failure cannot lose the
   bytes.
4. On boundary ticks, record the alerts and ENE fetches the same way and PUT
   blobs when their digest is new that day.
5. PUT `CollectorTick` in a `finally`. Make it independent of the alpha CAS,
   as `archiveAlertsLiveness` is already.

Limits and cost per tick:

- Measured: gzip and sha256 of one tick's 8 payloads took **4–6 ms** in Bun
  `CompressionStream`, and 8.3 ms for gzip in Python zlib. Workerd uses native
  zlib, so the cost should be similar [inference].
- Memory: about 0.8 MB extra.
- R2 writes: 2 PUTs per minute, plus blobs on change.
- [Cloudflare limits](https://developers.cloudflare.com/workers/platform/limits/):
  Workers Paid gives a Cron Trigger 30 s of CPU by default (up to 5 min),
  128 MB of memory, and 10,000 subrequests. Free gives 10 ms of CPU. The
  current per-minute decode and inference cannot fit 10 ms, so the account is
  on Paid [inference, the plan was not checked].

Validation:

- TS types are generated from the JSON Schema with the same
  `json-schema-to-typescript` step as layer 1 (`bun run gen:model`). The
  recorder builds records only through those types.
- A vitest test validates recorder output with ajv against the exported
  schema. Python and TS fixtures follow the existing `gen_*_parity_fixture.py`
  pattern.
- No runtime ajv: ajv compiles validators with `new Function`, which Workers
  block [inference]. If the owner wants runtime checks, use ajv standalone
  validators that are precompiled at build time.

Failure behaviour:

- An upstream failure becomes a `FeedFetch` with its outcome. The Worker never
  retries inside the tick, which keeps today's 10 s budget.
- If a bundle or blob PUT fails, `payload_key` stays null and `store_error`
  is set. The tick record is still written, and the existing `writeFailures`
  counter in `archive/health/` goes up.
- If the tick record PUT fails, the bundle's index still holds the GTFS-RT
  metadata. Only the alerts and ENE metadata for that minute are lost, and the
  minute reads as a partial collector gap.
- Overlapping invocations for one cron minute write separate keys, because
  `started_at_ms` is in the key. Both are observations. Choosing one is
  layer 3's job.
- The recorder runs beside today's archives and does not replace them in
  step 2.

### 4.1 As built (slice 4)

The recorder is `worker/src/recorder.ts`; `worker/src/observation_keys.ts`
formats its keys, pinned to `keys.py` by
`tests/fixtures/parity_observation_keys.json`. The build follows §4 except
where listed here.

- **What is recorded.** Every invocation fetches the 8 GTFS-RT feeds and the
  alerts feed at its start. On a boundary tick, step 2 of the pipeline takes
  that same alerts payload, so the pipeline's 5-minute cadence is unchanged and
  the feed is fetched once. ENE, stations and entrances are recorded when the
  pipeline fetches them (hourly or daily, alpha CAS winner only). The static
  GTFS HEAD is recorded every minute. Its detached capture GET is not: it
  outlives the tick, and its bytes go to `archive/gtfs/`.
- **Bundle.** Every member is a gzip member, the index (member 0) included.
  A reader splits them in order (Python: `zlib.decompressobj(31)` and
  `unused_data`). A bundle is written every invocation, also when no GTFS-RT
  body was read, so the index records the failed attempts. The tick's and the
  index's `fetches`, and so the member numbers, follow `planned_sources`
  order, not the order the fetches finished in.
- **Outcomes.** A GTFS-RT body is `unparseable` when its top level or its
  FeedHeader is not well-formed protobuf. Entity bytes are left to the
  offline decoder. The alerts JSON must be an object whose `entity` is an
  array (absent means 0); ENE and Socrata JSON must be an array. Any other
  JSON, or a body that does not parse, is `unparseable`. The bytes are stored
  in every one of these cases.
- **Blob dedup.** A HEAD on the content-addressed key, then a PUT on a miss,
  filed under the UTC day of `finished_at_ms`. A failed HEAD counts as a
  miss, so the blob is written anyway. A per-day digest index object was
  rejected: it needs the same Class B read, adds a Class A write ($4.50/M) on
  every change, and needs a CAS against overlapping invocations. Two racing
  PUTs of one blob key write the same bytes.
- **Closing the tick.** Every fetch runs through `Recorder.attempt`, and
  `finish()` waits for every attempt still in flight before it writes the
  tick. An invocation that throws while the alerts fetch is pending still
  records that fetch and its blob, then rethrows.
- **Write failures.** A failed bundle, blob or tick PUT counts as
  `observations_bundle`, `observations_blob` or `observations_tick` in that
  minute's `archive/health/` record. The recorder finishes before that record
  is written. The `finally` covers only an invocation that throws.
- **Tests.** `worker/test/recorder.test.ts` drives the real handler against
  Miniflare R2. Miniflare's Node-side binding cannot serialize the `Headers`
  form of `onlyIf` that `conditionalPut` sends (3.20241230 and 4.20260508
  alike), so the test bucket passes it on as `{ etagDoesNotMatch: '*' }`.
  `tests/fixtures/recorder_output.json` is a tick and bundle index the
  recorder wrote; `tests/test_recorder_output.py` runs it through the
  Pydantic validators.
- **Cost per tick.** Measured in Node under vitest with Miniflare R2, not in
  workerd, which was not measured. Recording one minute (8 GTFS-RT feeds and
  the alerts feed from the 2026-10-20 fixtures: classify, sha256, gzip, R2
  calls) took 14–15 ms of CPU. A whole off-boundary tick took 39–55 ms.
- **Size per minute.** The bundle is 135 KB, the alerts blob 74 KB (stored
  only on a new digest) and the tick record 6.5 KB. At that rate all day the
  bundles are 194 MB/day. Scaled by the §1.1 weekly-mean factor of 0.73 they
  are 142 MB/day, the figure §6 started from. Tick records are about
  9.4 MB/day, not the 3.6 MB first assumed.

### 4.2 As built (slice 5)

The derive job is `momentarily.model.observations.derive`. `storage` is the
store interface (list a prefix, and get, put or delete a key). `bundle`
reads tick bundles and blobs. `alert_index` builds the AlertVersion index.
`run` derives one UTC day. `LocalStorage` maps keys to paths under a
directory. `training.derive_observations` holds the R2 store, so the model
package never imports boto3.

```
python -m momentarily.model.observations.derive --root <dir> --day YYYY-MM-DD \
    [--out <dir>] [--hours H-H] [--overwrite] [--report report.json]
murk exec -- uv run python -m training.derive_observations --day YYYY-MM-DD [...]
```

Both commands print one JSON report: fetches by outcome, fetches not decoded
by reason (`unparseable`, `empty_body`, `not_stored`, `no_record`), per
record type the inputs, rows, files and what happened to each file, every
decode failure (fetch id, source, payload key and member, error), every
missing or inconsistent input, and every conflicting file. Exit status 0
means clean. Exit status 1 means a fetch failed to decode, an input is
missing or inconsistent, or a file conflicts (see the overwrite rule). Exit
status 2 means bad arguments or no `--root`. A day with no fetch observed in
it writes and deletes nothing, reports `"status": "no_ticks"` and exits 0.
`--hours` derives only those hourly files. The daily `alert_version` file
needs the whole day.

- **Joining an invocation.** Ticks and bundles are filed by `scheduled_at`,
  records by `observed_at`. So the job also reads the hour before the day:
  an invocation dispatched at 23:59:50 observes after midnight. Its records
  go to the next day's 00 files, never to this day's. The fetches come from
  the tick record. When the tick record is missing, the GTFS-RT fetches come
  from the bundle index, and the minute's alerts and ENE fetches are lost
  (§4). When both exist, each fetch in the index must be in the tick
  unchanged, and each tick fetch with a member must be in the index. A
  GTFS-RT body that arrived after the bundle was written is in the tick only.
  Overlapping invocations of one minute are separate fetches, and all their
  records are kept. A key the store lists but cannot return is a missing
  input. The run goes on without it.
- **What is decoded.** Only a stored body of an `ok` or `empty` fetch is
  decoded. The report counts `unparseable` fetches, bodies that were not
  stored, and sources with no layer-2 record (`ene_equipments`, Socrata, the
  static GTFS HEAD). The recorder records a 200 with a zero-byte GTFS-RT body
  as `empty` (its wire scan counts 0 entities in 0 bytes). That body is no
  FeedMessage, because the header is required. So the job skips it as
  `empty_body`: no record, no presence evidence, no failure. Any other
  `DecodeError` is reported for its fetch and the run goes on. Decode
  failures and input problems in the carry hour (below) appear in the
  reports of both days, because both days' files depend on them.
- **Memory.** Records go to arrow rows as soon as a fetch is decoded, and
  each hour's files are written and released when the hour's last tick
  folder is read. On the fixture (11 invocations) peak RSS fell from 513 MB
  to 213–217 MB, with a 69 MB baseline. The rows held for the 11 invocations take
  24 MB of arrow memory (trip updates 20 MB). Scaled to 60 invocations, an
  hour holds about 130 MB of rows, and the peak is about 0.6 GB with the
  copies the writer makes [inference, not measured on a full hour].
- **Provenance.** `inputs` names every tick, bundle and blob a file's records
  came from. The digest is the sha256 of the stored object (the gzip bytes
  for bundles and blobs), not of the payload. `captured_at` is the latest
  `observed_at` that key gave the file. So `built_at` is the latest
  observation in the file. The `alert_version` file names every tick and
  payload of its window's evidence.
- **AlertVersion window and rows.** Day D's index covers
  `[D 00:00 − 1 h, D+1 00:00)`. The carry is one hour before the day, and
  nothing after. A version gets one row per *run*: a stretch of sightings
  with no evidence (`ok` or `empty`, decoded) lacking it in between. A
  version that leaves and comes back is two rows. The AlertVersion natural
  key is now `(source, alert_id, content_digest, first_seen)` (§3.2). The
  report counts such versions (`alert_versions_reappearing`). D's file holds
  every run with a sighting in D, and every run whose `gone_by` falls in D.
  A run seen and gone within the carry belongs to D−1. `absent_before` and
  `gone_by` are the nearest evidence fetches of the same source before the
  run's first and after its last sighting, null when the window has none.
  `gone_kind` is `superseded` when the `gone_by` presence holds the same id
  with another digest, else `removed`. So a run that ends with the alert
  gone, or replaced by another alert, is `removed`. A run whose id flips to
  other content is `superseded`, also when the first content comes back
  later.
- **Gaps.** The cron minute of a fetch is `scheduled_at // 60`. Two
  consecutive evidence fetches more than one minute apart leave the seconds
  strictly between them in `unobserved`. A failed fetch, a lost tick record
  and a minute with no invocation all make such a gap. When `gone_by` is
  null and the last evidence is before the window's last minute, the rest of
  the window is unobserved too.
- **Midnight.** A run that spans midnight is in both days' files. D's row
  has `gone_by` null. D+1's row starts in its carry. If the run started in
  the last hour of D, that row has the true `absent_before`. If it started
  earlier, the row has `absent_before` null, and its sightings overlap D's
  row in the carry hour, so a reader can join the two. A run whose last
  sighting is in D's last hour and whose `gone_by` is after midnight is
  closed in D+1's file. The one edge no file closes: no evidence for more
  than the carry across midnight. Then D's row ends with `gone_by` null and a
  trailing `unobserved` gap, and D+1's window holds none of its sightings.
- **`numbered` alerts.** Their ids change every poll (§3.5). So nearly every
  `numbered` version is seen once: `first_seen == last_seen`, `gone_by` the
  next poll, `gone_kind` `removed`. The index infers no identity across ids.
  Its `alert_type`, `created_at` and `updated_at` are null, because
  `MessageToDict` keys the extension `[transit_realtime.mercury_alert]`.
- **When a day is final.** The job reads nothing filed under D+1, so a
  rebuild of D gives the same bytes whether D+1 exists or not. D's files are
  final once every invocation dispatched before D+1 00:00 has written its
  tick and bundle, a few minutes after midnight UTC (one Worker invocation).
  A rebuild also needs D−1 22:00–24:00 and D unchanged in the store.
- **Overwrite rule.** Each file is built in a staging directory and compared
  with the stored file of the same key. No stored file: write it. Identical
  bytes: skip it. A stored file with another `decoder_version`: replace it.
  The new build has the same bytes under the stored file's `code_sha` and
  `code_dirty`: skip it, because only the code provenance differs. Any other
  difference, or a stored file that is not a readable derived file, is a
  conflict: the job keeps the stored file, lists it and exits 1.
  `--overwrite` replaces it. A stored file in the scope that the rebuild has
  no rows for follows the same rule: deleted with `--overwrite` or another
  `decoder_version`, else a conflict. The bytes also depend on the `pyarrow`
  version (§5.1). After a `pyarrow` upgrade, rebuilds of old days conflict
  until they run with `--overwrite`. The job owns raw-lineage files only: a
  stored file whose provenance inputs are not all ticks, bundles and blobs
  (a legacy backfill file) is never replaced or deleted, with or without
  `--overwrite`. Rows built for its key are a conflict and the job exits 1;
  if the rebuild has no rows for it, it is kept, listed in `foreign_kept`
  and the exit status is unaffected.
- **Nightly wiring.** The last derive step of `prune-nightly.yml` (05:00 UTC)
  derives the previous UTC day from R2. It runs after the prune, so a
  failure cannot block it, and with `if: !cancelled()`, so an earlier failed
  step does not skip it. A manual run with the `derive_day` input derives
  that day instead, to recover a missed night. Until the recorder deploys,
  the step is a no-op. The job timeout is now 45 minutes.
- **Replay fixture.** `scripts/gen_recorded_hour_fixture.py` packages the
  2026-10-20 sample payloads into the recorder's exact tick, bundle and blob
  format: ten cron minutes of hour 14, every feed every minute, ENE once. It
  synthesizes one HTTP 503, one overlapping invocation (3 s later, 3
  timeouts) and one lost tick record, and documents each. The output is
  25 objects, 1,638,600 bytes, and the generator is deterministic.
  `tests/test_observation_derive.py` pins the canonical JSON of the derived
  records per `DECODER_VERSION`, checks that a second run is byte-identical,
  and rebuilds the recorder's own bundle from `tests/fixtures/recorder_output.json`
  and the Worker test's bodies: every member hashes to its index entry.
- **The fixture day derived.** 11 invocations (one from the bundle index
  only), fetches 107 `ok`, 3 `timeout`, 1 `http_error`, 0 decode failures.

  | Record | Rows | Files | Inputs |
  |---|---|---|---|
  | `vehicle` | 6,832 | 1 | 21 |
  | `trip_update` | 7,967 | 1 | 21 |
  | `feed_header` | 84 | 1 | 21 |
  | `alert_presence` | 21 (10 alerts, 11 numbered) | 1 | 22 |
  | `equipment` | 249 | 1 | 3 |
  | `alert_version` | 190 (179 alerts, 11 numbered) | 1 | 22 |

- **Cost.** Measured on the fixture on an M-series laptop: 2.1–2.8 s for 11
  invocations, about 0.2–0.25 s each, almost all of it in the GTFS-RT decoder
  and the parquet row encoding. A whole day is about 1,440 invocations, so
  about 5–6 minutes of CPU plus about 2,900 R2 GETs [inference].

## 5. Backfill mapping

`Unknown(reason)` marks every field a legacy archive did not keep. Nothing is
inferred, including the feed of a trace row and `6` versus `6X`. Every
backfilled record carries a `legacy` lineage (archive, object key, row,
period; §3.4). Legacy fetch and tick facts never become `FeedFetch` or
`CollectorTick`, which stay live only.

| Legacy source (coverage) | Becomes | Kept | Permanently lost (Unknown) |
|---|---|---|---|
| `archive/trace/` (R2 only, 120 d; earliest days pruned from about 12-10 [inference: trace start ≈ 08-12]) | partial VehicleObservation, plus a LegacyFetchOutcome per feed and trace minute | `trip_id`, `stop_id`, `vehicle_ts`, `stop_seq` (when stopped), `stopped=true` → `current_status=1`, `observed_at`, `scheduled_at`, `fresh_feeds` → per-feed `succeeded`. `feed_digest`/`feed_etag` are not carried: they are the static feed identity, which `archive/gtfs/` holds, and all 2,880 objects that carry them hold one identical pair | `route_id` as published (folded: `6`/`7`/`F` may have been `6X`/`7X`/`FX`), `current_status` when not stopped (0, 2 or absent), `stop_seq` when moving, `is_assigned` (placeholder vehicles are indistinguishable), `source` feed of each row, `start_date`, `start_time`, `train_id`, `entity_id`, `entity_index`, `fetch_id`, vehicles with empty trip or stop ids, header timestamp, fetch failure reason, bytes |
| `archive/arrivals/` (R2 only, 120 d) | **not mapped** | — | truncated to 4 trains within 60 min, SKIPPED rows dropped, arrival/departure merged, route folded. It stays a grading input for `eta_grade.py`. |
| `archive/trip_updates/` (06-15..), `archive/vehicles/` (06-21..) | LegacyFetchOutcome only (5-min, per feed `succeeded`) | which feeds were fresh | every per-entity field. The counts and transitions are aggregates and stay what they are. |
| `archive/alerts/` (90 d. Mirror 07-21..) | AlertVersion | entity verbatim, `content_digest`, `alert_type`, `created_at`, `updated_at`, `first_seen` = `observed_at` (5-min resolution, a sighting that is normally the first) | `absent_before`, `last_seen`, `gone_by`, `gone_kind` (except a later `updated_at` for the same id, which proves `superseded` by that time), `unobserved`, `window`, content changes with no `updated_at` bump, feed header |
| `archive/alerts_liveness/` (09-02..) | LegacyFetchOutcome(`alerts`) | outcome (`succeeded`), wall-clock attempt time, `fetched_at` (`last_success_at`) | scheduled minute, HTTP status, digest, entity count, header timestamp. No AlertPresence before layer 2 starts. |
| `v1/predictions/` (90 d. Mirror 07-21..) | separate `LegacyRouteAlertState` (decision 5): per route row and 5-min tick, `primary_alert_type` | highest-`sort_order` alert active at that tick, planned work included, for routes with an inference row | alert ids, every lower-priority alert, routes without an inference row. It can bound a route-level alert end. It cannot bound an alert version end. |
| `archive/ene/` (90 d) | EquipmentObservation, plus a LegacyFetchOutcome for each success | full record content, hourly | `fetch_id`, failed fetches (no record), byte-exact payload (re-serialized), sub-hour changes |
| `archive/health/` (09-11..) | LegacyCollectorHealth | write-failure counts per invocation | planned sources, fetch timing |
| `archive/gtfs/` | adopted as is | everything | — |
| traversals, windows, movement_census, transitions | not layer 2 (derived) | — | — |

**Urgent:** the 90-day windows of `archive/alerts/`, `archive/ene/` and
`v1/predictions/` are deleting their first days (07-21/22) now. For those
days the local mirror is the only copy. The trace's first days follow in
about 7 weeks.

### 5.1 As built (slice 6)

The adapters live in `momentarily.model.observations.legacy`, one module per
row of the table above (`trace`, `fresh_feeds` for `archive/trip_updates` and
`archive/vehicles`, `alerts`, `alerts_liveness`, `predictions`, `ene`,
`health`). They need no `pyarrow`. `momentarily.model.observations.parquet`
writes any record type; the derive job is meant to reuse it.

```
python -m momentarily.model.observations.legacy --mirror <mirror-root> \
    --out <out-root> [--report report.json]
```

The CLI reads the mirror (local only) and writes
`<out>/model/observations/s1/derived/...` with the existing `keys`; no key
was added. `vehicle` and `equipment` files are hourly by `observed_at`.
`alert_version` files are daily by `first_seen`. The three legacy-only
records are daily by `observed_at`. A malformed object is skipped, listed in
the report, and makes the CLI exit 1. A missing mirror exits 2. A record that
a contract rejects stops the run. A file at a target key whose provenance
inputs are not all legacy archive objects (a raw-lineage file the derive job
wrote), or that is not a readable derived file, is never overwritten: it is
kept, listed under `conflicts` in the report, and makes the CLI exit 1.

**Legacy periods.** Each `start` is the first UTC day folder of the archive
in the mirror of 2026-10-20. `end` is null. The values are constants
(`legacy.common.PERIODS`), so a partial input cannot move them. For
`archive/alerts`, `archive/ene` and `v1/predictions` the start is the oldest
surviving day, not the first day written. The Worker wrote these archives
from about 2026-05-13, and the days before 07-21 were pruned before history
was kept.

| Archive | `start` |
|---|---|
| `archive/trip_updates` | 2026-06-15 |
| `archive/vehicles` | 2026-06-21 |
| `archive/alerts`, `archive/ene`, `v1/predictions` | 2026-07-21 (oldest surviving day) |
| `archive/trace` | 2026-08-12 |
| `archive/alerts_liveness` | 2026-09-02 |
| `archive/health` | 2026-09-11 |

**Deviations and findings.**

- **Trace `feed_digest` and `feed_etag` are not carried.** The owner decided
  this. They are the static feed identity, which `archive/gtfs/` already
  holds. All 2,880 objects that carry the fields hold one identical pair.
- **The trace `scheduled_at` is not the cron minute.** It is
  `floor(event.scheduledTime / 1000)`, the instant Cloudflare dispatched the
  invocation. It falls 14, 15, 19, 26 or 35 s past the minute, and 2,828 of
  3,330 objects have the same value as `observed_at`. `LegacyFetchOutcome`
  keeps it verbatim. Layer 3 must not assume one tick per minute. On
  2026-10-19 two invocations ran 5 s apart, at `scheduled_at` 1792429574
  and 1792429579 (17:06:14 and 17:06:19 UTC).
- **A stopped trace row without `stop_seq`** (18,141 rows) has
  `current_stop_sequence` null: the feed omitted the field. It is not
  `Unknown`. A moving row has both `current_status` and
  `current_stop_sequence` `Unknown`.
- **`ene_equipments`** objects give a `LegacyFetchOutcome` only, because the
  catalog is not a layer-2 record. ENE wall times are parsed in
  America/New_York. A blank or whitespace-only time is `None` (absent). A
  repeated, skipped or unparseable time is `Unknown`. The slice-3 decoder
  uses the same rule.
- **Alert versions.** A version is one distinct `(alert_id,
  content_digest)`. `first_seen` is the earliest sighting, and the lineage
  names that object. `gone_kind` is `superseded` when another version of the
  same id has a larger `updated_at`. The object that proves it is one of
  the file's provenance inputs, so `built_at` can fall after the file's
  day. The mirror has no re-sightings of an identical digest, so the number
  of versions equals the number of objects.
- **Writer.** zstd level 9, row groups of 2^20 rows, rows in natural-key
  order, and a duplicate key rejected. Every writer option that affects the
  bytes is pinned: format version, data page version and size, dictionary,
  statistics, compression and level. The key-value metadata holds the arrow
  schema and the `DerivedFileMetadata`. The writer rejects an output path
  that does not end in `file_key`, and any record outside the file's hour or
  day. The bytes are reproducible for a given `pyarrow` version. `pyarrow`
  has no upper bound, and `uv.lock` pins the version. The bytes also depend
  on `code_sha` and `code_dirty`. `columns.layout` is now cached per model.

**Full run** (2026-10-20, all of the mirror, about 130 s for each run).
Parse failures: none. Two runs gave byte-identical files (sha256 of all 2,550
files).

| Archive | Objects | Rows | Adapter s |
|---|---|---|---|
| `archive/trace` (3 days) | 3,330 | 1,845,677 vehicle + 26,640 fetch | 30.7 |
| `archive/trip_updates` | 36,547 | 292,376 fetch | 5.8 |
| `archive/vehicles` | 34,631 | 277,048 fetch | 6.8 |
| `archive/alerts` | 16,195 | 16,195 alert_version | 2.6 |
| `archive/alerts_liveness` | 13,609 | 13,609 fetch | 1.6 |
| `archive/ene` | 6,424 | 257,604 equipment + 6,424 fetch | 8.3 |
| `archive/health` | 51,944 | 51,944 health | 6.1 |
| `v1/predictions` | 26,211 | 760,119 route alert state | 12.8 |

| Record | Files | Bytes | Write s |
|---|---|---|---|
| `vehicle` | 56 | 9,953,714 | 37.7 |
| `equipment` | 2,142 | 33,864,175 | 5.8 |
| `alert_version` | 92 | 12,219,988 | 0.8 |
| `legacy_fetch_outcome` | 128 | 42,362,836 | 5.0 |
| `legacy_route_alert_state` | 92 | 11,139,853 | 4.7 |
| `legacy_collector_health` | 40 | 20,843,935 | 0.3 |

## 6. Retention and storage cost

R2 prices: storage $0.015/GB-month (10 GB free), Class A $4.50/M, Class B
$0.36/M. The bucket already holds about 17 GB (trace 10.8 GB at 120 d,
arrivals 3.7 GB, ENE 1.3 GB, and the rest). So the free tier is already used,
and every figure below is a marginal cost.

| Item | Per day | Proposed retention | Steady storage | $/month |
|---|---|---|---|---|
| GTFS-RT tick bundles (gzip) | 143 MB (weekly mean, §1.1); 194 MB at the measured slice-4 rate | 35 d | 5.0–6.8 GB | 0.08–0.10 |
| Nightly zstd `--long` repack of the bundles | 30 MB | keep (1 yr shown) | 11 GB/yr | 0.16 per retained year |
| Alerts blobs (gzip, on change) | 8.6–21 MB | keep (1 yr shown) | 3.1–7.7 GB/yr | 0.05–0.12 per year |
| ENE and Socrata blobs | ~1 MB | keep | 0.4 GB/yr | 0.01 |
| Tick records | 9.4 MB (measured, slice 4) | keep | 3.4 GB/yr | 0.05 per year |
| Derived vehicle + stop-time parquet | 55 MB | 365 d (rebuildable from raw) | 20 GB | 0.30 |
| Derived alert presence/versions, ENE, headers | < 2 MB [inference] | keep | < 1 GB/yr | < 0.01 |

R2 operations per month:

- Worker: about 2 PUTs per minute, plus blobs on a new digest (alerts, at
  least 118 changes a day; about 50 ENE and Socrata). That is about 96k
  Class A, or $0.43. Dedupe costs one HEAD (Class B) per JSON fetch: 1,440
  alerts plus about 75 ENE and Socrata a day, about 45k a month, or $0.016.
- Nightly derive: about 52k GETs ($0.02) and about 1.5k PUTs.

**Total in year one: about $1/month**, rising about $0.35/month for each further
year of raw, alerts and tick records kept.

Not repacking and keeping the gzip bundles for a year instead would cost
52 GB and $0.78/month. Money does not drive this choice. The repack exists so
that "keep raw forever" stays cheap.

Prune: `prune.py` lists every dated prefix in full every night. That costs
one Class A list per 1,000 keys retained, so its cost grows with the number of
keys, not bytes. Per-minute keys make this worse. Example: one year of tick
records plus bundles is 1.05M keys, or about 1,050 list calls a night (32k a
month, $0.14). The layer-2 prefixes should be pruned by listing day folders
with a delimiter and deleting only expired days, which costs O(expired days).
The 3650-day per-minute `archive/health/` prefix already grows by 1,434 keys
a day under the full listing.

## 7. Prerequisite: no `training` imports in the model package

`momentarily.model.network` imports `training.gtfs_static` (`Calendar`,
`feed_rows`, `gtfs_seconds`, `read_calendar`, `read_version`,
`fetch_gtfs_zip`), `training.provenance.code_provenance`, and
`training.traversal_archive` (`DayProvenance`, `decode_day`, which pulls in
`training.trace`, `load_r2` and boto3). The wheel packages only
`src/momentarily`.

Proposal:

1. Move `provenance.py` (standard library only) and the static-GTFS reader
   (standard library plus httpx, already a dependency) into the package, as
   `momentarily.provenance` and `momentarily.gtfs.static`.
2. Move the traversal-day codec (`encode_day`, `decode_day`, `DayProvenance`,
   and the `Traversal` row type) into `momentarily.archive.traversals`.
3. Point `training/` at the package, the direction it already uses
   (`training` imports `momentarily.mapping` and `momentarily.hmm` today).
   Use a clean cutover with no re-export shims.
4. Layer 2 code (`momentarily.model.observations`) imports only
   `momentarily.*`. The GTFS-RT decoder is new package code.

Acceptance: build the wheel, install it in an empty venv, and run
`python -c "import momentarily.model.network.build, momentarily.model.observations"`.

## 8. Decisions

Decided by the owner on 2026-10-20 unless marked open. The owner intends to
keep full history, possibly as a commercial dataset, so nothing raw is deleted
by age.

1. **Raw retention: decided.** Gzip tick bundles for 35 d, then a nightly
   zstd `--long` hourly repack kept indefinitely (30 MB/day, about $0.16/month
   per year).
2. **Alert polling cadence: decided.** The recorder fetches alerts every
   minute. Blobs are stored only on change. Today the Worker fetches alerts
   only on 5-minute boundaries, so first sighting trails `updated_at` by a
   median 143 s.
3. **Legacy archives: decided.** Stop date-based deletion of every archive and
   stream (not only `archive/alerts/` and `v1/predictions/`), and upload the
   mirror's days that R2 has already pruned. This lands on `main` first,
   because the nightly prune runs from there.
4. **Normalized storage format: decided.** Parquet-zstd (5.5 + 50 MB/day),
   with `pyarrow` as an optional extra of the derive job only.
5. **`LegacyRouteAlertState`: decided, with a condition.** Kept as a lossy
   legacy contract. Every record states that it is legacy and for which dates:
   its source (`v1/predictions`), the source's granularity (route, 5-minute
   tick, presence only), and the legacy period it belongs to (2026-07-21 up to
   the first day the recorder wrote `AlertPresence`). Layer 3 uses it only for
   dates without `AlertPresence`, and labels any bound it produces as derived
   from legacy data.
6. **GTFS-RT decoder: decided.** Google's `gtfs-realtime-bindings` plus the
   MTA's `nyct-subway.proto`, compiled with `protoc` and vendored into the
   package, with present-but-unmodelled fields recorded in `unmodeled_fields`
   (§3.5).
7. **Materialize TripUpdateObservation: decided.** Yes, as parquet, kept
   365 d. Older days are regenerated from the raw bundles, which are kept
   indefinitely.
8. **Poll faster than 1 min: decided.** No. Polling stays at 60 s. Revisit
   only if layer 3 shows that the 60 s bracket on arrivals and dwells matters
   to a product.
9. **Cutover: deferred.** Decide after 30 days of the recorder running side by
   side with the current archives.

## 9. Step 2 build plan

The slices are listed in order. Slices 3 and 4 can run in parallel after
slice 2.

1. **Package the shared modules (§7).** Acceptance: the wheel-only import test
   passes, `training/` contains no copy of the moved code, and the existing
   layer-1 and training tests pass unchanged.
2. **Contracts and generation.** Add `momentarily.model.observations.contracts`,
   the schema export, TS generation and fixtures, following layer 1.
   Acceptance: layer-1-style tests fail on a stale schema, stale TS, or a
   stray file. Ajv validates every fixture. A contract with a default is
   rejected.
3. **Python decoders.** Add GTFS-RT → VehicleObservation,
   TripUpdateObservation and FeedHeaderObservation. Add alerts JSON (and the
   `numbered` feed alerts) → AlertPresence. Add ENE → EquipmentObservation.
   Use a handful of the 2026-10-20 sample payloads as golden fixtures.
   Acceptance:
   - absent fields decode to `None`, never a default;
   - negative delays come out signed;
   - placeholder vehicles keep their future timestamp;
   - `route_id` is verbatim;
   - an injected unknown field shows in `unmodeled_fields`;
   - the digest of an alert entity equals the digest of the same entity
     re-serialized as `archive/alerts/` stores it.
4. **Worker recorder.** Write tick bundles, blobs and `CollectorTick`
   beside the existing pipeline. Acceptance:
   - vitest with Miniflare R2 checks one tick against the expected keys;
   - injected timeout, HTTP 500, empty 200, garbage 200 and a failed PUT each
     give the right `outcome` or `store_error`;
   - ajv validates the output;
   - a parity fixture shows that the gunzipped bundle members are
     byte-identical to the fetched bodies (sha256 equals `payload_sha256`).
5. **Offline derive job and AlertVersion index.** Acceptance (the **replay
   test**): a committed fixture hour of archived ticks and raw objects derives
   to records whose canonical bytes match a pinned digest. A second run is
   byte-identical. Changing the decoder version without updating the pin
   fails. AlertVersion: a scripted presence sequence (appear, persist, failed
   fetch, removal, supersede) gives the expected `first_seen`, `last_seen`,
   `gone_by`, `gone_kind` and `unobserved`.
6. **Backfill adapters (§5).** Acceptance:
   - a test asserts that every field outside each legacy source's "kept" list
     is `Unknown`;
   - the number of AlertVersions equals the number of `archive/alerts/`
     objects, minus re-sightings of an identical digest;
   - trace-derived observation counts equal trace row counts;
   - running the adapters twice over the mirror gives identical outputs.
7. **Retention.** Add day-folder pruning for `model/observations/s1/` and the
   nightly zstd repack. Acceptance: a dry run deletes only expired day
   folders. The number of list calls is bounded by the number of days, not
   keys. A repacked hour decodes to the same payload digests as its tick
   bundles.

## What was not verified

- Daily sizes scale one 10-minute weekday sample by vehicle counts. The share
  of 6X/7X and SS, rush-hour payload sizes, and night-time payload sizes were
  not sampled.
- How often the alerts feed changes in a busy period. The sample saw no
  change. 118 changes/day is a lower bound taken from the archive.
- Whether alert content changes without an `updated_at` bump. The current
  archive cannot show this.
- Workerd's gzip CPU cost. Bun and Python were measured, not Workerd.
- The account plan, current R2 usage, and the R2 operation counts. These
  need credentials, so the bucket total and the existing-operations estimate
  are derived from the mirror and the code.
- The trace and arrivals start dates in R2. These are not mirrored.
