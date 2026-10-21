# Data model, layer 1: conventions and network (`network/s1`)

The data model is a stack of contracts built beside the live pipeline. Layer 1
is the static network that every later layer keys off. Pydantic in
`src/momentarily/model/` is the source of truth; `schema/model/s1/*.schema.json`
is exported from it, and `shared/model/s1/*.ts` is generated from those
schemas. Nothing is hand-mirrored.

## Conventions (`momentarily.model.conventions`)

- **Instant**: integer UTC epoch seconds.
- **ServiceDate**: ISO date of a GTFS service day. It is never the UTC date of
  an instant.
- **GtfsSeconds**: seconds after noon minus 12 h on the service day. Values of
  86400 or more are legal (24:04:00 = 86640). Convert with
  `instant_of(service_date, seconds, tz)`, which subtracts 12 h in UTC
  seconds. On DST change days, that origin is 23:00 or 01:00 local time, not
  midnight.
- **Intervals are half-open**, `[start, end)`. GTFS's inclusive `end_date`
  values (`calendar.txt`, `feed_info.txt`) are stored as end + 1 day.
- **No defaults**: no contract field has a default value. A missing value is an
  explicit `null` (or `Unknown` with a reason), so a missing measurement can
  never read as a normal value. `Model` rejects a subclass that declares a
  default.
- **Provenance**: `code_sha`, `code_dirty`, `schema`, `inputs` (name to digest),
  `captured_at` (name to capture instant), `period_start` (the feed's first
  service day), `observed_from_utc_day`, `applied_retroactively`, and
  `built_at`. `built_at` is the capture time of the stations snapshot, not the
  wall clock, so a rebuild gives the same bytes. An input is retroactive when it
  was captured on or after the start of the period it describes. With
  `--day-feed-map`, that start is `observed_from_utc_day`, the first UTC archive
  day observed on the feed, compared with the capture time in UTC. Without it,
  the start is `period_start`, compared in the system timezone.

## Identities

| Thing | Identity |
| --- | --- |
| Feed | `feed_id` = sha256 hex of the GTFS zip bytes (the same key as `archive/gtfs/`) |
| Stations input | blake3 of the canonical snapshot file |
| Build | `build_id` = first 16 hex of blake3(`network/s1␟BUILDER_VERSION␟system_id␟feed_id␟stations_digest`[`␟overrides_digest`]) |
| Build content | `content_digest` = blake3 of the canonical uncompressed JSON (in `index.json`) |
| Pattern | `{route}-{code \| direction_id \| x}-{h}`, where h = first 12 hex of blake3(system␟route␟code␟stops…) |
| Segment | `{from_stop}>{to_stop}` (directed, platform level) |

A pattern is one (route, direction, ordered stop list). Shapes are an attribute
of a pattern, not part of its identity. Direction belongs to trips and patterns,
and `direction_id` can be null. Platforms carry only the NYC N/S code and the
rider-facing label for that side of the station.

## Building (local only, no credentials)

```sh
# 1. Snapshot the MTA Subway Stations dataset (data.ny.gov 39hk-dx4f) once.
uv run python -m momentarily.model.network snapshot-stations --out OUT
# 2. Build a network from a GTFS zip (or --live to fetch the current feed).
uv run python -m momentarily.model.network build --gtfs FEED.zip \
    --stations OUT/stations-<digest16>.json [--day-feed-map OUT/day_feed_map.json] --out OUT
# 3. Map each archived traversal day to the feed it was collected against.
#    (Inferred leads for unstamped days use OUT/index.json, so build first.)
uv run python -m momentarily.model.network day-feed-map \
    --traversals ARCHIVE/traversals --out OUT
```

The model package imports only `momentarily.*`, so these commands also run from
an installed wheel with no checkout. It shares the static-GTFS reader
(`momentarily.gtfs.static`), the code-provenance stamp
(`momentarily.provenance`), and the traversal-day codec
(`momentarily.archive.traversals`) with `training/`, which imports them from the
package. `tests/test_wheel.py` installs the built wheel in a fresh venv and runs
the CLI.

`build` writes `OUT/<build_id>.json.gz` (gzip mtime 0, sorted keys) and adds an
entry to `OUT/index.json`. A build_id names one network's data (everything but
provenance). If that file already exists with the same network data but
different provenance (another commit or a newer day map), it is replaced and
the index entry is updated. If its network data differs, or it cannot be read,
the build is refused: bump `BUILDER_VERSION` when a builder change alters the
output. A test pins the fixture's network digest to the current version.

The build fails, and lists every problem, when any of these is true:

- `routes.txt`, `trips.txt`, or `stop_times.txt` is empty, or a trip_id
  repeats;
- a GTFS parent station does not have exactly one stations-dataset row, or a
  dataset row has no GTFS station;
- a platform has no direction label. A platform with no N/S suffix gets no
  label from the dataset, so it needs `--overrides` (a JSON object of stop_id
  to label). An override that fills no gap also fails the build;
- a trip does not resolve to exactly one pattern;
- a pattern's segments do not chain from end to end.

## Keeping the map current (`update`, nightly)

The `day-feed-map` backfill reads `archive/traversals/`, which ends on
2026-10-18. Later days are added by `update`, which never reads traversals.
Its evidence, for each UTC day:

- the `gtfs_static` HEAD in every recorder tick under
  `model/observations/s1/ticks/<day>/` (docs/model-observations.md §4.1): the
  strong ETag the static feed URL served at that tick's `scheduled_at`. A
  failed HEAD, a weak or missing ETag, or a tick that does not parse is no
  evidence for that minute;
- the static-feed capture records `archive/gtfs/by_etag/<etag>.json`
  (`{"digest": ...}`). The Worker (`worker/src/gtfs_feed.ts`) writes one when
  a HEAD ETag differs from `archive/gtfs/latest.json`, keyed on the ETag of
  the GET that fetched the bytes, after storing `archive/gtfs/<digest>.zip`.
  The nightly `training.gtfs_archive` writes the same layout.

The join is ETag to `feed_id` through `by_etag`, accepted only when the zip it
names is held. Two ETags that name the same digest are one feed (the same
bytes served again). The capture records carry no time, and `latest.json`
only names the newest feed, so the ticks are the time series and the captures
only the join.

| Evidence on the day | Entry |
| --- | --- |
| one feed; no gap between successful HEADs, or from midnight to the first and the last to the next midnight, over `MAX_GAP_S` (15 min) | verified, `feed_id` |
| one feed, but a gap over 15 min | unverified, `inferred_feed_id` = that feed |
| more than one feed (a change during the day) | unverified, no feed |
| an ETag with no capture record, no held zip, or a zip that does not read | unverified, no feed |
| ticks, but no successful HEAD (the reason counts unreadable ticks) | unverified, no feed |
| no ticks | unverified, no feed |

The reason names the spans (`<feed8> <first>..<last>` in epoch seconds), as
the backfill does for a partly stamped day. A mid-day change is handled as
the backfill handles a day spanning two digests: no feed, no lead. The
recorder's measured cadence is 60 s (2026-10-21, 03:57 to 13:30 UTC: 573 of
578 gaps 60 s, the rest 0 s retries); 15 min leaves room for short outages.

`update` appends every day after the map's last one through `--through`
(default yesterday, UTC). It never rewrites a mapped day, so a day added
before it was complete stays as it was recorded; run it on finished days. It
then builds every feed that the map names (`feed_id` or `inferred_feed_id`)
and that has no `index.json` entry, from `archive/gtfs/<feed_id>.zip`
(checked against its digest), with no overrides, the newest
`stations-*.json` already in the output (by `retrieved_at`; the builder needs
no fresh snapshot), and the updated map as `--day-feed-map`. A failure stays
with its feed: a feed that does not build (for any error) is reported, the
map is still extended and published, the step exits 1, and the next run
retries the build. A captured zip that does not read is reported the same
way, and its days stay without a feed.

Writes never replace different bytes. A build file already present is kept
when it holds the same network (another commit's provenance) and is a
conflict otherwise. `index.json` and `day_feed_map.json` may only be replaced
by a document that keeps every existing entry unchanged. All checks run
before the first write; on a conflict nothing is written and the step exits
1. Builds are written first, then the index, then the map. In R2 a build is
put with `If-None-Match: *`, so one written by another run after the check
is also a conflict (exit 1, the index and map unwritten). A rerun with the
same inputs writes nothing.

```sh
# Local: both roots use the R2 key layout (a bucket mirror).
uv run python -m momentarily.model.network update --source MIRROR --out ROOT \
    [--through YYYY-MM-DD]
# Nightly, against R2 (.github/workflows/prune-nightly.yml):
uv run python -m training.network_update
```

The nightly step runs even when an earlier step failed (`!cancelled()`),
and resumes after the map's last day, so a missed night is caught up. It
prints day lines, build ids and object keys only.

## Regenerating the contracts

```sh
uv run python -m scripts.export_model_schema        # schema/model/s1 (all layers)
(cd worker && bun run gen:model)                    # shared/model/s1 (all layers)
uv run python -m scripts.gen_model_network_fixture  # tests/fixtures/model/*_s1.json
```

Later layers export into a subdirectory (`schema/model/s1/observations/`), and
the generated TypeScript mirrors that tree. If a schema under
`schema/model/s1` is out of date or stray, or a network fixture is stale,
`tests/test_model_network.py` fails. If the generated TypeScript is out of
date or a stray file is anywhere under `shared/model/s1`,
`worker/test/model_network.test.ts` fails. The worker test also validates the
fixture network, index, and day map with ajv. The observations fixtures
(`tests/fixtures/model/observations/`) are hand-written;
`tests/test_model_observations.py` round-trips them through Pydantic and the
parquet layout, and `worker/test/model_observations.test.ts` validates them
with ajv.
