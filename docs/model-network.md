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
  86400 or more are legal (24:04:00 = 86640).
- **Intervals are half-open**, `[start, end)`. GTFS's inclusive `end_date`
  values (`calendar.txt`, `feed_info.txt`) are stored as end + 1 day.
- **No defaults**: no contract field has a default value. A missing value is an
  explicit `null` (or `Unknown` with a reason), so a missing measurement can
  never read as a normal value. `Model` rejects a subclass that declares a
  default.
- **Provenance**: `code_sha`, `code_dirty`, `schema`, `inputs` (name to digest),
  `applied_retroactively`, and `built_at`. `built_at` is the capture time of the
  stations snapshot, not the wall clock, so a rebuild gives the same bytes.

## Identities

| Thing | Identity |
| --- | --- |
| Feed | `feed_id` = sha256 hex of the GTFS zip bytes (the same key as `archive/gtfs/`) |
| Stations input | blake3 of the canonical snapshot file |
| Build | `build_id` = first 16 hex of blake3(`network/s1␟feed_id␟stations_digest`[`␟overrides_digest`]) |
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
    --stations OUT/stations-<digest16>.json --out OUT
# 3. Map each archived traversal day to the feed it was collected against.
uv run python -m momentarily.model.network day-feed-map \
    --traversals ARCHIVE/traversals --out OUT
```

`build` writes `OUT/<build_id>.json.gz` (gzip mtime 0, sorted keys) and adds an
entry to `OUT/index.json`. The build fails, and lists every problem, when any of
these is true:

- a GTFS parent station does not have exactly one stations-dataset row, or a
  dataset row has no GTFS station;
- a directional platform has no direction label (`--overrides` can fill a gap
  with a JSON object of stop_id to label; an override that fills no gap also
  fails the build);
- a trip does not resolve to exactly one pattern;
- a pattern's segments do not chain from end to end.

## Regenerating the contracts

```sh
uv run python -m scripts.export_model_schema        # schema/model/s1
(cd worker && bun run gen:model)                    # shared/model/s1
uv run python -m scripts.gen_model_network_fixture  # tests/fixtures/model/network_s1.json
```

If a schema is out of date, `tests/test_model_network.py` fails. If the
generated TypeScript is out of date, `worker/test/model_network.test.ts` fails.
The worker test also validates the fixture build with ajv.
