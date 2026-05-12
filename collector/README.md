# Local MTA data collector

A tiny Docker container that polls the MTA developer gateway every 5 minutes (alerts) and hourly (elevators/escalators), appending every observation to a JSONL log on your local disk. The goal is to build a real-data corpus we'll train and validate the HMM regime model against — *before* the public Momentarily publisher is wired up.

This is local-only. Nothing is published; nothing leaves your machine.

## Prereqs

- Docker — either Docker Desktop or [Colima](https://github.com/abiosoft/colima) on macOS

That's it. The JSON feeds the collector polls are publicly accessible without authentication.

> An MTA API key will be needed later when we add protobuf trip-updates (real-time arrival ETAs), but not for v1 of the collector. See `.env.example` for the placeholder.

## Quickstart (Colima)

```bash
# Start Colima if it isn't running
colima start

# Build and run
docker compose -f collector/docker-compose.yml up -d --build

# Watch the logs
docker compose -f collector/docker-compose.yml logs -f
```

You should see lines like:

```
momentarily.collector: Collector starting. data_dir=/data poll_interval=300s ...
momentarily.collector: subway-alerts: appended 23 entities to 2026-05-11.jsonl
```

## Output

```
data/
├── alerts/YYYY-MM-DD.jsonl     # one line per alert observation per poll
├── ene/YYYY-MM-DD.jsonl        # one line per E&E feed snapshot per hourly poll
└── meta/
    ├── last_fetched.json       # latest successful fetch per feed (resume state)
    └── poll_log.jsonl          # every poll attempt with success/error info
```

Inspect the corpus from your host without docker exec:

```bash
# Number of alerts captured today
wc -l data/alerts/$(date -u +%Y-%m-%d).jsonl

# Last 5 alert observations
tail -n 5 data/alerts/$(date -u +%Y-%m-%d).jsonl | jq .

# Polls in the last hour
tail -n 12 data/meta/poll_log.jsonl | jq .
```

## Stopping / restarting

```bash
# Stop (data preserved)
docker compose -f collector/docker-compose.yml down

# Restart later — resumes from last_fetched.json with no data loss
docker compose -f collector/docker-compose.yml up -d
```

## What it's collecting

| Endpoint | Cadence | Why |
|---|---|---|
| `camsys/subway-alerts.json` | every 5 min | Mercury alerts feed — primary signal for HMM training |
| `nyct/nyct_ene.json` | every hour | Current elevator/escalator outages |
| `nyct/nyct_ene_upcoming.json` | every hour | Upcoming scheduled E&E outages |
| `nyct/nyct_ene_equipments.json` | every hour | Equipment registry (ADA pathway flags, location text) |

All endpoints are JSON-only (no protobuf). The collector matches what the eventual public Momentarily publisher will pull in v1, so the corpus seeds the publisher's initial HMM training.

## Disk usage

About 125 KB/day raw, ~45 MB/year. Trivial.

## Troubleshooting

**`docker compose` says it can't connect:** check `colima status`. If stopped, `colima start`.

**HTTP errors in the logs:** check `data/meta/poll_log.jsonl` — every poll attempt is logged there with success/error detail. If MTA is returning 4xx/5xx, the gateway may be having a hiccup; the collector retries on the next interval.

**No new entries in the alerts file:** confirm the container is actually running with `docker compose -f collector/docker-compose.yml ps`. If polls succeed but no entries appear, MTA may currently have zero active alerts — verify by tailing `data/meta/poll_log.jsonl` for `"status":"ok"` lines.

**Container restart loop:** the most common cause is a wrong `DATA_DIR` path or a permission issue on the bind-mounted volume. The container runs as user `collector` (uid 10001) and writes to `/data`. If your host directory has restrictive permissions, fix with `chmod -R u+rwX data/`.

## Related project tracker beads

- `homeassistant-mta-subway-5w0.8` — this work
- `homeassistant-mta-subway-5w0.5` — eventual publisher (will reuse this fetch logic)
- `homeassistant-mta-subway-5w0.6` — shadow rollout (corpus from here seeds Phase 1)
- `homeassistant-mta-subway-5w0.7` — HMM methodology that this corpus trains
