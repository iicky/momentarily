# Params rollback

How to revert `state/params.json` (and, when the same run touched them, its
sidecars) to a prior immutable snapshot after a bad trainer run ships live.

## When to use

- A weekly retrain (`training.train_em`, Sunday 05:00 UTC per
  `trainer/wrangler.toml`, or a manual
  `murk exec -- uv run python -m training.train_em`) published a fit that is
  live-wrong even though it passed every automated gate: the structural
  checks in `main` (severity floor, empty movement baseline, `MIN_DATA_DAYS`
  span) and the publish plausibility gate
  (`training.publish_params.implausible_params` — collapsed self-loop,
  per-route stationary-mix jump, fallback-fraction surge) only catch
  degenerate-but-non-empty fits and large jumps against the previously-live
  blob. A fit that is well-formed and close enough to the prior blob to clear
  those bounds, but still scores badly against real incidents, ships anyway.
- A standalone sidecar refresh — `training.backfill_service_baseline`,
  `training.ridership` (`ridership-weekly.yml`, Sundays 06:00 UTC), or
  `training.service_weight` (`service-weight-weekly.yml`, Sundays 06:30 UTC)
  — wrote a bad sidecar without touching `state/params.json` at all.
- **Not** for a first publish/bootstrap with no live blob to compare against
  (`--skip-plausibility`) — there is nothing to roll back to.

Rollback is a mitigation, not a fix: the next scheduled retrain runs again
regardless, and will re-ship the same bad fit unless the underlying data or
code issue is also addressed.

## Prerequisites

- R2 S3-compatible credentials: `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`,
  `R2_SECRET_ACCESS_KEY`, `R2_BUCKET` (`momentarily`), via `murk exec --` or
  exported directly — see [self-hosting.md](self-hosting.md). All commands
  below assume `murk exec --`; drop it and export the vars yourself if you
  don't use murk.
- The repo's Python deps installed (`uv sync`), run from the repository root.
- Nothing here needs a Worker deploy: the Worker reads `state/params.json`
  (and the sidecar keys) off a direct R2 bucket binding
  (`worker/wrangler.toml` `[[r2_buckets]]`), not an HTTP fetch, so there is no
  CDN cache in front of the objects this runbook rewrites.

## R2 keys and cache headers

Every live/versioned pair below is written by `training/publish_params.py`
(`training/ridership.py` and `training/service_weight.py` for the last two).
The `Cache-Control` header is what `_publish`/`write_baseline` set on the
object at write time; a rollback that uses `copy_object` with
`MetadataDirective='REPLACE'` (below) must repeat it explicitly, since
`CopyObject` does not infer it from the destination key.

| Artifact | Live key | Versioned prefix | Version suffix | Cache-Control |
| --- | --- | --- | --- | --- |
| params | `state/params.json` | `state/params/` | `trained_at` | `public, max-age=300, s-maxage=900` |
| segment_params | `state/segment_params.json` | `state/segment_params/` | `trained_at` | `no-store` |
| service_baseline | `state/service_baseline.json` | `state/service_baseline/` | `generated_at` (own stamp; equals `trained_at` on a full retrain, independent on a standalone `backfill_service_baseline`/refresh) | `no-store` |
| scheduled_headway | `state/scheduled_headway.json` | `state/scheduled_headway/` | `trained_at` | `no-store` |
| segment_dwell | `state/segment_dwell.json` | `state/segment_dwell/` | `trained_at` | `no-store` |
| ridership_baseline | `state/ridership_baseline.json` | `state/ridership_baseline/` | `generated_at` (own weekly stamp, independent of `trained_at`) | `public, max-age=300, s-maxage=900` |
| service_weight_baseline | `state/service_weight_baseline.json` | `state/service_weight_baseline/` | `generated_at` (own weekly stamp, independent of `trained_at`) | `public, max-age=300, s-maxage=900` |
| prov (private) | `state/prov/latest.json` | `state/prov/` | `trained_at` | `no-store` — audit trail; never rolled back |
| prov (public mirror) | `v1/prov/latest.json` | `v1/prov/` | `trained_at` | `public, max-age=31536000, immutable` — audit trail; never rolled back |

`segment_params`, `service_baseline` (on a full retrain), `scheduled_headway`,
and `segment_dwell` share `params.json`'s `trained_at` because
`training.train_em.main` publishes all four in the same run
(`write_params`/`write_service_baseline`/`write_segment_params`/
`write_segment_dwell`/`write_scheduled_headway` all take the one `trained_at`
computed for that run). `ridership_baseline` and `service_weight_baseline`
are refreshed on their own weekly schedule and version by their own
`generated_at` — check the sidecar's own JSON body for the value, don't
assume it matches `params.json`'s `trained_at`.

## Steps

### 1. Identify the bad live version

```bash
murk exec -- uv run python -c "
from training.r2_client import load_config, make_client
import json

cfg = load_config()
client = make_client(cfg)
live = json.loads(client.get_object(Bucket=cfg.bucket, Key='state/params.json')['Body'].read())
print('live trained_at:', live['trained_at'])
"
```

### 2. Find the prior good version

List `state/params/v<trained_at>.json` snapshots (the rollback trail —
`training.prune` keeps these for `PARAMS_RETENTION_DAYS = 180` days,
`training/prune.py`), newest last:

```bash
murk exec -- uv run python -c "
from training.r2_client import load_config, make_client
import re

cfg = load_config()
client = make_client(cfg)
keys = []
for page in client.get_paginator('list_objects_v2').paginate(Bucket=cfg.bucket, Prefix='state/params/'):
    for obj in page.get('Contents', []):
        keys.append(obj['Key'])
keys.sort(key=lambda k: int(re.search(r'v(\d+)\.json', k).group(1)))
for k in keys:
    print(k)
"
```

Pick the version immediately before the bad `trained_at` from step 1 (or
further back, if more than one recent run is suspect). Sanity-check it
before rolling back — e.g. fetch `state/params/v<good_trained_at>.json` and
skim its `training_corpus`/`hyperparams` for the window it was fit on.

### 3. Roll back the live pointer

```bash
murk exec -- uv run python -c "
from training.r2_client import load_config, make_client

cfg = load_config()
client = make_client(cfg)
GOOD_TRAINED_AT = 1788214744  # from step 2
client.copy_object(
    Bucket=cfg.bucket,
    CopySource={'Bucket': cfg.bucket, 'Key': f'state/params/v{GOOD_TRAINED_AT}.json'},
    Key='state/params.json',
    ContentType='application/json',
    CacheControl='public, max-age=300, s-maxage=900',
    MetadataDirective='REPLACE',
)
print('state/params.json rolled back to', GOOD_TRAINED_AT)
"
```

(Cloudflare R2 supports the same S3 `CopyObject`/`PutObject` API
`training/r2_client.py` already uses, so this needs no new tooling — no
`aws`-CLI profile is configured anywhere in this repo.)

### 4. Verify the pointer flipped

```bash
murk exec -- uv run python -c "
from training.r2_client import load_config, make_client
import json

cfg = load_config()
client = make_client(cfg)
live = json.loads(client.get_object(Bucket=cfg.bucket, Key='state/params.json')['Body'].read())
print('live trained_at:', live['trained_at'])
"
```

This must print the `GOOD_TRAINED_AT` from step 2/3, not the bad value from
step 1.

### 5. Repeat for each sidecar the same run also refreshed

For every sidecar in the table above that the bad run touched, copy its
`v<version>.json` (same `trained_at` for `segment_params`,
`scheduled_headway`, `segment_dwell`, and a full-retrain `service_baseline`;
its own `generated_at` for a standalone `service_baseline` refresh,
`ridership_baseline`, or `service_weight_baseline`) over its live key,
repeating that sidecar's exact `Cache-Control` from the table:

```bash
murk exec -- uv run python -c "
from training.r2_client import load_config, make_client

cfg = load_config()
client = make_client(cfg)
GOOD_VERSION = 1788214744  # trained_at or generated_at, per the table above
LIVE_KEY = 'state/segment_dwell.json'
VERSIONED_KEY = f'state/segment_dwell/v{GOOD_VERSION}.json'
CACHE_CONTROL = 'no-store'
client.copy_object(
    Bucket=cfg.bucket,
    CopySource={'Bucket': cfg.bucket, 'Key': VERSIONED_KEY},
    Key=LIVE_KEY,
    ContentType='application/json',
    CacheControl=CACHE_CONTROL,
    MetadataDirective='REPLACE',
)
print(LIVE_KEY, 'rolled back to', GOOD_VERSION)
"
```

Never roll back a sidecar that the bad run did not touch — its live object
is unrelated to the incident and a stray copy only discards genuinely newer
data.

## Cache invalidation

None of the objects this runbook rewrites are served over HTTP: `state/` is
private (`worker/wrangler.toml`'s Worker route gates public reads to `v1/`
only — see `docs/adr/0001-cloudflare-workers-r2-only-split-ts-python.md`),
and the Worker's own reads go through the `MOMENTARILY` R2 bucket binding
(`worker/src/params.ts` `loadParams`/`loadRidershipBaseline`/
`loadServiceWeightBaseline`), not a cached fetch. The `Cache-Control`
headers in the table above only matter if you ever read `state/*` keys
directly over R2's own S3 endpoint — they are not part of an edge cache to
invalidate.

The Worker's cron fires every minute (`worker/wrangler.toml`), with the
5-minute alerts/HMM/snapshot pipeline gated to run on 5-minute tick
boundaries. A rollback therefore takes effect on the Worker's very next
pipeline tick — **at most 5 minutes** after the copy in step 3/5 completes,
no separate purge step required.

## End-to-end verification

Confirm the change reached the live snapshot, not just the R2 object:

```bash
curl -s https://feed.momentarily.nyc/v1/snapshot.json | jq '.provenance.params'
```

(`.provenance.params.trained_at` is read from the `state/params.json` the
Worker just loaded, and `.provenance.params.key` is a pure string derivation
from it — `worker/src/params.ts`'s `versionedParamsKey(trainedAt)` — never a
pointer stored inside the params document. So copying a versioned object over
the live key is enough: the Worker names `state/params/v<trained_at>.json`
from the copied document's own `trained_at`.) `trained_at` must equal the
`GOOD_TRAINED_AT` from step 2, and `key` must equal
`state/params/v<GOOD_TRAINED_AT>.json`. Allow up to 5 minutes after step 3
for the next tick before checking.

If you self-host, substitute your own Worker URL — see
[self-hosting.md](self-hosting.md).
