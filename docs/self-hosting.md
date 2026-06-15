# Self-hosting a Momentarily publisher

You can run your own Momentarily-compatible publisher — your own snapshot URL,
your own R2 bucket, independent of the iicky-operated `feed.momentarily.nyc`
instance. This is the live architecture from [ADR 0001](adr/0001-cloudflare-workers-r2-only-split-ts-python.md):
a TypeScript Worker does per-tick inference and writes the snapshot to R2; a
weekly Python training Container re-fits the HMM; R2 is the only state store.

## What you need

- A **Cloudflare account on Workers Paid** — the weekly trainer runs in a
  [Container](https://developers.cloudflare.com/workers/runtime-apis/bindings/containers/),
  which needs the paid plan. (The publisher Worker and R2 themselves are light.)
- An **R2 bucket**.
- The **`wrangler`** CLI (`npm i -g wrangler` or `bunx wrangler`).
- **No MTA API key.** The upstream feeds this uses (`camsys/subway-alerts.json`,
  the `nyct/nyct_ene*` elevator/escalator feeds, the GTFS-RT trip-updates) are
  served keyless from the MTA gateway. You are still bound by the
  [MTA developer agreement](https://api.mta.info/#/DataFeedAgreement) and must
  attribute the MTA in your snapshot (the publisher already does).

Budget ~15 minutes.

## A note on secrets: this repo uses `murk`, you don't have to

Every `deploy`/`dev` script in `worker/`, `trainer/`, and `viz/` wraps `wrangler`
in [`murk exec`](https://www.npmjs.com/package/@iicky/murk-secrets) — the
author's encrypted vault that injects `CLOUDFLARE_ACCOUNT_ID` and the `R2_*`
credentials. **You don't need murk.** Either set those as environment variables
in your shell before running `wrangler` directly, or use `wrangler secret put`
(below). Anywhere a script says `murk exec -- wrangler …`, you can run
`wrangler …` yourself with the env vars set.

## 1. Create the R2 bucket

```bash
wrangler r2 bucket create my-momentarily
```

## 2. Point the config at your account

In `worker/wrangler.toml`:

- set `[[r2_buckets]] bucket_name` to your bucket (`my-momentarily`);
- either remove the `[[routes]]` custom-domain block (you'll get a free
  `*.workers.dev` URL) or replace `feed.momentarily.nyc` with a domain you've
  added to Cloudflare.

Export your account id (read by `wrangler` from the environment):

```bash
export CLOUDFLARE_ACCOUNT_ID=<your-account-id>
```

## 3. Deploy the publisher Worker

```bash
cd worker
wrangler deploy            # the npm "deploy" script wraps this in murk + a
                           # git-sha provenance define; running wrangler
                           # directly is fine, code_sha just falls back to "unknown"
```

The Worker fetches the feeds, runs the forward filter, and writes
`v1/snapshot.json` to R2 on its cron (every 5 minutes). Your snapshot is now
live at your Worker URL (`https://<worker>.workers.dev/v1/snapshot.json` or your
custom domain).

## 4. Deploy the weekly trainer

The trainer Container needs R2 S3 credentials at runtime. Create an R2 API token
(Account → R2 → Manage API Tokens) and set the four secrets:

```bash
cd trainer
wrangler secret put R2_ACCOUNT_ID
wrangler secret put R2_ACCESS_KEY_ID
wrangler secret put R2_SECRET_ACCESS_KEY
wrangler secret put R2_BUCKET          # my-momentarily
wrangler deploy
```

It runs Baum-Welch weekly (Sunday 05:00 UTC) and republishes `state/params.json`;
the Worker picks up new params on its next tick. Until the first run, the Worker
uses bootstrap params.

## 5. (Optional) Eval + calibration

`training/eval.py` grades the published forecasts and writes `v1/calibration.json`
for the dashboard. Run it on a schedule (GitHub Actions, like `.github/workflows/eval-daily.yml`,
or any cron) with the same `R2_*` environment variables set:

```bash
R2_ACCOUNT_ID=… R2_ACCESS_KEY_ID=… R2_SECRET_ACCESS_KEY=… R2_BUCKET=my-momentarily \
  uv run python -m training.eval --days 7
```

## 6. Point consumers at your feed

Anything that reads the Momentarily snapshot (the
[homeassistant-mta-subway](https://github.com/iicky/homeassistant-mta-subway)
integration, a dashboard, your own code) just needs your snapshot URL in place
of `feed.momentarily.nyc`. The schema is identical and path-versioned (`/v1/`).

## Local development

See the `dev` scripts in each subproject. They use `murk`; without it, set the
env vars yourself and run `wrangler dev` / `uv run …` directly. The Python
package's tests need no credentials (`uv run pytest`).
