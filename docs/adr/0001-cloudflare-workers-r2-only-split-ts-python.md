# 1. Cloudflare Workers + R2-only state + split TypeScript/Python

- Status: Accepted — live in production
- Date decided: 2026-05-12
- Last reconciled with deployed reality: 2026-06-14

## Context

Momentarily turns the MTA GTFS-Realtime Mercury alerts stream into a per-line
HMM estimate of each subway line's operational state (normal / disrupted /
suspended) plus an expected recovery time, published as a public JSON feed.

The original design (issues c72.3, c72.6) put the live publisher in **Python on
GitHub Actions cron**, writing the snapshot to R2 with boto3. That shape had two
problems:

1. GitHub Actions cron is best-effort with minute-plus jitter, and the MTA
   alerts feed moves on a 5-minute cadence — the live path wants a scheduler
   that actually fires every 5 minutes.
2. Running the whole Python toolkit (numpy, EM machinery) on every tick just to
   advance a forward filter is heavyweight for what is microseconds of math.

We needed a live path that fires reliably every 5 minutes, stays inside a free
tier, and keeps the per-tick work small — while still doing the expensive
Baum-Welch training somewhere it fits.

We considered a database for state (D1) versus object storage only (R2). D1's
free tier caps at 100K writes/day, which gets tight if we ever fan state out
per-alert; R2 has no such write-count limit and the state we need is small.

## Decision

Build on **all Cloudflare, R2 as the only state store, split by language**:

- **Live path — TypeScript on a Cloudflare Worker.** A Workers Cron Trigger
  fires every 5 minutes (`worker/wrangler.toml`, `crons = ["*/5 * * * *"]`),
  reads rolling state from R2, advances the HMM forward filter per route, and
  publishes the snapshot + grading streams back to R2. The forward filter is a
  TypeScript port of the Python reference (`worker/src/hmm.ts`). Inference only —
  no training in the Worker.

- **Training path — Python in a Cloudflare Container.** A separate cron-only
  Worker (`trainer/`) starts a container weekly (Sunday 05:00 UTC,
  `crons = ["0 5 * * SUN"]`) that runs `python -m training.train_em`:
  Baum-Welch EM over a rolling window of the R2 archive, writing fresh params
  back to R2. The Worker picks them up on its next tick.

- **Eval path — Python on GitHub Actions.** `training/eval.py` runs daily
  (06:00 UTC, `.github/workflows/eval-daily.yml`), grades the published
  predictions against observed outcomes, and writes `v1/eval.json` +
  `v1/calibration.json`.

- **State lives only in R2.** No database, no local files. `state/*` is the
  live rolling state, `archive/*` is the raw training corpus, `v1/*` is the
  public contract.

### Evolution from the original plan

The plan in c72/rbk described training and calibration as **laptop** work
("OFFLINE PATH — Python on laptop"). That was never deployed that way. Training
moved into a Cloudflare Container cron (`trainer/`) and eval into GitHub Actions,
so **live operation has no laptop dependency** — the whole pipeline is
serverless. The Python package (`src/momentarily/`, `training/`) remains the
reference implementation and the offline analysis toolkit (calibration notebooks,
`training/run_filter.py`), but it is not on the critical path.

## Realized topology

Three independent compute surfaces, all reading/writing the one R2 bucket
(`momentarily`):

| Surface | Where | Cadence | Writes |
| --- | --- | --- | --- |
| Publisher | Worker `momentarily-held` (`worker/`) | every 5 min | `v1/snapshot.json`, `v1/predictions/`, `v1/regime_transitions/`, `state/alpha.json`, `state/last_seen.json`, `archive/alerts/`, `archive/ene/` |
| Trainer | Container via Worker `momentarily-trainer` (`trainer/`) | Sunday 05:00 UTC | `state/params.json`, `state/params/v<trained_at>.json` |
| Eval | GitHub Actions (`eval-daily.yml`) | daily 06:00 UTC | `v1/eval.json`, `v1/calibration.json` |

The Worker's per-tick scheduled handler (`worker/src/index.ts`):

1. Read `state/last_seen.json`, `state/alpha.json`, `state/params.json` (with etags for compare-and-swap).
2. Fetch the MTA alerts feed; on failure fall back to the last good fetch.
3. Append new alert versions to `archive/alerts/YYYY-MM-DD/` (deduped by `(alert_id, updated_at)`).
4. Derive observations and run the forward filter per route; reseed alpha on a params-version change rather than resetting.
5. Persist `state/alpha.json` with an etag CAS — a losing tick skips the remaining publish steps.
6. Render and publish `v1/snapshot.json`.
7. Append the grading streams `v1/predictions/` and `v1/regime_transitions/`.
8. Hourly: fetch the elevator/escalator feeds into `archive/ene/`.
9. Persist `state/last_seen.json` (etag CAS).

## R2 layout

Public — served via the `feed.momentarily.nyc` custom domain. The domain is
bound to the **Worker**, not the bucket; the Worker's fetch handler gates reads
to the `v1/` prefix, so `state/` and `archive/` stay private
(`worker/wrangler.toml`).

```
v1/
  snapshot.json                       full system snapshot (max-age=60, s-maxage=300)
  eval.json                           self-grade metrics (max-age=300)
  calibration.json                    compact public calibration aggregate
  predictions/YYYY-MM-DD/<ts>.jsonl   one line per route per tick
  regime_transitions/YYYY-MM-DD/<ts>.jsonl  one line per regime flip
```

Private — Worker/trainer/eval only, never served:

```
state/
  last_seen.json                      alert dedupe map, feed freshness, cached station status
  alpha.json                          HMM posterior per route + published state
  params.json                         live trained params (Worker reads each tick)
  params/v<trained_at>.json           immutable versioned snapshots (rollback trail)
archive/
  alerts/YYYY-MM-DD/<updated_at>-<alert_id>.json   deduped alert versions (training corpus)
  ene/YYYY-MM-DD/HH0000-<source>.json              hourly elevator/escalator outages
```

## Language split and the shared contract

- **TypeScript** owns the live Worker: orchestration (`index.ts`), the forward
  filter (`hmm.ts`), alpha persistence (`alpha.ts`), snapshot rendering
  (`snapshot.ts`), grading streams (`grading.ts`), params reading (`params.ts`).
- **Python** owns training and analysis: `training/train_em.py` (Baum-Welch),
  `training/eval.py` (grading), `training/dwell.py`, plus the reference
  implementation under `src/momentarily/` (`hmm.py`, `schema.py`, `mapping.py`,
  `derive.py`).

The two sides meet at JSON-on-R2 contracts, not shared code:

- The Worker reads Python-written `state/params.json` (3×3 transition matrices,
  emissions by state and time-of-day bin, dwell quantiles) every tick.
- The Python trainer/eval read Worker-written `v1/predictions/` and
  `v1/regime_transitions/` to fit and grade.

Keeping these as versioned JSON schemas (rather than a shared library compiled
to two targets) is the seam that lets the two languages evolve independently.
The cost is that the forward-filter math exists twice (Python reference + TS
port) and the two must be kept in step by hand.

## Why this shape

1. **R2 is object storage, not a database.** Every write is a full-object PUT;
   there's no append-in-place. State is small (`state/*` is tens of KB), so a
   re-PUT each tick is fine, and `archive/` is append-only by day. That fit
   makes R2-only viable and sidesteps D1's 100K-writes/day free-tier cap.
2. **Worker CPU is enough for inference, not training.** A forward filter is
   microseconds; Baum-Welch EM is not. Splitting inference (Worker) from
   training (weekly container) keeps the hot path tiny and the heavy job off it.
3. **Workers Cron is a real scheduler.** Every-5-minute firing with low jitter,
   versus GitHub Actions cron's best-effort minute-plus jitter.
4. **It stays in the free tier.** The trainer container is "1/4 vCPU / 1 GiB —
   overkill for a ~90s EM run, and \$0 marginal inside the Workers Paid free
   allowance" (`trainer/wrangler.toml`). The Worker's ~288 invocations/day sit
   well under the 100K/day allowance, and public reads are R2 bytes-out behind
   the CDN edge cache, not Worker invocations.

## Consequences

Positive:

- No laptop and no always-on server in the live path — fully serverless.
- One storage system (R2) for live state, training corpus, and public contract.
- Inference and training scale and fail independently; a missed weekly train
  just means the Worker keeps using the last good `state/params.json`.
- The public surface is a plain CDN-cached JSON feed — cheap to read at scale.

Negative / watch items:

- **Duplicated HMM math** (Python reference + TypeScript port) must be kept in
  sync by hand.
- **state.json single-writer assumption.** Only the Worker writes `state/alpha.json`
  and `state/last_seen.json`; correctness relies on the cron firing sequentially.
  The etag compare-and-swap is the guard if that assumption is ever violated.
- **Public read traffic at scale.** If many consumers poll `v1/snapshot.json`
  every minute, R2 Class B (read) ops could pressure the free tier; the custom
  domain's CDN edge cache (via the cache-control headers the Worker sets) is what
  keeps origin reads low. Monitor.
- **Archive is deduped-by-version, not per-tick.** `archive/alerts/` rows are
  every distinct `(alert_id, updated_at)`, not every alert at every poll;
  training code reconstructs per-tick state from `active_period` intersections.

## Supersedes

- The Python-publisher-on-GitHub-Actions live path (c72.3 "build Python
  publisher", c72.6 local data-collection container as production source). The
  local collector's accumulated data became the bootstrap corpus for the first
  EM run; production archive is now the Worker's R2 writes.
