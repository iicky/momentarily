# Momentarily — local dashboard

A local-only Next.js app for *seeing* the Momentarily feed and judging the HMM.
Not deployed, not part of the publish path — run it on your machine.

```bash
cd viz
npm install
npm run dev        # http://localhost:3000
```

## Two views

**Status** (`/`) — glanceable "what's running right now". Reads the public
`https://feed.momentarily.nyc/v1/snapshot.json` (no credentials), polls every
60s. Route grid colored by line, per-line regime probabilities, recovery ETAs,
feed freshness, accessibility rollup. Click a line for the full inference.

**Models** (`/models`) — does the model deserve trust? Reads the prediction and
regime-transition history from R2 and scores the forecasts against what actually
happened:

- **Recovery-forecast reliability** — when the model said "P(normal in 30/60/120m)
  = x", did lines actually recover that fast in fraction x of cases? (diagonal =
  calibrated). Brier score per horizon.
- **Predicted vs actual recovery** — scatter of forecast median against the real
  time-to-normal, with IQR coverage (target ~50%).
- **Regime timeline vs reality** — per-line swimlane of inferred regimes.
- **Learned transition matrices** — the trained 3×3 per line.

Ground truth comes from the transition stream, not the model's own labels, so
these are a real test. Predictions whose outcome isn't yet observable in the
window are censored out.

### Credentials (Models view only)

The grading streams are timestamped JSONL; reading a window needs an R2 LIST,
which the public Worker doesn't expose. So Models reads R2 directly, using the
`R2_*` secrets already in the project's **murk** vault.

`npm run dev` handles this: it runs Next under `murk exec`, which finds the
vault at the repo root and injects the R2 secrets into the server's
environment. No hand-managed keys, no `.env`.

In a fresh git worktree, run `murk-wt link` once. murk keys its identity on the
vault's absolute path, so each worktree looks in a key slot of its own and only
the original checkout has one; without the link you get "MURK_KEY not set".
`.bb-env-setup.sh` does this automatically for bb-managed worktrees.

If a `MURK_KEY_FILE` is already exported in your shell it overrides the linked
key, and a stale one fails with "not a recipient" rather than anything about
keys. Unset it and retry.

Credential resolution order (`lib/r2.ts`): `process.env.R2_*` (murk exec / CI /
`.env.local`) → the [`@iicky/murk-secrets`](https://www.npmjs.com/package/@iicky/murk-secrets)
bindings reading `../.murk` in-process. The bindings path activates once that
package ships a prebuilt native binary for your platform; until then the
`murk exec` path covers it.

Status needs no credentials. To run it alone without murk: `npm run dev:plain`.

## Test

```bash
npm test     # verifies the calibration math (Node's built-in runner)
```
