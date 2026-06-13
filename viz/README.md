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
which the public Worker doesn't expose. So Models reads R2 directly:

```bash
cp .env.local.example .env.local   # then fill in an R2 API token
```

Until that's set, the Status view works fully and Models shows a setup notice.

## Test

```bash
npm test     # verifies the calibration math (Node's built-in runner)
```
