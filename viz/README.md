# Momentarily — local dashboard

A local-only Next.js app for *seeing* the Momentarily feed and judging the HMM.
Not deployed, not part of the publish path — run it on your machine.

```bash
cd viz
npm install
npm run dev        # http://localhost:3000
```

## Views

**Status** (`/`) — glanceable "what's running right now". Reads the public
`https://feed.momentarily.nyc/v1/snapshot.json` (no credentials), polls every
60s. Route grid colored by line, per-line regime probabilities, recovery ETAs,
feed freshness, accessibility rollup. Click a line for the full inference.

**Lines** (`/lines`) — one page per subway line. `/lines/<route>` lists the
line's stations in running order, each with its live service flow, ADA, and a
link to the station page. Pick a direction to walk the trip northbound or
southbound.

**Station** (`/stations/<gtfs_stop_id>`) — everything about one stop: the lines
it serves, borough, structure, coordinates and platform labels, ADA breakdown,
elevator/escalator outages, the live service-flow verdict, and every segment
that touches it.

**Map** (`/map`) — where the degradation is, across the whole system. One line
diagram, four overlays, one stroke per route per station pair, so the 4, 5 and 6
on the Lexington trunk are three separate strokes rather than one shared shape.
Hover for the reading, click to pin, scroll to zoom. Needs no credentials: the
geometry is a committed asset (see below).

The overlays answer different questions at *different spatial units*, and the
view names the unit under the map every time, because a reader who takes a
route-level reading for a segment measurement has been misled by the map rather
than by the feed:

- **Movement** — per `(route, direction, station pair)`, from
  `segment_flow.segments`. Direction selector is northbound / southbound /
  worst-of-both. A cell absent from the surface is *no reading*, drawn as the
  dimmed route colour and never as the healthy one — an absence of evidence is
  not a clean bill of health, and `viz/lib/segments.ts` has tests pinning that.
  A verdict lands only on the successor the cell names, because one cell key is
  shared by several drawn edges wherever a route branches or runs express.
  A healthy segment carries no recovery: there is nothing to forecast.
- **Supply** — per **route**, from `route_status[route].service_condition` and
  `service_ratio`. Every edge of a route wears its route's reading, so these
  strokes are dashed: nothing here was measured on the segment it is drawn on.
- **Scheduled time** — per drawn hop and direction, from the static timetable in
  `diagram.json`, selectable across NYCT's weekday / Saturday / Sunday classes
  and defaulting to today's in New York. A sequential violet-to-sky ramp, not
  the state colours: this is a magnitude, not a health verdict. Colour is rank
  within the class (the timetable writes almost every hop as 60, 90 or 120
  seconds, so equal intervals would collapse three quarters of the network into
  one bin) and the legend prints the real boundaries.
- **Trains** — per **stop**, from `v1/trains.json`, fetched separately from the
  snapshot and only while this overlay is selected. Filled disc = standing at
  the platform, ring = that plus the trains heading there, both by area. Nothing
  is drawn between stations: the feed names a stop, and which segment a moving
  train occupies is ambiguous at a branch.

**Trip** (`/map/trip`) — one line and direction at a time, drawing that trip's
pairwise segments on a geographic projection with the segment list beside it.
The system map answers "where is the network hurting"; this answers "what does
my ride look like end to end". Station coordinates come from NYS Open Data
39hk-dx4f; segment topology and canonical stop order come from the committed
diagram asset (below) — needs no credentials either.


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

## The diagram asset

`public/diagram.json` is the map's geometry — station positions, one edge per
(route, station pair), the `segment_flow` key that measures each edge in each
direction, and that hop's scheduled run time — plus the same static-GTFS
segment topology the trainer publishes to the credentialed
`state/segment_params.json`: `adjacency` (ranked successors per
`route|direction|from_stop`) and `route_stops` (scheduled stopping patterns,
most-run first). Carrying topology on the committed asset too is what lets
Lines, Station and the per-line Trip view read it with no R2 vault. Timing is
split by NYCT service class — `seconds: {weekday?, saturday?, sunday?}`, each
`{north?, south?}` in whole seconds — because the weekend timetable really is
a different schedule, not noise around the weekday one; a class or direction
the static timetable never gave a time for is absent from the object, never
published as 0. It's generated from the static GTFS feed and committed,
because the feed is a ~40 MB download and a full `stop_times` pass — not a
page load.

```bash
uv run python -m scripts.gen_diagram          # from the repo root
```

Regenerate after an MTA service change (new station, new branch, route
withdrawn). Output is deterministic for a given feed — no timestamp, sorted
keys, `provenance.code_sha`/`dirty` aside — so an empty diff means the
timetable didn't move. The producer is `training/diagram.py`; `lib/segments.ts`
is the one place that decides what the snapshot says about a segment, and
`lib/stations.ts` the one place that orders a line's stops, shared across the
Lines, Station and Trip views.

The layout is ours, derived from MTA's published stop coordinates and
timetable — not traced from MTA's map artwork. Edges run octilinear (the eight
compass bearings), with hops already within 10° of an axis left straight;
measured against this feed, every NYC hop is within 22° of an axis and the
median is 13°, so doglegging all of them would staircase the long gentle runs.

Stations are not snapped to a grid, which is this layout's ceiling: each hop is
routed on its own, so a long diagonal run still shows shallow sawtooth. Fixing
that means moving stations so consecutive hops share a bearing — a global
layout solve. Edges carry a `path` polyline, so when that lands it's a change
to `training/diagram.py` alone and the renderer doesn't move.
