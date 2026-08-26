# Engineering Journal

Append-only. When you discover something non-obvious that cost real effort to
find — a footgun, a wrong assumption, a number nobody would have guessed —
add an entry at the end. Never edit or delete an existing entry, even after
it's superseded; if a later finding changes the picture, add a new entry and
say which older one it revises.

## Format

Each entry is a `##` heading with a date and a short slug, an `origin:` line,
then the observation body:

    ## YYYY-MM-DD — short-slug
    origin: self | agent | artifact

    What was tried, what was found, and the number that proves it.

**`origin` values:**

- `self` — a human found this directly (debugging, reading code, reviewing output)
- `agent` — an AI agent found this while doing unrelated work
- `artifact` — surfaced by a build, test run, log, or generated report, not by a person or agent reasoning about it

## The rule

**Negative results always get recorded, and they MUST carry the number that
made them negative.** "Tried X, regressed" is not an acceptable entry — there
is nothing in it to check, reproduce, or compare against later. "Tried X,
held-out log-likelihood dropped from -812.4 to -1043.9" is acceptable. If you
can't state the number, the investigation isn't finished — go get the number,
or write down exactly which number you'd need and why it's not available yet.

---

## 2026-08-09 — worked-example: em-prior-strength-floor

origin: self

`training/train_em.py` fits the per-route HMM transition matrix with a
Dirichlet prior anchored at `prior_strength` pseudo-counts (default 100.0).
Tried dropping it to 20 to let thin-data routes move further from the global
prior: held-out log-likelihood on the March incident set fell from -812.4 to
-1043.9 over the same 20 EM iterations, and two routes (G, FS) started
oscillating instead of converging monotonically. Reverted to 100.0. Below
~40 the Dirichlet counts stop dominating the per-route transition counts for
anything but the highest-volume routes, so thin routes overfit their own
sparse incident history instead of leaning on the pooled prior. If this needs
revisiting, the number to chase is the per-route incident count above which
`prior_strength=20` stops hurting — untested here.


## 2026-08-11 — pooled-normal-dwell coverage is transition-gated

origin: agent

The pooled log-logistic AFT for normal-regime dwell was built to lift the
elapsed-conditioning fix off the four routes that cleared
`MIN_SAMPLES_FOR_EMPIRICAL`. It does, but not as far as the design implies: a
`train_em --dry-run` on the current 14-day window fits normal cells for 11 of
28 routes. `pooled_dwell_cells` reads `dwell_samples_by_cell`, which derives a
still-open regime from a route's *last transition record*, so the 17 routes
with zero transitions in the window supply neither an event nor a censored
observation and never reach the pooling step — the steadiest routes, which are
the ones the pooling exists to serve, are still the ones excluded. Those routes
carry 62% of normal route-ticks.

Counterfactual over 50,866 predictions (08-05..08-11), pooled curves vs the
live geometric projection, graded against the model's own condition at t+H:

| horizon | covered subset (n=19,217) | fleet (n=50,663) |
| --- | --- | --- |
| 30 min | Brier 0.01734 -> 0.00326 (-81%) | 0.01714 -> 0.01180 (-31%) |
| 60 min | 0.03746 -> 0.00392 (-90%) | 0.03895 -> 0.02623 (-33%) |
| 120 min | 0.06052 -> 0.00416 (-93%) | 0.06380 -> 0.04242 (-34%) |

On the covered subset mean predicted `p_normal_in_120` moves 0.7582 -> 0.9870
against an actual base rate of 0.9968, so the sharpness bias really is closed
where a cell exists. The number to chase is the fleet column: it only moves if
open regimes are reconstructed from prediction rows rather than from transition
records.


## 2026-08-11 — open regimes from prediction rows: 11 -> 29 normal cells

origin: agent

Follows the entry above, which named the number to chase. `_open_regimes` read
each route's still-open regime off its *last transition record*, so a route with
no transition in the window supplied neither an event nor a censored
observation. Sourcing open regimes from the prediction stream instead — every
live route is written every tick, and the row already carries
`regime_entered_at` — closes it.

Counterfactual on the same 14-day window (104 transitions, 122,438 predictions):
pooled normal-dwell cells go **11 -> 29**, gaining 2, 4, 6X, 7, A, C, E, F, FX,
G, GS, H, J, M, Q, R, SI, W. The 18 gained routes had held one regime for a
median ~450h, which is exactly why they never appeared in the transition stream.

The live blast radius is larger than the 11 suggests. Published params
(`trained_at=1785774438`) carry a `normal` cell for **4 of 28** routes — 6, B,
FS, H — so the other 24 run the geometric fallback. That is the source of the
overconfidence visible on the site: the 1 train, normal for 88h, publishes
`p_normal_in_30min=0.8814` against an empirical base rate of 0.999. With a
fitted cell the same route reads 0.9987, and the 18 newly covered routes average
0.9997.

Two footguns found while wiring it. Predictions load by whole-day prefix, so a
backdated `--end` reads rows past the censoring boundary and reports the regime
open *now* — `open_regimes_from_predictions` takes `window_end` and ignores
later rows. And `not_scheduled` must be dropped: it is not a regime the filter
dwells in, so banking scheduled downtime as a long healthy run would inflate the
very curve this corrects.


## 2026-08-11 — the forecast is anti-correlated with the arm consumers read

origin: agent

`training/eval.py` graded everything against `PredictionRecord.condition`, which
is the **alert-shadow**, not `published_condition`, the movement-primary arm the
site serves. `published_condition` was parsed into the record and never read.
`scorecard.py` had already migrated (7a15b1b); eval never did, so every headline
number described a model no consumer reads.

Grading the same `p_normal_in_H` forecast against each arm over 7 days:

| arm | n | Brier | BSS vs persistence | AUC |
| --- | --- | --- | --- | --- |
| shadow (`condition`), h30 | 55,390 | 0.01709 | -7.93 | **0.780** |
| movement (`published_condition`), h30 | 41,132 | 0.01605 | -16.37 | **0.145** |
| movement, h60 | 40,807 | 0.03640 | -40.27 | 0.144 |
| movement, h120 | 40,081 | 0.06059 | -55.48 | 0.142 |

AUC 0.142-0.145 is not weak, it is inverted: random is 0.5, and inverting the
forecast would score ~0.855. The mechanism is not subtle. Of the 29
movement-disrupted ticks in the window, **29/29 have shadow `condition ==
"normal"`** — `effectiveCondition` (worker/src/snapshot.ts) hard-returns `normal`
whenever `disruptiveAlertCount === 0`, so movement-disrupted-while-alerts-silent
is by construction the case the filter is most confident about. Mean
`p_normal_in_30min` is **0.9166 on movement-disrupted ticks vs 0.8780 on
movement-normal ticks**: the forecast is *more* sure of health exactly where
movement says the line is stalled.

Thin — 29 disrupted ticks (6:12, R:7, G:3, GS:2, J:2, M:1, W:1, 5:1) — so treat
the magnitude as provisional. The sign is not provisional; it has a mechanism.

Also note truth-map semantics. The movement truth must be dense over gradeable
ticks with `truth_default=None`, not sparse-with-absent-means-normal: 16.6% of
ticks publish `unknown` and 8.1% `not_scheduled`, and scoring those as calm
would credit the forecast for 24.7% of the window it was never tested on.
Gradeable share is 75.3%.


## 2026-08-11 — the movement arm was blind for three weeks and nothing noticed

origin: agent

Census of `published_condition` over its whole life (shipped 2026-07-11 in
15e4b65, 262,459 prediction rows to 2026-08-11, 97.8% carrying the field):

| period | unknown share | gradeable share |
| --- | --- | --- |
| 2026-07-11 | 3.7% | 92.2% |
| 2026-07-12 .. 2026-08-02 | ~100% | ~0% |
| 2026-08-03 | 64.1% | 34.9% |
| 2026-08-04 .. 2026-08-11 | 15-17% | 68-78% |

For 22 consecutive days the site published `unknown` for essentially every
route on every tick and no check fired. Coverage returns exactly at the
`trained_at=1785774438` retrain (2026-08-03T16:27Z), so the cause is an empty
movement advance-baseline in the published params — the same condition
`train_em` now warns about ("movement-primary condition will publish 'unknown'
for every route"). The warning exists; nothing downstream treats a fleet-wide
`unknown` as an outage.

Consequence for planning: the usable movement history is **8 days**
(2026-08-04 onward), not the 31 days the ship date implies. Any claim fitted on
"a month of movement data" is fitted on eight days.

Route-level movement episode census over the full window (73 episodes, all
closed, 17 routes): median duration **10 min**, min 5 min, max 155 min; 9 routes
clear MIN_VOTER_EVENTS=3 (3:19, GS:8, 4:7, 6:6, G:6, R:6, 1:5, D:3, J:3).

That median is the headline. The alert-shadow's episodes run to hundreds of
hours (one F "Severe Delays" held 459h, recorded in episodes.py). Movement
disruptions are a different phenomenon an order of magnitude shorter, so the
32.9-min recovery MAE the shadow reports is not a baseline the movement arm
should be compared against — most movement stalls are over in two ticks.


## 2026-08-11 — alert corroboration of movement episodes: 15.4% confirmed, confirmed slice sits at lead=0

origin: agent

`training/escalation.py`'s old tick-based `alert_confirmed_rate`/`evaporated_rate`
couldn't validate against real archived data (the escalation cohort filter and
the published-archive/offline-recompute source lock-in left it untested).
Rebuilt on `training/episodes.py` episodes (`confirmation_rate` + `lead_time`,
with `n_unconfirmed` reported and never turned into a rate) and ran it via
`murk exec` against real R2 archives, 2026-08-07..2026-08-11 (published_archive
source — the movement arm is live now, see the entry above): 40,455
predictions, 259 movement-disrupted episodes, canonical severity_floor=2 alert
truth as the corroborating set, ±30/60 min window around each episode's onset.

`confirmation_rate = 40/259 = 0.1544`; `n_unconfirmed = 219` (84.6% — not
disconfirmed, per the metric's own definition). Of the 40 confirmed, median
lead time is **0.0 min**, IQR `[0.0, 0.0]`: confirmation is almost entirely a
same-tick alert already active, not a later alert catching up to a movement
call that led it. Combined, the data does not show a "movement leads, alerts
confirm later" pattern in this window — it shows a small minority of
movement-disrupted episodes overlapping an alert that was already posted, and
the rest (84.6%) with no alert record either way, consistent with the 29/29
alert-silent finding two entries above. Number to chase next: split the 219
unconfirmed episodes by duration — short blips (movement-classifier noise) vs.
long stalls (a real alert-feed gap) look identical in this rollup.


## 2026-08-11 — static GTFS topology fixes segment-graph fragmentation: 44/52 chains vs 8/53 observed

origin: agent

`training/segments.py`'s `canonical_adjacency` (observed cross-tick modal
successor, gated at `MIN_SHARE>=0.5` in the live `train_em.write_segment_params`)
fragments the line graph. Measured directly against a real 14-day R2
vehicle-archive window (2026-07-28..2026-08-11, `canonical_adjacency` +
connected-components over each (route,direction)'s edge set): 53
(route,direction) groups have any observed adjacency; only **8/53** reduce to
one connected component under the `MIN_SHARE>=0.5` gate live today (33/53
without that gate, using only `canonical_adjacency`'s own `min_advances=3`
existence floor — so the reliability-share gate alone drops 25 previously-
connected groups to fragmented). This is the same failure mode previously reported
("2/53, Q north 45 links/14 components") but a different archive window, not
independently reproduced at that exact figure.

New `training/gtfs_static.py` parses the real NYCT static feed
(`https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip`, verified against
`http://web.mta.info/developers/data/nyct/subway/google_transit.zip` — a 301 to
the identical object, same ETag/Content-Length) directly: 52 (route,direction)
groups, 1797 (route,direction,from_stop) keys, **131** from_stops with more than
one static successor (real branch/express splits — Rockaway Park Shuttle +
Lefferts Blvd on the A, Dyre Ave on the 2/5 — confirmed by inspecting the
disjoint components, not archive noise). **44/52** groups reduce to one
connected component on the dominant-successor graph; the remaining 8 are
genuine branch tails that never reconverge (A south: 3 components; 5 north, 2
south, E north, N north, W south, R north, R south: 2 components each).

Combining both real datasets: of the 1795 (route,direction,from_stop) segments
with both a static successor and an observed cross-tick baseline this window,
**276 (15.4%)** were excluded from `segment_params.json` by the live
`MIN_SHARE>=0.5` gate despite the static timetable naming a real successor for
them — the number `MIN_SHARE`-as-existence-gate was actually costing.


## 2026-08-11 — movement regime debounce: debounce=2 erases 76% of route episodes

origin: agent

`training/movement_backfill.py` reconstructs the movement-regime transition
stream offline through `training.regime.replay_regimes`, from two independent
sources — `published_condition` (route scope only) and `archive/vehicles`
(route + segment scope, the only source that reaches segment granularity) —
and swept `debounce_ticks` in {1, 2, 3} against real R2 archives over the
usable window (2026-08-04..2026-08-11, via `murk exec`).

Route scope, from `published_condition` (this archived window predates the
regime clock landing this session, so it is the raw per-tick classifier call,
not a debounced regime):

| debounce_ticks | n episodes | median (min) | min | max | routes ≥3 eps | survival vs 1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 17 | 5.0 | 4.95 | 34.92 | 3 of 6 | 100% |
| 2 | 4 | 10.125 | 10.0 | 34.92 | 0 of 2 | 23.5% |
| 3 | 1 | 49.95 | 49.95 | 49.95 | 0 of 1 | 5.9% |

`debounce_ticks=2` (the shipped default, tuned for `derive_actual_recovery`'s
alert-count service metric, not movement) erases 13 of 17 episodes and every
episode under 10 minutes. The raw population's own median is 5.0 min — one
tick — so debounce=2 erases the median outright, not just a short tail;
debounce=3 leaves a single episode.

Segment scope, from `archive/vehicles` (the only source that reaches segment
keys; per-tick calls are already smoothed by a 6-tick/30-min trailing sum
before the regime clock runs at all — real archive median matched-trips per
segment leaf per tick is 1, ten times under `segments.py`'s
`MIN_MATCHED_TRIPS=3` floor):

| debounce_ticks | n episodes | median (min) | min | max | cells ≥3 eps | survival vs 1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 828 | 10.0 | 5.0 | 435 | 35 of 79 | 100% |
| 2 | 411 | 30.0 | 10.0 | 765 | 26 of 59 | 49.6% |
| 3 | 216 | 55.0 | 15.0 | 1090 | 25 of 49 | 26.1% |

Segment scope survives the same sweep far better (49.6% at debounce=2 vs
route's 23.5%) because the 30-min accumulation window already absorbs most of
the single-tick noise before the regime clock sees it — the extra debounce
is compounding on top of smoothing that already happened, not doing the same
job twice for free.

Cross-source agreement at debounce_ticks=1, route scope, over the same
window: of 88 `published_condition`-reconstructed transitions, 5 (5.7%)
matched an `archive/vehicles`-reconstructed transition (same route, same
new_state, `entered_at` within one tick); of 32 vehicles-reconstructed
transitions, 5 (15.6%) matched back. Agreement is low both directions. The
vehicles-source route reconstruction is thin by construction — it self-trains
`compute_advance_baseline` (min_samples=20 per (route, direction, tod_bin))
on only 8 days, versus whatever longer baseline window backed the deployed
Worker's own classifier — so this reads as a sample-size-limited finding, not
a settled disagreement between the two derivations.

Recommendation: **debounce_ticks=1** for both scopes, against the numbers
above — movement's own classifier already carries a per-tick significance
test (`classify_direction`'s binomial tail, alpha=0.05), and its real episode
population is dominated by single-tick events that `debounce_ticks=2` erases
almost entirely. The lead makes the final call.
## 2026-08-11 — debounce set to 1: the movement classifier is not noise-dominated

origin: self

Decision on the sweep in the entry above, which left the call open. Shipped
`DEBOUNCE_TICKS = 1` in both `training/regime.py` and `worker/src/regime.ts`.

The argument for debounce=2 is that a single-tick call is noise. That is
checkable, and it does not hold. Over 2026-08-04..08-11 the route classifier
opened **17 episodes in 65,685 route-ticks** (0.026%). If `classify_direction`
were firing at its nominal `CLASSIFY_ALPHA=0.05` binomial tail, the same window
would carry ~3,284 onsets — **193x** what was observed. The significance test is
not what binds; the `DISRUPTED_RATIO=0.5` posterior gate is, and it is far more
conservative than the alpha implies. A population that sparse is not one a
confirmation rule is filtering noise out of.

Against that, debounce=2 erases 13 of those 17 episodes and every episode under
10 min, when the population's own median is 5.0 min — one tick. It would remove
76% of what the site publishes today, and per the corroboration entry above we
have no evidence those detections are false: alert non-corroboration cannot
refute a movement call, and only 15.4% of movement episodes are corroborated at
all.

Segment scope also runs at 1. Its per-tick calls are already smoothed by a
30-min trailing accumulation before the clock sees them, so a debounce on top
compounds denoising that already happened (49.6% survival at 2).

What debounce=1 does NOT give up: the clock still back-dates on commit, an
abstention still holds a regime open rather than ending it, and a cell blind
for `MAX_IDLE_SEC` still evicts rather than resuming a regime it cannot vouch
for. Those three rules, not the confirmation count, are what make offline and
online segment identically. The debounce machinery stays and is still pinned at
2 by `tests/fixtures/parity_regime.json`.

The number to chase if this is revisited: an independent corroborator for
single-tick movement episodes. The trip-updates service metric
(`derive_actual_recovery`) is derivation-independent from vehicle positions and
was not used here.


## 2026-08-11 — clustered incidents: 0/1260 ticks show two disrupted segments adjacent

origin: agent

`training/incidents.py` (C7) clusters contiguous disrupted segments into
incidents on the premise that adjacent disruption overwhelmingly shares one
cause (a stuck train, a signal fault) rather than being independent draws.
Measured against the real 8-day archive (2026-08-04..08-11, `murk exec`):
segment states from `training.movement_backfill.segment_ticks_from_vehicle_bodies`
(this epic's canonical offline segment reconstruction — 6-tick/30-min trailing
accumulation, `classify_segment` against a self-trained baseline, the same
function the debounce entries above already validated), topology from the real
static GTFS successor graph (`training.gtfs_static.load_successors`: 1797
segments, 52 route-direction groups).

Of 2267 reconstructed ticks, 1260 (55.6%) had two or more disrupted segments
somewhere on the network at once. Of those 1260, **zero** produced a
multi-segment cluster — at `max_gap=0` and at `max_gap=1` alike. Mean cluster
size across every tick in the window: exactly 1.0. A same-route-direction cut
against an earlier (non-canonical, same window) reconstruction found the same
shape before the switch: of 130 route-direction groups with >=2 disrupted
segments at one tick, only 1 (0.77%) had an adjacent pair — ruling out "the
pairs are just on different lines" as the explanation. Over the full window,
`replay_incidents` reconstructs 872 incidents (870 completed, 2 still open at
the window boundary), median duration 600s (10 min = 2 ticks), median
footprint size **1.0 segment**.

This is a negative result against the stated premise: at 5-min tick /
30-min accumulation granularity, this real window essentially never shows two
adjacent segments reading disrupted at the SAME tick. It does not make the
incident-level composition wrong — summing/maxing independent per-segment
dwells would still misprice the rare tick where adjacency does occur, and 872
single-segment "incidents" is the correct degenerate output when there is
nothing to cluster, not a sign the clustering code is broken. But the
premise's imagined mechanism — a stuck train visibly stalling its physical
neighbor in the same window — is not what this classifier observes; a real
blockage's queueing effect on the segment behind it plausibly needs more than
one tick to become significant. Number to chase if this is revisited: a
LAGGED adjacency test (segment A disrupted at tick T predicting segment B
disrupted at tick T+k for small k), not attempted here.

## 2026-08-11 — v1/movement_transitions is empty: no real segment or route dwell fit yet

origin: agent

Went to run `training.segment_dwell.build_segment_dwell` on real archived
data via `murk exec --` to report segment-cell counts and an own/pooled
split. `list_objects_v2` on `v1/movement_transitions/` (no date filter, so
every day ever written) returns **KeyCount=0** — the stream that both the
route and segment regime clocks feed (`movementTransitions` +
`writeMovementTransitions` in `worker/src/grading.ts`) has never written a
single object to R2, at either scope.

This isn't a credentials or query problem: `v1/regime_transitions/` (the
older alert-driven filter stream) lists fine over the same window.
`state/movement_state.json` and `state/segment_flow.json` both exist and are
fresh (`LastModified` 2026-08-11 20:56 UTC, minutes before this check), so
the deployed Worker is actively computing regimes — `movement_state.json` is
only 378 bytes, consistent with a handful of routes still on their very
first cold-start entry. `writeMovementTransitions` only puts an object when
`changes.length > 0` (grading.ts), so a regime clock that has not yet
*changed* any cell's state writes nothing, even while it publishes
`state/*.json` every tick.

Net: the movement_transitions stream this session's dwell-curve work
(segment and, per the debounce entry above, route too) depends on has zero
real observations right now. `build_segment_dwell`/`compute_dwell_quantiles`
cannot be validated against production data until the deployed Worker has
run long enough, post-deploy, to commit at least one regime change per
scope. Tests in `tests/test_segment_dwell.py` exercise the estimator against
synthetic fixtures instead.


## 2026-08-11 — revises the entry above: 0/1260 is baseline-window-sensitive

origin: agent

Follow-up on "clustered incidents: 0/1260 ticks show two disrupted segments
adjacent" above, which self-trains `build_segment_baseline` on the FULL
8-day window before classifying every tick in it (same convention
`movement_backfill.py` uses for its own headline numbers). Reran each of the
8 days SEPARATELY — one self-trained baseline per single day — to check
whether the zero survives a shorter baseline window:

| day | disrupted calls | ticks w/ >=2 disrupted | adjacent (gap0) | share |
| --- | --- | --- | --- | --- |
| 08-04 | 3391 | 286 | 11 | 3.8% |
| 08-05 | 4236 | 286 | 15 | 5.2% |
| 08-06 | 3949 | 287 | 26 | 9.1% |
| 08-07 | 3601 | 285 | 0 | 0.0% |
| 08-08 | 4559 | 287 | 0 | 0.0% |
| 08-09 | 5089 | 287 | 0 | 0.0% |
| 08-10 | 4379 | 287 | 0 | 0.0% |
| 08-11 | 5210 | 251 | 37 | 14.7% |

Four of eight single-day baselines reproduce the zero; the other four don't,
and the non-zero share ranges 3.8-14.7% with no visible day-of-week pattern
in a single week. The 8-day-baseline run (this entry's parent) isn't wrong —
each segment leaf gets far more supporting data and therefore a more stable
`p0`, which is the more defensible baseline — but "zero" should be read as
"near-zero and sensitive to how much data trains the baseline," not as a
hard floor. A 1-day baseline is thin enough that some of what it flags
`disrupted` is baseline noise, and even the days that DO show adjacency stay
near mean cluster size ~1.0 (a handful of adjacent pairs among thousands of
singleton calls) — so the qualitative finding (adjacency is rare, nowhere
close to "overwhelming") holds on every single day measured; only the exact
count moves. Number to chase if this matters later: rerun the
8-day-baseline classification but score it against a HELD-OUT week (not
overlapping the classified window) to remove the self-training confound
entirely — not attempted here.


## 2026-08-11 — IRT/BMT/IND division does not earn a hierarchy level: eta² 0.16, p=0.081

origin: agent

New `training/divisions.py`: a static 28-route IRT/BMT/IND map (cross-checked
against the real GTFS `routes.txt` for the route-id space and against two
independent sources — Wikipedia's A/B Division articles and nycsubway.org's
"Subway FAQ" line-by-line lineage — for the division assignment; SI excluded,
it is a disconnected non-subway railway, never part of the three-division
network), plus a one-way ANOVA (between/within variance, eta-squared, and a
distribution-free permutation p-value, all pure stdlib) over per-route fitted
dwell scales.

Ran on the ALERT arm (`training.eval.load_transitions` + `pooled_dwell`),
not movement: movement route-scope coverage is 17 episodes over 6 routes
(prior entry), too thin to answer this; the alert arm fits 29 routes over
the same 14-day window `train_em` uses live.

`partially_pooled_dwell` on that window, grouped by division (n=28 mapped
routes, SI dropped):

| division | n | mean log-scale | within-division variance |
| --- | --- | --- | --- |
| BMT | 9 | 10.899 | 0.027 |
| IND | 9 | 10.841 | 0.023 |
| IRT | 10 | 10.529 | 0.401 |

MS_between=0.381, MS_within=0.161, F=2.373, **eta²=0.1596** (division
nominally explains 16% of fitted-log-scale variance), permutation
p=**0.0808** (10,000 draws, seed 0) — fails conventional significance with
only 3 groups to compare.

The 16% does not survive inspection. Only 6 of the 28 routes clear
`MIN_VOTER_EVENTS` and fit their own MLE (`source="own"`) instead of being
shrunk to the population MAP; the other 22 are single long right-censored
observations that land within a narrow band (log-scale 10.83-11.26)
regardless of division, because in this window they were all censored by
a similarly-long stretch of continuous `normal`, not because of any
division-level pattern. Of the 6 own-voters, 4 are IRT (1, 3, 5, 6 — route 3
alone contributes log-scale 9.057 off 12 real exits, the single biggest
outlier in the dataset) and only 1 each land in BMT (FS) and IND (B). IRT's
apparent separation is 4 specific routes' this-window churn, not a
division-wide effect — and BMT vs IND, the two divisions closest in the
classification scheme, are statistically indistinguishable here (10.899 vs
10.841, within each other's within-division noise). Restricting to the 6
own-voters directly (n=1,1,4 per division — too degenerate to trust, but
telling): eta² swings to 0.272 and F *drops* to 0.561 (p=0.529), the
opposite of what a real division effect would do with less shrinkage noise.

**Verdict: no.** A division level does not earn its place — the measured
effect is small (eta²=0.16), not significant (p=0.081) at n=28 split three
ways, and what signal exists traces to which specific routes happened to
churn during this 14-day window, not to IRT/BMT/IND membership. Re-run this
exact module once movement or alert coverage is wide enough that most
routes clear `MIN_VOTER_EVENTS` on their own MLE — the number to chase is
the `n_own`/`n_pooled` split (6/23 here) approaching parity.


## 2026-08-11 — movement dwell_movement fit: 31 cells (8 own, 23 pooled), disrupted median 5.1-6.8min matches the 17-episode count exactly

origin: agent

Ran `training.train_em._movement_dwell` for real over 2026-08-04..2026-08-11
(`murk exec`, dry-run — nothing written to R2). The committed
`v1/movement_transitions` stream is empty for every date in the window (0
keys under every `v1/movement_transitions/<date>/` prefix checked), so every
cell in this run came from the `reconstruct_movement_transitions` fallback:
88 route-scope transitions rebuilt from the live `published_condition` ticks
through the same regime clock (`DEBOUNCE_TICKS=1`) the Worker runs online.

31 (route, state) cells total: 25 `normal`, 6 `disrupted`, 0 `suspended`.
8 own-fit (`n >= MIN_VOTER_EVENTS=3`: C/R/Z/G/6 normal, R/G/6 disrupted),
23 pooled-parent. 25 of 28 known routes get a `normal` cell (the 3 missing —
6X, 7X, GS — published `unknown` for nearly the entire window per their own
prediction-stream history, so the clock never opened a regime for them, not
a bug); 6 of 28 get a `disrupted` cell (6, G, J, M, R, W); 0 get `suspended`.

Median `median_sec`: `disrupted` cells range 303-410s (5.05-6.83min, median
of medians 307s) — narrower than and inside the previously-measured 4.95-
34.92min route-episode range, so no runaway-median bug. `normal` cells range
10,108-272,317s, dominated by censored routes still holding their first-ever
observed regime at window end (min 10,108s is Z's own-fit fitted median off
11 real exits; the 272,317s value repeats across most zero-event routes
because they share one open-regime start: the point published_condition
became usable, 2026-08-04).

Cross-check against the debounce=1 episode census from the movement-backfill
entry (17 route episodes across 6 routes): summing `n` over the 6 disrupted
cells here gives 5+3+5+1+2+1 = **17**, and the 3 that clear
`MIN_VOTER_EVENTS` (R=5, G=3, 6=5) are exactly the "3 of 6 routes with >=3
episodes" cited there. Same source data, two independent code paths
(census vs. dwell fit), same numbers.


## 2026-08-11 — movement exit-destination split: 100% to-normal, closed by measurement not code

origin: agent

`movementRecovery` (worker/src/snapshot.ts) reads a disrupted/suspended
regime's exit probability straight as `p_normal_in_H`, unlike the alert-HMM
branch in the same file which multiplies by a trained `toNormal` share. That
was flagged as a simplification, not proven immaterial. Measured it via
`murk exec` + `training.movement_backfill.reconstruct_movement_transitions`,
route scope, `debounce_ticks=1`, per `from_state`, both independent sources.

`source=vehicles` (2026-06-21..2026-08-12, 53 days, VEHICLE-ONLY —
`load_r2.derive_movement_state` can never return `not_scheduled`, so this
source is structurally biased against non-normal exits; trusted only as a
high-volume cross-check, not the vocabulary answer):

| from_state | n exits | to_normal | dest breakdown |
| --- | --- | --- | --- |
| disrupted | 213 | 1.000 | normal=213 |
| suspended | 0 | n/a | (no completed episode in window) |

19 routes contribute to the 213, every one individually 100% to normal (7:45,
G:29, M:23, R:16, W:14, 6:14, E:11, J:10, D:10, N:9, A:8, C:5, 4:5, F:3, Q:3,
B:3, H:2, 1:1, 2:1, 5:1).

`source=predictions` (2026-08-04..2026-08-12, 8 days, the faithful published
vocabulary — trusted for the `not_scheduled` question the vehicle source
structurally cannot answer):

| from_state | n exits | to_normal | dest breakdown |
| --- | --- | --- | --- |
| disrupted | 17 | 1.000 | normal=17 |
| suspended | 0 | n/a | (no completed episode in window) |

Per-route on the faithful source: R:5, 6:5, G:3, J:2, M:1, W:1 — every route
100% to normal, none below the pooled figure. (This 17 is the same population
the 2026-08-11 dwell-fit entry above counted by a different code path — same
numbers, second confirmation.)

`suspended` completed zero route-scope episodes in either window — not "rare
and mostly normal", genuinely absent from both the 53-day high-volume archive
and the 8-day faithful stream. `disrupted` never exited to `suspended` or
`not_scheduled` either, on the source that can express `not_scheduled`.

Decision rule from the assignment: >=0.98 pooled to-normal on the faithful
source, no route materially below it. Measured 1.000 (17/17) with zero
exceptions across 6 routes, corroborated by 1.000 (213/213) across 19 routes
on the larger source. Verdict: **immaterial today, no code change.** Did not
build `movement_exit_split` — a trained-share params block, a worker
accessor, and a multiply in `movementRecovery` would all be exercising a
split that has not once fired in 61 days of combined archive. Replaced the
"no trained split exists" comment in `movementRecovery` with these numbers
and this date so the claim is checked, not assumed. Number to chase if this
ever needs revisiting: the first observed `suspended`-from or
disrupted-to-`not_scheduled` route-scope transition, whenever it happens.


## 2026-08-12 — 53-day incident clustering settles it: still doesn't hold, but not literally zero (3 real incidents)

origin: agent

Follow-up on "clustered incidents: 0/1260 ticks show two disrupted segments
adjacent" and its baseline-window-sensitivity revision (both 2026-08-11
above). Both were underpowered: 8 days, one baseline. Re-ran
`training.incidents.measure_premise` over the full vehicle archive
(2026-06-21..2026-08-12, 53 days, `murk exec`), scored against TWO
baselines built from the same 53-day vehicle archive: `self` (`load_r2.
build_segment_baseline`, self-trained on the measured window — the prior
run's weakness, still run here so the two can be compared directly) and
`published` (the live `state/segment_params.json`, `trained_at=1786492506`,
2532 cells — what the Worker actually classifies against, via the new
`published_baseline_cells` + `segment_ticks_with_baseline`).

Pooled 53-day result, both baselines:

| baseline | candidate ticks (>=2 disrupted) | adjacent ticks gap0 | adjacent ticks gap1 | share ticks gap0 | mean cluster size gap0 | share disrupted-segments-in-multi gap0 | gap1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| self (53-day self-trained) | 5257 | 21 | 21 | 0.40% | 1.0016 | 0.274% | 0.285% |
| published (live segment_params.json) | 5489 | 21 | 21 | 0.38% | 1.0016 | 0.271% | 0.283% |

gap=0 and gap=1 give the IDENTICAL 21 adjacent ticks on both baselines —
loosening `DEFAULT_MAX_GAP` still has no support. The two baselines land
within ~4% of each other on the denominator (5257 vs 5489 candidate ticks —
the published baseline calls slightly more ticks judgeable) and produce the
exact same 21 numerator. (8963 total reconstructed ticks either way; 5257
of them, 58.7%, have >=2 disrupted segments somewhere on the network — same
order of magnitude as the 8-day run's 55.6%, not a new finding on its own.)

Per-week breakdown (self-trained baseline; published tracks it within 1-2
ticks/week everywhere it differs):

| week | candidate ticks | adjacent ticks gap0 | share |
| --- | --- | --- | --- |
| 06-21..06-27 | 0 | 0 | n/a (no disruption observed yet) |
| 06-28..07-04 | 0 | 0 | n/a |
| 07-05..07-11 | 20 | 0 | 0.0% |
| 07-12..07-18 | 948 | 11 | 1.16% |
| 07-19..07-25 | 1185 | 9 | 0.76% |
| 07-26..08-01 | 1304 | 1 | 0.08% |
| 08-02..08-08 | 1255 | 0 | 0.0% |
| 08-09..08-12 | 545 | 0 | 0.0% |

5 of 8 weeks are exactly zero; all 21 adjacent ticks fall in a 3-week span
(07-12..08-01). Traced identity through the whole window with
`advance_incidents` (not just counting candidate ticks) to see how many
DISTINCT real incidents that span represents: exactly **3**, agreed on by
both baselines down to the segment keys and timestamps —

| incident | segments | first seen (UTC) | last seen (UTC) | span |
| --- | --- | --- | --- | --- |
| 1 | 7\|south\|701S, 7\|south\|702S | 2026-07-16 04:20 | 2026-07-16 07:55 | 215 min |
| 2 | A\|north\|A48N, A51N, A55N | 2026-07-24 18:15 | 2026-07-24 19:45 | 90 min |
| 3 | 7\|south\|701S, 7\|south\|702S | 2026-07-27 12:30 | 2026-07-27 12:45 | 15 min |

Two of the three are the SAME 7|south segment pair recurring (a specific
chronically slow spot, not a generic clustering tendency); the third is the
one 3-segment footprint in the whole 53 days, on the A up in the 140s-170s.
Zero incidents anywhere else on the network, on either baseline, in 53
days.

**Verdict: does not hold — confirmed, not overturned, but the literal
"zero" from the 8-day run is overturned.** 53 days surfaced 3 real
multi-segment incidents an 8-day window was simply too short to ever see;
they're genuine, not noise (both a self-trained-on-53-days baseline and the
independently-trained published baseline agree on the same 3, to the
minute). But 3 incidents against 2532 segment cells over 53 days, and
0.27-0.29% of disrupted-segment observations, is not "overwhelming" by any
reading — the premise's core claim (that adjacent clustering is the
dominant case, not the exception) still fails. **What changed the number:
the longer window, not the baseline.** Self-trained-on-53-days and
published converge almost exactly (same 21 ticks, same 3 incidents,
matching timestamps); baseline choice moved only the candidate-tick
denominator by ~4% and never flipped a single classification that mattered
to the verdict. Left `training.incidents.path_incident_durations` unwired
(non-goal: `train_em.py`/`snapshot.ts` publish nothing new here) and
rewrote the module docstring with this finding and window so the next
reader doesn't re-derive it from scratch.
## 2026-08-12 — suspended is not covered by the exit-split measurement

origin: agent

Revises the entry above ("movement exit-destination split: 100% to-normal,
closed by measurement not code"). Its measurement stands and its conclusion
for `disrupted` stands; its title is wrong on two counts, and the second one
was a live unsoundness.

The measurement covered `disrupted` at n=213 and n=17, both 100% to normal.
It covered `suspended` at **n=0**. Zero observations is not a measurement of
1.0 — it is the absence of one, and the entry then let a suspended cell's exit
probability be read directly as `p_normal_in_H` on that basis. The two states
are not interchangeable here: `disrupted` means trains are moving slowly and
recovering means they speed up, whereas `suspended` means there are no trains
on the route at all, and a route resuming service can plausibly come back
degraded before it comes back normal. Nothing in the archive says otherwise
because nothing in the archive says anything.

Fixed in code, so it was not closed by measurement alone: `movementRecovery`
now serves only the states whose exit destination has actually been observed
(`normal`, which needs no split at all since an exit from normal is never to
normal, and `disrupted`). Anything else returns null and falls through to the
alert arm, which has a trained transition matrix and a real `toNormal` share.
Pinned by a test that fails when the guard is removed (verified by deleting
the line: 8 pass / 1 fail, restored: 9 pass).

Live blast radius today is zero — the published fit carries 25 `normal` and 6
`disrupted` cells and **no `suspended` cell**, so the guarded path was
unreachable in production. It would have become reachable the first time a
suspended movement episode completed and pooling produced a cell for it, which
is exactly when a silently-wrong `p_normal_in_H` would have started shipping.

The number to chase if this is revisited: the to-normal share for exits from
`suspended`, which needs at least one completed suspended route-scope episode
to exist before it can be estimated at all.

## 2026-08-11 — movement-curve forecast counterfactual: AUC 0.150→0.762 at 30min, still inverted at 120min

origin: agent

Counterfactual over the archived window where the movement arm is usable
(2026-08-04..2026-08-12, 67,309 rows; 456 excluded because they already carry
`recovery_source=='movement'` — produced by today's deployed code, not the old
alert-driven path, so they aren't a fair "old" sample — leaving 66,853 graded
rows). For each row, reconstructed the movement regime `(state, entered_at)`
per (route, tick) by replaying `published_condition` through
`training.regime.advance_regimes` (debounce=1, the same primitive
`training.regime.replay_regimes`/`movement_backfill` use, just keeping every
intermediate tick instead of only the final one), then recomputed
`p_normal_in_H` with the LIVE `dwell_movement` curve via
`training.dwell.p_leave_by`, mirroring `movementRecovery`'s exact guards:
normal → `1 - p_leave_by` (survival), disrupted → `p_leave_by` directly, any
other state/no curve/no clock/schedule-precedence → fall back to the archived
(old) value. Graded against `published_condition` at t+H via
`training.eval.calibrate(..., truth_default=None)` — unknown/not_scheduled
ticks drop out rather than scoring as calm.

Movement path served 50,722/66,853 rows (75.9%; 50,696 normal + 26
disrupted); 11,131 (16.7%) fell back for no `published_condition` reading that
tick, 5,000 (7.5%) for a state the movement path doesn't serve
(not_scheduled/suspended); zero rows fell back for a missing curve, a
non-positive clock, or schedule precedence.

| horizon | n | AUC old→new | Brier old→new | BSS vs persistence old→new |
| --- | --- | --- | --- | --- |
| 30min | 50,061 | 0.150→0.762 | 0.01587→0.00066 | -16.27→+0.29 |
| 60min | 49,831 | 0.150→0.493 | 0.03609→0.00102 | -39.87→-0.16 |
| 120min | 49,201 | 0.150→0.397 | 0.06004→0.00178 | -58.08→-0.75 |

Positive-class (published-normal at t+H) base rate is ~99.95% at every
horizon. The mean-split that named the bug, `p_normal_in_30min`: old 0.9209
disrupted vs 0.8788 normal — backwards, same sign as the originally-reported
0.9166/0.8780 — vs new 0.9856 disrupted vs 0.9932 normal, sign flipped:
correctly less confident where the line is stalled. The disrupted-current-tick
sample is thin (n=26, entirely on the 6 routes with a fitted disrupted curve —
0 fell back to no-curve), so treat the AUC magnitude as provisional.

**The inversion is gone at 30min (0.150→0.762) and no longer inverted at
60min (0.493, essentially chance) but is STILL INVERTED at 120min (0.397 <
0.5)** — independently re-derived by hand (matched-sample rebuild + a
from-scratch Mann-Whitney) to rule out a scoring bug. New beats a naive
persistence baseline only at 30min; at 60 and 120min it loses to persistence,
far less catastrophically than old (BSS -0.16/-0.75 vs old's -39.87/-58.08)
but it still loses. Likely driver: of the movement-path "normal" rows, 71.4%
carry `entered_at` pinned to that route's very first in-window reading — the
regime almost certainly predates the 8-day archive window (pre-window data is
92.8% `unknown` and any real signal is evicted by the 1-hour idle rule before
it could bridge to the window anyway), so true elapsed time is understated
throughout the window and the understatement compounds as the horizon grows.

Self-consistency caveat: both the truth (`published_condition` replayed
through the regime clock) and the new forecast (the movement dwell curve
trained on `published_condition` transitions) derive from the movement arm.
This is the correct apples-to-apples comparison — the 0.142-0.145 figure that
named the bug was computed the same way, against the same arm — but it is not
evidence of independent skill, and does not validate the movement classifier
itself.


## 2026-08-11 — independent-truth grade of the movement forecast: trip-updates MAE 251.8min, movement CRPS underwater (skill -0.10, -0.36 held out)

origin: agent

Grades the movement arm against truth it did NOT come from — the counterfactual
entry above grades it against `published_condition` (the arm's own truth, same
source problem the eval-honesty doc warns about). Two independent checks, plus
a live coverage/adoption/corroboration refresh.

**Archive coverage, verified (not assumed identical):** `archive/trip_updates/`
is 59 days, 2026-06-15..2026-08-12 — starts 6 days EARLIER than
`archive/vehicles/` (53 days, 2026-06-21..2026-08-12). `v1/predictions/` is 91
days, 2026-05-14..2026-08-12. Largest common window for Part 1 is therefore
2026-06-15..2026-08-12, gated by trip_updates.

**Part 1 — trip-updates `assigned_n` truth (`training.load_r2.derive_actual_recovery`,
`training.eval.independent_recovery_metrics`, defaults).** 16,674 trip-updates
ticks → 132 (route, tod_bin) baseline cells (>=20 samples) → 1,054 independent
disruption intervals (median 70min, min 10min, max 4,880min/81.3h — this truth
is dominated by long service reductions, not movement-scale minutes). Graded
`recovery_minutes` against them: **n=362 graded ticks (49 regimes), MAE=251.8min,
RMSE=490.6min, IQR coverage=21.0%**; per-regime macro-average MAE=178.2min,
coverage=17.2%; 19,798 schedule-sourced rows excluded per the existing arm
contract. Window coverage: 39.1% unknown, 56.4% gradeable (this window still
contains the 22-day unknown gap below).

Signed bias (predicted − actual_remaining, same matching as
`independent_recovery_metrics`, sign kept): mean **-140.8min**, median
**-29.7min** — `recovery_minutes` under-predicts the trip-updates truth's
remaining time on average. Direction, stated plainly per the semantic gap: the
two signals measure different things (trip-updates counts dispatched trains,
movement times station-to-station advance), so this is not automatically an
arm error — but the sign says the movement arm calls "recovered" *before* the
fleet count does, the "thinned but flowing" case, not the "full fleet crawling"
case. MAE at 251.8min is large in absolute terms, but a truth series whose
median disruption alone is 70min (and tail runs to 81h of planned-work-scale
reductions) is a different population than the movement arm's own minutes-long
episodes (see Part 2) — the two truths are not commensurate, which is itself
part of the honest answer here, not an excuse for the number.

**Part 2 — episode-level CRPS/PIT, `training.scorecard.episode_recovery` +
`movement_dwell_lookup_from_params`, movement episodes from
`training.episodes.extract_episodes` over `training.load_r2.build_movement_truth`
(causal baseline: 2026-06-21..07-04, 3,826 bodies -> 241 cells; graded window
2026-07-05..08-12, 10,936 bodies -> 242,252 truth cells, mirroring
`training.review`'s own causal-baseline convention).** 130 movement episodes
extracted (0 standing, all peak_state=disrupted — no suspended movement
episode exists to grade, confirming the earlier 0-suspended finding again),
median duration 5.0min, min 4.95-5.0min\*, max 585min, 20 routes. Live
`dwell_movement` (trained_at=2026-08-11 23:55 UTC, window 2026-07-29..08-11)
covers disrupted curves for only 6 of those routes (6, G, J, M, R, W —
matching the live-params fact); the other 14 routes' episodes have no curve:
**n_scored=76, n_no_curve=54, n_censored=0**.

CRPS **mean=5.17min vs baseline(climatology)=4.69min → skill=-0.103**: the
movement dwell curve scores 10.3% WORSE than just predicting each episode's
own empirical duration distribution. Baseline confirmed built over the graded
movement-episode population itself (`episode_recovery`'s own contract,
`recovery_dist.py:160`) — never the alert-shadow's hundreds-of-hours
population; the 76 scored durations here are minutes, not hours, so no
cross-population dilution occurred. `mean_pit=0.494` (0.006 below 0.5,
negligibly pessimistic) — but the 10-bin histogram is bimodal, not uniform:
`[0,0,38,0,14,0,0,2,8,14]` — 50% of mass sits at PIT∈[0.2,0.3) and 29% at
PIT>=0.8, canceling to a near-0.5 mean while masking two real failure modes
(recovers much faster than predicted, and recovers much slower than
predicted) in roughly equal measure. The module's own verdict text calls this
"Well calibrated" on the mean but flags it directly: "scores 10% worse than
the dead-simple baseline — calibrated, yet no sharper than guessing the
average. Calibration isn't skill."

Checked whether this is a training-window leakage artifact (the live params'
14-day training window, 2026-07-29..08-11, overlaps 14 of our 39 graded
days): split by episode onset against that boundary. Held-out (onset before
2026-07-29, predates the params entirely): **n=26, skill=-0.361** — WORSE, not
better. In-sample-or-later (onset >= 2026-07-29): n=50, skill=-0.117. The
negative skill is not explained by grading the model against its own training
data — it holds, and deepens, out of sample.

**Part 3 — deployed system health, live R2 (all times UTC; UTC date rolls at
the point the codebase's deploy commits land in NY time, so "today"=08-12,
"before deploy"=08-11).**

Coverage, last 14 days + today:

| date | unknown share | gradeable share |
| --- | --- | --- |
| 07-30 | 100.0% | 0.02% |
| 07-31 | 99.9% | 0.08% |
| 08-01 | 100.0% | 0% |
| 08-02 | 100.0% | 0% |
| 08-03 | 64.1% | 34.9% |
| 08-04 | 16.9% | 78.4% |
| 08-05 | 17.3% | 77.8% |
| 08-06 | 17.1% | 78.2% |
| 08-07 | 16.8% | 78.3% |
| 08-08 | 16.3% | 70.8% |
| 08-09 | 16.5% | 68.9% |
| 08-10 | 15.3% | 77.6% |
| 08-11 | 16.3% | 78.1% |
| 08-12 (today, n=522, partial day) | 13.8% | 82.8% |

Coverage is healthy now: today's 13.8% unknown / 82.8% gradeable sits inside
(slightly better than) the stable ~15-17%/~78% band every day has held since
the 07-12..08-02 empty-baseline gap closed on 08-03. Nothing resembling the
22-day 100%-unknown blackout has recurred.

Adoption, today (n=522, post-deploy) vs before deploy (n=8,352, 08-11, almost
entirely pre-deploy — the deploy landed ~23:48 UTC 08-11):

| field | value | before deploy | today |
| --- | --- | --- | --- |
| recovery_source | movement | 0.29% (24) | **82.76% (432)** |
| recovery_source | hmm | 99.71% (8,328) | 17.24% (90) |
| condition_source | movement | 83.66% (6,987) | 86.21% (450) |
| condition_source | unknown | 16.34% (1,365) | 13.79% (72) |
| published_condition | disrupted | 0.14% (12) | 0% (0) |

`condition_source` was already movement-majority BEFORE today's deploy — this
matches the bug's own framing exactly: published status was already
movement-derived, only recovery/forecast were gated behind the alert-HMM.
`recovery_source` is the field that moved, 0.29% -> 82.8%. This is adoption,
not performance: zero disrupted rows exist today (0/522) to grade the new
forecast's live skill against, so nothing above is a skill claim.

Alert corroboration, refreshed (`training.escalation.corroborate_episodes` +
`confirmation_summary`, 2026-08-04..08-12, 9 days vs the prior 5-day
08-07..08-11 window; `esc_source=published_archive`, coverage/usable gates
both passed so no offline-recompute fallback needed): **361 movement-disrupted
episodes (up from 259), confirmation_rate=41/361=11.4% (down from 15.4%/259),
n_unconfirmed=320**, median lead time among the 41 confirmed = 0.0min (IQR
[0.0, 0.0]) — same shape as before, confirmation is almost entirely a
same-tick alert already posted, not a movement call leading a later alert.
Per `training/escalation.py`'s own contract, `n_unconfirmed` is not divided
into a rate and alert absence is not computed as a false-positive/precision/
specificity number against movement — that would resurrect the asymmetry the
module exists to avoid.

**Bottom line:** post-deploy live data cannot grade the new code yet (435-522
rows so far today, zero disrupted). Both counterfactual checks that CAN run —
this entry's independent-truth ones and the AUC counterfactual above — agree
the fix is real but incomplete: `p_normal_in_H` uninverted at 30min in the
sibling's counterfactual, but the movement dwell curve's own recovery-time
distribution scores below climatology here (CRPS skill -0.10, -0.36 held
out), and the trip-updates truth disagrees with `recovery_minutes` by a
251.8min MAE that this window's differing truth-population scale only
partially explains. Deployed and directionally correct is not the same claim
as calibrated or skillful; this entry's numbers are the ones to revisit once
live disrupted data exists to grade directly.

\*The 4.95min figure is from the constraints' prior 8-day/predictions-sourced
measurement, not this run (whose vehicle-sourced episodes bottom out at one
tick, 5.0min, by construction — `TICK_SECONDS=300`).


## 2026-08-11 — forecast horizon inversion: projection defect, not left-censoring (trustworthy-clock subset still inverted, AUC 0.352 at 120min)

origin: agent

Splits the counterfactual two entries above ("movement-curve forecast
counterfactual: AUC 0.150→0.762 at 30min, still inverted at 120min") by
clock trustworthiness, to answer whether the 60/120min inversion is
left-censoring (fixes itself with runtime) or a genuine horizon-projection
defect (needs a code/model fix). Same method, same window
(2026-08-04..08-12, 66,888 graded rows today — the archive grew a few rows
since that entry's 66,853), same live params (`trained_at`
2026-08-11T23:55:06Z) — but every row is additionally tagged by replaying
`published_condition` through `training.regime.advance_regimes` (debounce=1)
and comparing its regime's `entered_at` to that route's own first in-window
reading: `trustworthy` if the regime demonstrably started inside the window,
`censored` if the regime was already open at window start (true start
unknown, elapsed understated). Of the 50,722 movement-path rows, 36,184
(71.4% — matches the diagnosed figure exactly) are censored, 14,538 (28.6%)
are trustworthy.

| subset | 30min AUC / BSS (n) | 60min AUC / BSS (n) | 120min AUC / BSS (n) |
| --- | --- | --- | --- |
| all (sanity) | 0.762 / +0.29 (50,061) | 0.493 / -0.16 (49,831) | 0.398 / -0.75 (49,297) |
| trustworthy | 0.728 / **+0.40** (14,278) | 0.395 / **-0.00** (14,053) | 0.352 / **-0.53** (13,606) |
| censored | 0.624 / -0.32 (35,783) | 0.449 / -0.63 (35,778) | 0.317 / -1.30 (35,690) |

**Interpretation rule applied as specified: the trustworthy subset does NOT
recover AUC above 0.5 at 60 or 120min (0.395, 0.352) — it is still inverted
with a demonstrably real, in-window, uncensored clock. Verdict: PROJECTION
DEFECT, not left-censoring** — the 60/120min inversion will not fix itself
as the deployed system accumulates runtime. Left-censoring is real and does
matter, but only at 30min: trustworthy beats persistence there (BSS +0.40)
while censored loses (BSS -0.32), a clean split. By 60min censoring stops
predicting forecast quality at all (BSS -0.00 vs -0.63) and by 120min
trustworthy is still decisively negative (-0.53) — waiting for more routes'
clocks to "age in" will not turn that positive.

Honesty check on the deciding number: AUC is a discordant-pair statistic and
the positive-class base rate is ~99.95%, so it lives on the thin negative
(not-normal-at-t+H) tail. Counted directly: all/trustworthy/censored carry
26/19/7 negative rows at 30min, 26/15/11 at 60min, 25/9/16 at 120min — the
trustworthy subset's 120min AUC=0.352 rests on just **9** discordant rows,
thinner than the 25 behind the original headline number. Treat that single
AUC value as directionally right, not statistically final. The BSS columns
are a sturdier witness for the same verdict: Brier is a proper score over
the full n (13,606-35,783 per cut, not just the discordant tail), and it is
decisively negative in every 60/120min cut regardless of censoring status —
three independently-thinned populations agreeing that the forecast loses to
naive persistence is not the kind of thing 9 rows can flip.

A non-monotonic-in-elapsed diagnostic sharpens the "not censoring" case
further: bucketing the trustworthy-normal rows by elapsed time at 120min,
AUC is 0.290 for 6-24h elapsed (n=5,406, the worst band), 0.829 for 24-48h
(n=1,778, the best), and 0.472 for 48h+ (n=3,361, chance). A pure
understated-elapsed story predicts smooth degradation with elapsed; this is
a trough at 6-24h next to a strong band at 24-48h, more consistent with the
fitted normal-dwell curve's shape being wrong in specific places (plausibly
a missing time-of-day/periodicity covariate) than with a uniform clock-age
effect.

**Experiment 3 (sanity: is the horizon math itself buggy?):** checked
`p_normal_in_H` is non-increasing in H, for every one of the 25 live
`dwell_movement` normal cells, across 950 elapsed values per cell (0 up to
10x each cell's own max curve value, into the tail-extrapolation regime) and
48 horizons (5-240min, 5min steps) — 45,600 total comparisons, **0
violations**. The horizon-projection code is not buggy in the trivial
monotonicity sense; the defect is in what the curve says P(stay normal) is
conditional on elapsed, not in the H-step math.

**Experiment 2 (how long is "patience", if it mattered):** 25 routes carry a
live `dwell_movement` normal cell (matches the known count). 14 of 25 (56%)
share one identical pooled-parent curve — median 85,057s (23.6h), q75
592,496s (164.6h) — so most routes' "typical normal regime" figure is a
shared cross-route average, not their own history; own-fit routes range from
Z at 16,006s (4.45h) to M at 77,495s (21.5h). The distribution of per-route
median normal dwell across all 25 routes has p50 = p75 = 85,057s (23.6h),
both landing on the pooled plateau since it covers >50% of routes. Verified
the deploy boundary directly from `recovery_source` adoption, hour-binned:
0% at every hour through 2026-08-11T22:00 UTC, 6.9% at 23:00, 82.8% by
2026-08-12T00:00 — confirms the ~23:48 UTC 08-11 deploy time cited two
entries above. Deploy + 85,057s = **2026-08-12T23:25:37Z** for both the
50%-of-routes and the 75%-of-routes milestone — about one day, not the
~75h a prior fit (cited in this investigation's brief) had suggested. That
optimism is moot given the verdict above: reaching "trustworthy" in a day
does not fix an AUC that is already inverted for the trustworthy subset
today.

**Recommendation:** keep publishing `p_normal_in_30min` — real, repeatedly
measured skill (BSS +0.29 to +0.40 across every cut run today and in the
entry above). Stop publishing `p_normal_in_60min` and `p_normal_in_120min`
as skillful forecasts: both lose to naive persistence in every population
cut tested (BSS -0.00 to -1.30), the loss is not a censoring artifact (the
uncensored subset loses too), and Experiment 3 rules out a fixable
monotonicity bug — this needs a model-form change (the elapsed-conditional
normal-dwell curve itself), not more runtime.

Commands run (`murk exec -- uv run python <scratch script>`, PYTHONPATH set
to the repo root; script lived only in /tmp, deleted after the run, nothing
left in the repo): loaded `v1/predictions/` for every date 2026-08-04
through 2026-08-12 inclusive (`training.eval.load_predictions`),
`state/params.json`
(`training.r2_client.get_object_bytes`), replayed
`training.regime.advance_regimes` tick-by-tick over `published_condition`,
recomputed `p_normal_in_H` via `training.dwell.p_leave_by` against
`training.scorecard.movement_dwell_lookup_from_params`, graded via
`training.eval.calibrate(..., truth_default=None)`.


## 2026-08-12 — movement recovery distribution: model form, not data volume (mixture fix cuts the CRPS deficit from -0.088 to -0.023 using the same data)

origin: agent

**Verdict: MODEL FORM.** The decisive number: refitting the shipped
continuous log-logistic dwell curve as a one-tick point-mass mixed with a
conditional continuous tail — same 06-21..07-04 training data, same
held-out 97-episode population, zero additional days — cuts the CRPS-skill
deficit from **-0.088 to -0.023** (a 74% reduction). Meanwhile the causal
training-window sweep below shows no monotonic gain from more data at all
(skill oscillates -0.14 to -0.24 from 7 to 35 days of training, and gets
*worse* from day 21 to day 28). Four experiments, detailed below, all point
the same way: not data volume, not pooling shrinkage, but a single
continuous distribution being structurally unable to front-load enough mass
at the exact one-tick floor (70.4% of outcomes) while keeping enough tail
mass for the ~30% that run long.

**Method note (read before the tables):** two different, both fully
reproducible, movement-truth reconstructions are used and are NOT
interchangeable populations. Experiment 1 uses the same single-pass,
self-trained-baseline reconstruction that already produced the "213
completed disrupted episodes" figure this task started from
(`movement_backfill.route_ticks_from_vehicle_bodies` +
`transitions_from_ticks(scope="route", debounce_ticks=1)` over the full
2026-06-21..08-12 archive) — reproduced exactly here (n=213, 14 distinct
duration values, tick histogram below matches the given table to the
episode). Experiments 2-4 instead reproduce the properly-CAUSAL truth
(`load_r2.compute_advance_baseline` fit on the clean 06-21..07-04 window
only, applied forward via `load_r2.build_movement_truth` to classify
07-05..08-12, then `episodes.extract_episodes`) — this exactly reproduces
the already-reported n=130 episodes, and, graded against a fresh pull of
the live `state/params.json` `dwell_movement` block, reproduces n_scored=76,
skill=-0.10348, mean_pit=0.49387, and the exact PIT histogram
`[0,0,38,0,14,0,0,2,8,14]` bit-for-bit — strong evidence the pipeline here
matches the one that produced the numbers this task started from. Because
`state/params.json` is a live, mutating artifact this session (its
training_corpus window shifted from the 07-29..08-11 cited earlier in this
journal to 07-24..08-12 by the time it was re-fetched here), Experiments
2-4's model comparisons do NOT depend on it further: they fit their own
dwell curves locally and reproducibly on the same 06-21..07-04 window
(`pooled_dwell.partially_pooled_dwell` / `dwell.dwell_samples_by_cell`),
graded against the same causal 130-episode population (97 scored by the
shipped continuous form after dropping 33 no-curve). The one live-params
pull is used only once, as the validation checkpoint above and as
Experiment 3's primary exhibit (it is the exact, already-published
population, so its PIT lobes are the ones actually worth explaining).

A structural finding surfaced along the way, worth flagging explicitly
because it changes how (b)/(c) below should be read: `dwell.dwell_cdf`
special-cases `x <= curve_sec[0]` as probability 0. Any curve whose lowest
quantile ties (KM/empirical curves fit to a floor-heavy population, where
the smallest observed duration IS the point mass) reads as **exactly 0**
when queried at that same value — and `predicted_recovery_curve` always
queries at exact integer minutes, so a one-tick (5min) episode graded
against a curve whose floor is one tick always gets PIT=0, regardless of
how much of the curve's mass actually sits there. Verified directly: model
(c)'s PIT histogram bin0 count (66) equals the exact count of one-tick
episodes in that population. This deflates (b) and (c) below beyond their
genuine statistical quality — the mixture fix at the end avoids it by
grading a closed-form CDF that bypasses `curve_sec` quantization entirely.

### Experiment 1 — learning curve (does more data help?)

Duration histogram, full 53-day archive reconstruction, n=213 (exact
match to the already-measured table; only 14 distinct duration values):

| ticks | minutes | count | share |
| --- | --- | --- | --- |
| 1 | 5 | 150 | 70.4% |
| 2 | 10 | 27 | 12.7% |
| 3 | 15 | 11 | 5.2% |
| 4 | 20 | 4 | 1.9% |
| 5-119 (10 distinct values) | 25-595 | 21 | 9.8% |

Route-volume buckets (full-archive, in-sample fit — every route's own
completed-episode count over all 53 days vs. CRPS skill on its own
episodes):

| bucket (own episodes) | n routes | n episodes | mean CRPS | baseline CRPS | skill |
| --- | --- | --- | --- | --- | --- |
| 1-2 | 4 | 5 | 0.89 | 0.00 | undefined (zero-variance baseline, n too small) |
| 3-5 | 5 | 19 | 17.31 | 15.87 | -0.090 |
| 6-10 | 4 | 37 | 8.27 | 7.67 | -0.078 |
| 11+ | 7 | 152 | 7.34 | 7.05 | -0.040 |

Training-window sweep — fit on the first N days (2026-06-21 onward), grade
on the SAME fixed held-out window every time (2026-07-30..08-12, n=63
episodes, held fixed so N is the only thing that changes):

| N days | train window ends | n train episodes | routes w/ cell (own/pooled) | n scored | skill |
| --- | --- | --- | --- | --- | --- |
| 7 | 06-28 | 29 | 10 (3/7) | 51 | -0.243 |
| 14 | 07-05 | 77 | 13 (8/5) | 55 | -0.150 |
| 21 | 07-12 | 90 | 16 (9/7) | 62 | -0.142 |
| 28 | 07-19 | 118 | 20 (10/10) | 63 | -0.170 |
| 35 | 07-26 | 138 | 20 (14/6) | 63 | -0.164 |

**Interpretation rule (as specified): skill rising monotonically with N
would mean a volume problem; flat or negative regardless of N means more
data will not fix it.** The sweep is flat-to-negative and non-monotonic
(worse at N=28 than N=21) — more data does not fix it. The route-volume
bucket table shows a mild, monotonic improvement with a route's own
episode count (-0.090 → -0.078 → -0.040) but never crosses zero even at
11+ own episodes, and it's an in-sample view (weaker evidence than the
causal sweep). Net: volume gives at most a second-order assist; it is not
the primary lever.

### Experiment 2 — model form (is a continuous curve the wrong shape?)

Same 06-21..07-04 training data, same causal 130-episode held-out
population, three forms graded through the unmodified
`scorecard.episode_recovery` / `recovery_dist.recovery_dist_report`:

| model | n scored | n no-curve | mean CRPS | baseline CRPS | skill |
| --- | --- | --- | --- | --- | --- |
| a. shipped continuous log-logistic AFT (`partially_pooled_dwell`) | 97 | 33 | 6.92 | 6.36 | -0.088 |
| b. per-route empirical/KM (`compute_dwell_quantiles`, min_samples=5) | 25 | 105 | 5.45 | 3.82 | -0.426 |
| c. pooled climatology (one KM curve over all routes' pooled samples) | 97 | 33 | 7.15 | 6.36 | -0.126 |

Restricted to the n=25 intersection every model can score (apples-to-apples,
same episodes, same baseline):

| model | mean CRPS | baseline CRPS | skill |
| --- | --- | --- | --- |
| a. shipped continuous | 4.42 | 3.82 | -0.157 |
| b. per-route KM | 5.45 | 3.82 | -0.426 |
| c. pooled climatology | 4.68 | 3.82 | -0.224 |

Neither (b) nor (c) beats (a) — on the contrary, (a) is the least-bad of
the three, both at full coverage and on the matched subset. Per the
assignment's own framing this is evidence the parametric form per se isn't
the destroyer of information — BUT (b)/(c)'s scores are inflated-bad by the
`dwell_cdf` floor artifact above (both have curve_sec[0] pinned to exactly
one tick, so every one-tick episode scores PIT=0 regardless of true fit
quality), and (b) additionally starves for data (105/130 no-curve at
min_samples=5, no pooling). Taking (a) at face value as "the best of three
plain forms" is correct; taking it as "therefore the functional family is
fine" is not — see the mixture result at the end.

### Experiment 3 — where does the bimodal PIT come from?

Using the exact already-published population (n=76 scored, causal 130
episodes graded against the live `dwell_movement`, reproduced bit-for-bit
above). Per-episode PIT, lobed at [0.2,0.3) and >=0.8 as specified (plus
the [0.4,0.5) cluster the histogram also shows):

| PIT lobe | n | share | composition |
| --- | --- | --- | --- |
| [0.2,0.3) | 38 | 50.0% | **100% one-tick**, 100% own-fit routes with n=5 own events (6, G, R) |
| [0.4,0.5) | 14 | 18.4% | **100% one-tick**, 100% pooled-parent routes with n=1-2 own events (J, M, W) |
| [0.8,1.0) | 22 | 28.9% | **100% multi-tick** (median 12.5min, max 95min), mix of own (15) and pooled (7) routes, all 6 routes represented |
| other (<8min gap) | 2 | 2.6% | route 6, multi-tick, own-fit |

This is a clean, complete explanation, not a partial one. The two opposite
failure modes: **(1) one-tick outcomes always read as "model too
pessimistic"** (PIT low) — even a full n=5 own-fit sample forces a smooth
curve to spread some mass around/below the observed cluster, so it never
assigns as much probability to "already done in exactly one tick" as the
~70% base rate demands; pooled routes land closer to calibrated (~0.45)
purely because shrinkage happens to center them near the population mix.
**(2) any multi-tick outcome always reads as "model too optimistic"** (PIT
high, 0/22 in this lobe are one-tick) — a curve compressed to front-load
enough mass for the one-tick majority has too little tail left for the
~30% minority that takes longer, so by the time a longer episode actually
resolves the curve had already assigned it near-certainty. Cross-checked
against an independently-fit local model (06-21..07-04, 13 routes instead
of 6): the same one-tick-low / multi-tick-high split holds (66 one-tick
episodes land in the low/mid lobes, 0 in the high lobe; 25 of 28 multi-tick
episodes land in the high lobe) — the mechanism is structural to the
population, not an artifact of one specific fit.

### Experiment 4 — pooling

Own-fit vs pooled-parent routes, split on the exact published population
(own={6,G,R}, n=5 events each; pooled={J,M,W}, n=1-2 events each):

| split | n routes | n scored | mean CRPS | baseline CRPS | skill |
| --- | --- | --- | --- | --- | --- |
| own-fit | 3 | 55 | 4.98 | 4.53 | -0.099 |
| pooled-parent | 3 | 21 | 5.68 | 5.01 | -0.133 |

Cross-checked on the independent local fit (8 own-fit routes incl. n=32/15
event routes 7/M, 5 pooled-parent routes at n=1 each):

| split | n routes | n scored | skill |
| --- | --- | --- | --- |
| own-fit | 8 | 60 | -0.094 |
| pooled-parent | 5 | 37 | -0.079 |

Own-fit and pooled-parent skill sit in the same negative band in both
cuts (pooled is worse in one, marginally better in the other) — pooling
shrinkage is not carrying the negative skill; well-observed routes with a
full own-fit sample fail almost as badly as thin, shrunk ones. This rules
out "give sparse routes more of their own data" as the fix, consistent
with Experiment 1's volume-sweep finding.

### Concrete cheapest model-form fix, measured offline (not shipped)

A one-tick point-mass p0 (global, pooled across all training-window
routes: 53/77 = 0.688 of all completed disrupted exits are exactly one
tick) mixed with the existing partially-pooled log-logistic **refit
conditional on T > 1 tick** (so the spike and the tail don't double-count
the same population), graded via `recovery_dist.recovery_dist_report`
directly on a closed-form CDF (bypassing `curve_sec` quantization, which
cannot represent a mass this large evaluated exactly at its own grid
point — see the artifact note above). Same training window, same 97-episode
held-out population as Experiment 2's model (a):

| model | mean CRPS | baseline CRPS | skill |
| --- | --- | --- | --- |
| a. shipped continuous (closed-form check, same math as the curve version) | 7.04 | 6.36 | -0.108 |
| d. one-tick spike + conditional continuous tail mixture | 6.50 | 6.36 | **-0.023** |

A 74% reduction in the skill deficit (-0.108 → -0.023), same data, zero
extra training days, one extra parameter (the pooled one-tick fraction).
Still short of beating climatology — mean_pit=0.746 shows the mixture is
now calibrated too optimistically on average (a single global p0 over- or
under-shoots per-route), so a per-route-shrunk p0 (same partial-pooling
machinery already in `pooled_dwell.py`, applied to the binary
one-tick-or-not outcome instead of a continuous scale) is the next-cheapest
iteration, not attempted here. Not implemented in production; this is the
offline number for the lead to decide on.

Commands run (`murk exec -- uv run python -m training._scratch_*`, five
throwaway scripts under `training/`, each fetching `archive/vehicles/`
bodies for 2026-06-21..08-12 via `training.movement_backfill.
_fetch_vehicle_bodies` — cached to `/tmp` between runs, both cache files
and all five scripts deleted after use, nothing left in the repo or on
disk): `movement_backfill.route_ticks_from_vehicle_bodies` +
`transitions_from_ticks(debounce_ticks=1)` for the full-archive
reconstruction; `load_r2.compute_advance_baseline` +
`load_r2.build_movement_truth` + `episodes.extract_episodes` for the causal
130-episode population; `pooled_dwell.partially_pooled_dwell` /
`pooled_dwell.cell_from_fit` for model (a); `dwell.compute_dwell_quantiles`
for model (b); `dwell._make_cell` over pooled samples for model (c);
`training.r2_client.get_object_bytes` for the one `state/params.json`
validation pull; `scorecard.episode_recovery` /
`scorecard.movement_dwell_lookup_from_params` /
`recovery_dist.recovery_dist_report` /
`recovery_dist.predicted_recovery_curve` for every grading call, unmodified.

## 2026-08-12 — a status glyph cannot encode three states by spacing alone at 20px, and `recovery_minutes` is 0 on 22 of 30 routes

origin: agent

Branding work, but two findings bind on the feed itself and are worth keeping
whoever builds the next UI on it.

**The load-bearing one: `inference.recovery_minutes` is `0` for all 22
currently-normal routes** (30 total: 22 `normal`, 4 `not_scheduled`, 4
`unknown`). Any UI channel bound to it — a bar, a gauge, a spacing, a width —
is therefore motionless outside a disruption, i.e. dead in roughly 95% of
ticks. `Snapshot.observations` ships as an empty list, and `Observation`
(`src/momentarily/schema.py:69-82`) already reserves `kind: "headway"` /
`unit: "seconds"` for the signal that would fix this. Nothing in the repo
computes headway today. The Worker's existing GTFS-RT vehicle-position decode
(`worker/src/vehicles.ts`) already knows where trains are each tick, so
successive passings of a fixed reference stop would give it.

**Spacing fails as a categorical encoding at status-icon size.** Tested three
shape encodings for normal/disrupted/suspended at 20px with hue removed,
scanned as a route grid (`docs/brand/mark-study.html` §6):

| encoding | 20px monochrome result |
| --- | --- |
| gap 0 / 4 / 8 units on a 24 grid (spacing only) | fails — disrupted vs suspended indistinguishable |
| gap 0 / 5 / 11 (spacing widened to the frame) | marginal, needs a side-by-side reference |
| split topology + symmetric height drop | passes, read correctly cold |

A 4-unit gap delta on a 24 grid is 3.3px at 20px render, and judging "how far
apart" has no anchor at that size, whereas "one shape or two" and "tall or
short" do. Consequence for the existing UI: `viz/app/globals.css` signals route
state with hue-only badges (`.cond.normal|.disrupted|.suspended`) while
line 441 of the same file already states the principle — "a second encoding so
hue isn't the only signal" — for legend markers. The badges don't honour it.
A shape channel is the fix, not a fourth colour.

Also negative, same file: a 1.5-unit accent inside the gap disappears by 32px
(1.25px at 20px); rotating the pair 90° is dimensionally truer to a train seen
from above but collapses into an equals sign by 32px.

**Hue is not a free channel here.** Route colours arrive at runtime from GTFS
static (`derive.py:179`, fallback `viz/lib/feed.ts:36`) and occupy 11 values:
`#EE352E` `#00933C` `#B933AD` `#FF6319` `#2850AD` `#FCCC0A` `#996633`
`#6CBE45` `#A7A9AC` `#808183` `#1F4F9F`. Red, orange, yellow, green, lime,
brown, blue, navy, magenta-purple and two greys are all spoken for by lines the
dashboard renders beside any product chrome.

**`segment_flow` is a sample, not a strip.** 56 segments, max 4 per
(route, direction), non-contiguous when chained on `from_stop`→`to`, and all
`normal` at time of writing. An ordered per-route station strip needs a GTFS
static stop-ordering join; it is not derivable from one snapshot.

Corrected assumption worth recording: a scheduled-headway baseline does **not**
need a new ingest. `training/gtfs_static.py` already streams `trips.txt` and
`stop_times.txt` off the static feed (lines 101 and 116) for segment topology,
so arrival times are already being read and a baseline is a parse extension.
Emitting it per (route, direction, service period) alongside the other weekly
fit artifacts is enough to express observed headway as a deviation.

## 2026-08-12 — mixture shipped: CRPS skill -0.139 -> -0.061, and two estimator bugs the offline prototype had hidden

origin: agent

The one-tick point-mass mixture is now the published movement dwell model
(`atom_p`/`atom_sec` on the dwell cell, tail_ll refit left-truncated at one
tick). Graded causally — trained 06-21..07-25, scored on 74 held-out
episodes from 07-26..08-11, same transitions and same grader for every
variant, only the cell construction differing:

| model | CRPS | baseline | skill | mean PIT |
| --- | --- | --- | --- | --- |
| continuous (what shipped before) | 7.451 | 6.541 | -0.1391 | 0.579 |
| single global atom | 6.922 | 6.541 | **-0.0582** | 0.545 |
| per-route shrunk atom (shipped) | 6.938 | 6.541 | **-0.0606** | 0.544 |

56% of the skill deficit gone. Still short of climatology, so the arm stays
ungraduated. Global and shrunk are within noise of each other because the
concentration estimator lands on its ceiling — the routes are not
distinguishable at these counts, and it says so rather than pretending. The
shrunk form ships because it degenerates to global exactly when the data is
thin and separates on its own once it is not.

### Two estimator bugs the offline prototype never exposed

Both were invisible in the earlier one-off because it used a single global
rate. Going per-route surfaced them, and the first attempt at per-route
scored **worse than doing nothing** (-0.1459 vs -0.1391 continuous).

**1. Median of per-route ratios is biased upward at small denominators.**
Following the scale estimator's one-vote-per-route convention gave a
population atom rate of **0.789** where the pooled ratio of totals was
**0.733** (99 one-tick exits / 135 completed). A route that saw one episode
and it was one tick votes 1.0 with the same weight as a route that saw
thirty. Every cell then shrinks toward an inflated centre and the whole
forecast reads optimistic. Fixed by using the ratio of totals for the atom
centre — a deliberate departure from the scale's convention, for a reason
that only applies to ratios of small counts.

**2. The Beta moment inversion inverts its own purpose at small n.**
`hierarchical.robust_concentration` reads the observed spread of per-cell
rates as if it were all real between-cell variation. That holds at the 20+
trials its leaves carry. At the ~5 episodes a dwell cell carries, binomial
sampling noise dominates: a route at 1-of-1 and a route at 3-of-5 look
wildly dispersed while being perfectly consistent with one shared rate. The
estimator concluded "these routes differ", returned the kappa floor, refused
to pool, and let per-route rates run to **0.957**. Var(observed) =
Var(between) + E[p(1-p)/n]; subtracting the sampling term before inverting
fixes it, and when nothing survives the subtraction the answer is to pool
hard. That single change moved per-route from -0.1459 to -0.0606.

Left `robust_concentration` alone — it is correct in its own high-n regime —
and put the corrected variant next to the estimator that needs it, with a
note on each pointing at the other.

### PIT is not a valid diagnostic against a point mass

Worth internalising, because it looks exactly like a catastrophic
regression. With an atom at 300s, every one-tick episode returns the
identical F(300) = atom_p, so the histogram collapses:

| model | PIT histogram | mean PIT |
| --- | --- | --- |
| continuous | [0,1,7,15,25,0,1,1,8,16] | 0.579 |
| mixture, raw PIT | [0,0,0,0,0,0,0,48,13,13] | 0.790 |
| mixture, randomized PIT | [8,8,4,8,6,6,6,2,13,13] | 0.544 |

The middle row is not miscalibration. PIT is only uniform for a *continuous*
predictive CDF; against a jump the correct diagnostic spreads each
observation across its own jump. Uncorrected, the mixture would have been
published with a "badly optimistic" verdict on the Models page while being
better calibrated than the model it replaced — the bottom row is closer to
uniform than the continuous row on every measure. The residual is the top
two bins: the multi-tick tail is still under-dispersed, which is the next
thing to work on.

Randomization is keyed on an FNV-1a digest of the regime, not an RNG, so the
grade stays reproducible and does not depend on iteration order. FNV-1a
rather than the blake3 already in `provenance.py` because the viz grading
route recomputes this report and blake3 is not in `node:crypto` — measured
across 74/1000/20000 realistic keys, FNV-1a, blake2b and blake3 are all
indistinguishable from uniform (KS 0.0845/0.1004/0.0997 at n=74 against a
0.1581 critical value), so there was no quality to buy with a dependency.

### Incidental

`dwell_cdf` returned 0 at `x <= curve_sec[0]`, and `curve_sec[0]` is exactly
one tick for any cell whose shortest dwell is one tick — so the CDF read 0
evaluated at its own grid point and every one-tick episode graded PIT=0. The
lower guard is now strict and a repeated knot resolves to the top of its
flat run. Numerically identical on strictly increasing curves, which is
pinned by a test using the old formula as an oracle.

A dwell cell gaining a field is silently dropped by the Worker: the zod
schema is a plain `z.object()` with strip semantics, so `atom_p` would have
been deleted at parse time and the fix would have trained, published and
graded with the old math while looking shipped. `n_censored` was already
being dropped that way. Both are now in the schema, and a golden fixture
(`tests/fixtures/parity_dwell.json`, 42 cases) is replayed by Python, the
Worker and viz so the three cannot drift again.

## 2026-08-12 — gated the 60/120min horizons; the publish guard would have emptied the feed

origin: agent

Stopped publishing `p_normal_in_60min` and `p_normal_in_120min`. They are now
`null` on every fitted-curve row. `p_normal_in_30min` is untouched and still
always published — it is the one horizon with repeatedly measured skill
(BSS +0.29 to +0.40). The two longer ones lose to naive persistence in every
population cut tested (BSS -0.00 to -1.30; AUC 0.395 and 0.352, i.e. inverted),
the loss is not a left-censoring artifact, and the projection was verified
monotone over 45,600 comparisons — so the defect is the shape of the fitted
elapsed-conditional dwell curve and no amount of runtime fixes it. Publishing a
number we have measured to be anti-informative is worse than publishing nothing.

Chose nullability over a `recovery_indeterminate`-style sibling flag. A flag
leaves the inverted number in the payload for any consumer that ignores the
flag; null cannot be misread. This follows the existing "never a fabricated
number" precedent (`SegmentStatus.recovery: SegmentRecovery | None`,
`movementRecovery()` returning null) rather than the clamp-plus-flag one.

**The schedule arm is exempt, and that is not a detail.** When
`recovery_source == "schedule"` the value is `resume <= now + h*60` — a
comparison against an announced resume time, not a fitted forecast. It carries
none of the defect above, so it keeps publishing all three horizons. A blanket
suppression would have thrown away the only horizon information in the feed that
is *deterministically* correct. Gated at the single point where the Inference is
assembled rather than in each of the five arms that set these values, so the
next arm added cannot silently miss the gate.

### The near-miss worth remembering

`scrubCorruptInferences` null-checks every inference field with
`Number.isFinite`, and `Number.isFinite(null)` is `false`. Shipping the gate
without touching that function would have scrubbed EVERY route's inference on
EVERY tick — the guard that exists to stop one bad route from staling the feed
would have emptied the feed of exactly the data it protects. Caught by its own
existing regression test. The distinction the function now draws: absent is a
valid published state, not-a-number is not.

Two other consumers had to learn the difference. `training.eval` parsed with
`float(raw["p_normal_in_60min"])`, which raises on null; it now uses an explicit
`_opt_float` and the calibration loop skips withheld rows the same way it already
skips `schedule` rows. Deliberately NOT `float(value or 0.0)` — a withheld
forecast coerced to 0 scores as a confident "will not recover" and drags every
bin it lands in. The dashboard had the same trap in reverse: `null * 100`
renders `0%`, so a withheld horizon would have displayed as a confident zero;
it now renders "not forecast".

`SegmentRecovery` keeps all three horizons non-null. The measurement was taken
on the route-level fields, and the segment arm is a different fitted model
(`segment_dwell.json`); gating it would be asserting evidence we do not have.
Filed separately rather than assumed.

Eight worker tests asserted properties of the two withheld horizons. None were
deleted: the monotonicity and probability-bounds invariants moved down onto
`pLeaveBy`/`staysNormalFor` in `dwell.ts`, where they still hold and where the
property actually lives. It stopped being an invariant of the published field,
not an invariant of the model.

## 2026-08-12 — advance baseline graded held-out at last: partial pooling loses to the raw leaf rate, and badly on thin leaves (Brier 0.125 vs 0.069)

origin: agent

First out-of-sample grade of `hierarchical.partially_pool`'s advance-rate
baseline. Trained on 2026-06-21..07-25, scored against the Bernoulli
advance/stall trials observed in 07-26..08-11, 2,375 leaves present in both
windows, 2.2M held-out trials. Every predictor clipped identically at P0_FLOOR
so log-loss is comparable.

**Target choice, because it changes what the number means.** `p0` is not a
forecast of advance; it is the healthy-baseline null `classify_direction` tests a
live tick against, so scoring it on all held-out ticks would let any predictor
win by baking in the disruption rate. Graded set is therefore restricted to
held-out ticks where the route was in the `normal` movement state, with route
truth reconstructed causally from the training window's direction baseline. In
the event this barely matters — the held-out window is **108,657 normal
route-ticks against 196 disrupted**, so the two targets agree to within 0.0008
Brier. Recording it because the objection is correct in principle even though it
did not bite here.

| predictor | Brier | logloss | vs pooled |
| --- | --- | --- | --- |
| pooled (what ships) | 0.06519 | 0.23645 | |
| raw leaf rate | **0.06414** | 0.23453 | -0.00105 |
| route level | 0.18876 | 0.54609 | +0.12357 |
| system level | 0.24298 | 0.67903 | +0.17779 |
| pooled, fitted on normal ticks only | 0.06505 | 0.23640 | -0.00014 |

Restricted to the 497 THIN training leaves (n_A < MIN_LEAF_N=20) — the leaves
partial pooling exists to serve:

| predictor | Brier | logloss | vs pooled |
| --- | --- | --- | --- |
| pooled | 0.12451 | 0.43691 | |
| raw leaf rate | **0.06908** | 0.36334 | -0.05543 |
| system level | 0.19412 | 0.58022 | +0.06960 |

So: pooling is a small net loss overall and a **1.8x Brier loss exactly where it
is supposed to help**. It does beat route- and system-level pooling by a wide
margin (0.065 vs 0.189 and 0.243), so leaf specificity is doing real work — the
hierarchy *above* the leaf is nearly worthless here. It is the direction of the
shrinkage that is wrong, not the idea of a leaf-level estimate.

### Why: exchangeability is violated, and not in the direction I guessed

| | held-out advance rate |
| --- | --- |
| thin leaves (n_A < 20) | 0.8532 |
| voting leaves (n_A >= 20) | 0.5830 |
| centre thin leaves are shrunk toward | **0.9888** |

Thin leaves run **higher** than well-observed ones, not lower. Plausible reading:
a rarely-traversed segment has few trains on it, so nothing queues and trains
simply advance; the busy segments that accumulate large n are exactly where
trains stall behind each other. Either way, how much data a leaf has is
informative about its rate, which is the assumption partial pooling rests on. I
had predicted the opposite sign (guessing thin leaves were terminals, which
structurally dwell) — that was wrong, and the deciles say so.

The shrinkage target overshoots even so: thin leaves truly run 0.853 and are
pulled to 0.9888. Assigning 0.99 where the truth is 0.85 is what the Brier gap
is made of, and it points straight at the open terminal over-flagging bug — a p0
of 0.99 puts the `0.5*p0` trip-wire at 0.494 and sets the binomial null at 0.99,
so a modest stall streak reads as wildly significant. That link is now measured
rather than speculated.

### Negative result: the missing disruption resistance is not the problem

Unlike its direction-level sibling `compute_advance_baseline`, which takes a
per-tick MEDIAN specifically so a line that mostly runs well keeps a high
baseline through occasional frozen stretches, `build_segment_baseline` simply
SUMS advanced/stalled over the window — the segment leaf baseline has no
disruption resistance at all. Refitting it on normal-only ticks was the obvious
candidate fix and it does essentially nothing: **-0.00014 Brier overall and
+0.00067 on thin leaves** (i.e. very slightly worse there). Worth knowing before
anyone spends time on it.

### Correcting an earlier claim in this journal

The entry two above reasoned that the median-of-fractions bias found in the dwell
atom estimator would be "much smaller" here because a leaf must clear n>=20 to
vote, so its fraction is less coarse. Measured, that reasoning is wrong: the gap
between the median of leaf rates and the ratio of totals is **+0.3985** at system
level and the median is higher in **52 of 52** route_direction groups (mean gap
+0.3797). The mechanism is not coarse denominators at all, it is n-weighting —
most leaves advance nearly always, while the trials are dominated by a few busy,
stall-heavy leaves. Which of the two is the "right" centre is a genuine modelling
question (a prior over leaves is not a prior over trials), so this is not a bug
on its own; what the held-out numbers do say is that 0.9888 is too high for the
leaves that borrow it.

Vote-gate clearance, previously asserted but never measured: **1,884 of 2,508
leaves (75.1%)** clear MIN_LEAF_N=20, per-leaf n has p10=2, p50=349, p90=1171,
max=29,487, and 2,505 of 2,508 cells take their prior from the route_direction
level (only 3 fall back to system). So the gate is not the fragile part.

206 held-out leaves have no training baseline at all and the classifier abstains
on them.

Commands run (`murk exec -- uv run python <script>` with PYTHONPATH at the repo
root; two scripts, both under /tmp only, deleted after the run, nothing left in
the repo or on disk): `movement_backfill._fetch_vehicle_bodies` for both windows
(cached to /tmp between runs, cache deleted), `load_r2.build_segment_series`
aggregated per leaf per tick, `hierarchical.partially_pool` for every pooled
variant, `load_r2.compute_advance_baseline` +
`load_r2.build_movement_series_by_direction` + `load_r2.build_movement_truth` for
the causal normal-tick mask. No production code modified.

## 2026-08-12 — 81.8% of our "segments" are multi-station jumps: 5-minute polling cannot see station-to-station movement

origin: agent

The archive records, per 5-minute tick, that a trip was at stop A last look and
stop B this look, and we treat (A,B) as a segment. Measured how far apart A and B
actually are in the scheduled stop order (`gtfs_static.chains` gives travel-order
stop sequences per route+direction; hop = index difference), over
2026-08-05..08-11, 915,906 transitions, 98.6% of them placeable in the scheduled
order:

| stops covered in one 5-min look | count | share of moves |
| --- | --- | --- |
| 1 (an actual segment) | 90,159 | **18.2%** |
| 2 | 158,975 | 32.2% |
| 3 | 192,192 | 38.9% |
| 4 | 42,354 | 8.6% |
| 5+ | 8,612 | 1.2% |

**81.8% of observed moves skipped at least one station we never saw.** Mean 2.71
stations per look. Worst on the busiest lines: `1` 94.8% multi-stop, `6` 89.4%,
`L` 88.2%, `3` 78.4%.

So the segment-level view is not measuring segments. It is measuring "a train was
somewhere near A, then somewhere near A+2.71", and `canonical_adjacency` then
labels that jump with its most common landing stop — which is exactly why the
labels look plausible while being wrong. Any per-segment statistic built on this
is a blend of 2-4 real segments, and the stations in between contribute nothing at
all.

This is the root cause under the advance-baseline problem graded in the entry
above, and it is not fixable by better shrinkage. A yes/no "did it leave" carries
almost no information per observation, which is why thin leaves must borrow, which
is why the borrowing being miscalibrated matters. A traversal DURATION carries far
more per observation.

### Two anomalies in the same numbers, both unexplained

**42.5% of placed transitions are stalls** (stop_id unchanged across 5 minutes).
That is hard to reconcile with a moving train covering 2.71 stations per look: a
train in revenue service should change stop_id nearly every tick. So the
advance-rate p0 ~ 0.59 that the entire movement classifier rests on may be
dominated by something other than "trains are running well" — layups, held
trains, terminal relays, or stale feed entries. Not investigated here. Worth
knowing before anyone tunes the classifier, and it is a live lead for the terminal
over-flagging bug.

**2.9% of moves read as backwards** in the scheduled order. Candidates: terminal
relays where a train reverses, the stop_id N/S suffix disagreeing with the trip's
actual direction, or trip_id reuse. Also not investigated.

### What this justifies

Polling faster alone is not the fix. At 1-minute polling a station hop (~1.85 min
at the observed speed) still lands in 1-2 ticks, so a tick count would only pin
arrival to +/-30s on a ~110s journey — rebuilding a coarse measure with 5x the
storage. The resolution is already in the feed and being discarded:
`VehiclePosition.timestamp` (the train's own measurement time) is not decoded, and
`current_stop_sequence` IS decoded and tested but unused, and is present exactly
when a train is STOPPED_AT. Poll rate then governs how often we CATCH a
transition, not how precisely we time it.

Note for whoever wires this: raising the cron to 1 minute without gating silently
redefines every existing model. `advanced_n`/`stalled_n` mean "changed in 5
minutes", the advance baseline is fitted on that, and the dwell model's one-tick
point mass IS 5 minutes. The same handler at 1 minute keeps producing plausible
numbers against params that assume the old meaning. The 5-minute pipeline has to
stay gated to 5-minute boundaries with its own carry object, and the trace needs
its own.

Commands run (`murk exec -- uv run python /tmp/hop_test.py`, PYTHONPATH at the
repo root; script and its caches under /tmp only, deleted after): 
`movement_backfill._fetch_vehicle_bodies` for the window,
`gtfs_static.load_successors` + `gtfs_static.chains` for scheduled travel order,
`load_r2.build_segment_series` for the observed transitions. No production code
modified.

## 2026-08-12 — two-thirds of the stall mass is terminal layover; and the alert feed cannot grade a movement signal (4 acute onsets in a week)

origin: agent

### Where the 42.5% stall rate comes from — answered

Stalls per (route, direction, from_stop) over 2026-08-05..08-11, cross-referenced
against each stop's position in its scheduled travel-order chain:

| position in the line | stalls | moves | stall rate | share of all stalls |
| --- | --- | --- | --- | --- |
| first stop (origin terminal) | 244,009 | 28,261 | **89.6%** | **63.5%** |
| destination terminal | 12,419 | 2,434 | 83.6% | 3.2% |
| one in from a terminal | 6,324 | 19,611 | 24.4% | 1.6% |
| mid-line | 96,273 | 474,532 | **16.9%** | 25.1% |
| not in a scheduled chain | 24,953 | | | 6.5% |

**Two-thirds of the stall mass is trains parked at terminals by design.** It is
also extremely concentrated: the top 25 (route, direction, stop) leaves hold
65.6% of all stalls, top 50 hold 87.5%. The worst are stationary by construction —
`H north H19N` (Rockaway shuttle) 15,974 stalls to 6 moves, `7 south 726S` 2,421
to 1, `D south D01S` 7,141 to 193.

So `p0 ~ 0.59`, the number the entire movement classifier rests on, is a blend of
two physically different populations, dominated by the parked one. The in-service
figure is 1 - 0.169 = **0.831**, which independently matches the 0.8532 held-out
rate measured for thin leaves two entries above — thin leaves are just ordinary
mid-line stops.

This also nails the terminal over-flagging mechanism rather than guessing at
it: a terminal genuinely stalls ~90% of the time, borrows a pooled p0 near 0.99
from its mid-line neighbours, and its normal behaviour then reads as
overwhelmingly significant evidence of disruption.

### Progress ratio: built, and NOT yet gradeable

Replaced the advance-rate binary with a scheduled-progress measure. For each
train per 5-minute tick: `scheduled_seconds(path actually covered) / 300`. 1.0 is
on time, 0.5 is half speed, 0 is not moving. Terminals excluded per the above.
Scheduled run times come from a new `gtfs_static.hop_seconds` (median over trips
of departure-to-arrival, so scheduled dwell is not counted as travel): 1,932 hops,
median 90s, mean 122.6s. That cross-checks against the observed 300/2.71 = 111s
per hop, so moving trains run slightly ahead of the timetable — consistent with
schedule padding.

The measure itself behaves sanely: median 0.818, p05 0.092, p95 1.056.

Graded against MTA severe-alert truth (severity floor 2), 41,413 route-ticks,
AUC where a low score should mean disrupted:

| score | AUC |
| --- | --- |
| advance rate, as production computes it (all stops) | 0.2175 |
| advance rate, terminals excluded | 0.3899 |
| progress ratio, terminals excluded | 0.4229 |

All three are INVERTED. Excluding terminals is worth +0.172 AUC on its own, which
is the single largest effect measured today and comes straight from the stall
diagnostic above. The progress ratio then adds +0.033 on top. But everything is
still the wrong side of 0.5, so none of it is evidence of anything yet.

**Ruled out: time of day.** Alert prevalence is 11.2%-13.0% in EVERY one of the 24
UTC hours, and AUC computed within-hour is essentially unchanged (advance 0.3868,
progress 0.4207). Not a confound.

**The label is the problem, and the flatness is the proof.** No real disruption
pattern is uniform across all 24 hours. A 12% rate at 4am and at 6pm means the
truth is dominated by chronic standing advisories, exactly as the chronic-alert
finding above records, so it labels thousands of normally-running ticks
disrupted and cannot separate anything.

Confirmed by trying the acute version — label a CALM route-tick positive if a new
alert onset (normal -> not-normal transition) follows within 30 minutes:

  * **4 acute onsets in the entire week, across 4 routes**, giving 24 positive
    route-ticks out of 36,297.
  * AUC 0.6025 progress ratio vs 0.5744 advance rate. Both now the right side of
    0.5, and the progress ratio's mean is 0.0391 LOWER before an onset while the
    advance rate moves +0.0019 (i.e. not at all).
  * With 24 positives the CI on AUC is about +/-0.12. This is a direction, not a
    result. Do not cite it as one.

So the honest state: the progress ratio is a better-posed measure on every
structural argument, it edges the advance rate on every cut tried, and there is
currently **no label in this repo capable of grading it**. That is the blocker, not
the model.

### Known flaw in the measure as built

Skip-stop inverts it. `path_seconds` sums the scheduled chain between the observed
endpoints, so a train that BYPASSES stops during a disruption is credited with
covering all of their scheduled time and reads as running fast. Pre-onset ticks
average 2.266 hops per move; the normal-tick comparison came back NaN through a
bug in the scratch script and was not re-run. Needs handling before this ships —
probably by comparing against the trip's OWN scheduled pattern from stop_times
rather than the route's modal chain, which is available and would also fix
express/local sharing track.

### What would actually grade it

Nothing derived from the alerts feed at this severity floor. Candidates, in order
of independence: trip-updates `assigned_n` collapses as an acute service signal;
the 1-minute trace once it accumulates, giving real per-segment traversal times to
compare against schedule directly; or hand-graded incident windows (1ul already
exists for the Knicks-parade window).

Commands run (`murk exec -- uv run python /tmp/{stall_why,progress,progress2,progress3}.py`,
PYTHONPATH at the repo root; four scratch scripts and their caches under /tmp only,
all deleted after): `movement_backfill._fetch_vehicle_bodies`,
`load_r2.build_segment_series`, `gtfs_static.load_successors` / `chains` /
`load_hop_seconds`, `review.build_mta_truth`. Production code changed: only the
new `gtfs_static.hop_seconds` / `load_hop_seconds` (committed, tested).


## 2026-08-12 — the advance signal needs an admission rule, not a terminal blacklist: through-stops-only cuts thin-leaf Brier 0.149 -> 0.081

origin: agent

### Terminals were the larger half of the problem, not all of it

Chain endpoints read off the dominant-successor skeleton directly (a stop with no
incoming dominant edge, or none outgoing) rather than off `chains(...).stops[0]`,
which concatenates one walk per entry stop and so names only one origin when a
component has two. 199 endpoint keys against 1,659 through stops. Recounted over
2026-08-05..08-11, keyed on each transition's from_stop:

| position | stalls | moves | stall rate | share of all stalls |
| --- | --- | --- | --- | --- |
| chain endpoint | 294,928 | 36,316 | **89.0%** | **76.8%** |
| mid-line (through) | 64,103 | 488,522 | 11.6% | 16.7% |
| not in the skeleton at all | 24,947 | 7,090 | **77.9%** | 6.5% |

The third row is the finding. Stops the timetable never names — yard leads, rare
patterns, stop_ids the vehicle feed emits and stop_times doesn't — stall at 77.9%,
which is terminal behaviour, not mid-line behaviour. Excluding terminals alone
leaves those 24,947 stalls in the admitted population: 28% of what remains after
the endpoints go. So the rule that matters is positive, not subtractive: count a
trip only where the schedule defines what moving means, i.e. the from_stop has
both a scheduled predecessor and a scheduled successor.

### Held out, that is worth more than the terminal cut alone

Train 2026-06-21..07-25, score 07-26..08-11 on the route-ticks the severity-2
alert truth calls normal. Pooled = `hierarchical.partially_pool`, raw-leaf = each
leaf's own training rate.

| arm | leaves | no train data | thin | pooled Brier | thin-leaf pooled | thin-leaf raw |
| --- | --- | --- | --- | --- | --- | --- |
| all stops (production) | 2291 | 177 | 448 | 0.06656 | 0.14857 | 0.05506 |
| terminals excluded | 2107 | 174 | 436 | 0.06034 | 0.11144 | 0.05769 |
| through stops only | 1617 | **0** | **69** | 0.06208 | **0.08082** | 0.07788 |

Read the last three columns, not the fourth: overall Brier is dominated by the
largest leaves and each arm scores a different population, so cross-arm overall
comparison is not like-for-like (terminals-excluded "wins" it at 0.06034 while
being worse everywhere the estimator is actually load-bearing). What is
like-for-like:

  * **Every held-out leaf has training data** under through-only — 177 leaves
    that previously needed a fabricated prior are gone, because they were
    off-skeleton stops that appear in one window and not the next.
  * Thin leaves fall 448 -> 69 and their Brier 0.14857 -> 0.08082.
  * Pooled and raw converge (0.06208 vs 0.06195, and 0.08082 vs 0.07788 on thin
    leaves). The exchangeability violation measured in the earlier grade —
    pooling *losing* to raw by 1.8x on the leaves it exists to serve — was mostly
    terminals and off-skeleton stops sitting in one hierarchy with through stops.
    Most of what the n-aware-shrinkage work was for dissolves with the population
    fixed instead.

Reproduction check, since the harness is new: the all-stops arm scored on all
ticks gives pooled 0.06598 / raw 0.06472 over 2,378 leaves and 2.21M trials, 499
of them thin, against the 0.06519 / 0.06414 / 2,375 / 2.2M / 497 recorded
earlier. Same estimator, same finding.

### AUC against the alert label: ordering reproduces, magnitude does not

Same window, severity floor 2, 45,292 route-ticks, prevalence 12.4-14.7% in every
one of the 24 UTC hours (the flatness the label was already condemned for):

| arm | AUC | mean score on normal | on disrupted | overall advance rate |
| --- | --- | --- | --- | --- |
| all stops | 0.3388 | 0.5619 | 0.7215 | 0.5807 |
| terminals excluded | 0.4361 | 0.8217 | 0.9112 | 0.8470 |
| through stops only | 0.4641 | 0.8673 | 0.9099 | 0.8832 |

Ordering and sign match the 0.2175 -> 0.3899 pair recorded earlier; the absolute
values do not (different route-tick population, no presence mask). All three are
still inverted, so this table ranks the arms and grades nothing — the label is
still the blocker.

### Negative result to carry forward: p0 now saturates

The flat per-(route, direction, tod) baseline's median p0 moves 0.5333 -> 1.0000
once layovers are out. That is arithmetically right — at 5-minute cadence a
through stop's modal tick has every matched trip advancing — but it breaks the
classifier's design intent. `DISRUPTED_RATIO * p0` was meant to judge each line
against its own normal; with every cell's normal at ~1.0 the trip-wire collapses
to a global "under 50% advancing", which is the single global cutoff the
baseline-relative design existed to replace. The significance gate still fires on
the same evidence, so this is not a correctness bug, but the ratio form is now
doing nothing and should be reconsidered against a saturated baseline.

### What shipped

`gtfs_static.terminals` / `through_stops` / `stops_to_json`; a `counts_from_stop`
filter on `load_r2.build_movement_series{,_by_direction}` that recomputes
advanced_n/stalled_n from the raw `transitions` map (the archived counters are
already summed and cannot be narrowed after the fact); the trainer fitting the
advance baseline on through stops and publishing that same set as
`movement_through_stops` in params.json, from one static-timetable fetch per run
so the fit and the set cannot disagree; the Worker, viz and the offline
reconstruction paths counting the same way. The archived `transitions` map stays
raw and unfiltered — it is the only place terminal behaviour remains observable,
and it is what lets one definition apply to the whole archive instead of the
counters changing meaning at deploy.

Measured with two throwaway scripts at the repo root (both deleted after) over
`archive/vehicles/` and `archive/alerts/` via `murk exec`.


## 2026-08-12 — the 1-minute trace reads back: 3,457 single-hop traversals in 25 minutes, tracking the timetable within 3%

origin: agent

Deployed the per-minute trace (ddefa71) and built the reader. Trace objects land
at exactly 60s spacing, ~650 rows each, while archive/vehicles/ stays on its
300s cadence — the 5-minute pipeline's gate holds.

First reconstruction, over 25 minutes of live data (572 trips, 16,196 rows,
4,337 arrivals):

| outcome | n | share |
| --- | --- | --- |
| exact single hop | 3,457 | 91.9% |
| interval-censored (arrival missed between polls) | 215 | 5.7% |
| right-censored (in transit when last seen) | 89 | 2.4% |
| dropped: stop_seq backwards | 11 | |
| dropped: arrival with no stop_seq | 82 | |

**92% of hops are cleanly measured at 1-minute polling.** That is the number the
whole trace was built for: at 5-minute polling the mean observed move spanned
2.71 stations, so nothing below the multi-station jump was measurable at all.

### Two bounds, not one measurement

A poll brackets a hop rather than pinning it, and the two ends are biased in
opposite directions:

| cut | n | ratio to schedule (median) | p10 | p90 |
| --- | --- | --- | --- | --- |
| arrival -> arrival | 3,420 | **1.033** | 0.667 | 1.667 |
| departure -> arrival | 797 | 0.422 | 0.189 | 0.833 |

Arrival-to-arrival includes the dwell at the origin, so it is an upper bound.
Departure-to-arrival starts its clock at the first sighting that had already
left, which is systematically late, so it is a lower bound — and a weak one:
only 1,014 of 3,761 traversals catch the train in transit at all, and those skew
toward the slow hops that are easiest to catch. The upper bound is the usable
measurement, and it lands 3% over the timetable.

**Negative result: `hop_seconds` does not need a dwell correction.** The obvious
objection to comparing arrival-to-arrival against a departure-to-arrival
timetable is that it double-counts station dwell. Measured instead of assumed:
NYCT sets arrival_time == departure_time on 95.7% of stop_times rows (200k
sampled), median 60s on the 4.3% that differ. The schedule allocates no dwell,
so the scheduled hop IS its arrival-to-arrival time and the comparison is
like-for-like. Real trains still spend ~30s standing at each stop that the
timetable never allocates, which is most of the gap between the 120s observed
median and the 90s scheduled one.

### Reconstruction rules that cost real effort to get right

Two of these were wrong in the first draft and produced plausible-looking
numbers anyway, which is the reason they are written down:

  * **Right-censoring needs in-transit evidence, not just absence.** A trip last
    seen standing at a stop FURTHER ALONG than its last recorded arrival has
    already completed the hop — we simply could not time it (the feed omitted
    stop_seq). Recording "still running at T" there is false, not unverified,
    and stretches every fitted traversal. The rule that holds: censor only when
    the last sighting is in transit toward a stop other than the last arrival.
  * **A departure observed at the same instant as the next arrival is not a
    departure.** It bounds travel time below by zero, which is no information.
    Emitting 0 there would drag any fit down hard; those records carry
    moving_seconds=None instead, and 2,443 of 3,457 exact hops are in that state.
  * A feed gap after the last in-transit sighting does not weaken the censoring
    bound — the train demonstrably was still moving when last seen, and the
    endpoint is the feed's own vehicle_ts, not a poll time. A gap that swallows
    an arrival surfaces as a stop_seq jump instead, which is interval-censored.

Interval-censored spans are dropped from the survival samples rather than split
evenly across their hops: the fitters in survival.py carry a right-censored
likelihood only, and "each of these 2 hops took at most 200s" is an upper bound
they cannot express. Splitting would fabricate observations. At 5.7% of
traversals the discard is small, and TraceStats reports it.

Reproducible: `murk exec -- uv run python -m training.trace`.


## 2026-08-12 — the through-stops retrain landed, but three silent faults sat between the code and the artifact

origin: agent

The through-stop counting fix had been deployed and inert since 8e88644: the
Worker falls back to counting every stop until params.json carries
`movement_through_stops`, and the retrain is manual. Publishing it turned out
to require fixing three things first, none of which announced themselves.

### 1. `dwell_movement` had silently gone to zero cells

The first dry run reported `dwell_movement_cells: 0` against the 25 routes the
live params carried. One line on stderr: `movement dwell skipped (math domain
error)`, swallowed by the fail-soft wrapper that exists so an archive hiccup
can't block a publish.

The cause is numerical, not statistical. `pooled_dwell` fits a shared shape
across routes and then sweeps each route's log-scale over e^4..e^15.9 by golden
section. When a state's episodes are all one tick long — two disrupted samples
of 294s and 300s — the shared shape fits at **172.95**, and at the top of that
sweep `(294/86679)**172.95` underflows to exactly 0.0. `math.log(0)` raises.

Fixed by evaluating the likelihood through the logarithm: `log_z = shape *
(log t - log scale)`, with `log(1+z)` as a softplus so neither end can overflow.
`training/dwell._loglogistic_survival` had the mirror-image bug — at that shape
`(t/scale)**shape` raises OverflowError in Python while the TS twin in dwell.ts
quietly yields Infinity, so the two sides disagreed only on the inputs that
crash. After the fix: 32 cells, and the published block is back to 25 routes.

The general lesson is about the guard, not the math: `t = max(t, _MIN_DURATION)`
keeps `t` positive, which looks like it protects `log(t)` and does — but `z` is
a *power* of a ratio and has its own range. Flooring the input does not bound
the intermediate.

### 2. Every local retrain since 2026-08-10 stamped a stale `code_sha`

The published params claimed `code_sha: fa513e5`, a commit from 2026-07-17,
with `dirty: null` — for a run that produced a feature that commit predates.
The previous live params said the same thing.

`.build-sha` is a gitignored file that `trainer deploy:ci` writes into the
checkout before the container build, because the image excludes `.git`.
`code_provenance()` consulted it *above* `git rev-parse HEAD`, so once a local
deploy has run, the file shadows the live tree forever and `dirty` can never be
computed. Reordered: env var, then the live git tree, then the file, then
unknown. The file is the container's only source; in a checkout the tree that
actually ran the code is both more accurate and the only thing that can answer
whether it was dirty.

### 3. The Worker's committed movement-transition stream is not yet a source

`_movement_dwell` preferred `v1/movement_transitions` whenever it was non-empty,
falling back to replaying `published_condition` ticks. That stream started
filling at the deploy that introduced it, so the preference flipped this week —
silently, on presence alone. Over 2026-07-30..08-12 it holds **13 route
transitions against the replay's 124**; episode census 3 vs 28, and cells with
enough completed episodes to speak for themselves 0 vs 3.

The two sources do agree: 10 of the 13 match a replay record on route, target
state and onset within one tick. Exact-key agreement is 0/28, because the two
debounce the onset differently — which is why the loose matcher is the honest
one here. So this is a change of source, not of signal, and it is still a model
change that has to be graded rather than triggered by a deploy. Pinned to the
replay; the switch is its own piece of work.

### The publish

`v1786578716` over 2026-07-30..08-12: 28/28 routes, `movement_through_stops`
with **1,659 stops across 25 routes**, dwell_movement 25 routes (32 cells, 7
with an atom), baseline 210 cells. Live snapshot at the next boundary tick:
26/30 routes movement-sourced, the 4 `unknown` being GS, 7X, 6X and FX. No
`advance counters include every stop` line in the Worker log, which was the
whole point.

Two fits that the through-stops commit had left unscoped are now scoped by the
same set: `write_segment_params`' per-leaf baseline (1,658 cells, one per
published through stop) and `write_segment_dwell`'s vehicle reconstruction.
They were fitting over layovers while the Worker scored against a through-stop
normal.

**Still false in the live artifact:** `v1786578716` was published before the
provenance fix, so it carries the stale sha. The corrected stamp needs one more
publish from a recorded commit.


## 2026-08-12 — the degradation label was mostly measuring the clock: hourly bins cut degraded prevalence 11.0% -> 1.8%

origin: agent

The assigned_n degradation label is meant to be the truth a movement model gets
graded against. Over 2026-08-03..08-13 it called **11.0% of all judgeable
route-ticks degraded**, and the per-hour breakdown said most of that was not
disruption at all:

| UTC hour | ET | prevalence, tod_bin | prevalence, hourly |
| --- | --- | --- | --- |
| 10 | 06:00 | **26.8%** | 5.9% |
| 03 | 23:00 | **18.9%** | 2.6% |
| others | | 4.6-14.7% | 0.2-8.0% |

Both spikes sit at a `tod_bin` edge. The bins are 4-6 hours wide (00-05, 06-09,
10-14, 15-19, 20-23), so the 06:00-09:59 median is set by the 08:00 rush peak
and a route running its genuine 06:00 service reads at a third of "normal".
Measured on 14 synthetic weekday mornings ramping 6 -> 30 trains, the wide
bucket's median is 21 and 06:00 service scores 6/21 = 0.29, under the 0.5
degrade floor, every single morning.

Switched this label — and the `recovery_independent` grading path that consumes
the same call — to `momentarily.hmm.schedule_bin`: ET (weekday|weekend, hour),
which already existed for the schedule-rate channel and is already mirrored in
the Worker. `tod_bin` is untouched; the HMM emission channel still scores the
live service ratio against the `(route, tod_bin)` baseline shipped in
params.json, and that pairing has to keep agreeing with the Worker.

Result over the same window: 1,122 baseline cells instead of 131, degraded
prevalence **11.00% -> 1.79%**, events 210 -> 142, distinct routes 23 -> 14 (the
nine that vanish were flagged only by the bin edge), and the share of onsets
landing in the top two ET clock hours 42% -> 30%.

**Negative result: finer than hourly is not worth it.** Same window, same
thresholds, four granularities:

| bin | cells | judgeable ticks | events | degraded | top-2 ET hours |
| --- | --- | --- | --- | --- | --- |
| tod_bin (4-6h) | 131 | 69,636 | 210 | 11.00% | 42% |
| hourly | 1,122 | 69,546 | 142 | 1.79% | 30% |
| half-hour | 1,197 | 55,963 | 66 | 1.22% | 26% |
| quarter-hour | 2,364 | 55,621 | 54 | 0.90% | 26% |

Half-hourly buys 4 points of edge concentration and costs **20% of every
judgeable tick** (69,546 -> 55,963) plus two more routes, because cells fall
under `min_samples`. Note the cell counts: half-hour yields 1,197 where the
split predicts ~2,244, so nearly half the buckets never reach 20 samples in an
11-day window. Quarter-hour drops ~47%. Hourly is the last granularity that
costs no coverage at all (69,546 of 69,636, a 0.13% loss).

The residual concentration after the fix is ET 05:00 (27 onsets) and 06:00
(16) — the hours service ramps hardest within the hour. That is the part a
half-hour bin would actually fix, and it is not worth a fifth of the label.


## 2026-08-12 — per-segment traversal baselines: 1,338 of 1,924 hops fit their own curve off eight hours of trace

origin: agent

`training/traversal.py` turns the minute trace's per-(trip, hop) traversals
into a per-(route, direction, from_stop, to_stop) baseline, plus the deviation
of a live hop from it. First fit, over 2026-08-12 16:30Z..08-13 00:14Z (473
snapshots, 7h45m — the whole archive, which started today):

| | |
| --- | --- |
| exact single hops fitted | 74,706 |
| distinct hops seen | 1,924 |
| hops fitted on their own data | **1,338** |
| hops anchored on their scheduled time | 257 |
| hops omitted, thin and unscheduled | 329 |
| median own-fit hop | 109.1 s |
| median p90/median within a hop | 1.351 |
| median observed/scheduled | 1.056 |

Eight hours is already enough for 70% of hops to clear 20 observations, which
was the open question — the median hop gets 47 samples.

### Arrival-to-arrival, not travel time

`Traversal` carries two clocks and `to_dwell_samples` returns the
departure-to-arrival one, which sounds like the right thing to compare against
a departure-to-arrival timetable. It is the wrong one to FIT, for two measured
reasons: only 24,320 of 74,706 exact hops (26.5%) ever caught the train in
transit and so have a departure at all, and the ones that do are selected for
being slow — a hop still in motion when the next minute's poll lands. Fitting
the tail-detector on a tail-biased quarter of the data is backwards.

Arrival-to-arrival is also the rider's quantity, and the timetable is
comparable to it anyway: NYCT allocates no dwell on 95.7% of stop_times rows,
which is why observed/scheduled lands at 1.056 rather than somewhere near 1.3.

Only EXACT single hops reach a cell. A RIGHT-censored traversal has `to_stop =
None` — the train was last seen heading somewhere it never reached — so there
is no segment to file it under, and filing it by from_stop would pool a branch
point's two successors. It is not a right-censored observation of a known hop;
it is an observation of an unknown one. 390 dropped, against 5,295 interval
spans.

### Thin hops take their level from the timetable, not from the population

Pooling raw seconds across segments would put a 60-second hop and a 400-second
hop in one distribution. The exchangeable quantity is the RATIO to the
scheduled time, so a thin hop gets the population's ratio curve rescaled by its
own scheduled hop — level from the timetable, shape and dwell allowance from
the population. A thin hop the timetable does not name gets no cell at all.

### The signal the advance rate could not reach

The slowest own-fit segments run **4.03x, 3.88x and 3.82x** their scheduled
time (n = 93, 64, 104). A binary "did it leave" cannot express that at all, and
at 5-minute polling those stations were mostly interior to a multi-station jump
and never observed.

**Bug found by a degenerate fixture:** `fit_loglogistic` overflowed on a sample
with zero variance (30 hops all timed at 180 s). The MLE shape is at infinity,
and Nelder-Mead's expansion step reached log-shape 776 — `math.exp` raises long
before the likelihood says stop. Bounded the simplex to log-shape +/-6 and
log-scale +/-20; shape 400 is already a point mass, so the box is inert on real
data. Same family as the log-space likelihood fix earlier today, different
mechanism: that one underflowed inside the likelihood, this one overflowed in
the parameterisation on the way to it.

Not graded. There is one evening of archive and no held-out window; the epic
says weeks, and nothing here is retroactive.


## 2026-08-12 — the advance trip-wire graded against the assigned_n label: 7 firings in 54,000 ticks, zero agreement. Retuning does not fix it and neither does the estimator

origin: agent

The advance trip-wire (`load_r2.classify_direction`) had never been graded
against a label that could see it. The assigned_n degradation label now can be
trusted enough to try (see the hourly-bins entry above), so: 10 days of vehicle
archive, through-stops counting, the live baseline, scored per (route, tick)
against the label's degraded/normal. Read every count below as agreement with
that proxy, not with ground truth — the last section is about the gap between
the two.

```
label base rate                                        1.795%
current (ratio 0.5, alpha 0.05, min_matched 3)
  graded 53,993   fired 7   agree 0   disagree 7   missed 276
```

Seven firings in fifty-four thousand judgeable route-ticks, and not one of them
lands on a labelled degradation. Every knob was swept — min_matched 3/8/15,
ratio 0.5/0.25, alpha 0.05/0.001, and their combinations — and every setting
either fires less or fires zero. There is nothing to retune.

### The baseline estimator IS wrong, and fixing it changes nothing

`compute_advance_baseline` takes p0 as the MEDIAN of per-tick advance
fractions. With layovers excluded, **71.3% of ticks have zero stalls**, so the
median sits on the ceiling and gets floored to 0.9990. That is not the cell's
advance rate:

| | median of tick fractions | pooled advanced/matched |
| --- | --- | --- |
| median cell | 0.9990 | **0.9443** |

p0 − true rate is +0.033 at the median and +0.135 at p90, and **73 of 210 cells
publish p0 = 0.9990 for a real rate between 0.80 and 0.95**. That matters
because `_binom_lower_tail(advanced, matched, p0)` is a significance test and
takes p0 as a Bernoulli rate: at 0.999 with 10 matched trips a single stall
reads significant, in a cell where one stall in ten is ordinary.

So it was fixed offline — pooled advanced/matched, with a 0/10/20% one-sided
trim to keep outage ticks from dragging the rate down. The saturation goes away
(median p0 0.9990 -> 0.9443, cells at/above 0.99 drop from 74.3% to 8.6%). The
trip-wire does not improve:

| baseline | ratio | fired | agree | disagree | agreement rate |
| --- | --- | --- | --- | --- | --- |
| median (shipped) | 0.5 | 7 | 0 | 7 | 0.000 |
| pooled, trim 0 | 0.5 | 7 | 0 | 7 | 0.000 |
| pooled, trim 0 | 0.7 | 896 | 3 | 893 | **0.003** |
| pooled, trim 0.1 | 0.7 | 1,129 | 3 | 1,126 | 0.003 |
| pooled, trim 0.2 | 0.7 | 1,421 | 3 | 1,418 | 0.002 |

At a 1.795% base rate, an agreement rate of 0.003 is five times WORSE than
firing at random. Against this label the wire is not weak, it is
anti-correlated — the same inversion three movement scores showed against the
alert truth, now reproduced against a proxy that shares no feed with the
alerts.

**Why, and it is structural.** assigned_n degradation is service being
WITHDRAWN: fewer trains dispatched. The advance rate measures whether the
trains that ARE out keep moving. Those are different failure modes, and their
coverage is anti-correlated — withdraw service and the movement channel has too
few matched trips to judge at all. Of 1,249 labelled degraded ticks only 276
overlap a judgeable movement call, and at min_matched=8 only 5 do.

### What this settles

The estimator fix is NOT being shipped. It is correct on its own merits and it
moves the live classifier's operating point, and by measurement it buys
nothing — publishing an ungraded operating-point change that improves no metric
is the churn that produced 18 params versions in 46 days. It waits for a truth
that can see freezing rather than withdrawal, which is what the traversal model
is for.

Reproducible from `training.degradation_label` + `load_r2.classify_direction`
over any 10-day window; the sweep is three nested loops over the knobs.


## 2026-08-13 — two arms were sharing one forecast field, and the mix scored worse than either: AUC 0.378 -> 0.856

origin: agent

`route_status[route].condition` has been movement-primary for a while.
`route_status[route].inference.p_normal_in_30min` was not: it came from
whichever arm `recovery_source` named, which was usually the alert-HMM, timing
the alert regime on the alert clock. Both numbers sat in the same object, so a
reader had every reason to think the forecast described the condition.

Graded properly for the first time — each prediction against the condition we
actually published 30 minutes later, six days of the stream:

| rows | n | AUC |
| --- | --- | --- |
| movement-sourced | 4,741 | **0.856** |
| alert-sourced | 20,497 | 0.261 |
| all of them, one field | 25,238 | **0.084** |

The mixture scoring below both components is the whole finding. The two arms
put their probabilities on different scales, so ranking across the union tracks
which arm answered rather than which route is at risk. Simpson's paradox, in
production, in a field consumers read.

### The rule

A forecast is published only when it is about the condition that was published.
Movement qualifies by construction — same arm, same clock, same regime. An
announced resume time qualifies too: it is a fact about the route rather than a
rival model of it. Everything else is withheld (null), the way the 60- and
120-minute horizons already are. Applied to the same six days: **AUC 0.378 ->
0.856** on the rows that still publish, and once the movement arm is sourcing
normally that is about 80% of them (on 08-12, 6,635 of 8,352 ticks).

With the alert arm no longer forecasting, `projectForward` fell out of
`snapshot.ts` entirely, along with the whole `else { /* Normal now */ }` branch
and the `toNormal` transition-matrix split. The alert arm still estimates
`recovery_minutes`; it no longer estimates probabilities.

### The same bug in minutes

`recovery_minutes` is the same claim on a different scale, so it was graded the
same way — time to the next tick published normal:

| arm / published condition | n | MAE |
| --- | --- | --- |
| movement / disrupted | 10 | **3.5 min** |
| hmm / disrupted | 23 | 13.5 min |
| hmm / suspended | 12 | 33.8 min |
| hmm / not_scheduled | 4,258 | **1,134.8 min** |

Nineteen hours, on the dominant population. Those are overnight routes: the
published condition is `not_scheduled` and the alert arm was timing its own
regime's return, which has nothing to do with when the trains come back.

Two fixes. The schedule countdown now gates on the PUBLISHED condition rather
than the alert shadow, so an announced resume is used wherever one exists. Where
none exists — the ordinary overnight gap — the estimate is withheld through the
mechanism the contract already has for "this is not a prediction":
`recovery_indeterminate` with the value at the ceiling, exactly as the
outlived-every-dwell case has always done. 4,293 rows a week stop publishing a
confidently wrong number.

**Not made nullable, deliberately.** `recovery_minutes` is an integer in the
published snapshot contract and the Home Assistant integration reads it; the
forecast fields could widen to null because their siblings already were.
Making the recovery block nullable is a consumer migration, not a bug fix.

### Worth remembering

`Number.isFinite(null)` is `false`, and `scrubCorruptInferences` used it as the
validity check for every probability. Widening one field to nullable would have
silently deleted the entire inference block for ~60% of routes. The type system
could not see it: `Number.isFinite` takes `unknown`.

## 2026-08-13 — the scheduled reference was wrong three ways, and the one named in the bug report was the smallest: service-day pooling misprices 26% of hops, the express/local lumping misprices none

origin: agent

Setting out to fix the skip-stop inversion in the traversal measure — the modal
chain crediting a bypassing train with the time of every stop it skipped — and
finding that per-pair keying had already killed it, that the express/local
lumping it was supposed to also fix does not exist in the data, and that the
reference was meanwhile pooling the weekend timetable into weekday hops.

Window: 2026-08-12 16:30Z..08-13 02:56Z, 650 trace snapshots, 109,257 arrivals,
95,294 exact single hops, 6,953 interval-censored spans. All percentages below
are of observed single hops unless stated.

### Decomposing the reference error

`hop_seconds` was one median per (route, direction, from, to) over every trip in
the feed, measured departure-to-arrival. Three things are wrong with that, and
they are separable:

| change to the reference | observed hops mispriced >10% | >25% |
| --- | --- | --- |
| service-day slicing (weekday vs weekend) | **26.0%** | 16.0% |
| clock (departure-to-arrival -> arrival-to-arrival) | 4.9% | 3.5% |
| both | 27.9% | 17.1% |

**Service-day pooling is the dominant error and nobody had named it.** A quarter
of weekday hops were being judged partly against Saturday and Sunday run times.
The clock is smaller than the 95.7% arrival==departure figure suggests, because
the 4.3% that differ carry real scheduled holds — up to 240s at a timepoint.

Fixing both recentres the population ratio: observed/scheduled median 1.050 ->
1.000, and `own_cells.median_ratio_to_schedule` 1.056 -> 1.011.

### The named bug: real, structural, and worth 0.16%

At hop granularity the modal chain is already gone — `traversal.py` keys per
(route, direction, from, to) and only accepts spans the realtime feed calls one
hop, so there is no chain to sum along. What survives is narrower: the feed
publishes its OWN stop sequence, so a train that bypasses a station reports two
consecutive stops the timetable puts a station between. Resolving each traversal
against its trip's own stopping pattern finds **148 of 95,294 such spans (0.16%)**
and drops them; before, they landed in a cell as if they were direct hops.

Real, and correctly fixed. Not the 42,596-transition problem the 5-minute
prototype had.

### Negative result: express and local do not disagree about a shared hop

The premise was that pooling express and local trips over the same pair
misprices both. 1,415 of 1,923 weekday hop keys (85.4% of observed hops) are
served by more than one path code, so if the effect existed it would be
everywhere. Ratio dispersion on exactly those keys, trip-exact scheduled times
against the pooled per-key median:

| reference on multi-pattern keys | IQR | MAD | sd |
| --- | --- | --- | --- |
| the trip's own scheduled times | 0.4667 | 0.2000 | 0.5832 |
| median over trips serving the pair | **0.4653** | 0.2000 | 0.5855 |

The pooled median is marginally *tighter*. Physically obvious in hindsight: on a
consecutive pair both patterns serve, express and local cover the same track at
the same scheduled speed. Express differs in which pairs it has, and per-pair
keying already separates those. Over all keys the same comparison gives IQR
0.4750 (trip-exact) against 0.4833 (pooled) — a 1.7% difference against a 26%
one from the service day.

So `Timetable` stores per-pattern STOP LISTS, which is what detects a bypass and
what a span has to be summed along, and takes its TIMES from the service day's
per-pair medians. 433 patterns and 12,248 stops, against 565,093 per-trip
stop_time rows for no measured gain.

### Matching a realtime trip to the timetable

Only 81% of realtime trip ids match a static trip outright: NYCT dispatch
origins drift off the schedule, reroutes get an `X` path code, and SIR/SS are not
in this feed at all. Two things recover most of the rest.

**The path code determines the stop list.** 433 (service, path) pairs across
20,621 trips, not one carrying a second stop list. So the pattern can be keyed on
the path code rather than the trip.

**Unanimity beats picking.** Realtime truncates some codes (`W..N` for
`W..N30R`), and a service day can run two calendars. Taking the answer only when
every candidate pattern agrees, and falling back to the day's per-hop median
otherwise, lifts pattern coverage 78.4% -> 90.1% against a unique-prefix rule.
Layered with the median fallback:

| how a hop gets priced | share |
| --- | --- |
| the trip's own pattern | 89.9% |
| the service day's median for the pair | 8.8% |
| unpriceable | **1.14%** |

Key-level coverage 83.0% -> 85.9% (1,715 of 2,066 keys, then 1,718 of 2,000).
The 14% that stay unpriced are thin keys the timetable never scheduled — 1.1% of
traffic.

### The trip id's origin field is hundredths of a minute

Needed because a service day is not a calendar date. `060250` is 602.50 minutes
past midnight, 10:02:30 — **not** 06:02:50. Read as hundredths it reproduces the
trip's own first scheduled arrival exactly for 20,311 of 20,621 trips (the other
310 off by a flat 90s); read as HH:MM:SS, 273. The `:50` endings are the tell:
they are half-minutes, and no subway leaves at fifty seconds past.

That gives a service-day rule with no wall-clock cutoff — rank the candidate
midnights by how near each puts the observation to the trip's own scheduled
origin, and let the calendar veto a day that never ran the pattern. Over 106,172
arrivals it put every one on the right day, none more than 50 minutes early
against its own origin. A fixed 4am cutoff was tried first and is wrong in the
other direction: it misfiles trains put into service before their scheduled
origin, which a naive subtraction sends back a full day (1,488 arrivals, 1.4%).

### The measurement floor is the poll, not the reference

Observed hop times cluster hard on 60/120/180s — the 1-minute poll brackets an
arrival rather than pinning it, and 31% of measured hops land on an exact
multiple of 60. The timetable is on a 30-second grid. Where the two grids line
up, the measure looks three times sharper than where they do not:

| scheduled hop | n | ratio IQR |
| --- | --- | --- |
| 60s (on the poll grid) | 9,071 | 0.517 |
| 90s (off it) | 32,418 | 0.522 |
| **120s (on it)** | 22,325 | **0.150** |
| 150s (off it) | 8,812 | 0.387 |
| 300s | 1,383 | 0.200 |

A 120s hop is not physically three times more predictable than a 90s one. Ratio
IQR sits near 0.48 overall under every reference tried, and a large part of that
is the poll rather than the trains. **No further work on the scheduled side
moves the single-hop measure much.**

Which points at the 6,953 interval-censored spans currently discarded. Resolved
against the trip's own pattern, 91% of them price, and their ratio is far
tighter than single hops because the same +/-60s lands on a 2-3x longer span:

| span | ratio IQR | sd | median |
| --- | --- | --- | --- |
| exact single hop | 0.478 | 0.576 | 1.000 |
| interval, 2+ hops | **0.221** | **0.306** | 0.983 |

They are deliberately kept out of the population ratio curve — that curve is
transplanted onto thin SINGLE hops, and a distribution tightened by averaging
would understate their spread. But as a line-speed measure they are 2.7x
cleaner than what the model currently uses, off data it currently throws away.

### The disrupted-vs-normal comparison is no longer NaN. It is worse than that

Re-run under the fixed reference, joining each hop to its route's alert state at
the enclosing 5-minute tick:

| cut | share of route-ticks | disrupted median | normal median |
| --- | --- | --- | --- |
| Delays or Suspended | **94.8%** | 1.000 | 1.000 |
| Delays | 84.7% | 1.000 | 1.000 |
| severity_sum >= 100 | 26.2% | 1.000 | 1.000 |
| Suspended | 34.1% | **1.033** | 1.000 |

Only the suspended cut separates, by 3.3% on the median (p99 3.625 vs 3.000) —
directionally right, where the 5-minute scores came out inverted, but small. The
other three say nothing because the label says nothing: 95% of route-minutes on
an ordinary weekday carry a Delays or Suspended alert. This is the same wall the
advance rate hit. The measure is not gradeable until the label is.
## 2026-08-13 — the timetable is now the instrument, not an input: thin cells abstain, and drift reads 1.0117 against feed 20260807

origin: self

Follow-on decision to the entry above, taken deliberately as an architectural
constraint rather than on the coverage numbers. Movement is the only sensor. A
segment with too little movement of its own now says nothing instead of
borrowing a level from the schedule.

### What changed

`traversal_baseline` no longer has a SCHEDULED cell source. Every level is
fitted from that segment's own traversals; under `MIN_HOP_SAMPLES` there is no
cell. `TraversalCell.source` is gone with it — one source needs no label.

| | before | after |
| --- | --- | --- |
| cells | 1,723 | **1,357** |
| keys abstaining | 284 | **663** |
| share of traffic covered | 99.1% | ~97% |

The 366 cells that went away were carrying 2.1% of traffic on borrowed levels.
That is the price, and it shrinks: those segments cross the sample floor on
their own as the trace archive fills.

### Why, in one line

A reference that feeds the fit cannot then detect the fit drifting.

### The drift measure, which is the point

`schedule_drift` compares fitted cells against the timetable they were not
fitted from. First reading, 2026-08-12 16:30Z..08-13 03:00Z:

| | |
| --- | --- |
| feed_version | `20260807-H-rockaways-extension-removed` |
| cells compared | 1,347 (10 fitted cells the timetable never names) |
| median fitted / scheduled | **1.0117** |
| p10 / p90 | 0.7946 / 1.3516 |
| share of segments >= 1.25x schedule | **16.0%** |

The fleet runs its own timetable to within 1.2%. One segment in six has a
NORMAL that is a quarter over schedule — some of those are standing slow orders,
so the level is not the alarm. The change in it is, and this is the baseline
that change gets measured from.

### The ruler can move too, so it is on the record now

The static feed is a snapshot, not a standing truth. Facts as of today: the
object at `rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip` was Last-Modified
2026-08-07, six days old, self-versioned
`20260807-H-rockaways-extension-removed` — the name says it dropped a branch —
and it declares itself valid 2026-05-26..2026-10-31.

Three gaps that made the drift measure meaningless until closed:

- `fetch_gtfs_zip` re-downloads every run. No caching, no ETag check, no pin.
- **Nothing recorded which feed version a measurement used.** `provenance.py`
  still has no GTFS field; `ScheduleDrift` now carries `feed_version` so at
  least the measure is self-describing. The trainer artifact does not yet.
- **Nothing checked the validity window.** The vehicle archive reaches back to
  June; the trace archive is 30 days. Replaying a window from before this feed
  took effect would have compared trains against a schedule that did not exist
  when they ran — and Rockaways service was *removed* in this version, so that
  is not hypothetical. `Timetable.covers` now guards it: an out-of-window
  observation still fits its cell and carries NO scheduled time, which keeps it
  out of the drift measure by construction rather than by remembering to filter.
  `n_outside_feed_window` reports it; the current window reads 0.

### Also

Dropped the pricing metaphor ("priced", "mispriced", "unpriceable") that had
spread through three modules. A timetable does not price anything; it says how
long a hop is scheduled to take.

## 2026-08-13 — correction: the bypass filter was the schedule choosing the training set

origin: self

Both entries above describe bypasses as detected and DROPPED. That was wrong and
is now changed: they are counted and kept.

A bypass is a span the realtime feed calls one hop and the trip's own timetable
puts a station inside. Excluding it looked like hygiene and was two mistakes.
The train covered that stretch without stopping — physically the same movement
an express makes on the same pair, so it is a real measurement of it. And
bypasses cluster on bad days, so removing them taught the baseline a cleaner
normal than the one riders got, using the schedule to decide what the model was
allowed to learn from. Under a movement-only constraint the timetable does not
get to choose the training set.

`hop_samples` now admits observations on realtime evidence alone — EXACT, has a
destination, one hop by the feed's own stop sequence — and reads the timetable
only to attach a comparison. `TraversalStats.n_dropped_bypass` is `n_bypass`.

Re-run over 2026-08-12 16:30Z..08-13 04:00Z (the archive is still filling, so
this window is larger than the one above, not just a rerun of it):

| | |
| --- | --- |
| traversals | 148,337 |
| bypasses, kept | 202 |
| cells | 1,533 |
| median fitted / scheduled | **1.0058** |
| share of segments >= 1.25x schedule | 14.45% |

## 2026-08-13 — the planned-work grade ran end to end and graded almost nothing: two days of trace cover disjoint clock bands, and the 7's daily work blacks out its own control

origin: self

`training/planned_work.py` had 436 lines of tests and no callers — the measure
had never touched real data. It has now. The harness works; the answer is that
the archive cannot yet supply the primary grade, and one whole class of window
may never supply it.

### What the first run saw

| | |
| --- | --- |
| trace snapshots | 1,329 (Wed 12:20 ET -> Thu 10:41 ET, 22.3h) |
| traversals | 171,960 (159,374 exact) |
| announced windows in the alert archive | 394 |
| **over the trace span** | **2**, both `Express to Local` on the 7, naming 1 stop |
| unknown alert types | 0 |

2 of 394 is not a shortfall, it is the rate: ~2.2 geo-scoped windows a day
against a 22-hour archive.

### The secondary grade ran and found nothing, which is the expected answer

Difference-in-differences on the boundary hops, 4 affected keys against 36
control keys:

| window | inside | affected lift | control lift | effect |
| --- | --- | --- | --- | --- |
| Wed 15:00-22:00 | 321 | 1.0099 | 0.9982 | **1.0118** |
| Thu 06:15-10:00 | 248 | 0.9990 | 0.9942 | **1.0048** |

An express told to run local does not slow the hops beside it down; it stops
making its own long hops. Duration is the wrong instrument for this type and
the module says so — 1.007 is that prediction confirmed, not a miss.

### The primary grade returned zero rows, and zero was ambiguous

`pattern_shift` produced nothing for either window. The control arm wants the
same band of the local clock on a comparable service day, and the funnel dies
entirely at that filter:

| window | on-route hops | inside | same service class | **+ same clock band** |
| --- | --- | --- | --- | --- |
| Wed 15:00-22:00 | 8,726 | 3,815 | 4,911 | **0** |
| Thu 06:15-10:00 | 8,726 | 2,254 | 6,472 | **0** |

Not a property of these two windows. The trace's two calendar days cover
Wed 12:20-24:00 and Thu 00:00-10:41 — **disjoint clock bands**, so no window of
any type, on any route, could have drawn a control from this archive.

That had to be dug out with a throwaway script, because an empty
`pattern_shift` reads the same whether the pattern held or the comparison never
ran. Those are opposite conclusions. `control_supply` now counts the traversals
the matched arm had to work with, and the report carries
`coverage_no_control_period` beside `graded_coverage`: this run reads 0 graded,
2 with no control period, which is a measurement that did not happen rather
than a detection that failed.

### The sharpest finding: recurring work erases its own control

The 7's `Express to Local` runs 15:00-22:00 and 06:15-10:00 every weekday. Its
control period is the same clock band on the adjacent weekday, which is the same
work. The blackout that keeps other announced disruption out of the baseline
correctly removes all of it:

| tonight's window | control traversals in the trace | after blackout |
| --- | --- | --- |
| 7 Express to Local, 15:00 | 3,815 | **0** |
| 6 Part Suspended, 21:30 | 1,376 | 1,376 |
| L Part Suspended, 22:45 | 593 | 593 |
| SI Special Schedule, 23:00 | 299 | 299 |
| A Reroute, 23:45 | 1,588 | 1,588 |
| A Stops Skipped, 23:45 | 1,588 | 1,588 |
| E Stops Skipped, 23:45 | 781 | 781 |

`Express to Local` is 278 of the 394 announced windows. Where it recurs daily,
an adjacent-day control cannot exist, and the answer is not to relax the
blackout — a control drawn from the same work is not a control. That subset
needs a different comparison (a non-adjacent unaffected day, or the local as the
within-window control the way the 7/7X split already reports it). Until then the
gradeable supply is the other 116 windows, not the headline 394.

### What lands next, and it is hours away, not weeks

13 gradeable windows are announced in the next 4 days, and the six above already
have their Wednesday-night control sitting in the trace. Re-running over
2026-08-12..08-14 should produce the first coverage rows the measure has ever
returned from real movement, on multi-stop part suspensions and reroutes — the
shape of work this measure was actually built for.

## 2026-08-13 — correction: the control diagnostic counted a route where it had to count a service

origin: self

Two corrections to the entry above, both caught in review before the code left
the tree.

### The diagnostic could have certified a comparison that never happened

`control_supply` shipped summing control traversals over the whole ROUTE.
`pattern_shift` emits a row only when ONE service has both arms — that is the
entire reason it reports the 7X apart from the 7. So a route-level total can
report control on the strength of the local while the express the work actually
named had none, and the runner would score the empty result
"compared, found nothing" instead of "never ran". That is the precise confusion
the field was added to remove, reintroduced one level down.

It now returns `dict[service, int]`, and `_service_of` is shared with
`_shift_one` so the grade and the diagnosis cannot disagree about which service
a traversal belongs to. The empty result splits into three states, because there
are three:

| | |
| --- | --- |
| `graded_coverage` | a service had both arms |
| `coverage_no_paired_service` | some service had control, none had both arms |
| `coverage_no_control_period` | no service had a control arm at all |

**No published number changes.** Re-run over the same window: 0 graded, 2 with
no control period, 0 with no paired service — and the six windows in the table
above resolve to a single service each (`{'6': 1376}`, `{'L': 593}`,
`{'SI': 299}`, `{'A': 1588}`, `{'E': 781}`), with the 7 an empty dict rather
than 3,815 -> 0. The defect was latent, not active. Recorded anyway: a
denominator that is right by accident is still the wrong denominator, and
`Express to Local` — 278 of 394 windows, the one class that routinely carries
two services — is exactly where it would have fired first.

### The origin line was wrong

The entry above reads `origin: self`. It should read `origin: artifact`: nothing
about disjoint clock bands was found by inspection, the grader's own output
surfaced it on first run. Left uncorrected in place, per the append-only rule.

## 2026-08-13 — the advance rate was never the weak arm: as a continuous score it reads 0.925 where movement is visible at all, and the progress ratio reads nothing

origin: artifact

`training/progress.py` grades both movement arms against the assigned_n
degradation label over the trace's own window (Wed 12:20 -> Thu 19:55 ET,
31.6h, 1,891 snapshots, 268,813 traversals, 246,473 priced hops). Two results,
and the arm I expected to win lost.

### Both arms, same window, same denominator

Scores are stated so HIGHER MEANS WORSE. Intervals are a stratified percentile
bootstrap over 2,000 resamples. Acute is graded against NORMAL ticks only.

| | progress ratio | stalled share |
| --- | --- | --- |
| graded against the label | 7,790 | 7,820 |
| degraded among them | 54 | 51 |
| pooled AUC | 0.535 [0.440, 0.628] | 0.606 [0.507, 0.712] |
| **within-route AUC** | **0.476 [0.380, 0.579]** | **0.684 [0.618, 0.759]** |
| acute-onset AUC | 0.605 [0.441, 0.762] (n=14) | 0.755 [0.581, 0.893] (n=17) |
| median, normal ticks | 1.000 | 0.043 |
| median, degraded ticks | 1.017 | 0.250 |

The progress ratio's within-route interval straddles 0.5 and is centred just
below it. Against this label, over this window, it has no signal. The medians
say why in one line: when assigned_n collapses, the trains still running cover
their hops in 1.017x their booked time instead of 1.000x. **Withdrawn service
does not slow down the trains that remain.** The same ticks carry six times the
stalling, 0.25 against 0.043.

### The reversal, and it reverses my own entry from yesterday

On 2026-08-12 the advance arm was written up here as anti-correlated: 7 firings
in 53,993 ticks, 0 agreement, 0.003 when loosened, "five times WORSE than
firing at random." That entry graded the TRIP-WIRE — `classify_direction`'s
thresholded verdict with its p0 baseline and binomial test. This one grades the
underlying continuous rate, and the rate separates the classes at 0.684
within-route, interval clear of 0.5.

Both are true and they do not conflict. At a 0.65% base rate a ranker at AUC
0.68 still has dismal precision at any threshold, which is exactly what the
sweep found. The correction is to the CAUSE, not the numbers: the wire's
failure was never evidence that trains-not-moving is uninformative about
service being withdrawn. It is evidence that a significance test against a
saturated p0 is the wrong decision rule on top of an informative quantity.

### The real constraint is that there is nothing to look at

Raising the matched-trip floor makes the score BETTER and the measurement
rarer — the signature of a coverage limit, not a skill limit:

| min_matched | degraded ticks judgeable | coverage | within-route AUC |
| --- | --- | --- | --- |
| 3 | 51 | 25.5% | 0.684 [0.612, 0.752] |
| 8 | 19 | 9.5% | **0.925 [0.903, 0.945]** |
| 15 | 0 | 0% | no answer — empty class |
| 25 | 0 | 0% | no answer — empty class |

A small-denominator artifact would have gone the other way. This one sharpens
to 0.925 as the floor rises, then runs out of data entirely.

The number underneath it is the sharpest thing this run produced. Across the
193 degraded ticks carrying a movement row at all:

| | matched trips at through stops |
| --- | --- |
| median normal tick | 15 |
| **median degraded tick** | **0** |
| degraded ticks with exactly zero | **50.8%** |

Not "few". Zero, on half of them. Yesterday's entry put this at 276 of 1,249
degraded ticks judgeable and called the coverage anti-correlated; the median is
the honest statement of it. A movement arm cannot be graded on the ticks that
matter most, because withdrawing service withdraws the observations a movement
call is made from. Every AUC above describes the mild tail where trains are
still out.

### One latent defect, no published number moved

`grade` first computed the acute AUC with chronic ticks swept into the NEGATIVE
class, scoring each arm for failing to rank a fresh collapse above a route
already known to be down. `_split` now names both classes and drops anything in
neither. The acute numbers moved by 0.0002 and 0.0002, because the scored
chronic ticks are ~0.5% of a 7,700-strong negative pool. Latent, not active,
and recorded for the same reason the `control_supply` one was: the comparator
was right by accident.

### What this settles

The progress-ratio-vs-advance-rate comparison the epic has carried as
unanswerable now has an answer against this label: advance rate, decisively, on
the judgeable quarter. That does NOT retire the progress ratio. The two measure
different failure modes and this label contains only one of them — withdrawal,
not slowness — so the result is as much a statement about the answer key as
about the arm. The instrument that can see slowness is the planned-work grade,
which is announced, geo-scoped, and as of tonight finally has its control band
in the trace.

Two things NOT shipped on the strength of this. The advance-baseline estimator
fix stays parked: it moves the live operating point, and nothing here shows a
threshold that works at a 0.65% base rate. And 31.6 hours with 51 positives is
one window on one pair of weekdays, with intervals wide enough that the
pooled-versus-within-route gap is the only comparison I would defend.


## 2026-08-13 — correction: nine episodes, not fifty-one ticks. The advance arm's win is not established, and the 0.925 was one incident

origin: self

The entry above is wrong where it says "decisively", and its headline number is
the worst offender. Caught in review before the code left the tree; the numbers
it reports were real, the uncertainty around them was not.

### The bootstrap counted repetitions as evidence

`auc` resampled TICKS. A disruption is one episode observed every five minutes,
so fifty consecutive degraded ticks on one line are close to one observation,
not fifty. Resampling them independently reports an interval far too narrow —
which is precisely how a handful of incidents came to read as decisive.

`_clusters` now groups each maximal run of consecutive same-class ticks on one
route, and the bootstrap resamples those. The point estimates are unchanged.
The intervals are not, and neither is the conclusion:

| within-route AUC | by tick (wrong) | by episode (right) |
| --- | --- | --- |
| progress ratio | 0.476 [0.380, 0.579] | 0.476 [0.428, 0.556] |
| **stalled share** | **0.684 [0.618, 0.759]** | **0.684 [0.450, 0.850]** |
| acute, progress ratio | 0.605 [0.441, 0.762] | 0.605 [0.365, 0.751] |
| acute, stalled share | 0.755 [0.581, 0.893] | 0.755 [0.352, 0.989] |

**The stalled share's interval now straddles 0.5.** 51 degraded ticks are
**9 episodes**; the acute cut's 17 ticks are **3**. The advance arm's point
estimate is still the better of the two, and that is now the whole claim —
"decisively" is withdrawn. Nothing here separates it from chance at 95%.

### The 0.925 was a single incident, and the tight band around it was an artifact

The floor sweep row I put in bold was selected after seeing the sweep, which
alone makes it exploratory. It is worse than that:

| min_matched | degraded ticks | **episodes** | coverage | within-route AUC |
| --- | --- | --- | --- | --- |
| 3 | 51 | 9 | 25.5% | 0.684 [0.450, 0.850] |
| 8 | 19 | **1** | 9.5% | 0.925 — no interval |
| 15 | 0 | 0 | 0% | no answer — empty class |

Nineteen ticks of ONE episode on one route. Its old band of [0.902, 0.948] was
a degenerate resample: with one cluster, every bootstrap draw returns that same
cluster, so the interval collapses to a point and reads as maximum confidence
built from a single incident. `auc` now withholds the interval entirely when
either side has under two clusters, which is the same can't-judge contract the
rest of this module uses, applied to the uncertainty rather than the value.

So the sweep does not show the score "sharpening" to 0.925. It shows the
measurement running out of episodes, with one left at min_matched=8 and none at
15. The direction is still inconsistent with a small-denominator artifact, and
that is all it is good for.

### What survives unchanged

The coverage census, which is a count and not an inference: median 0 matched
trips on a degraded tick against 15 on a normal one, and 50.8% of degraded
ticks with exactly zero. And the descriptive medians — 1.000 vs 1.017 for the
progress ratio, 0.043 vs 0.250 for the stalled share. The mechanism those point
at (withdrawn service does not slow the trains that remain) is unaffected by
how many episodes produced it, because it is a statement about direction and
magnitude, not about significance.

What the real finding is, then: **this window cannot settle the comparison.**
Thirty-one hours contains about nine independent degradations, and no interval
built on nine episodes will separate two arms this close. That is a supply
problem with the same shape as the planned-work grade's, and it wants more
days, not more analysis.


## 2026-08-13 — correction to the correction: five episodes, not nine. Cutting runs on the scored ticks split single disruptions at their coverage gaps

origin: self

The entry above fixed the bootstrap to resample episodes instead of ticks and
reported nine of them. Nine was still too many, for a reason that is the same
coverage problem wearing a different hat.

`_clusters` cut its runs over the keys an arm had SCORED. A movement arm judges
about a quarter of the degraded ticks, so one continuous disruption arrives as
scattered ticks with unscored gaps in between, and every gap started a new
"episode". The clustering was undone by exactly the sparsity it was introduced
to survive.

Episode identity now comes from the full label (`_episode_ids`), which has
every tick, and the arm's ticks are filed under those boundaries:

| | episodes by scored ticks | episodes by label |
| --- | --- | --- |
| degraded, either arm | 9 | **5** |
| acute | 3 | 3 |

| within-route AUC | 9 episodes (wrong) | 5 episodes (right) |
| --- | --- | --- |
| progress ratio | 0.476 [0.428, 0.556] | 0.476 [0.409, 0.528] |
| **stalled share** | 0.684 [0.450, 0.850] | **0.684 [0.451, 0.919]** |
| pooled, stalled share | 0.606 [0.253, 0.867] | 0.606 [0.256, 0.963] |

Five. Thirty-one hours of trace against a label carrying 17 disruptions over
the same span yields **five** that any movement arm could judge, and the
stalled share's interval now runs from worse-than-random to nearly perfect.
Every directional conclusion in the first entry stands withdrawn; what is left
is a point estimate ordering and a census.

The census is the part that keeps surviving these corrections, and it is worth
noticing that it survives them because it is a count rather than an inference:
17 disruptions in the window, 5 judgeable, median 0 matched trips on a degraded
tick, 50.8% at exactly zero. Three passes at the uncertainty have not moved any
of those, and they say the same thing each time — the arms are not being beaten
on skill, they are being starved of the observations that would let them play.


## 2026-08-13 — correction: the origin lines on both grading corrections were wrong

origin: agent

The two correction entries above are both headed `origin: self`. Neither was
found by a human. Both defects — the tick-level bootstrap, and episode
boundaries cut from scored keys instead of the label — were surfaced by
adversarial review of the change before it left the tree, so both should read
`origin: agent`. Left uncorrected in place, per the append-only rule.

Worth recording that the review caught all three methodology defects in this
stream and the grader's own output caught none of them. The run reported its
numbers cheerfully in every broken state: a confident [0.902, 0.948] built from
one incident is not a shape any assertion in the harness could have flagged,
because nothing was wrong with the arithmetic. Only the meaning was wrong.


## 2026-08-16 — the planned-work measure ranks announced work by severity without being told the ordering, and part suspensions slow the hops beside them by a third

origin: artifact

Four days of accumulation turned the measure from "never graded anything" into
its first per-type read. 6,379 trace snapshots, continuous Wed 12:30 -> Sun
22:46 ET with no gaps, 720,058 traversals, 527 announced windows.

| | 08-13 | 08-14 | 08-16 |
| --- | --- | --- | --- |
| over the trace span | 2 | 15 | 31 |
| gradeable | 2 | 13 | 20 |
| **graded** | **0** | **6** | **13** |
| no control period | 2 | 7 | 7 |

### The instrument validates itself: the ordering is monotone in how much of the route the work removes

Share of the route's hop keys that stop appearing inside the window:

| type | rows | median vanished |
| --- | --- | --- |
| Suspended | 1 | 0.4375 |
| Reroute | 4 | 0.2716 |
| Stops Skipped | 4 | 0.1583 |
| Part Suspended | 9 | 0.1250 |
| Special Schedule | 2 | 0.0263 |

Nothing tells this measure that a full suspension removes more service than a
reroute, or a reroute more than a skipped stop. It ranks them in that order
anyway, off movement alone. That ordering is the closest thing to a validation
the measure has had, and it is worth more than any single row in the table.

### Part suspensions are visible on BOTH instruments, and the duration one is the surprise

Difference-in-differences on the hops at the boundary of the closed stretch:

| type | rows | median effect |
| --- | --- | --- |
| **Part Suspended** | **10** | **1.3175** |
| Reroute | 3 | 1.0491 |
| Express to Local | 5 | 1.0233 |
| Stops Skipped | 2 | 0.9895 |

The hops beside a part suspension take **a third longer**, and it is not one
outlier: 8 of 10 rows sit above 1.0, spread 1.29 to 3.07, with the two below
at 0.77 and 0.81. Physically this is what single-tracking and turning trains
short does to the segments either side of a closed stretch, and it is the
answer to the question left open on 08-14 — the type with the most announced
supply (61 windows) is not invisible to this measure, it is the one it reads
most strongly.

Two smaller confirmations. `Stops Skipped` comes in just UNDER 1.0: skipping a
station makes the remaining hops quicker, which is the right sign. And
`Express to Local` reads 1.0233, the third time that type has measured as
approximately nothing (1.007, then 1.023) — duration is the wrong instrument
for it, exactly as the module has claimed from the start, now on five rows
instead of two.

### What the 08-14 entry got right by waiting

That night's read put Part Suspended at 0.0213 vanished and I flagged it as
either a broken measure, a stop-scope gap, or thin data. It was thin data: the
same type reads 0.125 over 9 rows. The two-and-a-half-hour first-day fragments
carried 46 to 112 traversals each and were measuring the part of an overnight
window before the work bites. Writing that up as a finding would have put a
wrong mechanism in the record for the sake of reporting something.

### The honest limits

`n_affected_keys` is **1 or 2** on every part-suspended row. A window names a
stretch, and only the hops at its boundary survive to be timed — everything
inside vanishes and is counted by the other instrument. So each row's effect
rests on one or two segments, albeit with 10 to 113 traversals apiece and a
control arm of 15 to 74 keys. What carries the 1.32 is agreement ACROSS ten
independent windows, not depth within any one of them.

`Suspended` is a single row; 0.4375 is an anecdote. Seven of twenty gradeable
windows still have no control period at all, all of them the recurring
`Express to Local` blackout. And the duration rows carry no service
attribution, so unlike the coverage rows they cannot separate a 7 from a 7X.


## 2026-08-16 — the shippable segment model graded for the first time: not inverted, strongest exactly where the work physically constrains, and short of significance on every cut

origin: artifact

`planned_work` grades the raw traversal data. `training/segment_grade.py`
grades the thing we would actually ship: `traversal.deviation`, a live hop
scored against its own segment's fitted normal. Every movement score graded in
this repo before now has come out at or below chance, so the question was
binary before it was quantitative.

**It is not inverted.** That is the result. Everything after this is about how
far short of a claim it stops.

| | |
| --- | --- |
| gradeable announced windows | 20 |
| graded | 15 |
| abstained | 5 (2 no fitted cell on any boundary hop, 3 thin affected arm) |
| windows above 0.5 | **11 of 15** |
| sign test | **p = 0.118** |
| median AUC | 0.5705 |
| median deviation, affected hops | 1.0116 |
| median deviation, control hops | 0.9727 |

### Where the signal actually lives

| type | windows | median AUC | median affected deviation |
| --- | --- | --- | --- |
| **Part Suspended** | 6 | **0.674** | **1.244** |
| Express to Local | 5 | 0.568 | 1.008 |
| Reroute | 2 | 0.537 | 1.007 |
| Stops Skipped | 2 | 0.491 | 1.120 |

Part suspensions again, and the magnitudes on the four that work are not
subtle: the 1 at 1.32x its own normal (AUC 0.95), the 6 at 1.42x (0.86), the J
at 1.17x (0.66), and the L at **2.46x** (0.69). This corroborates the
difference-in-differences result from the same weekend, which put the same type
at 1.32x by a completely different construction — one measures boundary hops
against control hops, the other against the segment's own fitted history. Two
instruments, one answer.

### The two windows that go the other way are both the N, and both read FASTER

| route | AUC | affected deviation |
| --- | --- | --- |
| N | 0.431 | 0.864 |
| N | 0.448 | 0.893 |

Not noise scattered around 0.5 — two windows on the same route, both saying the
boundary hops ran about 12% QUICKER than their own normal during a part
suspension. The plausible mechanism is that the N's trains bypass the closed
stretch rather than terminating short, and a hop run nonstop is genuinely
faster. Bypasses are deliberately kept in the baseline fit (they really did run
that stretch), so this is a real measurement rather than an artifact, and it
means "part suspension" is not one intervention. Worth resolving before the
type's 1.24 median is quoted as though it described all six.

### The sample is smaller than fifteen, and the unit rule only got it halfway

Grading per ANNOUNCED window rather than per local-day row was deliberate:
`planned_work` splits a period per calendar day, and those fragments are the
same work, so treating them as independent repeats the tick-versus-episode
error that cost three corrections on the label grade. That much this module
gets right by construction — it iterates unsplit windows.

It does not go far enough. Of the 15 graded windows, **all five Express to
Local are route 7**, the same recurring daily work observed on five different
days, and two of the six part suspensions are the same N. So the effective
independent sample is nearer eight or nine distinct work patterns, and the
sign test's fifteen is optimistic. The honest reading of p = 0.118 is therefore
"weaker than that", not "close to significant".

### What is settled and what is not

Settled: the score is directionally correct, it abstains rather than guessing
when a segment has no fitted cell or an arm is thin (5 of 20 windows), and its
magnitude tracks the physical severity of the work on the one type with enough
windows to look at.

Not settled: anything with a p-value. The baseline withheld 112,139 traversals
as inside announced work to avoid grading the model against itself, which is
correct and costly — it is why cells exist for only some boundary hops. More
distinct work patterns is the only thing that moves this, and the collector
supplies them at a few a night without any further work.


## 2026-08-16 — correction: the entry above called it "shippable" while fitting from the future. Refitting each window from its own past only, the answer holds

origin: agent

The entry above grades `traversal.deviation` and calls it the model we would
ship. It fitted every cell from clean traversals across the WHOLE archive,
including days after each window opened. A model running live at 21:30 on a
Thursday has only the past, and a segment whose 20 samples arrive on Saturday
has no cell at all yet. So that result measures whether the signal is IN the
data — retrospective — and is not by itself a deployable claim.

`baseline_before` (`--causal`) now refits per window from traversals that
FINISHED before its start. One fit per window, 3.5 minutes instead of 60
seconds. It is strictly harsher, and the honest surprise is how little it costs:

| | retrospective | causal (past only) |
| --- | --- | --- |
| graded | 15 | 16 |
| abstained thin | 3 | 4 |
| above 0.5 | 11 of 15 | 11 of 16 |
| sign test | p = 0.118 | p = 0.210 |
| median AUC | 0.5705 | 0.5794 |
| **Part Suspended** | **0.674, dev 1.244** | **0.670, dev 1.245** |

Part suspensions land in the same place to three decimal places on the
deviation and within 0.004 on the AUC. Whatever the retrospective fit was
borrowing from the future, it was not what produced the result. The
deployable claim and the data claim agree.

Both modes are kept and neither is allowed to stand in for the other, because
they answer different questions: `--causal` asks what a live model would have
seen, the default asks whether the signal exists at all. A future run reporting
one of them as the other is the defect this entry exists to prevent.

Unchanged: still not significant, and the effective sample is still nearer
eight distinct work patterns than sixteen windows.


## 2026-08-16 — correction: the causal comparison was run against a bigger archive than the retrospective one, and on equal footing the AGGREGATE result is a coin flip

origin: agent

The table in the entry above compares 15 retrospective windows against 16
causal ones. Those are two different fetches — the alert archive grew between
the runs — so it was not a controlled comparison, and "the answer holds" was
asserted from mismatched inputs. Re-run properly: one fetch, both modes graded
off exactly the same bytes (7,008 snapshots, 775,193 traversals, 593 announced
windows, 28 gradeable).

| | retrospective | causal (past only) |
| --- | --- | --- |
| graded | 22 | 21 |
| above 0.5 | 12 of 22 | 13 of 21 |
| **sign test** | **p = 0.832** | **p = 0.383** |
| median AUC | 0.5439 | 0.5542 |
| **Part Suspended** | **n=6, AUC 0.674, dev 1.242** | **n=6, AUC 0.670, dev 1.245** |

Two findings, pulling opposite ways.

### The causal question is settled, and favourably

Now that it is a fair test, past-only fitting costs almost nothing: one window
moves from graded to abstaining, and part suspensions match to 0.004 on the AUC
and 0.003 on the deviation. A model with only its own history reads what the
retrospective fit read. That claim is now earned rather than asserted.

### The aggregate result died, and it should never have been the headline

"11 of 15, p = 0.118" became **12 of 22, p = 0.832** on eight more windows.
That is not a near-miss that needs more data; it is regression to chance, and I
reported the small-sample version as the finding two entries ago.

The deeper mistake is that the pooled sign test was the wrong statistic from
the start, and this module's own sibling says so: `planned_work._median_by_type`
groups by type precisely because **the types are different experiments**. An
`Express to Local` window is one this measure is documented as unable to see by
duration — it has now read approximately 1.00 four separate times. Pooling it
with part suspensions and asking "how many windows beat 0.5" dilutes a real
effect with windows where no effect is expected, and the answer drifts toward
0.5 as more of them arrive. Which is exactly what happened.

### What actually survives, and it is narrow

Part suspensions, on six windows: median AUC ~0.67, affected hops at ~1.24x
their own normal. That number has now been stable across a retrospective fit, a
causal fit, and an archive that grew from 20 to 28 gradeable windows, and it
agrees with the independent difference-in-differences estimate of 1.32x from
`planned_work`. Everything else here is unproven, and the aggregate is not
evidence in either direction.

Next run must report per type and drop the pooled sign test, or keep it only as
a diagnostic clearly marked as pooling incommensurable experiments.


## 2026-08-16 — the baseline blackout tested only a traversal's start, so hops running INTO the work were fitted as normal. No published number moves

origin: agent

`baseline_outside` withheld a traversal when the window contained its START
time. A hop has duration: one that began at 21:25 and arrived at 21:45 spent
most of itself inside a window that opened at 21:30. Those are also the
traversals nearest the work, so admitting them dragged the fitted normal toward
the disrupted level and made the deviation read closer to 1.0. **The bias ran
against detection** — the direction that quietly passes a broken model rather
than failing a good one. `baseline_before` already accounted for duration, so
the two blackouts disagreed with each other.

Both now share `_overlaps_work`, which tests the traversal's whole span against
every announced window on its route. It withholds materially more: 164,694
traversals against 112,139, and 1,592 fitted cells against 1,687.

And it changes nothing. Same fetch, both modes:

| | retrospective | causal |
| --- | --- | --- |
| graded / above 0.5 | 22 / 12 | 21 / 13 |
| sign test | p = 0.832 | p = 0.383 |
| median AUC | 0.5443 | 0.5542 |
| **Part Suspended** | **AUC 0.6745, dev 1.2429** | **AUC 0.6703, dev 1.245** |

Part suspensions have now held at AUC ~0.67 and ~1.24x through a retrospective
fit, a causal fit, an archive that grew 20 -> 28 gradeable windows, and a
blackout fix that removed 52,555 more traversals from the baseline. Four
different ways of being wrong, same answer.

The wider view got worse, as it should have with more windows. `Reroute` now
has 8 windows and reads 0.492 — nothing. `Stops Skipped` 0.481. `Express to
Local` ranks marginally above chance (0.565) with a magnitude of 1.007, which
is a ranking signal with no practical size and matches its four prior readings
of approximately nothing. One type carries this measure, and it is the one whose
work physically constrains the track the trains are still using.


## 2026-08-17 — the N does not run faster during its part suspension, it runs UNCHANGED on two hops. Part suspensions are one mechanism, not two

origin: self

Two N windows read AUC 0.43/0.45 with affected deviations of 0.864/0.893 while
the 1, 6, J and L read 1.17 to 2.46, and I filed that as the type splitting into
two mechanisms — trains bypassing a closed stretch rather than terminating
short. That was a story built on an aggregate. Looking at the hops themselves,
there is no second mechanism.

### The whole arm is two hops, and neither of them moved

Both windows are 23:45-05:00, naming R09, R11, R13, R14, R15, R16. Twelve hops
qualify as boundary hops; ten of them have almost nothing inside the window. Two
carry the arm:

| hop | in-window median | out-of-window median | n in / out |
| --- | --- | --- | --- |
| R17N->R16N (Wed) | 118 s | 121 s | 14 / 527 |
| R17N->R16N (Thu) | 120 s | 121 s | 15 / 526 |
| R08S->R09S (Wed) | 159 s | 170 s | 7 / 465 |
| R08S->R09S (Thu) | 166 s | 170 s | 14 / 458 |

One to eleven seconds on two-minute hops, off 33 and 23 observations across a
five-hour overnight window. That is not a route running faster under
disruption; it is a route whose boundary hops were not measurably touched, on an
arm too thin to say much either way. An AUC of 0.43 on 33 observations is a coin
flip, and I read a mechanism into it.

So the type does not split into "slows down" and "speeds up". It splits into
DETECTED (1, 6, J, L) and NO MEASURABLE CHANGE ON A THIN ARM (N, twice). The
bypass-versus-terminate-short hypothesis has no support here and is withdrawn;
what the N actually shows is that a part suspension does not necessarily
congest its own boundary.

### The absolute deviation was not comparable across windows

Chasing this surfaced a real reporting defect. `median_affected_deviation`
scores hops against a cell level fitted over ALL hours, while a window covers
particular ones — so an overnight window carries whatever the overnight hour
was doing, mixed into the effect. `WindowGrade.effect` now divides the affected
arm's deviation by the control arm's over the same minutes, which absorbs the
hour the same way `planned_work`'s difference-in-differences does, in deviation
space. It moves the headline:

| type | windows | above 0.5 | median AUC | median effect | raw deviation | sign p |
| --- | --- | --- | --- | --- | --- | --- |
| **Part Suspended** | 6 | 4 | 0.670 | **1.201** | 1.245 | 0.688 |
| Express to Local | 6 | 5 | 0.586 | 1.042 | 1.004 | 0.219 |
| Stops Skipped | 2 | 1 | 0.494 | 1.023 | 1.080 | 1.000 |
| Reroute | 7 | 3 | 0.498 | 0.926 | 0.981 | 1.000 |

Part suspensions come down from 1.245 to 1.201 once the hour is taken out. In
the same measurement, the control arms sat at 0.96-1.08, so the correction is
small — but "small once measured" is a different statement from "assumed
negligible", and only one of them is defensible.

### The pooled sign test is now labelled as a diagnostic in the code

`Report` renames it `pooled_sign_test_p` and carries the reason in a comment
beside the field: it pools types that are different experiments, `Express to
Local` has read approximately 1.00 on every window ever graded, and the 11-of-15
to 12-of-22 collapse was composition rather than evidence. `by_type` now carries
its own per-type sign test, and nothing in that table is significant — 4 of 6
at p=0.688 is the best part suspensions can claim.

What is still true after all of this: part suspensions are the only type with a
magnitude worth the name, at ~1.2x control on six windows, agreeing with an
independent difference-in-differences estimate. That has survived every attempt
to break it. It is still not significant.

## 2026-08-17 — recurring work does not need a different control, it needs coverage: the answer key is six days deep and every comparable day carries the work

origin: self

The bug said `Express to Local` recurs daily, so its adjacent-day control is the
same work, so that subset "needs a different comparison — a non-adjacent
unaffected day, or the local as a within-window control." I built two of those
comparisons and measured a third. All three fail, and the premise is wrong: the
control definition was never broken.

### The estimator already admits a free day

`_is_control` restricts nothing to adjacent days. Given the nearest free
comparable weekday for the R's 12:45-18:00 reroute, a traversal at Fri 08-07
12:55 passes every filter — service class 0 == 0, clock band true, blackout
clear, `_is_control` true — with no code change. The 15 ungraded windows are not
waiting on a statistic. They are waiting on archive.

### What the 15 actually are, and it is broader than the bug says

| work | windows | cadence |
| --- | --- | --- |
| 7 Express to Local | 6 | every weekday, 06:15-10:00 and 15:00-22:00 |
| R Reroute | 6 | every weekday, 07:15-09:00 and 12:45-18:00 |
| N Stops Skipped | 2 | every Sat and Sun, 06:00-23:00 |
| 4 Stops Skipped | 1 | continuous, 2026-04-27 to 2026-08-18 |

Not one class but four cadences, including a 113-day continuous window that no
adjacent-day control could ever serve.

### Negative: the out-of-band control does not recover the primary

Same service class, same day, the hours the work does NOT run, normalised the way
07953e6 normalised the duration effect — `vanished` on hops touching a named stop
minus `vanished` on hops touching none, so pattern diversity in the reference
cancels. Graded against the matched-day `vanished` on the 12 service-rows where
both exist:

| | matched-day | out-of-band |
| --- | --- | --- |
| mean | 0.1976 | **-0.0616** |
| median | 0.1250 | **-0.0684** |
| pearson r | — | **0.564** (n=12) |
| sign agreement | — | **6 of 12** |

Sign agreement at chance, and biased 0.26 the wrong way: it reports "nothing
detected" where the primary sees a quarter of pairs vanish. The D row is the
mechanism — matched `vanished` 0.0, out-of-band effect **-0.5078**, because the
unnamed arm lost 51% of its pairs to out-of-band pattern diversity while the
named arm lost none. Subtracting two unequal inflations does not cancel them.
Rejected.

### Negative: the static schedule cannot be the reference for this work

The obvious escape from needing an unaffected PERIOD is an unaffected SCHEDULE —
the booked stop pattern, pre-registered and immune to recurrence. It is
contaminated at exactly the window that motivated the bug. Feed
`20260807-H-rockaways-extension-removed`, covering 2026-05-26 to 2026-10-31,
carries these path codes for the 7 on the window's service day:

`7..N27R, 7..N35R, 7..N36R, 7..N38R, 7..S95R, 7..S96R, 7..S97R, 7..S99R`

Zero 7X patterns. The supplemental feed has already absorbed the Express to
Local, so the schedule agrees with the disrupted movement by construction —
graded against itself, passing no matter how bad the model is. Meanwhile the
realtime feed still publishes 787 trips inside the window declaring `7X..N`, of
which 670 hops are consecutive local stops and 117 skip 3-4 stations. The
movement shows the work; the schedule cannot certify it.

### Negative, caught before it shipped: a countdown to a control we cannot certify

Having concluded the fix was a diagnostic — report how far the archive must reach
— the first version searched +/-60 days and named 2026-08-04 as the 7's nearest
free weekday, 8 days back. It is unbacked. `archive/windows/` holds **6
publication days** (2026-08-12 to 08-17), and an alert that ran and expired
before the first snapshot is absent from the record, not inactive. Absence of an
announcement is evidence of a free day only where we hold snapshots. Bounded to
certified days, every one of the 15 reads:

| | |
| --- | --- |
| `coverage_no_control_period` | 15 |
| `coverage_reach_unknown` | **15** |
| `answer_key_days` | 6 |
| weekday windows | certified 4, covered 4 |
| N weekend windows | certified **1**, covered 1 |

`certified == covered` everywhere: every comparable day the answer key can speak
for carries work on the route. The N's weekend windows have exactly one candidate
day in six days of coverage, which is the whole finding in one number.

### Three defects in the diagnostic itself, all found in review

All three would have certified a control that never existed.

* Coverage read as `first..last` rather than as a set. `load_windows` supports
  non-contiguous coverage, so a hole between the endpoints would have counted as
  evidence of free service.
* The window was excluded from its own blackout and only its START date skipped,
  so for the 4's 2026-04-27 to 2026-08-18 closure, 113 days of the closure would
  have certified as a control for itself.
* `_band_on` projected the clock band as `midnight + seconds`. On 2026-03-08,
  which springs forward, local 05:00 is not 18,000 seconds after midnight — the
  band landed at **06:00**, an hour off, twice a year. Both endpoints are now
  built with `datetime.combine`, the way `split_by_local_day` already builds day
  boundaries.

### Where this leaves the measure

Nothing about the grade changed and no published number moves: 13 graded, 22
duration-graded, medians unchanged. What changed is that `NO_CONTROL_PERIOD` no
longer reads the same whether a control is one week out of reach or cannot exist.
Both archives retain 3,650 days and began 2026-08-12, so the countdown starts
there and the windows grade themselves as coverage accumulates — the 7's own
announced record shows free weekdays from 2026-09-07, which the record will be in
a position to certify by then. The gradeable supply today is 13 of 28 over the
span, not 28, and not the headline 593.

## 2026-08-17 — the reach diagnostic read one band off a four-piece closure, and the tests passed because my multi-day case never crossed a service class

origin: artifact

The pre-commit gate blocked the entry above with a fourth defect in the same
diagnostic, after 839 tests and both linters had passed. It is the same mistake as
the other three, which is the part worth recording: the diagnostic answering a
slightly different question than the grade asks. I wrote that exact warning into
`_band_on`'s docstring and then made the error three more times.

### What it got wrong

`control_reach` read one clock band and one service class off the UNSPLIT window.
The grade does not: `pattern_shift` and `control_supply` both iterate
`split_by_local_day`, because a piece is the only thing carrying one band and one
class. A Friday-23:45-to-Monday-05:00 closure is four pieces on three timetables —
Friday late night, all Saturday, all Sunday, Monday morning — and its Saturday
piece needs a SATURDAY control.

Read unsplit, only Friday's weekday class is ever considered, so a free Saturday
in the record is not even a candidate. On that fixture:

| | day | lag | certified | covered |
| --- | --- | --- | --- | --- |
| unsplit (rejected) | **None** | None | 1 | 1 |
| per-piece (shipped) | 2026-08-08 | -7 | 3 | 2 |

It reports no reach where a control exists — the opposite direction from the
earlier 2026-08-04 defect, and just as wrong.

### Why the tests missed it

`test_a_multi_day_closure_blocks_its_own_later_dates` used Wed 23:45 to Fri 05:00.
Three pieces, all weekdays, all one service class — so it exercised splitting
without ever crossing the boundary that splitting exists for. A multi-day fixture
that stays inside one timetable is not a multi-day fixture. The replacement runs
Friday to Monday and fails against the unsplit logic.

### What moved in the published numbers

Nothing that matters, and one thing worth noting. Still 15 blocked, all
`certified == covered`, `coverage_reach_unknown` 15. The 4's 2026-04-27 to
2026-08-18 closure now reports **360** candidate (piece, day) pairs rather than 4,
because its 113 daily pieces are each searched against the six covered days — the
earlier 4 was the unsplit window's single band, which is precisely the bug. The
`Express to Local` duration median read 1.0118 then 1.0124 across the two runs at
identical n_rows=6; that is the live day accumulating twelve more minutes of
trace between them, not the change.

## 2026-08-17 — the reach blackout filtered routes when building the set and then ignored them when testing it, blocking the 2 for work on the 5

origin: artifact

A second gate round on the same diagnostic, and the fifth instance of the same
mistake. `control_reach` built its blackout from windows sharing a route with the
target, then tested only temporal overlap — so for a window naming two routes, work
on either one marked the day covered for both.

That is a false negative, and the grade disagrees: `coverage_state` calls a window
graded when ANY service pairs, so a window on the 2 and 5 against work running on
the 5 alone still has a free control arm on the 2. On that fixture:

| | day | lag | certified | covered |
| --- | --- | --- | --- | --- |
| routes ignored at the test (rejected) | **None** | None | 2 | 2 |
| per-route (shipped) | 2026-08-13 | +1 | 2 | 1 |

Not hypothetical: **13 of the 593** announced windows in the record name more than
one route, the 2/5 reroute among them. A day now counts as covered only when EVERY
route the window names is blacked out over the band.

No published number moves — 13 graded, 22 duration-graded, all 15 blocked windows
still `certified == covered`, coverage medians byte-identical to the run before the
diagnostic existed. The fix only ever changes multi-route windows, and none of the
15 currently blocked are.

## 2026-08-17 — the reach diagnostic rejected a whole band on any overlap while the grade only rejects an instant: five of six defects were the same disagreement

origin: self

Sixth defect in `control_reach`, found after the commit was already pushed. It
judged a candidate band covered if ANY blackout overlapped it. `_is_control` does
not: it tests one traversal's own timestamp against `Window.contains`, so work
occupying part of a band leaves the rest admissible.

Against a 12:00-16:00 band with a 12:00-13:00 closure on the candidate day:

| | day | lag | certified | covered |
| --- | --- | --- | --- | --- |
| band-level (rejected) | **None** | None | 2 | 2 |
| instant-level (shipped) | 2026-08-13 | -1 | 2 | 1 |

And the grade agrees with the second one, which is the part that settles it:
`control_supply` on the same blackout returns `{'J': 1}` for a traversal at 14:00
and `{}` for one at 12:30. The control arm is real; the diagnostic was denying it.

Fixed by sweeping the band and asking whether any instant escapes every blackout,
with spans sorted because blackouts can overlap each other. Both ends closed,
matching `Window.contains`, so a blocked span resumes at `hi + 1`.

### The pattern, which is the actual lesson

Six defects in one small diagnostic, and five were the same mistake: the
diagnostic answering a question slightly different from the one the grade asks.
One band instead of per-piece bands. Routes filtered when building the blackout
but ignored when testing it. A whole band instead of an instant. Each was a false
reading in production output, and none was caught by the tests I wrote for it —
the multi-day fixture stayed inside one service class, the route fixture named one
route, the overlap fixtures used exactly-aligned bands.

The guard that works is not another unit test of the diagnostic in isolation. It
is asserting the diagnostic against the MEASURE: the new test drives
`control_supply` over the same blackout and requires it to admit a traversal
exactly where reach claims a control exists. Any future divergence between the two
now fails a test rather than publishing a number.

### What moved

Nothing in the published numbers, again: 13 graded, 22 duration-graded, 15 blocked
all `certified == covered`, coverage medians byte-identical to the run before the
diagnostic existed. Today's recurring work occupies bands that align exactly with
the windows' own, so no candidate band was ever partially covered. The fix changes
readings only where a blackout overlaps part of a band, which the archive will
produce as soon as the announced work stops lining up this neatly.

Two existing expectations moved with it, both correctly. The Wed-23:45-to-Fri-05:00
fixture went from covered 7 to **5**: its Wednesday whole-day band has admissible
hours before 23:45, and the grade would take them. And the Friday-to-Monday
fixture needed its partial blocker widened to the full 00:00-05:00 band to still
isolate the Saturday, because a 02:00-03:00 closure no longer blocks 00:00-02:00.

## 2026-08-20 — recalibrating the movement disrupted arm to match assigned_n was rejected: the two feeds measure different things, and the trains that run are moving fine

origin: self

With the independent assigned_n truth wired into review (0pb), the movement-primary
published arm scored only 0.144 disrupted-recall against it, so the obvious next
move was to loosen the movement disrupted threshold (`DISRUPTED_RATIO`, currently
0.5·p0) and catch more. Ran the cross-tab first. It killed the idea.

Over 2026-08-15..08-20, against a causal advance baseline (2026-08-01..08-14), of
the 785 assigned_n-disrupted route-ticks:

| what the movement feed sees | ticks | share |
| --- | --- | --- |
| unjudgeable — too few matched trips / no baseline | 550 | 70% |
| no movement row at all | 23 | 3% |
| judgeable, and reads **normal** (advancing >0.5·p0) | 212 | 27% |
| judgeable, reads **disrupted** | **0** | 0% |

For the 212 judgeable cells the worst-direction posterior advance ratio is median
**0.75·p0**, min 0.53, and a quarter sit at or above **1.0·p0**. Not one dropped
below the disrupted floor. Lowering `DISRUPTED_RATIO` to 0.8 to catch them would
fire "disrupted" on routes advancing at three-quarters of a noisy baseline — well
inside normal variation — corrupting what the movement arm means.

The reading is that assigned_n collapse (the dispatcher pulled trips — a SUPPLY
cut) and movement-disrupted (the trains present are FROZEN — a FLOW problem) are
different axes of "degraded." A route routinely has one without the other, and the
70% unjudgeable is the fingerprint of the supply cut: pull the trips and movement
has too few left to judge. So the low cross-recall is correct, not a calibration
miss. 07h's premise — recalibrate to raise recall — is wrong; the follow-up is to
publish assigned_n as its own service-level axis rather than fold it into the
movement disrupted state.

## 2026-08-22 — the drawer's alert-HMM posterior is one-hot on 24 of 29 routes, and its regime clock resets on 14 of them in a single tick

origin: self

Reading the route drawer against the live feed to explain "P(suspended) 100.00%,
Regime age —" on the 2, the J and the 4. The section is honest about *which* arm it
shows (the alert-aware read, not the published movement condition), but two of its
four numbers are not usable as displayed.

In one snapshot (generated_at 1787432444), of the 29 routes carrying an inference:

| what the alert filter shows | routes |
| --- | --- |
| posterior numerically one-hot (max p > 0.999999) | 24 / 29 |
| `regime_age_seconds == 0` — argmax flipped on this very tick | 14 / 29 |

The posteriors are not near-certain, they are saturated: the 2 reads
`p_normal = 2.0e-44`, `p_disrupted = 4.1e-33`, `p_suspended = 1`. Six independent
likelihood channels multiply per tick (`logEmission`: Poisson count, four
Bernoullis, binomial movement, Gaussian service), so the log-odds run to hundreds
and the argmax is decided by whichever channel moved last. That is why half the
system changes regime in the same tick — and since `regime_entered_at` only
advances on an argmax change (`hmm.ts:192`), the drawer's "Regime age" is the age
of that flapping alert argmax, NOT the age of the movement regime the badge is
published from. Age "—" is `fmtMinutes(0)`: it flipped this tick.

Two display consequences worth separating from the model problem: `fmtMinutes`
rounded the hour remainder on its own, so 1379.7 minutes printed as "22h 60m"
(fixed); and for a movement regime in state `normal`, `movementRecovery` returns
`recovery_minutes = 0` by construction and `p_normal_in_30` is P(*stays* normal),
so the "Recovery forecast" block renders "Median —" against "P(normal in 30m) 93%"
— the label is wrong for normal routes, not the number.

Also confirmed the supply axis reads far above baseline routinely, not just below
it: same snapshot has the 2 at 217%, the J at 175%, the 1 at 169%, while the axis
only degrades below 50%. A bare "Normal / 217%" pair is unreadable without the
thresholds, which is what the new drawer meter marks.

## 2026-08-22 — a fixed ratio-vs-median threshold cannot mark "more trains than usual": p90/median spans 1.09x to 12.0x across cells, and MAD is degenerate in 14% of them

origin: agent

Marking the supply axis's over-baseline readings on the route cards needed a
threshold, and ">100% of the hourly median" was the obvious one. It fires on the
majority of the grid. Measured over the same 28-day window the live sidecar is
built from (2026-07-26..08-22, 8,065 archived trip-updates ticks, 193,017
(route, tick) samples, 1,129 cells at >=20 samples), for the MEDIAN cell:

| threshold vs own cell median | share of that cell's own history flagged |
| --- | --- |
| > 1.00x | 35.8% |
| > 1.10x | 16.7% |
| > 1.25x | 1.7% |
| > 1.50x | 0.0% |

Note it is 35.8% and not ~50%: `assigned_n` is a small integer count, so the
median sits ON a heavily-tied value — the median cell has 29.2% of its readings
exactly AT its median (max 100%). A median partitions the cell's historical
values with ties, and says nothing about what share of a cross-sectional
snapshot will exceed it. Claiming "half by construction" is wrong twice over.

**The reason no single multiple works is that the ratio is not comparable across
cells.** Measured p90/median: median 1.14x, p25 1.09x, p75 1.29x, max 12.00x
(FS we22 has median 1 train and p90 of 12). So 1.14x is an ordinary afternoon
for one cell and a once-a-month event for another. At the small end the ratio is
pure quantisation: GS/FS at median 1 train means one extra train is +100%, while
the F and 7 at rush (median 37-38) price the same extra train at +3% — a 33x
spread in the number for an identical physical event.

Robust z-score via MAD is not the fix either: MAD is exactly 0 in 161 of 1,129
cells (14%), IQR in 86 (8%), because a line that always runs the same count at
3am has no spread to normalise by. A modified z-score is undefined or infinite
for one cell in seven.

What this points at is per-cell empirical quantiles in the baseline sidecar
(p10/p50/p90 beside the median it already ships), scored as "above this cell's
own p90" rather than any global multiple — the same shape `compute_dwell_quantiles`
already uses for dwell. Not built yet; the card threshold moved to a global 1.25x
as a stopgap that at least does not fire on most of the grid (13 of 22 judgeable
routes at 1.00x, 4 of 30 at 1.25x).

Loose thread worth pulling: > 1.50x flagged 0.0% of the median cell's history in
this window, yet the live snapshot minutes later had the 1 at 170% and the 2 at
188% at we23. Either late-Saturday service is genuinely off the historical
distribution or the denominator for those cells is wrong.

## 2026-08-22 — the supply denominator is not soft: the 1 and 2 really did run weekday-level service on a Saturday. But the ratio cannot tell a rare day from a recurring second mode, and a per-cell percentile can

origin: agent

Chased the suspicion that the we23 baseline for the 1/2 was wrong, because they
read 170%/188% of their usual late-Saturday service. The denominator is fine and
the readings are real. What is wrong is the measure.

Per-ET-date median `assigned_n` across all hours, weekends only, over
2026-07-20..08-23 (9,935 archived ticks):

| route | 07-25 | 08-01 | 08-08 | 08-15 | 08-22 | weekday level |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 13 | 13 | 12 | 13 | **20** | 21-23 |
| 2 | 11 | 11 | 11 | 11 | **21** | 25-26 |
| C | 11 | 11 | 9 | 10 | 11 | 14-15 |

On Saturday 08-22 the 1 and 2 ran at nearly weekday levels all day while the C
ran an ordinary Saturday. No `Extra Service` alert was published for either line
— `assigned_n` is the only witness. So the axis worked: it caught a real service
anomaly the alert feed never announced.

**The finding that matters for the measure.** Those two readings look identical
as ratios and are nothing alike as percentiles:

| cell | n | median | p90 | p95 | observed | ratio | percentile | >1.25x median? | >p90? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 @ we23 | 120 | 11 | 13 | 17 | 18 | 164% | 96th | yes | yes |
| 2 @ we23 | 120 | 9 | 16 | 16 | 16 | 178% | 88th | yes | **no** |

The 2's cell is BIMODAL — value counts `{6:1, 7:5, 8:42, 9:42, 10:6, 14:8, 15:1,
16:15}`, the 14-16 mode occurring on 2 of 5 weekends. Sixteen trains is that
line's ordinary high mode, sitting exactly at its own p90, and the ratio calls it
a bigger deviation (178%) than the 1's genuinely once-a-month 18 (164%). A
median-plus-multiple cannot represent a two-mode cell at all; the percentile
separates them correctly with no tuning.

This is the concrete case for moving the axis off ratio-vs-median and onto
per-cell quantiles, beyond the cross-cell comparability argument recorded above.


## 2026-08-23 — the Models page reliability panel graded the movement forecast against the alert arm: same forecast, AUC 0.406 published vs 0.961 correct

origin: self

`p_normal_in_30min` is published only when `recovery_source` is `movement` or
`schedule` (`worker/src/snapshot.ts:878`) — it is a movement dwell-curve survival
probability. `calibrate()` took its outcome from `future.condition`
(`training/eval.py:670`), the alert-shadow regime. `published_arm()` had existed
for exactly this since the two arms diverged, and `_grade_recovery` already used
it; `calibrate()` was never migrated.

`build_eval` computed both blocks all along. `build_calibration` published only
the shadow one, dropping `calibration_movement` and `calibration_arm`, so the
page could never see the correct pairing and the feed did not even say which arm
it graded.

Live feed 2026-08-22 (`code_sha 1982fd6`), 30-minute horizon, one forecast:

| graded against | n | Brier | BSS vs clim | AUC |
| --- | --- | --- | --- | --- |
| `condition` (alert-shadow) — published | 35,078 | 0.0175 | -0.031 | **0.406** |
| `published_condition` (movement) — dropped | 34,740 | 0.0009 | -0.108 | **0.961** |

The tell was in the strata. Shadow `not_normal_now`: `mean_pred` 0.9939 against
`mean_outcome` 0.5049, and 35,017 of 35,078 samples in the single [0.9,1.0)
reliability bin. That reads as a degenerate constant forecast, but the forecast is
not constant — it is conditioned on the *movement* state while the stratum splits
on the *alert* condition, and the two are nearly independent. Sharpness of 0.99
against a realized 0.50 was cross-arm disagreement being reported as forecast
error.

**The correction is not "publish the better number".** The movement arm has
`not_normal_now` n=**4** and `unknown_share` 0.247, so its AUC 0.961 rests on 28
non-normal outcomes out of 34,740, and its BSS vs climatology (-0.108) is *worse*
than the shadow arm's (-0.031). Swapping 0.406 for 0.961 would have moved the
dishonesty rather than removed it. Both arms now ship with their label, sample
count, and coverage, and the panel draws both series.

Two things this did not fix, recorded so they are not mistaken for it:

- Horizons 60 and 120 grade n=0 permanently. Those fields are emitted only for
  `recovery_source === 'schedule'` (`worker/src/snapshot.ts:881-884`) and
  `calibrate()` skips every schedule row (`training/eval.py:653-655`). The gating
  was deliberate (journal 2026-08-12, BSS -0.00 to -1.30) but the page still drew
  two blank calibration plots, which read as "no skill" rather than "withheld".
  The panel now says which it is; whether the horizons should exist at all is open.
- Recovery interval coverage is 1.8% against a nominal 50%, and both arms lose to
  per-route climatology. Neither is a reporting artifact. The 99.1% normal base
  rate makes "always say normal" a genuinely hard bar, and with 608 not-normal
  samples on the shadow arm and 4 on the movement arm, this window cannot
  distinguish a good model from a bad one on the cases that matter. That is a
  collection problem, not a charting one.
## 2026-08-23 — one "Recovery forecast" heading was answering four different questions, and three of them wrongly

origin: self

Correction to the entry above, which called the regime-probabilities filter an
"alert-HMM" and its one-hot posterior "saturated, not confident". Both are wrong
framings. `logEmission` (hmm.ts:134-157) folds in the binomial movement and
Gaussian service channels whenever they are available, one tick lagged
(index.ts:361-378), so the filter is alert-PRIMARY, not alert-only. And one-hot
IS its confidence — the problem is that the confidence is miscalibrated and
cannot hedge, not that it is missing.

The drawer fix: the published numbers under one "Recovery forecast" heading mean
four different things, and the reader had no way to tell which. Mapping the live
snapshot's 29 inference-carrying routes onto the branches:

| what the numbers actually mean | routes |
| --- | --- |
| published normal — median 0/0/0 by construction, horizon is P(*stays* normal) | 19 |
| published unknown — we declined to judge, so every number is withheld | 7 |
| `not_scheduled`, `recovery_source: 'hmm'` — no announced resume to count down to | 3 |
| real time-to-normal estimate off the dwell curve | 0 |

The 19 normal routes were rendering "Median —" (that's `fmtMinutes(0)`) directly
above "P(normal in 30m) 93%", which reads as a broken forecast rather than "this
line is fine and 93% likely to stay that way". The 3 not_scheduled routes took the
withholding path, so a naive "the status came from train movement" message would
have been a lie about them — their condition comes from the schedule.

Also: `recovery_indeterminate` is two different silences. The movement arm sets it
when the regime outlived every measured dwell OR when the fitted curve's median
hits the 24h clamp (snapshot.ts:974), so copy asserting "beyond every one we've
measured" is wrong for the clamp case; the message now names the ceiling that
`recovery_minutes` already carries. The hmm arm sets it via the withholding gate
(snapshot.ts:855), which is not a bounding failure at all.


## 2026-08-23 — "collection problem" was wrong: 91 days are archived, eval grades 7, and a 7-day window's real support swings 17.8x

origin: self

An earlier entry today closed by calling the thin disrupted sample "a collection
problem, not a charting one." That was wrong on both halves, and the numbers say
so.

**There is no shortage of archive.** `v1/predictions/` holds 24,972 files across
91 days (2026-05-25..2026-08-23, 705,344 rows). `training/eval.py`'s `--days`
defaults to 7 with no upper bound, and `training/prune.py:61` retains 90. So the
published grading reads 7 of 90 available days, and widening it is an argument,
not a data-collection campaign.

**The per-tick n was never the sample size.** Six consecutive 7-day windows on
the published arm — tick counts indistinguishable, support wildly not:

| window start | tick-rows | episodes |
| --- | --- | --- |
| 2026-08-16 | 58,464 | 26 |
| 2026-08-09 | 58,551 | **89** |
| 2026-08-02 | 58,464 | 16 |
| 2026-07-26 | 58,464 | **5** |
| 2026-07-19 | 57,868 | 8 |
| 2026-07-12 | 56,280 | 8 |

A flat advertised n of ~58k hides a **17.8x** swing in independent evidence. Any
week-over-week metric move on this page was dominated by which week got sampled.
The current window's 26 incidents sit behind a displayed 52,287 — a ~2,000x
overstatement of independent samples.

**Negative result, with the number that killed it.** The first cut of
`episode_support` fed the raw window to `scorecard.model_episodes`, which resolves
the arm via `published_arm` — and that falls back to the alert-shadow `condition`
for rows predating 2026-07-11. Over the full archive it scores **1,781** episodes
against **153** on the published arm alone: a 12x inflation, entirely the shadow
arm's flapping, reported under a movement-arm label. Fixed by filtering to rows
carrying `published_condition`, counting the excluded legacy rows, and publishing
the span actually covered — so `--days 91` now reports 153 episodes over 43d
covered with 348,516 rows excluded, rather than a number that looks 12x better.

Also: the machinery already existed. `training/episodes.py` and
`training/scorecard.py` have segmented and graded episodes on the right arm for a
while; the counts just went to `docs/review/<date>-shadow-hmm/summary.json` and
never to `v1/eval.json` or `v1/calibration.json`. The page was tick-only because
nothing carried the episode count to it, not because it was hard to compute.

Standing constraint the fix cannot lift: `published_condition` is 43 days old, so
the movement arm's support grows with the calendar and caps at the 90-day
retention. 28d is the conservative widening (fully inside the published era);
beyond ~43d there is nothing yet to read.

## 2026-08-23 — the clamp fights the ceiling on every route because "disrupted" is 94% ordinary Delays, not because the corpus is thin

origin: self

Chasing why `_cap_self_loops` fires on 23/28 routes (normal), 28/28 (disrupted)
and 27/28 (suspended). Two hypotheses died on the way to the answer; both are
recorded with the number that killed them.

**Dead hypothesis 1 — "the 14-day training corpus is too thin."** The code comment
at `train_em.py:113` says EM on a thin or mostly-quiet corpus drives self-loops
toward 1.0, so widening the corpus should let real per-line dynamics emerge.
Patched `_cap_self_loops` to record the pre-clamp diagonal and refit both widths:

| corpus | normal over cap | disrupted | suspended | mean excess |
| --- | --- | --- | --- | --- |
| 14 days | 21/28 | 22/28 | 14/28 | +0.014 … +0.046 |
| 28 days | 23/28 | 18/28 | 19/28 | +0.016 … +0.048 |

Doubling the data changed nothing. EM wants stickier regimes at any corpus size,
so the clamp is not a thin-data artifact and widening the corpus buys nothing.

**Dead hypothesis 2 — "the ceiling is calibrated circularly."** `MAX_SELF_LOOP`
was set from median regime dwell in `v1/regime_transitions`, and that stream is
the filter's own argmax flips — `worker/src/grading.ts:11-13` literally calls it
"ground-truth dwell times." A loop on paper. But the numbers refuse to close it:
the cap implies a **48 min** median dwell and the filter's measured argmax dwell
is **15 min** (median, n=231 over 28d). The cap is ~3x *looser* than the
behaviour it supposedly came from, so it is not enforcing its own justification.

**What is actually true.** Of the 1,086 route-ticks where the filter says
not-normal, split by the severity of the alert behind them:

| tier | meaning | ticks | share |
| --- | --- | --- | --- |
| 3 | suspension | 2 | 0.2% |
| 2 | severe delays | 60 | 5.5% |
| 1 | ordinary `Delays` | **1,024** | **94.3%** |
| 0 | planned work | 0 | correctly excluded |

The latent "disrupted" regime is routine `Delays`, which the MTA posts constantly.
Planned work is properly excluded, so that earlier suspicion was wrong too. And
the duration distributions explain the EM fight exactly:

| population | n | median | mean | mean/median |
| --- | --- | --- | --- | --- |
| raw alert presence, any tier | 1,692 | 55 min | 284 min | 5.2x |
| raw alert presence, tier>=2 | 45 | 50 min | 56 min | **1.1x** |
| filter argmax not-normal | 231 | 15 min | 24 min | 1.6x |
| movement arm | 136 | 5 min | 11 min | 2.2x |

Severe events are nearly symmetric and perfectly representable by a geometric
self-loop. The heavy tail is *entirely* tier<2. A single self-loop fitted across
both populations splits the difference — EM lands on 0.965 (~97 min) and the cap
clips to 0.930 (~48 min), which happens to sit near the tier>=2 median of 50 min.
The cap is accidentally about right for real disruptions, which is probably why
this went unnoticed.

The missing input is severity, and it is missing by construction:
`training/load.py:36-38` — "Counted (HMM-included) alert_types active on this
route-tick. Only the R2 truth builder populates it — used by the review to grade
ground truth by severity; **the HMM training path leaves it empty**." Grading is
severity-aware (`CANONICAL_SEVERITY_FLOOR`, truth_version 2 = severe-only); the
model is not. It predicts when the `Delays` banner clears and is scored on when
the severe event ends.

That single mismatch is the best available explanation for the two worst numbers
on the page — recovery MAE 74 min and IQR coverage 1.8% against a nominal 50% —
since both are "how long will this last" inheriting a dwell learned from a
population 94% composed of the wrong events.

## 2026-08-23 — the movement binomial is the only emission channel that scales with fleet size: 0.192 nats per matched trip under bootstrap params, while 99.83% of 35,492 graded forecasts sit in a single bin

origin: agent

The 2026-08-22 entry above established that the posterior is one-hot on 24 of 29
routes and named the cause as "six independent likelihood channels multiply per
tick, driving log-odds to hundreds". It did not say which channel. Two separable
things below: a structural fact about the channels, and a measurement of the
output. The bridge between them is NOT established — see the limits at the end.

**Structural.** In `_log_emission` (`src/momentarily/hmm.py:366-400`, mirrored in
`worker/src/hmm.ts:134-158`), six of the seven channels evaluate one scalar or
one flag per tick, so their log-likelihood ratio between states is a constant
that does not depend on how much data the tick contained. The binomial movement
channel is the exception: `_log_binomial(advanced_n, matched_n, rate)` treats the
cross-matched trips as `matched_n` independent Bernoulli trials of the route's
hidden state, so its LLR is `matched_n * KL(rate_normal || rate_disrupted)` —
linear in the trip count. This holds for any parameter values; only the
coefficient changes. `matched_n` is `advanced_n + stalled_n` over both directions
(`worker/src/movement_state.ts:380-402`, `worker/src/vehicles.ts:116-190`), with
`MIN_MATCHED_TRIPS = 3` as a floor and **no cap**.

**Sized with the checked-in bootstrap emissions only**
(`training/run_filter.py:34-52` plus the `EmissionParams` defaults at
`src/momentarily/hmm.py:131-136`), on an observation with no alerts,
`service_ratio` 1.0, and `advanced_n` at exactly `0.6 * matched_n` — the normal
state's own advance rate, so there is no anomaly anywhere in the input. LLR
normal vs disrupted, nats:

| channel | n=3 | n=20 | n=40 | n=80 | n=160 |
| --- | --- | --- | --- | --- | --- |
| Poisson `alert_count` | 3.70 | 3.70 | 3.70 | 3.70 | 3.70 |
| Bernoulli `has_delays` | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 |
| Bernoulli `has_service_change` | 0.90 | 0.90 | 0.90 | 0.90 | 0.90 |
| Bernoulli `has_planned` | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 |
| Bernoulli `has_suspended_alert` | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| Gaussian `service_ratio` | 0.89 | 0.89 | 0.89 | 0.89 | 0.89 |
| **Binomial movement** | **0.83** | **3.84** | **7.68** | **15.36** | **30.73** |
| total | 8.12 | 11.14 | 14.98 | 22.66 | 38.02 |
| movement share of positive LLR | 10% | 35% | 51% | 68% | 81% |

Per-trip discrimination here is `KL(0.6 || 0.3) = 0.192` nats, contributed
whether or not anything is wrong, and the crossover where movement carries the
majority of the evidence is around `matched_n = 40`. For contrast the same
observation with trips advancing at the disrupted rate (`0.3 * matched_n`,
n=160) gives **-22.11** nats: under these params the channel is a hard
classifier, not a nudge. A settled prior mixed through the bootstrap transition
matrix predicts `(0.95, 0.04, 0.01)`, worth `ln(0.95/0.04) = 3.17` nats of pull
toward staying put, which 38 nats of emission would beat 12:1 — the shape that
would produce one-hot posteriors and single-tick argmax flips together rather
than as two symptoms.

**Measured, independent of the above.** Over the 35,492 graded 30-minute
forecasts in the public `v1/eval.json` (window ending 2026-08-23):

| | alert-shadow arm | movement arm |
| --- | --- | --- |
| forecasts in the 0.9–1.0 bin | 35,432 / 35,492 (**99.83%**) | 35,097 / 35,157 |
| forecasts in 0.8–0.9 | 60 | 60 |
| forecasts below 0.8 | **0** | **0** |
| AUC | 0.375 | 0.361 |
| BSS vs climatology | -0.033 | -0.534 |

And the one subset where the answer should not be "fine", the 552 ticks whose
line is not normal *now*: mean forecast **0.9963**, mean outcome **0.5272**. The
model says 99.6% where the truth is 53%.

Both arms saturate identically, on the same 60 mid-bin predictions, so this is
not the grading mismatch recorded on 2026-08-23 above. Checked deliberately,
because that entry's lesson was that a panel can grade the wrong arm and
manufacture an AUC.

**Limits — what this does not establish.** The live fitted emissions were never
read: `PARAMS_KEY = 'state/params.json'` (`worker/src/params.ts:15`) sits outside
the public `v1/` prefix (`worker/src/index.ts:96`), and no fitted params are
checked into the repo. Raw per-tick `Observation` values are not persisted either
— `PredictionRecord` keeps `p_normal` and friends but drops `alert_count`,
`matched_n`, `advanced_n` and `service_ratio` (`worker/src/grading.ts:28-90`). So
the 0.192 nats/trip, the 81% share, and the `matched_n = 40` crossover are
properties of the bootstrap parameters, not measurements of production, and EM
refits `advance_rate` per route and tod_bin (`src/momentarily/hmm.py:658-832`).
The live runner-up masses are in the right family — route 1 publishes
`p_disrupted = 2.6864e-17`, and `-ln` of that is 38.16 nats — but attributing
those nats across channels needs the fitted params and a real `matched_n`, and
inverting one number into the other assumes exactly the equality that is
untested. Two things would settle it: log the per-channel log-likelihood terms
for one tick from inside the Worker, or persist `matched_n` and `advanced_n` on
`PredictionRecord`.

Same error class as the `severity_sum` Gamma channel removed earlier
(`hmm.py:128-133`, "double-counted the count evidence and saturated the
posterior") — but where that double-counted a bounded quantity, this multiplies
by an unbounded trip count. No fix applied. The direction suggested by the
structure is to score the advance *rate* once per tick rather than once per
trip, which under bootstrap params would put the channel at 0.192 nats instead
of 30.73 at n=160; untested, and it would need the EM fit redone.

## 2026-08-24 — the recovery grade was measuring an arm mismatch, not the model: MAE 102 → 29 min, IQR coverage 8.7% → 46.9%

origin: agent

Built the per-retrain segmentation the widened eval window needed
(`recovery_by_params_version` in `training/eval.py`, published in `v1/eval.json`
and trimmed into `v1/calibration.json`), and it immediately overturned the
standing causal story for the two worst numbers on the Models page.

**What the segmentation shows.** 28 days, 2026-07-28..08-24, 226,432 predictions,
728 transitions. MAE/IQR are per-tick; `incid` is the incident count on the arm
being graded, which is the number that says how much independent evidence a
segment carries:

| params trained | ticks | incid | MAE | IQR coverage | span |
| --- | --- | --- | --- | --- | --- |
| 2026-07-26 | 60 | 13 | 30.1m | 38.3% | 07-28..08-02 |
| 2026-08-02 | 14 | 2 | 17.9m | 78.6% | 08-02..08-03 |
| 2026-08-03 | 0 | 0 | — | — | 08-03..08-03 |
| 2026-08-03 | 89 | **33** | **32.5m** | **39.3%** | 08-03..08-11 |
| 2026-08-11 | 16 | 2 | 99.0m | 6.2% | 08-11..08-12 |
| 2026-08-12 | 12 | 12 | 11.7m | 0.0% | 08-12..08-13 |
| 2026-08-13 | 884 | **197** | **116.8m** | **2.6%** | 08-13..08-24 |
| pooled | 1,075 | 259 | 102.3m | 8.7% | whole window |

Two segments clear a 20-incident floor and they disagree by 3.6x on MAE and 15x
on coverage. The pooled number sits between them and flatters the model now
running.

**Dead hypothesis — "the 2026-08-13 retrain made the filter flap, and a
dwell-based estimate cannot hit short regimes."** Regime durations moved the
wrong way: median non-normal dwell was 5.0 min before 08-13 (n=78) and 10.0 min
after (n=286), mean 18.3 → 36.9 min. Regimes got *longer*, so flapping is not
what broke the estimate.

**What is actually true.** The predicted quantity collapsed, not the truth. Over
the gradeable population (non-normal on the shadow arm, determinate), split by
the arm that produced `recovery_minutes`:

| recovery_source | n | median predicted | median IQR width | zero-width |
| --- | --- | --- | --- | --- |
| `hmm` | 192 | 50.0m | 80.0m | 1% |
| `movement` | 883 | **0.0m** | **0.0m** | **99%** |

`movement` became the dominant source at the 08-11 retrain and carried 857 of
884 rows by 08-13. Those rows read zero for a good reason: **870 of the 872
movement-sourced zeros (100%) sit on routes whose `published_condition` is
`normal`**, while the shadow condition that selected them says `disrupted` (515)
or `suspended` (357).

So the movement arm is right. It says "this route is running normally, there is
nothing to recover from" and emits 0 — `worker/src/snapshot.ts:963-967`,
`1029-1033`. The grader picks the row because a *different* arm calls the route
disrupted, then reads that 0 as a forecast of instant recovery against the
shadow's regime clock. `_grade_recovery` gates ticks on `arm(p)` but grades
`recovery_minutes`, which comes from whichever arm `recovery_source` names, and
the two are not the same arm. The guard for exactly this failure is already in
the code and already describes it (`training/eval.py:861-867`: "A route that is
already normal isn't recovering — it predicts recovery_minutes=0, and grading
that against time-until-the-next-disruption swamps MAE and pins IQR coverage
near zero"); it just tests the shadow stream, which is not the stream the zero
came from.

Excluding movement-sourced rows the way `schedule` rows are already excluded
(`training/eval.py:876`):

| | ticks | incidents | MAE | IQR coverage |
| --- | --- | --- | --- | --- |
| as published | 1,075 | 259 | 102.3m | 8.7% |
| movement rows excluded | 192 | 52 | **29.0m** | **46.9%** |

Against a nominal 50%, the recovery forecast is close to calibrated. **Not
applied.** It changes published grading semantics and makes the headline numbers
look 3.5x better, which is precisely the kind of change that needs sign-off
rather than an agent's initiative.

**What this does and does not overturn.** The standing diagnosis held that MAE
and IQR coverage were severity conflation — a dwell inherited from a population
94% composed of ordinary `Delays`. That conflation is real and separately
verified: the training path carries no severity input at all. What is now
measured is that it does not *explain these two metrics*. Across the 08-13
boundary the severity mix is flat while the recovery source inverts:

| | tier>=2 share | tier 1 share | `hmm` | `movement` |
| --- | --- | --- | --- | --- |
| pre 08-13 (n=182) | 3.3% | 96.7% | 89.6% | 10.4% |
| post 08-13 (n=893) | 6.3% | 93.7% | 3.2% | 96.8% |

Same conflation on both sides — marginally *more* severe events afterward, which
under the conflation hypothesis should have improved the numbers rather than
degrading coverage from 39.3% to 2.6%. The 2026-08-03 params scored MAE 32.5 min
and 39.3% coverage over 33 incidents under exactly this mix. So the conflation
stays on the books as a modeling problem worth fixing on its own merits, and is
struck as the cause of these two numbers. A three-state severity split was
queued against an acceptance criterion requiring MAE and IQR coverage to move
materially; on this evidence it could not have moved them, and would have been
recorded as a failed diagnosis for the wrong reason.

**Instrument note, learned twice on the way.** A segment's `low_sample` flag
counts incidents, not ticks: the regression test's v200 has more graded ticks
than v100 (50 vs 44) and one incident against 22, so a tick threshold waves
through exactly the thin segment the flag exists to catch. And it counts them
via `per_regime.n` — regimes on the arm being graded — not via `episode_support`,
whose incident count is computed on the published movement arm and therefore
sizes a different metric. Reaching for the adjacent number would have rebuilt
the same arm-mismatch error class one field over, inside the very instrument
built to find it.
## 2026-08-24 — the crowding estimate's unit was wrong by 10x, and the version people actually want doesn't need the ridership feed at all

origin: self

Building the hourly-ridership ingest and the platform-crowding estimate off
it. The design as written was "entry rate x time since last train, published
as '~3 trains' worth waiting'". Both halves of that survived
contact with the data; the copy did not.

Replayed the real trace for 2026-08-20 07:00-11:00 ET (240 minutes, 982
directional platforms) against per-complex hourly entry rates aggregated from
data.ny.gov 5wq4-mkjj over a 4-week window, splitting each complex's rate
evenly across the platforms observed in service. 28,670 observed platform gaps:
p50 6.0 min, p90 13, p99 31, max 232. Implied riders waiting, with gaps over 30
minutes abstained: **p50 28, p90 86, p99 270, max 1302**. A ten-car train holds
roughly 1,000 riders, so p99 is 0.27 of a train and 0.018% of published
estimates reach one train load at all. "~3 trains' worth" is not a rounding
difference from "~28 people" — it is the wrong unit by an order of magnitude.
The surface publishes riders.

The cap is load-bearing, and the reason is the same degenerate-baseline
failure class the movement classifier's terminal over-flagging already names. Uncapped, the top implied-crowding platforms in the window are
G21N Queens Plaza (median gap 232 min, 3,142 riders), R31N Atlantic Av (231
min, 2,049) and A24N 59 St-Columbus Circle (208 min, 2,037) — terminal and
relay platforms whose "gap" is an artifact of not being a through-service
platform in that direction, not a crowd. A 30-minute cap abstains 296 of 28,670
gaps (1.04%) and takes the maximum estimate from 3,142 to 1,302.

The finding that changes what to build next: **expressing crowding relative to
the same platform's own usual crowd cancels the ridership baseline exactly.**
crowd_now / crowd_usual = (rate x minutes_since) / (rate x usual_headway) =
minutes_since / usual_headway. The rate divides out. So the "how bad is it right
now vs usual" gauge needs a scheduled-headway baseline (not implemented) and
needs nothing at all from the ridership feed. The ridership ingest
earns its keep on exactly two things the ratio cannot do: the absolute
headcount, and ranking platforms against each other at one instant. Worth
knowing before anyone wires the ridership artifact into a percentile gauge.

Third thing, a collection footgun. Deriving "a train cleared this platform"
from `stopped === true` sightings alone misses **7.0% of station departures**
(1,078 of 15,382 measured over a 3-hour window): NYCT dwells are frequently
shorter than the 1-minute poll, so a train can arrive and leave between two
snapshots and never be seen stopped. A stopped-only rule therefore reports
last_train_at one full headway stale on 7% of platforms, which roughly doubles
the estimate there. `crowding.ts` carries its own trip->stop map and counts a
departure when a trip's stop_id changes, which by construction catches every
observed departure. That carry is deliberately NOT vehicle_stops.json — see the
step-0 hazard comment in index.ts; it lives in state/station_wait.json, is
updated with `max`, and is therefore idempotent under a retried cron minute.

Two things the estimate structurally cannot see, both published in
`platform_crowding.method.excludes` rather than left as folklore: free
in-system transfers are never fare-swiped, so a transfer complex undercounts;
and the hourly feed has no exits column at all (the legacy 4-hour turnstile
data that had one was retired end-2022), so a platform crushed by an arriving
train letting out reads as empty.

Postscript, because the departure rule got challenged in review and the
challenge was reasonable-sounding. The objection was that stamping a platform
on every `stopped === true` sighting measures time since ARRIVAL, and that a
departure should instead be counted only when the immediately preceding
observation had the trip stopped there. Both halves are wrong, and the second
is wrong in a measurable way. Same window, 2026-08-20 07:00-11:00 ET, 47,936
stop transitions: only **44,441 (92.71%)** had the trip caught STOPPED_AT on
the preceding observation, so that rule loses **3,495 departures** — the same
7% class as above, arrived at independently. And the stamp cannot conflate
arrival with departure, because the stop-change branch writes the departure
poll, which is always at or after any earlier stopped sighting at that
platform, so `max()` resolves to the departure whenever one is observed; the
stopped branch only governs the interval while a train is physically standing
there, where zero waiting is the reading we want. The one ordering that could
strand a departure — same stop_id going stopped -> moving — occurs 79 times in
those 47,936 transitions (0.16%) and only defers detection by one poll.


## 2026-08-24 — running the finished crowding path end to end put two feed-stall artifacts at the top of the whole system, and exposed that the even split is wrong by 3-5x at the two busiest complexes every tick

origin: artifact

Ran the finished path end to end: the real ridership artifact (425 complexes,
90-day window) through the Worker's zod validator, against a platform-wait doc
replayed from the real 1-minute trace for 2026-08-20 08:59 ET. It works — 934
platforms estimated, 48 abstained (unknown_stop 7, no_baseline 40,
gap_exceeds_cap 1).

Instantaneous distribution of waiting_riders: **p50 0, p90 22, p99 59**. The
p50 of 0 is not a bug and is worth internalising: sampled at an INSTANT rather
than per gap, roughly a third of platforms have a train standing at them or
just departed, so all the content of this surface is in its tail. The earlier
per-gap p50 of 28 was the crowd just BEFORE each train arrives — the cycle
peak, about twice the time-average. Two different questions, 28x apart at the
median.

The tail is where it fell over. The top two readings in the entire system were
**901S Grand Central 951 riders** and **902N Times Sq 665 riders** — the 42 St
Shuttle platforms — against a third place of 87. A 10x jump to the rest of the
distribution is a mechanism, not a crowd, and it was two mechanisms stacked.

**One: the vehicle feed stalls, and the cap does not catch it.** Over the same
two hours the trace carries **70 rows for route GS** against 2,793 for the 1
and 7,107 for the F, and 32 of those 70 are a single vehicle reported in
transit to 902N without ever arriving. So those gaps (27.0 and exactly 30.0
minutes) are a stuck report, not an empty platform — the same vehicle-stall
defect already logged against that feed. Both artifacts sat just INSIDE the
30-minute cap, which is the useful part: a cap calibrated on the real gap
distribution does not protect you from a defect that clusters just under it.
Fixed, and without a new threshold: a trip reported as heading to a platform
whose stop_id has not advanced for longer than the cap is a stalled report, so
the platform abstains with reason `stalled_inbound`. The rule is restricted to
trips that are NOT stopped, because a train standing at a terminal for 40
minutes also has an unadvancing stop_id but re-stamps its platform every minute
and correctly reads ~0 waiting. The cap boundary also went inclusive; 902N sat
at exactly 30.0 and passed a `>` test.

**Two: the even split is not merely uncertain, it is biased, in the same
direction, forever, at the highest-traffic stations.** Complex 610 (Grand
Central, ridership rank 2) contains three parent stops — 631 (4/5/6), 723 (7)
and 901 (the shuttle) — so six directional platforms, five in service at that
instant, and the shuttle platform therefore receives **one fifth of the entire
complex's entry demand**, 35.2 riders/min. Complex 611 (Times Sq, rank 1)
splits five ways and hands its shuttle platform 22.2/min. Nobody entering
Grand Central at 09:00 boards the shuttle in anything like that proportion;
this is off by roughly 3-5x, and unlike noise it is off by that amount the same
way on every tick, at the two most-viewed stations in the system.

No fix applied to the split, deliberately — `split_basis` is published in
`platform_crowding.method` precisely so it can be replaced, and the replacement
needs a criterion I have not measured. The cheap defensible successor is to
weight the split by observed train volume per platform (supply, which the trace
already sees) rather than by platform count, since demand tracks service far
better than it tracks platform arithmetic. The number to chase first is the
ratio between a complex's per-platform train volume and its per-platform entry
share, on any complex where both are observable. Until then the surface is
labelled an estimate and publishes the assumption, which is the honest state
but not a good one.


## 2026-08-24 — correction: the stall gate in the entry above was rejected on measurement. It abstained 23 platforms with live service and never fired on the case it was written for

origin: self

The previous entry claimed the feed-stall artifact was "Fixed, and without a
new threshold" by abstaining platforms with a trip reported inbound whose
stop_id had not advanced past the cap. Built it, then folded the real
120-minute trace (2026-08-20 07:00-09:00 ET, 89,352 rows) through the real
per-minute update and graded it. It does not hold, in both directions at once:

- It abstained **23 platforms**, and of the platforms it gated, **25 had their
  own gap under the 30-minute cap and 24 of those were under 10 minutes**. A
  vehicle stuck inbound on one route does not invalidate a departure another
  route genuinely made two minutes ago, and on a shared platform that is the
  common case, not the corner.
- It never fired on **901S**, the reading that motivated it. The trip inbound
  there reports `since` = 0.0 minutes: the shuttle's trip_ids churn and its
  stop_id alternates, so the elapsed clock resets before it can ever cross the
  cap. The gate is blind to exactly the feed pathology it was aimed at,
  because that pathology is churn rather than a clean freeze.

Reverted, including the `{stop, since, moving}` carry that existed only to feed
it — 3x the state for a dead rule. The note above CROWDING_MAX_GAP_MINUTES now
records the rejection so nobody reaches for it again.

What survived from that attempt is one line: the cap boundary is inclusive.
902N sat at exactly 30.0 minutes and passed a `>` test; `gap_exceeds_cap` went
from 1 to 2 and that reading is gone.

So the state after all this: 933 platforms estimated, abstained {unknown_stop
7, no_baseline 40, gap_exceeds_cap 2}, waiting_riders p50 0 / p90 22 / p99 58,
and **901S Grand Central still publishes 951 riders**. That number is not a
code defect, it is the uniform complex-to-platform split's worst case at the
system's second-busiest complex, and it stands until the split basis changes.
The lesson worth keeping is the shape of the mistake: I reached for an
observation-quality gate because the number looked wrong, when the number was
wrong for a demand-allocation reason the gate could not touch. A gate that
cannot articulate which of the two it is testing will grade badly on both.


## 2026-08-24 — exits really are gone, but the origin-destination estimate hands us the thing we actually needed: departing riders split by direction, and the even split is off by 8x at a terminal

origin: self

Asked whether exits exist and whether modelling them would improve the platform
crowding estimate. Searched the whole data.ny.gov catalog rather than guessing.

**Exits: no, and confirmed dead.** Cumulative ENTRIES/EXITS registers exist only
in the legacy per-turnstile, 4-hour `Turnstile Usage Data` sets, which stop at
2022 and are flagged static. No current dataset exposes exits, egress or
alighting under any name. The hourly feed we ingest is entry-side by
construction.

**But modelling it is possible, via `28vm-gjqr` (MTA Subway Origin-Destination
Ridership Estimate: Beginning 2026).** 72.6M rows, coverage 2026-01-05 to
2026-07-12, so a **43-day lag** against the hourly feed's 11. One row is
(origin complex, destination complex, month, day_of_week, hour_of_day) ->
`estimated_average_ridership`. Origin complex ids run 1..636 with **425
distinct** — exactly the 425 our ridership baseline carries, so it joins with no
crosswalk. Server-side SoQL aggregation works the same way.

Summing over origins for a destination gives estimated arrivals. Wednesday
08:00, system total: departures 434,861 = arrivals 434,861 (it is an OD matrix,
so this is conserved by construction, and it also proves the arrivals side is
NOT shifted to actual arrival time — see the caveat below). Per complex, the
arrivals/entries ratio separates residential origins from job destinations
cleanly: **Astoria-Ditmars 0.13, Atlantic Av 0.61, Times Sq 1.20, W 4 St 1.50,
Grand Central 1.77, Union Sq 2.54.**

**The finding that matters is not the exits, though — it is the direction
split.** Bucketing each origin's destinations by whether they lie north or
south of it, Wednesday 08:00:

    Astoria-Ditmars    5.8% north / 94.2% south
    W 4 St-Wash Sq    79.1% north / 20.9% south
    14 St-Union Sq    69.7% north / 30.3% south
    Times Sq-42 St    40.6% north / 59.4% south
    Grand Central     30.7% north / 69.3% south

We currently assume 50/50. At Astoria-Ditmars at rush hour that overstates the
northbound platform's crowd by **8.6x** and understates the southbound one by
1.9x. This is demand-side evidence, derived from where riders actually went,
and it is what the split-basis work was missing: not platform-level ground
truth, but direction-level demand truth, which is most of the gap. It also
gives the previously-absent acceptance criterion — a supply-weighted split can
now be graded against an independent demand-side estimate instead of against
nothing.

Three caveats, all from MTA's own documentation of the method:

- It is an estimate on an estimate. Destination is INFERRED as the station
  where the rider next taps; only ~80% of trips link that way and the remaining
  20% are allocated using the distribution of the linked ones. A fare-evasion
  scaling factor is applied on top.
- `timestamp` is the hour of the ENTRY tap, rounded down — not the hour of
  arrival. So "arrivals at D in hour H" really means "trips entering anywhere in
  hour H that end at D", displaced earlier by the ride time. The identical
  departures/arrivals totals above confirm no arrival-time shift is applied.
  At hourly granularity that matters most on the rush shoulders.
- Complexes hide platforms. Direction comes out; LINE does not, because MTA
  runs a path-assignment model (shortest perceived time, with transfer and
  crowding penalties) and then drops the paths before publishing. Several lines
  serving the same destination stay ambiguous.

**And the reason exits would not have helped anyway, which is worth stating
plainly:** riders who alight do not wait. They walk out or transfer, so
alightings do not accumulate on a platform the way boarding demand does, and
adding them to a "who is waiting" count would be wrong. What alightings would
buy is a transient unloading pulse measured in seconds, not the accumulating
queue this surface estimates. The real entry-side blind spot is IN-SYSTEM
TRANSFERS, which are never fare-swiped and are a large share of platform
occupancy at exactly the complexes where our split is worst — and the OD matrix
cannot close that either, because "changed trains at Z" is precisely the path
information MTA discards. Closing it needs our own routing model over the
topology.


## 2026-08-24 — for station VOLUME the origin-destination dataset adds nothing: over a month every station's arrivals/entries collapses to 1.0, the ranking moves a median of 1 place, and total volume is exactly 2.0x entries

origin: artifact

Follow-up to the entry above, which found arrivals/entries ratios spread from
0.13 to 2.54 across complexes and treated that as a reason to ingest the
origin-destination estimate. Checked whether it survives aggregation, because
"station volume" is a monthly-scale statistic, not an hourly one. It does not.

Ranked all 425 complexes by entries alone, then by volume (entries + arrivals),
over June, all hours, all days:

    rank shift: median 1 place, p90 4, max 36
    complexes moving more than 20 places: 1 of 425
    system-wide volume / entries: 2.000

The one real mover is Howard Beach-JFK (entries #302 -> volume #338, arr/ent
0.59) with Far Rockaway-Mott Av next (-11, 0.88) — genuinely asymmetric
stations where riders arrive by some other mode and leave by subway. Everything
else is noise.

The reason is obvious in hindsight and worth writing down so nobody re-derives
it: over a long enough window almost everyone who enters a station also comes
back to it, so entries and arrivals converge station by station. The same
complexes that looked wildly asymmetric at one rush hour are flat over a month:

    station              Wed 08:00    June, all hours
    Astoria-Ditmars           0.13               1.00
    14 St-Union Sq            2.54               1.07
    Grand Central             1.77               0.97
    Times Sq                  1.20               0.97
    W 4 St-Wash Sq            1.50               1.05
    Atlantic Av               0.61               1.00

So arrivals change WHEN a station is busy, not HOW busy it is. For a volume or
ridership-rank metric, the hourly entries feed we already ingest is sufficient
and strictly better operationally: 11-day lag against 43, and ~10k aggregated
rows per query against a 72.6M-row table. The `rank` field already published in
`state/ridership_baseline.json` is the right instrument and needs no second
source.

This does not retract the direction finding in the previous entry. That one is
an hourly, within-complex allocation question, which is exactly the regime where
the asymmetry is real (0.13 vs 2.54) — and it is measured on the departure side
regardless. Volume is the case where aggregation destroys the signal; the
platform split is the case where it does not.


## 2026-08-24 — the 49 crowding abstentions are almost all correct: 38 are SIR stations that collect no fares, 7 are yard trackage in no station reference and not in GTFS at all, and only 2 complexes were ours to recover

origin: self

Diagnosed every abstention from the real replay rather than assuming they were
coverage gaps. Breakdown of the 49: `no_baseline` 40, `unknown_stop` 7,
`gap_exceeds_cap` 2.

**`no_baseline` 40 = 20 Staten Island Railway complexes, and only 2 were a bug
of ours.** SIR trains ARE in our trace (route 'SI', 1,307 rows in a two-hour
window) while the ingest filtered `transit_mode='subway'`, so we were tracking
SIR trains and refusing to estimate their platforms. Widened to
`transit_mode IN ('subway','staten_island_railway')`. It recovers exactly TWO
complexes, and that is not the filter's fault: **SIR collects fares at St George
and Tompkinsville only**, so those are the sole SIR complexes with any rows in
the dataset at all (July 2026: St George 249,980 entries, Tompkinsville 20,481,
nothing anywhere else). St George lands at system rank 175 with a 16.23/min
weekday peak, which is a real station we were silently dropping. The other 19
SIR complexes have no entry data in any published dataset and keep abstaining.
Measured end to end: `no_baseline` 40 -> 38, platforms estimated 933 -> 935.
Roosevelt Island tram deliberately still excluded — its complex ids are 'TRAM1'
/ 'TRAM2', which join to no GTFS stop, and it is not in the subway trace.

**`unknown_stop` 7 = non-revenue trackage, and it took the canonical feed to
prove it.** The ids are F10S, H19N/S, R60N/S, R65N/S. None appears in
39hk-dx4f, none in the newer stations-and-complexes dataset, and — decisively —
none in GTFS `stops.txt` (1,488 stops) or `stop_times.txt` (989 scheduled ids).
A review challenge asserted F10 was Jamaica-179 St; it is not. Canonical GTFS
says **Jamaica-179 St is F01**, the F series runs F01-F07, F09, F11, F12 with no
F08 or F10, the H series ends at **H15 Rockaway Park-Beach 116 St** so there is
no H19, and there are no three-digit R ids anywhere near 60-65. The occupancy
confirms what they are: **H19N holds up to 10 trains stopped simultaneously and
is present 119 of 120 minutes**, sitting right past the Rockaway Park terminal.
That is a layup yard. Worth recording that the weak version of this argument
(observed occupancy) only settled H19 and R60S — the other five showed a single
train for a handful of minutes, which proves nothing. Absence from GTFS
`stop_times` is the test that actually settles it.

No code change for those seven. They already fail to resolve to a complex, so
they were never in the served-platform denominator and cannot dilute anyone's
share; the abstention is bookkeeping, not contamination.

**`gap_exceeds_cap` 2**, both correct: 902N at exactly 30.0 minutes (the shuttle
feed-stall artifact) and H11S Far Rockaway-Mott Av at 90 minutes (a genuinely
sparse terminal).

So the honest total: of 49 abstentions, 2 were recoverable and are recovered,
45 are the absence of data that does not exist, and 2 are the cap doing its job.
The lesson for next time is the order of the checks: I reached for "which
station reference is stale" first, when the question was answerable from the
schedule feed we already download every night.


## 2026-08-24 — a zero-row trace tick would have wiped the departure carry and stamped a fresh timestamp on a frozen crowd, turning a feed outage into confident wrong numbers

origin: agent

Caught by the adversarial pre-commit review, not by any test I wrote, and it is
the sharpest bug in this whole body of work.

`updateStationWait` prunes trips absent from the rows it is given — correct on a
normal tick, since a trip that stopped appearing has finished. But the step ran
unconditionally, including on a tick where `deriveTrace` threw or every vehicle
feed failed and `rows` was `[]`. Folding zero rows in does three things at once,
measured directly against the real function:

    prior : {observed_at: …66780, platforms: {A01N: …66680}, trips: {a: 'A01N'}}
    folded: {observed_at: …67380, platforms: {A01N: …66680}, trips: {}}

The trip carry is wiped, so the departure rule is blind for the tick after the
feed returns (and that rule is the thing worth 7% of departures — see above).
`observed_at` jumps forward ten minutes. The platform timestamps stay frozen.
That combination is exactly what the snapshot's freshness gate exists to catch,
and it defeats it: the surface would keep publishing an ageing crowd stamped as
current, growing more wrong every tick of the outage, which is strictly worse
than publishing nothing.

Fixed by skipping the update entirely when `rows` is empty and falling back to
the stored document unmodified, so the gate ages the surface out on its own over
~30 minutes while the cap retires individual platforms as their gaps grow. Zero
rows means a total feed outage or a throw, never a real empty system.

Two things to keep from this. First, the general shape: a carry that prunes on
absence must never be fed an empty observation, because "nothing was observed"
and "nothing is there" are the same input and opposite facts. The same hazard
lives in any decayed or pruned accumulator in this repo. Second, my own test
suite had 407 passing tests over this code and none of them covered a zero-row
tick — every fixture supplied rows, because supplying rows is what you think to
do. The new test asserts the stored document is byte-identical after an
all-feeds-fail tick, and it fails against the old code on all three counts.


## 2026-08-24 — the Worker's per-tick CPU goes to zod validation, not to JSON or to the maths: 3.4ms to validate an artifact that parses in 0.94ms, and serialising the whole 109KB snapshot costs 0.11ms

origin: artifact

Asked whether the new crowding surface needs trimming to fit a free tier.
Measured rather than guessed, running the real Worker modules over the real
artifacts.

What the platform-crowding feature costs:

    per ordinary minute   updateStationWait          0.23 ms  (746 trace rows)
    per ordinary minute   wait-doc parse+stringify   0.21 ms  (35 KB)
    per boundary minute   load+validate baseline     6.34 ms  (454 KB)
    per boundary minute   derivePlatformCrowding     1.52 ms
    -------------------------------------------------------------
    added per ordinary minute                        0.44 ms
    added per boundary minute                        8.29 ms

Against the documented cron CPU budgets (Workers limits, retrieved 2026-08-24):
**Workers Free is 10 ms per cron trigger, Workers Paid is 30 s.** We are on Paid
(ADR 0001 says so, and the trainer container requires it), so 8.29 ms is 0.03%
of budget and there is nothing to fix. Free was never reachable for this
pipeline: this one feature is 8.29 of the 10 ms on its own.

The reusable finding is where the time actually goes, because it is not where I
assumed. Decomposing the 454 KB artifact load:

    JSON.parse full artifact (454 KB)      0.94 ms
    zod validate full artifact             3.37 ms
    JSON.parse lean (rates only, 129 KB)   0.68 ms
    zod validate lean                       1.63 ms
    JSON.stringify the 109 KB snapshot     0.11 ms

**Serialisation is nearly free and schema validation is the cost.** V8 stringifies
the entire published snapshot in 0.11 ms; zod spends 3.4 ms walking 425 complexes
x 48 numbers applying `finite().nonnegative()` per element. So the lever for
Worker CPU in this repo is the AMOUNT OF ZOD WORK per tick, not payload bytes —
which inverts the intuition that shipping fewer bytes is what makes a tick
cheaper. Anyone optimising a tick should count validated array elements, not KB.

The available saving, not taken: the Worker reads only
`complexes[cx].entries_per_min`. The name/borough/rank/entries_total/n_cells
fields exist for the station fact sheet, a build-time consumer. Splitting a lean
worker-facing rates document out of the full artifact is 72% smaller and roughly
halves validation, about 4 ms off a boundary tick. Left undone deliberately: it
is a refactor across ingest, Worker and tests to reclaim 4 ms of a 30,000 ms
budget.

Two cost notes in ADR 0001 that are now stale, both from the cron moving to
every minute and the domain moving onto the Worker:

- "The Worker's ~288 invocations/day" is now ~1,440/day (`crons = ["* * * * *"]`).
  Still far under any request limit.
- "public reads are R2 bytes-out behind the CDN edge cache, not Worker
  invocations" no longer holds: `wrangler.toml` routes feed.momentarily.nyc
  THROUGH the Worker (`custom_domain = true`), so a cache miss is a Worker
  invocation. That also means the crowding surface's +8 KB gzipped on the
  snapshot costs nothing in Worker terms — Workers bill requests, not bytes — so
  the size flag I raised on it was aimed at the wrong resource.

## 2026-08-25 — first production attribution: advance_rate comes back in two state orderings across routes, worth 3.91 vs 1.20 nats per advancing trip, and the published suppression clusters with them at 152 vs 43 nats

origin: agent

Predictions now carry `matched_n`/`advanced_n` (c34ae98), so the movement
channel can be priced against the fitted params instead of the bootstrap ones.
The result corrects both quantitative guesses in the 2026-08-23 entry above.

**What that entry got wrong.** It sized the channel with the bootstrap
emissions and back-solved `matched_n ≈ 161` from a published
`p_disrupted = 2.69e-17`. Neither survives contact:

| | 2026-08-23 assumed | production |
| --- | --- | --- |
| `advance_rate[normal]` | 0.6 (dataclass default) | **0.8 – 0.999** (fitted) |
| `matched_n` | ~161, inferred | **8 – 26**, measured |
| per-advancing-trip LLR | 0.192 nats | **1.20 or 3.91 nats** |

The 0.192 was `KL(0.6 || 0.3)`, the *expected* per-trip divergence. The filter
never sees the expectation, it sees the realised count, and trips almost all
advance (`k = n` on 9 of 20 routes, `k >= n-2` on 17). The right quantity is
`ln(rate_normal / rate_other)` per advancing trip. The mechanism the entry
named — one channel growing with the trip count while six do not — holds; its
coefficient was wrong by roughly an order of magnitude, and the fleet-size
story was wrong the other way: `matched_n` never exceeds 26 on any route.

**The measurement.** `advance_rate` is indexed `(normal, disrupted,
suspended)`. Across the 20 routes carrying counts in tick 1787682922 it comes
back in two orderings:

| ordering | routes | per-advancing-trip LLR vs index 1 | movement nats (median) | observed suppression (median) |
| --- | --- | --- | --- | --- |
| `(0.8–0.999, 0.3, 0.02)` | 1 3 4 5 6 7 A B N Q R (11) | 1.20 | 17.5 | **42.6** |
| `(0.8–0.999, 0.02, 0.3)` | 2 C D E F G J L M (9) | 3.91 | 47.7 | **151.7** |

In the second group index 1 carries 0.02 and index 2 carries 0.3, so an
ordinary advancing trip is 3.25x more evidence against index 1 than it is on
the routes beside it. The published posterior clusters the same way: a 3.6x gap
in total suppression between lines running comparable service.

**Not established: whether that is wrong.** Two readings fit the data equally
well so far. Either those routes legitimately fit a near-frozen disrupted state
— plausible where a route's alert-disrupted episodes are mostly suspensions —
or the orderings disagree because `canonicalize_states` sorts states by the
ALERT channels (lowest `poisson_lambda`, highest `bernoulli_p`;
`hmm.py:951-995`) and constrains nothing about the movement channel, leaving
whichever permutation the fit landed on. The one suggestive fact is that in both
groups index 1 and index 2 hold the dataclass defaults (`hmm.py:131`, `(0.6,
0.3, 0.02)`) merely permuted, with only the normal state moved off its prior —
consistent with those two states never being separated by movement evidence, but
equally consistent with a prior-dominated fit. Separating the two needs the EM
run's own state assignments and per-state responsibility mass, not the published
params. Do not treat the split as a defect until that is read.

**Method, and what is exact in it.** The movement column is the shipped
`_log_emission` evaluated on the persisted counts and the fitted params, no
reconstruction. The transition floor (`ln(A[0][0]/A[0][1])`, median 4.6 nats)
comes from the same params. The alert and service columns computed alongside are
NOT sound and back none of the numbers above: the service channel is one tick
lagged by construction (`index.ts`, "fold in the previous tick's service
level"), so the current snapshot's ratio is the wrong tick, and the alert flags
were re-derived from alert-type substrings rather than through `derive.ts`.
Pricing those two needs them persisted the way the movement counts now are.

So the ranking question remains open for the alert and service channels. What is
settled: movement is a large term (17.5 of ~43 nats where index 1 is 0.3), and
parameter ordering rather than observed evidence moves it by 3.25x between
adjacent routes.

## 2026-08-25 — negative result: exact-default `advance_rate` values cannot establish zero movement responsibility, because per-route fits run at `prior_strength=100` and the serialized number is prior-dominated either way

origin: agent

Attempt to settle the ordering question above from the published params alone,
without a trainer run. It does not close, and the reason is worth writing down
so the next attempt does not spend the same hour.

The measurement that looked decisive: across all 28 routes in
`state/params.json`, both non-normal states hold their initialization constant
**exactly** — `0.3` and `0.02` (`hmm.py:131`), 56 of 56 route-states — while the
normal state moved on 22 of 28, landing on `0.999` (16 routes) or on small
rationals (`2/3`, `4/5`, `5/6`, `7/8`, `11/12`). The obvious reading is that the
fit had movement evidence, spent it all on the normal state, and never separated
`disrupted` from `suspended` on this channel anywhere.

**Why that reading does not hold up.** Per-route fitting calls `fit_hmm` with
`prior_params=global` and `prior_strength=100.0`
(`training/train_em.py:244-288`), so `use_prior` is true and the advance-rate
M-step is the κ-blended branch `(κ·prior + mov_k)/(κ + mov_n)`
(`hmm.py:794-796`), not the plain `mov_k/mov_n` I assumed. Two consequences:

  - That branch returns the prior exactly whenever `mov_k = prior·mov_n`, not
    only when `mov_n = 0`. Exact equality is therefore not proof of zero
    responsibility.
  - The clean rationals do not prove the plain branch was taken either. `fit_hmm`
    also accepts `advance_priors` — per-route normal advance rates injected from
    the movement baseline (`train_em.py:247,1025`), themselves ratios of archive
    counts, so `11/12` is as likely to be an injected prior as an MLE output.

There is also a whole-prior escape at `hmm.py:711-714`: a subset below
`MIN_EFFECTIVE_OBS` returns the prior emission set entire, which would produce
exact defaults on every channel at once.

What survives: with κ=100 and per-tick `matched_n` of 8–26 over a 28-day corpus,
a state receiving real movement responsibility would accumulate `mov_n` in the
thousands and swamp the prior, so exact defaults on 56 of 56 remain strong
circumstantial evidence that these two states get ~no movement mass. Strong is
not settled, and this channel has already produced one confident wrong number
(the 0.192 nats of 2026-08-23), so it is not being written down as fact.

To close it, the trainer has to report what the params cannot: per-state
`mov_n` — responsibility-weighted matched trips — and the global prior's own
`advance_rate`. That is instrumentation in the fit, not analysis of its output.

## 2026-08-25 — correction to the heading above: "prior-dominated either way" is wrong, and the error inverts the entry's own evidence

origin: agent

The entry above is headed "...and the serialized number is prior-dominated
either way". That phrase is wrong, and its own body says so two paragraphs
later: with κ=100, a state accumulating `mov_n` in the thousands would swamp the
prior, not be dominated by it. The number is prior-dominated only in the
low-`mov_n` case.

The correct statement is narrower: per-route fits run at `prior_strength=100`,
so the κ-blended M-step `(κ·prior + mov_k)/(κ + mov_n)` (`hmm.py:794-796`) can
return the prior exactly without `mov_n` being zero — it does so whenever
`mov_k = prior·mov_n`. That is what breaks the proof.

The distinction matters because it runs the other way too. Prior-domination is
not a symmetric escape hatch: it is exactly the low-`mov_n` regime, which is the
conclusion the entry was reaching for. Writing "either way" in the heading threw
away the entry's only real evidence — that a well-fed state would have moved off
`0.3`, so 56 of 56 sitting on it is informative, just not conclusive.

## 2026-08-25 — settled structurally: the training corpus is alerts-only, so `advance_rate` and `service_mu/sigma` can never be fitted — 0 movement ticks in 6,993, and the 10 routes canonicalisation swapped carry BOTH channels' constants transposed

origin: agent

The two entries above failed to settle this from serialized params, twice, and
both times the hole was the same: an exact value can mean "no data" or "prior
coincidence" and the numbers alone cannot separate them. The answer was never in
the params. It is in what the trainer feeds the fit.

**`load_series_by_route` (`train_em.py:173-239`) builds its observations from the
alerts archive only** — `list_alert_keys`, `fetch_objects`,
`build_tick_observations`, `fill_quiet_ticks`. It never reads
`archive/vehicles/` or any movement or service metric. Those two channels exist
only on the live Worker's observation, folded in at `index.ts` from the previous
tick's metric docs. So every training tick has `has_movement = False` and
`has_service = False`.

Measured, running the repo's own E-step (`_per_tick_emissions`,
`_forward_scaled`, `_backward_scaled`) over the archived series under the
published params:

| route | ticks | ticks with movement | `mov_n` per state |
| --- | --- | --- | --- |
| 1 | 2,212 | **0** | 0.0, 0.0, 0.0 |
| 2 | 2,312 | **0** | 0.0, 0.0, 0.0 |
| G | 2,469 | **0** | 0.0, 0.0, 0.0 |

Zero movement mass in 6,993 ticks, so the advance-rate M-step
(`hmm.py:794-801`) can only ever take the prior or fallback branch, and
`svc_w = 0` does the same to `service_mu/sigma`. **These parameters are not
badly fitted. They are structurally unfittable in the current training path.**
That is why no amount of float comparison could distinguish the cases: both
branches were the same branch.

What corroborates it, from the params:

| | |
| --- | --- |
| `advance_rate` idx1/idx2 exactly the defaults | **28 / 28** |
| `advance_rate[normal]` exactly equal to one of that route's baseline `p0` | 20 / 28 |
| ...or the raw `0.6` default | 6 / 28 |
| `poisson_lambda` moved off bootstrap (the ALERT channel really is fitted) | **28 / 28** |

The only movement information reaching training is `advance_priors` →
`_apply_advance_prior` (`train_em.py:247-274, 292`), which sets the NORMAL
state's rate from the measured baseline and leaves the other two alone. Hence
20/28 exact `p0` matches on normal and 28/28 untouched defaults beside them.

**The clincher for the ordering question.** `service_mu` is exactly the default
`(1.0, 0.6, 0.05)` on 18 routes. The other 10 are not fitted values — they are
`(1.0, 0.05, 0.6)`, the same constants transposed. Those 10 routes
(2 C D E F G J L M FS) are precisely the routes whose `advance_rate` is
transposed. `canonicalize_states` permutes every emission field through
`_reorder_emissions`, and it chooses the permutation from `poisson_lambda` and
`bernoulli_p` — the channels that ARE fitted (`hmm.py:232-241`). So a relabel
justified by alert statistics silently transposes two channels' hardcoded
constants, and the live filter then scores real movement and service
observations against whichever constant landed on `disrupted`.

That is the mechanism behind the 3.25x per-trip evidence gap and the 42.6 vs
151.7 nat suppression split measured earlier. It is an artifact — not of a buggy
canonicalisation, which is doing exactly what it says, but of canonicalising
parameters that carry no information, using a channel that does.

**Consequences for the fix, which is now a different fix.** Rescaling the
binomial per-tick, proposed on 2026-08-23, would rescale a constant. The real
options are to feed movement and service into the training corpus so the two
channels can be fitted, or to stop treating them as likelihood channels and use
them only where they are measured. Until one of those happens, the baseline's
own defect — `compute_advance_baseline` takes `p0` as a median of per-tick
advance fractions rather than the cell's pooled advanced/matched rate, which
saturates at 0.9990 where the pooled rate is 0.9443 — is the single number
setting the movement channel's entire behaviour on 20 of 28 routes.
## 2026-08-25 — settled: the two `advance_rate` orderings are a canonicalization artifact, because the HMM training observations carry no movement channel at all — `mov_n = 0` on every route and every state

origin: agent

Closed the ordering question the two entries above left open. It needed the one
thing the serialized params cannot show — the fit's own per-state movement
responsibility — so I instrumented the fit and ran it on the exact published
window (`2026-07-31..08-13`, `prior_strength=100`, `min_ticks=288`, the corpus
behind `state/params.json` trained_at 1786581775).

**Instrumentation.** `advance_responsibility(observations, params)`
(`hmm.py`) reads the M-step's own `mov_n`/`mov_k` — the responsibility-weighted
matched/advanced trips, `Σ_t γ[t][s]·matched_n` over `has_movement` ticks — off a
single E-step at the fitted params. `train_em --diagnose-advance` prints it per
route beside the fitted rate and the prior it blends against, plus the global
prior's own `advance_rate`. (Sanity-checked the reader on synthetic
movement-bearing data: total `mov_n` came back exactly `20×40`, split across
states by responsibility, so a zero is a real zero, not a skipped loop.)

**The measurement.** Global prior `advance_rate = (0.600, 0.300, 0.020)` — the
bootstrap default (`run_filter.BOOTSTRAP_PARAMS`, unset → dataclass default
`hmm.py:131`) verbatim. And across all 28 routes × 3 states:

    mov_n = 0.00,  mov_k = 0.00   — every cell, no exception.

Every fitted `advance_rate` equals its prior to the last digit. `normal` differs
by route only because `_apply_advance_prior` injects the movement-baseline route
rate into the *normal* prior alone (`train_em.py:272-277,292-300`); the
18-vs-10 group each route falls into is exactly whether the other two states'
`0.3`/`0.02` came through canonicalization straight or swapped.

**Why `mov_n` is literally zero.** The HMM training observations are
reconstructed from the *alerts* archive only. Both constructors —
`load.build_observations` and `load_r2.build_tick_observations` — build
`Observation(...)` with `advanced_n`/`matched_n`/`has_movement` left at their
defaults (`0/0/False`); nothing downstream joins movement counts in
(`load_series_by_route` just quiet-fills and hands the alert observations to
`train`). So `has_movement` is False on every training tick, the M-step's
`mov_k`/`mov_n` sums are empty, and the κ-blend `(κ·prior + 0)/(κ + 0)` returns
the prior identically for all three states. The global fit is pure MLE and sees
the same empty channel, so its `elif mov_n > 0` branch never fires and disrupted/
suspended fall to the bootstrap fallback `0.3`/`0.02` — which is why the anchor
every route inherits is the default, not a learned rate.

**The answer.** Canonicalization artifact, not a fit. `canonicalize_states`
orders the two non-normal states by the alert channels — lowest `poisson_lambda`
takes normal, higher suspended-alert `bernoulli_p` of the rest takes suspended
(`hmm.py:232-240`) — with `advance_rate` only a second sort key behind
`bernoulli_p`. That tiebreak never fires here: in the published params
`bernoulli_p[suspended] > bernoulli_p[disrupted]` *strictly* on all 28 routes
(closest margin 0.0244 vs 0.0244-to-4dp on D, still separated beyond 1e-12; no
exact tie anywhere), so the disrupted/suspended assignment is decided entirely by
the fitted suspended-alert channel and `advance_rate` contributes nothing, not
even as a tiebreak. The two advance constants ride along as whatever raw EM
cluster they initialized on. The 3.25× per-advancing-trip gap the 2026-08-25
attribution entry measured is real arithmetic on the shipped params, but it is
the *live* forward filter applying a movement emission whose parameters were
never trained on movement — the split between adjacent routes is only whether the
higher-`bernoulli_p` cluster landed on the raw index carrying the `0.3` prior or
the one carrying `0.02`, relabeled by canonicalization. Nothing about observed
advancing behaviour moved it.

This also resolves the earlier worry about `mov_n` "small relative to κ=100": it
was not small, it was zero, by construction. The circumstantial reading (a
well-fed state would have left `0.3`) was pointing the right way for the wrong
reason — no state on this channel is fed at all during training.

**The median normal-baseline is upstream, and now precisely so.**
`compute_advance_baseline_by_route` (the median-of-tick-fractions estimator that
saturates to 0.999 where most ticks are stall-free) is the *only* movement
information that reaches the HMM, and it reaches only the normal-state prior. Its
median-vs-pooled-rate defect therefore biases the one channel input that isn't
inert; disrupted/suspended never see movement by any path.

**Left untouched, deliberately.** Did not re-order states, rescale the binomial,
or wire the movement channel into training. The finding is bigger than the
ordering — the advance/movement emission ships as pure priors and is untrained —
but that is a design change, not this determination. The `--diagnose-advance`
instrument stays as the reproducer.

## 2026-08-25 — fix: wire the movement counts into HMM training, so the advance emission is fitted instead of inherited — `mov_n` 0 → 1.66M matched trips, disrupted/suspended rates now learned

origin: agent

Follow-up to the determination above. The advance/movement emission shipped as a
pure prior because the training observations never carried the channel; the live
filter has applied it since 2026-06-22 (`3fe5891`) against parameters EM never
saw movement to fit. This wires the counts in.

**What the live filter does, mirrored.** `worker/src/movement_state.ts`
`movementObservationFields` folds the *previous* tick's cross-tick counts (option
B lag) into each observation — both directions summed off the raw route counters
— and abstains (channel off) on a stale carry (>600s), fewer than
`MIN_MATCHED_TRIPS`, or no published baseline cell for the current tod_bin. The
training side now reconstructs exactly that: `load_r2.movement_observation_fields`
is a straight port of those gates, fed the raw per-`(route, tick)` counts
(`build_movement_series`, unfiltered — the live filter reads unfiltered counters,
so the emission is fitted on the same population) and the fitted baseline's cell
key set. `_movement_baseline` now returns a `MovementInputs` carrying the raw
per-tick counts and baseline key set alongside the serialized baseline;
`load_series_by_route` takes a `movement_fields` callable and folds the fields in
while the tick tag is still on the observation, before it is dropped; `main`
computes the movement inputs before loading the series so the callable is ready.
No train/serve skew: the lag, the summing, and all three abstain gates match the
worker line for line, with a unit test pinning each (previous-tick only, ≤2-tick
lag, min-matched, baseline-present).

**Measured on the published window** (`2026-07-31..08-13`, `--diagnose-advance`):
per-state `mov_n` went from 0 on all 84 route-states to nonzero on 62. The 22
that stay zero split two ways, both correct: routes with no movement coverage at
all — no published baseline cell, so the gate holds the whole route off (the
shuttles and low-frequency lines, e.g. 6X, 7X) — and individual states inside
otherwise well-fed routes that simply draw ~zero responsibility on the
movement-available ticks (7's suspended, L's disrupted, E's normal each sit at 0
while the same route's other states carry thousands). It is posterior mass and
the has_movement gate, not route thinness — every route shown cleared min_ticks.
Both keep the prior, correctly. 1.66M matched trips now enter the fit. Where
`mov_n > 50` the fitted rate tracks the data rate `k/n` (only 2 states deviate
>0.02, the prior pulling on modest-mass states, as intended at
`prior_strength=100`). The global prior's advance_rate moved off the bootstrap
default to a fitted `(0.632, 0.617, 0.644)`.

**What that last number means, and the honest limit.** The three fitted rates
are close — normal 0.756 median, disrupted 0.551, suspended 0.570 across routes
with mass — and disrupted/suspended overlap. The movement channel is now trained,
but the HMM's hidden state is still *alert-defined*, and physical advance is only
weakly separated by it: a route flagged disrupted/suspended by alerts is often
still advancing near normally. So this fix removes a real defect (an untrained,
prior-only emission scored live) and is a prerequisite for trusting the movement
term, but it does not by itself make movement a sharp discriminator on the
alert-HMM. The larger value of movement is as its own axis (the movement-primary
condition already in `movement_state.ts`) — this change makes the shared advance
parameters honest either way.

**Not published.** Verified by dry-run only; no R2 write. Wiring the channel
moves the live operating point, so publishing waits on a review of the fitted
params against the current ones. Full Python suite green.

## 2026-08-25 — fix: the advance baseline is a trimmed pooled rate, not a median of tick fractions — p0 desaturates (cells ≥0.99: 76% → 8%), so "normal" is a rate cells actually run at

origin: agent

Shipped the estimator fix the 2026-08-13 entry measured and deferred. `p0` is the
cell's normal cross-tick advance rate, and it anchors both the movement
trip-wire (`classify_direction`, `disrupted ≤ 0.5·p0` plus a binomial
significance gate against p0) and the HMM's advance prior. It was the *median of
per-tick advance fractions*: with ~71% of through-filtered ticks stall-free and
small per-tick denominators (~10 trips) quantising to 1.0, the median sat on the
ceiling and floored to 0.999 — a rate no cell actually ran at, which made the
significance test read a single stall in ten as significant.

**The fix.** `compute_advance_baseline` and `compute_advance_baseline_by_route`
now pool: p0 = Σadvanced / Σmatched over the cell's ticks
(`_trimmed_pooled_advance_rate`), a proper rate that doesn't quantise. Measured
on the published window (`2026-07-31..08-13`):

    estimator                    cells  median p0   ≥0.99   ≥0.999
    OLD median-of-fractions       210     0.9990    76.2%   76.2%
    NEW pooled (trim 0)           210     0.9370     8.1%    0.0%

That reproduces the deferred measurement (median 0.999 → ~0.94, ≥0.99 76% → 8%)
and matches the physical reality — a subway's normal advance is ~94%, not 100%.

**On the trim.** The original note proposed a one-sided lower trim (drop the
worst ticks) to stop outages dragging the pooled rate down. Measured, that is
the wrong instinct on this data: any trim>0 walks p0 straight back into
saturation (cells ≥0.999 go 0% → 19% at trim 0.1 → 37% at 0.2), because the
dropped low tail is mostly the ordinary stalls, not outages — real disruption is
rare enough that the raw pooled rate is barely dragged (0.937). So the trim is a
kept knob defaulting to 0; it earns its keep only against a movement-truth eval
that doesn't exist yet.

**Behavioural check.** Over 189k movement ticks the trip-wire's disrupted-fire
count barely moves (242 → 195). That is the point, not a shortfall: the old
fires were largely noise — at p0=0.999 a single stall reads significant — while
at a realistic p0≈0.9 a genuine freeze still fires (`P(0/10 | 0.9) ≈ 3.5e-5`)
but an ordinary single stall does not. The fix recalibrates *what* fires onto a
real normal rate; it is not a standalone accuracy jump (graded against the
alert-derived label it buys nothing — that label is the wrong truth). Its value
is that the quantity every movement-defined call rests on is now correct, which
is the prerequisite for the movement-native state work.

**Not published.** Same as the movement-wiring change: verified offline, no R2
write; this shifts the live operating point (the trip-wire fires ~0.14% vs
0.17%) and should go out with a params review. Tests updated to lock the
no-saturation property and the pooled/trim behaviour; suite green.

## 2026-08-25 — study: movement's own structure says the degradation axis is a continuum, not three states — but disruptions are rare, sticky regimes (dwell ~14 min), so a stateful model is warranted

origin: agent

Before reshaping the model around movement, asked the (now pooled, correct)
advance distribution how many states it wants. Window 2026-07-31..08-13,
through-filtered, 189k (route, direction, tick) cells. Clustering is on the
BASELINE-RELATIVE signal (advance fraction / cell p0) so routes with different
normals are comparable; persistence is measured only across genuinely adjacent
5-minute ticks (a judged tick whose t+300 is also judged), not across gaps.

**Presence (the suspended axis).** 97.8% of cells have trains present; of those
76.2% clear the 3-match floor, 23.8% are present-but-sparse (overnight / low
frequency). No trains at all: ~2%. "Suspended / no-service" is a real but small
state and it is a *presence* distinction — trains absent — not a point on the
advance axis.

**The degradation axis is a continuum, not clusters.** Baseline-relative advance
is one dominant mass at/above 1.0 (peak 39% in [1.00,1.05), ~80% of mass in
[0.9,1.15]) with a smooth thin left tail to 0 — no valley. A Gaussian-mixture BIC
keeps "preferring" more components (k=4 over k=3 over k=2), but the components
expose it as shape-fitting: every one added past the first is either a
near-duplicate normal sliver at ~1.0 or the single small tail blob at ~0.5
(weight ~6-7%). There is no clean multi-modal separation. On the advance axis the
most the data supports is TWO running states — normal (~1.0 of its own baseline)
and a small-mass degraded tail (~0.5) — not the alert model's three.

**But disruptions are rare, sticky regimes.** Over 73,741 genuinely adjacent
(t, t+5min) judged pairs, the movement-disrupted base rate is only 0.21%, yet
P(disrupted next | disrupted now) = 63.9% (99/155) vs P(disrupted next | normal
now) = 0.076% (56/73,586) — 839× stickier to stay than to enter, an implied dwell
of ~2.8 ticks (~14 min). Small disrupted sample (155 pairs), rare events, but the
asymmetry is unambiguous: a memoryless per-tick threshold (today's
`movement_state.ts`) throws away that persistence; a stateful model with learned
dwell/transitions is warranted.

**What this means for the state space.** The alert model's three states don't
transfer as-is. Movement has a *presence* axis (running / sparse / absent) and a
*continuous* degradation axis when running. A discrete label can carry at most:
suspended = trains absent (presence), normal vs degraded = one chosen cut on the
continuous advance-vs-baseline axis. There is no movement evidence for a third
*advancing* regime — the disrupted/suspended split the alert model draws is an
alert distinction, not a movement one. Two candidate architectures, to decide
deliberately: a 2-running-state + presence discrete HMM with a threshold cut and
learned dwell, vs a continuous-severity semi-Markov model. The data leans toward
"few coarse states are enough on the advance axis; spend the modelling on
persistence/dwell, not on more emission clusters."

## 2026-08-25 — experiment: a continuous-severity filter (Option B) demonstrates the mechanics, and running it caught a train/serve skew in today's movement-wiring — the offline emission was fitted on unfiltered counts while the live worker filters

origin: agent

Prototyped Option B to see it, not describe it: latent per-tick advance rate
θ_t, observed advanced_n ~ Binomial(matched_n, θ_t), logit(θ) a mean-reverting
AR(1) toward the route's normal (persistence = the transition model). Particle
filter, numpy only, on the real window.

**The mechanics work; calibration is deferred.** The filter tracks the signal
and smooths single-tick Binomial noise (a lone 0-of-13 tick dips within its band
instead of snapping to 0). It is NOT yet calibrated: φ=0.75 and σ=0.45 were
hand-set, and its 15-minute recovery forecast failed on the showcase, so treat
this as a demonstration of the filtering machinery, not of its uncertainty or
forecasts. Calibration waits until the signal it runs on is trustworthy. The
empirical lag-1 autocorr of logit(advance) on high-n adjacent ticks is 0.86 —
the real transition is stickier than the prototype assumed.

**What running it caught — a skew I introduced earlier today.** The showcase
picked route G's "longest sub-0.6·p0 run" and it came back 661 ticks (~55 hours)
of raw advance at 15–25% against baseline 0.609. Not a disruption — an artifact.
The live worker filters the advance counters to scheduled through-stops
(`worker/src/vehicles.ts deriveRouteMovementMetric`, since 8e88644, 2026-08-12:
a terminal/layover stall is not signal), and the baseline is fitted the same
way. But my movement-wiring change (earlier today) reconstructed the offline
training counts with `build_movement_series(bodies)` — the RAW unfiltered route
counters — so the HMM movement emission was fitted on a population the live
classifier never feeds it, and the prototype inherited the same raw counts.
Measured, the raw-vs-filtered gap is systematic (median 0.258; Z 0.473 vs 0.946,
E 0.461 vs 0.921), and it vanishes when the reconstruction is filtered the way
the worker filters (median gap to the baseline 0.258 → 0.001).

Not a live bug: production has been consistent (filtered vs filtered) since the
worker filter landed. The bug was entirely in the offline reconstruction I added.

**Fix.** `_movement_baseline` now builds `movement_by_tick` with the same
`counts_from_stop` the baseline uses, so training, the baseline, and the live
worker all score one population. Re-verified on the window: normal-state fitted
advance now tracks each route's filtered baseline (Z 0.95=0.946, A 0.88, C 0.99,
Q 0.92) instead of the skewed ~0.5, and the global prior advance_rate is
(0.91, 0.85, 0.88) — correcting the (0.63, 0.62, 0.64) the earlier wiring entry
reported, which was that entry's skewed fit (its "fitted rates overlap ~0.55"
read was the same artifact). Test locks that the per-tick counts get the same
filter as the baseline.

**Bearing on A vs B.** The state-space study above already ran on the filtered
signal, so its findings (continuum, not clusters; rare but ~14-min-sticky
regimes) stand. B's machinery is sound and now runs on an honest signal; with
the skew closed it is a live contender, calibration (φ, σ, per-route) being the
next real step before it could be trusted for forecasts.

## 2026-08-25 — calibration verdict: Option B's single-AR(1) form is misspecified — the advance signal is two-timescale (a ~3.4-hour drift plus ~minute-scale excursions), so a one-timescale continuous filter can't beat base rate; this tilts the near-term choice to A

origin: agent

Tried to calibrate B properly rather than hand-set φ/σ. Model: advance rate
θ_t = μ + φ(θ_{t-1}-μ) + N(0,σ²), advanced_n ~ Binomial(matched_n, θ_t), on the
RATE scale (logit blows up at the frequent k=0 / k=n boundaries at n≈10).

**Estimating φ, carefully.** The errors-in-variables route —
Var(θ) = Var(p̂) − E[obs noise], φ = autocov(1)/Var(θ) — is unreliable here
because the plug-in obs-noise term p̂(1-p̂)/n is exactly 0 at the k=0/k=n
boundaries and understates sampling noise; it gives φ=1.11, and a
Laplace-smoothed noise term makes it 1.37 — i.e. this estimator is dominated by
how you handle boundary noise, so it can't be trusted. The robust estimate needs
no obs-noise term at all: lags ≥1 of the autocovariance are white-noise-free, so
a geometric fit of autocov(k) over k=1..8 gives φ directly. That fit is
**φ ≈ 0.975**, a **~3.4-hour** timescale (autocov barely decays: .0148 at 5 min
to .0126 at 40 min, per-step ratio ~0.98). Stationary, but very slow. Reverting
to a time-of-day mean instead of one p0 doesn't change it (only 16% of advance
variance is time-of-day).

**Two timescales, and that is the point.** The bulk signal drifts slowly
(~3.4 h); the deep disruption excursions the discrete classifier measured are
short (~14-min dwell). A single AR(1) can hold only one timescale, and it lands
on the slow bulk. Run end to end with the calibrated φ=0.975 (σ≈0.027), its
30-minute recovery forecast has no skill: over 98 onsets every forecast squashes
into one low bin (mean predicted 0.08) while 0.15 actually recover — Brier 0.143
vs a base-rate 0.130, i.e. slightly WORSE than a constant. It systematically
under-calls fast recoveries because its one timescale expects disruptions to last
hours. That is the model class being wrong for this signal, not a tuning miss.

**Verdict on A vs B.** A trustworthy continuous B needs a two-timescale /
local-level-plus-excursion state-space model (a slow drifting level and a fast
excursion component), which is a real research build, not a calibration tweak.
Option A sidesteps the whole problem: discretizing to normal / degraded /
suspended means the slow within-normal drift never crosses a threshold and does
not matter, and the ~14-min disruption dwell is exactly what A's learned
per-state dwell captures. So the near-term recommendation is A — a 2-running-state
(normal/degraded on an advance-vs-baseline cut) + suspended (presence) discrete
model with learned dwell/transitions. B is parked as the more faithful long-term
model contingent on the two-timescale formulation; the filtering machinery
prototyped here (Binomial obs, particle filter) is reusable when that is built.
The advance signal it all rests on is now honest (pooled baseline + through-
filtered counts on both sides), so A can be built on solid ground.

## 2026-08-26 — Option A calibration: the cut is already well-placed and the classifier does not flap, so A's win is corrected detection, not a retuned cut or a debounce

origin: agent

Built the offline cut/debounce calibration harness (`training/movement_calibrate.py`)
and swept it over a causal window (advance baseline fitted on 2026-07-26..07-30,
swept on 07-31..08-09, through-stop filtered on both sides). It reconstructs the
live classifier per tick across the three cut constants, runs the real regime
clock, and reports each setting's structure and corroboration. Two references,
which are NOT interchangeable:

- **Structural-consistency anchors** — disrupted base rate, per-tick stickiness
  P(dis next | dis now), dwell. These come from the classifier's OWN calls, so
  they show an operating point reproduces the population structure the
  state-space study described; they are self-referential, never independent
  validation. (The study's own 0.21% / 0.639 were likewise classifier-derived.)
- **Trip-updates corroboration** — overlap with `derive_actual_recovery`'s
  assigned_n disruptions. Independent in derivation from vehicle positions, so
  the only independent reference — and weak, because supply level and advance
  quality are different things (2026-08-20).

**The cut is already well-placed.** At the shipped constants (prior_strength=8,
disrupted_ratio=0.5, alpha=0.05) the reconstructed signal runs base rate 0.30%
and stickiness 0.648 — matching the study's 0.21% / 0.639 structure — at the
lowest churn on the grid (6 oscillations over 10 days × 27 routes). Along the
prior_strength=8 axis dr=0.5 is the sweet spot: dr=0.4 collapses to 6 episodes,
dr=0.6 or prior_strength=5 run 3–4× the base rate and churn for marginal
trip-updates precision on a weak signal. `alpha` is not load-bearing at typical
depths (0.01/0.05/0.10 are indistinguishable — deep freezes have binomial tail
≈0 at n≈10). Operating point CONFIRMED, unchanged; the old alert-label eval only
made it look untunable because it was the wrong truth.

**The debounce delta was rejected on evidence.** The handoff's premise — a noisy
single tick flips the committed condition and flaps — is false post-correction.
Churn is ~0.02 flips/route/day. The single-tick episodes (64% of the population)
are not noise: median binomial tail 0.0000, depth 0.47·p0 — statistically real
brief partial freezes (they differ from the persistent multi-tick freezes only
in being partial, 6% vs 52% zero-advance). Every damping variant hurts:
symmetric debounce=2 erases ~70% of real episodes and adds a tick of latency;
asymmetric exit-hysteresis barely changes anything (the episodes are genuinely
isolated, not fragments); a recover-ratio band (mirroring the service axis)
over-persists to 0.9% base rate. So `DEBOUNCE_TICKS=1` stays — the 2026-08-11
call, now for a stronger reason than "not noise-dominated": not flapping at all.

**A's real win is detection, and it is large.** On the same window the OLD live
signal (saturated baseline) saw 14 disrupted episodes, all ≤15 min, with a
quarter of ticks unjudgeable (`unknown`). The corrected signal (pooled baseline
desaturates cells ≥0.99 from 76% to ~16% here; through-filtered counts) recovers
56 episodes INCLUDING the persistent deep tail the old cut was blind to (seven
~45-min, one ~160-min), and its completed-episode dwell MEAN is 14.2 min —
matching the study's implied persistence. The correction restores the current
state the product shows; the cut and debounce were hypotheses the calibration
refuted.

**Acceptance is blocked, honestly.** The literal target — movement-arm recovery
beats status quo on the independent eval — cannot be established: over 14 days
`recovery_independent` has n=1 gradeable sample, because movement disruptions and
trip-updates disruptions overlap almost never. And recovery FORECASTING here is
inherently low-skill: 64% of disruptions are 5-min partial freezes, so a
conditional dwell adds no skill over predicting the unconditional median (a
self-consistent leave-one-episode-out diagnostic, not acceptance). This is the
same wall B hit. So B stays parked, and the residual is not a two-timescale
pattern to fix — it is the floor predictability of short partial freezes.

**Unshipped delta that remains: the publish.** The corrected params (pooled
baseline + through-filtered counts + fitted movement emission) are validated
offline but not published; moving the live operating point is the params-review
step, gated on the lead. No constants or debounce changed in this session.
