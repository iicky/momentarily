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