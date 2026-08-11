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