<!--
Go/no-go memo template for the HMM shadow-validation review (momentarily-c72.4
graduation decision). After running `python -m training.review`, copy this into
that run's docs/review/<date>-shadow-hmm/ dir as memo.md and fill every <…> from
its summary.json + the 4 PNGs. NOTE docs/review/ is gitignored (local artifacts),
so record the FINAL decision in the momentarily-c72.4 bead — same as the smp
review did. Delete this comment in the copy.

Decision rule of thumb: graduate a field only if it is (a) calibrated, (b) beats
the trivial baselines (persistence AND climatology) where it claims skill, and
(c) holds up against an INDEPENDENT recovery truth, not just the model's own
argmax. The 2026-05-17 (smp) review was NO-GO on recovery MAE ~7h + mid-range
miscalibration; vk0 fixed the eval harness and the recovery estimator, so this
rerun is the first honest read.
-->

# HMM shadow validation — go/no-go memo

- **Date:** &lt;YYYY-MM-DD&gt;
- **Decision:** &lt;GO | NO-GO | PARTIAL — which fields graduate&gt;
- **Params under review:** `trained_at = <summary.current_params.trained_at>` (must be **1781491265** for the vk0.12 rerun — if not, the segment is the wrong model)
- **Code:** `code_sha = <provenance>` · **Window:** `<window.start>..<window.end>` (`<days>` days)
- **Artifacts:** `reliability.png`, `confusion.png`, `changepoint_alignment.png`, `recovery_by_route.png`, `summary.json` (this dir)

## 0. Data sufficiency (gate before reading anything else)

| Check | Value | Enough? |
| --- | --- | --- |
| `current_params.n_predictions` (this model only) | &lt;n&gt; | &lt;y/n&gt; |
| `recovery.overall.n` (graded recovery ticks) | &lt;n&gt; | &lt;y/n&gt; |
| `recovery.per_regime.n` (distinct regimes) | &lt;n&gt; | &lt;y/n&gt; |
| `recovery_clearance.n_disruptions` (up0) | &lt;n&gt; | &lt;y/n&gt; |
| `recovery_independent` present? (xum, ~06-28) | &lt;yes/null&gt; | — |

If these are thin, the rest is directional only — say so and don't graduate on it.

## 1. Calibration & probabilistic skill (`reliability.png`, `summary.calibration`)

| Horizon | Brier | BSS vs persistence | BSS vs climatology | Verdict |
| --- | --- | --- | --- | --- |
| 30 min | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; |
| 60 min | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; |
| 120 min | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; |

- Reliability curve on/off the diagonal: &lt;…&gt;
- **Key question:** does `p_normal_in_H` beat BOTH baselines (BSS > 0 vs persistence *and* climatology)? A horizon that loses to persistence is not ready. (Prior runs lost skill at 120 min — watch that row.)

## 2. Regime confusion (`confusion.png`, `summary.confusion`)

HMM `condition` vs MTA-derived state (row-normalized):

- normal precision/recall: &lt;…&gt;
- disrupted precision/recall: &lt;…&gt;
- suspended precision/recall: &lt;…&gt;
- Worst confusion cell and whether it matters for a consumer: &lt;…&gt;

## 3. Changepoint alignment (`changepoint_alignment.png`, `summary.changepoint_alignment`)

- Matched within ±30 min: `<n_matched>/<n_total>` (`<pct>`)
- Median |delta|: &lt;…&gt; min · mean delta: &lt;…&gt; min (sign = lead/lag)
- **Sanity check:** post-vk0.2 the truth grid now carries recoveries, so the match rate should be far above the old 25/1401. A still-low rate means either real lag or a remaining eval artifact — diagnose which.

## 4. Recovery accuracy — three truths side by side

The headline. Grade `recovery_minutes` against each independent-ness tier.

| Truth source | n | MAE (min) | RMSE (min) | IQR coverage (target ≈0.5) |
| --- | --- | --- | --- | --- |
| Argmax / self (`recovery`, per-tick) | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; |
| Argmax / self (`recovery.per_regime`) | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; |
| **Alert-feed clearance** (`recovery_clearance`, up0) | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; |
| **Trip-updates service** (`recovery_independent`, xum) | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; |

- Do the independent truths agree with the self metric, or is the self metric flattered by grading against its own argmax? &lt;…&gt;
- IQR coverage near 0.5 = honest intervals; far below = overconfident. &lt;…&gt;
- Per-route / per-alert-type outliers worth flagging (`recovery.by_route`, `by_alert_type`): &lt;…&gt;

## 5. Decision

**&lt;GO | NO-GO | PARTIAL&gt;**

Graduate (move from shadow `inference` to a stable published surface):
- `condition`: &lt;graduate / hold&gt; — because &lt;…&gt;
- `recovery_minutes` (+ low/high band): &lt;graduate / hold&gt; — because &lt;…&gt;
- `p_normal_in_H`: &lt;graduate / hold, which horizons&gt; — because &lt;…&gt;

Gating rationale tying back to §1–4: &lt;…&gt;

## 6. Caveats & follow-ups

- Known data-coverage limits (e.g. route 7 only ~8h/day in trip-updates; thin routes): &lt;…&gt;
- xum trip-updates truth matures ~2026-06-28 — re-read §4 then if it was null here.
- New beads filed: &lt;ids&gt;
