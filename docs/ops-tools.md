# Operator tools

The `training/` package holds two kinds of entrypoint. Most are stages of the
weekly training container, run for you by CI and the publish workflow. The
modules below are the exception: manual-run grading, evaluation, calibration,
and one-off diagnostics that are **not** wired into any workflow. They exist to
answer a question against the archive and are run by hand, so their zero
coverage is intentional, not a gap.

Every tool that reads or writes R2 needs the `R2_*` credentials in its
environment — run it under `murk exec -- <command>` (or export the vars
yourself). Tools that only touch local files or public open-data APIs need no
credentials. R2 prefixes are relative to the state bucket; `archive/*` prefixes
are the immutable per-tick history, `v1/*` the published prediction/transition
stream, `state/*` the live inference sidecars.

| Module | Purpose | Example invocation | R2 reads / writes |
| --- | --- | --- | --- |
| `backtest` | Tier-1 decision-gate backtest: KM-residual vs geometric `p_normal` forecast on a held-out window. | `uv run python -m training.backtest --eval-days 3 --horizon 6 --out out/` | reads `v1/predictions/`, `archive/vehicles/`; writes none (local `--out` `summary.json`) |
| `incidents` | Incident-level duration for clustered contiguous disrupted segments (parked — the one-cause-per-incident premise did not survive measurement). | `uv run python -m training.incidents --days 8` | reads `archive/vehicles/`, `state/segment_params.json`; writes none |
| `major_incidents` | Join the MTA's official Major Incidents open-data log to our archive — the one external, common-mode-independent truth source. | `uv run python -m training.major_incidents --json` | reads `archive/trace/` (+ NYS Open Data `ereg-mcvp` API); writes none |
| `movement_dwell_grade` | Causal grading of the published movement recovery-dwell forecast against climatology (CRPS). | `uv run python -m training.movement_dwell_grade --train-start 2026-08-01 --train-end 2026-08-14 --eval-end 2026-08-21` | reads `v1/predictions/` (or `archive/vehicles/` per `--source`); writes none |
| `movement_hsmm_grade` | Causal grading of an explicit-duration (negative-binomial) sojourn against the geometric movement debounce. | `uv run python -m training.movement_hsmm_grade --train-days 14 --eval-days 7` | reads `v1/predictions/` (or `archive/vehicles/`), `archive/trip_updates/`; writes none |
| `od_direction` | Origin-destination direction-demand diagnostic: sizes the direction bias the platform-crowding surface leaves uncorrected. | `uv run python -m training.od_direction --out od_direction.json` | reads NYS Open Data `28vm-gjqr` API only; writes none (local `--out` JSON) |
| `online_fdr` | Replay online-FDR control (LORD++ and ADDIS) over the fleet's archived movement-detector p-values. | `uv run python -m training.online_fdr --days 21 --fit-days 14` | reads `archive/vehicles/`, `archive/trip_updates/`; writes none |
| `segment_grade` | Grade the live `traversal.deviation` segment score against announced planned work. | `uv run python -m training.segment_grade --start-date 2026-08-12 --end-date 2026-08-16` | reads `archive/trace/`, `archive/windows/` (+ static GTFS zip); writes none |
| `service_night_gate_eval` | Does the supply baseline's independent-night gate tighten false alarms on bimodal weekend late-night cells? | `uv run python -m training.service_night_gate_eval --score-days 21 --fit-days 35` | reads `archive/trip_updates/`, `archive/alerts/`; writes none |
| `station_maintenance` | Per-station elevator/escalator maintenance scorecard sidecar for the Station page. | `uv run python -m training.station_maintenance` | reads `archive/ene/`, `archive/windows/`; writes none (committed `viz/public/station_maintenance.json`) |
| `backfill_service_baseline` | Backfill the supply-baseline sidecar from trip-updates without touching `params.json` (no retrain). | `uv run python -m training.backfill_service_baseline --dry-run` | reads `archive/trip_updates/`, `params.json`; writes `state/service_baseline.json` (+ versioned `state/service_baseline/`) |
| `movement_calibrate` | Sweep the movement disrupted-arm cut constants and regime debounce, reporting the diagnostics that decide an operating point. | `uv run python -m training.movement_calibrate --baseline-start 2026-08-01 --eval-start 2026-08-08 --eval-end 2026-08-11` | reads `archive/vehicles/`, `archive/trip_updates/`; writes none |
| `run_filter` | Demo: run the HMM forward filter over one route's observation history and print its regime trajectory. | `uv run python -m training.run_filter --route 6` | reads local collector archive (`./data`) only; writes none |
