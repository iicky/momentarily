// Mirrors the Worker's published contracts so the parser stays honest.
// Sources: worker/src/snapshot.ts (Snapshot, RouteStatusOut, Inference, SystemStatus),
//          worker/src/grading.ts (PredictionRecord, TransitionRecord).
// Keep field names in lockstep with those files.

export type Condition = "normal" | "disrupted" | "suspended" | "unknown";

export interface Inference {
  condition: string;
  recovery_minutes: number;
  is_disrupted: boolean;
  p_normal: number;
  p_disrupted: number;
  p_suspended: number;
  regime_entered_at: number;
  regime_age_seconds: number;
  recovery_minutes_low: number;
  recovery_minutes_high: number;
  // Dwell estimate saturated the ceiling — recovery bounds are clamped/meaningless.
  recovery_indeterminate: boolean;
  p_normal_in_30min: number | null;
  // All three horizons are withheld (null) rather than publish a number
  // known to be wrong, for different reasons per horizon:
  // 60/120min: model-derived forecasts scored worse than naive persistence
  // (AUC 0.395 / 0.352, BSS as low as -1.30 — see journal.md:1040-1051).
  // 30min: withheld whenever the forecast arm isn't the arm that produced
  // `condition` above. Graded against the published condition,
  // movement-sourced forecasts score AUC 0.856, hmm-sourced score AUC
  // 0.261, and mixing the two scores AUC 0.084 — worse than either, because
  // the two arms' probabilities aren't on the same scale.
  p_normal_in_60min: number | null;
  p_normal_in_120min: number | null;
  model_warming_up: boolean;
  // Which arm produced the recovery numbers. "movement" and "schedule" forecast
  // the PUBLISHED condition; "hmm" is the alert filter's own regime, which is
  // exactly when every horizon above is withheld — so this is also how a reader
  // (and the drawer) tells a bounded-out forecast from a withheld one.
  recovery_source: "hmm" | "schedule" | "movement";
  // Announced end of a planned-work window, set only by the schedule arm.
  resumes_at: number | null;
  // now has passed resumes_at but the alert is still up.
  overdue: boolean;
}

export interface DirectionAlerts {
  alerts: string[];
  primary_alert_type: string | null;
}

export interface RouteStatus {
  route_id: string;
  alerts: string[];
  condition: string;
  // Where `condition` came from: 'movement' | 'schedule' | 'unknown' | 'hmm'.
  condition_source: string;
  // When `condition` began (epoch s) — the badge's own clock, how long the
  // published state has held. Non-null only when condition_source === 'movement';
  // null for 'schedule'/'unknown'/'hmm'. NOT inference.regime_entered_at (which
  // times the HMM argmax); the drawer must never fill this from that clock.
  condition_entered_at: number | null;
  // Supply axis (assigned_n vs its hourly baseline), distinct from `condition`
  // (flow). 'normal' | 'degraded' | 'unknown'.
  service_condition: string;
  // Magnitude behind service_condition (assigned_n / hourly baseline), or null
  // when unjudgeable.
  service_ratio: number | null;
  // Cell's own p10/median at this hour — this route's usual low end. Null
  // whenever the cell has no quantiles, its median is <= 0, or the route is
  // unjudgeable this tick (same conditions as service_ratio, plus no quantiles).
  service_low_ratio: number | null;
  // Cell's own p90/median at this hour — this route's usual high end. Null
  // under the same conditions as service_low_ratio.
  service_high_ratio: number | null;
  // Where service_ratio sits within this cell's own same-daypart baseline, as a
  // 0-100 percentile (worker movement_state.servicePercentile). Low = fewer
  // trains than usual for this time of week; exact at the cell's p10/median/p90,
  // saturating at 90 above its p90. A percentile of the baseline, NOT a forecast.
  // Null under the same conditions as service_low_ratio.
  service_percentile: number | null;
  category: string;
  primary_alert_type: string | null;
  label: string;
  by_direction: {
    northbound: DirectionAlerts;
    southbound: DirectionAlerts;
  };
  inference: Inference | null;
}

export interface Freshness {
  subway_alerts: number | null;
  lirr_alerts: number | null;
  mnr_alerts: number | null;
  bus_alerts: number | null;
  path_alerts: number | null;
  ferry_alerts: number | null;
  ene: number | null;
  stations_static: number | null;
}

export interface SystemStatus {
  by_mode: Record<
    string,
    { routes_with_alerts: string[]; alert_count: number; severity_max: number }
  >;
  accessibility: {
    elevators_out: number;
    escalators_out: number;
    ada_pathways_degraded: number;
  };
  overall_label: string;
  condition: string | null;
  lines_disrupted_count: number;
  most_degraded_line: string | null;
  most_recovered_line: string | null;
}

export interface CompatRoute {
  id: string;
  name: string;
  color: string;
  status: string;
}

// --- Station + segment surfaces (mirrors worker/src/snapshot.ts + schema) ---

export interface Station {
  gtfs_stop_id: string;
  station_complex_id: string | null;
  name: string;
  borough: string | null;
  routes_served: string[];
  ada: 0 | 1 | 2;
  ada_northbound: boolean;
  ada_southbound: boolean;
}

export interface StationStatus {
  station_complex_id: string;
  alerts: string[];
  ada_status: string;
  elevators_total: number;
  elevators_out: number;
  escalators_total: number;
  escalators_out: number;
  earliest_elevator_return: number | null;
  oldest_outage_since: number | null;
}

export interface SegmentRecovery {
  recovery_minutes: number;
  recovery_minutes_low: number;
  recovery_minutes_high: number;
  recovery_indeterminate: boolean;
  p_normal_in_30min: number | null;
  p_normal_in_60min: number;
  p_normal_in_120min: number;
}

export interface SegmentStatus {
  route: string;
  direction: string;
  from_stop: string;
  // Successor stop from the trainer's adjacency, null when the topology doc
  // wasn't available that tick. Load-bearing for attribution: a cell key names
  // only its from_stop, so at a branch or express split several drawn edges
  // claim it, and `to` is what says which hop the reading is actually about.
  to: string | null;
  // "quiet" is the quiet-normal call: too little scheduled through this cell
  // right now for an empty window to be evidence of anything.
  status: "normal" | "quiet" | "disrupted";
  entered_at: number;
  // Null on a NORMAL cell -- a healthy segment has nothing to forecast, so
  // this is never fabricated for one. Populated on a DISRUPTED cell only
  // once a trained dwell curve exists and its regime clock has started.
  recovery: SegmentRecovery | null;
}

// `segments` carries EVERY judged cell, normal and disrupted alike -- a key
// absent from it was never judged this tick, never a healthy read by
// omission. The whole surface is null when the Worker couldn't read its
// segment state.
//
// SIZING (why this is one dict, not the normal/disrupted split shipped and
// reverted the same day, 2026-08-23): measured on the live feed that day --
// base snapshot minus segment_flow 179.9 KB, a bare segment record (incl.
// its key) 151 B, a record carrying a recovery block 382 B. A normal cell
// is always bare (SegmentStatus.recovery is null on healthy track), so the
// shipped policy's ~701 judged cells/tick (~3 typically disrupted) totals
// ~284.0 KB, under the 300 KB line with ~120 KB (~814 bare records) of
// headroom before that line is even in question -- ~16% above today's
// population. Revisit only if judged volume climbs toward that ceiling or
// the disrupted share grows well past today's ~0.4%; not before.
export interface SegmentFlow {
  observed_at: number;
  segments: Record<string, SegmentStatus>;
}

export interface StationServiceFlow {
  // "quiet" when every segment touching the station is quiet-normal — nothing
  // much scheduled here right now, so neither flowing nor degraded.
  status: "flowing" | "quiet" | "degraded";
  worst_deficit: number;
  worst_segment: [string, string] | null;
  routes: string[];
  n_segments: number;
  worst_recovery: SegmentRecovery | null;
}

export interface StationFlow {
  observed_at: number;
  stations: Record<string, StationServiceFlow>;
}

// --- Platform crowding (mirrors worker/src/snapshot.ts + schema) ---

// One platform's estimate. The two inputs travel with the answer on purpose:
// waiting_riders is only correct as of PlatformCrowding.observed_at, and a
// consumer polling every 60s has to re-derive it against its own clock.
export interface PlatformCrowdingEstimate {
  last_train_at: number;
  // This platform's ASSUMED share of its complex's entry rate for the current
  // (weekday/weekend, hour) cell — see PlatformCrowdingMethod.split_basis.
  entries_per_min: number;
  waiting_riders: number;
}

// The constants and admitted assumptions behind every estimate in the surface,
// published rather than documented so a reader can reproduce the arithmetic.
export interface PlatformCrowdingMethod {
  formula: string;
  split_basis:
    | "uniform_over_served_platforms"
    | "scheduled_service_over_served_platforms";
  max_gap_minutes: number;
  served_window_minutes: number;
  excludes: string[];
  baseline_generated_at: number;
  baseline_window_start: string;
  baseline_window_end: string;
  // Present since the scheduled-service split; older snapshots omit them.
  service_weight_generated_at?: number | null;
  service_weight_feed_version?: string | null;
}

export interface PlatformCrowding {
  observed_at: number;
  method: PlatformCrowdingMethod;
  // Keyed by DIRECTIONAL GTFS stop id ('127N'). The parent station is the key
  // with its N/S suffix stripped (undirected(), same rule as the worker), and
  // its metadata lives in `stations` under that id. Platforms that cannot be
  // estimated are ABSENT, not zeroed; the reason is counted in `abstained`.
  platforms: Record<string, PlatformCrowdingEstimate>;
  n_platforms: number;
  abstained: Record<string, number>;
}

// Full alert record from snap.alerts — the resolvable detail behind the ids
// carried on RouteStatus.alerts. Mirrors worker/src/derive.ts AlertOut.
export interface Alert {
  id: string;
  alert_type: string;
  source: string;
  sort_order: number;
  active_period: Array<{ start?: number; end?: number }>;
  header_text: { translation: Array<{ text: string; language: string }> } | null;
  informed_entities: Array<{ route_id: string; direction_id?: number }>;
}

// --- Train position surface ---
// Published as its OWN object at v1/trains.json, a sibling of the snapshot
// rather than a field on it: measured at 36.2KB against a 185.8KB snapshot
// whose canonical consumer (a Home Assistant integration) can't use train
// positions, so it doesn't belong in the shared payload.
//
// Mirrors worker/src/snapshot.ts TrainsOut / TrainPositionOut. Aggregated: one
// entry per distinct (route, direction, stop, stopped) tuple, with `n`
// counting the trains that match it.

export interface TrainPosition {
  // Base route; the 6X/7X/FX express variants are folded into 6/7/F.
  route: string;
  // null when the feed's trip descriptor doesn't determine a direction.
  direction: "north" | "south" | null;
  // The DIRECTIONAL stop id exactly as NYCT reports it. Its meaning depends on
  // `stopped`: NYCT names the stop a train is HEADING TO while in transit, and
  // the stop it is AT once STOPPED_AT. There is deliberately no segment here —
  // which segment a moving train occupies is ambiguous at a branch, so the
  // surface refuses to guess and a consumer must not either.
  stop: string;
  // true = standing at the platform, false = en route toward it.
  stopped: boolean;
  // How many trains share this tuple.
  n: number;
}

// Build provenance for a published object: the git commit the Worker was
// deployed from, whether that deploy had uncommitted changes, and which
// component wrote the object.
export interface Provenance {
  code_sha: string;
  dirty: boolean | null;
  producer: string;
  // Identity of the trained params behind the snapshot's inference — trained_at
  // is the model version stamp, key the immutable versioned R2 object it maps to
  // (state/params/v<trained_at>.json). Both null means bootstrap params (no
  // params.json published yet). Present on the snapshot; absent on trains.json,
  // which carries no model, so it stays optional on this shared shape.
  params?: ParamsProvenance | null;
}

export interface ParamsProvenance {
  trained_at: number | null;
  key: string | null;
}

// Its own clock. On a tick where every vehicle feed fails the Worker skips
// rewriting the object rather than publishing an empty one, so a served body
// can be stale — `observed_at` is the only thing that says how stale, and it
// is NOT the snapshot's generated_at.
export interface Trains {
  observed_at: number;
  provenance: Provenance;
  // NYCT splits realtime vehicles across line-group feeds. `expected_feeds` is
  // the constant list the Worker polls every tick (currently 8: ace, bdfm, g,
  // jz, nqrw, l, numbered, si); `fresh_feeds` is the subset that decoded.
  //
  // When they differ, `positions` is a PARTIAL read: the routes behind a failed
  // feed have no entries, and that is silence rather than an empty platform.
  // The object deliberately carries no feed-name -> route-id mapping, because
  // NYCT's grouping has shuttle edge cases nobody here can verify, and a
  // guessed mapping would turn silence into a wrong per-route claim. So a
  // consumer may say how many feeds reported and which are missing, and may
  // NOT say which routes are affected.
  fresh_feeds: string[];
  expected_feeds: string[];
  positions: TrainPosition[];
}

export interface Snapshot {
  schema_version: string;
  generated_at: number;
  // Which code and which trained params produced this snapshot — see Provenance.
  // params.trained_at names the model version live right now, verifiable straight
  // off the public feed without reading R2.
  provenance: Provenance;
  attribution: string;
  freshness: Freshness;
  alerts: Alert[];
  route_status: Record<string, RouteStatus>;
  system: SystemStatus;
  compat: { subwaynow_routes: Record<string, CompatRoute> };
  stations: Record<string, Station>;
  station_status: Record<string, StationStatus>;
  station_flow: StationFlow | null;
  segment_flow: SegmentFlow | null;
  platform_crowding: PlatformCrowding | null;
}

// --- Grading streams (Phase B) ---

export interface PredictionRecord {
  ts: number;
  route: string;
  condition: string;
  p_normal: number;
  p_disrupted: number;
  p_suspended: number;
  regime_entered_at: number;
  recovery_minutes: number;
  recovery_minutes_low: number;
  recovery_minutes_high: number;
  recovery_indeterminate: boolean;
  p_normal_in_30min: number | null;
  // 60/120min are withheld (null) on records where the forecast came from a
  // fitted model — they measured worse than naive persistence (see
  // journal.md:1040-1051). Schedule-sourced rows are unaffected.
  // 30min is withheld when `recovery_source`'s arm isn't the one that
  // produced `condition`: graded against the published condition, mixing
  // movement- and hmm-sourced forecasts scores AUC 0.084 (worse than either
  // arm alone, since their probabilities aren't on the same scale).
  p_normal_in_60min: number | null;
  p_normal_in_120min: number | null;
  primary_alert_type: string | null;
  params_version: number;
  // Optional: present only on records written after schedule-based recovery
  // shipped. "schedule" rows are deterministic resume lookups, excluded from
  // HMM calibration and graded for adherence instead. "movement" rows come from
  // the movement dwell curve (worker/src/grading.ts:57 writes all three).
  recovery_source?: "hmm" | "schedule" | "movement";
  resumes_at?: number | null;
}

export interface TransitionRecord {
  ts: number;
  route: string;
  prev_state: string;
  new_state: string;
  regime_entered_at: number;
  exited_at: number;
  dwell_sec: number;
  alert_type_at_entry: string | null;
}

// --- /api/grading response (Phase B) ---

export interface HeatmapEntry {
  route: string;
  transition: number[][]; // 3x3, rows = from-state
}

export interface GradingResponse {
  configured: boolean;
  // "streams" = full credentialed read of the grading history; "calibration" =
  // aggregate-only view served from the public v1/calibration.json (no R2
  // credentials). Drilldowns (scatter, detection, schedule, swimlane, per-route
  // filtering) are absent in the calibration view.
  source?: "streams" | "calibration";
  error?: string;
  window: { days: number; from: string; to: string };
  counts: {
    predictionFiles: number;
    predictionRecords: number;
    transitionFiles: number;
    transitionRecords: number;
    alertFiles: number;
    alertVersions: number;
    alertsCapped: boolean;
    pointsCapped: boolean;
  };
  routes: string[];
  states: string[];
  // ReliabilityResult[] for horizons 30/60/120 and RecoveryResult — typed
  // structurally on the client to avoid importing server modules.
  reliability: unknown[];
  recovery: unknown;
  // ResumeChurnResult / AdherenceResult / DetectionLatencyResult — typed
  // structurally on the client.
  resumeChurn: unknown;
  adherence: unknown;
  detectionLatency: unknown;
  timelines: unknown[];
  heatmap: HeatmapEntry[];
  paramsTrainedAt: number | null;
  // Per-state self-loop ceiling the fit was clamped to (train_em.MAX_SELF_LOOP),
  // in `states` order. A diagonal sitting on its ceiling is a hyperparameter,
  // not a learned rate, so the heatmap marks those cells rather than presenting
  // them as this line's dynamics. Absent on feeds published before the cap shipped.
  paramsSelfLoopCap?: number[] | null;
  // Effective sample support behind every per-tick count above: the number of
  // independent incidents in the window. Ticks inside one regime are
  // autocorrelated, so a flat tick count can hide a 17.8x swing in real support.
  // Typed structurally on the client. Absent on older feeds and on the streams view.
  episodeSupport?: unknown;
  // The running model's own recovery grade, isolated from the retrains before
  // it. `recovery` above is pooled across every params version in the window,
  // so it belongs to no single model and cannot say whether a retrain helped.
  // Carries its own n_graded and a low_sample flag, because the newest version
  // holds roughly a quarter of the window by construction. Calibration view
  // only; typed structurally on the client.
  currentParams?: unknown;
  // When the underlying feed was generated (public aggregate). null on the
  // credentialed streams view, which reads live up to "now".
  generatedAt?: number | null;
  // RecoveryDistResult — recovery-time CDF + CRPS/PIT report. Streams view only;
  // typed structurally on the client.
  recoveryDist?: unknown;
  // RegimeBands — per-line stacked probability series over the window. Streams
  // view only (the aggregate feed carries no per-tick history); typed
  // structurally on the client.
  regimeBands?: unknown;
  // DriftDoc from the calibration feed — input-drift signals (unmapped
  // alert_type rate, emission-channel PSI). Absent on the streams view and on
  // older feeds; typed structurally on the client.
  drift?: unknown;
}
