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
  to: string | null;
  status: "normal" | "disrupted";
  entered_at: number;
  recovery: SegmentRecovery | null;
}

export interface SegmentFlow {
  observed_at: number;
  segments: Record<string, SegmentStatus>;
}

export interface StationServiceFlow {
  status: "flowing" | "degraded";
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
  split_basis: "uniform_over_served_platforms";
  max_gap_minutes: number;
  served_window_minutes: number;
  excludes: string[];
  baseline_generated_at: number;
  baseline_window_start: string;
  baseline_window_end: string;
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

export interface Snapshot {
  schema_version: string;
  generated_at: number;
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
