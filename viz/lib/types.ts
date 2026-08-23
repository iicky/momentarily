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
  // HMM calibration and graded for adherence instead.
  recovery_source?: "hmm" | "schedule";
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
