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
  p_normal_in_30min: number;
  p_normal_in_60min: number;
  p_normal_in_120min: number;
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

export interface Snapshot {
  schema_version: string;
  generated_at: number;
  attribution: string;
  freshness: Freshness;
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
  p_normal_in_30min: number;
  p_normal_in_60min: number;
  p_normal_in_120min: number;
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
}
