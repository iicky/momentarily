// Public-feed path for the Models tab: read the compact v1/calibration.json the
// trainer publishes (training/eval.py) instead of LISTing the raw grading
// streams through the credentialed R2 S3 API. This is what lets a browser-only /
// hosted viz draw the aggregate reliability, recovery, and transition charts
// with no R2 credentials — the public Worker doesn't expose LIST, but it does
// serve this single object.

import type { HeatmapEntry } from "./types";

// Same public feed as lib/feed.ts; kept as a value import-free local so the
// mappers below can be unit-tested under `node --test` without pulling the
// snapshot module's runtime deps. Override with NEXT_PUBLIC_FEED_BASE.
const FEED_BASE =
  process.env.NEXT_PUBLIC_FEED_BASE ?? "https://feed.momentarily.nyc";

export interface CalibrationRecoveryStats {
  n: number;
  mae_min: number | null;
  rmse_min: number | null;
  iqr_coverage: number | null;
}

export interface DriftDoc {
  unmapped_alert_type: {
    n_typed_ticks: number;
    unmapped_rate: number;
    unmapped_types: Record<string, number>;
    by_route: Record<string, number>;
  };
  emission_channels: {
    available: boolean;
    cells_scored?: number;
    cells_skipped_thin?: number;
    psi_threshold?: number;
    routes_drifted?: string[];
    by_route?: Record<
      string,
      {
        max_alert_count_psi: number;
        max_flag_delta: number;
        max_flag_delta_channel: string | null;
        n_cells: number;
        significant: boolean;
      }
    >;
  };
}

export interface CalibrationDoc {
  generated_at: number;
  window: { start: number; end: number };
  predictions_seen: number;
  transitions_seen: number;
  // Present only on calibration.json published after the drift work; older
  // feeds omit it, so the panel is gated on its presence.
  drift?: DriftDoc;
  calibration: {
    horizon_min: number;
    n: number;
    brier: number | null;
    brier_persistence: number | null;
    brier_climatology: number | null;
    bss_persistence: number | null;
    bss_climatology: number | null;
    bins: {
      bin_lo: number;
      bin_hi: number;
      n: number;
      mean_pred: number | null;
      mean_outcome: number | null;
    }[];
  }[];
  recovery: {
    overall: CalibrationRecoveryStats;
    per_regime: CalibrationRecoveryStats;
  };
  transition_matrices: {
    trained_at: number | null;
    states: string[];
    routes: Record<string, number[][]>;
  };
}

export async function fetchCalibration(base = FEED_BASE): Promise<CalibrationDoc> {
  const res = await fetch(`${base}/v1/calibration.json`, { cache: "no-store" });
  if (!res.ok) throw new Error(`calibration fetch failed: ${res.status}`);
  return res.json();
}

// Reshape the published bins into the same ReliabilityResult the client charts
// expect (bin midpoint, predicted/observed means). excludedSchedule isn't
// carried in the aggregate — the published Brier is over whatever the grader
// scored — so it's reported as 0.
export interface AggregateReliability {
  horizonMin: number;
  bins: { p: number; predictedMean: number; observedFreq: number; n: number }[];
  brier: number;
  n: number;
  excludedSchedule: number;
}

export function calibrationReliability(doc: CalibrationDoc): AggregateReliability[] {
  return doc.calibration.map((c) => ({
    horizonMin: c.horizon_min,
    n: c.n,
    brier: c.brier ?? NaN,
    excludedSchedule: 0,
    bins: c.bins.map((b) => ({
      p: (b.bin_lo + b.bin_hi) / 2,
      predictedMean: b.mean_pred ?? NaN,
      observedFreq: b.mean_outcome ?? NaN,
      n: b.n,
    })),
  }));
}

export function calibrationHeatmap(doc: CalibrationDoc): HeatmapEntry[] {
  return Object.entries(doc.transition_matrices.routes)
    .map(([route, transition]) => ({ route, transition }))
    .filter((h) => h.transition.length === 3)
    .sort((a, b) => a.route.localeCompare(b.route, undefined, { numeric: true }));
}
