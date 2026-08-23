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

export interface CalibrationStratum {
  n: number;
  brier: number | null;
  bss_persistence: number | null;
  // Sharpness vs realized rate. The pair is the tell for a degenerate forecast:
  // mean_pred 0.99 against mean_outcome 0.50 is not a calibration nit, it is a
  // forecast that ignores the state it is conditioned on. Optional: feeds
  // published before the sharpness fields landed carry only n and skill.
  mean_pred?: number | null;
  mean_outcome?: number | null;
  auc?: number | null;
}

export interface CalibrationHorizon {
  horizon_min: number;
  n: number;
  brier: number | null;
  brier_persistence: number | null;
  brier_climatology: number | null;
  bss_persistence: number | null;
  bss_climatology: number | null;
  auc?: number | null;
  excluded_schedule?: number;
  // Split by the current condition at T — "normal_now" (the sticky-regime case)
  // vs "not_normal_now" (the recovery forecast), so the UI can show which slice
  // drags the overall skill negative.
  by_current?: Record<string, CalibrationStratum>;
  bins: {
    bin_lo: number;
    bin_hi: number;
    n: number;
    mean_pred: number | null;
    mean_outcome: number | null;
  }[];
}

export interface MovementCoverage {
  n_ticks: number;
  unknown_share: number;
  gradeable_share: number;
  by_condition: Record<string, number>;
}

/** Effective sample support behind the per-tick counts. Ticks inside one regime
 * are autocorrelated, so the episode count — not the tick count — is how much
 * independent evidence the window carries. Measured on the published arm, six
 * consecutive 7-day windows carried indistinguishable tick counts (56k-58k) and
 * 5 to 89 episodes: a 17.8x spread in real support behind a flat advertised n. */
export interface EpisodeSupport {
  graded_arm: string;
  n_episodes: number;
  n_left_censored: number;
  n_right_censored: number;
  n_standing: number;
  standing_tick_share: number;
  tick_rows: number;
  // Rows dropped for predating the published arm; a widened window can exclude
  // most of its own span this way.
  excluded_pre_arm_rows?: number;
  // The span the arm actually covers, which can be far shorter than the window.
  covered?: { start: number; end: number } | null;
}

export interface CalibrationDoc {
  generated_at: number;
  window: { start: number; end: number };
  predictions_seen: number;
  transitions_seen: number;
  // Absent on feeds published before effective support shipped.
  episode_support?: EpisodeSupport;
  // Present only on calibration.json published after the drift work; older
  // feeds omit it, so the panel is gated on its presence.
  drift?: DriftDoc;
  // One entry per horizon. `auc` is the metric Brier cannot supply: when the
  // outcome is ~99% one class, a degenerate forecast still posts a small Brier,
  // and only rank discrimination shows it is pointed the wrong way.
  calibration: CalibrationHorizon[];
  // Which condition arm `calibration` graded against. Absent on feeds published
  // before both arms shipped, where the block is the alert-shadow one.
  calibration_arm?: string;
  // The SAME forecast graded against the published movement-primary arm — the
  // one p_normal_in_H is actually a forecast of. Absent on older feeds.
  calibration_movement?: {
    graded_arm: string;
    // How much of the window the movement arm could judge, over the scope this
    // block covers. `unknown_share` is the fraction of ticks it had no reading
    // for; those are dropped rather than scored as calm, so a high share means
    // this arm's n is a thin slice of the window.
    coverage?: MovementCoverage;
    horizons: CalibrationHorizon[];
  };
  recovery: {
    overall: CalibrationRecoveryStats;
    per_regime: CalibrationRecoveryStats;
  };
  transition_matrices: {
    trained_at: number | null;
    states: string[];
    routes: Record<string, number[][]>;
    // Per-state self-loop ceiling the fit was clamped to, in `states` order.
    // Absent on feeds published before the cap shipped.
    self_loop_cap?: number[] | null;
  };
  // Per-route reliability + recovery summary (training/eval.build_by_line), so
  // the Models tab can filter by line straight off this static feed. Absent on
  // feeds published before the per-line work.
  by_line?: Record<
    string,
    {
      n_predictions: number;
      calibration: CalibrationHorizon[];
      // Null when the publisher had no movement truth map; absent on older feeds.
      calibration_movement?: CalibrationHorizon[] | null;
      // THIS route's movement coverage. Never substitute the window-aggregate
      // rate here: the chip renders as the selected route's own coverage, and
      // the aggregate can be true of no individual route.
      movement_coverage?: MovementCoverage | null;
      // THIS route's incident count, for the same reason: the line filter also
      // swaps the adjacent prediction count to this route's, so the aggregate
      // would claim the whole system's incidents support one line.
      episode_support?: EpisodeSupport | null;
      recovery: {
        overall: CalibrationRecoveryStats;
        per_regime: CalibrationRecoveryStats;
      };
    }
  >;
}

export async function fetchCalibration(base = FEED_BASE): Promise<CalibrationDoc> {
  const res = await fetch(`${base}/v1/calibration.json`, { cache: "no-store" });
  if (!res.ok) throw new Error(`calibration fetch failed: ${res.status}`);
  return res.json();
}

// Reshape the published bins into the shape the reliability chart draws (bin
// midpoint, predicted/observed means), carrying the skill scores, AUC, and the
// normal-now/not-normal-now decomposition.
//
// One horizon can be graded by more than one arm. p_normal_in_H is a movement-arm
// forecast, so the movement grading is the one that answers "was the forecast
// right"; the shadow grading answers "does the alert filter agree with it". They
// disagreed by 0.55 AUC on the 2026-08-22 feed, so the chart draws both rather
// than picking — see training/eval.build_calibration.
export interface ReliabilityStratum {
  n: number;
  bss: number | null;
  meanPred: number | null;
  meanOutcome: number | null;
}

export interface ReliabilityArm {
  /** Feed-supplied arm label, e.g. "published_condition (movement-primary)". */
  arm: string;
  /** True for the movement arm: the forecast's own target, not the filter's. */
  isForecastTarget: boolean;
  n: number;
  brier: number;
  auc: number | null;
  skillPersistence: number | null;
  skillClimatology: number | null;
  /** Fraction of ticks this arm had no reading for and therefore dropped. */
  unknownShare?: number;
  bins: { p: number; predictedMean: number; observedFreq: number; n: number }[];
  decomp?: {
    normalNow?: ReliabilityStratum;
    notNormalNow?: ReliabilityStratum;
  };
}

export interface AggregateReliability {
  horizonMin: number;
  excludedSchedule: number;
  arms: ReliabilityArm[];
}

const SHADOW_FALLBACK_ARM = "condition (alert-shadow)";

function stratum(s: CalibrationStratum | undefined): ReliabilityStratum | undefined {
  if (!s) return undefined;
  return {
    n: s.n,
    bss: s.bss_persistence,
    meanPred: s.mean_pred ?? null,
    meanOutcome: s.mean_outcome ?? null,
  };
}

export function reshapeArm(
  c: CalibrationHorizon,
  arm: string,
  isForecastTarget: boolean,
  unknownShare?: number,
): ReliabilityArm {
  const nn = stratum(c.by_current?.normal_now);
  const xn = stratum(c.by_current?.not_normal_now);
  return {
    arm,
    isForecastTarget,
    n: c.n,
    brier: c.brier ?? NaN,
    auc: c.auc ?? null,
    skillPersistence: c.bss_persistence,
    skillClimatology: c.bss_climatology,
    unknownShare,
    decomp: nn || xn ? { normalNow: nn, notNormalNow: xn } : undefined,
    bins: c.bins.map((b) => ({
      p: (b.bin_lo + b.bin_hi) / 2,
      predictedMean: b.mean_pred ?? NaN,
      observedFreq: b.mean_outcome ?? NaN,
      n: b.n,
    })),
  };
}

/** Pair the two gradings by horizon. The movement arm leads: it is the one the
 * forecast is about, so it should be read first even when it is the thinner
 * sample. Horizons come from the shadow block, which is always published. */
function reshapeReliability(
  shadow: CalibrationHorizon[],
  movement: CalibrationHorizon[] | null | undefined,
  shadowArm: string,
  movementArm: string,
  unknownShare?: number,
): AggregateReliability[] {
  return shadow.map((c) => {
    // Three horizons; a linear scan beats building a lookup for each call.
    const m = (movement ?? []).find((x) => x.horizon_min === c.horizon_min);
    return {
      horizonMin: c.horizon_min,
      excludedSchedule: c.excluded_schedule ?? 0,
      arms: [
        ...(m ? [reshapeArm(m, movementArm, true, unknownShare)] : []),
        reshapeArm(c, shadowArm, false),
      ],
    };
  });
}

export function calibrationReliability(doc: CalibrationDoc): AggregateReliability[] {
  return reshapeReliability(
    doc.calibration,
    doc.calibration_movement?.horizons,
    doc.calibration_arm ?? SHADOW_FALLBACK_ARM,
    doc.calibration_movement?.graded_arm ?? "",
    doc.calibration_movement?.coverage?.unknown_share,
  );
}

/** Routes the static feed carries a per-line breakdown for, sorted. */
export function calibrationRoutes(doc: CalibrationDoc): string[] {
  return Object.keys(doc.by_line ?? {}).sort((a, b) =>
    a.localeCompare(b, undefined, { numeric: true }),
  );
}

/** Per-line reliability, recovery, and effective support from the static feed, or
 * null when the feed has no breakdown for this route (thin line, or a
 * pre-per-line feed). Every field is route-scoped: the caller swaps the whole
 * counts line to this route, so an aggregate leaking through here reads as a
 * claim about the selected line. */
export function calibrationForLine(
  doc: CalibrationDoc,
  route: string,
): {
  reliability: AggregateReliability[];
  recovery: CalibrationDoc["recovery"];
  episodeSupport: EpisodeSupport | undefined;
} | null {
  const bl = doc.by_line?.[route];
  if (!bl) return null;
  return {
    reliability: reshapeReliability(
      bl.calibration,
      bl.calibration_movement,
      doc.calibration_arm ?? SHADOW_FALLBACK_ARM,
      doc.calibration_movement?.graded_arm ?? "",
      // This route's own coverage, never the window aggregate. Undefined on
      // feeds published before per-route coverage, which suppresses the chip
      // rather than showing a rate that isn't about this line.
      bl.movement_coverage?.unknown_share,
    ),
    recovery: bl.recovery,
    // Undefined, never the aggregate: a route with no published support must show
    // no incident count rather than the system's.
    episodeSupport: bl.episode_support ?? undefined,
  };
}

export function calibrationHeatmap(doc: CalibrationDoc): HeatmapEntry[] {
  return Object.entries(doc.transition_matrices.routes)
    .map(([route, transition]) => ({ route, transition }))
    .filter((h) => h.transition.length === 3)
    .sort((a, b) => a.route.localeCompare(b.route, undefined, { numeric: true }));
}
