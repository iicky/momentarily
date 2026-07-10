/**
 * HMM parameters: read from R2 if present, else use bootstrap defaults.
 *
 * Python training writes params.json to r2://momentarily/state/params.json.
 * If absent (first deploy, training hasn't run yet), the Worker falls back to
 * a small set of hand-picked params so the forward filter still produces
 * sensible output.
 */

import { z } from 'zod';

import type { EmissionParams, HMMParams } from './hmm';
import { N_TOD_BINS } from './hmm';

const PARAMS_KEY = 'state/params.json';

// Semantic bounds, not just finiteness: a malformed-but-finite trainer upload
// (negative mass, a transition row that doesn't sum to 1, a probability > 1)
// would otherwise pass shape validation and feed invalid numbers straight into
// the forward filter. Each domain gets the tightest vector that still admits
// every legitimate trained value.
const STOCHASTIC_EPS = 1e-3; // row-sum tolerance; EM output is exact to float

const nonNeg = z.number().finite().nonnegative();
const positive = z.number().finite().positive();
const prob = z.number().finite().min(0).max(1);

// Poisson rate ≥ 0 (a state may genuinely never emit events).
const RateVec3 = z.tuple([nonNeg, nonNeg, nonNeg]);
// Gamma shape/rate must be strictly positive for a proper density.
const PosVec3 = z.tuple([positive, positive, positive]);
// Bernoulli emission probabilities in [0, 1].
const ProbVec3 = z.tuple([prob, prob, prob]);
// A discrete distribution over the 3 states: each in [0, 1] and summing to 1.
const StochasticVec3 = z
  .tuple([prob, prob, prob])
  .refine((v) => Math.abs(v[0] + v[1] + v[2] - 1) <= STOCHASTIC_EPS, {
    message: 'must sum to 1',
  });

const EmissionParamsSchema = z.object({
  poisson_lambda: RateVec3,
  gamma_alpha: PosVec3,
  gamma_beta: PosVec3,
  bernoulli_p: ProbVec3,
  bernoulli_p_delays: ProbVec3,
  bernoulli_p_service_change: ProbVec3,
  bernoulli_p_planned: ProbVec3,
  // Per-state matched-trip advance rate. Optional for back-compat with
  // params.json written before the movement channel (vhh.4).
  advance_rate: ProbVec3.optional(),
  // Per-state service-ratio Gaussian (assigned_n / baseline): mu is a ratio
  // (>=0, may exceed 1), sigma a std (>=0). Optional for back-compat with
  // params.json written before the service channel.
  service_mu: RateVec3.optional(),
  service_sigma: RateVec3.optional(),
});

const DwellQuantilesSchema = z.object({
  n: z.number().int().nonnegative(),
  q25_sec: z.number().int().nonnegative(),
  median_sec: z.number().int().nonnegative(),
  q75_sec: z.number().int().nonnegative(),
  // Empirical P(dwell <= horizon). Optional for back-compat with params.json
  // written before the recovery-probability work.
  recover_by_30: z.number().min(0).max(1).optional(),
  recover_by_60: z.number().min(0).max(1).optional(),
  recover_by_120: z.number().min(0).max(1).optional(),
  // Full dwell distribution as quantiles at evenly spaced probabilities — lets
  // the Worker condition recovery outputs on elapsed regime age (see
  // worker/src/dwell.ts). Optional for back-compat. See momentarily-vk0.1.
  curve_sec: z.array(z.number().nonnegative()).min(2).optional(),
  // [shape, scale] of a log-logistic fit to this cell's dwells. pLeaveBy uses it
  // to extrapolate the tail past the last observed quantile instead of the
  // constant-hazard exponential patch. Optional for back-compat with older
  // params.json. See momentarily-gtq.5.
  tail_ll: z.tuple([z.number().positive(), z.number().positive()]).optional(),
});

// Per-route, per-prev-state empirical dwell quantiles from the trainer.
// Keys are the same state names the worker uses: "normal" / "disrupted" /
// "suspended". Cells the trainer didn't include (sample size below its
// floor) simply aren't here and the worker falls back to its geometric
// estimate. See momentarily-w97.
const DwellByStateSchema = z.record(z.string(), DwellQuantilesSchema).optional();

// Cause-segmented dwell: state -> alert_type -> quantiles. Layered on top of
// dwell_quantiles; the worker prefers a (state, alert_type) cell and falls back
// to the (state) aggregate when one is absent. See momentarily-alu.
const DwellByStateAlertSchema = z
  .record(z.string(), z.record(z.string(), DwellQuantilesSchema))
  .optional();

const HMMParamsSchema = z.object({
  transition: z.tuple([StochasticVec3, StochasticVec3, StochasticVec3]),
  initial: StochasticVec3,
  emissions: EmissionParamsSchema,
  emissions_by_bin: z.array(EmissionParamsSchema).length(N_TOD_BINS).optional(),
  dwell_quantiles: DwellByStateSchema,
  dwell_quantiles_by_alert: DwellByStateAlertSchema,
});

export interface DwellQuantiles {
  n: number;
  q25_sec: number;
  median_sec: number;
  q75_sec: number;
  recover_by_30?: number | undefined;
  recover_by_60?: number | undefined;
  recover_by_120?: number | undefined;
  curve_sec?: number[] | undefined;
  tail_ll?: [number, number] | undefined;
}

export type DwellByState = Record<string, DwellQuantiles>;

// state -> alert_type -> quantiles
export type DwellByStateAlert = Record<string, Record<string, DwellQuantiles>>;

// Per-(route, direction, tod_bin) advance-rate baseline (momentarily-vhh.3/vhh.5).
// p0 is the cell's normal cross-tick advance fraction; alpha/beta carry it as a
// Beta prior for the movement emission. The Worker uses it live to gate and
// score the movement channel.
const AdvanceBaselineCellSchema = z.object({
  p0: prob,
  alpha: positive,
  beta: positive,
  n: z.number().int().nonnegative(),
});
// route -> direction -> tod_bin (stringified int) -> cell
const MovementBaselineSchema = z.record(
  z.string(),
  z.record(z.string(), z.record(z.string(), AdvanceBaselineCellSchema)),
);

export type AdvanceBaselineCell = z.infer<typeof AdvanceBaselineCellSchema>;
export type MovementBaseline = z.infer<typeof MovementBaselineSchema>;

// route -> tod_bin (stringified int) -> median assigned_n. The Worker divides
// live assigned_n by this to form the service ratio the emission scores.
const ServiceBaselineSchema = z.record(z.string(), z.record(z.string(), nonNeg));
export type ServiceBaseline = z.infer<typeof ServiceBaselineSchema>;

// route -> schedule_bin (e.g. `wd06`) -> in-service rate in [0,1]: the share of
// usable ticks the route was running at that (weekend, hour) bin. The Worker
// reads it to split a no-service reading into suspended vs not_scheduled.
const ScheduleRateSchema = z.record(z.string(), z.record(z.string(), prob));
export type ScheduleRate = z.infer<typeof ScheduleRateSchema>;

const TrainedParamsWrapperSchema = z.object({
  schema_version: z.string(),
  trained_at: z.number().finite(),
  // Validate each route separately (below) so one bad route doesn't drop the
  // whole upload — the others should still apply.
  routes: z.record(z.string(), z.unknown()),
  // Validated separately too, so a malformed baseline degrades the movement
  // channel only, not the whole params upload.
  movement_baseline: z.unknown().optional(),
  // Validated separately too, so a malformed service baseline degrades the
  // service channel only, not the whole params upload.
  service_baseline: z.unknown().optional(),
  // Validated separately too, so a malformed schedule rate degrades the
  // suspended/not_scheduled split only, not the whole params upload.
  schedule_rate: z.unknown().optional(),
});

// The three "kind of disruption" flags (delays/service_change/planned) all
// indicate `disrupted`, not `suspended` — only has_suspended_alert
// (bernoulli_p) should pull hard toward suspended. Before this, all three
// leaned suspended, so any persistent planned-work/delay alert drifted routes
// into `suspended`. See momentarily-x5b.
const BOOTSTRAP_EMISSIONS: EmissionParams = {
  poisson_lambda: [0.3, 4.0, 12.0],
  gamma_alpha: [1.0, 3.0, 6.0],
  gamma_beta: [2.0, 0.4, 0.2],
  bernoulli_p: [0.001, 0.05, 0.95],
  bernoulli_p_delays: [0.02, 0.6, 0.35],
  bernoulli_p_service_change: [0.02, 0.6, 0.4],
  bernoulli_p_planned: [0.05, 0.6, 0.35],
};

export const BOOTSTRAP_PARAMS: HMMParams = {
  transition: [
    [0.95, 0.04, 0.01],
    [0.08, 0.9, 0.02],
    [0.02, 0.1, 0.88],
  ],
  initial: [0.9, 0.08, 0.02],
  emissions: BOOTSTRAP_EMISSIONS,
};

/**
 * Per-route trained params from Python. When a route is missing — or its
 * entry failed shape validation — the Worker uses the global bootstrap.
 *
 * `dwell` carries the optional empirical regime-dwell quantiles sidecar
 * (sample-based, computed from v1/regime_transitions). The Worker uses these
 * to set recovery_minutes when present; absent cells fall back to the
 * geometric dwell from the trained transition self-loop.
 */
export interface TrainedParams {
  schema_version: string;
  trained_at: number;
  routes: Record<string, HMMParams>;
  dwell: Record<string, DwellByState>;
  // Cause-segmented dwell sidecar, route -> state -> alert_type -> quantiles.
  // Preferred over `dwell` when the current regime's alert_type has a cell.
  dwellByAlert: Record<string, DwellByStateAlert>;
  // Per-(route, direction, tod_bin) advance-rate baseline for the movement
  // channel. Empty until the trainer has ~2wk of by_direction archive.
  movementBaseline: MovementBaseline;
  // Per-(route, tod_bin) assigned_n baseline for the service emission channel.
  serviceBaseline: ServiceBaseline;
  // Per-(route, schedule_bin) in-service rate; the Worker splits a no-service
  // reading into suspended (normally runs now) vs not_scheduled (rush-only gap).
  scheduleRate: ScheduleRate;
}

/**
 * Strip optional emissions_by_bin when absent so the result is assignable to
 * HMMParams under exactOptionalPropertyTypes.
 */
function toHMMParams(p: z.infer<typeof HMMParamsSchema>): HMMParams {
  if (p.emissions_by_bin !== undefined) {
    return {
      transition: p.transition,
      initial: p.initial,
      emissions: p.emissions,
      emissions_by_bin: p.emissions_by_bin,
    };
  }
  return {
    transition: p.transition,
    initial: p.initial,
    emissions: p.emissions,
  };
}

/**
 * Validate the trained-params document. A failed wrapper (wrong top-level
 * shape) returns null and the Worker falls back to bootstrap for every route.
 * A failed *route* is dropped from the returned map and that single route
 * falls back to bootstrap via paramsForRoute, so one bad upload row can't
 * NaN-poison the rest of the fleet. See momentarily-30o.
 */
export function parseTrainedParams(data: unknown): TrainedParams | null {
  const wrapper = TrainedParamsWrapperSchema.safeParse(data);
  if (!wrapper.success) {
    console.error('params.json wrapper invalid; using bootstrap:', wrapper.error.issues);
    return null;
  }
  const routes: Record<string, HMMParams> = {};
  const dwell: Record<string, DwellByState> = {};
  const dwellByAlert: Record<string, DwellByStateAlert> = {};
  let dropped = 0;
  for (const [routeId, raw] of Object.entries(wrapper.data.routes)) {
    const parsed = HMMParamsSchema.safeParse(raw);
    if (parsed.success) {
      routes[routeId] = toHMMParams(parsed.data);
      if (parsed.data.dwell_quantiles) {
        dwell[routeId] = parsed.data.dwell_quantiles;
      }
      if (parsed.data.dwell_quantiles_by_alert) {
        dwellByAlert[routeId] = parsed.data.dwell_quantiles_by_alert;
      }
    } else {
      dropped += 1;
      console.warn(
        `params.json route ${routeId} failed validation; falling back to bootstrap:`,
        parsed.error.issues,
      );
    }
  }
  if (dropped > 0) {
    console.warn(`params.json: ${dropped} route(s) dropped; bootstrap will fill in`);
  }

  // Movement baseline is optional and validated on its own: a malformed baseline
  // disables the movement channel but leaves the rest of the params intact.
  let movementBaseline: MovementBaseline = {};
  if (wrapper.data.movement_baseline !== undefined) {
    const parsed = MovementBaselineSchema.safeParse(wrapper.data.movement_baseline);
    if (parsed.success) {
      movementBaseline = parsed.data;
    } else {
      console.warn('params.json movement_baseline invalid; movement channel off:', parsed.error.issues);
    }
  }

  // Service baseline, validated on its own like the movement baseline.
  let serviceBaseline: ServiceBaseline = {};
  if (wrapper.data.service_baseline !== undefined) {
    const parsed = ServiceBaselineSchema.safeParse(wrapper.data.service_baseline);
    if (parsed.success) {
      serviceBaseline = parsed.data;
    } else {
      console.warn('params.json service_baseline invalid; service channel off:', parsed.error.issues);
    }
  }

  // Schedule rate, validated on its own like the baselines.
  let scheduleRate: ScheduleRate = {};
  if (wrapper.data.schedule_rate !== undefined) {
    const parsed = ScheduleRateSchema.safeParse(wrapper.data.schedule_rate);
    if (parsed.success) {
      scheduleRate = parsed.data;
    } else {
      console.warn('params.json schedule_rate invalid; suspended/not_scheduled split off:', parsed.error.issues);
    }
  }

  return {
    schema_version: wrapper.data.schema_version,
    trained_at: wrapper.data.trained_at,
    routes,
    dwell,
    dwellByAlert,
    movementBaseline,
    serviceBaseline,
    scheduleRate,
  };
}

/**
 * Load trained params from R2. Returns null if not yet present (first deploy
 * before Python EM has written anything) or if the document is malformed.
 */
export async function loadParams(bucket: R2Bucket): Promise<TrainedParams | null> {
  const obj = await bucket.get(PARAMS_KEY);
  if (!obj) return null;
  try {
    return parseTrainedParams(await obj.json());
  } catch (err) {
    console.error('params.json parse failed; using bootstrap:', err);
    return null;
  }
}

/**
 * Resolve params for a specific route: trained params if available, else
 * bootstrap.
 */
export function paramsForRoute(
  trained: TrainedParams | null,
  routeId: string,
): HMMParams {
  return trained?.routes?.[routeId] ?? BOOTSTRAP_PARAMS;
}

/**
 * Advance-rate baseline for a (route, direction, tod_bin) cell, or null when the
 * trainer hasn't established one yet. A null baseline is the signal to drop the
 * movement channel out for that cell (Observation.has_movement = false).
 */
export function advanceBaselineFor(
  trained: TrainedParams | null,
  routeId: string,
  direction: string,
  todBin: number,
): AdvanceBaselineCell | null {
  return trained?.movementBaseline?.[routeId]?.[direction]?.[String(todBin)] ?? null;
}

/**
 * Median assigned_n for a (route, tod_bin) cell, or null when the trainer hasn't
 * established one yet. A null baseline drops the service channel out for that
 * cell (Observation.has_service = false).
 */
export function serviceBaselineFor(
  trained: TrainedParams | null,
  routeId: string,
  todBin: number,
): number | null {
  return trained?.serviceBaseline?.[routeId]?.[String(todBin)] ?? null;
}

/**
 * In-service rate for a (route, schedule_bin) cell, or null when the trainer
 * hasn't established one yet. A null rate means "unknown schedule" — the caller
 * keeps a no-service reading as suspended rather than downgrading it.
 */
export function scheduleRateFor(
  trained: TrainedParams | null,
  routeId: string,
  scheduleBin: string,
): number | null {
  return trained?.scheduleRate?.[routeId]?.[scheduleBin] ?? null;
}

/**
 * Empirical dwell quantiles for a regime, most-specific first:
 *   1. (route, state, alertType) — cause-segmented, when alertType is given
 *      and the trainer has that cell.
 *   2. (route, state) — the aggregate across causes.
 *   3. null — caller falls back to its analytic (geometric self-loop) estimate.
 *
 * The cause-conditioned cell is preferred because dwell under e.g. planned work
 * is structurally different from delays; conditioning tightens the interval.
 * See momentarily-alu.
 */
export function dwellForRouteState(
  trained: TrainedParams | null,
  routeId: string,
  state: string,
  alertType: string | null = null,
): DwellQuantiles | null {
  if (alertType !== null) {
    const byCause = trained?.dwellByAlert?.[routeId]?.[state]?.[alertType];
    if (byCause) return byCause;
  }
  return trained?.dwell?.[routeId]?.[state] ?? null;
}
