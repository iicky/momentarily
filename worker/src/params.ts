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

const Vec3Schema = z.tuple([
  z.number().finite(),
  z.number().finite(),
  z.number().finite(),
]);

const EmissionParamsSchema = z.object({
  poisson_lambda: Vec3Schema,
  gamma_alpha: Vec3Schema,
  gamma_beta: Vec3Schema,
  bernoulli_p: Vec3Schema,
  bernoulli_p_delays: Vec3Schema,
  bernoulli_p_service_change: Vec3Schema,
  bernoulli_p_planned: Vec3Schema,
});

const DwellQuantilesSchema = z.object({
  n: z.number().int().nonnegative(),
  q25_sec: z.number().int().nonnegative(),
  median_sec: z.number().int().nonnegative(),
  q75_sec: z.number().int().nonnegative(),
});

// Per-route, per-prev-state empirical dwell quantiles from the trainer.
// Keys are the same state names the worker uses: "normal" / "disrupted" /
// "suspended". Cells the trainer didn't include (sample size below its
// floor) simply aren't here and the worker falls back to its geometric
// estimate. See momentarily-w97.
const DwellByStateSchema = z.record(z.string(), DwellQuantilesSchema).optional();

const HMMParamsSchema = z.object({
  transition: z.tuple([Vec3Schema, Vec3Schema, Vec3Schema]),
  initial: Vec3Schema,
  emissions: EmissionParamsSchema,
  emissions_by_bin: z.array(EmissionParamsSchema).length(N_TOD_BINS).optional(),
  dwell_quantiles: DwellByStateSchema,
});

export interface DwellQuantiles {
  n: number;
  q25_sec: number;
  median_sec: number;
  q75_sec: number;
}

export type DwellByState = Record<string, DwellQuantiles>;

const TrainedParamsWrapperSchema = z.object({
  schema_version: z.string(),
  trained_at: z.number().finite(),
  // Validate each route separately (below) so one bad route doesn't drop the
  // whole upload — the others should still apply.
  routes: z.record(z.string(), z.unknown()),
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
  let dropped = 0;
  for (const [routeId, raw] of Object.entries(wrapper.data.routes)) {
    const parsed = HMMParamsSchema.safeParse(raw);
    if (parsed.success) {
      routes[routeId] = toHMMParams(parsed.data);
      if (parsed.data.dwell_quantiles) {
        dwell[routeId] = parsed.data.dwell_quantiles;
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
  return {
    schema_version: wrapper.data.schema_version,
    trained_at: wrapper.data.trained_at,
    routes,
    dwell,
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
 * Empirical dwell quantiles for a (route, state) cell. Returns null when the
 * trainer didn't include one — caller should fall back to its analytic
 * (geometric self-loop) estimate.
 */
export function dwellForRouteState(
  trained: TrainedParams | null,
  routeId: string,
  state: string,
): DwellQuantiles | null {
  return trained?.dwell?.[routeId]?.[state] ?? null;
}
