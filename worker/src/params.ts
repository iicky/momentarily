/**
 * HMM parameters: read from R2 if present, else use bootstrap defaults.
 *
 * Python training writes params.json to r2://momentarily/state/params.json.
 * If absent (first deploy, training hasn't run yet), the Worker falls back to
 * a small set of hand-picked params so the forward filter still produces
 * sensible output.
 */

import type { EmissionParams, HMMParams } from './hmm';

const PARAMS_KEY = 'state/params.json';

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
 * Per-route trained params from Python. When a route is missing, the Worker
 * uses the global bootstrap params.
 *
 * Schema is intentionally loose — Python is the producer, we accept what it
 * sends and validate the shape minimally. Adding a JSON Schema artifact and
 * Zod-validating is a follow-up.
 */
export interface TrainedParams {
  schema_version: string;
  trained_at: number;
  routes: Record<string, HMMParams>;
}

/**
 * Load trained params from R2. Returns null if not yet present (first deploy
 * before Python EM has written anything).
 */
export async function loadParams(bucket: R2Bucket): Promise<TrainedParams | null> {
  const obj = await bucket.get(PARAMS_KEY);
  if (!obj) return null;
  try {
    const data = (await obj.json()) as TrainedParams;
    if (!data || typeof data !== 'object' || !data.routes) {
      console.error('params.json shape invalid; using bootstrap');
      return null;
    }
    return data;
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
