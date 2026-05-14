/**
 * Per-line Hidden Markov Model over transit service state.
 *
 * Port of src/momentarily/hmm.py. Inference only (forward filter, projection,
 * dwell estimation, hysteresis + Unknown). Training stays in Python.
 *
 * Three hidden states: normal | disrupted | suspended. Observations per tick:
 *   - alert_count            (Poisson per state)
 *   - severity_sum           (Gamma per state)
 *   - has_suspended_alert    (Bernoulli per state)
 *   - has_delays             (Bernoulli per state)
 *   - has_service_change     (Bernoulli per state)
 *   - has_planned            (Bernoulli per state)
 *
 * Emissions can be conditioned on TOD bin via `HMMParams.emissionsByBin`.
 */

import { logBernoulli, logGamma, logPoisson } from './math';

export const STATES = ['normal', 'disrupted', 'suspended'] as const;
export type State = (typeof STATES)[number];
export const N_STATES = 3;

// TOD bins (UTC) — match hmm.py's tod_bin() exactly.
export const N_TOD_BINS = 5;
export function tod_bin(epochSeconds: number): number {
  const hour = Math.floor(epochSeconds / 3600) % 24;
  if (hour < 5) return 0;
  if (hour < 13) return 1;
  if (hour < 17) return 2;
  if (hour < 23) return 3;
  return 4;
}

export interface Observation {
  alert_count: number;
  severity_sum: number;
  has_suspended_alert: boolean;
  has_delays: boolean;
  has_service_change: boolean;
  has_planned: boolean;
  tod_bin: number;
}

type Vec3 = readonly [number, number, number];
type Matrix3x3 = readonly [Vec3, Vec3, Vec3];

export interface EmissionParams {
  poisson_lambda: Vec3;
  gamma_alpha: Vec3;
  gamma_beta: Vec3;
  bernoulli_p: Vec3;
  bernoulli_p_delays: Vec3;
  bernoulli_p_service_change: Vec3;
  bernoulli_p_planned: Vec3;
}

export interface HMMParams {
  transition: Matrix3x3;
  initial: Vec3;
  emissions: EmissionParams;
  /** When set (length N_TOD_BINS), emissions are looked up by obs.tod_bin. */
  emissions_by_bin?: readonly EmissionParams[];
}

export interface FilterState {
  probabilities: Vec3;
  regime_entered_at: number;
  last_updated_at: number;
}

// ---------------------------------------------------------------------------
// Forward filter
// ---------------------------------------------------------------------------

function emissionsFor(params: HMMParams, obs: Observation): EmissionParams {
  if (!params.emissions_by_bin) return params.emissions;
  const idx = Math.max(0, Math.min(N_TOD_BINS - 1, obs.tod_bin));
  return params.emissions_by_bin[idx] ?? params.emissions;
}

function logEmission(obs: Observation, em: EmissionParams): Vec3 {
  return [0, 1, 2].map(
    (s) =>
      logPoisson(obs.alert_count, em.poisson_lambda[s]!)
      + logGamma(obs.severity_sum, em.gamma_alpha[s]!, em.gamma_beta[s]!)
      + logBernoulli(obs.has_suspended_alert, em.bernoulli_p[s]!)
      + logBernoulli(obs.has_delays, em.bernoulli_p_delays[s]!)
      + logBernoulli(obs.has_service_change, em.bernoulli_p_service_change[s]!)
      + logBernoulli(obs.has_planned, em.bernoulli_p_planned[s]!),
  ) as unknown as Vec3;
}

export function forwardUpdate(
  state: FilterState,
  obs: Observation,
  params: HMMParams,
  now: number,
): FilterState {
  const prior = state.probabilities;
  const a = params.transition;

  // Predict: predicted[s] = Σ_s' prior[s'] · A[s', s]
  const predicted: Vec3 = [0, 1, 2].map((s) =>
    [0, 1, 2].reduce((acc, sp) => acc + prior[sp]! * a[sp]![s]!, 0),
  ) as unknown as Vec3;

  const logEmis = logEmission(obs, emissionsFor(params, obs));
  const logUnnorm: Vec3 = [0, 1, 2].map(
    (s) => (predicted[s]! > 0 ? Math.log(predicted[s]!) : -Infinity) + logEmis[s]!,
  ) as unknown as Vec3;

  const maxLog = Math.max(logUnnorm[0], logUnnorm[1], logUnnorm[2]);
  let post: Vec3;
  if (maxLog === -Infinity) {
    post = prior;
  } else {
    const scaled = [0, 1, 2].map((s) => Math.exp(logUnnorm[s]! - maxLog));
    const total = scaled[0]! + scaled[1]! + scaled[2]!;
    post = [scaled[0]! / total, scaled[1]! / total, scaled[2]! / total] as Vec3;
  }

  const prevArgmax = argmax(prior);
  const newArgmax = argmax(post);
  const regimeEnteredAt =
    newArgmax !== prevArgmax ? now : state.regime_entered_at;

  return {
    probabilities: post,
    regime_entered_at: regimeEnteredAt,
    last_updated_at: now,
  };
}

// ---------------------------------------------------------------------------
// Projection + dwell
// ---------------------------------------------------------------------------

export function projectForward(
  state: FilterState,
  params: HMMParams,
  ticksAhead: number,
): Vec3 {
  if (ticksAhead < 0) throw new Error('ticksAhead must be >= 0');
  if (ticksAhead === 0) return state.probabilities;

  const a = params.transition;
  let current: Vec3 = state.probabilities;
  for (let i = 0; i < ticksAhead; i += 1) {
    current = [0, 1, 2].map((s) =>
      [0, 1, 2].reduce((acc, sp) => acc + current[sp]! * a[sp]![s]!, 0),
    ) as unknown as Vec3;
  }
  return current;
}

const LARGE_DWELL = 10_000;

export function expectedDwellTicks(
  state: FilterState,
  params: HMMParams,
): { median: number; q25: number; q75: number } {
  const argmaxIdx = argmax(state.probabilities);
  const selfLoop = params.transition[argmaxIdx]![argmaxIdx]!;

  if (selfLoop >= 1.0) {
    return { median: LARGE_DWELL, q25: LARGE_DWELL, q75: LARGE_DWELL };
  }
  if (selfLoop <= 0) {
    return { median: 1, q25: 1, q75: 1 };
  }

  const logSelf = Math.log(selfLoop);
  const quantile = (q: number): number => {
    const target = 1 - q;
    if (target <= 0) return LARGE_DWELL;
    return Math.max(1, Math.ceil(Math.log(target) / logSelf));
  };
  return { median: quantile(0.5), q25: quantile(0.25), q75: quantile(0.75) };
}

// ---------------------------------------------------------------------------
// Hysteresis + Unknown overlay
// ---------------------------------------------------------------------------

export const PUBLISHED_UNKNOWN = 'unknown' as const;
export type PublishedLabel = State | typeof PUBLISHED_UNKNOWN;

export const HYSTERESIS_TICKS = 2;

export interface PublishedState {
  label: PublishedLabel;
  pending_state: State;
  pending_streak: number;
  last_updated_at: number;
}

export function initialPublishedState(state: FilterState): PublishedState {
  const idx = argmax(state.probabilities);
  return {
    label: STATES[idx]!,
    pending_state: STATES[idx]!,
    pending_streak: HYSTERESIS_TICKS,
    last_updated_at: state.last_updated_at,
  };
}

/**
 * Advance one tick with hysteresis + Unknown handling.
 *
 * obs=null signals a feed gap: posterior is preserved, label flips to
 * "unknown". On the first real observation after a gap, the new label is
 * published immediately (no extra lag).
 */
export function forwardStep(
  state: FilterState,
  published: PublishedState,
  obs: Observation | null,
  params: HMMParams,
  now: number,
): { state: FilterState; published: PublishedState } {
  if (obs === null) {
    return {
      state,
      published: { ...published, label: PUBLISHED_UNKNOWN, last_updated_at: now },
    };
  }

  const newState = forwardUpdate(state, obs, params, now);
  const newArgmax = STATES[argmax(newState.probabilities)]!;

  const streak =
    newArgmax === published.pending_state ? published.pending_streak + 1 : 1;

  const cameFromUnknown = published.label === PUBLISHED_UNKNOWN;
  const newLabel: PublishedLabel =
    cameFromUnknown || streak >= HYSTERESIS_TICKS ? newArgmax : published.label;

  return {
    state: newState,
    published: {
      label: newLabel,
      pending_state: newArgmax,
      pending_streak: streak,
      last_updated_at: now,
    },
  };
}

// ---------------------------------------------------------------------------
// Small helper
// ---------------------------------------------------------------------------

function argmax(v: Vec3): 0 | 1 | 2 {
  if (v[0]! >= v[1]! && v[0]! >= v[2]!) return 0;
  if (v[1]! >= v[2]!) return 1;
  return 2;
}
