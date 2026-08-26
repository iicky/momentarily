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
import { N_TOD_BINS, schedule_bin } from './hmm';

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
  // params.json written before the movement channel.
  advance_rate: ProbVec3.optional(),
  // Per-state service-ratio Gaussian (assigned_n / baseline): mu is a ratio
  // (>=0, may exceed 1), sigma a std (>=0). Optional for back-compat with
  // params.json written before the service channel.
  service_mu: RateVec3.optional(),
  service_sigma: RateVec3.optional(),
});

const DwellQuantilesSchema = z.object({
  n: z.number().int().nonnegative(),
  // Right-censored episode count (still open when the trainer fit this cell).
  // Was already on the wire and silently stripped by this schema's default
  // STRIP semantics — add it so the parsed shape stops disagreeing with what
  // the trainer actually sends.
  n_censored: z.number().optional(),
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
  // worker/src/dwell.ts). Optional for back-compat.
  curve_sec: z.array(z.number().nonnegative()).min(2).optional(),
  // [shape, scale] of a log-logistic fit to this cell's dwells. pLeaveBy uses it
  // to extrapolate the tail past the last observed quantile instead of the
  // constant-hazard exponential patch. Optional for back-compat with older
  // params.json. When atom_p/atom_sec are both present, this is instead the
  // SAME log-logistic LEFT-TRUNCATED at atom_sec (see worker/src/dwell.ts
  // mixtureSurvival/mixtureQuantile) — the two atom fields change what this
  // one means, they don't add an independent third mode.
  tail_ll: z.tuple([z.number().positive(), z.number().positive()]).optional(),
  // P(dwell == atom_sec) and the tick it sits at (300s = one publisher tick).
  // Both optional, and only meaningful together: a cell where most episodes
  // end on the very first tick can't be fit by a single continuous curve
  // without either flattening the tail or missing the spike, so the trainer
  // instead mixes a point mass here with tail_ll refit conditional on
  // T > atom_sec. Either field missing — older params.json, or a cell the
  // mixture fit skipped — means the pure curve_sec/tail_ll path applies
  // unchanged.
  atom_p: z.number().optional(),
  atom_sec: z.number().int().optional(),
});

// Per-route, per-prev-state empirical dwell quantiles from the trainer.
// Keys are the same state names the worker uses: "normal" / "disrupted" /
// "suspended". Cells the trainer didn't include (sample size below its
// floor) simply aren't here and the worker falls back to its geometric
// estimate.
const DwellByStateSchema = z.record(z.string(), DwellQuantilesSchema).optional();

// Cause-segmented dwell: state -> alert_type -> quantiles. Layered on top of
// dwell_quantiles; the worker prefers a (state, alert_type) cell and falls back
// to the (state) aggregate when one is absent.
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
  n_censored?: number | undefined;
  q25_sec: number;
  median_sec: number;
  q75_sec: number;
  recover_by_30?: number | undefined;
  recover_by_60?: number | undefined;
  recover_by_120?: number | undefined;
  curve_sec?: number[] | undefined;
  tail_ll?: [number, number] | undefined;
  atom_p?: number | undefined;
  atom_sec?: number | undefined;
}

export type DwellByState = Record<string, DwellQuantiles>;

// state -> alert_type -> quantiles
export type DwellByStateAlert = Record<string, Record<string, DwellQuantiles>>;

// Per-(route, direction, tod_bin) advance-rate baseline.
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

// route -> direction -> stop ids with both a scheduled predecessor and a
// scheduled successor (training.gtfs_static.through_stops). Co-versioned
// with movement_baseline — same trainer run, same fit — so the two update
// or fall back together. Flattened to `${route}|${direction}|${stop}` keys
// below, the shape deriveRouteMovementMetric (vehicles.ts) tests per trip.
const MovementThroughStopsSchema = z.record(
  z.string(),
  z.record(z.string(), z.array(z.string())),
);
export type MovementThroughStops = z.infer<typeof MovementThroughStopsSchema>;

// route -> tod_bin (stringified int) -> median assigned_n. The Worker divides
// live assigned_n by this to form the service ratio the emission scores.
const ServiceBaselineSchema = z.record(z.string(), z.record(z.string(), nonNeg));
export type ServiceBaseline = z.infer<typeof ServiceBaselineSchema>;

// route -> schedule_bin (e.g. `wd06`) -> in-service rate in [0,1]: the share of
// usable ticks the route was running at that (weekend, hour) bin. The Worker
// reads it to split a no-service reading into suspended vs not_scheduled.
const ScheduleRateSchema = z.record(z.string(), z.record(z.string(), prob));
export type ScheduleRate = z.infer<typeof ScheduleRateSchema>;

// Per-route, per-movement-state empirical dwell quantiles from the movement
// clock (training.regime), keyed the same way as `dwell_quantiles` above but
// off the movement transitions stream instead of the alert HMM's. Top-level,
// like the baselines — not nested under the per-route HMM params, since the
// movement clock isn't part of HMMParamsSchema.
const DwellMovementSchema = z.record(z.string(), z.record(z.string(), DwellQuantilesSchema));
export type DwellMovement = z.infer<typeof DwellMovementSchema>;

const TrainedParamsWrapperSchema = z.object({
  schema_version: z.string(),
  trained_at: z.number().finite(),
  // Validate each route separately (below) so one bad route doesn't drop the
  // whole upload — the others should still apply.
  routes: z.record(z.string(), z.unknown()),
  // Validated separately too, so a malformed baseline degrades the movement
  // channel only, not the whole params upload.
  movement_baseline: z.unknown().optional(),
  // Validated separately too; malformed or absent falls back to null (count
  // every stop) rather than dropping the whole params upload. Co-versioned
  // with movement_baseline — see gtfs_static.through_stops in
  // training/gtfs_static.py and the StopFilter training/load_r2.py builds
  // from it.
  movement_through_stops: z.unknown().optional(),
  // Validated separately too, so a malformed service baseline degrades the
  // service channel only, not the whole params upload.
  service_baseline: z.unknown().optional(),
  // Validated separately too, like service_baseline; feeds the published
  // service-degradation axis (finer, schedule_bin-keyed).
  service_baseline_hourly: z.unknown().optional(),
  // Validated separately too, so a malformed schedule rate degrades the
  // suspended/not_scheduled split only, not the whole params upload.
  schedule_rate: z.unknown().optional(),
  // Validated separately too, so a malformed movement-dwell sidecar disables
  // the movement recovery arm only, not the whole params upload. Absent
  // entirely for params.json written before the movement recovery arm existed.
  dwell_movement: z.unknown().optional(),
});

// The three "kind of disruption" flags (delays/service_change/planned) all
// indicate `disrupted`, not `suspended` — only has_suspended_alert
// (bernoulli_p) should pull hard toward suspended. Before this, all three
// leaned suspended, so any persistent planned-work/delay alert drifted routes
// into `suspended`.
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
  // Route|direction|stop keys admitted by the movement advance filter (see
  // vehicles.ts deriveRouteMovementMetric). Null means "no filter": absent or
  // malformed movement_through_stops, or a params.json written before the
  // field existed. NEVER an empty set — that would silently zero every
  // route's advance/stall count instead of leaving the filter off.
  // Co-versioned with movementBaseline: the trainer fits the baseline
  // against exactly this stop set.
  throughStops: ReadonlySet<string> | null;
  // Per-(route, tod_bin) assigned_n baseline for the service emission channel.
  serviceBaseline: ServiceBaseline;
  // Per-(route, schedule_bin) assigned_n baseline for the published
  // service-degradation axis. Finer than serviceBaseline (hourly) so a supply
  // cut is judged against the same-hour normal, not a wide tod block's core.
  serviceBaselineHourly: ServiceBaseline;
  // Per-(route, schedule_bin) in-service rate; the Worker splits a no-service
  // reading into suspended (normally runs now) vs not_scheduled (rush-only gap).
  scheduleRate: ScheduleRate;
  // Per-(route, movement-state) empirical dwell quantiles off the movement
  // clock. Empty until the trainer publishes dwell_movement; the Worker falls
  // back to the alert-HMM dwell (`dwell` above) when a cell is absent.
  dwellMovement: DwellMovement;
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
 * NaN-poison the rest of the fleet.
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

  // Through-stop set for the movement channel, co-versioned with
  // movement_baseline. Absent or malformed => null, never an empty set — see
  // TrainedParams.throughStops above. Mirrors training/gtfs_static.through_stops
  // and the StopFilter training/load_r2.py builds from it;
  // deriveRouteMovementMetric (vehicles.ts) applies the same filter live.
  let throughStops: ReadonlySet<string> | null = null;
  if (wrapper.data.movement_through_stops !== undefined) {
    const parsed = MovementThroughStopsSchema.safeParse(wrapper.data.movement_through_stops);
    if (parsed.success) {
      const keys = new Set<string>();
      for (const [routeId, byDirection] of Object.entries(parsed.data)) {
        for (const [direction, stops] of Object.entries(byDirection)) {
          for (const stopId of stops) keys.add(`${routeId}|${direction}|${stopId}`);
        }
      }
      // A well-formed but empty doc ({}, or every route empty) means the same
      // as absent: count every stop. Keeping the empty Set would admit no stop
      // at all and zero every route's advance/stall counters.
      if (keys.size > 0) throughStops = keys;
      else {
        console.warn(
          'params.json movement_through_stops is empty; advance counters include every stop',
        );
      }
    } else {
      console.warn(
        'params.json movement_through_stops invalid; advance counters include every stop:',
        parsed.error.issues,
      );
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

  // Hourly service baseline, validated on its own like service_baseline.
  let serviceBaselineHourly: ServiceBaseline = {};
  if (wrapper.data.service_baseline_hourly !== undefined) {
    const parsed = ServiceBaselineSchema.safeParse(wrapper.data.service_baseline_hourly);
    if (parsed.success) {
      serviceBaselineHourly = parsed.data;
    } else {
      console.warn('params.json service_baseline_hourly invalid; service axis off:', parsed.error.issues);
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

  // Movement dwell, validated on its own like the baselines.
  let dwellMovement: DwellMovement = {};
  if (wrapper.data.dwell_movement !== undefined) {
    const parsed = DwellMovementSchema.safeParse(wrapper.data.dwell_movement);
    if (parsed.success) {
      dwellMovement = parsed.data;
    } else {
      console.warn('params.json dwell_movement invalid; movement recovery arm off:', parsed.error.issues);
    }
  }

  return {
    schema_version: wrapper.data.schema_version,
    trained_at: wrapper.data.trained_at,
    routes,
    dwell,
    dwellByAlert,
    movementBaseline,
    throughStops,
    serviceBaseline,
    serviceBaselineHourly,
    scheduleRate,
    dwellMovement,
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
 * Median assigned_n for a (route, schedule_bin) cell, the denominator of the
 * published service-degradation axis. Null when the trainer hasn't established
 * one yet — the axis then reads 'unknown' for that cell.
 */
export function serviceBaselineHourlyFor(
  trained: TrainedParams | null,
  routeId: string,
  scheduleBin: string,
): number | null {
  return trained?.serviceBaselineHourly?.[routeId]?.[scheduleBin] ?? null;
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

/**
 * Empirical dwell quantiles for a (route, movement-state) cell off the
 * movement clock — see C2/C3. Optional in params: absent trainer output or a
 * missing cell returns null (never throws), and the caller falls back to the
 * alert-HMM dwell / geometric estimate.
 */
export function movementDwellFor(
  trained: TrainedParams | null,
  routeId: string,
  state: string,
): DwellQuantiles | null {
  return trained?.dwellMovement?.[routeId]?.[state] ?? null;
}

// ---------------------------------------------------------------------------
// Ridership baseline: per-station-complex entries/min by
// (weekday/weekend, hour) cell, written weekly by training/ridership.py from
// the MTA hourly ridership dataset (5wq4-mkjj). Its own R2 object, like
// segment_params.json — never folded into params.json, so a late or failed
// ridership run can never perturb the HMM filter's params version. Read
// fresh each tick alongside the other baselines; feeds the live per-platform
// crowding estimate (crowding.ts).
// ---------------------------------------------------------------------------

const RIDERSHIP_BASELINE_KEY = 'state/ridership_baseline.json';

// Complex-wide entries/min for each of the 24 local hours, one array per
// weekday/weekend class. Always exactly 24 long — the trainer emits one
// cell per hour whether or not that hour had enough samples (see
// training/ridership.py); a document that's short an hour is malformed,
// not merely sparse, and must be rejected rather than index out of range.
const HourlyRatesSchema = z.array(nonNeg).length(24);

const RidershipComplexSchema = z.object({
  name: z.string(),
  borough: z.string().nullable(),
  entries_per_min: z.object({
    wd: HourlyRatesSchema,
    we: HourlyRatesSchema,
  }),
  entries_total: nonNeg,
  transfers_total: nonNeg,
  rank: z.number().int().positive(),
  n_cells: z.number().int().nonnegative(),
});

const RidershipBaselineSchema = z.object({
  schema_version: z.string(),
  generated_at: z.number(),
  source: z.object({
    dataset: z.string(),
    url: z.string(),
    transit_mode: z.string(),
    window_start: z.string(),
    window_end: z.string(),
    latest_hour: z.string(),
    weekday_days: z.number().int().nonnegative(),
    weekend_days: z.number().int().nonnegative(),
  }),
  complexes: z.record(z.string(), RidershipComplexSchema),
  n_complexes: z.number().int().nonnegative(),
});
export type RidershipBaselineDoc = z.infer<typeof RidershipBaselineSchema>;

/**
 * Load the ridership baseline from R2. Null when absent (before the first
 * weekly training/ridership.py run) or malformed — a bad complex row, a
 * short entries_per_min array, a negative rate all fail semantic bounds, not
 * just shape. The crowding surface then abstains entirely rather than
 * publish off a document that failed validation. Never throws into the tick.
 */
export async function loadRidershipBaseline(
  bucket: R2Bucket,
): Promise<RidershipBaselineDoc | null> {
  const obj = await bucket.get(RIDERSHIP_BASELINE_KEY);
  if (!obj) return null;
  try {
    return RidershipBaselineSchema.parse(await obj.json());
  } catch (err) {
    console.error('ridership_baseline.json invalid; platform crowding off:', err);
    return null;
  }
}

/**
 * Entries/min for a station complex at the current (weekday/weekend, hour)
 * cell, or null when the baseline has no rate for that complex — the
 * crowding surface then abstains with 'no_baseline' for every platform in
 * it, never fabricating a rate. Parses schedule_bin's `wd06`/`we22` key
 * rather than re-deriving the weekday/local-hour split: this is the only
 * place that rule is applied to ridership, so it can never drift from the
 * HMM's own notion of the current schedule cell.
 */
export function ridershipRateFor(
  baseline: RidershipBaselineDoc | null,
  complexId: string,
  epochSeconds: number,
): number | null {
  const complex = baseline?.complexes?.[complexId];
  if (!complex) return null;
  const bin = schedule_bin(epochSeconds);
  const cls = bin.slice(0, 2) as 'wd' | 'we';
  const hour = parseInt(bin.slice(2), 10);
  return complex.entries_per_min[cls][hour] ?? null;
}