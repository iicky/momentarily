/**
 * Render and publish the two public artifacts: the snapshot, and its sibling
 * `trains` object.
 *
 * Shape matches src/momentarily/schema.py's Snapshot. Surfaces whose upstream
 * source isn't wired yet (observations, stations, bridges, tunnels) emit as
 * empty placeholders so the schema_version=1 contract stays honored. alerts,
 * routes, and equipment are populated from the data already fetched each tick.
 *
 * Output is publicly readable at https://feed.momentarily.nyc/v1/snapshot.json
 * via the R2 custom domain. Cache headers per ADR (max-age=60, s-maxage=300).
 *
 * `trains` (buildTrains/publishTrains below) is published separately, at
 * v1/trains.json, rather than embedded as a Snapshot field: at ~700
 * concurrent trips it adds ~36KB (measured on a realistic vehicle set, see
 * the worker's vehicles.test.ts) to every fetch, and the canonical snapshot
 * consumer (homeassistant-mta-subway, polling many installs every few
 * minutes) never reads it — charging every install that bandwidth forever
 * for a feature only the /map overlay uses is the wrong trade. It carries
 * the same cache policy and its own observed_at + provenance, so a consumer
 * holding only that object can still say which build produced it and how
 * stale it is.
 */

import type { RouteRoll } from './alpha';
import type { ParamsProvenance, Provenance } from './buildinfo';
import { codeProvenance } from './buildinfo';
import { CROWDING_MAX_GAP_MINUTES, CROWDING_SERVED_WINDOW_MINUTES, derivePlatformCrowding } from './crowding';
import type {
  AlertOut,
  AlertRef,
  DirectionAlerts,
  RouteSnapshot,
} from './derive';
import { buildRoutes, metaForRoute } from './derive';
import { conditionalRecovery, pLeaveBy } from './dwell';
import { HYSTERESIS_TICKS, N_STATES, PUBLISHED_UNKNOWN, STATES } from './hmm';
import type { PublishedLabel } from './hmm';
import { NO_ALERTS_FALLBACK, categoryForLabel, coarseStatus } from './mapping';
import type { DwellQuantiles, RidershipBaselineDoc, ServiceWeightBaselineDoc, TrainedParams } from './params';
import { dwellForRouteState, movementDwellFor, paramsForRoute, versionedParamsKey } from './params';
import { servicePercentile } from './movement_state';
import type { EquipmentOut, StationStatus } from './stations';
import type { StationOut } from './stations_static';
import type {
  SegmentCondition,
  SegmentDwellDoc,
  SegmentFlowDoc,
  SegmentParamsDoc,
  StationFlowDoc,
  StationWaitDoc,
} from './state';
import type { TrainPosition } from './vehicles';

// Above this, the geometric dwell estimate is uninformative — a trained
// self-loop ≈ 1 means the model has no evidence the regime ever ends (typical
// of open-ended planned work). Clamp + flag rather than publish "34 days".
const MAX_RECOVERY_MINUTES = 1440;

// Fast-attack threshold for surfacing `condition`. When the filter is
// highly confident in a state that disagrees with the hysteresis-gated
// published label, we surface the filter's view instead of the lagged
// label. Hysteresis still protects the underlying publish state machine
// from flapping on ambiguous evidence; this only governs what consumers
// see.
const FAST_ATTACK_PROB = 0.9;

// Movement state is carried from the prior tick, normally ~5 min old. If the
// vehicle feeds stall, don't keep publishing a frozen reading indefinitely —
// past this age the route falls back to the alert/HMM condition. Six ticks.
const MAX_MOVEMENT_STATE_AGE_SEC = 1800;

const SNAPSHOT_KEY = "v1/snapshot.json";
const TRAINS_KEY = "v1/trains.json";

// Shared by publishSnapshot and publishTrains so the two public artifacts
// can never drift on cache policy — see the ADR reference in the file
// header comment for where these numbers come from.
const PUBLIC_CACHE_CONTROL = "public, max-age=60, s-maxage=300";

export const SCHEMA_VERSION = "1";

export const ATTRIBUTION =
  "Snapshot built from MTA GTFS-RT feeds via api.mta.info. " +
  "Published by Momentarily (https://feed.momentarily.nyc). " +
  "Not affiliated with the MTA.";

// The HMM/alert forecast for a route — recovery timing (recovery_minutes,
// p_normal_in_H) plus the alert-derived regime read. With movement-primary
// publishing this is the SHADOW: its `condition`/`is_disrupted`/probabilities are
// the alert view, not the published current state (route_status.condition).
interface Inference {
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
  // True when the dwell estimate saturated MAX_RECOVERY_MINUTES — the regime
  // is so persistent the model can't bound when it ends. recovery_minutes and
  // its bounds are clamped to the ceiling in that case.
  recovery_indeterminate: boolean;
  // A forecast about the PUBLISHED condition, or nothing. The condition in
  // route_status is movement-primary; this number comes from whichever arm
  // recovery_source names, and the two are not always the same arm. Graded
  // against the condition actually published 30 minutes later, the
  // movement-sourced rows score AUC 0.856 and the alert-sourced rows 0.261 —
  // but publishing them in one field scores 0.084, WORSE than either, because
  // the arms put their probabilities on different scales and the mixed
  // ranking tracks which arm answered rather than the risk. So it is null
  // whenever the forecast arm is not the arm that decided the condition.
  p_normal_in_30min: number | null;
  // Withheld for a different reason: these two measured worse than naive
  // persistence in every population cut (BSS -0.00 to -1.30, AUC 0.395 and
  // 0.352 — inverted), and the cause is the shape of the fitted
  // elapsed-conditional dwell curve, not censoring and not the horizon
  // projection, so more runtime will not fix it. They are null whenever the
  // value would come from a fitted curve, and survive only when
  // recovery_source === 'schedule', where the answer is a deterministic
  // comparison against an announced resume time. Mirrors
  // src/momentarily/schema.py Inference.
  p_normal_in_60min: number | null;
  p_normal_in_120min: number | null;
  model_warming_up: boolean;
  // Where recovery_minutes comes from: "schedule" is a deterministic lookup of
  // the planned-work resume time (no model uncertainty); "movement" is the
  // movement-clock dwell curve (preferred whenever the published condition is
  // movement-sourced and a curve exists); "hmm" is the alert-regime dwell
  // estimate, used only as the fallback. The grader excludes "schedule" rows
  // from HMM calibration.
  recovery_source: "hmm" | "schedule" | "movement";
  // Announced resume time (epoch s) for schedule recovery; null for hmm.
  resumes_at: number | null;
  // now has passed resumes_at but the planned alert is still active — recovery
  // is clamped to 0 rather than counting down past the announced time.
  overdue: boolean;
}

interface RouteStatusOut {
  route_id: string;
  alerts: string[];
  // Severity axis — the published current state, movement-primary: observed from
  // train movement where judgeable, a planned "No Scheduled Service" alert where
  // flagged, else 'unknown'. Alerts never assert disruption here — that lives on
  // the shadow (inference) and cause (category / primary_alert_type) axes.
  condition: string;
  // Where `condition` came from this tick: 'movement' (observed from train
  // positions), 'schedule' (a planned "No Scheduled Service" alert), or 'unknown'
  // (movement can't judge — an honest coverage gap, never an alert fallback).
  condition_source: string;
  // Supply axis — assigned_n against its own hourly baseline, one-tick lagged
  // like `condition`. 'normal' | 'degraded' | 'unknown'. Distinct from
  // `condition` (flow): a route's trips can be pulled (degraded here) while the
  // trains still running advance fine (normal there), and the reverse.
  service_condition: string;
  // Magnitude behind service_condition: assigned_n / its hourly baseline this
  // tick. null when unjudgeable (service_condition then 'unknown'). Raw, not
  // debounced — service_condition is the debounced regime over this.
  service_ratio: number | null;
  // Cell p10/median and p90/median — service_ratio's own spread, on the same
  // scale so both can render as ticks on one meter. null whenever service_ratio
  // would be null, plus one more case: the cell has no published quantiles.
  service_low_ratio: number | null;
  service_high_ratio: number | null;
  // Where service_ratio sits within this cell's own same-daypart baseline, as a
  // 0-100 percentile (movement_state.servicePercentile). Low = fewer trains than
  // usual for this daypart; exact at the cell's p10/median/p90, saturating at 90
  // above its p90. A percentile of the baseline, NOT a forecast. null under the
  // same conditions as service_low_ratio (no reading or no published quantiles).
  service_percentile: number | null;
  // Cause axis — our vocabulary, derived from the MTA alert_type.
  category: string;
  primary_alert_type: string | null;
  // Soft-deprecated: now derivable from condition + category. Kept for
  // existing consumers and the compat layer.
  label: string;
  by_direction: {
    northbound: DirectionAlerts;
    southbound: DirectionAlerts;
  };
  inference: Inference | null;
}

interface Freshness {
  subway_alerts: number | null;
  lirr_alerts: number | null;
  mnr_alerts: number | null;
  bus_alerts: number | null;
  path_alerts: number | null;
  ferry_alerts: number | null;
  ene: number | null;
  stations_static: number | null;
}

interface Accessibility {
  elevators_out: number;
  escalators_out: number;
  ada_pathways_degraded: number;
}

interface ModeRollup {
  routes_with_alerts: string[];
  alert_count: number;
  severity_max: number;
}

interface SystemStatus {
  by_mode: Record<string, ModeRollup>;
  accessibility: Accessibility;
  overall_label: string;
  condition: string | null;
  lines_disrupted_count: number;
  most_degraded_line: string | null;
  most_recovered_line: string | null;
}

interface CompatRouteSummary {
  north: string | null;
  south: string | null;
}

interface CompatServiceChangeSummary {
  both: string[];
  north: string[];
  south: string[];
}

interface CompatRoute {
  id: string;
  name: string;
  color: string;
  status: string;
  scheduled: boolean;
  direction_statuses: CompatRouteSummary | null;
  delay_summaries: CompatRouteSummary | null;
  service_irregularity_summaries: CompatRouteSummary | null;
  service_change_summaries: CompatServiceChangeSummary | null;
}

interface Compat {
  subwaynow_routes: Record<string, CompatRoute>;
}

// Expected recovery off a dwell curve conditioned on a regime clock. Same
// field names as Inference's recovery block (in minutes, not seconds) so a
// segment's or station's recovery is directly comparable to a route's.
interface SegmentRecovery {
  recovery_minutes: number;
  recovery_minutes_low: number;
  recovery_minutes_high: number;
  recovery_indeterminate: boolean;
  p_normal_in_30min: number;
  p_normal_in_60min: number;
  p_normal_in_120min: number;
}

// Per-segment published status: the debounced regime + clock, successor
// stop, and expected recovery for one (route, direction, from_stop) cell.
interface SegmentStatusOut {
  route: string;
  direction: string;
  from_stop: string;
  to: string | null;
  // 'quiet' is the quiet-normal call: the timetable runs too little here right
  // now for an empty window to mean anything (segment_flow.ts classifyThroughput).
  status: SegmentCondition;
  entered_at: number;
  recovery: SegmentRecovery | null;
}

// The segment-flow surface: sibling of StationFlow, keyed the same way as
// segment_flow.json/segment_params.json (`route|direction|from_stop`).
// `segments` carries EVERY judged cell, normal and disrupted alike — a key
// absent from it was never judged this tick, never a healthy read by
// omission. That single-dict membership check is the whole honesty
// property this surface rests on.
//
// SIZING (why this is one dict, not the normal/disrupted split shipped and
// reverted the same day, 2026-08-23): measured on the live feed that day —
// base snapshot minus segment_flow 179.9 KB, a bare segment record (incl.
// its key) 151 B, a record carrying a recovery block 382 B. A normal cell
// is always bare (SegmentStatusOut.recovery is null on healthy track), so
// the shipped policy's ~701 judged cells/tick (~3 typically disrupted)
// totals ~284.0 KB, under the 300 KB line with ~120 KB (~814 bare records)
// of headroom before that line is even in question — ~16% above today's
// population. The split existed only for a policy that was measured and
// REJECTED before shipping: decay=0.98, ~1199 judged cells/tick (see
// segment_flow.ts's module docstring). Revisit only if judged volume
// climbs toward that headroom ceiling or the disrupted share grows well
// past today's ~0.4%; not before.
interface SegmentFlowOut {
  observed_at: number;
  segments: Record<string, SegmentStatusOut>;
}

// StationFlowDoc's per-station entry, additively carrying the expected
// recovery of its already-selected worst_segment.
interface StationServiceFlowOut {
  // 'quiet' when every segment touching the station is quiet — nothing
  // scheduled here right now, which is neither flowing nor degraded.
  status: StationFlowDoc['stations'][string]['status'];
  worst_deficit: number;
  worst_segment: [string, string] | null;
  routes: string[];
  n_segments: number;
  worst_recovery: SegmentRecovery | null;
}

interface StationFlowOut {
  observed_at: number;
  stations: Record<string, StationServiceFlowOut>;
}

// One dot per distinct (route, direction, stop, stopped) tuple, folded by
// vehicles.ts's trainPositions() from the ~700 concurrent in-service trips.
// `stop` carries NYCT's usual duality — the stop a train is heading to while
// moving, the stop it is at once stopped — and this surface deliberately
// never infers which segment a moving train occupies (ambiguous at branch/
// express points); see trainPositions()'s doc comment for the full rationale.
// Part of PublishedTrains (buildTrains/publishTrains below), NOT the
// Snapshot — see that pair's doc comment for why it's a sibling artifact.
interface TrainPositionOut {
  route: string;
  direction: "north" | "south" | null;
  stop: string;
  stopped: boolean;
  n: number;
}

// The platform-crowding surface: entries_per_min * minutes_since_last_train,
// split evenly across a station complex's currently-served platforms. See
// the crowding contract for the full derivation; crowding.ts computes
// everything here except `method`, which this module attaches from its own
// constants plus the baseline document's provenance.
interface PlatformCrowdingEstimateOut {
  last_train_at: number;
  entries_per_min: number;
  waiting_riders: number;
}

interface PlatformCrowdingMethodOut {
  formula: string;
  // Which rule split each complex's rate across its served platforms.
  // 'scheduled_service_over_served_platforms' when the service_weight baseline
  // is loaded (weighted where a complex is fully covered this hour, even
  // otherwise); 'uniform_over_served_platforms' when it is absent and every
  // complex splits evenly. See crowding.ts derivePlatformCrowding.
  split_basis: 'uniform_over_served_platforms' | 'scheduled_service_over_served_platforms';
  max_gap_minutes: number;
  served_window_minutes: number;
  excludes: string[];
  baseline_generated_at: number;
  baseline_window_start: string;
  baseline_window_end: string;
  // Provenance of the scheduled-service split weights: the service_weight
  // baseline's own generated_at and GTFS feed_version, or null when the
  // baseline was absent and the split fell back to uniform.
  service_weight_generated_at: number | null;
  service_weight_feed_version: string | null;
}

interface PlatformCrowdingOut {
  observed_at: number;
  method: PlatformCrowdingMethodOut;
  platforms: Record<string, PlatformCrowdingEstimateOut>;
  n_platforms: number;
  abstained: Record<string, number>;
}

interface Snapshot {
  schema_version: string;
  generated_at: number;
  provenance: Provenance;
  attribution: string;
  supported_modes: string[];
  freshness: Freshness;
  alerts: AlertOut[];
  observations: unknown[];
  routes: Record<string, unknown>;
  stations: Record<string, StationOut>;
  equipment: EquipmentOut[];
  bridges: unknown[];
  tunnels: unknown[];
  route_status: Record<string, RouteStatusOut>;
  station_status: Record<string, unknown>;
  // Per-station service flow ("is my station moving"), rolled up from the segment
  // movement model. Distinct from station_status (accessibility/alerts). Null when
  // absent or stale.
  station_flow: StationFlowOut | null;
  // Per-segment service flow ("is this stretch of track moving"), the same
  // segment movement model station_flow rolls up. Null when absent or stale.
  segment_flow: SegmentFlowOut | null;
  // Estimated riders waiting per platform, derived from the ridership
  // baseline and this tick's platform-wait state. Null before the ridership
  // baseline is published, before the first vehicle tick after deploy, or
  // when stale.
  platform_crowding: PlatformCrowdingOut | null;
  system: SystemStatus;
  compat: Compat;
}

// Identity of the trained params behind this snapshot's inference, for the
// provenance block. Both fields null means the Worker fell back to bootstrap
// params (no params.json published yet) — an honest "no model version", not a
// missing field. `key` is derived, never read: the versioned object the live
// pointer's trained_at maps to, so a consumer can pin the exact params.
function paramsProvenance(trained: TrainedParams | null): ParamsProvenance {
  if (trained === null) return { trained_at: null, key: null };
  return {
    trained_at: trained.trained_at,
    key: versionedParamsKey(trained.trained_at),
  };
}

export function buildSnapshot(args: {
  generatedAt: number;
  alertsFreshness: number;
  routeSnapshots: Map<string, RouteSnapshot>;
  rolls: Record<string, RouteRoll>;
  trainedParams: TrainedParams | null;
  tickSeconds: number;
  /** Cached station_status, refreshed on hourly E&E fetches. Empty when
   * E&E hasn't been parsed yet (e.g. before the first hourly tick after
   * deploy). */
  stationStatuses?: Record<string, StationStatus>;
  eneFreshness?: number | null;
  /** Alerts active this tick — the atomic objects route_status IDs resolve
   * against. Empty only on a true alerts-feed gap. */
  alerts?: AlertOut[];
  /** Elevators/escalators currently out, cached from the hourly E&E fetch. */
  equipment?: EquipmentOut[];
  /** Static station metadata, cached from the daily 39hk-dx4f fetch. */
  stations?: Record<string, StationOut>;
  /** Epoch the served station metadata was fetched, or null before first fetch. */
  stationsStaticFreshness?: number | null;
  /** Last tick's movement regimes, keyed by route. Routes present here have
   * their published `condition` observed from train movement; absent routes
   * publish 'unknown'. Null/undefined before the first vehicle tick after
   * deploy. Lagged one tick (~5 min) — see state.MOVEMENT_STATE_KEY. */
  movementStates?: {
    observed_at: number;
    regimes: Record<string, { state: string; entered_at: number }>;
    // Per-route service-level regimes; the published service_condition (supply
    // axis). Absent on docs from before the axis or a params set with no hourly
    // service baseline — service_condition then reads 'unknown'.
    service_regimes?:
      Record<string, { state: string; entered_at: number }> | undefined;
    // Per-route raw service ratio behind service_condition. Same lifecycle as
    // service_regimes; absent -> service_ratio null.
    service_ratios?: Record<string, number> | undefined;
    // Per-route quantile-derived low/high ratios behind service_low_ratio/
    // service_high_ratio. Same lifecycle as service_ratios, plus absence when
    // the trainer hasn't published per-cell quantiles at all.
    service_quantile_ratios?: Record<string, { low: number; high: number }> | undefined;
  } | null;
  /** Last tick's per-station service flow, one-tick lagged like movementStates.
   * Null/undefined before the first vehicle tick after deploy. */
  stationFlow?: StationFlowDoc | null;
  /** Last tick's per-segment regimes (route|direction|from_stop cells), one-
   * tick lagged like stationFlow — segment_flow.json's regimes, written at
   * step 8b. Null/undefined before the first vehicle tick after deploy. */
  segmentFlow?: SegmentFlowDoc | null;
  /** Segment topology (successor stop + reliability annotation). Trainer-
   * owned and read fresh, not tick-lagged. Null when absent. */
  segmentParams?: SegmentParamsDoc | null;
  /** Per-segment dwell curves the segment recovery is conditioned on. Null
   * until the trainer publishes segment_dwell.json — segments then publish
   * status without recovery, never a fabricated number. */
  segmentDwell?: SegmentDwellDoc | null;
  /** This tick's own platform-wait state (state/station_wait.json),
   * threaded through in-memory from index.ts step 0 rather than read back
   * from R2 — unlike stationFlow/segmentFlow, it is NOT one-tick-lagged.
   * Null/undefined before the first vehicle tick after deploy, or when
   * step 0's read/update/write failed this tick. */
  stationWait?: StationWaitDoc | null;
  /** Per-station-complex entries/min baseline from training/ridership.py.
   * Read fresh each tick, not tick-lagged — it only changes weekly. Null
   * until the trainer's first run, or when the document fails validation;
   * platform_crowding then publishes null rather than estimate off nothing. */
  ridershipBaseline?: RidershipBaselineDoc | null;
  /** Per-directional-platform scheduled-service baseline from
   * training/service_weight.py. Read fresh each tick, not tick-lagged — it
   * only changes weekly. Null before the first run or when the document fails
   * validation; the crowding split then falls back to even-over-served rather
   * than abstaining (unlike a missing ridership baseline, which is fatal to
   * the surface). */
  serviceWeightBaseline?: ServiceWeightBaselineDoc | null;
}): Snapshot {
  const route_status: Record<string, RouteStatusOut> = {};

  // Use movement state only while it's reasonably fresh; a long vehicle-feed
  // gap shouldn't pin a stale condition on the public surface.
  const movementFresh =
    args.movementStates != null &&
    args.generatedAt - args.movementStates.observed_at <=
      MAX_MOVEMENT_STATE_AGE_SEC;
  const movementStates = movementFresh ? args.movementStates : null;
  const stationFlowFresh =
    args.stationFlow != null &&
    args.generatedAt - args.stationFlow.observed_at <=
      MAX_MOVEMENT_STATE_AGE_SEC;
  const segmentFlow = args.segmentFlow ?? null;
  const segmentFlowFresh =
    segmentFlow != null &&
    args.generatedAt - segmentFlow.observed_at <= MAX_MOVEMENT_STATE_AGE_SEC;
  const segmentFlowOut =
    segmentFlow != null && segmentFlowFresh
      ? buildSegmentFlowOut(
          segmentFlow,
          args.segmentParams ?? null,
          args.segmentDwell ?? null,
          args.generatedAt,
        )
      : null;
  const stationFlowOut =
    stationFlowFresh && args.stationFlow != null
      ? buildStationFlowOut(args.stationFlow, segmentFlowOut)
      : null;

  // The freshest-decaying published surface: unlike stationFlow/segmentFlow
  // (a one-tick-lagged read, gated the same way), stationWait is this same
  // tick's own just-written doc, so under normal operation this gate is
  // never the reason it's absent — it exists to catch the case step 0's
  // read/write failed and left stationWait stale or null. The gate matters
  // more here than anywhere else: a stale station_flow just shows a wrong
  // status, but a stale platform_crowding shows a wrong and ever-more-wrong
  // number, because the crowd it describes keeps growing at entries_per_min
  // for every minute it goes unrefreshed.
  const stationWaitFresh =
    args.stationWait != null
    && args.generatedAt - args.stationWait.observed_at <= MAX_MOVEMENT_STATE_AGE_SEC;
  const platformCrowdingOut =
    stationWaitFresh && args.stationWait != null && args.ridershipBaseline != null
      ? buildPlatformCrowdingOut(
          args.stationWait,
          args.ridershipBaseline,
          args.serviceWeightBaseline ?? null,
          args.stations ?? {},
          args.generatedAt,
        )
      : null;

  // Publish every route we have alpha for — good-service lines get their
  // inference too. Union with current routeSnapshots in case a route just got
  // its first alert this tick (alpha entry written after buildSnapshot reads).
  const allRouteIds = new Set<string>([
    ...Object.keys(args.rolls),
    ...args.routeSnapshots.keys(),
    ...Object.keys(movementStates?.regimes ?? {}),
  ]);

  for (const routeId of allRouteIds) {
    const snap = args.routeSnapshots.get(routeId);
    const roll = args.rolls[routeId];
    const activeAlerts = snap?.active_alert_ids ?? [];
    const schedule: ScheduleFacts = {
      isNotScheduled: snap?.is_not_scheduled ?? false,
      hasRealtimeAlert: snap?.has_realtime_alert ?? false,
      scheduledResumeAt: snap?.scheduled_resume_at ?? null,
    };
    const movementRegime = movementStates?.regimes[routeId] ?? null;
    const serviceRegime = movementStates?.service_regimes?.[routeId] ?? null;
    const inference: Inference | null = roll
      ? buildInference(
          roll,
          args.generatedAt,
          args.tickSeconds,
          routeId,
          args.trainedParams,
          snap?.observation.alert_count ?? 0,
          schedule,
          movementRegime,
        )
      : null;

    const label = snap?.coarse_label ?? NO_ALERTS_FALLBACK;
    // Current state is movement-primary: train movement is the published answer to
    // "is this route disrupted right now". Alerts never assert the condition — the
    // alert-derived read lives on as the shadow (inference.condition) and the cause
    // (category / primary_alert_type). A planned "No Scheduled Service" alert is the
    // one exception: a planned non-run wins. When movement can't judge (cold start,
    // feed gap, thin/absent signal) the condition is 'unknown' — an honest coverage
    // gap, never an alert-derived fallback.
    const { condition, source: condition_source } = resolvePublishedCondition(
      schedule,
      movementRegime,
    );
    const serviceRatio = movementStates?.service_ratios?.[routeId] ?? null;
    const serviceLowRatio =
      movementStates?.service_quantile_ratios?.[routeId]?.low ?? null;
    const serviceHighRatio =
      movementStates?.service_quantile_ratios?.[routeId]?.high ?? null;
    route_status[routeId] = {
      route_id: routeId,
      alerts: activeAlerts,
      condition,
      condition_source,
      service_condition: serviceRegime?.state ?? "unknown",
      service_ratio: serviceRatio,
      service_low_ratio: serviceLowRatio,
      service_high_ratio: serviceHighRatio,
      service_percentile: servicePercentile(
        serviceRatio,
        serviceLowRatio,
        serviceHighRatio,
      ),
      category: categoryForLabel(label),
      primary_alert_type: snap?.primary_alert_type ?? null,
      label,
      by_direction: snap?.by_direction ?? {
        northbound: { alerts: [], primary_alert_type: null },
        southbound: { alerts: [], primary_alert_type: null },
      },
      inference,
    };
  }

  const system = buildSystemStatus(
    route_status,
    args.routeSnapshots,
    args.stationStatuses ?? {},
  );
  const compat = buildCompat(route_status, args.routeSnapshots);

  return {
    schema_version: SCHEMA_VERSION,
    generated_at: args.generatedAt,
    provenance: { ...codeProvenance(), params: paramsProvenance(args.trainedParams) },
    attribution: ATTRIBUTION,
    supported_modes: ["subway"],
    freshness: {
      subway_alerts: args.alertsFreshness,
      lirr_alerts: null,
      mnr_alerts: null,
      bus_alerts: null,
      path_alerts: null,
      ferry_alerts: null,
      ene: args.eneFreshness ?? null,
      stations_static: args.stationsStaticFreshness ?? null,
    },
    alerts: args.alerts ?? [],
    observations: [],
    routes: buildRoutes(),
    stations: args.stations ?? {},
    equipment: args.equipment ?? [],
    bridges: [],
    tunnels: [],
    route_status,
    station_status: args.stationStatuses ?? {},
    station_flow: stationFlowOut,
    segment_flow: segmentFlowOut,
    platform_crowding: platformCrowdingOut,
    system,
    compat,
  };
}

function buildSystemStatus(
  routeStatuses: Record<string, RouteStatusOut>,
  routeSnapshots: Map<string, RouteSnapshot>,
  stationStatuses: Record<string, StationStatus>,
): SystemStatus {
  const routes_with_alerts: string[] = [];
  let alert_count = 0;
  let severity_max = 0;
  for (const [routeId, rs] of Object.entries(routeStatuses)) {
    if (rs.alerts.length > 0) {
      routes_with_alerts.push(routeId);
      alert_count += rs.alerts.length;
    }
    // severity_max is the alert severity of the worst movement-confirmed
    // disruption — alerts explain a disruption's severity but never assert one, so
    // a route reading normal/unknown/not_scheduled folds no alert severity in.
    if (rs.condition !== "disrupted" && rs.condition !== "suspended") continue;
    const snap = routeSnapshots.get(routeId);
    if (snap && snap.severity_max > severity_max)
      severity_max = snap.severity_max;
  }
  routes_with_alerts.sort();

  let lines_disrupted_count = 0;
  let most_degraded_line: string | null = null;
  let mostDegradedScore = -1;
  let most_recovered_line: string | null = null;
  let mostRecoveredEnteredAt = -1;
  for (const [routeId, rs] of Object.entries(routeStatuses)) {
    const inf = rs.inference;
    // Count what's published: `condition` is the movement-primary current state
    // (a movement-only route has inference null). Ranking within (most degraded /
    // recovered) still uses the HMM's continuous probabilities and regime age when
    // present — a movement-only disrupted route scores a flat 1.
    const disrupted =
      rs.condition === "disrupted" || rs.condition === "suspended";
    if (disrupted) {
      lines_disrupted_count += 1;
      const score = inf ? inf.p_disrupted + inf.p_suspended : 1;
      if (score > mostDegradedScore) {
        mostDegradedScore = score;
        most_degraded_line = routeId;
      }
    } else if (
      rs.condition === "normal" &&
      inf &&
      inf.regime_entered_at > mostRecoveredEnteredAt
    ) {
      mostRecoveredEnteredAt = inf.regime_entered_at;
      most_recovered_line = routeId;
    }
  }

  return {
    by_mode: {
      subway: { routes_with_alerts, alert_count, severity_max },
    },
    accessibility: buildAccessibility(stationStatuses),
    overall_label:
      routes_with_alerts.length === 0
        ? "All systems normal"
        : `Alerts on ${routes_with_alerts.length} subway lines`,
    condition: null,
    lines_disrupted_count,
    most_degraded_line,
    most_recovered_line,
  };
}

function buildAccessibility(
  stationStatuses: Record<string, StationStatus>,
): Accessibility {
  let elevators_out = 0;
  let escalators_out = 0;
  let ada_pathways_degraded = 0;
  for (const s of Object.values(stationStatuses)) {
    elevators_out += s.elevators_out;
    escalators_out += s.escalators_out;
    if (s.ada_status === "ada_degraded") ada_pathways_degraded += 1;
  }
  return { elevators_out, escalators_out, ada_pathways_degraded };
}

function buildCompat(
  routeStatuses: Record<string, RouteStatusOut>,
  routeSnapshots: Map<string, RouteSnapshot>,
): Compat {
  const subwaynow_routes: Record<string, CompatRoute> = {};
  for (const [routeId, rs] of Object.entries(routeStatuses)) {
    const meta = metaForRoute(routeId);
    const snap = routeSnapshots.get(routeId);
    const refs = snap?.alerts ?? [];

    const direction_statuses: CompatRouteSummary = {
      north: rs.by_direction.northbound.primary_alert_type
        ? coarseStatus(rs.by_direction.northbound.primary_alert_type)
        : null,
      south: rs.by_direction.southbound.primary_alert_type
        ? coarseStatus(rs.by_direction.southbound.primary_alert_type)
        : null,
    };

    const northRefs = refs.filter(
      (r) => r.direction_id === 0 || r.direction_id === null,
    );
    const southRefs = refs.filter(
      (r) => r.direction_id === 1 || r.direction_id === null,
    );

    const delayKeywords = ["delay"];
    const irregularityKeywords = ["slow", "reroute", "skip"];
    const changeKeywords = ["service change", "suspend", "express", "local"];

    const delay_summaries: CompatRouteSummary = {
      north: firstHeaderMatching(northRefs, delayKeywords),
      south: firstHeaderMatching(southRefs, delayKeywords),
    };
    const service_irregularity_summaries: CompatRouteSummary = {
      north: firstHeaderMatching(northRefs, irregularityKeywords),
      south: firstHeaderMatching(southRefs, irregularityKeywords),
    };
    const service_change_summaries: CompatServiceChangeSummary = {
      both: headersMatching(refs, changeKeywords),
      north: [],
      south: [],
    };

    // not_scheduled renders as a scheduled gap so the HomeAssistant integration
    // doesn't choke on an unknown status. Otherwise `status` stays the alert-derived
    // coarse label (shadow) — this legacy compat surface intentionally lags the
    // movement-primary route_status.condition; a movement-derived status mapping is
    // deferred to the HA integration work so its contract isn't changed blind.
    const notScheduled = rs.condition === "not_scheduled";
    subwaynow_routes[routeId] = {
      id: routeId,
      name: meta.name,
      color: meta.color,
      status: notScheduled ? "Not Scheduled" : rs.label,
      scheduled: !notScheduled,
      direction_statuses,
      delay_summaries,
      service_irregularity_summaries,
      service_change_summaries,
    };
  }
  return { subwaynow_routes };
}

function firstHeaderMatching(
  refs: AlertRef[],
  keywords: string[],
): string | null {
  for (const r of refs) {
    if (!r.header_text) continue;
    const typeLower = r.alert_type.toLowerCase();
    if (keywords.some((k) => typeLower.includes(k))) return r.header_text;
  }
  return null;
}

function headersMatching(refs: AlertRef[], keywords: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const r of refs) {
    if (!r.header_text) continue;
    const typeLower = r.alert_type.toLowerCase();
    if (!keywords.some((k) => typeLower.includes(k))) continue;
    if (seen.has(r.header_text)) continue;
    seen.add(r.header_text);
    out.push(r.header_text);
  }
  return out;
}

interface ScheduleFacts {
  isNotScheduled: boolean;
  hasRealtimeAlert: boolean;
  scheduledResumeAt: number | null;
}

// Last known movement-clock reading for a route: the SAME (state, entered_at)
// buildSnapshot publishes as condition/condition_source === 'movement'. null
// when movement can't judge (no reading, or the reading fell outside
// MAX_MOVEMENT_STATE_AGE_SEC upstream).
interface MovementRegime {
  state: string;
  entered_at: number;
}

// Which arm decides the published condition. buildSnapshot reads the condition
// off it and buildInference gates the forecast on it, so the precedence lives
// here once — a second copy that drifted would publish a forecast about a
// condition nobody was shown.
type ConditionSource = "schedule" | "movement" | "unknown";

function resolvePublishedCondition(
  schedule: ScheduleFacts,
  movementRegime: MovementRegime | null,
): { condition: string; source: ConditionSource } {
  if (schedule.isNotScheduled)
    return { condition: "not_scheduled", source: "schedule" };
  if (movementRegime !== null) {
    return { condition: movementRegime.state, source: "movement" };
  }
  return { condition: "unknown", source: "unknown" };
}

// atom_p/atom_sec activate the mixture closed form in pLeaveBy/conditionalRecovery
// only together, and only when atom_p is a genuine probability — exactly 0 or 1
// would either no-op or swallow the whole tail, not the intended point mass. Any
// other combination (fields absent, older params.json, an out-of-range value)
// returns undefined and callers fall back to the pure curve_sec/tail_ll path,
// byte-for-byte what it was before this mixture was added.
function atomFor(cell: DwellQuantiles): { p: number; sec: number } | undefined {
  if (cell.atom_p === undefined || cell.atom_sec === undefined)
    return undefined;
  if (!(cell.atom_p > 0 && cell.atom_p < 1)) return undefined;
  // A non-positive atom location would place the point mass at or before t=0 and
  // apply it to every elapsed value. dwell.ts rejects it too; this keeps the
  // rejection visible at the boundary where params first become a forecast.
  if (!(cell.atom_sec > 0)) return undefined;
  return { p: cell.atom_p, sec: cell.atom_sec };
}

function buildInference(
  roll: RouteRoll,
  now: number,
  tickSeconds: number,
  routeId: string,
  trained: TrainedParams | null,
  disruptiveAlertCount: number,
  schedule: ScheduleFacts,
  movementRegime: MovementRegime | null,
): Inference {
  const probs = roll.filter.probabilities;
  const params = paramsForRoute(trained, routeId);

  // Set only by the two arms that can also decide the published condition —
  // see the gate at the return. The alert arm still estimates recovery_minutes
  // below, but it no longer produces a forecast: on its own clock and its own
  // regime, that number described a condition nobody was shown.
  let p_normal_in_30: number | null = null;
  let p_normal_in_60: number | null = null;
  let p_normal_in_120: number | null = null;

  const argmaxIdx = argmaxOf(probs);

  const condition = resolveCondition(roll, disruptiveAlertCount, schedule);

  // Recovery_minutes is "time until back to normal," picked in order of
  // preference:
  //   1. Schedule countdown — a planned-work alert announces its own resume
  //      time; deterministic, no model uncertainty.
  //   2. Movement curve + movement clock — when the PUBLISHED condition came
  //      from movement (see buildSnapshot), condition on how long the
  //      movement regime has run, off the movement dwell curve. This is the
  //      regime consumers are actually shown; the HMM's own condition can
  //      silently disagree with it (effectiveCondition's zero-alert guard).
  //   3. Alert-HMM empirical dwell from the regime_transitions stream —
  //      heavy-tailed reality, not a geometric approximation. Prefers the
  //      cause-conditioned (route, condition, alert_type_at_entry) cell,
  //      falling back to the (route, condition) aggregate.
  //   4. Geometric dwell from the trained transition self-loop — works
  //      everywhere but saturates at the clamp ceiling for any route with
  //      sustained planned-work alerts.
  let recovery_minutes = 0;
  let recovery_minutes_low = 0;
  let recovery_minutes_high = 0;
  let recovery_indeterminate = false;
  let recovery_source: "hmm" | "schedule" | "movement" = "hmm";
  let resumes_at: number | null = null;
  let overdue = false;

  // What consumers are actually shown. Every arm below is judged against it,
  // not against the alert-HMM shadow: `condition` above hard-returns normal
  // whenever there are no disruptive alerts, so a route published
  // not_scheduled overnight used to look "normal" here and lose its
  // deterministic countdown to an alert-regime dwell estimate.
  const publishedCondition = resolvePublishedCondition(
    schedule,
    movementRegime,
  ).condition;

  // Whether "when is it back" is even a question for this route. Published
  // normal: nothing to recover from. Published unknown: we declined to judge,
  // and answering anyway would assert the disruption we just said we could not
  // see. Both the schedule countdown and the withholding gate below key off
  // this one predicate so they cannot disagree about `unknown`.
  const publishedNotNormal =
    publishedCondition !== "normal" && publishedCondition !== "unknown";

  // A planned-work disruption announces its own resume time (the window end),
  // so recovery is a deterministic schedule lookup, not a dwell estimate — for
  // ALL planned_work, not just no-service. Real-time alerts have no trustworthy
  // end and keep HMM recovery; when both are present the real-time alert wins.
  const scheduleRecovery =
    publishedNotNormal &&
    !schedule.hasRealtimeAlert &&
    schedule.scheduledResumeAt !== null;

  // Movement curve, conditioned on the movement clock — only when the
  // published condition actually came from movement (mirrors buildSnapshot's
  // own condition_source precedence: schedule.isNotScheduled beats movement).
  // null whenever there's no movement regime, no clock (entered_at <= 0), or
  // the trainer hasn't published dwell_movement for this cell yet.
  const movement =
    !schedule.isNotScheduled && movementRegime !== null
      ? movementRecovery(
          trained,
          routeId,
          movementRegime.state,
          movementRegime.entered_at,
          now,
        )
      : null;

  if (scheduleRecovery) {
    const resume = schedule.scheduledResumeAt!;
    recovery_source = "schedule";
    resumes_at = resume;
    // now has passed the announced resume but the alert is still active this
    // tick — clamp to 0 rather than count down past it. Next tick an extension
    // or a newly-posted real-time alert takes over via precedence.
    overdue = now >= resume;
    const remaining = Math.max(0, Math.round((resume - now) / 60));
    recovery_minutes = remaining;
    recovery_minutes_low = remaining;
    recovery_minutes_high = remaining;
    // It's back at the announced time: P(normal in k) is 1 once the window end
    // falls within k minutes, else 0.
    const within = (mins: number): number =>
      resume <= now + mins * 60 ? 1 : 0;
    p_normal_in_30 = within(30);
    p_normal_in_60 = within(60);
    p_normal_in_120 = within(120);
  } else if (movement !== null) {
    recovery_source = "movement";
    recovery_minutes = movement.recovery_minutes;
    recovery_minutes_low = movement.recovery_minutes_low;
    recovery_minutes_high = movement.recovery_minutes_high;
    recovery_indeterminate = movement.recovery_indeterminate;
    p_normal_in_30 = movement.p_normal_in_30;
    p_normal_in_60 = movement.p_normal_in_60;
    p_normal_in_120 = movement.p_normal_in_120;
  } else if (condition !== "normal") {
    const clamp = (m: number): number => Math.min(m, MAX_RECOVERY_MINUTES);
    const empirical = dwellForRouteState(
      trained,
      routeId,
      condition,
      roll.alert_type_at_entry,
    );
    if (empirical !== null) {
      const secToMin = (s: number): number => Math.round(s / 60);
      // Condition on how long the regime has already lasted: for heavy-tailed
      // dwells the unconditional quantiles/fractions are only correct at
      // elapsed=0, so recovery is the *remaining* time.
      const elapsedSec = Math.max(0, now - roll.filter.regime_entered_at);

      if (empirical.curve_sec !== undefined) {
        const curve = empirical.curve_sec;
        const tail = empirical.tail_ll;
        const atom = atomFor(empirical);
        const conditional = conditionalRecovery(curve, elapsedSec, tail, atom);
        if (conditional !== null) {
          recovery_minutes = clamp(secToMin(conditional.median_sec));
          recovery_minutes_low = clamp(secToMin(conditional.q25_sec));
          recovery_minutes_high = clamp(secToMin(conditional.q75_sec));
          recovery_indeterminate = recovery_minutes >= MAX_RECOVERY_MINUTES;
        } else {
          // Outlived every observed dwell — no trustworthy recovery time.
          recovery_minutes = MAX_RECOVERY_MINUTES;
          recovery_minutes_low = MAX_RECOVERY_MINUTES;
          recovery_minutes_high = MAX_RECOVERY_MINUTES;
          recovery_indeterminate = true;
        }
      } else {
        // Pre-curve params.json: unconditional cell values (legacy behavior
        // until the trainer republishes with curve_sec).
        recovery_minutes = clamp(secToMin(empirical.median_sec));
        recovery_minutes_low = clamp(secToMin(empirical.q25_sec));
        recovery_minutes_high = clamp(secToMin(empirical.q75_sec));
        recovery_indeterminate = recovery_minutes >= MAX_RECOVERY_MINUTES;
        // Legacy cells carry unconditional recover_by_* fractions; those were
        // the alert arm's forecast and are no longer published.
      }
    } else {
      const selfLoop = params.transition[argmaxIdx]![argmaxIdx]!;
      const dwellTicks = dwellQuantiles(selfLoop);
      const dwellToMinutes = (t: number): number =>
        Math.round((t * tickSeconds) / 60);
      const rawMedian = dwellToMinutes(dwellTicks.median);
      recovery_indeterminate = rawMedian >= MAX_RECOVERY_MINUTES;
      recovery_minutes = clamp(rawMedian);
      recovery_minutes_low = clamp(dwellToMinutes(dwellTicks.q25));
      recovery_minutes_high = clamp(dwellToMinutes(dwellTicks.q75));
    }
  }

  // The filter is still settling when: the route just appeared (regime younger
  // than the hysteresis window), the published label hasn't cleared hysteresis,
  // or we're recovering from a feed gap ("unknown").
  const model_warming_up =
    roll.published.label === PUBLISHED_UNKNOWN ||
    roll.published.pending_streak < HYSTERESIS_TICKS ||
    now - roll.filter.regime_entered_at < HYSTERESIS_TICKS * tickSeconds;
  // The forecast is published only when it is about the condition we publish.
  // Both arms below are: movement shares the published condition's arm, clock
  // and regime, and the schedule countdown can only fire when the published
  // condition is already not normal. Anything else is the alert arm
  // forecasting its own regime on its own clock, which is not the regime
  // consumers were shown. Gated here, at the single point where the inference
  // is assembled, rather than in each arm: there are five ways these values
  // get set and a per-arm gate would silently miss the next one added.
  const forecastsThePublishedCondition =
    recovery_source === "movement" || recovery_source === "schedule";

  // recovery_minutes is the same claim in minutes, so it gets the same gate,
  // wherever the question arises at all. Measured over 6 days of the prediction
  // stream: alert-arm estimates shown against schedule-published not_scheduled
  // routes missed by a mean of 1,135 minutes over 4,258 rows (median 680) —
  // they were timing the alert regime's return, not the route's. Withheld the
  // way the outlived-every-dwell case already is: recovery_indeterminate says
  // the number is not a prediction and the value carries the ceiling, rather
  // than a fabricated estimate. Not null, because recovery_minutes is an
  // integer in the published snapshot contract that external consumers read.
  if (publishedNotNormal && !forecastsThePublishedCondition) {
    recovery_minutes = MAX_RECOVERY_MINUTES;
    recovery_minutes_low = MAX_RECOVERY_MINUTES;
    recovery_minutes_high = MAX_RECOVERY_MINUTES;
    recovery_indeterminate = true;
  }

  return {
    condition,
    recovery_minutes,
    // Shadow-HMM disruption flag: whether the alert-derived regime reads a live
    // disruption. The PUBLISHED disruption is route_status.condition (movement-
    // primary); this tracks the HMM view that also anchors the recovery forecast.
    // normal (incl. planned-only, zero realtime alerts) and not_scheduled never count.
    is_disrupted: condition !== "normal" && condition !== "not_scheduled",
    p_normal: probs[0],
    p_disrupted: probs[1],
    p_suspended: probs[2],
    regime_entered_at: roll.filter.regime_entered_at,
    regime_age_seconds: Math.max(0, now - roll.filter.regime_entered_at),
    recovery_minutes_low,
    recovery_minutes_high,
    recovery_indeterminate,
    p_normal_in_30min: forecastsThePublishedCondition ? p_normal_in_30 : null,
    // Same gate, plus the measured one: only the schedule arm's deterministic
    // countdown survives at these horizons — see the Inference interface above.
    p_normal_in_60min:
      forecastsThePublishedCondition && recovery_source === "schedule"
        ? p_normal_in_60
        : null,
    p_normal_in_120min:
      forecastsThePublishedCondition && recovery_source === "schedule"
        ? p_normal_in_120
        : null,
    model_warming_up,
    recovery_source,
    resumes_at,
    overdue,
  };
}

interface MovementRecoveryResult {
  recovery_minutes: number;
  recovery_minutes_low: number;
  recovery_minutes_high: number;
  recovery_indeterminate: boolean;
  p_normal_in_30: number;
  p_normal_in_60: number;
  p_normal_in_120: number;
}

// Movement states whose exit destination has actually been measured, so an
// exit probability can honestly be read as P(normal). Anything outside this
// table falls back to the alert arm rather than assuming an unobserved split.
const MOVEMENT_SPLIT_MEASURED: Record<string, true> = {
  normal: true,
  disrupted: true,
};

/**
 * Recovery + p_normal_in_H off the movement curve, conditioned on the
 * movement clock (elapsed = now - entered_at) — the SAME regime consumers see
 * as condition/condition_source==='movement'. Returns null when there's no
 * usable clock or no trained curve for this (route, state) cell, so the
 * caller falls back to the alert-HMM path.
 *
 * dwell_movement is route-scope, all-cause (C2) — no trained split of "exits
 * to normal" vs "exits to a worse state" exists for the movement clock the
 * way the HMM transition matrix supplies one for the alert arm (see toNormal
 * below). Measured instead of assumed (2026-08-11, murk exec +
 * training.movement_backfill route-scope reconstruction): every completed
 * disrupted-regime exit went to normal — 17/17 on the faithful
 * published_condition source (2026-08-04..2026-08-12, 6 routes, no route
 * below 100%) and 213/213 on the higher-volume vehicle-archive source
 * (2026-06-21..2026-08-12, 19 routes). A disrupted cell's exit probability is
 * therefore read directly as p_normal_in_H, and a normal cell needs no split
 * at all: exits from normal are never "to normal", so p_normal_in_H is the
 * plain survival function.
 *
 * suspended is deliberately NOT served here. It completed no route-scope
 * episode in either window (n=0), and no observations is not evidence of a
 * 100% return to normal — a route resuming from having no trains at all can
 * plausibly come back degraded before it comes back normal. Those fall through
 * to the alert arm, which has a trained transition matrix to split on.
 */
function movementRecovery(
  trained: TrainedParams | null,
  routeId: string,
  state: string,
  enteredAt: number,
  now: number,
): MovementRecoveryResult | null {
  if (enteredAt <= 0) return null;
  if (MOVEMENT_SPLIT_MEASURED[state] !== true) return null;
  const cell = movementDwellFor(trained, routeId, state);
  if (cell?.curve_sec === undefined) return null;
  const curve = cell.curve_sec;
  const tail = cell.tail_ll;
  const atom = atomFor(cell);
  const elapsedSec = Math.max(0, now - enteredAt);

  if (state === "normal") {
    const staysNormalFor = (horizonSec: number): number =>
      1 - pLeaveBy(curve, elapsedSec, horizonSec, tail, atom);
    return {
      recovery_minutes: 0,
      recovery_minutes_low: 0,
      recovery_minutes_high: 0,
      recovery_indeterminate: false,
      p_normal_in_30: staysNormalFor(1800),
      p_normal_in_60: staysNormalFor(3600),
      p_normal_in_120: staysNormalFor(7200),
    };
  }

  const clamp = (m: number): number => Math.min(m, MAX_RECOVERY_MINUTES);
  const secToMin = (s: number): number => Math.round(s / 60);
  const conditional = conditionalRecovery(curve, elapsedSec, tail, atom);
  let recovery_minutes: number;
  let recovery_minutes_low: number;
  let recovery_minutes_high: number;
  let recovery_indeterminate: boolean;
  if (conditional !== null) {
    recovery_minutes = clamp(secToMin(conditional.median_sec));
    recovery_minutes_low = clamp(secToMin(conditional.q25_sec));
    recovery_minutes_high = clamp(secToMin(conditional.q75_sec));
    recovery_indeterminate = recovery_minutes >= MAX_RECOVERY_MINUTES;
  } else {
    // Outlived every observed dwell — no trustworthy recovery time.
    recovery_minutes = MAX_RECOVERY_MINUTES;
    recovery_minutes_low = MAX_RECOVERY_MINUTES;
    recovery_minutes_high = MAX_RECOVERY_MINUTES;
    recovery_indeterminate = true;
  }
  return {
    recovery_minutes,
    recovery_minutes_low,
    recovery_minutes_high,
    recovery_indeterminate,
    p_normal_in_30: pLeaveBy(curve, elapsedSec, 1800, tail, atom),
    p_normal_in_60: pLeaveBy(curve, elapsedSec, 3600, tail, atom),
    p_normal_in_120: pLeaveBy(curve, elapsedSec, 7200, tail, atom),
  };
}

/**
 * Recovery for a DISRUPTED segment cell off its own dwell curve,
 * conditioned on the segment regime clock (elapsed = now - entered_at) —
 * the same math as movementRecovery, sourced from segment_dwell.json
 * instead of dwell_movement. Returns null when there's no usable clock
 * (entered_at <= 0) or no trained curve for this (key, state) cell, so the
 * caller publishes status without a fabricated recovery. Never called for
 * a NORMAL cell — buildSegmentFlowOut publishes `recovery: null` for one
 * directly, without a curve lookup: a healthy segment has nothing to
 * forecast, so there is no "recovery from normal" to compute.
 */
function segmentRecovery(
  dwell: SegmentDwellDoc | null,
  key: string,
  state: string,
  enteredAt: number,
  now: number,
): SegmentRecovery | null {
  if (enteredAt <= 0) return null;
  const cell = dwell?.cells[key]?.[state];
  if (cell === undefined) return null;
  const curve = cell.curve_sec;
  const tail = cell.tail_ll;
  const elapsedSec = Math.max(0, now - enteredAt);

  const clamp = (m: number): number => Math.min(m, MAX_RECOVERY_MINUTES);
  const secToMin = (s: number): number => Math.round(s / 60);
  const conditional = conditionalRecovery(curve, elapsedSec);
  let recovery_minutes: number;
  let recovery_minutes_low: number;
  let recovery_minutes_high: number;
  let recovery_indeterminate: boolean;
  if (conditional !== null) {
    recovery_minutes = clamp(secToMin(conditional.median_sec));
    recovery_minutes_low = clamp(secToMin(conditional.q25_sec));
    recovery_minutes_high = clamp(secToMin(conditional.q75_sec));
    recovery_indeterminate = recovery_minutes >= MAX_RECOVERY_MINUTES;
  } else {
    // Outlived every observed dwell — no trustworthy recovery time.
    recovery_minutes = MAX_RECOVERY_MINUTES;
    recovery_minutes_low = MAX_RECOVERY_MINUTES;
    recovery_minutes_high = MAX_RECOVERY_MINUTES;
    recovery_indeterminate = true;
  }
  return {
    recovery_minutes,
    recovery_minutes_low,
    recovery_minutes_high,
    recovery_indeterminate,
    p_normal_in_30min: pLeaveBy(curve, elapsedSec, 1800, tail),
    p_normal_in_60min: pLeaveBy(curve, elapsedSec, 3600, tail),
    p_normal_in_120min: pLeaveBy(curve, elapsedSec, 7200, tail),
  };
}

/**
 * Per-segment published surface: debounced status + clock + successor stop +
 * expected recovery, keyed the same way as segment_flow.json/
 * segment_params.json (`route|direction|from_stop`) — every judged cell,
 * normal and disrupted alike. A NORMAL cell's recovery is always null: a
 * healthy segment has nothing to forecast, so its curve is never even
 * looked up. Sibling of station_flow; caller gates freshness before
 * calling this.
 */
function buildSegmentFlowOut(
  flow: SegmentFlowDoc,
  params: SegmentParamsDoc | null,
  dwell: SegmentDwellDoc | null,
  now: number,
): SegmentFlowOut {
  const segments: Record<string, SegmentStatusOut> = {};
  for (const [key, regime] of Object.entries(flow.regimes)) {
    const parts = key.split("|");
    const route = parts[0] ?? "";
    const direction = parts[1] ?? "";
    const from_stop = parts[2] ?? "";
    segments[key] = {
      route,
      direction,
      from_stop,
      to: params?.adjacency[key]?.to ?? null,
      status: regime.state,
      entered_at: regime.entered_at,
      recovery:
        regime.state === "disrupted"
          ? segmentRecovery(dwell, key, regime.state, regime.entered_at, now)
          : null,
    };
  }
  return { observed_at: flow.observed_at, segments };
}

/**
 * Recovery for a station's already-selected worst touching segment (see
 * deriveStationFlow). worst_segment is only a (from, to) stop pair — when
 * more than one route shares that physical track (common on shared
 * trackage) more than one live segment key can match; they describe the
 * same movement, so picking the lexicographically first key is a stable
 * tie-break, not a second "worst" determination. A worst_segment that
 * currently reads NORMAL still has an entry in `segments` like any other
 * judged cell, but its `recovery` is always null — there is nothing more
 * informative to say about a segment that isn't disrupted than that it
 * isn't, so this returns null for that case exactly as it does for "never
 * judged".
 */
function worstSegmentRecovery(
  worstSegment: readonly [string, string] | null,
  segments: Record<string, SegmentStatusOut>,
): SegmentRecovery | null {
  if (worstSegment === null) return null;
  const [from, to] = worstSegment;
  for (const key of Object.keys(segments).sort()) {
    const s = segments[key]!;
    if (s.from_stop === from && s.to === to) return s.recovery;
  }
  return null;
}

/**
 * station_flow, additive: every entry keeps its existing fields and gains
 * worst_recovery — the expected recovery of the same worst_segment
 * deriveStationFlow already picked, off the fresh segment surface above.
 */
function buildStationFlowOut(
  stationFlow: StationFlowDoc,
  segmentFlowOut: SegmentFlowOut | null,
): StationFlowOut {
  const stations: Record<string, StationServiceFlowOut> = {};
  for (const [sid, s] of Object.entries(stationFlow.stations)) {
    stations[sid] = {
      ...s,
      worst_recovery: segmentFlowOut
        ? worstSegmentRecovery(s.worst_segment, segmentFlowOut.segments)
        : null,
    };
  }
  return { observed_at: stationFlow.observed_at, stations };
}

/**
 * Attach `method` (the constants plus the baseline's own provenance) to
 * crowding.ts's per-platform derivation, producing the published
 * PlatformCrowdingOut surface. Caller gates freshness before calling this.
 */
function buildPlatformCrowdingOut(
  stationWait: StationWaitDoc,
  ridershipBaseline: RidershipBaselineDoc,
  serviceWeights: ServiceWeightBaselineDoc | null,
  stations: Record<string, StationOut>,
  now: number,
): PlatformCrowdingOut {
  const result = derivePlatformCrowding(
    stationWait,
    ridershipBaseline,
    serviceWeights,
    stations,
    now,
  );
  return {
    ...result,
    method: {
      formula: 'entries_per_min * minutes_since_last_train',
      split_basis: serviceWeights
        ? 'scheduled_service_over_served_platforms'
        : 'uniform_over_served_platforms',
      max_gap_minutes: CROWDING_MAX_GAP_MINUTES,
      served_window_minutes: CROWDING_SERVED_WINDOW_MINUTES,
      excludes: ['in-system transfers', 'exits'],
      baseline_generated_at: ridershipBaseline.generated_at,
      baseline_window_start: ridershipBaseline.source.window_start,
      baseline_window_end: ridershipBaseline.source.window_end,
      service_weight_generated_at: serviceWeights?.generated_at ?? null,
      service_weight_feed_version: serviceWeights?.source.feed_version ?? null,
    },
  };
}

function argmaxOf(v: readonly [number, number, number]): 0 | 1 | 2 {
  if (v[0] >= v[1] && v[0] >= v[2]) return 0;
  if (v[1] >= v[2]) return 1;
  return 2;
}

/**
 * Decide which label to surface to consumers as `condition`.
 *
 *   - "unknown" published label (post-feed-gap) → use filter argmax
 *   - filter very confident (max p ≥ FAST_ATTACK_PROB) and disagrees with
 *     published.label → use filter argmax (skip the hysteresis lag)
 *   - otherwise → use the hysteresis-gated published.label
 *
 * The underlying publish state machine still respects HYSTERESIS_TICKS;
 * this only governs what we render.
 */
/**
 * Apply the condition precedence on top of the HMM label:
 *   1. real-time disruptive alert (lmm:alert:*) active → HMM condition (live
 *      reality wins, even if a planned alert is also active)
 *   2. else active planned "No Scheduled Service" → not_scheduled (off-timetable,
 *      not broken)
 *   3. else → HMM condition / normal
 */
function resolveCondition(
  roll: RouteRoll,
  disruptiveAlertCount: number,
  schedule: ScheduleFacts,
): string {
  if (schedule.hasRealtimeAlert)
    return effectiveCondition(roll, disruptiveAlertCount);
  if (schedule.isNotScheduled) return "not_scheduled";
  return effectiveCondition(roll, disruptiveAlertCount);
}

function effectiveCondition(
  roll: RouteRoll,
  disruptiveAlertCount: number,
): PublishedLabel {
  // Consistency guardrail: every disruption signal the filter sees is derived
  // from real-time alerts, so with zero real-time disruptive alerts the honest
  // condition is `normal` — planned/scheduled work never reads disrupted. It
  // also stops a stale or over-confident filter from publishing `disrupted` with
  // nothing live to explain it, and keeps system.overall_label consistent with
  // lines_disrupted_count.
  if (disruptiveAlertCount === 0) return "normal";
  const argmaxState = STATES[argmaxOf(roll.filter.probabilities)]!;
  if (roll.published.label === PUBLISHED_UNKNOWN) return argmaxState;
  const peakProb =
    roll.filter.probabilities[argmaxOf(roll.filter.probabilities)];
  if (peakProb >= FAST_ATTACK_PROB && argmaxState !== roll.published.label) {
    return argmaxState;
  }
  return roll.published.label;
}

function dwellQuantiles(selfLoop: number): {
  median: number;
  q25: number;
  q75: number;
} {
  const LARGE = 10_000;
  if (selfLoop >= 1.0) return { median: LARGE, q25: LARGE, q75: LARGE };
  if (selfLoop <= 0) return { median: 1, q25: 1, q75: 1 };
  const logSelf = Math.log(selfLoop);
  const q = (qv: number): number => {
    const target = 1 - qv;
    return target <= 0
      ? LARGE
      : Math.max(1, Math.ceil(Math.log(target) / logSelf));
  };
  return { median: q(0.5), q25: q(0.25), q75: q(0.75) };
}

/**
 * Document-level corruption that makes the whole snapshot unusable to ANY
 * consumer (missing version, no timestamp, no provenance). These block the
 * publish. Per-route inference problems do NOT belong here — one bad route must
 * never black out the entire feed; those are scrubbed instead (below).
 */
export function snapshotFatalViolations(s: Snapshot): string[] {
  const v: string[] = [];
  if (!s.schema_version) v.push("schema_version is empty");
  if (!Number.isFinite(s.generated_at) || s.generated_at <= 0) {
    v.push(`generated_at invalid: ${s.generated_at}`);
  }
  if (!s.provenance || typeof s.provenance.code_sha !== "string") {
    v.push("provenance.code_sha missing");
  }
  return v;
}

/**
 * Cross-surface consistency checks that don't corrupt a consumer but signal a
 * wiring regression — the rollups counting things the detail arrays then drop.
 * Warn-only: a self-contradictory feed is worse than a stale one only if it
 * also blacks out, so we log and keep publishing. The class of bug this catches
 * is exactly what shipped `alert_count: 14` next to `alerts: []`.
 */
export function snapshotConsistencyWarnings(s: Snapshot): string[] {
  const w: string[] = [];
  const subwayAlerts = s.system.by_mode.subway?.alert_count ?? 0;
  if (subwayAlerts > 0 && s.alerts.length === 0) {
    w.push(`system.alert_count=${subwayAlerts} but alerts[] is empty`);
  }
  const out =
    s.system.accessibility.elevators_out +
    s.system.accessibility.escalators_out;
  if (out > 0 && s.equipment.length === 0) {
    w.push(`accessibility reports ${out} units out but equipment[] is empty`);
  }
  return w;
}

/**
 * Null out any route inference carrying a non-finite (NaN/Infinity) number —
 * the only kind of value that genuinely poisons a consumer (it serializes to
 * `null` and breaks a numeric reader). The inference field is already nullable,
 * so a scrubbed route ships in a valid degraded state and the rest of the feed
 * publishes normally. Marginal floats (e.g. 1.0000001) are finite and ship
 * as-is — we do NOT range-check, because that once stalled the whole feed.
 *
 * A DELIBERATELY WITHHELD horizon is a different thing entirely: the 60- and
 * 120-minute forecasts are null by design on every fitted-curve row (see the
 * Inference interface), and null is their valid published value. Treating that
 * as corruption would scrub every inference on every tick and empty the feed of
 * exactly the data it exists to carry — which is what the regression test below
 * this function is guarding. Absent is fine; not-a-number is not.
 * Mutates `s` in place; returns the route ids scrubbed (for logging).
 */
export function scrubCorruptInferences(s: Snapshot): string[] {
  const scrubbed: string[] = [];
  const finiteOrWithheld = (v: number | null): boolean =>
    v === null || Number.isFinite(v);
  for (const [routeId, rs] of Object.entries(s.route_status)) {
    const inf = rs.inference;
    if (!inf) continue;
    const allFinite =
      Number.isFinite(inf.p_normal) &&
      Number.isFinite(inf.p_disrupted) &&
      Number.isFinite(inf.p_suspended) &&
      finiteOrWithheld(inf.p_normal_in_30min) &&
      finiteOrWithheld(inf.p_normal_in_60min) &&
      finiteOrWithheld(inf.p_normal_in_120min) &&
      Number.isFinite(inf.recovery_minutes) &&
      Number.isFinite(inf.recovery_minutes_low) &&
      Number.isFinite(inf.recovery_minutes_high);
    if (!allFinite) {
      rs.inference = null;
      scrubbed.push(routeId);
    }
  }
  return scrubbed;
}

export async function publishSnapshot(
  bucket: R2Bucket,
  snapshot: Snapshot,
): Promise<void> {
  // Scoped fail-safe: scrub corrupt per-route inferences (and keep publishing
  // everything else), and only refuse to publish on document-level corruption —
  // so the CDN keeps serving the last-good snapshot in that rare case. A single
  // bad route can never stale the whole feed.
  const scrubbed = scrubCorruptInferences(snapshot);
  if (scrubbed.length > 0) {
    console.warn(
      `publish: scrubbed non-finite inference on ${scrubbed.length} route(s): ${scrubbed.join(", ")}`,
    );
  }
  const fatal = snapshotFatalViolations(snapshot);
  if (fatal.length > 0) {
    throw new Error(
      `snapshot fatally malformed, not publishing: ${fatal.join("; ")}`,
    );
  }
  const inconsistencies = snapshotConsistencyWarnings(snapshot);
  if (inconsistencies.length > 0) {
    console.warn(
      `publish: snapshot consistency: ${inconsistencies.join("; ")}`,
    );
  }
  await bucket.put(SNAPSHOT_KEY, JSON.stringify(snapshot), {
    httpMetadata: {
      contentType: "application/json",
      cacheControl: PUBLIC_CACHE_CONTROL,
    },
  });
}

// The trains.json artifact's shape — see the file header comment for why
// this is a sibling of Snapshot rather than a field on it. Self-describing
// like Snapshot itself: its own observed_at, and the same provenance block
// (code_sha/dirty/producer) Snapshot carries, so a consumer holding only
// this object can still say which build produced it.
//
// fresh_feeds/expected_feeds exist because `positions` alone cannot
// distinguish "zero trains right now" from "some NYCT line-group feeds
// failed to decode, so those routes are silently missing" — Promise
// .allSettled in index.ts SKIPS a rejected feed rather than throwing, so
// nothing about a partial vehicle set is exceptional from trainPositions()'s
// point of view. fresh_feeds names which feeds decoded this tick (same
// convention archive.ts's archiveVehicleMetric/archiveTripUpdateMetric/
// archiveTraceRows already use); expected_feeds is the full constant list,
// same order, so a consumer can diff the two without hardcoding NYCT's feed
// grouping. fresh_feeds.length < expected_feeds.length means `positions` is
// a PARTIAL read — some line groups are silently absent, not actually
// empty — and only fresh_feeds.length === expected_feeds.length makes an
// empty `positions` a genuine "zero trains observed".
export interface PublishedTrains {
  observed_at: number;
  provenance: Provenance;
  fresh_feeds: string[];
  expected_feeds: string[];
  positions: TrainPositionOut[];
}

export function buildTrains(
  observedAt: number,
  positions: TrainPosition[],
  freshFeeds: readonly string[],
  expectedFeeds: readonly string[],
): PublishedTrains {
  return {
    observed_at: observedAt,
    provenance: codeProvenance(),
    fresh_feeds: [...freshFeeds],
    expected_feeds: [...expectedFeeds],
    positions,
  };
}

/**
 * Publish trains.json alongside (never gating, never gated on) snapshot.json.
 *
 * index.ts is responsible for the fail-soft contract, and it has two distinct
 * failure shapes to honor, not one:
 *   - NO feed decoded this tick (freshFeeds empty): index.ts must not call
 *     this at all. An all-feeds-failed publish would otherwise write
 *     {positions: []} — indistinguishable from "zero trains in NYC", a
 *     fabrication this surface exists specifically to avoid. The object is
 *     left un-rewritten; a consumer sees the last-good read, with its own
 *     observed_at to judge staleness, never a fabricated empty one.
 *   - SOME feeds decoded (freshFeeds a strict subset of expectedFeeds):
 *     index.ts calls this normally. The object IS published, flagged
 *     partial via fresh_feeds/expected_feeds, rather than withheld — a
 *     partial map is still useful, as long as it says so.
 *
 * Unlike publishSnapshot there's no fatal-violation gate here: TrainPosition
 * has no derived floating-point fields that could go non-finite.
 */
export async function publishTrains(
  bucket: R2Bucket,
  trains: PublishedTrains,
): Promise<void> {
  await bucket.put(TRAINS_KEY, JSON.stringify(trains), {
    httpMetadata: {
      contentType: "application/json",
      cacheControl: PUBLIC_CACHE_CONTROL,
    },
  });
}

// Re-export for the entrypoint to use, no need to import N_STATES directly.
export { N_STATES };
export const TICK_SECONDS = 300;
export const NO_ALERTS = NO_ALERTS_FALLBACK;
