/**
 * Snapshot-shape parity: a snapshot built by the Worker must validate against
 * schema/snapshot.schema.json — the JSON Schema generated from the Pydantic
 * model (the source of truth). If Python adds or retypes a contract field and
 * the TS `buildSnapshot` output isn't updated to match, this fails.
 *
 * Regenerate the schema with: uv run python -m scripts.export_schema
 */

import Ajv2020 from 'ajv/dist/2020';
import { describe, expect, test } from 'vitest';

import schema from '../../schema/snapshot.schema.json';
import type { RouteRoll } from '../src/alpha';
import type { RouteSnapshot } from '../src/derive';
import { TICK_SECONDS, buildSnapshot } from '../src/snapshot';

const ajv = new Ajv2020({ allErrors: true, strict: false });
const validate = ajv.compile(schema);

function check(snapshot: unknown): void {
  const ok = validate(snapshot);
  expect(
    ok,
    `snapshot failed schema/snapshot.schema.json:\n${JSON.stringify(validate.errors, null, 2)}`,
  ).toBe(true);
}

// A route snapshot carrying `n` active alerts, so the disrupted-condition path
// is reachable (the consistency guardrail forces `normal` when there are none).
function snapMapWithAlerts(routeId: string, n: number): Map<string, RouteSnapshot> {
  const ids = Array.from({ length: n }, (_, i) => `lmm:alert:${i}`);
  const m = new Map<string, RouteSnapshot>();
  m.set(routeId, {
    route_id: routeId,
    observation: {
      alert_count: n,
      severity_sum: n * 5,
      has_suspended_alert: false,
      has_delays: n > 0,
      has_service_change: false,
      has_planned: false,
      tod_bin: 0,
    },
    active_alert_ids: ids,
    alerts: [],
    severity_max: n > 0 ? 5 : 0,
    primary_alert_type: n > 0 ? 'Delays' : null,
    coarse_label: n > 0 ? 'Delays' : 'Good Service',
    by_direction: {
      northbound: { alerts: ids, primary_alert_type: n > 0 ? 'Delays' : null },
      southbound: { alerts: [], primary_alert_type: null },
    },
    has_realtime_alert: n > 0,
    is_not_scheduled: false,
    scheduled_resume_at: null,
  });
  return m;
}

describe('Worker snapshot conforms to the Pydantic-generated schema', () => {
  test('empty snapshot validates', () => {
    check(
      buildSnapshot({
        generatedAt: 1_700_000_000,
        alertsFreshness: 1_700_000_000,
        routeSnapshots: new Map(),
        rolls: {},
        trainedParams: null,
        tickSeconds: TICK_SECONDS,
      }),
    );
  });

  test('a populated observations surface validates, enum and all', () => {
    // The headway surface is the one thing in the snapshot fed by the GTFS-RT
    // protobuf, and Observation.direction is a CLOSED vocabulary in the
    // Pydantic model (ObservationDirection), so it reaches the JSON Schema as
    // an enum. Validating a real populated surface here is what keeps
    // headway.ts's HeadwayObservation from drifting off schema.py's
    // Observation — the two are hand-mirrored.
    const snap = buildSnapshot({
      generatedAt: 1_700_000_300,
      alertsFreshness: 1_700_000_300,
      routeSnapshots: new Map(),
      rolls: {},
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      vehiclePositionsFreshness: 1_700_000_300,
      headway: {
        observed_at: 1_700_000_300,
        reference_at: 1_699_000_000,
        reference_trained_at: 1_699_000_000,
        reference_stops: { '1|north': '121N', '1|south': '122S' },
        reference_fallbacks: {},
        cells: {
          '1|north': {
            stop_id: '121N',
            passings: [
              { at: 1_700_000_060, trip: 'T1' },
              { at: 1_700_000_300, trip: 'T2' },
            ],
          },
          '1|south': {
            stop_id: '122S',
            passings: [
              { at: 1_699_999_940, trip: 'S1' },
              { at: 1_700_000_250, trip: 'S2' },
            ],
          },
        },
        trips: {},
        gaps: [],
      },
    });
    check(snap);
    expect(snap.observations.length).toBe(2);
    expect(snap.observations.map((o) => o.direction)).toEqual(['north', 'south']);
    expect(snap.observations[0]).toMatchObject({
      entity_ref: 'subway_route:1',
      kind: 'headway',
      unit: 'seconds',
      value: 240,
      stop_id: '121N',
    });
    expect(snap.freshness.vehicle_positions).toBe(1_700_000_300);
  });

  test('snapshot carries a provenance block (falls back to unknown undeployed)', () => {
    const snap = buildSnapshot({
      generatedAt: 1_700_000_000,
      alertsFreshness: 1_700_000_000,
      routeSnapshots: new Map(),
      rolls: {},
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
    });
    check(snap);
    // __GIT_SHA__ isn't defined under vitest, so the typeof guard yields the
    // fallback — the point is the field exists and is well-formed.
    expect(snap.provenance).toEqual({
      code_sha: 'unknown',
      dirty: null,
      producer: 'worker',
      // Bootstrap params (trainedParams: null): the identity block is present
      // but null, an honest "no model version" rather than a missing field.
      params: { trained_at: null, key: null },
    });
  });

  test('provenance names the params version and its immutable versioned key', () => {
    const snap = buildSnapshot({
      generatedAt: 1_700_000_000,
      alertsFreshness: 1_700_000_000,
      routeSnapshots: new Map(),
      rolls: {},
      trainedParams: {
        schema_version: '1',
        trained_at: 1787787983,
        routes: {},
        dwell: {},
        dwellByAlert: {},
        movementBaseline: {},
        throughStops: null,
        serviceBaselineHourly: null,
      } as unknown as Parameters<typeof buildSnapshot>[0]['trainedParams'],
      tickSeconds: TICK_SECONDS,
    });
    check(snap);
    expect(snap.provenance.params).toEqual({
      trained_at: 1787787983,
      key: 'state/params/v1787787983.json',
    });
  });

  test('provenance.prov_ref: public PROV url when the served params carry one', () => {
    const snap = buildSnapshot({
      generatedAt: 1_700_000_000,
      alertsFreshness: 1_700_000_000,
      routeSnapshots: new Map(),
      rolls: {},
      trainedParams: {
        schema_version: '1',
        trained_at: 1787787983,
        routes: {},
        dwell: {},
        dwellByAlert: {},
        movementBaseline: {},
        throughStops: null,
        serviceBaselineHourly: null,
        // The params doc recorded its PROV sidecar's state/ key: a run that
        // emitted a PROV document. The snapshot points at the public mirror URL.
        provRef: 'state/prov/v1787787983.json',
      } as unknown as Parameters<typeof buildSnapshot>[0]['trainedParams'],
      tickSeconds: TICK_SECONDS,
    });
    check(snap);
    expect(snap.provenance.prov_ref).toBe(
      'https://feed.momentarily.nyc/v1/prov/v1787787983.json',
    );
  });

  test('provenance.prov_ref: absent for params trained before the emitter existed', () => {
    const snap = buildSnapshot({
      generatedAt: 1_700_000_000,
      alertsFreshness: 1_700_000_000,
      routeSnapshots: new Map(),
      rolls: {},
      trainedParams: {
        schema_version: '1',
        // The current live params (1788229972) predate the PROV emitter, so
        // parseTrainedParams yields provRef: null and no reference is fabricated.
        trained_at: 1788229972,
        routes: {},
        dwell: {},
        dwellByAlert: {},
        movementBaseline: {},
        throughStops: null,
        serviceBaselineHourly: null,
        provRef: null,
      } as unknown as Parameters<typeof buildSnapshot>[0]['trainedParams'],
      tickSeconds: TICK_SECONDS,
    });
    check(snap);
    // Absent, never null — an unpublished PROV doc is a missing field.
    expect('prov_ref' in snap.provenance).toBe(false);
  });

  test('provenance.prov_ref: absent when params failed to load (bootstrap)', () => {
    // trainedParams === null covers both a first-deploy bootstrap and a
    // read/parse failure of params.json: no params identity, so no PROV to walk.
    const snap = buildSnapshot({
      generatedAt: 1_700_000_000,
      alertsFreshness: 1_700_000_000,
      routeSnapshots: new Map(),
      rolls: {},
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
    });
    check(snap);
    expect('prov_ref' in snap.provenance).toBe(false);
  });

  test('system.accessibility sums elevators/escalators across station_status', () => {
    const snap = buildSnapshot({
      generatedAt: 1_700_000_000,
      alertsFreshness: 1_700_000_000,
      routeSnapshots: new Map(),
      rolls: {},
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
      stationStatuses: {
        '601': {
          station_complex_id: '601',
          alerts: [],
          ada_status: 'ada_degraded',
          elevators_total: 3,
          elevators_out: 2,
          escalators_total: 1,
          escalators_out: 1,
          earliest_elevator_return: null,
          oldest_outage_since: null,
        },
        '602': {
          station_complex_id: '602',
          alerts: [],
          ada_status: 'operational',
          elevators_total: 2,
          elevators_out: 1,
          escalators_total: 0,
          escalators_out: 0,
          earliest_elevator_return: null,
          oldest_outage_since: null,
        },
      },
    });
    check(snap);
    expect(snap.system.accessibility).toEqual({
      elevators_out: 3,
      escalators_out: 1,
      ada_pathways_degraded: 1,
    });
  });

  test('snapshot with an inferred route validates', () => {
    const roll: RouteRoll = {
      filter: {
        probabilities: [0.1, 0.3, 0.6],
        regime_entered_at: 1_699_999_000,
        last_updated_at: 1_700_000_000,
      },
      published: {
        label: 'suspended',
        pending_state: 'suspended',
        pending_streak: 3,
        last_updated_at: 1_700_000_000,
      },
      alert_type_at_entry: null,
    };
    check(
      buildSnapshot({
        generatedAt: 1_700_000_000,
        alertsFreshness: 1_700_000_000,
        routeSnapshots: new Map(),
        rolls: { '1': roll },
        trainedParams: null,
        tickSeconds: TICK_SECONDS,
      }),
    );
  });

  test('effectiveCondition: ambiguous filter (max p < 0.9) keeps hysteresis label', () => {
    const snap = buildSnapshot({
      generatedAt: 1_700_000_000,
      alertsFreshness: 1_700_000_000,
      routeSnapshots: snapMapWithAlerts('1', 1),
      rolls: {
        '1': {
          filter: {
            probabilities: [0.35, 0.6, 0.05],  // disrupted leading but < 0.9
            regime_entered_at: 1_699_999_700,
            last_updated_at: 1_700_000_000,
          },
          published: {
            label: 'normal',
            pending_state: 'disrupted',
            pending_streak: 1,
            last_updated_at: 1_700_000_000,
          },
          alert_type_at_entry: null,
        },
      },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
    });
    expect(snap.route_status['1']!.inference!.condition).toBe('normal');
  });

  test('effectiveCondition: confident filter (max p >= 0.9) overrides stale label', () => {
    const snap = buildSnapshot({
      generatedAt: 1_700_000_000,
      alertsFreshness: 1_700_000_000,
      routeSnapshots: snapMapWithAlerts('1', 1),
      rolls: {
        '1': {
          filter: {
            probabilities: [0.05, 0.94, 0.01],  // disrupted at 0.94 >= FAST_ATTACK_PROB
            regime_entered_at: 1_700_000_000,
            last_updated_at: 1_700_000_000,
          },
          published: {
            label: 'normal',           // hysteresis-lagged
            pending_state: 'disrupted',
            pending_streak: 1,
            last_updated_at: 1_700_000_000,
          },
          alert_type_at_entry: null,
        },
      },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
    });
    expect(snap.route_status['1']!.inference!.condition).toBe('disrupted');
  });

  test('effectiveCondition: confident filter agreeing with label is a no-op', () => {
    const snap = buildSnapshot({
      generatedAt: 1_700_000_000,
      alertsFreshness: 1_700_000_000,
      routeSnapshots: snapMapWithAlerts('1', 1),
      rolls: {
        '1': {
          filter: {
            probabilities: [0.05, 0.94, 0.01],
            regime_entered_at: 1_699_999_000,
            last_updated_at: 1_700_000_000,
          },
          published: {
            label: 'disrupted',
            pending_state: 'disrupted',
            pending_streak: 5,
            last_updated_at: 1_700_000_000,
          },
          alert_type_at_entry: null,
        },
      },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
    });
    expect(snap.route_status['1']!.inference!.condition).toBe('disrupted');
  });

  test('effectiveCondition: unknown label falls back to filter argmax', () => {
    const snap = buildSnapshot({
      generatedAt: 1_700_000_000,
      alertsFreshness: 1_700_000_000,
      routeSnapshots: snapMapWithAlerts('1', 1),
      rolls: {
        '1': {
          filter: {
            probabilities: [0.5, 0.3, 0.2],  // normal leads, NOT above FAST_ATTACK
            regime_entered_at: 1_699_999_000,
            last_updated_at: 1_700_000_000,
          },
          published: {
            label: 'unknown',           // post-feed-gap
            pending_state: 'normal',
            pending_streak: 1,
            last_updated_at: 1_700_000_000,
          },
          alert_type_at_entry: null,
        },
      },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
    });
    expect(snap.route_status['1']!.inference!.condition).toBe('normal');
  });

  test('guardrail: confident disrupted filter with zero active alerts publishes normal', () => {
    const snap = buildSnapshot({
      generatedAt: 1_700_000_000,
      alertsFreshness: 1_700_000_000,
      routeSnapshots: new Map(), // no active alerts on the route
      rolls: {
        '1': {
          filter: {
            probabilities: [0.02, 0.97, 0.01], // filter latched in disrupted
            regime_entered_at: 1_699_980_000,
            last_updated_at: 1_700_000_000,
          },
          published: {
            label: 'disrupted',
            pending_state: 'disrupted',
            pending_streak: 5,
            last_updated_at: 1_700_000_000,
          },
          alert_type_at_entry: null,
        },
      },
      trainedParams: null,
      tickSeconds: TICK_SECONDS,
    });
    // No alert to explain a disruption → the shadow HMM condition is gated
    // to normal, is_disrupted is false, and recovery collapses to 0.
    expect(snap.route_status['1']!.inference!.condition).toBe('normal');
    expect(snap.route_status['1']!.inference!.is_disrupted).toBe(false);
    expect(snap.route_status['1']!.inference!.recovery_minutes).toBe(0);
  });

  test('an alert-arm inference publishes recovery but withholds every forecast horizon', () => {
    const trained = {
      schema_version: '1',
      trained_at: 1,
      routes: {},
      dwell: {
        '1': {
          disrupted: {
            n: 50,
            q25_sec: 600,
            median_sec: 1800,
            q75_sec: 5400,
            recover_by_30: 0.4,
            recover_by_60: 0.7,
            recover_by_120: 0.95,
          },
        },
      },
      dwellByAlert: {},
      dwellMovement: {},
      movementBaseline: {},
      throughStops: null,
      serviceBaseline: {},
      serviceBaselineHourly: {},
      scheduleRate: {},
      provRef: null,
    };
    const snap = buildSnapshot({
      generatedAt: 1_700_000_000,
      alertsFreshness: 1_700_000_000,
      routeSnapshots: snapMapWithAlerts('1', 1),
      rolls: {
        '1': {
          filter: {
            probabilities: [0.05, 0.94, 0.01],
            regime_entered_at: 1_699_990_000,
            last_updated_at: 1_700_000_000,
          },
          published: {
            label: 'disrupted',
            pending_state: 'disrupted',
            pending_streak: 5,
            last_updated_at: 1_700_000_000,
          },
          alert_type_at_entry: 'Delays',
        },
      },
      trainedParams: trained,
      tickSeconds: TICK_SECONDS,
    });
    const inf = snap.route_status['1']!.inference!;
    // The alert arm still estimates recovery from the empirical cell, but it
    // publishes no forecast: no movement reading here, so the published
    // condition is 'unknown' and a probability sourced from the alert regime
    // would describe something else entirely.
    expect(snap.route_status['1']!.condition).toBe('unknown');
    expect(inf.recovery_source).toBe('hmm');
    expect(inf.p_normal_in_30min).toBeNull();
    expect(inf.p_normal_in_60min).toBeNull();
    expect(inf.p_normal_in_120min).toBeNull();
    // An unreadable route has nothing to recover from, so the estimate is not
    // clamped to the indeterminate ceiling either.
    expect(inf.recovery_indeterminate).toBe(false);
  });
});
