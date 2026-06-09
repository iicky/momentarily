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
    expect(snap.route_status['1']!.condition).toBe('normal');
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
    expect(snap.route_status['1']!.condition).toBe('disrupted');
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
    expect(snap.route_status['1']!.condition).toBe('disrupted');
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
    expect(snap.route_status['1']!.condition).toBe('normal');
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
    // No alert to explain a disruption → condition and inference are gated to
    // normal, is_disrupted is false, and recovery collapses to 0.
    expect(snap.route_status['1']!.condition).toBe('normal');
    expect(snap.route_status['1']!.inference!.condition).toBe('normal');
    expect(snap.route_status['1']!.inference!.is_disrupted).toBe(false);
    expect(snap.route_status['1']!.inference!.recovery_minutes).toBe(0);
  });

  test('p_normal_in_X uses empirical recovery fractions when a dwell cell exists', () => {
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
    // Empirical recovery curve replaces the geometric projection.
    expect(inf.p_normal_in_30min).toBe(0.4);
    expect(inf.p_normal_in_60min).toBe(0.7);
    expect(inf.p_normal_in_120min).toBe(0.95);
  });
});
