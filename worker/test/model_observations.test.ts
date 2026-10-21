/**
 * Observations layer (schema observations/s1) across languages.
 *
 * Every fixture under tests/fixtures/model/observations/ (hand-written from the
 * live feeds, round-tripped through Pydantic by tests/test_model_observations.py)
 * must validate against the Pydantic-exported JSON Schema and read through the
 * generated types. The generated-TS drift check for every layer lives in
 * model_network.test.ts.
 */

/// <reference types="node" />

import Ajv2020, { type ValidateFunction } from 'ajv/dist/2020';
import { describe, expect, test } from 'vitest';

import alertPresenceSchema from '../../schema/model/s1/observations/alert_presence.schema.json';
import alertVersionSchema from '../../schema/model/s1/observations/alert_version.schema.json';
import collectorTickSchema from '../../schema/model/s1/observations/collector_tick.schema.json';
import fileMetadataSchema from '../../schema/model/s1/observations/derived_file_metadata.schema.json';
import equipmentSchema from '../../schema/model/s1/observations/equipment_observation.schema.json';
import feedHeaderSchema from '../../schema/model/s1/observations/feed_header_observation.schema.json';
import healthSchema from '../../schema/model/s1/observations/legacy_collector_health.schema.json';
import legacyFetchSchema from '../../schema/model/s1/observations/legacy_fetch_outcome.schema.json';
import legacySchema from '../../schema/model/s1/observations/legacy_route_alert_state.schema.json';
import bundleIndexSchema from '../../schema/model/s1/observations/tick_bundle_index.schema.json';
import tripUpdateSchema from '../../schema/model/s1/observations/trip_update_observation.schema.json';
import vehicleSchema from '../../schema/model/s1/observations/vehicle_observation.schema.json';
import type { CollectorTick } from '../../shared/model/s1/observations/collector_tick';
import type { LegacyRouteAlertState } from '../../shared/model/s1/observations/legacy_route_alert_state';
import type { TripUpdateObservation } from '../../shared/model/s1/observations/trip_update_observation';
import type { VehicleObservation } from '../../shared/model/s1/observations/vehicle_observation';
import alertPresence from '../../tests/fixtures/model/observations/alert_presence.json';
import alertVersion from '../../tests/fixtures/model/observations/alert_version.json';
import collectorTick from '../../tests/fixtures/model/observations/collector_tick.json';
import fileMetadata from '../../tests/fixtures/model/observations/derived_file_metadata.json';
import equipment from '../../tests/fixtures/model/observations/equipment_observation.json';
import feedHeader from '../../tests/fixtures/model/observations/feed_header_observation.json';
import health from '../../tests/fixtures/model/observations/legacy_collector_health.json';
import legacyFetch from '../../tests/fixtures/model/observations/legacy_fetch_outcome.json';
import legacy from '../../tests/fixtures/model/observations/legacy_route_alert_state.json';
import bundleIndex from '../../tests/fixtures/model/observations/tick_bundle_index.json';
import tripUpdate from '../../tests/fixtures/model/observations/trip_update_observation.json';
import vehicle from '../../tests/fixtures/model/observations/vehicle_observation.json';

const ajv = new Ajv2020({ allErrors: true, strict: false });

const DOCUMENTS: Record<string, { schema: object; records: object[] }> = {
  collector_tick: { schema: collectorTickSchema, records: collectorTick },
  tick_bundle_index: { schema: bundleIndexSchema, records: bundleIndex },
  vehicle_observation: { schema: vehicleSchema, records: vehicle },
  trip_update_observation: { schema: tripUpdateSchema, records: tripUpdate },
  feed_header_observation: { schema: feedHeaderSchema, records: feedHeader },
  alert_presence: { schema: alertPresenceSchema, records: alertPresence },
  alert_version: { schema: alertVersionSchema, records: alertVersion },
  equipment_observation: { schema: equipmentSchema, records: equipment },
  legacy_route_alert_state: { schema: legacySchema, records: legacy },
  legacy_fetch_outcome: { schema: legacyFetchSchema, records: legacyFetch },
  legacy_collector_health: { schema: healthSchema, records: health },
  derived_file_metadata: { schema: fileMetadataSchema, records: fileMetadata },
};

function parse<T>(validate: ValidateFunction<T>, doc: unknown): T {
  if (!validate(doc)) throw new Error(JSON.stringify(validate.errors, null, 2));
  return doc;
}

describe('observation fixtures conform to their exported schemas', () => {
  for (const [name, { schema, records }] of Object.entries(DOCUMENTS)) {
    test(`${name}: every record validates; a missing or extra field does not`, () => {
      const validate = ajv.compile(schema);
      expect(records.length).toBeGreaterThan(0);
      for (const record of records) {
        expect(validate(record), `${name}: ${JSON.stringify(validate.errors)}`).toBe(true);
        const [first] = Object.keys(record);
        const missing = Object.fromEntries(Object.entries(record).filter(([k]) => k !== first));
        expect(validate(missing), `${name} without ${first}`).toBe(false);
        expect(validate({ ...record, extra: true }), `${name} with an extra field`).toBe(false);
      }
    });
  }
});

describe('the generated types carry the three states', () => {
  test('a placeholder vehicle reads as unassigned-unknown, not false, with a future timestamp', () => {
    const validate = ajv.compile<VehicleObservation>(vehicleSchema);
    const [assigned, placeholder] = vehicle.map((v) => parse(validate, v));
    expect(assigned?.trip?.nyct_is_assigned).toBe(true);
    expect(placeholder?.trip?.nyct_is_assigned).toBeNull();
    expect(placeholder?.current_status).toBeNull();
    expect(placeholder?.vehicle_timestamp).toBeGreaterThan(placeholder?.observed_at ?? Infinity);
    expect(validate({ ...placeholder, current_status: 7 })).toBe(false);
  });

  test('route ids stay as published', () => {
    const validate = ajv.compile<VehicleObservation>(vehicleSchema);
    const routes = vehicle.map((v) => parse(validate, v).trip?.route_id);
    expect(routes).toContain('6X');
  });

  test('legacy rows are told apart by lineage alone', () => {
    const validate = ajv.compile<VehicleObservation>(vehicleSchema);
    const rows = vehicle.map((v) => parse(validate, v));
    const legacyRows = rows.filter((r) => r.lineage.kind === 'legacy');
    expect(legacyRows.map((r) => (r.lineage.kind === 'legacy' ? r.lineage.archive : null))).toEqual([
      'archive/trace',
    ]);
    const [first] = rows;
    expect(validate({ ...first, lineage: { ...first?.lineage, kind: 'derived' } })).toBe(false);
  });

  test('a negative delay validates and stays negative', () => {
    const validate = ajv.compile<TripUpdateObservation>(tripUpdateSchema);
    const update = parse(validate, tripUpdate[0]);
    expect(update.delay).toBeLessThan(0);
    expect(update.stop_times.map((s) => s.arrival_delay)).toEqual([-49, -39]);
  });

  test('an Unknown needs its reason', () => {
    const validate = ajv.compile(alertVersionSchema);
    const legacyVersion = alertVersion[1];
    expect(validate(legacyVersion)).toBe(true);
    expect(validate({ ...legacyVersion, gone_by: { unknown: true } })).toBe(false);
    expect(validate({ ...legacyVersion, gone_by: { unknown: false, reason: 'x' } })).toBe(false);
  });

  test('a fetch that failed still records its outcome', () => {
    const validate = ajv.compile<CollectorTick>(collectorTickSchema);
    const tick = parse(validate, collectorTick[0]);
    const outcomes = Object.fromEntries(tick.fetches.map((f) => [f.source, f.outcome]));
    expect(outcomes).toMatchObject({ si: 'timeout', g: 'http_error', jz: 'empty', l: 'ok' });
    expect(tick.fetches.find((f) => f.source === 'si')?.finished_at_ms).toBeNull();
  });
});

describe('legacy route alert state says what it is', () => {
  const validate = ajv.compile<LegacyRouteAlertState>(legacySchema);
  const [record] = legacy;

  test('the lineage, its period and the granularity are required', () => {
    const state = parse(validate, record);
    expect(state.lineage).toMatchObject({ kind: 'legacy', archive: 'v1/predictions' });
    expect(state.granularity).toEqual({
      scope: 'route',
      tick_seconds: 300,
      content: 'primary_alert_type_presence',
    });
    const { lineage: _lineage, ...unmarked } = state;
    expect(validate(unmarked)).toBe(false);
    const { period: _period, ...undated } = state.lineage;
    expect(validate({ ...state, lineage: undated })).toBe(false);
    expect(validate({ ...state, lineage: { ...state.lineage, kind: 'raw' } })).toBe(false);
  });

  test('the granularity cannot claim a finer source', () => {
    const state = parse(validate, record);
    expect(validate({ ...state, granularity: { ...state.granularity, tick_seconds: 60 } })).toBe(false);
    expect(validate({ ...state, granularity: { ...state.granularity, scope: 'alert' } })).toBe(false);
  });
});
