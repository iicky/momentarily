/**
 * Events layer (schema events/s1) across languages.
 *
 * Every fixture under tests/fixtures/model/events/records/ (round-tripped
 * through Pydantic by tests/test_model_events.py) must validate against the
 * Pydantic-exported JSON Schema and read through the generated types. The
 * generated-TS drift check for every layer lives in model_network.test.ts.
 */

/// <reference types="node" />

import Ajv2020, { type ValidateFunction } from 'ajv/dist/2020';
import { describe, expect, test } from 'vitest';

import coverageSchema from '../../schema/model/s1/events/coverage_interval.schema.json';
import headwaySchema from '../../schema/model/s1/events/headway.schema.json';
import openStateSchema from '../../schema/model/s1/events/open_state.schema.json';
import traversalSchema from '../../schema/model/s1/events/segment_traversal.schema.json';
import stopEventSchema from '../../schema/model/s1/events/stop_event.schema.json';
import supplySchema from '../../schema/model/s1/events/supply_observation.schema.json';
import tripMatchSchema from '../../schema/model/s1/events/trip_match.schema.json';
import type { CoverageInterval } from '../../shared/model/s1/events/coverage_interval';
import type { Headway } from '../../shared/model/s1/events/headway';
import type { StopEvent } from '../../shared/model/s1/events/stop_event';
import coverage from '../../tests/fixtures/model/events/records/coverage_interval.json';
import headway from '../../tests/fixtures/model/events/records/headway.json';
import openState from '../../tests/fixtures/model/events/records/open_state.json';
import traversal from '../../tests/fixtures/model/events/records/segment_traversal.json';
import stopEvent from '../../tests/fixtures/model/events/records/stop_event.json';
import supply from '../../tests/fixtures/model/events/records/supply_observation.json';
import tripMatch from '../../tests/fixtures/model/events/records/trip_match.json';

const ajv = new Ajv2020({ allErrors: true, strict: false });

const DOCUMENTS: Record<string, { schema: object; records: object[] }> = {
  trip_match: { schema: tripMatchSchema, records: tripMatch },
  stop_event: { schema: stopEventSchema, records: stopEvent },
  segment_traversal: { schema: traversalSchema, records: traversal },
  headway: { schema: headwaySchema, records: headway },
  open_state: { schema: openStateSchema, records: openState },
  coverage_interval: { schema: coverageSchema, records: coverage },
  supply_observation: { schema: supplySchema, records: supply },
};

function parse<T>(validate: ValidateFunction<T>, doc: unknown): T {
  if (!validate(doc)) throw new Error(JSON.stringify(validate.errors, null, 2));
  return doc;
}

describe('event fixtures conform to their exported schemas', () => {
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

describe('the generated types carry the layer-3 distinctions', () => {
  test('reported_planned and stopped are separate service evidence', () => {
    const validate = ajv.compile<Headway>(headwaySchema);
    const [stopped, planned] = headway.map((h) => parse(validate, h));
    expect(stopped?.follower_service_evidence).toBe('stopped');
    expect(planned?.follower_service_evidence).toBe('reported_planned');
    expect(validate({ ...planned, follower_service_evidence: 'service_not_separable' })).toBe(false);
  });

  test('a "last"-family dwell is an Unknown with its reason, not a number or null', () => {
    const validate = ajv.compile<StopEvent>(stopEventSchema);
    const last = parse(validate, stopEvent[0]);
    expect(last.family).toBe('last');
    expect(last.dwell).toMatchObject({ unknown: true });
    expect(validate({ ...last, dwell: { unknown: true } })).toBe(false);
  });

  test('a left-censored bound has a null after', () => {
    const validate = ajv.compile<StopEvent>(stopEventSchema);
    const first = stopEvent.map((s) => parse(validate, s)).find((s) => s.flags.includes('first_of_run'));
    expect(first?.arrival?.after).toBeNull();
    expect(first?.arrival?.uncovered_s).toBeNull();
  });

  test('coverage states include not_stored', () => {
    const validate = ajv.compile<CoverageInterval>(coverageSchema);
    const states = coverage.map((c) => parse(validate, c).state);
    expect(states).toContain('not_stored');
    expect(validate({ ...coverage[0], state: 'low_supply' })).toBe(false);
  });
});
