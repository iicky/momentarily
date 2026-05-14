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
});
